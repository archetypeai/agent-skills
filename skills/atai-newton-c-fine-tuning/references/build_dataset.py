#!/usr/bin/env python3
"""Turn a folder of labelled clips into the four files a Newton C fine-tune needs.

    <src>/train/<Class>/*.mp4  ->  train.jsonl      fine-tuning `messages` (training_files)
    <src>/val/<Class>/*.mp4    ->  val.jsonl        fine-tuning `messages` (validation_files)
    <src>/test/<Class>/*.mp4   ->  test_closed.jsonl  InferenceRecord, prompt lists the labels
                                   test_open.jsonl    InferenceRecord, no labels in the prompt
                                   test_labels.json   ground truth, in line order

The two APIs take DIFFERENT formats: training is OpenAI chat `messages` with frames
as image_url data URIs; batch inference is `{system, prompt, inputs:[{type, format,
data}]}`. Both carry the same N evenly spaced JPEG frames per clip.

Needs ffmpeg + ffprobe on PATH; nothing to pip install.

    python3 build_dataset.py --src data/UCF101_subset --out dataset
    python3 build_dataset.py --src clips --out dataset --n-frames 8 --max-side 448
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path

SYSTEM = "You are a video action-recognition assistant."
QUESTION = "What action is the person performing in this video? Answer with the action name."
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def closed_prompt(classes: list[str]) -> str:
    return ("What action is the person performing in this video? "
            f"Answer with exactly one of these labels: {', '.join(classes)}.")


def frame_times(duration: float, n: int) -> list[float]:
    """Centres of n equal segments: never the very first or last frame."""
    return [duration * (i + 0.5) / n for i in range(n)]


def sample_frames(video: Path, n: int, max_side: int) -> list[str]:
    """n evenly spaced frames as base64 JPEG, longest side <= max_side."""
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        text=True).strip())
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(frame_times(dur, n)):
            jpg = Path(tmp) / f"{i}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
                            "-frames:v", "1", "-vf", scale, "-q:v", "3", "-y", str(jpg)], check=True)
            if jpg.exists():
                out.append(base64.b64encode(jpg.read_bytes()).decode())
    if not out:
        raise RuntimeError(f"no frames from {video}")
    return out


def training_record(frames: list[str], label: str) -> dict:
    content = [{"type": "text", "text": QUESTION}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in frames]
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
        {"role": "assistant", "content": label},
    ]}


def inference_record(frames: list[str], prompt: str) -> dict:
    return {"system": SYSTEM, "prompt": prompt,
            "inputs": [{"type": "image", "format": "base64", "data": b} for b in frames]}


def clips(src: Path, split: str, classes: list[str]) -> list[tuple[Path, str]]:
    return [(p, c) for c in classes for p in sorted((src / split / c).iterdir())
            if p.suffix.lower() in VIDEO_EXT]


def write_jsonl(path: Path, records) -> int:
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=448)
    a = ap.parse_args()

    for split in ("train", "val", "test"):
        if not (a.src / split).is_dir():
            raise SystemExit(f"missing {a.src / split}/<Class>/*.mp4")
    classes = sorted(d.name for d in (a.src / "train").iterdir() if d.is_dir())
    a.out.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, list[str]] = {}

    def frames(p: Path) -> list[str]:
        if p not in cache:
            cache[p] = sample_frames(p, a.n_frames, a.max_side)
        return cache[p]

    for split, name in (("train", "train.jsonl"), ("val", "val.jsonl")):
        rows = clips(a.src, split, classes)
        n = write_jsonl(a.out / name, (training_record(frames(p), c) for p, c in rows))
        print(f"{name}: {n} clips")

    test = clips(a.src, "test", classes)
    write_jsonl(a.out / "test_closed.jsonl", (inference_record(frames(p), closed_prompt(classes)) for p, _ in test))
    write_jsonl(a.out / "test_open.jsonl", (inference_record(frames(p), QUESTION) for p, _ in test))
    (a.out / "test_labels.json").write_text(json.dumps({"classes": classes, "labels": [c for _, c in test]}))
    print(f"test: {len(test)} clips -> test_closed.jsonl, test_open.jsonl, test_labels.json")
    print(f"classes ({len(classes)}): {', '.join(classes)}")


if __name__ == "__main__":
    main()
