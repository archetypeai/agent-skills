"""
Video examples for Newton C 2.6 fusion model via /query (prod).

C 2.6 genuinely reasons over video on /query — unlike the C 2.4 / 2.5 text
checkpoints, which accept an .mp4 but ignore the sampled frames (they reply
"I can't see videos"). Two working paths, both on c2_6_8b_fp8:

  (A) Pass the .mp4 directly by file_id + `max_frames`. GPQ decodes and
      samples the video server-side. Simplest — no client-side video tooling.
      This is the recommended default.

  (B) Sample frames yourself client-side and send them as multiple
      `data.base64_img` events with `multi_image: true`. Use when you want
      control over which/how many frames, or already have frames. Requires a
      local video decoder (ffmpeg) to sample. `multi_image: true` is REQUIRED
      — without it, multiple image inputs return 400 query_failed.

Both were verified to read the sample assembly clip and return the correct
PASS/FAIL inspection. The demo uses a worker-assembly inspection prompt.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY
    python video_query.py
    # or point at your own .mp4:
    python video_query.py /path/to/your.mp4
"""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import requests

from _common import MODEL, banner, client, extract_text, upload_file

DEFAULT_VIDEO = Path(__file__).parent / "sample_assets" / "1_pass_2_pass_3_pass_B.mp4"

# A worker-assembly inspection prompt: PASS/FAIL per step. Ground truth for the
# sample clip (named 1_pass_2_pass_3_pass) is all three steps PASS.
ASSEMBLY_PROMPT = (
    "You are observing a worker doing an assembly job.\n"
    "Available Components: O-ring (black), Cap (blue), Manifold (black), Wrench\n"
    "Evaluate each step. Answer PASS if the action was performed according to "
    "the description, otherwise FAIL.\n"
    "Step 1: attaching / placing / aligning the o-ring to the cap.\n"
    "Step 2: attaching / inserting / aligning the cap to the manifold.\n"
    "Step 3: using a wrench to tighten the cap.\n"
    "Respond in EXACTLY this format:\n"
    "Step 1: PASS or FAIL\nDescription: <1-2 sentences>\n"
    "Step 2: PASS or FAIL\nDescription: <1-2 sentences>\n"
    "Step 3: PASS or FAIL\nDescription: <1-2 sentences>\n"
    "Summary: <2-3 sentences>"
)
QUERY = "Evaluate the assembly steps shown in the attached video."


def _post_query(body: dict, timeout: int = 600) -> tuple[str, int]:
    """Single /query POST. Returns (text, elapsed_ms)."""
    endpoint, headers = client()
    t0 = time.time()
    r = requests.post(f"{endpoint}/query", headers=headers, json=body, timeout=timeout)
    ms = int((time.time() - t0) * 1000)
    if not r.ok:
        sys.exit(f"Query failed [{r.status_code}]: {r.text[:400]}")
    return extract_text(r.json()), ms


def example_mp4_direct(video_path: Path) -> None:
    """Path A — upload the .mp4, let GPQ decode + sample it server-side."""
    banner("1. Video via .mp4 file_id + max_frames (server-side sampling)")
    file_id = upload_file(video_path)
    print(f"Uploaded → file_id={file_id}")

    body = {
        "query": QUERY,
        "instruction_prompt": ASSEMBLY_PROMPT,
        "file_ids": [file_id],
        "model": MODEL,
        "max_new_tokens": 500,
        "max_frames": 32,  # frames GPQ samples uniformly from the video (default 32)
        "sanitize": False,
    }
    text, ms = _post_query(body)
    print(f"[{ms} ms]\n{text}\n")


def example_max_frames_tradeoff(video_path: Path) -> None:
    """Same clip at two frame budgets — more frames = more temporal detail, more latency."""
    banner("2. max_frames tradeoff (8 vs 32 frames)")
    file_id = upload_file(video_path)
    for n_frames in (8, 32):
        body = {
            "query": QUERY,
            "instruction_prompt": ASSEMBLY_PROMPT,
            "file_ids": [file_id],
            "model": MODEL,
            "max_new_tokens": 500,
            "max_frames": n_frames,
            "sanitize": False,
        }
        text, ms = _post_query(body)
        print(f"--- max_frames={n_frames:>2}  [{ms} ms] ---\n{text}\n")


def _sample_frames(video_path: Path, n: int = 8) -> list[str] | None:
    """Sample ~n frames uniformly with ffmpeg. Returns base64 PNGs, or None if ffmpeg is missing."""
    if shutil.which("ffmpeg") is None:
        return None
    tmp = tempfile.mkdtemp(prefix="vframes_")
    fps = "0.3"
    try:  # aim for ~n frames across the clip's duration
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        fps = f"{max(n / float(out), 0.01):.4f}"
    except Exception:
        pass
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(video_path), "-vf", f"fps={fps}",
         f"{tmp}/f_%03d.png"],
        check=True, timeout=120,
    )
    frames = sorted(Path(tmp).glob("f_*.png"))[:n]
    return [base64.b64encode(p.read_bytes()).decode("ascii") for p in frames]


def example_client_sampled_frames(video_path: Path) -> None:
    """Path B — sample frames client-side, send as base64_img events + multi_image."""
    banner("3. Client-sampled frames via base64_img events + multi_image=true")
    frames = _sample_frames(video_path, n=8)
    if frames is None:
        print(
            "ffmpeg not found — skipping the client-sampling demo.\n"
            "Install ffmpeg to run this path, or just use example 1 (.mp4 direct).\n"
        )
        return
    print(f"Sampled {len(frames)} frames client-side")

    body = {
        "query": QUERY,
        "instruction_prompt": ASSEMBLY_PROMPT,
        "file_ids": [],
        "model": MODEL,
        "max_new_tokens": 500,
        "multi_image": True,  # REQUIRED — treat the frames as ONE multi-frame input
        "sanitize": False,
        "events": [
            {"type": "data.base64_img", "event_data": {"contents": b, "mime_type": "image/png"}}
            for b in frames
        ],
    }
    text, ms = _post_query(body)
    print(f"[{ms} ms]\n{text}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help=f"Path to an .mp4 file. Defaults to {DEFAULT_VIDEO.name}.",
    )
    args = parser.parse_args()
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    example_mp4_direct(video_path)
    example_max_frames_tradeoff(video_path)
    example_client_sampled_frames(video_path)


if __name__ == "__main__":
    main()
