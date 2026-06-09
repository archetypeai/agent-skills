"""
Video examples for Newton C 2.6 fusion model via /query (prod).

C 2.6 reasons over video on /query: attach an .mp4 by file_id and set
`max_frames`, and GPQ decodes + uniformly samples the clip server-side before
the model sees the frames. No client-side video tooling needed. This is the
capability that distinguishes C 2.6 from the C 2.4 / 2.5 text checkpoints,
which accept an .mp4 but ignore the frames ("I can't see videos").

Note: `multi_image` is NOT a video knob — it switches the model to multi-image
mode (multiple attached images treated as independent images, not video
frames). For video, send the .mp4 and let `max_frames` sample it.

The demo uses a worker-assembly PASS/FAIL inspection prompt; ground truth for
the sample clip (named 1_pass_2_pass_3_pass) is all three steps PASS.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY
    python video_query.py
    # or point at your own .mp4:
    python video_query.py /path/to/your.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

from _common import MODEL, banner, client, extract_text, upload_file

DEFAULT_VIDEO = Path(__file__).parent / "sample_assets" / "1_pass_2_pass_3_pass_B.mp4"

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
    """Upload the .mp4, let GPQ decode + sample it server-side."""
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


if __name__ == "__main__":
    main()
