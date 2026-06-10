"""
Video examples for Newton C 2.6 fusion model via /query (prod).

C 2.6 reasons over video on /query, two ways:

  (a) Attach an .mp4 by file_id and set `max_frames` (default 32, max 64) —
      GPQ decodes + uniformly samples the clip server-side. No client-side
      video tooling needed.

  (b) Send frames you sampled yourself: N x `data.base64_img` events with
      `multi_image: false` PLUS a `query_metadata` block carrying the
      video-metadata triple (`raw_fps`, `frames_indices`,
      `total_num_frames`). Without query_metadata this shape fails with
      400 query_failed. Example 3 below demonstrates it.

This is the capability that distinguishes C 2.6 from the C 2.4 / 2.5 text
checkpoints, which accept an .mp4 but ignore the frames ("I can't see
videos").

Note: `multi_image: true` is NOT a video knob — it switches the model to
multi-image mode (attached images treated as independent images, not video
frames). See image_query.py for that.

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
import base64
import sys
import time
from pathlib import Path

import requests

from _common import MODEL, banner, client, extract_text, query, upload_file

ASSETS = Path(__file__).parent / "sample_assets"
DEFAULT_VIDEO = ASSETS / "1_pass_2_pass_3_pass_B.mp4"
SAMPLE_FRAMES = [ASSETS / "assembly_before.png", ASSETS / "assembly_after.png"]

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


def example_frame_list(frame_paths: list[Path]) -> None:
    """Client-sampled frames as ONE video: base64 events + query_metadata."""
    banner(
        "3. Video via client-sampled frames (base64 events + query_metadata)\n"
        f"   {', '.join(p.name for p in frame_paths)}"
    )
    events = [
        {
            "type": "data.base64_img",
            "event_data": {"contents": base64.b64encode(p.read_bytes()).decode("ascii")},
        }
        for p in frame_paths
    ]
    n = len(events)
    text, _, ms = query(
        user_query=(
            "This is a short video of an assembly task. Describe the sequence "
            "of actions in order, in two sentences."
        ),
        instruction_prompt="Refer to the frames as one continuous video.",
        multi_image=False,  # false = the images are frames of ONE video
        events=events,
        # Required for the frame-list video path; omitting it → 400 query_failed.
        query_metadata={
            "raw_fps": 1.0,
            "frames_indices": list(range(n)),
            "total_num_frames": n,
        },
        max_new_tokens=150,
    )
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
    if all(p.exists() for p in SAMPLE_FRAMES):
        example_frame_list(SAMPLE_FRAMES)


if __name__ == "__main__":
    main()
