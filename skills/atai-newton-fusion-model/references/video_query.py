"""
Video examples on Model Serving: Newton-C 2.6 (default) or Newton-Fusion 1.0.

Video goes in as an encoded clip (.mp4); the model's video reader samples up
to 16 frames at 224 px. That budget is fixed: `max_frames` in the request is
refused, and a session cannot change it either. Four ways in:

  1. /inference/query — upload via the files API, put the `fil_...` uid in
     `file_ids`. The only way for clips too big to send inline (the full
     7.8 MB sample works this way).
  2. /inference/query — the clip inline: `file_ids: ["base64:<mp4 bytes>"]`.
  3. /inference/chat/completions — a `video_url` part with a data URI.
  4. /inference/responses — an `input_video` part with a data URI.

Inline clips count toward the request body: keep them under ~1.5 MB as
base64 (a ~10 MB body hung). The small re-encode in sample_assets is for the
inline examples.

No frame lists: `data.base64_img` events, `multi_image`, and
`query_metadata` are gone. If you have frames, mux them into an .mp4 first
(e.g. `ffmpeg -framerate 2 -i f%03d.jpg -c:v libx264 -pix_fmt yuv420p clip.mp4`).

The demo uses a worker-assembly PASS/FAIL inspection prompt; ground truth for
the sample clip (named 1_pass_2_pass_3_pass) is all three steps PASS.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY and ATAI_API_ENDPOINT
    python video_query.py                       # Newton-C 2.6
    python video_query.py --model fusion        # Newton-Fusion 1.0
    python video_query.py /path/to/your.mp4     # upload path uses your clip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from archetypeai.api_client import ArchetypeAI

from _common import banner, chat, data_uri, make_client, model_arg, query, read_base64, resolve_model, respond, upload_file

ASSETS = Path(__file__).parent / "sample_assets"
DEFAULT_VIDEO = ASSETS / "1_pass_2_pass_3_pass_B.mp4"
SMALL_VIDEO = ASSETS / "1_pass_2_pass_3_pass_B_small.mp4"

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
DESCRIBE = "Describe what the person does in this video, in order. Three sentences."


def example_upload_query(client: ArchetypeAI, video_path: Path, **model_kw) -> None:
    banner(f"1. /inference/query with an uploaded clip (fil_ uid)\n   {video_path.name}")
    file_uid = upload_file(client, video_path)
    print(f"Uploaded → file_uid={file_uid}")
    text, _, elapsed_ms = query(
        client,
        user_query=QUERY,
        system_prompt=ASSEMBLY_PROMPT,
        file_ids=[file_uid],
        max_new_tokens=500,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_inline_query(client: ArchetypeAI, video_path: Path, **model_kw) -> None:
    banner(f"2. /inference/query with the clip inline (base64:)\n   {video_path.name}")
    text, _, elapsed_ms = query(
        client,
        user_query=DESCRIBE,
        file_ids=[f"base64:{read_base64(video_path)}"],
        max_new_tokens=300,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_chat_video(client: ArchetypeAI, video_path: Path, **model_kw) -> None:
    banner(f"3. /inference/chat/completions with a video_url part\n   {video_path.name}")
    text, _, elapsed_ms = chat(
        client,
        [
            {"type": "video_url", "video_url": {"url": data_uri(video_path)}},
            {"type": "text", "text": DESCRIBE},
        ],
        max_tokens=300,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_responses_video(client: ArchetypeAI, video_path: Path, **model_kw) -> None:
    banner(f"4. /inference/responses with an input_video part\n   {video_path.name}")
    text, _, elapsed_ms = respond(
        client,
        [
            {"type": "input_video", "video_url": data_uri(video_path)},
            {"type": "input_text", "text": DESCRIBE},
        ],
        max_output_tokens=300,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help=f"Clip for the upload example. Defaults to {DEFAULT_VIDEO.name}.",
    )
    model_arg(parser)
    args = parser.parse_args()
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")
    model_kw = {"model": resolve_model(args.model), "reasoning": args.reasoning}

    client = make_client()
    print(f"model={model_kw['model']} reasoning={args.reasoning}\n")
    example_upload_query(client, video_path, **model_kw)
    example_inline_query(client, SMALL_VIDEO, **model_kw)
    example_chat_video(client, SMALL_VIDEO, **model_kw)
    example_responses_video(client, SMALL_VIDEO, **model_kw)


if __name__ == "__main__":
    main()
