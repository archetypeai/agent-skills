"""
Video examples on the Inference API: Newton-C 2.6 (default) or Newton-Fusion 1.0.

Video goes in as an encoded clip (.mp4), inline. The model's video reader
samples up to 16 frames at 224 x 224. That budget is fixed: `max_frames` in
the request is refused, and a session cannot change it either. Three ways in:

  1. /inference/query — the clip inline: `file_ids: ["base64:<mp4 bytes>"]`,
     here with a worker-assembly PASS/FAIL inspection prompt.
  2. /inference/chat/completions — a `video_url` part with a data URI.
  3. /inference/responses — an `input_video` part with a data URI.

Inline clips count toward the request body, which is capped at 2 MiB: keep the
clip under about 1.4 MB. Uploaded files can't be used on production
(/inference/query refuses `fil_` uids and filenames). Because the model only
sees 16 frames at 224 px, shrinking a clip loses nothing:
    ffmpeg -i in.mp4 -vf "scale=448:-2,fps=4" -c:v libx264 -crf 28 -an small.mp4
For long or large clips you can't shrink or cut, or for more than 16 frames,
use the Direct Query API skill (atai-newton-fusion-model).

No frame lists: `data.base64_img` events, `multi_image`, and
`query_metadata` are refused. If you have frames, mux them into an .mp4 first
(e.g. `ffmpeg -framerate 2 -i f%03d.jpg -c:v libx264 -pix_fmt yuv420p clip.mp4`).

Ground truth for the sample clip (1_pass_2_pass_3_pass) is all three steps PASS.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY and ATAI_API_ENDPOINT
    python video_query.py                       # Newton-C 2.6
    python video_query.py --model fusion        # Newton-Fusion 1.0
    python video_query.py /path/to/your.mp4     # your clip (under ~1.4 MB)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from archetypeai.api_client import ArchetypeAI

from _common import banner, chat, data_uri, make_client, model_arg, query, read_base64, resolve_model, respond

# Shared with the Direct Query skill (atai-newton-fusion-model) rather than copied.
ASSETS = Path(__file__).resolve().parent.parent.parent / "atai-newton-fusion-model" / "references" / "sample_assets"
DEFAULT_VIDEO = ASSETS / "1_pass_2_pass_3_pass_B_small.mp4"

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


def example_inline_query(client: ArchetypeAI, video_path: Path, **model_kw) -> None:
    banner(f"1. /inference/query with the clip inline (base64:) — PASS/FAIL check\n   {video_path.name}")
    text, _, elapsed_ms = query(
        client,
        user_query=QUERY,
        system_prompt=ASSEMBLY_PROMPT,
        file_ids=[f"base64:{read_base64(video_path)}"],
        max_new_tokens=500,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_chat_video(client: ArchetypeAI, video_path: Path, **model_kw) -> None:
    banner(f"2. /inference/chat/completions with a video_url part\n   {video_path.name}")
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
    banner(f"3. /inference/responses with an input_video part\n   {video_path.name}")
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
        help=f"Clip for every example (under ~1.4 MB). Defaults to {DEFAULT_VIDEO.name}.",
    )
    model_arg(parser)
    args = parser.parse_args()
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")
    model_kw = {"model": resolve_model(args.model), "reasoning": args.reasoning}

    client = make_client()
    print(f"model={model_kw['model']} reasoning={args.reasoning}\n")
    example_inline_query(client, video_path, **model_kw)
    example_chat_video(client, video_path, **model_kw)
    example_responses_video(client, video_path, **model_kw)


if __name__ == "__main__":
    main()
