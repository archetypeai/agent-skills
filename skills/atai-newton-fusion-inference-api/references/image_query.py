"""
Image examples on the Inference API: Newton-C 2.6 (default) or Newton-Fusion 1.0.

Images go inline, as data URIs, on the OpenAI-compatible endpoints:

  1. /inference/chat/completions — an `image_url` part.
  2. /inference/responses — an `input_image` part, here with a JSON-extraction
     prompt.
  3. Several independent images in one call (before/after): several
     `image_url` parts in one chat message.

/inference/query refuses images in every form on production (a
`data.base64_img` event, `base64:<image>` in `file_ids`, an uploaded file by
`fil_` uid or by filename, `multi_image: true`), each with a bare 400. To
send an uploaded image, or one at full resolution, use the Direct Query API
skill (atai-newton-fusion-model) instead.

Inline images count toward the request body, which is capped at 2 MiB. Every
image is resized to 224 x 224 before the model sees it, so a large image
gains nothing: crop to the region you care about instead.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY and ATAI_API_ENDPOINT
    python image_query.py                       # Newton-C 2.6
    python image_query.py --model fusion        # Newton-Fusion 1.0
    python image_query.py /path/to/your.jpg     # your image in examples 1-2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archetypeai.api_client import ArchetypeAI

from _common import banner, chat, data_uri, make_client, model_arg, resolve_model, respond

# Shared with the Direct Query skill (atai-newton-fusion-model) rather than copied.
ASSETS = Path(__file__).resolve().parent.parent.parent / "atai-newton-fusion-model" / "references" / "sample_assets"
DEFAULT_IMAGE = ASSETS / "wind-turbines-small.jpg"
BEFORE_IMAGE = ASSETS / "assembly_before.png"
AFTER_IMAGE = ASSETS / "assembly_after.png"

ANALYST = "You are a careful visual analyst."


def example_chat_inline(client: ArchetypeAI, image_path: Path, **model_kw) -> None:
    banner(f"1. /inference/chat/completions with an inline image_url\n   {image_path.name}")
    text, _, elapsed_ms = chat(
        client,
        [
            {"type": "image_url", "image_url": {"url": data_uri(image_path)}},
            {
                "type": "text",
                "text": (
                    "Describe this image in three short bullet points. Focus on what "
                    "is visually present, not what it might mean."
                ),
            },
        ],
        system_prompt=f"{ANALYST} Reply with exactly three bullet points starting with '- '. No preamble.",
        max_tokens=300,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_responses_json(client: ArchetypeAI, image_path: Path, **model_kw) -> None:
    banner(f"2. /inference/responses with input_image — structured extraction\n   {image_path.name}")
    text, _, elapsed_ms = respond(
        client,
        [
            {"type": "input_image", "image_url": data_uri(image_path)},
            {
                "type": "input_text",
                "text": "Extract the dominant visual properties of this image. Respond with ONLY the JSON object, no fences.",
            },
        ],
        instructions=(
            "You output a single JSON object with this shape:\n"
            '{"primary_subject": "<short label>", '
            '"colors": ["<color1>", "<color2>", "<color3>"], '
            '"text_visible": <true|false>, '
            '"composition": "<one short sentence>"}\n'
            "Output only the JSON object."
        ),
        max_output_tokens=300,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")
    try:
        print(f"Parsed JSON: primary_subject={json.loads(text).get('primary_subject')!r}\n")
    except json.JSONDecodeError as decode_error:
        print(f"WARN: model output did not parse as JSON: {decode_error}\n")


def example_before_after(client: ArchetypeAI, before_path: Path, after_path: Path, **model_kw) -> None:
    banner(
        "3. Two independent images in one chat message (before/after)\n"
        f"   {before_path.name} + {after_path.name}"
    )
    text, _, elapsed_ms = chat(
        client,
        [
            {"type": "image_url", "image_url": {"url": data_uri(before_path)}},
            {"type": "image_url", "image_url": {"url": data_uri(after_path)}},
            {
                "type": "text",
                "text": (
                    "Image 1 is a workbench BEFORE an assembly task; image 2 is the "
                    "same bench DURING/AFTER. What has changed, and what is the "
                    "worker doing in image 2?"
                ),
            },
        ],
        system_prompt=(
            "You compare two independent photos of a workbench. Answer in two short "
            "paragraphs: (1) what changed, (2) what action is in progress in the second image."
        ),
        max_tokens=400,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help=f"Image for examples 1-2. Defaults to {DEFAULT_IMAGE.name}.",
    )
    model_arg(parser)
    args = parser.parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")
    model_kw = {"model": resolve_model(args.model), "reasoning": args.reasoning}

    client = make_client()
    print(f"model={model_kw['model']} reasoning={args.reasoning}\n")
    example_chat_inline(client, image_path, **model_kw)
    example_responses_json(client, image_path, **model_kw)
    example_before_after(client, BEFORE_IMAGE, AFTER_IMAGE, **model_kw)


if __name__ == "__main__":
    main()
