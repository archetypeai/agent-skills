"""
Image examples on Model Serving: Newton-C 2.6 (default) or Newton-Fusion 1.0.

Three ways to attach an image, one per API:

  1. /inference/query — upload via the files API, then put the `fil_...` uid
     in `file_ids`. Best when the same image is reused across queries, and
     the only way for images too big to send inline.
  2. /inference/chat/completions — an `image_url` part with a data URI.
  3. /inference/responses — an `input_image` part with a data URI, here with
     a JSON-extraction prompt.
  4. Several independent images in one call (before/after): put several
     `image_url` parts in one chat message. `multi_image` on /query is refused.

Inline images count toward the request body: keep them under ~1.5 MB as
base64 (a 2.6 MB body got 413). The small JPEG in sample_assets is for the
inline examples; the full PNG goes through the upload path.

Refused on Model Serving (bare 400, no reason given): `data.base64_img`
events, `base64:<image>` in file_ids, the upload's filename in file_ids,
`multi_image: true`, and a `fil_` uid as a chat `image_url`.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY and ATAI_API_ENDPOINT
    python image_query.py                       # Newton-C 2.6
    python image_query.py --model fusion        # Newton-Fusion 1.0
    python image_query.py /path/to/your.png     # upload path uses your image
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archetypeai.api_client import ArchetypeAI

from _common import banner, chat, data_uri, make_client, model_arg, query, resolve_model, respond, upload_file

ASSETS = Path(__file__).parent / "sample_assets"
DEFAULT_IMAGE = ASSETS / "wind-turbines.png"
SMALL_IMAGE = ASSETS / "wind-turbines-small.jpg"
BEFORE_IMAGE = ASSETS / "assembly_before.png"
AFTER_IMAGE = ASSETS / "assembly_after.png"

ANALYST = "You are a careful visual analyst."


def example_upload_query(client: ArchetypeAI, image_path: Path, **model_kw) -> None:
    banner(f"1. /inference/query with an uploaded image (fil_ uid)\n   {image_path.name}")
    file_uid = upload_file(client, image_path)
    print(f"Uploaded → file_uid={file_uid}")
    text, _, elapsed_ms = query(
        client,
        user_query=(
            "Describe this image in three short bullet points. Focus on what "
            "is visually present, not what it might mean."
        ),
        system_prompt=f"{ANALYST} Reply with exactly three bullet points starting with '- '. No preamble.",
        file_ids=[file_uid],
        max_new_tokens=300,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_chat_inline(client: ArchetypeAI, image_path: Path, **model_kw) -> None:
    banner(f"2. /inference/chat/completions with an inline image_url\n   {image_path.name}")
    text, _, elapsed_ms = chat(
        client,
        [
            {"type": "image_url", "image_url": {"url": data_uri(image_path)}},
            {"type": "text", "text": "What is in this image? One sentence."},
        ],
        system_prompt=f"{ANALYST} Reply with exactly one sentence. No preamble.",
        max_tokens=200,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_responses_json(client: ArchetypeAI, image_path: Path, **model_kw) -> None:
    banner(f"3. /inference/responses with input_image — structured extraction\n   {image_path.name}")
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
        "4. Two independent images in one chat message (before/after)\n"
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
        help=f"Image for the upload example. Defaults to {DEFAULT_IMAGE.name}.",
    )
    model_arg(parser)
    args = parser.parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")
    model_kw = {"model": resolve_model(args.model), "reasoning": args.reasoning}

    client = make_client()
    print(f"model={model_kw['model']} reasoning={args.reasoning}\n")
    example_upload_query(client, image_path, **model_kw)
    example_chat_inline(client, SMALL_IMAGE, **model_kw)
    example_responses_json(client, SMALL_IMAGE, **model_kw)
    example_before_after(client, BEFORE_IMAGE, AFTER_IMAGE, **model_kw)


if __name__ == "__main__":
    main()
