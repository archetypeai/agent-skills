"""
Image example for Newton C 2.6 fusion model via /query (prod).

Two paths to attach an image to a /query call:

  (a) Upload via the official files API, then reference the filename in
      `file_ids`. Best when the same image is reused across multiple queries.

  (b) Inline base64 in a `data.base64_img` event in the /query body.
      Best for one-shot queries where the upload roundtrip is wasted.

Both paths reach the same fusion model and produce comparable description
quality.

Attaching MORE than one image requires choosing a mode (example 4 here uses
the first):

  * `multi_image: true` — multi-image mode: each attachment is an
    independent image (before/after, multi-view). Max 16 images.
  * `multi_image: false` + a `query_metadata` block — the images are frames
    of ONE video; see video_query.py. Without query_metadata, a multi-image
    request fails with 400 query_failed.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY and ATAI_API_ENDPOINT
    python image_query.py
    # or point at your own image:
    python image_query.py /path/to/your.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from archetypeai import utils
from archetypeai.api_client import ArchetypeAI

from _common import banner, make_client, query, upload_file

ASSETS = Path(__file__).parent / "sample_assets"
DEFAULT_IMAGE = ASSETS / "wind-turbines.png"
BEFORE_IMAGE = ASSETS / "assembly_before.png"
AFTER_IMAGE = ASSETS / "assembly_after.png"


def example_file_upload(client: ArchetypeAI, image_path: Path) -> None:
    banner(f"1. Image via file_ids (upload, then reference)\n   {image_path.name}")
    file_id = upload_file(client, image_path)
    print(f"Uploaded → file_id={file_id}")

    text, _, elapsed_ms = query(
        client,
        user_query=(
            "Describe this image in three short bullet points. Focus on what "
            "is visually present, not what it might mean."
        ),
        instruction_prompt=(
            "You are a careful visual analyst. Reply with exactly three "
            "bullet points starting with '- '. No preamble."
        ),
        file_ids=[file_id],
        max_new_tokens=300,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_inline_base64(client: ArchetypeAI, image_path: Path) -> None:
    banner(f"2. Image via inline base64 event\n   {image_path.name}")
    encoded_image = utils.base64_encode(str(image_path))

    text, _, elapsed_ms = query(
        client,
        user_query="What is in this image? One sentence.",
        instruction_prompt="Reply with exactly one sentence. No preamble.",
        max_new_tokens=200,
        events=[
            {
                "type": "data.base64_img",
                "event_data": {"contents": encoded_image},
            }
        ],
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_structured_extraction(client: ArchetypeAI, image_path: Path) -> None:
    banner(f"3. Structured extraction from an image\n   {image_path.name}")
    file_id = upload_file(client, image_path)

    text, _, elapsed_ms = query(
        client,
        user_query=(
            "Extract the dominant visual properties of this image as JSON. "
            "Respond with ONLY the JSON object, no fences."
        ),
        instruction_prompt=(
            "You output a single JSON object with this shape:\n"
            '{"primary_subject": "<short label>", '
            '"colors": ["<color1>", "<color2>", "<color3>"], '
            '"text_visible": <true|false>, '
            '"composition": "<one short sentence>"}\n'
            "Output only the JSON object."
        ),
        file_ids=[file_id],
        max_new_tokens=300,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_multi_image(client: ArchetypeAI, before_path: Path, after_path: Path) -> None:
    """Two independent images in one call: requires `multi_image: true`."""
    banner(
        "4. Multiple images via multi_image: true (before/after comparison)\n"
        f"   {before_path.name} + {after_path.name}"
    )
    file_ids = [upload_file(client, before_path), upload_file(client, after_path)]
    print(f"Uploaded → file_ids={file_ids}")

    text, _, elapsed_ms = query(
        client,
        user_query=(
            "Image 1 is a workbench BEFORE an assembly task; image 2 is the "
            "same bench DURING/AFTER. Compare them: what has changed, and "
            "what is the worker doing in image 2?"
        ),
        instruction_prompt=(
            "You compare two independent photos of a workbench. Answer in "
            "two short paragraphs: (1) what changed between the images, "
            "(2) what action is in progress in the second image."
        ),
        file_ids=file_ids,
        max_new_tokens=400,
        multi_image=True,  # independent images; without it the list means "video frames" and needs query_metadata
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help=f"Path to an image file. Defaults to {DEFAULT_IMAGE.name}.",
    )
    args = parser.parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    client = make_client()
    example_file_upload(client, image_path)
    example_inline_base64(client, image_path)
    example_structured_extraction(client, image_path)
    if BEFORE_IMAGE.exists() and AFTER_IMAGE.exists():
        example_multi_image(client, BEFORE_IMAGE, AFTER_IMAGE)


if __name__ == "__main__":
    main()
