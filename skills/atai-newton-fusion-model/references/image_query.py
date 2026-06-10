"""
Image example for Newton C 2.6 fusion model via /query (prod).

Two paths to attach an image to a /query call:

  (a) Upload via /v0.5/files, then reference the filename in `file_ids`.
      Best when the same image is reused across multiple queries.

  (b) Inline base64 in a `data.base64_img` event in the /query body.
      Best for one-shot queries where the upload roundtrip is wasted.

Both paths reach the same fusion model and produce comparable description
quality.

Attaching MORE than one image additionally requires `multi_image: true`
(example 4). That flag puts the model in multi-image mode: each attachment
is an independent image (before/after, multi-view) — NOT frames of a video.
Without the flag, a multi-image request fails with 400 query_failed. For
video, see video_query.py (.mp4 + max_frames).

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY
    python image_query.py
    # or point at your own image:
    python image_query.py /path/to/your.png
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import sys
import time
from pathlib import Path

import requests

from _common import (
    DEFAULT_ENDPOINT,
    MODEL,
    banner,
    client,
    extract_text,
    query,
    upload_file,
)

ASSETS = Path(__file__).parent / "sample_assets"
DEFAULT_IMAGE = ASSETS / "wind-turbines.png"
BEFORE_IMAGE = ASSETS / "assembly_before.png"
AFTER_IMAGE = ASSETS / "assembly_after.png"


def example_file_upload(image_path: Path) -> None:
    banner(f"1. Image via file_ids (upload, then reference)\n   {image_path.name}")
    file_id = upload_file(image_path)
    print(f"Uploaded → file_id={file_id}")

    text, _, ms = query(
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
    print(f"[{ms} ms]\n{text}\n")


def example_inline_base64(image_path: Path) -> None:
    banner(f"2. Image via inline base64 event\n   {image_path.name}")
    endpoint, headers = client()

    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"

    body = {
        "query": "What is in this image? One sentence.",
        "instruction_prompt": "Reply with exactly one sentence. No preamble.",
        "file_ids": [],
        "model": MODEL,
        "max_new_tokens": 200,
        "sanitize": False,
        "events": [
            {
                "type": "data.base64_img",
                "event_data": {"contents": b64, "mime_type": mime},
            }
        ],
    }

    t0 = time.time()
    r = requests.post(f"{endpoint}/query", headers=headers, json=body, timeout=120)
    ms = int((time.time() - t0) * 1000)
    if not r.ok:
        sys.exit(f"Query failed [{r.status_code}]: {r.text[:400]}")
    text = extract_text(r.json())
    print(f"[{ms} ms]\n{text}\n")


def example_structured_extraction(image_path: Path) -> None:
    banner(f"3. Structured extraction from an image\n   {image_path.name}")
    file_id = upload_file(image_path)

    text, _, ms = query(
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
    print(f"[{ms} ms]\n{text}\n")


def example_multi_image(before_path: Path, after_path: Path) -> None:
    """Two independent images in one call: requires `multi_image: true`."""
    banner(
        "4. Multiple images via multi_image: true (before/after comparison)\n"
        f"   {before_path.name} + {after_path.name}"
    )
    file_ids = [upload_file(before_path), upload_file(after_path)]
    print(f"Uploaded → file_ids={file_ids}")

    text, _, ms = query(
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
        multi_image=True,  # required for >1 image; omitting it → 400 query_failed
    )
    print(f"[{ms} ms]\n{text}\n")


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

    example_file_upload(image_path)
    example_inline_base64(image_path)
    example_structured_extraction(image_path)
    if BEFORE_IMAGE.exists() and AFTER_IMAGE.exists():
        example_multi_image(BEFORE_IMAGE, AFTER_IMAGE)


if __name__ == "__main__":
    main()
