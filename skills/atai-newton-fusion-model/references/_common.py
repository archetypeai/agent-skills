"""
Shared helpers for the /query example scripts, built on the official
Archetype AI python client (https://github.com/archetypeai/python-client).

Each script creates one client via `make_client()` and passes it as the
first argument to `query()` / `upload_file()`.

Credential lookup order (first hit wins):
  1. ATAI_API_KEY / ATAI_API_ENDPOINT env vars. BOTH are required — there
     is no default endpoint, so a wrong-endpoint mistake fails loudly.
  2. python-dotenv `find_dotenv()` walk — starts from cwd and walks up.
     Catches .env at any ancestor (repo root, project root, etc.).
  3. Sibling .env file next to this _common.py (i.e. references/.env).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from archetypeai.api_client import ArchetypeAI

MODEL = "Newton::c2_6_8b_fp8_260424d7a55d5e"


def _try_load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except ImportError:
        return
    # Walk up from cwd to find any ancestor .env (repo root, etc.).
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
    # Also pick up references/.env if it exists, without overriding what we just loaded.
    sibling = Path(__file__).parent / ".env"
    if sibling.exists():
        load_dotenv(sibling, override=False)


def make_client() -> ArchetypeAI:
    """Resolve credentials and return an official ArchetypeAI client.

    Both ATAI_API_KEY and ATAI_API_ENDPOINT are required. The endpoint is
    deliberately NOT defaulted — pointing at the wrong deployment should
    fail loudly here, not silently at query time.
    """
    _try_load_dotenv()
    api_key = os.environ.get("ATAI_API_KEY")
    if not api_key:
        sys.exit(
            "ATAI_API_KEY is not set. Either export it, or copy .env.example "
            "to .env (in the cwd or alongside this file) and fill it in."
        )
    api_endpoint = os.environ.get("ATAI_API_ENDPOINT")
    if not api_endpoint:
        sys.exit(
            "ATAI_API_ENDPOINT is not set. Export it (e.g. "
            "https://api.u1.archetypeai.app/v0.5) or add it to your .env — "
            "there is no default."
        )
    api_endpoint = api_endpoint.rstrip("/")
    if not api_endpoint.endswith("/v0.5"):
        api_endpoint = api_endpoint + "/v0.5"
    return ArchetypeAI(api_key, api_endpoint=api_endpoint)


def upload_file(client: ArchetypeAI, local_path: str | Path) -> str:
    """Upload a file via the official files API. Returns the file_id (== filename)."""
    path = Path(local_path)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    response_data = client.files.local.upload(str(path))
    file_id = response_data.get("file_id")
    if not file_id:
        sys.exit(f"No file_id in upload response: {json.dumps(response_data)[:400]}")
    return file_id


def query(
    client: ArchetypeAI,
    user_query: str,
    *,
    instruction_prompt: str = "",
    file_ids: list[str] | None = None,
    max_new_tokens: int = 512,
    max_frames: int | None = None,
    multi_image: bool | None = None,
    events: list[dict[str, Any]] | None = None,
    query_metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], int]:
    """
    Single /query call. Returns (text, raw_json, elapsed_ms).

    The system turn goes in `instruction_prompt`. C 2.6 honors only this
    field; the legacy `system_prompt` field is inert on this checkpoint
    (verified: a directive sent in `system_prompt` alone is ignored, while
    the same directive in `instruction_prompt` is obeyed), so we don't send
    it.

    Multiple images go one of two ways:
      * `multi_image=True` — multi-image mode: each image is independent.
      * `multi_image=False` + `query_metadata` (raw_fps / frames_indices /
        total_num_frames) — the images are frames of ONE video. Without the
        query_metadata triple this shape fails with 400 query_failed.
    """
    body: dict[str, Any] = {
        "query": user_query,
        "instruction_prompt": instruction_prompt,
        "file_ids": file_ids or [],
        "model": MODEL,
        "max_new_tokens": max_new_tokens,
    }
    if max_frames is not None:
        body["max_frames"] = max_frames
    if multi_image is not None:
        body["multi_image"] = multi_image
    if events:
        body["events"] = events
    if query_metadata:
        body["query_metadata"] = query_metadata

    start_time = time.time()
    payload = client.requests_post(
        f"{client.api_endpoint}/query",
        data_payload=json.dumps(body),
        additional_headers={"Content-Type": "application/json"},
    )
    elapsed_ms = int((time.time() - start_time) * 1000)
    return extract_text(payload), payload, elapsed_ms


def extract_text(payload: dict[str, Any]) -> str:
    """Walk the documented response shape: response.response[0]."""
    response_field = payload.get("response")
    if isinstance(response_field, dict):
        inner = response_field.get("response")
        if isinstance(inner, list) and inner:
            return inner[0] or ""
        if isinstance(inner, str):
            return inner
    if isinstance(response_field, list) and response_field:
        return response_field[0] or ""
    if isinstance(response_field, str):
        return response_field
    if isinstance(payload.get("text"), str):
        return payload["text"]
    return ""


def banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
