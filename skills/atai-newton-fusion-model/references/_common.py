"""
Shared helpers for the three /query example scripts.

Keeps each per-modality example single-purpose and readable while still
deduplicating the auth / upload / extract-response code.

Credential lookup order (first hit wins):
  1. ATAI_API_KEY env var.
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

import requests

MODEL = "Newton::c2_6_8b_fp8_260424d7a55d5e"
DEFAULT_ENDPOINT = "https://api.u1.archetypeai.app/v0.5"


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


def client() -> tuple[str, dict[str, str]]:
    """Resolve credentials + endpoint and return (endpoint, headers)."""
    _try_load_dotenv()
    api_key = os.environ.get("ATAI_API_KEY")
    if not api_key:
        sys.exit(
            "ATAI_API_KEY is not set. Either export it, or copy .env.example "
            "to .env (in the cwd or alongside this file) and fill it in."
        )
    endpoint = os.environ.get("ATAI_API_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
    if not endpoint.endswith("/v0.5"):
        endpoint = endpoint + "/v0.5"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    return endpoint, headers


def upload_file(local_path: str | Path, timeout: int = 300) -> str:
    """POST a file to /v0.5/files. Returns the file_id (== filename)."""
    endpoint, headers = client()
    upload_headers = {"Authorization": headers["Authorization"]}  # no Content-Type
    path = Path(local_path)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    with path.open("rb") as f:
        r = requests.post(
            f"{endpoint}/files",
            headers=upload_headers,
            files={"file": (path.name, f)},
            timeout=timeout,
        )
    if not r.ok:
        sys.exit(f"Upload failed [{r.status_code}]: {r.text[:400]}")
    data = r.json()
    file_id = data.get("file_id") or data.get("id")
    if not file_id:
        sys.exit(f"No file_id in upload response: {json.dumps(data)[:400]}")
    return file_id


def query(
    user_query: str,
    *,
    instruction_prompt: str = "",
    file_ids: list[str] | None = None,
    max_new_tokens: int = 512,
    sanitize: bool = False,
    multi_image: bool = False,
    timeout: int = 600,
) -> tuple[str, dict[str, Any], int]:
    """
    Single /query call. Returns (text, raw_json, elapsed_ms).

    The system turn goes in `instruction_prompt`. C 2.6 honors only this
    field; the legacy `system_prompt` field is inert on this checkpoint
    (verified: a directive sent in `system_prompt` alone is ignored, while
    the same directive in `instruction_prompt` is obeyed), so we don't send
    it.

    `multi_image=True` is required when attaching more than one image — it
    puts the model in multi-image mode (each image is independent, NOT video
    frames). Without it, a multi-image request fails with 400 query_failed.
    """
    endpoint, headers = client()
    body = {
        "query": user_query,
        "instruction_prompt": instruction_prompt,
        "file_ids": file_ids or [],
        "model": MODEL,
        "max_new_tokens": max_new_tokens,
        "sanitize": sanitize,
    }
    if multi_image:
        body["multi_image"] = True
    t0 = time.time()
    r = requests.post(f"{endpoint}/query", headers=headers, json=body, timeout=timeout)
    elapsed_ms = int((time.time() - t0) * 1000)
    if not r.ok:
        sys.exit(f"Query failed [{r.status_code}]: {r.text[:400]}")
    payload = r.json()
    return extract_text(payload), payload, elapsed_ms


def extract_text(payload: dict[str, Any]) -> str:
    """Walk the documented response shape: response.response[0]."""
    resp = payload.get("response")
    if isinstance(resp, dict):
        inner = resp.get("response")
        if isinstance(inner, list) and inner:
            return inner[0] or ""
        if isinstance(inner, str):
            return inner
    if isinstance(resp, list) and resp:
        return resp[0] or ""
    if isinstance(resp, str):
        return resp
    if isinstance(payload.get("text"), str):
        return payload["text"]
    return ""


def banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
