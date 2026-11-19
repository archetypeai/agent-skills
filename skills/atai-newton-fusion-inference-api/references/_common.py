"""
Shared helpers for the Inference API example scripts, built on the official
Archetype AI python client (https://github.com/archetypeai/python-client).

Two models, three APIs, all under `{root}/api/inference`:

  * `newton-c-2.6`      — Newton-C 2.6, 8B dense. The default.
  * `newton-fusion-1.0` — Newton-Fusion 1.0, 35B MoE (~3B active per token),
                          a reasoning model.

  * `POST /inference/query`            — Direct Query shape (`query`, `file_ids`)
  * `POST /inference/chat/completions` — OpenAI Chat Completions shape
  * `POST /inference/responses`        — OpenAI Responses shape

Each script creates one client via `make_client()` and passes it as the first
argument to the helpers.

Credential lookup order (first hit wins):
  1. ATAI_API_KEY / ATAI_API_ENDPOINT env vars. BOTH are required — there
     is no default endpoint, so a wrong-endpoint mistake fails loudly.
  2. python-dotenv `find_dotenv()` walk — starts from cwd and walks up.
  3. Sibling .env file next to this _common.py (i.e. references/.env).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from archetypeai.api_client import ArchetypeAI

C26 = "newton-c-2.6"
FUSION = "newton-fusion-1.0"
DEFAULT_MODEL = C26
MODELS = {"c26": C26, "fusion": FUSION}

# Fusion reasoning. `/no_think` at the end of the user's text keeps the
# <think> block empty, but ONLY when the request carries an image or video;
# on text-only requests Fusion reasons anyway. `/think` does not force
# reasoning, so reasoning mode sends no switch. Whenever Fusion reasons it can
# outrun the budget: on a short text question it reasoned for ~7-9k characters
# (2048 tokens ran out), and on a dense video it looped counting objects. A
# mild repetition penalty fixed both in every test. Even so, a 14-line log
# took ~9k characters of reasoning (~37 s), so reasoning gets 4096 tokens.
# The penalty only takes effect on /inference/query: Chat Completions and
# Responses accept it and ignore it (measured on production, 2026-11-15).
NO_THINK = " /no_think"
REASONING_PARAMS = {"repetition_penalty": 1.1}
REASONING_MIN_TOKENS = 4096

# Media go inline (base64 / data URIs) and count toward the request body, which
# is capped at 2 MiB: a 1.4 MiB body worked, 2.1 MiB hung with no answer and
# 3.2 MiB got 413 (production, 2026-11-15). Uploaded files can't be used:
# /inference/query refuses `fil_` uids and filenames alike on production.
INLINE_LIMIT_BYTES = 1_800_000
# Whole request body: 2 MiB is the server's cap (2.1 MiB hung, 3.2 MiB got 413).
BODY_LIMIT_BYTES = 2_000_000


def _try_load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except ImportError:
        return
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
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
            "ATAI_API_ENDPOINT is not set. Export it or add it to your .env — "
            "there is no default."
        )
    api_endpoint = re.sub(r"/v[0-9]+(\.[0-9]+)*$", "", api_endpoint.rstrip("/"))
    api_endpoint = re.sub(r"/api(/lens|/inference)?$", "", api_endpoint) + "/api"
    # The first call after a model replica has been idle can take ~40 s.
    return ArchetypeAI(api_key, api_endpoint=api_endpoint, request_timeout_sec=180)


def resolve_model(name: str) -> str:
    """Accept a short name (`c26`, `fusion`) or a full model alias."""
    return MODELS.get(name, name)


def read_base64(local_path: str | Path) -> str:
    """Base64 of a local file, refusing anything too big to send inline."""
    data = Path(local_path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    if len(encoded) > INLINE_LIMIT_BYTES:
        sys.exit(
            f"{Path(local_path).name} is {len(encoded) / 1e6:.1f} MB as base64; inline "
            f"request bodies over 2 MiB are refused (413) or hang, and uploads aren't "
            "accepted. Shrink it first (e.g. ffmpeg -i in.mp4 -vf \"scale=448:-2,fps=4\" "
            "-c:v libx264 -crf 28 -an small.mp4), or use the Direct Query API skill "
            "(atai-newton-fusion-model), which takes uploaded files."
        )
    return encoded


def data_uri(local_path: str | Path) -> str:
    """`data:<mime>;base64,...` for chat/completions and responses media parts."""
    mime = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{read_base64(local_path)}"


def split_thinking(text: str) -> tuple[str, str]:
    """Return (answer, thinking) from a Fusion reply; C 2.6 has no think block.

    Raises RuntimeError when the think block never closed: the token budget
    ran out mid-reasoning and there is no answer.
    """
    match = re.match(r"\s*<think>([\s\S]*?)(</think>\s*|$)", text)
    if not match:
        # The chat template may already have opened the block: reasoning, then </think>.
        thinking, closed, answer = text.partition("</think>")
        return (answer.strip(), thinking.strip()) if closed else (text.strip(), "")
    if not match.group(2):
        raise RuntimeError(
            "Fusion ran out of tokens while reasoning and gave no answer. "
            "Raise the token budget or add /no_think (works with media only)."
        )
    return text[match.end():].strip(), match.group(1).strip()


def _fusion_text(text: str, model: str, reasoning: bool) -> str:
    return text if model != FUSION or reasoning else text + NO_THINK


def _fusion_params(model: str, reasoning: bool, max_tokens: int, has_media: bool) -> tuple[int, dict[str, Any]]:
    """Fusion reasons in reasoning mode AND on any text-only request (where
    /no_think is ignored), so both get a budget big enough to finish."""
    if model != FUSION:
        return max_tokens, {}
    if reasoning or not has_media:
        return max(max_tokens, REASONING_MIN_TOKENS), dict(REASONING_PARAMS)
    return max_tokens, {}


def _post(client: ArchetypeAI, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    data = json.dumps(body)
    if len(data) > BODY_LIMIT_BYTES:
        raise ValueError(
            f"Request body is {len(data) / 1e6:.1f} MB; the Inference API caps it at 2 MiB "
            "(bigger bodies hang or get 413). Send fewer or smaller media, or use the Direct "
            "Query API skill (atai-newton-fusion-model), which takes uploaded files."
        )
    start_time = time.time()
    payload = client.requests_post(
        f"{client.api_endpoint}/inference/{path}",
        data_payload=data,
        additional_headers={"Content-Type": "application/json"},
    )
    return payload, int((time.time() - start_time) * 1000)


def query(
    client: ArchetypeAI,
    user_query: str,
    *,
    model: str = DEFAULT_MODEL,
    system_prompt: str = "",
    file_ids: list[str] | None = None,
    max_new_tokens: int = 512,
    reasoning: bool = False,
    **params: Any,
) -> tuple[str, dict[str, Any], int]:
    """
    One `POST /inference/query`. Returns (answer, raw_json, elapsed_ms).

    The system turn goes in `system_prompt`; `instruction_prompt` is only
    appended to the query text. `file_ids` takes a `base64:<mp4>` inline clip;
    images and uploaded files are refused here (send images to chat()/respond()).
    Extra sampling knobs (temperature,
    top_p, top_k, do_sample, repetition_penalty) pass through `**params`.
    """
    max_new_tokens, extra = _fusion_params(model, reasoning, max_new_tokens, bool(file_ids))
    body: dict[str, Any] = {
        "model": model,
        "query": _fusion_text(user_query, model, reasoning),
        "file_ids": file_ids or [],
        "max_new_tokens": max_new_tokens,
        **extra,
        **params,
    }
    if system_prompt:
        body["system_prompt"] = system_prompt
    payload, elapsed_ms = _post(client, "query", body)
    answer, _ = split_thinking(extract_text(payload))
    return answer, payload, elapsed_ms


def chat(
    client: ArchetypeAI,
    content: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    system_prompt: str = "",
    max_tokens: int = 512,
    reasoning: bool = False,
    **params: Any,
) -> tuple[str, dict[str, Any], int]:
    """One `POST /inference/chat/completions` with a single user turn.

    `content` is a list of OpenAI parts: `{"type": "text", ...}`,
    `{"type": "image_url", "image_url": {"url": data_uri(...)}}`,
    `{"type": "video_url", "video_url": {"url": data_uri(...)}}`.
    """
    max_tokens, extra = _fusion_params(model, reasoning, max_tokens, any(p.get("type") != "text" for p in content))
    content = [
        {**part, "text": _fusion_text(part["text"], model, reasoning)} if part.get("type") == "text" else part
        for part in content
    ]
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, **extra, **params}
    payload, elapsed_ms = _post(client, "chat/completions", body)
    answer, _ = split_thinking(extract_text(payload))
    return answer, payload, elapsed_ms


def respond(
    client: ArchetypeAI,
    content: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    instructions: str = "",
    max_output_tokens: int = 512,
    reasoning: bool = False,
    **params: Any,
) -> tuple[str, dict[str, Any], int]:
    """One `POST /inference/responses` with a single user turn.

    `content` parts: `{"type": "input_text", ...}`,
    `{"type": "input_image", "image_url": data_uri(...)}`,
    `{"type": "input_video", "video_url": data_uri(...)}`.
    """
    max_output_tokens, extra = _fusion_params(
        model, reasoning, max_output_tokens, any(p.get("type") != "input_text" for p in content)
    )
    content = [
        {**part, "text": _fusion_text(part["text"], model, reasoning)} if part.get("type") == "input_text" else part
        for part in content
    ]
    body: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_output_tokens,
        **extra,
        **params,
    }
    if instructions:
        body["instructions"] = instructions
    payload, elapsed_ms = _post(client, "responses", body)
    answer, _ = split_thinking(extract_text(payload))
    return answer, payload, elapsed_ms


def extract_text(payload: dict[str, Any]) -> str:
    """Model text from any of the three response shapes."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        return (choices[0].get("message") or {}).get("content") or ""
    output = payload.get("output")
    if isinstance(output, list):
        return "".join(
            part.get("text", "")
            for item in output
            for part in (item.get("content") or [])
            if isinstance(part, dict)
        )
    response_field = payload.get("response")
    if isinstance(response_field, dict):
        if response_field.get("success") is False:
            raise RuntimeError(f"Model returned success=false: {json.dumps(response_field)[:300]}")
        inner = response_field.get("response")
        if isinstance(inner, list) and inner:
            return inner[0] or ""
        if isinstance(inner, str):
            return inner
    return ""


def model_arg(parser) -> None:
    """Shared --model / --reasoning flags for the example scripts."""
    parser.add_argument(
        "--model",
        default="c26",
        help="c26 (default), fusion, or a full model alias.",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Fusion only: let the model reason before answering (slower).",
    )


def banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
