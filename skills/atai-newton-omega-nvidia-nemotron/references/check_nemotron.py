"""
Preflight for NVIDIA Nemotron on NVIDIA's hosted API — run this before wiring
Nemotron into an Omega pipeline.

It checks, in order:
  1. NVIDIA_API_KEY is set.
  2. The model is listed (GET /v1/models — needs no key, so a listing alone
     proves nothing about your key).
  3. A one-line chat call succeeds with reasoning off, and reports latency and
     reasoning tokens (should be 0).

A failure prints what the status usually means: 403 = wrong key type (NGC),
404 = model not available to your account, 410 = model retired.

    python check_nemotron.py
    python check_nemotron.py --model nvidia/nemotron-3-super-120b-a12b
    python check_nemotron.py --list          # just list Nemotron chat models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from _common import (
    NemotronError,
    banner,
    chat_body,
    load_dotenv_if_available,
    nemotron_model,
    nvidia_endpoint,
)

# Not chat models: embedding, reward, safety and document-parsing variants.
NON_CHAT_MARKERS = ("embed", "reward", "safety", "parse", "guard")


def list_nemotron_models() -> list[str]:
    with urllib.request.urlopen(f"{nvidia_endpoint()}/models", timeout=30) as response:
        data = json.load(response)
    ids = [m["id"] for m in data.get("data", []) if "nemotron" in m["id"].lower()]
    return sorted(i for i in ids if not any(marker in i.lower() for marker in NON_CHAT_MARKERS))


def ping(model: str) -> dict:
    body = chat_body("Reply with exactly: OK", "ping", max_tokens=16)
    body["model"] = model
    request = urllib.request.Request(
        f"{nvidia_endpoint()}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}", "Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise NemotronError(exc.code, exc.read().decode(errors="ignore")) from exc
    usage = data.get("usage") or {}
    return {
        "reply": (data["choices"][0]["message"].get("content") or "").strip(),
        "latency_s": round(time.time() - started, 2),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="model id (default: NEMOTRON_MODEL or the skill default)")
    parser.add_argument("--list", action="store_true", help="only list Nemotron chat models")
    args = parser.parse_args()
    load_dotenv_if_available()
    model = args.model or nemotron_model()

    banner("Nemotron chat models listed by the NVIDIA API")
    models = list_nemotron_models()
    for model_id in models:
        print(("→ " if model_id == model else "  ") + model_id)
    if args.list:
        return
    print()
    if model not in models:
        print(f"! {model} is not in the listing — it may be retired or misspelled.")

    banner(f"Calling {model}")
    if not os.environ.get("NVIDIA_API_KEY"):
        sys.exit("NVIDIA_API_KEY is not set. Create one at https://build.nvidia.com/settings/api-keys.")
    try:
        result = ping(model)
    except NemotronError as exc:
        sys.exit(str(exc))
    print(f"reply: {result['reply']!r}")
    print(f"latency: {result['latency_s']} s")
    print(f"reasoning tokens: {result['reasoning_tokens']} (0 = reasoning off, as intended)")
    print("\nOK — set NEMOTRON_MODEL to this id if it isn't the default.")


if __name__ == "__main__":
    main()
