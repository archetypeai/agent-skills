"""
Shared helpers for pairing the Omega encoder (Archetype AI Newton, /query) with
NVIDIA Nemotron (NVIDIA's hosted, OpenAI-compatible API).

Omega SENSES: it turns a sensor window into embeddings, and a client-side KNN
turns those into a verdict. Nemotron EXPLAINS: it reads the verdict plus plain
window statistics and writes operator-facing text. Nemotron never sees an
embedding — vectors mean nothing to a language model.

The Omega half mirrors `atai-newton-omega-model/references/_common.py` (one
/query per channel, fanned out). The Nemotron half uses only the standard
library, so there is no OpenAI SDK dependency.

Credential lookup (first hit wins): env vars, then a .env found by walking up
from cwd, then a .env next to this file.
  ATAI_API_KEY, ATAI_API_ENDPOINT   required for Omega (no default endpoint)
  NVIDIA_API_KEY                    required for Nemotron (nvapi-..., from
                                    https://build.nvidia.com/settings/api-keys)
  NEMOTRON_MODEL, NVIDIA_API_ENDPOINT   optional overrides
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

OMEGA_MODEL = "OmegaEncoder::omega_embeddings_1_4"
EMBED_DIM = 768
TIME_COLUMNS = {"timestamp", "time", "ts", "datetime", "date"}

NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NEMOTRON_TIMEOUT = 60

# What each NVIDIA status usually means in practice — the API's own messages are terse.
NVIDIA_ERROR_HINTS = {
    401: "NVIDIA_API_KEY was rejected. Create a key at https://build.nvidia.com/settings/api-keys.",
    403: (
        "Authorization failed. The usual cause is an NGC key (used to pull containers): it also "
        "starts with nvapi- but can't call hosted models. Create a key at "
        "https://build.nvidia.com/settings/api-keys."
    ),
    404: "This model isn't available to your account. Run check_nemotron.py to list models and pick another.",
    410: "This model has been retired. Pick a current one with check_nemotron.py and set NEMOTRON_MODEL.",
    429: "Rate limited (the free tier is about 40 requests/minute). Cache results and retry later.",
}


class NemotronError(RuntimeError):
    def __init__(self, status: int, detail: str):
        hint = NVIDIA_ERROR_HINTS.get(status, "")
        super().__init__(f"Nemotron request failed ({status}): {detail[:300]}" + (f"\n  → {hint}" if hint else ""))
        self.status = status


def load_dotenv_if_available() -> None:
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


# ---------------------------------------------------------------- Omega (/query)

def make_client():
    """Official ArchetypeAI client. Both ATAI_API_KEY and ATAI_API_ENDPOINT are required."""
    from archetypeai.api_client import ArchetypeAI

    load_dotenv_if_available()
    api_key = os.environ.get("ATAI_API_KEY")
    if not api_key:
        sys.exit("ATAI_API_KEY is not set. Export it or copy .env.example to .env and fill it in.")
    api_endpoint = os.environ.get("ATAI_API_ENDPOINT")
    if not api_endpoint:
        sys.exit("ATAI_API_ENDPOINT is not set (e.g. https://api.u1.archetypeai.app/v0.5) — there is no default.")
    api_endpoint = api_endpoint.rstrip("/")
    if not api_endpoint.endswith("/v0.5"):
        api_endpoint += "/v0.5"
    return ArchetypeAI(api_key, api_endpoint=api_endpoint)


def _embed_channel(client, channel_values: list[float]) -> list[float]:
    body = {
        "query": "",
        "model": OMEGA_MODEL,
        "normalize_input": False,
        "events": [{"type": "data.numeric_array", "event_data": {"contents": [channel_values]}}],
    }
    payload = client.requests_post(
        f"{client.api_endpoint}/query",
        data_payload=json.dumps(body),
        additional_headers={"Content-Type": "application/json"},
    )
    response = payload.get("response")
    vector = response.get("response") if isinstance(response, dict) else response
    if vector and isinstance(vector[0], list):
        vector = vector[0]
    return vector or []


def embed(client, window: list[list[float]], max_workers: int = 8) -> list[list[float]]:
    """Embed a channel-first `[channels x timesteps]` window: one /query per channel, in parallel.

    Pre-scale the window with a scaler fit once on your reference pool; this sends
    normalize_input=false so cross-window amplitude survives (see atai-newton-omega-model).
    """
    with ThreadPoolExecutor(max_workers=min(max_workers, len(window))) as executor:
        return list(executor.map(lambda channel: _embed_channel(client, channel), window))


def read_series(csv_path: str | Path, time_columns: set[str] = TIME_COLUMNS) -> tuple[list[str], list[list[float]]]:
    """Read a sensor CSV into (channel_names, channel-first series), dropping time columns."""
    with open(csv_path, newline="") as file_handle:
        reader = csv.reader(file_handle)
        header = next(reader)
        keep = [i for i, name in enumerate(header) if name.strip().lower() not in time_columns]
        rows = [[float(row[i]) for i in keep] for row in reader if row]
    if not rows:
        sys.exit(f"No rows in {csv_path}")
    names = [header[i] for i in keep]
    return names, [[row[c] for row in rows] for c in range(len(keep))]


def window_at(series: list[list[float]], start: int, size: int) -> list[list[float]]:
    return [channel[start:start + size] for channel in series]


# ---------------------------------------------------------------- window statistics

def channel_stats(values: list[float]) -> dict[str, float]:
    """Plain, unit-bearing statistics an LLM can reason about (unlike embeddings)."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    rms = math.sqrt(sum(v * v for v in values) / n)
    peak = max(abs(v) for v in values)
    kurtosis = (sum((v - mean) ** 4 for v in values) / n) / (var ** 2) if var > 0 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "crest_factor": round(peak / rms, 2) if rms > 0 else 0.0,
        "kurtosis": round(kurtosis, 2),
    }


def window_stats(names: list[str], window: list[list[float]]) -> dict[str, dict[str, float]]:
    return {name: channel_stats(channel) for name, channel in zip(names, window)}


# ---------------------------------------------------------------- Nemotron (NVIDIA hosted API)

def nemotron_model() -> str:
    return os.environ.get("NEMOTRON_MODEL") or NEMOTRON_MODEL


def nvidia_endpoint() -> str:
    return (os.environ.get("NVIDIA_API_ENDPOINT") or NVIDIA_ENDPOINT).rstrip("/")


def chat_body(system: str, user: str, *, max_tokens: int = 800, temperature: float = 0.2,
              reasoning: bool = False) -> dict[str, Any]:
    """Chat-completions body. Nemotron 3 turns reasoning off via the chat template —
    `/no_think` in the prompt is ignored."""
    return {
        "model": nemotron_model(),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": reasoning},
    }


def nemotron_chat(system: str, user: str, *, retries: int = 2, **kwargs) -> str:
    """One chat completion. Returns the reply text; raises NemotronError with a hint on failure."""
    load_dotenv_if_available()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("NVIDIA_API_KEY is not set. Create one at https://build.nvidia.com/settings/api-keys.")
    request = urllib.request.Request(
        f"{nvidia_endpoint()}/chat/completions",
        data=json.dumps(chat_body(system, user, **kwargs)).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=NEMOTRON_TIMEOUT) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            # Retry only transient failures; 4xx other than 429 won't change on retry.
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise NemotronError(exc.code, exc.read().decode(errors="ignore")) from exc
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return strip_reasoning(content)


def strip_reasoning(text: str) -> str:
    """Drop a <think>…</think> block if a model emits one despite reasoning being off."""
    while "<think>" in text and "</think>" in text:
        start, end = text.index("<think>"), text.index("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


def parse_json(text: str, kind: type = dict):
    """Extract the first JSON object (kind=dict) or array (kind=list) from a reply.

    Tolerates code fences and stray prose around it; raises ValueError if none parses.
    """
    open_char, close_char = ("{", "}") if kind is dict else ("[", "]")
    start, end = text.find(open_char), text.rfind(close_char)
    if start == -1 or end <= start:
        raise ValueError(f"No JSON {kind.__name__} in reply: {text[:200]!r}")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, kind):
        raise ValueError(f"Expected JSON {kind.__name__}, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------- the brief pattern

BRIEF_SYSTEM = """You are a reliability engineer reviewing a machine-health alert.

An embedding classifier (Omega encoder + KNN) has labelled a sensor window. You get the classifier's verdict and vote, statistics for the flagged window, and the same statistics for a healthy baseline. You do NOT get the raw embeddings or the classifier's reasoning.

Return ONLY a JSON object, no prose or code fences:
{"observation": "...", "likely_cause": "...", "checks": ["...", "..."]}

Rules:
- observation: the clearest difference from the baseline, citing numbers from the data. Max 160 characters.
- likely_cause: the most specific hypothesis the numbers support, phrased as a hypothesis ("consistent with...", "suggests..."). Don't just restate that the window differs. Max 140 characters.
- checks: 2-3 concrete things to inspect, each max 70 characters.
- Use only the signals given. Do not invent alarms, codes, frequencies or readings.
- Round numbers for reading at a glance."""


def build_brief_request(context: str, verdict: str, votes: dict[str, int],
                        flagged: dict, baseline: dict) -> str:
    return json.dumps({
        "asset_context": context,
        "classifier": {"verdict": verdict, "knn_votes": votes},
        "flagged_window": flagged,
        "healthy_baseline": baseline,
    })


def validate_brief(obj: dict) -> dict:
    """Keep only well-formed briefs; raise ValueError otherwise. Never show unvalidated model output."""
    observation = str(obj.get("observation") or "").strip()
    cause = str(obj.get("likely_cause") or "").strip()
    checks = [str(c).strip() for c in obj.get("checks") or [] if str(c).strip()][:3]
    if not observation or not cause or not checks:
        raise ValueError(f"Incomplete brief: {obj!r}")
    return {"observation": observation, "likely_cause": cause, "checks": checks}


def nemotron_brief(context: str, verdict: str, votes: dict[str, int], flagged: dict, baseline: dict) -> dict:
    reply = nemotron_chat(BRIEF_SYSTEM, build_brief_request(context, verdict, votes, flagged, baseline))
    return validate_brief(parse_json(reply, dict))


def banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
