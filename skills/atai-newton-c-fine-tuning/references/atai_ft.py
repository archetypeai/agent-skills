"""Stdlib-only client for Newton C fine-tuning and batch inference (platform API 2.0).

Two surfaces, both under {root}/api:

    fine-tuning   POST /fine-tuning/jobs, GET .../{id}, .../{id}/events|checkpoints,
                  POST .../{id}/cancel
    batch         POST /batch/jobs (pipeline_key "activity-detection"),
                  GET .../{id}, .../{id}/events|outputs
    files         POST /files/uploads/initiate -> PUT presigned parts -> .../complete

Reads ATAI_API_KEY and ATAI_API_ENDPOINT from the environment, then from ./.env
(the environment wins). The endpoint may be the bare root or end in /api.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CLIENT = "agent_skills"  # X-Atai-Client: attributes these calls to the skills
BASE_VARIANT = "newton/c:2.5.1-8b-base"
FT_TERMINAL = {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED", "CANCELED", "STOPPED"}
FT_SUCCESS = {"SUCCEEDED", "COMPLETED"}
BATCH_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "CANCELED", "STOPPED"}


# ---------------------------------------------------------------- config

def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            os.environ.setdefault(k.strip(), v)


def api_base(endpoint: str) -> str:
    """{root}/api for any accepted endpoint form: bare root, /api, or a legacy /vX.Y."""
    e = endpoint.strip().rstrip("/")
    e = re.sub(r"/v[0-9]+(\.[0-9]+)*$", "", e)
    e = re.sub(r"/api$", "", e)
    return e + "/api"


def base() -> str:
    load_dotenv()
    ep = os.environ.get("ATAI_API_ENDPOINT", "")
    if not ep.strip():
        sys.exit("set ATAI_API_ENDPOINT (see .env.example)")
    return api_base(ep)


def api_key() -> str:
    load_dotenv()
    key = os.environ.get("ATAI_API_KEY")
    if not key:
        sys.exit("set ATAI_API_KEY (see .env.example)")
    return key


# ---------------------------------------------------------------- http

def request(method: str, path: str, body=None, *, params: dict | None = None,
            raw: bool = False, allow_status: bool = False, timeout: int = 120):
    """Call {base}{path}. GETs retry network errors and 502/503/504; creates are sent once."""
    url = base() + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    tries = 5 if method == "GET" else 1
    for attempt in range(tries):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {api_key()}")
        req.add_header("X-Atai-Client", CLIENT)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
                out = payload if raw else json.loads(payload or b"null")
                return (resp.status, out) if allow_status else out
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code not in (502, 503, 504) or attempt + 1 == tries:
                if allow_status:
                    return e.code, detail
                sys.exit(f"{method} {path} failed ({e.code}): {detail[:1000]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            if attempt + 1 == tries:
                if allow_status:
                    return 0, str(e)
                sys.exit(f"{method} {path} failed: {e}")
        time.sleep(5 * 2 ** attempt)


def list_all(path: str, get=None, limit: int = 100) -> list:
    """Every item of a cursor-paginated list, oldest first.

    Platform 2.0 lists are {data, has_more, next_cursor}, newest first; the next
    (older) page is `after=<next_cursor>`. `offset` is ignored and there is no total.
    """
    get = get or (lambda p, params: request("GET", p, params=params))
    items: list = []
    after = None
    while True:
        params = {"limit": limit, **({"after": after} if after else {})}
        page = get(path, params)
        items.extend(page.get("data") or [])
        after = page.get("next_cursor")
        if not page.get("has_more") or not after:
            break
    return list(reversed(items))


# ---------------------------------------------------------------- files

def upload_name(path: str | Path, prefix: str = "newton_c") -> str:
    """A content-addressed file name. file_id IS the name, in an org-wide namespace,
    and a name that already exists is reused (409), so a fixed name would silently
    serve stale contents after you rebuild the file."""
    p = Path(path)
    digest = hashlib.sha1(p.read_bytes()).hexdigest()[:8]
    return f"{prefix}_{p.stem}_{digest}{p.suffix}"


def upload_jsonl(path: str | Path, name: str | None = None) -> str:
    """Upload a JSONL file with the multipart presigned flow; return its file_id.

    The simple POST /files rejects application/x-ndjson, so every JSONL goes this way.
    """
    p = Path(path)
    name = name or upload_name(p)
    status, init = request("POST", "/files/uploads/initiate", {
        "filename": name, "file_type": "application/x-ndjson", "num_bytes": p.stat().st_size,
    }, allow_status=True)
    if status == 409:
        return name  # same content already uploaded under this name
    if status != 200 and status != 201:
        sys.exit(f"upload initiate {name} failed ({status}): {str(init)[:500]}")
    upload_id, parts, done = init["upload_id"], init["parts"], []
    try:
        with p.open("rb") as f:
            for part in parts:
                f.seek(part["offset"])
                chunk = f.read(part["length"])
                req = urllib.request.Request(part["url"], data=chunk, method="PUT")  # presigned: no auth
                with urllib.request.urlopen(req, timeout=600) as resp:
                    etag = resp.headers.get("ETag", "").strip('"')
                done.append({"part_number": part["part_number"], "part_token": etag})
    except Exception:
        request("POST", f"/files/uploads/{upload_id}/abort", allow_status=True)
        raise
    out = request("POST", f"/files/uploads/{upload_id}/complete", {"parts": done})
    return out.get("file_id") or out.get("file_name") or name


def fetch_ref(ref: str) -> str:
    """Download a batch output. A ref is a root path (`/api/files/fil_…/content`):
    strip its /api so it isn't doubled on {base}. Absolute URLs are presigned."""
    if ref.startswith("/"):
        return request("GET", ref_path(ref), raw=True).decode()
    with urllib.request.urlopen(ref, timeout=120) as resp:
        return resp.read().decode()


def ref_path(ref: str) -> str:
    return re.sub(r"^/(api|v[0-9]+(\.[0-9]+)*)(?=/)", "", ref)


# ---------------------------------------------------------------- fine-tuning

def fine_tune_body(train_id: str, val_id: str, name: str, *, n_epochs="auto", seed=None) -> dict:
    """POST /fine-tuning/jobs body. `name` and `validation_files` are required (422
    without them). There is no base-model selector: `model: "newton"` is C 2.5.1-8b,
    and a `base_model_tag` in the body is ignored."""
    f = lambda fid: {"type": "plain", "file_id": fid, "format": "jsonl"}  # noqa: E731
    body = {
        "model": "newton",
        "name": name,
        "method": {"type": "lora", "lora": {"hyperparameters": {
            "n_epochs": n_epochs, "batch_size": "auto", "learning_rate_multiplier": "auto"}}},
        "training_files": [f(train_id)],
        "validation_files": [f(val_id)],
    }
    if seed is not None:
        body["seed"] = seed
    return body


def best_checkpoint(checkpoints: list[dict]) -> dict | None:
    """The checkpoint with the lowest validation eval_loss (the val split's job)."""
    scored = [c for c in checkpoints if (c.get("metrics") or {}).get("eval_loss") is not None]
    if scored:
        return min(scored, key=lambda c: c["metrics"]["eval_loss"])
    return checkpoints[-1] if checkpoints else None


# ---------------------------------------------------------------- batch inference

def inference_job_body(file_id: str, name: str, checkpoint_id: str | None = None,
                       variant: str = BASE_VARIANT, max_new_tokens: int = 64) -> dict:
    """POST /batch/jobs body for activity-detection.

    The fine-tuned model is the BASE variant plus the checkpoint on the
    `worker.checkpoint` input port. `config.checkpoint_id` is accepted and silently
    ignored (you get the base model), and a checkpoint as `model_variant` is rejected.
    """
    inputs = {"worker.data": [{"kind": "file", "file_id": file_id}]}
    if checkpoint_id:
        inputs["worker.checkpoint"] = [{"kind": "checkpoint", "checkpoint_id": checkpoint_id}]
    return {
        "name": name,
        "pipeline_type": "batch",
        "pipeline_key": "activity-detection",
        "inputs": inputs,
        "parameters": {"worker": {"parallelism": 1, "config": {
            "model_variant": variant,
            "generation": {"do_sample": False, "max_new_tokens": max_new_tokens},
        }}},
    }


def predictions(job_id: str) -> list[str]:
    """A finished batch job's predictions, ordered by line_index (= input line order)."""
    rows: list[tuple[int, str]] = []
    for out in list_all(f"/batch/jobs/{job_id}/outputs"):
        for line in fetch_ref(out["data"]["ref"]).splitlines():
            if line.strip():
                obj = json.loads(line)
                rows.append((obj.get("line_index", len(rows)), str(obj.get("prediction", ""))))
    return [p for _, p in sorted(rows)]


# ---------------------------------------------------------------- polling

def watch(path: str, terminal: set[str], interval: int = 30) -> dict:
    """Poll a job and print new events until it reaches a terminal status."""
    seen: set = set()
    last = None
    while True:
        job = request("GET", path)
        status = job.get("status")
        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] status {status}", flush=True)
            last = status
        status_code, page = request("GET", f"{path}/events", params={"limit": 50}, allow_status=True)
        if status_code == 200:
            for e in reversed(page.get("data") or []):
                key = e.get("id") or (e.get("created_at"), e.get("message"))
                if key not in seen:
                    seen.add(key)
                    print(f"  {e.get('event_type') or e.get('type', '')}: {e.get('message') or '-'}", flush=True)
        if status in terminal:
            return job
        time.sleep(interval)


# ---------------------------------------------------------------- state

STATE = Path("ft-state.json")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kv) -> dict:
    s = {**load_state(), **kv}
    STATE.write_text(json.dumps(s, indent=2) + "\n")
    return s
