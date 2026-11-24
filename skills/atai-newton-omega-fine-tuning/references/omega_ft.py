"""Stdlib-only client for fine-tuning an Omega head and running it (platform API 2.0).

    fine-tuning   POST /fine-tuning/jobs (model "omega", method "head"), GET .../{id},
                  .../{id}/events|checkpoints, POST .../{id}/cancel
    batch         POST /batch/jobs (pipeline_key "machine-state-classification"),
                  GET .../{id}, .../{id}/events|outputs
    files         POST /files/uploads/initiate -> PUT presigned parts -> .../complete

Reads ATAI_API_KEY and ATAI_API_ENDPOINT from the environment, then from ./.env
(the environment wins). The endpoint may be the bare root or end in /api.
"""
from __future__ import annotations

import csv
import hashlib
import io
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
FT_TERMINAL = {"SUCCEEDED", "COMPLETED", "FAILED", "CANCELLED", "CANCELED", "STOPPED"}
FT_SUCCESS = {"SUCCEEDED", "COMPLETED"}
BATCH_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "CANCELED", "STOPPED"}
PIPELINE = "machine-state-classification"


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

def upload_name(path: str | Path, prefix: str = "omega") -> str:
    """A content-addressed file name. file_id IS the name, in an org-wide namespace,
    and a name that already exists is reused (409), so a fixed name would silently
    serve stale contents after you rebuild the file."""
    p = Path(path)
    digest = hashlib.sha1(p.read_bytes()).hexdigest()[:8]
    return f"{prefix}_{p.stem}_{digest}{p.suffix}"


def upload_csv(path: str | Path, name: str | None = None) -> str:
    """Upload a CSV with the multipart presigned flow; return its file_id (= its name)."""
    p = Path(path)
    name = name or upload_name(p)
    status, init = request("POST", "/files/uploads/initiate", {
        "filename": name, "file_type": "text/csv", "num_bytes": p.stat().st_size,
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


# ---------------------------------------------------------------- data layout

def class_files(root: str | Path, split: str) -> list[tuple[Path, str]]:
    """(csv, class) for <root>/<split>/<class>/*.csv, sorted."""
    d = Path(root) / split
    if not d.is_dir():
        sys.exit(f"missing {d}/<class>/*.csv")
    out = [(p, c.name) for c in sorted(d.iterdir()) if c.is_dir() for p in sorted(c.glob("*.csv"))]
    if not out:
        sys.exit(f"no CSVs under {d}/<class>/")
    return out


def data_columns(csv_path: str | Path, timestamp_column: str = "timestamp") -> list[str]:
    with open(csv_path) as f:
        return [c for c in next(csv.reader(f)) if c != timestamp_column]


# ---------------------------------------------------------------- fine-tuning

def n_shot_file(file_id: str, label: str) -> dict:
    """An Omega training/validation file: one class per file, the class in `label`."""
    return {"type": "n_shot", "file_id": file_id, "label": label, "format": "csv"}


def head_body(train: list[tuple[str, str]], val: list[tuple[str, str]], name: str, columns: list[str],
              *, window: int = 128, step: int = 4, n_epochs=30, batch_size=32, seed=None) -> dict:
    """POST /fine-tuning/jobs body for a head on frozen Omega. `train`/`val` are
    (file_id, class) pairs. The request names no Omega version: the platform uses its
    default and reports it as the job's `base_model_tag`."""
    body = {
        "model": "omega",
        "name": name,
        "method": {"type": "head", "head": {
            "hyperparameters": {"batch_size": batch_size, "n_epochs": n_epochs},
            "reader": {"window_size": window, "step_size": step, "data_columns": columns},
        }},
        "training_files": [n_shot_file(f, c) for f, c in train],
        "validation_files": [n_shot_file(f, c) for f, c in val],
    }
    if seed is not None:
        body["seed"] = seed
    return body


def best_checkpoint(checkpoints: list[dict]) -> dict | None:
    """The head worker saves a checkpoint each time validation macro-F1 improves, so the
    highest step is the best on validation."""
    return max(checkpoints, key=lambda c: c.get("step", 0)) if checkpoints else None


def collapsed(checkpoints: list[dict], n_classes: int) -> bool:
    """True when the only checkpoint scores like a one-class predictor (macro-F1 1/n):
    the head never learned. Typical when there are too few training windows."""
    if len(checkpoints) != 1:
        return False
    f1 = (checkpoints[0].get("metrics") or {}).get("macro_f1")
    one_class = 2 / (n_classes * (n_classes + 1))  # 0.333 for two balanced classes
    return f1 is not None and abs(f1 - one_class) < 0.02


# ---------------------------------------------------------------- batch inference

def inference_body(inference: list[str], n_shots: list[tuple[str, str]], checkpoint_id: str, name: str,
                   columns: list[str], *, window: int = 128, step: int = 1,
                   timestamp_column: str = "timestamp") -> dict:
    """POST /batch/jobs body that classifies `inference` files with a fine-tuned head.

    The head goes on the `worker.fine_tune_checkpoint` input. `worker.n_shots` is a
    REQUIRED input of the pipeline (it is the KNN reference when no checkpoint is
    attached); with a checkpoint the worker ignores it. `window` must equal the
    fine-tune's window: the head only understands embeddings from that window.
    """
    return {
        "name": name,
        "pipeline_type": "batch",
        "pipeline_key": PIPELINE,
        "inputs": {
            "worker.inference": [{"file_id": f} for f in inference],
            "worker.n_shots": [{"file_id": f, "metadata": {"class": c}} for f, c in n_shots],
            "worker.fine_tune_checkpoint": [{"kind": "checkpoint", "checkpoint_id": checkpoint_id,
                                             "metadata": {}}],
        },
        "parameters": {"worker": {"parallelism": 1, "config": {
            "model_type": "omega_1_5_base",
            "reader_config": {"window_size": window, "step_size": step,
                              "timestamp_column": timestamp_column, "data_columns": columns},
            "batch_size": 32,
        }}},
    }


def outputs(job_id: str) -> list[tuple[str, str]]:
    """(input file id, CSV text) for every output of a finished batch job. Each output
    entry names the input it came from (`input_file_id`); its CSV is `timestamp,Prediction`,
    one row per window (the timestamp of the window's last row)."""
    return [(o.get("input_file_id", ""), fetch_ref(o["data"]["ref"]))
            for o in list_all(f"/batch/jobs/{job_id}/outputs")]


def predictions(csv_text: str) -> list[str]:
    """The `Prediction` column of one output CSV."""
    return [row["Prediction"] for row in csv.DictReader(io.StringIO(csv_text))]


def score(pairs: list[tuple[str, str]]) -> dict:
    """Accuracy, confusion matrix and per-class precision/recall/F1 from (actual, predicted)."""
    classes = sorted({a for a, _ in pairs} | {p for _, p in pairs})
    conf = {a: {p: 0 for p in classes} for a in classes}
    for a, p in pairs:
        conf[a][p] += 1
    per = {}
    for c in classes:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in classes if o != c)
        fn = sum(conf[c][o] for o in classes if o != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0}
    n = len(pairs)
    correct = sum(conf[c][c] for c in classes)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0, "confusion": conf,
            "per_class": per, "macro_f1": sum(v["f1"] for v in per.values()) / len(per) if per else 0.0}


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

STATE = Path("omega-ft-state.json")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kv) -> dict:
    s = {**load_state(), **kv}
    STATE.write_text(json.dumps(s, indent=2) + "\n")
    return s
