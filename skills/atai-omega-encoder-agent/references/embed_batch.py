#!/usr/bin/env python3
"""Embed role files in batch with the Omega Encoder agent (Agent Lifecycle 2.0, developer key).

    python embed_batch.py --roles roles --role library validation --window 1024 512 256
    python embed_batch.py --roles roles --role test delivery --window frozen

Role files are the data-prep skill's output (`atai-omega-encoder-agent-data-prep`): one file
per segment, `timestamp` + channels, `rows` rows each, and a manifest.json.

For each window size W:
  project (reused by name) -> copy of the canonical `omega-embeddings` blueprint, with
  `overrides.values` window_size = step_size = W -> package -> batch deployment -> one batch
  session per role over that role's files.
Each file yields rows / W windows; they are matched back to their file by start timestamp
(the data-prep preflight guarantees no two files' time ranges overlap). `frozen` is the
window `detect.py select` froze in <eval>/settings.json.

Writes <out>/embed/w<W>/<role>.npz: files, emb [segments, windows, channels, 768], window.
Ids (project, uploads, and per window the blueprint, package, deployment and sessions) are
kept in <out>/embed/state.json, so a rerun reuses everything it can and an unchanged role
is not embedded again. A different API key or endpoint starts fresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atai_api as A  # noqa: E402

_LOCK = threading.Lock()
SHIPPED_WINDOW = 1024


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class State:
    def __init__(self, out: str):
        self.path = os.path.join(out, "embed", "state.json")
        self.out = out
        st = json.load(open(self.path)) if os.path.exists(self.path) else {}
        ident = hashlib.sha256(f"{A.api_base()}|{os.environ['ATAI_API_KEY']}".encode()).hexdigest()[:12]
        if st and st.get("identity") != ident:
            log("API key or endpoint changed since the last run: starting fresh (old ids and embeddings discarded)")
            for d, _, names in os.walk(os.path.join(out, "embed")):
                for n in names:
                    if n.endswith(".npz"):
                        os.remove(os.path.join(d, n))
            st = {}
        st["identity"] = ident
        st.setdefault("windows", {})
        self.d = st
        self.save()

    def save(self) -> None:
        with _LOCK:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            json.dump(self.d, open(self.path, "w"), indent=1)


def ensure_project(st: State, name: str) -> str:
    if "project" not in st.d:
        found = [p for p in A.list_all(f"{A.api_base()}/projects") if p["name"] == name]
        st.d["project"] = found[0]["id"] if found else A.request(
            "POST", f"{A.api_base()}/projects", body={"name": name, "description": "Omega Encoder agent embeddings"})["id"]
        st.save(); log(f"project {st.d['project']}")
    return st.d["project"]


def ensure_package(st: State, window: int, project_name: str, prefix: str) -> dict:
    """The window's blueprint copy (window and step set at copy time) and its package."""
    ws = st.d["windows"].setdefault(str(window), {})
    key = f"{prefix}-w{window}"
    if "blueprint" not in ws:
        project = ensure_project(st, project_name)
        found = [b for b in A.request("GET", f"{A.agents_base()}/blueprints?project_id={project}&key={key}")["data"]
                 if b.get("blueprint_key") == key]
        bp = found[0] if found else A.request(
            "POST", f"{A.agents_base()}/blueprints/omega-embeddings/copy",
            body={"project_id": project, "blueprint_key": key,
                  "overrides": {"values": {"window_size": window, "step_size": window}}})
        ws["blueprint"] = bp["id"]; st.save(); log(f"blueprint {key} {bp['id']}")
    values = A.request("GET", f"{A.agents_base()}/blueprints/{ws['blueprint']}")["document"]["values"]
    if (values["window_size"], values["step_size"]) != (window, window):
        sys.exit(f"blueprint {key} has window {values['window_size']}, step {values['step_size']}; expected {window}")
    A.ready_package(ws, ws["blueprint"], key, st.save, log)    # a failed one is rebuilt, once
    return ws


def ensure_batch(st: State, window: int, project_name: str, prefix: str) -> dict:
    ws = ensure_package(st, window, project_name, prefix)
    dpl = ws.get("deployment")
    if not dpl or A.status_of(A.deployments_url(dpl)) not in ("ready", "deploying"):
        ws["deployment"] = A.deploy(ws["package"], "batch")["id"]; st.save()   # saved before waiting
        log(f"batch deployment {ws['deployment']} (window {window})")
    A.deployment_ready(ws["deployment"])
    return ws


def cache_key(path: str) -> str:
    """Path plus content hash: a rebuilt role file is uploaded again, never reused stale."""
    return f"{os.path.abspath(path)}:{hashlib.sha1(open(path, 'rb').read()).hexdigest()[:12]}"


def upload_all(st: State, paths: list[str], jobs: int = 4) -> list[str]:
    cache = st.d.setdefault("uploads", {})
    keys = {p: cache_key(p) for p in paths}
    todo = [p for p in paths if keys[p] not in cache]
    if todo:
        log(f"uploading {len(todo)} files")
        with ThreadPoolExecutor(jobs) as pool:
            for key, fid in pool.map(lambda p: (keys[p], A.upload_file(p)["file_id"]), todo):
                with _LOCK:
                    cache[key] = fid
                st.save()                       # each finished upload survives a later failure
    return [cache[keys[p]] for p in paths]


def fetch_all_results(session_id: str, expected: int, settle_s: int = 600) -> list[dict]:
    """Every window the session wrote, re-fetched until there are exactly `expected`: a
    completed session can briefly return a partial result. /results is paged (100 by
    default; one output per input file), so next_cursor is followed."""
    deadline = time.time() + settle_s
    while True:
        items, cursor = [], None
        while True:
            page = A.request("GET", f"{A.sessions_url(session_id)}/results?limit=1000"
                             + (f"&after={cursor}" if cursor else ""))
            items += page.get("data", [])
            cursor = page.get("next_cursor")
            if not page.get("has_more") or not cursor:
                break
        rows = []
        for it in items:
            ref = (it.get("data") or {}).get("ref") or it.get("ref")
            payload = A._download(ref) if ref else None
            if payload:
                rows += [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        if len(rows) == expected:
            return rows
        if time.time() > deadline:
            raise Unfinished(f"{session_id}: {len(rows)} windows after {settle_s}s, expected {expected}")
        log(f"  {session_id}: {len(rows)}/{expected} windows; waiting")
        time.sleep(15)


class Unfinished(Exception):
    """A role's session that failed, or whose output stayed short."""


def map_windows(rows: list[dict], files: list[dict], window: int, step: float, channels: list[str],
                per_file: int) -> np.ndarray:
    """Batch output rows -> [files, windows, channels, 768]. Window k of a file starts k * window
    rows after the file's first timestamp; outputs carry the channels by their column names."""
    by_start = {round(r["start_timestamp"], 4): r for r in rows}
    out, missing = [], []
    for f in files:
        starts = [round(f["start"] + k * window * step, 4) for k in range(per_file)]
        if any(s not in by_start for s in starts):
            missing.append(os.path.basename(f["path"]))
            continue
        out.append([[by_start[s][c] for c in channels] for s in starts])
    if missing:
        raise Unfinished(f"no window at the expected start times of: {missing[:5]}")
    return np.asarray(out, dtype=np.float32)


def run_role(st: State, a, m: dict, window: int, role: str) -> None:
    files = [f for f in m["files"] if f["role"] == role]
    if not files:
        log(f"{role}: no files in the manifest"); return
    paths = [os.path.join(a.roles, f["path"]) for f in files]
    per_file = m["rows"] // window
    expected = len(files) * per_file
    out_path = os.path.join(a.out, "embed", f"w{window}", f"{role}.npz")
    run_key = hashlib.sha1("|".join(sorted(cache_key(p) for p in paths)).encode()).hexdigest()[:16]
    if os.path.exists(out_path) and str(np.load(out_path)["run_key"]) == run_key:
        log(f"{role} @ {window}: already embedded ({expected} windows)"); return
    ws = ensure_batch(st, window, a.project, a.key_prefix)
    fids = upload_all(st, paths)
    key = f"{role}:{run_key}"
    sid = ws.setdefault("sessions", {}).get(key)
    if not sid:
        sid = A.open_session(ws["deployment"], fids)["id"]
        ws["sessions"][key] = sid; st.save()
        log(f"{role} @ {window}: batch session {sid} over {len(fids)} files, {expected} windows expected")
    t = time.time()
    status = A.poll_session(sid, timeout_s=6 * 3600, interval_s=20)
    try:
        if status != "completed":
            raise Unfinished(f"session {sid} ended {status}")
        emb = map_windows(fetch_all_results(sid, expected), files, window, m["step_s"], m["channels"], per_file)
    except Unfinished as e:                  # end and forget it: the next run opens a new session
        try:
            A.end_session(sid)
        except SystemExit as err:
            log(f"  could not end {sid}: {err}")
        ws["sessions"].pop(key, None); st.save()
        raise Unfinished(f"{role} @ {window}: {e}; rerun to start a new session") from None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, files=np.asarray([os.path.basename(f["path"]) for f in files]), emb=emb,
                        run_key=np.asarray(run_key), window=np.asarray(window))
    log(f"{role} @ {window}: {expected} windows {emb.shape} in {time.time() - t:.0f}s -> {out_path}")


def windows_arg(values: list[str], rows: int, eval_dir: str) -> list[int]:
    out = []
    for v in values:
        if v == "frozen":
            p = os.path.join(eval_dir, "settings.json")
            if not os.path.exists(p):
                sys.exit(f"--window frozen needs {p}: run detect.py select first")
            out.append(int(json.load(open(p))["window"]))
        elif v.isdigit() and int(v) > 0 and rows % int(v) == 0:
            out.append(int(v))
        else:
            sys.exit(f"--window takes divisors of {rows} (the rows per segment) or 'frozen' (got {v})")
    return list(dict.fromkeys(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roles", required=True, help="the data-prep skill's role files (with manifest.json)")
    ap.add_argument("--role", nargs="+", required=True, choices=["library", "validation", "test", "delivery"])
    ap.add_argument("--window", nargs="+", default=[str(SHIPPED_WINDOW)], help="window sizes, or 'frozen'")
    ap.add_argument("--out", default="out")
    ap.add_argument("--eval", default="eval_out", help="where detect.py writes settings.json")
    ap.add_argument("--project", default="Omega Encoder agent")
    ap.add_argument("--key-prefix", default="omega-encoder", help="blueprint copies are <prefix>-w<W>")
    a = ap.parse_args()
    A.load_dotenv()
    m = json.load(open(os.path.join(a.roles, "manifest.json")))
    windows = windows_arg(a.window, m["rows"], a.eval)
    log(f"endpoint {A.api_base()}")
    st = State(a.out)
    unfinished = []
    for window in windows:
        for role in a.role:
            try:
                run_role(st, a, m, window, role)
            except Unfinished as e:              # the other roles and windows go on
                log(str(e))
                unfinished.append(f"{role} @ {window}")
    if unfinished:
        sys.exit(f"{len(unfinished)} without a complete output ({', '.join(unfinished)}): rerun to start new sessions")


if __name__ == "__main__":
    main()
