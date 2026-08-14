#!/usr/bin/env python3
"""Run the managed OSM agent end-to-end against a PRE-PACKAGED bundle: upload a
sensor CSV, resolve the maintained "OSM Quick Start" bundle, run it, poll until
terminal, download the per-window predictions — and, if the input ships a
`<input>_labels.csv` ground-truth sidecar, score the run.

No bundle creation, no classifier URI: the platform ships canonical
quick-start bundles (classifier + windowing already pinned). This script just
finds one and runs it. It resolves the bundle **by name** so it is portable
across dev/staging/prod (the bundle *id* changes per environment; the name
does not); pass `--bundle-id` to skip the lookup.

Stdlib-only. The bundle API is plural everywhere (as of 2026-08-11):
`GET /agents/bundles[?query=…]`, `GET /agents/bundles/{id}`,
`POST /agents/bundles`, `POST /agents/bundles/{id}/run` — the singular forms
now return 404.

Auth / endpoint come from the environment (a local .env is loaded if present):
  ATAI_API_KEY        Bearer token (required).
  ATAI_API_ENDPOINT   Deployment root (required). A trailing /vX.Y is
                      tolerated and stripped: the agent API is versionless
                      (mounted at /agents), the files API at /v0.5/files.

Usage:
  python3 run_osm_agent.py                      # default sample slice
  python3 run_osm_agent.py --csv my_slice.csv
  python3 run_osm_agent.py --embeddings         # + Newton Omega embedding per window
  python3 run_osm_agent.py --bundle-id bnd_...  # skip name lookup (pin an id)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "sample_data", "volve_states_opt_slice_04.csv")

# Canonical pre-packaged quick-start bundles, resolved by name (portable across
# environments). The base name is a substring of the embeddings name, so the
# lookup selects the EXACT name match, not just the first hit.
BUNDLE_NAME = "OSM Quick Start (Volve Six State)"
BUNDLE_NAME_EMBEDDINGS = "OSM Quick Start (Volve Six State, Embeddings)"

POLL_INTERVAL_S = 15
TIMEOUT_S = 2 * 60 * 60   # ~2 min uncontended, ~30 min contended; keep generous margin


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def host_base():
    """Deployment root with any /vX.Y suffix stripped."""
    endpoint = os.environ.get("ATAI_API_ENDPOINT", "").rstrip("/")
    if not endpoint:
        sys.exit("ATAI_API_ENDPOINT is not set")
    return re.sub(r"/v[0-9]+(\.[0-9]+)*$", "", endpoint)


def request(method, url, body=None, headers=None, retries=4):
    """Call the API, retrying transient network failures.

    Polls run for the better part of an hour, and a single DNS hiccup or
    socket timeout would otherwise kill the client while the platform job
    carries on unattended. Only *network* errors retry; an HTTP status is a
    real answer from the server and still exits.
    """
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            sys.exit(f"{method} {url} failed ({e.code}): {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                sys.exit(f"{method} {url} failed after {retries + 1} attempts: {e}")
            wait = 5 * (2 ** attempt)
            print(f"  network error ({e}); retrying in {wait}s [{attempt + 1}/{retries}]")
            time.sleep(wait)


def upload_file(path):
    """POST a file to /v0.5/files as multipart/form-data (stdlib only)."""
    boundary = uuid.uuid4().hex
    filename = os.path.basename(path)
    with open(path, "rb") as f:
        content = f.read()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: text/csv\r\n\r\n",
        content,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(f"{host_base()}/v0.5/files", data=body, method="POST")
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"file upload failed ({e.code}): {e.read().decode(errors='replace')}")


def resolve_bundle(agents, name):
    """Find the pre-packaged bundle id for an exact name, via the plural
    read endpoint's case-insensitive substring search. Portable across
    environments: the name is stable, the id is not."""
    q = urllib.parse.urlencode({"query": name, "limit": 50})
    res = request("GET", f"{agents}/bundles?{q}")
    rows = res.get("data", []) if isinstance(res, dict) else (res or [])
    exact = [b for b in rows if b.get("name") == name]
    if not exact:
        seen = [b.get("name") for b in rows]
        sys.exit(f"no bundle named {name!r} found (query returned {seen}). "
                 f"Is it published in this environment? Pass --bundle-id to "
                 f"pin one, or contact support@archetypeai.dev.")
    # Prefer the canonical (platform-published) bundle over any same-named
    # org bundle that would otherwise shadow it.
    for bundle in exact:
        if bundle.get("is_canonical"):
            return bundle["id"]
    return exact[0]["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=DEFAULT_CSV, help="input CSV to run inference on")
    parser.add_argument("--embeddings", action="store_true",
                        help="use the variant bundle that also emits the Newton "
                             "Omega encoder embedding for each window — one "
                             "embedding_{variate} column per sensor channel, "
                             "each a 768-d vector")
    parser.add_argument("--bundle-id",
                        help="run this bundle id directly, skipping the name lookup")
    parser.add_argument("--window-size", type=int, default=16,
                        help="window the bundle's classifier was fit with; used "
                             "only for the local steady-state scoring cut (16)")
    parser.add_argument("--output", default="osm-output.csv",
                        help="local path to save the classified-windows CSV")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("ATAI_API_KEY"):
        sys.exit("ATAI_API_KEY is not set")
    agents = f"{host_base()}/agents"

    # 1. Upload the input CSV. Source connectors use the returned file_id
    #    (the filename), NOT the fil_ uid.
    print(f"uploading {args.csv} ...")
    uploaded = upload_file(args.csv)
    file_id = uploaded["file_id"]
    print(f"  file_id={file_id}  file_uid={uploaded.get('file_uid')}")

    # 2. Resolve the pre-packaged bundle. By name (portable) unless an id is
    #    pinned. The bundle already pins the classifier and windowing.
    name = BUNDLE_NAME_EMBEDDINGS if args.embeddings else BUNDLE_NAME
    if args.bundle_id:
        bundle_id = args.bundle_id
        print(f"using bundle {bundle_id}")
    else:
        bundle_id = resolve_bundle(agents, name)
        print(f"resolved {name!r} -> {bundle_id}")

    # 3. Run the bundle. No sink is given, so the runner writes one output per
    #    input, named after the input file.
    print("starting agent run ...")
    agent = request("POST", f"{agents}/bundles/{bundle_id}/run", body={
        "connectors": {"source": [{"type": "file", "id": file_id}]},
    })
    agent_id = agent["id"]
    print(f"  agent_id={agent_id}  status={agent['status']}")

    # 4. Poll until terminal, echoing new audit-log events as they arrive.
    seen_events = 0
    deadline = time.time() + TIMEOUT_S
    while True:
        agent = request("GET", f"{agents}/instances/{agent_id}")
        events = request("GET", f"{agents}/instances/{agent_id}/events")["data"]
        for event in events[seen_events:]:
            print(f"  [{event['level']}] {event['created_at']}  {event['message']}")
        seen_events = len(events)
        if agent["status"] not in ("running", "paused"):
            break
        if time.time() > deadline:
            sys.exit(f"timed out after {TIMEOUT_S}s; agent {agent_id} still {agent['status']}")
        time.sleep(POLL_INTERVAL_S)

    print(f"agent finished: status={agent['status']}")

    # 5. Fetch the run results and download the output CSV. Checked even on
    #    status=failed: a flaky job poller can mark a successful run failed —
    #    if the output is there, the run succeeded.
    results = request("GET", f"{agents}/instances/{agent_id}/results")
    if not results["data"]:
        sys.exit(f"run produced no output: status={agent['status']} "
                 f"error={agent.get('error')}")
    if agent["status"] != "completed":
        print(f"  note: status={agent['status']} ({agent.get('error')}) "
              f"but output exists — treating as succeeded")
    # The results envelope is {data, has_more, next_cursor} — no `total` field
    # (removed when the list envelopes were standardized).
    print(f"results ({len(results['data'])}):")
    for output in results["data"]:
        print(f"  {output['data']['filename']}  ({output['data']['num_bytes']} bytes)")

    filename = results["data"][0]["data"]["filename"]
    req = urllib.request.Request(f"{host_base()}/v0.5/files/download/{filename}")
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    with urllib.request.urlopen(req) as resp, open(args.output, "wb") as f:
        f.write(resp.read())
    print(f"saved output to {args.output}")

    # 6. If the input ships a ground-truth sidecar (the sample slice does),
    #    score the predictions: each window is judged against the label of its
    #    end row (the row its timestamp points at). Stdlib only.
    labels_csv = args.csv.replace(".csv", "_labels.csv")
    if os.path.exists(labels_csv):
        evaluate(args.output, labels_csv, args.window_size)


def evaluate(pred_csv, labels_csv, window):
    import csv
    truth_rows = list(csv.DictReader(open(labels_csv)))
    truth = {int(float(r["DATE_TIME"])): r["label"] for r in truth_rows}

    # steady windows: all rows one label, no timestamp seams inside
    ts = [int(float(r["DATE_TIME"])) for r in truth_rows]
    labs = [r["label"] for r in truth_rows]
    steady_ts = set()
    for i in range(len(ts) - window + 1):
        w_labs, w_ts = labs[i:i + window], ts[i:i + window]
        if all(l == w_labs[0] for l in w_labs) and \
           all(1 <= b - a <= 60 for a, b in zip(w_ts, w_ts[1:])):
            steady_ts.add(w_ts[-1])

    per_class = {}
    n = correct = n_steady = correct_steady = invalid = 0
    for row in csv.DictReader(open(pred_csv)):
        # window-end key: current blueprint emits `finish_timestamp`; older
        # runs emitted `timestamp` — accept either
        t = int(float(row.get("finish_timestamp") or row["timestamp"]))
        pred, true = row["predicted_state"], truth.get(t)
        if pred == "INVALID_STATE" or true is None:
            invalid += 1
            continue
        stats = per_class.setdefault(true, {"tp": 0, "fn": 0})
        per_class.setdefault(pred, {"tp": 0, "fn": 0}).setdefault("fp", 0)
        n += 1
        if pred == true:
            correct += 1
            stats["tp"] += 1
        else:
            stats["fn"] += 1
            per_class[pred]["fp"] = per_class[pred].get("fp", 0) + 1
        if t in steady_ts:
            n_steady += 1
            correct_steady += pred == true
    if not n:
        print(f"\nno windows scored vs {os.path.basename(labels_csv)} "
              f"({invalid} invalid/unmatched skipped) — timestamps don't line up?")
        return
    print(f"\nevaluation vs {os.path.basename(labels_csv)} "
          f"({n} scored windows, {invalid} invalid/unmatched skipped):")
    print(f"  accuracy: {correct/n:.4f}   steady-state accuracy: "
          f"{correct_steady/max(n_steady,1):.4f} ({n_steady} windows)")
    f1s = []
    for state in sorted(per_class):
        s = per_class[state]
        tp, fn, fp = s["tp"], s["fn"], s.get("fp", 0)
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        print(f"  {state:<14} precision={prec:.4f} recall={rec:.4f} "
              f"f1={f1:.4f} (n={tp+fn})")
    print(f"  macro-F1: {sum(f1s)/len(f1s):.4f}")


if __name__ == "__main__":
    main()
