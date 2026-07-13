#!/usr/bin/env python3
"""Run the managed OSM agent end-to-end: upload a sensor CSV, create a bundle
from the canonical `osm` blueprint pinning a fitted classifier, run it, poll
until terminal, download the per-window predictions — and, if the input ships
a `<input>_labels.csv` ground-truth sidecar, score the run.

Stdlib-only. Each step maps to an operation in ../openapi.yaml:
  1. Upload a CSV of sensor records to the platform files API.
  2. Create a bundle from the `osm` blueprint, pointing its
     `fit-classifier` artifact at a fitted classifier on S3. (create_bundle)
  3. Run the bundle with the uploaded file bound as the source connector.
     Each run creates a new agent instance.                    (run_bundle)
  4. Poll the agent until it reaches a terminal state, surfacing the
     audit-log events as they appear.            (get_agent, list_agent_events)
  5. Fetch the run results and download the output CSV.  (get_agent_results)

Auth / endpoint come from the environment (a local .env is loaded if present):
  ATAI_API_KEY        Bearer token (required).
  ATAI_API_ENDPOINT   Deployment root (required). A trailing /vX.Y is
                      tolerated and stripped: the agent API is versionless
                      (mounted at /agents), the files API at /v0.5/files.

Usage:
  python3 run_osm_agent.py --classifier s3://... --window-size 16
  python3 run_osm_agent.py --csv my_slice.csv --classifier s3://... \
      --window-size 64 --step-size 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "sample_data", "volve_states_opt_slice_04.csv")
# The published six-state Volve classifier this skill's sample data matches
# (fitted at window_size=16, step_size=1 — the defaults below).
DEFAULT_CLASSIFIER = (
    "s3://atai-platform-dev-platform-data-us-west-2/"
    "osm_classifier_tmp/volve-states-classifier-20260710T063913Z.safetensors"
)
BLUEPRINT_KEY = "osm"
POLL_INTERVAL_S = 15
TIMEOUT_S = 45 * 60


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


def request(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        sys.exit(f"{method} {url} failed ({e.code}): {detail}")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=DEFAULT_CSV, help="input CSV to run inference on")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                        help="s3:// path to the fitted classifier safetensors")
    parser.add_argument("--window-size", type=int, default=16,
                        help="MUST match the classifier's fit config — a mismatch "
                             "silently degrades accuracy instead of erroring")
    parser.add_argument("--step-size", type=int, default=1,
                        help="rows between window starts (1 = score every row)")
    parser.add_argument("--name", default="OSM agent run",
                        help="human label for the bundle/agent")
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

    # 2. Create a bundle pinning the canonical OSM blueprint. The
    #    `fit-classifier` artifact slot must be an s3:// URI (platform file
    #    ids and https URLs fail with ENOENT).
    print(f"creating bundle from blueprint '{BLUEPRINT_KEY}' ...")
    bundle = request("POST", f"{agents}/bundle", body={
        "blueprint": BLUEPRINT_KEY,
        "name": args.name,
        "description": "OSM skill reference run",
        "values": {"window_size": args.window_size, "step_size": args.step_size},
        "artifacts": {"fit-classifier": args.classifier},
    })
    print(f"  bundle_id={bundle['id']}  status={bundle['status']}")

    # 3. Run the bundle. No sink is given, so the runner writes one output per
    #    input, named after the input file.
    print("starting agent run ...")
    agent = request("POST", f"{agents}/bundle/{bundle['id']}/run", body={
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
    print(f"results ({results['total']}):")
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
        t = int(float(row["timestamp"]))
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
