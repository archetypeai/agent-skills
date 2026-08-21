#!/usr/bin/env python3
"""Run the managed OSM agent end-to-end against a PRE-PACKAGED bundle: upload a
sensor CSV, resolve the maintained "OSM Quick Start" bundle, run it, poll until
terminal, download the per-window predictions — and, if the input ships a
`<input>_labels.csv` ground-truth sidecar, score the run.

Built on the official Archetype AI python client (`pip install archetypeai`),
which owns auth, endpoint resolution, and the Agents API surface. The client
mounts the versionless `/agents` and the versioned `/v0.5/files` itself, so a
`/vX.Y` suffix on ATAI_API_ENDPOINT is handled for you.

No bundle creation, no classifier URI: the platform ships canonical quick-start
bundles (classifier + windowing already pinned). This script just finds one and
runs it. It resolves the bundle **by name** so it is portable across
deployments (the bundle *id* is deployment-specific; the name is not); pass
`--bundle-id` to skip the lookup.

Auth / endpoint come from the environment (a local .env is loaded if present):
  ATAI_API_KEY        Bearer token (required).
  ATAI_API_ENDPOINT   Deployment root (required).

Usage:
  python3 run_osm_agent.py                      # default sample slice
  python3 run_osm_agent.py --csv my_slice.csv
  python3 run_osm_agent.py --embeddings         # + Newton Omega embedding per window
  python3 run_osm_agent.py --bundle-id bnd_...  # skip name lookup (pin an id)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

try:
    from archetypeai import ArchetypeAI
except ModuleNotFoundError:  # the only third-party dependency
    sys.exit("This runner needs the official Archetype AI client:\n"
             "    pip install -r requirements.txt   (from this directory)\n"
             "    pip install archetypeai           (or just the package)")

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "sample_data", "volve_states_opt_slice_04.csv")

# Canonical pre-packaged quick-start bundles, resolved by name (portable across
# deployments). Query is a substring match, so a PREFIX of these names matches
# both variants — always select the exact name, never the first result.
BUNDLE_NAME = "OSM Quick Start (Volve Six State)"
BUNDLE_NAME_EMBEDDINGS = "OSM Quick Start (Volve Six State, Embeddings)"

POLL_INTERVAL_S = 15
TIMEOUT_S = 2 * 60 * 60   # ~2 min uncontended, ~30 min contended; generous margin


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def versioned(endpoint):
    """Return the endpoint in the form the client expects: WITH the /vX.Y suffix.

    The client uses api_endpoint verbatim for the files API (/v0.5/files) and
    strips the version itself for the versionless agents API. Passing a bare
    root therefore breaks uploads while bundle calls keep working — an empty
    `ApiError: {}` that looks like anything but a missing version path.

    Accept either form so one .env works for every skill in this repo: the
    model skills ship ATAI_API_ENDPOINT with /v0.5, the agent skills without.
    """
    endpoint = endpoint.rstrip("/")
    return endpoint if re.search(r"/v[0-9]+(\.[0-9]+)*$", endpoint) else f"{endpoint}/v0.5"


def resolve_bundle(client, name):
    """Find the pre-packaged bundle id for an EXACT name.

    `query` is a case-insensitive substring match over name and id, so a prefix
    of a bundle name returns every variant — querying "OSM Quick Start" returns
    the Embeddings bundle too, and it sorts newest-first. Taking the first row
    would silently run the wrong bundle (314 MB of embeddings instead of a
    221 KB prediction file), so match the name exactly and prefer the canonical
    (platform-published) bundle over any same-named org bundle.
    """
    rows = client.agents.bundles.list(query=name, limit=50).get("data", [])
    exact = [b for b in rows if b.get("name") == name]
    if not exact:
        seen = [b.get("name") for b in rows]
        sys.exit(f"no bundle named {name!r} found (query returned {seen}). "
                 f"Is it published in this deployment? Pass --bundle-id to "
                 f"pin one, or contact support@archetypeai.dev.")
    for bundle in exact:
        if bundle.get("is_canonical"):
            return bundle["id"]
    return exact[0]["id"]


def watch(client, agent_id):
    """Poll until terminal, echoing new audit events as they arrive.

    Deliberately NOT client.agents.instances.wait_until_done(): that returns as
    soon as `status` is terminal, and `status` is not trustworthy here. A run
    whose output exists can report `failed` (the job poller flakes), so the
    caller must judge success from /results, not from this return value.
    """
    deadline = time.time() + TIMEOUT_S
    seen = set()
    status = "running"
    while status == "running" and time.time() < deadline:
        status = client.agents.instances.get(agent_id).get("status")
        for event in client.agents.instances.get_events(agent_id).get("data", []):
            marker = f"{event.get('created_at')}{event.get('message')}"
            if marker not in seen:
                seen.add(marker)
                print(f"  [{event.get('level', 'info')}] {event.get('created_at', '')}  "
                      f"{event.get('message', '')}")
        if status == "running":
            time.sleep(POLL_INTERVAL_S)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=DEFAULT_CSV, help="input CSV to run inference on")
    parser.add_argument("--embeddings", action="store_true",
                        help="run the Embeddings bundle: adds one 768-d "
                             "embedding_{variate} column per channel (~1,400x "
                             "the output size)")
    parser.add_argument("--bundle-id",
                        help="skip name resolution and run this bundle id")
    parser.add_argument("--window-size", type=int, default=16,
                        help="the bundle's window size, for scoring only")
    parser.add_argument("--output", default="osm-output.csv",
                        help="local path to save the classified-windows CSV")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("ATAI_API_KEY")
    endpoint = os.environ.get("ATAI_API_ENDPOINT")
    if not api_key:
        sys.exit("ATAI_API_KEY is not set")
    if not endpoint:
        sys.exit("ATAI_API_ENDPOINT is not set")
    client = ArchetypeAI(api_key, api_endpoint=versioned(endpoint))

    # 1. Upload the input CSV. Source connectors use the returned file_id (the
    #    filename), NOT the fil_ uid — client.agents.bundles.run() builds the
    #    data ref for us.
    print(f"uploading {args.csv} ...")
    uploaded = client.files.local.upload(args.csv)
    file_id = uploaded["file_id"]
    print(f"  file_id={file_id}  file_uid={uploaded.get('file_uid')}")

    # 2. Resolve the pre-packaged bundle. By name (portable) unless an id is
    #    pinned. The bundle already pins the classifier and windowing.
    name = BUNDLE_NAME_EMBEDDINGS if args.embeddings else BUNDLE_NAME
    if args.bundle_id:
        bundle_id = args.bundle_id
        print(f"using bundle {bundle_id}")
    else:
        bundle_id = resolve_bundle(client, name)
        print(f"resolved {name!r} -> {bundle_id}")

    # 3. Run the bundle. No sink, so the runner writes one output per input.
    print("starting agent run ...")
    agent = client.agents.bundles.run(bundle_id, source=[file_id])
    agent_id = agent["id"]
    print(f"  agent_id={agent_id}  status={agent.get('status')}")

    # 4. Poll until terminal, streaming audit events.
    status = watch(client, agent_id)

    # 5. Fetch results and download the output CSV. Checked even on
    #    status=failed: a flaky job poller can mark a successful run failed —
    #    if the output is there, the run succeeded.
    results = client.agents.instances.get_results(agent_id)
    outputs = results.get("data", [])
    if not outputs:
        sys.exit(f"run produced no output: status={status}")
    if status != "completed":
        print(f"  note: status={status} but output exists — treating as succeeded")
    print(f"results ({len(outputs)}):")
    for output in outputs:
        inner = output["data"]
        print(f"  {inner['filename']}  ({inner['num_bytes']} bytes)")

    filename = outputs[0]["data"]["filename"]
    client.files.local.download(filename, args.output)
    print(f"saved output to {args.output}")

    # 6. If the input ships a ground-truth sidecar (the sample slice does),
    #    score the predictions: each window is judged against the label of its
    #    end row (the row its timestamp points at).
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
