#!/usr/bin/env python3
"""Run the platform's OSM demo blueprint end to end, then score the output.

    python3 run_osm_agent.py                         # playground-osm-drilling on the sample slice
    python3 run_osm_agent.py --csv my_slice.csv      # your own prepared CSV
    python3 run_osm_agent.py --blueprint-id blp_...  # a blueprint in your project (Path 2's)

By default this copies the platform's demo blueprint `playground-osm-drilling` into a project
(created on first use, reused after), packages the copy, deploys the package as `batch`,
runs one session over the input file and downloads its output. The session, deployment
and package are deleted afterwards; the project and the copy are kept for the next run.

`playground-osm-drilling` is the canonical `osm` blueprint with a six-state classifier the
platform fitted on Volve drilling telemetry, its windowing (window and step 40) and
`output_embeddings` pinned. Its input validation keeps the blueprint default
`sample_rate_interval_tolerance: 0.05`, so the input must be on an exact sample grid: a
window whose row gaps differ by more than 5% comes back INVALID_STATE.

If the input ships a `<input>_labels.csv` sidecar (timestamp, label), the run is scored:
each window against the label of its end row.

Stdlib only, on the same HTTP helper as Path 2 (osm_lifecycle/atai_http.py). Reads
ATAI_API_KEY and ATAI_API_ENDPOINT from the environment or the nearest .env.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "osm_lifecycle"))
from atai_http import (TERMINAL, agents, check_auth, create_package, deploy, deployment_ready, end,  # noqa: E402
                       lifecycle_url, load_dotenv, open_session, package_ready, project_url, request,
                       result_url, upload_file)

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_CSV = os.path.join(SCRIPT_DIR, "sample_data", "drilling_volve_f14.csv")

# The platform's OSM demo blueprint, resolved by key: its blp_ id differs per deployment.
DEMO_BLUEPRINT = "playground-osm-drilling"
# Only a blueprint in a project can be packaged and deployed, so the demo is copied into
# this project under this key once, and the copy reused.
PROJECT = "OSM skill Path 1"
COPY_KEY = "playground-osm-drilling-skill"
# The demo's window, for scoring when the blueprint doesn't say.
DEFAULT_WINDOW = 40

# The platform can report a run completed while its output file is still being written
# (a known issue), so the output counts as complete only once its last window ends at
# the input's last row.
END_SLACK_S = 60
OUTPUT_WAIT_S = 600


# --------------------------------------------------------------------------
# lifecycle: project -> copy -> package -> batch deployment -> session
# --------------------------------------------------------------------------

def project_id(name: str = PROJECT) -> str:
    """Your project of that name, created on first use."""
    found, after = [], ""
    while True:                                    # every page: the cursor is opaque, passed back as is
        page = request("GET", f"{project_url()}?limit=1000{after}")
        found += [p for p in page.get("data", []) if p["name"] == name]
        if not page.get("has_more"):
            break
        after = f"&after={page['next_cursor']}"
    if found:
        return found[0]["id"]
    pid = request("POST", project_url(), {"name": name})["id"]
    print(f"created project {pid} ({name!r})")
    return pid


def demo_copy(pid: str) -> dict:
    """The project's copy of the demo blueprint, copied on first use. Copying needs only the
    right to modify the project. A key in a project resolves only with ?project_id=."""
    found = [bp for bp in request("GET", f"{agents()}/blueprints?project_id={pid}&key={COPY_KEY}")["data"]
             if bp.get("blueprint_key") == COPY_KEY]
    if found:
        return found[0]
    bp = request("POST", f"{agents()}/blueprints/{DEMO_BLUEPRINT}/copy",
                 {"project_id": pid, "blueprint_key": COPY_KEY})
    print(f"copied {DEMO_BLUEPRINT} into {pid} as {COPY_KEY} ({bp['id']})")
    return bp


def blueprint_window(blueprint_id: str) -> int:
    """The window the blueprint runs at, for the steady-state view when scoring."""
    doc = request("GET", f"{agents()}/blueprints/{blueprint_id}").get("document") or {}
    return int((doc.get("values") or {}).get("window_size") or DEFAULT_WINDOW)


def upload(path: str) -> str:
    """Upload under a timestamped name and return the file id.

    File ids ARE filenames, so re-uploading a name replaces the record and orphans the
    object any in-flight run already resolved."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    return upload_file(path, f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.csv")["file_id"]


def poll(session_id: str, timeout_s: int = 2 * 3600, interval_s: int = 5) -> dict:
    """The session, once terminal. A session's `status` is authoritative and reaches
    completed, failed or cancelled, with `error` saying why."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        ses = request("GET", lifecycle_url("sessions", session_id))
        if ses.get("status") != last:
            last = ses.get("status")
            print(f"  {last}")
        if last in TERMINAL:
            return ses
        time.sleep(interval_s)
    return {"status": "timeout"}


def last_events(session_id: str, n: int = 5) -> list:
    """The session's latest events, newest first."""
    return request("GET", lifecycle_url("sessions", session_id) + "/events?limit=" + str(n)).get("data", [])


# csv.reader caps one field at 128 KB by default. An `embedding_{variate}`
# column is a whole vector in a single quoted field, so the cap is reachable —
# and it raises rather than truncating, which at least fails loudly.
csv.field_size_limit(1 << 31)


def tail_row(path):
    """A CSV's header and last row, read from the file's end.

    The window grows until it holds a complete final record. A fixed window is
    correct only while every row fits inside it, and the failure is silent: a
    window narrower than one row yields a fragment of a field, which still
    parses as a number and reads as an output that stopped early. An
    Embeddings row already runs to ~75 KB, so the margin is narrower than it
    looks. Growing also reads LESS in the common case — 4 KiB rather than a
    megabyte.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = next(csv.reader([f.readline().decode()]))
        window = 4096
        while True:
            f.seek(max(0, size - window))
            lines = f.read().strip().split(b"\n")
            # A leading partial line proves the trailing one is whole.
            if len(lines) > 1 or window >= size:
                return header, next(csv.reader([lines[-1].decode()]))
            window *= 4


def last_window_end(output_csv):
    """The output's last `finish_timestamp`, for logging."""
    try:
        header, row = tail_row(output_csv)
        return row[header.index("finish_timestamp")]
    except (IndexError, StopIteration, ValueError):
        return "?"


def output_reaches_end(output_csv, input_csv):
    """Whether the output's last window ends at the input's last row: within one window span
    (an output row's finish - start; rows after the last whole window are never output) plus
    END_SLACK_S. None when the timestamps aren't epoch numbers (no check then)."""
    try:
        header, row = tail_row(output_csv)
        finish = float(row[header.index("finish_timestamp")])
        span = finish - float(row[header.index("start_timestamp")])
        end = float(tail_row(input_csv)[1][0])
    except (ValueError, IndexError, StopIteration):
        return None
    return end - finish <= max(span, 0) + END_SLACK_S


ARTIFACT_WAIT_S = 600          # how long to wait for artifacts after a run reports completed


def artifact_status(item):
    """An output's own completion status, or None if the platform reports none.

    A run reports `completed` while its artifacts are still being written. One
    measured instance: an output held 16,018 of an eventual 50,547 bytes at the
    moment its run reported completed -- 318 rows of ~997. Reading it gives a
    well-formed file that is simply short, so nothing downstream notices. The
    console shows the same pairing, run COMPLETED beside artifact PARTIAL.
    """
    return (item.get("data", {}).get("file_attributes", {})
                .get("job_output", {}).get("status"))


def complete_results(fetch, wait_s=ARTIFACT_WAIT_S):
    """The run's outputs, once every one reports complete.

    Needs nothing but the session id -- unlike a content check, which has to
    compare the output against the input and so only works for the client that
    produced the run.
    """
    deadline = time.time() + wait_s
    waited = False
    while True:
        items = fetch()
        pending = [i for i in items if artifact_status(i) not in (None, "complete")]
        if not pending:
            if waited:
                print("  every artifact reports complete")
            return items
        if time.time() > deadline:
            print(f"  WARNING {len(pending)} artifact(s) still incomplete after "
                  f"{wait_s // 60} min; collecting anyway")
            return items
        waited = True
        print(f"  {len(items) - len(pending)} of {len(items)} artifacts ready; waiting")
        time.sleep(15)


def fetch(ref: str, out_path: str) -> None:
    req = urllib.request.Request(result_url(ref))
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    with urllib.request.urlopen(req, timeout=300) as resp, open(out_path, "wb") as fh:
        fh.write(resp.read())


def download_results(session_id: str, out_path: str, input_csv: str) -> str | None:
    """The session's output, downloaded again every 15 s while it ends before the input does."""
    items = complete_results(
        lambda: request("GET", lifecycle_url("sessions", session_id) + "/results").get("data", []))
    if not items:
        print("  /results is EMPTY — the session completed but produced nothing.")
        return None
    # Each item nests its fields one level down: {id, port_name, data: {ref, filename,
    # num_bytes, ...}}. `ref` is a root path, /api/files/<fil_…>/content, which already
    # carries /api: result_url resolves it against the deployment root.
    inner = items[0].get("data") or {}
    if not inner.get("ref"):
        print(f"  results present but no ref in the entry: {items[0]!r}")
        return None
    print(f"  results ({len(items)}): {inner.get('filename', '?')}  ({inner.get('num_bytes', '?')} bytes)")
    deadline = time.time() + OUTPUT_WAIT_S
    while True:
        fetch(inner["ref"], out_path)
        if output_reaches_end(out_path, input_csv) is not False:
            return out_path
        if time.time() > deadline:
            print(f"  WARNING output still ends early {OUTPUT_WAIT_S // 60} min after the run finished: "
                  f"last window {last_window_end(out_path)}, input ends {tail_row(input_csv)[1][0]}")
            return out_path
        print(f"  finished; output still being written (last window {last_window_end(out_path)}, "
              f"input ends {tail_row(input_csv)[1][0]})")
        time.sleep(15)


def run(blueprint_id: str, file_id: str, out_path: str, input_csv: str, name: str) -> str | None:
    """Package the blueprint, deploy it as batch, run one session over the file, download its
    output; then delete the session, deployment and package, which each hold the next."""
    t0 = time.time()
    pkg = dpl = ses = None
    try:
        pkg = create_package(blueprint_id, name)
        print(f"package {pkg['id']}")              # each id printed before waiting on it
        package_ready(pkg["id"])
        print(f"  ready ({time.time() - t0:.0f} s)")
        dpl = deploy(pkg["id"], "batch")
        print(f"batch deployment {dpl['id']}")
        deployment_ready(dpl["id"])
        ses = open_session(dpl["id"], [file_id])
        print(f"session {ses['id']}")
        done = poll(ses["id"])
        if done.get("status") != "completed":
            for ev in last_events(ses["id"]):
                print(f"  event: {ev}")
            sys.exit(f"session did not complete: {done.get('status')} {done.get('error') or ''}")
        print(f"  completed ({time.time() - t0:.0f} s from packaging)")
        return download_results(ses["id"], out_path, input_csv)
    finally:
        deleted, left = [], []
        for kind, obj in (("sessions", ses), ("deployments", dpl), ("packages", pkg)):
            if not obj:
                continue
            try:                                   # each tried even when one before it fails
                end(kind, obj["id"])
                deleted.append(obj["id"])
            except (RuntimeError, SystemExit, OSError) as e:
                left.append(obj["id"])
                print(f"could not delete {obj['id']}: {str(e).splitlines()[0][:200]}")
        if deleted:
            print(f"deleted {', '.join(deleted)} (the project and blueprint stay)")
        if left:
            print(f"left on the platform: {', '.join(left)}; delete them with DELETE /agents/<kind>/<id>")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=DEFAULT_CSV, help="input CSV to run inference on")
    parser.add_argument("--labels", default=None,
                        help="ground-truth sidecar (default: <csv stem>_labels.csv)")
    parser.add_argument("--blueprint-id", default=None,
                        help="run this blueprint (blp_…, in one of your projects, e.g. Path 2's "
                             f"promoted one) instead of the project's copy of {DEMO_BLUEPRINT}")
    parser.add_argument("--project", default=PROJECT,
                        help=f"the project to copy {DEMO_BLUEPRINT} into (default {PROJECT!r})")
    parser.add_argument("--window-size", type=int, default=None,
                        help="window for the steady-state view when scoring "
                             f"(default: the blueprint's own, {DEFAULT_WINDOW} for the demo)")
    parser.add_argument("--name", default="OSM Path 1", help="name of the package built for the run")
    parser.add_argument("--output", default="osm-output.csv",
                        help="local path to save the classified-windows CSV")
    parser.add_argument("--score-only", metavar="OUTPUT_CSV", default=None,
                        help="skip the platform and score an output you already have")
    args = parser.parse_args()

    labels = args.labels or args.csv.rsplit(".", 1)[0] + "_labels.csv"

    if args.score_only:
        if not os.path.exists(labels):
            sys.exit(f"no label sidecar at {labels}")
        evaluate(args.score_only, labels, args.window_size or DEFAULT_WINDOW)
        return

    load_dotenv()
    check_auth()

    if args.blueprint_id:
        blueprint_id = args.blueprint_id
        print(f"blueprint {blueprint_id}")
    else:
        pid = project_id(args.project)
        blueprint_id = demo_copy(pid)["id"]
        print(f"blueprint {blueprint_id} ({COPY_KEY}, the copy of {DEMO_BLUEPRINT} in {pid})")
    window = args.window_size or blueprint_window(blueprint_id)

    print(f"uploading {args.csv}")
    file_id = upload(args.csv)
    print(f"  file_id={file_id}")

    if not run(blueprint_id, file_id, args.output, args.csv, args.name):
        sys.exit("no results to score")
    print(f"  saved {args.output}")
    summarize(args.output)

    if os.path.exists(labels):
        evaluate(args.output, labels, window)
    else:
        print(f"  no labels at {labels} — skipping scoring")


def summarize(pred_csv):
    """Windows, invalid windows and predicted states, so an unlabelled run says something."""
    rows = list(csv.DictReader(open(pred_csv)))
    states = {}
    for r in rows:
        states[r["predicted_state"]] = states.get(r["predicted_state"], 0) + 1
    print(f"  {len(rows)} windows, {states.get('INVALID_STATE', 0)} invalid; predicted: "
          + ", ".join(f"{s} {n}" for s, n in sorted(states.items(), key=lambda x: -x[1])))


def evaluate(pred_csv, labels_csv, window):
    truth_rows = list(csv.DictReader(open(labels_csv)))
    tcol = next(iter(truth_rows[0]))          # the timestamp column: DATE_TIME, timestamp, ...
    truth = {int(float(r[tcol])): r["label"] for r in truth_rows}

    # steady windows: all rows one label, no timestamp seams inside
    ts = [int(float(r[tcol])) for r in truth_rows]
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
