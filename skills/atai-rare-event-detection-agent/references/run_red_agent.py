#!/usr/bin/env python3
"""Run the platform's RED demo blueprint end to end, then score the output.

    python3 run_red_agent.py                         # playground-red on the bundled slice
    python3 run_red_agent.py --csv my_slice.csv      # your own prepared CSV
    python3 run_red_agent.py --blueprint-id blp_...  # a blueprint in your project (Path 2's)
    python3 run_red_agent.py --score-only red-output.csv

By default this copies the platform's demo blueprint `playground-red` into a project
(created on first use, reused after), packages the copy, deploys the package as `batch`,
runs one session over the input file and downloads its output. The session, deployment
and package are deleted afterwards; the project and the copy are kept for the next run.

`playground-red` is the canonical `red` blueprint with a pump-breakdown classifier the
platform fitted, its windowing (window and step 200) and `output_embeddings` pinned. Its
classes are `pump_nshot_normal` and `pump_nshot_pump_breakdown`; the scorer matches them
to the sidecar's `normal` and `pump_breakdown`. For another asset, fit a classifier with
Path 2 and pass its promoted blueprint's id.

Scoring reports three views, because window averages mislead when the positive
class is under 1% of windows: window-level P/R/F1 under the design's majority
rule, per-incident detection with latency, and the false-alarm rate measured
only on windows containing zero fault rows.

Stdlib only, on the same HTTP helper as Path 2 (red_lifecycle/atai_http.py). Reads
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
sys.path.insert(0, os.path.join(SCRIPT_DIR, "red_lifecycle"))
from atai_http import (TERMINAL, agents, check_auth, create_package, deploy, end,  # noqa: E402
                       lifecycle_url, load_dotenv, open_session, project_url, request,
                       result_url, upload_file)

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_CSV = os.path.join(SCRIPT_DIR, "sample_data", "pump_eval_inc04.csv")

# The platform's RED demo blueprint, resolved by key: its blp_ id differs per deployment.
DEMO_BLUEPRINT = "playground-red"
# Only a blueprint in a project can be packaged and deployed, so the demo is copied into
# this project under this key once, and the copy reused.
PROJECT = "RED skill Path 1"
COPY_KEY = "playground-red-skill"
# The demo's window, for scoring when the blueprint doesn't say.
DEFAULT_WINDOW = 200
# Class names are NOT hardcoded: `normal` is the conventional baseline label and
# the rare-event class is inferred from the label sidecar, so this runner works
# for any catalog. Override with --normal-class / --fault-class.
DEFAULT_NORMAL = "normal"


# --------------------------------------------------------------------------
# lifecycle: project -> copy -> package -> batch deployment -> session
# --------------------------------------------------------------------------

def project_id(name: str = PROJECT) -> str:
    """Your project of that name, created on first use."""
    found = [p for p in request("GET", f"{project_url()}?limit=1000")["data"] if p["name"] == name]
    if found:
        return found[0]["id"]
    pid = request("POST", project_url(), {"name": name})["id"]
    print(f"created project {pid} ({name!r})")
    return pid


def demo_copy(pid: str) -> dict:
    """The project's copy of the demo blueprint, copied on first use. Copying needs only the
    right to modify the project. A key in a project resolves only with ?project_id=."""
    found = request("GET", f"{agents()}/blueprints?project_id={pid}&key={COPY_KEY}")["data"]
    if found:
        return found[0]
    bp = request("POST", f"{agents()}/blueprints/{DEMO_BLUEPRINT}/copy",
                 {"project_id": pid, "blueprint_key": COPY_KEY})
    print(f"copied {DEMO_BLUEPRINT} into {pid} as {COPY_KEY} ({bp['id']})")
    return bp


def blueprint_window(blueprint_id: str) -> int:
    """The window the blueprint runs at, to rebuild each output row's span when scoring."""
    doc = request("GET", f"{agents()}/blueprints/{blueprint_id}").get("document") or {}
    return int((doc.get("values") or {}).get("window_size") or DEFAULT_WINDOW)


def upload(path: str) -> str:
    """Upload under a timestamped name and return the file id.

    File ids ARE filenames, so re-uploading a name replaces the record and orphans the
    object any in-flight run already resolved: that run then dies naming a UUID and
    nothing else."""
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


# The platform can report a run completed while its output file is still being
# written (a known issue), so the output counts as complete only once its last
# window ends at the input's end: within one window span (an output row's
# finish - start) plus a minute.
END_SLACK_S = 60
OUTPUT_WAIT_S = 600


# csv.reader caps one field at 128 KB by default. An `embedding_{variate}`
# column is a whole vector in a single quoted field, so the cap is reachable —
# and it raises rather than truncating, which at least fails loudly.
csv.field_size_limit(1 << 31)


def tail_row(path: str) -> tuple[list[str], list[str]]:
    """A CSV's header and last row, read from the file's end.

    The window grows until it holds a complete final record. A fixed window is
    correct only while every row fits inside it, and the failure is silent: a
    window narrower than one row yields a fragment of a field, which still
    parses as a number and reads as an output that stopped early. An
    embeddings row already runs to ~85 KB, so the margin is narrower than it
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


def output_reaches_end(output_csv: str, input_csv: str) -> bool | None:
    """Whether the output's last window ends at the input's last row. None when
    the output carries no window span or the timestamps aren't numbers."""
    try:
        head, row = tail_row(output_csv)
        finish = float(row[head.index("finish_timestamp")])
        span = finish - float(row[head.index("start_timestamp")])
        end = float(tail_row(input_csv)[1][0])
    except (ValueError, IndexError, StopIteration):
        return None
    return end - finish <= max(span, 0) + END_SLACK_S


def wait_until_complete(save, output_csv: str, input_csv: str) -> None:
    """save() downloads the output; again every 15 s while it ends early."""
    deadline = time.time() + OUTPUT_WAIT_S
    while output_reaches_end(output_csv, input_csv) is False:
        if time.time() > deadline:
            print(f"  WARNING: the output still ends before the input "
                  f"{OUTPUT_WAIT_S // 60} min after the run finished; rerun to download it again")
            return
        print("  run finished; output still being written, downloading again in 15 s")
        time.sleep(15)
        save()


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


def download_results(session_id: str, out_path: str, input_csv: str | None = None) -> str | None:
    def _fetch():
        return request("GET", lifecycle_url("sessions", session_id) + "/results").get("data", [])

    items = complete_results(_fetch)
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
    fetch(inner["ref"], out_path)
    if input_csv:
        wait_until_complete(lambda: fetch(inner["ref"], out_path), out_path, input_csv)
    return out_path


def run(blueprint_id: str, file_id: str, out_path: str, input_csv: str, name: str) -> str | None:
    """Package the blueprint, deploy it as batch, run one session over the file, download its
    output; then delete the session, deployment and package, which each hold the next."""
    t0 = time.time()
    pkg = create_package(blueprint_id, name)
    print(f"package {pkg['id']} ready ({time.time() - t0:.0f} s)")
    dpl = ses = None
    try:
        dpl = deploy(pkg["id"], "batch")
        print(f"batch deployment {dpl['id']} ready")
        ses = open_session(dpl["id"], file_id)
        print(f"session {ses['id']}")
        done = poll(ses["id"])
        if done.get("status") != "completed":
            for ev in last_events(ses["id"]):
                print(f"  event: {ev}")
            sys.exit(f"session did not complete: {done.get('status')} {done.get('error') or ''}")
        print(f"  completed ({time.time() - t0:.0f} s from packaging)")
        return download_results(ses["id"], out_path, input_csv)
    finally:
        for kind, obj in (("sessions", ses), ("deployments", dpl), ("packages", pkg)):
            if obj:
                end(kind, obj["id"])
        print("deleted the session, deployment and package (the project and blueprint stay)")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pick(row: dict, *candidates: str) -> str | None:
    for c in candidates:
        if c in row:
            return c
    return None


def same_class(predicted: str, label: str) -> bool:
    """Whether an output class is the sidecar's label. A blueprint's class names can carry a
    prefix the sidecar doesn't: `playground-red` predicts `pump_nshot_pump_breakdown` for the
    sidecar's `pump_breakdown`."""
    return predicted == label or predicted.endswith("_" + label)


def score(output_path: str, labels_path: str, window_size: int,
          normal: str = DEFAULT_NORMAL, fault: str | None = None) -> None:
    """Score per-window predictions against a row-aligned label sidecar."""
    preds = read_csv_rows(output_path)
    truth = read_csv_rows(labels_path)
    if not preds:
        print("output file has no rows")
        return

    # Infer the rare-event class from the sidecar: whatever is not `normal`.
    if fault is None:
        others = sorted({r["label"] for r in truth} - {normal})
        if len(others) > 1:
            print(f"multiple non-{normal} labels {others} — pass --fault-class")
            return
        # An all-normal slice is legitimate: score it, expect no positives.
        fault = others[0] if others else "\x00_no_fault_label"
    NORMAL, FAULT = normal, fault

    pred_col = pick(preds[0], "predicted_state", "predicted_label", "predicted_class",
                    "prediction", "label")
    # Newer blueprints emit the window span as finish_/start_timestamp; older
    # ones a single `timestamp`. Either way the row is keyed to the window END.
    ts_col = pick(preds[0], "finish_timestamp", "timestamp", "time")
    inv_col = pick(preds[0], "invalid", "invalid_state")
    if not pred_col or not ts_col:
        print(f"could not find prediction/timestamp columns in {list(preds[0])}")
        return

    # Row index by timestamp, so each output row (keyed to its window end) can
    # be mapped back to the window's row span. The platform emits timestamps as
    # floats ("1530962520.0") while the sidecar carries integer seconds, so
    # join on the numeric value rather than the string.
    def as_ts(value: str) -> int:
        return int(float(value))

    index_of = {as_ts(r["timestamp"]): i for i, r in enumerate(truth)}
    labels = [r["label"] for r in truth]

    tp = fp = fn = tn = 0
    invalid = unmatched = 0
    fault_windows_pred: list[tuple[int, int]] = []
    normal_window_total = 0
    false_alarms = 0

    for row in preds:
        if inv_col and str(row.get(inv_col, "")).strip().lower() in ("true", "1"):
            invalid += 1
            continue
        end = index_of.get(as_ts(row[ts_col]))
        if end is None:
            unmatched += 1
            continue
        lo = max(0, end - window_size + 1)
        span = labels[lo:end + 1]
        if not span:
            unmatched += 1
            continue
        n_fault = sum(1 for s in span if s == FAULT)
        # Design-doc labelling: majority wins; if neither holds a majority the
        # rare event takes precedence over normal.
        truth_label = FAULT if n_fault * 2 >= len(span) else NORMAL
        is_fault_pred = same_class(row[pred_col].strip(), FAULT)

        if truth_label == FAULT and is_fault_pred:
            tp += 1
        elif truth_label == FAULT and not is_fault_pred:
            fn += 1
        elif truth_label == NORMAL and is_fault_pred:
            fp += 1
        else:
            tn += 1

        if n_fault == 0:
            normal_window_total += 1
            if is_fault_pred:
                false_alarms += 1
        if is_fault_pred:
            fault_windows_pred.append((lo, end + 1))

    scored = tp + fp + fn + tn
    print(f"\nevaluation vs {os.path.basename(labels_path)} "
          f"({scored} scored windows, {invalid} invalid, {unmatched} unmatched, "
          f"window {window_size})")
    if not scored:
        return

    def prf(tp_: int, fp_: int, fn_: int) -> tuple[float, float, float]:
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    fp_p, fp_r, fp_f = prf(tp, fp, fn)
    tn_p, tn_r, tn_f = prf(tn, fn, fp)
    print(f"  accuracy: {(tp + tn) / scored:.4f}")
    print(f"  {FAULT:<16} precision={fp_p:.4f} recall={fp_r:.4f} f1={fp_f:.4f} "
          f"(n={tp + fn})")
    print(f"  {NORMAL:<16} precision={tn_p:.4f} recall={tn_r:.4f} f1={tn_f:.4f} "
          f"(n={tn + fp})")
    print(f"  macro-F1: {(fp_f + tn_f) / 2:.4f}")
    print(f"  confusion: tp={tp} fp={fp} fn={fn} tn={tn}")
    if normal_window_total:
        print(f"  false-alarm rate on all-normal windows: "
              f"{false_alarms / normal_window_total:.4f} "
              f"({false_alarms}/{normal_window_total})")

    # Incident-level detection: did we catch each contiguous fault episode?
    episodes, start = [], None
    for i, lab in enumerate(labels):
        if lab == FAULT and start is None:
            start = i
        elif lab != FAULT and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, len(labels)))

    if episodes:
        print(f"  incident-level detection ({len(episodes)} episode(s) in this slice):")
        for a, b in episodes:
            overlapping = [w for w in fault_windows_pred if w[0] < b and a < w[1]]
            if overlapping:
                first_end = min(w[1] for w in overlapping) - 1
                latency = max(0, first_end - a)
                print(f"    rows {a}-{b} (len {b - a}): DETECTED, "
                      f"{len(overlapping)} window(s), first at row {first_end} "
                      f"(+{latency} samples after onset)")
            else:
                print(f"    rows {a}-{b} (len {b - a}): MISSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="prepared input CSV (defaults to the bundled sample "
                         "slice, resolved next to this script). A ground-truth "
                         "sidecar named <stem>_labels.csv next to it enables "
                         "scoring.")
    ap.add_argument("--labels", default=None,
                    help="ground-truth sidecar (default: <csv stem>_labels.csv)")
    ap.add_argument("--blueprint-id", default=None,
                    help="run this blueprint (blp_…, in one of your projects, e.g. Path 2's "
                         f"promoted one) instead of the project's copy of {DEMO_BLUEPRINT}")
    ap.add_argument("--project", default=PROJECT,
                    help=f"the project to copy {DEMO_BLUEPRINT} into (default {PROJECT!r})")
    ap.add_argument("--window-size", type=int, default=None,
                    help="the window to rebuild each row's span with when scoring "
                         f"(default: the blueprint's own, {DEFAULT_WINDOW} for the demo)")
    ap.add_argument("--name", default="RED Path 1", help="name of the package built for the run")
    ap.add_argument("--output", default="red-output.csv")
    ap.add_argument("--normal-class", default=DEFAULT_NORMAL,
                    help="baseline class name in the label sidecar")
    ap.add_argument("--fault-class", default=None,
                    help="rare-event class name; inferred from the sidecar when omitted")
    ap.add_argument("--score-only", metavar="OUTPUT_CSV",
                    help="skip the platform, score an already-downloaded output")
    args = ap.parse_args()

    labels = args.labels or args.csv.rsplit(".", 1)[0] + "_labels.csv"

    if args.score_only:
        if not os.path.exists(labels):
            sys.exit(f"no label sidecar at {labels}")
        score(args.score_only, labels, args.window_size or DEFAULT_WINDOW,
              args.normal_class, args.fault_class)
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

    if os.path.exists(labels):
        score(args.output, labels, window, args.normal_class, args.fault_class)
    else:
        print(f"  no labels at {labels} — skipping scoring")


if __name__ == "__main__":
    main()
