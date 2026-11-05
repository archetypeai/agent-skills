#!/usr/bin/env python3
"""Run the platform's AD demo blueprint end to end, then score the output.

    python3 run_ad_agent.py                         # playground-ad on the bundled slice
    python3 run_ad_agent.py --csv my_slice.csv      # your own prepared CSV
    python3 run_ad_agent.py --blueprint-id blp_...  # a blueprint in your project (Path 2's)
    python3 run_ad_agent.py --score-only ad-output.csv

By default this copies the platform's demo blueprint `playground-ad` into a project
(created on first use, reused after), packages the copy, deploys the package as `batch`,
runs one session over the input file and downloads its output. The session, deployment
and package are deleted afterwards; the project and the copy are kept for the next run.

`playground-ad` is the canonical `ad` blueprint with a detector the platform fitted on one
bearing's healthy baseline, its threshold (1.762), windowing and input-validation settings
pinned, and `output_embeddings` on. The bundled sample slice is from that same bearing: a
detector encodes ONE asset's notion of normal. For another asset, fit one with Path 2 and
pass its promoted blueprint's id.

Scoring is deliberately not precision/recall: run-to-failure data has no per-window ground
truth. See `score()` and SKILL.md.

Stdlib only, on the same HTTP helper as Path 2 (ad_lifecycle/atai_http.py). Reads
ATAI_API_KEY and ATAI_API_ENDPOINT from the environment or the nearest .env.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "ad_lifecycle"))
from atai_http import (TERMINAL, agents, check_auth, create_package, deploy, deployment_ready, end,  # noqa: E402
                       lifecycle_url, load_dotenv, open_session, package_ready, project_url, request,
                       result_url, upload_file)

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_CSV = os.path.join(SCRIPT_DIR, "sample_data",
                           "bearing_eval_set2_brg1_transition.csv")

# The platform's AD demo blueprint, resolved by key: its blp_ id differs per deployment.
DEMO_BLUEPRINT = "playground-ad"
# Only a blueprint in a project can be packaged and deployed, so the demo is copied into
# this project under this key once, and the copy reused.
PROJECT = "AD skill Path 1"
COPY_KEY = "playground-ad-skill"
# The demo detector's threshold, for scoring when the blueprint doesn't say.
DEFAULT_THRESHOLD = 1.762
# Three consecutive snapshots over the line. One window is noise; the same
# rule an independent offline scorer used, so numbers stay comparable.
SUSTAIN = 3


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


def blueprint_threshold(blueprint_id: str) -> float:
    """The threshold the blueprint labels at, for scoring the same way client-side."""
    doc = request("GET", f"{agents()}/blueprints/{blueprint_id}").get("document") or {}
    return float((doc.get("values") or {}).get("threshold") or DEFAULT_THRESHOLD)


def upload(path: str) -> str:
    """Upload under a timestamped name and return the file id.

    File ids ARE filenames, so re-uploading a name replaces the record and orphans the
    object any in-flight run already resolved: that run then dies naming a UUID and
    nothing else."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    return upload_file(path, f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.csv")["file_id"]


def poll(session_id: str, timeout_s: int = 3600, interval_s: int = 5) -> dict:
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
        # The single most important check in this script. A run whose windows
        # were all invalidated reports `completed` with NO results.
        print("  /results is EMPTY — the session completed but produced nothing.")
        print("  Almost always validate_monotonic_timestamps: every window was")
        print("  marked invalid. See SKILL.md.")
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


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score(output_path: str, labels_path: str, threshold: float) -> dict:
    """Lead time and crossing rate — NOT precision/recall.

    Run-to-failure data has no per-window ground truth: the binary labels in
    circulation are a hand-placed cut on a gradual curve, so precision against
    them mostly measures where that line was drawn. What IS ground truth is the
    documented outcome and the position of each window in the asset's life.
    """
    rows = list(csv.DictReader(open(output_path)))
    labels = list(csv.DictReader(open(labels_path)))
    if not rows:
        return {"error": "output has no rows"}

    scores = [float(r["anomaly_score"]) for r in rows if r.get("anomaly_score")]
    n_invalid = sum(1 for r in rows if str(r.get("invalid", "")).lower() == "true")

    # Labels are one row per WINDOW and carry the snapshot index explicitly;
    # windows-per-snapshot is whatever the ratio says. Indexing by position
    # instead reads the wrong part of the asset's life.
    per_snapshot: dict[int, float] = {}
    for lab in labels:
        idx = int(lab["snapshot_index"])
        per_snapshot.setdefault(idx, float(lab["operating_hours_to_end"]))
    wps = max(1, len(labels) // max(1, len(per_snapshot)))

    over_by_snapshot: dict[int, bool] = {}
    for i, r in enumerate(rows):
        snap = i // wps
        val = float(r.get("anomaly_score", 0) or 0) > threshold
        over_by_snapshot[snap] = over_by_snapshot.get(snap, False) or val

    ordered = sorted(over_by_snapshot.items())
    run = 0
    detected_at = None
    for snap, over in ordered:
        if over:
            run += 1
            if run >= SUSTAIN:
                detected_at = snap - SUSTAIN + 1
                break
        else:
            run = 0

    crossings = sum(1 for s in scores if s > threshold)
    return {
        "windows": len(rows),
        "invalid": n_invalid,
        "snapshots": len(ordered),
        "median_score": round(statistics.median(scores), 4) if scores else None,
        "max_score": round(max(scores), 4) if scores else None,
        "crossings": crossings,
        "crossing_rate": round(crossings / len(scores), 4) if scores else None,
        "detected": detected_at is not None,
        "detected_at_snapshot": detected_at,
        "lead_hours": per_snapshot.get(detected_at) if detected_at is not None else None,
    }


def report(res: dict, threshold: float) -> None:
    print()
    print("=" * 66)
    print(f" Scored — threshold {threshold}, sustained {SUSTAIN} snapshots")
    print("=" * 66)
    if "error" in res:
        print(f"  {res['error']}")
        return
    print(f"  windows           {res['windows']:,}   invalid {res['invalid']}")
    print(f"  snapshots         {res['snapshots']:,}")
    print(f"  score median/max  {res['median_score']} / {res['max_score']}")
    print(f"  crossings         {res['crossings']}  ({res['crossing_rate']:.2%} of windows)")
    if res["detected"]:
        lead = res["lead_hours"]
        print(f"  DETECTED          snapshot {res['detected_at_snapshot']}"
              + (f"  ->  {lead:.1f} operating hours before end of record" if lead is not None else ""))
    else:
        print("  not detected      no sustained crossing")
    print()
    print("  Lead time is to the END OF THE RECORD, not to fault onset —")
    print("  degradation is already underway when the detector fires.")


def run(blueprint_id: str, file_id: str, out_path: str, input_csv: str, name: str) -> str | None:
    """Package the blueprint, deploy it as batch, run one session over the file, download its
    output; then delete the session, deployment and package, which each hold the next."""
    t0 = time.time()
    pkg = dpl = ses = None
    try:
        pkg = create_package(blueprint_id, name)
        package_ready(pkg["id"])
        print(f"package {pkg['id']} ready ({time.time() - t0:.0f} s)")
        dpl = deploy(pkg["id"], "batch")
        deployment_ready(dpl["id"])
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="prepared input CSV (defaults to the bundled sample "
                         "slice, resolved next to this script)")
    ap.add_argument("--labels", default=None,
                    help="ground-truth sidecar (default: <csv stem>_labels.csv)")
    ap.add_argument("--blueprint-id", default=None,
                    help="run this blueprint (blp_…, in one of your projects, e.g. Path 2's "
                         f"promoted one) instead of the project's copy of {DEMO_BLUEPRINT}")
    ap.add_argument("--project", default=PROJECT,
                    help=f"the project to copy {DEMO_BLUEPRINT} into (default {PROJECT!r})")
    ap.add_argument("--threshold", type=float, default=None,
                    help="scoring threshold (default: the blueprint's own, 1.762 for the demo)")
    ap.add_argument("--name", default="AD Path 1", help="name of the package built for the run")
    ap.add_argument("--output", default="ad-output.csv")
    ap.add_argument("--score-only", metavar="OUTPUT_CSV", default=None,
                    help="skip the platform and score an output you already have")
    args = ap.parse_args()

    labels = args.labels or args.csv.rsplit(".", 1)[0] + "_labels.csv"

    if args.score_only:
        threshold = args.threshold or DEFAULT_THRESHOLD
        report(score(args.score_only, labels, threshold), threshold)
        return

    load_dotenv()
    check_auth()

    size_mb = os.path.getsize(args.csv) / 1048576
    if size_mb > 50:
        print(f"WARNING: {args.csv} is {size_mb:.0f} MiB. Inputs over ~50 MiB aborted in")
        print("  ConcatColumnsNode on older runtimes — see SKILL.md. Split it if this one does.")

    if args.blueprint_id:
        blueprint_id = args.blueprint_id
        print(f"blueprint {blueprint_id}")
    else:
        pid = project_id(args.project)
        blueprint_id = demo_copy(pid)["id"]
        print(f"blueprint {blueprint_id} ({COPY_KEY}, the copy of {DEMO_BLUEPRINT} in {pid})")
    threshold = args.threshold or blueprint_threshold(blueprint_id)

    print(f"uploading {args.csv} ({size_mb:.1f} MiB)")
    file_id = upload(args.csv)
    print(f"  file_id={file_id}")

    if not run(blueprint_id, file_id, args.output, args.csv, args.name):
        sys.exit("no results to score")
    print(f"  saved {args.output}")

    if os.path.exists(labels):
        report(score(args.output, labels, threshold), threshold)
    else:
        print(f"  no labels at {labels} — skipping scoring")


if __name__ == "__main__":
    main()
