#!/usr/bin/env python3
"""Path 2, step 5: serve the promoted classifier over /query, checked against the delivery.

deliver.py runs the classifier in batch: upload a file, run a session, download the output.
Agent serving keeps an agent loaded and answers a window of rows in one round trip, as a
live system would ask it every minute for the latest `window_size` readings. This replays
delivery files that way and checks every answer against deliver.py's batch output:

  1. a package of the promoted blueprint, as it is (the delivery's step: a package has no
     values); the platform supplies the serving connectors for a serving deployment;
  2. that package deployed as `serving` (the agent placed on a worker and loaded), and a
     session opened on the deployment, queried at POST /agents/sessions/{id}/query;
  3. each file replayed one step at a time: every query holds the latest `window_size`
     rows (--windows-per-query N sends N consecutive windows at once, `step` apart), so
     the served windows are exactly the ones batch scored. By default only the windows
     ending within --context rows of the file's incidents (--whole for all);
  4. every answer against batch: `predicted_state`, every class probability to within
     1e-4, and per file each incident's detection delay and the false alarms (score.py's
     rules), served and batch;
  5. the session cancelled, the deployment stopped, and the session, deployment and package
     deleted, however it ends (Ctrl-C, SIGTERM, SIGHUP included). Each id goes to
     <out>/serve_state.json before the script waits on it and stays there until then, so
     cleanup.py finds them after a kill -9.

A served agent keeps nothing between queries, so a query must carry the whole window; the
timestamps in its answer are row indices within the query.
Needs deliver.py's output. Writes <out>/serving/<files>_wpq<N>.csv
(a row per window) and .json (the summary).

    python serve.py --roles roles --out out
"""
import argparse
import json
import os
import signal
import statistics
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import (check_auth, create_package, deploy, deployment_ready, end as end_one, lifecycle_url,  # noqa: E402
                       load_dotenv, open_session, package_ready, request, root, same_deployment, wait_for)
from common import load_manifest, log  # noqa: E402
from score import read_output, score_windows  # noqa: E402

MODEL = "red"          # the name a query gives in `model`
MAX_WPQ = 16


def query_body(chunk, model=MODEL):
    """One query: a list per channel of the chunk's rows (window + (windows - 1) * step of them)."""
    return {"model": model, "query": "", "events": [{"type": "data.numeric_array",
            "event_data": {"contents": [chunk[:, c].tolist() for c in range(chunk.shape[1])]}}]}


def span(labels, background, window, context, n_windows, step=1):
    """(first window, windows) ending within `context` rows of the file's event rows;
    every window when there are none. Window k ends on row k * step + window - 1."""
    rows = [i for i, lab in enumerate(labels) if lab != background]
    if not rows:
        return 0, n_windows
    first = max(0, -(-(rows[0] - context - window + 1) // step))
    last = min(n_windows - 1, (rows[-1] + context - window + 1) // step)
    return first, last - first + 1


def outcome(preds, first, window, labels, background, step=1):
    """Each incident's delay (rows) and the false alarms, score.py's rules, for windows
    first, first + 1, ... (`step` rows apart) predicting `preds`."""
    r = score_windows(labels, [((first + k) * step, (first + k) * step + window, p) for k, p in enumerate(preds)],
                      background)
    return {"incidents": [i["delay_rows"] for i in r["incidents"]], "false_alarms": r["false_alarms"]}


def end(ex):
    """End the session, then the deployment, then delete the package; an id already gone
    counts as done. Each is tried even when one before it fails: a session that won't delete
    must not leave the serving deployment holding its worker."""
    ok = True
    for kind in ("sessions", "deployments", "packages"):
        obj = ex.get(kind[:-1])
        if not obj:
            continue
        try:
            end_one(kind, obj)
        except RuntimeError as e:
            log(f"cleanup: {e}; cleanup.py retries it")
            ok = False
    return ok


def stop(signum, _frame):
    sys.exit(f"stopped by signal {signum}")       # so `finally` ends the session and deployment


def main(argv=None):
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="red")
    ap.add_argument("--files", nargs="+", help="delivery files by name, e.g. inc04 (default: all)")
    ap.add_argument("--context", type=int, default=120,
                    help="replay the windows ending within this many rows of an event row (default 120)")
    ap.add_argument("--whole", action="store_true", help="replay every window of each file")
    ap.add_argument("--windows-per-query", type=int, default=1, help=f"1 = live, one window per step; up to {MAX_WPQ}")
    a = ap.parse_args(argv)
    load_dotenv()
    check_auth(log)
    m = load_manifest(a.roles)
    test = json.load(open(os.path.join(a.out, "test_state.json")))
    if not same_deployment(test):
        sys.exit(f"the blueprint in {a.out}/test_state.json is on {test['endpoint']}, a different deployment from {root()}")
    bp = test["blueprint"]
    channels, window, background = m["channels"], bp["values"]["window_size"], m["background"]
    step = bp["values"]["step_size"]
    classes = m["classes"]
    files = {os.path.basename(f["file"]).replace(".csv", ""): f for f in m["delivery"]}

    plan = []                                    # (name, X, labels, batch, first window, windows)
    for name in a.files or files:
        f = files.get(name) or sys.exit(f"{name} is not a delivery file: {', '.join(files)}")
        data = pd.read_csv(os.path.join(a.roles, f["file"]))
        labels = pd.read_csv(os.path.join(a.roles, f["labels"])).label.tolist()
        out = os.path.join(a.out, "delivery", os.path.basename(f["file"]))
        if not os.path.exists(out):
            sys.exit(f"no {out}: run deliver.py first")
        batch = read_output(out)
        if len(batch) != (len(data) - window) // step + 1:
            sys.exit(f"{name}: the delivery output has {len(batch)} windows, expected "
                     f"{(len(data) - window) // step + 1} at step {step}")
        first, n = (0, len(batch)) if a.whole else span(labels, background, window, a.context, len(batch), step)
        plan.append((name, data[channels].to_numpy(dtype=float), labels, batch, first, n))
    wpq = max(1, min(a.windows_per_query, MAX_WPQ))
    log(f"{len(plan)} file(s), {sum(p[5] for p in plan):,} windows at step {step}, {wpq} per query; model {bp['key']}")

    sf = os.path.join(a.out, "serve_state.json")
    if os.path.exists(sf):
        sys.exit(f"{sf} holds an earlier replay's session, deployment and package: run cleanup.py first")
    ex = {"endpoint": root(), "package": None, "deployment": None, "session": None}
    served, latency = {}, []
    t0 = time.time()
    try:
        # each id is recorded before waiting on it, so cleanup.py can delete what fails
        for kind, make, ready in (("package", lambda: create_package(bp["id"], f"{a.name} serving"), package_ready),
                                  ("deployment", lambda: deploy(ex["package"], "serving"), deployment_ready),
                                  ("session", lambda: open_session(ex["deployment"]), None)):
            ex[kind] = make()["id"]
            json.dump(ex, open(sf, "w"))
            if ready:
                ready(ex[kind])
        log(f"package {ex['package']} · serving deployment {ex['deployment']} · session {ex['session']}")
        wait_for("sessions", ex["session"], "running", "pending")
        for name, X, _, _, first, n in plan:
            got = served[name] = []
            for k in range(0, n, wpq):
                count = min(wpq, n - k)
                lo = first + k
                t = time.time()
                answer = request("POST", lifecycle_url("sessions", ex["session"]) + "/query",
                                 query_body(X[lo * step:(lo + count - 1) * step + window]), idempotent=True)
                latency.append(time.time() - t)
                records = (answer.get("response") or {}).get("data")
                if not isinstance(records, list) or len(records) != count:
                    sys.exit(f"{name} window {lo}: expected {count} records, got {str(answer)[:300]}")
                got.extend(records)
            log(f"  {name}: {n} windows, {time.time() - t0:.0f} s so far")
    finally:
        if end(ex):
            os.remove(sf)
            log(f"ended and deleted {ex['session']}, {ex['deployment']}, {ex['package']}")
    elapsed = time.time() - t0

    rows, per_file = [], []
    for name, _, labels, batch, first, n in plan:
        ref, got = batch[first:first + n], served[name]
        for k, (s, b) in enumerate(zip(got, ref)):
            diff = max(abs(float(s[f"p_{c}"]) - float(b[f"p_{c}"])) for c in classes)
            rows.append({"file": name, "window": first + k, "end_row": (first + k) * step + window - 1,
                         "batch_state": b["predicted_state"], "served_state": s["predicted_state"],
                         **{f"served_p_{c}": float(s[f"p_{c}"]) for c in classes},
                         "max_p_diff": diff, "served_invalid": str(s["invalid"]).lower() == "true",
                         "same_state": s["predicted_state"] == b["predicted_state"], "same_p": diff < 1e-4})
        per_file.append({"file": name, "windows": n,
                         "end_rows": [first * step + window - 1, (first + n - 1) * step + window - 1],
                         "served": outcome([s["predicted_state"] for s in got], first, window, labels, background, step),
                         "batch": outcome([b["predicted_state"] for b in ref], first, window, labels, background, step)})

    df = pd.DataFrame(rows)
    sdir = os.path.join(a.out, "serving")
    os.makedirs(sdir, exist_ok=True)
    stem = os.path.join(sdir, f"{'_'.join(p[0] for p in plan)}{'_whole' if a.whole else ''}_wpq{wpq}")
    df.to_csv(stem + ".csv", index=False)
    ms = sorted(x * 1000 for x in latency)
    summary = {"blueprint": bp["key"], "step": step, "files": per_file,
               "windows": len(df), "windows_per_query": wpq, "queries": len(latency),
               "same_state": int(df.same_state.sum()), "same_p": int(df.same_p.sum()),
               "max_p_diff": float(df.max_p_diff.max()), "invalid": int(df.served_invalid.sum()),
               "latency_ms": {"median": round(statistics.median(ms)), "p95": round(ms[int(0.95 * (len(ms) - 1))]),
                              "max": round(ms[-1])},
               "replay_s": round(elapsed)}
    json.dump(summary, open(stem + ".json", "w"), indent=1)

    def fmt(o):
        det = ", ".join("missed" if d is None else f"+{d} rows" for d in o["incidents"]) or "no incident"
        return f"{det}, {o['false_alarms']} false alarm(s)"

    print(f"\nserved {len(df):,} windows of {len(plan)} file(s) in {len(latency):,} queries, {elapsed:.0f} s")
    print(f"  against the delivery (batch): predicted_state identical {summary['same_state']:,}/{len(df):,}, "
          f"probabilities within 1e-4 {summary['same_p']:,}/{len(df):,} "
          f"(max difference {summary['max_p_diff']:.1e}); invalid {summary['invalid']}")
    print(f"  latency per query: median {summary['latency_ms']['median']} ms, p95 {summary['latency_ms']['p95']} ms, "
          f"max {summary['latency_ms']['max']} ms")
    for r in per_file:
        print(f"  {r['file']} (windows ending on rows {r['end_rows'][0]}-{r['end_rows'][1]}): "
              f"served {fmt(r['served'])}; batch {fmt(r['batch'])}")
    print(f"  -> {stem}.csv / .json")
    return 0 if summary["same_state"] == len(df) else 1


if __name__ == "__main__":
    sys.exit(main())
