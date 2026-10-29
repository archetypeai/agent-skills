#!/usr/bin/env python3
"""Path 2, step 5: serve the promoted detector over /query, checked against the delivery.

deliver.py runs the detector in batch: upload a file, run an agent, download the output.
Agent serving keeps an agent loaded and answers a window of samples in one round trip,
as a live system would ask it for the latest `window_size` samples. This replays delivery
recordings that way and checks every answer against deliver.py's batch output:

  1. a bundle of the promoted blueprint, as it is, at step 1 (bundle values override a
     blueprint's): the platform supplies the serving connectors when it runs a bundle in
     serving mode;
  2. that bundle run with {"mode": "serving"};
  3. each recording replayed one sample at a time: every query holds the latest
     `window_size` samples (--windows-per-query N sends N consecutive windows at once).
     A served agent keeps nothing between queries and numbers the samples itself, so
     recordings are replayed one by one: a window across a join would not be marked
     invalid as batch marks it;
  4. every answer against batch: `predicted_label`, and `anomaly_score` to within 1e-4
     relative (float32 noise is ~2e-5), plus each recording's first sustained alarm
     (3 consecutive anomaly windows), served and batch;
  5. the agent cancelled and deleted, and the bundle deleted, however it ends (Ctrl-C,
     SIGTERM, SIGHUP included). Their ids stay in <out>/serve_state.json until then, so
     cleanup.py finds them after a kill -9.

Needs deliver.py's output at step 1 (its default). Writes
<out>/serving/<file>_wpq<N>.csv (a row per window) and .json (the summary).

    python serve.py --roles roles --out out --recordings normal_48h_run43 idv05_run3
"""
import argparse
import json
import os
import signal
import statistics
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import agents, check_auth, load_dotenv, request, root, same_deployment  # noqa: E402
from common import load_manifest, log  # noqa: E402

MODEL = "ad"          # the name a query gives in `model`
SUSTAIN = 3
MAX_WPQ = 16


def query_body(chunk, model=MODEL):
    """One query: a list per channel of the chunk's rows (window + windows - 1 of them)."""
    return {"model": model, "query": "", "events": [{"type": "data.numeric_array",
            "event_data": {"contents": [chunk[:, c].tolist() for c in range(chunk.shape[1])]}}]}


def segments(labels, window):
    """(recording, first row, windows) for each recording of a delivery file."""
    out = []
    for rec, rows in labels.groupby("recording", sort=False):
        n = len(rows) - window + 1
        if n > 0:
            out.append((rec, int(rows.index[0]), n))
    return out


def first_alarm(labels):
    """Index of the window that completes the first run of SUSTAIN anomaly windows."""
    run = 0
    for i, lab in enumerate(labels):
        run = run + 1 if lab == "anomaly" else 0
        if run == SUSTAIN:
            return i
    return None


def end(ex):
    """Delete the agent and the bundle; an id already gone counts as done."""
    calls = [("POST", f"/instances/{ex['agent']}/cancel", {}), ("DELETE", f"/instances/{ex['agent']}", None)]
    for method, path, body in (calls if ex["agent"] else []) + [("DELETE", f"/bundles/{ex['bundle']}", None)]:
        try:
            request(method, f"{agents()}{path}", body)
        except RuntimeError as e:
            if "(404)" not in str(e):
                log(f"cleanup: {e}; cleanup.py retries it")
                return False
    return True


def stop(signum, _frame):
    sys.exit(f"stopped by signal {signum}")       # so `finally` deletes the agent and bundle


def main(argv=None):
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="ad")
    ap.add_argument("--file", help="a delivery file (default: the first one that isn't calibration)")
    ap.add_argument("--recordings", nargs="+", help="the file's recordings to replay (default: all)")
    ap.add_argument("--windows-per-query", type=int, default=1, help=f"1 = live, one window per sample; up to {MAX_WPQ}")
    ap.add_argument("--max-windows", type=int, help="replay only the first N windows of each recording")
    a = ap.parse_args(argv)
    load_dotenv()
    check_auth(log)
    m = load_manifest(a.roles)
    test = json.load(open(os.path.join(a.out, "test_state.json")))
    if not same_deployment(test):
        sys.exit(f"the blueprint in {a.out}/test_state.json is on {test['endpoint']}, a different deployment from {root()}")
    deliver = json.load(open(os.path.join(a.out, "deliver_state.json")))
    if deliver.get("values", {}).get("step_size") != 1:
        sys.exit("serve.py compares against deliver.py's output at step 1: rerun deliver.py with --step 1")
    bp = test["blueprint"]
    channels, window = m["channels"], bp["values"]["window_size"]
    f = next((d for d in m["delivery"] if (a.file and d["file"].endswith(a.file)) or
              (not a.file and not d.get("calibration"))), None) or sys.exit(f"no delivery file {a.file or ''}")
    data = pd.read_csv(os.path.join(a.roles, f["file"]))
    labels = pd.read_csv(os.path.join(a.roles, f["labels"]))
    batch = pd.read_csv(os.path.join(a.out, "delivery", os.path.basename(f["file"])))
    if len(batch) != len(data) - window + 1:
        sys.exit(f"{f['file']}: the delivery output has {len(batch)} windows, expected {len(data) - window + 1}")
    segs = [s for s in segments(labels, window) if not a.recordings or s[0] in a.recordings]
    if not segs:
        sys.exit(f"none of {a.recordings} in {f['file']}")
    segs = [(rec, first, min(n, a.max_windows or n)) for rec, first, n in segs]
    wpq = max(1, min(a.windows_per_query, MAX_WPQ))
    X = data[channels].to_numpy(dtype=float)
    ts = data[m["timestamp_column"]].to_numpy(dtype=float)
    log(f"{f['file']}: {len(segs)} recording(s), {sum(s[2] for s in segs):,} windows, {wpq} per query; "
        f"model {bp['key']}")

    sf = os.path.join(a.out, "serve_state.json")
    if os.path.exists(sf):
        sys.exit(f"{sf} holds an earlier replay's agent and bundle: run cleanup.py first")
    bundle = request("POST", f"{agents()}/bundles", {"blueprint": bp["key"], "name": f"{a.name} serving",
                                                      "values": {"step_size": 1}})
    ex = {"endpoint": root(), "bundle": bundle["id"], "agent": None}
    json.dump(ex, open(sf, "w"))
    served, latency = {}, []
    t0 = time.time()
    try:
        ex["agent"] = request("POST", f"{agents()}/bundles/{ex['bundle']}/run", {"mode": "serving"})["id"]
        json.dump(ex, open(sf, "w"))
        log(f"bundle {ex['bundle']} · agent {ex['agent']}")
        while (status := request("GET", f"{agents()}/instances/{ex['agent']}").get("status")) != "running":
            if status in ("failed", "cancelled", "canceled", "completed") or time.time() - t0 > 600:
                sys.exit(f"agent {ex['agent']} is {status}, not running")
            time.sleep(3)
        for rec, first, n in segs:
            got = served[rec] = []
            for k in range(0, n, wpq):
                count = min(wpq, n - k)
                t = time.time()
                answer = request("POST", f"{agents()}/instances/{ex['agent']}/query",
                                 query_body(X[first + k:first + k + window + count - 1]))
                latency.append(time.time() - t)
                records = (answer.get("response") or {}).get("data")
                if not isinstance(records, list) or len(records) != count:
                    sys.exit(f"{rec} window {k}: expected {count} records, got {str(answer)[:300]}")
                got.extend(records)
            log(f"  {rec}: {n} windows, {time.time() - t0:.0f} s so far")
    finally:
        if end(ex):
            os.remove(sf)
            log(f"cancelled and deleted {ex['agent']}, deleted {ex['bundle']}")
    elapsed = time.time() - t0

    rows, per_rec = [], []
    for rec, first, n in segs:
        ref, got = batch.iloc[first:first + n], served[rec]
        for k, (s, b) in enumerate(zip(got, ref.itertuples())):
            rows.append({"recording": rec, "window": k, "finish_timestamp": ts[first + k + window - 1],
                         "batch_label": b.predicted_label, "served_label": s["predicted_label"],
                         "batch_score": b.anomaly_score, "served_score": float(s["anomaly_score"]),
                         "served_invalid": str(s["invalid"]).lower() == "true",
                         "same_label": s["predicted_label"] == b.predicted_label,
                         "same_score": abs(float(s["anomaly_score"]) / b.anomaly_score - 1) < 1e-4})
        lab = labels.label.to_numpy()[first:first + n + window - 1]
        onset = int(np.argmax(lab == "anomaly")) if (lab == "anomaly").any() else None
        alarm = {}
        for who, labs in (("served", [s["predicted_label"] for s in got]), ("batch", ref.predicted_label.tolist())):
            i = first_alarm(labs)
            if i is None:
                alarm[who] = None
            elif onset is None or i + window - 1 < onset:
                alarm[who] = f"a false alarm {(ts[first + i + window - 1] - ts[first]) / 3600:.2f} h in"
            else:
                alarm[who] = f"{(ts[first + i + window - 1] - ts[first + onset]) / 3600:.2f} h after onset"
        per_rec.append({"recording": rec, "windows": n, "first_alarm": alarm})

    df = pd.DataFrame(rows)
    sdir = os.path.join(a.out, "serving")
    os.makedirs(sdir, exist_ok=True)
    stem = os.path.join(sdir, f"{os.path.basename(f['file']).replace('.csv', '')}_wpq{wpq}")
    df.to_csv(stem + ".csv", index=False)
    ms = sorted(x * 1000 for x in latency)
    summary = {"blueprint": bp["key"], "file": f["file"], "recordings": per_rec,
               "windows": len(df), "windows_per_query": wpq, "queries": len(latency),
               "same_label": int(df.same_label.sum()), "same_score": int(df.same_score.sum()),
               "max_score_rel_diff": float((df.served_score / df.batch_score - 1).abs().max()),
               "invalid": int(df.served_invalid.sum()),
               "latency_ms": {"median": round(statistics.median(ms)), "p95": round(ms[int(0.95 * (len(ms) - 1))]),
                              "max": round(ms[-1])},
               "replay_s": round(elapsed)}
    json.dump(summary, open(stem + ".json", "w"), indent=1)

    print(f"\nserved {len(df):,} windows of {len(segs)} recording(s) in {len(latency):,} queries, {elapsed:.0f} s")
    print(f"  against the delivery (batch): predicted_label identical {summary['same_label']:,}/{len(df):,}, "
          f"anomaly_score within 1e-4 {summary['same_score']:,}/{len(df):,} "
          f"(max relative difference {summary['max_score_rel_diff']:.1e}); invalid {summary['invalid']}")
    print(f"  latency per query: median {summary['latency_ms']['median']} ms, p95 {summary['latency_ms']['p95']} ms, "
          f"max {summary['latency_ms']['max']} ms")
    for r in per_rec:
        print(f"  {r['recording']:<18} first sustained alarm: served {r['first_alarm']['served'] or 'none'}; "
              f"batch {r['first_alarm']['batch'] or 'none'}")
    print(f"  -> {stem}.csv / .json")
    return 0 if summary["same_label"] == len(df) else 1


if __name__ == "__main__":
    sys.exit(main())
