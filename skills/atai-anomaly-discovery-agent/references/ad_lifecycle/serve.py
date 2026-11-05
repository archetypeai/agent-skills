#!/usr/bin/env python3
"""Path 2, step 5: serve the promoted detector over /query, checked against the delivery.

deliver.py runs the detector in batch: upload a file, run an agent, download the output.
Agent serving keeps an agent loaded and answers a window of samples in one round trip,
as a live system would ask it for the latest `window_size` samples. This replays delivery
recordings that way and checks every answer against deliver.py's batch output:

  1. a bundle of the promoted blueprint, as it is, at the delivery's step (bundle values
     override a blueprint's): the platform supplies the serving connectors when it runs a
     bundle in serving mode;
  2. that bundle run with {"mode": "serving"};
  3. each recording replayed one step at a time: every query holds the latest
     `window_size` samples (--windows-per-query N sends N consecutive windows at once).
     At step 1 that is a query per sample, as live; at a larger step (deliver.py --step,
     for a deployment that can't deliver at step 1 in time) the windows batch scored.
     A served agent keeps nothing between queries and numbers the samples itself, so
     recordings are replayed one by one, and a recording with a gap (a segment seam) one
     gap-free piece at a time: a window across a join or a gap would not be marked
     invalid as batch marks it;
  4. every answer against batch: `predicted_label`, and `anomaly_score` to within 1e-4
     relative (float32 noise is ~2e-5), plus each recording's first sustained alarm
     (3 consecutive anomaly windows), served and batch;
  5. the agent cancelled and deleted, and the bundle deleted, however it ends (Ctrl-C,
     SIGTERM, SIGHUP included). Their ids stay in <out>/serve_state.json until then, so
     cleanup.py finds them after a kill -9.

Needs deliver.py's output (step 1 by default; any step works). Writes
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


def gaps(ts):
    """True where a row follows a gap: a step more than 5% over the file's sample interval, as
    the platform's sampling-rate check sees it (the seam between joined recordings, or between
    the segments of one recording that prepare.py split at a gap)."""
    dt = np.diff(ts)
    return np.r_[False, dt > 1.05 * np.median(dt)]


def operating_s(ts):
    """Seconds of operation since the first row, a gap counted as one sample interval."""
    dt = np.diff(ts)
    return np.r_[0.0, np.cumsum(np.where(dt > 1.05 * np.median(dt), np.median(dt), dt))]


def segments(labels, window, step=1, ts=None):
    """(recording, first window's first row, windows) for each gap-free piece of each recording
    of a delivery file: the windows batch scored inside it (they tile the file from row 0 at
    `step`). A window across a gap is invalid in batch, and a served agent can't see the gap,
    so each piece is replayed on its own."""
    cut = gaps(ts) if ts is not None else np.zeros(len(labels), bool)
    out = []
    for rec, rows in labels.groupby("recording", sort=False):
        lo, hi = int(rows.index[0]), int(rows.index[0]) + len(rows)
        starts = [lo, *(lo + 1 + np.flatnonzero(cut[lo + 1:hi]))]
        for a, b in zip(starts, [*starts[1:], hi]):
            first = -(-a // step) * step
            n = (b - window - first) // step + 1
            if n > 0:
                out.append((rec, int(first), int(n)))
    return out


def cap(segs, max_windows):
    """The first `max_windows` windows of each recording, across its pieces in order: a
    recording stops where its count runs out, so no later piece is replayed after a cut one."""
    if not max_windows:
        return segs
    left, out = {}, []
    for rec, first, n in segs:
        k = min(n, left.setdefault(rec, max_windows))
        if k > 0:
            out.append((rec, first, k))
            left[rec] -= k
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
    ap.add_argument("--max-windows", type=int, help="replay only the first N windows of each recording (across its gap-free pieces)")
    a = ap.parse_args(argv)
    load_dotenv()
    check_auth(log)
    m = load_manifest(a.roles)
    test = json.load(open(os.path.join(a.out, "test_state.json")))
    if not same_deployment(test):
        sys.exit(f"the blueprint in {a.out}/test_state.json is on {test['endpoint']}, a different deployment from {root()}")
    deliver = json.load(open(os.path.join(a.out, "deliver_state.json")))
    bp = test["blueprint"]
    channels, window = m["channels"], bp["values"]["window_size"]
    step = deliver.get("values", {}).get("step_size") or bp["values"]["step_size"]
    f = next((d for d in m["delivery"] if (a.file and d["file"].endswith(a.file)) or
              (not a.file and not d.get("calibration"))), None) or sys.exit(f"no delivery file {a.file or ''}")
    data = pd.read_csv(os.path.join(a.roles, f["file"]))
    labels = pd.read_csv(os.path.join(a.roles, f["labels"]))
    batch = pd.read_csv(os.path.join(a.out, "delivery", os.path.basename(f["file"])))
    if len(batch) != (len(data) - window) // step + 1:
        sys.exit(f"{f['file']}: the delivery output has {len(batch)} windows, expected "
                 f"{(len(data) - window) // step + 1} at step {step}")
    X = data[channels].to_numpy(dtype=float)
    ts = data[m["timestamp_column"]].to_numpy(dtype=float)
    segs = [s for s in segments(labels, window, step, ts) if not a.recordings or s[0] in a.recordings]
    if not segs:
        sys.exit(f"none of {a.recordings} in {f['file']}")
    segs = cap(segs, a.max_windows)
    wpq = max(1, min(a.windows_per_query, MAX_WPQ))
    recs = list(dict.fromkeys(s[0] for s in segs))
    log(f"{f['file']}: {len(recs)} recording(s) in {len(segs)} gap-free piece(s), {sum(s[2] for s in segs):,} windows, "
        f"{wpq} per query, step {step}; model {bp['key']}")

    sf = os.path.join(a.out, "serve_state.json")
    if os.path.exists(sf):
        sys.exit(f"{sf} holds an earlier replay's agent and bundle: run cleanup.py first")
    bundle = request("POST", f"{agents()}/bundles", {"blueprint": bp["key"], "name": f"{a.name} serving",
                                                      "values": {"step_size": step}})
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
        for j, (rec, first, n) in enumerate(segs):
            got = served[j] = []
            for k in range(0, n, wpq):
                count = min(wpq, n - k)
                t = time.time()
                chunk = X[first + k * step:first + (k + count - 1) * step + window]
                answer = request("POST", f"{agents()}/instances/{ex['agent']}/query", query_body(chunk), idempotent=True)
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

    rows, alarm_rows, windows = [], {rec: {} for rec in recs}, dict.fromkeys(recs, 0)
    for j, (rec, first, n) in enumerate(segs):
        ref, got = batch.iloc[first // step:first // step + n], served[j]
        for k, (s, b) in enumerate(zip(got, ref.itertuples())):
            rows.append({"recording": rec, "window": windows[rec] + k, "finish_timestamp": ts[first + k * step + window - 1],
                         "batch_label": b.predicted_label, "served_label": s["predicted_label"],
                         "batch_score": b.anomaly_score, "served_score": float(s["anomaly_score"]),
                         "served_invalid": str(s["invalid"]).lower() == "true",
                         "same_label": s["predicted_label"] == b.predicted_label,
                         "same_score": abs(float(s["anomaly_score"]) / b.anomaly_score - 1) < 1e-4})
        windows[rec] += n
        # an alarm is sustained within a piece: batch abstains across a gap, so no run spans one
        for who, labs in (("served", [s["predicted_label"] for s in got]), ("batch", ref.predicted_label.tolist())):
            i = first_alarm(labs)
            if i is not None and who not in alarm_rows[rec]:
                alarm_rows[rec][who] = first + i * step + window - 1

    op, lab = operating_s(ts), labels.label.to_numpy()
    per_rec = []
    for rec in recs:
        pieces = [(first, n) for r, first, n in segs if r == rec]
        lo, hi = pieces[0][0], pieces[-1][0] + (pieces[-1][1] - 1) * step + window
        onset = lo + int(np.argmax(lab[lo:hi] == "anomaly")) if (lab[lo:hi] == "anomaly").any() else None
        alarm = {}
        for who in ("served", "batch"):
            row = alarm_rows[rec].get(who)
            if row is None:
                alarm[who] = None
            elif onset is None or row < onset:
                alarm[who] = f"a false alarm {(op[row] - op[lo]) / 3600:.2f} h in"
            else:
                alarm[who] = f"{(op[row] - op[onset]) / 3600:.2f} h after onset"
        per_rec.append({"recording": rec, "pieces": len(pieces), "windows": sum(n for _, n in pieces),
                        "first_alarm": alarm})

    df = pd.DataFrame(rows)
    sdir = os.path.join(a.out, "serving")
    os.makedirs(sdir, exist_ok=True)
    stem = os.path.join(sdir, f"{os.path.basename(f['file']).replace('.csv', '')}_wpq{wpq}")
    df.to_csv(stem + ".csv", index=False)
    ms = sorted(x * 1000 for x in latency)
    summary = {"blueprint": bp["key"], "file": f["file"], "step": step, "recordings": per_rec,
               "windows": len(df), "windows_per_query": wpq, "queries": len(latency),
               "same_label": int(df.same_label.sum()), "same_score": int(df.same_score.sum()),
               "max_score_rel_diff": float((df.served_score / df.batch_score - 1).abs().max()),
               "invalid": int(df.served_invalid.sum()),
               "latency_ms": {"median": round(statistics.median(ms)), "p95": round(ms[int(0.95 * (len(ms) - 1))]),
                              "max": round(ms[-1])},
               "replay_s": round(elapsed)}
    json.dump(summary, open(stem + ".json", "w"), indent=1)

    print(f"\nserved {len(df):,} windows of {len(recs)} recording(s) in {len(latency):,} queries, {elapsed:.0f} s")
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
