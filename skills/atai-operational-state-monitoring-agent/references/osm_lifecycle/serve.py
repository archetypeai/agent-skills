#!/usr/bin/env python3
"""Path 2, step 5: serve the promoted model over /query, checked against the delivery.

deliver.py runs the model in batch: upload the files, run an agent, download the outputs.
Agent serving keeps an agent loaded and answers a window in one round trip, as a machine's
controller would ask it about its latest `window_size` readings. This replays the delivery
files that way and checks every answer against deliver.py's batch output for the same window:

  1. a bundle of the promoted blueprint, as it is: the platform supplies the serving
     connectors when it runs a bundle in serving mode, so no served copy of the blueprint
     is made (creating a blueprint takes the admin role on Staging). The bundle passes
     `data_columns` and `timestamp_column`, which the osm blueprint's source templates
     from values and batch reads from the CSV header;
  2. that bundle run with {"mode": "serving"};
  3. each file replayed a window at a time: every query holds one window's rows of every
     channel, one list per channel (--windows-per-query N sends N consecutive windows).
     A served agent keeps nothing between queries, so a query carries the whole window;
  4. every answer against batch: `predicted_state`, every `p_<state>` to within 1e-4, and
     the macro-F1 against the held-back labels, served and batch. Exits 1 if more than
     0.1% of the states differ: a window near a tie can swap one neighbour;
  5. the agent cancelled and deleted, and the bundle deleted, however it ends (Ctrl-C,
     SIGTERM, SIGHUP included). Their ids stay in <out>/serve_state.json until then, so
     cleanup.py finds them after a kill -9.

Dev and Staging only for now. Needs deliver.py's outputs in <out>/delivery/. Writes
<out>/serving/serve_<N>files[_first<M>]_wpq<W>.csv (a row per window) and .json (the summary).

    python serve.py --roles roles --out out
    python serve.py --roles roles --out out --files warm_fast --max-windows 50 --windows-per-query 8
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
from score import scores  # noqa: E402

MODEL = "osm"         # the name a query gives in `model` (a served agent does not check it)
MAX_WPQ = 16


def query_body(chunk, model=MODEL):
    """One query: a list per channel of the chunk's rows (windows x window_size of them)."""
    return {"model": model, "query": "", "events": [{"type": "data.numeric_array",
            "event_data": {"contents": [chunk[:, c].tolist() for c in range(chunk.shape[1])]}}]}


def windows_of(batch, ts_ms, window):
    """The first row of each batch window, from its finish_timestamp; exits if they don't tile."""
    row = {t: i for i, t in enumerate(ts_ms)}
    ends = [row.get(round(float(v) * 1000)) for v in batch.finish_timestamp]
    if None in ends:
        sys.exit("a batch window's finish_timestamp matches no row of its delivery file")
    firsts = np.array(ends) - window + 1
    starts = np.round(batch.start_timestamp.astype(float).to_numpy() * 1000).astype(np.int64)
    if (firsts < 0).any() or (ts_ms[firsts] != starts).any():
        sys.exit("batch windows don't span window_size rows")
    return firsts


def chunks(firsts, window, wpq):
    """(index, count) per query: up to wpq consecutive windows; a gap in the tiling (an
    invalid window left out) ends a query."""
    out, k = [], 0
    while k < len(firsts):
        n = 1
        while n < wpq and k + n < len(firsts) and firsts[k + n] == firsts[k] + n * window:
            n += 1
        out.append((k, n))
        k += n
    return out


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
    ap.add_argument("--name", default="osm", help="names the serving bundle")
    ap.add_argument("--files", nargs="+", metavar="NAME", help="delivery files whose name contains NAME (default: all)")
    ap.add_argument("--max-windows", type=int, help="replay only the first N windows of each file")
    ap.add_argument("--windows-per-query", type=int, default=1, help=f"1 = live, one window a query; up to {MAX_WPQ}")
    args = ap.parse_args(argv)
    load_dotenv()
    check_auth(log)
    # match the deployment, not one host: dev is api.dev.* and platform.dev.*, staging platform.stage.*
    if not any(d in root() for d in (".dev.", ".stage.")):
        sys.exit("agent serving is on dev and staging only for now "
                 "(e.g. https://platform.dev.u1.archetypeai.app, https://platform.stage.u1.archetypeai.app)")
    manifest = load_manifest(args.roles)
    states = manifest["states"]
    ts_path = os.path.join(args.out, "test_state.json")
    if not os.path.exists(ts_path):
        sys.exit(f"no {ts_path}: run promote_and_test.py first")
    test = json.load(open(ts_path))
    if not same_deployment(test):
        sys.exit(f"the blueprint in {ts_path} is on {test['endpoint']}, a different deployment from {root()}")
    promoted = test["blueprint"]["key"]
    window = request("GET", f"{agents()}/blueprints/{promoted}")["document"]["values"]["window_size"]
    files = manifest["delivery"]["files"]
    if args.files:
        files = [f for f in files if any(s in f["file"] for s in args.files)] or sys.exit(f"no delivery file matches {args.files}")

    plan = []                                    # (name, X, labels per row, batch, first rows)
    channels = manifest["channels"]
    for f in files:
        name = os.path.basename(f["file"])
        out = os.path.join(args.out, "delivery", name)
        if not os.path.exists(out):
            sys.exit(f"no {out}: run deliver.py first")
        data = pd.read_csv(os.path.join(args.roles, f["file"]), engine="pyarrow")
        ts_ms = np.round(data.timestamp.to_numpy() * 1000).astype(np.int64)
        lpath = f.get("labels") or f["file"].replace("delivery/", "delivery_labels/", 1)
        labels = pd.read_csv(os.path.join(args.roles, lpath), dtype=str).label.to_numpy(dtype=object)
        batch = pd.read_csv(out, dtype={"predicted_state": str, "invalid": str})
        batch = batch[batch.invalid.str.lower() != "true"].reset_index(drop=True)
        if args.max_windows:
            batch = batch.head(args.max_windows)
        plan.append((name, data[channels].to_numpy(dtype=np.float32), labels, batch, windows_of(batch, ts_ms, window)))
    wpq = max(1, min(args.windows_per_query, MAX_WPQ))
    log(f"{len(plan)} file(s), {sum(len(p[3]) for p in plan):,} windows of {window} rows, {wpq} per query; "
        f"model {promoted}")

    sf = os.path.join(args.out, "serve_state.json")
    if os.path.exists(sf):
        sys.exit(f"{sf} exists: an earlier serve.py's agent may still be up; run cleanup.py --yes first")
    bundle = request("POST", f"{agents()}/bundles", {
        "blueprint": promoted, "name": f"{args.name} serve",
        "values": {"data_columns": channels, "timestamp_column": "timestamp"}})
    ex = {"endpoint": root(), "agent": None, "bundle": bundle["id"]}
    json.dump(ex, open(sf, "w"), indent=1)
    served, latency = {}, []
    t0 = time.time()
    try:
        ex["agent"] = request("POST", f"{agents()}/bundles/{ex['bundle']}/run", {"mode": "serving"})["id"]
        json.dump(ex, open(sf, "w"), indent=1)
        log(f"bundle {ex['bundle']} · agent {ex['agent']}")
        while (status := request("GET", f"{agents()}/instances/{ex['agent']}").get("status")) != "running":
            if status in ("failed", "cancelled", "canceled", "completed") or time.time() - t0 > 600:
                sys.exit(f"agent {ex['agent']} is {status}, not running")
            time.sleep(3)
        for name, X, _, batch, firsts in plan:
            got = served[name] = []
            for k, count in chunks(firsts, window, wpq):
                t = time.time()
                answer = request("POST", f"{agents()}/instances/{ex['agent']}/query",
                                 query_body(X[firsts[k]:firsts[k] + count * window]))
                latency.append(time.time() - t)
                records = (answer.get("response") or {}).get("data")
                if not isinstance(records, list) or len(records) != count:
                    sys.exit(f"{name} window {k}: expected {count} records, got {str(answer)[:300]}")
                got.extend(records)
            log(f"  {name}: {len(firsts)} windows, {time.time() - t0:.0f} s so far")
    finally:
        if end(ex):
            os.remove(sf)
            log(f"cancelled and deleted {ex['agent']}, deleted {ex['bundle']}")
    elapsed = time.time() - t0

    rows = []
    for name, _, labels, batch, firsts in plan:
        for k, (s, b) in enumerate(zip(served[name], batch.itertuples())):
            diff = max(abs(float(s[f"p_{c}"]) - float(getattr(b, f"p_{c}"))) for c in states)
            rows.append({"file": name, "window": k, "finish_timestamp": b.finish_timestamp,
                         "label": labels[firsts[k] + window - 1],
                         "batch_state": b.predicted_state, "served_state": s["predicted_state"],
                         **{f"served_p_{c}": float(s[f"p_{c}"]) for c in states},
                         "max_p_diff": diff, "served_invalid": str(s["invalid"]).lower() == "true",
                         "same_state": s["predicted_state"] == b.predicted_state, "same_p": diff < 1e-4})
    df = pd.DataFrame(rows)
    y = df.label.to_numpy(dtype=object)
    sdir = os.path.join(args.out, "serving")
    os.makedirs(sdir, exist_ok=True)
    stem = os.path.join(sdir, f"serve_{len(plan)}files{f'_first{args.max_windows}' if args.max_windows else ''}_wpq{wpq}")
    df.to_csv(stem + ".csv", index=False)
    ms = sorted(x * 1000 for x in latency)
    summary = {"blueprint": promoted, "files": [p[0] for p in plan], "windows": len(df),
               "windows_per_query": wpq, "queries": len(latency),
               "same_state": int(df.same_state.sum()), "same_p": int(df.same_p.sum()),
               "max_p_diff": float(df.max_p_diff.max()), "invalid": int(df.served_invalid.sum()),
               "served": scores(y, df.served_state.to_numpy(dtype=object), states),
               "batch": scores(y, df.batch_state.to_numpy(dtype=object), states),
               "latency_ms": {"median": round(statistics.median(ms)), "p95": round(ms[int(0.95 * (len(ms) - 1))]),
                              "max": round(ms[-1])},
               "replay_s": round(elapsed)}
    json.dump(summary, open(stem + ".json", "w"), indent=1)

    print(f"\nserved {len(df):,} windows of {len(plan)} file(s) in {len(latency):,} queries, {elapsed:.0f} s")
    print(f"  against the delivery (batch): predicted_state identical in {summary['same_state']:,}/{len(df):,}, "
          f"probabilities within 1e-4 in {summary['same_p']:,}/{len(df):,} "
          f"(max difference {summary['max_p_diff']:.1e}); invalid {summary['invalid']}")
    print(f"  latency per query: median {summary['latency_ms']['median']} ms, "
          f"p95 {summary['latency_ms']['p95']} ms, max {summary['latency_ms']['max']} ms")
    for who in ("served", "batch"):
        r = summary[who]
        print(f"  {who:<7} macro-F1 {r['macro_f1']:.4f}  " + "  ".join(f"{s} {r['f1'][s]:.2f}" for s in states))
    print(f"  -> {stem}.csv / .json")
    return 0 if summary["same_state"] >= 0.999 * len(df) else 1


if __name__ == "__main__":
    sys.exit(main())
