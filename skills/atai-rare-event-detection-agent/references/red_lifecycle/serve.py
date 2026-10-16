#!/usr/bin/env python3
"""Path 2, step 5: serve the promoted classifier over /query, checked against the delivery.

deliver.py runs the classifier in batch: upload a file, run an agent, download the output.
Agent serving keeps an agent loaded and answers a window of rows in one round trip, as a
live system would ask it every minute for the latest `window_size` readings. This replays
delivery files that way and checks every answer against deliver.py's batch output:

  1. the served copy of the promoted blueprint, made once and reused
     (`<key>-served-records`): the same prototypes, encoder and values, step 1, with the
     CSV source and sink swapped for an `atai-records` serving endpoint;
  2. a bundle of it, run with {"mode": "serving"};
  3. each file replayed one row at a time: every query holds the latest `window_size`
     rows (--windows-per-query N sends N consecutive windows at once). By default only
     the windows ending within --context rows of the file's incidents (--whole for all);
  4. every answer against batch: `predicted_state`, every class probability to within
     1e-4, and per file each incident's detection delay and the false alarms (score.py's
     rules), served and batch;
  5. the agent cancelled and deleted, and the bundle deleted, however it ends (Ctrl-C,
     SIGTERM, SIGHUP included). Their ids stay in <out>/serve_state.json until then, so
     cleanup.py finds them after a kill -9.

A served agent keeps nothing between queries, so a query must carry the whole window; the
timestamps in its answer are row indices within the query. Dev only for now. Needs
deliver.py's output at step 1 (its default). Writes <out>/serving/<files>_wpq<N>.csv (a row
per window) and .json (the summary).

    python serve.py --roles roles --out out
"""
import argparse
import copy
import json
import os
import signal
import statistics
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import agents, check_auth, load_dotenv, request, root  # noqa: E402
from common import load_manifest, log  # noqa: E402
from score import read_output, score_windows  # noqa: E402

MODEL = "red"          # the name a query gives in `model`
MAX_WPQ = 16


def served_document(src, channels, timestamp_column, model=MODEL):
    """The promoted blueprint's document with an atai-records serving source and sink, at step 1."""
    doc = copy.deepcopy(src["document"])
    doc.pop("blueprint_id", None)
    doc["values"]["step_size"] = 1
    endpoint = {"format": "serving", "kind": "http", "protocols": ["atai-records"]}
    doc["connectors"] = {
        "source": {"key": "RecordsSource", "config": {
            **endpoint, "models": [model],
            "data_columns": [{"dtype": "f32", "names": channels}], "timestamp_column": timestamp_column}},
        "sink": {"key": "RecordsSink", "config": dict(endpoint)},
    }
    doc["name"] = f"{src['name']} (served, records)"
    return doc


def served_blueprint(promoted, channels, timestamp_column):
    key = f"{promoted}-served-records"
    try:
        request("GET", f"{agents()}/blueprints/{key}")
        return key
    except RuntimeError as e:
        if "(404)" not in str(e):
            raise
    doc = served_document(request("GET", f"{agents()}/blueprints/{promoted}"), channels, timestamp_column)
    bp = request("POST", f"{agents()}/blueprints", {
        "blueprint_key": key, "name": doc["name"], "document": doc,
        "description": f"{promoted} served with atai-records, step 1."})
    log(f"created served blueprint {key} ({bp['id']})")
    return key


def query_body(chunk, model=MODEL):
    """One query: a list per channel of the chunk's rows (window + windows - 1 of them)."""
    return {"model": model, "query": "", "events": [{"type": "data.numeric_array",
            "event_data": {"contents": [chunk[:, c].tolist() for c in range(chunk.shape[1])]}}]}


def span(labels, background, window, context, n_windows):
    """(first window, windows) ending within `context` rows of the file's event rows;
    every window when there are none. Window k ends on row k + window - 1."""
    rows = [i for i, lab in enumerate(labels) if lab != background]
    if not rows:
        return 0, n_windows
    first = max(0, rows[0] - context - window + 1)
    last = min(n_windows - 1, rows[-1] + context - window + 1)
    return first, last - first + 1


def outcome(preds, first, window, labels, background):
    """Each incident's delay (rows) and the false alarms, score.py's rules, for windows
    first, first + 1, ... predicting `preds`."""
    r = score_windows(labels, [(first + k, first + k + window, p) for k, p in enumerate(preds)], background)
    return {"incidents": [i["delay_rows"] for i in r["incidents"]], "false_alarms": r["false_alarms"]}


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
    ap.add_argument("--name", default="red")
    ap.add_argument("--files", nargs="+", help="delivery files by name, e.g. inc04 (default: all)")
    ap.add_argument("--context", type=int, default=120,
                    help="replay the windows ending within this many rows of an event row (default 120)")
    ap.add_argument("--whole", action="store_true", help="replay every window of each file")
    ap.add_argument("--windows-per-query", type=int, default=1, help=f"1 = live, one window per row; up to {MAX_WPQ}")
    a = ap.parse_args(argv)
    load_dotenv()
    if "api.dev." not in root():
        sys.exit("agent serving is on dev only for now")
    check_auth(log)
    m = load_manifest(a.roles)
    test = json.load(open(os.path.join(a.out, "test_state.json")))
    deliver = json.load(open(os.path.join(a.out, "deliver_state.json")))
    if deliver.get("values", {}).get("step_size") != 1:
        sys.exit("serve.py compares against deliver.py's output at step 1: rerun deliver.py with --step 1")
    bp = test["blueprint"]
    channels, window, background = m["channels"], bp["values"]["window_size"], m["background"]
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
        if len(batch) != len(data) - window + 1:
            sys.exit(f"{name}: the delivery output has {len(batch)} windows, expected {len(data) - window + 1}")
        first, n = (0, len(batch)) if a.whole else span(labels, background, window, a.context, len(batch))
        plan.append((name, data[channels].to_numpy(dtype=float), labels, batch, first, n))
    wpq = max(1, min(a.windows_per_query, MAX_WPQ))
    log(f"{len(plan)} file(s), {sum(p[5] for p in plan):,} windows, {wpq} per query; model {bp['key']}")

    key = served_blueprint(bp["key"], channels, m["timestamp_column"])
    sf = os.path.join(a.out, "serve_state.json")
    if os.path.exists(sf):
        sys.exit(f"{sf} holds an earlier replay's agent and bundle: run cleanup.py first")
    ex = {"bundle": request("POST", f"{agents()}/bundles", {"blueprint": key, "name": f"{a.name} serving"})["id"],
          "agent": None}
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
        for name, X, _, _, first, n in plan:
            got = served[name] = []
            for k in range(0, n, wpq):
                count = min(wpq, n - k)
                lo = first + k
                t = time.time()
                answer = request("POST", f"{agents()}/instances/{ex['agent']}/query",
                                 query_body(X[lo:lo + window + count - 1]))
                latency.append(time.time() - t)
                records = (answer.get("response") or {}).get("data")
                if not isinstance(records, list) or len(records) != count:
                    sys.exit(f"{name} window {lo}: expected {count} records, got {str(answer)[:300]}")
                got.extend(records)
            log(f"  {name}: {n} windows, {time.time() - t0:.0f} s so far")
    finally:
        if end(ex):
            os.remove(sf)
            log(f"cancelled and deleted {ex['agent']}, deleted {ex['bundle']}")
    elapsed = time.time() - t0

    rows, per_file = [], []
    for name, _, labels, batch, first, n in plan:
        ref, got = batch[first:first + n], served[name]
        for k, (s, b) in enumerate(zip(got, ref)):
            diff = max(abs(float(s[f"p_{c}"]) - float(b[f"p_{c}"])) for c in classes)
            rows.append({"file": name, "window": first + k, "end_row": first + k + window - 1,
                         "batch_state": b["predicted_state"], "served_state": s["predicted_state"],
                         **{f"served_p_{c}": float(s[f"p_{c}"]) for c in classes},
                         "max_p_diff": diff, "served_invalid": str(s["invalid"]).lower() == "true",
                         "same_state": s["predicted_state"] == b["predicted_state"], "same_p": diff < 1e-4})
        per_file.append({"file": name, "windows": n, "end_rows": [first + window - 1, first + n + window - 2],
                         "served": outcome([s["predicted_state"] for s in got], first, window, labels, background),
                         "batch": outcome([b["predicted_state"] for b in ref], first, window, labels, background)})

    df = pd.DataFrame(rows)
    sdir = os.path.join(a.out, "serving")
    os.makedirs(sdir, exist_ok=True)
    stem = os.path.join(sdir, f"{'_'.join(p[0] for p in plan)}{'_whole' if a.whole else ''}_wpq{wpq}")
    df.to_csv(stem + ".csv", index=False)
    ms = sorted(x * 1000 for x in latency)
    summary = {"blueprint": bp["key"], "served_blueprint": key, "files": per_file,
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
