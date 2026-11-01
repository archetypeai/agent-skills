#!/usr/bin/env python3
"""Path 2, step 6: the training windows' embeddings, and a 2-D map of them a live app can draw.

    python library_embeddings.py --roles roles --out out                    # batch: one run per training file
    python library_embeddings.py --roles roles --out out --serving          # a serving agent, a query per window
    python library_embeddings.py --roles roles --out out --serving --agent agt_...   # one already running
    python library_embeddings.py --roles roles --out out --view             # the map again, no platform calls
    python library_embeddings.py --roles roles --out out --view-dir ../my-app/static

The promoted classifier is one prototype per class (`strategy: centroid`), but it is a registry
artifact with no download route, and trials and evals return metrics and predictions, not
embeddings. So the training windows are encoded again on the promoted blueprint with
`output_embeddings: true`, a bundle value: no new blueprint, the same encoder on the same
windows (the trial's window, tiled from each file's row 0 at --step). Each class's mean stands
in for its prototype.

Two routes, the same points:
  batch (default)  a bundle at --step, one run per training file, --parallel at a time. The
                   platform cuts the windows, the outputs are files. Each run takes a job slot.
  --serving        a serving agent, one window per /query, cut here exactly as the platform
                   cuts them. One agent for every file: no job slots or queue, and on the free
                   trial no 2-hour cap. With --agent, an agent of the promoted blueprint
                   already running with output_embeddings (a live app's) is used and left
                   running; otherwise one is started and deleted at the end, however the run
                   ends (its ids in <out>/library_serving_state.json until then).

--step 4 (the default) keeps a quarter of the training windows: the same cloud, a quarter of
the download (the pump's 12,059 windows at step 1 are 255 MB of CSV).

The map is PCA 1-2 of the embeddings (the channels' 768 values each, concatenated in the
manifest's order), PC 1 turned so the background class sits on its positive side and PC 2 so
the first event class does. A served window lands in the same coordinates: (e - mean) . row.
It is a drawing, not the classifier: the classifier compares in every dimension.

Writes <out>/library_embeddings/<file>.csv (a row per window: timestamps, prediction,
embeddings), <out>/library_embeddings.json (the summary) and
<view-dir>/red-embedding-landscape.json (default <out>; the format the pump quickstart's
app-serving/ reads from its static/). Batch ids go to <out>/library_state.json, so a rerun
resumes and cleanup.py deletes them; a rerun for another blueprint stops rather than mixing
two classifiers' windows.
"""
import argparse
import csv
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import agents, check_auth, load_dotenv, request, root, same_deployment, upload_file  # noqa: E402
from common import load_manifest, log, stamp  # noqa: E402
from deliver import download, poll  # noqa: E402
from serve import MODEL, end, query_body, stop  # noqa: E402

WORKERS = 4
VIEW = "red-embedding-landscape.json"
# Checkpointing off: with it on, a run's output is written in pieces as it goes, and a file of
# 768-value cells per channel is large enough to make that slow.
CHECKPOINTING_OFF = {"checkpointing": {"trigger_after_input_byte_size": 0, "trigger_after_delay_ms": 0}}


def starts(rows, window, step):
    """First row of each window, as the platform tiles a file: from row 0, every `step` rows."""
    return list(range(0, rows - window + 1, step)) if rows >= window else []


def out_path(out, f):
    return os.path.join(out, "library_embeddings", os.path.basename(f["file"]))


def run_batch(a, files, key, step, state, save):
    """One batch run per training file, --parallel at a time; each output saved once complete."""
    if not state.get("bundle"):
        values = {"output_embeddings": True, "step_size": step, "runner_config": CHECKPOINTING_OFF}
        state["bundle"] = request("POST", f"{agents()}/bundles", {
            "blueprint": key, "name": f"{a.name} library embeddings", "values": values})["id"]
        save()
        log(f"bundle {state['bundle']} on {key} (step {step}, embeddings on)")
    runs = state.setdefault("runs", {})

    def one(f):
        src, dst = os.path.join(a.roles, f["file"]), out_path(a.out, f)
        if os.path.exists(dst):
            return True
        if f["file"] not in runs:
            fid = upload_file(src, f"{a.name}-lib-{os.path.basename(src)[:-4]}-{stamp()}.csv")["file_id"]
            runs[f["file"]] = request("POST", f"{agents()}/bundles/{state['bundle']}/run",
                                      {"connectors": {"source": [{"type": "file", "id": fid, "format": "csv"}]}})["id"]
            save()
            log(f"  {f['file']}: run {runs[f['file']]}")
        status = poll(runs[f["file"]])
        if download(runs[f["file"]], dst, src, a.timestamp_column):
            log(f"  {f['file']}: complete")
            return True
        log(f"  {f['file']}: {status}, no complete output; rerun to retry")
        if status in ("failed", "canceled", "cancelled"):
            runs.pop(f["file"])
            save()
        return False
    with ThreadPoolExecutor(a.parallel) as pool:
        return all(pool.map(one, files))


def serving_agent(a, key, sp):
    """A serving agent of the promoted blueprint with embeddings on, once running. Its ids are
    in `sp` from the moment each exists, so a kill leaves them for cleanup.py."""
    ex = {"endpoint": root(), "agent": None, "bundle": None}
    ex["bundle"] = request("POST", f"{agents()}/bundles", {"blueprint": key, "name": f"{a.name} library serving",
                                                           "values": {"output_embeddings": True}})["id"]
    json.dump(ex, open(sp, "w"), indent=1)
    ex["agent"] = request("POST", f"{agents()}/bundles/{ex['bundle']}/run", {"mode": "serving"})["id"]
    json.dump(ex, open(sp, "w"), indent=1)
    t0 = time.time()
    while (s := request("GET", f"{agents()}/instances/{ex['agent']}").get("status")) != "running":
        if s in ("failed", "cancelled", "canceled", "completed") or time.time() - t0 > 900:
            sys.exit(f"serving agent {ex['agent']} is {s}, not running")
        time.sleep(3)
    log(f"serving agent {ex['agent']} running")
    return ex


def run_serving(a, files, agent, window, step, channels):
    """Each training file's windows, one /query each, written in the batch output's shape."""
    def query(chunk):
        rec = request("POST", f"{agents()}/instances/{agent}/query", query_body(chunk, a.model),
                      idempotent=True)["response"]["data"][0]
        missing = [c for c in channels if f"embedding_{c}" not in rec]
        if missing:
            sys.exit(f"the agent's answer has no embedding_{missing[0]}: its bundle needs output_embeddings: true")
        return rec
    t0, n = time.time(), 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for f in files:
            dst = out_path(a.out, f)
            if os.path.exists(dst):
                continue
            df = pd.read_csv(os.path.join(a.roles, f["file"]))
            X, ts = df[channels].to_numpy(float), df[a.timestamp_column].to_numpy(float)
            first = starts(len(X), window, step)
            recs = list(pool.map(query, [X[i:i + window] for i in first]))
            rows = [{"start_timestamp": ts[i], "finish_timestamp": ts[i + window - 1],
                     "predicted_state": r.get("predicted_state"), "invalid": r.get("invalid", False),
                     **{f"embedding_{c}": json.dumps(r[f"embedding_{c}"]) for c in channels}}
                    for i, r in zip(first, recs)]
            pd.DataFrame(rows).to_csv(dst + ".part", index=False)
            os.replace(dst + ".part", dst)
            n += len(rows)
            log(f"  {f['file']}: {len(rows):,} windows")
    log(f"served {n:,} windows in {time.time() - t0:.0f} s")


def read_embeddings(path, channels):
    """[windows, channels x 768] in the channels' order; invalid windows left out."""
    csv.field_size_limit(1 << 30)
    vecs = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("finish_timestamp") == "finish_timestamp" or str(r.get("invalid")).lower() == "true":
                continue
            missing = [c for c in channels if f"embedding_{c}" not in r]
            if missing:
                sys.exit(f"{path}: no embedding_{missing[0]} column: was the bundle made with output_embeddings?")
            vecs.append(np.concatenate([np.asarray(json.loads(r[f"embedding_{c}"]), dtype=np.float32) for c in channels]))
    return np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)


def landscape(classes, background, by_class, window, step, n_fitted):
    """PCA 1-2 of every class's windows: PC 1 turned toward the background class, PC 2 toward
    the first event class; each class's mean (its stand-in prototype) projected the same way."""
    present = [c for c in classes if len(by_class.get(c, ()))]
    X = np.concatenate([by_class[c] for c in present]).astype(np.float64)
    y = np.concatenate([np.full(len(by_class[c]), classes.index(c)) for c in present])
    mean = X.mean(axis=0)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:2].copy()
    means = {c: by_class[c].astype(np.float64).mean(axis=0) - mean for c in present}
    event = next((c for c in present if c != background), None)
    for k, c in ((0, background), (1, event)):
        if c in means and means[c] @ components[k] < 0:
            components[k] = -components[k]
    coords = Xc @ components.T
    var = S ** 2 / max(1, len(X) - 1)
    protos = [(means[c] @ components.T).tolist() if c in means else None for c in classes]
    return {
        "classes": classes,
        "prototypes": [[round(float(v), 3) for v in p] if p else None for p in protos],
        "prototype_source": "recomputed: the mean of each class's training windows",
        "dim": int(X.shape[1]),
        "n": int(len(X)),
        "n_fitted": int(n_fitted),
        "window_size": int(window),
        "step_size": int(step),
        "explained_variance": [round(float(v), 4) for v in var[:2] / var.sum()],
        "mean": [round(float(v), 5) for v in mean],
        "components": [[round(float(v), 6) for v in row] for row in components],
        "shots": [{"x": round(float(coords[i, 0]), 3), "y": round(float(coords[i, 1]), 3), "c": int(y[i])}
                  for i in range(len(X))],
    }


def write_view(a, m, meta, t0):
    """The map from the saved outputs in <out>/library_embeddings/."""
    channels, classes, window, step = m["channels"], m["classes"], meta["window_size"], meta["step_size"]
    by_class, per_file, short, n_fitted = {}, {}, [], 0
    for f in m["training"]:
        v = read_embeddings(out_path(a.out, f), channels)
        if len(v):
            by_class.setdefault(f["class"], []).append(v)
        per_file[f["file"]] = len(v)
        want = len(starts(f["n_rows"], window, step))
        n_fitted += max(0, f["n_rows"] - window + 1)
        if len(v) != want:
            short.append(f"{f['file']} {len(v)}/{want}")
    if not by_class:
        sys.exit(f"no embeddings in {a.out}/library_embeddings/")
    view = landscape(classes, m["background"], {c: np.concatenate(v) for c, v in by_class.items()},
                     window, step, n_fitted)
    view_dir = a.view_dir or a.out
    os.makedirs(view_dir, exist_ok=True)
    with open(os.path.join(view_dir, VIEW), "w") as fh:
        json.dump({**view, "blueprint": meta["blueprint"], "route": meta["route"], "channels": channels}, fh,
                  separators=(",", ":"))
    per_class = {c: int(sum(len(v) for v in by_class.get(c, []))) for c in classes}
    summary = {**meta, "windows": view["n"], "per_file": per_file, "per_class": per_class, "dim": view["dim"],
               "explained": view["explained_variance"], "prototypes": view["prototypes"],
               "seconds": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(a.out, "library_embeddings.json"), "w"), indent=1)
    ev = view["explained_variance"]
    print(f"\nlibrary: {view['n']:,} windows of {n_fitted:,} ({', '.join(f'{c} {n:,}' for c, n in per_class.items())}), "
          f"window {window}, step {step} ({meta['route']})")
    print(f"  PCA 1-2 explain {ev[0]:.1%} and {ev[1]:.1%} -> {os.path.join(view_dir, VIEW)}")
    if short:
        print(f"  windows short of the tiling (invalid, or an output cut short): {', '.join(short[:5])}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="red", help="prefixes the uploads and names the bundles")
    ap.add_argument("--step", type=int, default=4, help="stride over the training files (default 4)")
    ap.add_argument("--serving", action="store_true", help="embed through a serving agent, a query per window")
    ap.add_argument("--agent", help="with --serving: an agent already running the promoted blueprint with embeddings")
    ap.add_argument("--model", default=MODEL, help="the `model` a query names")
    ap.add_argument("--parallel", type=int, default=3, help="batch runs at a time (the free trial runs 3 jobs)")
    ap.add_argument("--view", action="store_true", help="only rebuild the map from <out>/library_embeddings/")
    ap.add_argument("--view-dir", help=f"where {VIEW} goes (default <out>)")
    a = ap.parse_args(argv)
    m = load_manifest(a.roles)
    a.timestamp_column = m["timestamp_column"]
    t0 = time.time()
    if a.view:
        meta = json.load(open(os.path.join(a.out, "library_embeddings.json")))
        return write_view(a, m, {k: meta[k] for k in ("blueprint", "route", "window_size", "step_size")}, t0)
    load_dotenv()
    check_auth(log)
    ts_path = os.path.join(a.out, "test_state.json")
    if not os.path.exists(ts_path):
        sys.exit(f"no {ts_path}: run promote_and_test.py first")
    test = json.load(open(ts_path))
    if not same_deployment(test):
        sys.exit(f"the blueprint in {ts_path} is on {test['endpoint']}, a different deployment from {root()}")
    key = test["blueprint"]["key"]
    meta = {"blueprint": key, "route": "serving" if a.serving else "batch",
            "window_size": test["blueprint"]["values"]["window_size"], "step_size": a.step}
    sp = os.path.join(a.out, "library_state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    if not same_deployment(state):
        sys.exit(f"{sp} is from {state['endpoint']}, a different deployment from {root()}: move it aside")
    if state.get("blueprint", key) != key or state.get("step", a.step) != a.step:
        sys.exit(f"{sp} and {a.out}/library_embeddings/ are for {state.get('blueprint')} at step {state.get('step')}, "
                 f"not {key} at step {a.step}: move them aside (cleanup.py deletes the runs) so the map holds "
                 f"one classifier's windows")
    state.update({"endpoint": root(), "blueprint": key, "step": a.step})

    def save():
        json.dump(state, open(sp, "w"), indent=1)
    save()
    os.makedirs(os.path.join(a.out, "library_embeddings"), exist_ok=True)
    if a.serving:
        ssp = os.path.join(a.out, "library_serving_state.json")
        if os.path.exists(ssp):
            sys.exit(f"{ssp} exists: an earlier run's serving agent may still be up; run cleanup.py --yes first")
        if a.agent and (b := request("GET", f"{agents()}/instances/{a.agent}").get("blueprint")) != key:
            sys.exit(f"agent {a.agent} runs {b}, not the promoted {key}")
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        ex = None
        try:
            agent = a.agent or (ex := serving_agent(a, key, ssp))["agent"]
            run_serving(a, m["training"], agent, meta["window_size"], a.step, m["channels"])
        finally:
            if ex is None and os.path.exists(ssp):      # failed while starting: delete what it made
                ex = json.load(open(ssp))
            if ex and end(ex):
                os.remove(ssp)
                log(f"deleted serving agent {ex['agent']} and bundle {ex['bundle']}")
    elif not run_batch(a, m["training"], key, a.step, state, save):
        sys.exit("not every training file has a complete output: rerun to resume")
    write_view(a, m, meta, t0)
    if state.get("bundle"):
        log(f"next: python cleanup.py --out {a.out} --yes deletes the batch runs and their bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
