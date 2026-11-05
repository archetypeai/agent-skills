#!/usr/bin/env python3
"""Path 2, step 6: the library's embeddings, and a 2-D view of them a live app can draw.

    python library_embeddings.py --roles roles --out out                    # batch: one session per library file
    python library_embeddings.py --roles roles --out out --serving          # a serving session, a query per window
    python library_embeddings.py --roles roles --out out --serving --session ses_...   # one already running
    python library_embeddings.py --roles roles --out out --view             # the view again, no platform calls
    python library_embeddings.py --roles roles --out out --view-dir ../my-app/static

The promoted detector is fitted on the library's windows, but it is a registry artifact with
no download route, so those windows' embeddings can't be read back. They can be recomputed:
a package carries no values, so an embeddings blueprint comes first, a 1-trial optimization
at the promoted trial's values with `output_embeddings: true`, promoted (resumable through
<out>/library_state.json). The same encoder on the same windows (the trial's window and step,
tiled from each file's row 0).

Two routes, the same points (SKAB through bundles, 2026-10-31: 1,950 windows both ways,
embeddings within 4.0e-6, scores within 2e-7):
  batch (default)  a package of the embeddings blueprint, a batch deployment, one session per
                   library file. The platform cuts the windows, the outputs are files. Each
                   session takes a job slot.
  --serving        a serving deployment and one session, a window per /query, cut here
                   exactly as the platform cuts them: no job slots or queue, and on the free
                   trial no 2-hour cap. With --session, a session already running the
                   embeddings blueprint (a live app's) is used and left running; otherwise the
                   package, deployment and session are made and deleted at the end, however
                   the run ends (their ids in <out>/library_serving_state.json until then).

The view is PCA 1-2 of the library's embeddings (the channels' 768 values each, concatenated
in the manifest's order), each axis at unit spread. A served window lands in the same
coordinates: (e - mean) . row. It is a drawing, not the detector: the detector scores in more
dimensions, with its own PCA and the per-channel features.

Writes <out>/library_embeddings/<file>.csv (a row per window: timestamps, score, embeddings),
<out>/library_embeddings.json (the summary) and <view-dir>/library-view.{json,bin} (default
<out>). Batch ids go to <out>/library_state.json, so a rerun resumes and cleanup.py deletes
the sessions, deployment and package (the embeddings blueprint stays). A rerun for another
blueprint or step stops rather than mixing two detectors' windows: it checks
library_state.json and library_embeddings.json. A rerun with nothing new to embed starts
nothing, and keeps the route the outputs were made by.
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
from atai_http import (agents, check_auth, create_package, deploy, deployment_ready, lifecycle_url,  # noqa: E402
                       list_trials, load_dotenv, open_session, package_ready, request, root, same_deployment,
                       upload_all, upload_file, wait, wait_for)
from common import base_values, cache_path, example, load_manifest, log, role_paths, stamp  # noqa: E402
from deliver import download, poll  # noqa: E402
from promote_and_test import promote  # noqa: E402
from serve import MODEL, end, query_body, stop  # noqa: E402

WORKERS = 4


def starts(rows, window, step):
    """First row of each window, as the platform tiles a file: from row 0, every `step` rows."""
    return list(range(0, rows - window + 1, step)) if rows >= window else []


def out_path(out, f):
    return os.path.join(out, "library_embeddings", os.path.basename(f["file"]))


def embeddings_blueprint(a, m, test, state, save):
    """{id, key} of a blueprint at the promoted trial's values with output_embeddings on: a
    1-trial optimization of the project's copy of `ad`, then promoted. Resumable through `state`."""
    if state.get("embeddings_blueprint"):
        return state["embeddings_blueprint"]
    vals = test["blueprint"]["values"]
    if not state.get("embeddings_optimization"):
        base = json.load(open(os.path.join(a.out, "project_state.json")))["base"]
        lib, val = role_paths(a.roles, m, "library"), role_paths(a.roles, m, "validation")
        ids = upload_all(lib + val, a.roles, cache_path(a.out), prefix=a.name, log=log)
        pinned = {k: v for k, v in vals.items() if k != "n_neighbors"}
        o = request("POST", f"{agents()}/optimizations", {
            "name": f"{a.name} embeddings {stamp()}", "blueprint_id": base["id"], "primary": "anomaly.macro_f1",
            "overrides": {"values": {**base_values(m), **pinned, "output_score": True, "output_embeddings": True}},
            # the search space can't be empty: k, with the trial's one value
            "search_space": {"parameters": {"n_neighbors": {
                "kind": "value", "spec": {"type": "categorical", "values": [vals["n_neighbors"]]}}}},
            "budget": {"max_trials": 1},
            "training_examples": [example(os.path.basename(p), ids[p], training=True) for p in lib],
            "validation_examples": [example(os.path.basename(p), ids[p]) for p in val]})
        state["embeddings_optimization"] = o["id"]
        save()
        log(f"optimization {o['id']}: 1 trial at {test['blueprint']['key']}'s values, embeddings on")
    opt_id = state["embeddings_optimization"]
    opt = wait(f"{agents()}/optimizations/{opt_id}", opt_id, log=log)
    trial = next((t for t in list_trials(opt_id) if t["status"] == "completed"), None)
    if opt["status"] != "completed" or not trial:
        sys.exit(f"{opt_id} is {opt['status']} with no completed trial: move {a.out}/library_state.json aside to try again")
    bp = promote({"optimization": opt_id, "pinned": {k: v for k, v in vals.items() if k != "n_neighbors"}},
                 trial, f"{a.name}-embeddings")
    state["embeddings_blueprint"] = {"id": bp["id"], "key": bp["key"]}
    save()
    log(f"promoted {trial['id']} -> {bp['key']} ({bp['id']})")
    return state["embeddings_blueprint"]


def run_batch(a, files, emb, window, step, state, save):
    """One batch session per library file, --parallel at a time; each output saved once complete."""
    if not state.get("deployment"):
        if not state.get("package"):
            state["package"] = create_package(emb["id"], f"{a.name} library embeddings")["id"]
            save()
        package_ready(state["package"])
        state["deployment"] = deploy(state["package"], "batch")["id"]
        save()
        deployment_ready(state["deployment"])
        log(f"package {state['package']} of {emb['key']} (window {window}, step {step}, embeddings on), "
            f"batch deployment {state['deployment']}")
    runs = state.setdefault("runs", {})

    def one(f):
        src, dst = os.path.join(a.roles, f["file"]), out_path(a.out, f)
        if os.path.exists(dst):
            return True
        if f["file"] not in runs:
            fid = upload_file(src, f"{a.name}-lib-{os.path.basename(src)[:-4]}-{stamp()}.csv")["file_id"]
            runs[f["file"]] = open_session(state["deployment"], fid)["id"]
            save()
            log(f"  {f['file']}: session {runs[f['file']]}")
        status = poll(runs[f["file"]])
        if download(runs[f["file"]], dst, src, a.timestamp_column):
            return True
        log(f"  {f['file']}: {status}, no complete output; rerun to retry")
        if status in ("failed", "canceled", "cancelled"):
            runs.pop(f["file"])
            save()
        return False
    with ThreadPoolExecutor(a.parallel) as pool:
        return all(pool.map(one, files))


def serving_session(a, emb, sp):
    """A serving session on the embeddings blueprint `emb`, once running: its package, a serving
    deployment, a session. Each id is in `sp` before the script waits on it, so a kill leaves
    them for cleanup.py."""
    ex = {"endpoint": root(), "package": None, "deployment": None, "session": None}
    for kind, make, ready in (("package", lambda: create_package(emb["id"], f"{a.name} library serving"), package_ready),
                              ("deployment", lambda: deploy(ex["package"], "serving"), deployment_ready),
                              ("session", lambda: open_session(ex["deployment"]), None)):
        ex[kind] = make()["id"]
        json.dump(ex, open(sp, "w"), indent=1)
        if ready:
            ready(ex[kind])
    wait_for("sessions", ex["session"], "running", "pending", timeout_s=900)
    log(f"serving session {ex['session']} running (deployment {ex['deployment']}, package {ex['package']})")
    return ex


def session_blueprints(session_id):
    """The blueprint ids a session runs: its deployment's package's."""
    dpl = request("GET", lifecycle_url("sessions", session_id))["deployment_id"]
    pkg = request("GET", lifecycle_url("deployments", dpl))["package_id"]
    return request("GET", lifecycle_url("packages", pkg)).get("blueprint_ids", [])


def run_serving(a, files, session, window, step, channels):
    """Each library file's windows, one /query each, written in the batch output's shape."""
    def query(chunk):
        rec = request("POST", lifecycle_url("sessions", session) + "/query", query_body(chunk, a.model),
                      idempotent=True)["response"]["data"][0]
        missing = [c for c in channels if f"embedding_{c}" not in rec]
        if missing:
            sys.exit(f"the session's answer has no embedding_{missing[0]}: its blueprint needs output_embeddings: true")
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
            rows = [{"start_timestamp": ts[i], "finish_timestamp": ts[i + window - 1], "invalid": r.get("invalid", False),
                     "anomaly_score": r["anomaly_score"],
                     **{f"embedding_{c}": json.dumps(r[f"embedding_{c}"]) for c in channels}}
                    for i, r in zip(first, recs)]
            pd.DataFrame(rows).to_csv(dst + ".part", index=False)
            os.replace(dst + ".part", dst)
            n += len(rows)
            log(f"  {f['file']}: {len(rows):,} windows")
    log(f"served {n:,} windows in {time.time() - t0:.0f} s")


def read_windows(path, channels):
    """[windows, channels x 768] in the channels' order, and the scores; invalid windows left out."""
    csv.field_size_limit(1 << 30)
    vecs, scores = [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("invalid")).lower() == "true":
                continue
            vecs.append(np.concatenate([np.asarray(json.loads(r[f"embedding_{c}"]), dtype=np.float32) for c in channels]))
            scores.append(float(r["anomaly_score"]))
    return np.stack(vecs), np.array(scores)


def pca2(X):
    """mean [dim], rows [2, dim] (each axis at unit spread), ref [n, 2], explained [2]."""
    mean = X.mean(axis=0)
    Xc = X - mean
    if len(X) <= X.shape[1]:            # through the n x n Gram matrix: fewer windows than dimensions
        val, vec = np.linalg.eigh(Xc @ Xc.T)
        top = np.argsort(val)[::-1][:2]
        axes = (Xc.T @ vec[:, top]) / np.sqrt(val[top])
    else:
        val, vec = np.linalg.eigh(Xc.T @ Xc)
        top = np.argsort(val)[::-1][:2]
        axes = vec[:, top]
    var = val[top] / (len(X) - 1)
    rows = (axes / np.sqrt(var)).T
    return mean, rows, Xc @ rows.T, var / Xc.var(axis=0, ddof=1).sum()


def write_view(a, m, meta, t0):
    channels = m["channels"]
    vecs, scores, per_file, short = [], [], {}, []
    for f in m["library"]:
        v, s = read_windows(out_path(a.out, f), channels)
        vecs.append(v)
        scores.append(s)
        per_file[f["file"]] = len(v)
        want = len(starts(f["rows"], meta["window_size"], meta["step_size"]))
        if len(v) != want:
            short.append(f"{f['file']} {len(v)}/{want}")
    X, sc = np.concatenate(vecs).astype(np.float64), np.concatenate(scores)
    n, dim = X.shape
    mean, rows, ref, explained = pca2(X)
    view_dir = a.view_dir or a.out
    os.makedirs(view_dir, exist_ok=True)
    layout, offset = {}, 0
    with open(os.path.join(view_dir, "library-view.bin"), "wb") as fh:
        for k, arr in {"ref": ref, "score": sc, "mean": mean, "pc0": rows[0], "pc1": rows[1]}.items():
            arr = np.ascontiguousarray(arr, dtype="<f4")
            fh.write(arr.tobytes())
            layout[k] = {"offset": offset, "shape": list(arr.shape)}
            offset += arr.nbytes
    thr = meta["threshold"]
    with open(os.path.join(view_dir, "library-view.json"), "w") as fh:
        json.dump({"note": ("The promoted detector's library, recomputed with output_embeddings on, in PCA 1-2 of "
                            "its embeddings, each axis at unit spread. For drawing only."),
                   "blueprint": meta["blueprint"], "route": meta["route"], "channels": channels,
                   "embeddingDim": dim // len(channels), "totalDim": dim, "nReference": n,
                   "windowSize": meta["window_size"], "stepSize": meta["step_size"], "threshold": thr,
                   "explainedVariance": [round(float(e), 4) for e in explained],
                   "layout": layout, "dtype": "float32-le"}, fh, indent=1)
    summary = {**meta, "windows": n, "per_file": per_file, "dim": dim,
               "score_min": float(sc.min()), "score_median": float(np.median(sc)), "score_max": float(sc.max()),
               "over_threshold": float((sc > thr).mean()), "explained": [float(e) for e in explained],
               "view_bytes": offset, "seconds": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(a.out, "library_embeddings.json"), "w"), indent=1)
    print(f"\nlibrary: {n:,} windows from {len(m['library'])} files, window {meta['window_size']}, "
          f"step {meta['step_size']} ({meta['route']})")
    print(f"  scores {sc.min():.3f}-{sc.max():.3f} (median {np.median(sc):.3f}); "
          f"{(sc > thr).mean():.2%} over the threshold {thr:.3f}")
    print(f"  PCA 1-2 explain {explained[0]:.1%} and {explained[1]:.1%} -> {view_dir}/library-view.bin "
          f"({offset / 1e3:.0f} KB), library-view.json")
    if short:
        print(f"  windows short of the tiling (invalid, or an output cut short): {', '.join(short[:5])}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="ad", help="prefixes the uploads and names the platform objects")
    ap.add_argument("--serving", action="store_true", help="embed through a serving session, a query per window")
    ap.add_argument("--session", help="with --serving: a session already running the embeddings blueprint")
    ap.add_argument("--model", default=MODEL, help="the `model` a query names")
    ap.add_argument("--parallel", type=int, default=3, help="batch sessions at a time")
    ap.add_argument("--view", action="store_true", help="only rebuild the view from <out>/library_embeddings/")
    ap.add_argument("--view-dir", help="where library-view.{json,bin} go (default <out>)")
    a = ap.parse_args(argv)
    m = load_manifest(a.roles)
    a.timestamp_column = m["timestamp_column"]
    t0 = time.time()
    if a.view:
        meta = json.load(open(os.path.join(a.out, "library_embeddings.json")))
        return write_view(a, m, {k: meta[k] for k in ("blueprint", "route", "window_size", "step_size", "threshold")}, t0)
    load_dotenv()
    check_auth(log)
    test = json.load(open(os.path.join(a.out, "test_state.json")))
    if not same_deployment(test):
        sys.exit(f"the blueprint in {a.out}/test_state.json is on {test['endpoint']}, a different deployment from {root()}")
    key, values = test["blueprint"]["key"], test["blueprint"]["values"]
    meta = {"blueprint": key, "route": "serving" if a.serving else "batch", "window_size": values["window_size"],
            "step_size": values["step_size"], "threshold": float(values["threshold"])}
    # library_embeddings.json outlives a moved or deleted library_state.json
    summary = os.path.join(a.out, "library_embeddings.json")
    made = json.load(open(summary)) if os.path.exists(summary) else {}
    if made and (made.get("blueprint"), made.get("step_size")) != (key, meta["step_size"]):
        sys.exit(f"{summary} and <out>/library_embeddings/ are for {made.get('blueprint')} at step "
                 f"{made.get('step_size')}, not the promoted {key} at step {meta['step_size']}: move them aside "
                 f"so the view holds one detector's windows")
    sp = os.path.join(a.out, "library_state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    if not same_deployment(state):
        sys.exit(f"{sp} is from {state['endpoint']}, a different deployment from {root()}: move it aside")
    if state.get("blueprint", key) != key:
        sys.exit(f"{sp} and <out>/library_embeddings/ are for {state['blueprint']}, not the promoted {key}: "
                 f"move them aside (cleanup.py deletes its sessions) so the view holds one detector's windows")
    if state.get("bundle"):
        sys.exit(f"{sp} is from a bundle run: move it and {a.out}/library_embeddings/ aside")
    state.update({"endpoint": root(), "blueprint": key})

    def save():
        json.dump(state, open(sp, "w"), indent=1)
    save()
    os.makedirs(os.path.join(a.out, "library_embeddings"), exist_ok=True)
    if all(os.path.exists(out_path(a.out, f)) for f in m["library"]):
        log("every library file has its embeddings: nothing started")
        meta["route"] = made.get("route", meta["route"])         # keep how they were made
    elif a.serving:
        ssp = os.path.join(a.out, "library_serving_state.json")
        if os.path.exists(ssp):
            sys.exit(f"{ssp} exists: an earlier run's serving session may still be up; run cleanup.py --yes first")
        emb = embeddings_blueprint(a, m, test, state, save)
        if a.session and emb["id"] not in (bps := session_blueprints(a.session)):
            sys.exit(f"session {a.session} runs {bps}, not the embeddings blueprint {emb['key']} ({emb['id']})")
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        ex = None
        try:
            session = a.session or (ex := serving_session(a, emb, ssp))["session"]
            run_serving(a, m["library"], session, meta["window_size"], meta["step_size"], m["channels"])
        finally:
            if ex is None and os.path.exists(ssp):      # failed while starting: delete what it made
                ex = json.load(open(ssp))
            if ex and end(ex):
                os.remove(ssp)
                log(f"deleted session {ex['session']}, deployment {ex['deployment']} and package {ex['package']}")
    else:
        emb = embeddings_blueprint(a, m, test, state, save)
        if not run_batch(a, m["library"], emb, meta["window_size"], meta["step_size"], state, save):
            sys.exit("not every library file has a complete output: rerun to resume")
    write_view(a, m, meta, t0)
    if state.get("package"):
        log(f"next: python cleanup.py --out {a.out} --yes deletes the batch sessions, deployment and package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
