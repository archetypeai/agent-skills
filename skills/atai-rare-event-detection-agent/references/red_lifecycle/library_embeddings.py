#!/usr/bin/env python3
"""Path 2, step 6: the training windows' embeddings, and a 2-D map of them a live app can draw.

    python library_embeddings.py --roles roles --out out                    # batch: one session per training file
    python library_embeddings.py --roles roles --out out --serving          # a serving session, a query per window
    python library_embeddings.py --roles roles --out out --serving --session ses_...   # one already running
    python library_embeddings.py --roles roles --out out --view             # the map again, no platform calls
    python library_embeddings.py --roles roles --out out --view-dir ../my-app/static

The promoted classifier is one prototype per class (`strategy: centroid`), but it is a registry
artifact with no download route, and trials and evals return metrics and predictions, not
embeddings. So the training windows are encoded again. A package carries no values, so an
embeddings blueprint comes first: a 1-trial optimization of the project's copy of `red` at the
promoted trial's values, with `output_embeddings: true` and `step_size` --step, promoted
(resumable through <out>/library_state.json). The same encoder on the same windows (the
trial's window, tiled from each file's row 0 at --step). Each class's mean stands in for its
prototype.

Two routes, the same points:
  batch (default)  a package of the embeddings blueprint, a batch deployment, one session per
                   training file, --parallel at a time. The platform cuts the windows, the
                   outputs are files. Each session takes a job slot.
  --serving        a serving deployment of the embeddings blueprint and one session, a window
                   per /query, cut here exactly as the platform cuts them: no job slots or
                   queue, and on the free trial no 2-hour cap. With --session, a session
                   already running the embeddings blueprint (a live app's) is used and left
                   running; otherwise the package, deployment and session are made and
                   deleted at the end, however the run ends (their ids in
                   <out>/library_serving_state.json until then).

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
resumes and cleanup.py deletes the sessions, deployment and package (the embeddings blueprint
stays). A rerun for another blueprint or step stops rather than mixing two classifiers'
windows: it checks library_state.json and library_embeddings.json. A rerun that embeds
nothing new starts nothing and keeps the route the outputs were made by.

If a batch session hits a time limit (DeadlineExceeded: the free trial stops a job after 2 hours),
a plain rerun hits it again: rerun with a larger --step (fewer windows), or --serving.
"""
import argparse
import csv
import json
import os
import signal
import sys
import threading
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
VIEW = "red-embedding-landscape.json"


def starts(rows, window, step):
    """First row of each window, as the platform tiles a file: from row 0, every `step` rows."""
    return list(range(0, rows - window + 1, step)) if rows >= window else []


def out_path(out, f):
    return os.path.join(out, "library_embeddings", os.path.basename(f["file"]))


def embeddings_blueprint(a, m, test, state, save):
    """{id, key} of a blueprint at the promoted trial's values with output_embeddings on and
    step_size --step: a 1-trial optimization of the project's copy of `red`, then promoted.
    Resumable through `state`."""
    if state.get("embeddings_blueprint"):
        return state["embeddings_blueprint"]
    vals = {**test["blueprint"]["values"], "step_size": a.step}
    pinned = {k: v for k, v in vals.items() if k != "metric"}
    if not state.get("embeddings_optimization"):
        base = json.load(open(os.path.join(a.out, "project_state.json")))["base"]
        train = [(os.path.join(a.roles, f["file"]), f["class"]) for f in m["training"]]
        val = role_paths(a.roles, m, "validation")
        ids = upload_all([p for p, _ in train] + val, a.roles, cache_path(a.out), prefix=a.name, log=log)
        o = request("POST", f"{agents()}/optimizations", {
            "name": f"{a.name} embeddings {stamp()}", "blueprint_id": base["id"],
            "primary": "event.foreground_macro_f1",
            "overrides": {"values": {**base_values(m), **pinned, "output_embeddings": True}},
            # the search space can't be empty: the metric, with the trial's one value
            "search_space": {"parameters": {"metric": {
                "kind": "value", "spec": {"type": "categorical", "values": [vals["metric"]]}}}},
            "budget": {"max_trials": 1},
            "training_examples": [example(os.path.basename(p), ids[p], c) for p, c in train],
            "validation_examples": [example(os.path.basename(p), ids[p]) for p in val]})
        state["embeddings_optimization"] = o["id"]
        save()
        log(f"optimization {o['id']}: 1 trial at {test['blueprint']['key']}'s values, step {a.step}, embeddings on")
    opt_id = state["embeddings_optimization"]
    opt = wait(f"{agents()}/optimizations/{opt_id}", opt_id, log=log)
    trial = next((t for t in list_trials(opt_id) if t["status"] == "completed"), None)
    if opt["status"] != "completed" or not trial:
        sys.exit(f"{opt_id} is {opt['status']} with no completed trial: move {a.out}/library_state.json aside to try again")
    bp = promote({"optimization": opt_id, "pinned": pinned}, trial, f"{a.name}-embeddings")
    state["embeddings_blueprint"] = {"id": bp["id"], "key": bp["key"]}
    save()
    log(f"promoted {trial['id']} -> {bp['key']} ({bp['id']})")
    return state["embeddings_blueprint"]


def run_batch(a, files, emb, step, state, save):
    """One batch session per training file, --parallel at a time; each output saved once complete.
    Returns (every file complete, files embedded now)."""
    todo = [f for f in files if not os.path.exists(out_path(a.out, f))]
    if not todo:
        return True, 0
    if not state.get("deployment"):
        if not state.get("package"):
            state["package"] = create_package(emb["id"], f"{a.name} library embeddings")["id"]
            save()
        package_ready(state["package"])
        state["deployment"] = deploy(state["package"], "batch")["id"]
        save()
        deployment_ready(state["deployment"])
        log(f"package {state['package']} of {emb['key']} (step {step}, embeddings on), "
            f"batch deployment {state['deployment']}")
    runs = state.setdefault("runs", {})
    lock = threading.Lock()

    def one(f):
        src, dst = os.path.join(a.roles, f["file"]), out_path(a.out, f)
        if f["file"] not in runs:
            fid = upload_file(src, f"{a.name}-lib-{os.path.basename(src)[:-4]}-{stamp()}.csv")["file_id"]
            ses = open_session(state["deployment"], fid)["id"]
            with lock:
                runs[f["file"]] = ses
                save()
            log(f"  {f['file']}: session {ses}")
        status = poll(runs[f["file"]])
        if download(runs[f["file"]], dst, src, a.timestamp_column):
            log(f"  {f['file']}: complete")
            return True
        log(f"  {f['file']}: {status}, no complete output; rerun to retry")
        if status in ("failed", "canceled", "cancelled"):
            error = str(request("GET", lifecycle_url("sessions", runs[f["file"]])).get("error") or "")
            if "DeadlineExceeded" in error:
                log(f"  {f['file']}: {error[:200]} A plain rerun hits it again: rerun with a larger --step "
                    f"(fewer windows) or --serving (no job time limit)")
            with lock:
                runs.pop(f["file"])
                save()
        return False
    with ThreadPoolExecutor(a.parallel) as pool:
        return all(pool.map(one, todo)), len(todo)


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
    """Each training file's windows, one /query each, written in the batch output's shape."""
    def query(chunk):
        rec = request("POST", lifecycle_url("sessions", session) + "/query", query_body(chunk, a.model),
                      idempotent=True)["response"]["data"][0]
        missing = [c for c in channels if f"embedding_{c}" not in rec]
        if missing:
            sys.exit(f"the session's answer has no embedding_{missing[0]}: its blueprint needs output_embeddings: true")
        return rec
    t0, n, written = time.time(), 0, 0
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
            n, written = n + len(rows), written + 1
            log(f"  {f['file']}: {len(rows):,} windows")
    log(f"served {n:,} windows in {time.time() - t0:.0f} s")
    return written


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
                sys.exit(f"{path}: no embedding_{missing[0]} column: was its blueprint promoted with output_embeddings?")
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
        n_fitted += len(starts(f["n_rows"], window, meta.get("trial_step", 1)))   # the windows the prototypes averaged
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
        print(f"  windows off the tiling, got/expected (invalid windows, an output cut short, or another "
              f"step's outputs): {', '.join(short[:5])}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="red", help="prefixes the uploads and names the platform objects")
    ap.add_argument("--step", type=int, default=4, help="stride over the training files (default 4)")
    ap.add_argument("--serving", action="store_true", help="embed through a serving session, a query per window")
    ap.add_argument("--session", help="with --serving: a session already running the embeddings blueprint")
    ap.add_argument("--model", default=MODEL, help="the `model` a query names")
    ap.add_argument("--parallel", type=int, default=3, help="batch sessions at a time (the free trial runs 3 jobs)")
    ap.add_argument("--view", action="store_true", help="only rebuild the map from <out>/library_embeddings/")
    ap.add_argument("--view-dir", help=f"where {VIEW} goes (default <out>)")
    a = ap.parse_args(argv)
    m = load_manifest(a.roles)
    a.timestamp_column = m["timestamp_column"]
    t0 = time.time()
    if a.view:
        meta = json.load(open(os.path.join(a.out, "library_embeddings.json")))
        return write_view(a, m, {k: meta[k] for k in ("blueprint", "route", "window_size", "step_size", "trial_step")
                                 if k in meta}, t0)
    load_dotenv()
    check_auth(log)
    ts_path = os.path.join(a.out, "test_state.json")
    if not os.path.exists(ts_path):
        sys.exit(f"no {ts_path}: run promote_and_test.py first")
    test = json.load(open(ts_path))
    if not same_deployment(test):
        sys.exit(f"the blueprint in {ts_path} is on {test['endpoint']}, a different deployment from {root()}")
    key, values = test["blueprint"]["key"], test["blueprint"]["values"]
    meta = {"blueprint": key, "route": "serving" if a.serving else "batch", "window_size": values["window_size"],
            "step_size": a.step, "trial_step": values.get("step_size", 1)}
    summary_path = os.path.join(a.out, "library_embeddings.json")
    made = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    if made and (made.get("blueprint"), made.get("step_size")) != (key, a.step):
        sys.exit(f"{summary_path} and {a.out}/library_embeddings/ are for {made.get('blueprint')} at step "
                 f"{made.get('step_size')}, not {key} at step {a.step}: move them aside so the map holds "
                 f"one classifier's windows")
    sp = os.path.join(a.out, "library_state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    if not same_deployment(state):
        sys.exit(f"{sp} is from {state['endpoint']}, a different deployment from {root()}: move it aside")
    if state.get("blueprint", key) != key or state.get("step", a.step) != a.step:
        sys.exit(f"{sp} and {a.out}/library_embeddings/ are for {state.get('blueprint')} at step {state.get('step')}, "
                 f"not {key} at step {a.step}: move them aside (cleanup.py deletes the sessions) so the map holds "
                 f"one classifier's windows")
    if state.get("bundle"):
        sys.exit(f"{sp} is from a bundle run: move it and {a.out}/library_embeddings/ aside")
    state.update({"endpoint": root(), "blueprint": key, "step": a.step})

    def save():
        json.dump(state, open(sp, "w"), indent=1)
    save()
    os.makedirs(os.path.join(a.out, "library_embeddings"), exist_ok=True)
    pending = [f for f in m["training"] if not os.path.exists(out_path(a.out, f))]
    if a.serving and not pending:
        written = 0
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
            written = run_serving(a, m["training"], session, meta["window_size"], a.step, m["channels"])
        finally:
            if ex is None and os.path.exists(ssp):      # failed while starting: delete what it made
                ex = json.load(open(ssp))
            if ex and end(ex):
                os.remove(ssp)
                log(f"deleted session {ex['session']}, deployment {ex['deployment']} and package {ex['package']}")
    elif not pending:
        complete, written = True, 0
    else:
        emb = embeddings_blueprint(a, m, test, state, save)
        complete, written = run_batch(a, m["training"], emb, a.step, state, save)
        if not complete:
            sys.exit("not every training file has a complete output: rerun to resume")
    if not written and made.get("route"):         # nothing new: keep how the outputs were made
        meta["route"] = made["route"]
    write_view(a, m, meta, t0)
    if state.get("package"):
        log(f"next: python cleanup.py --out {a.out} --yes deletes the batch sessions, deployment and package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
