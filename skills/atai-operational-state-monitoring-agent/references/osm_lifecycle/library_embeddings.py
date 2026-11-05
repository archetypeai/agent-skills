#!/usr/bin/env python3
"""Path 2, step 6: the library's embeddings, and a 2-D view of them a live app can draw.

    python library_embeddings.py --roles roles --out out              # one batch run per library file, then the view
    python library_embeddings.py --roles roles --out out --view       # the view again from the saved outputs, no platform calls
    python library_embeddings.py --roles roles --out out --view-dir ../my-app/static

The promoted model is a kNN over the library's windows, but it is a registry artifact with
no download route, so its reference windows can't be read back. They can be recomputed by
running each library file (one per state) through the same encoder, on the same windows
(its window and step), with `output_embeddings: true`, which returns each window's Omega
embedding, 768 values per channel.

A package runs a blueprint exactly as promoted and can't turn an output on, so this first
makes a blueprint that has it on: a 1-trial optimization at the promoted trial's own point
(window, step, k, metric, weights), on the same library and search-validation files, with
`output_embeddings: true` pinned, then promoted. Its kNN is refitted, but the embeddings
depend only on the encoder and the windowing, so they are the promoted model's. Then a
package of it, one batch deployment, and a session per library file.

The view is the two directions that best separate the states: each embedding scaled to unit
length (the model compares by cosine), PCA to 50 dimensions, then linear discriminant analysis
to 2, each axis scaled to unit spread. The two largest directions of spread (PCA 1-2) are
about the data's variance, not its states, and mixed two of LARCO's four. The view holds the
points with their state, the mean and the two projection rows, so a served window's embedding
lands in the same coordinates: ((e / |e|) - mean) . row. It is a drawing, not the model: the
kNN decides in every dimension.

Writes <out>/library_embeddings/<state>.csv (the batch outputs, large), <out>/library_embeddings.json
(the summary) and <view-dir>/library-view.{json,bin} (default <out>). Ids go to
<out>/library_state.json, so a rerun resumes and cleanup.py deletes the sessions, deployment
and package (the embeddings blueprint and its optimization stay).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import (TERMINAL, agents, check_auth, create_package, deploy, deployment_ready, lifecycle_url,  # noqa: E402
                       list_trials, load_dotenv, open_session, package_ready, request, root, same_deployment,
                       trial_setting, upload_all, wait)
from common import cache_path, example, load_manifest, log, role_paths, stamp  # noqa: E402
from deliver import END_SLACK_MS, OUTPUT_WAIT_S, fetch, time_range, to_ms  # noqa: E402
from optimize import pinned_values, search_space  # noqa: E402
from promote_and_test import promote  # noqa: E402


def embeddings_blueprint(args, manifest, test, state, save):
    """{id, key} of a blueprint at the promoted trial's point with output_embeddings on: a
    1-trial optimization in the project, then promoted. Resumable through `state`."""
    if state.get("embeddings_blueprint"):
        return state["embeddings_blueprint"]
    if not state.get("embeddings_optimization"):
        opt, tid = test["model"]["optimization"], test["model"]["trial"]
        w, st, k, metric, weights = trial_setting(next(t for t in list_trials(opt) if t["id"] == tid))
        base = json.load(open(os.path.join(args.out, "project_state.json")))["base"]
        bp = request("GET", f"{agents()}/blueprints/{base['id']}")
        search = set(manifest.get("search_validation") or [f["recording"] for f in manifest["validation"]["files"]])
        library = role_paths(args.roles, manifest, "library")
        validation = role_paths(args.roles, manifest, "validation", search)
        ids = upload_all(library + validation, args.roles, cache_path(args.out), prefix=args.name, log=log)
        training = [example(os.path.basename(p)[:-4], ids[p], os.path.basename(p).split("__")[0]) for p in library]
        o = request("POST", f"{agents()}/optimizations", body={
            "name": f"{args.name} embeddings {stamp()}", "blueprint_id": bp["id"], "objective": "macro_f1",
            "search_space": search_space([w], [st], [k], [metric], [weights]), "budget": {"max_trials": 1},
            "training_examples": training,
            "validation_examples": [example(os.path.basename(p)[:-4], ids[p]) for p in validation],
            "overrides": pinned_values(bp, training, manifest, {"output_embeddings": True})})
        state.update({"endpoint": root(), "embeddings_optimization": o["id"]})
        save()
        log(f"optimization {o['id']}: 1 trial at {tid}'s point (w {w}, step {st}, k {k}, {metric}, {weights}), embeddings on")
    opt = wait(f"{agents()}/optimizations/{state['embeddings_optimization']}", state["embeddings_optimization"],
               every_s=30, log=log)
    trial = next((t for t in list_trials(opt["id"]) if t["status"] == "completed"), None)
    if opt["status"] != "completed" or not trial:
        sys.exit(f"{opt['id']} is {opt['status']} with no completed trial: move {args.out}/library_state.json aside to try again")
    bp = promote(opt["id"], trial, f"{args.name}-embeddings")
    state["embeddings_blueprint"] = {"id": bp["id"], "key": bp["blueprint_key"]}
    save()
    return state["embeddings_blueprint"]


def outputs(session):
    """The session's output rows, header first, from every result file it wrote."""
    rows, header = [], None
    for item in request("GET", lifecycle_url("sessions", session) + "/results").get("data") or []:
        body = list(csv.reader(fetch(item["data"]["ref"]).splitlines()))
        if body:
            header = header or body[0]
            rows += body[1:]
    return header, rows


def run_all(args, files, state, save, ensure_deployment):
    """One batch session per library file; each output saved once it reaches its file's end."""
    out_dir = os.path.join(args.out, "library_embeddings")
    done = lambda f: os.path.exists(os.path.join(out_dir, f"{f['state']}.csv"))  # noqa: E731
    if all(done(f) for f in files):
        return
    ensure_deployment()
    paths = {f["state"]: os.path.join(args.roles, f["file"]) for f in files}
    ids = upload_all(list(paths.values()), args.roles, cache_path(args.out), prefix=args.name, log=log)
    for f in files:
        if not done(f) and f["state"] not in state["runs"]:
            ses = open_session(state["deployment"], [ids[paths[f["state"]]]])
            state["runs"][f["state"]] = ses["id"]
            save()
            log(f"  {f['state']}: session {ses['id']}")
    os.makedirs(out_dir, exist_ok=True)
    pending = {f["state"] for f in files if not done(f)}
    deadline = {}
    while pending:
        for s in sorted(pending):
            aid = state["runs"][s]
            if s not in deadline:
                status = request("GET", lifecycle_url("sessions", aid))["status"]
                if status not in TERMINAL:
                    continue
                if status != "completed":
                    state["runs"].pop(s)
                    save()
                    sys.exit(f"{s}: session {aid} {status}; rerun to start it again")
                deadline[s] = time.time() + OUTPUT_WAIT_S
            header, rows = outputs(aid)
            last = max((to_ms(r[header.index("finish_timestamp")]) for r in rows), default=0) if header else 0
            # a run can say completed before its output is fully written (deliver.py)
            if time_range(paths[s])[1] - last <= END_SLACK_MS:
                with open(os.path.join(out_dir, f"{s}.csv"), "w", newline="") as fh:
                    csv.writer(fh).writerows([header] + rows)
                log(f"  {s}: {len(rows):,} windows")
                pending.discard(s)
            elif time.time() > deadline[s]:
                state["runs"].pop(s)
                save()
                sys.exit(f"{s}: session {aid} completed {OUTPUT_WAIT_S // 60} min ago but its output stops early; "
                         f"rerun to start it again")
        if pending:
            time.sleep(15 if pending <= set(deadline) else 30)


def read_embeddings(path, channels):
    """[windows, channels x 768] in the channels' order, invalid windows left out."""
    csv.field_size_limit(1 << 30)
    vecs = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("invalid")).lower() == "true":
                continue
            missing = [c for c in channels if f"embedding_{c}" not in r]
            if missing:
                sys.exit(f"{path}: no embedding_{missing[0]} column: was its blueprint promoted with output_embeddings?")
            vecs.append(np.concatenate([np.asarray(json.loads(r[f"embedding_{c}"]), dtype=np.float32) for c in channels]))
    return np.stack(vecs)


def project(X, codes, k=50, shrink=1e-3):
    """The 2 discriminant axes of the unit-length embeddings, through PCA to k dimensions:
    mean [dim], rows [2, dim] (each axis at unit spread), ref [n, 2], and the share of
    windows nearest their own state's centre in 2-D."""
    codes = np.asarray(codes)
    states = np.unique(codes)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    mean = Xn.mean(axis=0)
    Xc = Xn - mean
    # PCA through the n x n Gram matrix: the windows are fewer than the dimensions
    eigval, eigvec = np.linalg.eigh(Xc @ Xc.T)
    top = np.argsort(eigval)[::-1][:min(k, len(X) - 1)]
    k = len(top)
    P = (Xc.T @ eigvec[:, top]) / np.sqrt(eigval[top])           # unit principal axes, [dim, k]
    Z = Xc @ P
    m = np.array([Z[codes == i].mean(axis=0) for i in states])
    Sw = sum((Z[codes == i] - m[j]).T @ (Z[codes == i] - m[j]) for j, i in enumerate(states))
    Sw += shrink * np.trace(Sw) / k * np.eye(k)
    Sb = sum((codes == i).sum() * np.outer(m[j] - Z.mean(0), m[j] - Z.mean(0)) for j, i in enumerate(states))
    w, V = np.linalg.eig(np.linalg.solve(Sw, Sb))
    A = P @ V[:, np.argsort(-w.real)[:2]].real                    # [dim, 2]
    A /= (Xc @ A).std(axis=0)
    ref = Xc @ A
    if not (np.isfinite(A).all() and np.isfinite(ref).all()):
        sys.exit("the projection is not finite: check the embeddings for NaN or all-zero windows")
    centroids = np.array([ref[codes == i].mean(axis=0) for i in states])
    nearest = states[((ref[:, None] - centroids[None]) ** 2).sum(-1).argmin(1)]
    return mean, A.T, ref, float((nearest == codes).mean())


def write_view(args, manifest, meta, t0):
    """The view from the saved outputs in <out>/library_embeddings/."""
    states, channels = manifest["states"], manifest["channels"]
    vecs, codes = [], []
    for f in manifest["library"]["files"]:
        v = read_embeddings(os.path.join(args.out, "library_embeddings", f"{f['state']}.csv"), channels)
        vecs.append(v)
        codes += [states.index(f["state"])] * len(v)
    X = np.concatenate(vecs).astype(np.float64)
    n, dim = X.shape
    log(f"{n:,} windows x {dim:,} values")
    mean, rows, ref, separated = project(X, codes)

    view_dir = args.view_dir or args.out
    os.makedirs(view_dir, exist_ok=True)
    pieces = {"ref": ref, "state": np.array(codes, dtype=np.float32), "mean": mean, "pc0": rows[0], "pc1": rows[1]}
    layout, offset = {}, 0
    with open(os.path.join(view_dir, "library-view.bin"), "wb") as fh:
        for k, a in pieces.items():
            a = np.ascontiguousarray(a, dtype="<f4")
            fh.write(a.tobytes())
            layout[k] = {"offset": offset, "shape": list(a.shape)}
            offset += a.nbytes
    per_state = {s: codes.count(i) for i, s in enumerate(states)}
    with open(os.path.join(view_dir, "library-view.json"), "w") as fh:
        json.dump({
            "note": ("The promoted model's library, recomputed in batch with output_embeddings on, on the 2 "
                     "directions that best separate the states: unit-length embeddings, PCA to 50, LDA to 2. "
                     "For drawing only."),
            "blueprint": meta["blueprint"], "channels": channels, "states": states,
            "embeddingDim": dim // len(channels), "totalDim": dim, "nReference": n, "perState": per_state,
            "windowSize": meta["window_size"], "stepSize": meta["step_size"],
            "normalize": True, "projection": "lda", "axes": ["LD 1", "LD 2"], "separated": round(separated, 3),
            "layout": layout, "dtype": "float32-le"}, fh, indent=1)
    summary = {**meta, "windows": n, "per_state": per_state, "dim": dim, "projection": "lda",
               "separated_2d": round(separated, 3), "view_bytes": offset, "seconds": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(args.out, "library_embeddings.json"), "w"), indent=1)
    print(f"\nlibrary: {n:,} windows ({', '.join(f'{s} {c}' for s, c in per_state.items())}), "
          f"window {meta['window_size']}, step {meta['step_size']}")
    print(f"  2 discriminant axes: {separated:.0%} of the windows nearest their own state's centre")
    print(f"  -> {view_dir}/library-view.bin ({offset / 1e3:.0f} KB), library-view.json")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="osm", help="prefixes the uploads and names the platform objects")
    ap.add_argument("--view", action="store_true", help="only rebuild the view from <out>/library_embeddings/")
    ap.add_argument("--view-dir", help="where library-view.{json,bin} go (default <out>)")
    args = ap.parse_args(argv)
    manifest = load_manifest(args.roles)
    t0 = time.time()
    if args.view:
        meta = json.load(open(os.path.join(args.out, "library_embeddings.json")))
        return write_view(args, manifest, {k: meta[k] for k in ("blueprint", "window_size", "step_size")}, t0)
    load_dotenv()
    check_auth(log)
    ts_path = os.path.join(args.out, "test_state.json")
    if not os.path.exists(ts_path):
        sys.exit(f"no {ts_path}: run promote_and_test.py first")
    test = json.load(open(ts_path))
    key = test["blueprint"]["key"]
    # by id: a key registered in a project resolves only with ?project_id=
    values = request("GET", f"{agents()}/blueprints/{test['blueprint']['id']}")["document"]["values"]
    meta = {"blueprint": key, "window_size": values["window_size"], "step_size": values["step_size"]}

    sp = os.path.join(args.out, "library_state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    if not same_deployment(state):
        sys.exit(f"{sp} is from {state['endpoint']}, a different deployment from {root()}: "
                 f"move it aside, or set .env to that deployment")
    save = lambda: json.dump(state, open(sp, "w"), indent=1)  # noqa: E731

    if state.get("bundle"):
        sys.exit(f"{sp} is from a bundle run: move it and {args.out}/library_embeddings/ aside")

    def ensure_deployment():
        if not state.get("deployment"):
            emb = embeddings_blueprint(args, manifest, test, state, save)
            if not state.get("package"):
                state.update({"endpoint": root(), "package": create_package(emb["id"], f"{args.name} library embeddings")["id"],
                              "blueprint": key, "runs": {}})
                save()
            package_ready(state["package"])
            state["deployment"] = deploy(state["package"], "batch")["id"]
            save()
            deployment_ready(state["deployment"])
            log(f"package {state['package']} of {emb['key']} (window {meta['window_size']}, step {meta['step_size']}, "
                f"embeddings on), batch deployment {state['deployment']}")
    run_all(args, manifest["library"]["files"], state, save, ensure_deployment)
    write_view(args, manifest, meta, t0)
    log(f"next: python cleanup.py --out {args.out} --yes deletes the sessions, deployment and package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
