#!/usr/bin/env python3
"""Path 2, step 6: the library's embeddings, and a 2-D view of them a live app can draw.

    python library_embeddings.py --roles roles --out out              # one batch run per library file, then the view
    python library_embeddings.py --roles roles --out out --serving    # a serving agent, one query per window
    python library_embeddings.py --roles roles --out out --serving --agent agt_...   # one already running
    python library_embeddings.py --roles roles --out out --view       # the view again from the saved outputs, no platform calls
    python library_embeddings.py --roles roles --out out --view-dir ../my-app/static

The promoted model is a kNN over the library's windows, but it is a registry artifact with
no download route, so its reference windows can't be read back. They can be recomputed: a
bundle on the promoted blueprint with `output_embeddings: true` runs each library file (one
per state) through the same encoder, on the same windows (its window and step), and returns
each window's Omega embedding, 768 values per channel.

--serving gets the same embeddings from a serving agent instead, one window per /query, cut
here as the platform cuts them: from each file's row 0 at the blueprint's step, leaving out
the windows that cross a join between the file's pieces (training skips those). One agent for
the whole library: no job slots, queue or 2-hour cap on the free trial. With --agent, an
agent of the promoted blueprint already running with output_embeddings (a live app's) is used
and left running; otherwise one is started and deleted at the end (ids in
<out>/library_serving_state.json until then). Measured on the AD skill's SKAB library
(2026-10-31): the same 1,950 windows as batch, embeddings within 4.0e-6.

The view is the two directions that best separate the states: each embedding scaled to unit
length (the model compares by cosine), PCA to 50 dimensions, then linear discriminant analysis
to 2, each axis scaled to unit spread. The two largest directions of spread (PCA 1-2) are
about the data's variance, not its states, and mixed two of LARCO's four. The view holds the
points with their state, the mean and the two projection rows, so a served window's embedding
lands in the same coordinates: ((e / |e|) - mean) . row. It is a drawing, not the model: the
kNN decides in every dimension.

Writes <out>/library_embeddings/<state>.csv (the batch outputs, large), <out>/library_embeddings.json
(the summary) and <view-dir>/library-view.{json,bin} (default <out>). Ids go to
<out>/library_state.json, so a rerun resumes and cleanup.py deletes them. A rerun for another
blueprint stops rather than mixing two models' windows: it checks library_state.json and,
because cleanup.py deletes that, library_embeddings.json too. A rerun with nothing new to
embed starts no agent or run, and keeps the route the outputs were made by.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import TERMINAL, agents, check_auth, load_dotenv, request, root, same_deployment, upload_all  # noqa: E402
from common import cache_path, load_manifest, log  # noqa: E402
from deliver import END_SLACK_MS, OUTPUT_WAIT_S, fetch, time_range, to_ms  # noqa: E402
from serve import MODEL, end, query_body, stop  # noqa: E402

WORKERS = 4


def firsts(rows, window, step, pieces=None):
    """First row of each window the platform cuts from a library file: from row 0 every
    `step` rows, leaving out windows that cross a join between pieces."""
    bounds = [(p["row"], p["row"] + p["rows"]) for p in pieces] if pieces else [(0, rows)]
    return [i for i in range(0, rows - window + 1, step) if any(a <= i and i + window <= b for a, b in bounds)]


def start_serving(args, key, channels, sp):
    """A serving agent of the promoted blueprint with embeddings on, once running. Its ids are
    in `sp` from the moment each exists, so a kill leaves them for cleanup.py."""
    ex = {"endpoint": root(), "agent": None, "bundle": None}
    ex["bundle"] = request("POST", f"{agents()}/bundles", {
        "blueprint": key, "name": f"{args.name} library serving",
        "values": {"data_columns": channels, "timestamp_column": "timestamp", "output_embeddings": True}})["id"]
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


def run_serving(args, files, agent, window, step, channels):
    """Each library file's windows, one /query each, saved in the batch output's shape."""
    out_dir = os.path.join(args.out, "library_embeddings")
    os.makedirs(out_dir, exist_ok=True)

    def query(chunk):
        rec = request("POST", f"{agents()}/instances/{agent}/query", query_body(chunk, args.model),
                      idempotent=True)["response"]["data"][0]
        if missing := [c for c in channels if f"embedding_{c}" not in rec]:
            sys.exit(f"the agent's answer has no embedding_{missing[0]}: its bundle needs output_embeddings: true")
        return rec
    t0, n = time.time(), 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for f in files:
            dst = os.path.join(out_dir, f"{f['state']}.csv")
            if os.path.exists(dst):
                continue
            with open(os.path.join(args.roles, f["file"]), newline="") as fh:
                rows = list(csv.DictReader(fh))
            X = np.array([[float(r[c]) for c in channels] for r in rows], dtype=np.float32)
            first = firsts(len(rows), window, step, f.get("pieces"))
            recs = list(pool.map(query, [X[i:i + window] for i in first]))
            with open(dst + ".part", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["start_timestamp", "finish_timestamp", "invalid", *[f"embedding_{c}" for c in channels]])
                for i, r in zip(first, recs):
                    w.writerow([rows[i]["timestamp"], rows[i + window - 1]["timestamp"], str(r.get("invalid", False)).lower(),
                                *[json.dumps(r[f"embedding_{c}"]) for c in channels]])
            os.replace(dst + ".part", dst)
            n += len(first)
            log(f"  {f['state']}: {len(first):,} windows")
    log(f"served {n:,} windows in {time.time() - t0:.0f} s")


def outputs(agent):
    """The run's output rows, header first, from every result file it wrote."""
    rows, header = [], None
    for item in request("GET", f"{agents()}/instances/{agent}/results").get("data") or []:
        body = list(csv.reader(fetch(item["data"]["filename"]).splitlines()))
        if body:
            header = header or body[0]
            rows += body[1:]
    return header, rows


def run_all(args, files, state, save, ensure_bundle):
    """One batch run per library file; each output saved once it reaches its file's end."""
    out_dir = os.path.join(args.out, "library_embeddings")
    done = lambda f: os.path.exists(os.path.join(out_dir, f"{f['state']}.csv"))  # noqa: E731
    if all(done(f) for f in files):
        return
    ensure_bundle()
    paths = {f["state"]: os.path.join(args.roles, f["file"]) for f in files}
    ids = upload_all(list(paths.values()), args.roles, cache_path(args.out), prefix=args.name, log=log)
    for f in files:
        if not done(f) and f["state"] not in state["runs"]:
            run = request("POST", f"{agents()}/bundles/{state['bundle']}/run",
                          {"connectors": {"source": [{"type": "file", "id": ids[paths[f['state']]]}]}})
            state["runs"][f["state"]] = run["id"]
            save()
            log(f"  {f['state']}: run {run['id']}")
    os.makedirs(out_dir, exist_ok=True)
    pending = {f["state"] for f in files if not done(f)}
    deadline = {}
    while pending:
        for s in sorted(pending):
            aid = state["runs"][s]
            if s not in deadline:
                status = request("GET", f"{agents()}/instances/{aid}")["status"]
                if status not in TERMINAL:
                    continue
                if status != "completed":
                    state["runs"].pop(s)
                    save()
                    sys.exit(f"{s}: run {aid} {status}; rerun to start it again")
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
                sys.exit(f"{s}: run {aid} completed {OUTPUT_WAIT_S // 60} min ago but its output stops early; "
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
                sys.exit(f"{path}: no embedding_{missing[0]} column: was the bundle made with output_embeddings?")
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
            "note": (f"The promoted model's library, recomputed {meta.get('route', 'in batch')} with output_embeddings on, on the 2 "
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
    ap.add_argument("--name", default="osm", help="prefixes the uploads and names the bundle")
    ap.add_argument("--serving", action="store_true", help="embed through a serving agent, a query per window")
    ap.add_argument("--agent", help="with --serving: an agent already running the promoted blueprint with embeddings")
    ap.add_argument("--model", default=MODEL, help="the `model` a query names")
    ap.add_argument("--view", action="store_true", help="only rebuild the view from <out>/library_embeddings/")
    ap.add_argument("--view-dir", help="where library-view.{json,bin} go (default <out>)")
    args = ap.parse_args(argv)
    manifest = load_manifest(args.roles)
    t0 = time.time()
    if args.view:
        meta = json.load(open(os.path.join(args.out, "library_embeddings.json")))
        return write_view(args, manifest, {k: meta[k] for k in ("blueprint", "window_size", "step_size", "route")
                                           if k in meta}, t0)
    load_dotenv()
    check_auth(log)
    ts_path = os.path.join(args.out, "test_state.json")
    if not os.path.exists(ts_path):
        sys.exit(f"no {ts_path}: run promote_and_test.py first")
    key = json.load(open(ts_path))["blueprint"]["key"]
    values = request("GET", f"{agents()}/blueprints/{key}")["document"]["values"]
    meta = {"blueprint": key, "window_size": values["window_size"], "step_size": values["step_size"],
            "route": "through a serving agent" if args.serving else "in batch"}
    # cleanup.py deletes library_state.json but leaves the outputs and their summary
    summary = os.path.join(args.out, "library_embeddings.json")
    made = json.load(open(summary)) if os.path.exists(summary) else {}
    sp = os.path.join(args.out, "library_state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    for what, bp, step in ((summary, made.get("blueprint"), made.get("step_size")),
                           (sp, state.get("blueprint"), meta["step_size"])):
        if bp and (bp, step) != (key, meta["step_size"]):
            sys.exit(f"{what} and {args.out}/library_embeddings/ are for {bp} at step {step}, not the promoted "
                     f"{key} at step {meta['step_size']}: move them aside (cleanup.py deletes the runs) so the "
                     f"view holds one model's windows")
    out_dir = os.path.join(args.out, "library_embeddings")
    if all(os.path.exists(os.path.join(out_dir, f"{f['state']}.csv")) for f in manifest["library"]["files"]):
        log("every library file has its embeddings: no agent or run started")
        meta["route"] = made.get("route", meta["route"])         # keep how they were made
        return write_view(args, manifest, meta, t0)

    if args.serving:
        ssp = os.path.join(args.out, "library_serving_state.json")
        if os.path.exists(ssp):
            sys.exit(f"{ssp} exists: an earlier run's serving agent may still be up; run cleanup.py --yes first")
        if args.agent and (b := request("GET", f"{agents()}/instances/{args.agent}").get("blueprint")) != key:
            sys.exit(f"agent {args.agent} runs {b}, not the promoted {key}")
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        ex = None
        try:
            agent = args.agent or (ex := start_serving(args, key, manifest["channels"], ssp))["agent"]
            run_serving(args, manifest["library"]["files"], agent, meta["window_size"],
                        meta["step_size"], manifest["channels"])
        finally:
            if ex is None and os.path.exists(ssp):      # failed while starting: delete what it made
                ex = json.load(open(ssp))
            if ex and end(ex):
                os.remove(ssp)
                log(f"deleted serving agent {ex['agent']} and bundle {ex['bundle']}")
        return write_view(args, manifest, meta, t0)

    if not same_deployment(state):
        sys.exit(f"{sp} is from {state['endpoint']}, a different deployment from {root()}: "
                 f"move it aside, or set .env to that deployment")
    save = lambda: json.dump(state, open(sp, "w"), indent=1)  # noqa: E731

    def ensure_bundle():
        if not state.get("bundle"):
            b = request("POST", f"{agents()}/bundles", {
                "blueprint": key, "name": f"{args.name} library embeddings", "values": {"output_embeddings": True}})
            state.update({"endpoint": root(), "bundle": b["id"], "blueprint": key, "runs": {}})
            save()
            log(f"bundle {b['id']} on {key} (window {meta['window_size']}, step {meta['step_size']}, embeddings on)")
    run_all(args, manifest["library"]["files"], state, save, ensure_bundle)
    write_view(args, manifest, meta, t0)
    log(f"next: python cleanup.py --out {args.out} --yes deletes the runs and their bundle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
