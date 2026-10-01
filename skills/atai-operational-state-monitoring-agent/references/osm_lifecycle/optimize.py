#!/usr/bin/env python3
"""Path 2, step 1: fit and score OSM settings on your role files with the Optimizations API.

Every library file (roles/library/<state>__library.csv) becomes a training example
labelled by its state; each search-validation file becomes a validation example scored
on its `label` column, each window paired with its last row. The platform embeds every
window with Omega, fits a kNN on the library, and reports macro-F1 per trial.

The search space is the product of the values given:
  values  (kind "value"):   --windows, --steps           how the data is windowed
  fitting (kind "fitting"): --k, --metrics, --weights    the kNN
With --max-trials below the product's size the platform samples: a random search.
By default max_trials = the size, so every point runs once. Trials run one after
another. Step > window skips records and needs --allow-gaps.

Writes <out>/optimize_<optimization id>.json (the optimization, every trial, a summary).

    python optimize.py --roles roles --dry-run                   # the plan: no uploads, no jobs
    python optimize.py --roles roles --k 5 31 --background       # 2 trials, detached
    python optimize.py --resume opt_...                          # collect a run whose poller died
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import agents, check_auth, list_trials, load_dotenv, report_f1, request, trial_setting, upload_all, wait  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from common import cache_path, example, load_manifest, log, role_paths, stamp  # noqa: E402


def search_space(windows, steps, k, metrics, weights):
    one = lambda vals: {"type": "categorical", "values": list(vals)}  # noqa: E731
    return {"parameters": {
        "window_size": {"kind": "value", "spec": one(windows)},
        "step_size": {"kind": "value", "spec": one(steps)},
        "k_neighbors": {"kind": "fitting", "spec": one(k)},
        "metric": {"kind": "fitting", "spec": one(metrics)},
        "weights": {"kind": "fitting", "spec": one(weights)},
    }}


def grid(windows, steps, k, metrics, weights):
    return list(itertools.product(windows, steps, k, metrics, weights))


def gap_pairs(points):
    """(window, step) pairs with step > window: records between windows are skipped."""
    return sorted({(w, st) for w, st, *_ in points if st > w})


def summarize(trials, states):
    rows = []
    for t in sorted(trials, key=lambda t: t["trial_number"]):
        w, st, k, metric, weights = trial_setting(t)
        f1, n = report_f1(t.get("metrics_report"))
        rows.append({"trial": t["trial_number"], "id": t["id"], "status": t["status"], "window": w, "step": st,
                     "k": k, "metric": metric, "weights": weights, "macro_f1": t.get("objective_value"),
                     "f1": f1, "windows_scored": n, "error": t.get("error")})
    for r in sorted(rows, key=lambda r: -(r["macro_f1"] if r["macro_f1"] is not None else -1)):
        m = "   -  " if r["macro_f1"] is None else f"{r['macro_f1']:.4f}"
        print(f"  #{r['trial']:<3} w={r['window']:<5} step={r['step']:<5} k={r['k']:<3} {r['metric']:<6} {r['weights']:<8} "
              f"{r['status']:<9} macro-F1 {m}  " + "  ".join(f"{s} {r['f1'].get(s, 0):.2f}" for s in states)
              + f"  windows {r['windows_scored']:,}")
        if r["error"]:
            print(f"        error: {r['error']}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles", help="the role-file folder (with manifest.json)")
    ap.add_argument("--out", default="out", help="where results, the upload cache and logs go")
    ap.add_argument("--name", default="osm", help="a short name for platform objects")
    ap.add_argument("--windows", type=int, nargs="+", default=[512])
    ap.add_argument("--steps", type=int, nargs="+", default=[512])
    ap.add_argument("--k", type=int, nargs="+", default=[5, 31])
    ap.add_argument("--metrics", nargs="+", default=["cosine"])
    ap.add_argument("--weights", nargs="+", default=["uniform"])
    ap.add_argument("--max-trials", type=int)
    ap.add_argument("--allow-gaps", action="store_true", help="allow step > window (skips the records between windows)")
    ap.add_argument("--blueprint", default="osm", help="blueprint key or blp_ id to optimize (default: the canonical osm)")
    ap.add_argument("--upload-jobs", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", metavar="OPT_ID")
    add_background_flag(ap, default_log="{out}/optimize.log")
    args = ap.parse_args()
    maybe_detach(args)

    manifest = load_manifest(args.roles)
    states = manifest["states"]
    search = set(manifest.get("search_validation") or [f["recording"] for f in manifest["validation"]["files"]])
    library = role_paths(args.roles, manifest, "library")
    validation = role_paths(args.roles, manifest, "validation", search)
    points = grid(args.windows, args.steps, args.k, args.metrics, args.weights)
    gaps = gap_pairs(points)
    if gaps and not args.allow_gaps:
        sys.exit(f"step > window would skip records for {gaps}: use step <= window, or --allow-gaps to do it deliberately")
    space = search_space(args.windows, args.steps, args.k, args.metrics, args.weights)
    n = min(args.max_trials or len(points), len(points))
    log(f"{len(library)} library files ({', '.join(states)}), {len(validation)} validation files; "
        f"{n} trial(s) of a {len(points)}-point space ({'random search' if n < len(points) else 'every point'})")
    if args.dry_run:
        print(json.dumps(space, indent=1))
        return

    load_dotenv()
    check_auth(log)
    if args.resume:
        opt_id = args.resume
    else:
        ids = upload_all(library + validation, args.roles, cache_path(args.out), prefix=args.name, jobs=args.upload_jobs, log=log)
        blueprint_id = request("GET", f"{agents()}/blueprints/{args.blueprint}")["id"]
        body = {
            "name": f"{args.name} optimize {stamp()}", "blueprint_id": blueprint_id, "objective": "macro_f1",
            "search_space": space, "budget": {"max_trials": args.max_trials or len(points)},
            "training_examples": [example(os.path.basename(p)[:-4], ids[p], os.path.basename(p).split("__")[0])
                                  for p in library],
            "validation_examples": [example(os.path.basename(p)[:-4], ids[p]) for p in validation],
        }
        opt_id = request("POST", f"{agents()}/optimizations", body=body)["id"]
        log(f"optimization {opt_id} created (collect later with --resume {opt_id})")

    opt = wait(f"{agents()}/optimizations/{opt_id}", opt_id, every_s=60, log=log,
               progress=lambda o: o.get("progress", {}))
    trials = list_trials(opt_id)
    print(f"\noptimization {opt_id}: {opt['status']}")
    if opt.get("error"):
        print(f"  error from the platform: {opt['error']}")
    rows = summarize(trials, states)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"optimize_{opt_id}.json")
    with open(path, "w") as f:
        json.dump({"optimization": opt, "trials": trials, "summary": rows, "search_space": space,
                   "validation": [os.path.relpath(p, args.roles) for p in validation]}, f, indent=1)
    log(f"wrote {path}; next: python promote_and_test.py --roles {args.roles} --out {args.out} --name {args.name}")


if __name__ == "__main__":
    main()
