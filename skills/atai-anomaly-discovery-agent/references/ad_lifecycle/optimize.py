#!/usr/bin/env python3
"""Path 2, step 1: fit and score AD settings on your role files with the Optimizations API.

Every library file becomes a training example, every row labelled `normal` by a
constant; every validation file a validation example scored on its `label` column, each
window paired with its last row. Per trial the platform embeds every window with Omega,
fuses per-channel features, fits normalisation, PCA and a Local Outlier Factor detector
on the library, and reports macro-F1 on validation at the trial's threshold.

Each option takes one value (pinned in `overrides.values` for every trial) or several
(searched, `kind: value`):
  --windows, --steps   window_size, step_size: both Optimize legs use the one step
  --k                  n_neighbors (always in the search space: it can't be empty)
  --metrics            l1, l2, cosine
  --pca                pca_components (PCA is always on in `ad`)
  --threshold          one value, or LOW:HIGH for a float range
  --features           feature_names, a list: pinned only; search another set with a
                       second optimization
The platform draws distinct points at random: a budget at or above the grid's size runs
every point once (a grid), below it, or with a threshold range, a random search. By
default a grid of up to DEFAULT_TRIALS points runs in full; a bigger grid or a threshold
range gets a DEFAULT_TRIALS-trial random search: with the few settings that matter, a
dozen random points usually land close to the best. --max-trials sets the budget.

Writes <out>/optimize_<optimization id>.json and <out>/optimize_state.json.

    python optimize.py --roles roles --dry-run                     # the plan: no uploads, no jobs
    python optimize.py --roles roles --out out --name ad-sample    # the default 2-trial search
    python optimize.py --resume opt_...                            # collect a run whose poller died
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import agents, check_auth, list_trials, load_dotenv, report_anomaly, request, trial_values, upload_all, wait  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from common import base_values, cache_path, example, load_manifest, log, role_paths, stamp  # noqa: E402

# The TEP quickstart's search: the full example's 24-trial winner, with k 5 and 10.
DEFAULTS = {"windows": [48], "steps": [8], "k": [5, 10], "metrics": ["l2"], "pca": [16],
            "threshold": "1.215", "features": ["std", "range", "slope", "median"]}


DEFAULT_TRIALS = 12


def budget(size, max_trials=None):
    """Trials to run: --max-trials if given (never more than a grid), else the whole grid
    up to DEFAULT_TRIALS points, else (a bigger grid, or a float range: size None) a
    DEFAULT_TRIALS-trial random search."""
    if size is None:
        return max_trials or DEFAULT_TRIALS
    return min(max_trials or min(size, DEFAULT_TRIALS), size)


def build(a):
    """(search_space, pinned values, grid size)."""
    space, pinned, size = {}, {}, 1
    for name, vals in (("n_neighbors", a.k), ("window_size", a.windows), ("step_size", a.steps),
                       ("metric", a.metrics), ("pca_components", a.pca)):
        if len(vals) > 1 or name == "n_neighbors":
            space[name] = {"kind": "value", "spec": {"type": "categorical", "values": list(vals)}}
            size *= len(vals)
        else:
            pinned[name] = vals[0]
    if ":" in a.threshold:
        lo, hi = (float(v) for v in a.threshold.split(":"))
        space["threshold"] = {"kind": "value", "spec": {"type": "float_range", "min": lo, "max": hi}}
        size = None
    else:
        pinned["threshold"] = float(a.threshold)   # a categorical list of floats is a 422
    pinned["feature_names"] = list(a.features)
    return {"parameters": space}, pinned, size


def show(trials):
    rows = []
    for t in sorted(trials, key=lambda t: t.get("objective_value") or -1, reverse=True):
        r = report_anomaly(t.get("metrics_report"))
        rows.append({"id": t["id"], "status": t["status"], "values": trial_values(t), **r})
        print(f"  {t['id']}  {t['status']:<9} {json.dumps(trial_values(t))}  " + (
            f"macro-F1 {r['macro_f1']:.4f}  anomaly F1 {r['anomaly_f1']:.4f}  recall {r['recall']:.3f}  "
            f"precision {r['precision']:.3f}  windows {r['windows']:,}" if r else ""))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="ad", help="prefixes every upload; use your own")
    ap.add_argument("--windows", type=int, nargs="+", default=DEFAULTS["windows"])
    ap.add_argument("--steps", type=int, nargs="+", default=DEFAULTS["steps"])
    ap.add_argument("--k", type=int, nargs="+", default=DEFAULTS["k"])
    ap.add_argument("--metrics", nargs="+", default=DEFAULTS["metrics"])
    ap.add_argument("--pca", type=int, nargs="+", default=DEFAULTS["pca"])
    ap.add_argument("--threshold", default=DEFAULTS["threshold"], help="a value, or LOW:HIGH")
    ap.add_argument("--features", nargs="+", default=DEFAULTS["features"])
    ap.add_argument("--max-trials", type=int,
                    help=f"the trial budget (default: every point of a grid up to {DEFAULT_TRIALS}, "
                         f"else a {DEFAULT_TRIALS}-trial random search)")
    ap.add_argument("--primary", default="anomaly.macro_f1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", metavar="OPT_ID", help="collect an existing optimization instead of creating one")
    add_background_flag(ap, "{out}/optimize.log")
    a = ap.parse_args(argv)
    maybe_detach(a)
    load_dotenv()
    os.makedirs(a.out, exist_ok=True)
    state_file = os.path.join(a.out, "optimize_state.json")
    state = json.load(open(state_file)) if os.path.exists(state_file) else {"runs": []}

    if a.resume:
        opt_id = a.resume
    else:
        m = load_manifest(a.roles)
        lib, val = role_paths(a.roles, m, "library"), role_paths(a.roles, m, "validation")
        space, pinned, size = build(a)
        trials = budget(size, a.max_trials)
        log(f"library {len(lib)} files, validation {len(val)} files; pinned {json.dumps(pinned)}")
        log(f"searching {sorted(space['parameters'])}: {trials} trial(s)"
            + (" (a float range: random search)" if size is None else
               f" of a {size}-point grid" + (" (random search)" if trials < size else "")))
        gaps = sorted({(w, s) for w, s in itertools.product(a.windows, a.steps) if s > w})
        if gaps:
            sys.exit(f"step > window skips rows between windows: {gaps}")
        if a.dry_run:
            return 0
        check_auth(log)
        ids = upload_all(lib + val, a.roles, cache_path(a.out), prefix=a.name, log=log)
        bp = request("GET", f"{agents()}/blueprints/ad")
        body = {"name": f"{a.name} {stamp()}", "blueprint_id": bp["id"], "primary": a.primary,
                "overrides": {"values": {**base_values(m), **pinned}}, "search_space": space,
                "budget": {"max_trials": trials},
                "training_examples": [example(os.path.basename(p), ids[p], training=True) for p in lib],
                "validation_examples": [example(os.path.basename(p), ids[p]) for p in val]}
        opt_id = request("POST", f"{agents()}/optimizations", body)["id"]
        state["runs"].append({"optimization": opt_id, "pinned": pinned, "space": space})
        json.dump(state, open(state_file, "w"), indent=1)
        log(f"optimization {opt_id}")
    opt = wait(f"{agents()}/optimizations/{opt_id}", opt_id, log=log,
               progress=lambda o: json.dumps(o.get("progress") or {}))
    trials = list_trials(opt_id)
    print(f"\noptimization {opt_id}: {opt['status']}")
    rows = show(trials)
    json.dump({"optimization": opt, "trials": trials, "summary": rows},
              open(os.path.join(a.out, f"optimize_{opt_id}.json"), "w"), indent=1)
    if opt["status"] == "completed":
        print(f"\nnext: python promote_and_test.py --roles {a.roles} --out {a.out} --name {a.name}")
    return 0 if opt["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
