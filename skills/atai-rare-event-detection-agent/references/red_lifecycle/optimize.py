#!/usr/bin/env python3
"""Path 2, step 1: fit and score RED settings on your role files with the Optimizations API.

Each training file becomes a training example labelled by its one class (a constant);
each validation file a validation example scored on its `label` column. Per trial the
platform embeds every window with Omega 1.5, averages each class's embeddings into a
prototype, and scores validation; the ranking objective is --primary
(default `event.foreground_macro_f1`: the F1 over the event classes, background apart).

What can be searched (`kind: value`):
  --metrics    the distance to the prototypes: cosine, l1, l2, dot (the classifier's only
               fit setting; the blueprint defaults to l1)
  --windows    window_size: a list, or LOW:HIGH (an inclusive integer range)
  --steps      step_size: a list, or LOW:HIGH. Both Optimize legs use the trial's step
One value pins a setting in `overrides.values`; `metric` is always in the search space
(it can't be empty). Trials that share window and step share one encoder pass, so the
metric is nearly free and each window or step adds a pass over every file.

The platform draws distinct points at random: a budget at or above the grid's size runs
every point (a grid), below it, or with a range, a random search. By default a grid of up
to DEFAULT_TRIALS points runs in full, and a bigger grid or a range gets a
DEFAULT_TRIALS-trial random search. --max-trials sets the budget.

Writes <out>/optimize_<optimization id>.json and <out>/optimize_state.json.

    python optimize.py --roles roles --dry-run                       # the plan: no uploads, no jobs
    python optimize.py --roles roles --out out --name red-sample     # window 64, step 1, four metrics
    python optimize.py --roles roles --out out --windows 16 32 64    # a 12-point grid
    python optimize.py --resume opt_...                              # collect a run whose poller died
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import agents, check_auth, list_trials, load_dotenv, report_event, request, trial_values, upload_all, wait  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from common import base_values, cache_path, example, load_manifest, log, role_paths, stamp  # noqa: E402

DEFAULTS = {"windows": ["64"], "steps": ["1"], "metrics": ["cosine", "l1", "l2", "dot"]}
DEFAULT_TRIALS = 12


def spec(values, cast=int):
    """(search spec or None, pinned value or None, number of points or None for a range)."""
    if len(values) == 1 and ":" in values[0]:
        lo, hi = (int(v) for v in values[0].split(":"))
        return {"type": "int_range", "min": lo, "max": hi}, None, None
    vals = [cast(v) for v in values]
    if len(vals) == 1:
        return None, vals[0], 1
    return {"type": "categorical", "values": vals}, None, len(vals)


def build(a):
    """(search_space, pinned values, grid size or None)."""
    space, pinned, size = {}, {}, 1
    for name, vals, cast in (("metric", a.metrics, str), ("window_size", a.windows, int), ("step_size", a.steps, int)):
        s, pin, n = spec(vals, cast)
        if name == "metric" and s is None:                # the space can't be empty
            s = {"type": "categorical", "values": [pin]}
        if s is None:
            pinned[name] = pin
        else:
            space[name] = {"kind": "value", "spec": s}
            size = None if n is None or size is None else size * (n or 1)
    return {"parameters": space}, pinned, size


def budget(size, max_trials=None):
    """--max-trials if given (never more than a grid), else the whole grid up to
    DEFAULT_TRIALS points, else a DEFAULT_TRIALS-trial random search."""
    if size is None:
        return max_trials or DEFAULT_TRIALS
    return min(max_trials or min(size, DEFAULT_TRIALS), size)


def show(trials, background):
    rows = []
    for t in sorted(trials, key=lambda t: t.get("objective_value") or -1, reverse=True):
        r = report_event(t.get("metrics_report"), background)
        rows.append({"id": t["id"], "status": t["status"], "values": trial_values(t),
                     **{k: v for k, v in r.items() if k not in ("class_names", "confusion_matrix")}})
        print(f"  {t['id']}  {t['status']:<9} {json.dumps(trial_values(t))}  " + (
            f"event F1 {r['foreground_macro_f1']:.4f}  macro-F1 {r['macro_f1']:.4f}  recall {r['recall']:.3f}  "
            f"precision {r['precision']:.3f}  windows {r['windows']:,}" if r else ""))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="red", help="prefixes every upload; use your own")
    ap.add_argument("--windows", nargs="+", default=DEFAULTS["windows"])
    ap.add_argument("--steps", nargs="+", default=DEFAULTS["steps"])
    ap.add_argument("--metrics", nargs="+", default=DEFAULTS["metrics"])
    ap.add_argument("--max-trials", type=int,
                    help=f"the trial budget (default: every point of a grid up to {DEFAULT_TRIALS}, "
                         f"else a {DEFAULT_TRIALS}-trial random search)")
    ap.add_argument("--primary", default="event.foreground_macro_f1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", metavar="OPT_ID", help="collect an existing optimization instead of creating one")
    add_background_flag(ap, "{out}/optimize.log")
    a = ap.parse_args(argv)
    maybe_detach(a)
    load_dotenv()
    os.makedirs(a.out, exist_ok=True)
    state_file = os.path.join(a.out, "optimize_state.json")
    state = json.load(open(state_file)) if os.path.exists(state_file) else {"runs": []}
    m = load_manifest(a.roles)

    if a.resume:
        opt_id = a.resume
    else:
        train = [(os.path.join(a.roles, f["file"]), f["class"]) for f in m["training"]]
        val = role_paths(a.roles, m, "validation")
        space, pinned, size = build(a)
        trials = budget(size, a.max_trials)
        steps = [pinned["step_size"]] if "step_size" in pinned else []
        wins = [pinned["window_size"]] if "window_size" in pinned else []
        if steps and wins and steps[0] > wins[0]:
            sys.exit(f"step {steps[0]} > window {wins[0]} skips rows between windows")
        log(f"training {len(train)} files ({', '.join(sorted({c for _, c in train}))}), validation {len(val)} files; "
            f"pinned {json.dumps(pinned)}")
        log(f"searching {sorted(space['parameters'])}: {trials} trial(s)"
            + (" (a range: random search)" if size is None else
               f" of a {size}-point grid" + (" (random search)" if trials < size else "")))
        if a.dry_run:
            print(json.dumps(space, indent=1))
            return 0
        check_auth(log)
        ids = upload_all([p for p, _ in train] + val, a.roles, cache_path(a.out), prefix=a.name, log=log)
        bp = request("GET", f"{agents()}/blueprints/red")
        body = {"name": f"{a.name} {stamp()}", "blueprint_id": bp["id"], "primary": a.primary,
                "overrides": {"values": {**base_values(m), **pinned}}, "search_space": space,
                "budget": {"max_trials": trials},
                "training_examples": [example(os.path.basename(p), ids[p], c) for p, c in train],
                "validation_examples": [example(os.path.basename(p), ids[p]) for p in val]}
        opt_id = request("POST", f"{agents()}/optimizations", body)["id"]
        state["runs"].append({"optimization": opt_id, "pinned": pinned, "space": space, "trials": trials})
        json.dump(state, open(state_file, "w"), indent=1)
        log(f"optimization {opt_id}")
    opt = wait(f"{agents()}/optimizations/{opt_id}", opt_id, log=log,
               progress=lambda o: json.dumps(o.get("progress") or {}))
    trials = list_trials(opt_id)
    print(f"\noptimization {opt_id}: {opt['status']}")
    rows = show(trials, m["background"])
    json.dump({"optimization": opt, "trials": trials, "summary": rows},
              open(os.path.join(a.out, f"optimize_{opt_id}.json"), "w"), indent=1)
    if opt["status"] == "completed":
        print(f"\nnext: python promote_and_test.py --roles {a.roles} --out {a.out} --name {a.name}")
    return 0 if opt["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
