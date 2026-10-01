#!/usr/bin/env python3
"""Path 2, step 2: promote the best trial to a blueprint, then test it once with the Evals API.

  1. Promote: POST /agents/optimizations/{opt}/trials/{trial}/promote turns a trial into
     a blueprint with its setting and fitted kNN attached: a reusable model, no S3, no
     classifier artifact to build. The best completed trial by validation macro-F1 is
     used (from <out>/optimize_*.json) unless --trial names one. The blueprint key ends
     in the trial's id, so a rerun with the same setting never reuses an older model.
  2. Test: one eval (POST /agents/evals) over every test file, each window paired with
     the `label` at its last row. The report is pooled over the files.

Every id is saved in <out>/test_state.json as soon as it exists, so a rerun resumes
instead of promoting or testing again. This is the one-shot test: once its number is
seen, the setting is frozen. Writes <out>/test.json.

    python promote_and_test.py --roles roles --background
    python promote_and_test.py --trial otr_... --background
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import agents, check_auth, list_trials, load_dotenv, report_f1, request, trial_setting, upload_all, wait  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from common import cache_path, example, load_manifest, log, role_paths, stamp  # noqa: E402

REPORT_WAIT_S = 600            # how long to wait for the metrics report after the platform says completed


def blueprint_key(trial, name="osm"):
    """Unique per trial: `<name>-w<window>-s<step>-<metric>-k<k>-<weights>-<last 6 of the trial id>`."""
    w, st, k, metric, weights = trial_setting(trial)
    return f"{name}-w{w}-s{st}-{metric}-k{k}-{weights}-{trial['id'][-6:]}".lower()


def best_trial(runs, trial_id=None):
    """(optimization id, trial) of the best completed trial across saved runs, or of trial_id."""
    trials = [(r["optimization"]["id"], t) for r in runs for t in r["trials"]
              if t.get("status") == "completed" and t.get("objective_value") is not None]
    if trial_id:
        trials = [(o, t) for o, t in trials if t["id"] == trial_id]
    if not trials:
        return None
    return max(trials, key=lambda ot: ot[1]["objective_value"])


def promote(opt, trial, name):
    key = blueprint_key(trial, name)
    try:
        return request("GET", f"{agents()}/blueprints/{key}")      # promoted on an earlier, interrupted run
    except RuntimeError:
        pass
    w, st, k, metric, weights = trial_setting(trial)
    bp = request("POST", f"{agents()}/optimizations/{opt}/trials/{trial['id']}/promote", body={
        "blueprint_key": key, "name": f"{name} OSM (w{w}, s{st}, {metric}, k{k}, {weights})",
        "description": f"Promoted from {opt} / {trial['id']}: window {w}, step {st}, {metric}, k {k}, {weights}."})
    log(f"promoted {trial['id']} -> {bp['blueprint_key']} ({bp['id']})")
    return bp


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="osm", help="the blueprint key's prefix")
    ap.add_argument("--trial", metavar="OTR_ID", help="promote this trial instead of the best one")
    ap.add_argument("--upload-jobs", type=int, default=3)
    add_background_flag(ap, default_log="{out}/test.log")
    args = ap.parse_args()
    maybe_detach(args)
    load_dotenv()
    check_auth(log)
    manifest = load_manifest(args.roles)
    states = manifest["states"]
    state_path = os.path.join(args.out, "test_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}

    def save():
        os.makedirs(args.out, exist_ok=True)
        json.dump(state, open(state_path, "w"), indent=1)

    if "model" in state:
        opt = state["model"]["optimization"]
        trial = next(t for t in list_trials(opt) if t["id"] == state["model"]["trial"])
    else:
        runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.out, "optimize_opt_*.json")))]
        found = best_trial(runs, args.trial)
        if not found:
            sys.exit(f"no completed trial in {args.out}/optimize_*.json: run optimize.py first")
        opt, trial = found
        state["model"] = {"optimization": opt, "trial": trial["id"]}
        save()
    w, st, k, metric, weights = trial_setting(trial)
    log(f"model: {opt} / {trial['id']} (w {w}, step {st}, k {k}, {metric}, {weights}; "
        f"validation macro-F1 {trial['objective_value']:.4f})")
    if "blueprint" not in state:
        bp = promote(opt, trial, args.name)
        state["blueprint"] = {"id": bp["id"], "key": bp["blueprint_key"]}
        save()

    paths = role_paths(args.roles, manifest, "test")
    ids = upload_all(paths, args.roles, cache_path(args.out), prefix=args.name, jobs=args.upload_jobs, log=log)
    if "eval" not in state:
        ev = request("POST", f"{agents()}/evals", body={
            "blueprint_id": state["blueprint"]["id"], "name": f"{args.name} test {stamp()}", "emit_predictions": False,
            "examples": [example(os.path.basename(p)[:-4], ids[p]) for p in paths]})
        state["eval"] = ev["id"]
        save()
        log(f"eval {ev['id']}: {len(paths)} test file(s)")
    ev = wait(f"{agents()}/evals/{state['eval']}", state["eval"], every_s=60, log=log)
    # the platform can report completed before the eval's metrics report is there: the
    # eval counts as completed here only once it is
    deadline = time.time() + REPORT_WAIT_S
    if ev["status"] == "completed" and not ev.get("metrics_report"):
        log(f"{state['eval']}: platform: completed; waiting for its metrics report")
    while ev["status"] == "completed" and not ev.get("metrics_report") and time.time() < deadline:
        time.sleep(30)
        ev = request("GET", f"{agents()}/evals/{state['eval']}")
    json.dump(ev, open(os.path.join(args.out, f"test_{state['eval']}.json"), "w"), indent=1)
    if ev["status"] != "completed":
        sys.exit(f"eval {state['eval']} ended {ev['status']}: {ev.get('error')}")
    if not ev.get("metrics_report"):
        sys.exit(f"eval {state['eval']} completed but has no metrics report after {REPORT_WAIT_S // 60} min: "
                 f"a platform problem (report the eval id); rerun this command to check again")

    f1, n = report_f1(ev["metrics_report"])
    macro = ev["metrics_report"]["primary"]["value"]
    res = {"model": {**state["model"], "blueprint": state["blueprint"], "window": w, "step": st, "k": k,
                     "metric": metric, "weights": weights},
           "eval": state["eval"], "macro_f1": round(macro, 4), "f1": {s: round(f1.get(s, 0.0), 4) for s in states},
           "windows": n, "confusion": ev["metrics_report"]["targets"]["state"]["confusion_matrix"],
           "class_names": ev["metrics_report"]["targets"]["state"]["class_names"]}
    json.dump(res, open(os.path.join(args.out, "test.json"), "w"), indent=1)
    print(f"\ntest, {len(paths)} files: macro-F1 {macro:.4f}  " + "  ".join(f"{s} {res['f1'][s]:.2f}" for s in states)
          + f"  windows {n:,}")
    log(f"wrote {os.path.join(args.out, 'test.json')}; next: python deliver.py --roles {args.roles} --out {args.out} --name {args.name}")


if __name__ == "__main__":
    main()
