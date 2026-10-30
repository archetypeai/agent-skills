#!/usr/bin/env python3
"""Path 2, step 2: promote the best trial to a blueprint, then test it once with the Evals API.

1. Promote the best completed trial across optimize.py's runs (a tie goes to the
   optimization that tried more points; --trial picks one): POST
   /agents/optimizations/{opt}/trials/{trial}/promote with a blueprint_key unique per
   trial. The blueprint pins the trial's values, `events` and `background`, and carries
   the fitted prototypes as a registry artifact (`aar_…`).
2. Test once: POST /agents/evals with the blueprint and every test file, with predictions.
   An eval takes no value overrides: it runs at the trial's window and step.
3. Check before reporting. The platform scores each row once: a window answers for the
   rows since the previous window ended (all its rows for the first), collapsed to their
   last row. Every prediction's expected label must equal that, read from the file's
   `label` column (windows tile each file from row 0 at step_size; a window's place in its
   file is its rank by `key.predicted`, its [start, end] in epoch seconds), and the
   offline confusion matrix must equal the platform's.
4. Score as the design does (score.py): each window by the majority of its rows; each
   incident detected or missed, and how soon; false alarms on windows with no event row.

Every id goes to <out>/test_state.json, so a rerun resumes. Once you have seen the test
number, the setting is frozen.

    python promote_and_test.py --roles roles --out out --name red-sample
"""
import argparse
import json
import os
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import agents, check_auth, files_url, list_trials, load_dotenv, report_event, request, root, same_deployment, upload_all, wait  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from common import cache_path, example, load_manifest, log, role_paths, stamp  # noqa: E402
from score import prf, score_windows  # noqa: E402


def best(runs, trials_of=list_trials):
    """(run, trial): the best completed trial; a tie goes to the run that tried more points."""
    top = None
    for r in runs:
        ts = trials_of(r["optimization"])
        tried = sum(t["status"] == "completed" for t in ts)
        for t in ts:
            if t["status"] == "completed" and t.get("objective_value") is not None:
                key = (t["objective_value"], tried)
                if top is None or key > top[0]:
                    top = (key, r, t)
    if not top:
        sys.exit("no completed trial: run optimize.py first")
    return top[1], top[2]


def promote(run, trial, name):
    vals = {**run["pinned"], **(trial.get("trial_values") or {}).get("values", {})}
    if trial.get("promoted_blueprint"):
        bp = trial["promoted_blueprint"]
        return {"id": bp["id"], "key": bp["blueprint_key"], "trial": trial["id"], "values": vals}
    bp = request("POST", f"{agents()}/optimizations/{run['optimization']}/trials/{trial['id']}/promote", {
        "blueprint_key": f"{name}-{trial['id'][-6:]}",
        "name": f"{name} (window {vals.get('window_size')}, step {vals.get('step_size')}, {vals.get('metric')})",
        "description": f"Promoted from {run['optimization']}, trial {trial['id']}."})
    return {"id": bp["id"], "key": bp["blueprint_key"], "trial": trial["id"], "values": vals}


def check(roles, m, ev, preds, window, step):
    """Pair predictions with rows; exit on any mismatch; return the design-rule score per file."""
    by_ex = {}
    for p in preds:
        by_ex.setdefault(p["example"], []).append(p)
    names = report_event(ev["metrics_report"], m["background"])["class_names"]
    offline = [[0] * len(names) for _ in names]
    per_file, abstained = {}, 0
    for f in m["test"]:
        name = os.path.basename(f["file"])
        labels = pd.read_csv(os.path.join(roles, f["file"]), usecols=["label"]).label.tolist()
        starts = list(range(0, len(labels) - window + 1, step))
        # `input` is 0 on every record; the window's own [start, end] orders them
        got = sorted(by_ex.get(name, []), key=lambda p: p["key"]["predicted"][0])
        if len(got) != len(starts):
            sys.exit(f"{name}: {len(got)} predictions for {len(starts)} windows at window {window}, step {step}")
        windows = []
        for k, (s, p) in enumerate(zip(starts, got)):
            gold = labels[s + window - 1]           # the span's last row: at step 1, the window's last row
            if p["expected"] != gold:
                sys.exit(f"{name} window {k}: platform expected {p['expected']}, the label column says {gold}: misaligned")
            if p["outcome"] == "missed" or p["predicted"] is None:   # the classifier abstained
                abstained += 1
                continue
            offline[names.index(gold)][names.index(p["predicted"])] += 1
            windows.append((s, s + window, p["predicted"]))
        per_file[name] = score_windows(labels, windows, m["background"])
    platform = report_event(ev["metrics_report"], m["background"])["confusion_matrix"]
    if offline != platform:
        sys.exit(f"offline confusion {offline} != platform {platform} (rows = gold, {names})")
    return per_file, abstained


def request_raw(url):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="red")
    ap.add_argument("--trial", help="promote this trial instead of the best")
    add_background_flag(ap, "{out}/test.log")
    a = ap.parse_args(argv)
    maybe_detach(a)
    load_dotenv()
    check_auth(log)
    m = load_manifest(a.roles)
    sf = os.path.join(a.out, "test_state.json")
    state = json.load(open(sf)) if os.path.exists(sf) else {}
    if not same_deployment(state):
        sys.exit(f"{sf} is from {state['endpoint']}, a different deployment from {root()}: "
                 f"move it aside to promote and test here, or set .env to that deployment")
    state["endpoint"] = root()
    if "blueprint" not in state:
        saved = json.load(open(os.path.join(a.out, "optimize_state.json")))
        if not same_deployment(saved):
            sys.exit(f"{a.out}/optimize_state.json is from {saved['endpoint']}, a different deployment from {root()}: "
                     f"run optimize.py here first, or set .env to that deployment")
        runs = saved["runs"]
        run, trial = best(runs)
        if a.trial:
            run, trial = next((r, t) for r in runs for t in list_trials(r["optimization"]) if t["id"] == a.trial)
        state["blueprint"] = promote(run, trial, a.name)
        json.dump(state, open(sf, "w"), indent=1)
        log(f"promoted {trial['id']} -> {state['blueprint']['key']} ({state['blueprint']['id']})")
    bp = state["blueprint"]
    if "eval" not in state:
        test = role_paths(a.roles, m, "test")
        ids = upload_all(test, a.roles, cache_path(a.out), prefix=a.name, log=log)
        ev = request("POST", f"{agents()}/evals", {
            "blueprint_id": bp["id"], "name": f"{a.name} test {stamp()}", "emit_predictions": True,
            "examples": [example(os.path.basename(p), ids[p]) for p in test]})
        state["eval"] = ev["id"]
        json.dump(state, open(sf, "w"), indent=1)
        log(f"eval {ev['id']}: {len(test)} test files")
    ev = wait(f"{agents()}/evals/{state['eval']}", state["eval"], log=log)
    if ev["status"] != "completed" or not ev.get("metrics_report"):
        sys.exit(f"eval {state['eval']} ended {ev['status']}: {ev.get('error')}")
    pred = next(x["id"] for x in ev["output_artifacts"] if x.get("metadata", {}).get("kind") == "predictions")
    raw = request_raw(f"{files_url()}/download/{pred}")
    with open(os.path.join(a.out, "test-predictions.ndjson"), "wb") as fh:
        fh.write(raw)
    preds = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    v = bp["values"]
    per_file, abstained = check(a.roles, m, ev, preds, v["window_size"], v["step_size"])
    r = report_event(ev["metrics_report"], m["background"])
    print(f"\ntest, {len(m['test'])} files (window {v['window_size']}, step {v['step_size']}, {v.get('metric')}; "
          f"{abstained} abstained)")
    print(f"  platform (each row once): event F1 {r['foreground_macro_f1']:.4f}  macro-F1 {r['macro_f1']:.4f}  "
          f"recall {r['recall']:.4f}  precision {r['precision']:.4f}")
    pooled = {k: sum(x[k] for x in per_file.values()) for k in ("tp", "fn", "fp", "tn", "false_alarms", "normal_windows")}
    p, rec, f1 = prf(pooled)
    print(f"  design rule (each window by majority): event F1 {f1:.4f}  precision {p:.4f}  recall {rec:.4f}")
    print(f"  false alarms {pooled['false_alarms']:,} / {pooled['normal_windows']:,} windows with no event row "
          f"({pooled['false_alarms'] / max(pooled['normal_windows'], 1):.2%})")
    for name, x in per_file.items():
        for i in x["incidents"]:
            when = f"+{i['delay_rows']} rows" if i["detected"] else "MISSED"
            print(f"    {name:<14} {i['class']} {i['length']:>6,} rows  {when:<11}  false alarms "
                  f"{x['false_alarms']:>4} / {x['normal_windows']:,}")
    json.dump({"eval": state["eval"], "blueprint": bp, "abstained": abstained,
               "platform": {k: x for k, x in r.items() if k not in ("class_names", "confusion_matrix")},
               "design_rule": {**pooled, "event_f1": f1}, "files": per_file},
              open(os.path.join(a.out, "test.json"), "w"), indent=1)
    print(f"\nnext: python deliver.py --roles {a.roles} --out {a.out} --name {a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
