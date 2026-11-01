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

--max-windows N tests in pieces, for a deployment that stops a job after a time limit (the
free trial: 2 hours): each test file is split into pieces of at most N windows that overlap
by window - step rows (so every window lies whole in one piece), one eval per piece, at most
--parallel at a time (a queued job may already be on the clock). The pieces' predictions
are joined per file and checked as one eval's; the platform's confusion matrix is their sum,
and its F1s, accuracy and detection precision/recall are recomputed from the sum. The plan is
saved with its N: a rerun at another N cuts a new one, keeping only the evals whose file and
rows are unchanged, so old and new boundaries never mix. Piece files are named by row range.

Every id goes to <out>/test_state.json, so a rerun resumes. Once you have seen the test
number, the setting is frozen.

    python promote_and_test.py --roles roles --out out --name red-sample
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import TERMINAL, agents, check_auth, files_url, list_trials, load_dotenv, report_event, request, root, same_deployment, upload_all, wait  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from common import cache_path, example, load_manifest, log, role_paths, stamp  # noqa: E402
from deliver import pieces  # noqa: E402
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
    url = f"{agents()}/optimizations/{run['optimization']}/trials/{trial['id']}/promote"
    body = {"name": f"{name} (window {vals.get('window_size')}, step {vals.get('step_size')}, {vals.get('metric')})",
            "description": f"Promoted from {run['optimization']}, trial {trial['id']}."}
    try:
        bp = request("POST", url, {"blueprint_key": f"{name}-{trial['id'][-6:]}", **body})
    except RuntimeError as e:
        # choosing the key needs the admin role (a free-trial org has none): let the platform choose it
        if "(403)" not in str(e) or "blueprint_key" not in str(e):
            raise
        bp = request("POST", url, body)
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


def summed_report(reports, background):
    """One metrics_report from several evals': the confusion matrices summed, and the
    aggregates recomputed from the sum (the per-eval ones can't be averaged): per-class F1,
    macro and foreground macro F1, accuracy, and detection recall/precision (any event
    class against the background)."""
    ts = [r["targets"]["event"] for r in reports]
    names = ts[0]["class_names"]
    k = len(names)
    cm = [[sum(t["confusion_matrix"][g][q] for t in ts) for q in range(k)] for g in range(k)]
    f1 = []
    for c in range(k):
        tp, fp, fn = cm[c][c], sum(cm[g][c] for g in range(k)) - cm[c][c], sum(cm[c]) - cm[c][c]
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    b = names.index(background)
    ev = [c for c in range(k) if c != b]
    tp = sum(cm[g][q] for g in ev for q in ev)
    fn, fp = sum(cm[g][b] for g in ev), sum(cm[b][q] for q in ev)
    agg = {"macro_f1": sum(f1) / k, "foreground_macro_f1": sum(f1[c] for c in ev) / len(ev),
           "accuracy": sum(cm[c][c] for c in range(k)) / max(1, sum(map(sum, cm))),
           "detection_recall": tp / (tp + fn) if tp + fn else 0.0,
           "detection_precision": tp / (tp + fp) if tp + fp else 0.0}
    return {"targets": {"event": {"class_names": names, "confusion_matrix": cm, "aggregate": agg}}}


def piece_plan(a, m, window, step):
    """Every test file cut into pieces of at most --max-windows windows, each written once
    to <out>/test_pieces/<file>_rows<lo>-<hi>.csv."""
    pdir = os.path.join(a.out, "test_pieces")
    os.makedirs(pdir, exist_ok=True)
    plan = []
    for f in m["test"]:
        df = pd.read_csv(os.path.join(a.roles, f["file"]))
        stem = os.path.splitext(os.path.basename(f["file"]))[0]
        for lo, hi in pieces(len(df), window, step, a.max_windows):
            path = os.path.join(pdir, f"{stem}_rows{lo}-{hi}.csv")
            if not os.path.exists(path):
                df.iloc[lo:hi].to_csv(path, index=False)
            plan.append({"file": os.path.basename(f["file"]), "path": path, "rows": [lo, hi]})
    return plan


def test_in_pieces(a, m, bp, state, sf):
    """One eval per piece of at most --max-windows windows, --parallel at a time; returns
    (a stand-in eval with the summed report, the joined predictions)."""
    v = bp["values"]
    window, step = v["window_size"], v["step_size"]
    if "pieces" not in state or state.get("max_windows") != a.max_windows:
        old = {(x["file"], tuple(x["rows"])): x for x in state.get("pieces", []) if x.get("eval")}
        plan = piece_plan(a, m, window, step)
        kept = 0
        for x in plan:
            if (o := old.get((x["file"], tuple(x["rows"])))):
                x["eval"], x["status"] = o["eval"], o.get("status")
                kept += 1
        state["pieces"], state["max_windows"] = plan, a.max_windows
        json.dump(state, open(sf, "w"), indent=1)
        log(f"{len(plan)} pieces of at most {a.max_windows} windows"
            + (f"; {kept} eval(s) kept from the earlier plan" if kept else ""))
    ids = upload_all([x["path"] for x in state["pieces"]], a.out, cache_path(a.out), prefix=a.name, log=log)
    while True:
        running = 0
        for x in state["pieces"]:
            if x.get("eval") and x.get("status") not in TERMINAL:
                x["status"] = request("GET", f"{agents()}/evals/{x['eval']}").get("status")
                if x["status"] in TERMINAL:
                    log(f"{os.path.basename(x['path'])}: eval {x['eval']} {x['status']}")
            running += bool(x.get("eval")) and x.get("status") not in TERMINAL
        for x in state["pieces"]:
            if running >= a.parallel:
                break
            if not x.get("eval"):
                name = os.path.basename(x["path"])
                ev = request("POST", f"{agents()}/evals", {
                    "blueprint_id": bp["id"], "name": f"{a.name} test {name} {stamp()}", "emit_predictions": True,
                    "examples": [example(name, ids[x["path"]])]})
                x["eval"], x["status"] = ev["id"], ev.get("status")
                running += 1
                log(f"{name}: eval {ev['id']} (rows {x['rows'][0]:,}-{x['rows'][1]:,})")
        json.dump(state, open(sf, "w"), indent=1)
        failed = [x for x in state["pieces"] if x.get("status") in TERMINAL and x["status"] != "completed"]
        if failed:
            for x in failed:
                ev = request("GET", f"{agents()}/evals/{x['eval']}")
                log(f"{os.path.basename(x['path'])}: eval {x['eval']} {x['status']}: {ev.get('error')}")
                x.pop("eval"), x.pop("status")
            json.dump(state, open(sf, "w"), indent=1)
            sys.exit("rerun to retry the failed pieces, or rerun with a smaller --max-windows (a new plan)")
        if all(x.get("status") == "completed" for x in state["pieces"]):
            break
        time.sleep(30)

    reports, preds = [], []
    for x in state["pieces"]:
        ev = request("GET", f"{agents()}/evals/{x['eval']}")
        if not ev.get("metrics_report"):
            log(f"{os.path.basename(x['path'])}: eval {x['eval']} completed without a metrics report")
            x.pop("eval"), x.pop("status")
            json.dump(state, open(sf, "w"), indent=1)
            sys.exit("rerun to test that piece again")
        reports.append(ev["metrics_report"])
        pred = next(o["id"] for o in ev["output_artifacts"] if o.get("metadata", {}).get("kind") == "predictions")
        for line in request_raw(f"{files_url()}/download/{pred}").decode().splitlines():
            if line.strip():
                p = json.loads(line)
                p["example"] = x["file"]          # windows sort by their epoch start across pieces
                preds.append(p)
    ev = {"metrics_report": summed_report(reports, m["background"])}
    with open(os.path.join(a.out, "test-predictions.ndjson"), "w") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in preds)
    return ev, preds


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
    ap.add_argument("--max-windows", type=int, default=0,
                    help="test in pieces of at most this many windows, one eval each (0: one eval)")
    ap.add_argument("--parallel", type=int, default=3, help="evals at a time, with --max-windows")
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
    if a.max_windows:
        ev, preds = test_in_pieces(a, m, bp, state, sf)
        state["eval"] = [x["eval"] for x in state["pieces"]]
    elif "eval" not in state:
        test = role_paths(a.roles, m, "test")
        ids = upload_all(test, a.roles, cache_path(a.out), prefix=a.name, log=log)
        ev = request("POST", f"{agents()}/evals", {
            "blueprint_id": bp["id"], "name": f"{a.name} test {stamp()}", "emit_predictions": True,
            "examples": [example(os.path.basename(p), ids[p]) for p in test]})
        state["eval"] = ev["id"]
        json.dump(state, open(sf, "w"), indent=1)
        log(f"eval {ev['id']}: {len(test)} test files")
    if not a.max_windows:
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
