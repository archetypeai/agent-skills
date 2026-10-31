#!/usr/bin/env python3
"""Path 2, step 2: promote the best trial to a blueprint, then test it once with the Evals API.

1. Promote the best completed trial across optimize.py's runs (or --trial):
   POST /agents/optimizations/{opt}/trials/{trial}/promote with a blueprint_key unique per
   trial (choosing one takes the admin role: without it, the platform's default,
   `<base key>-<trial id>`). The blueprint lands in the optimization's project, carries the
   fitted detector (a registry artifact, `aar_…`) and pins the trial's values.
2. Test once: POST /agents/evals with the blueprint and every test file, at the trial's own
   threshold and step (an eval takes no value overrides), with predictions.
3. Check the predictions before reporting: each window's expected label equals its
   sidecar's label at the window's last row (windows tile each file from row 0 at
   step_size; a window's place in its file is its rank by `key.predicted`, its
   [start, end] in epoch seconds), and the offline confusion matrix equals the platform's.

Every id goes to <out>/test_state.json, so a rerun resumes. Once you have seen the test
number, the setting is frozen.

    python promote_and_test.py --roles roles --out out --name ad-sample
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import agents, check_auth, files_url, list_trials, load_dotenv, report_anomaly, request, root, same_deployment, upload_all, wait  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from common import cache_path, example, load_manifest, log, role_paths, stamp  # noqa: E402


def best(out):
    saved = json.load(open(os.path.join(out, "optimize_state.json")))
    if not same_deployment(saved):
        sys.exit(f"{out}/optimize_state.json is from {saved['endpoint']}, a different deployment from {root()}: "
                 f"run optimize.py here first, or set .env to that deployment")
    runs = saved["runs"]
    top = None
    for r in runs:
        for t in list_trials(r["optimization"]):
            if t["status"] == "completed" and t.get("objective_value") is not None:
                if top is None or t["objective_value"] > top[1]["objective_value"]:
                    top = (r, t)
    if not top:
        sys.exit("no completed trial: run optimize.py first")
    return top


def promote(run, trial, name):
    vals = {**run["pinned"], **(trial.get("trial_values") or {}).get("values", {})}
    if trial.get("promoted_blueprint"):
        bp = trial["promoted_blueprint"]
        return {"id": bp["id"], "key": bp["blueprint_key"], "trial": trial["id"], "values": vals}
    url = f"{agents()}/optimizations/{run['optimization']}/trials/{trial['id']}/promote"
    body = {"name": f"{name} (w {vals.get('window_size')}, step {vals.get('step_size')}, k {vals.get('n_neighbors')}, "
                    f"{vals.get('metric')}, PCA {vals.get('pca_components')}, threshold {vals.get('threshold')})",
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
    """Pair predictions with sidecar rows; raise on any mismatch; return per-type window detection."""
    by_ex = {}
    for p in preds:
        by_ex.setdefault(p["example"], []).append(p)
    offline = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    per_type, abstained = {}, 0
    for f in m["test"]:
        name = os.path.basename(f["file"])
        side = pd.read_csv(os.path.join(roles, f["labels"]), dtype=str).fillna("")
        ends = [side.iloc[i + window - 1] for i in range(0, len(side) - window + 1, step)]
        # `input` is 0 on every record; the window's own [start, end] orders them
        got = sorted(by_ex.get(name, []), key=lambda p: p["key"]["predicted"][0])
        if len(got) != len(ends):
            sys.exit(f"{name}: {len(got)} predictions for {len(ends)} windows at window {window}, step {step}")
        for i, (row, p) in enumerate(zip(ends, got)):
            if p["expected"] != row["label"]:
                sys.exit(f"{name} window {i}: platform expected {p['expected']}, sidecar {row['label']}: misaligned")
            if p["outcome"] == "missed" or p["predicted"] is None:   # the detector abstained
                abstained += 1
                continue
            g, pr = row["label"], p["predicted"]
            offline[("t" if g == pr else "f") + ("p" if pr == "anomaly" else "n")] += 1
            if g == "anomaly":
                d = per_type.setdefault(row["type"] or row["recording"], [0, 0])
                d[0] += pr == "anomaly"
                d[1] += 1
    r = report_anomaly(ev["metrics_report"])
    platform = {k: r[k] for k in ("tp", "fn", "fp", "tn")}
    if offline != platform:
        sys.exit(f"offline confusion {offline} != platform {platform}")
    return per_type, abstained


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="ad")
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
        run, trial = best(a.out)
        if a.trial:
            trial = next(t for r in [run] for t in list_trials(r["optimization"]) if t["id"] == a.trial)
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
    per_type, abstained = check(a.roles, m, ev, preds, v["window_size"], v["step_size"])
    r = report_anomaly(ev["metrics_report"])
    print(f"\ntest, {len(m['test'])} files: macro-F1 {r['macro_f1']:.4f}  anomaly F1 {r['anomaly_f1']:.4f}  "
          f"recall {r['recall']:.3f}  precision {r['precision']:.3f}  windows {r['windows']:,} "
          f"({abstained} abstained)")
    print(f"  false alarms on normal windows: {r['fp']:,} / {r['fp'] + r['tn']:,} "
          f"({r['fp'] / max(r['fp'] + r['tn'], 1):.2%})")
    for k, (hit, n) in sorted(per_type.items()):
        print(f"  {k:<14} window detection {hit:>5,} / {n:>5,}  {hit / n:.3f}")
    json.dump({"eval": state["eval"], "blueprint": bp, **r, "abstained": abstained,
               "detection_by_type": {k: {"flagged": h, "windows": n} for k, (h, n) in per_type.items()}},
              open(os.path.join(a.out, "test.json"), "w"), indent=1)
    print(f"\nnext: python deliver.py --roles {a.roles} --out {a.out} --name {a.name}")
    return 0


def request_raw(url):
    import urllib.request
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


if __name__ == "__main__":
    sys.exit(main())
