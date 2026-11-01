#!/usr/bin/env python3
"""Path 2, cleanup: delete the agents and bundles deliver.py, serve.py, library_embeddings.py
and a serving app left on the platform.

    python cleanup.py --out out           # list what would be deleted, and what stays
    python cleanup.py --out out --yes     # delete it

From the ids in <out>/:
  app_serving_state.json   a serving app's agents and bundle, if its bridge was killed
                           before it could delete them (the bridge pattern in SKILL.md
                           writes this file).
  library_state.json       library_embeddings.py's batch runs and their bundle.
  library_serving_state.json  the serving agent library_embeddings.py --serving started, if it
                           was killed before it could delete it.
  serve_state.json         a serving agent and its bundle, if serve.py was killed before it
                           could delete them. A running serving agent holds a serving worker
                           until it is cancelled, or idle for 20 minutes.
  delivery/runs.json       the delivery agents and their bundle. The bundle's own listing
                           (GET /agents/instances?bundle_id=) also finds runs a rerun replaced.

Agents go first (a bundle that still has an agent answers 409), a running one is cancelled
before it is deleted, and an id already gone (404) counts as done. Progress goes to
<out>/cleanup_state.json, so a rerun picks up where it stopped.

What stays: the promoted blueprint (deleting a blueprint takes an org admin's key; serve.py
serves it as it is, with no copy), and the optimizations and eval (records your results
cite). After a cleanup, deliver.py --resume can't collect again: its bundle is gone.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import TERMINAL, agents, check_auth, list_all, load_dotenv, request, root, same_deployment  # noqa: E402
from common import log  # noqa: E402

STATES = {"app": "app_serving_state.json", "library": "library_state.json", "serving": "serve_state.json",
          "library serving": "library_serving_state.json", "delivery": os.path.join("delivery", "runs.json")}


def call(method, path, body=None):
    """The response, or None when the object is already gone."""
    try:
        return request(method, f"{agents()}{path}", body)
    except RuntimeError as e:
        if "(404)" in str(e):
            return None
        raise


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def known_agents(st):
    """The agent ids a state file names, in any of the scripts' shapes."""
    ids = {st.get("agent")} | set(st.get("agents") or [])
    ids |= set((st.get("runs") or {}).values()) if isinstance(st.get("runs"), dict) else {
        r["agent"] for r in st.get("runs") or []}
    return {a for a in ids if a}


def targets(found, runs_of):
    """(what, kind, id) to delete, each agent before its bundle. `found` maps what -> state."""
    out = []
    for what, st in found.items():
        if not st.get("bundle"):
            continue
        ids = known_agents(st) | (set() if what in ("serving", "library serving") else runs_of(st["bundle"]))
        out += [(what, "instances", a) for a in sorted(ids)] + [(what, "bundles", st["bundle"])]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out")
    ap.add_argument("--yes", action="store_true", help="delete; without it, only list")
    a = ap.parse_args(argv)
    load_dotenv()
    check_auth(log)
    found = {what: load(os.path.join(a.out, f)) for what, f in STATES.items()}
    for what, st in found.items():
        if not same_deployment(st):
            sys.exit(f"{STATES[what]} is from {st['endpoint']}, a different deployment from {root()}: "
                     f"set .env to that deployment")
    todo_all = targets(found, lambda b: {r["id"] for r in list_all(f"{agents()}/instances?bundle_id={b}")})
    done_path = os.path.join(a.out, "cleanup_state.json")
    gone = set(load(done_path).get("deleted", []))
    todo = [t for t in todo_all if t[2] not in gone]
    for what in STATES:
        mine = [t for t in todo if t[0] == what]
        print(f"{what}: {sum(t[1] == 'instances' for t in mine)} agent(s), "
              f"{sum(t[1] == 'bundles' for t in mine)} bundle(s) to delete")
    print(f"already deleted: {len(gone & {t[2] for t in todo_all})}")
    test = load(os.path.join(a.out, "test_state.json"))
    if test.get("blueprint"):
        key = test["blueprint"]["key"]
        print(f"stays: blueprint {key} (an org admin can delete it), "
              f"and the optimizations and eval")
    if not a.yes:
        print("\nnothing deleted: rerun with --yes")
        return 0

    for k, (_, kind, obj) in enumerate(todo, 1):
        if kind == "instances":
            info = call("GET", f"/instances/{obj}")
            if info and info.get("status") not in TERMINAL:
                call("POST", f"/instances/{obj}/cancel", {})
        call("DELETE", f"/{kind}/{obj}")
        gone.add(obj)
        if k % 25 == 0 or k == len(todo):
            json.dump({"deleted": sorted(gone)}, open(done_path, "w"), indent=1)
            log(f"deleted {k}/{len(todo)}")
    for what in ("app", "library", "serving", "library serving"):   # delivery/runs.json stays: it records the outputs' runs
        if found[what].get("bundle"):
            os.remove(os.path.join(a.out, STATES[what]))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
