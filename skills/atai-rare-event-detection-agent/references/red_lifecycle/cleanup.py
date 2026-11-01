#!/usr/bin/env python3
"""Path 2, cleanup: delete the agents and bundles deliver.py, serve.py and library_embeddings.py left.

    python cleanup.py --out out           # list what would be deleted, and what stays
    python cleanup.py --out out --yes     # delete it

From the ids in <out>/:
  serve_state.json     a serving agent and its bundle, if serve.py was killed before it could
                       delete them. A running serving agent holds a serving worker until it
                       is cancelled, or idle for 20 minutes.
  deliver_state.json   the delivery agents and their bundle. The state holds each file's last
                       attempt; the bundle's own listing (GET /agents/instances?bundle_id=)
                       also finds the attempts a rerun replaced.
  library_state.json   library_embeddings.py's batch runs and their bundle; the bundle's own
                       listing also finds a run the state dropped (failed, or replaced).
  library_serving_state.json  a serving agent library_embeddings.py --serving started, if it
                       was killed before it could delete it.

Agents go first (a bundle that still has an agent answers 409), a running one is cancelled
before it is deleted, and an id already gone (404) counts as done. Progress goes to
<out>/cleanup_state.json, so a rerun picks up where it stopped.

What stays: the promoted blueprint (deleting a blueprint takes an org admin's key; serve.py
serves it as it is, with no copy), and the optimizations and eval (records your results cite). After a cleanup,
rerunning deliver.py needs <out>/deliver_state.json moved aside: its bundle is gone.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import TERMINAL, agents, check_auth, list_all, load_dotenv, request, root, same_deployment  # noqa: E402
from common import log  # noqa: E402


def call(method, path, body=None):
    """The response, or None when the object is already gone."""
    try:
        return request(method, f"{agents()}{path}", body)
    except RuntimeError as e:
        if "(404)" in str(e):
            return None
        raise


def load(out, name):
    p = os.path.join(out, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def targets(serve, deliver, bundle_runs, library=None, library_runs=frozenset(), library_serving=None):
    """(what, kind, id) to delete, each agent before its bundle."""
    library = library or {}
    out = []
    for what, st, extra in (("serving", serve, set()), ("delivery", deliver, bundle_runs),
                            ("library", {"bundle": library.get("bundle"), "agents": library.get("runs", {})},
                             set(library_runs)),
                            ("library serving", library_serving or {}, set())):
        if not st.get("bundle"):
            continue
        ids = {a for a in ([st.get("agent")] + list(st.get("agents", {}).values())) if a} | extra
        out += [(what, "instances", a) for a in sorted(ids)] + [(what, "bundles", st["bundle"])]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out")
    ap.add_argument("--yes", action="store_true", help="delete; without it, only list")
    a = ap.parse_args(argv)
    load_dotenv()
    check_auth(log)
    serve, deliver, test = load(a.out, "serve_state"), load(a.out, "deliver_state"), load(a.out, "test_state")
    library, library_serving = load(a.out, "library_state"), load(a.out, "library_serving_state")
    for st in (serve, deliver, library, library_serving):
        if not same_deployment(st):
            sys.exit(f"ids in {a.out} are from {st['endpoint']}, a different deployment from {root()}: "
                     f"set .env to that deployment")
    runs = set()
    if deliver.get("bundle"):
        runs = {r["id"] for r in list_all(f"{agents()}/instances?bundle_id={deliver['bundle']}")}
    library_runs = set()
    if library.get("bundle"):
        library_runs = {r["id"] for r in list_all(f"{agents()}/instances?bundle_id={library['bundle']}")}
    todo_all = targets(serve, deliver, runs, library, library_runs, library_serving)
    done_path = os.path.join(a.out, "cleanup_state.json")
    gone = set(load(a.out, "cleanup_state").get("deleted", []))
    todo = [t for t in todo_all if t[2] not in gone]
    for what in ("serving", "delivery", "library", "library serving"):
        mine = [t for t in todo if t[0] == what]
        print(f"{what}: {sum(t[1] == 'instances' for t in mine)} agent(s), "
              f"{sum(t[1] == 'bundles' for t in mine)} bundle(s) to delete")
    print(f"already deleted: {len(gone & {t[2] for t in todo_all})}")
    if test.get("blueprint"):
        key = test["blueprint"]["key"]
        print(f"stays: blueprint {key} (an org admin can delete it), "
              f"and the optimizations and eval {test.get('eval', '')}".rstrip())
    if not a.yes:
        print("\nnothing deleted: rerun with --yes")
        return 0

    for k, (what, kind, obj) in enumerate(todo, 1):
        if kind == "instances":
            info = call("GET", f"/instances/{obj}")
            if info and info.get("status") not in TERMINAL:
                call("POST", f"/instances/{obj}/cancel", {})
        call("DELETE", f"/{kind}/{obj}")
        gone.add(obj)
        if k % 25 == 0 or k == len(todo):
            json.dump({"deleted": sorted(gone)}, open(done_path, "w"), indent=1)
            log(f"deleted {k}/{len(todo)}")
    if serve.get("bundle"):
        os.remove(os.path.join(a.out, "serve_state.json"))
    if library.get("bundle"):
        os.remove(os.path.join(a.out, "library_state.json"))
    if library_serving.get("bundle"):
        os.remove(os.path.join(a.out, "library_serving_state.json"))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
