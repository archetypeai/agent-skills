#!/usr/bin/env python3
"""Path 2, cleanup: delete the sessions, deployments and packages deliver.py, serve.py,
library_embeddings.py and a serving app left on the platform.

    python cleanup.py --out out           # list what would be deleted, and what stays
    python cleanup.py --out out --yes     # delete it

From the packages recorded in <out>/:
  app_serving_state.json   a serving app's, if its bridge was killed before it could delete
                           them (the bridge pattern in SKILL.md writes this file).
  library_state.json       library_embeddings.py's (its embeddings blueprint is kept for a rerun).
  serve_state.json         serve.py's, if it was killed before it could delete them. A
                           serving deployment holds a serving worker until it is stopped.
  delivery/runs.json       deliver.py's.
For each package, the platform's own listings (GET /agents/deployments?package_id=, then
GET /agents/sessions?deployment_id=) give its deployments and their sessions, so ones a
script never recorded, or a rerun replaced, are found too.

Sessions go first, then deployments, then the package: each holds the next. A session still
going is cancelled, and a deployment still up stopped, before it is deleted; an id already
gone (404) counts as done. Progress goes to <out>/cleanup_state.json, so a rerun picks up
where it stopped.

What stays: the project, its copy of the blueprint, the promoted blueprints, and the
optimizations and eval (records your results cite). After a cleanup, deliver.py --resume
can't collect again: its sessions are gone.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import check_auth, end, lifecycle_url, list_all, load_dotenv, request, root, same_deployment  # noqa: E402
from common import log  # noqa: E402

STATES = {"app": "app_serving_state.json", "library": "library_state.json", "serving": "serve_state.json",
          "delivery": os.path.join("delivery", "runs.json")}
KINDS = ("sessions", "deployments", "packages")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def targets(packages, deployments_of, sessions_of):
    """(what, kind, id) to delete, each session before its deployment and each deployment
    before its package. `packages` is [(what, package id)]; deployments_of / sessions_of list
    a package's deployments and a deployment's sessions."""
    out = []
    for what, pkg in packages:
        dpls = deployments_of(pkg)
        out += [(what, "sessions", s) for d in dpls for s in sessions_of(d)]
        out += [(what, "deployments", d) for d in dpls] + [(what, "packages", pkg)]
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
        if st.get("bundle"):
            sys.exit(f"{STATES[what]} holds bundle ids from before packages: delete them with the bundle-era cleanup.py")

    def exists(pkg):
        try:
            return bool(request("GET", lifecycle_url("packages", pkg)))
        except RuntimeError as e:
            if "(404)" in str(e):
                return False
            raise
    packages = [(what, st["package"]) for what, st in found.items() if st.get("package") and exists(st["package"])]
    ids = lambda kind, **q: [r["id"] for r in list_all(lifecycle_url(kind) + "?" + "&".join(f"{k}={v}" for k, v in q.items()))]  # noqa: E731
    todo_all = targets(packages, lambda p: ids("deployments", package_id=p), lambda d: ids("sessions", deployment_id=d))
    done_path = os.path.join(a.out, "cleanup_state.json")
    gone = set(load(done_path).get("deleted", []))
    todo = [t for t in todo_all if t[2] not in gone]
    for what in STATES:
        mine = [t for t in todo if t[0] == what]
        print(f"{what}: " + ", ".join(f"{sum(t[1] == k for t in mine)} {k}" for k in KINDS) + " to delete")
    print(f"already deleted: {len(gone & {t[2] for t in todo_all})}")
    test = load(os.path.join(a.out, "test_state.json"))
    if test.get("blueprint"):
        print(f"stays: blueprint {test['blueprint']['key']}, the optimizations and eval, "
              f"and the project with its copy of the blueprint")
    if not a.yes:
        print("\nnothing deleted: rerun with --yes")
        return 0

    for k, (_, kind, obj) in enumerate(todo, 1):
        end(kind, obj)
        gone.add(obj)
        if k % 25 == 0 or k == len(todo):
            json.dump({"deleted": sorted(gone)}, open(done_path, "w"), indent=1)
            log(f"deleted {k}/{len(todo)}")
    for what in ("app", "serving"):       # delivery/runs.json stays: it records the outputs' sessions
        if found[what].get("package"):
            os.remove(os.path.join(a.out, STATES[what]))
    lib = found["library"]
    if lib.get("package"):                # keep the embeddings blueprint for a rerun; forget what was deleted
        json.dump({k: v for k, v in lib.items() if k not in ("package", "deployment", "runs")},
                  open(os.path.join(a.out, STATES["library"]), "w"), indent=1)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
