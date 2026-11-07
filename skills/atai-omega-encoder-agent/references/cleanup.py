#!/usr/bin/env python3
"""Delete everything embed_batch.py and serve.py made on the platform.

    python cleanup.py                 # list what would be deleted
    python cleanup.py --yes           # delete it

Works from the project, not just the state file, so it also finds what a killed run left
behind: each deployment's sessions, then the deployments, the packages, the blueprint
copies and the project, in that order (a deployment must go before its package and a
package before its blueprint, or the delete is a 409). A serving deployment still up is
named first: it holds a worker until it is ended. Uploaded files are kept.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atai_api as A  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="Omega Encoder agent")
    ap.add_argument("--out", default="out")
    ap.add_argument("--yes", action="store_true", help="actually delete")
    a = ap.parse_args()
    A.load_dotenv()
    projects = [p["id"] for p in A.request("GET", f"{A.api_base()}/projects?limit=1000")["data"] if p["name"] == a.project]
    if not projects:
        print("nothing left")
        return
    for prj in projects:
        packages = A.request("GET", f"{A.packages_url()}?project_id={prj}&limit=1000")["data"]
        deployments = A.request("GET", f"{A.deployments_url()}?project_id={prj}&limit=1000")["data"]
        sessions = [s for d in deployments
                    for s in A.request("GET", f"{A.sessions_url()}?deployment_id={d['id']}&limit=1000")["data"]]
        blueprints = [b for b in A.request("GET", f"{A.agents_base()}/blueprints?project_id={prj}&limit=1000")["data"]
                      if b.get("project_id") == prj]
        print(f"project {prj}: {len(sessions)} sessions, {len(deployments)} deployments, "
              f"{len(packages)} packages, {len(blueprints)} blueprints")
        for d in deployments:
            if d["kind"] == "serving" and d["status"] not in ("stopped", "failed"):
                print(f"  serving deployment {d['id']} is {d['status']}: it holds a worker until it is ended")
        if not a.yes:
            continue
        failed = []

        def attempt(label, fn):
            try:
                fn()
            except SystemExit as e:              # atai_api exits on an HTTP error
                failed.append(f"{label}: {str(e)[:200]}")

        for s in sessions:
            attempt(f"session {s['id']}", lambda s=s: A.end_session(s["id"]))
        for d in deployments:
            attempt(f"deployment {d['id']}", lambda d=d: A.end_deployment(d["id"]))
        for p in packages:
            attempt(f"package {p['id']}", lambda p=p: A.request("DELETE", A.packages_url(p["id"]), raw=True))
        for b in blueprints:
            attempt(f"blueprint {b['blueprint_key']}",
                    lambda b=b: A.request("DELETE", f"{A.agents_base()}/blueprints/{b['id']}", raw=True))
        attempt(f"project {prj}", lambda: A.request("DELETE", f"{A.api_base()}/projects/{prj}", raw=True))
        if failed:
            print(f"project {prj}: {len(failed)} objects could not be deleted:")
            for f in failed:
                print(f"  {f}")
        else:
            print(f"deleted project {prj} and everything in it")
    state = os.path.join(a.out, "embed", "state.json")
    if a.yes and os.path.exists(state):
        st = json.load(open(state))
        json.dump({k: st[k] for k in ("identity", "uploads") if k in st}, open(state, "w"), indent=1)
    if not a.yes:
        print("dry run; pass --yes to delete")


if __name__ == "__main__":
    main()
