#!/usr/bin/env python3
"""Delete what discover.py made on the platform. A dry run unless --yes.

    python cleanup.py --out out           # what would be deleted
    python cleanup.py --out out --yes     # delete it

From the ids in <out>/discover_state.json, in the order each one frees the next: sessions
(cancelled first if still going; also found from the deployment, so ones never recorded go
too), the deployment (stopped first), the package, the copy of od, then the project if
nothing else is left in it. Uploaded files stay. An id already gone counts as done, so a
rerun finishes an interrupted cleanup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_api  # noqa: E402


def call(method: str, path: str, body=None) -> bool:
    """True when the call worked or the object is already gone (404)."""
    try:
        od_api.request(method, path, body)
        return True
    except SystemExit as e:
        if " -> 404" in str(e):
            return True
        print(f"    {e}")
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--yes", action="store_true", help="delete; without it, only list")
    a = ap.parse_args(argv)
    path = a.out / "discover_state.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: nothing recorded to delete")
    st = json.loads(path.read_text())
    od_api.check_key()
    if st.get("endpoint") and st["endpoint"] != od_api.root():
        raise SystemExit(f"{path} is for {st['endpoint']}, not {od_api.root()}")
    sessions = {r["session"] for r in st.get("runs", {}).values() if r.get("session")}
    if st.get("deployment"):
        try:
            sessions |= {s["id"] for s in od_api.request(
                "GET", f"/agents/sessions?deployment_id={st['deployment']}&limit=1000")["data"]}
        except SystemExit:
            pass
    plan = [*(("session", s) for s in sorted(sessions)), ("deployment", st.get("deployment")),
            ("package", st.get("package")), ("blueprint", st.get("blueprint")), ("project", st.get("project"))]
    plan = [(k, v) for k, v in plan if v]
    for kind, oid in plan:
        print(f"  {kind:<10} {oid}")
    if not a.yes:
        print(f"\ndry run: {len(plan)} objects; rerun with --yes to delete them")
        return 0
    ok = True
    for kind, oid in plan:
        if kind == "session":
            try:
                status = od_api.request("GET", f"/agents/sessions/{oid}").get("status")
            except SystemExit:
                status = None
            if status in {"pending", "running", "paused"}:
                call("POST", f"/agents/sessions/{oid}/cancel", {})
            done = call("DELETE", f"/agents/sessions/{oid}")
        elif kind == "deployment":
            call("POST", f"/agents/deployments/{oid}/stop", {})
            done = call("DELETE", f"/agents/deployments/{oid}")
        elif kind == "package":
            done = call("DELETE", f"/agents/packages/{oid}")
        elif kind == "blueprint":
            done = call("DELETE", f"/agents/blueprints/{oid}")
        else:
            left = od_api.request("GET", f"/agents/blueprints?project_id={oid}&limit=10")["data"]
            if left:
                print(f"  kept    project {oid}: it still holds {len(left)} other blueprint(s)")
                continue
            done = call("DELETE", f"/projects/{oid}")
        print(f"  {'deleted' if done else 'FAILED '} {kind} {oid}")
        ok &= done
    if ok:
        path.rename(path.with_suffix(".deleted.json"))
        print(f"\nall deleted; {path} moved aside")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
