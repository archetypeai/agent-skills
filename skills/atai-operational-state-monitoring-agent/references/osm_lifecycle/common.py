"""Shared bits for the Path 2 scripts: logging, the role-file manifest, the upload cache, the project."""
import datetime
import json
import os
import sys


def log(msg):
    print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def load_manifest(roles):
    """The role-file manifest (contract: atai-operational-state-monitoring-agent-data-prep/references/role-files.md)."""
    path = os.path.join(roles, "manifest.json")
    if not os.path.exists(path):
        sys.exit(f"no {path}: build the role files first (atai-operational-state-monitoring-agent-data-prep)")
    m = json.load(open(path))
    if not m.get("states"):
        sys.exit(f"{path} has no `states`: not an osm-role-files manifest")
    return m


def role_paths(roles, manifest, role, subset=None):
    """Absolute paths of a role's files, optionally only those of the given recordings."""
    return [os.path.join(roles, f["file"]) for f in manifest[role]["files"]
            if subset is None or f["recording"] in subset]


def cache_path(out):
    return os.path.join(out, "uploads.json")


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def example(name, file_id, state=None):
    """A training example (constant ground truth: the file's state) or a scored example
    (the `label` column, each window paired with its last row)."""
    gt = ({"from": {"constant": state}} if state
          else {"from": {"column": "label"}, "downsampling": "last_record"})
    return {"name": name, "inputs": [{"type": "file", "id": file_id, "format": "csv"}], "ground_truth": {"state": gt}}


def base_blueprint(out, project_name, source, base_key, log=print):
    """The project and its copy of `source` (the canonical `osm` by default), which the optimizations search.

    Everything after it lands in the project: an optimization in its blueprint's project, and
    so do the blueprints it promotes and their packages, deployments and sessions. A deployment
    needs one, and a canonical blueprint is outside any, while POST /optimizations takes no
    project_id. Copying needs only the right to modify the project, not the admin role.
    An existing project of that name, and an existing copy under `base_key` in it, are reused.

    Writes <out>/project_state.json; returns {project, id, key}."""
    from atai_http import agents, project_url, request, root, same_deployment
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "project_state.json")
    state = json.load(open(path)) if os.path.exists(path) else {}
    if not same_deployment(state):
        sys.exit(f"{path} is from {state['endpoint']}, a different deployment from {root()}: move it aside")
    state["endpoint"] = root()
    if "project" not in state:
        found = [p for p in request("GET", f"{project_url()}?limit=1000")["data"] if p["name"] == project_name]
        state["project"] = found[0]["id"] if found else request("POST", project_url(), {"name": project_name})["id"]
        json.dump(state, open(path, "w"), indent=1)
        log(f"project {state['project']} ({project_name})")
    if "base" not in state:
        found = request("GET", f"{agents()}/blueprints?project_id={state['project']}&key={base_key}")["data"]
        if found:
            bp = found[0]
            log(f"reusing {base_key} in {state['project']} ({bp['id']})")
        else:
            bp = request("POST", f"{agents()}/blueprints/{source}/copy", {"project_id": state["project"], "blueprint_key": base_key})
            log(f"copied {source} into {state['project']} as {base_key} ({bp['id']})")
        state["base"] = {"id": bp["id"], "key": bp["blueprint_key"]}
        json.dump(state, open(path, "w"), indent=1)
    return {"project": state["project"], **state["base"]}
