"""Shared bits for the Path 2 scripts: logging, the role-file manifest, the upload cache."""
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
