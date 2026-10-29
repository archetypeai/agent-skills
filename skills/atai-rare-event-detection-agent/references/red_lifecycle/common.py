"""Shared bits for the Path 2 scripts: logging, the role-file manifest, examples, values."""
import datetime
import json
import os
import sys


def log(msg):
    print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def load_manifest(roles):
    """The role-file manifest (contract: atai-rare-event-detection-agent-data-prep/references/role-files.md)."""
    path = os.path.join(roles, "manifest.json")
    if not os.path.exists(path):
        sys.exit(f"no {path}: build the role files first (atai-rare-event-detection-agent-data-prep)")
    m = json.load(open(path))
    if m.get("contract") != "red-role-files/v1":
        sys.exit(f"{path} is not a red-role-files/v1 manifest")
    return m


def role_paths(roles, manifest, role):
    return [os.path.join(roles, f["file"]) for f in manifest[role]]


def cache_path(out):
    return os.path.join(out, "uploads.json")


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def example(name, file_id, cls=None):
    """A training example (every row the file's one class, by a constant) or a scored one
    (the `label` column). The platform scores each row once: a window answers for the rows
    since the previous window ended, so at step 1 its label is its last row's."""
    gt = {"from": {"constant": cls}} if cls else {"from": {"column": "label"}, "downsampling": "last_record"}
    return {"name": name, "inputs": [{"type": "file", "id": file_id, "format": "csv"}], "ground_truth": {"event": gt}}


def base_values(manifest):
    """Values every request pins. `events` and `background` are required by the `red`
    blueprint; since its 2026-10-12 re-seed it also applies its own window (1024) and step
    (256) unless given, so the scripts always pin or search those too."""
    return {"timestamp_column": manifest["timestamp_column"], "data_columns": manifest["channels"],
            "events": manifest["classes"], "background": manifest["background"]}
