"""Shared bits for the Path 2 scripts: logging, the role-file manifest, examples."""
import datetime
import json
import os
import sys


def log(msg):
    print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def load_manifest(roles):
    """The role-file manifest (contract: atai-anomaly-discovery-agent-data-prep/references/role-files.md)."""
    path = os.path.join(roles, "manifest.json")
    if not os.path.exists(path):
        sys.exit(f"no {path}: build the role files first (atai-anomaly-discovery-agent-data-prep)")
    m = json.load(open(path))
    if m.get("contract") != "ad-role-files/v1":
        sys.exit(f"{path} is not an ad-role-files/v1 manifest")
    return m


def role_paths(roles, manifest, role):
    return [os.path.join(roles, f["file"]) for f in manifest[role]]


def cache_path(out):
    return os.path.join(out, "uploads.json")


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def example(name, file_id, training=False):
    """A training example (every row `normal`, by a constant) or a scored example (the
    `label` column, each window paired with its last row)."""
    gt = ({"from": {"constant": "normal"}, "downsampling": "last_record"} if training
          else {"from": {"column": "label"}, "downsampling": "last_record"})
    return {"name": name, "inputs": [{"type": "file", "id": file_id, "format": "csv"}], "ground_truth": {"anomaly": gt}}


def base_values(manifest):
    """Values every request pins: the columns to read. Since the 2026-10-12 `ad` re-seed the
    blueprint also applies its own window (1024), step (256) and features (seven) unless given,
    so the scripts always pin or search those too."""
    return {"timestamp_column": manifest["timestamp_column"], "data_columns": manifest["channels"]}
