#!/usr/bin/env python3
"""Path 2, step 3: run the tested model over new recordings, as a customer would.

Creates one bundle from the blueprint promote_and_test.py made (<out>/test_state.json),
unchanged, and runs it over the delivery files (no labels) in batches: one run by
default, --files-per-run N for smaller batches (a failed run then costs less). Each run
is one agent instance; the platform works through a run's files one at a time, and
through several runs one at a time too.

A run's outputs don't name their inputs, so every output row is matched to its delivery
file by `finish_timestamp` (recordings must not overlap in time). Writes one predictions
CSV per delivery file to <out>/delivery/, and <out>/delivery/runs.json (bundle, runs,
each file's time range) as each run starts, so a rerun collects instead of starting over.

    python deliver.py --roles roles --background
    python deliver.py --roles roles --resume          # collect the runs in runs.json
"""
import argparse
import csv
import datetime
import io
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import TERMINAL, agents, list_all, load_dotenv, request, root, upload_all  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from common import cache_path, load_manifest, log, stamp  # noqa: E402


def to_ms(value):
    """A timestamp as epoch ms: epoch seconds (`1694181225.45`), epoch ms, or ISO 8601."""
    value = str(value).strip()
    try:
        v = float(value)
        return round(v if v > 1e11 else v * 1000)
    except ValueError:
        return round(datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def time_range(path):
    """First and last timestamp of a role file, in epoch ms, without reading it all."""
    with open(path, "rb") as f:
        f.readline()
        first = f.readline().split(b",", 1)[0]
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 4096))
        last = f.read().strip().split(b"\n")[-1].split(b",", 1)[0]
    return to_ms(first.decode()), to_ms(last.decode())


def match_rows(rows, finish_col, ranges):
    """{file: [rows]} by each row's finish time, and how many rows matched no file."""
    ranges = sorted(ranges)
    out, unmatched = {}, 0
    for row in rows:
        t = to_ms(row[finish_col])
        name = next((n for a, b, n in ranges if a <= t <= b), None)
        if name is None:
            unmatched += 1
        else:
            out.setdefault(name, []).append(row)
    return out, unmatched


def fetch(filename):
    req = urllib.request.Request(f"{root()}/v0.5/files/download/{filename}")
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read().decode("utf-8", errors="replace")


def start(args, manifest, state_path, test_state):
    files = [f["file"] for f in manifest["delivery"]["files"]]
    paths = [os.path.join(args.roles, f) for f in files]
    ids = upload_all(paths, args.roles, cache_path(args.out), prefix=args.name, jobs=args.upload_jobs, log=log)
    key = test_state["blueprint"]["key"]
    bundle = request("POST", f"{agents()}/bundles", body={
        "blueprint": key, "name": f"{args.name} delivery {stamp()}",
        "description": "the tested model, run over the delivery recordings"})
    log(f"bundle {bundle['id']} from {key}")
    per = args.files_per_run or len(files)
    state = {"bundle": bundle["id"], "blueprint": key, "runs": [],
             "files": {f: dict(zip(("first_ms", "last_ms"), time_range(p))) for f, p in zip(files, paths)}}
    for i in range(0, len(files), per):
        batch = files[i:i + per]
        run = request("POST", f"{agents()}/bundles/{bundle['id']}/run", body={
            "connectors": {"source": [{"type": "file", "id": ids[os.path.join(args.roles, f)]} for f in batch]}})
        state["runs"].append({"agent": run["id"], "files": batch})
        json.dump(state, open(state_path, "w"), indent=1)
        log(f"  run {run['id']} over {len(batch)} file(s) ({len(state['runs'])} of {-(-len(files) // per)})")
    return state


def collect(args, state, state_path):
    pending = {r["agent"] for r in state["runs"] if r.get("status") not in TERMINAL}
    while pending:
        for aid in sorted(pending):
            a = request("GET", f"{agents()}/instances/{aid}")
            if a["status"] in TERMINAL:
                pending.discard(aid)
                r = next(r for r in state["runs"] if r["agent"] == aid)
                r["status"], r["error"] = a["status"], a.get("error")
                log(f"  {aid} {a['status']}" + (f" ({a.get('error')})" if a.get("error") else ""))
        json.dump(state, open(state_path, "w"), indent=1)
        if pending:
            time.sleep(60)

    ranges = [(v["first_ms"], v["last_ms"], n) for n, v in state["files"].items()]
    per_file, header, unmatched = {}, None, 0
    for r in state["runs"]:
        # a "failed" status can hide a successful run: collect whatever /results holds
        for item in list_all(f"{agents()}/instances/{r['agent']}/results"):
            body = list(csv.reader(io.StringIO(fetch(item["data"]["filename"]))))
            if not body:
                continue
            if header is None:
                header = body[0]
            elif body[0] != header:
                sys.exit(f"outputs disagree on columns: {body[0]} vs {header}")
            got, miss = match_rows(body[1:], header.index("finish_timestamp"), ranges)
            unmatched += miss
            for name, rows in got.items():
                per_file.setdefault(name, []).extend(rows)
    out_dir = os.path.join(args.out, "delivery")
    fin = header.index("finish_timestamp") if header else 0
    for name, rows in per_file.items():
        rows.sort(key=lambda row: to_ms(row[fin]))
        with open(os.path.join(out_dir, os.path.basename(name)), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    inv = header.index("invalid") if header and "invalid" in header else None
    n = sum(map(len, per_file.values()))
    bad = sum(1 for v in per_file.values() for row in v if inv is not None and row[inv].lower() == "true")
    log(f"{len(per_file)} of {len(state['files'])} files have predictions: {n:,} windows ({bad:,} invalid); "
        f"{unmatched:,} rows matched no file -> {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="osm")
    ap.add_argument("--files-per-run", type=int, default=0, help="files per run (default 0: all in one run)")
    ap.add_argument("--upload-jobs", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="collect the runs in runs.json")
    add_background_flag(ap, default_log="out/deliver.log")
    args = ap.parse_args()
    maybe_detach(args)
    load_dotenv()
    manifest = load_manifest(args.roles)
    os.makedirs(os.path.join(args.out, "delivery"), exist_ok=True)
    state_path = os.path.join(args.out, "delivery", "runs.json")
    if args.resume or os.path.exists(state_path):
        if not args.resume:
            log(f"{state_path} exists: collecting those runs (move it aside to deliver again)")
        state = json.load(open(state_path))
    else:
        test_state = os.path.join(args.out, "test_state.json")
        if not os.path.exists(test_state):
            sys.exit("no promoted blueprint yet: run promote_and_test.py first")
        state = start(args, manifest, state_path, json.load(open(test_state)))
    collect(args, state, state_path)
    log(f"next: python score.py --roles {args.roles} --out {args.out}")


if __name__ == "__main__":
    main()
