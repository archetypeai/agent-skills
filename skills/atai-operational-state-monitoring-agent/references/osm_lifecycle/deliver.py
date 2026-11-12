#!/usr/bin/env python3
"""Path 2, step 3: run the tested model over new recordings, as a customer would.

Packages the blueprint promote_and_test.py made (<out>/test_state.json), unchanged (a
package carries no values), deploys it as `batch`, and runs it over the delivery files
(no labels) in sessions: one by default, --files-per-run N for smaller batches (a failed
session then costs less). Each session is one run over its files, given as source
connectors; the platform works through a session's files one at a time.

A run's outputs don't name their inputs, so every output row is matched to its delivery
file by `finish_timestamp` (recordings must not overlap in time). Writes one predictions
CSV per delivery file to <out>/delivery/, and <out>/delivery/runs.json (package,
deployment, sessions, each file's time range) as each session starts, so a rerun collects
instead of starting over. A rerun also replaces what can't finish: a failed package is
rebuilt once, and a session that failed or ended `completed` with an output short of its
file's end is ended and dropped, and its files get a new session.

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
from atai_http import (TERMINAL, check_auth, deploy, deployment_ready, end, lifecycle_url, list_all,  # noqa: E402
                       load_dotenv, open_session, ready_package, request, result_url, root, same_deployment,
                       upload_all)
from background import add_background_flag, maybe_detach  # noqa: E402
from common import cache_path, load_manifest, log, stamp  # noqa: E402

# The platform reports a run completed while an output is still being written. Each
# output carries its own status at data.file_attributes.job_output.status (`partial`
# -> `complete`), which is what the console renders; that is the primary gate. The
# timestamp check stays as a second opinion: a run counts as completed here only once
# every file's predictions reach that file's end, which also catches an output that is
# complete but does not cover its input.
END_SLACK_MS = 60_000          # a file's last prediction lands within a window of its end
OUTPUT_WAIT_S = 600            # how long to wait for outputs after the platform says completed


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


def fetch(ref):
    """A result file, by its `ref`: a root path (`/api/files/<fil_…>/content`) or a URL."""
    req = urllib.request.Request(result_url(ref))
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read().decode("utf-8", errors="replace")


def start(args, manifest, state_path, test_state, state=None):
    """The package and batch deployment (once), then a session for every delivery file no run
    holds yet: all of them on a first run, those of dropped sessions on a rerun."""
    files = [f["file"] for f in manifest["delivery"]["files"]]
    paths = [os.path.join(args.roles, f) for f in files]
    key = test_state["blueprint"]["key"]
    if state is None:
        state = {"endpoint": root(), "blueprint": key, "runs": [],
                 "files": {f: dict(zip(("first_ms", "last_ms"), time_range(p))) for f, p in zip(files, paths)}}

    def save():
        json.dump(state, open(state_path, "w"), indent=1)

    save()
    if "deployment" not in state:
        ready_package(state, test_state["blueprint"]["id"], f"{args.name} delivery {stamp()}", save, log)
        state["deployment"] = deploy(state["package"], "batch")["id"]
        save()
        log(f"package {state['package']} from {key}, batch deployment {state['deployment']}")
        deployment_ready(state["deployment"])
    held = {f for r in state["runs"] for f in r["files"]}
    todo = [f for f in files if f not in held]
    if not todo:
        return state
    ids = upload_all([os.path.join(args.roles, f) for f in todo], args.roles, cache_path(args.out),
                     prefix=args.name, jobs=args.upload_jobs, log=log)
    per = args.files_per_run or len(todo)
    for i in range(0, len(todo), per):
        batch = todo[i:i + per]
        ses = open_session(state["deployment"], [ids[os.path.join(args.roles, f)] for f in batch])
        state["runs"].append({"session": ses["id"], "files": batch})
        save()
        log(f"  session {ses['id']} over {len(batch)} file(s) ({i // per + 1} of {-(-len(todo) // per)})")
    return state


def drop_unfinished(state, state_path):
    """End and forget every session that failed or ended `completed` with an output short of
    a file's end, so the next run opens new sessions for its files."""
    keep = []
    for r in state["runs"]:
        failed = r.get("status") in TERMINAL and r.get("status") != "completed"
        short = (r.get("error") or "").startswith("outputs incomplete")
        if not (failed or short):
            keep.append(r)
            continue
        try:
            end("sessions", r["session"])
        except RuntimeError as e:
            log(f"  could not end {r['session']}: {e}; cleanup.py retries it")
        log(f"  dropped {r['session']} ({r.get('status')}{', ' + r['error'] if r.get('error') else ''}): "
            f"a rerun opens a new session for its {len(r['files'])} file(s)")
    if len(keep) != len(state["runs"]):
        state["runs"] = keep
        json.dump(state, open(state_path, "w"), indent=1)


def artifact_status(item):
    """An output's own completion status, or None if the platform does not report one."""
    return (item.get("data", {}).get("file_attributes", {})
                .get("job_output", {}).get("status"))


def download(runs, ranges):
    """The runs' output rows, matched to delivery files: ({file: [rows]}, header, unmatched).

    Outputs the platform has not finished writing are skipped, not parsed: their rows
    would be a prefix, and a prefix is indistinguishable from a complete file once parsed."""
    per_file, header, unmatched = {}, None, 0
    for r in runs:
        for item in list_all(lifecycle_url("sessions", r["session"]) + "/results"):
            status = artifact_status(item)
            if status is not None and status != "complete":
                continue
            body = list(csv.reader(io.StringIO(fetch(item["data"]["ref"]))))
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
    return per_file, header, unmatched


def incomplete(names, state, per_file, header):
    """The files among `names` whose predictions stop more than END_SLACK_MS before the file's end."""
    if not header:
        return list(names)
    fin = header.index("finish_timestamp")
    return [n for n in names if not per_file.get(n)
            or state["files"][n]["last_ms"] - max(to_ms(row[fin]) for row in per_file[n]) > END_SLACK_MS]


def collect(args, state, state_path):
    ranges = [(v["first_ms"], v["last_ms"], n) for n, v in state["files"].items()]
    pending = {r["session"] for r in state["runs"] if r.get("status") not in TERMINAL}
    finishing = {}                 # run -> deadline, once the platform says completed
    while pending:
        for aid in sorted(pending):
            r = next(r for r in state["runs"] if r["session"] == aid)
            if aid not in finishing:
                a = request("GET", lifecycle_url("sessions", aid))
                if a["status"] not in TERMINAL:
                    continue
                if a["status"] != "completed":
                    pending.discard(aid)
                    r["status"], r["error"] = a["status"], a.get("error")
                    log(f"  {aid} {a['status']}" + (f" ({a.get('error')})" if a.get("error") else ""))
                    continue
                finishing[aid] = time.time() + OUTPUT_WAIT_S
            per_file, header, _ = download([r], ranges)
            short = incomplete(r["files"], state, per_file, header)
            n = len(r["files"])
            if not short:
                pending.discard(aid)
                r["status"], r["error"] = "completed", None
                log(f"  {aid} completed: all {n} outputs complete")
            elif time.time() > finishing[aid]:
                pending.discard(aid)
                r["status"], r["error"] = "completed", f"outputs incomplete: {', '.join(short)}"
                log(f"  {aid} WARNING platform completed {OUTPUT_WAIT_S // 60} min ago but {len(short)} of {n} "
                    f"outputs still end early ({', '.join(os.path.basename(x) for x in short)}); "
                    f"a rerun downloads them again, then replaces the session")
            else:
                log(f"  {aid} platform: completed; outputs {n - len(short)} of {n} complete "
                    f"({', '.join(os.path.basename(x) for x in short)} still being written)")
        json.dump(state, open(state_path, "w"), indent=1)
        if pending:
            time.sleep(15 if pending <= set(finishing) else 60)

    # a "failed" status can hide a successful run: collect whatever every run's /results holds
    per_file, header, unmatched = download(state["runs"], ranges)
    done = [r for r in state["runs"] if r.get("status") == "completed"]
    short = incomplete([n for r in done for n in r["files"]], state, per_file, header)
    for r in done:
        if not set(r["files"]) & set(short) and (r.get("error") or "").startswith("outputs incomplete"):
            r["error"] = None      # a --resume found them complete
    json.dump(state, open(state_path, "w"), indent=1)
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
        f"{unmatched:,} rows matched no file; incomplete outputs: "
        f"{[os.path.basename(x) for x in short] or 'none'} -> {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="osm")
    ap.add_argument("--files-per-run", type=int, default=0, help="files per run (default 0: all in one run)")
    ap.add_argument("--upload-jobs", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="collect the runs in runs.json")
    add_background_flag(ap, default_log="{out}/deliver.log")
    args = ap.parse_args()
    maybe_detach(args)
    load_dotenv()
    check_auth(log)
    manifest = load_manifest(args.roles)
    os.makedirs(os.path.join(args.out, "delivery"), exist_ok=True)
    state_path = os.path.join(args.out, "delivery", "runs.json")
    test_path = os.path.join(args.out, "test_state.json")
    if not os.path.exists(test_path):
        sys.exit("no promoted blueprint yet: run promote_and_test.py first")
    test_state = json.load(open(test_path))
    if not same_deployment(test_state):
        sys.exit(f"the blueprint in {test_path} is on {test_state['endpoint']}, "
                 f"a different deployment from {root()}: run promote_and_test.py here first")
    state = None
    if args.resume or os.path.exists(state_path):
        if not args.resume:
            log(f"{state_path} exists: collecting those runs (move it aside to deliver again)")
        state = json.load(open(state_path))
        if not same_deployment(state):
            sys.exit(f"{state_path} is from {state['endpoint']}, a different deployment from {root()}: "
                     f"move {os.path.dirname(state_path)}/ aside to deliver here, or set .env to that deployment")
        if state.get("blueprint") != test_state["blueprint"]["key"]:
            sys.exit(f"{state_path} is for {state.get('blueprint')}: move it aside to deliver "
                     f"{test_state['blueprint']['key']}")
        collect(args, state, state_path)            # download what the saved runs have, first
        drop_unfinished(state, state_path)
    before = [r["session"] for r in (state or {}).get("runs", [])]
    state = start(args, manifest, state_path, test_state, state)
    if [r["session"] for r in state["runs"]] != before:
        collect(args, state, state_path)
    log(f"next: python score.py --roles {args.roles} --out {args.out}")


if __name__ == "__main__":
    main()
