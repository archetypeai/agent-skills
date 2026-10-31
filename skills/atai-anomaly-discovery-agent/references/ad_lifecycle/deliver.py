#!/usr/bin/env python3
"""Path 2, step 3: deliver the promoted blueprint in batch, one session per delivery file.

POST /agents/packages packages the promoted blueprint (its encoder and fitted detector
resolved once); POST /agents/deployments deploys it as `batch`; then POST /agents/sessions
with the file as a source connector runs it over each delivery file, --parallel at a time.
A package carries no values: it runs at the trial's own window, step and threshold, with
`output_score` on as optimize.py pinned it. There is no step-1 delivery: choose the step at
Optimize. score.py rescores at any threshold offline, from the scores.

--max-windows-per-run N splits a longer file into sessions of at most N windows, for a
deployment that caps a run's time (the free trial: 2 hours). Each run's rows overlap the
next by window - step, so every window lies whole in one run; the runs' outputs are joined
into the file's, the same windows as one run gives. Off by default: a run per file.

Each output is downloaded again every 15 s until its last window reaches the input's last
timestamp: a session can report `completed` while its output is still being written. A
network outage is waited out (every 60 s, up to an hour), not counted as a failed file.

Writes <out>/delivery/<file>.csv (split runs' inputs and outputs in <out>/delivery-runs/)
and <out>/deliver_state.json (resume: finished outputs are skipped, running sessions are
polled, not started again; a file whose session failed or was cancelled without a complete
output gets a new session).

    python deliver.py --roles roles --out out --name ad-sample
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai_http import (TERMINAL, check_auth, create_package, deploy, lifecycle_url, load_dotenv, open_session,  # noqa: E402
                       request, result_url, root, same_deployment, upload_file)
from background import add_background_flag, maybe_detach  # noqa: E402
from common import load_manifest, log, stamp  # noqa: E402

NETWORK_WAIT_S, NETWORK_RETRIES = 60, 60
lock = threading.Lock()


def last_ts(path, col):
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 4096))
        line = fh.read().decode().strip().splitlines()[-1]
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    return float(line.split(",")[header.index(col)])


def pieces(rows, window, step, max_windows):
    """[(first row, end row)] of each run of a file. A file of more than max_windows windows
    is split into runs of at most that many: each starts on a multiple of the step and
    overlaps the next by window - step rows, so every window lies whole in one run and the
    runs' outputs, joined, are the file's. One run when max_windows is 0 or the file fits."""
    n = (rows - window) // step + 1
    if not max_windows or n <= max_windows:
        return [(0, rows)]
    return [(k * step, min(rows, (k + max_windows - 1) * step + window)) for k in range(0, n, max_windows)]


def join(parts, dst, window, step):
    """Concatenate the runs' outputs into dst; exit unless each run gave its windows, in order."""
    header, rows = None, []
    for (first, end), path in parts:
        with open(path) as fh:
            r = csv.reader(fh)
            header = next(r)
            got = list(r)
        if len(got) != (end - first - window) // step + 1:
            sys.exit(f"{path}: {len(got)} windows for rows {first}-{end}, expected {(end - first - window) // step + 1}")
        rows += got
    fin = [float(x[header.index("finish_timestamp")]) for x in rows]
    if any(b <= a for a, b in zip(fin, fin[1:])):
        sys.exit(f"{dst}: the joined windows are not in time order")
    with open(dst + ".part", "w", newline="") as fh:
        csv.writer(fh).writerows([header] + rows)
    os.replace(dst + ".part", dst)


def covered(out_path, source_end):
    """True once the output's last window ends within ~1.5 windows of the input's end."""
    with open(out_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return False
    end = max(float(r["finish_timestamp"]) for r in rows)
    win = max(float(r["finish_timestamp"]) - float(r["start_timestamp"]) for r in rows[:64])
    return source_end - end <= 1.5 * win + 1e-6


def poll(session_id, timeout_s=3 * 3600):
    """The session's status, once terminal. A session reaches a terminal status and carries
    `error`; its /events say what happened (opened, dispatched as a job, completed)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = request("GET", lifecycle_url("sessions", session_id)).get("status")
        if st in TERMINAL:
            return st
        time.sleep(20)
    return "timeout"


def artifact_status(item):
    """An output's own completion status, or None if the platform does not report one."""
    return (item.get("data", {}).get("file_attributes", {})
                .get("job_output", {}).get("status"))


def download(session_id, dst, source, ts_col):
    """The run's output, once it covers the input; None after 10 min of short downloads.

    The platform reports a run completed while its output is still being written; the
    output's own status (`partial` -> `complete`) is the primary gate, the coverage
    check a second opinion."""
    end = last_ts(source, ts_col)
    for _ in range(40):
        res = request("GET", lifecycle_url("sessions", session_id) + "/results").get("data", [])
        if res and artifact_status(res[0]) in (None, "complete"):
            import urllib.request
            req = urllib.request.Request(result_url(res[0]["data"]["ref"]))
            req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
            with urllib.request.urlopen(req, timeout=300) as resp, open(dst + ".part", "wb") as fh:
                fh.write(resp.read())
            if covered(dst + ".part", end):
                os.replace(dst + ".part", dst)
                return dst
        time.sleep(15)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--name", default="ad")
    ap.add_argument("--max-windows-per-run", type=int, default=0, metavar="N",
                    help="split a longer file into runs of at most N windows (default 0: a run per file)")
    ap.add_argument("--parallel", type=int, default=3)
    add_background_flag(ap, "{out}/deliver.log")
    a = ap.parse_args(argv)
    maybe_detach(a)
    load_dotenv()
    check_auth(log)
    m = load_manifest(a.roles)
    test = json.load(open(os.path.join(a.out, "test_state.json")))
    if not same_deployment(test):
        sys.exit(f"the blueprint in {a.out}/test_state.json is on {test['endpoint']}, "
                 f"a different deployment from {root()}: run promote_and_test.py here first")
    key, bp_id = test["blueprint"]["key"], test["blueprint"]["id"]
    sf = os.path.join(a.out, "deliver_state.json")
    state = json.load(open(sf)) if os.path.exists(sf) else {}
    if state.get("blueprint", key) != key:
        sys.exit(f"{sf} is for {state['blueprint']}: move it aside to deliver {key}")
    if not same_deployment(state):
        sys.exit(f"{sf} is from {state['endpoint']}, a different deployment from {root()}: "
                 f"move it aside to deliver here, or set .env to that deployment")
    state["endpoint"] = root()
    state["blueprint"] = key

    def save():
        with lock:
            json.dump(state, open(sf, "w"), indent=1)

    if "bundle" in state:
        sys.exit(f"{sf} is from a bundle delivery: move it aside to deliver through a package")
    if "package" not in state:
        state["package"] = create_package(bp_id, f"{a.name} delivery")["id"]
        save()
        log(f"package {state['package']} on {key}")
    if "deployment" not in state:
        state["deployment"] = deploy(state["package"], "batch")["id"]
        save()
        log(f"batch deployment {state['deployment']}")
    out_dir = os.path.join(a.out, "delivery")
    os.makedirs(out_dir, exist_ok=True)
    window, step = test["blueprint"]["values"]["window_size"], test["blueprint"]["values"]["step_size"]

    def one(job):
        """One session: job's source to its output (resume: a finished output is kept)."""
        name, src, dst = job
        if os.path.exists(dst):
            return dst
        runs = state.setdefault("sessions", {})
        if name not in runs:
            fid = upload_file(src, f"{a.name}-{os.path.basename(src).replace('.csv', '')}-{stamp()}.csv")["file_id"]
            ses = open_session(state["deployment"], fid)
            with lock:
                runs[name] = ses["id"]
            save()
            log(f"{name}: session {ses['id']}")
        status = poll(runs[name])
        got = download(runs[name], dst, src, m["timestamp_column"])
        if not got and status in ("failed", "canceled", "cancelled"):   # forget the dead session: a rerun opens a new one
            error = str(request("GET", lifecycle_url("sessions", runs[name])).get("error") or "")
            if "DeadlineExceeded" in error:
                log(f"{name}: {error[:200]} A plain rerun hits it again: rerun with --max-windows-per-run N "
                    f"(the same windows in sessions of N, each well under the limit)")
            elif error:
                log(f"{name}: {error[:300]}")
            with lock:
                del runs[name]
            save()
        log(f"{name}: {status}; " + ("output complete" if got else "no complete output: rerun to retry"))
        return got

    def safe(job):
        for attempt in range(NETWORK_RETRIES + 1):
            try:
                return one(job)
            except Exception as e:   # noqa: BLE001
                text = f"{type(e).__name__} {e}"
                if attempt < NETWORK_RETRIES and any(k in text for k in ("URLError", "urlopen", "timed out", "Connection")):
                    log(f"{job[0]}: network error ({e}); retrying in {NETWORK_WAIT_S} s")
                    time.sleep(NETWORK_WAIT_S)
                    continue
                log(f"{job[0]}: {text}; rerun to retry")
                return None

    jobs, split = [], {}            # split: a file's output -> [((first, end), the run's output)]
    for f in m["delivery"]:
        src = os.path.join(a.roles, f["file"])
        dst = os.path.join(out_dir, os.path.basename(f["file"]))
        if os.path.exists(dst):
            continue
        with open(src) as fh:
            header, *lines = fh.readlines()
        parts = pieces(len(lines), window, step, a.max_windows_per_run)
        if len(parts) == 1:
            jobs.append((f["file"], src, dst))
            continue
        pdir = os.path.join(a.out, "delivery-runs")
        os.makedirs(pdir, exist_ok=True)
        stem = os.path.basename(f["file"])[:-4]
        split[dst] = []
        for i, (first, end) in enumerate(parts):
            psrc = os.path.join(pdir, f"{stem}_run{i + 1:02d}.csv")
            if not os.path.exists(psrc):
                with open(psrc, "w") as fh:
                    fh.writelines([header] + lines[first:end])
            jobs.append((f"{f['file']} rows {first}-{end}", psrc, psrc[:-4] + "_out.csv"))
            split[dst].append(((first, end), psrc[:-4] + "_out.csv"))
        log(f"{f['file']}: {len(parts)} runs of at most {a.max_windows_per_run:,} windows")
    with ThreadPoolExecutor(a.parallel) as pool:
        list(pool.map(safe, jobs))
    for dst, parts in split.items():
        if all(os.path.exists(p) for _, p in parts):
            join(parts, dst, window, step)
    done = sum(os.path.exists(os.path.join(out_dir, os.path.basename(f["file"]))) for f in m["delivery"])
    log(f"{done} of {len(m['delivery'])} delivery files have complete outputs in {out_dir}")
    if done == len(m["delivery"]):
        print(f"\nnext: python score.py --roles {a.roles} --out {a.out}")
    return 0 if done == len(m["delivery"]) else 1


if __name__ == "__main__":
    sys.exit(main())
