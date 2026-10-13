#!/usr/bin/env python3
"""Path 2, step 3: deliver the promoted blueprint as a bundle, one agent run per delivery file.

POST /agents/bundles {"blueprint": <your key>, "values": {...}} creates the deployable agent.
Its values override the promoted blueprint's: by default it adds `step_size: 1` (a score
for every sample; --step to change), `output_score: true` and checkpointing off. --threshold
labels windows at another threshold with the same detector (an operating point, not a new
detector). Then POST /agents/bundles/{id}/run per delivery file, --parallel at a time.

Each output is downloaded again every 15 s until its last window reaches the input's last
timestamp: a run can report `completed` while its output is still being written. A
network outage is waited out (every 60 s, up to an hour), not counted as a failed file.

Writes <out>/delivery/<file>.csv and <out>/deliver_state.json (resume: finished outputs
are skipped, started agents are polled, not started again).

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
from atai_http import TERMINAL, agents, check_auth, load_dotenv, request, root, upload_file  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from common import load_manifest, log, stamp  # noqa: E402

CHECKPOINTING_OFF = {"checkpointing": {"trigger_after_input_byte_size": 0, "trigger_after_delay_ms": 0}}
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


def covered(out_path, source_end):
    """True once the output's last window ends within ~1.5 windows of the input's end."""
    with open(out_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return False
    end = max(float(r["finish_timestamp"]) for r in rows)
    win = max(float(r["finish_timestamp"]) - float(r["start_timestamp"]) for r in rows[:64])
    return source_end - end <= 1.5 * win + 1e-6


def poll(agent_id, timeout_s=3 * 3600):
    """Status, until terminal. An error-level log entry is terminal too: a pod can die while
    `status` still reads `running`. (Match the level, not the word "terminated": the happy
    path logs "Agent execution terminated successfully".)"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        a = request("GET", f"{agents()}/instances/{agent_id}")
        if a.get("status") in TERMINAL:
            return a["status"]
        logs = request("GET", f"{agents()}/instances/{agent_id}/logs").get("data", [])
        if any(e.get("level") == "error" for e in logs):
            return "failed"
        time.sleep(20)
    return "timeout"


def download(agent_id, dst, source, ts_col):
    """The run's output, once it covers the input; None after 10 min of short downloads."""
    end = last_ts(source, ts_col)
    for _ in range(40):
        res = request("GET", f"{agents()}/instances/{agent_id}/results").get("data", [])
        if res:
            name = res[0]["data"]["filename"]
            import urllib.request
            req = urllib.request.Request(f"{root()}/v0.5/files/download/{name}")
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
    ap.add_argument("--step", type=int, default=1, help="step_size for delivery (the trial's, if 0)")
    ap.add_argument("--threshold", type=float, help="label windows at this threshold instead of the trial's")
    ap.add_argument("--parallel", type=int, default=3)
    add_background_flag(ap, "{out}/deliver.log")
    a = ap.parse_args(argv)
    maybe_detach(a)
    load_dotenv()
    check_auth(log)
    m = load_manifest(a.roles)
    test = json.load(open(os.path.join(a.out, "test_state.json")))
    key = test["blueprint"]["key"]
    sf = os.path.join(a.out, "deliver_state.json")
    state = json.load(open(sf)) if os.path.exists(sf) else {}
    if state.get("blueprint", key) != key:
        sys.exit(f"{sf} is for {state['blueprint']}: move it aside to deliver {key}")
    state["blueprint"] = key

    def save():
        with lock:
            json.dump(state, open(sf, "w"), indent=1)

    if "bundle" not in state:
        values = {"output_score": True, "runner_config": CHECKPOINTING_OFF}
        if a.step:
            values["step_size"] = a.step
        if a.threshold is not None:
            values["threshold"] = a.threshold
        b = request("POST", f"{agents()}/bundles", {"blueprint": key, "name": f"{a.name} delivery", "values": values})
        state["bundle"], state["values"] = b["id"], values
        save()
        log(f"bundle {b['id']} on {key}: {json.dumps({k: v for k, v in values.items() if k != 'runner_config'})}")
    out_dir = os.path.join(a.out, "delivery")
    os.makedirs(out_dir, exist_ok=True)

    def one(f):
        src = os.path.join(a.roles, f["file"])
        dst = os.path.join(out_dir, os.path.basename(f["file"]))
        if os.path.exists(dst):
            return dst
        runs = state.setdefault("agents", {})
        if f["file"] not in runs:
            fid = upload_file(src, f"{a.name}-{os.path.basename(src).replace('.csv', '')}-{stamp()}.csv")["file_id"]
            agent = request("POST", f"{agents()}/bundles/{state['bundle']}/run",
                            {"connectors": {"source": [{"type": "file", "id": fid, "format": "csv"}]}})
            with lock:
                runs[f["file"]] = agent["id"]
            save()
            log(f"{f['file']}: agent {agent['id']}")
        status = poll(runs[f["file"]])
        got = download(runs[f["file"]], dst, src, m["timestamp_column"])
        log(f"{f['file']}: {status}; " + ("output complete" if got else "no complete output: rerun to retry"))
        return got

    def safe(f):
        for attempt in range(NETWORK_RETRIES + 1):
            try:
                return one(f)
            except Exception as e:   # noqa: BLE001
                text = f"{type(e).__name__} {e}"
                if attempt < NETWORK_RETRIES and any(k in text for k in ("URLError", "urlopen", "timed out", "Connection")):
                    log(f"{f['file']}: network error ({e}); retrying in {NETWORK_WAIT_S} s")
                    time.sleep(NETWORK_WAIT_S)
                    continue
                log(f"{f['file']}: {text}; rerun to retry")
                return None

    with ThreadPoolExecutor(a.parallel) as pool:
        got = list(pool.map(safe, m["delivery"]))
    done = sum(1 for g in got if g)
    log(f"{done} of {len(got)} delivery files have complete outputs in {out_dir}")
    if done == len(got):
        print(f"\nnext: python score.py --roles {a.roles} --out {a.out}")
    return 0 if done == len(got) else 1


if __name__ == "__main__":
    sys.exit(main())
