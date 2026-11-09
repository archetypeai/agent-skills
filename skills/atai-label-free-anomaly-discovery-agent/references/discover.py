#!/usr/bin/env python3
"""Run the Label-Free Anomaly Discovery agent (`od`) over CSV files, in batch.

    python discover.py --files sample_data/metropt3_2020-07-08_2020-07-16.csv --window 360
    python discover.py --files a.csv b.csv --window 120 --channels p1,p2,t1 --out out
    python discover.py --files a.csv --window 360 --rerun          # new sessions even where an output exists

Agent Lifecycle 2.0, every object in one project:
  1. project       `--project` (reused if one by that name exists)
  2. blueprint     a copy of `od` with these values: the timestamp column, the channels
                   (default: every column but the timestamp), window_size and step_size
                   (default step = window: no overlap), and any `--values` JSON. The rest
                   is od's default: Omega 1.5 + 7 statistics per channel, per-channel PCA 5,
                   LOF (L1) with K = 20% of the file's windows, threshold 1.5.
  3. package       the copy, its encoder resolved once
  4. deployment    kind `batch`
  5. sessions      one per file; od fits LOF on that file's valid windows and scores them
Nothing is trained ahead of time and no label is sent. Ids go to <out>/discover_state.json,
so a rerun resumes. Writes <out>/discover/od-<file>.csv:
  finish_timestamp, start_timestamp, predicted_label, invalid, anomaly_score
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import od_api  # noqa: E402


def log(msg: str) -> None:
    print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def build_values(header: list[str], timestamp_col: str, channels: str | None, window: int, step: int | None,
                 extra: dict) -> dict:
    """The copy's `overrides.values`."""
    if timestamp_col not in header:
        raise SystemExit(f"no column {timestamp_col!r} in the input (columns: {header[:10]})")
    chans = [c.strip() for c in channels.split(",")] if channels else [c for c in header if c != timestamp_col]
    missing = [c for c in chans if c not in header]
    if missing:
        raise SystemExit(f"channels not in the input: {missing}")
    if not 16 <= window <= 1024:
        raise SystemExit(f"--window {window}: od's encoder takes 16 to 1,024 samples")
    return {"timestamp_column": timestamp_col, "data_columns": chans, "window_size": window,
            "step_size": step or window, **extra}


def expected_windows(path: Path, values: dict) -> tuple[int, float]:
    """How many windows od emits for a file, and the finish timestamp of the last."""
    ts = pd.read_csv(path, usecols=[values["timestamp_column"]])[values["timestamp_column"]].to_numpy()
    w, s = values["window_size"], values["step_size"]
    n = (len(ts) - w) // s + 1 if len(ts) >= w else 0
    return n, float(ts[(n - 1) * s + w - 1]) if n else float("nan")


def check_output(out: Path, src: Path, values: dict) -> str:
    """'' if the output has a row per window of the input, ending where the last window ends."""
    o = pd.read_csv(out)
    n, last = expected_windows(src, values)
    if len(o) != n:
        return f"{len(o)} windows, expected {n}"
    if float(o.finish_timestamp.iloc[-1]) != last:
        return "the last window does not end where the input's last full window does"
    return ""


def setup(st: dict, state_path: Path, project: str, key: str, values: dict) -> None:
    """Project, blueprint copy, package and deployment: made once, reused after."""
    od_api.check_key()
    if st.get("endpoint") and st["endpoint"] != od_api.root():
        raise SystemExit(f"{state_path} is for {st['endpoint']}; use another --out for {od_api.root()}")
    st["endpoint"] = od_api.root()
    if st.get("values") and st["values"] != values:
        raise SystemExit(f"{state_path} was made with other values ({json.dumps(st['values'])[:200]}…); "
                         "use another --out (and --key), or delete the old objects with cleanup.py")

    def save():
        state_path.write_text(json.dumps(st, indent=1))
    if "project" not in st:
        st["project"] = od_api.find_project(project) or od_api.request(
            "POST", "/projects", {"name": project, "description": "Label-Free Anomaly Discovery"})["id"]
        save()
    log(f"project    {st['project']} ({project})")
    if "blueprint" not in st:
        found = od_api.request("GET", f"/agents/blueprints?project_id={st['project']}&key={key}")["data"]
        if found:
            raise SystemExit(f"blueprint key {key!r} already exists in the project; pick another --key")
        st["blueprint"] = od_api.request("POST", "/agents/blueprints/od/copy", {
            "project_id": st["project"], "blueprint_key": key, "overrides": {"values": values}})["id"]
        st["values"] = values
        save()
    log(f"blueprint  {st['blueprint']} ({key}: a copy of od, window {values['window_size']}, step {values['step_size']})")
    if "package" not in st:
        st["package"] = od_api.request("POST", "/agents/packages", {
            "blueprint_ids": [st["blueprint"]], "target": {"kind": "platform"}, "name": key})["id"]
        save()
    od_api.wait(f"/agents/packages/{st['package']}", {"ready"}, label=f"package    {st['package']}")
    if "deployment" not in st or od_api.request("GET", f"/agents/deployments/{st['deployment']}").get(
            "status") in od_api.TERMINAL_BAD:
        st["deployment"] = od_api.request("POST", "/agents/deployments",
                                          {"package_id": st["package"], "kind": "batch"})["id"]
        save()
    od_api.wait(f"/agents/deployments/{st['deployment']}", {"ready"}, label=f"deployment {st['deployment']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", nargs="+", type=Path, required=True, help="prepared CSV files, one session each")
    ap.add_argument("--window", type=int, required=True, help="window_size in rows (16-1,024)")
    ap.add_argument("--step", type=int, help="step_size in rows (default: the window, no overlap)")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--channels", help="comma-separated (default: every column but the timestamp)")
    ap.add_argument("--values", default="{}", help='more od values as JSON, e.g. \'{"threshold": 2.0}\'')
    ap.add_argument("--project", default="Label-Free Anomaly Discovery")
    ap.add_argument("--key", default="od-lfad", help="the copy's blueprint key, unique in the project")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--rerun", action="store_true", help="new sessions even where an output exists")
    a = ap.parse_args(argv)

    for f in a.files:
        if not f.exists():
            raise SystemExit(f"{f} not found")
    header = list(pd.read_csv(a.files[0], nrows=0).columns)
    values = build_values(header, a.timestamp_col, a.channels, a.window, a.step, json.loads(a.values))
    a.out.mkdir(parents=True, exist_ok=True)
    state_path = a.out / "discover_state.json"
    st = json.loads(state_path.read_text()) if state_path.exists() else {}
    setup(st, state_path, a.project, a.key, values)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs = st.setdefault("runs", {})

    def out_of(src: Path) -> Path:
        return a.out / "discover" / f"od-{src.stem}.csv"

    for src in a.files:                                # start every session, then collect
        out, run = out_of(src), runs.get(str(src), {})
        if out.exists() and not a.rerun and not check_output(out, src, values):
            log(f"{src.name}: {out} already complete (--rerun to redo)")
            continue
        if a.rerun or "session" not in run or run.get("status") in od_api.TERMINAL_BAD:
            fid = od_api.upload(src, f"od-{src.stem}-{stamp}.csv")
            ses = od_api.request("POST", "/agents/sessions", {
                "deployment_id": st["deployment"], "connectors": {"source": [{"type": "file", "id": fid}]}})["id"]
            runs[str(src)] = {"file_id": fid, "session": ses, "started": stamp, "output": str(out)}
            state_path.write_text(json.dumps(st, indent=1))
            log(f"{src.name}: uploaded {fid}, session {ses}")

    for src in a.files:
        out, run = out_of(src), runs.get(str(src))
        if not run or (out.exists() and not a.rerun and not check_output(out, src, values)):
            continue
        t0 = time.time()
        ses = od_api.wait(f"/agents/sessions/{run['session']}", {"completed"}, every_s=10, timeout_s=3600,
                          label=f"{src.name}: session {run['session']}")
        run["status"] = ses["status"]
        out.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(40):                            # a completed session can still be writing its output
            out.write_bytes(od_api.download_result(run["session"]))
            problem = check_output(out, src, values)
            if not problem:
                break
            time.sleep(15)
        else:
            raise SystemExit(f"{src.name}: output incomplete: {problem}")
        o = pd.read_csv(out)
        inv = o.invalid.astype(str).str.lower().eq("true")
        log(f"{src.name}: {len(o)} windows, {int(inv.sum())} invalid (cross a gap), "
            f"{int((o.predicted_label == 'anomaly').sum())} over the threshold; {time.time() - t0:.0f} s -> {out}")
        state_path.write_text(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
