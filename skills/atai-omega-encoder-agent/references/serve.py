#!/usr/bin/env python3
"""Serve the Omega Encoder agent: one segment per query, scored live, checked against batch.

    python serve.py --roles roles                    # every delivery segment, in time order
    python serve.py --roles roles --limit 10 --role test

1. The frozen window's package (from embed_batch.py), deployed as **serving**.
2. A serving session whose `values` set window_size and step_size to the frozen window. A
   session's values apply to it alone: one serving deployment can serve any window, a
   session per window size. (Batch sessions refuse `values`: there, the window is set on
   the blueprint copy.)
3. Each segment, in time order, sent as one `POST /api/agents/sessions/{id}/query`: one list
   per channel, `rows` numbers each. The answer is one record per window, `channel_0..N` in
   the order the lists were sent (not the column names), 768 numbers each. It is scored at
   once by the frozen kNN against the batch library.
4. Every answer checked against batch embeddings of the same segments, when embed_batch.py
   has embedded the role: cosine per window and channel, score difference, alarm agreement.
5. The session and the deployment are ended however the run ends (Ctrl-C, SIGTERM): a
   serving deployment holds a worker until it is ended.

Writes <out>/serve/w<W>/<role>.npz (batch's layout; score it with
`detect.py deliver --source serving`) and <eval>/serving/<role>.{csv,json}.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atai_api as A  # noqa: E402
import detect as D  # noqa: E402
import embed_batch as E  # noqa: E402


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def query_body(df: pd.DataFrame, channels: list[str]) -> dict:
    """One segment: a list per channel. No timestamps: the agent numbers the samples itself."""
    return {"query": "", "events": [{"type": "data.numeric_array",
            "event_data": {"contents": [df[c].astype(float).tolist() for c in channels]}}]}


def windows_of(answer: dict, n_channels: int) -> np.ndarray:
    """[windows, channels, 768] from one answer, in window order."""
    records = (answer.get("response") or {}).get("data")
    if not isinstance(records, list) or not records:
        raise ValueError(f"unexpected answer: {str(answer)[:300]}")
    records = sorted(records, key=lambda r: r.get("start_timestamp", 0))
    return np.asarray([[r[f"channel_{c}"] for c in range(n_channels)] for r in records], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--role", default="delivery", choices=["validation", "test", "delivery"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="out")
    ap.add_argument("--eval", default="eval_out")
    ap.add_argument("--project", default="Omega Encoder agent")
    ap.add_argument("--key-prefix", default="omega-encoder")
    a = ap.parse_args()
    p = D.Paths(a.roles, a.out, a.eval)
    m = p.manifest
    cfg = D.frozen(p)
    window = cfg["window"]
    A.load_dotenv()
    log(f"endpoint {A.api_base()}")
    files = sorted((f for f in m["files"] if f["role"] == a.role), key=lambda f: f["start"])[:a.limit]
    detector = D.frozen_detector(p, cfg)
    log(f"frozen settings: {D.describe(cfg)}, threshold {cfg['threshold']:.4f}")

    st = E.State(a.out)
    ws = E.ensure_package(st, window, a.project, a.key_prefix)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    dpl = sid = None
    rows, embs = [], []
    try:
        dpl = A.request("POST", A.deployments_url(), body={
            "package_id": ws["package"], "kind": "serving", "sessions": {"max": 2}})["id"]
        st.d["serving"] = {"deployment": dpl}; st.save()
        A._wait_for(A.deployments_url(dpl), "ready", "deploying", 600)
        log(f"serving deployment {dpl} of {ws['package']} ready")
        sid = A.request("POST", A.sessions_url(), body={
            "deployment_id": dpl, "values": {"window_size": window, "step_size": window}})["id"]
        st.d["serving"]["session"] = sid; st.save()
        A.wait_running(sid)
        log(f"serving session {sid} with values window_size={window}, step_size={window}: running")
        for i, f in enumerate(files, 1):
            df = pd.read_csv(os.path.join(a.roles, f["path"]))
            t = time.time()
            answer = A.request("POST", f"{A.sessions_url(sid)}/query", body=query_body(df, m["channels"]))
            latency = time.time() - t
            emb = windows_of(answer, len(m["channels"]))
            if len(emb) != m["rows"] // window:
                sys.exit(f"{f['path']}: {len(emb)} windows, expected {m['rows'] // window}")
            score = float(detector.score(D.joint(emb[None], m["channels"], cfg["channels"]))[0])
            name = os.path.basename(f["path"])
            rows.append({"file": name, "score": score, "alarm": score > cfg["threshold"], "latency_s": round(latency, 3)})
            embs.append(emb)
            log(f"  {i:>3}/{len(files)} {name:<28} score {score:.4f}{'  ALARM' if score > cfg['threshold'] else ''}"
                f"  ({latency * 1000:.0f} ms)")
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        for label, ident, fn in (("session", sid, A.end_session), ("deployment", dpl, A.end_deployment)):
            if ident:
                try:
                    fn(ident)
                except BaseException as e:   # atai_api exits on an HTTP error
                    log(f"could not end the serving {label} {ident}: {str(e)[:200]} (run cleanup.py)")
        st.d.pop("serving", None); st.save()
        if dpl:
            log(f"ended serving session {sid} and deployment {dpl}")
    if not rows:
        return

    served = pd.DataFrame(rows)
    emb = np.asarray(embs)
    out_dir = os.path.join(a.out, "serve", f"w{window}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{a.role}.npz"), files=np.asarray(served["file"].tolist(), dtype=str), emb=emb,
                        window=np.asarray(window))
    lat = served["latency_s"].to_numpy()
    summary = {"window": window, "segments": len(served), "alarms": int(served["alarm"].sum()),
               "latency_s": {"median": float(np.median(lat)), "p95": float(np.quantile(lat, 0.95)), "max": float(lat.max())}}
    print(f"\n{len(served)} {a.role} segments served in {len(served)} queries, window {window} "
          f"({emb.shape[1]} per segment); {summary['alarms']} alarm(s)")
    print(f"  latency per query: median {summary['latency_s']['median'] * 1000:.0f} ms, "
          f"p95 {summary['latency_s']['p95'] * 1000:.0f} ms, max {summary['latency_s']['max'] * 1000:.0f} ms")
    batch = p.embeddings(a.role, window)
    if os.path.exists(batch):
        z = np.load(batch)
        idx = {str(n): i for i, n in enumerate(z["files"])}
        b = z["emb"][[idx[n] for n in served["file"]]]
        cos = (b * emb).sum(-1) / np.linalg.norm(b, axis=-1) / np.linalg.norm(emb, axis=-1)
        bs = detector.score(D.joint(b, m["channels"], cfg["channels"]))
        same = int(((bs > cfg["threshold"]) == served["alarm"].to_numpy()).sum())
        rel = np.abs(bs - served["score"].to_numpy()) / np.abs(bs)
        served["batch_score"] = bs
        summary["against_batch"] = {"cosine_min": float(cos.min()), "alarm_agreement": same,
                                    "score_max_relative_difference": float(rel.max())}
        print(f"  against batch: embeddings cosine min {cos.min():.6f} over {cos.size} window-channels; "
              f"alarm identical in {same}/{len(served)}; score max relative difference {rel.max():.1e}")
    else:
        print(f"  {batch} not found: embed the role in batch to check serving against it")
    ev = os.path.join(a.eval, "serving")
    os.makedirs(ev, exist_ok=True)
    served.to_csv(os.path.join(ev, f"{a.role}.csv"), index=False)
    json.dump(summary, open(os.path.join(ev, f"{a.role}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
