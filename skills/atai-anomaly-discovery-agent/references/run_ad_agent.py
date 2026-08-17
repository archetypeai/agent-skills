#!/usr/bin/env python3
"""Run the managed Anomaly Discovery agent end to end, then score the output.

    python3 run_ad_agent.py                       # bundled sample, default detector
    python3 run_ad_agent.py --csv my_asset.csv --detector s3://.../fit-detector.safetensors
    python3 run_ad_agent.py --bundle-id bnd_...   # reuse a bundle you already have
    python3 run_ad_agent.py --score-only ad-output.csv

Unlike the OSM and RED runners there is no pre-packaged bundle to resolve by
name: a detector encodes ONE asset's notion of normal, so you supply one and
this creates a bundle from the canonical `ad` blueprint around it. Bundles are
reused by name when one already exists, so repeated runs do not litter the org.

Scoring is deliberately not precision/recall — run-to-failure data has no
per-window ground truth. See `score()` and SKILL.md.

Stdlib only. Reads ATAI_API_KEY and ATAI_API_ENDPOINT from ./.env or the
environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

# The detector published by the anomaly-discovery-agent-example repo, fitted on
# set 2's healthy baseline. It scores the bundled sample, which is from that
# same bearing — a detector belongs to one asset.
DEFAULT_DETECTOR = (
    "s3://atai-platform-dev-platform-data-us-west-2/jos/jobs/"
    "job_0331mvzhvr9r6tyd3hm1gts3qg/agent/worker-0/outputs/output/fit-detector.safetensors"
)
DEFAULT_THRESHOLD = 1.762
# Three consecutive snapshots over the line. One window is noise; this is the
# same rule the example repo's offline scorer uses, so numbers are comparable.
SUSTAIN = 3


# --------------------------------------------------------------------------
# environment / transport
# --------------------------------------------------------------------------

def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def api_base() -> str:
    endpoint = os.environ.get("ATAI_API_ENDPOINT", "").rstrip("/")
    if not endpoint:
        sys.exit("ATAI_API_ENDPOINT is not set (see .env.example)")
    return endpoint


def agents_base() -> str:
    # The Agent API is versionless: /agents, never /vX.Y/agents.
    base = api_base()
    for suffix in ("/v0.5", "/v0.4", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return f"{base}/agents"


def require_key() -> str:
    key = os.environ.get("ATAI_API_KEY", "")
    if not key:
        sys.exit("ATAI_API_KEY is not set (see .env.example)")
    return key


def request(method: str, url: str, body=None, raw: bool = False, retries: int = 3):
    """One HTTP call, retrying transient network errors (not HTTP errors)."""
    data = None
    headers = {"Authorization": f"Bearer {require_key()}"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = resp.read()
                return payload if raw else (json.loads(payload) if payload else {})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            raise SystemExit(f"{method} {url} -> HTTP {exc.code}\n  {detail}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise SystemExit(f"{method} {url} failed after {retries} attempts: {last}")


def upload_file(path: str) -> str:
    """Upload and return the file id. Names are timestamped on purpose.

    File ids ARE filenames, so re-uploading a name replaces the record and
    orphans the object any in-flight run already resolved — that run then dies
    naming a UUID and nothing else.
    """
    stem = os.path.basename(path).rsplit(".", 1)[0]
    rename = f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.csv"
    boundary = uuid.uuid4().hex
    with open(path, "rb") as fh:
        content = fh.read()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{rename}"\r\n'.encode(),
        b"Content-Type: text/csv\r\n\r\n",
        content,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{api_base()}/v0.5/files",
        data=body,
        headers={
            "Authorization": f"Bearer {require_key()}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.loads(resp.read())
    return (out.get("data") or out).get("file_id", rename)


# --------------------------------------------------------------------------
# bundle / run
# --------------------------------------------------------------------------

def find_bundle(name: str) -> str | None:
    """Reuse a bundle of this name if one exists. NOTE the plural endpoint."""
    res = request("GET", f"{agents_base()}/bundles?query={urllib.parse.quote(name)}&limit=100")
    for b in res.get("data", res) if isinstance(res, dict) else res:
        if (b.get("name") or "") == name:
            return b.get("id")
    return None


def create_bundle(name: str, detector: str, threshold: float, embeddings: bool) -> str:
    body = {
        "blueprint": "ad",
        "name": name,
        "values": {
            "threshold": threshold,
            # Required above 1 kHz. Timestamps are u64 milliseconds, so faster
            # sampling means consecutive samples share one — and left at the
            # default `true`, EVERY window is invalidated while the run still
            # reports `completed`.
            "validate_monotonic_timestamps": False,
            # No single tolerance describes burst-sampled data, where the
            # within-burst and between-burst intervals differ by orders of
            # magnitude.
            "sample_rate_interval_tolerance": None,
            "max_temporal_gap": 60.0,
            "output_score": True,
            "output_embeddings": embeddings,
        },
        # Must be an s3:// URI: the files API rejects safetensors and the
        # detector node resolves artifact strings as filesystem/S3 paths only.
        # A wrong KEY is accepted here and fails ~30 s into the run.
        "artifacts": {"ad-detector": detector},
    }
    res = request("POST", f"{agents_base()}/bundles", body=body)
    b = res.get("data", res)
    return b["id"]


def run_bundle(bundle_id: str, file_id: str) -> str:
    res = request(
        "POST",
        f"{agents_base()}/bundles/{bundle_id}/run",
        body={"connectors": {"source": [{"type": "file", "id": file_id}]}},
    )
    r = res.get("data", res)
    return r.get("id") or r.get("agent_id")


def poll(agent_id: str, timeout_s: int = 3600, interval_s: int = 15) -> str:
    """Poll to a terminal state, treating /logs as authoritative.

    `status` is NOT reliable: pods have terminated with Error (exit=1) while
    status still read `running` hours later. An error-level or pod.terminated
    log event is terminal regardless. Note /logs, not /events — the latter
    carries only lifecycle info.
    """
    deadline = time.time() + timeout_s
    seen: set[str] = set()
    while time.time() < deadline:
        info = request("GET", f"{agents_base()}/instances/{agent_id}")
        status = (info.get("data", info) or {}).get("status", "unknown")

        logs = request("GET", f"{agents_base()}/instances/{agent_id}/logs").get("data", [])
        terminal = None
        for entry in reversed(logs):
            key = str(entry.get("id"))
            if key in seen:
                continue
            seen.add(key)
            level = (entry.get("level") or "").lower()
            msg = str(entry.get("message") or "")[:160]
            low = msg.lower()

            # Match on the LEVEL and on explicit failure text, not on the word
            # "terminated" — the happy path also logs "Agent execution
            # terminated successfully", and treating that as a failure means the
            # runner reports every successful run as failed.
            if level in ("error", "critical") or "exit=1" in low or "error (exit" in low:
                print(f"  [error] {msg}")
                terminal = "failed"
                continue
            if "terminated successfully" in low or "completed successfully" in low:
                print(f"  [{level}] {msg}")
                terminal = terminal or "completed"
                continue
            if msg:
                print(f"  [{level}] {msg}")
        if terminal:
            return terminal

        if status in ("completed", "succeeded", "failed", "error", "cancelled"):
            return status
        time.sleep(interval_s)
    return "timeout"


def download_results(agent_id: str, out_path: str) -> str | None:
    res = request("GET", f"{agents_base()}/instances/{agent_id}/results")
    items = res.get("data", res) if isinstance(res, dict) else res
    if not items:
        # The single most important check in this script. A run whose windows
        # were all invalidated reports `completed` with NO results.
        print("  /results is EMPTY — the run reported success but produced nothing.")
        print("  Almost always validate_monotonic_timestamps: every window was")
        print("  marked invalid. See SKILL.md, step 2.")
        return None
    # The reference is nested one level: each item is
    #   {id, port_name, data: {ref, filename, num_bytes, ...}, created_at}
    # so the filename lives at item["data"]["filename"], NOT at the top level.
    # Reading it from the top gives an empty name and a 404 on a bare
    # /files/download/ URL.
    inner = items[0].get("data") or {}
    name = (inner.get("filename")
            or inner.get("ref", "").rsplit("/", 1)[-1]
            or items[0].get("name", ""))
    if not name:
        print(f"  results present but no filename in the reference: {items[0]!r}")
        return None
    print(f"  results ({len(items)}): {name}  ({inner.get('num_bytes', '?')} bytes)")
    blob = request("GET", f"{api_base()}/v0.5/files/download/{urllib.parse.quote(name)}", raw=True)
    with open(out_path, "wb") as fh:
        fh.write(blob)
    return out_path


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score(output_path: str, labels_path: str, threshold: float) -> dict:
    """Lead time and crossing rate — NOT precision/recall.

    Run-to-failure data has no per-window ground truth: the binary labels in
    circulation are a hand-placed cut on a gradual curve, so precision against
    them mostly measures where that line was drawn. What IS ground truth is the
    documented outcome and the position of each window in the asset's life.
    """
    rows = list(csv.DictReader(open(output_path)))
    labels = list(csv.DictReader(open(labels_path)))
    if not rows:
        return {"error": "output has no rows"}

    scores = [float(r["anomaly_score"]) for r in rows if r.get("anomaly_score")]
    n_invalid = sum(1 for r in rows if str(r.get("invalid", "")).lower() == "true")

    # Labels are one row per WINDOW and carry the snapshot index explicitly;
    # windows-per-snapshot is whatever the ratio says. Indexing by position
    # instead reads the wrong part of the asset's life.
    per_snapshot: dict[int, float] = {}
    for lab in labels:
        idx = int(lab["snapshot_index"])
        per_snapshot.setdefault(idx, float(lab["operating_hours_to_end"]))
    wps = max(1, len(labels) // max(1, len(per_snapshot)))

    over_by_snapshot: dict[int, bool] = {}
    for i, r in enumerate(rows):
        snap = i // wps
        val = float(r.get("anomaly_score", 0) or 0) > threshold
        over_by_snapshot[snap] = over_by_snapshot.get(snap, False) or val

    ordered = sorted(over_by_snapshot.items())
    run = 0
    detected_at = None
    for snap, over in ordered:
        if over:
            run += 1
            if run >= SUSTAIN:
                detected_at = snap - SUSTAIN + 1
                break
        else:
            run = 0

    crossings = sum(1 for s in scores if s > threshold)
    return {
        "windows": len(rows),
        "invalid": n_invalid,
        "snapshots": len(ordered),
        "median_score": round(statistics.median(scores), 4) if scores else None,
        "max_score": round(max(scores), 4) if scores else None,
        "crossings": crossings,
        "crossing_rate": round(crossings / len(scores), 4) if scores else None,
        "detected": detected_at is not None,
        "detected_at_snapshot": detected_at,
        "lead_hours": per_snapshot.get(detected_at) if detected_at is not None else None,
    }


def report(res: dict, threshold: float) -> None:
    print()
    print("=" * 66)
    print(f" Scored — threshold {threshold}, sustained {SUSTAIN} snapshots")
    print("=" * 66)
    if "error" in res:
        print(f"  {res['error']}")
        return
    print(f"  windows           {res['windows']:,}   invalid {res['invalid']}")
    print(f"  snapshots         {res['snapshots']:,}")
    print(f"  score median/max  {res['median_score']} / {res['max_score']}")
    print(f"  crossings         {res['crossings']}  ({res['crossing_rate']:.2%} of windows)")
    if res["detected"]:
        lead = res["lead_hours"]
        print(f"  DETECTED          snapshot {res['detected_at_snapshot']}"
              + (f"  ->  {lead:.1f} operating hours before end of record" if lead is not None else ""))
    else:
        print("  not detected      no sustained crossing")
    print()
    print("  Lead time is to the END OF THE RECORD, not to fault onset —")
    print("  degradation is already underway when the detector fires.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="sample_data/bearing_eval_set2_brg1_transition.csv")
    ap.add_argument("--labels", default=None,
                    help="ground-truth sidecar (default: <csv stem>_labels.csv)")
    ap.add_argument("--detector", default=DEFAULT_DETECTOR,
                    help="s3:// URI of the fitted detector")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--bundle-id", default=None, help="reuse an existing bundle")
    ap.add_argument("--bundle-name", default="AD skill sample (set2 brg1)")
    ap.add_argument("--embeddings", action="store_true",
                    help="also emit per-window Omega embeddings (much larger output)")
    ap.add_argument("--output", default="ad-output.csv")
    ap.add_argument("--score-only", metavar="OUTPUT_CSV", default=None,
                    help="skip the platform and score an output you already have")
    args = ap.parse_args()

    labels = args.labels or args.csv.rsplit(".", 1)[0] + "_labels.csv"

    if args.score_only:
        report(score(args.score_only, labels, args.threshold), args.threshold)
        return

    load_dotenv()
    require_key()

    size_mb = os.path.getsize(args.csv) / 1048576
    if size_mb > 50:
        print(f"WARNING: {args.csv} is {size_mb:.0f} MiB. Inputs over ~50 MiB abort in")
        print("  ConcatColumnsNode at a checkpoint boundary — see SKILL.md. Split it.")

    print(f"uploading {args.csv} ({size_mb:.1f} MiB)")
    file_id = upload_file(args.csv)
    print(f"  file_id={file_id}")

    bundle_id = args.bundle_id
    if not bundle_id:
        bundle_id = find_bundle(args.bundle_name)
        if bundle_id:
            print(f"reusing bundle {bundle_id} ({args.bundle_name!r})")
        else:
            bundle_id = create_bundle(args.bundle_name, args.detector,
                                      args.threshold, args.embeddings)
            print(f"created bundle {bundle_id} ({args.bundle_name!r})")

    agent_id = run_bundle(bundle_id, file_id)
    print(f"agent {agent_id} — polling (status is not authoritative; watching /logs)")
    status = poll(agent_id)
    print(f"  status={status}")
    if status not in ("completed", "succeeded"):
        sys.exit(f"run did not complete: {status}")

    if not download_results(agent_id, args.output):
        sys.exit("no results to score")
    print(f"  saved {args.output}")

    if os.path.exists(labels):
        report(score(args.output, labels, args.threshold), args.threshold)
    else:
        print(f"  no labels at {labels} — skipping scoring")


if __name__ == "__main__":
    main()
