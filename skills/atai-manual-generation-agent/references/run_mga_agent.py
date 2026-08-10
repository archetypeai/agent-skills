#!/usr/bin/env python3
"""Run the managed Manual Generation agent end to end, and score its output.

Stdlib only — no pip install, no virtualenv. Put a .env next to where you run:

    ATAI_API_KEY=<dev API key>
    ATAI_API_ENDPOINT=https://api.dev.u1.archetypeai.app   # NO /v0.5 suffix

    python3 run_mga_agent.py --video procedure.mp4
    python3 run_mga_agent.py --video procedure.mp4 --blueprint mga    # active blueprint
    python3 run_mga_agent.py --dry-run --video procedure.mp4
    python3 run_mga_agent.py --score out.jsonl --reference steps.csv  # offline

Budget ~15 minutes per run: ~7.5 min of model download and load happens on EVERY
run (nothing is cached), then roughly 2.5x realtime processing. The platform
serializes these, so concurrent submissions queue rather than parallelize.

SIX THINGS THAT WILL BITE YOU (all verified on dev, 2026-08-10)

  1. `POST /agents/bundle` is 404. The endpoint is PLURAL: /agents/bundles.
  2. Starting a run returns HTTP 202, not 201. Treat only 201 as success and you
     report a failure while the agent runs unattended with nothing collecting its
     output.
  3. Poll /logs, NOT /events. /events carries only "run started" and "dispatched
     to JOS"; /logs is what the console shows and the only place errors appear.
  4. The instance `status` field lies in BOTH directions — `running` 20+ minutes
     after a pod exited 1, and `running` after a job completed. Judge terminality
     from the log stream.
  5. Values are accepted whether or not they do anything. A key absent from the
     blueprint's `values`, or present but never referenced as ${values.<key>},
     is stored on the bundle and silently ignored. This script preflights that.
  6. Videos longer than ~5 minutes fail outright, several minutes in.

THE OUTPUT-TOKEN CAP

The active `mga` blueprint hardcodes max_new_tokens: 256 and does not expose it,
which truncates the manual — on a 173 s video, 6 steps covering half of it, the
last cut mid-clause. A superseded version (BLUEPRINT_WITH_PARAMS below) wires
max_new_tokens and prompt, and with them the same video yields 10-19 clean steps.

That superseded blueprint is `is_active: false`: usable for diagnosis, but nothing
built on it can ship, and it may be removed without notice. It is the default here
only so the skill demonstrates working behaviour. When the values are restored on
the active blueprint, set BLUEPRINT_DEFAULT = BLUEPRINT_ACTIVE.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

# Active canonical mga (is_canonical: true, is_active: true) — what ships.
BLUEPRINT_ACTIVE = "blp_0zmce3qnjp8659dp1pszdqbtx4"
# Superseded, but still accepts bundles and runs. Wires max_new_tokens, prompt,
# temperature, top_k, top_p, repetition_penalty, seed, stop. DIAGNOSIS ONLY.
BLUEPRINT_WITH_PARAMS = "blp_76kyqm4vjp9pt8tvfz8tks7x6t"
BLUEPRINT_DEFAULT = BLUEPRINT_WITH_PARAMS

# Says what to COVER, never how to FORMAT. ManualGenerationResultsParserNode owns
# the output template: a prompt that specifies its own format makes the parser
# return zero steps, which is why `prompt` was exposed once and rolled back.
DEFAULT_PROMPT = (
    "Generate a concise, ordered list of every distinct step performed in this "
    "video, covering the procedure from the first action to the last. Include "
    "brief steps and steps that are repeated. Use up to 20 steps. Keep each step "
    "to at most 15 words. Use both what is shown and what is said. Do not invent "
    "steps that are not shown or stated in the video."
)
TERMINAL_EVENTS = ("pod.terminated", "job.completed", "job.failed", "job.canceled")


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def env() -> tuple[str, str]:
    load_dotenv()
    key = os.environ.get("ATAI_API_KEY")
    endpoint = (os.environ.get("ATAI_API_ENDPOINT") or "").rstrip("/")
    if not key or not endpoint:
        sys.exit("set ATAI_API_KEY and ATAI_API_ENDPOINT (see .env.example)")
    # Tolerate a /vX.Y suffix: this script mounts /agents and /v0.5/files itself.
    for suffix in ("/v0.5", "/v0.4", "/v1"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    return key, endpoint


def api(method: str, url: str, body=None, raw: bool = False):
    key, _ = env()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
            # 202 is success for /run — see gotcha 2.
            if resp.status not in (200, 201, 202):
                sys.exit(f"{method} {url} unexpected status {resp.status}")
            return payload if raw else json.loads(payload or b"null")
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url} failed ({e.code}): {e.read().decode(errors='replace')}")


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def upload(path: str) -> str:
    """POST the video as multipart/form-data.

    The DECLARED Content-Type is checked against a MIME allowlist, not the bytes,
    so an .mp4 announced as anything else is rejected. Reads the file into memory;
    stream it in chunks if yours is very large.
    """
    key, endpoint = env()
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{os.path.basename(path)}"\r\n'.encode(),
        b"Content-Type: video/mp4\r\n\r\n",
        _read_bytes(path),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(f"{endpoint}/v0.5/files", data=body, method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.loads(resp.read())["file_id"]
    except urllib.error.HTTPError as e:
        sys.exit(f"upload failed ({e.code}): {e.read().decode(errors='replace')}")


def inert_values(bp: dict, values: dict) -> list[str]:
    """Which of `values` will this blueprint silently ignore?

    Honoured means declared in `values` AND referenced as ${values.<key>} by some
    node or connector config. The API accepts anything and drops the rest, so this
    is the only way to know before spending a run. Reading the document rather
    than hardcoding key names means this keeps working when a blueprint changes.
    """
    doc = bp["document"]
    wired = json.dumps({"nodes": doc.get("nodes"),
                        "connectors": doc.get("connectors")})
    return [k for k in values
            if k not in doc.get("values", {})
            or ("${values." + k + "}") not in wired]


def watch(agent_id: str, timeout_s: int = 3600) -> str:
    """Stream /logs until a terminal event. Ignores `status` entirely."""
    _, endpoint = env()
    seen: set[str] = set()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rows = sorted(api("GET", f"{endpoint}/agents/instances/{agent_id}"
                               f"/logs?limit=500").get("data", []),
                      key=lambda r: str(r.get("created_at")))
        for r in rows:
            marker = f"{r.get('created_at')}|{r.get('event_type')}|{r.get('message')}"
            if marker not in seen:
                seen.add(marker)
                print(f"  {str(r.get('created_at'))[11:19]} "
                      f"{str(r.get('level')):7} {r.get('message')}")
        if rows and str(rows[-1].get("event_type")) in TERMINAL_EVENTS:
            return "failed" if any(str(r.get("level")) == "ERROR"
                                   for r in rows) else "completed"
        time.sleep(20)
    return "timeout"


def fetch_results(agent_id: str, out_path: str) -> str | None:
    _, endpoint = env()
    items = api("GET", f"{endpoint}/agents/instances/{agent_id}/results").get("data") or []
    if not items:
        print("  no results (a failed run leaves none)")
        return None
    ref = (items[0].get("data") or {}).get("ref") or items[0].get("ref")
    if not ref:
        print("  results carried no download ref")
        return None
    # Run-output refs are RELATIVE platform paths that resolve under /v0.5 and need
    # the bearer token. They do not expire.
    payload = api("GET", f"{endpoint}/v0.5{ref}", raw=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(payload)
    print(f"  saved {out_path} ({len(payload)} bytes)")
    return out_path


def load_manual(path: str) -> list[dict]:
    with open(path) as f:
        return json.loads(f.read().strip().splitlines()[0])["results"]


def show(path: str) -> None:
    steps = load_manual(path)
    print(f"\nmanual — {len(steps)} steps (step field 0-{len(steps) - 1})")
    for s in steps:
        print(f"  [{s['timestamp_start']:>6.1f} - {s['timestamp_end']:>6.1f}s] "
              f"(frames {s['frame_start']:.0f}-{s['frame_end']:.0f})  {s['instruction']}")
    # A final step without terminal punctuation is the token-cap signature.
    if steps and steps[-1]["instruction"].rstrip()[-1] not in ".!?":
        print("\n  WARNING: the last step is cut mid-clause — generation hit the "
              "output\n  token cap. On the active blueprint that is a hardcoded "
              "max_new_tokens: 256.")


def score(out_path: str, ref_path: str) -> None:
    """Reference-step recall + temporal IoU. Offline; no API key needed.

    Deliberately no precision: a reference step list is typically a documented
    procedure rather than an inventory of the video, so a predicted step with no
    reference match is not necessarily wrong.
    """
    steps = load_manual(out_path)
    per: dict[int, list[tuple[float, float]]] = {}
    with open(ref_path) as f:
        for row in csv.reader(f):
            if len(row) == 3:
                per.setdefault(int(row[0]), []).append((float(row[1]), float(row[2])))
    # One interval per reference step: earliest start, latest end across annotators.
    ref = {k: (min(a for a, _ in v), max(b for _, b in v)) for k, v in per.items()}

    def iou(a, b):
        ov = min(a[1], b[1]) - max(a[0], b[0])
        return ov / ((a[1] - a[0]) + (b[1] - b[0]) - ov) if ov > 0 else 0.0

    # A spoken caution cannot satisfy an action step, but it occupies a span — and
    # MGA tiles the timeline with no gaps, so a purely temporal match would credit
    # it and inflate recall. Exclude advisories from candidacy.
    advisory = ("never ", "do not ", "don't ", "avoid ", "be careful")
    cands = [(s["timestamp_start"], s["timestamp_end"], s["instruction"])
             for s in steps
             if not s["instruction"].strip().lower().startswith(advisory)]

    print(f"\nscored against {os.path.basename(ref_path)} "
          f"({len(ref)} reference steps, {len(steps)} predicted)")
    hits = []
    for k in sorted(ref):
        best, txt = 0.0, "— none —"
        for a, b, t in cands:
            v = iou(ref[k], (a, b))
            if v > best:
                best, txt = v, t
        hits.append(best)
        print(f"  step {k:>2} {ref[k][0]:6.1f}-{ref[k][1]:<6.1f} IoU {best:4.2f}  {txt[:56]}")
    for thr in (0.1, 0.3, 0.5):
        n = sum(1 for v in hits if v >= thr)
        print(f"  recall @ IoU>={thr}: {n}/{len(ref)} ({100 * n / len(ref):.0f}%)")
    matched = [v for v in hits if v > 0]
    if matched:
        print(f"  mean IoU over matched steps: {sum(matched) / len(matched):.2f}")
    print("  (recall only — a predicted step with no reference match is not "
          "scored as\n   a false positive; see sample_data/README.md)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", help="path to an .mp4 under ~5 minutes with audio")
    ap.add_argument("--blueprint", default=BLUEPRINT_DEFAULT,
                    help=f"blueprint id or key (default {BLUEPRINT_DEFAULT}; pass "
                         f"'mga' for the active one, whose output is truncated)")
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=2048,
                    help="inert on the active blueprint, which hardcodes 256")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="NEVER specify an output format here — the results parser "
                         "owns the template and will return zero steps")
    ap.add_argument("--name", default="manual generation run")
    ap.add_argument("--output", default="mga-output.jsonl")
    ap.add_argument("--score", metavar="JSONL", help="offline: score an output")
    ap.add_argument("--reference", metavar="CSV",
                    help="reference steps: <step>,<start_sec>,<end_sec>")
    ap.add_argument("--show", metavar="JSONL", help="offline: print a manual")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.show:
        show(args.show)
        return
    if args.score:
        if not args.reference:
            sys.exit("--score needs --reference")
        show(args.score)
        score(args.score, args.reference)
        return
    if not args.video:
        ap.error("--video is required (or use --show / --score)")

    _, endpoint = env()
    values = {"max_frames": args.max_frames,
              "max_new_tokens": args.max_new_tokens,
              "prompt": args.prompt}

    if args.dry_run:
        print(f"POST {endpoint}/agents/bundles")
        print(json.dumps({"blueprint": args.blueprint, "name": args.name,
                          "values": values}, indent=2))
        print(f"POST {endpoint}/agents/bundles/<id>/run")
        print(json.dumps({"connectors": {"source": [
            {"type": "file", "id": os.path.basename(args.video),
             "format": "mp4"}]}}, indent=2))
        return

    if not os.path.exists(args.video):
        sys.exit(f"no video at {args.video}")

    # Preflight BEFORE uploading: an ignored value should surface in seconds, not
    # after 15 minutes of GPU.
    bp = api("GET", f"{endpoint}/agents/blueprints/{args.blueprint}")
    print(f"blueprint {bp['id']} (key={bp['blueprint_key']}, active={bp['is_active']})")
    if not bp["is_active"]:
        print("  NOTE: superseded blueprint — usable for diagnosis, cannot ship.")
    ignored = inert_values(bp, values)
    if ignored:
        print(f"  WARNING: {ignored} not wired on this blueprint; accepted and ignored.")
        for k in ignored:
            values.pop(k)

    print(f"uploading {args.video} ...")
    file_id = upload(args.video)
    print(f"  file_id={file_id}")

    # No `artifacts` map: the mga blueprint pins newton-fusion and whisper itself.
    bundle = api("POST", f"{endpoint}/agents/bundles",
                 body={"blueprint": args.blueprint, "name": args.name,
                       "values": values})
    print(f"  bundle_id={bundle['id']}  status={bundle.get('status')}")

    agent = api("POST", f"{endpoint}/agents/bundles/{bundle['id']}/run",
                body={"connectors": {"source": [
                    {"type": "file", "id": file_id, "format": "mp4"}]}})
    print(f"started run\n  agent_id={agent['id']}\n"
          f"  watching /logs (~7.5 min of model loading first) ...")

    verdict = watch(agent["id"])
    print(f"\nrun {verdict}")
    saved = fetch_results(agent["id"], args.output)
    if saved:
        show(saved)
        if args.reference:
            score(saved, args.reference)
    if verdict == "failed":
        sys.exit(f"run failed — full log: "
                 f"{endpoint}/agents/instances/{agent['id']}/logs")


if __name__ == "__main__":
    main()
