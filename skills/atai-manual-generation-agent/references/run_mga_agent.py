#!/usr/bin/env python3
"""Run the managed Manual Generation agent end to end, and score its output.

Built on the official archetypeai python client (`pip install -r
requirements.txt`). Put a .env next to where you run:

    ATAI_API_KEY=<your API key>
    ATAI_API_ENDPOINT=https://api.u1.archetypeai.app   # /v0.5 suffix optional

    python3 run_mga_agent.py --video procedure.mp4
    python3 run_mga_agent.py --video procedure.mp4 --blueprint mga    # active blueprint
    python3 run_mga_agent.py --dry-run --video procedure.mp4
    python3 run_mga_agent.py --score out.json --reference steps.csv   # offline

Budget ~15 minutes per run: ~7.5 min of model download and load happens on EVERY
run (nothing is cached), then roughly 2.5x realtime processing. Run one at a
time. Concurrent submissions queue only when other workloads hold the workers;
when they don't, they come up as concurrent pods and contend — one was SIGKILLed
mid-load. Other tenants' work is invisible to you, so neither outcome is
predictable and there is no serialization to rely on.

SIX THINGS THAT WILL BITE YOU (verified end to end; the same video reproduces an identical 18-step manual run to run)

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

CHECK WHICH OF YOUR VALUES ARE REAL

Twice in five days the setting that decided whether the manual was usable was one
the blueprint did not expose — max_new_tokens (hardcoded at 256), then prompt
(hardcoded to "10 steps or less"). Both are exposed again today. The lesson is
not either defect but that an unwired value is accepted with HTTP 201 and echoed
straight back at you, so it costs a 15-minute run to notice. This script
preflights every value and warns.

max_new_tokens is SHARED WITH THE REASONING BLOCK. The model reasons before it
answers, so if generation ends inside that block you get no manual, reported as
success: job.completed, no ERROR row, `results: []`, 38 bytes. ALWAYS COUNT YOUR
STEPS.

There is no threshold to memorise. A 2026-08-13 measurement on a 173 s video read
0 steps at 2048/4096 and 18 at 16384/32768/65536; the same video on 2026-08-20
returned the full 18-step manual at EVERY budget from 2048 to 65536, all
18/18 identical in content (five different md5s — deterministic in content, not
in bytes). Treat the older numbers as dated, keep the empty-output check, and see
SKILL.md, "`max_new_tokens` and the reasoning block".

file_id IS THE FILENAME, and it is a mutable pointer in an ORG-WIDE namespace.
Uploading the same name again repoints it and orphans the previous object. A run
resolves inputs at submit time but does not fetch the bytes until ~7.5 min of
model loading later, so a colleague uploading the same filename in that window
kills your run with `S3 object not found` — blaming storage, not the overwrite.
upload_name() suffixes every upload for this reason.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid

try:
    from archetypeai import ArchetypeAI
except ModuleNotFoundError:  # the only third-party dependency
    sys.exit("This runner needs the official Archetype AI client:\n"
             "    pip install -r requirements.txt   (from this directory)\n"
             "    pip install archetypeai           (or just the package)")

sys.stdout.reconfigure(line_buffering=True)

_CLIENT = None


def client() -> "ArchetypeAI":
    """The official client, built once from the environment."""
    global _CLIENT
    if _CLIENT is None:
        key, endpoint = env()
        _CLIENT = ArchetypeAI(key, api_endpoint=versioned(endpoint))
    return _CLIENT


def versioned(endpoint: str) -> str:
    """Return the endpoint in the form the client expects: WITH the /vX.Y suffix.

    The client uses api_endpoint verbatim for the files API (/v0.5/files) and
    strips the version itself for the versionless agents API. Passing a bare
    root therefore breaks uploads while bundle calls keep working — an empty
    `ApiError: {}` that points at nothing.

    Accept either form so one .env works for every skill in this repo: the
    model skills ship ATAI_API_ENDPOINT with /v0.5, the agent skills without.
    """
    endpoint = endpoint.rstrip("/")
    return endpoint if re.search(r"/v[0-9]+(\.[0-9]+)*$", endpoint) else f"{endpoint}/v0.5"

# Target the blueprint KEY, never an id.
#
# A key resolves to whatever is canonical and active. A pinned id does not survive
# republication, which is frequent — and a superseded id fails LATE: reading it,
# bundling and submitting all succeed, then the pod dies about a second in with
# `resolving blueprint: invalid config for 1 node(s)`.
#
# Pin an id only to reproduce a specific past run.
BLUEPRINT_DEFAULT = "mga"

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


def upload_name(path: str) -> str:
    """A collision-proof name to upload under: <stem>-<UTC stamp>-<4 hex>.

    file_id IS the filename and it is a mutable pointer in an ORG-WIDE namespace,
    so a plain basename lets anyone else in your org destroy your in-flight run
    (and you theirs). The stem keeps it recognisable, the stamp sorts, the random
    tail is what actually guarantees uniqueness.
    """
    stem, ext = os.path.splitext(os.path.basename(path))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stem}-{stamp}-{uuid.uuid4().hex[:4]}{ext or '.mp4'}"


def upload(path: str, name: str | None = None) -> str:
    """Upload the video and return its file_id.

    The client uploads under the file's own basename, so an explicit name means
    staging a copy first — which is how every run gets a unique, timestamped
    name (file ids ARE filenames, so re-using one replaces the record and
    orphans the object any in-flight run already resolved).
    """
    if name and name != os.path.basename(path):
        staged = os.path.join(tempfile.mkdtemp(), name)
        shutil.copyfile(path, staged)
        path = staged
    return client().files.local.upload(path)["file_id"]


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
        rows = sorted(client().agents.instances.get_logs(agent_id, limit=500)
                      .get("data", []),
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
    items = client().agents.instances.get_results(agent_id).get("data") or []
    if not items:
        print("  no results (a failed run leaves none)")
        return None
    inner = items[0].get("data") or {}
    name = inner.get("filename") or (inner.get("ref") or "").rsplit("/", 1)[-1]
    if not name:
        print("  results carried no download ref")
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    client().files.local.download(name, out_path)
    print(f"  saved {out_path} ({os.path.getsize(out_path)} bytes)")
    return out_path


def load_manual(path: str) -> list[dict]:
    """Steps from an MGA output file.

    MGA writes ONE JSON DOCUMENT per run — the results metadata reports
    `file_extension: "json"` and the body parses as a single object. Line-delimited
    bodies are still accepted so older files keep working.
    """
    raw = open(path, encoding="utf-8").read().strip()
    if not raw:
        return []
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        doc = [json.loads(l) for l in raw.splitlines() if l.strip()]
    rec = doc[0] if isinstance(doc, list) else doc
    return rec.get("results", [])


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
                    help=f"blueprint id or KEY (default {BLUEPRINT_DEFAULT!r}). A key "
                         f"tracks whatever is canonical and active; a pinned id can "
                         f"stop resolving when the blueprint is republished")
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=16384,
                    help="output budget (default 16384, matching the blueprint). A "
                         "FLOOR, not a ceiling: the model reasons out of this budget, "
                         "so 4096 returns an EMPTY manual rather than a short one, and "
                         "nothing above 16384 changes the answer")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="the instruction, honoured as ${values.prompt}; "
                         "the preflight warns if that stops being true. "
                         "Say what to COVER, never how to format — the "
                         "results parser owns the template and a format-overriding "
                         "prompt makes it return zero steps")
    ap.add_argument("--name", default=None,
                    help="bundle name. Defaults to 'mga <video> mnt<budget>', which "
                         "is what distinguishes your runs from each other in the "
                         "console — bundles CANNOT be renamed later (PATCH/PUT "
                         "return 405).")
    ap.add_argument("--output", default="mga-output.json")
    ap.add_argument("--score", metavar="FILE", help="offline: score an output")
    ap.add_argument("--reference", metavar="CSV",
                    help="reference steps: <step>,<start_sec>,<end_sec>")
    ap.add_argument("--show", metavar="FILE", help="offline: print a manual")
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

    # A bundle's name is set at creation and cannot be changed afterwards, so it
    # is the only handle that tells two runs apart in the console. Derive it from
    # the inputs that actually differ between runs. Done before --dry-run prints
    # the payload, so the dry run shows exactly what a real run would send.
    if args.name is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        args.name = f"mga {stem} mnt{args.max_new_tokens}"

    if args.dry_run:
        print(f"POST {endpoint}/agents/bundles")
        print(json.dumps({"blueprint": args.blueprint, "name": args.name,
                          "values": values}, indent=2))
        print(f"POST {endpoint}/agents/bundles/<id>/run")
        print(json.dumps({"connectors": {"source": [
            {"type": "file", "id": upload_name(args.video),
             "format": "mp4"}]}}, indent=2))
        return

    if not os.path.exists(args.video):
        sys.exit(f"no video at {args.video}")

    # Preflight BEFORE uploading: an ignored value should surface in seconds, not
    # after 15 minutes of GPU.
    bp = client().agents.blueprints.get(args.blueprint)
    # NOT include_yaml=True: the published client accepts the kwarg and the
    # call then fails with an empty `ApiError: {}`. `document` is returned
    # regardless, which is all the preflight below needs.
    print(f"blueprint {bp['id']} (key={bp['blueprint_key']}, active={bp['is_active']})")
    if not bp["is_active"]:
        print("  WARNING: superseded blueprint. It will read back, bundle and start "
              "normally, then the pod may die ~1s in with 'resolving blueprint: "
              "invalid config'. Target the key 'mga' instead.")
    ignored = inert_values(bp, values)
    print(f"  honoured: {[k for k in values if k not in ignored]}")
    if ignored:
        print(f"  WARNING: {ignored} not wired on this blueprint; accepted and ignored.")
        if "prompt" in ignored:
            print("    The blueprint's own instruction applies instead:\n"
                  f"    {bp['document']['connectors']['source']['config'].get('default_text')!r}")
        for k in ignored:
            values.pop(k)

    print(f"uploading {args.video} ...")
    file_id = upload(args.video, upload_name(args.video))
    print(f"  file_id={file_id}")

    # No `artifacts` map: the mga blueprint pins newton-fusion and whisper itself.
    bundle = client().agents.bundles.create(
        blueprint=args.blueprint, name=args.name, values=values)
    print(f"  bundle_id={bundle['id']}  status={bundle.get('status')}")

    agent = client().agents.bundles.run(
        bundle["id"], source=[{"type": "file", "id": file_id, "format": "mp4"}])
    print(f"started run\n  agent_id={agent['id']}\n"
          f"  watching /logs (~7.5 min of model loading first) ...")

    verdict = watch(agent["id"])
    print(f"\nrun {verdict}")
    saved = fetch_results(agent["id"], args.output)
    if saved:
        # A run can complete with results: [] — no ERROR row, no failed status.
        # Seen at max_new_tokens 2048; 4096 on the same input produced 10 steps.
        n = sum(len(json.loads(l).get("results", []))
                for l in open(saved) if l.strip())
        if n == 0:
            print("\n  WARNING: the run completed but produced ZERO steps. This is "
                  "not reported as a failure anywhere. Retry with a larger "
                  "--max-new-tokens before assuming the video is at fault.")
        show(saved)
        if args.reference:
            score(saved, args.reference)
    if verdict == "failed":
        sys.exit(f"run failed — full log: "
                 f"{endpoint}/agents/instances/{agent['id']}/logs")


if __name__ == "__main__":
    main()
