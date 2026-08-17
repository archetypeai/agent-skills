#!/usr/bin/env python3
"""Run the managed Task Verification Agent (TVA) over the Agent API.

Stdlib only — no third-party imports, no GPU, no atai_core. Drop a .env next to
where you run this:

    ATAI_API_KEY=<dev API key>
    ATAI_API_ENDPOINT=https://api.dev.u1.archetypeai.app

    # verify a recording against a procedure
    python3 run_tva_agent.py --video assembly.mp4 --sop sop.txt

    # print the exact payloads and stop — no API calls, no key needed
    python3 run_tva_agent.py --video assembly.mp4 --sop sop.txt --dry-run

    # score or re-read an output offline
    python3 run_tva_agent.py --score out.json
    python3 run_tva_agent.py --score out.json --labels 1_pass_2_pass_3_fail

WHAT MAKES TVA DIFFERENT FROM MGA: the reference procedure arrives AT RUNTIME.
The `tva` blueprint wires `source.text -> prepare.in` (PrepareSOPNode), so a run
sends TWO source inputs — the video and the SOP as a `.txt`. MGA's blueprint sets
`text_extensions: []` and takes its instruction from a bundle value, so its
instruction is fixed for the life of a bundle; TVA's SOP travels with each run,
and ONE BUNDLE SERVES EVERY SOP.

MultiModalSource routes the two inputs by their `format` field, not by position.

THE TWO FAILURES THAT COST THE MOST, both verified on dev 2026-08-11:

  1. A blueprint whose SINK cannot be instantiated. The canonical `tva` was
     republished with `format: json/per-request`, for which no connector is
     registered, and every run dies at graph instantiation AFTER loading both
     models — 996 s for zero output, with `status: ready` on the bundle and the
     cause only in /logs. check_sink() below catches it in about a second.

  2. The token budget spent inside the model's <think> block. f1-0 reasons before
     answering, and the reasoning shares `max_new_tokens`. At 2048 a clip with a
     SKIPPED step produced `results: []` while reporting `job.completed` with no
     ERROR row. Clean clips fit in 2048; the ones containing a defect do not — so
     the failure correlates with the inputs you care about. Default here is 8192.

Both are silent at the HTTP layer. Neither is visible in the `status` field.
"""
from __future__ import annotations

import argparse
import hashlib
import datetime
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

# Log event types that end a run. The instance `status` field is NOT usable for
# this — it has read `running` 20+ minutes after a pod died with exit=1.
TERMINAL_EVENTS = ("pod.terminated", "job.completed", "job.failed", "job.canceled")

# The statuses TaskVerificationResultsParserNode emits.
STATUSES = ("PASSED", "FAILED", "MISSING")

# Sink formats observed to produce output. BOTH are good.
#
# An earlier version of this file listed `json/per-request` as broken, because the
# canonical blueprint shipped with it during an ~18-hour window when no connector was
# registered for it. That was fixed on 2026-08-12, and the denylist then did real
# damage: it refused every run against the WORKING canonical blueprint while sounding
# certain. A preflight built on a measured constant goes stale in the worst direction.
#
# So: warn on an unrecognised format, never refuse, and say the list may be stale.
KNOWN_GOOD_SINK_FORMATS = {"jsonl/per-request", "json/per-request"}
KNOWN_BROKEN_SINK_FORMATS: set[str] = set()

# Default output budget. Deliberately NOT the ceiling and NOT the blueprint default
# (16384): the failure mode is repetition, so a bigger budget is more room to LOOP.
# Measured on a clip with two skipped steps: 5760 returned correct verdicts, 8192
# returned results:[], 16384 returned results:[] identically. See SKILL.md.
DEFAULT_MAX_NEW_TOKENS = 5760

# Where a project is expected to keep its procedure. Matches the worked example's
# layout, so a `sop/` directory beside the script is picked up with no flag. When it
# is absent — a fresh skill checkout — fall back to the SOP shipped in sample_data,
# and say which one is in force rather than guessing silently.
DEFAULT_SOP_PATH = "sop/oring-numbered.txt"
_ENDPOINT_NOTED: set[str] = set()   # env() runs per API call; warn once
BUNDLED_SOP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sample_data", "oring-numbered.txt")


def resolve_sop(path: str | None) -> str:
    """The SOP to run: an explicit --sop, else sop/oring-numbered.txt, else the sample."""
    if path:
        return path
    if os.path.exists(DEFAULT_SOP_PATH):
        print(f"  --sop not given -> {DEFAULT_SOP_PATH}")
        return DEFAULT_SOP_PATH
    print(f"  --sop not given and no {DEFAULT_SOP_PATH} here -> the bundled sample\n"
          f"    {BUNDLED_SOP_PATH}")
    return BUNDLED_SOP_PATH


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def env() -> tuple[str, str]:
    """(key, endpoint). The endpoint is normalised to bare — no /v0.5 suffix.

    The Agent API is mounted WITHOUT a version prefix while files live under
    /v0.5, so this script mounts each itself. A user pasting the /query base URL
    would otherwise produce /v0.5/agents/... and 404 everywhere.
    """
    load_dotenv()
    key = os.environ.get("ATAI_API_KEY")
    endpoint = (os.environ.get("ATAI_API_ENDPOINT") or "").rstrip("/")
    if not key:
        sys.exit("ATAI_API_KEY is not set (put it in .env)")
    if not endpoint:
        sys.exit("ATAI_API_ENDPOINT is not set (put it in .env)")
    endpoint = re.sub(r"/v\d+(\.\d+)?$", "", endpoint).rstrip("/")
    if ("api.dev" not in endpoint and "api.stage" not in endpoint
            and endpoint not in _ENDPOINT_NOTED):
        _ENDPOINT_NOTED.add(endpoint)
        print(f"  NOTE: {endpoint} is not the Dev or Staging endpoint. The "
              f"/agents API is verified on those two; Prod returns 404 for "
              f"every /agents path.")
    return key, endpoint


def api(method: str, url: str, body=None, raw: bool = False, retries: int = 4,
        allow_status: bool = False):
    """Call the API, retrying only NETWORK errors. An HTTP status is an answer.

    Runs are polled for up to an hour, and a DNS hiccup should not kill the
    client while the job carries on burning GPU unattended.
    """
    key, _ = env()
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {key}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
                # 202 is success when starting a run — see start_run().
                if not allow_status and resp.status not in (200, 201, 202):
                    sys.exit(f"{method} {url} unexpected status {resp.status}")
                out = payload if raw else json.loads(payload or b"null")
                return (resp.status, out) if allow_status else out
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if allow_status:
                return e.code, detail
            sys.exit(f"{method} {url} failed ({e.code}): {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                if allow_status:
                    return 0, str(e)
                sys.exit(f"{method} {url} failed after {retries + 1} attempts: {e}")
            wait = 5 * (2 ** attempt)
            print(f"  network error ({e}); retrying in {wait}s [{attempt + 1}/{retries}]")
            time.sleep(wait)


def remote_bytes(file_id: str) -> bytes | None:
    """The object currently stored under `file_id`, or None if absent."""
    _, endpoint = env()
    status, payload = api("GET", f"{endpoint}/v0.5/files/download/{file_id}",
                          raw=True, allow_status=True)
    return payload if status == 200 and isinstance(payload, bytes) else None


def run_suffix() -> str:
    """`<UTC timestamp>-<4 hex>`, generated ONCE per run and shared by its inputs.

    An org shares ONE FLAT file namespace and `file_id` IS the basename, so two people
    running the same clip write to the same object. A run pins its inputs at
    input-resolution time and dev can queue for an hour, so the second upload destroys
    the first's queued run — surfacing minutes later, inside a run that already started,
    as `S3 object not found` with `job.completed` on the job.

    Each part earns its place:
      the original stem  recognisable in an org-wide file list
      the UTC timestamp  sorts and greps; which upload was yours, and when
      4 hex              what actually guarantees it — two people starting in the same
                         second is precisely the case being fixed
    """
    return (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-" + uuid.uuid4().hex[:4])


def pair_names(video: str, sop: str, suffix: str | None = None) -> tuple[str, str]:
    """The ids to upload under, as (video, sop). Stems MATCH and name the SOP.

    A run's inputs arrive as ONE FLAT LIST of file ids with nothing saying which text
    belongs to which video, so the pipeline infers it from the names — by SUBSTRING,
    not equality: a `<stem>.txt` pairs with any video whose name CONTAINS `<stem>`.
    With a single SOP, as here, it applies to the video whether or not the names
    relate at all, so for a 1:1 run naming cannot break pairing.

    It still earns its keep. Including the SOP's stem records which procedure a run
    was checked against — the only such record, since the platform's file list holds
    only whatever was uploaded last. And `suffix` (see run_suffix) makes the pair
    unique across CONCURRENT USERS, which is the part that prevents lost runs.

    BOTH halves get the same suffix — a per-file suffix would break the very pairing
    this exists to protect.
    """
    v = os.path.splitext(os.path.basename(video))[0]
    s = os.path.splitext(os.path.basename(sop))[0]
    stem = f"{v}-{s}" + (f"-{suffix}" if suffix else "")
    return f"{stem}.mp4", f"{stem}.txt"


def upload(path: str, rename: str | None = None) -> str:
    """POST a file to /v0.5/files, streamed from disk. Returns the file_id.

    The DECLARED Content-Type is enforced against a MIME allowlist, not the bytes:
    `video/mp4` for the clip, `text/plain` for the SOP. The returned `file_id` is
    the BASENAME — that is what connectors.source[].id wants, not the `fil_...`
    `file_uid` in the same response.

    IDEMPOTENT: if an object of the same name already holds the same bytes, this
    SKIPS the upload. That is not an optimisation. A run pins its inputs at
    input-resolution time and dev can queue for an hour, so re-uploading the same
    name in that window kills whatever is already queued — it fails minutes later,
    inside the run, as `S3 object not found` with `job.completed` on the job.
    """
    _, endpoint = env()
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}")
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    boundary = uuid.uuid4().hex
    name = rename or os.path.basename(path)
    local = open(path, "rb").read()
    if remote_bytes(name) == local:
        print(f"  {name}: identical object already on the platform — skipping upload")
        return name
    preamble = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
    ])
    epilogue = f"\r\n--{boundary}--\r\n".encode()
    with open(path, "rb") as fh:
        body = _ChainedReader([preamble, fh, epilogue])
        req = urllib.request.Request(f"{endpoint}/v0.5/files", data=body,
                                     method="POST")
        req.add_header("Authorization", f"Bearer {env()[0]}")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Content-Length",
                       str(len(preamble) + os.path.getsize(path) + len(epilogue)))
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read())["file_id"]
        except urllib.error.HTTPError as e:
            sys.exit(f"upload of {path} failed ({e.code}): "
                     f"{e.read().decode(errors='replace')}")


class _ChainedReader:
    """A read()-able concatenating bytes chunks and file objects, so a large
    video is streamed rather than held in memory twice."""

    def __init__(self, parts):
        self._parts = list(parts)

    def read(self, amt: int = -1) -> bytes:
        out = b""
        while self._parts and (amt < 0 or len(out) < amt):
            want = -1 if amt < 0 else amt - len(out)
            head = self._parts[0]
            chunk = head[:want] if isinstance(head, (bytes, bytearray)) else head.read(want)
            if isinstance(head, (bytes, bytearray)):
                self._parts[0] = head[len(chunk):]
                if not self._parts[0]:
                    self._parts.pop(0)
            elif not chunk:
                self._parts.pop(0)
            out += chunk
        return out


def load_sop(path: str) -> list[str]:
    """SOP steps, one per line. Blank lines and # comments are dropped.

    PrepareSOPNode "accumulates the SOP step lines into a single prepared
    instruction", so THE LINE BREAKS ARE THE STEP BOUNDARIES and nothing else is.
    A wrapped line silently becomes two steps.
    """
    if not os.path.exists(path):
        sys.exit(f"no SOP file at {path} — write one step per line")
    with open(path) as fh:
        steps = [l.strip() for l in fh
                 if l.strip() and not l.lstrip().startswith("#")]
    if not steps:
        sys.exit(f"{path} has no step lines")
    return steps


def inert_values(doc: dict, values: dict) -> list[str]:
    """Which of `values` the blueprint will accept and silently ignore.

    A value is honoured only if it is declared in the blueprint's `values` AND
    referenced as ${values.<key>} by some node or connector. Setting an unwired
    value returns 201 and echoes it straight back, so nothing in the response
    tells you. Reads the document, so it needs no maintenance.
    """
    wired = json.dumps({"nodes": doc.get("nodes"),
                        "connectors": doc.get("connectors")})
    return [k for k in values
            if k not in doc.get("values", {})
            or ("${values." + k + "}") not in wired]


def check_sink(doc: dict) -> str | None:
    """Warn if the blueprint's sink cannot be instantiated. See module docstring.

    This is the cheapest check in the file and the one that saves the most: the
    platform validates the graph AFTER downloading and loading both models, so a
    bad sink costs the full ~7-minute cold start and leaves no results.
    """
    fmt = ((doc.get("connectors") or {}).get("sink") or {}).get("config", {}).get("format")
    if fmt in KNOWN_BROKEN_SINK_FORMATS:
        return (f"sink format {fmt!r} has NO REGISTERED CONNECTOR on dev. This run "
                f"would load both models (~7 min) and then die at graph "
                f"instantiation with no results. Target a blueprint whose sink is "
                f"one of {sorted(KNOWN_GOOD_SINK_FORMATS)}.")
    if fmt not in KNOWN_GOOD_SINK_FORMATS:
        return (f"sink format {fmt!r} has not been seen to work. If the run dies "
                f"at 'instantiating graph', this is why.")
    return None


def start_run(bundle_id: str, video_id: str, sop_id: str) -> str:
    """Start a run with BOTH inputs. Returns the agent id.

    Both go in the same `connectors.source` list and MultiModalSource routes them
    by `format`: `mp4` reaches source.video, `txt` reaches source.text. Order does
    not matter. Omit the text input and the run starts, then the prompt generator
    waits on an instruction that never arrives.

    Returns 202, not 201.
    """
    _, endpoint = env()
    agent = api("POST", f"{endpoint}/agents/bundles/{bundle_id}/run",
                body={"connectors": {"source": [
                    {"type": "file", "id": video_id, "format": "mp4"},
                    {"type": "file", "id": sop_id, "format": "txt"},
                ]}})
    return agent["id"]


def poll(agent_id: str, timeout_s: int = 3600, interval_s: int = 20) -> str:
    """Stream /logs until terminal. Returns completed / failed / timeout, or
    'empty-reasoning-overflow' when the parser reports it never got an answer.

    Polls /logs, NOT /events — /events carries only "run started" and
    "dispatched to JOS". Ignores the `status` field entirely.
    """
    _, endpoint = env()
    seen: set[str] = set()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rows = sorted(
            api("GET", f"{endpoint}/agents/instances/{agent_id}/logs?limit=500")
            .get("data", []), key=lambda r: str(r.get("created_at")))
        for r in rows:
            marker = f"{r.get('created_at')}|{r.get('event_type')}|{r.get('message')}"
            if marker in seen:
                continue
            seen.add(marker)
            print(f"  {str(r.get('created_at'))[11:19]} "
                  f"{str(r.get('level')):7} {str(r.get('event_type')):26} "
                  f"{r.get('message')}")
        if rows and str(rows[-1].get("event_type")) in TERMINAL_EVENTS:
            if any(str(r.get("level")) == "ERROR" for r in rows):
                return "failed"
            # A WARN naming the reasoning block means the budget was spent before
            # any verdict was emitted. The job still reports completed and
            # /results still returns a record — with an empty results list.
            for r in rows:
                if str(r.get("level")) == "WARN" and \
                        "reasoning block" in str(r.get("message", "")):
                    return "empty-reasoning-overflow"
            return "completed"
        time.sleep(interval_s)
    return "timeout"


def fetch_results(agent_id: str, out_path: str) -> bytes | None:
    """Save the first output. The `ref` is RELATIVE, resolves under /v0.5, and
    needs the bearer token; an absolute ref is presigned and must not get it.
    Run outputs do not expire, so this is re-fetchable later."""
    _, endpoint = env()
    results = api("GET", f"{endpoint}/agents/instances/{agent_id}/results")
    items = results.get("data") or []
    if not items:
        print("  no results (a failed run leaves none)")
        return None
    inner = items[0].get("data") or {}
    ref = inner.get("ref") or items[0].get("ref")
    if not ref:
        print("  results carried no download ref")
        return None
    if ref.startswith("http"):
        with urllib.request.urlopen(ref, timeout=300) as resp:
            payload = resp.read()
    else:
        payload = api("GET", f"{endpoint}/v0.5{ref}", raw=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(payload)
    return payload


def parse_labels(text: str) -> dict[int, bool]:
    """{step: performed_correctly} from `1_pass_2_fail_3_pass`, 1-based.

    A convention, not a platform feature: encoding ground truth in the filename
    keeps labels visible in `ls` and impossible to desynchronise from the media.
    """
    return {int(m.group(1)): m.group(2).lower() == "pass"
            for m in re.finditer(r"(\d+)_(pass|fail)", text, re.IGNORECASE)}


def step_number(row: dict) -> int | None:
    """The parser's `step`, converted to the 1-based numbering humans write.

    `step` is 0-BASED and undocumented: a 3-step SOP returns 0, 1, 2. A scorer
    that assumes 1-based fails QUIETLY — two steps still line up and the last
    reads as missing, which looks like a model that gave up rather than a join
    bug.
    """
    s = row.get("step")
    return None if s is None else int(s) + 1


def show(path: str, labels: str | None = None) -> int:
    """Print a verification report, and score it if labels are available.

    Returns the number of verdicts, so a caller can tell an empty run from a
    real one — which neither the job status nor the HTTP layer will do.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    records = [json.loads(l) for l in raw.decode(errors="replace").splitlines()
               if l.strip()]
    total_rows = 0
    for rec in records:
        rows = rec.get("results", []) or []
        total_rows += len(rows)
        print(f"\nverification report for {rec.get('id', '?')} — "
              f"{len(rows)} verdict(s)")
        if not rows:
            print("  NO VERDICTS. The run reported job.completed and returned an\n"
                  "  empty results list. The usual cause is the token budget being\n"
                  "  spent inside the model's <think> block, so nothing was ever\n"
                  "  emitted. Check /logs for 'generation ended inside the model's\n"
                  "  reasoning block' and raise --max-new-tokens. Clips where a step\n"
                  "  was SKIPPED need a bigger budget than clean ones.")
        for r in rows:
            # MISSING rows carry no timestamps by design — there is no interval to
            # point at for a step that never happened.
            ts, te = r.get("timestamp_start"), r.get("timestamp_end")
            span = f"[{ts:>6.1f} - {te:>6.1f}s]" if ts is not None and te is not None \
                else "[   no interval   ]"
            print(f"  step {step_number(r)} {span} {str(r.get('status')):8} "
                  f"{r.get('reason', '')}")

        truth = parse_labels(labels or str(rec.get("id") or "") or
                             os.path.basename(path))
        if truth:
            print("\n  scored against labels "
                  f"({'/'.join('%d=%s' % (k, 'pass' if v else 'fail') for k, v in sorted(truth.items()))}):")
            by_step = {step_number(r): r for r in rows}
            correct = 0
            for step, expected in sorted(truth.items()):
                row = by_step.get(step)
                status = str(row.get("status")).upper() if row else "NOT REPORTED"
                # PASSED is the only status meaning "done correctly". A step the
                # run never reported is a MISSING ANSWER, never a correct FAIL —
                # crediting it would score an empty output as perfect on every
                # fail-labelled clip.
                ok = row is not None and (status == "PASSED") == expected
                correct += ok
                print(f"    step {step}: expected {'PASS' if expected else 'FAIL':4} "
                      f" got {status:12} [{'OK' if ok else 'MISS'}]")
            print(f"  {correct}/{len(truth)} correct")
            if all(v for v in truth.values()):
                print("  NOTE: every step is labelled pass, so this clip cannot\n"
                      "  distinguish a working detector from one that always says\n"
                      "  PASSED. Score clips containing a skipped step too.")
    print(f"\nmd5 {hashlib.md5(raw).hexdigest()}  ({len(raw)} bytes)")
    return total_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", help="the recording to verify (.mp4)")
    ap.add_argument("--sop", default=None,
                    help=f"reference procedure, ONE STEP PER LINE (.txt). Defaults to "
                         f"{DEFAULT_SOP_PATH} if it exists, else the SOP bundled in "
                         f"references/sample_data/.")
    ap.add_argument("--blueprint", default="tva",
                    help="blueprint id or KEY (default: tva). A key resolves to "
                         "whatever is canonical and active, but check the sink — "
                         "the canonical version has shipped un-instantiable.")
    ap.add_argument("--max-frames", type=int, default=64,
                    help="frames uniformly sampled across the whole video "
                         "(blueprint default 16; 64 is the reader batch size)")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help=f"output budget, SHARED with the model's <think> block "
                         f"(default {DEFAULT_MAX_NEW_TOKENS}). Too low returns an "
                         f"empty result on exactly the clips that contain a defect.")
    ap.add_argument("--name", default="task verification run")
    ap.add_argument("--output", default=None, help="where to save the result JSON")
    ap.add_argument("--score", metavar="FILE",
                    help="print and score an existing output, then stop — offline")
    ap.add_argument("--labels", metavar="1_pass_2_fail_3_pass",
                    help="ground truth for --score, if not in the filename or id")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact payloads and stop — no API calls")
    ap.add_argument("--force", action="store_true",
                    help="run even if the sink cannot be instantiated")
    args = ap.parse_args()

    if args.score:
        show(args.score, args.labels)
        return
    if not args.video:
        sys.exit("--video is required (or use --score to read an existing output)")
    args.sop = resolve_sop(args.sop)

    steps = load_sop(args.sop)
    values = {"max_frames": args.max_frames,
              "max_new_tokens": args.max_new_tokens}
    clip = os.path.splitext(os.path.basename(args.video))[0]
    out_path = args.output or f"tva-output-{clip}.json"

    if args.dry_run:
        _, endpoint = env()
        print(f"POST {endpoint}/agents/bundles")
        print(json.dumps({"blueprint": args.blueprint, "name": args.name,
                          "values": values}, indent=2))
        print(f"\nPOST {endpoint}/agents/bundles/<id>/run")
        v_name, s_name = pair_names(args.video, args.sop, run_suffix())
        print(f"# stems match ({os.path.splitext(v_name)[0]}): the pipeline pairs a "
              f"video with its SOP by stem, and the -<UTC>-<hex> tail keeps concurrent "
              f"users from\n# overwriting each other's inputs. Generated per run, so "
              f"the REAL run will differ.")
        print(json.dumps({"connectors": {"source": [
            {"type": "file", "id": v_name, "format": "mp4"},
            {"type": "file", "id": s_name, "format": "txt"},
        ]}}, indent=2))
        print(f"\nSOP ({len(steps)} steps, one per line):")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")
        return

    _, endpoint = env()

    # Preflight BEFORE uploading. Both checks are free and each one catches a
    # failure that otherwise costs the full ~7-minute cold start.
    bp = api("GET", f"{endpoint}/agents/blueprints/{args.blueprint}")
    doc = bp["document"]
    print(f"blueprint {bp['id']} (key={bp.get('blueprint_key')}, "
          f"active={bp.get('is_active')})")
    if not bp.get("is_active"):
        print("  NOTE: this blueprint is superseded — results cannot ship.")
    inert = inert_values(doc, values)
    if inert:
        print(f"  WARNING: {inert} is not wired here and will be silently ignored.")
    for k, v in sorted(values.items()):
        print(f"  {k:16} {str(v):8} -> "
              f"{'NOT WIRED — ignored' if k in inert else 'honoured'}"
              f"   (blueprint default: {doc.get('values', {}).get(k)})")
    sink_warning = check_sink(doc)
    if sink_warning:
        print(f"  ** {sink_warning}")
        if not args.force:
            sys.exit("refusing to spend ~7 min of GPU on a run that cannot "
                     "produce output. Pass --force to do it anyway.")
    print(f"\n  SOP in force ({len(steps)} steps):")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s}")

    print(f"\nuploading ...")
    v_name, s_name = pair_names(args.video, args.sop, run_suffix())
    video_id = upload(args.video, rename=v_name)
    sop_id = upload(args.sop, rename=s_name)
    print(f"  stems match ({os.path.splitext(v_name)[0]}) — nothing else writes to "
          f"this pair")
    print(f"  video={video_id}  sop={sop_id}")

    # No `artifacts` map: the tva blueprint pins newton-fusion:1.0 and
    # whisper:large-v3 itself. Note the PLURAL /bundles — the singular 404s.
    bundle = api("POST", f"{endpoint}/agents/bundles",
                 body={"blueprint": args.blueprint, "name": args.name,
                       "values": values})
    print(f"bundle {bundle['id']} status={bundle.get('status')}")

    agent_id = start_run(bundle["id"], video_id, sop_id)
    print(f"run {agent_id}\n  watching /logs (~7 min of model loading first) ...")

    verdict = poll(agent_id)
    print(f"\nrun {verdict}")
    payload = fetch_results(agent_id, out_path)
    if payload is not None:
        print(f"  saved {out_path} ({len(payload)} bytes)")
        show(out_path)
    if verdict != "completed":
        sys.exit(f"run {verdict} — full log: "
                 f"{endpoint}/agents/instances/{agent_id}/logs")


if __name__ == "__main__":
    main()
