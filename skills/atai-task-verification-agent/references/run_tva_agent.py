#!/usr/bin/env python3
"""Run the managed Task Verification Agent (TVA) over the Agents API.

Stdlib only — no third-party imports, no GPU, no atai_core. Drop a .env next to
where you run this:

    ATAI_API_KEY=<your API key>
    ATAI_API_ENDPOINT=https://platform.u1.archetypeai.app

    # verify a recording against a procedure (a batch session)
    python3 run_tva_agent.py --video assembly.mp4 --sop sop.txt

    # print the exact payloads and stop — no API calls, no key needed
    python3 run_tva_agent.py --video assembly.mp4 --sop sop.txt --dry-run

    # load once, then one /query per clip (a serving deployment, released on exit)
    python3 run_tva_agent.py --serve --video a.mp4 b.mp4 --sop sop.txt

    # delete the sessions, deployments and packages this script recorded
    python3 run_tva_agent.py --cleanup          # list
    python3 run_tva_agent.py --cleanup --yes    # delete

    # score or re-read an output offline
    python3 run_tva_agent.py --score out.json
    python3 run_tva_agent.py --score out.json --labels 1_pass_2_pass_3_fail

THE LIFECYCLE (Agent Lifecycle 2.0). For one setting — max_frames, size,
max_new_tokens — the script finds or makes, in order, and records in
tva-lifecycle.json so the next run at that setting reuses them:

    project     "TVA skill"            found by name, else created
    copy        tva-f64-s224-t5760     `tva` copied into the project, the values set on it
    package     of that copy           its models resolved once
    deployment  batch (or serving)     of that package

then opens a SESSION per clip. A package, a deployment and a batch session take no
values (a batch session given them is a 422), so the values live on the copy: one
copy per setting.

WHAT MAKES TVA DIFFERENT FROM MGA: the reference procedure arrives AT RUNTIME.
The `tva` blueprint wires `source.text -> prepare.in` (PrepareSOPNode), so a session
gets TWO source inputs — the video and the SOP as a `.txt`. MGA's blueprint sets
`text_extensions: []` and takes its instruction from a value, fixed for the life of
whatever carries it; TVA's SOP travels with each session, and ONE DEPLOYMENT SERVES
EVERY SOP. MultiModalSource routes the two inputs by `format`, not by position.

THE FAILURE THAT COSTS THE MOST: the token budget spent inside the model's <think>
block. The session still ends `completed`, with `results: []` in its output, and a
session has no /logs to say why. Count the verdicts every time; this script exits
non-zero when there are none.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.stdout.reconfigure(line_buffering=True)

# A session's terminal statuses.
TERMINAL = {"completed", "failed", "cancelled"}

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

# Default output budget, always set on the copy. A plain copy runs at the blueprint's
# own default (8192 since 2026-10-15), so runs through this script stay comparable to
# the measurements in SKILL.md only if it is sent. It was chosen in 2026-08 when 5760
# was the only budget returning rows on a 21-step SOP; a 12-cell sweep on 2026-08-20
# found no budget-dependent difference on the shipped 3-step SOP, so treat this as a
# stable default rather than a safe one. See SKILL.md, "The output budget is shared
# with the reasoning block".
DEFAULT_MAX_NEW_TOKENS = 5760

# Where a project is expected to keep its procedure: a `sop/` directory beside the
# script is picked up with no flag. When it is absent — a fresh skill checkout —
# fall back to the SOP shipped in sample_data, and say which one is in force rather
# than guessing silently.
DEFAULT_SOP_PATH = "sop/oring-numbered.txt"
SAMPLE_SOP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sample_data", "oring-numbered.txt")

PROJECT = "TVA skill"
STATE_PATH = "tva-lifecycle.json"

# A serving query: the whole request is capped at 2 MiB (the video goes inline as
# base64), each query has a 58 s budget on the platform, and a request past this is lost.
SERVING_BODY_CAP = 2 * 1024 * 1024
QUERY_TIMEOUT_S = 120
# A serving deployment took 693 s to become ready on 2026-11-06; leave room.
SERVING_READY_S = 1800


def resolve_sop(path: str | None) -> str:
    """The SOP to run: an explicit --sop, else sop/oring-numbered.txt, else the sample."""
    if path:
        return path
    if os.path.exists(DEFAULT_SOP_PATH):
        print(f"  --sop not given -> {DEFAULT_SOP_PATH}")
        return DEFAULT_SOP_PATH
    print(f"  --sop not given and no {DEFAULT_SOP_PATH} here -> the shipped sample\n"
          f"    {SAMPLE_SOP_PATH}")
    return SAMPLE_SOP_PATH


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def api_root(endpoint: str) -> str:
    """Return the deployment root, with any /api or /vX.Y suffix removed.

    Accept every form so one .env works for every skill in this repo: the
    model skills ship ATAI_API_ENDPOINT with /v0.5, the agent skills without,
    and the console shows it bare.
    """
    endpoint = re.sub(r"/api$", "", endpoint.rstrip("/"))
    return re.sub(r"/v[0-9]+(\.[0-9]+)*$", "", endpoint)


def env() -> tuple[str, str]:
    """(key, root). The endpoint is normalised to a bare deployment root.

    Every surface lives under one versionless /api prefix on it — /api/projects,
    /api/agents, /api/files. A user pasting the /query base URL would otherwise
    produce /v0.5/api/agents/... and 404 everywhere.
    """
    key = os.environ.get("ATAI_API_KEY")
    endpoint = (os.environ.get("ATAI_API_ENDPOINT") or "").rstrip("/")
    if not key:
        sys.exit("ATAI_API_KEY is not set (put it in .env)")
    if not endpoint:
        sys.exit("ATAI_API_ENDPOINT is not set (put it in .env)")
    return key, api_root(endpoint)


def agents_url(path: str = "") -> str:
    return f"{env()[1]}/api/agents{path}"


def request(method: str, url: str, body=None, raw: bool = False,
            allow_status: bool = False, timeout: int = 120, retries: int = 4):
    """Call the API. Returns the parsed JSON (or bytes with raw=True).

    allow_status=True returns (status, body) instead of exiting on an HTTP error:
    a 404 during cleanup is a finding, not a crash. Network errors are retried for
    GET and DELETE only — a repeated POST /sessions would start a second run.
    """
    key, _ = env()
    data = json.dumps(body).encode() if body is not None else None
    attempts = retries + 1 if method in ("GET", "DELETE") else 1
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {key}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
                out = payload if raw else json.loads(payload or b"null")
                return (resp.status, out) if allow_status else out
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if allow_status:
                return e.code, detail
            sys.exit(f"{method} {url} failed ({e.code}): {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == attempts - 1:
                if allow_status:
                    return 0, str(e)
                sys.exit(f"{method} {url} failed: {e}")
            wait = 5 * (2 ** attempt)
            print(f"  network error ({e}); retrying in {wait}s")
            time.sleep(wait)


def run_suffix() -> str:
    """`<UTC timestamp>-<4 hex>`, generated ONCE per run and shared by its inputs.

    An org shares ONE FLAT file namespace and `file_id` IS the basename, so two people
    running the same clip write to the same object. A session pins its inputs when it
    resolves them and can queue for an hour, so the second upload destroys the first's
    queued input — surfacing minutes later, inside a run that already started, as
    `S3 object not found`.

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

    A session's inputs arrive as ONE FLAT LIST of file ids with nothing saying which
    text belongs to which video, so the pipeline infers it from the names — by
    SUBSTRING, not equality: a `<stem>.txt` pairs with any video whose name CONTAINS
    `<stem>`. With a single SOP, as here, it applies to the video whether or not the
    names relate at all, so for a 1:1 run naming cannot break pairing.

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


def upload(path: str, name: str) -> str:
    """POST a file to <root>/api/files under `name`. Returns the file_id.

    The DECLARED Content-Type is enforced against a MIME allowlist, not the bytes:
    `video/mp4` for the clip, `text/plain` for the SOP. The returned `file_id` is
    the BASENAME — that is what connectors.source[].id wants, not the `fil_...`
    `file_uid` in the same response.

    Every upload here carries a per-run <UTC>-<hex> suffix, so the name never
    already exists and nothing queued can be overwritten.
    """
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}")
    key, root = env()
    ctype = "video/mp4" if name.endswith(".mp4") else "text/plain"
    boundary = uuid.uuid4().hex
    with open(path, "rb") as fh:
        content = fh.read()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(), content,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(f"{root}/api/files", data=body, method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"upload of {path} failed ({e.code}): {e.read().decode(errors='replace')}")
    if not out.get("is_valid", True) or not out.get("file_id"):
        sys.exit(f"upload of {path} was rejected: {json.dumps(out)}")
    return out["file_id"]


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


def sop_text(steps: list[str]) -> str:
    return "\n".join(steps) + "\n"


def inert_values(doc: dict, values: dict) -> list[str]:
    """Which of `values` the blueprint will accept and silently ignore.

    A value is honoured only if it is declared in the blueprint's `values` AND
    referenced as ${values.<key>} by some node or connector. A copy refuses an
    undeclared key with a 400, but accepts a declared one that nothing references,
    so nothing in the response tells you. Reads the document, so it needs no
    maintenance.
    """
    wired = json.dumps({"nodes": doc.get("nodes"),
                        "connectors": doc.get("connectors")})
    return [k for k in values
            if k not in doc.get("values", {})
            or ("${values." + k + "}") not in wired]


def check_sink(doc: dict) -> str | None:
    """Warn if the blueprint's sink cannot be instantiated.

    The platform validates the graph AFTER downloading and loading both models, so
    a bad sink costs the full ~7-minute cold start and leaves no results.
    """
    fmt = ((doc.get("connectors") or {}).get("sink") or {}).get("config", {}).get("format")
    if fmt in KNOWN_BROKEN_SINK_FORMATS:
        return (f"sink format {fmt!r} has NO REGISTERED CONNECTOR here. This run "
                f"would load both models (~7 min) and then die at graph "
                f"instantiation with no results. Copy a blueprint whose sink is "
                f"one of {sorted(KNOWN_GOOD_SINK_FORMATS)}.")
    if fmt not in KNOWN_GOOD_SINK_FORMATS:
        return (f"sink format {fmt!r} has not been seen to work. If the session "
                f"fails at 'instantiating graph', this is why.")
    return None


# --------------------------------------------------------------------------
# lifecycle: project -> copy -> package -> deployment -> session
# --------------------------------------------------------------------------

def setting_key(values: dict, source: str = "tva") -> str:
    """`tva-f64-s224-t5760`: the copy's key, named after its values.

    A copy of anything but `tva` is prefixed with the source's last six characters,
    so two sources at one setting never share a key."""
    prefix = "tva" if source == "tva" else "tva-" + re.sub(r"[^a-z0-9]", "", source.lower())[-6:]
    return f"{prefix}-f{values['max_frames']}-s{values['size']}-t{values['max_new_tokens']}"


def copy_body(project_id: str, key: str, values: dict) -> dict:
    """A copy takes `overrides.values`: the one place left to set them."""
    return {"project_id": project_id, "blueprint_key": key, "name": f"TVA {key}",
            "overrides": {"values": values}}


def package_body(blueprint_id: str, key: str) -> dict:
    return {"blueprint_ids": [blueprint_id], "target": {"kind": "platform"}, "name": f"TVA {key}"}


def deployment_body(package_id: str, kind: str) -> dict:
    return {"package_id": package_id, "kind": kind}


def session_body(deployment_id: str, video_id: str | None = None,
                 sop_id: str | None = None) -> dict:
    """A batch session over one video AND its SOP, or a serving session with neither.

    Both inputs go in the same `connectors.source` list and MultiModalSource routes
    them by `format`: `mp4` reaches source.video, `txt` reaches source.text. Order
    does not matter. Omit the text input and the session starts, then the prompt
    generator waits on an instruction that never arrives. No `values`: a batch
    session given them is refused with a 422.
    """
    body = {"deployment_id": deployment_id}
    if video_id:
        body["connectors"] = {"source": [
            {"type": "file", "id": video_id, "format": "mp4"},
            {"type": "file", "id": sop_id, "format": "txt"},
        ]}
    return body


def query_body(video: bytes, sop: str) -> dict:
    """A serving query: the SOP as `query`, the video inline."""
    return {"model": "tv", "query": sop, "file_ids": ["base64:" + base64.b64encode(video).decode()]}


def load_state() -> dict:
    """tva-lifecycle.json, per deployment root: ids made on one root don't exist on another."""
    state = json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else {}
    mine = state.setdefault(env()[1], {})
    mine.setdefault("settings", {})
    mine.setdefault("sessions", {})
    return state


def save_state(state: dict) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, STATE_PATH)


def _alive(url: str) -> dict | None:
    status, obj = request("GET", url, allow_status=True)
    return obj if status == 200 and isinstance(obj, dict) else None


def wait_for(url: str, want: str, waiting: str, timeout_s: int, every_s: int = 5) -> dict:
    """Poll while the status is `waiting`; exit unless it then reaches `want`."""
    t0 = time.time()
    obj = request("GET", url)
    while obj["status"] == waiting:
        if time.time() - t0 > timeout_s:
            sys.exit(f"{obj['id']} still {waiting} after {timeout_s} s")
        time.sleep(every_s)
        obj = request("GET", url)
    if obj["status"] != want:
        sys.exit(f"{obj['id']} is {obj['status']}, not {want}: {obj.get('error')}")
    return obj


def project_id(state: dict) -> str:
    mine = state[env()[1]]
    if "project" not in mine:
        root = env()[1]
        found = [p for p in request("GET", f"{root}/api/projects?limit=1000")["data"]
                 if p["name"] == PROJECT]
        mine["project"] = found[0]["id"] if found else request(
            "POST", f"{root}/api/projects",
            body={"name": PROJECT, "description": "Task Verification (TVA) skill runs"})["id"]
        save_state(state)
        print(f"  project {mine['project']} ({PROJECT})")
    return mine["project"]


def agent(values: dict, kind: str, source: str = "tva") -> dict:
    """The setting's record, with a `ready` deployment of `kind` in it.

    Reuses whatever tva-lifecycle.json holds that still answers, and rebuilds only
    what is gone. Every id is recorded BEFORE anything waits on it, so --cleanup
    finds a package or deployment whose client died mid-wait.
    """
    key = setting_key(values, source)
    state = load_state()
    pid = project_id(state)
    rec = state[env()[1]]["settings"].setdefault(key, {"source": source, "values": values})

    if not rec.get("blueprint") or not _alive(agents_url(f"/blueprints/{rec['blueprint']}")):
        # A key in a project resolves only with ?project_id=.
        found = request("GET", agents_url(f"/blueprints?project_id={pid}&key={key}"))["data"]
        if found:
            rec["blueprint"] = found[0]["id"]
            print(f"  reusing copy {key} ({rec['blueprint']})")
        else:
            rec["blueprint"] = request("POST", agents_url(f"/blueprints/{source}/copy"),
                                       body=copy_body(pid, key, values))["id"]
            print(f"  copied {source} as {key} ({rec['blueprint']}) with {json.dumps(values)}")
        rec.pop("package", None)
        save_state(state)

    pkg = _alive(agents_url(f"/packages/{rec['package']}")) if rec.get("package") else None
    if not pkg or pkg["status"] not in ("ready", "building"):
        pkg = request("POST", agents_url("/packages"), body=package_body(rec["blueprint"], key))
        rec["package"] = pkg["id"]
        for k in ("batch", "serving"):
            rec.pop(k, None)
        save_state(state)
        print(f"  package {pkg['id']} building (models resolved once) ...")
    if pkg["status"] != "ready":
        wait_for(agents_url(f"/packages/{rec['package']}"), "ready", "building", 900)

    dpl = _alive(agents_url(f"/deployments/{rec[kind]}")) if rec.get(kind) else None
    if not dpl or dpl["status"] not in ("ready", "deploying"):
        dpl = request("POST", agents_url("/deployments"), body=deployment_body(rec["package"], kind))
        rec[kind] = dpl["id"]
        save_state(state)
        print(f"  {kind} deployment {dpl['id']}"
              + (" (places and loads the agent on a GPU worker: ~12 min)" if kind == "serving" else ""))
    if dpl["status"] != "ready":
        wait_for(agents_url(f"/deployments/{rec[kind]}"), "ready", "deploying",
                 SERVING_READY_S if kind == "serving" else 900, every_s=10)
    return {"key": key, **rec}


def record_session(session_id: str, **info) -> None:
    """Note a session as soon as it exists, for --cleanup."""
    state = load_state()
    state[env()[1]]["sessions"][session_id] = {
        **info, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save_state(state)


def local_time(stamp) -> str:
    """An event's UTC `created_at` as local HH:MM:SS, like the script's own lines."""
    try:
        return datetime.datetime.fromisoformat(str(stamp).replace("Z", "+00:00")) \
            .astimezone().strftime("%H:%M:%S")
    except ValueError:
        return str(stamp)[11:19]


def print_events(session_id: str, seen: set) -> None:
    """Print the session's events not printed yet. /events is newest first."""
    status, body = request("GET", agents_url(f"/sessions/{session_id}/events?limit=100"),
                           allow_status=True)
    if status != 200 or not isinstance(body, dict):
        return
    for ev in reversed(body.get("data") or []):
        marker = f"{ev.get('created_at')}|{ev.get('message')}"
        if marker not in seen:
            seen.add(marker)
            print(f"  {local_time(ev.get('created_at'))} "
                  f"{str(ev.get('level', 'info')):7} {ev.get('message')}")


def poll(session_id: str, timeout_s: int = 3600, interval_s: int = 20) -> str:
    """Poll a batch session until it ends, printing its events. Returns its status
    (`completed`, `failed`, `cancelled`) or `timeout`.

    A session has no /logs, only /events: opened, dispatched, failed with a reason —
    not the model-load timings and parser messages an instance's /logs carried. So
    a run whose budget went on reasoning still ends `completed`; count the verdicts.
    """
    seen: set[str] = set()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ses = request("GET", agents_url(f"/sessions/{session_id}"))
        print_events(session_id, seen)
        if ses["status"] in TERMINAL:
            if ses.get("error"):
                print(f"  {session_id}: {ses['error']}")
            return ses["status"]
        time.sleep(interval_s)
    return "timeout"


# The platform can report a run completed while its output file is still being
# written (a known issue). A partly written JSON document doesn't parse, so the
# output counts as complete only once it does.
OUTPUT_WAIT_S = 600


def wait_until_parses(save, out_path: str) -> bool:
    """save() downloads the output; again every 15 s while it isn't valid JSON."""
    deadline = time.time() + OUTPUT_WAIT_S
    while True:
        try:
            json.load(open(out_path, encoding="utf-8"))
            return True
        except (ValueError, UnicodeDecodeError):
            pass
        if time.time() > deadline:
            print(f"  WARNING: the output still isn't valid JSON {OUTPUT_WAIT_S // 60} min "
                  f"after the run finished; rerun to download it again")
            return False
        print("  run finished; output still being written, downloading again in 15 s")
        time.sleep(15)
        save()


ARTIFACT_WAIT_S = 600          # how long to wait for artifacts after a session reports completed


def artifact_status(item):
    """An output's own completion status, or None if the platform reports none.

    A run reports `completed` while its artifacts are still being written. One
    measured instance: an output held 16,018 of an eventual 50,547 bytes at the
    moment its run reported completed -- 318 rows of ~997. Reading it gives a
    well-formed file that is simply short, so nothing downstream notices.
    """
    return (item.get("data", {}).get("file_attributes", {})
                .get("job_output", {}).get("status"))


def complete_results(fetch, wait_s=ARTIFACT_WAIT_S):
    """The session's outputs, once every one reports complete."""
    deadline = time.time() + wait_s
    waited = False
    while True:
        items = fetch()
        pending = [i for i in items if artifact_status(i) not in (None, "complete")]
        if not pending:
            if waited:
                print("  every artifact reports complete")
            return items
        if time.time() > deadline:
            print(f"  WARNING {len(pending)} artifact(s) still incomplete after "
                  f"{wait_s // 60} min; collecting anyway")
            return items
        waited = True
        print(f"  {len(items) - len(pending)} of {len(items)} artifacts ready; waiting")
        time.sleep(15)


def result_url(ref: str, root: str) -> str:
    """Where a result `ref` resolves. A session's ref is a ROOT path,
    `/api/files/<fil_…>/content`, which already carries /api: appended to an
    `…/api` endpoint it becomes /api/api/… and 404s. An absolute ref is used as is."""
    if ref.startswith("http"):
        return ref
    return f"{root}{ref}" if ref.startswith("/api/") else f"{root}/api{ref}"


def fetch_results(session_id: str, out_path: str) -> bytes | None:
    """Save the session's output. Returns the bytes, or None when there are none."""
    key, root = env()
    items = complete_results(
        lambda: request("GET", agents_url(f"/sessions/{session_id}/results")).get("data") or [])
    if not items:
        print("  no results (a failed session leaves none)")
        return None
    if len(items) > 1:
        print(f"  WARNING: {len(items)} result files returned; saving only the first written")
    inner = items[-1].get("data") or {}           # newest first: the last is the first written
    ref = inner.get("ref")
    if not ref:
        print("  results carried no download ref")
        return None
    url = result_url(ref, root)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def save() -> None:
        req = urllib.request.Request(url)
        if not ref.startswith("http"):            # a presigned URL must not get the bearer header
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=300) as resp, open(out_path, "wb") as fh:
            fh.write(resp.read())

    save()
    wait_until_parses(save, out_path)
    return open(out_path, "rb").read()


def end(kind: str, obj_id: str) -> bool:
    """Cancel a session still going, or stop a deployment still up, then delete it;
    a package is deleted as it is. True once it is gone (a 404 counts).

    Never send DELETE to a session expecting a no-op: it removes the session and its
    results. Cancel is /cancel."""
    url = agents_url(f"/{kind}/{obj_id}")
    status, obj = request("GET", url, allow_status=True)
    if status == 404:
        return True
    if status == 200 and isinstance(obj, dict):
        if kind == "sessions" and obj.get("status") not in TERMINAL:
            request("POST", f"{url}/cancel", body={}, allow_status=True)
        if kind == "deployments" and obj.get("status") not in ("stopped", "failed"):
            request("POST", f"{url}/stop", body={}, allow_status=True)
    status, _ = request("DELETE", url, raw=True, allow_status=True)
    return 200 <= status < 300 or status == 404


def forget(ids: set) -> None:
    """Drop deleted ids from tva-lifecycle.json."""
    state = load_state()
    mine = state[env()[1]]
    for rec in mine["settings"].values():
        for k in ("package", "batch", "serving"):
            if rec.get(k) in ids:
                rec.pop(k)
    mine["sessions"] = {s: x for s, x in mine["sessions"].items() if s not in ids}
    save_state(state)


def cleanup(yes: bool) -> int:
    """Sessions, then deployments, then packages: each holds the next.

    Starts from each recorded package and lists its deployments and their sessions,
    so ones this script never recorded are found too. The project and its copies
    stay; the next run at a setting packages its copy again."""
    state = load_state()
    mine = state[env()[1]]
    targets = []
    for key, rec in mine["settings"].items():
        pkg = rec.get("package")
        if not pkg or not _alive(agents_url(f"/packages/{pkg}")):
            continue
        dpls = [d["id"] for d in request(
            "GET", agents_url(f"/deployments?package_id={pkg}&limit=1000"))["data"]]
        targets += [("sessions", s["id"]) for d in dpls for s in request(
            "GET", agents_url(f"/sessions?deployment_id={d}&limit=1000"))["data"]]
        targets += [("deployments", d) for d in dpls] + [("packages", pkg)]
    for kind in ("sessions", "deployments", "packages"):
        print(f"  {sum(t[0] == kind for t in targets)} {kind}")
    if mine.get("project"):
        print(f"stays: project {mine['project']} ({PROJECT}) and its copies of tva "
              f"({', '.join(mine['settings']) or 'none'})")
    if not targets:
        print("nothing recorded is still on the platform")
        return 0
    if not yes:
        print("nothing deleted: rerun with --cleanup --yes")
        return 0
    gone = {obj for kind, obj in targets if end(kind, obj)}
    forget(gone)
    print(f"deleted {len(gone)} of {len(targets)}")
    return 0 if len(gone) == len(targets) else 1


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
    real one — which neither the session status nor the HTTP layer will do.
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
            print("  NO VERDICTS. The session reported completed and returned an\n"
                  "  empty results list. The usual cause is the token budget being\n"
                  "  spent inside the model's <think> block, so nothing was ever\n"
                  "  emitted. A session has no /logs to confirm it. Change\n"
                  "  --max-new-tokens (one step, either way) and re-run: see SKILL.md,\n"
                  "  'The output budget is shared with the reasoning block'.")
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


def dry_run(args, values: dict, steps: list[str]) -> None:
    """Print every request body in order, with no API call and no key needed."""
    endpoint = os.environ.get("ATAI_API_ENDPOINT")
    root = api_root(endpoint) if endpoint else "{root}"
    key = setting_key(values, args.blueprint)
    print(f"once per setting (reused while {STATE_PATH} holds it):\n")
    print(f"POST {root}/api/projects            # unless a project named {PROJECT!r} exists")
    print(json.dumps({"name": PROJECT, "description": "Task Verification (TVA) skill runs"}, indent=2))
    print(f"\nPOST {root}/api/agents/blueprints/{args.blueprint}/copy")
    print(json.dumps(copy_body("<the project>", key, values), indent=2))
    print(f"\nPOST {root}/api/agents/packages     # then poll until ready")
    print(json.dumps(package_body("<the copy>", key), indent=2))
    kind = "serving" if args.serve else "batch"
    print(f"\nPOST {root}/api/agents/deployments  # then poll until ready")
    print(json.dumps(deployment_body("<the package>", kind), indent=2))
    if args.serve:
        print(f"\nPOST {root}/api/agents/sessions     # then poll until running")
        print(json.dumps(session_body("<the deployment>"), indent=2))
        for video in args.video:
            size = os.path.getsize(video) if os.path.exists(video) else 0
            over = size * 4 / 3 > SERVING_BODY_CAP
            print(f"\nPOST {root}/api/agents/sessions/<id>/query   # {os.path.basename(video)}"
                  + ("  ** over the 2 MiB request cap after base64: re-encode it smaller"
                     if over else ""))
            print(json.dumps({"model": "tv", "query": sop_text(steps),
                              "file_ids": [f"base64:<{size:,} bytes of mp4>"]}, indent=2))
        return
    print("\nper clip:")
    for video in args.video:
        v_name, s_name = pair_names(video, args.sop, run_suffix())
        print(f"\nPOST {root}/api/agents/sessions")
        print(f"# stems match ({os.path.splitext(v_name)[0]}): the -<UTC>-<hex> tail keeps "
              f"concurrent users from\n# overwriting each other's inputs. Generated per run, "
              f"so the REAL run will differ.")
        print(json.dumps(session_body("<the deployment>", v_name, s_name), indent=2))
    print(f"\nSOP ({len(steps)} steps, one per line):")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")


def preflight(source: str, values: dict, force: bool) -> None:
    """Read the source blueprint before anything is uploaded or built."""
    bp = request("GET", agents_url(f"/blueprints/{source}"))
    doc = bp.get("document") or {}
    print(f"blueprint {bp['id']} (key={bp.get('blueprint_key')}, active={bp.get('is_active')})")
    if not bp.get("is_active"):
        print("  NOTE: this blueprint is superseded.")
    inert = inert_values(doc, values)
    for k, v in sorted(values.items()):
        print(f"  {k:16} {str(v):8} -> "
              f"{'NOT WIRED — ignored' if k in inert else 'honoured'}"
              f"   (blueprint default: {doc.get('values', {}).get(k)})")
    sink_warning = check_sink(doc)
    if sink_warning:
        print(f"  ** {sink_warning}")
        if not force:
            sys.exit("refusing to spend ~7 min of GPU on a run that cannot "
                     "produce output. Pass --force to do it anyway.")


def run_batch(args, values: dict, steps: list[str]) -> None:
    """One batch session per clip on the setting's batch deployment."""
    rec = agent(values, "batch", source=args.blueprint)
    print(f"  {rec['key']}: copy {rec['blueprint']}  package {rec['package']}  "
          f"deployment {rec['batch']}")
    failed = []
    for video in args.video:
        clip = os.path.splitext(os.path.basename(video))[0]
        out_path = args.output if args.output and len(args.video) == 1 \
            else f"tva-output-{clip}.json"
        print(f"\n{clip}: uploading ...")
        v_name, s_name = pair_names(video, args.sop, run_suffix())
        video_id = upload(video, v_name)
        sop_id = upload(args.sop, s_name)
        print(f"  video={video_id}  sop={sop_id}")

        session_id = request("POST", agents_url("/sessions"),
                             body=session_body(rec["batch"], video_id, sop_id))["id"]
        record_session(session_id, setting=rec["key"], kind="batch", clip=clip, out=out_path)
        print(f"session {session_id}\n  watching its status and /events "
              f"(~7 min of model loading first) ...")
        try:
            status = poll(session_id)
        except KeyboardInterrupt:
            # Never leave a run on the GPU: an abandoned run blocks everyone behind it.
            print(f"\ninterrupted — cancelling {session_id}")
            request("POST", agents_url(f"/sessions/{session_id}/cancel"), body={},
                    allow_status=True)
            sys.exit(130)
        print(f"\nsession {status}")
        payload = fetch_results(session_id, out_path)
        n = 0
        if payload is not None:
            print(f"  saved {out_path} ({len(payload)} bytes)")
            n = show(out_path)
        if status != "completed" or n == 0:
            failed.append(f"{clip}: {status}" + ("" if n else ", no verdicts"))
    if failed:
        sys.exit("\n".join(failed) + f"\n  events: {agents_url('/sessions/<id>/events')}")


def run_serving(args, values: dict, steps: list[str]) -> None:
    """A serving deployment and one session on it, one /query per clip, then release.

    The deployment holds the GPU worker, not the session: both are cancelled /
    stopped and deleted on exit, Ctrl-C included, unless --keep."""
    sop = sop_text(steps)
    bodies = {}
    for video in args.video:
        with open(video, "rb") as fh:
            body = query_body(fh.read(), sop)
        size = len(json.dumps(body).encode())
        if size > SERVING_BODY_CAP:
            sys.exit(f"{video}: the query would be {size:,} bytes, over the 2 MiB cap. "
                     f"Re-encode it smaller first, e.g.\n  ffmpeg -i {video} -vf scale=640:480 "
                     f"-c:v libx264 -crf 28 -an small.mp4")
        bodies[video] = body

    t0 = time.time()
    key = setting_key(values, args.blueprint)
    session = deployment = None
    try:
        rec = agent(values, "serving", source=args.blueprint)
        deployment = rec["serving"]
        print(f"  serving deployment {deployment} ready after {time.time() - t0:.0f} s")
        session = request("POST", agents_url("/sessions"), body=session_body(deployment))["id"]
        record_session(session, setting=rec["key"], kind="serving")
        wait_for(agents_url(f"/sessions/{session}"), "running", "pending", SERVING_READY_S)
        print(f"  session {session} running")
        seen: set[str] = set()
        for video, body in bodies.items():
            clip = os.path.splitext(os.path.basename(video))[0]
            t = time.time()
            for attempt in range(4):
                code, answer = request("POST", agents_url(f"/sessions/{session}/query"),
                                       body=body, allow_status=True, timeout=QUERY_TIMEOUT_S)
                if code == 200 or not (code >= 500 or code == 0):
                    break
                print_events(session, seen)
                print(f"  {code or 'no answer'}; retrying in 15 s")
                time.sleep(15)
            if code == 409:
                sys.exit(f"the session no longer answers (409): {str(answer)[:200]}")
            if code != 200:
                sys.exit(f"query failed ({code}): {str(answer)[:300]}")
            rows = (answer.get("response") or {}).get("data")
            out_path = args.output if args.output and len(args.video) == 1 \
                else f"tva-serve-{clip}.json"      # never over a batch run's output
            with open(out_path, "w") as fh:
                # batch's {"id", "results"} document, one line, so --score reads both
                fh.write(json.dumps({"id": clip, "results": rows or []}) + "\n")
            print(f"\n{clip}: {time.time() - t:.1f} s -> {out_path}")
            show(out_path)
    finally:
        if args.keep:
            print(f"\nleft running: session {session}, deployment {deployment} — it holds a "
                  f"GPU worker until --cleanup --yes")
        else:
            # a deployment that failed to start (no capacity) is recorded but never returned
            deployment = deployment or load_state()[env()[1]]["settings"].get(key, {}).get("serving")
            gone = {i for kind, i in (("sessions", session), ("deployments", deployment))
                    if i and end(kind, i)}
            forget(gone)
            print(f"\nreleased: {', '.join(sorted(gone))} ({time.time() - t0:.0f} s in all)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", nargs="+", help="the recording(s) to verify (.mp4)")
    ap.add_argument("--sop", default=None,
                    help=f"reference procedure, ONE STEP PER LINE (.txt). Defaults to "
                         f"{DEFAULT_SOP_PATH} if it exists, else the SOP shipped in "
                         f"references/sample_data/.")
    ap.add_argument("--blueprint", default="tva",
                    help="the blueprint to copy into the project, by id or key "
                         "(default: tva). Check the sink — the canonical version has "
                         "shipped un-instantiable.")
    ap.add_argument("--max-frames", type=int, default=64,
                    help="frames uniformly sampled across the whole video (1..64)")
    ap.add_argument("--size", type=int, default=224,
                    help="each frame is resized to size x size (224..2048)")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help=f"output budget, SHARED with the model's <think> block "
                         f"(default {DEFAULT_MAX_NEW_TOKENS})")
    ap.add_argument("--serve", action="store_true",
                    help="a serving deployment: load once, one /query per --video, then "
                         "release it")
    ap.add_argument("--keep", action="store_true",
                    help="with --serve: leave the session and deployment up (holds a GPU)")
    ap.add_argument("--cleanup", action="store_true",
                    help="list the sessions, deployments and packages recorded in "
                         f"{STATE_PATH}; delete them with --yes")
    ap.add_argument("--yes", action="store_true", help="with --cleanup: delete")
    ap.add_argument("--output", default=None,
                    help="where to save the result JSON, for one --video (default "
                         "tva-output-<clip>.json, or tva-serve-<clip>.json with --serve)")
    ap.add_argument("--score", metavar="FILE",
                    help="print and score an existing output, then stop — offline")
    ap.add_argument("--labels", metavar="1_pass_2_fail_3_pass",
                    help="ground truth for --score, if not in the filename or id")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact payloads and stop — no API calls")
    ap.add_argument("--force", action="store_true",
                    help="run even if the sink cannot be instantiated")
    args = ap.parse_args()

    load_dotenv()

    if args.score:
        show(args.score, args.labels)
        return
    if args.cleanup:
        sys.exit(cleanup(args.yes))
    if not args.video:
        sys.exit("--video is required (or --score an existing output, or --cleanup)")
    args.sop = resolve_sop(args.sop)
    steps = load_sop(args.sop)
    values = {"max_frames": args.max_frames, "size": args.size,
              "max_new_tokens": args.max_new_tokens}

    if args.dry_run:
        dry_run(args, values, steps)
        return

    preflight(args.blueprint, values, args.force)
    print(f"\n  SOP in force ({len(steps)} steps):")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s}")
    print(f"\nthe agent for {setting_key(values, args.blueprint)} ...")
    if args.serve:
        run_serving(args, values, steps)
    else:
        run_batch(args, values, steps)


if __name__ == "__main__":
    main()
