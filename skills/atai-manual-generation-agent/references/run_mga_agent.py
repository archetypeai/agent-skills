#!/usr/bin/env python3
"""Run the managed Manual Generation agent end to end, and score its output.

Stdlib only: no client library, nothing to pip install. Put a .env next to where
you run:

    ATAI_API_KEY=<your API key>
    ATAI_API_ENDPOINT=https://platform.u1.archetypeai.app   # /api optional

    python3 run_mga_agent.py --video procedure.mp4                  # batch: a session per video
    python3 run_mga_agent.py --video a.mp4 b.mp4 --serve            # serving: load once, a /query per video
    python3 run_mga_agent.py --dry-run --video procedure.mp4        # print every request, call nothing
    python3 run_mga_agent.py --cleanup                              # list what --cleanup --yes deletes
    python3 run_mga_agent.py --score out.json --reference steps.csv   # offline
    python3 run_mga_agent.py --score out.json --reference steps.csv --keywords kw.json

AGENT LIFECYCLE 2.0. A run is, inside a project:

    copy of `mga`   this run's max_frames, max_new_tokens and prompt set as overrides,
                    keyed mga-f<frames>-t<tokens>-p<md5(prompt)[:6]>, one per setting
    package         of that copy: its models resolved once; it carries no values
    deployment      of the package: `batch`, or `serving` (holds a GPU worker)
    session         batch: one per video; serving: queried over /query

The first three are made once per setting and reused by every later run at it;
--state (mga-lifecycle.json) keeps their ids, and every session's, the moment
each exists. `--cleanup --yes` deletes sessions, then deployments, then packages.
Only Archetype AI's dev deployment serves these calls so far; production, staging
and the free trial keep bundles and instances until their next platform release.
Check: `GET /api/agents/sessions?limit=1` answers 200, not 404.

Budget ~10 minutes per batch session (10.6 min on dev, 2026-11-05): ~7 min of
model download and load on EVERY session (nothing is cached), then roughly 1.1x
realtime processing. Run one at a time. A serving deployment took 881 s to start
(dev, 2026-11-06) and then answered a 173 s video in 27.8 s.

SEVEN THINGS THAT WILL BITE YOU

  1. Set the values on the copy. A package, a deployment and a batch session take
     none (a session given `values` is a 422), and a plain copy runs at the
     blueprint's defaults: max_new_tokens 8192 and an instruction that caps the
     manual at "10 steps or less" (dev, 2026-11). This script always sets all three.
  2. A copy refuses an undeclared value (400) but accepts a declared one that no
     node references, and ignores it. The preflight reads the blueprint and warns.
  3. A session has /events, not /logs. Events say what the platform did (opened,
     dispatched, failed with a reason); the model-load timings and parser warnings
     are not there. An empty manual still ends `completed`: count the steps.
  4. A session's result `ref` is a root path, `/api/files/<id>/content`. Appended
     to an .../api endpoint it 404s as /api/api/...
  5. A serving deployment holds a GPU worker until it is stopped, not the session.
     --serve stops and deletes it on every way out, Ctrl-C and kill included.
  6. Videos longer than ~5 minutes fail outright, several minutes in.
  7. file_id IS THE FILENAME, a mutable pointer in an ORG-WIDE namespace. Uploading
     the same name again repoints it and orphans the previous object. A session
     does not fetch the bytes until ~7 min of model loading later, so a colleague
     uploading the same filename in that window kills your run with
     `S3 object not found`. upload_name() suffixes every upload for this reason.

max_new_tokens is SHARED WITH THE REASONING BLOCK. The model reasons before it
answers, so if generation ends inside that block you get no manual, reported as
success: `completed`, no error, `results: []`, 38 bytes. ALWAYS COUNT YOUR STEPS.
There is no threshold to memorise; see SKILL.md, "`max_new_tokens` and the
reasoning block".
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime
import hashlib
import json
import mimetypes
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.stdout.reconfigure(line_buffering=True)

# Copy from the blueprint KEY, never an id.
#
# A key resolves to whatever is canonical and active. A pinned id does not survive
# republication, which is frequent — and a superseded id fails LATE: reading it
# succeeds, then the run dies about a second in with
# `resolving blueprint: invalid config for 1 node(s)`.
#
# Pin an id only to reproduce a specific past run.
BLUEPRINT_DEFAULT = "mga"
PROJECT_DEFAULT = "MGA manuals"
STATE_DEFAULT = "mga-lifecycle.json"
DEFAULT_MAX_NEW_TOKENS = 16384

# Says what to COVER, never how to FORMAT. ManualGenerationResultsParserNode owns
# the output template: a prompt that specifies its own format makes the parser
# return zero steps.
#
# Replaced 2026-10-27. The previous one ("Generate
# a concise, ordered list ... Use up to 20 steps. Keep each step to at most 15
# words ...") looped on a silent 80 s mouse-assembly clip on production: 16 steps,
# 2 distinct texts, "Take the battery out of the mouse." 15 times on a wired mouse.
# This one gave 17 distinct steps on the same clip, same values. On the 173 s tire
# video the old prompt's texts also sat ~5 s early (0/11 reference steps at
# IoU>=0.5 by text, against 6/11 here). "concise" plus a word cap drove the loop;
# naming the kinds of detail wanted and requiring each step to differ from the one
# before is what stops it. Both pairs are in sample_data/.
DEFAULT_PROMPT = (
    "Generate an ordered list of every distinct step performed in this video, in "
    "order. Include brief steps and steps that are repeated. Write each step as a "
    "direct, actionable instruction, as if telling the reader what to do. Do not "
    "narrate or describe what is happening in the video. Capture the details "
    "needed to repeat the action exactly: control and part names, printed and "
    "on-screen text quoted word for word, part numbers, model and serial numbers, "
    "tool types and sizes, quantities, and which way things move. For every "
    "component being handled, state where it is using grid terms such as "
    "top-left, top-right, bottom-left, bottom-right, center-left, or "
    "upper-center. Where several similar actions are performed, such as four "
    "screws on one panel, treat each as its own step and say what distinguishes "
    "it. Where the video shows how to tell the step worked, say that. Base every "
    "step on what is visible; use the spoken narration where there is any, and "
    "rely on the picture alone where the video is silent. Each step must describe "
    "a different action from the one before it — where two moments look alike, "
    "state what changed between them. Do not invent steps, parts, or details that "
    "are not shown or stated in the video."
)
# A session's terminal statuses.
TERMINAL = {"completed", "failed", "cancelled", "canceled"}
# A serving query carries the video inline, and the whole body is capped at 2 MiB.
QUERY_BODY_CAP = 2 * 1024 * 1024
QUERY_TIMEOUT_S = 120          # the platform's budget is 58 s; past this the request is lost
QUERY_RETRIES = 3


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = v.strip()
            # A quoted value would otherwise keep its quotes: `"https://…"` is not a URL.
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            os.environ.setdefault(k.strip(), v)


def api_root(endpoint: str) -> str:
    """Return the deployment root, with any /api or /vX.Y suffix removed.

    Accept every form so one .env works for every skill in this repo: the
    model skills ship ATAI_API_ENDPOINT with /v0.5, the agent skills without,
    and the console shows it bare.
    """
    endpoint = re.sub(r"/api$", "", endpoint.strip().rstrip("/"))
    return re.sub(r"/v[0-9]+(\.[0-9]+)*$", "", endpoint)


def root() -> str:
    """The deployment root. Every surface lives under {root}/api."""
    load_dotenv()
    endpoint = os.environ.get("ATAI_API_ENDPOINT") or ""
    if not endpoint.strip():
        sys.exit("set ATAI_API_ENDPOINT (see .env.example)")
    return api_root(endpoint)


def api_key() -> str:
    load_dotenv()
    key = os.environ.get("ATAI_API_KEY")
    if not key:
        sys.exit("set ATAI_API_KEY (see .env.example)")
    return key


def agents_url(path: str = "") -> str:
    """{root}/api/agents/<path>: blueprints, packages, deployments, sessions."""
    return f"{root()}/api/agents" + (f"/{path}" if path else "")


def request(method: str, url: str, body=None, *, raw: bool = False,
            allow_status: bool = False, timeout: int = 120):
    """Call the API with the bearer token.

    GET and DELETE retry network errors and 502/503/504 (a ten-minute poll should
    not die on one blip); a POST is sent once, since a repeat could build a second
    package. allow_status=True returns (status, body) instead of exiting on an
    HTTP error: a 404 is a finding, not a crash.
    """
    data = json.dumps(body).encode() if body is not None else None
    tries = 5 if method in ("GET", "DELETE") else 1
    for attempt in range(tries):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {api_key()}")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
                out = payload if raw else json.loads(payload or b"null")
                return (resp.status, out) if allow_status else out
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code not in (502, 503, 504) or attempt + 1 == tries:
                if allow_status:
                    return e.code, detail
                sys.exit(f"{method} {url} failed ({e.code}): {detail}")
            reason = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt + 1 == tries:
                if allow_status:
                    return 0, str(e)
                sys.exit(f"{method} {url} failed: {e}")
            reason = str(getattr(e, "reason", e))
        wait = 5 * 2 ** attempt
        print(f"  {reason}; retrying in {wait} s")
        time.sleep(wait)


def alive(url: str) -> dict | None:
    """The object at `url`, or None if it is gone."""
    status, obj = request("GET", url, allow_status=True)
    return obj if status == 200 and isinstance(obj, dict) else None


def list_all(url: str) -> list[dict]:
    """Every item of a paginated listing. The cursor is opaque: pass it back verbatim."""
    items, cursor = [], ""
    sep = "&" if "?" in url else "?"
    while True:
        page = request("GET", f"{url}{sep}limit=1000{cursor}")
        items += page.get("data") or []
        if not page.get("has_more"):
            return items
        cursor = f"&after={page['next_cursor']}"


def wait_for(url: str, want: str, waiting: str, timeout_s: int, every_s: int = 5) -> dict:
    """Poll while the status is `waiting`; exit unless it then reaches `want`.

    A package goes building -> ready, a deployment deploying -> ready, a serving
    session pending -> running."""
    t0 = time.time()
    while True:
        obj = request("GET", url)
        if obj["status"] == want:
            return obj
        if obj["status"] != waiting:
            sys.exit(f"{obj.get('id')} is {obj['status']}, not {want}: {obj.get('error')}")
        if time.time() - t0 > timeout_s:
            sys.exit(f"{obj.get('id')} still {waiting} after {timeout_s} s")
        time.sleep(every_s)


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


class _ChainedReader:
    """A read()-able that concatenates bytes and file objects, so a video streams
    from disk instead of being held in memory twice inside a multipart body."""

    def __init__(self, parts):
        self._parts = list(parts)

    def read(self, amt: int = -1) -> bytes:
        out = b""
        while self._parts and (amt < 0 or len(out) < amt):
            want = -1 if amt < 0 else amt - len(out)
            head = self._parts[0]
            if isinstance(head, bytes):
                chunk = head if want < 0 else head[:want]
                self._parts[0] = head[len(chunk):]
                if not self._parts[0]:
                    self._parts.pop(0)
            else:
                chunk = head.read(want)
                if not chunk:
                    self._parts.pop(0)
            out += chunk
        return out


def upload(path: str, name: str) -> str:
    """POST the video to {root}/api/files as multipart; return its file_id.

    The declared Content-Type is checked against an allowlist, not the bytes, so
    an .mp4 goes as video/mp4."""
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path)[0] or "video/mp4"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    with open(path, "rb") as fh:
        req = urllib.request.Request(f"{root()}/api/files", method="POST",
                                     data=_ChainedReader([head, fh, tail]))
        req.add_header("Authorization", f"Bearer {api_key()}")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Content-Length", str(len(head) + os.path.getsize(path) + len(tail)))
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read())["file_id"]
        except urllib.error.HTTPError as e:
            sys.exit(f"upload of {path} failed ({e.code}): {e.read().decode(errors='replace')}")


def inert_values(bp: dict, values: dict) -> list[str]:
    """Which of `values` will this blueprint silently ignore?

    Honoured means declared in `values` AND referenced as ${values.<key>} by some
    node or connector config. A copy refuses an undeclared key with a 400 but
    accepts a declared one that nothing references, so this is the only way to
    know before spending a run. Reading the document rather than hardcoding key
    names means this keeps working when a blueprint changes.
    """
    doc = bp["document"]
    wired = json.dumps({"nodes": doc.get("nodes"),
                        "connectors": doc.get("connectors")})
    return [k for k in values
            if k not in doc.get("values", {})
            or ("${values." + k + "}") not in wired]


# ---------------------------------------------------------------------------
# The lifecycle: project -> copy -> package -> deployment -> session
# ---------------------------------------------------------------------------

def setting_key(values: dict, source: str = BLUEPRINT_DEFAULT) -> str:
    """`mga-f64-t16384-pc647bb`: the copy's key, and the setting's name in the state file.

    The prompt is too long for a key, so it goes in as the first 6 hex of its md5.
    A copy of anything but `mga` is prefixed with the source's last six
    characters, so two sources at one setting never share a key."""
    prefix = "mga" if source == "mga" else "mga-" + re.sub(r"[^a-z0-9]", "", source.lower())[-6:]
    prompt = values.get("prompt")
    p = hashlib.md5(prompt.encode()).hexdigest()[:6] if prompt else "default"
    return (f"{prefix}-f{values.get('max_frames', 'd')}"
            f"-t{values.get('max_new_tokens', 'd')}-p{p}")


def copy_body(project_id: str, key: str, values: dict) -> dict:
    """Where a bundle's `values` went: a copy of the blueprint takes them as overrides."""
    return {"project_id": project_id, "blueprint_key": key, "name": f"MGA {key}",
            "overrides": {"values": values}}


def package_body(blueprint_id: str, key: str) -> dict:
    """One blueprint, for the platform. A package carries no values."""
    return {"blueprint_ids": [blueprint_id], "target": {"kind": "platform"}, "name": f"MGA {key}"}


def session_body(deployment_id: str, file_id: str | None = None) -> dict:
    """A batch session over one video, or (no file) a serving session.

    The video goes in `connectors.source` exactly as a bundle run took it. MGA
    takes no text input: the instruction is the copy's `prompt` value."""
    body = {"deployment_id": deployment_id}
    if file_id:
        body["connectors"] = {"source": [{"type": "file", "id": file_id, "format": "mp4"}]}
    return body


def load_state(path: str) -> dict:
    """The ids this script made, for one deployment of the platform."""
    state = json.load(open(path)) if os.path.exists(path) else {}
    if state.get("endpoint") and api_root(state["endpoint"]) != root():
        sys.exit(f"{path} records ids on {state['endpoint']}, not {root()}: "
                 f"pass another --state, or move it aside")
    state["endpoint"] = root()
    state.setdefault("settings", {})
    state.setdefault("sessions", {})
    return state


def save_state(state: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, path)


def find_project(state: dict, path: str, name: str) -> str:
    """The project, found by name or created; everything else lands in it."""
    if not state.get("project"):
        found = [p for p in list_all(f"{root()}/api/projects") if p.get("name") == name]
        state["project"] = found[0]["id"] if found else request(
            "POST", f"{root()}/api/projects",
            {"name": name, "description": "Manual Generation agent runs"})["id"]
        save_state(state, path)
        print(f"  project {state['project']} ({name})")
    return state["project"]


def ensure_deployment(values: dict, kind: str, args, state: dict) -> tuple[str, dict]:
    """The setting's record, with a `ready` deployment of `kind` in it.

    Reuses whatever the state file holds that still answers and rebuilds only
    what is gone. Every id is saved BEFORE anything waits on it, so --cleanup
    finds a package or deployment whose client died."""
    key = setting_key(values, args.blueprint)
    project = find_project(state, args.state, args.project)
    rec = state["settings"].setdefault(key, {"source": args.blueprint, "values": values})

    if not rec.get("blueprint") or not alive(agents_url(f"blueprints/{rec['blueprint']}")):
        found = request("GET", agents_url(f"blueprints?project_id={project}&key={key}")).get("data") or []
        if found:
            rec["blueprint"] = found[0]["id"]
            print(f"  reusing copy {key} ({rec['blueprint']})")
        else:
            rec["blueprint"] = request("POST", agents_url(f"blueprints/{args.blueprint}/copy"),
                                       copy_body(project, key, values))["id"]
            print(f"  copied {args.blueprint} as {key} ({rec['blueprint']})")
        rec.pop("package", None)
        save_state(state, args.state)

    pkg = alive(agents_url(f"packages/{rec['package']}")) if rec.get("package") else None
    if not pkg or pkg["status"] not in ("ready", "building"):
        if pkg:                                    # failed: don't strand it
            request("DELETE", agents_url(f"packages/{pkg['id']}"), raw=True, allow_status=True)
        pkg = request("POST", agents_url("packages"), package_body(rec["blueprint"], key))
        rec["package"] = pkg["id"]
        rec.pop("batch", None)
        rec.pop("serving", None)
        save_state(state, args.state)
        print(f"  package {pkg['id']} building ...")
    if pkg["status"] != "ready":
        wait_for(agents_url(f"packages/{rec['package']}"), "ready", "building", 900)

    dpl = alive(agents_url(f"deployments/{rec[kind]}")) if rec.get(kind) else None
    if not dpl or dpl["status"] not in ("ready", "deploying"):
        if dpl:
            end("deployments", dpl["id"])
        dpl = request("POST", agents_url("deployments"), {"package_id": rec["package"], "kind": kind})
        rec[kind] = dpl["id"]
        save_state(state, args.state)
        print(f"  {kind} deployment {dpl['id']}"
              + (" (placing and loading the agent on a GPU worker; ~15 min)" if kind == "serving" else ""))
    if dpl["status"] != "ready":
        wait_for(agents_url(f"deployments/{rec[kind]}"), "ready", "deploying", 1800, every_s=10)
    return key, rec


def open_session(deployment_id: str, args, state: dict, **info) -> str:
    session_id = request("POST", agents_url("sessions"), session_body(deployment_id, info.get("file_id")))["id"]
    state["sessions"][session_id] = {
        **info, "deployment": deployment_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save_state(state, args.state)
    return session_id


def end(kind: str, obj_id: str) -> bool:
    """Cancel a session still going, or stop a deployment still up, then delete it;
    a package is deleted as it is. True once it is gone (a 404 counts)."""
    status, obj = request("GET", agents_url(f"{kind}/{obj_id}"), allow_status=True)
    if status == 404:
        return True
    if status == 200 and kind == "sessions" and obj.get("status") not in TERMINAL:
        request("POST", agents_url(f"sessions/{obj_id}/cancel"), {}, allow_status=True)
    if status == 200 and kind == "deployments" and obj.get("status") not in ("stopped", "failed"):
        request("POST", agents_url(f"deployments/{obj_id}/stop"), {}, allow_status=True)
    status, _ = request("DELETE", agents_url(f"{kind}/{obj_id}"), raw=True, allow_status=True)
    return 200 <= status < 300 or status == 404


def local_time(stamp) -> str:
    """An event's UTC `created_at` as local HH:MM:SS, like the script's own lines."""
    try:
        return datetime.datetime.fromisoformat(str(stamp).replace("Z", "+00:00")) \
            .astimezone().strftime("%H:%M:%S")
    except ValueError:
        return str(stamp)[11:19]


def poll_session(session_id: str, timeout_s: int = 3600, every_s: int = 20) -> str:
    """Poll a batch session until it ends, printing its events as they appear.

    Returns `completed`, `failed`, `cancelled`, or `timeout`. A session has no
    /logs, only /events (newest first): what the platform did, not the model-load
    timings and parser warnings. An empty manual still ends `completed`."""
    seen: set[str] = set()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ses = request("GET", agents_url(f"sessions/{session_id}"))
        events = request("GET", agents_url(f"sessions/{session_id}/events?limit=100")).get("data") or []
        for ev in reversed(events):
            marker = f"{ev.get('created_at')}|{ev.get('message')}"
            if marker not in seen:
                seen.add(marker)
                print(f"  {local_time(ev.get('created_at'))} "
                      f"{str(ev.get('level', 'info')):7} {ev.get('message')}")
        if ses["status"] in TERMINAL:
            if ses.get("error"):
                print(f"  {session_id}: {ses['error']}")
            return ses["status"]
        time.sleep(every_s)
    return "timeout"


def result_url(ref: str) -> str:
    """A session result's `ref`: a root path `/api/files/<id>/content`, or absolute.

    Resolve it against the deployment root, not against .../api, or it 404s as
    /api/api/..."""
    if ref.startswith("http"):
        return ref
    return f"{root()}{ref}" if ref.startswith("/api/") else f"{root()}/api{ref}"


def fetch_ref(ref: str) -> bytes:
    url = result_url(ref)
    if not url.startswith(root()):                 # presigned: no bearer header
        with urllib.request.urlopen(url, timeout=300) as resp:
            return resp.read()
    return request("GET", url, raw=True, timeout=300)


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


def fetch_results(session_id: str, out_path: str) -> str | None:
    """Download the session's manual to out_path."""
    items = request("GET", agents_url(f"sessions/{session_id}/results")).get("data") or []
    if not items:
        print("  no results (a failed session leaves none)")
        return None
    ref = (items[-1].get("data") or {}).get("ref")     # newest first: the last is the first written
    if not ref:
        print("  results carried no download ref")
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def save():
        with open(out_path, "wb") as fh:
            fh.write(fetch_ref(ref))
    save()
    wait_until_parses(save, out_path)
    print(f"  saved {out_path} ({os.path.getsize(out_path)} bytes)")
    return out_path


def query_body(video: bytes, prompt: str) -> dict:
    """A serving query: the video inline, the prompt as `query` (the same body an
    instance query took)."""
    return {"model": "mga", "query": prompt,
            "file_ids": ["base64:" + base64.b64encode(video).decode()]}


class SessionGone(Exception):
    """The serving session no longer answers: cancelled, idle too long, or crashed."""


def ask(session_id: str, body: dict) -> dict:
    """POST /query, retrying 5xx and timeouts a few times (a cold agent can 504)."""
    for attempt in range(1, QUERY_RETRIES + 2):
        t = time.time()
        code, answer = request("POST", agents_url(f"sessions/{session_id}/query"), body,
                               allow_status=True, timeout=QUERY_TIMEOUT_S)
        if code == 200:
            print(f"  answered in {time.time() - t:.1f} s")
            return answer
        print(f"  {code or 'no answer'} after {time.time() - t:.0f} s: {str(answer)[:200]}")
        if code == 409:
            raise SessionGone(str(answer))
        if not (code == 0 or code >= 500) or attempt > QUERY_RETRIES:
            raise RuntimeError(f"query failed ({code}): {str(answer)[:300]}")
        time.sleep(15)


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


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
              "output\n  token cap. Raise --max-new-tokens.")


def score(out_path: str, ref_path: str, keywords_path: str | None = None) -> None:
    """Reference-step recall + temporal IoU. Offline; no API key needed.

    Deliberately no precision: a reference step list is typically a documented
    procedure rather than an inventory of the video, so a predicted step with no
    reference match is not necessarily wrong.

    Temporal IoU is blind to text: it credits whichever prediction overlaps the
    reference interval, whatever it says. A manual whose texts sit one step early
    still scores well — measured 2026-10-27, 7/11 at IoU>=0.5 by time and 0/11 once
    each reference step is matched to the prediction that actually names it. With
    keywords_path (JSON: {"<step>": ["phrase", ...]}) that text-matched score is
    reported too, and every time-only match whose text names nothing is flagged.
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

    words = ({int(k): [w.lower() for w in v] for k, v in json.load(open(keywords_path)).items()}
             if keywords_path else {})

    print(f"\nscored against {os.path.basename(ref_path)} "
          f"({len(ref)} reference steps, {len(steps)} predicted)")
    hits, text_hits, wrong_text = [], [], 0
    for k in sorted(ref):
        best, txt = 0.0, "— none —"
        for a, b, t in cands:
            v = iou(ref[k], (a, b))
            if v > best:
                best, txt = v, t
        hits.append(best)
        line = f"  step {k:>2} {ref[k][0]:6.1f}-{ref[k][1]:<6.1f} IoU {best:4.2f}  {txt[:56]}"
        if k in words:
            names = lambda t: any(w in t.lower() for w in words[k])
            text_hits.append(max((iou(ref[k], (a, b)) for a, b, t in cands if names(t)),
                                 default=0.0))
            if best > 0 and not names(txt):
                wrong_text += 1
                line += f"\n{'':28}^ text names a different action; text-matched IoU {text_hits[-1]:.2f}"
        print(line)
    for thr in (0.1, 0.3, 0.5):
        n = sum(1 for v in hits if v >= thr)
        print(f"  recall @ IoU>={thr}: {n}/{len(ref)} ({100 * n / len(ref):.0f}%)")
    matched = [v for v in hits if v > 0]
    if matched:
        print(f"  mean IoU over matched steps: {sum(matched) / len(matched):.2f}")
    if text_hits:
        n = sum(1 for v in text_hits if v >= 0.5)
        print(f"  text-matched recall @ IoU>=0.5: {n}/{len(text_hits)} — the prediction that "
              f"names the step, not\n   whichever overlaps it most")
        if wrong_text:
            print(f"  WARNING: {wrong_text} time-only match(es) name a different action, so "
                  f"the IoU recall above\n   may overstate this manual; check its timestamps "
                  f"against the video")
    print("  (recall only — a predicted step with no reference match is not "
          "scored as\n   a false positive; see sample_data/README.md)")


def output_path(args, video: str) -> str:
    """--output for one video; <output stem>-<video stem>.json for several.

    Without --output, serving writes mga-serve.json, never over a batch run's output.
    """
    out = args.output or ("mga-serve.json" if args.serve else "mga-output.json")
    if len(args.video) == 1:
        return out
    stem, ext = os.path.splitext(out)
    return f"{stem}-{os.path.splitext(os.path.basename(video))[0]}{ext or '.json'}"


def check_manual(path: str, args) -> None:
    """Print the manual, warn on zero steps, and score it if asked."""
    steps = load_manual(path)
    if not steps:
        # A session can complete with results: [] — no error, no failed status,
        # and no /logs to say why.
        print("\n  WARNING: the run completed but produced ZERO steps. This is not "
              "reported as a failure anywhere. Check the prompt names no output "
              "format, then retry with a larger --max-new-tokens before assuming "
              "the video is at fault.")
    show(path)
    if args.reference:
        score(path, args.reference, args.keywords)


def run_batch(values: dict, args, state: dict) -> int:
    """One batch session per video, one at a time, on the setting's batch deployment."""
    key, rec = ensure_deployment(values, "batch", args, state)
    print(f"  {key}: copy {rec['blueprint']}  package {rec['package']}  deployment {rec['batch']}")
    failed = 0
    for video in args.video:
        print(f"\nuploading {video} ...")
        file_id = upload(video, upload_name(video))
        print(f"  file_id={file_id}")
        session_id = open_session(rec["batch"], args, state, setting=key, kind="batch",
                                  file_id=file_id, video=video)
        print(f"started session {session_id}\n"
              f"  watching its status and /events (~7 min of model loading first) ...")
        try:
            verdict = poll_session(session_id)
        except KeyboardInterrupt:
            # An abandoned session holds a GPU and blocks everyone behind it.
            print(f"\ninterrupted — cancelling {session_id}")
            end("sessions", session_id)
            return 130
        print(f"\nsession {verdict}")
        saved = fetch_results(session_id, output_path(args, video))
        if saved:
            check_manual(saved, args)
        if verdict != "completed":
            failed += 1
            print(f"session {verdict} — {agents_url(f'sessions/{session_id}/events')}")
    return 1 if failed else 0


def run_serving(values: dict, args, state: dict) -> int:
    """A serving deployment and one session: the model loads once, then a /query
    per video. Both are released on every way out: the deployment holds a GPU."""
    bodies = {}
    for video in args.video:
        with open(video, "rb") as fh:
            bodies[video] = query_body(fh.read(), args.prompt)
        size = len(json.dumps(bodies[video]))
        if size > QUERY_BODY_CAP:
            sys.exit(f"{video}: the query would be {size:,} B, over the 2 MiB cap. Re-encode "
                     f"it smaller and keep the audio, e.g.\n  ffmpeg -i {video} -vf scale=-2:360,fps=8 "
                     f"-c:v libx264 -crf 32 -c:a aac -ac 1 -ar 16000 -b:a 24k small.mp4")
    signal.signal(signal.SIGTERM, _raise_interrupt)
    key = setting_key(values, args.blueprint)
    session_id, t0, failed = None, time.time(), 0
    try:
        key, rec = ensure_deployment(values, "serving", args, state)
        print(f"  serving deployment ready after {time.time() - t0:.0f} s")
        session_id = open_session(rec["serving"], args, state, setting=key, kind="serving")
        wait_for(agents_url(f"sessions/{session_id}"), "running", "pending", 1800, every_s=10)
        print(f"  session {session_id} running after {time.time() - t0:.0f} s")
        for video, body in bodies.items():
            print(f"\nquerying {video} ...")
            try:
                answer = ask(session_id, body)
            except RuntimeError as e:
                print(f"  FAILED: {e}")       # a bad video fails only its own query
                failed += 1
                continue
            rows = (answer.get("response") or {}).get("data")
            if not isinstance(rows, list):
                print(f"  no manual in the answer: {str(answer)[:300]}")
                failed += 1
                continue
            out = output_path(args, video)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as fh:
                json.dump({"id": os.path.splitext(os.path.basename(video))[0], "results": rows}, fh)
            print(f"  saved {out}")
            check_manual(out, args)
    except SessionGone as e:
        print(f"STOPPED: the session no longer answers ({e}). Run again to start a new one.")
        failed += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
        failed += 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        deployment = state["settings"].get(key, {}).get("serving")
        if session_id and end("sessions", session_id):
            state["sessions"].pop(session_id, None)
        if deployment and end("deployments", deployment):
            state["settings"][key].pop("serving", None)
        save_state(state, args.state)
        print(f"released: session {session_id or '-'}, serving deployment {deployment or '-'} "
              f"({time.time() - t0:.0f} s in all)")
    return 1 if failed else 0


def cleanup(args) -> int:
    """Delete sessions, then deployments, then packages: each holds the next.

    Starts from each recorded package and lists its deployments and their
    sessions, so ones this script never got to record are found too. The project
    and its copies stay; the next run at a setting packages its copy again."""
    state = load_state(args.state)
    targets = []                                   # (kind, id), each before what holds it
    for key, rec in state["settings"].items():
        pkg = rec.get("package")
        if not pkg or not alive(agents_url(f"packages/{pkg}")):
            continue
        deployments = [d["id"] for d in list_all(agents_url(f"deployments?package_id={pkg}"))]
        deployments += [rec[k] for k in ("batch", "serving") if rec.get(k) and rec[k] not in deployments]
        targets += [("sessions", s["id"]) for d in deployments
                    for s in list_all(agents_url(f"sessions?deployment_id={d}"))]
        targets += [("deployments", d) for d in deployments] + [("packages", pkg)]
    listed = {t[1] for t in targets}
    targets = [("sessions", s) for s in state["sessions"] if s not in listed] + targets

    for kind in ("sessions", "deployments", "packages"):
        n = sum(t[0] == kind for t in targets)
        if n:
            print(f"{n} {kind} to delete")
    if state.get("project"):
        print(f"stays: project {state['project']} and its copies "
              f"({', '.join(state['settings']) or 'none'})")
    if not targets:
        print("nothing recorded is still on the platform")
    elif not args.yes:
        print("nothing deleted: rerun with --cleanup --yes")
        return 0

    gone, stuck = set(), []
    for kind, obj in targets:
        (gone.add if end(kind, obj) else stuck.append)(obj)
    for rec in state["settings"].values():
        for k in ("package", "batch", "serving"):
            if rec.get(k) in gone:
                rec.pop(k)
    state["sessions"] = {s: x for s, x in state["sessions"].items() if s not in gone}
    save_state(state, args.state)
    print(f"deleted {len(gone)} of {len(targets)}")
    if stuck:
        print(f"would not delete: {', '.join(stuck)} — rerun to try again")
        return 1
    return 0


def dry_run(values: dict, args) -> None:
    """Every request a real run would send, with no key and no call."""
    load_dotenv()
    endpoint = api_root(os.environ["ATAI_API_ENDPOINT"]) if os.environ.get("ATAI_API_ENDPOINT") \
        else "{ATAI_API_ENDPOINT}"
    agents = f"{endpoint}/api/agents"
    key = setting_key(values, args.blueprint)
    kind = "serving" if args.serve else "batch"

    def show_req(method, url, body=None):
        print(f"{method} {url}")
        if body is not None:
            print(json.dumps(body, indent=2))

    print(f"once per setting (reused while {args.state} holds it):\n")
    show_req("GET", f"{agents}/blueprints/{args.blueprint}")
    print(f"GET {endpoint}/api/projects   (find {args.project!r} by name; if none:)")
    show_req("POST", f"{endpoint}/api/projects",
             {"name": args.project, "description": "Manual Generation agent runs"})
    print(f"GET {agents}/blueprints?project_id=<the project>&key={key}   (if none:)")
    show_req("POST", f"{agents}/blueprints/{args.blueprint}/copy", copy_body("<the project>", key, values))
    show_req("POST", f"{agents}/packages", package_body("<the copy>", key))
    show_req("POST", f"{agents}/deployments", {"package_id": "<the package>", "kind": kind})
    if kind == "batch":
        print("\nper video:\n")
        show_req("POST", f"{endpoint}/api/files", None)
        show_req("POST", f"{agents}/sessions",
                 session_body("<the deployment>", upload_name(args.video[0])))
        print(f"GET {agents}/sessions/<id>  and  /events, until terminal; then /results")
    else:
        print("\nonce per run:\n")
        show_req("POST", f"{agents}/sessions", session_body("<the deployment>"))
        print(f"\nper video:\n\nPOST {agents}/sessions/<id>/query")
        print(json.dumps({"model": "mga", "query": values["prompt"],
                          "file_ids": ["base64:<the mp4's bytes>"]}, indent=2))
        print(f"\non exit: POST /sessions/<id>/cancel, DELETE /sessions/<id>, "
              f"POST /deployments/<id>/stop, DELETE /deployments/<id>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", nargs="+", help="one or more .mp4 files under ~5 minutes with audio")
    ap.add_argument("--serve", action="store_true",
                    help="serving deployment + one session: the model loads once, then one "
                         "/query per video (each request capped at 2 MiB). Stopped and deleted "
                         "on exit: it holds a GPU worker")
    ap.add_argument("--blueprint", default=BLUEPRINT_DEFAULT,
                    help=f"blueprint to copy, by KEY (default {BLUEPRINT_DEFAULT!r}). A key "
                         f"tracks whatever is canonical and active; a pinned id can "
                         f"stop resolving when the blueprint is republished")
    ap.add_argument("--project", default=PROJECT_DEFAULT,
                    help=f"project to work in, found by name or created (default {PROJECT_DEFAULT!r})")
    ap.add_argument("--state", default=STATE_DEFAULT,
                    help=f"where the ids are kept, for reuse and --cleanup (default {STATE_DEFAULT})")
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help=f"output budget (default {DEFAULT_MAX_NEW_TOKENS}; the blueprint's own "
                         "default is 8192, so this is always set on the copy). A "
                         "FLOOR, not a ceiling: the model reasons out of this budget, "
                         "so a low budget has returned an EMPTY manual rather than a short one, and "
                         "nothing above 16384 changes the answer")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="the instruction, honoured as ${values.prompt}; "
                         "the preflight warns if that stops being true. "
                         "Say what to COVER, never how to format — the "
                         "results parser owns the template and a format-overriding "
                         "prompt makes it return zero steps")
    ap.add_argument("--output", default=None,
                    help="where the manual goes (default mga-output.json, or mga-serve.json "
                         "with --serve); with several videos, <stem>-<video>.json")
    ap.add_argument("--cleanup", action="store_true",
                    help="list the sessions, deployments and packages in --state; with --yes, delete them")
    ap.add_argument("--yes", action="store_true", help="with --cleanup: delete")
    ap.add_argument("--score", metavar="FILE", help="offline: score an output")
    ap.add_argument("--reference", metavar="CSV",
                    help="reference steps: <step>,<start_sec>,<end_sec>")
    ap.add_argument("--keywords", metavar="JSON",
                    help='with --score: {"<step>": ["phrase", ...]} naming each '
                         'reference step, to also score by the prediction whose TEXT '
                         'names it (temporal IoU alone is blind to text)')
    ap.add_argument("--show", metavar="FILE", help="offline: print a manual")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every request a run would send; no key, no calls")
    args = ap.parse_args()

    if args.show:
        show(args.show)
        return
    if args.score:
        if not args.reference:
            sys.exit("--score needs --reference")
        show(args.score)
        score(args.score, args.reference, args.keywords)
        return
    if args.cleanup:
        sys.exit(cleanup(args))
    if not args.video:
        ap.error("--video is required (or use --show / --score / --cleanup)")

    values = {"max_frames": args.max_frames,
              "max_new_tokens": args.max_new_tokens,
              "prompt": args.prompt}
    if args.dry_run:
        dry_run(values, args)
        return
    missing = [v for v in args.video if not os.path.exists(v)]
    if missing:
        sys.exit(f"no video at {', '.join(missing)}")

    # Preflight BEFORE building or uploading anything: an ignored value should
    # surface in seconds, not after 10 minutes of GPU.
    status, bp = request("GET", agents_url(f"blueprints/{args.blueprint}"), allow_status=True)
    if status != 200:
        sys.exit(f"blueprint {args.blueprint!r} is not visible to this key ({status}): {bp}")
    print(f"blueprint {bp['id']} (key={bp['blueprint_key']}, active={bp['is_active']})")
    if not bp["is_active"]:
        print("  WARNING: superseded blueprint. It reads back and copies normally, then the "
              "run may die ~1 s in with 'resolving blueprint: invalid config'. Use the key 'mga'.")
    ignored = inert_values(bp, values)
    print(f"  honoured: {[k for k in values if k not in ignored]}")
    if ignored:
        print(f"  WARNING: {ignored} not wired on this blueprint; it has no effect.")
        if "prompt" in ignored:
            print("    The blueprint's own instruction applies instead:\n"
                  f"    {bp['document'].get('values', {}).get('prompt')!r}")
        # A copy refuses an undeclared value with a 400; drop those.
        for k in ignored:
            if k not in bp["document"].get("values", {}):
                values.pop(k)

    state = load_state(args.state)
    print(f"\nthe agent for {setting_key(values, args.blueprint)} (ids in {args.state}) ...")
    sys.exit((run_serving if args.serve else run_batch)(values, args, state))


if __name__ == "__main__":
    main()
