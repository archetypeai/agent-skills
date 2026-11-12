#!/usr/bin/env python3
"""Minimal stdlib-only client for the Archetype AI Agents API (Agent Lifecycle 2.0).

Auth and endpoint come from the environment, with a local .env loaded if
present: ATAI_API_KEY, ATAI_API_ENDPOINT.

The lifecycle: a blueprint in a project is packaged (its models resolved once),
the package is deployed as `batch` or `serving`, and work runs in sessions on a
deployment. A batch session is one run over its input files; a serving session
answers `/query`, one request at a time, until it is cancelled.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

TERMINAL = {"completed", "failed", "cancelled"}

def find_dotenv(start: str | None = None) -> str | None:
    """The nearest .env, from the current folder upwards."""
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isfile(os.path.join(d, ".env")):
            return os.path.join(d, ".env")
        if os.path.dirname(d) == d:
            return None
        d = os.path.dirname(d)


def load_dotenv(path: str | None = None) -> None:
    """Fill ATAI_* from `path`, or the nearest .env upwards. A variable already set in the
    shell wins: check `env | grep ATAI` when a key meant for another deployment answers 401."""
    path = path or find_dotenv()
    if not path or not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]                    # KEY="value" as most .env files write it
        if not os.environ.get(key.strip()):        # the shell wins
            os.environ[key.strip()] = value


def api_base() -> str:
    """The deployment root with /api appended — every surface lives under it.

    Accepts a bare root, a /vX.Y suffix or /api itself, so one .env works
    whichever form you have.
    """
    endpoint = os.environ.get("ATAI_API_ENDPOINT", "").rstrip("/")
    if not endpoint:
        sys.exit("ATAI_API_ENDPOINT is not set")
    endpoint = re.sub(r"/api$", "", endpoint)
    endpoint = re.sub(r"/v[0-9]+(\.[0-9]+)*$", "", endpoint)
    return f"{endpoint}/api"


def agents_base() -> str:
    return f"{api_base()}/agents"


def packages_url(package_id: str = "") -> str:
    """A package pins one blueprint and the models it resolves to. It carries no
    values: a package runs its blueprint exactly as it was promoted, so every
    setting, outputs included, is chosen when the blueprint is made."""
    base = f"{agents_base()}/packages"
    return f"{base}/{package_id}" if package_id else base


def deployments_url(deployment_id: str = "") -> str:
    base = f"{agents_base()}/deployments"
    return f"{base}/{deployment_id}" if deployment_id else base


def sessions_url(session_id: str = "") -> str:
    base = f"{agents_base()}/sessions"
    return f"{base}/{session_id}" if session_id else base


def _wait_for(url: str, status: str, waiting: str, timeout_s: int, every_s: int = 3,
              exit: bool = True) -> dict:
    """Poll while `waiting`; exit unless it then reaches `status` (exit=False: return it as is)."""
    obj = request("GET", url)
    t0 = time.time()
    while obj["status"] == waiting:
        if time.time() - t0 > timeout_s:
            if not exit:
                return obj
            raise SystemExit(f"{obj['id']} still {waiting} after {timeout_s} s")
        time.sleep(every_s)
        obj = request("GET", url)
    if obj["status"] != status and exit:
        raise SystemExit(f"{obj['id']} is {obj['status']}, not {status}: {obj.get('error')}")
    return obj


def create_package(blueprint_id: str, name: str) -> dict:
    """Package one blueprint for the platform; it starts `building`. The package lands in
    its blueprint's project; deploying needs one. Save its id before `package_ready`, so a
    build that fails or is interrupted can still be deleted."""
    return request("POST", packages_url(), body={
        "blueprint_ids": [blueprint_id], "target": {"kind": "platform"}, "name": name})


def package_ready(package_id: str, timeout_s: int = 600) -> dict:
    return _wait_for(packages_url(package_id), "ready", "building", timeout_s, every_s=5)


def ready_package(ws: dict, blueprint_id: str, name: str, save, log=print) -> dict:
    """ws["package"] once ready, created (and saved) first if there is none. A failed package
    never becomes ready, so it is deleted and built again, once."""
    for attempt in range(2):
        if not ws.get("package"):
            ws["package"] = create_package(blueprint_id, name)["id"]
            save()
            log(f"package {ws['package']}")
        pkg = _wait_for(packages_url(ws["package"]), "ready", "building", 600, every_s=5, exit=False)
        if pkg["status"] == "ready":
            return pkg
        if pkg["status"] != "failed" or attempt:
            sys.exit(f"{ws['package']} is {pkg['status']}, not ready: {pkg.get('error')}")
        log(f"{ws['package']} failed ({pkg.get('error')}): deleting it and building a new one")
        if status_of(packages_url(ws["package"])) is not None:
            request("DELETE", packages_url(ws["package"]), raw=True)
        ws.pop("package")
        ws.pop("deployment", None)
        save()


def deploy(package_id: str, kind: str) -> dict:
    """Deploy a ready package as `batch` (ready at once) or `serving` (placed on a worker,
    usually within seconds; it holds the worker until stopped). It starts `deploying`: save
    its id before `deployment_ready`."""
    return request("POST", deployments_url(), body={"package_id": package_id, "kind": kind})


def deployment_ready(deployment_id: str, timeout_s: int = 600) -> dict:
    return _wait_for(deployments_url(deployment_id), "ready", "deploying", timeout_s)


def status_of(url: str) -> str | None:
    """An object's status, or None once it is gone (404)."""
    try:
        return request("GET", url)["status"]
    except SystemExit as e:
        if "(404)" in str(e):
            return None
        raise


def list_all(url: str) -> list:
    """Every item of a paginated list; the cursor is opaque and passed back as is."""
    items, after = [], ""
    sep = "&" if "?" in url else "?"
    while True:
        page = request("GET", f"{url}{sep}limit=1000{after}")
        items += page.get("data", [])
        if not page.get("has_more"):
            return items
        after = f"&after={page['next_cursor']}"


def open_session(deployment_id: str, file_ids: list[str] | None = None) -> dict:
    """A batch session over `file_ids` (one output per input), or a serving
    session when there are none."""
    body = {"deployment_id": deployment_id}
    if file_ids:
        body["connectors"] = {"source": [{"type": "file", "id": f, "format": "csv"} for f in file_ids]}
    return request("POST", sessions_url(), body=body)


def wait_running(session_id: str, timeout_s: int = 600) -> dict:
    """A serving session takes queries once it is `running`."""
    return _wait_for(sessions_url(session_id), "running", "pending", timeout_s)


def end_session(session_id: str) -> None:
    """Cancel a session that is still going, then delete it; one already gone counts as done."""
    status = status_of(sessions_url(session_id))
    if status is None:
        return
    if status not in TERMINAL:
        request("POST", f"{sessions_url(session_id)}/cancel", body={}, idempotent=True)
    request("DELETE", sessions_url(session_id), raw=True)


def end_deployment(deployment_id: str) -> None:
    """Stop a deployment that is still up, then delete it; one already gone counts as done."""
    status = status_of(deployments_url(deployment_id))
    if status is None:
        return
    if status not in ("stopped", "failed"):
        request("POST", f"{deployments_url(deployment_id)}/stop", body={}, idempotent=True)
    request("DELETE", deployments_url(deployment_id), raw=True)


def request(method: str, url: str, body=None, headers=None, raw: bool = False,
            retries: int = 4, idempotent: bool | None = None):
    """Call the API, retrying transient failures (network errors, 502/503/504) of calls that
    are safe to repeat: GET, HEAD and DELETE, and POSTs marked idempotent (cancel, stop, a
    serving query). A create POST never retries: a repeat could make a second object that
    nothing records, such as a serving deployment holding a worker. Any other HTTP status is
    the server's answer and exits.
    """
    if not (method in ("GET", "HEAD", "DELETE") if idempotent is None else idempotent):
        retries = 0
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = resp.read()
                return payload if raw else json.loads(payload or b"null")
        except urllib.error.HTTPError as e:
            if e.code not in (502, 503, 504) or attempt == retries:
                sys.exit(f"{method} {url} failed ({e.code}): {e.read().decode(errors='replace')}")
            print(f"  {method} {url}: HTTP {e.code}; retrying in {5 * (2 ** attempt)}s [{attempt + 1}/{retries}]")
            time.sleep(5 * (2 ** attempt))
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException) as e:
            # IncompleteRead / RemoteDisconnected (a body cut off mid-transfer) included
            if attempt == retries:
                sys.exit(f"{method} {url} failed after {retries + 1} attempts: {e}")
            wait = 5 * (2 ** attempt)
            print(f"  network error ({e}); retrying in {wait}s "
                  f"[{attempt + 1}/{retries}]")
            time.sleep(wait)


def upload_file(path: str, rename: str | None = None, retries: int = 4) -> dict:
    """POST a file to /api/files as multipart/form-data."""
    boundary = uuid.uuid4().hex
    filename = rename or os.path.basename(path)
    content = open(path, "rb").read()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: text/csv\r\n\r\n",
        content,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    # Retry network failures (a reset TLS handshake, a timeout) like request() does;
    # an HTTP status is a real answer and still exits.
    for attempt in range(retries + 1):
        req = urllib.request.Request(f"{api_base()}/files", data=body, method="POST")
        req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"upload of {path} failed ({e.code}): {e.read().decode(errors='replace')}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            if attempt == retries:
                sys.exit(f"upload of {path} failed after {retries + 1} attempts: {e}")
            wait = 5 * (2 ** attempt)
            print(f"  upload of {os.path.basename(path)}: network error ({e}); retrying in {wait}s "
                  f"[{attempt + 1}/{retries}]")
            time.sleep(wait)


def poll_session(session_id: str, timeout_s: int, interval_s: int = 20) -> str:
    """Poll a batch session until it ends, printing its events as they appear.

    Returns the final status, or `running` when the timeout came first."""
    deadline = time.time() + timeout_s
    seen: set[str] = set()
    status = "pending"
    while status not in TERMINAL and time.time() < deadline:
        time.sleep(interval_s)
        ses = request("GET", sessions_url(session_id))
        status = ses["status"]
        events = request("GET", f"{sessions_url(session_id)}/events").get("data", [])
        for ev in reversed(events):                    # newest first
            marker = f"{ev.get('created_at')}{ev.get('message')}"
            if marker not in seen:
                seen.add(marker)
                print(f"  [{ev.get('level', 'info')}] {ev.get('created_at', '')}  "
                      f"{session_id}: {ev.get('message', '')}")
        if ses.get("error"):
            print(f"  {session_id}: {ses['error']}")
    if status not in TERMINAL:
        print(f"  {session_id}: timed out after {timeout_s/60:.0f} min, still {status}")
    return status


def _download(ref: str) -> bytes | None:
    """Follow a result ref: an absolute presigned URL, or a platform path fetched
    with the bearer token. Network errors return None so the caller can poll
    again instead of dying mid-wait."""
    try:
        if ref.startswith("http"):
            with urllib.request.urlopen(ref, timeout=120) as resp:
                return resp.read()
        # A session's ref is a root path, `/api/files/<id>/content`, not one under api_base().
        base = api_base().removesuffix("/api") if ref.startswith("/api/") else api_base()
        return request("GET", f"{base}{ref}", raw=True)
    except (urllib.error.URLError, TimeoutError, ConnectionError,
            http.client.HTTPException) as e:          # incl. IncompleteRead mid-download
        print(f"  download failed ({e}); retrying")
        return None
