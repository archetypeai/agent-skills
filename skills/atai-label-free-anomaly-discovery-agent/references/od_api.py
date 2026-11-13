"""A small standard-library client for the Agent Lifecycle 2.0 calls the od scripts make.

project -> copy of `od` with your values -> package -> batch deployment -> a session per
input file -> its results. Every object lives in the project.

Credentials: ATAI_API_KEY and ATAI_API_ENDPOINT from the environment, else from the nearest
.env, looking upwards from the current directory, as the other agent skills read them: a
variable already set in the shell wins. Keys are per deployment, and one exported for another
deployment is the usual reason a first call fails, so every script prints the endpoint, the
key's last 4 characters and where each came from.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

TERMINAL_BAD = {"failed", "cancelled", "canceled", "stopped"}
_LOADED = False
_CHECKED = False


def find_dotenv(start: Path | None = None) -> Path | None:
    for d in [start or Path.cwd(), *(start or Path.cwd()).parents]:
        if (d / ".env").is_file():
            return d / ".env"
    return None


def load_dotenv() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    source, path = "the shell", find_dotenv()
    if path:
        for line in path.read_text().splitlines():
            m = re.match(r"\s*(ATAI_API_KEY|ATAI_API_ENDPOINT)\s*=\s*(.*)", line)
            if m and not os.environ.get(m[1]):          # the shell wins
                os.environ[m[1]] = m[2].strip().strip("'\"")
                source = str(path)
    key = os.environ.get("ATAI_API_KEY", "")
    print(f"using {os.environ.get('ATAI_API_ENDPOINT')} with key …{key[-4:]} (from {source})", flush=True)


def root() -> str:
    """The deployment root, whether ATAI_API_ENDPOINT ends in /api (or /vX.Y) or not."""
    load_dotenv()
    ep = os.environ.get("ATAI_API_ENDPOINT", "").rstrip("/")
    key = os.environ.get("ATAI_API_KEY", "")
    if not ep or not key or key.startswith("<"):
        raise SystemExit("set ATAI_API_KEY and ATAI_API_ENDPOINT in .env (see .env.example)")
    return re.sub(r"/(api|v\d+\.\d+)$", "", ep)


def api() -> str:
    return root() + "/api"


def _auth() -> dict:
    return {"Authorization": f"Bearer {os.environ['ATAI_API_KEY']}"}


def check_key() -> None:
    """One cheap call first, so a wrong key, endpoint or deployment says so plainly."""
    global _CHECKED
    if _CHECKED:
        return
    base = root()
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/api/agents/sessions?limit=1", headers=_auth()),
                                    timeout=60):
            pass
    except urllib.error.HTTPError as e:
        hint = {401: "the key is not accepted there (wrong key, or a key for another deployment)",
                403: "the key's account has no access there (an expired trial, or a missing role)",
                404: "this deployment doesn't serve Agent Lifecycle 2.0 (/api/agents/sessions)"}
        raise SystemExit(f"{base}: {e.code}, {hint.get(e.code, e.read()[:300].decode(errors='replace'))}. "
                         "Check ATAI_API_KEY and ATAI_API_ENDPOINT in .env.")
    _CHECKED = True


def request(method: str, path: str, body=None, raw: bool = False, retries: int = 4,
            idempotent: bool | None = None):
    """One call; a 4xx raises SystemExit with the platform's message. GET, HEAD and DELETE
    retry network errors and 5xx, and so do POSTs marked idempotent (cancel, stop); a create
    POST never does, since a repeat could make a second object that nothing records."""
    if not (method in ("GET", "HEAD", "DELETE") if idempotent is None else idempotent):
        retries = 0
    url = path if path.startswith("http") else api() + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {**_auth(), **({"Content-Type": "application/json"} if data else {})}
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data, headers, method=method), timeout=120) as r:
                b = r.read()
                return b if raw else (json.loads(b) if b else {})
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries:
                raise SystemExit(f"{method} {path} -> {e.code}: {e.read()[:800].decode(errors='replace')}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)


def upload(path: Path, name: str) -> str:
    """POST a CSV to /api/files; returns its file id (the name). Use a new name per upload:
    replacing a file a session is still reading can break that session."""
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            "Content-Type: text/csv\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(api() + "/files", body, method="POST",
                                 headers={**_auth(), "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["file_id"]


def wait(path: str, ok: set, every_s: int = 5, timeout_s: int = 1800, label: str = "") -> dict:
    """Poll until `status` is in `ok`; stop on a terminal failure (a session carries `error`)."""
    t0, last = time.time(), None
    while time.time() - t0 < timeout_s:
        obj = request("GET", path)
        s = obj.get("status")
        if s != last:
            print(f"    {label or path.rsplit('/', 1)[-1]}: {s}", flush=True)
            last = s
        if s in ok:
            return obj
        if s in TERMINAL_BAD:
            raise SystemExit(f"{path}: {s}: {json.dumps(obj.get('error') or obj)[:800]}")
        time.sleep(every_s)
    raise SystemExit(f"{path}: still {last} after {timeout_s} s")


def list_all(path: str) -> list:
    """Every item of a paginated list; the cursor is opaque and passed back as is."""
    items, after = [], ""
    sep = "&" if "?" in path else "?"
    while True:
        page = request("GET", f"{path}{sep}limit=1000{after}")
        items += page.get("data", [])
        if not page.get("has_more"):
            return items
        after = f"&after={page['next_cursor']}"


def find_project(name: str) -> str | None:
    found = [p for p in list_all("/projects") if p["name"] == name]
    return found[0]["id"] if found else None


def status_of(path: str) -> str | None:
    """An object's status, or None once it is gone (404)."""
    try:
        return request("GET", path).get("status")
    except SystemExit as e:
        if " -> 404" in str(e):
            return None
        raise


def end_session(session_id: str) -> None:
    """Cancel a session still going, then delete it; one already gone counts as done."""
    s = status_of(f"/agents/sessions/{session_id}")
    if s is None:
        return
    if s in {"pending", "running", "paused"}:
        request("POST", f"/agents/sessions/{session_id}/cancel", {}, idempotent=True)
    request("DELETE", f"/agents/sessions/{session_id}", raw=True)


def download_result(session_id: str) -> bytes:
    """A session's output file. Its `ref` is a root path (/api/files/…/content): join it to the root."""
    res = request("GET", f"/agents/sessions/{session_id}/results")["data"]
    if not res:
        raise SystemExit(f"session {session_id}: no results")
    ref = (res[0].get("data") or res[0])["ref"]
    return request("GET", root() + ref if ref.startswith("/api/") else api() + ref, raw=True)
