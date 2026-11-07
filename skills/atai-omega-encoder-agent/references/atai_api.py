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

def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip()   # .env wins over a stale shell export


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


def deployment(endpoint: str) -> str:
    """What deployment an endpoint addresses, whatever hostname it arrives under.

    A deployment can answer under two names, api.<deployment> and
    platform.<deployment>: an agent created through either is visible
    through the other.

    So state written through one name has to load through the other. The
    generation (api, platform) is everything left of the first dot; the rest
    names the deployment, which still separates u1 from u2.
    """
    host = re.sub(r"^[a-z]+://", "", endpoint.strip()).split("/")[0].split(":")[0]
    return re.sub(r"^(api|platform)\.", "", host).lower()


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


def _wait_for(url: str, status: str, waiting: str, timeout_s: int, every_s: int = 3) -> dict:
    obj = request("GET", url)
    t0 = time.time()
    while obj["status"] == waiting:
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"{obj['id']} still {waiting} after {timeout_s} s")
        time.sleep(every_s)
        obj = request("GET", url)
    if obj["status"] != status:
        raise SystemExit(f"{obj['id']} is {obj['status']}, not {status}: {obj.get('error')}")
    return obj


def create_package(blueprint_id: str, name: str, timeout_s: int = 600) -> dict:
    """Package one blueprint for the platform and wait until it is ready.

    The package lands in its blueprint's project; deploying needs one."""
    pkg = request("POST", packages_url(), body={
        "blueprint_ids": [blueprint_id], "target": {"kind": "platform"}, "name": name})
    return _wait_for(packages_url(pkg["id"]), "ready", "building", timeout_s, every_s=5)


def deploy(package_id: str, kind: str, timeout_s: int = 600) -> dict:
    """Deploy a package as `batch` or `serving` and wait until it is ready.

    A batch deployment is ready at once; a serving one is placed on a worker
    first, usually within seconds."""
    dpl = request("POST", deployments_url(), body={"package_id": package_id, "kind": kind})
    return _wait_for(deployments_url(dpl["id"]), "ready", "deploying", timeout_s)


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
    """Cancel a session that is still going, then delete it."""
    if request("GET", sessions_url(session_id))["status"] not in TERMINAL:
        request("POST", f"{sessions_url(session_id)}/cancel", body={})
    request("DELETE", sessions_url(session_id), raw=True)


def end_deployment(deployment_id: str) -> None:
    """Stop a deployment that is still up, then delete it."""
    if request("GET", deployments_url(deployment_id))["status"] not in ("stopped", "failed"):
        request("POST", f"{deployments_url(deployment_id)}/stop", body={})
    request("DELETE", deployments_url(deployment_id), raw=True)


def request(method: str, url: str, body=None, headers=None, raw: bool = False,
            retries: int = 4):
    """Call the API, retrying transient network failures.

    Polls run for hours, and a single DNS hiccup or socket timeout used to kill
    the whole client while the platform job carried on running unattended — the
    job then finished with nobody to download its output. Only *network* errors
    retry; an HTTP status is a real answer from the server and still exits.
    """
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
            sys.exit(f"{method} {url} failed ({e.code}): {e.read().decode(errors='replace')}")
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException) as e:
            # http.client.IncompleteRead / RemoteDisconnected: the body was cut off
            # mid-transfer. Seen once in ~120 Stage 8 downloads; it killed the worker
            # because it is not a URLError. Retry like any other network fault.
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


def _last_timestamp(csv_path: str) -> float | None:
    """Timestamp of the final sample in an input CSV, read from the last line."""
    last = None
    with open(csv_path, "r") as fh:
        for line in fh:
            if line.strip():
                last = line
    try:
        return float(last.split(",")[0])
    except (AttributeError, ValueError):
        return None


def _output_coverage(out_path: str, source_end: float) -> tuple[float, float]:
    """(fraction of the input covered, shortfall in seconds).

    Compares the last window's finish timestamp against the input's last sample.
    Deliberately derived from the output itself rather than from window/step:
    those come from the detector artifact, so a client that did not pass them
    does not know them, and this check has to work in the default path.
    """
    import csv as _csv
    with open(out_path) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        return 0.0, source_end
    fin = "finish_timestamp" if "finish_timestamp" in rows[0] else list(rows[0])[0]
    sta = "start_timestamp" if "start_timestamp" in rows[0] else None
    try:
        end = max(float(r[fin]) for r in rows if r.get(fin))
    except ValueError:
        return 0.0, source_end
    # One window may legitimately be missing: a tail shorter than the window is
    # dropped rather than padded. Measure a window from the output itself.
    win = 0.0
    if sta:
        try:
            win = max(float(r[fin]) - float(r[sta]) for r in rows[:64] if r.get(sta))
        except ValueError:
            win = 0.0
    shortfall = max(0.0, source_end - end - win * 1.5)
    return (end / source_end if source_end else 1.0), shortfall


def _first_result(session_id: str) -> tuple[list, dict, str | None]:
    """One /results call: (items, first item's inner data, its download ref).

    The `ref` is a presigned URL that expires (~20 min), so every retry has to ask
    /results again rather than reuse the previous one.
    """
    results = request("GET", f"{sessions_url(session_id)}/results")
    items = results.get("data", results if isinstance(results, list) else [])
    if not items:
        return [], {}, None
    inner = items[0].get("data") or {}
    ref = inner.get("ref") or items[0].get("ref") or items[0].get("url")
    return items, inner, ref


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


def download_results(session_id: str, out_path: str,
                     source_csv: str | None = None,
                     settle_s: int = 180, appear_s: int = 60) -> str | None:
    """Fetch /results and save the first output to out_path, or return None.

    WAITS FOR THE ARTIFACT TO FINISH SYNCING, which is not the same thing as the
    run finishing. The output row is created while the agent is still writing and
    is filled in afterwards, so a download issued the moment the status flips to
    `completed` can capture a prefix. Measured on a 298.94 MB run: the first fetch
    returned 10,652 of 12,648 windows -- 84.2%, a perfectly well-formed CSV with
    zero invalid windows and nothing to suggest it was short. It scored 1002.8
    lead hours against a true 27.0. Re-fetching the same artifact minutes later
    returned all 12,648 rows.

    The console shows this as artifact status PARTIAL -> COMPLETE, but /results
    carries no status field -- only filename, num_bytes and ref -- so a client
    cannot ask. It has to verify by content, which is what this does:

      * `source_csv` given: re-fetch until the last scored window reaches the end
        of that input. Needs no knowledge of window/step -- which come from the
        detector artifact, so the default path does not have them.
      * otherwise: poll until num_bytes stops changing, which is weaker (a stall
        mid-write looks like a finish) but is all that is available.

    Two separate budgets: `appear_s` for the /results row to exist at all (0 to
    try exactly once, for runs that finished long ago or failed), then `settle_s`
    for the artifact to catch up once it does. The file is written to a temporary
    name and only moved to out_path when it passes, so a caller never finds a
    prefix at the real path -- a short file there would be scored as complete by
    evaluate_anomalies.py and skipped as already-downloaded by run_ad_eval.py.
    """
    appear_deadline = time.time() + appear_s
    while True:
        items, inner, ref = _first_result(session_id)
        if items:
            break
        if time.time() >= appear_deadline:
            print("  no results returned")
            return None
        time.sleep(5)
    print(f"results ({len(items)}):")
    for item in items:
        d = item.get("data") or {}
        print(f"  {d.get('filename') or item.get('id')}  ({d.get('num_bytes', '?')} bytes)")
    if not ref:
        print("  results carried no download ref")
        return None

    source_end = _last_timestamp(source_csv) if source_csv else None
    tmp_path = out_path + ".part"
    deadline = time.time() + settle_s
    frac = short = seen_bytes = None
    while True:
        # The artifact's own status is the authoritative signal and needs nothing
        # to compare against. A run reports `completed` while its outputs are
        # still being written -- measured, about a minute of it -- and the
        # console shows the same pairing, run COMPLETED beside artifact PARTIAL.
        # An absent status reads as ready, so a deployment reporting none is not
        # blocked, and the checks below still apply.
        status = (inner.get("file_attributes", {}).get("job_output", {}).get("status"))
        if status is not None and status != "complete":
            print(f"  artifact reports {status}; waiting")
        elif source_end is None:
            # No input to compare against: fall back to waiting for num_bytes to
            # hold still across two polls. A stall mid-write looks the same,
            # which is why the status above is preferred when present.
            settled = inner.get("num_bytes") == seen_bytes
            seen_bytes = inner.get("num_bytes")
            payload = _download(ref) if settled else None
            if payload is not None:
                open(tmp_path, "wb").write(payload)
                break
        else:
            payload = _download(ref)
            if payload is not None:
                open(tmp_path, "wb").write(payload)
                frac, short = _output_coverage(tmp_path, source_end)
                if short <= 0:
                    print(f"  verified: output covers the full input ({frac * 100:.1f}%)")
                    break
                print(f"  artifact still syncing: covers {frac * 100:.1f}% of the "
                      f"input ({short:,.0f}s short), waiting")
        if time.time() >= deadline:
            _discard(tmp_path)
            if short is not None:
                print(f"  WARNING: output covers {frac * 100:.1f}% of the input, "
                      f"{short:,.0f}s short, after {settle_s}s. The artifact is "
                      f"incomplete -- scoring it would silently give a wrong lead "
                      f"time -- so nothing was saved. Re-download for {session_id} "
                      f"in a minute.")
            else:
                print(f"  WARNING: the artifact was still changing after {settle_s}s; "
                      f"nothing saved. Re-download for {session_id} in a minute.")
            return None
        time.sleep(10 if source_end else 5)
        items, inner, ref = _first_result(session_id)
        if not ref:
            print("  results disappeared mid-download")
            _discard(tmp_path)
            return None
    os.replace(tmp_path, out_path)
    print(f"saved output to {out_path}")
    return out_path


def _discard(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
