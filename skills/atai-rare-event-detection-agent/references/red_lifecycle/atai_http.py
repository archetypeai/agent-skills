"""Stdlib HTTP helpers for Path 2 (Optimize → promote → Evals → bundle → run).

The same helpers as the OSM and AD skills' Path 2, with RED's metrics report (target
`event`) and an upload cache keyed by file content.

The official `archetypeai` client does not yet cover the Optimizations API, promote
or the Evals API, so Path 2 talks to them directly. Move these calls onto the client
once it does; the rest of Path 2 only uses the functions below.

Auth / endpoint come from the environment, else from the nearest .env (the current folder
upwards, e.g. the repo root's):
    ATAI_API_KEY        required
    ATAI_API_ENDPOINT   required: the deployment root, with or without an /api or /vX.Y
                        suffix. Every surface lives under one versionless /api prefix
                        (<root>/api/agents, <root>/api/files)
"""
import hashlib
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid

TERMINAL = {"completed", "failed", "canceled", "cancelled"}   # the platform spells it "cancelled"
RETRY_CODES = (502, 503, 504)
MAX_WAIT_S = 600


SOURCE = {}          # ATAI_* variable -> "shell" or the .env path it came from


def find_dotenv(start=None):
    """The nearest .env from `start` (default: the current folder) upwards, else the nearest
    from this script's folder upwards, else None. So one .env at the repo root serves every
    skill, as the model skills' python-dotenv lookup does, while a closer one still wins."""
    for base in ([start] if start else [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]):
        d = os.path.abspath(base)
        while True:
            if os.path.isfile(os.path.join(d, ".env")):
                return os.path.join(d, ".env")
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None


def load_dotenv(path=None):
    """Fill ATAI_* from `path`, or from find_dotenv() when no path is given. A variable already
    set in the shell wins, and is reported as such by check_auth(): an exported key for another
    deployment is a common mix-up."""
    for k in ("ATAI_API_KEY", "ATAI_API_ENDPOINT"):
        if os.environ.get(k):
            SOURCE[k] = "shell"
    path = path or find_dotenv()
    if not path or not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            if not os.environ.get(key):         # unset, or set but empty: an empty export mustn't hide .env
                os.environ[key] = value.strip()
                SOURCE[key] = os.path.abspath(path)


def check_auth(log=print):
    """One cheap authenticated GET, before any upload: say where the endpoint and key came
    from, and fail at once on a wrong key. (A rejected large upload only shows up in Python
    as "Broken pipe", retried for minutes: the server closes before the body is sent.)"""
    ep = os.environ.get("ATAI_API_ENDPOINT", "")
    log(f"endpoint {root()} (from {SOURCE.get('ATAI_API_ENDPOINT', 'the environment')}), "
        f"key …{os.environ.get('ATAI_API_KEY', '')[-4:]} (from {SOURCE.get('ATAI_API_KEY', 'the environment')})")
    req = urllib.request.Request(f"{agents()}/blueprints?limit=1")
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    try:
        with urllib.request.urlopen(req, timeout=60):
            return
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            hint = (" An ATAI_API_KEY exported in your shell overrides .env: `unset ATAI_API_KEY ATAI_API_ENDPOINT`."
                    if "shell" in SOURCE.values() else "")
            sys.exit(f"the API key is not valid for {ep} (HTTP {e.code}). Keys are per deployment.{hint}")
        raise


def root():
    """The deployment root: the endpoint without any /api or /vX.Y suffix."""
    endpoint = os.environ.get("ATAI_API_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        sys.exit("ATAI_API_ENDPOINT is not set (see .env.example)")
    if not os.environ.get("ATAI_API_KEY"):
        sys.exit("ATAI_API_KEY is not set (see .env.example)")
    endpoint = re.sub(r"/api$", "", endpoint)
    return re.sub(r"/v\d+(\.\d+)*$", "", endpoint)


def api():
    return f"{root()}/api"


def deployment(endpoint):
    """What deployment an endpoint addresses, whatever hostname it arrives under.

    api.u1.archetypeai.app and platform.u1.archetypeai.app are one deployment under two
    names: an id made through either is visible through the other. So the generation
    (api, platform) left of the first dot is dropped; the rest still separates dev.u1
    from stage.u1 from u1."""
    host = re.sub(r"^[a-z]+://", "", endpoint.strip()).split("/")[0].split(":")[0]
    return re.sub(r"^(api|platform)\.", "", host).lower()


def same_deployment(saved):
    """False if a saved file (results, state) records an endpoint on another deployment, whose
    ids don't exist on this one. A file from before endpoints were recorded passes."""
    return not saved.get("endpoint") or deployment(saved["endpoint"]) == deployment(root())


def agents():
    return f"{api()}/agents"


def files_url():
    return f"{api()}/files"


def _unsent(e):
    """True when the request never reached the server (DNS failure, connection refused)."""
    reason = getattr(e, "reason", e)
    return isinstance(reason, (socket.gaierror, ConnectionRefusedError))


def _retrying(send, what, retry_http=True, only_unsent=False):
    """Call send() through brief network / gateway outages (5, 10, 20 … 60 s, up to 10 min).

    only_unsent: retry only failures that provably never reached the server."""
    delay, waited = 5, 0
    while True:
        try:
            return send()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if not retry_http or e.code not in RETRY_CODES or waited >= MAX_WAIT_S:
                raise RuntimeError(f"{what} failed ({e.code}): {detail}") from None
            reason = f"HTTP {e.code}"
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            # a DNS lookup failing while big uploads saturate the uplink lands here
            if waited >= MAX_WAIT_S or (only_unsent and not _unsent(e)):
                raise
            reason = str(getattr(e, "reason", e))
        print(f"  {what} unavailable ({reason}); retrying in {delay}s", file=sys.stderr)
        time.sleep(delay)
        waited += delay
        delay = min(delay * 2, 60)


def request(method, url, body=None, idempotent=None):
    """JSON request. GETs retry through outages; POSTs retry only a connection that never
    reached the server (a repeat of a delivered POST could duplicate an optimization),
    unless idempotent=True (a serving query: a repeat only asks again)."""
    data = json.dumps(body).encode() if body is not None else None

    def send():
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read() or b"null")
    retry = method == "GET" if idempotent is None else idempotent
    return _retrying(send, f"{method} {url}", retry_http=retry, only_unsent=not retry)


def upload_file(path, name=None):
    """POST a file to /api/files (multipart); returns the platform's JSON (with file_id).

    Connectors and examples reference `file_id` (the filename), not the `fil_` uid.
    A retry after a lost response can leave an unused copy on the platform; harmless."""
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        content = f.read()
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{name or os.path.basename(path)}"\r\n'.encode(),
        b"Content-Type: text/csv\r\n\r\n", content, f"\r\n--{boundary}--\r\n".encode(),
    ])

    def send():
        req = urllib.request.Request(files_url(), data=body, method="POST")
        req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    return _retrying(send, f"upload of {os.path.basename(path)}")


def cache_key(path, base):
    """By content: a rebuilt but identical role file reuses its upload. Keyed by deployment
    too: file ids from one deployment don't exist on another."""
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return f"{deployment(root())}|{os.path.relpath(path, base)}|{digest}"


def upload_all(paths, base, cache_path, prefix="red", jobs=3, log=print):
    """Platform file ids for `paths`, uploading only what the cache doesn't hold yet.

    The cache is keyed by deployment, path and content, and saved after every file, so an
    interrupted upload resumes. Keep `jobs` low: many parallel multi-hundred-MB uploads
    can saturate the uplink until DNS lookups fail (seen on dev)."""
    from concurrent.futures import ThreadPoolExecutor
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [p for p in paths if cache_key(p, base) not in cache]
    if todo:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        log(f"uploading {len(todo)} of {len(paths)} files "
            f"({sum(map(os.path.getsize, todo)) / 1e6:,.0f} MB; {len(paths) - len(todo)} cached)")
        t0 = time.time()

        def one(p):
            name = f"{prefix}-" + os.path.relpath(p, base).replace(os.sep, "__").replace(".csv", f"-{stamp}.csv")
            return cache_key(p, base), upload_file(p, name)["file_id"]
        with ThreadPoolExecutor(jobs) as pool:
            for done, (k, fid) in enumerate(pool.map(one, todo), 1):
                cache[k] = fid
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=1)
        log(f"  uploaded {len(todo)} ({time.time() - t0:.0f} s)")
    return {p: cache[cache_key(p, base)] for p in paths}


def list_all(url):
    """Every item of a paginated list endpoint. The cursor is opaque: pass it back verbatim."""
    items, cursor = [], ""
    sep = "&" if "?" in url else "?"
    while True:
        page = request("GET", f"{url}{sep}limit=1000{cursor}")
        items += page.get("data", [])
        if not page.get("has_more"):
            return items
        cursor = f"&after={page['next_cursor']}"


def list_trials(opt_id):
    return list_all(f"{agents()}/optimizations/{opt_id}/trials")


def wait(url, label, every_s=30, log=print, progress=lambda o: ""):
    """Poll GET url until its status is terminal; log each change of state."""
    last = None
    while True:
        obj = request("GET", url)
        state = (obj.get("status"), progress(obj))
        if state != last:
            log(f"{label}: {obj.get('status')} {state[1]}".rstrip())
            last = state
        if obj.get("status") in TERMINAL:
            return obj
        time.sleep(every_s)


def trial_values(trial):
    """A trial's sampled values (a RED trial reports every setting under `values`)."""
    tv = trial.get("trial_values") or {}
    return {**tv.get("fitting", {}), **tv.get("values", {})}


def report_event(report, background="normal"):
    """From a trial's or eval's metrics_report (target `event`; class_names alphabetical):
    the aggregate objectives, the confusion matrix (rows = gold), and event-vs-background
    recall / precision / false alarms, whichever event."""
    t = (report or {}).get("targets", {}).get("event", {})
    names, cm = t.get("class_names"), t.get("confusion_matrix")
    if not names or not cm:
        return {}
    b = names.index(background)
    ev = [i for i in range(len(names)) if i != b]
    tp = sum(cm[g][p] for g in ev for p in ev)
    fn = sum(cm[g][b] for g in ev)
    fp = sum(cm[b][p] for p in ev)
    tn = cm[b][b]
    agg = t.get("aggregate", {})
    return {**{k: agg.get(k) for k in ("macro_f1", "foreground_macro_f1", "accuracy",
                                       "detection_recall", "detection_precision")},
            "class_names": names, "confusion_matrix": cm,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn, "windows": sum(map(sum, cm))}
