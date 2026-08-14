#!/usr/bin/env python3
"""End-to-end RED agent run over the Agent API — stdlib only.

upload CSV -> resolve the pre-packaged "RED Quick Start" bundle by name ->
run -> poll -> download per-window predictions -> score against a
ground-truth sidecar. Nothing to fit and no classifier to supply: the
canonical bundle pins both. (To run your OWN classifier, create a bundle
from the `red` blueprint — SKILL.md, "Bring your own classifier".)

Scoring reports three views, because window averages mislead when the positive
class is under 1% of windows: window-level P/R/F1 under the design's majority
rule, per-incident detection with latency, and the false-alarm rate measured
only on windows containing zero fault rows.

Auth comes from the environment (a local .env is loaded if present):
  ATAI_API_KEY, ATAI_API_ENDPOINT

Usage:
  python3 run_red_agent.py                 # quick-start bundle, sample data
  python3 run_red_agent.py --embeddings    # + embedding_{variate} columns
  python3 run_red_agent.py --bundle-id bnd_...   # pin one environment's id
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def api_base() -> str:
    endpoint = os.environ.get("ATAI_API_ENDPOINT", "").rstrip("/")
    if not endpoint:
        sys.exit("ATAI_API_ENDPOINT is not set")
    return endpoint


def require_key() -> str:
    load_dotenv()
    key = os.environ.get("ATAI_API_KEY")
    if not key:
        sys.exit("ATAI_API_KEY is not set (put it in .env)")
    return key


def agents_base() -> str:
    return f"{api_base()}/agents"


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
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                sys.exit(f"{method} {url} failed after {retries + 1} attempts: {e}")
            wait = 5 * (2 ** attempt)
            print(f"  network error ({e}); retrying in {wait}s "
                  f"[{attempt + 1}/{retries}]")
            time.sleep(wait)


def upload_file(path: str, rename: str | None = None) -> dict:
    """POST a file to /v0.5/files as multipart/form-data."""
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
    req = urllib.request.Request(f"{api_base()}/v0.5/files", data=body, method="POST")
    req.add_header("Authorization", f"Bearer {os.environ['ATAI_API_KEY']}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"upload of {path} failed ({e.code}): {e.read().decode(errors='replace')}")


def resolve_bundle(agents: str, name: str) -> dict:
    """Resolve a bundle by its stable name, preferring canonical bundles.

    Bundle ids are environment-scoped — the same pre-packaged bundle carries a
    different bnd_… id in dev, staging and prod — so the NAME is the stable
    handle. `GET /agents/bundles?query=` does a case-insensitive substring
    search over name and id (`?name=`/`?search=` are silently ignored), so
    match the name exactly client-side; canonical (platform) bundles win over
    same-named org bundles.
    """
    page = request("GET", f"{agents}/bundles?query={urllib.parse.quote(name)}&limit=100")
    exact = [bundle for bundle in page.get("data", []) if bundle.get("name") == name]
    for bundle in exact:
        if bundle.get("is_canonical"):
            return bundle
    if exact:
        return exact[0]
    sys.exit(f"no bundle named '{name}' found in this environment — the "
             f"pre-packaged bundles may not be published here yet (verified on "
             f"Dev; Staging/Prod rollout pending). Pass --bundle-id for this "
             f"environment as a fallback, or contact support@archetypeai.dev.")


def poll_agent(agent_id: str, timeout_s: int, interval_s: int = 20) -> str:
    """Poll until terminal, streaming new audit events as they appear."""
    agents = agents_base()
    deadline = time.time() + timeout_s
    seen: set[str] = set()
    status = "running"
    while status == "running" and time.time() < deadline:
        time.sleep(interval_s)
        status = request("GET", f"{agents}/instances/{agent_id}").get("status")
        for ev in request("GET", f"{agents}/instances/{agent_id}/events").get("data", []):
            marker = f"{ev.get('created_at')}{ev.get('message')}"
            if marker not in seen:
                seen.add(marker)
                print(f"  [{ev.get('level', 'info')}] {ev.get('created_at', '')}  "
                      f"{ev.get('message', '')}")
        print(f"  {time.strftime('%H:%M:%S')} status={status}")
    return status



def download_results(agent_id: str, out_path: str) -> str | None:
    """Fetch /results and save the first output to out_path."""
    results = request("GET", f"{agents_base()}/instances/{agent_id}/results")
    items = results.get("data", results if isinstance(results, list) else [])
    if not items:
        print("  no results returned")
        return None
    # Each result nests the useful fields under an inner "data" object; the
    # `ref` there is a presigned URL that expires (~20 min), so download now.
    print(f"results ({len(items)}):")
    for item in items:
        inner = item.get("data") or {}
        print(f"  {inner.get('filename') or item.get('id')}  "
              f"({inner.get('num_bytes', '?')} bytes)")
    inner = items[0].get("data") or {}
    ref = inner.get("ref") or item.get("ref") or item.get("url")
    if not ref:
        print("  results carried no download ref")
        return None
    # A fitted-artifact ref is an absolute presigned S3 URL; a run-output ref is
    # a relative platform path that needs the API base and the bearer token.
    if ref.startswith("http"):
        with urllib.request.urlopen(ref) as resp:
            payload = resp.read()
    else:
        # Relative refs are rooted at the versioned files API: the ref reads
        # "/files/download/<name>", which resolves under /v0.5.
        payload = request("GET", f"{api_base()}/v0.5{ref}", raw=True)
    open(out_path, "wb").write(payload)
    print(f"saved output to {out_path}")
    return out_path


# Pre-packaged canonical bundles (classifier + windowing already pinned).
# Names are the stable, cross-environment handles; ids differ per environment.
QUICK_START_BUNDLE = "RED Quick Start (Pump Breakdown)"
QUICK_START_BUNDLE_EMBEDDINGS = "RED Quick Start (Pump Breakdown, Embeddings)"
# Class names are NOT hardcoded: `normal` is the conventional baseline label and
# the rare-event class is inferred from the label sidecar, so this runner works
# for any catalog. Override with --normal-class / --fault-class.
DEFAULT_NORMAL = "normal"
TIMEOUT_S = 2 * 60 * 60


def read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pick(row: dict, *candidates: str) -> str | None:
    for c in candidates:
        if c in row:
            return c
    return None


def score(output_path: str, labels_path: str, window_size: int,
          normal: str = DEFAULT_NORMAL, fault: str | None = None) -> None:
    """Score per-window predictions against a row-aligned label sidecar."""
    preds = read_csv_rows(output_path)
    truth = read_csv_rows(labels_path)
    if not preds:
        print("output file has no rows")
        return

    # Infer the rare-event class from the sidecar: whatever is not `normal`.
    if fault is None:
        others = sorted({r["label"] for r in truth} - {normal})
        if len(others) > 1:
            print(f"multiple non-{normal} labels {others} — pass --fault-class")
            return
        # An all-normal slice is legitimate: score it, expect no positives.
        fault = others[0] if others else "\x00_no_fault_label"
    NORMAL, FAULT = normal, fault

    pred_col = pick(preds[0], "predicted_state", "predicted_label", "predicted_class",
                    "prediction", "label")
    # Newer blueprints emit the window span as finish_/start_timestamp; older
    # ones a single `timestamp`. Either way the row is keyed to the window END.
    ts_col = pick(preds[0], "finish_timestamp", "timestamp", "time")
    inv_col = pick(preds[0], "invalid", "invalid_state")
    if not pred_col or not ts_col:
        print(f"could not find prediction/timestamp columns in {list(preds[0])}")
        return

    # Row index by timestamp, so each output row (keyed to its window end) can
    # be mapped back to the window's row span. The platform emits timestamps as
    # floats ("1530962520.0") while the sidecar carries integer seconds, so
    # join on the numeric value rather than the string.
    def as_ts(value: str) -> int:
        return int(float(value))

    index_of = {as_ts(r["timestamp"]): i for i, r in enumerate(truth)}
    labels = [r["label"] for r in truth]

    tp = fp = fn = tn = 0
    invalid = unmatched = 0
    fault_windows_pred: list[tuple[int, int]] = []
    normal_window_total = 0
    false_alarms = 0

    for row in preds:
        if inv_col and str(row.get(inv_col, "")).strip().lower() in ("true", "1"):
            invalid += 1
            continue
        end = index_of.get(as_ts(row[ts_col]))
        if end is None:
            unmatched += 1
            continue
        lo = max(0, end - window_size + 1)
        span = labels[lo:end + 1]
        if not span:
            unmatched += 1
            continue
        n_fault = sum(1 for s in span if s == FAULT)
        # Design-doc labelling: majority wins; if neither holds a majority the
        # rare event takes precedence over normal.
        truth_label = FAULT if n_fault * 2 >= len(span) else NORMAL
        pred_label = row[pred_col].strip()
        is_fault_pred = pred_label == FAULT

        if truth_label == FAULT and is_fault_pred:
            tp += 1
        elif truth_label == FAULT and not is_fault_pred:
            fn += 1
        elif truth_label == NORMAL and is_fault_pred:
            fp += 1
        else:
            tn += 1

        if n_fault == 0:
            normal_window_total += 1
            if is_fault_pred:
                false_alarms += 1
        if is_fault_pred:
            fault_windows_pred.append((lo, end + 1))

    scored = tp + fp + fn + tn
    print(f"\nevaluation vs {os.path.basename(labels_path)} "
          f"({scored} scored windows, {invalid} invalid, {unmatched} unmatched)")
    if not scored:
        return

    def prf(tp_: int, fp_: int, fn_: int) -> tuple[float, float, float]:
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    fp_p, fp_r, fp_f = prf(tp, fp, fn)
    tn_p, tn_r, tn_f = prf(tn, fn, fp)
    print(f"  accuracy: {(tp + tn) / scored:.4f}")
    print(f"  {FAULT:<16} precision={fp_p:.4f} recall={fp_r:.4f} f1={fp_f:.4f} "
          f"(n={tp + fn})")
    print(f"  {NORMAL:<16} precision={tn_p:.4f} recall={tn_r:.4f} f1={tn_f:.4f} "
          f"(n={tn + fp})")
    print(f"  macro-F1: {(fp_f + tn_f) / 2:.4f}")
    print(f"  confusion: tp={tp} fp={fp} fn={fn} tn={tn}")
    if normal_window_total:
        print(f"  false-alarm rate on all-normal windows: "
              f"{false_alarms / normal_window_total:.4f} "
              f"({false_alarms}/{normal_window_total})")

    # Incident-level detection: did we catch each contiguous fault episode?
    episodes, start = [], None
    for i, lab in enumerate(labels):
        if lab == FAULT and start is None:
            start = i
        elif lab != FAULT and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, len(labels)))

    if episodes:
        print(f"  incident-level detection ({len(episodes)} episode(s) in this slice):")
        for a, b in episodes:
            overlapping = [w for w in fault_windows_pred if w[0] < b and a < w[1]]
            if overlapping:
                first_end = min(w[1] for w in overlapping) - 1
                latency = max(0, first_end - a)
                print(f"    rows {a}-{b} (len {b - a}): DETECTED, "
                      f"{len(overlapping)} window(s), first at row {first_end} "
                      f"(+{latency} samples after onset)")
            else:
                print(f"    rows {a}-{b} (len {b - a}): MISSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="sample_data/pump_eval_inc04.csv",
                    help="prepared input CSV. A ground-truth sidecar named "
                         "<stem>_labels.csv next to it enables scoring.")
    ap.add_argument("--embeddings", action="store_true",
                    help=f"run '{QUICK_START_BUNDLE_EMBEDDINGS}' instead — each "
                         "prediction row also carries the Newton Omega encoder "
                         "embedding for its window: one embedding_{variate} "
                         "column per sensor channel, each a 768-d vector. "
                         "Expect a much larger output file: 740 MB vs 381 KB "
                         "on the sample slice")
    ap.add_argument("--bundle-name", default=None,
                    help="run a different bundle by its exact name (default: "
                         f"'{QUICK_START_BUNDLE}'). Canonical bundles win when "
                         "names collide.")
    ap.add_argument("--bundle-id", default=None,
                    help="pin a bundle by bnd_… id instead of resolving by "
                         "name. Ids are environment-scoped — use only when the "
                         "pre-packaged bundle isn't published in this "
                         "environment yet.")
    ap.add_argument("--window-size", type=int, default=64,
                    help="the window the bundle's classifier was fit with. Used "
                         "only to reconstruct window spans when scoring — the "
                         "agent inherits windowing from the classifier snapshot")
    ap.add_argument("--output", default="red-output.csv")
    ap.add_argument("--normal-class", default=DEFAULT_NORMAL,
                    help="baseline class name in the label sidecar")
    ap.add_argument("--fault-class", default=None,
                    help="rare-event class name; inferred from the sidecar when omitted")
    ap.add_argument("--score-only", metavar="OUTPUT_CSV",
                    help="skip the platform, score an already-downloaded output")
    args = ap.parse_args()

    labels_path = args.csv.replace(".csv", "_labels.csv")

    if args.score_only:
        if not os.path.exists(labels_path):
            sys.exit(f"no label sidecar at {labels_path}")
        score(args.score_only, labels_path, args.window_size or 1024,
              args.normal_class, args.fault_class)
        return

    if args.bundle_id and args.bundle_name:
        sys.exit("pass --bundle-id or --bundle-name, not both")
    if args.embeddings and (args.bundle_id or args.bundle_name):
        sys.exit("--embeddings picks a specific pre-packaged bundle; don't "
                 "combine it with --bundle-id/--bundle-name")

    require_key()
    agents = agents_base()

    print(f"uploading {args.csv} ...")
    uploaded = upload_file(args.csv)
    file_id = uploaded["file_id"]
    print(f"  file_id={file_id}")

    # The pre-packaged canonical bundle pins the classifier and its windowing —
    # nothing to fit, nothing to create. (To run your OWN classifier, create a
    # bundle from the `red` blueprint instead: SKILL.md, "Bring your own
    # classifier".)
    if args.bundle_id:
        bundle = request("GET", f"{agents}/bundles/{args.bundle_id}")
        print(f"using bundle {bundle['id']}  name='{bundle.get('name')}'")
    else:
        name = args.bundle_name or (
            QUICK_START_BUNDLE_EMBEDDINGS if args.embeddings else QUICK_START_BUNDLE)
        bundle = resolve_bundle(agents, name)
        print(f"resolved bundle '{name}' -> {bundle['id']}"
              f"{'  (canonical)' if bundle.get('is_canonical') else ''}")

    agent = request("POST", f"{agents}/bundles/{bundle['id']}/run", body={
        "connectors": {"source": [{"type": "file", "id": file_id, "format": "csv"}]},
    })
    agent_id = agent["id"]
    print(f"starting agent run ...\n  agent_id={agent_id}  status={agent.get('status')}")

    status = poll_agent(agent_id, TIMEOUT_S)
    print(f"agent finished: status={status}")
    # A 'failed' status can hide a successful run when the job poller flakes —
    # always check for results before concluding the run died.
    saved = download_results(agent_id, args.output)
    if status != "completed" and not saved:
        sys.exit(f"run ended '{status}' with no results — inspect "
                 f"{agents}/instances/{agent_id}/events")

    if saved and os.path.exists(labels_path):
        score(saved, labels_path, args.window_size or 1024,
              args.normal_class, args.fault_class)
    elif saved:
        print(f"(no label sidecar at {labels_path} — skipping scoring)")


if __name__ == "__main__":
    main()
