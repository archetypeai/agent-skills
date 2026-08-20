#!/usr/bin/env python3
"""End-to-end RED agent run over the Agents API.

Built on the official archetypeai python client (`pip install archetypeai`).

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
  python3 run_red_agent.py --bundle-id bnd_...   # pin one deployment's id
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request

try:
    from archetypeai import ArchetypeAI
except ModuleNotFoundError:  # the only third-party dependency
    sys.exit("This runner needs the official Archetype AI client:\n"
             "    pip install -r requirements.txt   (from this directory)\n"
             "    pip install archetypeai           (or just the package)")

sys.stdout.reconfigure(line_buffering=True)

_CLIENT = None


def client() -> ArchetypeAI:
    """The official client, built once from the environment.

    Owns auth, retries and endpoint mounting. See versioned() for why the
    endpoint is normalised before it is handed over.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = ArchetypeAI(require_key(), api_endpoint=versioned(api_base()))
    return _CLIENT


def versioned(endpoint: str) -> str:
    """Return the endpoint in the form the client expects: WITH the /vX.Y suffix.

    The client uses api_endpoint verbatim for the files API (/v0.5/files) and
    strips the version itself for the versionless agents API. Passing a bare
    root therefore breaks uploads while bundle calls keep working — an empty
    `ApiError: {}` that looks like anything but a missing version path.

    Accept either form so one .env works for every skill in this repo: the
    model skills ship ATAI_API_ENDPOINT with /v0.5, the agent skills without.
    """
    endpoint = endpoint.rstrip("/")
    return endpoint if re.search(r"/v[0-9]+(\.[0-9]+)*$", endpoint) else f"{endpoint}/v0.5"


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as handle:
        for line in handle:
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
    key = os.environ.get("ATAI_API_KEY")
    if not key:
        sys.exit("ATAI_API_KEY is not set (put it in .env)")
    return key


def upload_file(path: str, rename: str | None = None) -> dict:
    """Upload via the client. Source connectors take the returned file_id."""
    if rename:
        # The client uploads under the file's basename, so a rename means
        # staging a copy under the wanted name first.
        staged = os.path.join(tempfile.mkdtemp(), rename)
        shutil.copyfile(path, staged)
        path = staged
    return client().files.local.upload(path)


def resolve_bundle(name: str) -> dict:
    """Resolve a bundle by its stable name, preferring canonical bundles.

    Bundle ids are deployment-scoped — the same pre-packaged bundle carries a
    different bnd_… id in each deployment — so the NAME is the stable handle.
    `GET /agents/bundles?query=` does a case-insensitive substring search over
    name and id (`?name=`/`?search=` are silently ignored), so match the name
    exactly client-side; canonical (platform) bundles win over same-named org
    bundles.
    """
    page = client().agents.bundles.list(query=name, limit=100)
    exact = [bundle for bundle in page.get("data", []) if bundle.get("name") == name]
    for bundle in exact:
        if bundle.get("is_canonical"):
            return bundle
    if exact:
        return exact[0]
    sys.exit(f"no bundle named '{name}' found — the pre-packaged bundles may "
             f"not be published in the deployment you're pointed at. Pass "
             f"--bundle-id for it as a fallback, or contact "
             f"support@archetypeai.dev.")


def poll_agent(agent_id: str, timeout_s: int, interval_s: int = 20) -> str:
    """Poll until terminal, streaming new audit events as they appear.

    Deliberately NOT client.agents.instances.wait_until_done(): that returns as
    soon as `status` is terminal, and a run whose output exists can report
    `failed` when the job poller flakes. Success is judged from /results.
    """
    deadline = time.time() + timeout_s
    seen: set[str] = set()
    status = "running"
    while status == "running" and time.time() < deadline:
        time.sleep(interval_s)
        status = client().agents.instances.get(agent_id).get("status")
        for ev in client().agents.instances.get_events(agent_id).get("data", []):
            marker = f"{ev.get('created_at')}{ev.get('message')}"
            if marker not in seen:
                seen.add(marker)
                print(f"  [{ev.get('level', 'info')}] {ev.get('created_at', '')}  "
                      f"{ev.get('message', '')}")
        print(f"  {time.strftime('%H:%M:%S')} status={status}")
    return status



def download_results(agent_id: str, out_path: str) -> str | None:
    """Fetch /results and save the first output to out_path."""
    results = client().agents.instances.get_results(agent_id)
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
    filename = inner.get("filename")
    # A fitted-artifact ref is an absolute presigned S3 URL that expires (~20
    # min), so fetch it directly. A run output is a platform file — let the
    # client download it by name.
    if ref and ref.startswith("http"):
        with urllib.request.urlopen(ref) as resp:
            open(out_path, "wb").write(resp.read())
    elif filename:
        client().files.local.download(filename, out_path)
    else:
        print("  results carried no download ref")
        return None
    print(f"saved output to {out_path}")
    return out_path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "sample_data", "pump_eval_inc04.csv")

# Pre-packaged canonical bundles (classifier + windowing already pinned).
# Names are the stable handles across deployments; ids are deployment-specific.
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
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="prepared input CSV (defaults to the bundled sample "
                         "slice, resolved next to this script). A ground-truth "
                         "sidecar named <stem>_labels.csv next to it enables "
                         "scoring.")
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

    load_dotenv()
    require_key()

    print(f"uploading {args.csv} ...")
    uploaded = upload_file(args.csv)
    file_id = uploaded["file_id"]
    print(f"  file_id={file_id}")

    # The pre-packaged canonical bundle pins the classifier and its windowing —
    # nothing to fit, nothing to create. (To run your OWN classifier, create a
    # bundle from the `red` blueprint instead: SKILL.md, "Bring your own
    # classifier".)
    if args.bundle_id:
        bundle = client().agents.bundles.get(args.bundle_id)
        print(f"using bundle {bundle['id']}  name='{bundle.get('name')}'")
    else:
        name = args.bundle_name or (
            QUICK_START_BUNDLE_EMBEDDINGS if args.embeddings else QUICK_START_BUNDLE)
        bundle = resolve_bundle(name)
        print(f"resolved bundle '{name}' -> {bundle['id']}"
              f"{'  (canonical)' if bundle.get('is_canonical') else ''}")

    agent = client().agents.bundles.run(
        bundle["id"], source=[{"type": "file", "id": file_id, "format": "csv"}])
    agent_id = agent["id"]
    print(f"starting agent run ...\n  agent_id={agent_id}  status={agent.get('status')}")

    status = poll_agent(agent_id, TIMEOUT_S)
    print(f"agent finished: status={status}")
    # A 'failed' status can hide a successful run when the job poller flakes —
    # always check for results before concluding the run died.
    saved = download_results(agent_id, args.output)
    if status != "completed" and not saved:
        sys.exit(f"run ended '{status}' with no results — inspect the events "
                 f"for {agent_id}")

    if saved and os.path.exists(labels_path):
        score(saved, labels_path, args.window_size or 1024,
              args.normal_class, args.fault_class)
    elif saved:
        print(f"(no label sidecar at {labels_path} — skipping scoring)")


if __name__ == "__main__":
    main()
