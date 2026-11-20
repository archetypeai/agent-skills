#!/usr/bin/env python3
"""Build and run a custom agent (a blueprint you write from the platform's agent nodes)
on Agent Lifecycle 2.0. Admin only on production: blueprint creation needs the admin role.

    python custom_agent.py registry                       # node keys + ports
    python custom_agent.py registry OmegaEncoderNode      # one node: contract, config schema, defaults
    python custom_agent.py check my-agent.yaml            # every key and port exists on this platform
    python custom_agent.py create my-agent.yaml --key my-agent \
        --values '{"timestamp_column": "timestamp", "data_columns": ["a", "b"]}'
    python custom_agent.py batch --input data.csv --out results/
    python custom_agent.py serve-records --input data.csv --columns a,b --window 256 --step 128
    python custom_agent.py serve-video --input clip.mp4 --prompt "Summarize this video."
    # trainable graphs only (a fittable head + a `metrics:` section):
    python custom_agent.py create omega-ad-custom.yaml --key my-ad --no-package
    python custom_agent.py optimize --roles tep/roles --spec optimize/omega-ad-tep.json
    python custom_agent.py promote --key my-ad-best      # best trial -> blueprint + package
    python custom_agent.py eval --roles tep/roles --spec optimize/omega-ad-tep.json
    python custom_agent.py cleanup                        # stop/delete what this state file started

State (project, blueprint, package, runs) goes to --state (default custom_agent_state.json),
so each step picks up the IDs of the one before. Stdlib only; reads ATAI_API_KEY and
ATAI_API_ENDPOINT from the environment or a .env file next to the script or in the cwd.
"""
import argparse, base64, csv, json, os, re, sys, time, urllib.error, urllib.request, uuid

TERMINAL = ("completed", "failed", "cancelled")


# ---------------------------------------------------------------- environment + HTTP

def load_env():
    for d in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))
    key, api = os.environ.get("ATAI_API_KEY"), os.environ.get("ATAI_API_ENDPOINT")
    if not key or not api:
        sys.exit("set ATAI_API_KEY and ATAI_API_ENDPOINT (e.g. https://platform.u1.archetypeai.app/api)")
    return key, api.rstrip("/")


KEY, API = None, None


def call(method, path, body=None, *, url=None, raw=False, ok=(), headers=None, data=None):
    """One API call. Raises SystemExit with the platform's error unless its status is in `ok`."""
    h = {"Authorization": f"Bearer {KEY}"}
    if body is not None:
        data, h["Content-Type"] = json.dumps(body).encode(), "application/json"
    h.update(headers or {})
    req = urllib.request.Request(url or API + path, method=method, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = r.read()
            return out if raw else json.loads(out or b"{}")
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")
        if e.code in ok:
            return {"_status": e.code, **(json.loads(text) if text.startswith("{") else {"text": text})}
        sys.exit(f"{method} {path or url} -> {e.code}: {text[:800]}")


def wait(path, waiting, every=10, timeout=3600):
    t0, obj = time.time(), call("GET", path)
    while obj.get("status") in waiting:
        if time.time() - t0 > timeout:
            sys.exit(f"{path} still {obj['status']} after {timeout}s")
        time.sleep(every)
        obj = call("GET", path)
    return obj


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- state

class State(dict):
    def __init__(self, path):
        super().__init__(json.load(open(path)) if os.path.exists(path) else {})
        self.path = path

    def save(self):
        json.dump(self, open(self.path, "w"), indent=1)


# ---------------------------------------------------------------- registry + local check

def registry():
    reg = call("GET", "/agents/nodes/registry")
    return {n["key"]: n for n in reg.get("nodes", []) + reg.get("connectors", [])}


def ports(entry):
    c = entry.get("contract") or {}
    return ({p["name"]: p.get("type") for p in c.get("inputs", [])},
            {p["name"]: p.get("type") for p in c.get("outputs", [])})


def cmd_registry(a):
    reg = registry()
    if a.node:
        if a.node not in reg:
            sys.exit(f"{a.node} is not on this platform; keys: {', '.join(sorted(reg))}")
        print(json.dumps(reg[a.node], indent=1))
        return
    for k in sorted(reg):
        i, o = ports(reg[k])
        fmt = lambda d: ", ".join(f"{n}:{t}" for n, t in d.items()) or "-"
        print(f"{k:38} in [{fmt(i)}]  out [{fmt(o)}]")
        if a.describe:
            print(f"{'':38} {reg[k].get('description', '')}")


def blueprint_parts(text):
    """Pull node ids -> keys and edges out of the YAML without a YAML dependency.
    Good enough for a pre-check; the platform's own validation at create is the authority."""
    nodes, section, current = {}, None, None
    for line in text.splitlines():
        if re.match(r"^\S", line):
            section = line.split(":")[0].strip()
            continue
        m = re.match(r"^  (\w+):\s*$", line)
        if m and section in ("nodes", "connectors"):
            current = m.group(1)
            continue
        m = re.match(r"^    key:\s*(\S+)", line)
        if m and current and section in ("nodes", "connectors"):
            nodes[current] = m.group(1).strip("'\"")
    edges = re.findall(r"from:\s*([\w]+)\.([\w]+)\s*,\s*to:\s*([\w]+)\.([\w]+)", text)
    return nodes, edges


def check(text, reg):
    nodes, edges = blueprint_parts(text)
    problems = []
    for nid, key in nodes.items():
        if key not in reg:
            problems.append(f"node `{nid}`: `{key}` is not in this platform's registry")
    for fn, fp, tn, tp in edges:
        for nid, port, side in ((fn, fp, 1), (tn, tp, 0)):
            if nid not in nodes:
                problems.append(f"edge {fn}.{fp} -> {tn}.{tp}: no node `{nid}`")
            elif nodes[nid] in reg:
                names = ports(reg[nodes[nid]])[side]
                # nodes with configurable ports (TeeNode, ConcatColumnsNode, ...) name them in config
                if names and port not in names and not any(n in ("*", "...") for n in names):
                    cfg_ports = re.search(rf"^  {nid}:\s*\n(?:    .*\n)*?\s+ports:\s*\[([^\]]*)\]", text, re.M)
                    if not (cfg_ports and port in [p.strip() for p in cfg_ports.group(1).split(",")]):
                        problems.append(f"edge {fn}.{fp} -> {tn}.{tp}: `{nodes[nid]}` has no "
                                        f"{'output' if side else 'input'} `{port}` (has {', '.join(names)})")
    sources = [k for k in nodes.values() if k.endswith("Source")]
    sinks = [k for k in nodes.values() if k.endswith("Sink")]
    if len(sources) != 1 or len(sinks) != 1:
        problems.append(f"{len(sources)} source(s) and {len(sinks)} sink(s): a deployable agent needs exactly one of each")
    return nodes, edges, problems


def cmd_check(a):
    text = open(a.blueprint).read()
    nodes, edges, problems = check(text, registry())
    print(f"{len(nodes)} nodes/connectors, {len(edges)} edges")
    for p in problems:
        print("  ✗", p)
    print("ok" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


# ---------------------------------------------------------------- create

def cmd_create(a, st):
    text = open(a.blueprint).read()
    _, _, problems = check(text, registry())
    if problems and not a.force:
        sys.exit("pre-check failed (use --force to let the platform decide):\n  " + "\n  ".join(problems))
    if a.project:
        st["project"] = a.project
    if "project" not in st:
        st["project"] = call("POST", "/projects", {"name": a.project_name, "description": "custom agents"})["id"]
        st.save()
    log(f"project {st['project']}")
    template_key = f"{a.key}-template" if a.values else a.key
    r = call("POST", "/agents/blueprints", {"blueprint_key": template_key, "project_id": st["project"],
                                             "yaml_document": text}, ok=(400, 403, 409))
    if "_status" in r:
        sys.exit(f"create refused ({r['_status']}): {json.dumps(r.get('errors') or r)[:1200]}")
    st["template_blueprint" if a.values else "blueprint"] = r["id"]
    st.save()
    log(f"blueprint {template_key} {r['id']}")
    if a.values:
        # A package carries no values: pin them on a copy, and package the copy.
        r = call("POST", f"/agents/blueprints/{r['id']}/copy", {
            "project_id": st["project"], "blueprint_key": a.key,
            "overrides": {"values": json.loads(a.values)}}, ok=(400, 409))
        if "_status" in r:
            sys.exit(f"copy refused ({r['_status']}): {json.dumps(r.get('errors') or r)[:1200]}")
        st["blueprint"] = r["id"]
        st.save()
        log(f"blueprint {a.key} {r['id']} (values pinned)")
    if not a.no_package:
        r = call("POST", "/agents/packages", {"blueprint_ids": [st["blueprint"]],
                                              "target": {"kind": "platform"}, "name": a.key})
        st["package"] = r["id"]
        st.save()
        log(f"package {r['id']}")


# ---------------------------------------------------------------- upload

def upload(path, kind):
    """CSV: initiate -> PUT presigned URL -> complete; the file is then referenced by its name.
    Video: multipart POST /files. File names must be unique in the org, so add a timestamp."""
    base, ext = os.path.splitext(os.path.basename(path))
    name = f"{base}_{time.strftime('%Y%m%d%H%M%S')}{ext}"
    body = open(path, "rb").read()
    if kind == "csv":
        u = call("POST", "/files/uploads/initiate", {"filename": name, "file_type": "text/csv", "num_bytes": len(body)})
        req = urllib.request.Request(u["parts"][0]["url"], method="PUT", data=body)
        with urllib.request.urlopen(req, timeout=600) as r:
            etag = r.headers["ETag"]
        call("POST", f"/files/uploads/{u['upload_id']}/complete",
             {"parts": [{"part_number": 1, "part_token": etag}]})
        return name, name
    boundary = uuid.uuid4().hex
    data = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: video/mp4\r\n\r\n").encode() + body + f"\r\n--{boundary}--\r\n".encode()
    r = call("POST", "/files", data=data, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    if not r.get("is_valid", True):
        sys.exit(f"upload refused: {json.dumps(r)[:400]}")
    return r["file_id"], r.get("file_uid")


# ---------------------------------------------------------------- batch

def job_errors(job):
    ev = call("GET", f"/batch/jobs/{job}/events?limit=100", ok=(404,))
    return [e for e in ev.get("data") or [] if e.get("level") == "ERROR"], ev.get("data") or []


def cmd_batch(a, st):
    pkg = a.package or st.get("package") or sys.exit("no package: run create first or pass --package")
    kind = "mp4" if a.input.lower().endswith(".mp4") else "csv"
    if "batch_deployment" not in st or a.new_deployment:
        st["batch_deployment"] = call("POST", "/agents/deployments", {"package_id": pkg, "kind": "batch"})["id"]
        st.save()
    wait(f"/agents/deployments/{st['batch_deployment']}", ("deploying",), every=5)
    file_id, _ = upload(a.input, kind)
    ses = call("POST", "/agents/sessions", {"deployment_id": st["batch_deployment"],
                                            "connectors": {"source": [{"type": "file", "id": file_id, "format": kind}]}})["id"]
    st.setdefault("sessions", []).append(ses)
    st.save()
    log(f"batch deployment {st['batch_deployment']} session {ses} on {file_id}")
    t0, status, job = time.time(), None, None
    while True:
        s = call("GET", f"/agents/sessions/{ses}")
        status, job = s["status"], s.get("job_id") or job
        if status in TERMINAL:
            break
        # A session can stay `running` after its pod failed: the job's events are the signal.
        if job:
            errors, _ = job_errors(job)
            if errors:
                status = "job-failed"
                break
        if time.time() - t0 > a.timeout:
            status = "timeout"
            break
        time.sleep(15)
    log(f"{status} after {time.time() - t0:.0f}s (job {job})")
    if status != "completed":
        if job:
            for e in reversed(job_errors(job)[1][:25]):
                print(f"   {e.get('level'):7} {e.get('message')}")
        if status not in TERMINAL:
            call("POST", f"/agents/sessions/{ses}/cancel", {}, ok=(400, 404, 409))
            log("session cancelled")
        sys.exit(1)
    os.makedirs(a.out, exist_ok=True)
    root = API[:-4] if API.endswith("/api") else API
    for r in call("GET", f"/agents/sessions/{ses}/results")["data"]:
        d = r["data"]
        body = call("GET", None, url=root + d["ref"], raw=True)   # ref is a root path
        path = os.path.join(a.out, d.get("filename") or os.path.basename(d["ref"]))
        open(path, "wb").write(body)
        log(f"output {path} ({len(body)} bytes)")


# ---------------------------------------------------------------- serving

def serving(st, pkg, fn):
    """Serving deployment -> session -> fn(session) -> session and deployment removed."""
    srv = {}
    try:
        t0 = time.time()
        srv["deployment"] = call("POST", "/agents/deployments", {"package_id": pkg, "kind": "serving"})["id"]
        d = wait(f"/agents/deployments/{srv['deployment']}", ("deploying",), every=5, timeout=3600)
        log(f"serving deployment {srv['deployment']} {d['status']} after {time.time() - t0:.0f}s {d.get('error') or ''}")
        if d["status"] != "ready":
            sys.exit(1)
        srv["session"] = call("POST", "/agents/sessions", {"deployment_id": srv["deployment"]})["id"]
        s = wait(f"/agents/sessions/{srv['session']}", ("pending",), every=5)
        log(f"session {srv['session']} {s['status']}")
        if s["status"] != "running":
            sys.exit(1)
        return fn(srv["session"])
    finally:
        if srv.get("session"):
            call("POST", f"/agents/sessions/{srv['session']}/cancel", {}, ok=(400, 404, 409))
            call("DELETE", f"/agents/sessions/{srv['session']}", ok=(400, 404, 409))
        if srv.get("deployment"):
            d = call("GET", f"/agents/deployments/{srv['deployment']}")
            if d["status"] not in ("stopped", "failed"):
                call("POST", f"/agents/deployments/{srv['deployment']}/stop", {}, ok=(400, 404, 409))
                wait(f"/agents/deployments/{srv['deployment']}", ("ready", "stopping"), every=5)
            call("DELETE", f"/agents/deployments/{srv['deployment']}", ok=(400, 404, 409))
            log("serving session and deployment removed")


def cmd_serve_records(a, st):
    pkg = a.package or st.get("package")
    rows = list(csv.DictReader(open(a.input)))
    cols = a.columns.split(",")
    data = [[float(r[c]) for r in rows] for c in cols]
    n = (len(rows) - a.window) // a.step + 1
    if a.max_windows:
        n = min(n, a.max_windows)

    def run(ses):
        out, lat = [], []
        for k in range(n):
            lo = k * a.step
            body = {"query": "", "events": [{"type": "data.numeric_array",
                                             "event_data": {"contents": [ch[lo:lo + a.window] for ch in data]}}]}
            t = time.time()
            ans = call("POST", f"/agents/sessions/{ses}/query", body)
            lat.append(time.time() - t)
            out.append((ans.get("response") or {}).get("data"))
        lat.sort()
        log(f"{n} windows, median {lat[len(lat) // 2]:.2f}s / query")
        return out

    out = serving(st, pkg, run)
    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    with open(a.save, "w") as f:
        for recs in out:
            for r in recs or []:
                f.write(json.dumps(r) + "\n")
    log(f"served records -> {a.save}")


def cmd_serve_video(a, st):
    pkg = a.package or st.get("package")
    b64 = base64.b64encode(open(a.input, "rb").read()).decode()

    def run(ses):
        t = time.time()
        ans = call("POST", f"/agents/sessions/{ses}/query", {"query": a.prompt, "file_ids": [f"base64:{b64}"]})
        log(f"answered in {time.time() - t:.1f}s")
        return ans

    ans = serving(st, pkg, run)
    print(json.dumps(ans, indent=1)[:4000])
    if a.save:
        json.dump(ans, open(a.save, "w"), indent=1)


# ---------------------------------------------------------------- optimize -> promote -> eval
# Only for a trainable graph: a head with fittable state (AnomalyDetectorNode, ClassifierNode,
# MultiheadClassifierNode, OutlierDetectorNode) and a `metrics:` section naming what to score.

def upload_file(path, prefix):
    """Multipart POST /files (CSV or video); returns the file_id that examples reference."""
    name = f"{prefix}_{os.path.basename(path)}"
    boundary = uuid.uuid4().hex
    ctype = "video/mp4" if path.lower().endswith(".mp4") else "text/csv"
    with open(path, "rb") as f:
        payload = f.read()
    data = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    r = call("POST", "/files", data=data, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    if not r.get("is_valid", True):
        sys.exit(f"upload {path} refused: {json.dumps(r)[:400]}")
    return r["file_id"]


def role_files(st, roles, role):
    """Upload every CSV in <roles>/<role>/ once; names get a stamp (file names are unique per org)."""
    files = st.setdefault("files", {})
    stamp = st.setdefault("stamp", time.strftime("%Y%m%d%H%M%S"))
    out = []
    for fn in sorted(os.listdir(os.path.join(roles, role))):
        if fn.endswith(".csv"):
            k = f"{role}/{fn}"
            if k not in files:
                files[k] = upload_file(os.path.join(roles, role, fn), f"{stamp}_{role}")
                st.save()
            out.append((fn, files[k]))
    return out


def examples(pairs, ground_truth):
    return [{"name": n, "inputs": [{"type": "file", "id": f, "format": "csv"}], "ground_truth": ground_truth}
            for n, f in pairs]


def wait_job(path, label, every=30, timeout=4 * 3600):
    t0 = time.time()
    while True:
        obj = call("GET", path)
        if obj["status"] in TERMINAL:
            return obj
        if time.time() - t0 > timeout:
            sys.exit(f"{label} still {obj['status']} after {timeout}s")
        log(f"{label} {obj['status']} {json.dumps(obj.get('progress') or {})}")
        time.sleep(every)


def headline(report, target):
    agg = (((report or {}).get("targets") or {}).get(target) or {}).get("aggregate") or {}
    return {k: round(v, 4) for k, v in agg.items() if isinstance(v, (int, float))}


def load_spec(a):
    with open(a.spec) as f:
        spec = json.load(f)
    values = dict(spec.get("values") or {})
    man = os.path.join(a.roles, "manifest.json")
    if "data_columns" not in values and os.path.exists(man):
        with open(man) as f:
            m = json.load(f)
        values.setdefault("timestamp_column", m.get("timestamp_column"))
        values["data_columns"] = m.get("channels") or m.get("data_columns")
    return spec, values


def cmd_optimize(a, st):
    """Fit the head on library/ and search `values` on validation/. Optimizes the template blueprint
    itself (values come in as overrides), not a packaged copy."""
    spec, values = load_spec(a)
    bp = a.blueprint or st.get("template_blueprint") or st.get("blueprint") or sys.exit("no blueprint: run create")
    gt = spec["ground_truth"]
    if "optimization" not in st:
        body = {"name": spec.get("name", "custom agent optimization"), "blueprint_id": bp,
                "primary": spec["primary"], "overrides": {"values": values},
                "search_space": {"parameters": spec.get("search_space", {})},
                "budget": {"max_trials": spec.get("max_trials", 4)},
                "training_examples": examples(role_files(st, a.roles, "library"), gt["training"]),
                "validation_examples": examples(role_files(st, a.roles, "validation"), gt["scored"])}
        r = call("POST", "/agents/optimizations", body, ok=(400, 409, 422))
        if "_status" in r:
            sys.exit(f"optimization refused ({r['_status']}): {json.dumps(r.get('errors') or r)[:1200]}")
        st["optimization"] = r["id"]
        st.save()
    opt_id = st["optimization"]
    log(f"optimization {opt_id}")
    opt = wait_job(f"/agents/optimizations/{opt_id}", "optimization")
    trials = call("GET", f"/agents/optimizations/{opt_id}/trials?limit=100").get("data", [])
    target = spec["primary"].split(".")[0]
    for t in sorted(trials, key=lambda t: t.get("objective_value") or -1, reverse=True):
        log(f"  {t['id']} {t['status']} {json.dumps((t.get('trial_values') or {}).get('values'))} "
            f"objective={t.get('objective_value')} {headline(t.get('metrics_report'), target)}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "optimization.json"), "w") as f:
        json.dump({"optimization": opt, "trials": trials}, f, indent=1)
    if opt["status"] != "completed":
        sys.exit(f"optimization {opt['status']}: {opt.get('error')}")


def cmd_promote(a, st):
    """Promote a trial (default: the best completed one) to a blueprint that pins its values and
    carries the fitted head as a registry artifact; then package it."""
    opt_id = st.get("optimization") or sys.exit("no optimization: run optimize")
    trials = call("GET", f"/agents/optimizations/{opt_id}/trials?limit=100").get("data", [])
    done = [t for t in trials if t["status"] == "completed"]
    best = next((t for t in done if t["id"] == a.trial), None) if a.trial else \
        max(done, key=lambda t: t.get("objective_value") or -1, default=None)
    if not best:
        sys.exit("no completed trial to promote")
    r = call("POST", f"/agents/optimizations/{opt_id}/trials/{best['id']}/promote",
             {"name": a.name or f"{a.key} (promoted)", "blueprint_key": a.key}, ok=(400, 403, 409))
    if "_status" in r:
        sys.exit(f"promote refused ({r['_status']}): {json.dumps(r.get('errors') or r)[:1200]}")
    st.update(best_trial=best["id"], promoted=r["id"], blueprint=r["id"])
    st.save()
    log(f"promoted {best['id']} (objective {best.get('objective_value')}) -> {a.key} {r['id']}")
    if not a.no_package:
        st["package"] = call("POST", "/agents/packages", {"blueprint_ids": [r["id"]],
                                                         "target": {"kind": "platform"}, "name": a.key})["id"]
        st.pop("batch_deployment", None)
        st.save()
        log(f"package {st['package']}")


def cmd_eval(a, st):
    """Score the promoted blueprint once on the held-out test/ role, at its own pinned values."""
    spec, _ = load_spec(a)
    bp = st.get("promoted") or sys.exit("no promoted blueprint: run promote")
    if "eval" not in st:
        body = {"name": spec.get("name", "custom agent") + " test", "blueprint_id": bp,
                "examples": examples(role_files(st, a.roles, "test"), spec["ground_truth"]["scored"]),
                "emit_predictions": True}
        st["eval"] = call("POST", "/agents/evals", body)["id"]
        st.save()
    ev = wait_job(f"/agents/evals/{st['eval']}", "eval")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(ev, f, indent=1)
    log(f"eval {st['eval']} {ev['status']} {ev.get('error') or ''}")
    log(f"test: {headline(ev.get('metrics_report'), spec['primary'].split('.')[0])}")


# ---------------------------------------------------------------- cleanup

def cmd_cleanup(a, st):
    for ses in st.get("sessions", []):
        s = call("GET", f"/agents/sessions/{ses}", ok=(404,))
        if s.get("status") not in TERMINAL + (None,):
            call("POST", f"/agents/sessions/{ses}/cancel", {}, ok=(400, 404, 409))
    if st.get("batch_deployment"):
        call("POST", f"/agents/deployments/{st['batch_deployment']}/stop", {}, ok=(400, 404, 409))
        log(f"batch deployment {st['batch_deployment']} stopped")
    if a.packages and st.get("package"):
        call("DELETE", f"/agents/packages/{st['package']}", ok=(400, 404, 409))
        log(f"package {st['package']} deleted")


def main():
    global KEY, API
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", default="custom_agent_state.json")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("registry"); s.add_argument("node", nargs="?"); s.add_argument("--describe", action="store_true")
    s = sub.add_parser("check"); s.add_argument("blueprint")
    s = sub.add_parser("create"); s.add_argument("blueprint"); s.add_argument("--key", required=True)
    s.add_argument("--values", help="JSON object pinned on a copy (packages carry no values)")
    s.add_argument("--project"); s.add_argument("--project-name", default="Custom Agents")
    s.add_argument("--no-package", action="store_true"); s.add_argument("--force", action="store_true")
    s = sub.add_parser("batch"); s.add_argument("--input", required=True); s.add_argument("--out", default="results")
    s.add_argument("--package"); s.add_argument("--timeout", type=int, default=1800)
    s.add_argument("--new-deployment", action="store_true")
    s = sub.add_parser("serve-records"); s.add_argument("--input", required=True); s.add_argument("--columns", required=True)
    s.add_argument("--window", type=int, required=True); s.add_argument("--step", type=int, required=True)
    s.add_argument("--max-windows", type=int); s.add_argument("--package")
    s.add_argument("--save", default="results/served.jsonl")
    s = sub.add_parser("serve-video"); s.add_argument("--input", required=True); s.add_argument("--prompt", required=True)
    s.add_argument("--package"); s.add_argument("--save")
    s = sub.add_parser("optimize"); s.add_argument("--roles", required=True, help="dir with library/ validation/ test/")
    s.add_argument("--spec", required=True, help="JSON: primary, values, search_space, max_trials, ground_truth")
    s.add_argument("--blueprint"); s.add_argument("--out", default="results")
    s = sub.add_parser("promote"); s.add_argument("--key", required=True); s.add_argument("--name")
    s.add_argument("--trial"); s.add_argument("--no-package", action="store_true")
    s = sub.add_parser("eval"); s.add_argument("--roles", required=True); s.add_argument("--spec", required=True)
    s.add_argument("--out", default="results")
    s = sub.add_parser("cleanup"); s.add_argument("--packages", action="store_true", help="also delete the package")
    a = p.parse_args()
    KEY, API = load_env()
    st = State(a.state)
    {"registry": lambda: cmd_registry(a), "check": lambda: cmd_check(a), "create": lambda: cmd_create(a, st),
     "batch": lambda: cmd_batch(a, st), "serve-records": lambda: cmd_serve_records(a, st),
     "serve-video": lambda: cmd_serve_video(a, st), "optimize": lambda: cmd_optimize(a, st),
     "promote": lambda: cmd_promote(a, st), "eval": lambda: cmd_eval(a, st),
     "cleanup": lambda: cmd_cleanup(a, st)}[a.cmd]()


if __name__ == "__main__":
    main()
