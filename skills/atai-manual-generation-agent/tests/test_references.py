"""Network-free tests for the MGA agent skill references.

    python -m pytest skills/atai-manual-generation-agent/tests -q

No credentials, no network, no GPU. Everything here checks request shapes built
in memory, the lifecycle against a fake API, or the committed sample outputs.
"""
import contextlib
import importlib.util
import io
import json
import os
import tokenize
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "..", "references")
DATA = os.path.join(REFS, "sample_data")

spec = importlib.util.spec_from_file_location(
    "run_mga_agent", os.path.join(REFS, "run_mga_agent.py"))
mga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mga)

with open(os.path.join(REFS, "run_mga_agent.py")) as _f:
    RUNNER_SRC = _f.read()


def code_only(src: str) -> str:
    """Source with comments and string literals removed.

    The runner's docstrings deliberately discuss the things it must NOT do
    ("poll /logs, NOT /events"), so a naive grep over the whole file reports a
    violation for prose that is in fact the warning against it. These tests care
    about executable code.
    """
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


RUNNER_CODE = code_only(RUNNER_SRC)


class TestEndpointResolution(unittest.TestCase):
    """Any endpoint form reduces to the deployment root; every surface is under {root}/api."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("ATAI_API_KEY", "ATAI_API_ENDPOINT")}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_every_form_reduces_to_the_root(self):
        for form in ("", "/", "/api", "/api/", "/v0.5", "/v0.4", "/v1"):
            os.environ["ATAI_API_ENDPOINT"] = f"https://platform.u1.archetypeai.app{form}"
            self.assertEqual(mga.root(), "https://platform.u1.archetypeai.app", form)

    def test_agents_and_files_live_under_api(self):
        os.environ["ATAI_API_ENDPOINT"] = "https://platform.u1.archetypeai.app/api"
        self.assertEqual(mga.agents_url("sessions"),
                         "https://platform.u1.archetypeai.app/api/agents/sessions")

    def test_quoted_dotenv_value_is_unquoted(self):
        import tempfile
        os.environ.pop("ATAI_API_ENDPOINT", None)
        fd, path = tempfile.mkstemp(suffix=".env")
        with os.fdopen(fd, "w") as f:
            f.write('ATAI_API_ENDPOINT="https://platform.u1.archetypeai.app/api"\n')
        try:
            mga.load_dotenv(path)
            self.assertEqual(os.environ["ATAI_API_ENDPOINT"],
                             "https://platform.u1.archetypeai.app/api")
        finally:
            os.unlink(path)

    def test_session_result_ref_resolves_against_the_root(self):
        """A session's ref is a root path that already carries /api; appended to
        .../api it would 404 as /api/api/..."""
        os.environ["ATAI_API_ENDPOINT"] = "https://platform.u1.archetypeai.app/api"
        self.assertEqual(mga.result_url("/api/files/fil_x/content"),
                         "https://platform.u1.archetypeai.app/api/files/fil_x/content")
        self.assertEqual(mga.result_url("https://bucket.example/x?sig=1"),
                         "https://bucket.example/x?sig=1")


class TestWiredValueDetection(unittest.TestCase):
    """A value is honoured only if declared AND referenced as ${values.<key>}.

    This is the check that distinguishes 'accepted' from 'has any effect' — the
    API returns 201 either way.
    """

    # Mirrors the canonical blueprint as of 2026-08-11 (blp_5wtsdwmp2q9hm87pabye2h9c1a):
    # max_new_tokens is wired now; prompt still is not.
    ACTIVE = {"document": {
        "values": {"max_frames": 16, "size": 224, "max_new_tokens": 16384,
                   "prompt": "Generate a concise, ordered list of distinct steps."},
        "nodes": {"video_reader": {"config": {"max_frames": "${values.max_frames}",
                                              "size": "${values.size}"}},
                  "fusion": {"config": {"model": "${models.fusion}",
                                        "max_new_tokens": "${values.max_new_tokens}"}}},
        "connectors": {"source": {"config": {
            "default_text": "${values.prompt}", "text_extensions": []}}}}}

    # How the same blueprint looked 2026-08-11..08-12, when `prompt` was hardcoded.
    PROMPT_HARDCODED = {"document": {
        "values": {"max_frames": 16, "max_new_tokens": 16384},
        "nodes": {"fusion": {"config": {"max_new_tokens": "${values.max_new_tokens}"}}},
        "connectors": {"source": {"config": {
            "default_text": "Generate a concise, ordered list of distinct steps "
                            "with 10 steps or less.", "text_extensions": []}}}}}

    WITH_PARAMS = {"document": {
        "values": {"max_frames": 16, "max_new_tokens": 256, "prompt": "…"},
        "nodes": {"video_reader": {"config": {"max_frames": "${values.max_frames}"}},
                  "fusion": {"config": {"model": "${models.fusion}",
                                        "max_new_tokens": "${values.max_new_tokens}"}}},
        "connectors": {"source": {"config": {"default_text": "${values.prompt}"}}}}}

    def test_active_blueprint_honours_prompt_again(self):
        """Since 2026-08-12 prompt is wired as ${values.prompt} again."""
        ignored = mga.inert_values(
            self.ACTIVE, {"max_frames": 64, "max_new_tokens": 16384, "prompt": "x"})
        self.assertEqual(ignored, [])

    def test_hardcoded_prompt_shape_is_detected_as_inert(self):
        """The regression this preflight exists to catch, using the real shape it
        had while `prompt` was unreachable."""
        ignored = mga.inert_values(self.PROMPT_HARDCODED, {"prompt": "x"})
        self.assertEqual(ignored, ["prompt"])

    def test_active_blueprint_honours_max_frames_and_max_new_tokens(self):
        self.assertEqual(
            mga.inert_values(self.ACTIVE, {"max_frames": 64, "max_new_tokens": 4096}),
            [])

    def test_hardcoded_prompt_is_what_capped_the_step_count(self):
        text = self.PROMPT_HARDCODED["document"]["connectors"]["source"]["config"]
        self.assertIn("10 steps or less", text["default_text"])
        self.assertEqual(text["text_extensions"], [],
                         "text inputs disabled, so default_text could not be overridden")

    def test_superseded_blueprint_honoured_all_three(self):
        """Historic: this shape is how `prompt` used to be reachable. It no longer
        resolves — kept so inert_values is exercised against a wired prompt."""
        self.assertEqual(mga.inert_values(
            self.WITH_PARAMS,
            {"max_frames": 64, "max_new_tokens": 2048, "prompt": "x"}), [])

    def test_declared_but_unreferenced_counts_as_inert(self):
        """The subtle case: present in `values`, wired to nothing."""
        bp = {"document": {"values": {"ghost": 1},
                           "nodes": {"n": {"config": {"model": "x"}}},
                           "connectors": {}}}
        self.assertEqual(mga.inert_values(bp, {"ghost": 2}), ["ghost"])


class TestSampleOutputs(unittest.TestCase):
    """The three committed outputs, and the invariants of MGA's schema."""

    FILES = {
        "truncated": "mga-output-truncated-active-blueprint.json",
        "budget": "mga-output-max_new_tokens2048.json",
        "budget_prompt": "mga-output-max_new_tokens2048-coverage-prompt.json",
    }

    def steps(self, key):
        return mga.load_manual(os.path.join(DATA, self.FILES[key]))

    def test_expected_step_counts(self):
        self.assertEqual(len(self.steps("truncated")), 6)
        self.assertEqual(len(self.steps("budget")), 10)
        self.assertEqual(len(self.steps("budget_prompt")), 19)

    def test_step_field_is_zero_based_and_contiguous(self):
        for key in self.FILES:
            idx = [s["step"] for s in self.steps(key)]
            self.assertEqual(idx, list(range(len(idx))), key)

    def test_output_tiles_the_timeline_with_no_gaps(self):
        """MGA does segmentation, not detection: every span abuts the next."""
        for key in self.FILES:
            st = self.steps(key)
            for a, b in zip(st, st[1:]):
                self.assertAlmostEqual(a["timestamp_end"], b["timestamp_start"],
                                       places=6, msg=key)

    def test_frame_indices_are_source_frames(self):
        """frame = timestamp * fps exactly, and indices exceed max_frames."""
        for key in self.FILES:
            for s in self.steps(key):
                if s["timestamp_start"]:
                    self.assertAlmostEqual(
                        s["frame_start"] / s["timestamp_start"], 25.0, places=3, msg=key)
            self.assertGreater(max(s["frame_end"] for s in self.steps(key)), 64)

    def test_only_the_truncated_output_is_cut_mid_clause(self):
        """The token-cap signature: no terminal punctuation on the last step."""
        self.assertNotIn(self.steps("truncated")[-1]["instruction"][-1], ".!?")
        for key in ("budget", "budget_prompt"):
            self.assertIn(self.steps(key)[-1]["instruction"][-1], ".!?", key)

    def test_raising_the_budget_extends_coverage(self):
        cov = {k: max(s["timestamp_end"] for s in self.steps(k)) for k in self.FILES}
        self.assertLess(cov["truncated"], cov["budget"])
        self.assertLess(cov["budget"], cov["budget_prompt"])


class TestReferenceAnnotations(unittest.TestCase):
    REF = os.path.join(DATA, "40567_i2JWkDyg26A_reference_steps.csv")

    def test_shape_and_step_range(self):
        with open(self.REF) as f:
            rows = [l.strip().split(",") for l in f if l.strip()]
        self.assertTrue(all(len(r) == 3 for r in rows))
        steps = {int(r[0]) for r in rows}
        # 1-BASED, unlike MGA's 0-based `step` field — joining them naively is
        # off by one and fails silently.
        self.assertEqual(steps, set(range(1, 12)))

    def test_intervals_are_ordered(self):
        with open(self.REF) as f:
            for l in f:
                if l.strip():
                    _, a, b = l.strip().split(",")
                    self.assertLess(float(a), float(b))

    def test_two_annotators_disagree_on_step_1(self):
        """Documented caveat: a 17x difference on the same step."""
        with open(self.REF) as f:
            spans = [(float(a), float(b)) for s, a, b in
                     (l.strip().split(",") for l in f if l.strip())
                     if int(s) == 1]
        durations = sorted(b - a for a, b in spans)
        self.assertGreater(durations[-1] / durations[0], 10)


class TestScoringSemantics(unittest.TestCase):
    def test_advisories_are_excluded_from_matching(self):
        """Without this, tiled output makes every reference interval 'covered'."""
        self.assertIn("advisory", RUNNER_CODE)
        self.assertIn("startswith", RUNNER_CODE)

    def _score_output(self, filename, keywords=None):
        """Run the real scorer on a committed sample and capture what it prints."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mga.score(os.path.join(DATA, filename),
                      os.path.join(DATA, "40567_i2JWkDyg26A_reference_steps.csv"),
                      keywords and os.path.join(DATA, keywords))
        return buf.getvalue()

    def test_keywords_cover_every_reference_step(self):
        kw = json.load(open(os.path.join(DATA, "40567_i2JWkDyg26A_step_keywords.json")))
        self.assertEqual(sorted(map(int, kw)), list(range(1, 12)))

    def test_text_matched_score_is_reported_only_with_keywords(self):
        """Temporal IoU is blind to text; --keywords adds the text-matched score."""
        name = "mga-output-dev-old-prompt-18steps.json"
        self.assertNotIn("text-matched", self._score_output(name))
        out = self._score_output(name, "40567_i2JWkDyg26A_step_keywords.json")
        self.assertIn("text-matched recall @ IoU>=0.5: 7/11", out)

    def test_shifted_manual_scores_well_by_time_and_zero_by_text(self):
        """Production, 2026-10-27, previous prompt: every text ~5 s early. Temporal
        IoU cannot see it; matching on text can."""
        out = self._score_output("mga-output-prod-old-prompt-SHIFTED.json",
                                 "40567_i2JWkDyg26A_step_keywords.json")
        self.assertIn("recall @ IoU>=0.5: 7/11", out)
        self.assertIn("text-matched recall @ IoU>=0.5: 0/11", out)
        self.assertIn("WARNING: 10 time-only match(es)", out)

    def test_scoring_reports_recall_not_precision(self):
        """A reference procedure is not an inventory, so precision would mislead."""
        out = self._score_output("mga-output-max_new_tokens2048.json")
        self.assertIn("recall @ IoU>=", out)
        self.assertNotIn("precision", out.lower())

    def test_raising_the_budget_improves_measured_recall(self):
        """The headline claim, checked end to end through the scorer."""
        def recall(name):
            for line in self._score_output(name).splitlines():
                if "recall @ IoU>=0.1" in line:
                    return int(line.split(":")[1].split("/")[0])
            self.fail(f"no recall line for {name}")

        self.assertGreater(recall("mga-output-max_new_tokens2048.json"),
                           recall("mga-output-truncated-active-blueprint.json"))


class TestRequestShapes(unittest.TestCase):
    def test_no_bundle_or_instance_calls_remain(self):
        """Bundles and instances are being removed; everything goes through
        projects, copies, packages, deployments and sessions."""
        import re
        self.assertIsNone(re.search(r'agents_url\(f?["\'](bundles|instances)', RUNNER_SRC))
        self.assertIsNone(re.search(r'/logs[?"\']', RUNNER_SRC))
        self.assertNotIn("archetypeai import", RUNNER_SRC)
        self.assertIn('agents_url(f"blueprints/{args.blueprint}/copy")', RUNNER_SRC)

    def test_polls_session_events_not_logs(self):
        """A session has /events and no /logs."""
        self.assertIn('sessions/{session_id}/events?limit=100', RUNNER_SRC)

    def test_no_artifacts_map(self):
        """Unlike osm/red, the mga blueprint pins its own models."""
        self.assertNotIn('"artifacts"', RUNNER_SRC)

    def test_default_prompt_specifies_no_output_format(self):
        """A format-specifying prompt makes the parser return zero steps."""
        for token in ("markdown", "##", "###", "json", "format:"):
            self.assertNotIn(token, mga.DEFAULT_PROMPT.lower())

    def test_default_blueprint_is_a_key_not_a_pinned_id(self):
        """A pinned id stops resolving when the blueprint is republished: the run
        starts, then dies with `invalid config for 1 node(s)`."""
        self.assertEqual(mga.BLUEPRINT_DEFAULT, "mga")
        self.assertFalse(mga.BLUEPRINT_DEFAULT.startswith("blp_"))

    def test_default_output_budget_clears_the_reasoning_floor(self):
        """The model reasons out of this budget: 2048 and 4096 both returned
        `results: []` on a 173 s video with no ERROR row, while 16384 gave 18
        steps. Below the floor you get no manual, not a short one."""
        self.assertGreaterEqual(mga.DEFAULT_MAX_NEW_TOKENS, 16384)
        self.assertIn("default=DEFAULT_MAX_NEW_TOKENS", RUNNER_SRC)

    def test_output_is_one_json_document_not_jsonl(self):
        """MGA reports file_extension: "json" and the body is a single object.
        A line-oriented reader works by accident on one video and breaks on two."""
        for name in ("mga-output-current-16384.json",
                     "mga-output-prod-old-prompt-SHIFTED.json",
                     "mga-output-dev-old-prompt-18steps.json",
                     "mga-output-mouse-current-prompt.json",
                     "mga-output-mouse-old-prompt-LOOP.json",
                     "mga-output-current-4096-EMPTY.json",
                     "mga-output-truncated-active-blueprint.json",
                     "mga-output-max_new_tokens2048.json",
                     "mga-output-max_new_tokens2048-coverage-prompt.json"):
            raw = open(os.path.join(DATA, name), encoding="utf-8").read().strip()
            doc = json.loads(raw)                      # would raise if JSONL
            self.assertIsInstance(doc, dict)
            self.assertIn("results", doc)

    def test_loader_accepts_both_shapes(self):
        """Older files are line-delimited; do not strand anyone holding one."""
        import tempfile, os as _os
        rec = {"id": "x", "results": [{"step": 0, "instruction": "Do the thing."}]}
        for body in (json.dumps(rec), json.dumps(rec) + "\n"):
            fd, path = tempfile.mkstemp(suffix=".json")
            with _os.fdopen(fd, "w") as f:
                f.write(body)
            try:
                self.assertEqual(len(mga.load_manual(path)), 1)
            finally:
                _os.unlink(path)

    def test_current_defaults_sample_beats_the_historic_ones(self):
        """The shipped configuration should beat the no-prompt row, or the defaults
        are wrong. 16 steps against 10 (production, 2026-10-27)."""
        cur = mga.load_manual(os.path.join(DATA, "mga-output-current-16384.json"))
        old = mga.load_manual(os.path.join(DATA, "mga-output-max_new_tokens2048.json"))
        self.assertEqual(len(cur), 16)
        self.assertGreater(len(cur), len(old))

    def test_previous_prompt_looped_and_current_one_does_not(self):
        """Silent 80 s clip, production, same values: the previous prompt emitted 2
        distinct texts in 16 steps; the current one 17 distinct in 17."""
        loop = [s["instruction"] for s in mga.load_manual(
            os.path.join(DATA, "mga-output-mouse-old-prompt-LOOP.json"))]
        cur = [s["instruction"] for s in mga.load_manual(
            os.path.join(DATA, "mga-output-mouse-current-prompt.json"))]
        self.assertEqual((len(loop), len(set(loop))), (16, 2))
        self.assertEqual(len(set(cur)), len(cur))

    def test_default_prompt_avoids_what_drove_the_loop(self):
        """"concise" plus a per-step word cap drove the loop; the replacement
        requires each step to differ from the one before."""
        p = mga.DEFAULT_PROMPT.lower()
        self.assertNotIn("concise", p)
        self.assertNotIn("at most 15 words", p)
        self.assertIn("different action from the one before", p)

    def test_the_empty_sample_really_is_empty(self):
        """A run below the reasoning floor: completed, no error, no steps.
        Kept as a fixture because nothing else distinguishes it from success."""
        path = os.path.join(DATA, "mga-output-current-4096-EMPTY.json")
        self.assertEqual(mga.load_manual(path), [])
        self.assertLess(os.path.getsize(path), 100)

    def test_scoring_an_empty_manual_does_not_crash(self):
        """It should score an honest zero rather than raising on preds[-1]."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mga.score(os.path.join(DATA, "mga-output-current-4096-EMPTY.json"),
                      os.path.join(DATA, "40567_i2JWkDyg26A_reference_steps.csv"))
        self.assertIn("0/11", buf.getvalue())

    def test_uploads_are_uniquely_named(self):
        """file_id is a mutable, org-wide pointer: a plain basename lets a
        colleague's upload orphan this run's input mid-flight."""
        a = mga.upload_name("data/clip.mp4")
        b = mga.upload_name("data/clip.mp4")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("clip-") and a.endswith(".mp4"))
        self.assertNotEqual(a, "clip.mp4")

    def test_zero_step_output_is_called_out(self):
        """A run can complete with no steps and no failure signal anywhere."""
        self.assertIn("ZERO steps", RUNNER_SRC)


class FakeAPI:
    """Stands in for request(): an in-memory platform that records every call."""

    def __init__(self, deployment_status="ready"):
        self.calls, self.objs, self.n = [], {}, 0
        self.deployment_status = deployment_status

    def _new(self, prefix, **fields):
        self.n += 1
        obj = {"id": f"{prefix}_{self.n}", **fields}
        self.objs[obj["id"]] = obj
        return obj

    def __call__(self, method, url, body=None, *, raw=False, allow_status=False, timeout=120):
        path = url.split("/api", 1)[1]
        self.calls.append((method, path, body))
        code, out = self.route(method, path, body)
        if allow_status:
            return code, out
        assert code < 300, (method, path, code)
        return out

    def route(self, method, path, body):
        tail = path.rsplit("/", 1)[-1].split("?")[0]
        if method == "GET" and path.startswith("/projects"):
            return 200, {"data": [], "has_more": False}
        if method == "POST" and path == "/projects":
            return 201, self._new("prj")
        if method == "GET" and path.startswith("/agents/blueprints?"):
            return 200, {"data": []}
        if method == "POST" and path.endswith("/copy"):
            return 201, self._new("blp")
        if method == "POST" and path == "/agents/packages":
            return 201, self._new("pkg", status="building")
        if method == "POST" and path == "/agents/deployments":
            return 201, self._new("dpl", status="deploying")
        if method == "POST" and path == "/agents/sessions":
            return 201, self._new("ses", status="pending")
        if method == "POST" and path.endswith("/query"):
            return 200, {"response": {"data": [{"step": 0, "instruction": "Do it.",
                                                "frame_start": 0, "frame_end": 25,
                                                "timestamp_start": 0.0, "timestamp_end": 1.0}]}}
        if method == "POST" and tail in ("cancel", "stop"):
            obj = self.objs.get(path.split("/")[-2])
            if obj:
                obj["status"] = "cancelled" if tail == "cancel" else "stopped"
            return 200, {}
        if method == "DELETE":
            return (204, b"") if self.objs.pop(tail, None) else (404, "not found")
        if method == "GET" and "?" in path and path.startswith(("/agents/deployments", "/agents/sessions")):
            return 200, {"data": [], "has_more": False}
        if method == "GET" and tail in self.objs:
            obj = self.objs[tail]
            # Whatever was building or deploying is ready by the next look.
            status = obj.get("status")
            if status == "building":
                obj["status"] = "ready"
            elif status == "deploying":
                obj["status"] = self.deployment_status
            elif status == "pending":
                obj["status"] = "running"
            return 200, obj
        return 404, "not found"

    def posts(self):
        return [p for m, p, _ in self.calls if m == "POST"]


class TestLifecycle(unittest.TestCase):
    """Project -> copy -> package -> deployment -> session, against a fake API."""

    VALUES = {"max_frames": 64, "max_new_tokens": 16384, "prompt": mga.DEFAULT_PROMPT}

    def setUp(self):
        import tempfile
        from unittest import mock
        self._saved = os.environ.get("ATAI_API_ENDPOINT"), os.environ.get("ATAI_API_KEY")
        os.environ["ATAI_API_ENDPOINT"] = "https://platform.u1.archetypeai.app"
        os.environ["ATAI_API_KEY"] = "test-key"
        self.dir = tempfile.TemporaryDirectory()
        self.args = type("Args", (), {"blueprint": "mga", "project": "MGA manuals",
                                      "state": os.path.join(self.dir.name, "state.json"),
                                      "yes": True})()
        self.patches = [mock.patch.object(mga.time, "sleep"), mock.patch("builtins.print")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.dir.cleanup()
        for k, v in zip(("ATAI_API_ENDPOINT", "ATAI_API_KEY"), self._saved):
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def fake(self, **kw):
        from unittest import mock
        api = FakeAPI(**kw)
        p = mock.patch.object(mga, "request", api)
        p.start()
        self.addCleanup(p.stop)
        return api

    def test_setting_key_names_the_values(self):
        key = mga.setting_key(self.VALUES)
        self.assertRegex(key, r"^mga-f64-t16384-p[0-9a-f]{6}$")
        self.assertNotEqual(key, mga.setting_key({**self.VALUES, "prompt": "other"}))
        self.assertTrue(mga.setting_key(self.VALUES, "my-blueprint").startswith("mga-eprint-"))

    def test_values_travel_only_on_the_copy(self):
        """A package, a deployment and a batch session take no values (a 422)."""
        copy = mga.copy_body("prj_1", "k", self.VALUES)
        self.assertEqual(copy["overrides"], {"values": self.VALUES})
        self.assertNotIn("values", mga.package_body("blp_1", "k"))
        session = mga.session_body("dpl_1", "clip-x.mp4")
        self.assertNotIn("values", session)
        self.assertEqual(session["connectors"]["source"],
                         [{"type": "file", "id": "clip-x.mp4", "format": "mp4"}])
        self.assertEqual(mga.session_body("dpl_1"), {"deployment_id": "dpl_1"})

    def test_first_run_builds_everything_and_records_ids_before_waiting(self):
        api = self.fake()
        state = mga.load_state(self.args.state)
        key, rec = mga.ensure_deployment(self.VALUES, "batch", self.args, state)
        self.assertEqual(api.posts(), ["/projects", "/agents/blueprints/mga/copy",
                                       "/agents/packages", "/agents/deployments"])
        saved = json.load(open(self.args.state))
        self.assertEqual(saved["settings"][key]["package"], rec["package"])
        self.assertEqual(saved["settings"][key]["batch"], rec["batch"])
        copy = next(b for m, p, b in api.calls if p.endswith("/copy"))
        self.assertEqual(copy["blueprint_key"], key)

    def test_a_rerun_at_the_same_setting_builds_nothing(self):
        api = self.fake()
        mga.ensure_deployment(self.VALUES, "batch", self.args, mga.load_state(self.args.state))
        n = len(api.posts())
        mga.ensure_deployment(self.VALUES, "batch", self.args, mga.load_state(self.args.state))
        self.assertEqual(len(api.posts()), n)

    def test_state_from_another_deployment_is_refused(self):
        json.dump({"endpoint": "https://elsewhere.example"}, open(self.args.state, "w"))
        with self.assertRaises(SystemExit):
            mga.load_state(self.args.state)

    def test_cleanup_deletes_sessions_then_deployments_then_packages(self):
        api = self.fake()
        state = mga.load_state(self.args.state)
        key, rec = mga.ensure_deployment(self.VALUES, "batch", self.args, state)
        mga.open_session(rec["batch"], self.args, state, setting=key, kind="batch")
        self.assertEqual(mga.cleanup(self.args), 0)
        deleted = [p.split("/")[2] for m, p, _ in api.calls if m == "DELETE"]
        self.assertEqual(deleted, ["sessions", "deployments", "packages"])
        left = json.load(open(self.args.state))
        self.assertEqual(left["sessions"], {})
        self.assertNotIn("package", left["settings"][key])
        self.assertIn("blueprint", left["settings"][key])     # the copy stays

    def test_serving_releases_the_deployment_on_the_way_out(self):
        """The serving deployment holds the GPU worker, so it is stopped and deleted."""
        import tempfile
        api = self.fake()
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=self.dir.name, delete=False) as f:
            f.write(b"\0" * 1000)
        self.args.video = [f.name]
        self.args.prompt = mga.DEFAULT_PROMPT
        self.args.output = os.path.join(self.dir.name, "out.json")
        self.args.reference = self.args.keywords = None
        self.assertEqual(mga.run_serving(dict(self.VALUES), self.args, mga.load_state(self.args.state)), 0)
        self.assertEqual(len(mga.load_manual(self.args.output)), 1)
        posts = api.posts()
        self.assertIn("/agents/deployments", posts)
        self.assertTrue(any(p.endswith("/stop") for p in posts))
        deleted = [p.split("/")[2] for m, p, _ in api.calls if m == "DELETE"]
        self.assertEqual(deleted, ["sessions", "deployments"])
        self.assertNotIn("serving", json.load(open(self.args.state))["settings"][mga.setting_key(self.VALUES)])

    def test_serving_releases_even_when_the_deployment_fails(self):
        api = self.fake(deployment_status="failed")
        self.args.video, self.args.prompt = [], mga.DEFAULT_PROMPT
        with self.assertRaises(SystemExit):
            mga.run_serving(dict(self.VALUES), self.args, mga.load_state(self.args.state))
        self.assertIn("DELETE", [m for m, p, _ in api.calls if p.startswith("/agents/deployments/")])

    def test_query_retries_a_timeout_and_stops_on_409(self):
        from unittest import mock
        answers = iter([(504, "query_timeout"), (200, {"response": {"data": []}})])
        with mock.patch.object(mga, "request", lambda *a, **k: next(answers)):
            self.assertEqual(mga.ask("ses_1", {}), {"response": {"data": []}})
        with mock.patch.object(mga, "request", lambda *a, **k: (409, "no longer running")):
            with self.assertRaises(mga.SessionGone):
                mga.ask("ses_1", {})

    def test_poll_prints_events_oldest_first_and_returns_the_status(self):
        from unittest import mock
        events = {"data": [{"created_at": "2026-11-05T10:00:02Z", "message": "second"},
                           {"created_at": "2026-11-05T10:00:01Z", "message": "first"}]}

        def fake(method, url, body=None, **kw):
            return events if "/events" in url else {"status": "completed"}
        printed = []
        with mock.patch.object(mga, "request", fake), \
                mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            self.assertEqual(mga.poll_session("ses_1"), "completed")
        self.assertLess(printed.index(next(p for p in printed if "first" in p)),
                        printed.index(next(p for p in printed if "second" in p)))


class TestDryRun(unittest.TestCase):
    """--dry-run prints every request and needs no key."""

    def run_main(self, *argv):
        from unittest import mock
        buf = io.StringIO()
        env = {k: v for k, v in os.environ.items() if k not in ("ATAI_API_KEY", "ATAI_API_ENDPOINT")}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(mga.sys, "argv", ["run_mga_agent.py", *argv]), \
                mock.patch.object(mga, "request", side_effect=AssertionError("no calls")), \
                contextlib.redirect_stdout(buf):
            mga.main()
        return buf.getvalue()

    def test_batch_plan(self):
        out = self.run_main("--dry-run", "--video", "clip.mp4")
        self.assertIn("POST {ATAI_API_ENDPOINT}/api/agents/blueprints/mga/copy", out)
        self.assertIn('"kind": "batch"', out)
        self.assertIn('"connectors"', out)
        self.assertEqual(out.count('"values"'), 1)            # on the copy, nowhere else
        self.assertNotIn("bundles", out)

    def test_serving_plan(self):
        out = self.run_main("--dry-run", "--serve", "--video", "clip.mp4")
        self.assertIn('"kind": "serving"', out)
        self.assertIn("/sessions/<id>/query", out)
        self.assertIn("/deployments/<id>/stop", out)


class TestWaitsForCompleteOutput(unittest.TestCase):
    """The platform can report completed while the output is still being written."""

    def test_downloads_again_until_the_json_parses(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.json")
            open(out, "w").write('{"steps": [{"step": 1')    # cut off mid-write
            n = {"saves": 0}

            def save():
                n["saves"] += 1
                open(out, "w").write('{"steps": [{"step": 1}]}' if n["saves"] >= 2 else '{"steps": [')

            with mock.patch.object(mga.time, "sleep"), mock.patch("builtins.print"):
                self.assertTrue(mga.wait_until_parses(save, out))
            self.assertEqual(n["saves"], 2)


if __name__ == "__main__":
    unittest.main()
