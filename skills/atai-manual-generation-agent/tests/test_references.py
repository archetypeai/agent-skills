"""Network-free tests for the MGA agent skill references.

    python -m unittest discover -s skills/atai-manual-generation-agent/tests

No credentials, no network, no GPU. Everything here checks either request shapes
built in memory, or the committed sample outputs.
"""
import contextlib
import importlib.util
import io
import json
import os
import re
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
    """The Agents API is mounted without a version prefix; files live under /v0.5."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("ATAI_API_KEY", "ATAI_API_ENDPOINT")}
        os.environ["ATAI_API_KEY"] = "test-key"

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_bare_endpoint_kept(self):
        os.environ["ATAI_API_ENDPOINT"] = "https://api.dev.u1.archetypeai.app"
        self.assertEqual(mga.env()[1], "https://api.dev.u1.archetypeai.app")

    def test_version_suffix_stripped(self):
        # A user pasting the /query base URL should still work: the runner mounts
        # /agents and /v0.5/files itself.
        for suffix in ("/v0.5", "/v0.4", "/v1"):
            os.environ["ATAI_API_ENDPOINT"] = f"https://api.dev.u1.archetypeai.app{suffix}"
            self.assertEqual(mga.env()[1], "https://api.dev.u1.archetypeai.app")

    def test_trailing_slash_stripped(self):
        os.environ["ATAI_API_ENDPOINT"] = "https://api.dev.u1.archetypeai.app/"
        self.assertEqual(mga.env()[1], "https://api.dev.u1.archetypeai.app")


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
        """PLDEV-1730 landed 2026-08-12: prompt is wired as ${values.prompt}."""
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

    def _score_output(self, filename):
        """Run the real scorer on a committed sample and capture what it prints."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mga.score(os.path.join(DATA, filename),
                      os.path.join(DATA, "40567_i2JWkDyg26A_reference_steps.csv"))
        return buf.getvalue()

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
    def test_bundle_endpoint_is_plural(self):
        self.assertIn("/agents/bundles", RUNNER_SRC)
        self.assertNotIn('"/agents/bundle"', RUNNER_SRC)

    def test_run_accepts_202(self):
        # A code literal, so the raw source is the unambiguous place to look.
        # the client owns status handling now; 202 is accepted by it
        self.assertIn("client().agents.bundles.run(", RUNNER_SRC)

    def test_polls_logs_not_events(self):
        """/events carries only two coarse rows; errors appear only in /logs."""
        self.assertIn("get_logs(agent_id, limit=500)", RUNNER_SRC)
        # Any request URL ending in /events would appear as a quoted fragment.
        for frag in ('/events"', "/events'", "/events?"):
            self.assertNotIn(frag, RUNNER_SRC)

    def test_no_artifacts_map(self):
        """Unlike osm/red, the mga blueprint pins its own models."""
        self.assertNotIn('"artifacts"', RUNNER_SRC)

    def test_default_prompt_specifies_no_output_format(self):
        """A format-specifying prompt makes the parser return zero steps."""
        for token in ("markdown", "##", "###", "json", "format:"):
            self.assertNotIn(token, mga.DEFAULT_PROMPT.lower())

    def test_default_blueprint_is_a_key_not_a_pinned_id(self):
        """A pinned id stops resolving when the blueprint is republished: the run
        returns 202, then the pod dies with `invalid config for 1 node(s)`."""
        self.assertEqual(mga.BLUEPRINT_DEFAULT, "mga")
        self.assertFalse(mga.BLUEPRINT_DEFAULT.startswith("blp_"))

    def test_default_output_budget_clears_the_reasoning_floor(self):
        """The model reasons out of this budget: 2048 and 4096 both returned
        `results: []` on a 173 s video with no ERROR row, while 16384 gave 18
        steps. Below the floor you get no manual, not a short one."""
        default = re.search(r'"--max-new-tokens", type=int, default=(\d+)', RUNNER_SRC)
        self.assertIsNotNone(default)
        self.assertGreaterEqual(int(default.group(1)), 16384)

    def test_output_is_one_json_document_not_jsonl(self):
        """MGA reports file_extension: "json" and the body is a single object.
        A line-oriented reader works by accident on one video and breaks on two."""
        for name in ("mga-output-current-16384.json",
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
        """The shipped configuration should be the best row, or the defaults are
        wrong. 18 steps against 10, and IoU>=0.5 of 7/11 against 4/11."""
        cur = mga.load_manual(os.path.join(DATA, "mga-output-current-16384.json"))
        old = mga.load_manual(os.path.join(DATA, "mga-output-max_new_tokens2048.json"))
        self.assertEqual(len(cur), 18)
        self.assertGreater(len(cur), len(old))

    def test_the_empty_sample_really_is_empty(self):
        """A run below the reasoning floor: job.completed, no error, no steps.
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


if __name__ == "__main__":
    unittest.main()
