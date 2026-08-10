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
    """The Agent API is mounted without a version prefix; files live under /v0.5."""

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

    ACTIVE = {"document": {
        "values": {"max_frames": 16, "size": 224},
        "nodes": {"video_reader": {"config": {"max_frames": "${values.max_frames}",
                                              "size": "${values.size}"}},
                  "fusion": {"config": {"model": "${models.fusion}"}}},
        "connectors": {"source": {"config": {
            "default_text": "Generate a concise, ordered list of distinct steps "
                            "with 10 steps or less.", "text_extensions": []}}}}}

    WITH_PARAMS = {"document": {
        "values": {"max_frames": 16, "max_new_tokens": 256, "prompt": "…"},
        "nodes": {"video_reader": {"config": {"max_frames": "${values.max_frames}"}},
                  "fusion": {"config": {"model": "${models.fusion}",
                                        "max_new_tokens": "${values.max_new_tokens}"}}},
        "connectors": {"source": {"config": {"default_text": "${values.prompt}"}}}}}

    def test_active_blueprint_ignores_generation_values(self):
        ignored = mga.inert_values(
            self.ACTIVE, {"max_frames": 64, "max_new_tokens": 2048, "prompt": "x"})
        self.assertEqual(sorted(ignored), ["max_new_tokens", "prompt"])

    def test_active_blueprint_honours_max_frames(self):
        self.assertEqual(mga.inert_values(self.ACTIVE, {"max_frames": 64}), [])

    def test_superseded_blueprint_honours_all_three(self):
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
        "truncated": "mga-output-truncated-active-blueprint.jsonl",
        "budget": "mga-output-max_new_tokens2048.jsonl",
        "budget_prompt": "mga-output-max_new_tokens2048-coverage-prompt.jsonl",
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
        out = self._score_output("mga-output-max_new_tokens2048.jsonl")
        self.assertIn("recall @ IoU>=", out)
        self.assertNotIn("precision", out.lower())

    def test_raising_the_budget_improves_measured_recall(self):
        """The headline claim, checked end to end through the scorer."""
        def recall(name):
            for line in self._score_output(name).splitlines():
                if "recall @ IoU>=0.1" in line:
                    return int(line.split(":")[1].split("/")[0])
            self.fail(f"no recall line for {name}")

        self.assertGreater(recall("mga-output-max_new_tokens2048.jsonl"),
                           recall("mga-output-truncated-active-blueprint.jsonl"))


class TestRequestShapes(unittest.TestCase):
    def test_bundle_endpoint_is_plural(self):
        self.assertIn("/agents/bundles", RUNNER_SRC)
        self.assertNotIn('"/agents/bundle"', RUNNER_SRC)

    def test_run_accepts_202(self):
        # A code literal, so the raw source is the unambiguous place to look.
        self.assertIn("(200, 201, 202)", RUNNER_SRC)

    def test_polls_logs_not_events(self):
        """/events carries only two coarse rows; errors appear only in /logs."""
        self.assertIn("/logs?limit=", RUNNER_SRC)
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


if __name__ == "__main__":
    unittest.main()
