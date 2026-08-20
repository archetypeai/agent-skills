"""Network-free tests for the TVA agent skill references.

    python -m unittest discover -s skills/atai-task-verification-agent/tests

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
    "run_tva_agent", os.path.join(REFS, "run_tva_agent.py"))
tva = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tva)

with open(os.path.join(REFS, "run_tva_agent.py")) as _f:
    RUNNER_SRC = _f.read()

PASS_OUT = os.path.join(DATA, "tva-output-1_pass_2_pass_3_pass_A.json")
EMPTY_OUT = os.path.join(DATA, "tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.json")
SOP = os.path.join(DATA, "oring-numbered.txt")


def code_only(src: str) -> str:
    """Source with comments and string literals removed.

    The runner's docstrings deliberately discuss what it must NOT do ("poll
    /logs, NOT /events"), so a naive grep reports a violation for the warning
    against it. These tests care about executable code.
    """
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


RUNNER_CODE = code_only(RUNNER_SRC)


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


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
        self.assertEqual(tva.env()[1], "https://api.dev.u1.archetypeai.app")

    def test_version_suffix_stripped(self):
        # Someone pasting the /query base URL should still work; otherwise every
        # agent path becomes /v0.5/agents/... and 404s.
        for suffix in ("/v0.5", "/v0.4", "/v1"):
            os.environ["ATAI_API_ENDPOINT"] = \
                "https://api.dev.u1.archetypeai.app" + suffix
            self.assertEqual(tva.env()[1], "https://api.dev.u1.archetypeai.app")

    def test_trailing_slash_stripped(self):
        os.environ["ATAI_API_ENDPOINT"] = "https://api.dev.u1.archetypeai.app/"
        self.assertEqual(tva.env()[1], "https://api.dev.u1.archetypeai.app")

    def test_missing_key_exits(self):
        os.environ.pop("ATAI_API_KEY")
        os.environ["ATAI_API_ENDPOINT"] = "https://api.dev.u1.archetypeai.app"
        with self.assertRaises(SystemExit):
            tva.env()

    def test_prod_endpoint_warns_and_staging_does_not(self):
        # The /agents API is on Dev and Staging (both verified); Prod 404s on
        # every agent path, which reads like a missing blueprint rather than
        # the wrong host. The note prints ONCE per endpoint — env() runs on
        # every api() call, and a per-call note buried the poll output.
        tva._ENDPOINT_NOTED.clear()
        os.environ["ATAI_API_ENDPOINT"] = "https://api.u1.archetypeai.app"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tva.env()
            tva.env()
        self.assertEqual(buf.getvalue().count("NOTE:"), 1)
        self.assertIn("Prod returns 404", buf.getvalue())

        os.environ["ATAI_API_ENDPOINT"] = "https://api.stage.u1.archetypeai.app"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tva.env()
        self.assertEqual(buf.getvalue(), "")


class TestSinkPreflight(unittest.TestCase):
    """Both observed sink formats are good; only an UNKNOWN one warns.

    This suite previously asserted that `json/per-request` is broken, which was true
    for ~18 hours and then actively harmful: the denylist refused every run against
    the fixed canonical blueprint. The contract now under test is "warn, never
    refuse, and do not hard-code which format is broken".
    """

    @staticmethod
    def doc(fmt):
        return {"connectors": {"sink": {"key": "RecordsSink",
                                       "config": {"format": fmt}}}}

    def test_both_observed_formats_pass(self):
        self.assertIsNone(tva.check_sink(self.doc("jsonl/per-request")))
        self.assertIsNone(tva.check_sink(self.doc("json/per-request")))

    def test_no_format_is_hardcoded_as_broken(self):
        """A denylist is what went stale. Keep it empty."""
        self.assertEqual(tva.KNOWN_BROKEN_SINK_FORMATS, set())

    def test_unknown_format_still_warns(self):
        self.assertIsNotNone(tva.check_sink(self.doc("parquet/per-request")))

    def test_missing_sink_config_does_not_crash(self):
        self.assertIsNotNone(tva.check_sink({"connectors": {}}))
        self.assertIsNotNone(tva.check_sink({}))


class TestWiredValuePreflight(unittest.TestCase):
    """A value is honoured only if declared AND referenced as ${values.<key>}."""

    DOC = {
        "values": {"max_frames": 16, "max_new_tokens": 8192, "size": 224,
                   "reader_batch_size": 64},
        "nodes": {
            "fusion": {"config": {"max_new_tokens": "${values.max_new_tokens}"}},
            "video_reader": {"config": {"max_frames": "${values.max_frames}",
                                        "size": "${values.size}"}},
        },
        "connectors": {"source": {"config": {}}},
    }

    def test_wired_values_are_not_inert(self):
        self.assertEqual(
            tva.inert_values(self.DOC, {"max_frames": 64, "max_new_tokens": 8192}),
            [])

    def test_prompt_does_not_exist_on_tva(self):
        # Unlike mga, tva has no prompt value at all — the SOP replaces it.
        self.assertEqual(tva.inert_values(self.DOC, {"prompt": "..."}), ["prompt"])

    def test_declared_but_unreferenced_counts_as_inert(self):
        # reader_batch_size is in `values` but no node references it here.
        self.assertEqual(
            tva.inert_values(self.DOC, {"reader_batch_size": 32}),
            ["reader_batch_size"])

    def test_temporal_compression_control_is_not_exposed(self):
        # The design doc's headline algorithm; the fusion node wires only
        # model + max_new_tokens.
        self.assertEqual(
            tva.inert_values(self.DOC, {"min_temporal_similarity_threshold": 0.9}),
            ["min_temporal_similarity_threshold"])


class TestSOPParsing(unittest.TestCase):
    """PrepareSOPNode accumulates LINES: the line breaks are the step boundaries."""

    def test_sample_sop_is_three_steps(self):
        self.assertEqual(len(tva.load_sop(SOP)), 3)

    def test_blank_lines_and_comments_dropped(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("# a comment\n\nStep 1: do a thing\n\n  \nStep 2: do another\n")
            path = fh.name
        try:
            self.assertEqual(tva.load_sop(path),
                             ["Step 1: do a thing", "Step 2: do another"])
        finally:
            os.unlink(path)

    def test_empty_sop_exits(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("# only comments\n\n")
            path = fh.name
        try:
            with self.assertRaises(SystemExit):
                tva.load_sop(path)
        finally:
            os.unlink(path)

    def test_missing_sop_exits(self):
        with self.assertRaises(SystemExit):
            tva.load_sop(os.path.join(DATA, "does-not-exist.txt"))


class TestOutputSchema(unittest.TestCase):
    """Properties of the real committed output that consumers depend on."""

    def setUp(self):
        self.rows = load(PASS_OUT)[0]["results"]

    def test_three_verdicts(self):
        self.assertEqual(len(self.rows), 3)

    def test_step_field_is_zero_based(self):
        # Undocumented, and an off-by-one fails quietly rather than erroring.
        self.assertEqual(sorted(r["step"] for r in self.rows), [0, 1, 2])

    def test_step_number_converts_to_one_based(self):
        self.assertEqual(sorted(tva.step_number(r) for r in self.rows), [1, 2, 3])

    def test_statuses_are_from_the_documented_set(self):
        for r in self.rows:
            self.assertIn(r["status"], tva.STATUSES)

    def test_frame_indices_are_source_frames(self):
        # frame = timestamp * source_fps (30 here). Sampled space would cap at 63
        # at max_frames=64.
        for r in self.rows:
            self.assertAlmostEqual(r["frame_start"], r["timestamp_start"] * 30,
                                   delta=1.0)
            self.assertAlmostEqual(r["frame_end"], r["timestamp_end"] * 30,
                                   delta=1.0)
        self.assertGreater(max(r["frame_end"] for r in self.rows), 63)

    def test_every_row_carries_a_reason(self):
        for r in self.rows:
            self.assertTrue(r.get("reason", "").strip())

    def test_reason_cites_the_video_not_the_sop(self):
        # The SOP used for this run did not mention colour; the model did. That is
        # evidence the vision path is doing the work.
        joined = " ".join(r["reason"] for r in self.rows).lower()
        self.assertIn("blue", joined)
        with open(SOP) as fh:
            self.assertNotIn("blue", fh.read().lower().split("step 1")[0])

    def test_intervals_are_ordered_and_non_empty(self):
        for r in self.rows:
            self.assertLess(r["timestamp_start"], r["timestamp_end"])

    def test_output_does_not_tile_the_timeline_contiguously(self):
        # Unlike MGA: TVA emits one row per SOP step and may leave gaps. It is
        # verification, not segmentation. Here they happen to abut, so the
        # assertion is about the SHAPE: one row per step, not one per span.
        self.assertEqual(len(self.rows), len(tva.load_sop(SOP)))


class TestEmptyResultIsRecognised(unittest.TestCase):
    """job.completed + results:[] is the failure that correlates with defects."""

    def test_empty_output_parses_but_has_no_verdicts(self):
        recs = load(EMPTY_OUT)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["results"], [])

    def test_empty_output_is_tiny(self):
        self.assertLess(os.path.getsize(EMPTY_OUT), 100)

    def test_show_reports_zero_verdicts(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n = tva.show(EMPTY_OUT)
        self.assertEqual(n, 0)
        out = buf.getvalue()
        self.assertIn("NO VERDICTS", out)
        # It must name the cause and the fix, not just the absence.
        self.assertIn("reasoning block", out)
        self.assertIn("max-new-tokens", out)

    def test_show_reports_the_real_verdict_count(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n = tva.show(PASS_OUT)
        self.assertEqual(n, 3)

    def test_empty_output_scores_zero_not_partial_credit(self):
        # The trap: NOT REPORTED must never count as a correct FAIL, or an empty
        # output scores 100% on every clip where a step was skipped.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tva.show(EMPTY_OUT)
        self.assertIn("0/3 correct", buf.getvalue())


class TestLabelParsing(unittest.TestCase):
    def test_labels_are_one_based(self):
        self.assertEqual(tva.parse_labels("1_pass_2_pass_3_fail_A"),
                         {1: True, 2: True, 3: False})

    def test_labels_read_from_a_filename_with_extension(self):
        self.assertEqual(tva.parse_labels("1_fail_2_pass_3_pass_C.mp4"),
                         {1: False, 2: True, 3: True})

    def test_unlabelled_name_yields_nothing(self):
        self.assertEqual(tva.parse_labels("assembly_take_7.mp4"), {})

    def test_all_pass_clip_is_flagged_as_non_discriminating(self):
        # Three PASSED verdicts on an all-pass clip is what a model that always
        # says PASSED would produce, so the scorer must say so.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tva.show(PASS_OUT)
        self.assertIn("3/3 correct", buf.getvalue())
        self.assertIn("always says", buf.getvalue())


class TestRequestShapes(unittest.TestCase):
    """Shapes that have each cost someone real time."""

    def test_bundle_endpoint_is_plural(self):
        self.assertIn("/agents/bundles", RUNNER_CODE)

    def test_run_accepts_202(self):
        self.assertIn("202", RUNNER_CODE)

    def test_polls_logs_not_events(self):
        self.assertIn("/logs", RUNNER_CODE)
        self.assertNotIn("/events", RUNNER_CODE)

    def test_no_artifacts_map(self):
        # The tva blueprint pins its own models; unlike red/osm there is nothing
        # to fit and no artifact to pin.
        self.assertNotIn("artifacts", RUNNER_CODE)

    def test_run_sends_both_video_and_sop_inputs(self):
        """The whole point of TVA: two source inputs, routed by `format`.

        Asserted against the --dry-run payload rather than the source text,
        because code_only() strips plain string literals (f-strings survive
        tokenization, bare ones do not) — so grepping for "mp4" was testing the
        test helper, not the runner.
        """
        payload = self.dry_run_payload()
        source = payload["connectors"]["source"]
        self.assertEqual(len(source), 2)
        self.assertEqual({e["format"] for e in source}, {"mp4", "txt"})
        # Every entry is a file reference by basename, not a fil_ uid.
        for entry in source:
            self.assertEqual(entry["type"], "file")
            self.assertFalse(entry["id"].startswith("fil_"))

    def test_run_payload_has_no_artifacts_key(self):
        self.assertNotIn("artifacts", self.dry_run_payload())

    @staticmethod
    def dry_run_payload():
        """The run payload the runner would POST, captured from --dry-run."""
        import sys as _sys
        argv = _sys.argv
        env_saved = {k: os.environ.get(k) for k in
                     ("ATAI_API_KEY", "ATAI_API_ENDPOINT")}
        os.environ["ATAI_API_KEY"] = "test-key"
        os.environ["ATAI_API_ENDPOINT"] = "https://api.dev.u1.archetypeai.app"
        _sys.argv = ["run_tva_agent.py", "--video", "clip.mp4", "--sop", SOP,
                     "--dry-run"]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                tva.main()
        finally:
            _sys.argv = argv
            for k, v in env_saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        # The second JSON object in the output is the run payload.
        blocks = [b for b in buf.getvalue().split("\n{") if '"connectors"' in b]
        return json.loads("{" + blocks[0].split("\n\nSOP")[0])

    def test_default_budget_is_above_the_empty_result_value(self):
        # 2048 produced results:[] on a clip with a skipped step.
        self.assertGreaterEqual(tva.DEFAULT_MAX_NEW_TOKENS, 4096)

    def test_terminality_comes_from_log_events(self):
        for ev in ("pod.terminated", "job.completed", "job.failed", "job.canceled"):
            self.assertIn(ev, tva.TERMINAL_EVENTS)

    def test_runner_is_stdlib_only(self):
        third_party = ("requests", "numpy", "pandas", "httpx", "aiohttp",
                       "sklearn", "torch")
        for mod in third_party:
            self.assertNotIn(f"import {mod}", RUNNER_SRC)


if __name__ == "__main__":
    unittest.main()
