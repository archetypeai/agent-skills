"""Network-free tests for the TVA agent skill references.

    python -m unittest discover -s skills/atai-task-verification-agent/tests

No credentials, no network, no GPU. Everything here checks request shapes built in
memory, the whole lifecycle against an in-memory fake platform, or the committed
sample outputs.
"""
import contextlib
import importlib.util
import io
import json
import os
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


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


class TestEndpointResolution(unittest.TestCase):
    """The endpoint reduces to a bare deployment root; the runner appends /api."""

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
        os.environ["ATAI_API_ENDPOINT"] = "https://platform.example.com"
        self.assertEqual(tva.env()[1], "https://platform.example.com")

    def test_version_suffix_stripped(self):
        # Someone pasting the /query base URL should still work; otherwise every
        # agent path becomes /v0.5/agents/... and 404s.
        for suffix in ("/v0.5", "/v0.4", "/v1"):
            os.environ["ATAI_API_ENDPOINT"] = \
                "https://platform.example.com" + suffix
            self.assertEqual(tva.env()[1], "https://platform.example.com")

    def test_trailing_slash_stripped(self):
        os.environ["ATAI_API_ENDPOINT"] = "https://platform.example.com/"
        self.assertEqual(tva.env()[1], "https://platform.example.com")

    def test_missing_key_exits(self):
        os.environ.pop("ATAI_API_KEY")
        os.environ["ATAI_API_ENDPOINT"] = "https://platform.example.com"
        with self.assertRaises(SystemExit):
            tva.env()

    def test_env_does_not_editorialise_about_the_endpoint(self):
        # This used to print "Prod returns 404 for every /agents path" for any
        # endpoint that was not Dev or Staging. That is false — the production
        # deployment serves /agents, and the `tva` blueprint is active there. env()
        # resolves the endpoint and says nothing about which one it is.
        for endpoint in ("https://platform.u1.archetypeai.app",
                         "https://platform.u1.archetypeai.app/v0.5",
                         "https://platform.u1.archetypeai.app/api"):
            os.environ["ATAI_API_ENDPOINT"] = endpoint
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                key, resolved = tva.env()
            self.assertEqual(buf.getvalue(), "")
            self.assertEqual(resolved, "https://platform.u1.archetypeai.app")


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
    """completed + results:[] is the failure that correlates with defects."""

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


class TestDryRun(unittest.TestCase):
    """The request bodies, captured from --dry-run: no API call and no key."""

    @staticmethod
    def blocks(*extra):
        import sys as _sys
        argv = _sys.argv
        env_saved = {k: os.environ.pop(k, None) for k in ("ATAI_API_KEY", "ATAI_API_ENDPOINT")}
        _sys.argv = ["run_tva_agent.py", "--video", "clip.mp4", "--sop", SOP, "--dry-run", *extra]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                tva.main()
        finally:
            _sys.argv = argv
            for k, v in env_saved.items():
                if v is not None:
                    os.environ[k] = v
        out = buf.getvalue()
        bodies, i = {}, 0
        dec = json.JSONDecoder()
        while (i := out.find("POST ", i)) >= 0:
            path = out[i + 5:out.index("\n", i)].split()[0]
            j = out.index("\n{", i) + 1
            bodies[path.split("/api/", 1)[1]], end = dec.raw_decode(out, j)
            i = end
        return bodies

    def test_needs_no_key(self):
        self.assertIn("agents/sessions", self.blocks())

    def test_values_go_on_the_copy(self):
        copy = self.blocks()["agents/blueprints/tva/copy"]
        self.assertEqual(copy["overrides"]["values"],
                         {"max_frames": 64, "size": 224, "max_new_tokens": 5760})
        self.assertEqual(copy["blueprint_key"], "tva-f64-s224-t5760")

    def test_package_deployment_and_session_carry_no_values(self):
        b = self.blocks()
        for path in ("agents/packages", "agents/deployments", "agents/sessions"):
            self.assertNotIn("values", b[path])
        self.assertEqual(b["agents/packages"]["target"], {"kind": "platform"})
        self.assertEqual(b["agents/deployments"]["kind"], "batch")

    def test_session_sends_both_video_and_sop_inputs(self):
        source = self.blocks()["agents/sessions"]["connectors"]["source"]
        self.assertEqual(len(source), 2)
        self.assertEqual({e["format"] for e in source}, {"mp4", "txt"})
        for entry in source:
            # a file reference by basename, not a fil_ uid
            self.assertEqual(entry["type"], "file")
            self.assertFalse(entry["id"].startswith("fil_"))
        # both halves share one stem, so the pipeline pairs them
        self.assertEqual(*(os.path.splitext(e["id"])[0] for e in source))

    def test_no_artifacts_anywhere(self):
        # The tva blueprint pins its own models: nothing to fit, no artifact to pin.
        for body in self.blocks().values():
            self.assertNotIn("artifacts", json.dumps(body))

    def test_serving_session_has_no_connectors_and_the_query_carries_the_sop(self):
        b = self.blocks("--serve")
        self.assertEqual(b["agents/deployments"]["kind"], "serving")
        self.assertEqual(b["agents/sessions"], {"deployment_id": "<the deployment>"})
        query = b["agents/sessions/<id>/query"]
        self.assertEqual(query["model"], "tv")
        self.assertEqual(query["query"], tva.sop_text(tva.load_sop(SOP)))


class FakePlatform:
    """Answers the lifecycle calls in memory, and records them."""

    PREFIX = {"packages": "pkg", "deployments": "dpl", "sessions": "ses"}

    def __init__(self, results=None, session_status="completed"):
        self.calls, self.objs, self.n = [], {}, 0
        self.results = results
        self.session_status = session_status

    def new(self, prefix, **fields):
        self.n += 1
        obj = {"id": f"{prefix}_{self.n}", **fields}
        self.objs[obj["id"]] = obj
        return obj

    def request(self, method, url, body=None, raw=False, allow_status=False, timeout=120, retries=4):
        path, _, query = url.split("/api", 1)[1].partition("?")
        self.calls.append((method, path, body))
        out, code = self.route(method, path, body, query)
        return (code, out) if allow_status else out

    def route(self, method, path, body, query=""):
        parts = path.strip("/").split("/")
        if path == "/projects":
            return ({"data": []} if method == "GET" else self.new("prj")), 200
        if path.startswith("/agents/blueprints") and path.endswith("/copy"):
            return self.new("blp"), 201
        if path == "/agents/blueprints":
            return {"data": []}, 200
        if path == "/agents/blueprints/tva":
            return {"id": "blp_tva", "blueprint_key": "tva", "is_active": True,
                    "document": {"values": {"max_frames": 16, "size": 224, "max_new_tokens": 8192},
                                 "nodes": {"n": "${values.max_frames} ${values.size} "
                                                "${values.max_new_tokens}"},
                                 "connectors": {"sink": {"config": {"format": "json/per-request"}}}}}, 200
        kind = parts[1]
        if len(parts) == 2:
            if method == "GET":
                return {"data": [o for o in self.objs.values() if o["id"].startswith(self.PREFIX[kind])
                                 and o.get("parent") and o["parent"] in query]}, 200
            if kind == "packages":
                return self.new("pkg", status="building"), 201
            if kind == "deployments":
                return self.new("dpl", status="deploying", parent=body["package_id"]), 201
            if kind == "sessions":
                return self.new("ses", status="pending", parent=body["deployment_id"]), 201
        obj = self.objs.get(parts[2])
        if obj is None:
            return "not found", 404
        if method == "DELETE":
            del self.objs[obj["id"]]
            return b"", 204
        if len(parts) == 4:
            sub = parts[3]
            if sub == "events":
                return {"data": [{"created_at": "2026-11-06T12:00:00Z", "message": "opened"}]}, 200
            if sub == "results":
                return {"data": self.results or []}, 200
            if sub == "query":
                return {"response": {"data": [{"step": 0, "status": "PASSED"}]}}, 200
            obj["status"] = {"cancel": "cancelled", "stop": "stopped"}[sub]
            return obj, 200
        obj["status"] = {"pkg": "ready", "dpl": "ready",
                         "ses": self.session_status}.get(obj["id"][:3])
        if obj["id"].startswith("ses") and obj.get("query"):
            obj["status"] = "running"
        return obj, 200

    def posts(self):
        return [p for m, p, _ in self.calls if m == "POST"]


class TestLifecycleAgainstAFakePlatform(unittest.TestCase):
    """The whole batch path and the serving path, offline."""

    OUTPUT = b'{"id": "clip", "results": [{"step": 0, "status": "PASSED"}]}\n'

    def setUp(self):
        import tempfile
        from unittest import mock
        self.dir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.dir.name)
        self.video = os.path.join(self.dir.name, "1_pass_2_pass_3_pass_A.mp4")
        open(self.video, "wb").write(b"\0" * 1000)
        self.saved = {k: os.environ.get(k) for k in ("ATAI_API_KEY", "ATAI_API_ENDPOINT")}
        os.environ["ATAI_API_KEY"] = "test-key"
        os.environ["ATAI_API_ENDPOINT"] = "https://platform.example.com/api"
        self.patches = [mock.patch.object(tva.time, "sleep"),
                        mock.patch.object(tva, "upload", side_effect=lambda path, name: name),
                        mock.patch.object(tva.urllib.request, "urlopen", side_effect=self.download)]
        for p in self.patches:
            p.start()
        self.downloaded = []

    def tearDown(self):
        for p in self.patches:
            p.stop()
        os.chdir(self.cwd)
        self.dir.cleanup()
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def download(self, req, timeout=None):
        self.downloaded.append(req.full_url)
        return _Resp(self.OUTPUT)

    def run_main(self, fake, *argv):
        import sys as _sys
        from unittest import mock
        saved = _sys.argv
        _sys.argv = ["run_tva_agent.py", *argv]
        buf = io.StringIO()
        try:
            with mock.patch.object(tva, "request", side_effect=fake.request), \
                    contextlib.redirect_stdout(buf):
                tva.main()
        finally:
            _sys.argv = saved
        return buf.getvalue()

    RESULT = [{"data": {"ref": "/api/files/fil_1/content", "filename": "out.json"}}]

    def test_batch_builds_once_then_reuses(self):
        fake = FakePlatform(results=self.RESULT)
        out = self.run_main(fake, "--video", self.video, "--sop", SOP)
        self.assertEqual(fake.posts(), ["/projects", "/agents/blueprints/tva/copy",
                                        "/agents/packages", "/agents/deployments", "/agents/sessions"])
        self.assertIn("1 verdict(s)", out)
        # the ref is a root path: resolved against the root, not against .../api
        self.assertEqual(self.downloaded, ["https://platform.example.com/api/files/fil_1/content"])
        # a second run at the same setting builds nothing
        fake.calls.clear()
        self.run_main(fake, "--video", self.video, "--sop", SOP)
        self.assertEqual(fake.posts(), ["/agents/sessions"])
        state = json.load(open(tva.STATE_PATH))["https://platform.example.com"]
        self.assertEqual(len(state["sessions"]), 2)
        self.assertEqual(set(state["settings"]["tva-f64-s224-t5760"]),
                         {"source", "values", "blueprint", "package", "batch"})

    def test_empty_results_exit_nonzero_though_the_session_completed(self):
        type(self).OUTPUT, saved = b'{"id": "clip", "results": []}\n', type(self).OUTPUT
        try:
            with self.assertRaises(SystemExit) as cm:
                self.run_main(FakePlatform(results=self.RESULT), "--video", self.video, "--sop", SOP)
        finally:
            type(self).OUTPUT = saved
        self.assertIn("no verdicts", str(cm.exception))

    def test_failed_session_exits_nonzero(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_main(FakePlatform(session_status="failed"), "--video", self.video, "--sop", SOP)
        self.assertIn("failed", str(cm.exception))

    def test_cleanup_deletes_sessions_then_deployments_then_packages(self):
        fake = FakePlatform(results=self.RESULT)
        self.run_main(fake, "--video", self.video, "--sop", SOP)
        fake.calls.clear()
        with self.assertRaises(SystemExit):
            self.run_main(fake, "--cleanup")
        self.assertFalse([c for c in fake.calls if c[0] == "DELETE"])     # lists only
        fake.calls.clear()
        with self.assertRaises(SystemExit) as cm:
            self.run_main(fake, "--cleanup", "--yes")
        self.assertEqual(cm.exception.code, 0)
        deleted = [p.split("/")[2] for m, p, _ in fake.calls if m == "DELETE"]
        self.assertEqual(deleted, ["sessions", "deployments", "packages"])
        # the project and the copy stay
        self.assertTrue(any(k.startswith("prj") for k in fake.objs))
        self.assertTrue(any(k.startswith("blp") for k in fake.objs))

    def test_serving_releases_the_deployment_on_exit(self):
        fake = FakePlatform()
        orig = fake.new

        def new(prefix, **fields):
            obj = orig(prefix, **fields)
            if prefix == "ses":
                obj["query"] = True                  # a serving session reaches `running`
            return obj
        fake.new = new
        out = self.run_main(fake, "--serve", "--video", self.video, "--sop", SOP)
        self.assertIn("/agents/sessions/ses_5/query", fake.posts())
        session = [b for m, p, b in fake.calls if m == "POST" and p == "/agents/sessions"][0]
        self.assertEqual(session, {"deployment_id": "dpl_4"})
        self.assertIn(("POST", "/agents/deployments/dpl_4/stop", {}), fake.calls)
        self.assertEqual([p for m, p, _ in fake.calls if m == "DELETE"],
                         ["/agents/sessions/ses_5", "/agents/deployments/dpl_4"])
        self.assertIn("released", out)
        doc = load("tva-output-1_pass_2_pass_3_pass_A.json")[0]
        self.assertEqual(doc["results"], [{"step": 0, "status": "PASSED"}])


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestHelpers(unittest.TestCase):
    def test_result_ref_is_a_root_path(self):
        root = "https://platform.example.com"
        self.assertEqual(tva.result_url("/api/files/fil_1/content", root),
                         root + "/api/files/fil_1/content")
        self.assertEqual(tva.result_url("/files/download/x.json", root),
                         root + "/api/files/download/x.json")
        self.assertEqual(tva.result_url("https://s3.example.com/x?sig=1", root),
                         "https://s3.example.com/x?sig=1")

    def test_event_times_are_printed_in_local_time(self):
        import datetime
        utc = datetime.datetime(2026, 11, 6, 12, 0, 0, tzinfo=datetime.timezone.utc)
        self.assertEqual(tva.local_time("2026-11-06T12:00:00Z"),
                         utc.astimezone().strftime("%H:%M:%S"))

    def test_setting_key_names_the_values(self):
        values = {"max_frames": 32, "size": 448, "max_new_tokens": 8192}
        self.assertEqual(tva.setting_key(values), "tva-f32-s448-t8192")
        self.assertTrue(tva.setting_key(values, "blp_abc123def").startswith("tva-123def-"))

    def test_default_budget_is_above_the_empty_result_value(self):
        # 2048 produced results:[] on a clip with a skipped step.
        self.assertGreaterEqual(tva.DEFAULT_MAX_NEW_TOKENS, 4096)

    def test_no_bundles_instances_or_logs(self):
        for gone in ("/bundles", "/instances", "/logs\"", "get_logs", "archetypeai import"):
            self.assertNotIn(gone, RUNNER_SRC)

    def test_runner_is_stdlib_only(self):
        third_party = ("archetypeai", "requests", "numpy", "pandas", "httpx", "aiohttp",
                       "sklearn", "torch")
        for mod in third_party:
            self.assertNotIn(f"import {mod}", RUNNER_SRC)


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

            with mock.patch.object(tva.time, "sleep"), mock.patch("builtins.print"):
                self.assertTrue(tva.wait_until_parses(save, out))
            self.assertEqual(n["saves"], 2)


if __name__ == "__main__":
    unittest.main()
