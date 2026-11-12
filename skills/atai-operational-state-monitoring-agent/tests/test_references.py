"""Network-free unit tests for the OSM agent reference script.

Mocks the HTTP helper at the request boundary, so these run with NO credentials and
NO network. They lock in the verified invariants:
  * Every surface lives under one /api prefix: any /api or /vX.Y suffix on
    ATAI_API_ENDPOINT is stripped before appending /api. Both ATAI_API_KEY and
    ATAI_API_ENDPOINT are required — no default endpoint.
  * No bundle or instance call: Path 1 copies the demo blueprint `playground-osm-drilling`
    into a project (once), packages the copy, deploys it as batch and runs a session over
    the upload's file_id (the filename, not the fil_ uid), then deletes the session,
    deployment and package in that order, even after a failed session.
  * evaluate() scores with end-row labeling, skips INVALID_STATE windows,
    and restricts the steady-state cut to seam-free single-label windows.
  * The shipped sample log is on the exact grid the demo validates, and its labels
    sidecar lines up row for row.

Run:
    python -m unittest discover -s skills/atai-operational-state-monitoring-agent/tests
    # or, from this directory:
    python test_references.py
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import re
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REF = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REF))

import run_osm_agent  # noqa: E402

SAMPLE = REF / "sample_data"
STATES = {"drilling", "reaming", "off_bottom", "in_slips", "trip_in_slips", "shut_in"}


ENV = {"ATAI_API_KEY": "sk_test", "ATAI_API_ENDPOINT": "https://x.test"}


class FakeAPI:
    """Stands in for atai_http.request, routing by method and path prefix and recording calls."""

    def __init__(self, routes):
        self.calls = []
        self.routes = routes

    def __call__(self, method, url, body=None, idempotent=None):
        path = url.split("/api", 1)[1]
        self.calls.append((method, path, body))
        for (m, prefix), value in self.routes.items():
            if m == method and path.startswith(prefix):
                return value(path, body) if callable(value) else value
        raise AssertionError(f"unexpected {method} {path}")


class TestEndpointResolution(unittest.TestCase):
    ROOT = "https://platform.u1.archetypeai.app"

    def test_any_env_convention_reduces_to_one_root(self):
        for endpoint in (self.ROOT, self.ROOT + "/", self.ROOT + "/v0.5",
                         self.ROOT + "/v0.5/", self.ROOT + "/api",
                         self.ROOT + "/api/"):
            with mock.patch.dict(os.environ, {"ATAI_API_KEY": "k", "ATAI_API_ENDPOINT": endpoint}):
                self.assertEqual(run_osm_agent.agents(), self.ROOT + "/api/agents")

    def test_endpoint_and_key_required(self):
        for env in ({"ATAI_API_KEY": "k"}, {"ATAI_API_ENDPOINT": self.ROOT}):
            with mock.patch.dict(os.environ, env, clear=True), self.assertRaises(SystemExit):
                run_osm_agent.agents()

    def test_result_ref_resolves_against_the_root(self):
        """A session result's ref already carries /api."""
        with mock.patch.dict(os.environ, ENV):
            self.assertEqual(run_osm_agent.result_url("/api/files/fil_1/content"),
                             "https://x.test/api/files/fil_1/content")


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, ENV)
        patcher.start()
        self.addCleanup(patcher.stop)

    def api(self, routes):
        fake = FakeAPI(routes)
        patcher = mock.patch.object(run_osm_agent, "request", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def test_no_bundle_or_instance_calls(self):
        """Bundles and agent instances are being removed from the platform."""
        src = (REF / "run_osm_agent.py").read_text()
        for gone in ("/bundles", "/instances", "/logs", "archetypeai"):
            self.assertNotIn(gone, src)

    def test_demo_blueprint_is_resolved_by_key(self):
        """blp_ ids differ per deployment; the key resolves everywhere the demo is published."""
        src = (REF / "run_osm_agent.py").read_text()
        self.assertIn('DEMO_BLUEPRINT = "playground-osm-drilling"', src)
        self.assertEqual(re.findall(r"blp_[a-z0-9]{10,}", src), [])

    def test_demo_is_copied_into_the_project_once(self):
        api = self.api({("GET", "/agents/blueprints?"): {"data": []},
                        ("POST", "/agents/blueprints/playground-osm-drilling/copy"):
                            lambda p, b: {"id": "blp_copy", "blueprint_key": b["blueprint_key"]}})
        with mock.patch("builtins.print"):
            self.assertEqual(run_osm_agent.demo_copy("prj_1")["id"], "blp_copy")
        self.assertEqual(api.calls[-1][2], {"project_id": "prj_1",
                                            "blueprint_key": run_osm_agent.COPY_KEY})

    def test_existing_copy_is_reused(self):
        """A second copy under the same key answers 409; a key in a project resolves only
        with ?project_id=."""
        api = self.api({("GET", "/agents/blueprints?"): {"data": [
            {"id": "blp_old", "blueprint_key": run_osm_agent.COPY_KEY}]}})
        self.assertEqual(run_osm_agent.demo_copy("prj_1")["id"], "blp_old")
        self.assertEqual(api.calls, [("GET", "/agents/blueprints?project_id=prj_1"
                                             f"&key={run_osm_agent.COPY_KEY}", None)])

    def test_project_is_reused_by_name(self):
        self.api({("GET", "/projects"): {"data": [{"id": "prj_x", "name": "other"},
                                                   {"id": "prj_1", "name": run_osm_agent.PROJECT}]}})
        self.assertEqual(run_osm_agent.project_id(), "prj_1")

    def test_project_lookup_reads_every_page(self):
        pages = {"": {"data": [{"id": "prj_x", "name": "other"}], "has_more": True, "next_cursor": "c1"},
                 "c1": {"data": [{"id": "prj_2", "name": run_osm_agent.PROJECT}], "has_more": False}}
        self.api({("GET", "/projects"): lambda path, body: pages[path.split("after=")[1] if "after=" in path else ""]})
        self.assertEqual(run_osm_agent.project_id(), "prj_2")

    def test_scoring_window_comes_from_the_blueprint(self):
        self.api({("GET", "/agents/blueprints/blp_1"): {"document": {"values": {"window_size": 16}}}})
        self.assertEqual(run_osm_agent.blueprint_window("blp_1"), 16)

    def _lifecycle(self, status, ended, sessions, results=None):
        for name, fn in (("create_package", lambda bp, name: {"id": "pkg_1"}),
                         ("package_ready", lambda pid: {"id": pid, "status": "ready"}),
                         ("deploy", lambda pkg, kind: {"id": "dpl_1", "kind": kind}),
                         ("deployment_ready", lambda did: {"id": did, "status": "ready"}),
                         ("open_session", lambda dpl, fids: sessions.append(fids) or {"id": "ses_1"}),
                         ("poll", lambda sid: {"status": status, "error": "boom"}),
                         ("last_events", lambda sid: []),
                         ("end", lambda kind, oid: ended.append((kind, oid))),
                         ("download_results", lambda sid, out, inp: results)):
            patcher = mock.patch.object(run_osm_agent, name, fn)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_teardown_goes_session_deployment_package(self):
        """Each holds the next: a deployment with sessions, or a package with deployments,
        can't be deleted. The session's source is the upload's file_id."""
        ended, sessions = [], []
        self._lifecycle("completed", ended, sessions, results="out.csv")
        with mock.patch("builtins.print"):
            self.assertEqual(run_osm_agent.run("blp_1", "log.csv", "out.csv", "in.csv", "n"), "out.csv")
        self.assertEqual(sessions, [["log.csv"]])
        self.assertEqual(ended, [("sessions", "ses_1"), ("deployments", "dpl_1"), ("packages", "pkg_1")])

    def test_failed_session_still_tears_down(self):
        ended = []
        self._lifecycle("failed", ended, [])
        with mock.patch("builtins.print"), self.assertRaises(SystemExit) as cm:
            run_osm_agent.run("blp_1", "log.csv", "out.csv", "in.csv", "n")
        self.assertIn("boom", str(cm.exception))
        self.assertEqual([k for k, _ in ended], ["sessions", "deployments", "packages"])

    def test_failed_package_is_still_deleted(self):
        """A package that never becomes ready is recorded before the wait, so it is deleted too."""
        ended = []
        self._lifecycle("completed", ended, [])

        def failed(pid):
            raise SystemExit(f"{pid} is failed, not ready: build error")

        with mock.patch.object(run_osm_agent, "package_ready", failed), mock.patch("builtins.print"), \
                self.assertRaises(SystemExit):
            run_osm_agent.run("blp_1", "f.csv", "out.csv", "in.csv", "n")
        self.assertEqual(ended, [("packages", "pkg_1")])

    def test_download_reports_empty_results(self):
        self.api({("GET", "/agents/sessions/ses_1/results"): {"data": []}})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIsNone(run_osm_agent.download_results("ses_1", "/tmp/unused.csv", "in.csv"))
        self.assertIn("EMPTY", out.getvalue())


class TestOutputCompleteness(unittest.TestCase):
    """The platform can report completed while the output is still being written."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.input = os.path.join(self.dir, "in.csv")
        with open(self.input, "w") as f:
            f.write("timestamp,a\n1194200000,0\n1194213745,1\n")
        self.output = os.path.join(self.dir, "out.csv")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _write(self, last):
        """An output whose last window (200 s, the demo's 40 rows at 5 s) ends at `last`."""
        with open(self.output, "w") as f:
            f.write(f"finish_timestamp,start_timestamp\n1194200195.0,1194200000.0\n{last},{last - 195}\n")

    def test_output_reaches_the_inputs_last_row(self):
        """Rows after the last whole window are never output: within one window span counts."""
        self._write(1194213595.0)                 # 150 s short: the 30 rows no window holds
        self.assertTrue(run_osm_agent.output_reaches_end(self.output, self.input))
        self._write(1194210000.0)
        self.assertFalse(run_osm_agent.output_reaches_end(self.output, self.input))

    def test_downloads_again_until_complete(self):
        n = {"downloads": 0}

        def fetch(ref, out_path):
            self.assertEqual(ref, "/api/files/fil_1/content")
            n["downloads"] += 1
            self._write(1194213595.0 if n["downloads"] >= 3 else 1194210000.0)

        api = FakeAPI({("GET", "/agents/sessions/ses_1/results"):
                       {"data": [{"data": {"filename": "o.csv", "ref": "/api/files/fil_1/content"}}]}})
        with mock.patch.dict(os.environ, ENV), mock.patch.object(run_osm_agent, "request", api), \
                mock.patch.object(run_osm_agent, "fetch", fetch), \
                mock.patch.object(run_osm_agent.time, "sleep"), mock.patch("builtins.print"):
            self.assertTrue(run_osm_agent.download_results("ses_1", self.output, self.input))
        self.assertEqual(n["downloads"], 3)


class TestArtifactGate(unittest.TestCase):
    """A run reports completed while its artifacts are still being written."""

    @staticmethod
    def _item(status, name="out.csv"):
        data = {"filename": name}
        if status is not None:
            data["file_attributes"] = {"job_output": {"status": status}}
        return {"data": data}

    def test_waits_until_every_artifact_is_complete(self):
        """The run says completed; one artifact is still partial. Do not collect it."""
        calls = []
        rounds = [[self._item("complete"), self._item("partial")],
                  [self._item("complete"), self._item("complete")]]

        def fetch():
            calls.append(1)
            return rounds[min(len(calls) - 1, len(rounds) - 1)]

        with mock.patch.object(run_osm_agent.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            items = run_osm_agent.complete_results(fetch)
        self.assertEqual(len(calls), 2)          # polled again rather than taking the prefix
        self.assertTrue(all(run_osm_agent.artifact_status(i) == "complete" for i in items))

    def test_returns_immediately_when_all_complete(self):
        with contextlib.redirect_stdout(io.StringIO()):
            items = run_osm_agent.complete_results(lambda: [self._item("complete")])
        self.assertEqual(len(items), 1)

    def test_a_platform_that_reports_no_status_is_not_blocked(self):
        """Absent status must not stall a client: older deployments report none."""
        with contextlib.redirect_stdout(io.StringIO()):
            items = run_osm_agent.complete_results(lambda: [self._item(None)])
        self.assertEqual(len(items), 1)

    def test_gives_up_rather_than_hanging(self):
        out = io.StringIO()
        with mock.patch.object(run_osm_agent.time, "sleep"), contextlib.redirect_stdout(out):
            items = run_osm_agent.complete_results(lambda: [self._item("partial")], wait_s=0)
        self.assertEqual(len(items), 1)
        self.assertIn("still incomplete", out.getvalue())


class TestTailRow(unittest.TestCase):
    """The completeness check must survive a row wider than the tail window."""

    HEADER = "finish_timestamp,start_timestamp,predicted_state,embedding_a\n"

    def _write(self, directory, floats_per_row):
        """An output CSV whose embedding column is a quoted, comma-filled vector."""
        path = os.path.join(directory, "out.csv")
        vector = ",".join(["-0.123456"] * floats_per_row)
        with open(path, "w", newline="") as handle:
            handle.write(self.HEADER)
            for finish, start in ((1000.0, 900.0), (2000.0, 1900.0)):
                handle.write(f'{finish},{start},drilling,"[{vector}]"\n')
        return path

    def _input(self, directory, last_timestamp):
        path = os.path.join(directory, "in.csv")
        with open(path, "w", newline="") as handle:
            handle.write("DATE_TIME,sensor\n")
            handle.write(f"1500.0,1\n{last_timestamp},2\n")
        return path

    def test_reads_past_a_row_wider_than_a_small_window(self):
        """An Embeddings row is ~75 KB of quoted vector, far past a 4 KB tail.

        A window landing mid-vector yields a fragment of a float. It still
        parses as a number, so no guard catches it — it just reads as an
        output that stopped early, and the caller re-downloads a complete
        314 MB file until it gives up.
        """
        with tempfile.TemporaryDirectory() as directory:
            wide = self._write(directory, 2000)       # ~24 KB per row
            self.assertGreater(os.path.getsize(wide), 4096)
            header, row = tail_row_of(wide)
            self.assertEqual(row[header.index("finish_timestamp")], "2000.0")

    def test_narrow_rows_are_unaffected(self):
        with tempfile.TemporaryDirectory() as directory:
            header, row = tail_row_of(self._write(directory, 1))
            self.assertEqual(row[header.index("finish_timestamp")], "2000.0")

    def test_reads_a_row_wider_than_a_fixed_megabyte(self):
        """The window grows, so no row width is too wide.

        A fixed window is correct only while every row fits inside it, and the
        failure is silent — a fragment of a float still parses as a number and
        reads as an output that stopped early. This row is deliberately wider
        than the 1 MiB window this helper used to take.
        """
        with tempfile.TemporaryDirectory() as directory:
            wide = self._write(directory, 160_000)      # ~1.8 MB per row
            self.assertGreater(os.path.getsize(wide), 2 << 20)
            header, row = tail_row_of(wide)
            self.assertEqual(row[header.index("finish_timestamp")], "2000.0")
            self.assertEqual(len(row), len(header))

    def test_quoted_commas_do_not_shift_the_columns(self):
        """csv.reader must treat the vector as ONE field, not hundreds."""
        with tempfile.TemporaryDirectory() as directory:
            header, row = tail_row_of(self._write(directory, 50))
            self.assertEqual(len(row), len(header))

    def test_wide_output_counts_as_reaching_the_end(self):
        with tempfile.TemporaryDirectory() as directory:
            wide = self._write(directory, 2000)
            self.assertIs(run_osm_agent.output_reaches_end(
                wide, self._input(directory, 2000.0)), True)

    def test_short_output_still_counts_as_early(self):
        with tempfile.TemporaryDirectory() as directory:
            wide = self._write(directory, 2000)
            self.assertIs(run_osm_agent.output_reaches_end(
                wide, self._input(directory, 999999.0)), False)


def tail_row_of(path):
    return run_osm_agent.tail_row(path)


class TestEvaluate(unittest.TestCase):
    def _score(self, labels_rows, pred_rows, window=2):
        with tempfile.TemporaryDirectory() as td:
            labels_csv = os.path.join(td, "labels.csv")
            pred_csv = os.path.join(td, "pred.csv")
            with open(labels_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["DATE_TIME", "label"])
                w.writerows(labels_rows)
            with open(pred_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "predicted_state", "invalid"])
                w.writerows(pred_rows)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                run_osm_agent.evaluate(pred_csv, labels_csv, window)
            return out.getvalue()

    def test_end_row_labeling_and_invalid_skip(self):
        labels = [(10, "drilling"), (15, "drilling"), (20, "reaming"), (25, "reaming")]
        preds = [
            (15, "drilling", 0),        # correct (end row 15 is drilling)
            (20, "drilling", 0),        # wrong (end row 20 is reaming)
            (25, "INVALID_STATE", 1),   # skipped
            (99, "reaming", 0),         # unmatched timestamp → skipped
        ]
        report = self._score(labels, preds)
        self.assertIn("2 scored windows, 2 invalid/unmatched skipped", report)
        self.assertIn("accuracy: 0.5000", report)

    def test_steady_state_excludes_label_transitions(self):
        labels = [(10, "drilling"), (15, "drilling"), (20, "reaming"), (25, "reaming")]
        # window=2: steady windows end at 15 (drilling,drilling) and 25
        # (reaming,reaming); the window ending at 20 straddles the transition.
        preds = [(15, "drilling", 0), (20, "reaming", 0), (25, "reaming", 0)]
        report = self._score(labels, preds)
        self.assertIn("steady-state accuracy: 1.0000 (2 windows)", report)

    def test_steady_state_excludes_timestamp_seams(self):
        labels = [(10, "drilling"), (15, "drilling"), (200, "drilling")]
        # Δt 15→200 exceeds the 60 s seam bound: the window ending at 200 is
        # contiguous in label but not in time → only one steady window.
        preds = [(15, "drilling", 0), (200, "drilling", 0)]
        report = self._score(labels, preds)
        self.assertIn("steady-state accuracy: 1.0000 (1 windows)", report)


class TestSampleData(unittest.TestCase):
    LOG = SAMPLE / "drilling_volve_f14.csv"
    LABELS = SAMPLE / "drilling_volve_f14_labels.csv"

    def test_log_and_sidecar_line_up(self):
        with open(self.LOG) as f:
            data_rows = list(csv.DictReader(f))
        with open(self.LABELS) as f:
            label_rows = list(csv.DictReader(f))
        self.assertEqual([r["timestamp"] for r in data_rows],
                         [r["timestamp"] for r in label_rows])
        self.assertLessEqual({r["label"] for r in label_rows}, STATES)
        self.assertNotIn("label", data_rows[0])  # inference input carries no labels

    def test_log_is_on_an_exact_grid(self):
        """The demo keeps sample_rate_interval_tolerance 0.05: a window whose row gaps differ
        by more than 5% is INVALID_STATE."""
        with open(self.LOG) as f:
            ts = [int(r["timestamp"]) for r in csv.DictReader(f)]
        self.assertEqual({b - a for a, b in zip(ts, ts[1:])}, {5})


if __name__ == "__main__":
    unittest.main()
