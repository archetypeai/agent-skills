"""Network-free unit tests for the RED agent reference script.

Mocks the HTTP helper at the request boundary, so these run with NO credentials and NO
network. They lock in the invariants verified against dev:
  * Both ATAI_API_KEY and ATAI_API_ENDPOINT are required — no default endpoint.
  * No bundle or instance call: Path 1 copies the demo blueprint `playground-red` into a
    project (once), packages the copy, deploys it as batch and runs a session, then deletes
    the session, deployment and package in that order, even after a failed session.
  * score() applies the design's majority rule (ties to the rare event), skips
    INVALID_STATE windows, reports the false-alarm rate only over windows with
    zero fault rows, and reports incident-level detection separately — a slice
    can score high accuracy while missing its incident entirely. A blueprint's
    prefixed class names (`pump_nshot_pump_breakdown`) match the sidecar's.
  * The shipped slice satisfies what the blueprint validates: a strictly
    regular cadence, monotonic timestamps, and a sidecar aligned row-for-row.

Run:
    python -m unittest discover -s skills/atai-rare-event-detection-agent/tests
    # or, from this directory:
    python test_references.py
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REF = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REF))

import run_red_agent  # noqa: E402

SAMPLE = REF / "sample_data"
FAULT = "pump_breakdown"   # the class name in the shipped sample data


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


class EnvTests(unittest.TestCase):
    def test_endpoint_required(self):
        with mock.patch.dict(os.environ, {"ATAI_API_KEY": "k"}, clear=True):
            with self.assertRaises(SystemExit):
                run_red_agent.agents()

    def test_key_required(self):
        with mock.patch.dict(os.environ, {"ATAI_API_ENDPOINT": "https://x"}, clear=True):
            with self.assertRaises(SystemExit):
                run_red_agent.agents()

    def test_endpoint_reduces_to_one_root(self):
        """Every .env convention in this repo (bare, /api, /v0.5) reduces to the same root."""
        for endpoint in ("https://x", "https://x/", "https://x/v0.5",
                         "https://x/v0.5/", "https://x/api", "https://x/api/"):
            with mock.patch.dict(os.environ, {"ATAI_API_KEY": "k", "ATAI_API_ENDPOINT": endpoint}):
                self.assertEqual(run_red_agent.agents(), "https://x/api/agents", endpoint)

    def test_result_ref_resolves_against_the_root(self):
        """A session result's ref already carries /api."""
        with mock.patch.dict(os.environ, ENV):
            self.assertEqual(run_red_agent.result_url("/api/files/fil_1/content"),
                             "https://x.test/api/files/fil_1/content")


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, ENV)
        patcher.start()
        self.addCleanup(patcher.stop)

    def api(self, routes):
        fake = FakeAPI(routes)
        patcher = mock.patch.object(run_red_agent, "request", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def test_no_bundle_or_instance_calls(self):
        """Bundles and agent instances are being removed from the platform."""
        src = (REF / "run_red_agent.py").read_text()
        for gone in ("/bundles", "/instances", "/logs", "archetypeai"):
            self.assertNotIn(gone, src)

    def test_demo_blueprint_is_resolved_by_key(self):
        """blp_ ids differ per deployment; the key resolves everywhere the demo is published."""
        src = (REF / "run_red_agent.py").read_text()
        self.assertIn('DEMO_BLUEPRINT = "playground-red"', src)
        self.assertEqual(re.findall(r"blp_[a-z0-9]{10,}", src), [])

    def test_demo_is_copied_into_the_project_once(self):
        """Only a blueprint in a project can be packaged and deployed, and the demo is in none."""
        api = self.api({("GET", "/agents/blueprints?"): {"data": []},
                        ("POST", "/agents/blueprints/playground-red/copy"):
                            lambda p, b: {"id": "blp_copy", "blueprint_key": b["blueprint_key"]}})
        self.assertEqual(run_red_agent.demo_copy("prj_1")["id"], "blp_copy")
        self.assertEqual(api.calls[-1][2], {"project_id": "prj_1",
                                            "blueprint_key": run_red_agent.COPY_KEY})

    def test_existing_copy_is_reused(self):
        """A second copy under the same key answers 409; a key in a project resolves only
        with ?project_id=."""
        api = self.api({("GET", "/agents/blueprints?"): {"data": [{"id": "blp_old"}]}})
        self.assertEqual(run_red_agent.demo_copy("prj_1")["id"], "blp_old")
        self.assertEqual(api.calls, [("GET", "/agents/blueprints?project_id=prj_1"
                                             f"&key={run_red_agent.COPY_KEY}", None)])

    def test_project_is_reused_by_name(self):
        self.api({("GET", "/projects"): {"data": [{"id": "prj_x", "name": "other"},
                                                   {"id": "prj_1", "name": run_red_agent.PROJECT}]}})
        self.assertEqual(run_red_agent.project_id(), "prj_1")

    def test_scoring_window_comes_from_the_blueprint(self):
        """Rebuild each row's span at the window the blueprint runs at."""
        self.api({("GET", "/agents/blueprints/blp_1"): {"document": {"values": {"window_size": 64}}}})
        self.assertEqual(run_red_agent.blueprint_window("blp_1"), 64)

    def _lifecycle(self, status, ended, results=None):
        for name, fn in (("create_package", lambda bp, name: {"id": "pkg_1"}),
                         ("deploy", lambda pkg, kind: {"id": "dpl_1", "kind": kind}),
                         ("open_session", lambda dpl, fid: {"id": "ses_1"}),
                         ("poll", lambda sid: {"status": status, "error": "boom"}),
                         ("last_events", lambda sid: []),
                         ("end", lambda kind, oid: ended.append((kind, oid))),
                         ("download_results", lambda sid, out, inp: results)):
            patcher = mock.patch.object(run_red_agent, name, fn)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_teardown_goes_session_deployment_package(self):
        """Each holds the next: a deployment with sessions, or a package with deployments,
        can't be deleted."""
        ended = []
        self._lifecycle("completed", ended, results="out.csv")
        with mock.patch("builtins.print"):
            self.assertEqual(run_red_agent.run("blp_1", "f.csv", "out.csv", "in.csv", "n"), "out.csv")
        self.assertEqual(ended, [("sessions", "ses_1"), ("deployments", "dpl_1"), ("packages", "pkg_1")])

    def test_failed_session_still_tears_down(self):
        """A failed run must not strand a deployment and package on the platform."""
        ended = []
        self._lifecycle("failed", ended)
        with mock.patch("builtins.print"), self.assertRaises(SystemExit) as cm:
            run_red_agent.run("blp_1", "f.csv", "out.csv", "in.csv", "n")
        self.assertIn("boom", str(cm.exception))
        self.assertEqual([k for k, _ in ended], ["sessions", "deployments", "packages"])

    def test_download_reports_empty_results(self):
        self.api({("GET", "/agents/sessions/ses_1/results"): {"data": []}})
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            self.assertIsNone(run_red_agent.download_results("ses_1", "/tmp/unused.csv"))
        self.assertIn("EMPTY", buf.getvalue())


class ScoringTests(unittest.TestCase):
    """score() reads an output CSV and a labels sidecar off disk."""

    def _score(self, preds, labels, window_size=4):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.csv"
            lab = Path(d) / "in_labels.csv"
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["timestamp", "predicted_state",
                                                  "invalid", "p_normal",
                                                  f"p_{FAULT}"])
                w.writeheader()
                w.writerows(preds)
            with open(lab, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["timestamp", "label"])
                w.writeheader()
                w.writerows(labels)
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf):
                run_red_agent.score(str(out), str(lab), window_size)
            return buf.getvalue()

    def test_invalid_windows_are_skipped_not_scored(self):
        labels = [{"timestamp": 60 * i, "label": "normal"} for i in range(8)]
        preds = [{"timestamp": float(60 * i), "predicted_state": "INVALID_STATE",
                  "invalid": "true", "p_normal": 0, f"p_{FAULT}": 0}
                 for i in range(4, 8)]
        out = self._score(preds, labels)
        self.assertIn("4 invalid", out)
        self.assertIn("0 scored windows", out)

    def test_majority_rule_ties_go_to_the_rare_event(self):
        # window_size 4 ending at row 3 spans rows 0-3: two fault, two normal.
        labels = [{"timestamp": 60 * i, "label": FAULT if i < 2 else "normal"}
                  for i in range(4)]
        preds = [{"timestamp": 180.0, "predicted_state": FAULT, "invalid": "false",
                  "p_normal": 0, f"p_{FAULT}": 1}]
        out = self._score(preds, labels)
        self.assertIn("tp=1", out)  # a 50/50 tie counts as the rare event

    def test_high_accuracy_can_coexist_with_a_missed_incident(self):
        """The reason incident-level detection is reported separately."""
        labels = [{"timestamp": 60 * i, "label": "normal"} for i in range(40)]
        for i in (20, 21):                       # a 2-row incident
            labels[i]["label"] = FAULT
        preds = [{"timestamp": float(60 * i), "predicted_state": "normal",
                  "invalid": "false", "p_normal": 1, f"p_{FAULT}": 0}
                 for i in range(3, 40)]
        out = self._score(preds, labels)
        # The incident is reported MISSED even though accuracy stays high, which
        # is the whole reason incident-level detection is a separate view.
        self.assertIn("MISSED", out)
        acc = float(out.split("accuracy:")[1].split()[0])
        self.assertGreater(acc, 0.9)
        self.assertIn("fn=3", out)

    def test_prefixed_class_names_match_the_sidecar(self):
        """playground-red predicts `pump_nshot_pump_breakdown` for the sidecar's `pump_breakdown`."""
        labels = [{"timestamp": 60 * i, "label": FAULT if i >= 4 else "normal"} for i in range(8)]
        preds = [{"timestamp": 180.0, "predicted_state": "pump_nshot_normal", "invalid": "false",
                  "p_normal": 1, f"p_{FAULT}": 0},
                 {"timestamp": 420.0, "predicted_state": f"pump_nshot_{FAULT}", "invalid": "false",
                  "p_normal": 0, f"p_{FAULT}": 1}]
        out = self._score(preds, labels)
        self.assertIn("tp=1 fp=0 fn=0 tn=1", out)
        self.assertIn("DETECTED", out)

    def test_false_alarms_counted_only_on_zero_fault_windows(self):
        labels = [{"timestamp": 60 * i, "label": "normal"} for i in range(8)]
        labels[7]["label"] = FAULT               # last row is fault
        preds = [
            # window ending row 3 spans 0-3: no fault rows -> a real false alarm
            {"timestamp": 180.0, "predicted_state": FAULT, "invalid": "false",
             "p_normal": 0, f"p_{FAULT}": 1},
            # window ending row 7 spans 4-7: contains a fault row -> not counted
            {"timestamp": 420.0, "predicted_state": FAULT, "invalid": "false",
             "p_normal": 0, f"p_{FAULT}": 1},
        ]
        out = self._score(preds, labels)
        self.assertIn("1/1", out)                # one alarm over one clean window


class SampleDataTests(unittest.TestCase):
    """The shipped slices must satisfy what the blueprint validates."""

    INFERENCE = SAMPLE / "pump_eval_inc04.csv"
    LABELS = SAMPLE / "pump_eval_inc04_labels.csv"

    @staticmethod
    def _read(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def test_slice_and_sidecar_line_up(self):
        rows = self._read(self.INFERENCE)
        labels = self._read(self.LABELS)
        self.assertEqual(len(rows), len(labels))
        self.assertEqual([r["timestamp"] for r in rows],
                         [l["timestamp"] for l in labels])

    def test_cadence_is_strictly_regular(self):
        """Irregular timestamps make the blueprint emit INVALID_STATE."""
        ts = [int(r["timestamp"]) for r in self._read(self.INFERENCE)]
        self.assertEqual({b - a for a, b in zip(ts, ts[1:])}, {60})
        self.assertEqual(ts, sorted(ts), "timestamps must be monotonic")

    def test_slice_carries_both_classes_and_normal_context(self):
        labels = [r["label"] for r in self._read(self.LABELS)]
        self.assertEqual(set(labels), {"normal", FAULT})
        # normal context on both sides is what makes a false-alarm rate meaningful
        self.assertEqual(labels[0], "normal")
        self.assertEqual(labels[-1], "normal")
        self.assertLess(labels.count(FAULT) / len(labels), 0.10,
                        "the fault class should be rare — that is the point")


class OutputCompletenessTests(unittest.TestCase):
    """The platform can report completed while the output is still being written."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.input = os.path.join(self.dir, "in.csv")
        with open(self.input, "w") as f:
            f.write("timestamp,a\n1526962440,0\n1527490260,1\n")
        self.output = os.path.join(self.dir, "out.csv")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write(self, finish):
        with open(self.output, "w") as f:
            f.write("finish_timestamp,start_timestamp,predicted_state\n"
                    f"1526966220.0,1526962440.0,normal\n{finish},{finish - 61440},normal\n")

    def test_output_must_reach_the_inputs_end(self):
        self.write(1527490260.0)
        self.assertTrue(run_red_agent.output_reaches_end(self.output, self.input))
        self.write(1527000000.0)
        self.assertFalse(run_red_agent.output_reaches_end(self.output, self.input))

    def test_downloads_again_until_complete(self):
        n = {"downloads": 0}

        def fetch(ref, out_path):
            self.assertEqual(ref, "/api/files/fil_1/content")
            n["downloads"] += 1
            self.write(1527490260.0 if n["downloads"] >= 3 else 1527000000.0)

        api = FakeAPI({("GET", "/agents/sessions/ses_1/results"):
                       {"data": [{"data": {"filename": "o.csv", "ref": "/api/files/fil_1/content"}}]}})
        with mock.patch.dict(os.environ, ENV), mock.patch.object(run_red_agent, "request", api), \
                mock.patch.object(run_red_agent, "fetch", fetch), \
                mock.patch.object(run_red_agent.time, "sleep"), mock.patch("builtins.print"):
            run_red_agent.download_results("ses_1", self.output, self.input)
        self.assertEqual(n["downloads"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
