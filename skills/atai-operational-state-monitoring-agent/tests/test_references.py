"""Network-free unit tests for the OSM agent reference script.

Mocks urllib at the request boundary, so these run with NO credentials and
NO network. They lock in the verified invariants:
  * The agent API is versionless: any /vX.Y suffix on ATAI_API_ENDPOINT is
    stripped before mounting /agents (the files API keeps its own /v0.5).
  * Both ATAI_API_KEY and ATAI_API_ENDPOINT are required — no default
    endpoint.
  * The pre-packaged bundle is resolved by EXACT name via
    GET /agents/bundles?query=... with no bundle creation;
    --embeddings selects the variant, --bundle-id skips the lookup; the run
    body binds the source connector to the upload's file_id (the filename),
    not the fil_ uid.
  * evaluate() scores with end-row labeling, skips INVALID_STATE windows,
    and restricts the steady-state cut to seam-free single-label windows.
  * The shipped sample slice and its labels sidecar line up row-for-row and
    cover all six states.

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
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REF = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REF))

import run_osm_agent  # noqa: E402

SAMPLE = REF / "sample_data"
STATES = {"drilling", "reaming", "off_bottom", "in_slips", "trip_in_slips", "shut_in"}


class TestEndpointResolution(unittest.TestCase):
    def test_version_suffix_is_stripped(self):
        for endpoint in ("https://api.dev.u1.archetypeai.app",
                         "https://api.dev.u1.archetypeai.app/",
                         "https://api.dev.u1.archetypeai.app/v0.5",
                         "https://api.dev.u1.archetypeai.app/v0.5/"):
            with mock.patch.dict(os.environ, {"ATAI_API_ENDPOINT": endpoint}):
                self.assertEqual(run_osm_agent.host_base(),
                                 "https://api.dev.u1.archetypeai.app")

    def test_missing_endpoint_exits(self):
        with mock.patch.dict(os.environ, {"ATAI_API_ENDPOINT": ""}):
            with self.assertRaises(SystemExit):
                run_osm_agent.host_base()


class TestRequestBodies(unittest.TestCase):
    """Drive main() against a canned API and assert the request shapes."""

    def _run_main(self, argv):
        calls = []
        outputs = {
            ("GET", "/agents/bundles"): {"data": [
                {"id": "bnd_base", "name": run_osm_agent.BUNDLE_NAME},
                {"id": "bnd_emb", "name": run_osm_agent.BUNDLE_NAME_EMBEDDINGS},
            ]},
            ("POST", "/agents/bundles/bnd_base/run"): {"id": "agt_1", "status": "running"},
            ("POST", "/agents/bundles/bnd_emb/run"): {"id": "agt_1", "status": "running"},
            ("POST", "/agents/bundles/bnd_pinned/run"): {"id": "agt_1", "status": "running"},
            ("GET", "/agents/instances/agt_1"): {"id": "agt_1", "status": "completed"},
            ("GET", "/agents/instances/agt_1/events"): {"data": []},
            ("GET", "/agents/instances/agt_1/results"): {
                "total": 1,
                "data": [{"data": {"filename": "out.csv", "num_bytes": 3}}],
            },
        }

        def fake_request(method, url, body=None, headers=None):
            # reads carry a ?query=... string; key mocks on the path alone
            path = url.split("archetypeai.app", 1)[1].split("?", 1)[0]
            calls.append((method, path, body))
            return outputs[(method, path)]

        def fake_upload(path):
            calls.append(("UPLOAD", path, None))
            return {"file_id": os.path.basename(path), "file_uid": "fil_x"}

        def fake_urlopen(req):  # the output-CSV download
            return contextlib.closing(io.BytesIO(b"x,y"))

        env = {"ATAI_API_KEY": "sk_test",
               "ATAI_API_ENDPOINT": "https://api.dev.u1.archetypeai.app/v0.5"}
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.dict(os.environ, env), \
                mock.patch.object(run_osm_agent, "request", fake_request), \
                mock.patch.object(run_osm_agent, "upload_file", fake_upload), \
                mock.patch.object(run_osm_agent.urllib.request, "urlopen", fake_urlopen), \
                mock.patch.object(sys, "argv",
                                  ["run_osm_agent.py",
                                   "--output", os.path.join(td, "out.csv")] + argv), \
                contextlib.redirect_stdout(io.StringIO()):
            run_osm_agent.main()
        return calls

    def test_resolve_by_name_and_run_shapes(self):
        calls = self._run_main(["--csv", str(SAMPLE / "volve_states_opt_slice_04.csv")])
        paths = [p for m, p, b in calls]
        # resolved by name via the plural read endpoint; NO bundle creation
        self.assertIn("/agents/bundles", paths)
        self.assertNotIn("/agents/bundle", paths)  # no create, and no singular paths
        run = next((m, p, b) for m, p, b in calls if p.endswith("/run"))
        self.assertEqual(run[1], "/agents/bundles/bnd_base/run")  # exact-name match
        self.assertEqual(run[2]["connectors"]["source"],
                         [{"type": "file", "id": "volve_states_opt_slice_04.csv"}])

    def test_embeddings_selects_exact_variant(self):
        calls = self._run_main(["--csv", str(SAMPLE / "volve_states_opt_slice_04.csv"),
                                "--embeddings"])
        run = next((m, p, b) for m, p, b in calls if p.endswith("/run"))
        self.assertEqual(run[1], "/agents/bundles/bnd_emb/run")

    def test_bundle_id_override_skips_lookup(self):
        calls = self._run_main(["--csv", str(SAMPLE / "volve_states_opt_slice_04.csv"),
                                "--bundle-id", "bnd_pinned"])
        self.assertNotIn("/agents/bundles", [p for m, p, b in calls])
        run = next((m, p, b) for m, p, b in calls if p.endswith("/run"))
        self.assertEqual(run[1], "/agents/bundles/bnd_pinned/run")


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
    def test_slice_and_sidecar_line_up(self):
        with open(SAMPLE / "volve_states_opt_slice_04.csv") as f:
            data_rows = list(csv.DictReader(f))
        with open(SAMPLE / "volve_states_opt_slice_04_labels.csv") as f:
            label_rows = list(csv.DictReader(f))
        self.assertEqual(len(data_rows), len(label_rows))
        self.assertEqual([r["DATE_TIME"] for r in data_rows],
                         [r["DATE_TIME"] for r in label_rows])
        self.assertEqual({r["label"] for r in label_rows}, STATES)
        self.assertNotIn("label", data_rows[0])  # inference input carries no labels


if __name__ == "__main__":
    unittest.main()
