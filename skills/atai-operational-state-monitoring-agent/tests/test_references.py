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
    """The client wants the endpoint WITH its /vX.Y suffix.

    It uses api_endpoint verbatim for the files API and strips the version
    itself for the versionless agents API. A bare root therefore breaks uploads
    while bundle calls keep working. Both .env conventions in this repo must
    normalise to the same thing.
    """

    def test_bare_root_gains_the_version(self):
        for endpoint in ("https://api.u1.archetypeai.app",
                         "https://api.u1.archetypeai.app/"):
            self.assertEqual(run_osm_agent.versioned(endpoint),
                             "https://api.u1.archetypeai.app/v0.5")

    def test_versioned_endpoint_is_left_alone(self):
        for endpoint in ("https://api.u1.archetypeai.app/v0.5",
                         "https://api.u1.archetypeai.app/v0.5/"):
            self.assertEqual(run_osm_agent.versioned(endpoint),
                             "https://api.u1.archetypeai.app/v0.5")


class _FakeClient:
    """Stands in for ArchetypeAI, recording the calls main() makes.

    Mocks the client object rather than HTTP, so these tests assert the
    lifecycle the runner drives — resolve, run, poll, results, download — and
    stay network-free.
    """

    def __init__(self, bundles, calls):
        self.calls = calls
        outer = self

        class _Local:
            def upload(self, path):
                outer.calls.append(("upload", os.path.basename(path)))
                return {"file_id": os.path.basename(path), "file_uid": "fil_x"}

            def download(self, filename, out_path):
                outer.calls.append(("download", filename))
                with open(out_path, "w") as f:
                    f.write("x,y\n")
                return True

        class _Files:
            local = _Local()

        class _Bundles:
            def list(self, query=None, limit=None):
                outer.calls.append(("bundles.list", query))
                return {"data": bundles}

            def create(self, **kw):
                outer.calls.append(("bundles.create", kw))
                raise AssertionError("the quick-start flow must never create a bundle")

            def run(self, bundle_id, source=None, sink=None):
                outer.calls.append(("bundles.run", bundle_id, tuple(source or ())))
                return {"id": "agt_1", "status": "running"}

        class _Instances:
            def get(self, agent_id):
                return {"id": agent_id, "status": "completed"}

            def get_events(self, agent_id):
                return {"data": []}

            def get_results(self, agent_id):
                return {"data": [{"data": {"filename": "out.csv", "num_bytes": 3}}]}

        class _Agents:
            bundles = _Bundles()
            instances = _Instances()

        self.files = _Files()
        self.agents = _Agents()


class TestRequestBodies(unittest.TestCase):
    """Drive main() against a faked client and assert the lifecycle."""

    BUNDLES = [
        {"id": "bnd_emb", "name": "OSM Quick Start (Volve Six State, Embeddings)",
         "is_canonical": True},
        {"id": "bnd_base", "name": "OSM Quick Start (Volve Six State)",
         "is_canonical": True},
    ]

    def _run_main(self, argv):
        calls = []
        env = {"ATAI_API_KEY": "sk_test",
               "ATAI_API_ENDPOINT": "https://api.u1.archetypeai.app"}
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.dict(os.environ, env), \
                mock.patch.object(run_osm_agent, "ArchetypeAI",
                                  lambda *a, **kw: _FakeClient(self.BUNDLES, calls)), \
                mock.patch.object(sys, "argv",
                                  ["run_osm_agent.py",
                                   "--output", os.path.join(td, "out.csv")] + argv), \
                contextlib.redirect_stdout(io.StringIO()):
            run_osm_agent.main()
        return calls

    def test_resolves_exact_name_not_first_result(self):
        """A prefix query returns both variants, Embeddings first (newest-first).

        Taking data[0] would silently run the Embeddings bundle — a 314 MB
        output in place of a 221 KB one — so the runner matches the name
        exactly. The fake deliberately lists Embeddings first.
        """
        calls = self._run_main(["--csv", str(SAMPLE / "volve_states_opt_slice_04.csv")])
        run = next(c for c in calls if c[0] == "bundles.run")
        self.assertEqual(run[1], "bnd_base")
        self.assertEqual(run[2], ("volve_states_opt_slice_04.csv",))

    def test_embeddings_selects_exact_variant(self):
        calls = self._run_main(["--csv", str(SAMPLE / "volve_states_opt_slice_04.csv"),
                                "--embeddings"])
        run = next(c for c in calls if c[0] == "bundles.run")
        self.assertEqual(run[1], "bnd_emb")

    def test_bundle_id_override_skips_lookup(self):
        calls = self._run_main(["--csv", str(SAMPLE / "volve_states_opt_slice_04.csv"),
                                "--bundle-id", "bnd_pinned"])
        self.assertNotIn("bundles.list", [c[0] for c in calls])
        run = next(c for c in calls if c[0] == "bundles.run")
        self.assertEqual(run[1], "bnd_pinned")

    def test_source_is_the_file_id_not_the_uid(self):
        calls = self._run_main(["--csv", str(SAMPLE / "volve_states_opt_slice_04.csv")])
        run = next(c for c in calls if c[0] == "bundles.run")
        self.assertNotIn("fil_x", run[2])


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
