"""Network-free unit tests for Path 2 (references/osm_lifecycle/).

Mocks urllib at the request boundary, so these run with NO credentials and NO
network. They lock in the invariants verified on the dev deployment (2026-09-30):
  * The search space: window / step are `value` parameters, k / metric / weights
    are `fitting` parameters, every spec categorical; max_trials defaults to the
    product's size (every point once); step > window needs --allow-gaps.
  * Blueprint keys are unique per trial (a rerun never reuses an older model),
    and the promoted trial is the best completed one by objective_value.
  * Scoring pairs each prediction with the label at its window's last row,
    skips invalid windows, and counts an absent state as F1 = 0.
  * Timestamps parse from epoch seconds, epoch ms and ISO 8601; output rows are
    matched to delivery files by time range.
  * Uploads retry through a DNS failure; POSTs never retry once delivered.
  * The endpoint loses any /vX.Y suffix before /agents.

Run:
    python -m unittest discover -s skills/atai-operational-state-monitoring-agent/tests
"""
from __future__ import annotations

import io
import json
import os
import socket
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

LIFE = Path(__file__).resolve().parent.parent / "references" / "osm_lifecycle"
sys.path.insert(0, str(LIFE))

import atai_http  # noqa: E402
import deliver  # noqa: E402
import optimize  # noqa: E402
import promote_and_test  # noqa: E402

try:
    import pandas as pd  # noqa: E402
    import score  # noqa: E402
except ImportError:          # score.py needs pandas / numpy (osm_lifecycle/requirements.txt)
    pd = score = None


def trial(tid, value, status="completed", w=512, st=512, k=5, metric="cosine", weights="uniform"):
    return {"id": tid, "status": status, "objective_value": value, "trial_number": 1,
            "trial_values": {"values": {"window_size": w, "step_size": st},
                             "fitting": {"k_neighbors": k, "metric": metric, "weights": weights}}}


class TestSearchSpace(unittest.TestCase):
    def test_kinds_and_specs(self):
        space = optimize.search_space([512], [256, 512], [5, 31], ["cosine"], ["uniform"])["parameters"]
        self.assertEqual({p: v["kind"] for p, v in space.items()},
                         {"window_size": "value", "step_size": "value", "k_neighbors": "fitting",
                          "metric": "fitting", "weights": "fitting"})
        self.assertTrue(all(v["spec"]["type"] == "categorical" for v in space.values()))
        self.assertEqual(space["k_neighbors"]["spec"]["values"], [5, 31])

    def test_grid_size_is_the_default_budget(self):
        self.assertEqual(len(optimize.grid([256, 512], [512], [5, 31], ["cosine", "l1"], ["uniform"])), 8)

    def test_step_over_window_is_flagged(self):
        pts = optimize.grid([256, 512], [512], [5], ["cosine"], ["uniform"])
        self.assertEqual(optimize.gap_pairs(pts), [(256, 512)])
        self.assertEqual(optimize.gap_pairs(optimize.grid([512], [512], [5], ["cosine"], ["uniform"])), [])


class TestPromote(unittest.TestCase):
    def test_key_is_unique_per_trial(self):
        a = promote_and_test.blueprint_key(trial("otr_aaaaaaaaaa1abc12", 0.8), "larco")
        b = promote_and_test.blueprint_key(trial("otr_bbbbbbbbbb2xyz34", 0.8), "larco")
        self.assertEqual(a, "larco-w512-s512-cosine-k5-uniform-1abc12")
        self.assertNotEqual(a, b)

    def test_best_completed_trial_wins(self):
        runs = [{"optimization": {"id": "opt_1"}, "trials": [trial("otr_x1", 0.79), trial("otr_x2", 0.83),
                                                             trial("otr_x3", None, status="failed")]}]
        opt, t = promote_and_test.best_trial(runs)
        self.assertEqual((opt, t["id"]), ("opt_1", "otr_x2"))
        self.assertEqual(promote_and_test.best_trial(runs, "otr_x1")[1]["id"], "otr_x1")
        self.assertIsNone(promote_and_test.best_trial([{"optimization": {"id": "o"}, "trials": []}]))


class TestTimestamps(unittest.TestCase):
    def test_formats(self):
        ms = 1694181225450
        for v in ("1694181225.45", "1694181225.450", "1694181225450", "2023-09-08T13:53:45.450Z"):
            self.assertEqual(deliver.to_ms(v), ms, v)

    def test_rows_match_files_by_time(self):
        ranges = [(1000, 1999, "a.csv"), (3000, 3999, "b.csv")]
        rows = [["1.5", "x"], ["3.2", "y"], ["2.5", "z"]]
        got, unmatched = deliver.match_rows(rows, 0, ranges)
        self.assertEqual({k: len(v) for k, v in got.items()}, {"a.csv": 1, "b.csv": 1})
        self.assertEqual(unmatched, 1)

    def test_time_range_reads_ends_only(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("timestamp,a\n1687797871.590,0\n1687797871.595,1\n1687797871.600,2\n")
        try:
            self.assertEqual(deliver.time_range(f.name), (1687797871590, 1687797871600))
        finally:
            os.unlink(f.name)


@unittest.skipIf(score is None, "pandas / numpy not installed")
class TestScoring(unittest.TestCase):
    STATES = ["drain", "fill", "spin", "wash"]

    def test_last_record_pairing_and_invalid(self):
        labels = pd.DataFrame({"timestamp": ["10.000", "10.005", "10.010", "10.015"],
                               "label": ["wash", "wash", "spin", "spin"]})
        pred = pd.DataFrame({"finish_timestamp": ["10.005", "10.015", "10.01", "99.0"],
                             "predicted_state": ["wash", "spin", "wash", "fill"],
                             "invalid": ["false", "false", "true", "false"]})
        y, p, skipped = score.pairs(pred, labels)
        self.assertEqual(list(y), ["wash", "spin"])
        self.assertEqual(list(p), ["wash", "spin"])
        self.assertEqual(dict(skipped), {"invalid": 1, "no label at finish time": 1})

    def test_absent_state_counts_zero(self):
        r = score.scores(["wash", "spin"], ["wash", "spin"], self.STATES)
        self.assertEqual(r["f1"]["wash"], 1.0)
        self.assertEqual(r["f1"]["drain"], 0.0)
        self.assertEqual(r["macro_f1"], 0.5)          # two perfect states of four


class TestHttp(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"ATAI_API_KEY": "k", "ATAI_API_ENDPOINT": "https://x.test/v0.5/"})
        self.env.start()
        self.sleep = mock.patch.object(atai_http.time, "sleep")
        self.sleep.start()

    def tearDown(self):
        self.env.stop()
        self.sleep.stop()

    def test_versionless_agents(self):
        self.assertEqual(atai_http.agents(), "https://x.test/agents")
        self.assertEqual(atai_http.files_url(), "https://x.test/v0.5/files")

    def test_upload_retries_through_dns_failure(self):
        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = json.dumps({"file_id": "f.csv"}).encode()
        dns = urllib.error.URLError(socket.gaierror(8, "nodename nor servname provided"))
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("timestamp,a\n1,2\n")
        try:
            with mock.patch.object(atai_http.urllib.request, "urlopen", side_effect=[dns, dns, ok]) as op:
                self.assertEqual(atai_http.upload_file(f.name)["file_id"], "f.csv")
            self.assertEqual(op.call_count, 3)
        finally:
            os.unlink(f.name)

    def test_post_not_retried_after_delivery(self):
        timeout = urllib.error.URLError(TimeoutError("read timed out"))
        with mock.patch.object(atai_http.urllib.request, "urlopen", side_effect=[timeout]) as op:
            with self.assertRaises(urllib.error.URLError):
                atai_http.request("POST", "https://x.test/agents/optimizations", {"a": 1})
        self.assertEqual(op.call_count, 1)

    def test_get_retries_gateway_errors(self):
        bad = urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b""))
        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = b'{"status": "completed"}'
        with mock.patch.object(atai_http.urllib.request, "urlopen", side_effect=[bad, ok]):
            self.assertEqual(atai_http.request("GET", "https://x.test/agents/evals/e")["status"], "completed")

    def test_report_f1(self):
        rep = {"targets": {"state": {"class_names": ["drain", "wash"], "confusion_matrix": [[3, 1], [0, 6]]}}}
        f1, n = atai_http.report_f1(rep)
        self.assertEqual(n, 10)
        self.assertAlmostEqual(f1["drain"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
