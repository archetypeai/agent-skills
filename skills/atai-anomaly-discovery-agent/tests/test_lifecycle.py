"""Network-free unit tests for Path 2 (ad_lifecycle/).

    python -m unittest discover -s skills/atai-anomaly-discovery-agent/tests
"""
import argparse
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "ad_lifecycle"))
import atai_http  # noqa: E402
import deliver  # noqa: E402
import optimize  # noqa: E402
import promote_and_test  # noqa: E402
import score  # noqa: E402
import serve  # noqa: E402
import cleanup  # noqa: E402


def args(**kw):
    d = dict(windows=[48], steps=[8], k=[5, 10], metrics=["l2"], pca=[16], threshold="1.215",
             features=["std", "range"])
    d.update(kw)
    return argparse.Namespace(**d)


class SearchSpace(unittest.TestCase):
    def test_pinned_and_searched(self):
        space, pinned, size = optimize.build(args())
        self.assertEqual(sorted(space["parameters"]), ["n_neighbors"])
        self.assertEqual(size, 2)
        self.assertEqual(pinned["window_size"], 48)
        self.assertEqual(pinned["threshold"], 1.215)
        self.assertEqual(pinned["feature_names"], ["std", "range"])

    def test_n_neighbors_always_searched(self):
        space, pinned, size = optimize.build(args(k=[5]))
        self.assertIn("n_neighbors", space["parameters"])       # the space can't be empty
        self.assertNotIn("n_neighbors", pinned)
        self.assertEqual(size, 1)

    def test_threshold_range_is_a_float_range(self):
        space, pinned, size = optimize.build(args(threshold="1.0:2.0", windows=[32, 48]))
        self.assertEqual(space["parameters"]["threshold"]["spec"], {"type": "float_range", "min": 1.0, "max": 2.0})
        self.assertNotIn("threshold", pinned)
        self.assertIsNone(size)                                  # a float range: random search
        self.assertEqual(space["parameters"]["window_size"]["spec"]["values"], [32, 48])

    def test_budget_runs_small_grids_in_full_and_samples_big_ones(self):
        self.assertEqual(optimize.budget(2), 2)                  # the default: every point
        self.assertEqual(optimize.budget(48), optimize.DEFAULT_TRIALS)   # a random search
        self.assertEqual(optimize.budget(48, 48), 48)            # --max-trials forces the grid
        self.assertEqual(optimize.budget(2, 20), 2)              # never more than the grid
        self.assertEqual(optimize.budget(None), optimize.DEFAULT_TRIALS)  # a float range
        self.assertEqual(optimize.budget(None, 24), 24)


class Scoring(unittest.TestCase):
    def test_sustained(self):
        self.assertIsNone(score.sustained([True, True, False, True, True], 3))
        self.assertEqual(score.sustained([False, True, True, True], 3), 3)

    def test_auroc(self):
        self.assertAlmostEqual(score.auroc(np.array([2.0, 3.0]), np.array([1.0, 1.5])), 1.0)
        self.assertAlmostEqual(score.auroc(np.array([1.0]), np.array([1.0])), 0.5)

    def test_evaluate(self):
        ms = np.arange(20) * 60_000
        rows = pd.DataFrame({"ms": np.r_[ms, ms + 10**9],
                             "anomaly_score": np.r_[np.full(10, 1.0), np.full(10, 3.0), np.full(20, 1.0)],
                             "label": ["normal"] * 10 + ["anomaly"] * 10 + ["normal"] * 20,
                             "recording": ["f"] * 20 + ["n"] * 20, "type": ["idv01"] * 20 + [""] * 20})
        r = score.evaluate(rows, 2.0, 3, 60_000)
        t = r["by_type"]["idv01"]
        self.assertEqual(t["detected"], 1)
        self.assertAlmostEqual(t["median_delay_h"], 2 * 60 / 3600)     # third window after onset
        self.assertEqual(t["window_detection"], 1.0)
        self.assertEqual(r["false_alarms"], 0)
        self.assertEqual(r["normal_windows"], 30)


class PredictionCheck(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "labels"))
        lab = ["normal"] * 6 + ["anomaly"] * 6
        pd.DataFrame({"timestamp": [f"{i}.000" for i in range(12)], "label": lab,
                      "recording": ["r"] * 12, "type": ["idv01"] * 12}).to_csv(
            os.path.join(self.dir, "labels", "t_labels.csv"), index=False)
        self.m = {"test": [{"file": "test/t.csv", "labels": "labels/t_labels.csv"}]}
        # window 4, step 4: windows end on rows 3, 7, 11 -> normal, anomaly, anomaly
        self.preds = [{"example": "t.csv", "row_index": i, "gold": g, "predicted": p, "abstained": False}
                      for i, (g, p) in enumerate([("normal", "normal"), ("anomaly", "anomaly"), ("anomaly", "normal")])]

    def ev(self, tp, fn, fp, tn):
        return {"metrics_report": {"targets": {"anomaly": {
            "class_names": ["anomaly", "normal"], "confusion_matrix": [[tp, fn], [fp, tn]],
            "aggregate": {"macro_f1": 0.5, "foreground_macro_f1": 0.5}}}}}

    def test_pairs_and_matches(self):
        per_type, abstained = promote_and_test.check(self.dir, self.m, self.ev(1, 1, 0, 1), self.preds, 4, 4)
        self.assertEqual(per_type, {"idv01": [1, 2]})
        self.assertEqual(abstained, 0)

    def test_misaligned_gold_is_caught(self):
        bad = [dict(p, gold="normal") for p in self.preds]
        with self.assertRaises(SystemExit):
            promote_and_test.check(self.dir, self.m, self.ev(1, 1, 0, 1), bad, 4, 4)

    def test_confusion_mismatch_is_caught(self):
        with self.assertRaises(SystemExit):
            promote_and_test.check(self.dir, self.m, self.ev(2, 0, 0, 1), self.preds, 4, 4)


class TestEndpoints(unittest.TestCase):
    def test_api_prefix(self):
        for endpoint in ("https://x.test", "https://x.test/", "https://x.test/v0.5",
                         "https://x.test/api", "https://x.test/api/"):
            with mock.patch.dict(os.environ, {"ATAI_API_KEY": "k", "ATAI_API_ENDPOINT": endpoint}):
                self.assertEqual(atai_http.agents(), "https://x.test/api/agents")
                self.assertEqual(atai_http.files_url(), "https://x.test/api/files")

    def test_same_deployment(self):
        env = {"ATAI_API_KEY": "k", "ATAI_API_ENDPOINT": "https://platform.u1.archetypeai.app/api"}
        with mock.patch.dict(os.environ, env):
            self.assertTrue(atai_http.same_deployment({"endpoint": "https://api.u1.archetypeai.app"}))
            self.assertFalse(atai_http.same_deployment({"endpoint": "https://platform.stage.u1.archetypeai.app"}))
            self.assertTrue(atai_http.same_deployment({}))       # saved before endpoints were recorded

    def test_artifact_status(self):
        item = {"data": {"filename": "o.csv", "file_attributes": {"job_output": {"status": "partial"}}}}
        self.assertEqual(deliver.artifact_status(item), "partial")
        self.assertIsNone(deliver.artifact_status({"data": {"filename": "o.csv"}}))


class Serving(unittest.TestCase):
    SRC = {"name": "ad-sample", "document": {"blueprint_id": "blp_x", "values": {
        "step_size": 8, "window_size": 48},
        "connectors": {"source": {"key": "CsvSource"}, "sink": {"key": "CsvSink"}}}}

    def test_served_document(self):
        doc = serve.served_document(self.SRC, ["a", "b"], "timestamp")
        self.assertEqual(doc["values"]["step_size"], 1)
        self.assertNotIn("blueprint_id", doc)
        src = doc["connectors"]["source"]["config"]
        self.assertEqual(src["protocols"], ["atai-records"])
        self.assertEqual(src["models"], ["ad"])
        self.assertEqual(src["data_columns"], [{"dtype": "f32", "names": ["a", "b"]}])
        self.assertEqual(doc["connectors"]["sink"]["config"]["format"], "serving")
        self.assertEqual(self.SRC["document"]["values"]["step_size"], 8)     # the source is left alone

    def test_query_body(self):
        chunk = np.arange(6.0).reshape(3, 2)                                  # 3 rows, 2 channels
        ev = serve.query_body(chunk)["events"][0]
        self.assertEqual(ev["type"], "data.numeric_array")
        self.assertEqual(ev["event_data"]["contents"], [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])

    def test_segments(self):
        labels = pd.DataFrame({"recording": ["n"] * 10 + ["f"] * 6 + ["s"] * 3})
        self.assertEqual(serve.segments(labels, 4), [("n", 0, 7), ("f", 10, 3)])   # "s" is shorter than a window

    def test_first_alarm(self):
        self.assertEqual(serve.first_alarm(["normal", "anomaly", "anomaly", "anomaly"]), 3)
        self.assertIsNone(serve.first_alarm(["anomaly", "anomaly", "normal", "anomaly"]))


class Cleanup(unittest.TestCase):
    def test_agents_before_their_bundle(self):
        t = cleanup.targets({"bundle": "bnd_s", "agent": "agt_s"},
                            {"bundle": "bnd_d", "agents": {"f1": "agt_1", "f2": "agt_2"}}, {"agt_0", "agt_1"})
        self.assertEqual(t, [("serving", "instances", "agt_s"), ("serving", "bundles", "bnd_s"),
                             ("delivery", "instances", "agt_0"), ("delivery", "instances", "agt_1"),
                             ("delivery", "instances", "agt_2"), ("delivery", "bundles", "bnd_d")])

    def test_nothing_recorded(self):
        self.assertEqual(cleanup.targets({}, {"agents": {"f": "agt_1"}}, set()), [])
        self.assertEqual(cleanup.targets({"bundle": "bnd_s", "agent": None}, {}, set()),
                         [("serving", "bundles", "bnd_s")])


if __name__ == "__main__":
    unittest.main()
