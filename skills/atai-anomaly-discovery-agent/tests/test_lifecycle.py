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
        # records as Evals writes them, listed out of order: the window's [start, end] orders them
        windows = [("normal", "normal"), ("anomaly", "anomaly"), ("anomaly", "normal")]
        self.preds = [{"example": "t.csv", "input": 0, "key": {"predicted": [4.0 * i, 4.0 * i + 3]},
                       "expected": g, "predicted": p, "correct": g == p, "outcome": "matched"}
                      for i, (g, p) in reversed(list(enumerate(windows)))]

    def ev(self, tp, fn, fp, tn):
        return {"metrics_report": {"targets": {"anomaly": {
            "class_names": ["anomaly", "normal"], "confusion_matrix": [[tp, fn], [fp, tn]],
            "aggregate": {"macro_f1": 0.5, "foreground_macro_f1": 0.5}}}}}

    def test_pairs_and_matches(self):
        per_type, abstained = promote_and_test.check(self.dir, self.m, self.ev(1, 1, 0, 1), self.preds, 4, 4)
        self.assertEqual(per_type, {"idv01": [1, 2]})
        self.assertEqual(abstained, 0)

    def test_misaligned_gold_is_caught(self):
        bad = [dict(p, expected="normal") for p in self.preds]
        with self.assertRaises(SystemExit):
            promote_and_test.check(self.dir, self.m, self.ev(1, 1, 0, 1), bad, 4, 4)

    def test_abstention_is_counted_not_scored(self):
        preds = [dict(p, predicted=None, correct=False, outcome="missed") if p["key"]["predicted"][0] == 8.0 else p
                 for p in self.preds]
        per_type, abstained = promote_and_test.check(self.dir, self.m, self.ev(1, 0, 0, 1), preds, 4, 4)
        self.assertEqual(per_type, {"idv01": [1, 1]})
        self.assertEqual(abstained, 1)

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
    def test_query_body(self):
        chunk = np.arange(6.0).reshape(3, 2)                                  # 3 rows, 2 channels
        ev = serve.query_body(chunk)["events"][0]
        self.assertEqual(ev["type"], "data.numeric_array")
        self.assertEqual(ev["event_data"]["contents"], [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])

    def test_segments(self):
        labels = pd.DataFrame({"recording": ["n"] * 10 + ["f"] * 6 + ["s"] * 3})
        self.assertEqual(serve.segments(labels, 4), [("n", 0, 7), ("f", 10, 3)])   # "s" is shorter than a window
        # at step 3 the windows start on rows 0, 3, 6, ...: "f" (rows 10-15) holds only the one at 12
        self.assertEqual(serve.segments(labels, 4, 3), [("n", 0, 3), ("f", 12, 1)])

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


class Promote(unittest.TestCase):
    """Choosing the key can need the admin role (the free trial): promote again without one."""

    def promote(self, error):
        calls = []

        def fake(method, url, body=None):
            calls.append(body)
            if "blueprint_key" in body:
                raise error(url)
            return {"id": "blp_1", "blueprint_key": "ad-otr_abc123"}
        run = {"optimization": "opt_1", "pinned": {"window_size": 64}}
        trial = {"id": "otr_abc123", "trial_values": {"values": {}}}
        with mock.patch.object(promote_and_test, "request", fake), \
                mock.patch.object(promote_and_test, "agents", lambda: "https://x/api/agents"):
            return promote_and_test.promote(run, trial, "ad-sample"), calls

    def test_a_forbidden_key_falls_back_to_the_platforms(self):
        bp, calls = self.promote(lambda url: RuntimeError(f"POST {url} failed (403): choosing the `blueprint_key` of a promoted blueprint requires the admin role"))
        self.assertEqual(calls[0]["blueprint_key"], "ad-sample-abc123")
        self.assertNotIn("blueprint_key", calls[1])
        self.assertEqual(calls[0]["name"], calls[1]["name"])
        self.assertEqual(bp["key"], "ad-otr_abc123")

    def test_any_other_error_still_stops(self):
        with self.assertRaises(RuntimeError):
            self.promote(lambda url: RuntimeError(f"POST {url} failed (409): conflict"))

class DeliverResume(unittest.TestCase):
    """A file whose agent failed without a complete output gets a new agent on the next run."""

    def deliver(self, out, status):
        started = []

        def fake(method, url, body=None):
            assert url.endswith("/run"), url
            started.append(body)
            return {"id": f"agt_new{len(started)}"}
        with mock.patch.multiple(deliver, load_dotenv=lambda: None, check_auth=lambda log: None,
                                 load_manifest=lambda roles: {"delivery": [{"file": "delivery/d.csv"}], "timestamp_column": "t"},
                                 root=lambda: "https://x", same_deployment=lambda s: True,
                                 agents=lambda: "https://x/api/agents", request=fake,
                                 upload_file=lambda src, name: {"file_id": "fil_1"},
                                 poll=lambda agent: status, download=lambda *a: None):
            deliver.main(["--roles", "roles", "--out", out])
        return started, json.load(open(os.path.join(out, "deliver_state.json")))["agents"]

    def setUp(self):
        self.out = tempfile.mkdtemp()
        for name, state in (("test_state.json", {"endpoint": "https://x", "blueprint": {"key": "bp"}}),
                            ("deliver_state.json", {"endpoint": "https://x", "blueprint": "bp", "bundle": "bnd_1",
                                                    "values": {}, "agents": {"delivery/d.csv": "agt_dead"}})):
            json.dump(state, open(os.path.join(self.out, name), "w"))

    def test_a_failed_agent_is_replaced_on_the_next_run(self):
        started, runs = self.deliver(self.out, "failed")
        self.assertEqual(started, [])                       # this run only polls the dead agent
        self.assertEqual(runs, {})
        started, runs = self.deliver(self.out, "timeout")      # still running when this run stopped waiting
        self.assertEqual(len(started), 1)
        self.assertEqual(runs, {"delivery/d.csv": "agt_new1"})

    def test_a_completed_or_still_running_agent_is_kept(self):
        for status in ("completed", "timeout"):
            started, runs = self.deliver(self.out, status)
            self.assertEqual((started, runs), ([], {"delivery/d.csv": "agt_dead"}))


if __name__ == "__main__":
    unittest.main()
