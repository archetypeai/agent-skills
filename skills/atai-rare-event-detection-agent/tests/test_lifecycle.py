"""Network-free unit tests for Path 2 (references/red_lifecycle/).

    python -m pytest skills/atai-rare-event-detection-agent/tests -v
"""
import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "red_lifecycle"))
import atai_http  # noqa: E402
import deliver  # noqa: E402
import optimize  # noqa: E402
import promote_and_test  # noqa: E402
import score  # noqa: E402


def args(**kw):
    d = dict(windows=["64"], steps=["1"], metrics=["cosine", "l1", "l2", "dot"])
    d.update(kw)
    return argparse.Namespace(**d)


class SearchSpace(unittest.TestCase):
    def test_the_probe(self):
        space, pinned, size = optimize.build(args())
        self.assertEqual(sorted(space["parameters"]), ["metric"])
        self.assertEqual(pinned, {"window_size": 64, "step_size": 1})
        self.assertEqual(size, 4)

    def test_metric_always_searched(self):
        space, pinned, size = optimize.build(args(metrics=["dot"]))
        self.assertEqual(space["parameters"]["metric"]["spec"]["values"], ["dot"])   # the space can't be empty
        self.assertNotIn("metric", pinned)
        self.assertEqual(size, 1)

    def test_window_list_and_range(self):
        space, pinned, size = optimize.build(args(windows=["16", "32", "64"]))
        self.assertEqual(space["parameters"]["window_size"]["spec"], {"type": "categorical", "values": [16, 32, 64]})
        self.assertEqual(size, 12)
        space, pinned, size = optimize.build(args(windows=["16:84"]))
        self.assertEqual(space["parameters"]["window_size"]["spec"], {"type": "int_range", "min": 16, "max": 84})
        self.assertIsNone(size)

    def test_budget(self):
        self.assertEqual(optimize.budget(4), 4)                      # a small grid in full
        self.assertEqual(optimize.budget(48), optimize.DEFAULT_TRIALS)
        self.assertEqual(optimize.budget(48, 48), 48)
        self.assertEqual(optimize.budget(4, 20), 4)
        self.assertEqual(optimize.budget(None), optimize.DEFAULT_TRIALS)   # a range: random search


class Scoring(unittest.TestCase):
    L = ["normal"] * 10 + ["breakdown"] * 5 + ["normal"] * 10

    def test_window_truth_ties_go_to_the_event(self):
        self.assertEqual(score.window_truth(["normal", "normal", "breakdown", "breakdown"], 0, 4, "normal"), "breakdown")
        self.assertEqual(score.window_truth(["normal", "normal", "normal", "breakdown"], 0, 4, "normal"), "normal")

    def test_incidents_and_delay(self):
        w = 4
        wins = [(s, s + w, "breakdown" if 12 <= s + w - 1 <= 16 else "normal") for s in range(len(self.L) - w + 1)]
        r = score.score_windows(self.L, wins, "normal")
        self.assertEqual(r["incidents"], [{"rows": [10, 15], "length": 5, "class": "breakdown",
                                           "detected": True, "delay_rows": 2}])
        self.assertEqual(r["false_alarms"], 0)                       # windows 13-16 hold event rows
        self.assertEqual(score.incidents_of(["n", "a", "a", "b", "n"], "n"), [(1, 3, "a"), (3, 4, "b")])

    def test_a_missed_incident_and_false_alarms(self):
        w = 4
        wins = [(s, s + w, "breakdown" if s == 0 else "normal") for s in range(len(self.L) - w + 1)]
        r = score.score_windows(self.L, wins, "normal")
        self.assertFalse(r["incidents"][0]["detected"])
        self.assertEqual((r["false_alarms"], r["fp"]), (1, 1))

    def test_wrong_event_class_is_not_a_detection(self):
        wins = [(10, 14, "slip")]
        r = score.score_windows(self.L, wins, "normal")
        self.assertFalse(r["incidents"][0]["detected"])
        self.assertEqual(r["tp"], 1)                                 # event vs background, whichever event


class PromoteAndCheck(unittest.TestCase):
    def test_a_tie_goes_to_the_larger_search(self):
        trial = lambda tid, v: {"id": tid, "status": "completed", "objective_value": v}  # noqa: E731
        trials = {"opt_probe": [trial("a", 0.85)] * 4, "opt_grid": [trial("b", 0.85)] * 12,
                  "opt_worse": [trial("c", 0.84)] * 24}
        runs = [{"optimization": k} for k in trials]
        run, t = promote_and_test.best(runs, trials_of=lambda o: trials[o])
        self.assertEqual((run["optimization"], t["id"]), ("opt_grid", "b"))

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "test"))
        lab = ["normal"] * 6 + ["breakdown"] * 3 + ["normal"] * 3
        pd.DataFrame({"timestamp": range(12), "x": 0.0, "label": lab}).to_csv(
            os.path.join(self.dir, "test", "inc01.csv"), index=False)
        self.m = {"background": "normal", "test": [{"file": "test/inc01.csv"}]}
        # window 4, step 4: windows end on rows 3, 7, 11 -> gold normal, breakdown, normal (last row)
        # records as Evals writes them, listed out of order: the window's [start, end] orders them
        windows = [("normal", "normal"), ("breakdown", "breakdown"), ("normal", "normal")]
        self.preds = [{"example": "inc01.csv", "input": 0, "key": {"predicted": [4.0 * i, 4.0 * i + 3]},
                       "expected": g, "predicted": p, "correct": g == p, "outcome": "matched"}
                      for i, (g, p) in reversed(list(enumerate(windows)))]

    def ev(self, cm):
        return {"metrics_report": {"targets": {"event": {"class_names": ["breakdown", "normal"], "confusion_matrix": cm,
                                                         "aggregate": {"macro_f1": 1, "foreground_macro_f1": 1}}}}}

    def test_pairs_and_matches(self):
        per_file, abstained = promote_and_test.check(self.dir, self.m, self.ev([[1, 0], [0, 2]]), self.preds, 4, 4)
        self.assertEqual(abstained, 0)
        self.assertTrue(per_file["inc01.csv"]["incidents"][0]["detected"])

    def test_misaligned_gold_is_caught(self):
        bad = [dict(p, expected="breakdown") for p in self.preds]
        with self.assertRaises(SystemExit):
            promote_and_test.check(self.dir, self.m, self.ev([[1, 0], [0, 2]]), bad, 4, 4)

    def test_abstention_is_counted_not_scored(self):
        preds = [dict(p, predicted=None, correct=False, outcome="missed") if p["key"]["predicted"][0] == 8.0 else p
                 for p in self.preds]
        per_file, abstained = promote_and_test.check(self.dir, self.m, self.ev([[1, 0], [0, 1]]), preds, 4, 4)
        self.assertEqual(abstained, 1)

    def test_confusion_mismatch_is_caught(self):
        with self.assertRaises(SystemExit):
            promote_and_test.check(self.dir, self.m, self.ev([[1, 0], [1, 1]]), self.preds, 4, 4)


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


if __name__ == "__main__":
    unittest.main()
