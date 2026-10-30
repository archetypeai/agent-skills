"""Network-free unit tests for Path 2 (references/red_lifecycle/).

    python -m pytest skills/atai-rare-event-detection-agent/tests -v
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "red_lifecycle"))
import atai_http  # noqa: E402
import cleanup  # noqa: E402
import deliver  # noqa: E402
import optimize  # noqa: E402
import promote_and_test  # noqa: E402
import score  # noqa: E402
import serve  # noqa: E402


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


class Serving(unittest.TestCase):
    def test_query_body_is_a_list_per_channel(self):
        chunk = np.arange(12, dtype=float).reshape(6, 2)                  # 6 rows, 2 channels
        contents = serve.query_body(chunk)["events"][0]["event_data"]["contents"]
        self.assertEqual(contents, [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]])

    def test_span_around_the_event(self):
        lab = ["normal"] * 20 + ["breakdown"] * 5 + ["normal"] * 20      # 45 rows, window 4: 42 windows
        first, n = serve.span(lab, "normal", 4, 3, 42)
        self.assertEqual((first, first + 4 - 1), (14, 17))               # the first window ends 3 rows before onset
        self.assertEqual(first + n - 1 + 4 - 1, 27)                      # the last 3 rows after the last event row
        self.assertEqual(serve.span(["normal"] * 10, "normal", 4, 3, 7), (0, 7))   # no event: every window
        self.assertEqual(serve.span(lab, "normal", 4, 100, 42), (0, 42))           # clipped to the file

    def test_outcome_matches_score(self):
        lab = ["normal"] * 10 + ["breakdown"] * 5 + ["normal"] * 10
        preds = ["breakdown" if 12 <= 5 + k + 3 <= 16 else "normal" for k in range(10)]   # windows 5..14
        self.assertEqual(serve.outcome(preds, 5, 4, lab, "normal"), {"incidents": [2], "false_alarms": 0})
        self.assertEqual(serve.outcome(["breakdown"] + ["normal"] * 9, 0, 4, lab, "normal"),
                         {"incidents": [None], "false_alarms": 1})

    def test_cleanup_deletes_agents_before_bundles(self):
        t = cleanup.targets({"bundle": "bnd_s", "agent": "agt_s"},
                            {"bundle": "bnd_d", "agents": {"delivery/inc04.csv": "agt_d1"}}, {"agt_d0", "agt_d1"})
        self.assertEqual(t, [("serving", "instances", "agt_s"), ("serving", "bundles", "bnd_s"),
                             ("delivery", "instances", "agt_d0"), ("delivery", "instances", "agt_d1"),
                             ("delivery", "bundles", "bnd_d")])
        self.assertEqual(cleanup.targets({}, {}, set()), [])


class Promote(unittest.TestCase):
    """Choosing the key can need the admin role (the free trial): promote again without one."""

    def promote(self, error):
        calls = []

        def fake(method, url, body=None):
            calls.append(body)
            if "blueprint_key" in body:
                raise error(url)
            return {"id": "blp_1", "blueprint_key": "red-otr_abc123"}
        run = {"optimization": "opt_1", "pinned": {"window_size": 64}}
        trial = {"id": "otr_abc123", "trial_values": {"values": {}}}
        with mock.patch.object(promote_and_test, "request", fake), \
                mock.patch.object(promote_and_test, "agents", lambda: "https://x/api/agents"):
            return promote_and_test.promote(run, trial, "red-sample"), calls

    def test_a_forbidden_key_falls_back_to_the_platforms(self):
        bp, calls = self.promote(lambda url: RuntimeError(f"POST {url} failed (403): choosing the `blueprint_key` of a promoted blueprint requires the admin role"))
        self.assertEqual(calls[0]["blueprint_key"], "red-sample-abc123")
        self.assertNotIn("blueprint_key", calls[1])
        self.assertEqual(calls[0]["name"], calls[1]["name"])
        self.assertEqual(bp["key"], "red-otr_abc123")

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
