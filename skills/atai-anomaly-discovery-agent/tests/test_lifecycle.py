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

    def test_segments_split_at_a_gap_inside_a_recording(self):
        # "n" rows 0-9 and "f" rows 10-21, f with a one-day seam after its 6th row (a gap prepare.py split at):
        # batch marks the windows across it invalid, so the replay must not send them either
        labels = pd.DataFrame({"recording": ["n"] * 10 + ["f"] * 12})
        ts = np.r_[np.arange(16.0), 86415.0 + np.arange(6.0)]
        self.assertEqual(serve.segments(labels, 4, 1, ts), [("n", 0, 7), ("f", 10, 3), ("f", 16, 3)])
        self.assertEqual(serve.segments(labels, 4, 3, ts), [("n", 0, 3), ("f", 12, 1), ("f", 18, 1)])
        self.assertEqual(serve.segments(labels, 4, 1, np.arange(22.0)), [("n", 0, 7), ("f", 10, 9)])

    def test_operating_time_skips_a_gap(self):
        op = serve.operating_s(np.r_[np.arange(5.0) * 60, 86400.0 + np.arange(5.0) * 60])
        self.assertEqual(op.tolist(), [0, 60, 120, 180, 240, 300, 360, 420, 480, 540])   # the seam counts as one step

    def test_replay_of_a_recording_with_a_gap(self):
        """A mocked serving agent scores what it is sent; batch abstains across the gap. The replay
        must send only gap-free windows, so every answer matches batch and none is spliced."""
        window, step, gap_after = 4, 1, 8
        ts = np.r_[np.arange(gap_after) * 60.0, 86400.0 + np.arange(12 - gap_after) * 60.0]
        x = np.r_[np.zeros(gap_after), np.full(12 - gap_after, 10.0)]          # a level change across the gap
        lab = ["normal"] * 10 + ["anomaly"] * 2
        score = lambda w: 1.0 + float(np.ptp(w))                                # spliced windows would score 11
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "roles", "delivery"))
            os.makedirs(os.path.join(d, "roles", "labels"))
            os.makedirs(os.path.join(d, "out", "delivery"))
            pd.DataFrame({"timestamp": ts, "a": x}).to_csv(os.path.join(d, "roles", "delivery", "f.csv"), index=False)
            pd.DataFrame({"timestamp": ts, "label": lab, "recording": "r", "type": "t"}).to_csv(
                os.path.join(d, "roles", "labels", "f.csv"), index=False)
            batch = []
            for i in range(len(ts) - window + 1):
                inv = bool(np.diff(ts[i:i + window]).max() > 61)
                batch.append({"finish_timestamp": ts[i + window - 1], "predicted_label": "normal",
                              "invalid": str(inv).lower(), "anomaly_score": score(x[i:i + window])})
            pd.DataFrame(batch).to_csv(os.path.join(d, "out", "delivery", "f.csv"), index=False)
            json.dump({"contract": "ad-role-files/v1", "timestamp_column": "timestamp", "channels": ["a"],
                       "delivery": [{"file": "delivery/f.csv", "labels": "labels/f.csv"}]},
                      open(os.path.join(d, "roles", "manifest.json"), "w"))
            json.dump({"endpoint": "x", "blueprint": {"key": "bp", "values": {"window_size": window, "step_size": step}}},
                      open(os.path.join(d, "out", "test_state.json"), "w"))
            json.dump({"values": {"step_size": step}}, open(os.path.join(d, "out", "deliver_state.json"), "w"))
            sent = []

            def request(method, url, body=None, **_):
                if url.endswith("/query"):
                    chunk = np.array(body["events"][0]["event_data"]["contents"]).T
                    sent.append(chunk[:, 0].tolist())
                    return {"response": {"data": [{"anomaly_score": score(chunk[:, 0]), "predicted_label": "normal",
                                                   "invalid": "false"}]}}
                return {"id": "bnd" if url.endswith("/bundles") else "agt", "status": "running"}

            with mock.patch.object(serve, "request", request), mock.patch.object(serve, "load_dotenv"), \
                    mock.patch.object(serve, "check_auth"), mock.patch.object(serve, "same_deployment", return_value=True), \
                    mock.patch.object(serve, "end", return_value=True), mock.patch.object(serve, "log"):
                rc = serve.main(["--roles", os.path.join(d, "roles"), "--out", os.path.join(d, "out")])
            summary = json.load(open(os.path.join(d, "out", "serving", "f_wpq1.json")))
        self.assertEqual(rc, 0)
        self.assertEqual(len(sent), 5 + 1)                     # 5 windows before the gap, 1 after; 3 across it skipped
        self.assertTrue(all(np.ptp(w) == 0 for w in sent))    # no window splices the two sides
        self.assertEqual((summary["windows"], summary["same_score"]), (6, 6))
        self.assertEqual(summary["recordings"][0]["pieces"], 2)

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
    """A file whose agent failed without a complete output gets a new agent on the next run;
    a file longer than --max-windows-per-run goes in several runs whose outputs join to its own."""

    def deliver(self, status, *args):
        started = []

        def fake(method, url, body=None):
            if method == "GET":
                return {"error": "DeadlineExceeded: stopped after reaching the platform's maximum runtime of 2 hours."}
            assert url.endswith("/run"), url
            started.append(body)
            return {"id": f"agt_new{len(started)}"}

        def download(agent, dst, src, ts_col):     # a batch output: a window every `step` rows of src
            if status != "completed":
                return None
            ts = pd.read_csv(src)["t"].tolist()
            rows = [(ts[s + 3], ts[s], "x") for s in range(0, len(ts) - 4 + 1, 2)]
            pd.DataFrame(rows, columns=["finish_timestamp", "start_timestamp", "predicted"]).to_csv(dst, index=False)
            return dst
        with mock.patch.multiple(deliver, load_dotenv=lambda: None, check_auth=lambda log: None,
                                 load_manifest=lambda roles: {"delivery": [{"file": "delivery/d.csv"}], "timestamp_column": "t"},
                                 root=lambda: "https://x", same_deployment=lambda s: True,
                                 agents=lambda: "https://x/api/agents", request=fake,
                                 upload_file=lambda src, name: {"file_id": "fil_1"},
                                 poll=lambda agent: status, download=download):
            deliver.main(["--roles", self.roles, "--out", self.out, *args])
        return started, json.load(open(os.path.join(self.out, "deliver_state.json")))["agents"]

    def setUp(self):
        self.out, self.roles = tempfile.mkdtemp(), tempfile.mkdtemp()
        os.makedirs(os.path.join(self.roles, "delivery"))
        pd.DataFrame({"t": range(100, 120), "v": range(20)}).to_csv(os.path.join(self.roles, "delivery", "d.csv"), index=False)
        # window 4, step 2: 20 rows hold 9 windows
        for name, state in (("test_state.json", {"endpoint": "https://x", "blueprint": {"key": "bp", "values": {"window_size": 4, "step_size": 2}}}),
                            ("deliver_state.json", {"endpoint": "https://x", "blueprint": "bp", "bundle": "bnd_1",
                                                    "values": {"step_size": 2}, "agents": {"delivery/d.csv": "agt_dead"}})):
            json.dump(state, open(os.path.join(self.out, name), "w"))

    def test_a_failed_agent_is_replaced_on_the_next_run(self):
        started, runs = self.deliver("failed")
        self.assertEqual(started, [])                       # this run only polls the dead agent
        self.assertEqual(runs, {})
        started, runs = self.deliver("timeout")             # still running when this run stopped waiting
        self.assertEqual(len(started), 1)
        self.assertEqual(runs, {"delivery/d.csv": "agt_new1"})

    def test_a_completed_or_still_running_agent_is_kept(self):
        for status in ("completed", "timeout"):
            os.makedirs(os.path.join(self.out, "delivery"), exist_ok=True)
            started, runs = self.deliver(status)
            self.assertEqual((started, runs), ([], {"delivery/d.csv": "agt_dead"}))

    def test_pieces(self):
        self.assertEqual(deliver.pieces(20, 4, 2, 0), [(0, 20)])
        self.assertEqual(deliver.pieces(20, 4, 2, 9), [(0, 20)])
        # 9 windows at most 4 a run: windows 0-3, 4-7, 8; each run's rows overlap the next by window - step
        self.assertEqual(deliver.pieces(20, 4, 2, 4), [(0, 10), (8, 18), (16, 20)])
        self.assertEqual(deliver.pieces(10, 4, 1, 3), [(0, 6), (3, 9), (6, 10)])

    def test_split_runs_join_to_the_whole_files_output(self):
        whole = os.path.join(self.out, "delivery", "d.csv")
        os.makedirs(os.path.dirname(whole))
        state = json.load(open(os.path.join(self.out, "deliver_state.json")))
        state["agents"] = {}
        json.dump(state, open(os.path.join(self.out, "deliver_state.json"), "w"))
        self.deliver("completed")
        one = pd.read_csv(whole)
        os.remove(whole)
        started, runs = self.deliver("completed", "--max-windows-per-run", "4")
        self.assertEqual(len(started), 3)
        self.assertEqual(sorted(r for r in runs if " rows " in r), ["delivery/d.csv rows 0-10", "delivery/d.csv rows 16-20", "delivery/d.csv rows 8-18"])
        pd.testing.assert_frame_equal(pd.read_csv(whole), one)
        self.assertEqual(len(one), 9)


class QueryRetry(unittest.TestCase):
    """A serving query is retried after a timeout; any other POST is not (it could duplicate)."""

    def run_request(self, **kw):
        calls = []

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"ok": true}'

        def urlopen(req, timeout):
            calls.append(req)
            if len(calls) == 1:
                raise TimeoutError("The read operation timed out")
            return Resp()
        with mock.patch.dict(os.environ, {"ATAI_API_KEY": "k"}), \
                mock.patch.object(atai_http.urllib.request, "urlopen", urlopen), \
                mock.patch.object(atai_http.time, "sleep", lambda s: None):
            return atai_http.request("POST", "https://x/api/agents/instances/a/query", {"data": []}, **kw), calls

    def test_an_idempotent_post_is_retried(self):
        answer, calls = self.run_request(idempotent=True)
        self.assertEqual((answer, len(calls)), ({"ok": True}, 2))

    def test_a_plain_post_is_not(self):
        with self.assertRaises(TimeoutError):
            self.run_request()


if __name__ == "__main__":
    unittest.main()
