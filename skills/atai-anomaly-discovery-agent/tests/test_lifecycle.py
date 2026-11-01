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
import library_embeddings  # noqa: E402


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

    def test_calibration_note_when_delivery_normal_is_unlike_calibration(self):
        # SKAB, 2026-10-31: set for 1%, the delivery's normal windows crossed 4.94% (360 / 7,288)
        self.assertIn("4.94%", score.calibration_note({"false_alarm_rate": 360 / 7288, "false_alarms": 360}, 0.01))
        # TEP sample: 0.38% (6 / 1,591), under the target
        self.assertIsNone(score.calibration_note({"false_alarm_rate": 6 / 1591, "false_alarms": 6}, 0.01))
        # a target under 1% reads as itself, not "0%"
        self.assertIn("not the 0.5% it was set for", score.calibration_note({"false_alarm_rate": 0.02, "false_alarms": 50}, 0.005))
        # over twice the target but too few windows to say
        self.assertIsNone(score.calibration_note({"false_alarm_rate": 0.05, "false_alarms": 5}, 0.01))

    def test_sustained_resets_at_a_gap(self):
        self.assertIsNone(score.sustained([True, True, True], 3, pieces=[0, 0, 1]))
        self.assertEqual(score.sustained([True, True, True, True], 3, pieces=[0, 1, 1, 1]), 3)
        self.assertEqual(score.sustained([True, True, True], 3), 2)

    def _paired(self, invalid_at=()):
        """A delivery file of one incident, 1-minute samples, window 4, step 1, with a one-day gap
        after row 10. Onset at row 8; scores high from row 9. Batch abstains on the 3 windows
        across the gap, and on any row in `invalid_at`."""
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "out", "delivery"))
        os.makedirs(os.path.join(d, "roles", "labels"))
        ts = np.r_[np.arange(11) * 60.0, 86400.0 + np.arange(9) * 60.0]
        lab = ["normal"] * 8 + ["anomaly"] * 12
        pd.DataFrame({"timestamp": ts, "label": lab, "recording": "r", "type": "t"}).to_csv(
            os.path.join(d, "roles", "labels", "f.csv"), index=False)
        rows = []
        for end in range(3, 20):
            inv = (ts[end] - ts[end - 3] > 3 * 60) or end in invalid_at
            rows.append({"finish_timestamp": ts[end], "invalid": str(inv).lower(),
                         "anomaly_score": 3.0 if end >= 9 else 1.0})
        pd.DataFrame(rows).to_csv(os.path.join(d, "out", "delivery", "f.csv"), index=False)
        return score.pair(os.path.join(d, "roles"), os.path.join(d, "out"),
                          {"file": "delivery/f.csv", "labels": "labels/f.csv"})

    def test_pair_marks_pieces_and_operating_time(self):
        j, invalid, _ = self._paired()
        self.assertEqual(invalid, 3)                                        # the windows across the gap
        self.assertEqual(j.piece.tolist(), [0] * 8 + [1] * 6)
        self.assertEqual(j.op_ms.iloc[8] - j.op_ms.iloc[7], 4 * 60_000)    # rows 10 -> 14: the gap counts as one sample

    def test_a_run_does_not_span_a_gap_and_the_delay_skips_it(self):
        # rows 9 and 10 cross before the gap: 2 windows, not a run of 3; the run completes on row 16
        r = score.evaluate(self._paired()[0], 2.0, 3, 60_000)["by_type"]["t"]
        self.assertEqual(r["detected"], 1)
        # onset window ends on row 8; the run ends on row 16: 8 samples of operation, the gap one of them
        self.assertAlmostEqual(r["median_delay_h"], 8 * 60 / 3600)
        np.testing.assert_allclose(serve.operating_s(np.r_[np.arange(11) * 60.0, 86400.0 + np.arange(9) * 60.0])[16]
                                   - serve.operating_s(np.r_[np.arange(11) * 60.0, 86400.0 + np.arange(9) * 60.0])[8],
                                   8 * 60)                                  # the same as serve.py at step 1

    def test_an_invalid_window_inside_a_piece_does_not_split_it(self):
        # a window rejected for another reason (row 16) is dropped, as before; the run still completes
        j, _, _ = self._paired(invalid_at=(16,))
        self.assertEqual(j.piece.nunique(), 2)
        r = score.evaluate(j, 2.0, 3, 60_000)["by_type"]["t"]
        self.assertAlmostEqual(r["median_delay_h"], 9 * 60 / 3600)          # rows 14, 15, 17


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

    def test_max_windows_caps_each_recording_not_each_piece(self):
        segs = [("r", 0, 5), ("r", 10, 5), ("q", 20, 5)]
        self.assertEqual(serve.cap(segs, 3), [("r", 0, 3), ("q", 20, 3)])            # stops in piece 1 of r
        self.assertEqual(serve.cap(segs, 7), [("r", 0, 5), ("r", 10, 2), ("q", 20, 5)])
        self.assertEqual(serve.cap(segs, None), segs)

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

            env = {"ATAI_API_ENDPOINT": "https://x/api", "ATAI_API_KEY": "k"}
            with mock.patch.dict(os.environ, env), mock.patch.object(serve, "request", request), \
                    mock.patch.object(serve, "load_dotenv"), \
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

    def test_library_runs_and_a_serving_agent_it_left(self):
        # agt_dropped: a run the state forgot (it failed), still listed under the bundle
        t = cleanup.targets({}, {}, set(), {"bundle": "bnd_l", "runs": {"library/a.csv": "agt_a"}}, {"agt_dropped"},
                            {"bundle": "bnd_ls", "agent": "agt_ls"})
        self.assertEqual(t, [("library", "instances", "agt_a"), ("library", "instances", "agt_dropped"),
                             ("library", "bundles", "bnd_l"),
                             ("library serving", "instances", "agt_ls"), ("library serving", "bundles", "bnd_ls")])

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
        started, self.cancelled = [], []

        def fake(method, url, body=None):
            if method == "GET":
                return {"status": "completed" if status == "completed" else "running",
                        "error": "DeadlineExceeded: stopped after reaching the platform's maximum runtime of 2 hours."}
            if url.endswith("/cancel"):
                self.cancelled.append(url.split("/")[-2])
                return {}
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
        # the whole file's run (completed) is forgotten, not cancelled: this split never reads it
        self.assertEqual((sorted(runs), self.cancelled),
                         (["delivery/d.csv rows 0-10", "delivery/d.csv rows 16-20", "delivery/d.csv rows 8-18"], []))
        pd.testing.assert_frame_equal(pd.read_csv(whole), one)
        self.assertEqual(len(one), 9)

    def test_trial_run_size_scales_with_channels(self):
        self.assertEqual(deliver.trial_max_windows(8), 4_800)      # SKAB: ~60 min at ~80 windows a minute
        self.assertEqual(deliver.trial_max_windows(33), 1_100)     # TEP: its sample ran 1,000 in 24-34 min
        self.assertEqual(deliver.trial_max_windows(10_000), 100)

    def on_trial(self, *args, running=None, completed=False, keep_state=False):
        """deliver on the free trial: 249 windows (window 4, step 2), at most 100 a run."""
        pd.DataFrame({"t": range(1000, 1500), "v": range(500)}).to_csv(
            os.path.join(self.roles, "delivery", "d.csv"), index=False)
        sf = os.path.join(self.out, "deliver_state.json")
        state = json.load(open(sf))
        if not keep_state:
            state["agents"] = {"delivery/d.csv": running} if running else {}
        json.dump(state, open(sf, "w"))

        def download(agent, dst, src, ts_col):     # a batch output: a window every 2 rows of src
            if not completed:
                return None
            ts = pd.read_csv(src)["t"].tolist()
            pd.DataFrame([(ts[s + 3], ts[s]) for s in range(0, len(ts) - 3, 2)],
                         columns=["finish_timestamp", "start_timestamp"]).to_csv(dst, index=False)
            return dst
        started, self.cancelled = [], []

        def fake(method, url, body=None):
            if method == "GET":
                return {"status": "running"}               # the earlier split's agents are still going
            if url.endswith("/cancel"):
                self.cancelled.append(url.split("/")[-2])
                return {}
            started.append(body)
            return {"id": f"agt_new{len(started)}"}
        manifest = {"delivery": [{"file": "delivery/d.csv"}], "timestamp_column": "t", "channels": ["v"]}
        with mock.patch.multiple(deliver, load_dotenv=lambda: None, check_auth=lambda log: None,
                                 load_manifest=lambda roles: manifest, root=lambda: "https://try.archetypeai.app",
                                 same_deployment=lambda s: True, agents=lambda: "https://x/api/agents", request=fake,
                                 upload_file=lambda src, name: {"file_id": "fil_1"},
                                 poll=lambda agent: "completed" if completed else "timeout",
                                 download=download, TRIAL_CHANNEL_WINDOWS_PER_RUN=100):
            deliver.main(["--roles", self.roles, "--out", self.out, *args])
        return started, json.load(open(sf))["agents"]

    def test_the_free_trial_splits_a_long_file_from_the_start(self):
        started, runs = self.on_trial()
        self.assertEqual(len(started), 3)
        self.assertEqual(sorted(runs), ["delivery/d.csv rows 0-202", "delivery/d.csv rows 200-402",
                                        "delivery/d.csv rows 400-500"])

    def test_a_rerun_with_another_split_does_not_reuse_its_files(self):
        # an earlier run split at 50 windows and stopped waiting; a plain rerun splits at 100
        self.on_trial("--max-windows-per-run", "50")
        started, runs = self.on_trial(completed=True, keep_state=True)
        self.assertEqual(len(started), 2)            # rows 0-202 and 200-402; 400-500 is the same run in both
        # the 50-window split's other 4 runs are cancelled, not left holding job slots
        self.assertEqual(sorted(self.cancelled), ["agt_new1", "agt_new2", "agt_new3", "agt_new4"])
        self.assertEqual(sorted(runs), ["delivery/d.csv rows 0-202", "delivery/d.csv rows 200-402",
                                        "delivery/d.csv rows 400-500"])
        whole = pd.read_csv(os.path.join(self.out, "delivery", "d.csv"))
        self.assertEqual(len(whole), 249)
        self.assertEqual(list(whole.start_timestamp), list(range(1000, 1497, 2)))

    def test_zero_turns_the_trial_split_off(self):
        started, runs = self.on_trial("--max-windows-per-run", "0")
        self.assertEqual((len(started), list(runs)), (1, ["delivery/d.csv"]))

    def test_a_whole_file_run_already_started_is_kept(self):
        started, runs = self.on_trial(running="agt_running")
        self.assertEqual((started, runs), ([], {"delivery/d.csv": "agt_running"}))

    def test_other_deployments_are_not_split(self):
        with mock.patch.object(deliver, "deployment", lambda e: "u1.archetypeai.app"):
            started, runs = self.on_trial()
        self.assertEqual(list(runs), ["delivery/d.csv"])


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



class LibraryEmbeddings(unittest.TestCase):
    """The library's windows cut as the platform cuts them, embedded over /query, and PCA 1-2."""

    def test_windows_tile_from_row_0(self):
        self.assertEqual(len(library_embeddings.starts(600, 32, 4)), 143)     # SKAB's valve1_0 library file
        self.assertEqual(library_embeddings.starts(10, 4, 3), [0, 3, 6])
        self.assertEqual(library_embeddings.starts(3, 4, 1), [])

    def test_pca2_rows_project_the_reference(self):
        rng = np.random.default_rng(0)
        for n, dim in ((40, 100), (300, 20)):            # through the Gram matrix, and the covariance
            X = rng.normal(size=(n, dim)) * np.linspace(3, 0.1, dim)
            mean, rows, ref, explained = library_embeddings.pca2(X)
            np.testing.assert_allclose(ref, (X - mean) @ rows.T, atol=1e-9)
            np.testing.assert_allclose(ref.std(axis=0, ddof=1), [1, 1], rtol=1e-6)
            self.assertGreater(explained[0], explained[1])

    def serve(self, rec, *args, agent_bp="bp", out=None, failing_query=False):
        """main --serving against a fake platform; returns out, the windows asked, the other calls."""
        roles, out = tempfile.mkdtemp(), out or tempfile.mkdtemp()
        os.makedirs(os.path.join(roles, "library"))
        manifest = {"timestamp_column": "timestamp", "channels": ["a", "b"], "library": []}
        for name, rows in (("l1", 20), ("l2", 11)):
            pd.DataFrame({"timestamp": np.arange(rows) * 1.0, "a": np.arange(rows), "b": -np.arange(rows)}).to_csv(
                os.path.join(roles, "library", f"{name}.csv"), index=False)
            manifest["library"].append({"file": f"library/{name}.csv", "rows": rows})
        json.dump({"endpoint": "https://x", "blueprint": {"key": "bp", "values": {
            "window_size": 4, "step_size": 2, "threshold": 1.5}}}, open(os.path.join(out, "test_state.json"), "w"))
        asked, calls = [], []

        def fake(method, url, body=None, idempotent=None):
            if url.endswith("/query"):
                if failing_query:
                    raise RuntimeError("POST /query failed (500)")
                asked.append(body["events"][0]["event_data"]["contents"])
                return {"response": {"data": [rec(body["events"][0]["event_data"]["contents"][0][0])]}}
            calls.append((method, url.split("/api/agents")[1]))
            if url.endswith("/bundles"):
                return {"id": "bnd_s"}
            if url.endswith("/run"):
                return {"id": "agt_s"}
            if method == "GET":
                return {"status": "running", "blueprint": agent_bp}
            return {}
        with mock.patch.multiple(library_embeddings, load_dotenv=lambda: None, check_auth=lambda log: None,
                                 load_manifest=lambda roles: manifest, root=lambda: "https://x",
                                 same_deployment=lambda s: True, agents=lambda: "https://x/api/agents", request=fake), \
                mock.patch.multiple(serve, request=fake, agents=lambda: "https://x/api/agents"):
            library_embeddings.main(["--roles", roles, "--out", out, "--serving", *args])
        return out, asked, calls

    def test_serving_route_writes_the_batch_shape_and_the_view(self):
        out, asked, _ = self.serve(self.rec, "--agent", "agt_app")
        self.assertEqual(len(asked), 9 + 4)                        # (20-4)//2+1 and (11-4)//2+1 windows
        self.assertIn([[2, 3, 4, 5], [-2, -3, -4, -5]], asked)      # a list per channel, the window at row 2 (queries run 4 at a time)
        one = pd.read_csv(os.path.join(out, "library_embeddings", "l1.csv"))
        self.assertEqual(list(one.start_timestamp), [0, 2, 4, 6, 8, 10, 12, 14, 16])
        self.assertEqual(list(one.finish_timestamp)[:2], [3, 5])
        summary = json.load(open(os.path.join(out, "library_embeddings.json")))
        self.assertEqual((summary["route"], summary["windows"], summary["dim"]), ("serving", 13, 6))
        view = json.load(open(os.path.join(out, "library-view.json")))
        self.assertEqual(view["layout"]["ref"]["shape"], [13, 2])

    @staticmethod
    def rec(first):
        return {"anomaly_score": 1.0 + first / 100, "invalid": False,
                "embedding_a": [first, 1.0, 0.0], "embedding_b": [0.0, first * 2, 1.0]}

    def test_its_own_agent_is_deleted_however_the_run_ends(self):
        out, _, calls = self.serve(self.rec)
        self.assertIn(("DELETE", "/instances/agt_s"), calls)
        self.assertIn(("DELETE", "/bundles/bnd_s"), calls)
        self.assertFalse(os.path.exists(os.path.join(out, "library_serving_state.json")))
        out2 = tempfile.mkdtemp()                         # a query fails: the agent is deleted all the same
        with self.assertRaises(RuntimeError):
            self.serve(self.rec, out=out2, failing_query=True)
        self.assertFalse(os.path.exists(os.path.join(out2, "library_serving_state.json")))

    def test_a_left_serving_agent_or_another_blueprint_stops_it(self):
        out = tempfile.mkdtemp()
        json.dump({"bundle": "bnd_old", "agent": "agt_old"}, open(os.path.join(out, "library_serving_state.json"), "w"))
        with self.assertRaises(SystemExit) as e:
            self.serve(self.rec, out=out)
        self.assertIn("cleanup.py", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            self.serve(self.rec, "--agent", "agt_other", agent_bp="another-bp")
        self.assertIn("not the promoted bp", str(e.exception))
        out = tempfile.mkdtemp()
        json.dump({"blueprint": "old-bp", "bundle": "bnd_l"}, open(os.path.join(out, "library_state.json"), "w"))
        with self.assertRaises(SystemExit) as e:
            self.serve(self.rec, "--agent", "agt_app", out=out)
        self.assertIn("old-bp", str(e.exception))

    def test_an_agent_without_embeddings_is_refused(self):
        with self.assertRaises(SystemExit) as e:
            self.serve(lambda first: {"anomaly_score": 1.0}, "--agent", "agt_app")
        self.assertIn("output_embeddings", str(e.exception))

if __name__ == "__main__":
    unittest.main()

