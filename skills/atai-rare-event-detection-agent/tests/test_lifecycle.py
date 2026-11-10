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
import library_embeddings  # noqa: E402
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

    def test_span_and_outcome_at_a_step(self):
        lab = ["normal"] * 20 + ["breakdown"] * 5 + ["normal"] * 20      # 45 rows, window 4, step 2: 21 windows
        first, n = serve.span(lab, "normal", 4, 3, 21, step=2)
        self.assertEqual(first * 2 + 4 - 1, 17)                          # the first window ends 3 rows before onset
        self.assertEqual((first + n - 1) * 2 + 4 - 1, 27)                # and the last 3 rows after the last event row
        preds = ["breakdown" if 20 <= (first + k) * 2 + 3 <= 26 else "normal" for k in range(n)]
        self.assertEqual(serve.outcome(preds, first, 4, lab, "normal", step=2)["false_alarms"], 0)

    def test_cleanup_deletes_sessions_before_deployments_before_their_package(self):
        dpls = {"pkg_s": ["dpl_s"], "pkg_d": ["dpl_d"]}
        sess = {"dpl_s": ["ses_s"], "dpl_d": ["ses_1", "ses_2"]}
        t = cleanup.targets([("serving", "pkg_s"), ("delivery", "pkg_d")], dpls.get, sess.get)
        self.assertEqual(t, [("serving", "sessions", "ses_s"), ("serving", "deployments", "dpl_s"),
                             ("serving", "packages", "pkg_s"),
                             ("delivery", "sessions", "ses_1"), ("delivery", "sessions", "ses_2"),
                             ("delivery", "deployments", "dpl_d"), ("delivery", "packages", "pkg_d")])
        self.assertEqual(cleanup.targets([], lambda p: [], lambda d: []), [])


class Promote(unittest.TestCase):
    """Choosing the key needs the admin role outside projects: promote under the platform's unless asked."""

    def promote(self, key=None):
        calls = []

        def fake(method, url, body=None):
            calls.append(body)
            return {"id": "blp_1", "blueprint_key": body.get("blueprint_key", "red-otr_abc123")}
        run = {"optimization": "opt_1", "pinned": {"window_size": 64}}
        trial = {"id": "otr_abc123", "trial_values": {"values": {}}}
        with mock.patch.object(promote_and_test, "request", fake), \
                mock.patch.object(promote_and_test, "agents", lambda: "https://x/api/agents"):
            return promote_and_test.promote(run, trial, "red-sample", key), calls

    def test_the_platforms_key_by_default(self):
        bp, calls = self.promote()
        self.assertEqual(len(calls), 1)
        self.assertNotIn("blueprint_key", calls[0])
        self.assertEqual(bp["key"], "red-otr_abc123")

    def test_a_chosen_key_when_asked(self):
        bp, calls = self.promote("red-mine")
        self.assertEqual(calls[0]["blueprint_key"], "red-mine")
        self.assertEqual(bp["key"], "red-mine")

class DeliverResume(unittest.TestCase):
    """A file whose session failed without a complete output gets a new session on the next run;
    a file longer than --max-windows-per-run goes in several sessions whose outputs join to its own."""

    def deliver(self, status, *args):
        started, self.cancelled = [], []

        def fake(method, url, body=None):
            if method == "GET":
                running = status != "completed" or url.split("/")[-1] in getattr(self, "still_running", ())
                return {"status": "running" if running else "completed",
                        "error": "DeadlineExceeded: stopped after reaching the platform's maximum runtime of 2 hours."}
            if url.endswith("/cancel"):
                self.cancelled.append(url.split("/")[-2])
                return {}
            raise AssertionError(f"{method} {url}")

        def open_session(deployment, file_id):
            assert deployment == "dpl_1", deployment
            started.append(file_id)
            return {"id": f"ses_new{len(started)}"}

        def download(session, dst, src, ts_col):     # a batch output: a window every `step` rows of src
            if status != "completed":
                return None
            ts = pd.read_csv(src)["t"].tolist()
            rows = [(ts[s + 3], ts[s], "x") for s in range(0, len(ts) - 4 + 1, 2)]
            pd.DataFrame(rows, columns=["finish_timestamp", "start_timestamp", "predicted"]).to_csv(dst, index=False)
            return dst
        with mock.patch.multiple(deliver, load_dotenv=lambda: None, check_auth=lambda log: None,
                                 load_manifest=lambda roles: {"delivery": [{"file": "delivery/d.csv"}], "timestamp_column": "t"},
                                 root=lambda: "https://x", same_deployment=lambda s: True,
                                 request=fake, open_session=open_session,
                                 lifecycle_url=lambda kind, obj="": f"https://x/api/agents/{kind}/{obj}",
                                 upload_file=lambda src, name: {"file_id": "fil_1"},
                                 poll=lambda session: status, download=download):
            deliver.main(["--roles", self.roles, "--out", self.out, *args])
        return started, json.load(open(os.path.join(self.out, "deliver_state.json")))["sessions"]

    def setUp(self):
        self.out, self.roles = tempfile.mkdtemp(), tempfile.mkdtemp()
        os.makedirs(os.path.join(self.roles, "delivery"))
        pd.DataFrame({"t": range(100, 120), "v": range(20)}).to_csv(os.path.join(self.roles, "delivery", "d.csv"), index=False)
        # window 4, step 2: 20 rows hold 9 windows
        for name, state in (("test_state.json", {"endpoint": "https://x", "blueprint": {"key": "bp", "id": "blp_1", "values": {"window_size": 4, "step_size": 2}}}),
                            ("deliver_state.json", {"endpoint": "https://x", "blueprint": "bp", "package": "pkg_1",
                                                    "deployment": "dpl_1", "sessions": {"delivery/d.csv": "ses_dead"}})):
            json.dump(state, open(os.path.join(self.out, name), "w"))

    def test_a_failed_session_is_replaced_on_the_next_run(self):
        started, runs = self.deliver("failed")
        self.assertEqual(started, [])                       # this run only polls the dead session
        self.assertEqual(runs, {})
        started, runs = self.deliver("timeout")             # still running when this run stopped waiting
        self.assertEqual(len(started), 1)
        self.assertEqual(runs, {"delivery/d.csv": "ses_new1"})

    def test_a_completed_or_still_running_session_is_kept(self):
        for status in ("completed", "timeout"):
            os.makedirs(os.path.join(self.out, "delivery"), exist_ok=True)
            started, runs = self.deliver(status)
            self.assertEqual((started, runs), ([], {"delivery/d.csv": "ses_dead"}))

    def test_a_bundle_state_is_refused(self):
        sf = os.path.join(self.out, "deliver_state.json")
        json.dump({"endpoint": "https://x", "blueprint": "bp", "bundle": "bnd_1"}, open(sf, "w"))
        with self.assertRaises(SystemExit):
            self.deliver("completed")

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
        state["sessions"] = {}
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

    def test_a_rerun_at_another_split_cancels_the_old_runs_and_reuses_no_file(self):
        state = json.load(open(os.path.join(self.out, "deliver_state.json")))
        state["sessions"] = {}
        json.dump(state, open(os.path.join(self.out, "deliver_state.json"), "w"))
        self.deliver("timeout", "--max-windows-per-run", "4")     # rows 0-10, 8-18, 16-20, still going
        os.makedirs(os.path.join(self.out, "delivery"), exist_ok=True)
        self.still_running = {"ses_new1", "ses_new2", "ses_new3"}
        started, runs = self.deliver("completed", "--max-windows-per-run", "3")
        self.assertEqual(sorted(self.cancelled), ["ses_new1", "ses_new2", "ses_new3"])
        self.assertEqual(sorted(runs), ["delivery/d.csv rows 0-8", "delivery/d.csv rows 12-20",
                                        "delivery/d.csv rows 6-14"])
        whole = pd.read_csv(os.path.join(self.out, "delivery", "d.csv"))
        self.assertEqual(list(whole.start_timestamp), list(range(100, 117, 2)))   # 9 windows, none twice


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


class LibraryCleanup(unittest.TestCase):
    def test_library_packages_are_found_and_the_embeddings_blueprint_kept(self):
        out = tempfile.mkdtemp()
        json.dump({"endpoint": "https://x", "blueprint": "bp", "embeddings_blueprint": {"id": "blp_e", "key": "bp-e"},
                   "package": "pkg_l", "deployment": "dpl_l", "runs": {"training/a.csv": "ses_a"}},
                  open(os.path.join(out, "library_state.json"), "w"))
        json.dump({"endpoint": "https://x", "package": "pkg_ls", "deployment": "dpl_ls", "session": "ses_ls"},
                  open(os.path.join(out, "library_serving_state.json"), "w"))
        # ses_dropped: a session the state forgot (it failed), still listed under the deployment
        listed = {"deployments?package_id=pkg_l": ["dpl_l"], "sessions?deployment_id=dpl_l": ["ses_a", "ses_dropped"],
                  "deployments?package_id=pkg_ls": ["dpl_ls"], "sessions?deployment_id=dpl_ls": ["ses_ls"]}
        ended = []
        with mock.patch.multiple(cleanup, load_dotenv=lambda: None, check_auth=lambda log: None,
                                 same_deployment=lambda st: True, request=lambda method, url: {"id": url},
                                 lifecycle_url=lambda kind, obj="": f"{kind}/{obj}" if obj else kind,
                                 list_all=lambda url: [{"id": i} for i in listed.get(url, [])],
                                 end=lambda kind, obj: ended.append((kind, obj)), log=lambda msg: None), \
                mock.patch("builtins.print"):
            cleanup.main(["--out", out, "--yes"])
        self.assertEqual(ended, [("sessions", "ses_a"), ("sessions", "ses_dropped"), ("deployments", "dpl_l"),
                                 ("packages", "pkg_l"), ("sessions", "ses_ls"), ("deployments", "dpl_ls"),
                                 ("packages", "pkg_ls")])
        self.assertFalse(os.path.exists(os.path.join(out, "library_serving_state.json")))
        kept = json.load(open(os.path.join(out, "library_state.json")))
        self.assertEqual((kept["embeddings_blueprint"]["id"], "package" in kept, "runs" in kept), ("blp_e", False, False))


class TestInPieces(unittest.TestCase):
    """--max-windows: one eval per piece, the platform's confusion matrices summed."""

    def test_the_summed_report(self):
        ev = lambda cm: {"targets": {"event": {"class_names": ["normal", "leak"], "confusion_matrix": cm}}}  # noqa: E731
        t = promote_and_test.summed_report([ev([[5, 1], [0, 2]]), ev([[3, 0], [2, 4]])], "normal")["targets"]["event"]
        self.assertEqual((t["class_names"], t["confusion_matrix"]), (["normal", "leak"], [[8, 1], [2, 6]]))
        agg = t["aggregate"]
        leak_f1, normal_f1 = 2 * 6 / (2 * 6 + 1 + 2), 2 * 8 / (2 * 8 + 2 + 1)   # from the sum, not the pieces' mean
        self.assertAlmostEqual(agg["foreground_macro_f1"], leak_f1)
        self.assertAlmostEqual(agg["macro_f1"], (leak_f1 + normal_f1) / 2)
        self.assertAlmostEqual(agg["accuracy"], 14 / 17)
        self.assertAlmostEqual(agg["detection_recall"], 6 / 8)
        self.assertAlmostEqual(agg["detection_precision"], 6 / 7)
        r = atai_http.report_event({"targets": {"event": t}}, "normal")   # nothing None in test.json
        self.assertTrue(all(r[k] is not None for k in ("accuracy", "detection_recall", "detection_precision")))

    def run_pieces(self, out, max_windows, fail_rows=(), no_report=(), slow=None):
        """test_in_pieces against a fake platform; returns the evals created, as rows, and the result."""
        roles = os.path.join(out, "roles")
        os.makedirs(os.path.join(roles, "test"), exist_ok=True)
        pd.DataFrame({"timestamp": np.arange(20.0), "a": np.arange(20.0), "label": ["normal"] * 20}).to_csv(
            os.path.join(roles, "test", "inc01.csv"), index=False)
        m = {"test": [{"file": "test/inc01.csv"}], "background": "normal"}
        bp = {"id": "blp", "values": {"window_size": 4, "step_size": 1}}
        sf = os.path.join(out, "test_state.json")
        state = json.load(open(sf)) if os.path.exists(sf) else {}
        a = argparse.Namespace(out=out, roles=roles, max_windows=max_windows, parallel=3, name="t")
        # evals persist across runs, as on the platform; slow: rows -> polls still running
        self.made_by_wait = []                     # evals made so far, at each wait
        made, rows_of, polls = [], self.__dict__.setdefault("rows_of", {}), self.__dict__.setdefault("polls", {})
        slow = slow or {}

        def fake(method, url, body=None, idempotent=None):
            if method == "POST":
                eid = f"evl_{len(rows_of) + 1}"
                name = body["examples"][0]["name"]
                rows_of[eid] = name
                made.append(name)
                return {"id": eid, "status": "running"}
            eid = url.rsplit("/", 1)[1]
            polls[eid] = polls.get(eid, 0) + 1
            if any(f"rows{r[0]}-{r[1]}" in rows_of[eid] and polls[eid] <= n for r, n in slow.items()):
                return {"status": "running"}
            bad = any(f"rows{r[0]}-{r[1]}" in rows_of[eid] for r in fail_rows)
            report = None if any(f"rows{r[0]}-{r[1]}" in rows_of[eid] for r in no_report) else {
                "targets": {"event": {"class_names": ["leak", "normal"], "confusion_matrix": [[0, 0], [0, 1]]}}}
            return {"status": "failed" if bad else "completed", "error": "DeadlineExceeded" if bad else None,
                    "metrics_report": report, "output_artifacts": [{"id": f"p_{eid}", "metadata": {"kind": "predictions"}}]}
        with mock.patch.multiple(promote_and_test, request=fake, agents=lambda: "https://x/api/agents",
                                 files_url=lambda: "https://x/api/files", stamp=lambda: "T",
                                 request_raw=lambda url: b'{"w": 1}\n',
                                 upload_all=lambda paths, *a, **k: {p: os.path.basename(p) for p in paths}), \
                mock.patch.object(promote_and_test.time, "sleep", lambda s: self.made_by_wait.append(len(made))):
            result = promote_and_test.test_in_pieces(a, m, bp, state, sf)
        return made, result, json.load(open(sf))

    def test_a_new_max_windows_cuts_a_new_plan(self):
        out = tempfile.mkdtemp()
        with self.assertRaises(SystemExit) as e:                    # the second of two pieces fails
            self.run_pieces(out, 9, fail_rows=[(9, 20)])
        self.assertIn("smaller --max-windows", str(e.exception))
        made, (ev, preds), state = self.run_pieces(out, 5)          # a rerun at a smaller N: a new plan
        self.assertEqual([x["rows"] for x in state["pieces"]], [[0, 8], [5, 13], [10, 18], [15, 20]])
        self.assertEqual(made, ["inc01_rows0-8.csv", "inc01_rows5-13.csv", "inc01_rows10-18.csv", "inc01_rows15-20.csv"])
        self.assertEqual(state["max_windows"], 5)
        self.assertEqual(ev["metrics_report"]["targets"]["event"]["confusion_matrix"], [[0, 0], [0, 4]])
        self.assertEqual(len(preds), 4)

    def test_a_dropped_pieces_eval_still_going_holds_its_slot(self):
        out = tempfile.mkdtemp()
        with self.assertRaises(SystemExit):         # rows 9-20 fails while rows 0-12 is still going
            self.run_pieces(out, 9, fail_rows=[(9, 20)], slow={(0, 12): 4})
        self.assertEqual(json.load(open(os.path.join(out, "test_state.json")))["pieces"][0]["status"], "running")
        made, _, state = self.run_pieces(out, 5, slow={(0, 12): 4})     # a new plan drops rows 0-12
        self.assertEqual(len(made), 4)
        # rows 0-12 holds one of the 3 slots until it ends: 2 new evals at first, not 3
        self.assertEqual(self.made_by_wait[0], 2)
        self.assertEqual([d["eval"] for d in state["dropped"]], ["evl_1"])   # tracked, not waited for

    def test_an_unchanged_piece_keeps_its_eval(self):
        out = tempfile.mkdtemp()
        made, _, first = self.run_pieces(out, 17)                   # 17 windows: one piece, rows 0-20
        self.assertEqual(made, ["inc01_rows0-20.csv"])
        made, _, second = self.run_pieces(out, 20)                  # the same piece: no new eval
        self.assertEqual(made, [])
        self.assertEqual(second["pieces"][0]["eval"], first["pieces"][0]["eval"])

    def test_an_eval_without_a_report_is_tested_again(self):
        out = tempfile.mkdtemp()
        with self.assertRaises(SystemExit):
            self.run_pieces(out, 9, no_report=[(0, 12)])
        made, _, _ = self.run_pieces(out, 9)
        self.assertEqual(made, ["inc01_rows0-12.csv"])


class LibraryEmbeddings(unittest.TestCase):
    """The training windows cut as the platform cuts them, embedded over /query, and PCA 1-2."""

    def test_windows_tile_from_row_0(self):
        self.assertEqual(len(library_embeddings.starts(2760, 512, 4)), 563)   # PRONTO's training leak at 512, step 4
        self.assertEqual(library_embeddings.starts(10, 4, 3), [0, 3, 6])
        self.assertEqual(library_embeddings.starts(3, 4, 1), [])

    def test_the_map_faces_the_background_and_the_event(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(0, 1, (60, 12)) + np.r_[5, np.zeros(11)]
        leak = rng.normal(0, 1, (40, 12)) - np.r_[5, np.zeros(11)]
        v = library_embeddings.landscape(["normal", "leak"], "normal", {"normal": normal, "leak": leak}, 8, 4, 400)
        self.assertGreater(v["prototypes"][0][0], 0)                # normal on PC 1's positive side
        self.assertGreater(v["prototypes"][1][1], 0)                # the event on PC 2's
        X = np.vstack([normal, leak])
        rows, mean = np.array(v["components"]), np.array(v["mean"])
        shot = (X[0] - mean) @ rows.T                               # a served window lands where its shot is
        self.assertAlmostEqual(shot[0], v["shots"][0]["x"], places=2)
        self.assertEqual((v["n"], v["n_fitted"], len(v["shots"])), (100, 400, 100))
        self.assertEqual({s["c"] for s in v["shots"]}, {0, 1})

    EMB = {"id": "blp_emb", "key": "bp-embeddings"}

    def serve(self, rec, *args, session_bp="blp_emb", out=None, failing_query=False, failing_run=False, serving=True,
              step="2", trial_step=1):
        """main (--serving by default) against a fake platform; returns out, the windows asked, the
        other calls, the roles and the manifest. The embeddings blueprint is already made."""
        roles, out = tempfile.mkdtemp(), out or tempfile.mkdtemp()
        os.makedirs(os.path.join(roles, "training"))
        manifest = {"contract": "red-role-files/v1", "timestamp_column": "timestamp", "channels": ["a", "b"],
                    "classes": ["normal", "leak"], "background": "normal", "training": []}
        for name, cls, rows in (("normal", "normal", 20), ("leak_inc01", "leak", 11)):
            pd.DataFrame({"timestamp": np.arange(rows) * 1.0, "a": np.arange(rows), "b": -np.arange(rows)}).to_csv(
                os.path.join(roles, "training", f"{name}.csv"), index=False)
            manifest["training"].append({"file": f"training/{name}.csv", "class": cls, "n_rows": rows})
        json.dump({"endpoint": "https://x", "blueprint": {"key": "bp", "id": "blp_bp", "values": {
            "window_size": 4, "step_size": trial_step, "metric": "l1"}}}, open(os.path.join(out, "test_state.json"), "w"))
        lib = os.path.join(out, "library_state.json")
        if not os.path.exists(lib):
            json.dump({"endpoint": "https://x", "embeddings_blueprint": self.EMB}, open(lib, "w"))
        asked, calls = [], []

        def fake(method, url, body=None, idempotent=None, **_):
            if url.endswith("/query"):
                if failing_query:
                    raise RuntimeError("POST /query failed (500)")
                asked.append(body["events"][0]["event_data"]["contents"])
                return {"response": {"data": [rec(body["events"][0]["event_data"]["contents"][0][0])]}}
            path = url.split("/api/agents")[1]
            calls.append((method, path))
            if method == "POST" and path == "/deployments" and failing_run:
                raise RuntimeError("POST /deployments failed (500)")
            if method == "POST" and path in ("/packages", "/deployments", "/sessions"):
                return {"id": {"/packages": "pkg_s", "/deployments": "dpl_s", "/sessions": "ses_s"}[path]}
            if method == "GET" and path.startswith("/packages/"):
                return {"id": path.split("/")[-1], "status": "ready", "blueprint_ids": [session_bp]}
            if method == "GET" and path.startswith("/deployments/"):
                return {"id": path.split("/")[-1], "status": "ready", "package_id": "pkg_app"}
            if method == "GET" and path.startswith("/sessions/"):
                return {"id": path.split("/")[-1], "status": "running", "deployment_id": "dpl_app"}
            return {}
        fakes = dict(request=fake, agents=lambda: "https://x/api/agents", root=lambda: "https://x")
        with mock.patch.multiple(library_embeddings, load_dotenv=lambda: None, check_auth=lambda log: None,
                                 load_manifest=lambda roles: manifest, same_deployment=lambda s: True, **fakes), \
                mock.patch.multiple(atai_http, **fakes), mock.patch.object(atai_http.time, "sleep", lambda s: None):
            library_embeddings.main(["--roles", roles, "--out", out, *(["--serving"] if serving else []),
                                     "--step", step, *args])
        return out, asked, calls, roles, manifest

    @staticmethod
    def rec(first):
        return {"predicted_state": "leak" if first > 5 else "normal", "invalid": False,
                "embedding_a": [first, 1.0, 0.0], "embedding_b": [0.0, first * 2, (first % 3) * 1.0]}

    def test_serving_route_writes_the_batch_shape_and_the_map(self):
        out, asked, _, roles, manifest = self.serve(self.rec, "--session", "ses_app")
        self.assertEqual(len(asked), 9 + 4)                        # (20-4)//2+1 and (11-4)//2+1 windows
        self.assertIn([[2, 3, 4, 5], [-2, -3, -4, -5]], asked)      # a list per channel, the window at row 2
        one = pd.read_csv(os.path.join(out, "library_embeddings", "normal.csv"))
        self.assertEqual(list(one.start_timestamp), [0, 2, 4, 6, 8, 10, 12, 14, 16])
        self.assertEqual(list(one.finish_timestamp)[:2], [3, 5])
        summary = json.load(open(os.path.join(out, "library_embeddings.json")))
        self.assertEqual((summary["route"], summary["windows"], summary["dim"], summary["step_size"]),
                         ("serving", 13, 6, 2))
        self.assertEqual(summary["per_class"], {"normal": 9, "leak": 4})
        view = json.load(open(os.path.join(out, library_embeddings.VIEW)))
        self.assertEqual((view["classes"], view["n"], view["n_fitted"], view["window_size"]),
                         (["normal", "leak"], 13, 17 + 8, 4))
        self.assertEqual((len(view["components"]), len(view["components"][0]), len(view["mean"])), (2, 6, 6))
        before = open(os.path.join(out, library_embeddings.VIEW)).read()
        os.remove(os.path.join(out, library_embeddings.VIEW))       # --view: the map again, no platform calls
        with mock.patch.object(library_embeddings, "load_manifest", lambda roles: manifest), \
                mock.patch.object(library_embeddings, "request", side_effect=AssertionError("a platform call")):
            library_embeddings.main(["--roles", roles, "--out", out, "--view"])
        self.assertEqual(open(os.path.join(out, library_embeddings.VIEW)).read(), before)

    def test_its_own_session_deployment_and_package_are_deleted_however_the_run_ends(self):
        out, _, calls, _, _ = self.serve(self.rec)
        self.assertEqual([c[1] for c in calls if c[0] == "DELETE"], ["/sessions/ses_s", "/deployments/dpl_s", "/packages/pkg_s"])
        self.assertFalse(os.path.exists(os.path.join(out, "library_serving_state.json")))
        out2 = tempfile.mkdtemp()                         # a query fails: they are deleted all the same
        with self.assertRaises(RuntimeError):
            self.serve(self.rec, out=out2, failing_query=True)
        self.assertFalse(os.path.exists(os.path.join(out2, "library_serving_state.json")))

    def test_a_left_session_another_blueprint_or_step_stops_it(self):
        out = tempfile.mkdtemp()
        json.dump({"package": "pkg_old", "deployment": "dpl_old", "session": "ses_old"},
                  open(os.path.join(out, "library_serving_state.json"), "w"))
        with self.assertRaises(SystemExit) as e:
            self.serve(self.rec, out=out)
        self.assertIn("cleanup.py", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            self.serve(self.rec, "--session", "ses_other", session_bp="blp_another")
        self.assertIn("not the embeddings blueprint", str(e.exception))
        for old in ({"endpoint": "https://x", "blueprint": "old-bp", "package": "pkg_l"},
                    {"endpoint": "https://x", "blueprint": "bp", "step": 4, "package": "pkg_l"}):
            out = tempfile.mkdtemp()
            json.dump(old, open(os.path.join(out, "library_state.json"), "w"))
            with self.assertRaises(SystemExit) as e:
                self.serve(self.rec, "--session", "ses_app", out=out)
            self.assertIn("move them aside", str(e.exception))

    def test_a_session_without_embeddings_is_refused(self):
        with self.assertRaises(SystemExit) as e:
            self.serve(lambda first: {"predicted_state": "normal"}, "--session", "ses_app")
        self.assertIn("output_embeddings", str(e.exception))


    def test_a_failure_before_the_session_starts_deletes_what_it_made(self):
        out = tempfile.mkdtemp()
        with self.assertRaises(RuntimeError):
            self.serve(self.rec, out=out, failing_run=True)
        self.assertFalse(os.path.exists(os.path.join(out, "library_serving_state.json")))

    def test_another_step_stops_even_after_cleanup(self):
        out, *_ = self.serve(self.rec, "--session", "ses_app")
        os.remove(os.path.join(out, "library_state.json"))          # moved aside; the outputs stay
        with self.assertRaises(SystemExit) as e:
            self.serve(self.rec, "--session", "ses_app", out=out, step="3")
        self.assertIn("at step 2, not bp at step 3", str(e.exception))

    def test_nothing_new_keeps_the_route_and_starts_nothing(self):
        out, *_ = self.serve(self.rec, "--session", "ses_app")
        for serving in (False, True):                               # a batch rerun, then a serving one without --session
            _, asked, calls, _, _ = self.serve(self.rec, out=out, serving=serving)
            self.assertEqual((asked, [c for c in calls if c[0] in ("POST", "DELETE")]), ([], []))
            self.assertEqual(json.load(open(os.path.join(out, "library_embeddings.json")))["route"], "serving")

    def test_n_fitted_counts_the_trials_windows(self):
        out, *_ = self.serve(self.rec, "--session", "ses_app", trial_step=2)
        self.assertEqual(json.load(open(os.path.join(out, library_embeddings.VIEW)))["n_fitted"], 9 + 4)


class LibraryBatch(unittest.TestCase):
    """run_batch: one session per training file, a package only when there is something to embed."""

    def run_batch(self, outcome):
        roles, out = tempfile.mkdtemp(), tempfile.mkdtemp()
        os.makedirs(os.path.join(roles, "training"))
        os.makedirs(os.path.join(out, "library_embeddings"))
        files = [{"file": f"training/{n}.csv"} for n in ("normal", "leak_inc01")]
        for f in files:
            open(os.path.join(roles, f["file"]), "w").write("timestamp,a\n0,1\n")
        a = argparse.Namespace(roles=roles, out=out, name="t", parallel=2, timestamp_column="timestamp")
        state, made = {}, []

        def download(session, dst, src, ts):
            if outcome == "completed":
                open(dst, "w").write("x")
                return dst
            return None
        with mock.patch.multiple(library_embeddings, request=lambda method, url, body=None: {"error": "rpc error: DeadlineExceeded"},
                                 lifecycle_url=lambda kind, obj="": f"https://x/api/agents/{kind}/{obj}",
                                 create_package=lambda bp, name: made.append(("package", bp)) or {"id": "pkg_l"},
                                 deploy=lambda pkg, kind: made.append(("deployment", kind)) or {"id": "dpl_l"},
                                 package_ready=lambda pid: {}, deployment_ready=lambda did: {},
                                 open_session=lambda dpl, fid: made.append(("session", fid)) or {"id": f"ses_{len(made)}"},
                                 upload_file=lambda src, name: {"file_id": name}, poll=lambda session: outcome,
                                 download=download, stamp=lambda: "T"):
            got = library_embeddings.run_batch(a, files, {"id": "blp_emb", "key": "e"}, 4, state, lambda: None)
            again = library_embeddings.run_batch(a, files, {"id": "blp_emb", "key": "e"}, 4, state, lambda: None)
        return got, again, state, made

    def test_each_file_gets_a_session_then_nothing_more(self):
        got, again, state, made = self.run_batch("completed")
        self.assertEqual((got, again), ((True, 2), (True, 0)))      # the rerun embeds nothing and creates nothing
        self.assertEqual([m for m in made if m[0] != "session"], [("package", "blp_emb"), ("deployment", "batch")])
        self.assertEqual(sorted(state["runs"]), ["training/leak_inc01.csv", "training/normal.csv"])

    def test_a_failed_session_is_forgotten_with_a_hint(self):
        with mock.patch.object(library_embeddings, "log") as log:
            got, _, state, _ = self.run_batch("failed")
        self.assertEqual(got[0], False)
        self.assertEqual(state["runs"], {})                         # a rerun starts new sessions
        self.assertTrue(any("--serving" in str(c) for c in log.call_args_list))

if __name__ == "__main__":
    unittest.main()
