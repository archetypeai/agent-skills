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


class LibraryCleanup(unittest.TestCase):
    def test_library_runs_and_a_serving_agent_it_left(self):
        # agt_dropped: a run the state forgot (it failed), still listed under the bundle
        t = cleanup.targets({}, {}, set(), {"bundle": "bnd_l", "runs": {"training/a.csv": "agt_a"}}, {"agt_dropped"},
                            {"bundle": "bnd_ls", "agent": "agt_ls"})
        self.assertEqual(t, [("library", "instances", "agt_a"), ("library", "instances", "agt_dropped"),
                             ("library", "bundles", "bnd_l"),
                             ("library serving", "instances", "agt_ls"), ("library serving", "bundles", "bnd_ls")])


class TestInPieces(unittest.TestCase):
    """--max-windows: one eval per piece, the platform's confusion matrices summed."""

    def test_the_summed_report(self):
        ev = lambda cm: {"targets": {"event": {"class_names": ["normal", "leak"], "confusion_matrix": cm}}}  # noqa: E731
        names, cm, f1 = promote_and_test.summed_report([ev([[5, 1], [0, 2]]), ev([[3, 0], [2, 4]])])
        self.assertEqual((names, cm), (["normal", "leak"], [[8, 1], [2, 6]]))
        self.assertAlmostEqual(f1[1], 2 * 6 / (2 * 6 + 1 + 2))      # from the sum, not the pieces' mean
        self.assertAlmostEqual(f1[0], 2 * 8 / (2 * 8 + 2 + 1))


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

    def serve(self, rec, *args, agent_bp="bp", out=None, failing_query=False):
        """main --serving against a fake platform; returns out, the windows asked, the other calls."""
        roles, out = tempfile.mkdtemp(), out or tempfile.mkdtemp()
        os.makedirs(os.path.join(roles, "training"))
        manifest = {"contract": "red-role-files/v1", "timestamp_column": "timestamp", "channels": ["a", "b"],
                    "classes": ["normal", "leak"], "background": "normal", "training": []}
        for name, cls, rows in (("normal", "normal", 20), ("leak_inc01", "leak", 11)):
            pd.DataFrame({"timestamp": np.arange(rows) * 1.0, "a": np.arange(rows), "b": -np.arange(rows)}).to_csv(
                os.path.join(roles, "training", f"{name}.csv"), index=False)
            manifest["training"].append({"file": f"training/{name}.csv", "class": cls, "n_rows": rows})
        json.dump({"endpoint": "https://x", "blueprint": {"key": "bp", "values": {
            "window_size": 4, "step_size": 1}}}, open(os.path.join(out, "test_state.json"), "w"))
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
            library_embeddings.main(["--roles", roles, "--out", out, "--serving", "--step", "2", *args])
        return out, asked, calls, roles, manifest

    @staticmethod
    def rec(first):
        return {"predicted_state": "leak" if first > 5 else "normal", "invalid": False,
                "embedding_a": [first, 1.0, 0.0], "embedding_b": [0.0, first * 2, (first % 3) * 1.0]}

    def test_serving_route_writes_the_batch_shape_and_the_map(self):
        out, asked, _, roles, manifest = self.serve(self.rec, "--agent", "agt_app")
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

    def test_its_own_agent_is_deleted_however_the_run_ends(self):
        out, _, calls, _, _ = self.serve(self.rec)
        self.assertIn(("DELETE", "/instances/agt_s"), calls)
        self.assertIn(("DELETE", "/bundles/bnd_s"), calls)
        self.assertFalse(os.path.exists(os.path.join(out, "library_serving_state.json")))
        out2 = tempfile.mkdtemp()                         # a query fails: the agent is deleted all the same
        with self.assertRaises(RuntimeError):
            self.serve(self.rec, out=out2, failing_query=True)
        self.assertFalse(os.path.exists(os.path.join(out2, "library_serving_state.json")))

    def test_a_left_agent_another_blueprint_or_step_stops_it(self):
        out = tempfile.mkdtemp()
        json.dump({"bundle": "bnd_old", "agent": "agt_old"}, open(os.path.join(out, "library_serving_state.json"), "w"))
        with self.assertRaises(SystemExit) as e:
            self.serve(self.rec, out=out)
        self.assertIn("cleanup.py", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            self.serve(self.rec, "--agent", "agt_other", agent_bp="another-bp")
        self.assertIn("not the promoted bp", str(e.exception))
        for old in ({"blueprint": "old-bp", "bundle": "bnd_l"}, {"blueprint": "bp", "step": 4, "bundle": "bnd_l"}):
            out = tempfile.mkdtemp()
            json.dump(old, open(os.path.join(out, "library_state.json"), "w"))
            with self.assertRaises(SystemExit) as e:
                self.serve(self.rec, "--agent", "agt_app", out=out)
            self.assertIn("move them aside", str(e.exception))

    def test_an_agent_without_embeddings_is_refused(self):
        with self.assertRaises(SystemExit) as e:
            self.serve(lambda first: {"predicted_state": "normal"}, "--agent", "agt_app")
        self.assertIn("output_embeddings", str(e.exception))


if __name__ == "__main__":
    unittest.main()
