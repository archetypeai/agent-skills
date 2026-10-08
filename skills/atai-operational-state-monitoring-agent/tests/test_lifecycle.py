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



class TestNewerBlueprint(unittest.TestCase):
    """The osm blueprint on dev since 2026-10-05 takes `states` as a value and reports every
    trial setting under `values`; older blueprints don't. Both must work."""

    def test_trial_setting_reads_both_formats(self):
        newer = {"trial_values": {"values": {"window_size": 512, "step_size": 512, "k_neighbors": 5,
                                             "metric": "cosine", "weights": "uniform"}, "models": {}}}
        self.assertEqual(atai_http.trial_setting(trial("a", 0.8)), atai_http.trial_setting(newer))

    def test_states_sent_only_when_the_blueprint_asks(self):
        training = [{"ground_truth": {"state": {"from": {"constant": s}}}} for s in ("drain", "fill", "spin", "wash")]
        self.assertEqual(atai_http.states_override({"document": {"values": {"k_neighbors": 5}}}, training), {})
        self.assertEqual(atai_http.states_override({"document": {"values": {"states": "${required}"}}}, training),
                         {"overrides": {"values": {"states": ["drain", "fill", "spin", "wash"]}}})

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


    def test_incomplete_flags_outputs_that_end_early(self):
        state = {"files": {n: {"last_ms": 1_000_000} for n in ("a.csv", "b.csv", "c.csv")}}
        per_file = {"a.csv": [["998.0"]], "b.csv": [["500.0"]]}  # b ends 500 s early, c has none
        got = deliver.incomplete(["a.csv", "b.csv", "c.csv"], state, per_file, ["finish_timestamp"])
        self.assertEqual(got, ["b.csv", "c.csv"])

    def test_run_is_completed_only_once_its_outputs_are(self):
        calls = {"status": 0, "results": 0}

        def request(method, url, **kw):
            calls["status"] += 1
            return {"status": "completed"}

        def list_all(url):
            calls["results"] += 1
            return [{"data": {"filename": "out"}}]

        def fetch(name):        # the platform finishes writing the output on the 3rd download
            return f"finish_timestamp,predicted_state\n110.0,y\n{299.0 if calls['results'] >= 3 else 150.0},y\n"

        state = {"runs": [{"agent": "agt_x", "files": ["b.csv"]}],
                 "files": {"b.csv": {"first_ms": 100_001, "last_ms": 300_000}}}
        with tempfile.TemporaryDirectory() as out, \
                mock.patch.multiple(deliver, request=request, list_all=list_all, fetch=fetch, agents=lambda: "x"), \
                mock.patch.object(deliver.time, "sleep"), mock.patch.object(deliver, "log"):
            os.makedirs(os.path.join(out, "delivery"))
            deliver.collect(mock.Mock(out=out), state, os.path.join(out, "delivery", "runs.json"))
        self.assertEqual(calls["status"], 1)
        self.assertEqual((state["runs"][0]["status"], state["runs"][0]["error"]), ("completed", None))
        self.assertEqual(calls["results"], 4)     # 3 checks, then the final collection

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

    def test_check_auth_fails_fast_and_blames_an_exported_key(self):
        # a key for another deployment, exported in the shell, beats .env; a rejected large
        # upload only looks like "Broken pipe", so the check runs before any upload
        with tempfile.TemporaryDirectory() as d:
            env = os.path.join(d, ".env")
            open(env, "w").write("ATAI_API_KEY=sk_from_dotenv\nATAI_API_ENDPOINT=https://x.test\n")
            denied = urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b""))
            with mock.patch.dict(os.environ, {"ATAI_API_KEY": "sk_from_shell"}, clear=False):
                os.environ.pop("ATAI_API_ENDPOINT", None)
                atai_http.SOURCE.clear()
                atai_http.load_dotenv(env)
                self.assertEqual(os.environ["ATAI_API_KEY"], "sk_from_shell")         # the shell wins
                self.assertEqual(atai_http.SOURCE["ATAI_API_KEY"], "shell")
                self.assertEqual(os.environ["ATAI_API_ENDPOINT"], "https://x.test")   # filled from .env
                with mock.patch.object(atai_http.urllib.request, "urlopen", side_effect=denied):
                    with self.assertRaises(SystemExit) as cm:
                        atai_http.check_auth(log=lambda m: None)
                self.assertIn("unset ATAI_API_KEY", str(cm.exception))

    def test_an_empty_exported_variable_does_not_hide_dotenv(self):
        with tempfile.TemporaryDirectory() as d:
            env = os.path.join(d, ".env")
            open(env, "w").write("ATAI_API_KEY=sk_x\nATAI_API_ENDPOINT=https://from-dotenv.test\n")
            with mock.patch.dict(os.environ, {"ATAI_API_ENDPOINT": "", "ATAI_API_KEY": ""}, clear=False):
                atai_http.SOURCE.clear()
                atai_http.load_dotenv(env)
                self.assertEqual(os.environ["ATAI_API_ENDPOINT"], "https://from-dotenv.test")
                self.assertEqual(atai_http.SOURCE["ATAI_API_ENDPOINT"], env)

    def test_dotenv_is_found_upwards_and_the_nearest_wins(self):
        import run_osm_agent
        with tempfile.TemporaryDirectory() as root:
            deep = os.path.join(root, "skills", "x", "references", "osm_lifecycle")
            os.makedirs(deep)
            open(os.path.join(root, ".env"), "w").write("ATAI_API_ENDPOINT=https://root.test\n")
            self.assertEqual(atai_http.find_dotenv(deep), os.path.join(root, ".env"))   # the repo root's
            open(os.path.join(root, "skills", ".env"), "w").write("ATAI_API_ENDPOINT=https://near.test\n")
            self.assertEqual(atai_http.find_dotenv(deep), os.path.join(root, "skills", ".env"))  # nearer wins
            cwd = os.getcwd()
            try:
                os.chdir(deep)
                self.assertEqual(os.path.realpath(run_osm_agent.find_dotenv()),
                                 os.path.realpath(os.path.join(root, "skills", ".env")))
            finally:
                os.chdir(cwd)

    def test_report_f1(self):
        rep = {"targets": {"state": {"class_names": ["drain", "wash"], "confusion_matrix": [[3, 1], [0, 6]]}}}
        f1, n = atai_http.report_f1(rep)
        self.assertEqual(n, 10)
        self.assertAlmostEqual(f1["drain"], 6 / 7)


class TestSampleCommandsMatchThePrepSkill(unittest.TestCase):
    """Path 2's Step 0 repeats the prep skill's commands for its sample; they must not drift."""

    def test_step_0_commands_are_the_prep_skills(self):
        skills = Path(__file__).resolve().parent.parent.parent
        osm = (skills / "atai-operational-state-monitoring-agent" / "SKILL.md").read_text()
        prep = (skills / "atai-operational-state-monitoring-agent-data-prep" / "SKILL.md").read_text()
        block = osm.split("**To try Path 2 on the prep skill's sample**", 1)[1].split("```sh", 1)[1].split("```", 1)[0]
        commands = [line.split("#", 1)[0].strip() for line in block.splitlines() if line.startswith("python ")]
        self.assertEqual(len(commands), 4)
        prep_lines = {line.split("#", 1)[0].strip() for line in prep.splitlines()}
        for command in commands:
            self.assertIn(command, prep_lines, f"not in the prep skill's SKILL.md: {command}")


if __name__ == "__main__":
    unittest.main()
