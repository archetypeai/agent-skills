"""Network-free unit tests for the reference scripts (python -m unittest / pytest)."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REFERENCES = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REFERENCES))

import _common  # noqa: E402
import explain_knn  # noqa: E402

SAMPLE = REFERENCES / "sample_data"


def _fake_response(payload: dict):
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
    return response


def _http_error(code: int, body: str = "{}"):
    return urllib.error.HTTPError("https://example", code, "err", {}, io.BytesIO(body.encode()))


class ChatBodyTest(unittest.TestCase):
    def test_reasoning_off_via_chat_template(self):
        body = _common.chat_body("sys", "user")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("/no_think", body["messages"][0]["content"])

    def test_model_override(self):
        with mock.patch.dict(os.environ, {"NEMOTRON_MODEL": "nvidia/other"}):
            self.assertEqual(_common.chat_body("s", "u")["model"], "nvidia/other")

    def test_default_model(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_common.nemotron_model(), _common.NEMOTRON_MODEL)


class NemotronChatTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test"})
        patcher.start()
        self.addCleanup(patcher.stop)
        sleep = mock.patch.object(_common.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    def test_returns_content_without_think_block(self):
        reply = {"choices": [{"message": {"content": "<think>hmm</think> [1]"}}]}
        with mock.patch.object(_common.urllib.request, "urlopen", return_value=_fake_response(reply)):
            self.assertEqual(_common.nemotron_chat("s", "u"), "[1]")

    def test_403_explains_key_type(self):
        with mock.patch.object(_common.urllib.request, "urlopen", side_effect=_http_error(403)):
            with self.assertRaises(_common.NemotronError) as ctx:
                _common.nemotron_chat("s", "u")
        self.assertEqual(ctx.exception.status, 403)
        self.assertIn("NGC", str(ctx.exception))

    def test_410_explains_retirement(self):
        with mock.patch.object(_common.urllib.request, "urlopen", side_effect=_http_error(410)):
            with self.assertRaises(_common.NemotronError) as ctx:
                _common.nemotron_chat("s", "u")
        self.assertIn("retired", str(ctx.exception))

    def test_retries_transient_then_succeeds(self):
        reply = {"choices": [{"message": {"content": "ok"}}]}
        effects = [_http_error(503), _fake_response(reply)]
        with mock.patch.object(_common.urllib.request, "urlopen", side_effect=effects) as urlopen:
            self.assertEqual(_common.nemotron_chat("s", "u"), "ok")
        self.assertEqual(urlopen.call_count, 2)

    def test_does_not_retry_auth_failure(self):
        with mock.patch.object(_common.urllib.request, "urlopen", side_effect=_http_error(403)) as urlopen:
            with self.assertRaises(_common.NemotronError):
                _common.nemotron_chat("s", "u")
        self.assertEqual(urlopen.call_count, 1)


class ParseAndValidateTest(unittest.TestCase):
    def test_parse_object_inside_fences(self):
        text = 'Here:\n```json\n{"observation": "a", "likely_cause": "b", "checks": ["c"]}\n```'
        self.assertEqual(_common.parse_json(text, dict)["observation"], "a")

    def test_parse_array(self):
        self.assertEqual(_common.parse_json('x [{"a": 1}] y', list), [{"a": 1}])

    def test_parse_rejects_missing_json(self):
        with self.assertRaises(ValueError):
            _common.parse_json("no json here", dict)

    def test_validate_brief_trims_checks(self):
        brief = _common.validate_brief(
            {"observation": " a ", "likely_cause": "b", "checks": ["1", "", "2", "3", "4"]}
        )
        self.assertEqual(brief, {"observation": "a", "likely_cause": "b", "checks": ["1", "2", "3"]})

    def test_validate_brief_rejects_incomplete(self):
        for bad in ({"observation": "a", "likely_cause": "b", "checks": []},
                    {"observation": "", "likely_cause": "b", "checks": ["c"]},
                    {"likely_cause": "b", "checks": ["c"]}):
            with self.assertRaises(ValueError):
                _common.validate_brief(bad)

    def test_nemotron_brief_end_to_end_with_stubbed_chat(self):
        reply = '{"observation": "rms up", "likely_cause": "wear", "checks": ["inspect"]}'
        with mock.patch.object(_common, "nemotron_chat", return_value=reply) as chat:
            brief = _common.nemotron_brief("ctx", "degraded", {"degraded": 3}, {"b1": {}}, {"b1": {}})
        self.assertEqual(brief["checks"], ["inspect"])
        sent = json.loads(chat.call_args.args[1])
        self.assertEqual(sent["classifier"], {"verdict": "degraded", "knn_votes": {"degraded": 3}})
        self.assertNotIn("embedding", json.dumps(sent).lower())


class StatsTest(unittest.TestCase):
    def test_channel_stats_known_values(self):
        stats = _common.channel_stats([1.0, -1.0, 1.0, -1.0])
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["rms"], 1.0)
        self.assertEqual(stats["peak"], 1.0)
        self.assertEqual(stats["crest_factor"], 1.0)

    def test_constant_channel_does_not_divide_by_zero(self):
        stats = _common.channel_stats([0.0] * 8)
        self.assertEqual(stats["kurtosis"], 0.0)
        self.assertEqual(stats["crest_factor"], 0.0)


class SampleDataTest(unittest.TestCase):
    def test_incoming_is_two_256_windows_with_four_channels(self):
        names, series = _common.read_series(SAMPLE / "bearing_incoming.csv")
        self.assertEqual(names, ["bearing_1", "bearing_2", "bearing_3", "bearing_4"])
        self.assertEqual(len(series[0]), 2 * explain_knn.WINDOW)

    def test_incoming_is_disjoint_from_shot_files(self):
        def timestamps(path):
            with open(path) as handle:
                next(handle)
                return {line.split(",")[0] for line in handle if line.strip()}

        incoming = timestamps(SAMPLE / "bearing_incoming.csv")
        for shot in ("bearing_healthy.csv", "bearing_degraded.csv"):
            self.assertFalse(incoming & timestamps(SAMPLE / shot), shot)


class KnnTest(unittest.TestCase):
    def test_majority_vote(self):
        import numpy as np

        library = [(np.array([0.0, 0.0]), "healthy"), (np.array([0.1, 0.0]), "healthy"),
                   (np.array([5.0, 5.0]), "degraded")]
        label, votes = explain_knn.knn(np.array([0.05, 0.0]), library, k=3)
        self.assertEqual(label, "healthy")
        self.assertEqual(votes, {"healthy": 2, "degraded": 1})


if __name__ == "__main__":
    unittest.main()
