"""Network-free unit tests for the Omega reference scripts.

Mocks the official archetypeai client's transport, so these run with NO
credentials and NO network. They lock in the verified invariants:
  * embed() posts a `data.numeric_array` event with the channel-first window
    and model OmegaEncoder::omega_embeddings_1_4 (no file_ids, no prompt).
  * Both ATAI_API_KEY and ATAI_API_ENDPOINT are required — no default
    endpoint.
  * read_series drops the timestamp column (channels = sensors only).
  * the KNN vote and joint-feature shape behave as expected.

Run:
    python -m unittest discover -s skills/atai-newton-omega-model/tests
    # or, from this directory:
    python test_references.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REF = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REF))

import numpy as np  # noqa: E402

import _common  # noqa: E402
import classify_knn  # noqa: E402

from archetypeai.api_client import ArchetypeAI  # noqa: E402

SAMPLE = REF / "sample_data"


def _capturing_post(payload: dict, store: dict):
    """Return a fake ArchetypeAI.requests_post that records its args and returns `payload`."""

    def _requests_post(self, api_endpoint, data_payload, additional_headers={}):
        store.update(
            url=api_endpoint,
            json=json.loads(data_payload),
            headers={**self.auth_headers, **additional_headers},
        )
        return payload

    return _requests_post


class _Base(unittest.TestCase):
    def hermetic_env(self, **env: str) -> None:
        saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))
        os.environ.pop("ATAI_API_KEY", None)
        os.environ.pop("ATAI_API_ENDPOINT", None)
        os.environ.update(env)
        patcher = mock.patch.object(_common, "_try_load_dotenv", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def hermetic_client(self) -> ArchetypeAI:
        self.hermetic_env(
            ATAI_API_KEY="test-key", ATAI_API_ENDPOINT="https://example.test/v0.5"
        )
        return _common.make_client()


class TestMakeClient(_Base):
    def test_missing_key_exits(self):
        self.hermetic_env(ATAI_API_ENDPOINT="https://example.test/v0.5")
        with self.assertRaises(SystemExit):
            _common.make_client()

    def test_missing_endpoint_exits(self):
        # The endpoint is required just like the key — no silent default.
        self.hermetic_env(ATAI_API_KEY="test-key")
        with self.assertRaises(SystemExit):
            _common.make_client()

    def test_endpoint_normalized_and_headers(self):
        self.hermetic_env(ATAI_API_KEY="test-key", ATAI_API_ENDPOINT="https://example.test/")
        client = _common.make_client()
        self.assertTrue(client.api_endpoint.endswith("/v0.5"))
        self.assertEqual(client.auth_headers["Authorization"], "Bearer test-key")


class TestEmbedBody(_Base):
    def test_embed_request_shape(self):
        client = self.hermetic_client()
        window = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]  # 2 channels x 3 timesteps
        store: dict = {}
        payload = {"response": {"response": [[0.1] * 768, [0.2] * 768], "warning_messages": ["w"]}}
        with mock.patch.object(ArchetypeAI, "requests_post", _capturing_post(payload, store)):
            embeddings, warnings, _elapsed_ms = _common.embed(client, window, normalize_input=True)
        body = store["json"]
        self.assertEqual(body["model"], "OmegaEncoder::omega_embeddings_1_4")
        self.assertTrue(body["normalize_input"])
        self.assertEqual(body["events"][0]["type"], "data.numeric_array")
        self.assertEqual(body["events"][0]["event_data"]["contents"], window)  # channel-first, passed through
        self.assertNotIn("file_ids", body)
        self.assertNotIn("sanitize", body)
        self.assertTrue(store["url"].endswith("/query"))
        self.assertEqual(store["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 768)
        self.assertEqual(warnings, ["w"])


class TestExtractEmbeddings(unittest.TestCase):
    def test_nested(self):
        embeddings, warnings = _common.extract_embeddings(
            {"response": {"response": [[1.0]], "warning_messages": ["x"]}}
        )
        self.assertEqual(embeddings, [[1.0]])
        self.assertEqual(warnings, ["x"])

    def test_plain_list(self):
        embeddings, warnings = _common.extract_embeddings({"response": [[1.0]]})
        self.assertEqual(embeddings, [[1.0]])
        self.assertEqual(warnings, [])

    def test_empty(self):
        self.assertEqual(_common.extract_embeddings({}), ([], []))


class TestReadSeries(unittest.TestCase):
    def test_drops_timestamp_column(self):
        series = _common.read_series(SAMPLE / "bearing_healthy.csv")
        # CSV is timestamp + bearing_1..4 -> 4 channels, timestamp excluded
        self.assertEqual(len(series), 4)
        self.assertEqual(len(series[0]), 2000)

    def test_window_at_shape_and_bounds(self):
        series = _common.read_series(SAMPLE / "bearing_healthy.csv")
        window_values = _common.window_at(series, start=0, window=128)
        self.assertEqual(len(window_values), 4)
        self.assertEqual(len(window_values[0]), 128)
        with self.assertRaises(SystemExit):
            _common.window_at(series, start=1999, window=1024)


class TestWindowSizeThreading(_Base):
    def test_embed_start_sends_window_of_requested_length(self):
        # --window-size must control the actual window sliced and sent to the
        # API (for BOTH library and test windows); a regression here would
        # silently embed wrong-length windows.
        client = self.hermetic_client()
        store: dict = {}
        payload = {"response": {"response": [[0.1] * 8, [0.2] * 8], "warning_messages": []}}
        series = [[float(value) for value in range(200)], [float(value) for value in range(200)]]
        mean = np.zeros((2, 1))
        std = np.ones((2, 1))
        with mock.patch.object(ArchetypeAI, "requests_post", _capturing_post(payload, store)):
            start, joint_feature = classify_knn._embed_start(client, series, 0, mean, std, 64)
        contents = store["json"]["events"][0]["event_data"]["contents"]
        self.assertEqual(start, 0)
        self.assertEqual(len(contents), 2)  # channels preserved
        self.assertEqual(len(contents[0]), 64)  # exactly the requested window length
        self.assertEqual(joint_feature.shape, (16,))  # 2 channels x 8 dims, concatenated

    def test_embed_start_window_offset(self):
        client = self.hermetic_client()
        store: dict = {}
        payload = {"response": {"response": [[0.1] * 8, [0.2] * 8], "warning_messages": []}}
        series = [[float(value) for value in range(200)], [float(value) for value in range(200)]]
        mean = np.zeros((2, 1))
        std = np.ones((2, 1))
        with mock.patch.object(ArchetypeAI, "requests_post", _capturing_post(payload, store)):
            classify_knn._embed_start(client, series, 100, mean, std, 32)
        contents = store["json"]["events"][0]["event_data"]["contents"]
        self.assertEqual(len(contents[0]), 32)
        self.assertEqual(contents[0][0], 100.0)  # slice starts at the requested offset


class TestKnn(unittest.TestCase):
    def test_majority_vote(self):
        library = np.array([[0.0, 0.0], [0.1, 0.0], [9.0, 9.0]])
        labels = ["healthy", "healthy", "degraded"]
        self.assertEqual(
            classify_knn.knn_classify(np.array([0.05, 0.0]), library, labels, k_neighbors=3),
            "healthy",
        )
        self.assertEqual(
            classify_knn.knn_classify(np.array([9.1, 9.0]), library, labels, k_neighbors=1),
            "degraded",
        )


class TestScalerAndWindows(unittest.TestCase):
    def test_fit_scaler_per_channel(self):
        # 2 channels x 4 timesteps
        mean, std = classify_knn.fit_scaler([[0.0, 2.0, 4.0, 6.0], [10.0, 10.0, 10.0, 10.0]])
        self.assertEqual(mean.shape, (2, 1))
        self.assertAlmostEqual(float(mean[0, 0]), 3.0)
        self.assertAlmostEqual(float(mean[1, 0]), 10.0)
        self.assertGreater(float(std[0, 0]), 0)  # varying channel
        self.assertAlmostEqual(float(std[1, 0]), 1e-9, places=6)  # constant channel -> ~floor

    def test_contiguous_starts_skips_gap(self):
        # two contiguous blocks of 4, with a big jump between -> window=4 yields starts 0 and 4, not 2
        timestamps = [0, 1, 2, 3, 1000, 1001, 1002, 1003]
        self.assertEqual(classify_knn.contiguous_starts(timestamps, window=4), [0, 4])

    def test_read_series_and_time(self):
        timestamps, series = _common.read_series_and_time(SAMPLE / "bearing_healthy.csv")
        self.assertEqual(len(series), 4)  # timestamp dropped from channels
        self.assertEqual(len(timestamps), 2000)
        self.assertTrue(all(isinstance(timestamp, int) for timestamp in timestamps[:5]))

    def test_inference_file_has_no_label_column(self):
        # The shipped test input must not contain a label column (leak-proof).
        header = open(SAMPLE / "bearing_inference.csv").readline().strip().split(",")
        self.assertNotIn("label", [column_name.lower() for column_name in header])
        series = _common.read_series(SAMPLE / "bearing_inference.csv")
        self.assertEqual(len(series), 4)  # 4 bearing channels embedded, nothing else


class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        report = classify_knn.metrics(["healthy", "degraded"], ["healthy", "degraded"])
        self.assertEqual(
            (report["accuracy"], report["precision"], report["recall"], report["f1"]),
            (1.0, 1.0, 1.0, 1.0),
        )

    def test_false_positive_lowers_precision(self):
        # one healthy window mislabelled degraded -> recall stays 1, precision drops
        report = classify_knn.metrics(["degraded", "healthy"], ["degraded", "degraded"])
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["precision"], 0.5)
        self.assertEqual(report["tp"], 1)
        self.assertEqual(report["fp"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
