"""Network-free unit tests for the Omega reference scripts.

Mocks the Archetype AI API, so these run with NO credentials and NO network.
They lock in the verified invariants:
  * embed() posts a `data.numeric_array` event with the channel-first window
    and model OmegaEncoder::omega_embeddings_1_4 (no file_ids, no prompt).
  * read_series drops the timestamp column (channels = sensors only).
  * the KNN vote and joint-feature shape behave as expected.

Run:
    python -m unittest discover -s skills/atai-newton-omega-model/tests
    # or, from this directory:
    python test_references.py
"""

from __future__ import annotations

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

SAMPLE = REF / "sample_data"


class _Resp:
    def __init__(self, payload: dict, ok: bool = True):
        self._payload = payload
        self.ok = ok
        self.status_code = 200 if ok else 400
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


def _capturing_post(payload: dict, store: dict):
    def _post(url, headers=None, json=None, timeout=None):
        store.update(url=url, headers=headers, json=json)
        return _Resp(payload)

    return _post


class _Base(unittest.TestCase):
    def hermetic_env(self, **env: str) -> None:
        saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))
        os.environ.pop("ATAI_API_ENDPOINT", None)
        os.environ.update(env)
        p = mock.patch.object(_common, "_try_load_dotenv", lambda: None)
        p.start()
        self.addCleanup(p.stop)


class TestEmbedBody(_Base):
    def test_embed_request_shape(self):
        self.hermetic_env(ATAI_API_KEY="test-key")
        window = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]  # 2 channels x 3 timesteps
        store: dict = {}
        resp = {"response": {"response": [[0.1] * 768, [0.2] * 768], "warning_messages": ["w"]}}
        with mock.patch.object(_common.requests, "post", _capturing_post(resp, store)):
            embeddings, warnings, _ms = _common.embed(window, normalize_input=True)
        body = store["json"]
        self.assertEqual(body["model"], "OmegaEncoder::omega_embeddings_1_4")
        self.assertTrue(body["normalize_input"])
        self.assertEqual(body["events"][0]["type"], "data.numeric_array")
        self.assertEqual(body["events"][0]["event_data"]["contents"], window)  # channel-first, passed through
        self.assertNotIn("file_ids", body)
        self.assertTrue(store["url"].endswith("/query"))
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 768)
        self.assertEqual(warnings, ["w"])


class TestExtractEmbeddings(unittest.TestCase):
    def test_nested(self):
        emb, warn = _common.extract_embeddings({"response": {"response": [[1.0]], "warning_messages": ["x"]}})
        self.assertEqual(emb, [[1.0]])
        self.assertEqual(warn, ["x"])

    def test_plain_list(self):
        emb, warn = _common.extract_embeddings({"response": [[1.0]]})
        self.assertEqual(emb, [[1.0]])
        self.assertEqual(warn, [])

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
        w = _common.window_at(series, start=0, window=128)
        self.assertEqual(len(w), 4)
        self.assertEqual(len(w[0]), 128)
        with self.assertRaises(SystemExit):
            _common.window_at(series, start=1999, window=1024)


class TestKnn(unittest.TestCase):
    def test_majority_vote(self):
        lib = np.array([[0.0, 0.0], [0.1, 0.0], [9.0, 9.0]])
        labels = ["healthy", "healthy", "degraded"]
        self.assertEqual(classify_knn.knn_classify(np.array([0.05, 0.0]), lib, labels, k=3), "healthy")
        self.assertEqual(classify_knn.knn_classify(np.array([9.1, 9.0]), lib, labels, k=1), "degraded")


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
        ts = [0, 1, 2, 3, 1000, 1001, 1002, 1003]
        self.assertEqual(classify_knn.contiguous_starts(ts, window=4), [0, 4])

    def test_read_series_and_time(self):
        ts, series = _common.read_series_and_time(SAMPLE / "bearing_healthy.csv")
        self.assertEqual(len(series), 4)  # timestamp dropped from channels
        self.assertEqual(len(ts), 2000)
        self.assertTrue(all(isinstance(t, int) for t in ts[:5]))

    def test_inference_file_has_no_label_column(self):
        # The shipped test input must not contain a label column (leak-proof).
        header = open(SAMPLE / "bearing_inference.csv").readline().strip().split(",")
        self.assertNotIn("label", [h.lower() for h in header])
        series = _common.read_series(SAMPLE / "bearing_inference.csv")
        self.assertEqual(len(series), 4)  # 4 bearing channels embedded, nothing else


class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        m = classify_knn.metrics(["healthy", "degraded"], ["healthy", "degraded"])
        self.assertEqual((m["accuracy"], m["precision"], m["recall"], m["f1"]), (1.0, 1.0, 1.0, 1.0))

    def test_false_positive_lowers_precision(self):
        # one healthy window mislabelled degraded -> recall stays 1, precision drops
        m = classify_knn.metrics(["degraded", "healthy"], ["degraded", "degraded"])
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
