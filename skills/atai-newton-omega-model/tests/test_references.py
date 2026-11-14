"""Network-free unit tests for the Omega reference scripts.

Mocks the official archetypeai client's transport, so these run with NO
credentials and NO network. They lock in the verified invariants:
  * OmegaSession opens a Model Serving session with window_size/step_size,
    sends ALL channels of a window in one `data.numeric_array` query, splits
    the flat [channels x 768] row back into per-channel vectors, reopens the
    session on HTTP 409 session_not_running, and cancels it on exit.
  * Wide windows are split into requests of MAX_CHANNELS_PER_REQUEST channels.
  * success=false (input shorter than the window) raises.
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

from archetypeai.api_client import ApiError, ArchetypeAI  # noqa: E402

SAMPLE = REF / "sample_data"


def _fake_serving(store: dict, dims: int = 768, ended_once: bool = False, success: bool = True):
    """Return a fake ArchetypeAI.requests_post that plays Model Serving.

    Records every call in `store["calls"]` as (url, body-or-None). Session
    opens return ids s1, s2, ...; queries return one flat row of
    `channels * dims` values. With `ended_once`, the first query fails with
    HTTP 409 session_not_running, like an expired session.
    """
    state = {"opened": 0, "ended": ended_once}

    def _requests_post(self, api_endpoint, data_payload, additional_headers={}):
        body = json.loads(data_payload) if data_payload else None
        store.setdefault("calls", []).append((api_endpoint, body))
        store["headers"] = {**self.auth_headers, **additional_headers}
        if api_endpoint.endswith("/inference/sessions"):
            state["opened"] += 1
            return {"id": f"s{state['opened']}", "values": body["values"]}
        if api_endpoint.endswith("/cancel"):
            return {}
        if state["ended"] and "/sessions/" in api_endpoint:
            state["ended"] = False
            raise ApiError({"errors": [{"code": "session_not_running"}]})
        channels = body["events"][0]["event_data"]["contents"]
        row = [float(index) for index in range(len(channels) * dims)]
        return {"response": {"success": success, "response": [row] if success else []}}

    return _requests_post


def _queries(store: dict) -> list:
    return [(url, body) for url, body in store["calls"] if url.endswith("/query")]


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
        """Any .env convention reduces to one root with /api appended.

        Both surfaces live under /api, so one client serves query and files
        alike. Accepting every form lets a single .env serve the model skills
        and the agent skills.
        """
        for given in ("https://example.test", "https://example.test/",
                      "https://example.test/v0.5", "https://example.test/api",
                      "https://example.test/api/lens"):
            self.hermetic_env(ATAI_API_KEY="test-key", ATAI_API_ENDPOINT=given)
            client = _common.make_client()
            self.assertEqual(client.api_endpoint, "https://example.test/api", given)
        self.assertEqual(client.auth_headers["Authorization"], "Bearer test-key")


class TestOmegaSession(_Base):
    def test_session_lifecycle_and_body(self):
        client = self.hermetic_client()
        window = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]  # 2 channels x 3 timesteps
        store: dict = {}
        with mock.patch.object(ArchetypeAI, "requests_post", _fake_serving(store)):
            with _common.OmegaSession(client, window_size=64) as session:
                vectors = session.embed(window)
        open_url, open_body = store["calls"][0]
        self.assertEqual(open_url, "https://example.test/api/inference/sessions")
        self.assertEqual(open_body, {"model": "omega-embedding",
                                     "values": {"window_size": 64, "step_size": 64}})
        queries = _queries(store)
        self.assertEqual(len(queries), 1)  # all channels in ONE request
        url, body = queries[0]
        self.assertEqual(url, "https://example.test/api/inference/sessions/s1/query")
        self.assertEqual(body["model"], "omega-embedding")
        self.assertEqual(body["events"][0]["type"], "data.numeric_array")
        self.assertEqual(body["events"][0]["event_data"]["contents"], window)
        self.assertNotIn("normalize_input", body)
        self.assertTrue(store["calls"][-1][0].endswith("/inference/sessions/s1/cancel"))
        self.assertEqual(store["headers"]["Authorization"], "Bearer test-key")
        # The flat [2 x 768] row is split back into per-channel vectors.
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0][0], 0.0)
        self.assertEqual(vectors[1][0], 768.0)

    def test_reopens_ended_session(self):
        client = self.hermetic_client()
        store: dict = {}
        with mock.patch.object(ArchetypeAI, "requests_post", _fake_serving(store, ended_once=True)):
            with _common.OmegaSession(client, window_size=32) as session:
                vectors = session.embed([[0.0] * 32])
        self.assertEqual(len(vectors), 1)
        urls = [url for url, _ in store["calls"]]
        self.assertTrue(urls[-2].endswith("/sessions/s2/query"))  # retried on the new session
        self.assertTrue(urls[-1].endswith("/sessions/s2/cancel"))

    def test_wide_window_split_into_chunks(self):
        client = self.hermetic_client()
        store: dict = {}
        channel_count = _common.MAX_CHANNELS_PER_REQUEST + 6
        window = [[float(channel)] * 16 for channel in range(channel_count)]
        with mock.patch.object(ArchetypeAI, "requests_post", _fake_serving(store, dims=4)), \
                mock.patch.object(_common, "EMBED_DIM", 4):
            with _common.OmegaSession(client, window_size=16) as session:
                vectors = session.embed(window)
        sizes = [len(body["events"][0]["event_data"]["contents"]) for _, body in _queries(store)]
        self.assertEqual(sizes, [_common.MAX_CHANNELS_PER_REQUEST, 6])
        self.assertEqual(len(vectors), channel_count)

    def test_short_input_raises(self):
        client = self.hermetic_client()
        store: dict = {}
        with mock.patch.object(ArchetypeAI, "requests_post", _fake_serving(store, success=False)):
            with _common.OmegaSession(client, window_size=256) as session:
                with self.assertRaises(RuntimeError):
                    session.embed([[0.0] * 100])

    def test_window_size_range(self):
        client = self.hermetic_client()
        for bad in (8, 2048):
            with self.assertRaises(ValueError):
                _common.OmegaSession(client, window_size=bad)

    def test_embed_once_uses_inference_query(self):
        client = self.hermetic_client()
        store: dict = {}
        with mock.patch.object(ArchetypeAI, "requests_post", _fake_serving(store)):
            windows = _common.embed_once(client, [[0.0] * 1024])
        self.assertEqual([url for url, _ in store["calls"]],
                         ["https://example.test/api/inference/query"])
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0][0]), 768)


class TestReadSeries(unittest.TestCase):
    def test_drops_timestamp_column(self):
        series = _common.read_series(SAMPLE / "bearing_healthy.csv")
        # CSV is timestamp + bearing_1..4 -> 4 channels, timestamp excluded
        self.assertEqual(len(series), 4)
        self.assertEqual(len(series[0]), 2000)

    def test_window_at_shape_and_bounds(self):
        series = _common.read_series(SAMPLE / "bearing_healthy.csv")
        window_values = _common.window_at(series, start=0, window_size=128)
        self.assertEqual(len(window_values), 4)
        self.assertEqual(len(window_values[0]), 128)
        with self.assertRaises(SystemExit):
            _common.window_at(series, start=1999, window_size=1024)


class TestWindowSizeThreading(_Base):
    def _session(self, store, dims=8):
        client = self.hermetic_client()
        patchers = [mock.patch.object(ArchetypeAI, "requests_post", _fake_serving(store, dims=dims)),
                    mock.patch.object(_common, "EMBED_DIM", dims)]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return _common.OmegaSession(client, window_size=64)

    def test_embed_start_sends_window_of_requested_length(self):
        # --window-size must control the actual window sliced and sent to the
        # API (for BOTH library and test windows); a regression here would
        # silently embed wrong-length windows.
        store: dict = {}
        session = self._session(store)
        series = [[float(value) for value in range(200)], [float(value) for value in range(200)]]
        start, joint_feature = classify_knn._embed_start(
            session, series, 0, np.zeros((2, 1)), np.ones((2, 1)), 64)
        self.assertEqual(start, 0)
        queries = _queries(store)
        self.assertEqual(len(queries), 1)  # all channels in one request
        contents = queries[0][1]["events"][0]["event_data"]["contents"]
        self.assertEqual(len(contents), 2)
        self.assertEqual(len(contents[0]), 64)  # exactly the requested window length
        self.assertEqual(joint_feature.shape, (16,))  # 2 channels x 8 dims, concatenated

    def test_embed_start_window_offset(self):
        store: dict = {}
        session = self._session(store)
        series = [[float(value) for value in range(200)], [float(value) for value in range(200)]]
        classify_knn._embed_start(session, series, 100, np.zeros((2, 1)), np.ones((2, 1)), 32)
        contents = _queries(store)[0][1]["events"][0]["event_data"]["contents"]
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
        # two contiguous blocks of 4, with a big jump between -> window_size=4 yields starts 0 and 4, not 2
        timestamps = [0, 1, 2, 3, 1000, 1001, 1002, 1003]
        self.assertEqual(classify_knn.contiguous_starts(timestamps, window_size=4), [0, 4])

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


class TestComparisonTable(unittest.TestCase):
    def test_formats_one_row_per_run(self):
        rows = [
            {"window_size": 16, "n_library": 16, "n_windows": 1000,
             "accuracy": 0.91, "precision": 0.88, "recall": 1.0, "f1": 0.936},
            {"window_size": 1024, "n_library": 8, "n_windows": 1000,
             "accuracy": 0.946, "precision": 0.903, "recall": 1.0, "f1": 0.949},
        ]
        table = classify_knn.comparison_table(rows)
        lines = table.splitlines()
        self.assertEqual(len(lines), 4)  # header + rule + 2 rows
        self.assertIn("window", lines[0])
        self.assertIn("f1", lines[0])
        self.assertIn("16", lines[2])
        self.assertIn("0.936", lines[2])
        self.assertIn("1024", lines[3])
        self.assertIn("0.949", lines[3])

    def test_print_report_returns_metrics(self):
        report = classify_knn.print_report(["degraded", "healthy"], ["degraded", "healthy"])
        self.assertEqual(report["f1"], 1.0)
        self.assertEqual(report["accuracy"], 1.0)


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
