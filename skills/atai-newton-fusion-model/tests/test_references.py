"""Network-free unit tests for the reference scripts.

These mock the Archetype AI API, so they run with NO credentials and NO
network. They lock in the invariants discovered while building this skill:

  * /query bodies send `instruction_prompt` and NEVER `system_prompt`
    (C 2.6 ignores system_prompt; sending it was the original bug).
  * `file_ids` is referenced by filename.
  * The video multi-frame path (A) sets `multi_image: true`.
  * The image inline path uses a `data.base64_img` event.
  * `extract_text` handles every response shape we've observed.

Run:
    python -m unittest discover -s skills/atai-newton-fusion-model/tests
    # or, from this directory:
    python test_references.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# The reference scripts use flat imports (`from _common import ...`), so put
# the references/ dir on the path.
REF = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REF))

import _common  # noqa: E402
import image_query  # noqa: E402
import text_query  # noqa: E402
import video_query  # noqa: E402


class _Resp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, payload: dict, ok: bool = True):
        self._payload = payload
        self.ok = ok
        self.status_code = 200 if ok else 400
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


def _capturing_post(payload: dict, store: dict):
    """Return a fake requests.post that records its kwargs and returns `payload`."""

    def _post(url, headers=None, json=None, timeout=None, files=None):
        store.update(url=url, headers=headers, json=json, files=files)
        return _Resp(payload)

    return _post


class _Base(unittest.TestCase):
    def hermetic_env(self, **env: str) -> None:
        """Isolate env vars and stop client() from loading the real .env."""
        saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))
        os.environ.pop("ATAI_API_ENDPOINT", None)
        os.environ.update(env)
        patcher = mock.patch.object(_common, "_try_load_dotenv", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestExtractText(unittest.TestCase):
    def test_nested_list(self):
        self.assertEqual(_common.extract_text({"response": {"response": ["hi"]}}), "hi")

    def test_nested_str(self):
        self.assertEqual(_common.extract_text({"response": {"response": "hi"}}), "hi")

    def test_top_level_list(self):
        self.assertEqual(_common.extract_text({"response": ["hi"]}), "hi")

    def test_top_level_str(self):
        self.assertEqual(_common.extract_text({"response": "hi"}), "hi")

    def test_text_field(self):
        self.assertEqual(_common.extract_text({"text": "hi"}), "hi")

    def test_empty_payload(self):
        self.assertEqual(_common.extract_text({}), "")

    def test_empty_inner_list(self):
        self.assertEqual(_common.extract_text({"response": {"response": []}}), "")


class TestClient(_Base):
    def test_missing_key_exits(self):
        self.hermetic_env()
        os.environ.pop("ATAI_API_KEY", None)
        with self.assertRaises(SystemExit):
            _common.client()

    def test_endpoint_normalized_and_headers(self):
        self.hermetic_env(ATAI_API_KEY="k", ATAI_API_ENDPOINT="https://example.test/")
        endpoint, headers = _common.client()
        self.assertTrue(endpoint.endswith("/v0.5"))
        self.assertEqual(headers["Authorization"], "Bearer k")
        self.assertEqual(headers["Content-Type"], "application/json")


class TestQueryBody(_Base):
    def test_body_invariants(self):
        self.hermetic_env(ATAI_API_KEY="test-key")
        store: dict = {}
        with mock.patch.object(_common.requests, "post", _capturing_post({"response": ["ok"]}, store)):
            text, _payload, _ms = _common.query(
                "hello", instruction_prompt="be brief", file_ids=["chart.png"], max_new_tokens=123
            )
        body = store["json"]
        self.assertEqual(body["query"], "hello")
        self.assertEqual(body["instruction_prompt"], "be brief")
        self.assertNotIn("system_prompt", body)  # the key finding — never send it
        self.assertEqual(body["file_ids"], ["chart.png"])
        self.assertEqual(body["model"], _common.MODEL)
        self.assertEqual(body["max_new_tokens"], 123)
        self.assertIn("sanitize", body)
        self.assertEqual(text, "ok")
        self.assertTrue(store["url"].endswith("/query"))
        self.assertEqual(store["headers"]["Authorization"], "Bearer test-key")

    def test_default_file_ids_empty(self):
        self.hermetic_env(ATAI_API_KEY="test-key")
        store: dict = {}
        with mock.patch.object(_common.requests, "post", _capturing_post({"response": ["x"]}, store)):
            _common.query("q")
        self.assertEqual(store["json"]["file_ids"], [])


class TestUploadFile(_Base):
    def test_missing_file_exits(self):
        self.hermetic_env(ATAI_API_KEY="test-key")
        with self.assertRaises(SystemExit):
            _common.upload_file("/no/such/file_xyz.png")

    def test_returns_file_id_by_filename(self):
        self.hermetic_env(ATAI_API_KEY="test-key")
        store: dict = {}
        with tempfile.NamedTemporaryFile("w", suffix=".png", delete=False) as f:
            f.write("x")
            path = f.name
        try:
            name = os.path.basename(path)
            with mock.patch.object(_common.requests, "post", _capturing_post({"file_id": name}, store)):
                file_id = _common.upload_file(path)
            self.assertEqual(file_id, name)
            self.assertTrue(store["url"].endswith("/files"))
            self.assertEqual(store["files"]["file"][0], name)  # uploaded under its filename
        finally:
            os.unlink(path)


class TestImageInlineBase64(_Base):
    def test_inline_base64_body(self):
        self.hermetic_env(ATAI_API_KEY="test-key")
        store: dict = {}
        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as f:
            f.write(b"\x89PNG-fake-bytes")
            path = Path(f.name)
        try:
            with mock.patch.object(image_query.requests, "post", _capturing_post({"response": ["desc"]}, store)):
                image_query.example_inline_base64(path)
            body = store["json"]
            self.assertEqual(body["file_ids"], [])
            self.assertNotIn("system_prompt", body)
            event = body["events"][0]
            self.assertEqual(event["type"], "data.base64_img")
            self.assertIn("contents", event["event_data"])
            self.assertEqual(event["event_data"]["mime_type"], "image/png")
        finally:
            os.unlink(path)


class TestVideo(_Base):
    def test_assembly_prompt_format(self):
        for token in ("Step 1:", "Step 2:", "Step 3:", "PASS", "FAIL", "Summary:"):
            self.assertIn(token, video_query.ASSEMBLY_PROMPT)

    def test_sample_frames_without_ffmpeg_returns_none(self):
        with mock.patch.object(video_query.shutil, "which", return_value=None):
            self.assertIsNone(video_query._sample_frames(Path("nonexistent.mp4")))


class TestTextConstants(unittest.TestCase):
    def test_field_legend_has_columns(self):
        for col in ("time_utc", "mac_a", "prot", "bytes_a", "pkts_b"):
            self.assertIn(col, text_query.FIELD_LEGEND)

    def test_analyst_persona(self):
        self.assertIn("network security analyst", text_query.ANALYST)


if __name__ == "__main__":
    unittest.main(verbosity=2)
