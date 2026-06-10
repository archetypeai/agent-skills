"""Network-free unit tests for the reference scripts.

These mock the official archetypeai client's transport, so they run with NO
credentials and NO network. They lock in the invariants discovered while
building this skill:

  * /query bodies send `instruction_prompt` and NEVER `system_prompt`
    (C 2.6 ignores system_prompt; sending it was the original bug).
  * /query bodies never send `sanitize`.
  * Both ATAI_API_KEY and ATAI_API_ENDPOINT are required — no default
    endpoint.
  * `file_ids` is referenced by filename.
  * The .mp4 video path uses `max_frames` and never sets `multi_image`.
  * The frame-list video path sets `multi_image: false` AND carries the
    `query_metadata` triple (raw_fps / frames_indices / total_num_frames) —
    without query_metadata that shape 400s.
  * Multi-image (independent images) bodies set `multi_image: true`;
    single-image bodies omit the key entirely.
  * `extract_text` handles every response shape we've observed.

Run:
    python -m unittest discover -s skills/atai-newton-fusion-model/tests
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

# The reference scripts use flat imports (`from _common import ...`), so put
# the references/ dir on the path.
REF = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REF))

import _common  # noqa: E402
import image_query  # noqa: E402
import text_query  # noqa: E402
import video_query  # noqa: E402

from archetypeai.api_client import ArchetypeAI  # noqa: E402


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
        """Isolate env vars and stop make_client() from loading the real .env."""
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


class TestQueryBody(_Base):
    def test_body_invariants(self):
        client = self.hermetic_client()
        store: dict = {}
        with mock.patch.object(ArchetypeAI, "requests_post", _capturing_post({"response": ["ok"]}, store)):
            text, _payload, _elapsed_ms = _common.query(
                client, "hello", instruction_prompt="be brief", file_ids=["chart.png"], max_new_tokens=123
            )
        body = store["json"]
        self.assertEqual(body["query"], "hello")
        self.assertEqual(body["instruction_prompt"], "be brief")
        self.assertNotIn("system_prompt", body)  # the key finding — never send it
        self.assertNotIn("sanitize", body)  # internal flag — never leak it
        self.assertEqual(body["file_ids"], ["chart.png"])
        self.assertEqual(body["model"], _common.MODEL)
        self.assertEqual(body["max_new_tokens"], 123)
        self.assertEqual(text, "ok")
        self.assertTrue(store["url"].endswith("/query"))
        self.assertEqual(store["headers"]["Authorization"], "Bearer test-key")

    def test_default_file_ids_empty(self):
        client = self.hermetic_client()
        store: dict = {}
        with mock.patch.object(ArchetypeAI, "requests_post", _capturing_post({"response": ["x"]}, store)):
            _common.query(client, "q")
        self.assertEqual(store["json"]["file_ids"], [])

    def test_multi_image_flag(self):
        client = self.hermetic_client()
        store: dict = {}
        with mock.patch.object(ArchetypeAI, "requests_post", _capturing_post({"response": ["x"]}, store)):
            _common.query(client, "q", file_ids=["a.png", "b.png"], multi_image=True)
        self.assertIs(store["json"]["multi_image"], True)

    def test_multi_image_omitted_by_default(self):
        client = self.hermetic_client()
        store: dict = {}
        with mock.patch.object(ArchetypeAI, "requests_post", _capturing_post({"response": ["x"]}, store)):
            _common.query(client, "q", file_ids=["a.png"])
        self.assertNotIn("multi_image", store["json"])


class TestUploadFile(_Base):
    def test_missing_file_exits(self):
        client = self.hermetic_client()
        with self.assertRaises(SystemExit):
            _common.upload_file(client, "/no/such/file_xyz.png")

    def test_returns_file_id_by_filename(self):
        client = self.hermetic_client()
        store: dict = {}

        def fake_upload(filename):
            store["filename"] = filename
            return {"file_id": os.path.basename(filename), "file_uid": "fil_123"}

        with mock.patch.object(client.files.local, "upload", fake_upload):
            file_id = _common.upload_file(client, Path(__file__))
        self.assertEqual(file_id, Path(__file__).name)  # referenced by filename, not fil_... uid
        self.assertEqual(store["filename"], str(Path(__file__)))


class TestImageInlineBase64(_Base):
    def test_inline_base64_body(self):
        client = self.hermetic_client()
        store: dict = {}
        sample_image = REF / "sample_assets" / "wind-turbines.png"
        with mock.patch.object(ArchetypeAI, "requests_post", _capturing_post({"response": ["desc"]}, store)):
            image_query.example_inline_base64(client, sample_image)
        body = store["json"]
        self.assertEqual(body["file_ids"], [])
        self.assertNotIn("system_prompt", body)
        self.assertNotIn("sanitize", body)
        event = body["events"][0]
        self.assertEqual(event["type"], "data.base64_img")
        self.assertIn("contents", event["event_data"])


class TestMultiImage(_Base):
    def test_body_sets_multi_image_true(self):
        client = self.hermetic_client()
        store: dict = {}
        with mock.patch.object(image_query, "upload_file", lambda client, path: Path(path).name), \
                mock.patch.object(ArchetypeAI, "requests_post", _capturing_post({"response": ["ok"]}, store)):
            image_query.example_multi_image(client, Path("before.png"), Path("after.png"))
        body = store["json"]
        self.assertEqual(body["file_ids"], ["before.png", "after.png"])
        self.assertIs(body["multi_image"], True)  # required for >1 independent image
        self.assertNotIn("system_prompt", body)
        self.assertNotIn("max_frames", body)  # images, not video


class TestVideo(_Base):
    def test_assembly_prompt_format(self):
        for token in ("Step 1:", "Step 2:", "Step 3:", "PASS", "FAIL", "Summary:"):
            self.assertIn(token, video_query.ASSEMBLY_PROMPT)

    def test_mp4_body_uses_max_frames_not_multi_image(self):
        client = self.hermetic_client()
        store: dict = {}
        with mock.patch.object(video_query, "upload_file", lambda client, path: "clip.mp4"), \
                mock.patch.object(ArchetypeAI, "requests_post", _capturing_post({"response": ["ok"]}, store)):
            video_query.example_mp4_direct(client, Path("clip.mp4"))
        body = store["json"]
        self.assertEqual(body["file_ids"], ["clip.mp4"])
        self.assertEqual(body["max_frames"], 32)
        self.assertNotIn("multi_image", body)  # multi_image is multi-image mode, not video
        self.assertNotIn("system_prompt", body)
        self.assertNotIn("sanitize", body)
        self.assertEqual(body["instruction_prompt"], video_query.ASSEMBLY_PROMPT)

    def test_frame_list_body_has_query_metadata(self):
        client = self.hermetic_client()
        store: dict = {}
        frames = [
            REF / "sample_assets" / "assembly_before.png",
            REF / "sample_assets" / "assembly_after.png",
        ]
        with mock.patch.object(ArchetypeAI, "requests_post", _capturing_post({"response": ["ok"]}, store)):
            video_query.example_frame_list(client, frames)
        body = store["json"]
        self.assertIs(body["multi_image"], False)  # false = frames of ONE video
        self.assertEqual(len(body["events"]), 2)
        self.assertEqual(body["events"][0]["type"], "data.base64_img")
        # The triple that makes the frame-list video path work at all:
        self.assertEqual(
            body["query_metadata"],
            {"raw_fps": 1.0, "frames_indices": [0, 1], "total_num_frames": 2},
        )
        self.assertNotIn("system_prompt", body)


class TestTextConstants(unittest.TestCase):
    def test_field_legend_has_columns(self):
        for column in ("time_utc", "mac_a", "prot", "bytes_a", "pkts_b"):
            self.assertIn(column, text_query.FIELD_LEGEND)

    def test_analyst_persona(self):
        self.assertIn("network security analyst", text_query.ANALYST)


if __name__ == "__main__":
    unittest.main(verbosity=2)
