"""Network-free unit tests for the reference scripts.

These mock the official archetypeai client's transport, so they run with NO
credentials and NO network. They lock in the invariants discovered while
moving this skill to Model Serving:

  * Every call goes to `{api}/inference/{query,chat/completions,responses}`
    with a `model` alias (`newton-c-2.6` by default).
  * /query sends the system turn in `system_prompt`, never
    `instruction_prompt`, and never the refused knobs (`multi_image`,
    `max_frames`, `events`, `query_metadata`, `presence_penalty`).
  * Fusion without reasoning gets `/no_think` appended to the user text;
    C 2.6 never does. Whenever Fusion reasons (reasoning mode, or any
    text-only request) the budget is >= 4096 and `repetition_penalty` is set.
  * `upload_file` returns the `fil_` uid, not the filename.
  * Inline payloads over the limit are refused before sending.
  * `split_thinking` strips a closed <think> block and raises on an
    unclosed one; `extract_text` handles all three response shapes.
  * Both ATAI_API_KEY and ATAI_API_ENDPOINT are required — no default.

Run:
    python -m unittest discover -s skills/atai-newton-fusion-model/tests
    # or, from this directory:
    python test_references.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REF = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REF))

import _common  # noqa: E402
import compare_models  # noqa: E402
import image_query  # noqa: E402
import text_query  # noqa: E402
import video_query  # noqa: E402

from archetypeai.api_client import ArchetypeAI  # noqa: E402

QUERY_REPLY = {"response": {"response": ["ok"], "success": True}}
CHAT_REPLY = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
RESPONSES_REPLY = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}


def _capturing_post(payload: dict, store: dict):
    """Fake ArchetypeAI.requests_post that records its args and returns `payload`."""

    def _requests_post(self, api_endpoint, data_payload, additional_headers={}):
        store.update(url=api_endpoint, json=json.loads(data_payload))
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
        self.hermetic_env(ATAI_API_KEY="test-key", ATAI_API_ENDPOINT="https://example.test/v0.5")
        return _common.make_client()

    def capture(self, payload: dict) -> dict:
        store: dict = {}
        patcher = mock.patch.object(ArchetypeAI, "requests_post", _capturing_post(payload, store))
        patcher.start()
        self.addCleanup(patcher.stop)
        return store


class TestMakeClient(_Base):
    def test_missing_key_exits(self):
        self.hermetic_env(ATAI_API_ENDPOINT="https://example.test")
        with self.assertRaises(SystemExit):
            _common.make_client()

    def test_missing_endpoint_exits(self):
        self.hermetic_env(ATAI_API_KEY="k")
        with self.assertRaises(SystemExit):
            _common.make_client()

    def test_endpoint_normalized(self):
        for raw in (
            "https://example.test",
            "https://example.test/",
            "https://example.test/api",
            "https://example.test/api/inference",
            "https://example.test/api/lens",
            "https://example.test/v0.5",
        ):
            self.hermetic_env(ATAI_API_KEY="k", ATAI_API_ENDPOINT=raw)
            self.assertEqual(_common.make_client().api_endpoint, "https://example.test/api", raw)


class TestQueryBody(_Base):
    def test_c26_defaults(self):
        client = self.hermetic_client()
        store = self.capture(QUERY_REPLY)
        text, _, _ = _common.query(client, "hi", system_prompt="sys", file_ids=["fil_abc"])
        self.assertEqual(text, "ok")
        self.assertEqual(store["url"], "https://example.test/api/inference/query")
        body = store["json"]
        self.assertEqual(body["model"], "newton-c-2.6")
        self.assertEqual(body["query"], "hi")
        self.assertEqual(body["system_prompt"], "sys")
        self.assertEqual(body["file_ids"], ["fil_abc"])
        for refused in ("instruction_prompt", "multi_image", "max_frames", "events", "query_metadata", "presence_penalty"):
            self.assertNotIn(refused, body)

    def test_system_prompt_omitted_when_empty(self):
        client = self.hermetic_client()
        store = self.capture(QUERY_REPLY)
        _common.query(client, "hi")
        self.assertNotIn("system_prompt", store["json"])
        self.assertEqual(store["json"]["file_ids"], [])

    def test_fusion_with_media_appends_no_think(self):
        client = self.hermetic_client()
        store = self.capture(QUERY_REPLY)
        _common.query(client, "describe", model=_common.FUSION, file_ids=["fil_x"], max_new_tokens=300)
        body = store["json"]
        self.assertTrue(body["query"].endswith(" /no_think"))
        self.assertEqual(body["max_new_tokens"], 300)
        self.assertNotIn("repetition_penalty", body)

    def test_fusion_text_only_gets_reasoning_budget(self):
        client = self.hermetic_client()
        store = self.capture(QUERY_REPLY)
        _common.query(client, "hi", model=_common.FUSION, max_new_tokens=256)
        body = store["json"]
        self.assertGreaterEqual(body["max_new_tokens"], 4096)
        self.assertEqual(body["repetition_penalty"], 1.1)

    def test_fusion_reasoning(self):
        client = self.hermetic_client()
        store = self.capture(QUERY_REPLY)
        _common.query(client, "describe", model=_common.FUSION, file_ids=["fil_x"], reasoning=True)
        body = store["json"]
        self.assertEqual(body["query"], "describe")
        self.assertGreaterEqual(body["max_new_tokens"], 4096)
        self.assertEqual(body["repetition_penalty"], 1.1)

    def test_c26_never_gets_fusion_switches(self):
        client = self.hermetic_client()
        store = self.capture(QUERY_REPLY)
        _common.query(client, "hi", reasoning=True, max_new_tokens=100)
        body = store["json"]
        self.assertEqual(body["query"], "hi")
        self.assertEqual(body["max_new_tokens"], 100)
        self.assertNotIn("repetition_penalty", body)

    def test_sampling_params_pass_through(self):
        client = self.hermetic_client()
        store = self.capture(QUERY_REPLY)
        _common.query(client, "hi", temperature=0.2, do_sample=False)
        self.assertEqual(store["json"]["temperature"], 0.2)
        self.assertIs(store["json"]["do_sample"], False)

    def test_think_block_stripped(self):
        client = self.hermetic_client()
        self.capture({"response": {"response": ["<think>\nplan\n</think>\n\nanswer"], "success": True}})
        text, _, _ = _common.query(client, "hi", model=_common.FUSION)
        self.assertEqual(text, "answer")


class TestChatAndResponses(_Base):
    def test_chat_body(self):
        client = self.hermetic_client()
        store = self.capture(CHAT_REPLY)
        parts = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA"}}, {"type": "text", "text": "what?"}]
        text, _, _ = _common.chat(client, parts, model=_common.FUSION, system_prompt="sys", max_tokens=200)
        self.assertEqual(text, "ok")
        self.assertEqual(store["url"], "https://example.test/api/inference/chat/completions")
        body = store["json"]
        self.assertEqual(body["messages"][0], {"role": "system", "content": "sys"})
        user = body["messages"][1]["content"]
        self.assertEqual(user[0], parts[0])
        self.assertEqual(user[1]["text"], "what? /no_think")
        self.assertEqual(body["max_tokens"], 200)

    def test_chat_text_only_fusion_budget(self):
        client = self.hermetic_client()
        store = self.capture(CHAT_REPLY)
        _common.chat(client, [{"type": "text", "text": "hi"}], model=_common.FUSION, max_tokens=200)
        self.assertGreaterEqual(store["json"]["max_tokens"], 4096)
        self.assertNotIn("system", [m["role"] for m in store["json"]["messages"]])

    def test_responses_body(self):
        client = self.hermetic_client()
        store = self.capture(RESPONSES_REPLY)
        parts = [{"type": "input_video", "video_url": "data:video/mp4;base64,AA"}, {"type": "input_text", "text": "what?"}]
        text, _, _ = _common.respond(client, parts, instructions="sys", max_output_tokens=300)
        self.assertEqual(text, "ok")
        self.assertEqual(store["url"], "https://example.test/api/inference/responses")
        body = store["json"]
        self.assertEqual(body["model"], "newton-c-2.6")
        self.assertEqual(body["instructions"], "sys")
        self.assertEqual(body["input"], [{"role": "user", "content": parts}])
        self.assertEqual(body["max_output_tokens"], 300)


class TestSplitThinking(unittest.TestCase):
    def test_no_block(self):
        self.assertEqual(_common.split_thinking("plain"), ("plain", ""))

    def test_empty_block(self):
        self.assertEqual(_common.split_thinking("<think>\n\n</think>\n\nanswer"), ("answer", ""))

    def test_reasoning_block(self):
        self.assertEqual(_common.split_thinking("<think>step 1</think> final"), ("final", "step 1"))

    def test_unclosed_block_raises(self):
        with self.assertRaises(RuntimeError):
            _common.split_thinking("<think>counting cars: 1, 2, 3")


class TestExtractText(unittest.TestCase):
    def test_query_shape(self):
        self.assertEqual(_common.extract_text(QUERY_REPLY), "ok")

    def test_query_success_false_raises(self):
        with self.assertRaises(RuntimeError):
            _common.extract_text({"response": {"response": [], "success": False}})

    def test_chat_shape(self):
        self.assertEqual(_common.extract_text(CHAT_REPLY), "ok")

    def test_responses_shape(self):
        self.assertEqual(_common.extract_text(RESPONSES_REPLY), "ok")

    def test_empty_payload(self):
        self.assertEqual(_common.extract_text({}), "")


class TestUploadFile(_Base):
    def test_missing_file_exits(self):
        client = self.hermetic_client()
        with self.assertRaises(SystemExit):
            _common.upload_file(client, "/nonexistent/clip.mp4")

    def test_returns_fil_uid_not_filename(self):
        client = self.hermetic_client()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp, mock.patch.object(
            client.files.local, "upload", return_value={"file_id": "clip.mp4", "file_uid": "fil_123", "is_valid": True}
        ):
            self.assertEqual(_common.upload_file(client, tmp.name), "fil_123")


class TestInlinePayloads(unittest.TestCase):
    def test_data_uri_mime(self):
        uri = _common.data_uri(REF / "sample_assets" / "wind-turbines-small.jpg")
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        uri = _common.data_uri(REF / "sample_assets" / "1_pass_2_pass_3_pass_B_small.mp4")
        self.assertTrue(uri.startswith("data:video/mp4;base64,"))

    def test_large_inline_refused(self):
        with self.assertRaises(SystemExit):
            _common.read_base64(REF / "sample_assets" / "1_pass_2_pass_3_pass_B.mp4")

    def test_inline_samples_fit(self):
        for name in ("wind-turbines-small.jpg", "1_pass_2_pass_3_pass_B_small.mp4", "assembly_before.png", "assembly_after.png"):
            _common.read_base64(REF / "sample_assets" / name)


class TestScripts(unittest.TestCase):
    def test_resolve_model(self):
        self.assertEqual(_common.resolve_model("c26"), "newton-c-2.6")
        self.assertEqual(_common.resolve_model("fusion"), "newton-fusion-1.0")
        self.assertEqual(_common.resolve_model("newton-fusion-1.0"), "newton-fusion-1.0")

    def test_sample_paths_exist(self):
        for path in (
            image_query.DEFAULT_IMAGE,
            image_query.SMALL_IMAGE,
            image_query.BEFORE_IMAGE,
            image_query.AFTER_IMAGE,
            video_query.DEFAULT_VIDEO,
            video_query.SMALL_VIDEO,
        ):
            self.assertTrue(path.exists(), path)

    def test_assembly_prompt_format(self):
        for step in ("Step 1:", "Step 2:", "Step 3:", "Summary:"):
            self.assertIn(step, video_query.ASSEMBLY_PROMPT)

    def test_flow_log_has_legend_and_rows(self):
        self.assertTrue(text_query.FLOW_LOG.startswith(text_query.FIELD_LEGEND))
        self.assertEqual(text_query.FLOW_LOG.count("2019-10-19T"), 14)

    def test_compare_kinds(self):
        self.assertEqual(compare_models.kind_of(Path("a.png")), "image")
        self.assertEqual(compare_models.kind_of(Path("a.mp4")), "video")
        self.assertEqual(compare_models.kind_of(Path("a.txt")), "text")
        with self.assertRaises(SystemExit):
            compare_models.kind_of(Path("a.bin"))

    def test_compare_variants(self):
        self.assertEqual(compare_models.VARIANTS["c26"]["model"], "newton-c-2.6")
        self.assertTrue(compare_models.VARIANTS["fusion+reason"]["reasoning"])


if __name__ == "__main__":
    unittest.main()
