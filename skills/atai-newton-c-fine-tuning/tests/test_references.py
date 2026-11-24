"""Network-free tests for the atai-newton-c-fine-tuning references (python -m pytest)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "references"))

import atai_ft as api  # noqa: E402
import build_dataset as bd  # noqa: E402
import run_fine_tune as rft  # noqa: E402


class ApiBase(unittest.TestCase):
    def test_every_endpoint_form(self):
        for ep in ("https://platform.u1.archetypeai.app", "https://platform.u1.archetypeai.app/",
                   "https://platform.u1.archetypeai.app/api", "https://platform.u1.archetypeai.app/api/",
                   "https://platform.u1.archetypeai.app/v0.5"):
            self.assertEqual(api.api_base(ep), "https://platform.u1.archetypeai.app/api")

    def test_output_ref_is_not_doubled(self):
        self.assertEqual(api.ref_path("/api/files/fil_1/content"), "/files/fil_1/content")
        self.assertEqual(api.ref_path("/v0.5/files/x"), "/files/x")
        self.assertEqual(api.ref_path("/files/x"), "/files/x")


class Pagination(unittest.TestCase):
    def test_follows_cursor_and_returns_oldest_first(self):
        pages = {None: {"data": [5, 4], "has_more": True, "next_cursor": "c1"},
                 "c1": {"data": [3, 2], "has_more": True, "next_cursor": "c2"},
                 "c2": {"data": [1], "has_more": False, "next_cursor": None}}
        calls = []

        def get(path, params):
            calls.append(params.get("after"))
            return pages[params.get("after")]
        self.assertEqual(api.list_all("/x", get=get), [1, 2, 3, 4, 5])
        self.assertEqual(calls, [None, "c1", "c2"])


class Bodies(unittest.TestCase):
    def test_fine_tune_body_has_required_fields(self):
        b = api.fine_tune_body("tr.jsonl", "va.jsonl", "run1")
        self.assertEqual(b["model"], "newton")
        self.assertEqual(b["name"], "run1")
        self.assertEqual(b["validation_files"], [{"type": "plain", "file_id": "va.jsonl", "format": "jsonl"}])
        self.assertNotIn("base_model_tag", b)

    def test_checkpoint_goes_on_the_port_not_config(self):
        b = api.inference_job_body("f.jsonl", "j", "ckp_1")
        self.assertEqual(b["inputs"]["worker.checkpoint"], [{"kind": "checkpoint", "checkpoint_id": "ckp_1"}])
        cfg = b["parameters"]["worker"]["config"]
        self.assertNotIn("checkpoint_id", cfg)
        self.assertEqual(cfg["model_variant"], api.BASE_VARIANT)

    def test_baseline_has_no_checkpoint(self):
        self.assertNotIn("worker.checkpoint", api.inference_job_body("f.jsonl", "j")["inputs"])

    def test_best_checkpoint_is_lowest_eval_loss(self):
        cks = [{"id": "a", "metrics": {"eval_loss": 1.3}}, {"id": "b", "metrics": {"eval_loss": 1.1}},
               {"id": "c", "metrics": {}}]
        self.assertEqual(api.best_checkpoint(cks)["id"], "b")
        self.assertEqual(api.best_checkpoint([{"id": "z"}])["id"], "z")
        self.assertIsNone(api.best_checkpoint([]))

    def test_success_covers_both_spellings(self):
        self.assertEqual(api.FT_SUCCESS, {"SUCCEEDED", "COMPLETED"})
        self.assertTrue(api.FT_SUCCESS <= api.FT_TERMINAL)


class Records(unittest.TestCase):
    def test_training_record_shape(self):
        r = bd.training_record(["AAA", "BBB"], "Archery")
        roles = [m["role"] for m in r["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        user = r["messages"][1]["content"]
        self.assertEqual(user[0]["type"], "text")
        self.assertEqual(user[1]["image_url"]["url"], "data:image/jpeg;base64,AAA")
        self.assertEqual(r["messages"][2]["content"], "Archery")
        json.dumps(r)

    def test_inference_record_shape(self):
        r = bd.inference_record(["AAA"], "p")
        self.assertEqual(set(r), {"system", "prompt", "inputs"})
        self.assertEqual(r["inputs"], [{"type": "image", "format": "base64", "data": "AAA"}])

    def test_closed_prompt_lists_every_label(self):
        p = bd.closed_prompt(["A", "B"])
        self.assertIn("A, B", p)

    def test_frame_times_are_segment_centres(self):
        self.assertEqual(bd.frame_times(8.0, 4), [1.0, 3.0, 5.0, 7.0])


class Scoring(unittest.TestCase):
    classes = ["Basketball", "BasketballDunk", "ApplyEyeMakeup", "ApplyLipstick"]

    def test_longest_match_resolves_nested_labels(self):
        self.assertEqual(rft.predicted_class("BasketballDunk", self.classes), "BasketballDunk")
        self.assertEqual(rft.predicted_class("basketball", self.classes), "Basketball")
        self.assertEqual(rft.predicted_class("Apply eye makeup", self.classes), "ApplyEyeMakeup")
        self.assertIsNone(rft.predicted_class("vault", self.classes))

    def test_dunk_answer_is_wrong_for_basketball_clip(self):
        s = rft.score(["BasketballDunk", "Basketball"], ["Basketball", "Basketball"], self.classes)
        self.assertEqual(s["correct"], 1)
        self.assertEqual(s["per_class"]["Basketball"], [1, 2])


if __name__ == "__main__":
    unittest.main()
