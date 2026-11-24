"""Network-free tests for the atai-newton-omega-fine-tuning references (python -m pytest)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "references"))

import omega_ft as api  # noqa: E402


class Config(unittest.TestCase):
    def test_every_endpoint_form(self):
        for ep in ("https://platform.u1.archetypeai.app", "https://platform.u1.archetypeai.app/api/",
                   "https://platform.u1.archetypeai.app/v0.5"):
            self.assertEqual(api.api_base(ep), "https://platform.u1.archetypeai.app/api")

    def test_output_ref_is_not_doubled(self):
        self.assertEqual(api.ref_path("/api/files/fil_1/content"), "/files/fil_1/content")


class Pagination(unittest.TestCase):
    def test_follows_cursor_and_returns_oldest_first(self):
        pages = {None: {"data": [3, 2], "has_more": True, "next_cursor": "c"},
                 "c": {"data": [1], "has_more": False}}
        self.assertEqual(api.list_all("/x", get=lambda p, q: pages[q.get("after")]), [1, 2, 3])


class Data(unittest.TestCase):
    def test_class_files_and_columns(self):
        with tempfile.TemporaryDirectory() as d:
            for split, cls, name in (("train", "attack", "a.csv"), ("train", "normal", "n.csv")):
                p = Path(d, split, cls)
                p.mkdir(parents=True)
                (p / name).write_text("timestamp,s1,s2\n1,0.1,0.2\n")
            files = api.class_files(d, "train")
            self.assertEqual([c for _, c in files], ["attack", "normal"])
            self.assertEqual(api.data_columns(files[0][0]), ["s1", "s2"])

    def test_upload_name_changes_with_content(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "x.csv")
            p.write_text("a")
            first = api.upload_name(p)
            p.write_text("b")
            self.assertNotEqual(first, api.upload_name(p))
            self.assertTrue(first.startswith("omega_x_") and first.endswith(".csv"))


class Bodies(unittest.TestCase):
    def test_head_body(self):
        b = api.head_body([("t.csv", "normal")], [("v.csv", "attack")], "run", ["s1"], window=64, step=2)
        self.assertEqual(b["model"], "omega")
        self.assertEqual(b["method"]["type"], "head")
        self.assertEqual(b["method"]["head"]["reader"], {"window_size": 64, "step_size": 2, "data_columns": ["s1"]})
        self.assertEqual(b["training_files"], [{"type": "n_shot", "file_id": "t.csv", "label": "normal", "format": "csv"}])
        self.assertEqual(b["validation_files"][0]["label"], "attack")

    def test_inference_body_attaches_head_and_required_n_shots(self):
        b = api.inference_body(["e.csv"], [("v.csv", "normal")], "ckp_1", "j", ["s1"], window=64)
        self.assertEqual(b["pipeline_key"], "machine-state-classification")
        i = b["inputs"]
        self.assertEqual(i["worker.fine_tune_checkpoint"][0]["checkpoint_id"], "ckp_1")
        self.assertEqual(i["worker.n_shots"], [{"file_id": "v.csv", "metadata": {"class": "normal"}}])
        self.assertEqual(i["worker.inference"], [{"file_id": "e.csv"}])
        self.assertEqual(b["parameters"]["worker"]["config"]["reader_config"]["window_size"], 64)
        self.assertNotIn("classifier_config", b["parameters"]["worker"]["config"])


class Checkpoints(unittest.TestCase):
    def test_best_is_highest_step(self):
        cks = [{"id": "a", "step": 1}, {"id": "b", "step": 20}, {"id": "c", "step": 11}]
        self.assertEqual(api.best_checkpoint(cks)["id"], "b")
        self.assertIsNone(api.best_checkpoint([]))

    def test_collapse_is_the_one_class_score(self):
        self.assertTrue(api.collapsed([{"step": 1, "metrics": {"macro_f1": 0.3324}}], 2))
        self.assertFalse(api.collapsed([{"step": 1, "metrics": {"macro_f1": 0.96}}], 2))
        self.assertFalse(api.collapsed([{"step": 1}, {"step": 2}], 2))
        self.assertTrue(api.collapsed([{"step": 1, "metrics": {"macro_f1": 1 / 6}}], 3))


class Scoring(unittest.TestCase):
    def test_predictions_and_score(self):
        text = "timestamp,Prediction\n1,normal\n2,attack\n3,normal\n"
        self.assertEqual(api.predictions(text), ["normal", "attack", "normal"])
        r = api.score([("normal", "normal"), ("normal", "attack"), ("attack", "attack")])
        self.assertEqual((r["correct"], r["n"]), (2, 3))
        self.assertAlmostEqual(r["per_class"]["normal"]["recall"], 0.5)
        self.assertAlmostEqual(r["per_class"]["attack"]["precision"], 0.5)

    def test_one_class_predictor_scores_a_third(self):
        pairs = [("normal", "normal")] * 50 + [("attack", "normal")] * 50
        self.assertAlmostEqual(api.score(pairs)["macro_f1"], 1 / 3, places=3)


if __name__ == "__main__":
    unittest.main()
