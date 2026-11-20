"""Network-free tests for the custom blueprint skill references.

    python -m pytest skills/atai-custom-agent/tests -q

No credentials, no network. The example blueprints are checked against the committed
snapshot of production's node registry (references/node-registry.prod.json).
"""
import glob
import importlib.util
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "..", "references")

spec = importlib.util.spec_from_file_location("custom_agent", os.path.join(REFS, "custom_agent.py"))
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

with open(os.path.join(REFS, "node-registry.prod.json")) as _f:
    _reg = json.load(_f)
REG = {n["key"]: n for n in _reg["nodes"] + _reg["connectors"]}
BLUEPRINTS = sorted(glob.glob(os.path.join(REFS, "blueprints", "*.yaml")))


def text(name):
    with open(os.path.join(REFS, "blueprints", name)) as f:
        return f.read()


class ExampleBlueprints(unittest.TestCase):
    def test_examples_present(self):
        self.assertEqual([os.path.basename(b) for b in BLUEPRINTS],
                         ["newton-c-video-summary-batch.yaml", "newton-c-video-summary.yaml",
                          "omega-ad-custom.yaml", "omega-embed-stats.yaml"])

    def test_every_example_passes_the_check(self):
        for b in BLUEPRINTS:
            nodes, edges, problems = ca.check(text(os.path.basename(b)), REG)
            self.assertEqual(problems, [], os.path.basename(b))
            self.assertGreater(len(edges), 5)

    def test_batch_video_variant_ends_in_records_sink(self):
        nodes, _, _ = ca.check(text("newton-c-video-summary-batch.yaml"), REG)
        self.assertEqual(nodes["sink"], "RecordsSink")
        self.assertIn("ManualGenerationResultsParserNode", nodes.values())
        nodes, _, _ = ca.check(text("newton-c-video-summary.yaml"), REG)
        self.assertEqual(nodes["sink"], "TextSink")


class Trainable(unittest.TestCase):
    def test_ad_template_is_trainable(self):
        src = text("omega-ad-custom.yaml")
        nodes, _, _ = ca.check(src, REG)
        self.assertIn("AnomalyDetectorNode", nodes.values())
        self.assertIn("\nmetrics:", "\n" + src)
        self.assertIn("primary: anomaly.macro_f1", src)
        self.assertIn("threshold: ${required}", src)

    def test_zero_shot_examples_have_no_metrics(self):
        for name in ("omega-embed-stats.yaml", "newton-c-video-summary.yaml"):
            self.assertNotIn("\nmetrics:", "\n" + text(name))

    def test_spec_matches_blueprint_metrics(self):
        with open(os.path.join(REFS, "optimize", "omega-ad-tep.json")) as f:
            spec = json.load(f)
        self.assertEqual(spec["primary"], "anomaly.macro_f1")
        self.assertEqual(set(spec["ground_truth"]["training"]), {"anomaly"})
        for p in spec["search_space"].values():
            self.assertTrue(all(isinstance(v, (int, str)) and not isinstance(v, bool)
                                for v in p["spec"]["values"]))


class Check(unittest.TestCase):
    def test_unknown_node_key(self):
        bad = text("omega-embed-stats.yaml").replace("key: ChannelFeaturesNode", "key: ChannelStatsNode")
        _, _, problems = ca.check(bad, REG)
        self.assertTrue(any("ChannelStatsNode" in p for p in problems))

    def test_unknown_port(self):
        bad = text("omega-embed-stats.yaml").replace("to: sink.in }", "to: sink.input }")
        _, _, problems = ca.check(bad, REG)
        self.assertTrue(any("no input `input`" in p for p in problems))

    def test_configured_ports_are_accepted(self):
        # TeeNode / ConcatColumnsNode name their ports in config (ports: [emb, feat])
        _, edges, problems = ca.check(text("omega-embed-stats.yaml"), REG)
        self.assertIn(("tee", "emb", "encoder", "in"), edges)
        self.assertEqual(problems, [])

    def test_two_sinks_flagged(self):
        bad = text("omega-embed-stats.yaml").replace(
            "nodes:\n", "nodes:\n  sink2:\n    key: RecordsSink\n    config:\n      format: jsonl\n\n", 1)
        _, _, problems = ca.check(bad, REG)
        self.assertTrue(any("exactly one of each" in p for p in problems))

    def test_sandbox_only_node_flagged(self):
        bad = text("omega-embed-stats.yaml").replace("key: ValueInterpolationNode", "key: x_PassthroughNode")
        _, _, problems = ca.check(bad, REG)
        self.assertTrue(any("x_PassthroughNode" in p for p in problems))


class Outputs(unittest.TestCase):
    def test_millisecond_timestamps_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ms, sec = os.path.join(d, "ms.csv"), os.path.join(d, "s.csv")
            with open(ms, "w") as f:
                f.write("timestamp,a\n1577883600004,1\n")
            with open(sec, "w") as f:
                f.write("timestamp,a\n1577883600.004,1\n")
            ca.check_timestamps(sec)
            with self.assertRaises(SystemExit):
                ca.check_timestamps(ms)

    def test_invalid_share(self):
        csv_body = b"finish_timestamp,predicted_label,invalid\n1,INVALID_STATE,true\n2,normal,false\n"
        self.assertEqual(ca.invalid_share(csv_body), (0.5, 2))
        self.assertEqual(ca.invalid_share(b'{"invalid": true}\n'), (1.0, 1))
        self.assertIsNone(ca.invalid_share(b'{"score": 1}\n'))


if __name__ == "__main__":
    unittest.main()
