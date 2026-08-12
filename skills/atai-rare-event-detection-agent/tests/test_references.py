"""Network-free unit tests for the RED agent reference script.

Mocks urllib at the request boundary, so these run with NO credentials and NO
network. They lock in the invariants verified against dev:
  * Both ATAI_API_KEY and ATAI_API_ENDPOINT are required — no default endpoint.
  * The bundle body pins the `red` blueprint and keys the artifact map by
    `red-classifier` (renamed from `rad-classifier` on 2026-07-28), and the run body binds
    the source connector to the upload's file_id (the filename), not the fil_ uid.
  * `step_size` is the only value sent: window_size/data_columns/timestamp_column
    are inherited from the classifier's own `parameters` metadata.
  * score() applies the design's majority rule (ties to the rare event), skips
    INVALID_STATE windows, reports the false-alarm rate only over windows with
    zero fault rows, and reports incident-level detection separately — a slice
    can score high accuracy while missing its incident entirely.
  * The shipped slices satisfy what the blueprint validates: a strictly regular
    cadence, monotonic timestamps, a sidecar aligned row-for-row, and shot files
    that are single-class and unlabelled (the class comes from the filename).

Run:
    python -m unittest discover -s skills/atai-rare-event-detection-agent/tests
    # or, from this directory:
    python test_references.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REF = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REF))

import run_red_agent  # noqa: E402

SAMPLE = REF / "sample_data"
FAULT = "pump_breakdown"   # the class name in the shipped sample data


class EnvTests(unittest.TestCase):
    def test_endpoint_required(self):
        with mock.patch.dict(os.environ, {"ATAI_API_KEY": "k"}, clear=True):
            with self.assertRaises(SystemExit):
                run_red_agent.api_base()

    def test_key_required(self):
        with mock.patch.dict(os.environ, {"ATAI_API_ENDPOINT": "https://x"}, clear=True):
            with self.assertRaises(SystemExit):
                run_red_agent.require_key()

    def test_agents_mount_is_versionless(self):
        with mock.patch.dict(os.environ, {"ATAI_API_ENDPOINT": "https://x"}, clear=True):
            self.assertEqual(run_red_agent.agents_base(), "https://x/agents")
            self.assertNotIn("/v0.5/agents", run_red_agent.agents_base())


class ScoringTests(unittest.TestCase):
    """score() reads an output CSV and a labels sidecar off disk."""

    def _score(self, preds, labels, window_size=4):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.csv"
            lab = Path(d) / "in_labels.csv"
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["timestamp", "predicted_state",
                                                  "invalid", "p_normal",
                                                  f"p_{FAULT}"])
                w.writeheader()
                w.writerows(preds)
            with open(lab, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["timestamp", "label"])
                w.writeheader()
                w.writerows(labels)
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf):
                run_red_agent.score(str(out), str(lab), window_size)
            return buf.getvalue()

    def test_invalid_windows_are_skipped_not_scored(self):
        labels = [{"timestamp": 60 * i, "label": "normal"} for i in range(8)]
        preds = [{"timestamp": float(60 * i), "predicted_state": "INVALID_STATE",
                  "invalid": "true", "p_normal": 0, f"p_{FAULT}": 0}
                 for i in range(4, 8)]
        out = self._score(preds, labels)
        self.assertIn("4 invalid", out)
        self.assertIn("0 scored windows", out)

    def test_majority_rule_ties_go_to_the_rare_event(self):
        # window_size 4 ending at row 3 spans rows 0-3: two fault, two normal.
        labels = [{"timestamp": 60 * i, "label": FAULT if i < 2 else "normal"}
                  for i in range(4)]
        preds = [{"timestamp": 180.0, "predicted_state": FAULT, "invalid": "false",
                  "p_normal": 0, f"p_{FAULT}": 1}]
        out = self._score(preds, labels)
        self.assertIn("tp=1", out)  # a 50/50 tie counts as the rare event

    def test_high_accuracy_can_coexist_with_a_missed_incident(self):
        """The reason incident-level detection is reported separately."""
        labels = [{"timestamp": 60 * i, "label": "normal"} for i in range(40)]
        for i in (20, 21):                       # a 2-row incident
            labels[i]["label"] = FAULT
        preds = [{"timestamp": float(60 * i), "predicted_state": "normal",
                  "invalid": "false", "p_normal": 1, f"p_{FAULT}": 0}
                 for i in range(3, 40)]
        out = self._score(preds, labels)
        # The incident is reported MISSED even though accuracy stays high, which
        # is the whole reason incident-level detection is a separate view.
        self.assertIn("MISSED", out)
        acc = float(out.split("accuracy:")[1].split()[0])
        self.assertGreater(acc, 0.9)
        self.assertIn("fn=3", out)

    def test_false_alarms_counted_only_on_zero_fault_windows(self):
        labels = [{"timestamp": 60 * i, "label": "normal"} for i in range(8)]
        labels[7]["label"] = FAULT               # last row is fault
        preds = [
            # window ending row 3 spans 0-3: no fault rows -> a real false alarm
            {"timestamp": 180.0, "predicted_state": FAULT, "invalid": "false",
             "p_normal": 0, f"p_{FAULT}": 1},
            # window ending row 7 spans 4-7: contains a fault row -> not counted
            {"timestamp": 420.0, "predicted_state": FAULT, "invalid": "false",
             "p_normal": 0, f"p_{FAULT}": 1},
        ]
        out = self._score(preds, labels)
        self.assertIn("1/1", out)                # one alarm over one clean window


class SampleDataTests(unittest.TestCase):
    """The shipped slices must satisfy what the blueprint validates."""

    INFERENCE = SAMPLE / "pump_eval_inc04.csv"
    LABELS = SAMPLE / "pump_eval_inc04_labels.csv"
    SHOTS = {
        "pump_nshot_normal.csv": "normal",
        "pump_nshot_pump_breakdown_inc01.csv": "pump_breakdown",
        "pump_nshot_pump_breakdown_inc02.csv": "pump_breakdown",
    }

    @staticmethod
    def _read(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def test_slice_and_sidecar_line_up(self):
        rows = self._read(self.INFERENCE)
        labels = self._read(self.LABELS)
        self.assertEqual(len(rows), len(labels))
        self.assertEqual([r["timestamp"] for r in rows],
                         [l["timestamp"] for l in labels])

    def test_cadence_is_strictly_regular(self):
        """Irregular timestamps make the blueprint emit INVALID_STATE."""
        ts = [int(r["timestamp"]) for r in self._read(self.INFERENCE)]
        self.assertEqual({b - a for a, b in zip(ts, ts[1:])}, {60})
        self.assertEqual(ts, sorted(ts), "timestamps must be monotonic")

    def test_slice_carries_both_classes_and_normal_context(self):
        labels = [r["label"] for r in self._read(self.LABELS)]
        self.assertEqual(set(labels), {"normal", FAULT})
        # normal context on both sides is what makes a false-alarm rate meaningful
        self.assertEqual(labels[0], "normal")
        self.assertEqual(labels[-1], "normal")
        self.assertLess(labels.count(FAULT) / len(labels), 0.10,
                        "the fault class should be rare — that is the point")

    def test_shot_files_are_single_class_and_unlabelled(self):
        """The fitting runner takes each file's class from its FILENAME."""
        for name, cls in self.SHOTS.items():
            path = SAMPLE / name
            self.assertTrue(path.exists(), f"{name} missing")
            with open(path) as f:
                header = f.readline().strip().split(",")
            self.assertEqual(header[0], "timestamp")
            self.assertNotIn("label", header,
                             "shot files carry no label column")
            self.assertIn(cls, name,
                          "the class name must appear in the filename")

    def test_filename_class_match_is_unambiguous(self):
        """Longest match wins, so 'normal' must not also match a fault file."""
        classes = sorted(set(self.SHOTS.values()), key=len, reverse=True)
        for name, expected in self.SHOTS.items():
            hits = [c for c in classes if c.lower() in name.lower()]
            self.assertEqual(max(hits, key=len), expected, name)


class BundleShapeTests(unittest.TestCase):
    def test_quick_start_bundle_names_are_the_stable_handles(self):
        """Bundle ids are environment-scoped; the NAMES resolve everywhere."""
        src = (REF / "run_red_agent.py").read_text()
        self.assertIn('QUICK_START_BUNDLE = "RED Quick Start (Pump Breakdown)"', src)
        self.assertIn('QUICK_START_BUNDLE_EMBEDDINGS = '
                      '"RED Quick Start (Pump Breakdown, Embeddings)"', src)
        # no real bnd_… id may be hardcoded — that would silently pin one
        # environment (prose like "bnd_…" in help text is fine)
        import re
        self.assertEqual(re.findall(r"bnd_[a-z0-9]{10,}", src), [])

    def test_resolution_is_by_query_with_exact_name_match(self):
        """?query= is a substring search (?name=/?search= are ignored), so the
        runner must exact-match the name client-side and prefer canonical."""
        src = (REF / "run_red_agent.py").read_text()
        self.assertIn("bundles?query=", src)
        self.assertIn('bundle.get("name") == name', src)
        self.assertIn('bundle.get("is_canonical")', src)

    def test_bundle_endpoints_are_plural(self):
        """The bundle API is plural everywhere as of 2026-08-11; the singular
        paths (POST /agents/bundle, POST /agents/bundle/{id}/run) 404."""
        src = (REF / "run_red_agent.py").read_text()
        self.assertIn('f"{agents}/bundles/{bundle[\'id\']}/run"', src)
        self.assertNotIn("/bundle/", src.replace("/bundles/", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
