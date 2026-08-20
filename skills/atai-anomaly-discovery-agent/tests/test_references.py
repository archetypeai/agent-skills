#!/usr/bin/env python3
"""Network-free tests for the AD skill's references.

    python3 -m pytest skills/atai-anomaly-discovery-agent/tests -q

Every test here encodes a mistake that is easy to make and expensive to find:
the request shapes that 404 on the singular endpoint, the validation values that
silently invalidate every window, the label indexing that reads the wrong part
of an asset's life, and the scoring rule that must not be precision/recall.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import pathlib
import sys
import textwrap

import pytest

REFS = pathlib.Path(__file__).resolve().parent.parent / "references"
SAMPLE = REFS / "sample_data" / "bearing_eval_set2_brg1_transition.csv"
LABELS = REFS / "sample_data" / "bearing_eval_set2_brg1_transition_labels.csv"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_ad_agent", REFS / "run_ad_agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_ad_agent"] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()



class FakeClient:
    """Stands in for ArchetypeAI, recording calls and returning canned payloads.

    Mocks the client object rather than HTTP, so the tests assert the lifecycle
    the runner drives and stay network-free.
    """

    def __init__(self, calls=None, **payloads):
        self.calls = calls if calls is not None else {}
        p = payloads
        outer = self

        class _Local:
            def upload(self, path):
                outer.calls["upload"] = os.path.basename(path)
                return p.get("upload", {"file_id": os.path.basename(path)})

            def download(self, name, out_path):
                outer.calls["download"] = name
                with open(out_path, "w") as f:
                    f.write(p.get("download", "x"))
                return True

        class _Files:
            local = _Local()

        class _Bundles:
            def list(self, query=None, limit=None):
                outer.calls["list"] = query
                return p.get("list", {"data": []})

            def create(self, blueprint=None, name=None, values=None, artifacts=None,
                       description="", model=None):
                outer.calls["create"] = {"blueprint": blueprint, "name": name,
                                         "values": values, "artifacts": artifacts}
                return p.get("create", {"id": "bnd_test"})

            def run(self, bundle_id, source=None, sink=None):
                outer.calls["run"] = {"bundle_id": bundle_id, "source": source}
                return p.get("run", {"id": "agt_1"})

        class _Instances:
            def get(self, agent_id):
                return p.get("get", {"status": "completed"})

            def get_logs(self, agent_id, **kw):
                return p.get("logs", {"data": []})

            def get_results(self, agent_id, **kw):
                return p.get("results", {"data": []})

        class _Agents:
            bundles = _Bundles()
            instances = _Instances()

        self.files = _Files()
        self.agents = _Agents()


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------

def test_endpoint_is_normalised_for_the_client():
    """The client wants the /vX.Y suffix: verbatim for files, stripped for agents.

    A bare root breaks uploads with an empty ApiError while bundle calls keep
    working, so both .env conventions in this repo normalise to one string.
    """
    for endpoint in ("https://x.test", "https://x.test/", "https://x.test/v0.5",
                     "https://x.test/v0.5/"):
        assert runner.versioned(endpoint) == "https://x.test/v0.5", endpoint


def test_agent_paths_come_from_the_client():
    """The client owns the paths, so this runner cannot drift from them again.

    The API moved from singular /agents/bundle to plural /agents/bundles on
    2026-08-11 and the singular forms now 404.
    """
    src = (REFS / "run_ad_agent.py").read_text()
    assert "urllib.request.Request(" not in src
    for call in ("client().agents.bundles.list(",
                 "client().agents.bundles.create(",
                 "client().agents.bundles.run(",
                 "client().agents.instances.get_results("):
        assert call in src, call


def test_no_runs_endpoint_is_used():
    """There is no /agents/bundles/{id}/runs — a client must filter
    /agents/instances by bundle_id, and that paginates."""
    src = (REFS / "run_ad_agent.py").read_text()
    assert "/runs" not in src


# --------------------------------------------------------------------------
# bundle creation — the values that decide whether a run produces anything
# --------------------------------------------------------------------------

def test_bundle_body_disables_monotonic_validation(monkeypatch):
    """Left at the blueprint default of true, EVERY window is invalidated above
    1 kHz and the run still reports completed."""
    captured = {}

    monkeypatch.setattr(runner, "client", lambda: FakeClient(captured))
    runner.create_bundle("n", "s3://bucket/d.safetensors", 1.762, False)

    values = captured["create"]["values"]
    assert values["validate_monotonic_timestamps"] is False
    assert values["sample_rate_interval_tolerance"] is None
    assert values["output_score"] is True


def test_bundle_uses_the_ad_detector_artifact_key(monkeypatch):
    """A wrong key is accepted at creation and fails ~30 s into the run. The
    key is `ad-detector`; a second blueprint `ada` uses `ada-detector`."""
    captured = {}
    monkeypatch.setattr(runner, "client", lambda: FakeClient(captured))
    runner.create_bundle("n", "s3://bucket/d.safetensors", 1.762, False)
    assert list(captured["create"]["artifacts"]) == ["ad-detector"]
    assert captured["create"]["blueprint"] == "ad"


def test_quick_start_bundle_names_are_the_stable_handles():
    """Bundle ids are environment-scoped; the NAMES resolve everywhere."""
    src = (REFS / "run_ad_agent.py").read_text()
    assert 'QUICK_START_BUNDLE = "AD Quick Start (Bearing Breakdown)"' in src
    assert ('QUICK_START_BUNDLE_EMBEDDINGS = '
            '"AD Quick Start (Bearing Breakdown, Embeddings)"') in src
    # No real bnd_… id may be hardcoded — that would silently pin one
    # environment (prose like "bnd_…" in help text is fine).
    import re
    assert re.findall(r"bnd_[a-z0-9]{10,}", src) == []


def test_resolution_selects_the_exact_canonical_match(monkeypatch):
    """?query= is a substring search returning newest-first, so a PREFIX of the
    base name also matches the Embeddings variant. Exact-name matching with an
    is_canonical preference is what selects. (The full base name including its
    closing paren matches only itself — verified on Prod — but a prefix does
    not, which is why data[0] is never safe.)"""
    listing = {"data": [
        {"id": "bnd_emb", "name": "AD Quick Start (Bearing Breakdown, Embeddings)",
         "is_canonical": True},
        {"id": "bnd_copy", "name": "AD Quick Start (Bearing Breakdown)",
         "is_canonical": False},
        {"id": "bnd_base", "name": "AD Quick Start (Bearing Breakdown)",
         "is_canonical": True},
    ]}
    monkeypatch.setattr(runner, "client", lambda: FakeClient(list=listing))
    assert runner.find_bundle("AD Quick Start (Bearing Breakdown)") == "bnd_base"


def test_unresolved_bundle_points_at_support(monkeypatch):
    """The name-not-found path must give the user a way forward."""
    monkeypatch.setattr(runner, "client", lambda: FakeClient(list={"data": []}))
    with pytest.raises(SystemExit) as excinfo:
        runner.resolve_quick_start("AD Quick Start (Bearing Breakdown)")
    message = str(excinfo.value)
    assert "support@archetypeai.dev" in message
    assert "--bundle-id" in message


def test_run_request_shape(monkeypatch):
    captured = {}
    monkeypatch.setattr(runner, "client", lambda: FakeClient(captured))
    assert runner.run_bundle("bnd_1", "f.csv") == "agt_1"
    assert captured["run"]["bundle_id"] == "bnd_1"
    # the client builds the data ref; the source is the file_id, not the fil_ uid
    assert captured["run"]["source"] == ["f.csv"]


# --------------------------------------------------------------------------
# polling — status is not authoritative
# --------------------------------------------------------------------------

def test_poll_treats_an_error_log_as_terminal(monkeypatch):
    """Pods have terminated with Error (exit=1) while status read `running`
    for hours. A client polling on status alone spins to its timeout."""
    fake = FakeClient(
        get={"status": "running"},            # never terminal
        logs={"data": [{"id": "1", "level": "error", "message": "worker 0 exploded"}]})
    monkeypatch.setattr(runner, "client", lambda: fake)
    assert runner.poll("agt_1", timeout_s=5, interval_s=0) == "failed"


def test_download_reports_empty_results(monkeypatch, capsys):
    """An all-invalid run reports `completed` with NO results. That emptiness
    is the tell, not the runtime."""
    monkeypatch.setattr(runner, "client", lambda: FakeClient(results={"data": []}))
    assert runner.download_results("agt_1", "/tmp/unused.csv") is None
    assert "EMPTY" in capsys.readouterr().out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _write(tmp_path, scores, labels_hours):
    out = tmp_path / "out.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["finish_timestamp", "start_timestamp", "predicted_label", "invalid", "anomaly_score"])
        for i, s in enumerate(scores):
            w.writerow([i + 1, i, "anomaly" if s > 1.762 else "normal", "false", s])
    lab = tmp_path / "out_labels.csv"
    with lab.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["snapshot_index", "operating_hours_to_end", "in_reference", "failed", "defect_mode"])
        for i, h in enumerate(labels_hours):
            w.writerow([i, h, "false", "true", "outer_race"])
    return str(out), str(lab)


def test_sustained_rule_needs_three_consecutive(tmp_path):
    """One window over the line is noise. Two isolated spikes must not fire."""
    out, lab = _write(tmp_path, [1.0, 2.0, 1.0, 2.0, 1.0, 1.0], [9, 8, 7, 6, 5, 4])
    assert runner.score(out, lab, 1.762)["detected"] is False


def test_sustained_rule_fires_and_reports_lead(tmp_path):
    out, lab = _write(tmp_path, [1.0, 1.0, 2.0, 2.0, 2.0, 1.0], [9, 8, 7, 6, 5, 4])
    res = runner.score(out, lab, 1.762)
    assert res["detected"] is True
    assert res["detected_at_snapshot"] == 2
    assert res["lead_hours"] == 7          # hours at the FIRST snapshot of the run


def test_crossing_rate_is_over_windows(tmp_path):
    out, lab = _write(tmp_path, [1.0, 1.0, 1.0, 2.0], [4, 3, 2, 1])
    assert runner.score(out, lab, 1.762)["crossing_rate"] == 0.25


def test_score_reports_no_precision_or_recall(tmp_path):
    """Deliberate: the data has no per-window ground truth, so precision
    against a hand-placed cut mostly measures where that line was drawn."""
    out, lab = _write(tmp_path, [1.0, 2.0, 2.0, 2.0], [4, 3, 2, 1])
    keys = set(runner.score(out, lab, 1.762))
    assert not keys & {"precision", "recall", "f1", "accuracy"}


# --------------------------------------------------------------------------
# sample data integrity
# --------------------------------------------------------------------------

def test_sample_data_exists_and_has_expected_columns():
    assert SAMPLE.exists() and LABELS.exists()
    with SAMPLE.open() as fh:
        assert next(csv.reader(fh)) == ["timestamp", "vibration"]
    with LABELS.open() as fh:
        cols = next(csv.reader(fh))
    for required in ("snapshot_index", "operating_hours_to_end", "failed", "defect_mode"):
        assert required in cols


def test_sample_is_small_enough_to_avoid_the_checkpoint_abort():
    """Inputs over ~50 MiB abort in ConcatColumnsNode at a checkpoint boundary."""
    assert SAMPLE.stat().st_size < 50 * 1024 * 1024


def test_sample_spans_the_transition():
    """The slice must contain the crossing — an already-elevated tail would
    show a high score but not the thing worth seeing."""
    rows = list(csv.DictReader(LABELS.open()))
    hours = [float(r["operating_hours_to_end"]) for r in rows]
    assert max(hours) > min(hours)                     # it progresses
    snaps = {int(r["snapshot_index"]) for r in rows}
    assert min(snaps) == 0                             # renumbered to stand alone
    assert len(snaps) >= 100
    # 2 windows per snapshot at window 1024 / step 1024 over 2,048 rows
    assert len(rows) == 2 * len(snaps)
