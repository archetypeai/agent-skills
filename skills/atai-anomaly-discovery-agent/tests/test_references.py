#!/usr/bin/env python3
"""Network-free tests for the AD skill's references.

    python3 -m pytest skills/atai-anomaly-discovery-agent/tests -q

Every test here encodes a mistake that is easy to make and expensive to find:
the demo blueprint that must be copied into a project before it can run, the
teardown order the platform enforces, the empty or half-written output that still
reports `completed`, the label indexing that reads the wrong part of an asset's
life, and the scoring rule that must not be precision/recall.
"""
from __future__ import annotations

import csv
import importlib.util
import pathlib
import sys

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


@pytest.fixture(autouse=True)
def endpoint(monkeypatch):
    monkeypatch.setenv("ATAI_API_KEY", "sk_test")
    monkeypatch.setenv("ATAI_API_ENDPOINT", "https://x.test")



class FakeAPI:
    """Stands in for atai_http.request, routing by method and path and recording calls.

    Mocks the HTTP layer the runner drives, so the tests assert the lifecycle and stay
    network-free."""

    def __init__(self, routes):
        self.calls = []
        self.routes = routes

    def __call__(self, method, url, body=None, idempotent=None):
        path = url.split("/api", 1)[1]
        self.calls.append((method, path, body))
        for (m, prefix), value in self.routes.items():
            if m == method and path.startswith(prefix):
                return value(path, body) if callable(value) else value
        raise AssertionError(f"unexpected {method} {path}")


# --------------------------------------------------------------------------
# endpoints and paths
# --------------------------------------------------------------------------

def test_endpoint_reduces_to_one_root(monkeypatch):
    """Every .env convention in this repo (bare, /api, /v0.5) reduces to the same root."""
    for endpoint in ("https://x.test", "https://x.test/", "https://x.test/v0.5",
                     "https://x.test/v0.5/", "https://x.test/api",
                     "https://x.test/api/"):
        monkeypatch.setenv("ATAI_API_ENDPOINT", endpoint)
        assert runner.agents() == "https://x.test/api/agents", endpoint


def test_no_bundle_or_instance_calls():
    """Bundles and agent instances are gone: dev answers 404 for /agents/bundles* and
    /agents/instances*, and the rest of the platform drops them at its next release."""
    src = (REFS / "run_ad_agent.py").read_text()
    assert "/bundles" not in src and "/instances" not in src and "/logs" not in src


def test_demo_blueprint_is_resolved_by_key():
    """blp_ ids differ per deployment; the key resolves everywhere the demo is published."""
    src = (REFS / "run_ad_agent.py").read_text()
    assert 'DEMO_BLUEPRINT = "playground-ad"' in src
    import re
    assert re.findall(r"blp_[a-z0-9]{10,}", src) == []


# --------------------------------------------------------------------------
# project and copy
# --------------------------------------------------------------------------

def test_demo_is_copied_into_the_project_once(monkeypatch):
    """Only a blueprint in a project can be packaged and deployed, and the canonical demo
    is in none: copy it under COPY_KEY on the first run."""
    api = FakeAPI({("GET", "/agents/blueprints?"): {"data": []},
                     ("POST", "/agents/blueprints/playground-ad/copy"):
                         lambda p, b: {"id": "blp_copy", "blueprint_key": b["blueprint_key"]}})
    monkeypatch.setattr(runner, "request", api)
    assert runner.demo_copy("prj_1")["id"] == "blp_copy"
    method, path, body = api.calls[-1]
    assert body == {"project_id": "prj_1", "blueprint_key": runner.COPY_KEY}


def test_existing_copy_is_reused(monkeypatch):
    """A second copy under the same key answers 409; the lookup needs ?project_id=,
    since a key registered in a project resolves only with it."""
    api = FakeAPI({("GET", "/agents/blueprints?"): {"data": [{"id": "blp_old"}]}})
    monkeypatch.setattr(runner, "request", api)
    assert runner.demo_copy("prj_1")["id"] == "blp_old"
    assert api.calls == [("GET", f"/agents/blueprints?project_id=prj_1&key={runner.COPY_KEY}", None)]


def test_project_is_reused_by_name(monkeypatch):
    api = FakeAPI({("GET", "/projects"): {"data": [{"id": "prj_x", "name": "other"},
                                                     {"id": "prj_1", "name": runner.PROJECT}]}})
    monkeypatch.setattr(runner, "request", api)
    assert runner.project_id() == "prj_1"


def test_scoring_threshold_comes_from_the_blueprint(monkeypatch):
    """Score at the threshold the blueprint labels at, so a Path 2 blueprint scores at its own."""
    api = FakeAPI({("GET", "/agents/blueprints/blp_1"): {"document": {"values": {"threshold": 1.215}}}})
    monkeypatch.setattr(runner, "request", api)
    assert runner.blueprint_threshold("blp_1") == 1.215


# --------------------------------------------------------------------------
# run: package -> batch deployment -> session, torn down in reverse
# --------------------------------------------------------------------------

def _lifecycle(monkeypatch, status, ended, results=None):
    monkeypatch.setattr(runner, "create_package", lambda bp, name: {"id": "pkg_1"})
    monkeypatch.setattr(runner, "package_ready", lambda pid: {"id": pid, "status": "ready"})
    monkeypatch.setattr(runner, "deploy", lambda pkg, kind: {"id": "dpl_1", "kind": kind})
    monkeypatch.setattr(runner, "deployment_ready", lambda did: {"id": did, "status": "ready"})
    monkeypatch.setattr(runner, "open_session", lambda dpl, fid: {"id": "ses_1"})
    monkeypatch.setattr(runner, "poll", lambda sid: {"status": status, "error": "boom"})
    monkeypatch.setattr(runner, "last_events", lambda sid: [])
    monkeypatch.setattr(runner, "end", lambda kind, oid: ended.append((kind, oid)))
    monkeypatch.setattr(runner, "download_results", lambda sid, out, inp: results)


def test_teardown_goes_session_deployment_package(monkeypatch):
    """Each holds the next: a deployment with sessions, or a package with deployments,
    can't be deleted."""
    ended = []
    _lifecycle(monkeypatch, "completed", ended, results="out.csv")
    assert runner.run("blp_1", "f.csv", "out.csv", "in.csv", "n") == "out.csv"
    assert ended == [("sessions", "ses_1"), ("deployments", "dpl_1"), ("packages", "pkg_1")]


def test_failed_session_still_tears_down(monkeypatch):
    """A failed run must not strand a deployment and package on the platform."""
    ended = []
    _lifecycle(monkeypatch, "failed", ended)
    with pytest.raises(SystemExit) as excinfo:
        runner.run("blp_1", "f.csv", "out.csv", "in.csv", "n")
    assert "failed" in str(excinfo.value) and "boom" in str(excinfo.value)
    assert [k for k, _ in ended] == ["sessions", "deployments", "packages"]


def test_failed_package_is_still_deleted(monkeypatch):
    """A package that never becomes ready is recorded before the wait, so it is deleted too."""
    ended = []
    _lifecycle(monkeypatch, "completed", ended)

    def failed(pid):
        raise SystemExit(f"{pid} is failed, not ready: build error")

    monkeypatch.setattr(runner, "package_ready", failed)
    with pytest.raises(SystemExit):
        runner.run("blp_1", "f.csv", "out.csv", "in.csv", "n")
    assert ended == [("packages", "pkg_1")]


def test_download_reports_empty_results(monkeypatch, capsys):
    """An all-invalid run reports `completed` with NO results. That emptiness
    is the tell, not the runtime."""
    monkeypatch.setattr(runner, "request", FakeAPI({("GET", "/agents/sessions/ses_1/results"): {"data": []}}))
    assert runner.download_results("ses_1", "/tmp/unused.csv") is None
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


def _write_ad(path, finish):
    path.write_text("finish_timestamp,start_timestamp,predicted_label,invalid,anomaly_score\n"
                    f"360000.05,360000.0,normal,false,1.0\n{finish},{finish - 0.05},normal,false,1.0\n")


def test_output_must_reach_the_inputs_end(tmp_path):
    inp = tmp_path / "in.csv"
    inp.write_text("timestamp,vibration\n360000.0,0.1\n431400.099951,0.4\n")
    out = tmp_path / "out.csv"
    _write_ad(out, 431400.05)
    assert runner.output_reaches_end(str(out), str(inp))
    _write_ad(out, 400000.0)
    assert runner.output_reaches_end(str(out), str(inp)) is False


def test_downloads_again_until_complete(tmp_path, monkeypatch):
    """The platform can report completed while the output is still being written."""
    inp = tmp_path / "in.csv"
    inp.write_text("timestamp,vibration\n360000.0,0.1\n431400.099951,0.4\n")
    out = tmp_path / "out.csv"
    n = {"downloads": 0}

    def fetch(ref, out_path):
        assert ref == "/api/files/fil_1/content"
        n["downloads"] += 1
        _write_ad(out, 431400.05 if n["downloads"] >= 3 else 400000.0)

    monkeypatch.setattr(runner, "request", FakeAPI({("GET", "/agents/sessions/ses_1/results"):
        {"data": [{"data": {"filename": "o.csv", "ref": "/api/files/fil_1/content"}}]}}))
    monkeypatch.setattr(runner, "fetch", fetch)
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    assert runner.download_results("ses_1", str(out), str(inp))
    assert n["downloads"] == 3
