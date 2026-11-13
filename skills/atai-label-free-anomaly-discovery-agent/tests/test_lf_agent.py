"""Network-free tests for the Label-Free Anomaly Discovery agent scripts.

    pytest skills/atai-label-free-anomaly-discovery-agent/tests/ -v
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references")
sys.path.insert(0, REF)
import discover  # noqa: E402
import grade  # noqa: E402
import od_api  # noqa: E402
import review  # noqa: E402

SAMPLE = Path(REF) / "sample_data" / "metropt3_2020-07-08_2020-07-16.csv"


# --- the request ---------------------------------------------------------------

def test_values_default_to_every_column_and_no_overlap():
    v = discover.build_values(["timestamp", "a", "b"], "timestamp", None, 360, None, {})
    assert v == {"timestamp_column": "timestamp", "data_columns": ["a", "b"], "window_size": 360, "step_size": 360}
    v = discover.build_values(["t", "a", "b"], "t", "b", 64, 32, {"threshold": 2.0})
    assert v["data_columns"] == ["b"] and v["step_size"] == 32 and v["threshold"] == 2.0


@pytest.mark.parametrize("header,ts,chans,window", [
    (["time", "a"], "timestamp", None, 360),      # no timestamp column
    (["timestamp", "a"], "timestamp", "z", 360),  # unknown channel
    (["timestamp", "a"], "timestamp", None, 8),   # under Omega's 16
    (["timestamp", "a"], "timestamp", None, 2048),
])
def test_values_reject_bad_input(header, ts, chans, window):
    with pytest.raises(SystemExit):
        discover.build_values(header, ts, chans, window, None, {})


def input_file(path, n, step=10, gap_after=None):
    t = 1_600_000_000 + np.arange(n) * step
    if gap_after:
        t[gap_after:] += 3600
    pd.DataFrame({"timestamp": t, "a": np.arange(n, dtype=float)}).to_csv(path, index=False)
    return path


def od_output(path, src, values, rows=None):
    """What od writes: a row per window tiled from row 0."""
    ts = pd.read_csv(src).timestamp.to_numpy()
    w, s = values["window_size"], values["step_size"]
    n = (len(ts) - w) // s + 1 if rows is None else rows
    idx = np.arange(n) * s
    pd.DataFrame({"finish_timestamp": ts[idx + w - 1].astype(float), "start_timestamp": ts[idx].astype(float),
                  "predicted_label": "normal", "invalid": "false", "anomaly_score": 1.0}).to_csv(path, index=False)
    return path


def test_output_check_counts_windows_and_coverage(tmp_path):
    v = discover.build_values(["timestamp", "a"], "timestamp", None, 100, None, {})
    src = input_file(tmp_path / "in.csv", 1050)
    assert discover.expected_windows(src, v)[0] == 10
    assert discover.check_output(od_output(tmp_path / "ok.csv", src, v), src, v) == ""
    assert "expected 10" in discover.check_output(od_output(tmp_path / "short.csv", src, v, rows=7), src, v)
    v2 = {**v, "step_size": 50}
    assert discover.expected_windows(src, v2)[0] == 20


# --- credentials ----------------------------------------------------------------

@pytest.fixture
def fresh_env(monkeypatch, tmp_path):
    monkeypatch.setattr(od_api, "_LOADED", False)
    monkeypatch.setattr(od_api, "_CHECKED", False)
    for k in ("ATAI_API_KEY", "ATAI_API_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_shell_wins_over_dotenv_which_is_found_upwards(fresh_env, monkeypatch):
    (fresh_env / ".env").write_text('ATAI_API_KEY="sk-from-dotenv"\nATAI_API_ENDPOINT=https://x.example/api\n')
    sub = fresh_env / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    monkeypatch.setenv("ATAI_API_KEY", "sk-from-shell")
    assert od_api.root() == "https://x.example"           # not in the shell: from the .env
    assert os.environ["ATAI_API_KEY"] == "sk-from-shell"


def test_a_create_post_is_not_retried(fresh_env, monkeypatch):
    monkeypatch.setenv("ATAI_API_KEY", "sk-1")
    monkeypatch.setenv("ATAI_API_ENDPOINT", "https://x.example")
    calls = []

    def boom(req, timeout):
        calls.append(req.get_method())
        raise TimeoutError("lost")
    monkeypatch.setattr(od_api.urllib.request, "urlopen", boom)
    monkeypatch.setattr(od_api.time, "sleep", lambda s: None)
    with pytest.raises(TimeoutError):
        od_api.request("POST", "/agents/sessions", {"deployment_id": "dpl_1"})
    assert calls == ["POST"]
    with pytest.raises(TimeoutError):
        od_api.request("GET", "/agents/sessions/ses_1", retries=2)
    assert calls == ["POST", "GET", "GET", "GET"]


def test_endpoint_forms_and_placeholder(fresh_env, monkeypatch):
    monkeypatch.setenv("ATAI_API_KEY", "sk-1")
    for ep in ["https://x.example", "https://x.example/", "https://x.example/api", "https://x.example/v0.5"]:
        monkeypatch.setenv("ATAI_API_ENDPOINT", ep)
        assert od_api.root() == "https://x.example"
    monkeypatch.setenv("ATAI_API_KEY", "<your API key>")
    with pytest.raises(SystemExit):
        od_api.root()


def test_result_ref_is_joined_to_the_root(fresh_env, monkeypatch):
    monkeypatch.setenv("ATAI_API_KEY", "sk-1")
    monkeypatch.setenv("ATAI_API_ENDPOINT", "https://x.example/api")
    seen = []

    def fake(method, path, body=None, raw=False, retries=4):
        seen.append(path)
        return {"data": [{"data": {"ref": "/api/files/fil_1/content"}}]} if not raw else b"csv"
    monkeypatch.setattr(od_api, "request", fake)
    assert od_api.download_result("ses_1") == b"csv"
    assert seen[-1] == "https://x.example/api/files/fil_1/content"      # not /api/api/...


# --- post-processing ------------------------------------------------------------

def scored(flags, scores, step_s=3600):
    n = len(flags)
    start = 1_600_000_000 + np.arange(n) * step_s
    return pd.DataFrame({"start_timestamp": start.astype(float), "finish_timestamp": (start + step_s - 10).astype(float),
                         "predicted_label": np.where(flags, "anomaly", "normal"), "invalid": "false",
                         "anomaly_score": scores})


def test_episodes_merge_back_to_back_windows_only():
    o = scored([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], np.ones(10))
    assert review.episodes(o, max_gap_s=10) == [[1, 2], [4, 5, 6], [9]]


def test_auroc():
    assert grade.auroc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert grade.auroc(np.array([1.0]), np.array([1.0])) == 0.5
    assert np.isnan(grade.auroc(np.array([]), np.array([1.0])))
    assert grade.to_epoch("1970-01-01 00:01") == 60 and grade.to_epoch("120") == 120


def test_review_and_grade_on_a_synthetic_run(tmp_path, capsys):
    n, w = 60, 6                                     # 60 windows of 6 rows at 600 s: 1 h each
    t = 1_600_000_000 + np.arange(n * w) * 600
    x = np.random.default_rng(0).normal(0, 1, (n * w, 2))
    x[30 * w:33 * w, 0] += 8                         # an event in windows 30-32
    src = tmp_path / "in.csv"
    pd.DataFrame({"timestamp": t, "a": x[:, 0], "b": x[:, 1]}).to_csv(src, index=False)
    flags = np.zeros(n, bool)
    flags[30:33] = True
    flags[2] = True                                  # one flag 28 h before: outside --lead-h 24
    scores = np.where(flags, 3.0, 1.0)
    scores[30] = 5.0
    scores[2] = 2.0
    out = tmp_path / "out"
    (out / "discover").mkdir(parents=True)
    o = scored(flags, scores)
    o["start_timestamp"] = t[::w].astype(float)
    o["finish_timestamp"] = t[w - 1::w].astype(float)
    o.to_csv(out / "discover" / "od-in.csv", index=False)
    (out / "discover_state.json").write_text(json.dumps({
        "values": {"timestamp_column": "timestamp", "data_columns": ["a", "b"], "window_size": w, "step_size": w},
        "runs": {str(src): {"output": str(out / "discover" / "od-in.csv")}}}))
    assert review.main(["--out", str(out)]) == 0
    eps = json.loads((out / "review" / "review.json").read_text())["in.csv"]
    assert [e["windows"] for e in eps] == [3, 1] and eps[0]["stands_out"].startswith("a level +")
    ev = tmp_path / "events.csv"
    pd.DataFrame({"name": ["incident"], "start": [str(pd.Timestamp(t[30 * w], unit="s"))],
                  "end": [str(pd.Timestamp(t[33 * w - 1], unit="s"))]}).to_csv(ev, index=False)
    assert grade.main(["--out", str(out), "--events", str(ev)]) == 0
    g = json.loads((out / "grade" / "grade.json").read_text())["files"]["in.csv"]
    assert g["events"][0]["auroc"] == 1.0 and g["events"][0]["flagged"] == 3
    assert [e["event"] for e in g["episodes"]] == ["incident", None]


def test_sample_is_a_valid_od_input():
    d = pd.read_csv(SAMPLE)
    assert list(d.columns) == ["timestamp", "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
                               "Motor_current"]
    assert (np.diff(d.timestamp) % 10 == 0).all() and np.isfinite(d.iloc[:, 1:].to_numpy()).all()
    v = discover.build_values(list(d.columns), "timestamp", None, 360, None, {})
    assert discover.expected_windows(SAMPLE, v)[0] == 156
