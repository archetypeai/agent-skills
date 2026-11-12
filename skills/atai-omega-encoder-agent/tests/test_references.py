"""Offline tests for the Omega Encoder agent reference scripts: no API key, no network.

    pytest skills/atai-omega-encoder-agent/tests/ -v
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references")
sys.path.insert(0, REF)
import atai_api as A  # noqa: E402
import detect as D  # noqa: E402
import embed_batch as E  # noqa: E402
import serve as S  # noqa: E402

CH = ["a", "b", "c"]


def emb(n, w, shift=0.0, seed=0):
    """[n, w, channels, 768]: normal segments share a direction per window position."""
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(99).normal(0, 1, (1, w, len(CH), 768))
    return (base + shift + rng.normal(0, 0.3, (n, w, len(CH), 768))).astype(np.float32)


def test_knn_matches_brute_force():
    rng = np.random.default_rng(1)
    lib, x = rng.normal(size=(20, 4, 16)), rng.normal(size=(5, 4, 16))
    det = D.Detector(lib, 0, 3, "any")
    ref = lib.reshape(-1, 16) - det.mean
    for i in range(5):
        for j in range(4):
            d = np.sort(np.linalg.norm(ref - (x[i, j] - det.mean), axis=1))[:3].mean()
            assert np.isclose(det.window_scores(x)[i, j], d)
    pos = D.Detector(lib, 0, 1, "position")
    d = np.linalg.norm((lib[:, 2] - pos.mean) - (x[0, 2] - pos.mean), axis=1).min()
    assert np.isclose(pos.window_scores(x)[0, 2], d)
    assert np.allclose(det.score(x), det.window_scores(x).max(1))


def test_joint_normalises_each_channel():
    e = emb(3, 2)
    j = D.joint(e, CH, ["a", "c"])
    assert j.shape == (3, 2, 2 * 768)
    assert np.allclose(np.linalg.norm(j[..., :768], axis=-1), 1)


@pytest.fixture
def workspace(tmp_path):
    roles, out = tmp_path / "roles", tmp_path / "out"
    (roles / "labels").mkdir(parents=True)
    m = {"channels": CH, "rows": 1024, "step_s": 0.01, "candidate_windows": [1024, 256], "files": []}
    json.dump(m, open(roles / "manifest.json", "w"))
    files = {"library": [f"library_{i}.csv" for i in range(30)],
             "validation": [f"validation_{i}.csv" for i in range(16)]}
    classes = ["normal"] * 10 + ["fault"] * 6
    json.dump({"operations": [{"file": f, "class": c} for f, c in zip(files["validation"], classes)]},
              open(roles / "labels" / "validation.json", "w"))
    for w, per in ((1024, 1), (256, 4)):
        d = out / "embed" / f"w{w}"
        d.mkdir(parents=True)
        lib = emb(30, per, seed=1)
        val = np.concatenate([emb(10, per, seed=2), emb(6, per, shift=0.6 if w == 256 else 0.05, seed=3)])
        for role, e in (("library", lib), ("validation", val)):
            np.savez(d / f"{role}.npz", files=np.asarray(files[role]), emb=e, window=np.asarray(w))
    return D.Paths(str(roles), str(out), str(tmp_path / "eval"))


def test_select_chooses_the_separating_window_and_freezes_a_threshold(workspace):
    D.cmd_select(workspace, {"all": CH})
    cfg = json.load(open(os.path.join(workspace.eval, "settings.json")))
    assert cfg["window"] == 256 and cfg["validation_auroc"] > 0.9
    _, x = D.load(workspace, "validation", 256, CH)
    s = D.frozen_detector(workspace, cfg).score(x)
    assert np.isclose((s[:10] > cfg["threshold"]).mean(), 0.0, atol=0.11)   # ~5% of 10 normals
    assert cfg["thresholds"]["far_1pct"] >= cfg["thresholds"]["far_5pct"]


def test_map_windows_back_to_files_by_start():
    files = [{"path": "r/x_1.csv", "start": 100.0}, {"path": "r/x_2.csv", "start": 200.0}]
    rows = []
    for f in files:
        for k in range(4):
            rows.append({"start_timestamp": f["start"] + k * 2.56, **{c: [f["start"] + k] * 768 for c in CH}})
    rng = np.random.default_rng(0)
    out = E.map_windows(list(rng.permutation(rows)), files, 256, 0.01, CH, 4)
    assert out.shape == (2, 4, 3, 768)
    assert out[1, 3, 0, 0] == 203.0                       # file 2, window 4
    with pytest.raises(E.Unfinished):                     # the run drops that session and goes on
        E.map_windows(rows[:-1], files, 256, 0.01, CH, 4)


def test_a_create_post_is_not_retried_and_a_get_is(monkeypatch):
    monkeypatch.setenv("ATAI_API_KEY", "sk-1")
    calls = []

    def boom(req, timeout):
        calls.append(req.get_method())
        raise TimeoutError("lost")
    monkeypatch.setattr(A.urllib.request, "urlopen", boom)
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit):
        A.request("POST", "https://x.test/api/agents/deployments", body={"kind": "serving"})
    assert calls == ["POST"]
    with pytest.raises(SystemExit):
        A.request("GET", "https://x.test/api/agents/sessions/ses_1", retries=2)
    assert calls == ["POST", "GET", "GET", "GET"]


def test_the_shell_wins_over_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text('ATAI_API_KEY="sk-from-dotenv"\nATAI_API_ENDPOINT=https://x.example\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATAI_API_KEY", "sk-from-shell")
    monkeypatch.delenv("ATAI_API_ENDPOINT", raising=False)
    A.load_dotenv()
    assert os.environ["ATAI_API_KEY"] == "sk-from-shell"
    assert os.environ["ATAI_API_ENDPOINT"] == "https://x.example"


def test_windows_arg(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"window": 256}))
    assert E.windows_arg(["1024", "512", "frozen", "256"], 1024, str(tmp_path)) == [1024, 512, 256]
    with pytest.raises(SystemExit):
        E.windows_arg(["300"], 1024, str(tmp_path))


def test_query_body_and_answer_parsing():
    df = pd.DataFrame({"timestamp": np.arange(8) * 0.01, "a": np.arange(8.0), "b": -np.arange(8.0)})
    body = S.query_body(df, ["a", "b"])
    ev = body["events"][0]
    assert ev["type"] == "data.numeric_array" and ev["event_data"]["contents"][1][3] == -3.0
    answer = {"response": {"data": [{"start_timestamp": 4, "channel_0": [2] * 768, "channel_1": [3] * 768},
                                    {"start_timestamp": 0, "channel_0": [0] * 768, "channel_1": [1] * 768}]}}
    w = S.windows_of(answer, 2)
    assert w.shape == (2, 2, 768) and w[0, 1, 0] == 1 and w[1, 0, 0] == 2   # sorted by start
    with pytest.raises(ValueError):
        S.windows_of({"response": {"data": []}}, 2)


@pytest.mark.parametrize("endpoint", ["https://x.example/api", "https://x.example/", "https://x.example/v0.5"])
def test_api_base_normalises(monkeypatch, endpoint):
    monkeypatch.setenv("ATAI_API_ENDPOINT", endpoint)
    assert A.api_base() == "https://x.example/api"
