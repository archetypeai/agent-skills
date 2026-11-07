#!/usr/bin/env python3
"""Your own anomaly detector (kNN) over the Omega Encoder agent's embeddings.

    python detect.py select --roles roles [--groups groups.json]   # choose on validation, freeze
    python detect.py test --roles roles                             # score test once (batch)
    python detect.py deliver --roles roles [--source serving]       # score the delivery afterwards

The protocol, fixed before anything is scored:
- A segment (`rows` rows) is embedded as rows / W windows of W rows. A window's joint state
  is its channels' 768-d embeddings, each L2-normalised, concatenated; optionally PCA,
  fitted on the library's windows.
- A window's score is its mean distance to the k nearest library windows: any library
  window (match "any"), or only those at the same position in their segment (match
  "position"; meaningful when segments are aligned on an event). A segment's score is its
  worst window.
- `select` tries every window size embedded for both library and validation, every channel
  group (--groups: {"name": [channels]}; default one group of every channel), PCA, k and
  match, keeps the best validation AUROC (normal vs every anomaly), then sets the threshold
  to a FAR_TARGET (5%) false-alarm rate on validation's normal segments: data the library
  has not seen. 1% and 2% thresholds are reported beside it. All frozen in
  <eval>/settings.json.
- `test` and `deliver` read settings.json and never change it.

Labels come from <roles>/labels/<role>.json, read only by the command that scores that role.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

FAR_TARGET = 0.05
FAR_REPORTED = (0.01, 0.02, 0.05)
GRID = {"pca": [16, 32, 64, 0], "k": [1, 3, 5, 10]}


class Paths:
    def __init__(self, roles: str, out: str, eval_dir: str):
        self.roles, self.out, self.eval = roles, out, eval_dir
        self.manifest = json.load(open(os.path.join(roles, "manifest.json")))

    def embeddings(self, role: str, window: int, source: str = "batch") -> str:
        kind = "serve" if source == "serving" else "embed"
        return os.path.join(self.out, kind, f"w{window}", f"{role}.npz")


def joint(emb: np.ndarray, channels: list[str], use: list[str]) -> np.ndarray:
    """[segments, windows, channels, 768] -> joint states [segments, windows, d]."""
    e = emb[:, None] if emb.ndim == 3 else emb
    e = e[:, :, [channels.index(c) for c in use]]
    e = e / np.maximum(np.linalg.norm(e, axis=3, keepdims=True), 1e-8)
    return e.reshape(e.shape[0], e.shape[1], -1)


def load(p: Paths, role: str, window: int, use: list[str], source: str = "batch") -> tuple[list[str], np.ndarray]:
    path = p.embeddings(role, window, source)
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run " + ("serve.py" if source == "serving"
                                              else f"embed_batch.py --role {role} --window {window}"))
    z = np.load(path)
    if int(z["window"]) != window:
        sys.exit(f"{path} was embedded at window {int(z['window'])}, expected {window}")
    return [str(f) for f in z["files"]], joint(z["emb"], p.manifest["channels"], use)


def labels(p: Paths, role: str, files: list[str]) -> np.ndarray:
    ops = json.load(open(os.path.join(p.roles, "labels", f"{role}.json")))["operations"]
    lab = {o["file"]: o["class"] for o in ops}
    return np.asarray([lab[f] for f in files])


class Detector:
    """kNN over library windows; a segment scores as its worst window."""

    def __init__(self, lib: np.ndarray, pca: int, k: int, match: str = "any"):
        flat = lib.reshape(-1, lib.shape[-1])
        self.mean = flat.mean(0)
        self.proj = np.linalg.svd(flat - self.mean, full_matrices=False)[2][:pca].T if pca else None
        self.lib, self.k, self.match = self._project(lib), k, match

    def _project(self, x: np.ndarray) -> np.ndarray:
        x = x - self.mean
        return x @ self.proj if self.proj is not None else x

    @staticmethod
    def _knn(x: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
        d = np.sqrt(np.maximum((x ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None] - 2 * x @ ref.T, 0))
        return np.sort(d, axis=1)[:, :k].mean(1)

    def window_scores(self, x: np.ndarray) -> np.ndarray:
        x = self._project(x)
        n, w, _ = x.shape
        if self.match == "position":
            return np.stack([self._knn(x[:, j], self.lib[:, j], self.k) for j in range(w)], 1)
        return self._knn(x.reshape(n * w, -1), self.lib.reshape(-1, self.lib.shape[-1]), self.k).reshape(n, w)

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.window_scores(x).max(1)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report(cls: np.ndarray, score: np.ndarray, thresholds: dict) -> dict:
    normal = score[cls == "normal"]
    out = {"normal": int(len(normal)),
           "false_alarm_rate": {k: round(float((normal > t).mean()), 4) if len(normal) else None
                                for k, t in thresholds.items()}, "types": {}}
    anomalous = cls != "normal"
    out["overall"] = {"segments": int(anomalous.sum()), "auroc": round(auroc(score[anomalous], normal), 4),
                      "detected": {k: round(float((score[anomalous] > t).mean()), 4) if anomalous.any() else None
                                   for k, t in thresholds.items()}}
    for c in sorted(set(cls) - {"normal"}):
        s = score[cls == c]
        out["types"][c] = {"segments": int(len(s)), "auroc": round(auroc(s, normal), 4),
                           "detected": {k: round(float((s > t).mean()), 4) for k, t in thresholds.items()}}
    return out


def show(role: str, r: dict) -> None:
    fmt = lambda v: f"{v:>8.1%}" if v is not None else f"{'—':>8}"  # noqa: E731
    print(f"{role}: thresholds set on validation for " + " ".join(f"{k:>8}" for k in r["false_alarm_rate"]))
    print(f"  {'false alarms (normal)':<26}{r['normal']:>4}    " + " ".join(fmt(v) for v in r["false_alarm_rate"].values()))
    for name, t in [("all anomalies", r["overall"]), *r["types"].items()]:
        print(f"  {name:<26}{t['segments']:>4}    " + " ".join(fmt(v) for v in t["detected"].values())
              + f"   AUROC {t['auroc']:.3f}")


def describe(c: dict) -> str:
    return (f"window {c['window']}, channels {c['group']}, PCA {c['pca'] or 'none'}, k {c['k']}"
            + (f", match {c['match']}" if c.get("windows", 1) > 1 else ""))


def cmd_select(p: Paths, groups: dict) -> None:
    rows = p.manifest["rows"]
    windows = [w for w in p.manifest.get("candidate_windows", [rows])
               if all(os.path.exists(p.embeddings(r, w)) for r in ("library", "validation"))]
    if not windows:
        sys.exit("no window has both library and validation embedded: run embed_batch.py first")
    trials = []
    for window, (g, use) in itertools.product(windows, groups.items()):
        _, lib = load(p, "library", window, use)
        files, x = load(p, "validation", window, use)
        cls = labels(p, "validation", files)
        for pca, k, match in itertools.product(GRID["pca"], GRID["k"], ["any", "position"] if rows // window > 1 else ["any"]):
            if (pca and pca >= min(lib.shape[0] * lib.shape[1], lib.shape[2])) or k >= lib.shape[0] * (1 if match == "position" else lib.shape[1]):
                continue
            s = Detector(lib, pca, k, match).score(x)
            trials.append({"window": window, "windows": rows // window, "group": g, "pca": pca, "k": k, "match": match,
                           "auroc": auroc(s[cls != "normal"], s[cls == "normal"])})
    best = max(trials, key=lambda t: t["auroc"])
    use = groups[best["group"]]
    lib_files, lib = load(p, "library", best["window"], use)
    files, x = load(p, "validation", best["window"], use)
    cls = labels(p, "validation", files)
    s = Detector(lib, best["pca"], best["k"], best["match"]).score(x)
    normal = s[cls == "normal"]
    thresholds = {f"far_{int(f * 100)}pct": float(np.quantile(normal, 1 - f)) for f in FAR_REPORTED}
    cfg = {"chosen_on": "validation", **{k: best[k] for k in ("window", "windows", "group", "pca", "k", "match")},
           "channels": use, "validation_auroc": round(best["auroc"], 4), "far_target": FAR_TARGET,
           "threshold": thresholds[f"far_{int(FAR_TARGET * 100)}pct"], "thresholds": thresholds,
           "library_segments": len(lib_files), "windows_tried": windows}
    os.makedirs(p.eval, exist_ok=True)
    json.dump(cfg, open(os.path.join(p.eval, "settings.json"), "w"), indent=1)
    json.dump(trials, open(os.path.join(p.eval, "selection.json"), "w"), indent=1)
    r = report(cls, s, thresholds)
    json.dump({"settings": cfg, **r}, open(os.path.join(p.eval, "validation.json"), "w"), indent=1)
    print(f"library {len(lib_files)} segments, validation {len(files)}; windows {windows}; {len(trials)} settings tried")
    for w, g in itertools.product(windows, groups):
        b = max((t for t in trials if t["window"] == w and t["group"] == g), key=lambda t: t["auroc"], default=None)
        if b:
            print(f"  window {w:>5}  {g:<20} best AUROC {b['auroc']:.3f}  ({describe(b).split(', ', 2)[2]})")
    print(f"chosen and frozen: {describe(cfg)} (validation AUROC {best['auroc']:.3f}); threshold "
          f"{cfg['threshold']:.4f} = {FAR_TARGET:.0%} false alarms on validation's normal segments")
    show("validation", r)
    print(f"wrote {os.path.join(p.eval, 'settings.json')}")


def frozen(p: Paths) -> dict:
    path = os.path.join(p.eval, "settings.json")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run detect.py select first")
    return json.load(open(path))


def frozen_detector(p: Paths, cfg: dict) -> Detector:
    _, lib = load(p, "library", cfg["window"], cfg["channels"])
    return Detector(lib, cfg["pca"], cfg["k"], cfg["match"])


def cmd_score(p: Paths, role: str, source: str) -> None:
    cfg = frozen(p)
    files, x = load(p, role, cfg["window"], cfg["channels"], source)
    cls = labels(p, role, files)
    s = frozen_detector(p, cfg).score(x)
    r = report(cls, s, cfg["thresholds"])
    name = role if source == "batch" else f"{role}_{source}"
    json.dump({"settings": cfg, "source": source, **r}, open(os.path.join(p.eval, f"{name}.json"), "w"), indent=1)
    pd.DataFrame({"file": files, "class": cls, "score": s}).to_csv(os.path.join(p.eval, f"scores_{name}.csv"), index=False)
    print(f"settings (frozen by select): {describe(cfg)}, threshold {cfg['threshold']:.4f}; embeddings from {source}")
    show(role, r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["select", "test", "deliver"])
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--eval", default="eval_out")
    ap.add_argument("--groups", help='JSON file {"name": [channels]}; default: one group of every channel')
    ap.add_argument("--source", choices=["batch", "serving"], default="batch")
    a = ap.parse_args()
    p = Paths(a.roles, a.out, a.eval)
    groups = json.load(open(a.groups)) if a.groups else {"all": p.manifest["channels"]}
    unknown = sorted({c for cs in groups.values() for c in cs} - set(p.manifest["channels"]))
    if unknown:
        sys.exit(f"--groups names channels not in the manifest: {unknown}")
    if a.command == "select":
        cmd_select(p, groups)
    else:
        cmd_score(p, "test" if a.command == "test" else "delivery", a.source)


if __name__ == "__main__":
    main()
