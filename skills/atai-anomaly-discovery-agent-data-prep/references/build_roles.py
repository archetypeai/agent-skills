#!/usr/bin/env python3
"""Step 3: build the role files the AD platform receives, per the contract in role-files.md.

From the prepared recordings (prepare.py) and the index's roles:

  zscore_stats.json   per-channel mean / std over the LIBRARY recordings only
  library/            one file per recording segment, original timestamps, no label column:
                      Optimize labels every training row `normal` by a constant. One file
                      per segment keeps every window: joined, the windows across each
                      boundary would be lost
  validation/, test/  the role's normal-only recordings joined into <role>_normal.csv, its
                      incident recordings into <role>_incidents_NN.csv (--per-file
                      recordings each), with a `label` column. Recordings are joined in
                      index order with a --seam-s gap between them, so the platform marks
                      the windows across a boundary invalid (abstained) instead of mixing
                      two recordings
  delivery/           the same, without `label`; plus `calibration.csv`, an unlabelled copy
                      of validation's normal-only file, delivered so the scorer can set a
                      false-alarm-rate threshold from scores of known-normal data
  labels/             every scored file's sidecar: timestamp, label, recording, type
  manifest.json       every file, its recordings, row counts, the channels and settings

Every CSV: `timestamp` (epoch seconds, 3 decimals), the channels z-scored (--decimals),
then `label` where the role has one.

    python build_roles.py --index recordings.csv --prepared prepared --out roles
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_common import channels_of, load_prepared, read_index, stem, timestamp_strings  # noqa: E402


def zscore_stats(prepared, library):
    n, s1, s2 = 0, 0.0, 0.0
    for rec in library:
        df = load_prepared(prepared, rec)
        channels = channels_of(df)
        x = df[channels].to_numpy(np.float64)
        n, s1, s2 = n + len(x), s1 + x.sum(0), s2 + (x ** 2).sum(0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    std[std < 1e-12] = 1.0                       # a constant channel stays 0, never NaN
    return {"channels": channels, "mean": mean.round(8).tolist(), "std": std.round(8).tolist(), "rows": int(n),
            "from": "all rows of the library recordings", "recordings": sorted(library)}


def write_csv(d, path, channels, mean, std, label, decimals):
    out = pd.DataFrame({"timestamp": timestamp_strings(d.timestamp.to_numpy())})
    z = (d[channels].to_numpy(np.float64) - mean) / std
    for j, c in enumerate(channels):
        out[c] = z[:, j]
    if label:
        out["label"] = d.label.astype(str).to_numpy()
    out.to_csv(path, index=False, float_format=f"%.{decimals}f")


def joined(prepared, recs, types, seam_ms):
    """The recordings' segments in order, re-timed so each starts seam_ms after the last ended."""
    parts, t_next = [], None
    for rec in recs:
        df = load_prepared(prepared, rec)
        for _, seg in df.groupby("segment", sort=True):
            seg = seg.copy()
            if t_next is not None:
                seg["timestamp"] = seg.timestamp - seg.timestamp.iloc[0] + t_next
            seg["recording"], seg["type"] = rec, types.get(rec, "")
            t_next = int(seg.timestamp.iloc[-1]) + seam_ms
            parts.append(seg)
    return pd.concat(parts, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--prepared", default="prepared")
    ap.add_argument("--out", default="roles")
    ap.add_argument("--per-file", type=int, default=21, help="incident recordings per scored file")
    ap.add_argument("--seam-s", type=float, default=86400, help="timestamp gap between joined recordings")
    ap.add_argument("--decimals", type=int, default=4)
    a = ap.parse_args(argv)
    idx = read_index(a.index)
    by_role = {r: list(idx.recording[idx.role == r]) for r in ("library", "validation", "test", "delivery")}
    if not by_role["library"]:
        raise SystemExit("no library recordings")
    types = dict(zip(idx.recording, idx.type))
    report = json.load(open(os.path.join(a.prepared, "prepare_report.json")))
    seam_ms = int(a.seam_s * 1000)

    for d in ("library", "validation", "test", "delivery", "labels"):
        shutil.rmtree(os.path.join(a.out, d), ignore_errors=True)
        os.makedirs(os.path.join(a.out, d))
    stats = zscore_stats(a.prepared, by_role["library"])
    channels, mean, std = stats["channels"], np.array(stats["mean"]), np.array(stats["std"])
    with open(os.path.join(a.out, "zscore_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    manifest = {"contract": "ad-role-files/v1", "step_ms": report["step_ms"], "channels": channels,
                "timestamp_column": "timestamp", "label_column": "label", "labels": ["normal", "anomaly"],
                "timestamp_format": "epoch seconds, 3 decimals", "decimals": a.decimals, "seam_s": a.seam_s,
                "zscore_stats": "zscore_stats.json", "library": [], "validation": [], "test": [], "delivery": []}
    for rec in by_role["library"]:
        df = load_prepared(a.prepared, rec)
        if (df.label != "normal").any():
            raise SystemExit(f"library recording {rec} has non-normal rows: the detector would fit them as normal")
        for seg, part in df.groupby("segment", sort=True):
            fname = f"{stem(rec)}__seg{int(seg)}.csv"
            write_csv(part, os.path.join(a.out, "library", fname), channels, mean, std, False, a.decimals)
            manifest["library"].append({"file": f"library/{fname}", "recording": rec, "segment": int(seg),
                                        "rows": len(part)})
    print(f"library     {len(manifest['library'])} files from {len(by_role['library'])} recordings, "
          f"{sum(f['rows'] for f in manifest['library']):,} rows, all normal")

    def write_scored(role, name, recs, labelled, calibration=False):
        d = joined(a.prepared, recs, types, seam_ms)
        fname = f"{name}.csv"
        write_csv(d, os.path.join(a.out, role, fname), channels, mean, std, labelled, a.decimals)
        lab = f"labels/{name}_labels.csv"
        pd.DataFrame({"timestamp": timestamp_strings(d.timestamp.to_numpy()), "label": d.label.astype(str),
                      "recording": d.recording, "type": d["type"]}).to_csv(os.path.join(a.out, lab), index=False)
        entry = {"file": f"{role}/{fname}", "labels": lab, "recordings": list(recs), "rows": len(d),
                 "normal": int((d.label == "normal").sum()), "anomaly": int((d.label == "anomaly").sum())}
        if calibration:
            entry["calibration"] = True
        manifest[role].append(entry)
        return entry

    for role in ("validation", "test", "delivery"):
        recs = by_role[role]
        has_anomaly = {r: bool((load_prepared(a.prepared, r).label == "anomaly").any()) for r in recs}
        normal_only = [r for r in recs if not has_anomaly[r]]
        incidents = [r for r in recs if has_anomaly[r]]
        if role == "delivery":   # one stream of recordings, as they would arrive
            for i in range(0, len(recs), a.per_file):
                write_scored(role, f"delivery_{i // a.per_file + 1:02d}", recs[i:i + a.per_file], False)
            continue
        if normal_only:
            write_scored(role, f"{role}_normal", normal_only, True)
        for i in range(0, len(incidents), a.per_file):
            write_scored(role, f"{role}_incidents_{i // a.per_file + 1:02d}", incidents[i:i + a.per_file], True)
    val_normal = [f for f in manifest["validation"] if f["anomaly"] == 0]
    if val_normal:
        write_scored("delivery", "calibration", val_normal[0]["recordings"], False, calibration=True)
    for role in ("validation", "test", "delivery"):
        fs = manifest[role]
        print(f"{role:<11} {len(fs)} files from {sum(len(f['recordings']) for f in fs)} recordings, "
              f"{sum(f['normal'] for f in fs):,} normal / {sum(f['anomaly'] for f in fs):,} anomaly rows"
              + ("  (labels held back in labels/)" if role == "delivery" else ""))
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {a.out}/ (manifest.json); check it with: python preflight_roles.py --roles {a.out}")
    return manifest


if __name__ == "__main__":
    main()
