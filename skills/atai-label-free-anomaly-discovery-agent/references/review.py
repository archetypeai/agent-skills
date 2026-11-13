#!/usr/bin/env python3
"""Turn od's scores into episodes a person can read. Not part of the agent: local post-processing.

    python review.py --out out
    python review.py --out out --top 10

od returns a score and a label per window. For each file discover.py ran, this script
1. merges od's flagged windows (score over its threshold) into episodes when one ends
   where the next begins, ranked by their highest score;
2. describes each episode from the input file: for every channel, how far the episode's
   typical level (mean) and spread (std) sit from the file's usual window, as a robust z
   (median and interquartile range over the file's valid windows), and shows the few that
   stand out most. A description of the data, not od's reasoning: od does not say which
   channels drove a score.
No label is read. Writes <out>/review/review.md and review.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def episodes(o: pd.DataFrame, max_gap_s: float) -> list[list[int]]:
    """Indices of flagged windows, grouped when one starts within `max_gap_s` of the previous one's end."""
    out, cur = [], []
    for i in np.flatnonzero((o.predicted_label == "anomaly").to_numpy()):
        if cur and o.start_timestamp[i] - o.finish_timestamp[cur[-1]] > max_gap_s:
            out.append(cur)
            cur = []
        cur.append(int(i))
    return out + ([cur] if cur else [])


def window_features(d: pd.DataFrame, chans: list[str], window: int, step: int, n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        w = d.iloc[i * step:i * step + window]
        rows.append({f"{c} level": w[c].mean() for c in chans} | {f"{c} spread": w[c].std() for c in chans})
    return pd.DataFrame(rows)


def robust_z(feats: pd.DataFrame, valid: np.ndarray, idx: list[int]) -> pd.Series:
    base = feats[valid]
    iqr = (base.quantile(.75) - base.quantile(.25)).replace(0, np.nan)
    return ((feats.loc[idx].median() - base.median()) / iqr).dropna()


def when(ts: float) -> str:
    return pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d %H:%M")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--top", type=int, default=5, help="episodes shown per file")
    ap.add_argument("--show", type=int, default=3, help="features shown per episode")
    a = ap.parse_args(argv)
    st = json.loads((a.out / "discover_state.json").read_text())
    v = st["values"]
    ts_col, chans, window, step = v["timestamp_column"], v["data_columns"], v["window_size"], v["step_size"]
    md, js = ["# Episodes flagged by the Label-Free Anomaly Discovery agent", "",
              "Post-processing of the agent's output (not part of the agent).", ""], {}
    for src, run in st["runs"].items():
        o = pd.read_csv(run["output"])
        d = pd.read_csv(src).rename(columns={ts_col: "timestamp"})
        dt = float(np.median(np.diff(d.timestamp.to_numpy())))
        valid = ~o.invalid.astype(str).str.lower().eq("true").to_numpy()
        feats = window_features(d, chans, window, step, len(o))
        eps = sorted(episodes(o, dt), key=lambda e: -o.anomaly_score[e].max())
        name = Path(src).name
        head = (f"{name}: {int(valid.sum())} windows scored; {int((o.predicted_label == 'anomaly').sum())} "
                f"flagged in {len(eps)} episodes")
        print(f"=== {head}")
        md += [f"## {head}", "", "| # | from | to | windows | peak score | what stands out (robust z vs the usual window) |",
               "|---|---|---|---|---|---|"]
        js[name] = []
        for k, e in enumerate(eps, 1):
            z = robust_z(feats, valid, e)
            top = z.reindex(z.abs().sort_values(ascending=False).index)[:a.show]
            stand = ", ".join(f"{f} {x:+.0f}" for f, x in top.items())
            row = {"rank": k, "start": when(o.start_timestamp[e[0]]), "end": when(o.finish_timestamp[e[-1]]),
                   "start_epoch": float(o.start_timestamp[e[0]]), "end_epoch": float(o.finish_timestamp[e[-1]]),
                   "windows": len(e), "peak": round(float(o.anomaly_score[e].max()), 3), "stands_out": stand}
            js[name].append(row)
            if k <= a.top:
                print(f"  {k}. {row['start']} -> {row['end']}  {len(e):3d} windows  peak {row['peak']:.2f}  {stand}")
                md.append(f"| {k} | {row['start']} | {row['end']} | {len(e)} | {row['peak']:.2f} | {stand} |")
        if len(eps) > a.top:
            print(f"  ... {len(eps) - a.top} more in review.json")
        md.append("")
    (a.out / "review").mkdir(parents=True, exist_ok=True)
    (a.out / "review" / "review.md").write_text("\n".join(md) + "\n")
    (a.out / "review" / "review.json").write_text(json.dumps(js, indent=1))
    print(f"wrote {a.out}/review/review.md and review.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
