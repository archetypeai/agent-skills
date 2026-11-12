#!/usr/bin/env python3
"""Compare od's result with events someone knows about. Not part of the agent: local evaluation.

    python grade.py --out out --events sample_data/known_events.csv
    python grade.py --out out --events work_orders.csv --lead-h 12

`--events` is a CSV with `start` and `end` (times on the data's clock, or epoch seconds)
and optionally `name`: maintenance reports, work orders, a shift log. The agent never sees
it. Per file, for each event inside it:
- AUROC: the chance that a window overlapping the event outscores one that doesn't (1.0
  perfect, 0.5 chance). It grades the ranking, whatever the threshold;
- the ranks the event's windows reached, and how many od flagged;
- the first flag from `--lead-h` hours before the event's start to its end: an event
  often builds before anyone records it, so a flag shortly before counts as early.
Then each episode from review.py as matching an event or not. "Not recorded" is not the
same as a false alarm: records list what someone noticed, not every unusual stretch.
Writes <out>/grade/grade.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def to_epoch(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return pd.Timestamp(x).timestamp()


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a positive outscores a negative), ties counted half."""
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--lead-h", type=float, default=24)
    a = ap.parse_args(argv)
    st = json.loads((a.out / "discover_state.json").read_text())
    review_path = a.out / "review" / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    ev = pd.read_csv(a.events)
    events = [{"name": r.get("name", f"event {i + 1}"), "start": to_epoch(r["start"]), "end": to_epoch(r["end"]),
               "start_text": str(r["start"]), "end_text": str(r["end"])} for i, r in ev.iterrows()]
    lead = a.lead_h * 3600
    result = {"lead_h": a.lead_h, "files": {}}
    for src, run in st["runs"].items():
        name = Path(src).name
        o = pd.read_csv(run["output"])
        o = o[~o.invalid.astype(str).str.lower().eq("true")].reset_index(drop=True)
        s, flagged = o.anomaly_score.to_numpy(), (o.predicted_label == "anomaly").to_numpy()
        lo, hi = o.start_timestamp.min(), o.finish_timestamp.max()
        inside = [e for e in events if e["end"] >= lo and e["start"] <= hi]
        print(f"=== {name}: {len(o)} valid windows, {len(inside)} known event(s) inside")
        rank = pd.Series(-s).rank(method="min").to_numpy().astype(int)
        fr = {"events": [], "episodes": []}
        for e in inside:
            hit = ((o.finish_timestamp >= e["start"]) & (o.start_timestamp <= e["end"])).to_numpy()
            near = o[(o.finish_timestamp >= e["start"] - lead) & (o.start_timestamp <= e["end"]) & flagged]
            first = (float(near.finish_timestamp.iloc[0]) - e["start"]) / 3600 if len(near) else None
            row = {"event": e["name"], "start": e["start_text"], "end": e["end_text"], "windows": int(hit.sum()),
                   "auroc": round(auroc(s[hit], s[~hit]), 3), "ranks": sorted(rank[hit].tolist()),
                   "flagged": int((hit & flagged).sum()), "first_flag_h": None if first is None else round(first, 2)}
            fr["events"].append(row)
            det = "not flagged" if first is None else \
                f"first flag {abs(first):.1f} h {'before' if first < 0 else 'after'} its start"
            print(f"  {e['name']} ({e['start_text']} .. {e['end_text']}): {row['windows']} windows; AUROC {row['auroc']:.3f}; "
                  f"ranks {row['ranks'][:8]}{' ...' if len(row['ranks']) > 8 else ''}; "
                  f"{row['flagged']}/{row['windows']} flagged; {det}")
        for ep in review.get(name, []):
            match = [e["name"] for e in inside if ep["end_epoch"] >= e["start"] - lead and ep["start_epoch"] <= e["end"]]
            fr["episodes"].append({**{k: ep[k] for k in ("rank", "start", "end", "windows", "peak")},
                                   "event": match[0] if match else None})
        if fr["episodes"]:
            n_hit = sum(x["event"] is not None for x in fr["episodes"])
            print(f"  episodes: {len(fr['episodes'])}, {n_hit} match a known event, {len(fr['episodes']) - n_hit} not recorded")
            for x in fr["episodes"][:5]:
                print(f"    {x['rank']}. {x['start']}  {x['windows']:3d} windows  peak {x['peak']:.2f}  "
                      f"{x['event'] or 'not recorded'}")
        result["files"][name] = fr
    (a.out / "grade").mkdir(parents=True, exist_ok=True)
    (a.out / "grade" / "grade.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {a.out}/grade/grade.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
