"""Shared helpers for the RED data-prep scripts: reading, incidents, roles, channel rules.

A RED recording is one long multivariate time series with a per-row label: the
background class (`normal`) almost everywhere, and an event class (`pump_breakdown`,
...) on the rows of each incident. An **incident** is a maximal run of rows of one
event class; incidents are numbered 1, 2, ... in time order, and roles are assigned
per incident.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROLES = ("training", "validation", "test", "delivery")


def read_recording(path):
    return pd.read_parquet(path) if str(path).endswith((".parquet", ".pq")) else pd.read_csv(path)


def to_seconds(ts):
    """Epoch seconds (float) from epoch seconds, epoch milliseconds or ISO 8601 / datetimes."""
    s = pd.Series(ts)
    if pd.api.types.is_numeric_dtype(s):
        v = s.to_numpy(dtype=float)
        return v / 1000.0 if np.nanmedian(np.abs(v)) > 1e11 else v
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, utc=True, format="ISO8601")
    elif s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return (s - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def runs(mask):
    """(start, end) half-open index pairs of the True runs of a boolean array."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[::2], d[1::2]))


def parse_ids(text):
    """'1,2,4-7' -> [1, 2, 4, 5, 6, 7]; '' -> []."""
    out = []
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def load_prepared(prepared):
    report = json.load(open(os.path.join(prepared, "prepare_report.json")))
    df = pd.read_parquet(os.path.join(prepared, "recording.parquet"))
    return df, report


def channels_of(df):
    return [c for c in df.columns if c not in ("timestamp", "label", "incident", "segment")]


def assign(report, training, validation, test, delivery=()):
    """{incident id: role} with every id checked; delivery may repeat test incidents."""
    known = {i["id"] for i in report["incidents"]}
    roles = {}
    for role, ids in (("training", training), ("validation", validation), ("test", test)):
        for i in ids:
            if i not in known:
                raise SystemExit(f"incident {i} does not exist (incidents: 1..{len(known)})")
            if i in roles:
                raise SystemExit(f"incident {i} is in both {roles[i]} and {role}")
            roles[i] = role
    for i in delivery:
        if i not in known:
            raise SystemExit(f"delivery incident {i} does not exist")
        if roles.get(i) in ("training", "validation"):
            raise SystemExit(f"delivery incident {i} is a {roles[i]} incident: deliver test incidents or new ones")
    return roles


def slice_bounds(incidents, n_rows, context):
    """{id: (lo, hi)}: `context` rows either side of each incident, cut at the midpoint to
    a neighbouring incident, so no row is in two incidents' slices."""
    out = {}
    for k, inc in enumerate(incidents):
        a, b = inc["rows"]
        lo, hi = a - context, b + context
        if k > 0:
            lo = max(lo, (incidents[k - 1]["rows"][1] + a) // 2)
        if k + 1 < len(incidents):
            hi = min(hi, (b + incidents[k + 1]["rows"][0]) // 2)
        out[inc["id"]] = (max(lo, 0), min(hi, n_rows))
    return out


def build_end(report, roles, context):
    """The first row any held-out incident's slice can reach: everything before it is what a
    deployment has at build time (the training incidents and the normal around them)."""
    held = [i for i in report["incidents"] if roles.get(i["id"]) in ("validation", "test")]
    if not held:
        return report["n_rows"]
    bounds = slice_bounds(report["incidents"], report["n_rows"], context)
    return min(bounds[i["id"]][0] for i in held)


def audit_channels(df, event_mask, end, max_nan_fraction=0.01, max_nan_run=16,
                   zero_event_share=0.20, zero_normal_share=0.02):
    """Per channel, on rows [0, end): what a rule-based triage decides, and why.

    drop  `dead`      constant, or all missing
    drop  `dropout`   missing on more than max_nan_fraction of rows. When the missing rows
                      fall mostly inside events, the channel goes dark *because* of the
                      event: interpolated, the dropout itself would become the signature.
    drop  `long gap`  a missing run longer than max_nan_run rows (the `red` blueprint
                      repairs runs up to `window_interpolation_max_gap`, 16 by default)
    keep  `zero-saturation` (a warning): exactly 0 in a large share of event rows and almost
                      never otherwise. A real reading of a stopped machine, so it is kept,
                      but it makes the task "has it stopped?"
    """
    rows = {}
    ev = np.asarray(event_mask[:end], bool)
    for c in channels_of(df):
        v = df[c].to_numpy(dtype=float)[:end]
        isna = np.isnan(v)
        frac = float(isna.mean())
        na_in_events = float((isna & ev).sum() / max(isna.sum(), 1))
        na_of_event_rows = float((isna & ev).sum() / max(ev.sum(), 1))
        longest = max((b - a for a, b in runs(isna)), default=0)
        finite = v[~isna]
        r = {"nan_fraction": round(frac, 4), "nan_inside_events": round(na_in_events, 4),
             "event_rows_nan": round(na_of_event_rows, 4), "longest_nan_run": int(longest)}
        if finite.size == 0 or np.nanstd(finite) == 0:
            r["decision"], r["why"] = "drop", "dead"
        elif frac > max_nan_fraction:
            r["decision"] = "drop"
            r["why"] = "dropout tracks the events" if na_in_events > 0.5 and na_of_event_rows > 0.1 else "dropout"
        elif longest > max_nan_run:
            r["decision"], r["why"] = "drop", "long gap"
        else:
            r["decision"], r["why"] = "keep", ""
            zs_ev = float((v[ev] == 0).mean()) if ev.any() else 0.0
            zs_n = float((v[~ev] == 0).mean())
            if zs_ev >= zero_event_share and zs_n < zero_normal_share:
                r["why"] = f"zero-saturation: 0 in {zs_ev:.0%} of event rows, {zs_n:.2%} of others"
        rows[c] = r
    return rows


def separation(df, channels, train_rows, normal_rows):
    """Mean |z| of the training-event rows against the build-region normal rows, per channel."""
    out = {}
    for c in channels:
        x = df[c].interpolate(limit_direction="both").to_numpy(dtype=float)
        mu, sd = x[normal_rows].mean(), x[normal_rows].std(ddof=1) or 1.0
        out[c] = float(np.abs((x[train_rows] - mu) / sd).mean())
    return out
