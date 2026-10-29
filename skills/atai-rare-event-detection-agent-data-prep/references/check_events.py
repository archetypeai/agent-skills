#!/usr/bin/env python3
"""Step 2: can these incidents and channels make a RED agent? Read-only, before any role file.

RED learns one prototype per class from a few labelled incidents, so a handful of
decisions carry the whole result. For the roles you plan (--training, --validation,
--test: incident ids from prepare.py), this checks, on build-time data only (rows
before the first held-out incident's slice, so no choice peeks at what it is scored on):

  incidents   every role has some; each event class has a training incident;
  windows     which incidents can fill half a window at each candidate --windows size
              (the classifier compares whole windows: an event shorter than half a window
              barely moves one), and how many windows each training class yields;
  channels    the triage build_roles.py will apply: dead channels, channels that go dark
              during events (interpolated, the dropout would become the signature),
              missing runs longer than the platform repairs; and zero-saturation, a real
              reading of a stopped machine that makes the task easy (a warning);
  separation  per kept channel, mean |z| of the training incidents against build-time
              normal, and of the held-out incidents with the same statistics: separation
              that does not transfer is a channel to distrust.

PASS / WARN / FAIL per check; exits 1 on a FAIL.

    python check_events.py --prepared prepared --training 1,2 --validation 3 --test 4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from red_common import assign, audit_channels, build_end, load_prepared, parse_ids, separation  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepared", default="prepared")
    ap.add_argument("--training", required=True, help="incident ids, e.g. 1,2")
    ap.add_argument("--validation", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--windows", default="16,32,64,128,256", help="candidate window sizes")
    ap.add_argument("--context", type=int, default=4096, help="normal rows either side of a scored incident")
    ap.add_argument("--max-nan-fraction", type=float, default=0.01)
    ap.add_argument("--max-nan-run", type=int, default=16)
    ap.add_argument("--top", type=int, default=10, help="channels build_roles.py would select")
    a = ap.parse_args(argv)
    df, rep = load_prepared(a.prepared)
    roles = assign(rep, parse_ids(a.training), parse_ids(a.validation), parse_ids(a.test))
    incidents = rep["incidents"]
    windows = [int(w) for w in a.windows.split(",")]
    fails = warns = 0

    def say(level, msg):
        nonlocal fails, warns
        fails += level == "FAIL"
        warns += level == "WARN"
        print(f"{level:<5} {msg}")

    print("incidents")
    for inc in incidents:
        print(f"  #{inc['id']:<3} {inc['class']:<18} {inc['n_rows']:>7,} rows  {roles.get(inc['id'], '-')}")
    for role in ("training", "validation", "test"):
        if not any(r == role for r in roles.values()):
            say("FAIL", f"no {role} incident")
    events = rep["classes"][1:]
    for cls in events:
        if not any(i["class"] == cls and roles.get(i["id"]) == "training" for i in incidents):
            say("FAIL", f"event class {cls!r} has no training incident: it gets no prototype")
        if not any(i["class"] == cls and roles.get(i["id"]) in ("validation", "test") for i in incidents):
            say("WARN", f"event class {cls!r} is never scored (no validation or test incident)")

    print("\nwindows")
    for w in windows:
        scored = [i for i in incidents if roles.get(i["id"]) in ("validation", "test")]
        blind = [f"#{i['id']} ({i['n_rows']} rows)" for i in scored if 2 * i["n_rows"] < w]
        shots = {cls: sum(max(0, i["n_rows"] - w + 1) for i in incidents
                          if i["class"] == cls and roles.get(i["id"]) == "training") for cls in events}
        level = "WARN" if blind else "PASS"
        if any(n == 0 for n in shots.values()):
            level = "FAIL"
        say(level, f"window {w:>4}: training windows " + ", ".join(f"{c} {n:,}" for c, n in shots.items())
            + (f"; cannot fill half a window: {', '.join(blind)}" if blind else "; every scored incident can fill half"))

    end = build_end(rep, roles, a.context)
    event_mask = df["label"].to_numpy() != rep["background"]
    print(f"\nchannels (rows 0-{end:,}: before the first held-out incident's slice)")
    audit = audit_channels(df, event_mask, end, a.max_nan_fraction, a.max_nan_run)
    for c, r in audit.items():
        if r["decision"] == "drop":
            detail = {"dead": "constant or empty",
                      "dropout tracks the events": f"missing on {r['nan_fraction']:.1%} of rows, "
                                                   f"{r['nan_inside_events']:.0%} of them inside events",
                      "dropout": f"missing on {r['nan_fraction']:.1%} of rows",
                      "long gap": f"a missing run of {r['longest_nan_run']} rows"}[r["why"]]
            say("WARN" if r["why"] == "dropout tracks the events" else "PASS",
                f"{c}: dropped, {r['why']} ({detail})")
        elif r["why"]:
            say("WARN", f"{c}: kept, {r['why']}")
    kept = [c for c, r in audit.items() if r["decision"] == "keep"]
    print(f"  {len(kept)} of {len(audit)} channels kept")

    print("\nseparation (mean |z| against build-time normal)")
    train = np.zeros(len(df), bool)
    held = np.zeros(len(df), bool)
    for inc in incidents:
        lo, hi = inc["rows"]
        if roles.get(inc["id"]) == "training":
            train[lo:hi] = True
        elif roles.get(inc["id"]) in ("validation", "test"):
            held[lo:hi] = True
    if not train.any():
        print(f"\n{fails} FAIL, {warns} WARN")
        return 1
    normal = ~event_mask
    normal[end:] = False
    sep_t = separation(df, kept, train, normal)
    sep_h = separation(df, kept, held, normal) if held.any() else {}
    top = sorted(kept, key=sep_t.get, reverse=True)[:a.top]
    for c in sorted(kept, key=sep_t.get, reverse=True):
        t, h = sep_t[c], sep_h.get(c, float("nan"))
        mark = "  <- selected" if c in top else ""
        print(f"  {c:<14} training {t:5.2f}   held-out {h:5.2f}{mark}")
        if c in top and h < 0.5 * t:
            say("WARN", f"{c}: separation does not transfer ({t:.2f} on training incidents, {h:.2f} held out)")

    print(f"\n{fails} FAIL, {warns} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
