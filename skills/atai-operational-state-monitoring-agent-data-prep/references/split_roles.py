#!/usr/bin/env python3
"""Optional: assign roles to recordings by group, at random with a fixed seed.

Skip this if you already know which recordings go where (fill the index's `role`
column yourself). Otherwise: recordings with the same `group` (e.g. the same setting
run twice, the same asset-day) always land in the same role, so near-identical
recordings never sit on both sides of a split. Rows already marked `delivery` are
left alone: delivery is usually "another asset", chosen by you, not at random.

Groups are shuffled with --seed and dealt out by --fractions (library, validation,
test). Pick the seed once, before looking at any score, and never re-roll it. Then
check the result with check_states.py and preflight_roles.py: every scored role must
contain every state.

    python split_roles.py --index recordings.csv --fractions 0.6 0.2 0.2 --seed 20260928
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osm_common import read_index  # noqa: E402


def assign(groups, fractions, seed):
    """{group: role} for library / validation / test, by shuffled group order."""
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    total = sum(fractions)
    n_lib = round(len(names) * fractions[0] / total)
    n_val = round(len(names) * fractions[1] / total)
    roles = ["library"] * n_lib + ["validation"] * n_val + ["test"] * (len(names) - n_lib - n_val)
    return dict(zip(names, roles))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.6, 0.2, 0.2], metavar=("LIB", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", help="write here instead of overwriting --index")
    a = ap.parse_args(argv)
    idx = read_index(a.index).drop(columns=["path"])
    if "role" not in idx:
        idx["role"] = ""
    if "group" not in idx or not idx.group.str.len().any():
        idx["group"] = idx.recording            # no groups given: every recording is its own
    todo = idx.role != "delivery"
    role_of = assign(set(idx.group[todo]), a.fractions, a.seed)
    idx.loc[todo, "role"] = idx.group[todo].map(role_of)
    path = a.out or a.index
    idx.to_csv(path, index=False)
    for r in ("library", "validation", "test", "delivery"):
        m = idx.role == r
        print(f"  {r:<10} {idx.group[m].nunique():>3} groups, {m.sum():>3} recordings")
    print(f"wrote {path} (seed {a.seed})")
    return idx


if __name__ == "__main__":
    main()
