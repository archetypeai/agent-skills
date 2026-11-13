---
name: atai-omega-encoder-agent-data-prep
description: >
  Prepare sensor recordings for Archetype AI's Omega Encoder agent (the `omega-embeddings`
  blueprint), locally and with no API key: cut every recording into fixed-length segments,
  one per operation (a tightening, a cycle, a batch), aligned on the event that starts it,
  on an exact sample grid (gaps dropped, never interpolated); z-score from the library only;
  write one role file per segment (`timestamp` + channels and nothing else, a whole number of
  windows at every window size you will try, no two files' time ranges overlapping) with the
  labels held back and a manifest; then preflight them against every rule. Use when the user
  has recordings of repeated operations, normal ones for a library and a few labelled faults
  to choose settings and test on, and wants Omega embeddings from the Omega Encoder agent in
  batch or serving to build their own detector on (the next step is `atai-omega-encoder-agent`),
  or when a batch session's windows could not be matched back to files. Do NOT use for the
  Direct Query API (`atai-newton-omega-model-data-prep`), for role files for the AD, OSM or RED
  agents' Optimizations (`atai-anomaly-discovery-agent-data-prep`,
  `atai-operational-state-monitoring-agent-data-prep`, `atai-rare-event-detection-agent-data-prep`),
  or for video, image or text.
---

# Omega Encoder Agent Data Prep — Recordings → Role Files

The Omega Encoder agent embeds whatever numeric columns it is given, a window at a time,
and returns one 768-dimensional embedding per channel per window. It does not score, label
or check anything. So the shape of what goes in decides whether what comes out can be used:
a label column is embedded as a sensor, a segment that is not a whole number of windows
loses its tail (or, in serving, fails), and two files whose time ranges overlap can't be told
apart in a batch session's output.

This skill turns recordings into **role files** that avoid all of that:

```
index.csv ─► prepare.py ─► prepared/ ─► build_roles.py ─► roles/ ─► preflight_roles.py
(segments,   one cut per     exact grid,    z-score (library),   library/ validation/     every rule,
 roles,      segment,        fixed rows     one file per segment, test/ delivery/          PASS / FAIL
 groups,     aligned on                      labels held back,    labels/ manifest.json
 labels)     an event                        distinct times
```

The next step is [`atai-omega-encoder-agent`](../atai-omega-encoder-agent/SKILL.md): embed
the role files in batch at a few window sizes, choose a detector on validation, test once,
deliver, and serve. Worked example, the whole way: the
[AURSAD screwdriving quickstart](https://github.com/archetypeai/omega-encoder-example-aursad-quickstart).

## When to Apply

- Recordings of a **repeated operation**, one recording (or one segment of one) per
  operation: a tightening, a wash cycle, a machining pass, a batch. Each becomes one file and
  is scored as a whole.
- **Normal operations for a library**, and a few labelled faults to choose settings on
  (validation), report once (test) and score afterwards (delivery).
- Recordings whose clocks **restart** (per session, run, day) or **jitter**, with the odd
  skipped sample.

## Your input

An index CSV, one row per segment:

| column | |
|---|---|
| `segment` | a unique id, used in file names |
| `file` | the recording's path (CSV or Parquet), relative to the index: a `timestamp` column in seconds, then numeric channels |
| `role` | `library` · `validation` · `test` · `delivery` |
| `group` | what must stay in one role: a recording session, a run, an asset-day |
| `label` | `normal`, or the fault's name. The library must be all `normal` |

**Split by group, in time.** Faults recorded in campaigns (one kind this week, another the
next) make a random split learn the session instead of the fault. The shipped sample keeps
AURSAD's recording sessions apart: library 0–4, validation 5–8, test 9–13, delivery 14–18.

## The steps (on the shipped sample)

The sample is 100 AURSAD tightenings (CC BY 4.0, [NOTICE](references/sample_data/NOTICE.md)):
a UR3e robot with an OnRobot screwdriver at 100 Hz, 12 channels, each file one whole
tightening as recorded (its own length, its clock's jitter).

### 1. `prepare.py`: one cut per segment, on an exact grid

```bash
cd skills/atai-omega-encoder-agent-data-prep/references
python prepare.py --index sample_data/index.csv --out prepared --rows 1024 \
    --onset-channel torque_current --onset-above 0.05 --pre-rows 256
```

- **The cut:** `--rows` rows, starting `--pre-rows` before the first row where
  `--onset-channel` rises above `--onset-above`, so every tightening lines up on the moment
  the screwdriver starts. Without an onset channel, from the first row.
- **The step:** the mean of the regular timestamp differences, to 4 significant figures, or
  `--step` when the clock jitters by more than a few percent of a step.
- **Continuity:** a step more than `--max-step` (10) times the step, forward or back, is a gap
  or a restart, and the segment is dropped, never interpolated. A skipped or repeated sample
  is absorbed: timestamps are re-gridded (t0 + i × step), values are kept as recorded. How far
  the clock sat off the grid is written as `max_offset_steps`.

**Expect:** `kept 88 of 100 segments, 1024 rows each`, and 12 `too short` drops.

**Look at what was dropped.** Here they're mostly **missing-screw** tightenings: without a
screw the screwdriver stops early, so the sample is short. A cut that is too long silently
removes the fault you most want to catch. Check the drops per class, and if one class
dominates them, use fewer `--rows` (a smaller multiple of your windows) or fewer
`--pre-rows` (more of the cut after the onset).

### 2. `build_roles.py`: the role files

```bash
python build_roles.py --prepared prepared --out roles --windows 1024 512 256
```

- **Z-score with the library's mean and std**, every channel, so nothing about the scored
  roles leaks into the scale.
- **One file per segment**, `roles/<role>/<role>_<segment>.csv`: exactly `timestamp` + the
  channels.
- **Distinct time ranges.** A batch session's outputs are matched to files by each window's
  start timestamp. Recorders restart their clocks, so segments overlap in time; when any do,
  every segment is re-based to its own slot (segment i starts at i × 2 × rows × step), its grid
  unchanged. The original start stays in the manifest as `source_start`.
- **Labels held back** in `roles/labels/<role>.json`, read only by whatever scores that role.
  The library has none.
- **`manifest.json`:** channels, step, rows, candidate windows, the z-score, the groups in each
  role, and every file with its start and end. The agent skill's scripts read it.

**Expect:** `library 32, validation 19, test 20, delivery 17 files`, `timestamps re-based`.

### 3. `preflight_roles.py`: every rule

```bash
python preflight_roles.py --roles roles --k 10
```

**Expect:** `11 PASS, 0 WARN, 0 FAIL` over 528 file checks. Exit status is non-zero on FAIL.

## The rules, and what breaks

| rule | what happens otherwise |
|---|---|
| **Exactly `timestamp` + channels** | the agent embeds every numeric column but the first: a label, id or index column becomes a sensor, and leaks the answer into the embeddings |
| **`rows` a multiple of every window** | batch drops the tail; a serving query shorter than its session's window fails with a 503, not a clear error |
| **An exact grid** | the agent's windows count rows; a jittered clock makes a "10.24 s" window cover a different time in every file |
| **No overlapping time ranges** | batch outputs are matched to files by window start; overlapping files' windows can't be told apart |
| **Distinct file names** | an upload replaces a file of the same name, including one an earlier session still reads |
| **Library: normal only, z-score from it alone** | an anomaly in it is embedded as normal, and a scale fitted on the scored roles leaks them into the library |
| **Enough library windows for k** | the k nearest neighbours of a tiny library are mostly noise |
| **A group in one role** | leakage: the same session's quirks in the library and the test make test scores optimistic |
| **Both classes in validation and test** | AUROC and a false-alarm threshold need both |

## Pitfalls

- **A long cut drops the short faults** (Step 1). Check the drops per class.
- **Aligning on an event matters for scoring by position.** The agent skill's detector can
  compare each window only with library windows at the same position in their segment. That
  works only if segments start at the same point of the operation.
- **Don't re-grid across a gap.** A pause or a restart inside a cut is a different take: drop
  the segment rather than stitch it.
- **Z-score before the windows are chosen, not after.** The same role files serve every
  window size; nothing here depends on the window except the divisibility check.

## Local Setup

```bash
cd skills/atai-omega-encoder-agent-data-prep/references
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, pyarrow
```

No API key. Run the tests:

```bash
pip install pytest
pytest skills/atai-omega-encoder-agent-data-prep/tests/ -v
```

They build synthetic segments (jitter, an onset, a gap, a clock that jumps back, a skipped
sample, clocks that all start at 0), run the whole prep, and then **break the role files on
purpose**: a label column in a file, a short file, two files with the same time range, a group
in two roles; the preflight must fail each one.

## File Layout

```
atai-omega-encoder-agent-data-prep/
├── SKILL.md
├── references/
│   ├── prepare.py            # Step 1: cut, align, grid
│   ├── build_roles.py        # Step 2: z-score, role files, labels, manifest
│   ├── preflight_roles.py    # Step 3: every rule
│   ├── oe_common.py
│   ├── requirements.txt
│   └── sample_data/          # 100 AURSAD tightenings + index.csv + NOTICE.md (CC BY 4.0)
└── tests/
    └── test_oe_prep.py
```

## Data attribution

The sample is derived from AURSAD (Leporowski, Tola, Hansen, Iosifidis; Aarhus University,
Technicon ApS), [doi:10.5281/zenodo.4487073](https://doi.org/10.5281/zenodo.4487073),
CC BY 4.0; every change is listed in [sample_data/NOTICE.md](references/sample_data/NOTICE.md).
