---
name: atai-operational-state-monitoring-agent-data-prep
description: >
  Prepare labelled sensor recordings for Archetype AI's Operational State
  Monitoring (OSM) agent, locally and with no API key: put every recording on an
  exact sample grid (split at gaps, never interpolate across one), attach one
  state per row, check that the sensor can actually tell the states apart, split
  recordings into library / validation / test / delivery by group, z-score from
  the library only, and write the role files the OSM platform accepts (one
  training file per state, continuous scored files, held-back delivery labels,
  a manifest), then preflight them against the platform's rules. Use when the
  user has labelled time-series recordings and wants to train, test and deploy an
  OSM agent on them (the next step is `atai-operational-state-monitoring-agent`,
  Path 2), or when an OSM optimization or eval failed or silently never started.
  Do NOT use for running a maintained OSM bundle on a single CSV
  (`atai-operational-state-monitoring-agent`, Path 1), for client-side embedding
  + KNN (`atai-newton-omega-model-data-prep`), or for video / image / text.
---

# OSM Data Prep — Labelled Recordings → Role Files

The OSM agent learns operational states (drilling / tripping, fill / wash / spin /
drain, …) from **labelled recordings** and classifies every window of new ones. Before
anything reaches the platform, the recordings have to become **role files**: a small,
strict layout the Optimizations and Evals APIs accept. Most ways to get it wrong don't
error: an optimization with too many training files **never starts**, one scored window
across a time gap **fails the whole trial**, and a state missing from the test set
**scores 0**. This skill builds the layout and checks every rule before you upload.

```
recordings (any rate, gaps, jitter)          role files (the contract)
  + a per-row state label        prepare.py   ─┐
  + recordings.csv (index)   →   check_states │→  build_roles.py  →  preflight_roles.py  →  OSM agent skill, Path 2
                                 split_roles  ─┘
```

Everything here is **local and needs no API key**. The contract itself is in
[`references/role-files.md`](references/role-files.md); the platform rules in it were
measured on the dev deployment (2026-09-30) while building the LARCO washing-machine
examples ([quickstart](https://github.com/archetypeai/osm-agent-example-larco-quickstart),
[full](https://github.com/archetypeai/osm-agent-example-larco)).

## When to Apply

- You have **labelled recordings** (one state per row, or per stretch) from one or more
  assets, and want an OSM agent that recognises those states
- An OSM **optimization stays `pending`** forever, or **fails** with `eval-mode test data
  must not contain windows a validation node rejected` — preflight your role files
- You want to know **whether your states are separable** from the sensor before spending
  platform time on them

**Do not use this skill when:**
- You only want to run the maintained Volve bundle on one CSV — that's
  [`atai-operational-state-monitoring-agent`](../atai-operational-state-monitoring-agent/SKILL.md), Path 1
- You're embedding windows yourself over `/query` for a local KNN — use
  [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md)
  (its `FeaturePreparer` has no role here: the platform embeds)
- Your raw data first needs heavy cleaning (imputing short dropouts, diagnosing sensors):
  do that with [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md)'s
  `DataPreprocessor`, then come back here

## Your input

**One file per recording** (CSV or Parquet) with a timestamp column, the sensor channels
(numeric), and a **state label per row**:

```
timestamp,back.x,back.y,back.z,...,label
1689787212.103,0.5312,-0.0911,0.8401,...,fill
```

- **Timestamps:** epoch seconds, epoch milliseconds or ISO 8601 — any rate, jitter and
  gaps are fine; step 1 fixes them.
- **Labels:** turning your source's annotations into one state per row is the
  dataset-specific part, and it's yours. (In LARCO, the washing machine's own spin,
  water-flow and heater labels become fill / wash / spin / drain by a priority rule —
  see the [quickstart's `prep/states.py`](https://github.com/archetypeai/osm-agent-example-larco-quickstart/blob/main/prep/states.py).)

**An index,** `recordings.csv`:

| column | |
|---|---|
| `recording` | a unique id, used in file names |
| `file` | the recording's path, relative to the index |
| `role` | `library` · `validation` · `test` · `delivery` — or leave empty and run `split_roles.py` |
| `group` | optional: recordings that must stay in one role (the same setting run twice, one asset-day, …) |

**Roles:** the **library** trains the model; **validation** chooses its settings;
**test** is scored **once**, at the end; **delivery** is what you'd hand a customer
(e.g. another asset), run without labels and scored afterwards against the held-back ones.

## The steps (on the shipped sample)

[`references/sample_data/`](references/sample_data/) holds five short LARCO washing-machine
cycles (9 accelerometer channels at ~200 Hz, raw jittered timestamps, a state per row,
9 MB) and a filled-in `recordings.csv`. Everything below runs on it in about 10 s:

```bash
cd skills/atai-operational-state-monitoring-agent-data-prep/references
pip install -r requirements.txt
S=sample_data/recordings.csv
```

### 1. `prepare.py` — an exact grid, one state per row

```bash
python prepare.py --index $S --out prepared --hz 200 --abs-max 2.2
```

Per recording: read; drop rows with a missing value; remove impossible samples
(`--abs-max`); cut into segments where samples are more than `--max-gap-s` apart (default
1 s — **never interpolate across a gap**); resample each segment onto an exact `1000/hz`
ms grid by **cubic spline** in time relative to the segment start (linear dulls
high-frequency content: in LARCO it kept 82–84 % of a 46 Hz tone where cubic kept
97–99 %); label each grid row by the last raw sample at or before it; drop segments
shorter than `--min-rows`.

```
  becken_warm_15-min_40_2                     188,566 rows  1 segment(s)  raw 201.204 Hz, 0 gap(s) > 1.0 s
  ...
5 recordings, 1,032,568 rows -> prepared/
```

**Why an exact grid:** the platform checks each window's sample rate against its own mean
interval (`sample_rate_interval_tolerance`, default 0.05, *relative*). Jittered raw
timestamps pass or fail unpredictably; a gap inside a window fails it outright.

### 2. `check_states.py` — can the sensor see your states?

```bash
python check_states.py --index $S --prepared prepared --window 512
```

Leave-one-recording-out kNN on simple window features (per-channel log RMS + a 16-band
spectrum), training classes balanced. Per state: the recall on recordings it never saw,
and where the misses go.

```
  ok    drain            recall 0.70  (207 windows)  → mostly wash (20%)
  ok    fill             recall 0.94  (356 windows)  → mostly wash (6%)
  WARN  spin             recall 0.42  (167 windows)  → mostly wash (39%)
  ok    wash             recall 0.84  (410 windows)  → mostly spin (10%)
```

**Read it before committing to a state list.** A state the sensor can't see caps every
model's macro-F1 at (n−1)/n. In LARCO, "heating" (the heater on during wash) scored 0.62
balanced accuracy against wash — near chance — and was folded into wash; the four-state
agent then worked. But a low recall can also mean **the recordings differ**: here the
sample's library mixes two machines, and the second spins harder, so its spin looks like
nothing in the first — the same cross-unit effect the full LARCO example saw on delivery.
Look at which recordings the misses come from before merging a state.

### 3. `split_roles.py` — optional: roles by group, at random

```bash
python split_roles.py --index recordings.csv --fractions 0.6 0.2 0.2 --seed 20260928
```

Skip it if you assign roles yourself (the sample's index already has them). It keeps each
`group` in one role and leaves `delivery` rows alone. **Fix the seed before you look at
any score, and never re-roll it** — re-rolling until the test looks good leaks the test.

### 4. `build_roles.py` — the role files

```bash
python build_roles.py --index $S --prepared prepared --out roles --window 1024 --per-state 40
```

```
library   drain__library.csv                  40 windows, 32 pieces from 2 recordings
library   fill__library.csv                   40 windows, 40 pieces from 2 recordings
library   spin__library.csv                   40 windows, 4 pieces from 2 recordings
library   wash__library.csv                   40 windows, 40 pieces from 2 recordings
validation   1 files from 1 recordings, 193 windows of 1024 rows
test        1 files from 1 recordings, 183 windows of 1024 rows
delivery    1 files from 1 recordings, 203 windows of 1024 rows
```

- **z-score** from the library recordings only (never validation, test or delivery).
- **Library:** `--per-state` windows of `--window` rows per state, spread evenly over the
  library recordings that have the state and evenly within each, written as **one file
  per state** of continuous single-state pieces with real timestamps. A two-recording
  library runs out of spin at ~42 windows here, hence `--per-state 40`; the build warns
  when a state falls short. (The LARCO examples used 100 and 400 with 8 and 54 cycles.)
- **Validation / test / delivery:** one continuous file per recording segment; delivery
  without `label`, its labels held back in `delivery_labels/`.
- **`--window`** is the library cut; the platform window (chosen later in the Optimize
  search) should divide it (1024 → 256 / 512 / 1024), so piece seams fall on window
  boundaries.

### 5. `preflight_roles.py` — check every rule

```bash
python preflight_roles.py --roles roles --prepared prepared --index $S
```

Per file: header, timestamp format, **an exact grid** (no jump in a scored file; in a
library file, forward jumps only where a piece starts), finite values, labels, the delivery
sidecar row for row; with `--prepared`, every library piece is its file's state and every
value matches the prepared data z-scored. Per role: manifest vs disk, **no recording or
group in two roles**, the z-score from the library only, per-state library balance,
**every state in validation, the search subset, test and delivery**, and the **config size**
against the platform's 1 MiB limit. Exit 1 on any FAIL.

```
  PASS  grid                   7 pass, 0 warn, 0 fail
  ...
  PASS  all states: test
  PASS  config size            ~3 KiB for 5 examples
RESULT: PASS, ready for the OSM agent skill (Path 2)
```

Without `--prepared` (e.g. role files unpacked from an archive), the two checks that need
the prepared data are skipped and everything else runs.

Then: [`atai-operational-state-monitoring-agent`](../atai-operational-state-monitoring-agent/SKILL.md),
**Path 2** — Optimize → promote → test once with Evals → deliver → score.

## The rules, and what breaks

The full contract, with the manifest schema, is [`references/role-files.md`](references/role-files.md).
The ones that fail silently or expensively:

| rule | what happens otherwise |
|---|---|
| **One training file per state** (pieces inside) | the whole optimization config lives in one **1 MiB** ConfigMap, ~636 B per training example → ~1,500 files at most; over it the job **never dispatches, with no error** — trials stay `pending` |
| **No time jump in a scored file** | one scored window across a jump **fails the whole trial or eval**; loosening the sampling tolerance doesn't help, and the search space can't turn the check off |
| **Every state in every scored set** | a missing state counts as **F1 = 0** in the platform's macro-F1 |
| **Library-only z-score; a recording (group) in one role** | leakage: validation and test scores come out optimistic |
| **Choose on validation, test once** | re-choosing after seeing the test number makes it optimistic |

## Pitfalls

- **Balanced library, unbalanced world.** Every state gets the same number of library
  windows, but in real recordings some states are rare (LARCO's drain: 3 % of the time).
  kNN then over-predicts the rare states — high recall, low precision. Expect it; judge
  per-state precision as well as recall.
- **Short recordings are fine if they cover the states.** LARCO's 15-minute programs hold
  as much fill, spin and drain as a 3-hour cotton cycle — the long programs are long in
  one state (wash). Choose recordings by state coverage, not length.
- **Labels at a lower rate than the signal** (LARCO: 1 Hz labels, 200 Hz vibration): label
  each sample by the label period it falls in, and drop samples outside any labelled period
  before step 1 — the sample data was built that way.
- **Don't renumber timestamps to hide gaps.** A window across a faked seam silently mixes two
  unrelated moments. Split instead; the library's real-timestamp pieces are the one place
  jumps are allowed, because training skips the windows that cross them.

## Local Setup

```bash
cd skills/atai-operational-state-monitoring-agent-data-prep/references
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, pyarrow, scipy
```

No API key. Run the tests:

```bash
pip install pytest
pytest skills/atai-operational-state-monitoring-agent-data-prep/tests/ -v
```

They build synthetic recordings (jitter, a gap, three timestamp formats), run the whole prep,
and then **break the role files on purpose** — a jump in a scored file, a jump inside a
library piece, a missing state, a piece of the wrong state, wrong scaling, a bad timestamp
format, a recording in two roles — and require the preflight to fail each one. They also
check that `check_states.py` flags a state that shares its signal with another.

## File Layout

```
skills/atai-operational-state-monitoring-agent-data-prep/
├── SKILL.md                  ← this file
├── references/
│   ├── role-files.md         ← the contract: layout, rules and why, manifest schema
│   ├── osm_common.py         ← index, recordings, timestamps (s / ms / ISO)
│   ├── prepare.py            ← 1. exact grid, split at gaps, one state per row
│   ├── check_states.py       ← 2. can the sensor tell the states apart?
│   ├── split_roles.py        ← 3. optional: roles by group, fixed seed
│   ├── build_roles.py        ← 4. role files + manifest
│   ├── preflight_roles.py    ← 5. every platform rule, before any upload
│   ├── requirements.txt      ← numpy, pandas, pyarrow, scipy
│   └── sample_data/
│       ├── recordings.csv    ← the index, roles filled in
│       ├── *.parquet         ← 5 short LARCO cycles, raw timestamps, a state per row
│       └── README.md         ← provenance + CC BY 4.0 attribution
└── tests/
    └── test_osm_prep.py      ← synthetic end-to-end + deliberately broken role files (pytest)
```
