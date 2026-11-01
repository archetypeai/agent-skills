---
name: atai-anomaly-discovery-agent-data-prep
description: >
  Prepare sensor recordings for Archetype AI's Anomaly Discovery (AD) agent, locally
  and with no API key: put every recording on an exact sample grid (split at gaps,
  never interpolate across one), check that the normal-only library is a trustworthy
  picture of normal (purity, stability, agreement, enough windows for the detector),
  z-score from the library only, and write the role files the AD platform accepts
  (one library file per normal recording, validation and test files that join
  recordings with a gap the platform rejects, delivery files with the labels held
  back, a calibration file of known-normal data, a manifest), then preflight them
  against every rule. Use when the user has normal-operation recordings (and a few
  recorded incidents to choose settings and test on) and wants to fit, test and deploy
  an AD agent on them (the next step is `atai-anomaly-discovery-agent`, Path 2), or when
  an AD optimization or eval failed or scored oddly. Do NOT use for running the
  maintained AD Quick Start bundle on one CSV (`atai-anomaly-discovery-agent`, Path 1),
  for labelled operating states (`atai-operational-state-monitoring-agent-data-prep`),
  or for client-side embedding + KNN (`atai-newton-omega-model-data-prep`).
---

# AD Data Prep — Recordings → Role Files

The AD agent is fitted on **normal operation only** and flags whatever no longer looks
like it. Before anything reaches the platform, the recordings have to become **role
files**: a small, strict layout the Optimizations and Evals APIs accept. The ways to get
it wrong rarely error. A library with one drifting recording quietly widens "normal"; a
library too thin for the detector is silently clamped; two recordings joined without a
real gap are scored as one; a test set without incidents scores macro-F1 against a
missing class. This skill builds the layout and checks every rule before you upload.

```
recordings (any rate, gaps, jitter)
  + a per-row label (normal / anomaly)   prepare.py → check_baseline.py → build_roles.py → preflight_roles.py → AD agent skill, Path 2
  + recordings.csv (index)
```

Everything here is **local and needs no API key**. The contract is
[`references/role-files.md`](references/role-files.md). It is the layout of the
[TEP quickstart](https://github.com/archetypeai/anomaly-discovery-agent-example-tep-quickstart),
generalised; the platform rules in it were measured on the dev deployment (2026-10-12 →
10-13) building the IMS bearings and Tennessee Eastman examples.

## When to Apply

- You have **normal-operation recordings** of an asset or process, and want an AD agent
  that flags departures from them
- You have **a few recorded incidents** (or can label where past ones began) to choose
  the settings on and to test once; the detector itself never sees them
- An AD optimization or eval **scored oddly** (a class at 0, every window invalid, a
  suspicious false-alarm rate) and you want to check the role files

**Do not use this skill when:**
- You only want to run the maintained AD Quick Start bundle on one CSV — that's
  [`atai-anomaly-discovery-agent`](../atai-anomaly-discovery-agent/SKILL.md), Path 1
- Your labels are **operating states** (fill / wash / spin, drilling / tripping) — use
  [`atai-operational-state-monitoring-agent-data-prep`](../atai-operational-state-monitoring-agent-data-prep/SKILL.md)
- Your raw data first needs heavy cleaning (imputing dropouts, diagnosing sensors) — do
  that with [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md),
  then come back here

## Your input

**One file per recording** (CSV or Parquet): a timestamp column, the sensor channels
(numeric), and a per-row **`label`**: `normal`, or `anomaly` from where the incident starts.

```
timestamp,xmeas_1,xmeas_2,...,xmv_11,label
1700000000,0.25038,3674.0,...,22.21,normal
```

- **Timestamps:** epoch seconds, epoch milliseconds or ISO 8601 — any rate, jitter and gaps
  are fine; step 1 fixes them.
- **Labels:** where an incident starts is the dataset-specific part, and it's yours. The
  library needs no judgement: every row is `normal`. For run-to-failure data, where
  degradation has no sharp start, a label is a cut you place; score detection and false
  alarms, not just window precision (the AD agent skill's Path 2 scorer does).

**An index,** `recordings.csv`:

| column | |
|---|---|
| `recording` | a unique id, used in file names |
| `file` | the recording's path, relative to the index |
| `role` | `library` · `validation` · `test` · `delivery` |
| `group` | optional: recordings that must stay in one role (one asset-day, one run, …) |
| `type` | optional: the incident type (a fault, a failure mode); the delivery is scored per type |

**Roles:** the **library** is normal operation, and the only data the detector is fitted
on. **Validation** chooses the settings and sets the alarm threshold: some normal-only
recordings and some incidents. **Test** is scored **once**, at the end. **Delivery** is
what you would hand over (more operation, another asset), run without labels and scored
afterwards against the held-back ones.

There is no random role splitter here (unlike the OSM prep skill): the roles are not
interchangeable — the library must be normal, validation and test must hold incidents —
so you assign them. Keep a `group` in one role, and decide before looking at any score.

## The steps (on the shipped sample)

[`references/sample_data/`](references/sample_data/) holds 35 runs of the Tennessee
Eastman Process (33 variables every 3 minutes, a label per row, 4.8 MB) and a filled-in
`recordings.csv`: 10 fault-free runs as the library; then validation, test and delivery,
each with fault-free runs and one run of each of six faults. The same data and split as
the TEP quickstart, smaller. Everything below runs on it in about 3 s:

```bash
cd skills/atai-anomaly-discovery-agent-data-prep/references
pip install -r requirements.txt
S=sample_data/recordings.csv
```

### 1. `prepare.py` — an exact grid, one label per row

```bash
python prepare.py --index $S --out prepared --step-ms 180000
```

Per recording: read; drop rows with a missing value; optionally remove impossible
samples (`--abs-max`); cut into segments where samples are more than `--max-gap-s` apart
(default three steps — **never interpolate across a gap**); resample each segment onto an
exact `--step-ms` grid starting at its first sample (linear, or `--method cubic`); label
each grid row by the last raw sample at or before it; drop segments shorter than
`--min-rows`. Data already on such a grid comes out unchanged, as the sample does.

```
  normal_25h_run01                  500 rows  1 segment(s)  (normal 500)
  ...
35 recordings, 29,000 rows, 33 channels -> prepared/
```

**Why an exact grid:** the `ad` blueprint checks each window's sample rate against its own
mean interval (`sample_rate_interval_tolerance`, default 0.05, relative). Jittered
timestamps pass or fail unpredictably; a gap inside a window invalidates it.

### 2. `check_baseline.py` — is the library a trustworthy picture of normal?

```bash
python check_baseline.py --index $S --prepared prepared --window 48 --step 8
```

```
  PASS  purity       10 library recordings, every row `normal`
  PASS  stability    no recording drifts between halves beyond its peers (mean shift <= 0.50 sd, RMS change within ±25%)
  PASS  agreement    every recording's RMS within 25% of the median 0.988
  PASS  coverage     570 library windows at window 48, step 8 support k 20 and PCA 20 without clamping
RESULT: PASS (0 fail, 0 warn)
```

- **Purity** (FAIL): a non-normal row in the library is fitted as normal.
- **Stability** (WARN): a recording whose second half drifts from its first — run-in, a
  level shift, early degradation. Measured against the library's own variation, not a fixed
  number: healthy TEP runs change their RMS by up to ±23% between halves on their own, so a
  fixed threshold would flag good data. On the IMS bearings the same idea caught a level
  shift at a rig restart, and the baseline was moved past it.
- **Agreement** (WARN): one recording far from the rest stretches "normal" for all.
- **Coverage** (FAIL): the detector's `n_neighbors` and `pca_components` need more windows
  than that; the platform clamps both silently. Pass the largest `--k` / `--pca` you plan
  to search.

Only you can say whether a flagged recording is still acceptable operation: check it,
then keep it, cut it, or move its start.

### 3. `build_roles.py` — the role files

```bash
python build_roles.py --index $S --prepared prepared --out roles
```

```
library     10 files from 10 recordings, 5,000 rows, all normal
validation  2 files from 11 recordings, 5,760 normal / 4,800 anomaly rows
test        2 files from 7 recordings, 1,920 normal / 4,800 anomaly rows
delivery    2 files from 12 recordings, 6,720 normal / 4,800 anomaly rows  (labels held back in labels/)
```

- **z-score** from the library recordings only (never validation, test or delivery). With
  `--scale-by COLUMN`, once per value of that index column (an operating condition), each
  recording by its own condition's library statistics: see **Several operating levels** below.
- **Library:** one file per recording segment, no label column. Joined, the windows
  across each boundary would be lost; separate, each recording stays identifiable.
- **Validation / test:** the role's normal-only recordings joined into `<role>_normal.csv`,
  its incidents into `<role>_incidents_NN.csv` (`--per-file`, default 21), with a `label`
  column and a one-day gap (`--seam-s`) between recordings, and between the segments of a
  recording split at a gap: the platform abstains on the windows across it instead of
  mixing two recordings or bridging a gap.
- **Delivery:** the role's recordings as they would arrive, no label column, plus
  `calibration.csv` (validation's normal-only file again, unlabelled): the deployed agent
  scores it, and the scorer sets a false-alarm-rate threshold from those scores.
- **`labels/`:** every scored file's sidecar (timestamp, label, recording, type).

### 4. `preflight_roles.py` — check every rule

```bash
python preflight_roles.py --roles roles --prepared prepared --index $S --window 48 --step 8
```

Per file: manifest vs disk, header, timestamp format, finite values, an exact grid inside
each recording segment and a rejected gap between joined ones, labels against their sidecar row
for row, no label in library or delivery. Per role: both classes in validation and test,
no recording in two roles (calibration excepted), the z-score from the library only; with
`--index`, no group in two roles; with `--prepared`, every library row normal; with
`--window/--step`, library coverage; the config size against the platform's 1 MiB limit.

```
  PASS  manifest vs disk       16 files
  ...
  PASS  both classes: test     normal 1,920 / anomaly 4,800 rows
  PASS  library coverage       570 windows at window 48, step 8; k 20 / PCA 20 need 21
  PASS  config size            ~7 KiB for 12 examples (limit 1 MiB)
RESULT: PASS, 0 fail(s), 0 warning(s); ready for the AD agent skill (Path 2)
```

Then: [`atai-anomaly-discovery-agent`](../atai-anomaly-discovery-agent/SKILL.md),
**Path 2** — Optimize → promote → test once with Evals → deliver → score.

## The rules, and what breaks

The full contract, with the manifest schema: [`references/role-files.md`](references/role-files.md).
The ones that fail silently or expensively:

| rule | what happens otherwise |
|---|---|
| **Library: normal only** | an incident in it is fitted as normal, and the detector stops flagging that kind of change |
| **Enough library windows** for `n_neighbors` / `pca_components` | no error: both are silently clamped to the number of windows minus one, and you score a different model than the one you asked for |
| **A real gap between joined recordings** | a gap within 5% of the step passes the sampling-rate check, and windows splice two recordings |
| **Both classes in validation and test** | macro-F1 scores a missing class 0 |
| **Library-only z-score; a recording (group) in one role** | leakage: validation and test scores come out optimistic |
| **Choose on validation, test once** | re-choosing after seeing the test number makes it optimistic |

## Pitfalls

- **Pick the window from the data's time scale.** The `ad` blueprint's default window is
  1,024 samples; at TEP's 3-minute sampling that is 51 hours, longer than any run, so the
  defaults produce no valid window at all. Pass `--window` / `--step` to the checks as you
  plan to search them.
- **Every incident type in validation, or not?** The sample's validation holds one run of
  every fault type, like the TEP examples, so its test has no type the settings never saw.
  For an "unseen incidents" test, leave some types out of validation and report them
  separately; the detector itself never sees any incident either way.
- **Normal varies.** Don't trim the library down to its quietest stretch: everything outside
  it becomes an alarm. Cover the operating conditions the asset normally runs in.
- **Several operating levels: scale per level.** When normal runs at a few distinct set-points
  (flow rates, speeds, products), one z-score over all of them is dominated by the distance
  between levels, and a fault smaller than that distance looks like another level's normal.
  On SKAB's water pump (three flow levels, 126, 76 and 32 L/min; 2026-10-31) one scale left
  valve closures at chance (AUROC 0.50 and 0.55); scaling each level by its own library rows
  lifted them to 0.80 and 0.73, and validation macro-F1 from 0.58 to 0.79, with the same
  search. Add the level as an index column and pass `--scale-by` it; every condition needs
  library recordings, and `build_roles.py` stops if one has none. **Only for a condition known
  at run time** (a set-point, a recipe, a mode switch): the deployed agent must get data scaled
  the same way, so the condition can't come from the data or its labels. Scaling is per
  recording: a recording that changes set-point partway gets one condition's statistics, so
  split it into one recording per set-point in the index first. One detector still fits every
  level; the library should still cover each one.
- **Don't renumber timestamps to hide gaps.** A window across a faked seam mixes two moments.
  Split instead; joining with a one-day gap is safe because the platform rejects it.

## Local Setup

```bash
cd skills/atai-anomaly-discovery-agent-data-prep/references
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, pyarrow, scipy
```

No API key. Run the tests:

```bash
pip install pytest
pytest skills/atai-anomaly-discovery-agent-data-prep/tests/ -v
```

They build synthetic recordings (jitter, a gap), run the whole prep, check that the
baseline check fails an impure library and warns on a level shift, and then **break the
role files on purpose** — a label in a library or delivery file, a jump inside a library
file, a bad timestamp format, a label that disagrees with its sidecar, an unknown label, a
test set without incidents, a recording in two roles, a library too thin for the detector
— and require the preflight to fail each one.

## File Layout

```
skills/atai-anomaly-discovery-agent-data-prep/
├── SKILL.md                  ← this file
├── references/
│   ├── role-files.md         ← the contract: layout, rules and why, manifest schema
│   ├── ad_common.py          ← index, recordings, timestamps (s / ms / ISO)
│   ├── prepare.py            ← 1. exact grid, split at gaps, one label per row
│   ├── check_baseline.py     ← 2. purity, stability, agreement, coverage of the library
│   ├── build_roles.py        ← 3. role files + manifest
│   ├── preflight_roles.py    ← 4. every platform rule, before any upload
│   ├── requirements.txt      ← numpy, pandas, pyarrow, scipy
│   └── sample_data/
│       ├── recordings.csv    ← the index, roles, groups and types filled in
│       ├── *.parquet         ← 35 Tennessee Eastman runs (4.8 MB), a label per row
│       └── README.md         ← provenance + public-domain attribution
└── tests/
    └── test_ad_prep.py       ← synthetic end-to-end + deliberately broken role files (pytest)
```
