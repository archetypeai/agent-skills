---
name: atai-label-free-anomaly-discovery-agent-data-prep
description: >
  Prepare raw sensor data for Archetype AI's Label-Free Anomaly Discovery agent (the `od`
  blueprint), locally and with no API key: check the raw files (cadence, gaps, missing
  values, constant channels, frozen-logger stretches), put the data on an exact sample grid
  (short holes filled, gaps kept, never interpolated across), cut out frozen stretches, keep
  the analog channels, cut the files od will read (one per batch session), and check each
  against od's rules and against what LOF needs to work (enough gap-free windows, few lost
  to gaps, anomalies short enough to stay a minority, a window inside Omega's 16-1,024
  samples). Use when the user has unlabeled raw sensor CSVs and wants to run label-free
  anomaly discovery on them (the next step is `atai-label-free-anomaly-discovery-agent`),
  or when an od run returned mostly invalid windows or a ranking that looks wrong. Do NOT
  use to build role files for the Anomaly Discovery agent fitted on normal data
  (`atai-anomaly-discovery-agent-data-prep`), for labelled operating states
  (`atai-operational-state-monitoring-agent-data-prep`), for a named rare event
  (`atai-rare-event-detection-agent-data-prep`), or for client-side embeddings
  (`atai-newton-omega-model-data-prep`).
---

# Label-Free AD Data Prep: raw data → files od can read

The Label-Free Anomaly Discovery agent (`od`) reads one CSV per batch session and compares
each window of it with the file's other windows. So the prep has two jobs:

1. **Make files od reads correctly:** a timestamp column and numeric channels on an exact
   grid, with gaps left as gaps.
2. **Make files od can work on:** long enough for LOF to mean something, with the
   anomalies you care about a small share of each file, and a window that suits the
   machine.

There are no roles and no labels here: no library, no validation or test split, no
normalisation (od scales what it needs, on the file it scores). What matters is where each
file starts and ends.

## When to Apply

- Raw sensor CSVs (text or epoch timestamps, irregular sampling, gaps, maybe a frozen
  logger) that should go through `od`
- Deciding **how to cut** a long record into files, and **which window** to use
- An `od` run with many invalid windows, or a ranking dominated by something that isn't
  the machine (a frozen logger, a gap)

**Do not use this skill when:**
- You'll fit a detector on normal-only data and monitor new data: [`atai-anomaly-discovery-agent-data-prep`](../atai-anomaly-discovery-agent-data-prep/SKILL.md)
- The data has labelled states or labelled incidents to learn from: the OSM or RED data-prep skills

## Your input

One or more CSV files (plain or `.gz`) with a timestamp column and numeric channels.
- **Timestamps:** text (`2020-07-08 15:20:51`) or numbers; epoch seconds and milliseconds
  are told apart by size. Several files are concatenated and sorted.
- **Channels:** by default every numeric column except binary ones (0/1 switch or valve
  states: constant in most windows, they say little). `--channels a,b,c` chooses.
- **A label column, if any,** is left out of the output: `od` never reads one. Keep labels
  or maintenance records apart, for grading afterwards.

## The steps (on the shipped sample)

```sh
cd references
pip install -r requirements.txt
S=sample_data/metropt3_2020-07-08_2020-07-16.csv.gz
python preflight_raw.py --input $S --step-s 10
python prepare.py --input $S --step-s 10 --out prepared
python preflight.py --prepared prepared --window 360 --expect-event-h 4
```

The sample is nine days of a metro train's air compressor (MetroPT-3, CC BY 4.0) in the
dataset's own format: 56,944 rows, text timestamps, 7 analog and 8 digital channels.

### 1. `preflight_raw.py`: the raw data, before any decision

Read-only. Each WARN names the decision it leads to in step 2. **Expect:**

```
  PASS  channels                   7 kept: TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current
  INFO  left out                   COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses (binary or not chosen)
  WARN  cadence                    median step 10 s; 8.8% of steps off 10 s by over 5% -> prepare.py snaps rows to an exact grid
  WARN  gaps                       18 longer than 50 s (longest 14.2 h) -> kept as gaps; od marks a window that crosses one invalid
  PASS  frozen logger              none
RESULT: PASS  (8 PASS, 2 WARN, 0 FAIL, 3 INFO)
```

### 2. `prepare.py`: the grid, the frozen stretches, the files

1. **An exact grid.** Each row goes to a `--step-s` slot (the mean when two share one).
   `od` checks the spacing inside every window and marks a window invalid if it varies by
   more than 5%, so jittered timestamps would cost windows.
2. **Short holes filled, gaps kept.** Up to `--max-fill-s` (default 5 steps) missing is
   jitter and takes the last value; anything longer stays a gap. Never interpolate across
   a gap: invented data is the kind of stretch `od` would rank as unusual.
3. **Frozen stretches cut.** Every channel holding one value for `--frozen-min` (30) or
   more is the logger, not the machine. On the full MetroPT-3 record that is 170 hours, in
   stretches of 15–62 hours. Measured on three weeks of it (Apr 4–25, 2020, staging) with
   them left in: all 36 frozen hours were flagged and took ranks 3 to ~40, pushing most of
   the reported leak from ranks 3–36 down to 42 and below; the leak's AUROC fell from
   0.974 to 0.883.
4. **The files.** The whole span (default), consecutive pieces (`--split-days 21`), or
   chosen ranges (`--split 2020-04-04..2020-04-25 2020-07-01..2020-07-22`).

**Expect:**

```
grid: 56,944 raw rows -> 56,449 rows at 10 s; frozen-logger rows cut: 0 (0.0 h)
  metropt3_2020-07-08_2020-07-16.csv: 56,449 rows (156.8 h of data) in 19 continuous pieces
```

**Writes:** `prepared/<name>.csv` (`timestamp` in epoch seconds, then the channels, in their
own units) and `prepared/prepare_report.json`.

### 3. `preflight.py`: will od work on these files?

Read-only. `od`'s rules, then what LOF needs:

| check | why | fails or warns when |
|---|---|---|
| window size | Omega takes 16–1,024 samples | outside |
| schema, no label, finite, increasing, on the grid | what `od` reads | any broken |
| no frozen stretch | it would top the ranking | one is left |
| gap-free windows | LOF on few windows means little | under 50 |
| lost to gaps | windows that cross a gap are invalid | over 25% of them |
| size | `od` holds a whole file's windows in memory | over 20,000 windows |
| LOF neighbours (INFO) | an episode longer than ~K windows forms its own neighbourhood and hides | — |
| window vs events, K vs events (`--expect-event-h`) | a window longer than an event dilutes it; an event longer than ~K windows hides | either |

**Expect:**

```
  PASS  metropt3_2020-07-08_2020-07-16.csv: windows 156 windows, 142 gap-free
  PASS  metropt3_2020-07-08_2020-07-16.csv: lost to gaps 9% of windows cross a gap
  INFO  metropt3_2020-07-08_2020-07-16.csv: LOF neighbours K = 28: an episode longer than ~28 h would form its own neighbourhood and hide
  PASS  metropt3_2020-07-08_2020-07-16.csv: window vs events window 1 h, events ~4 h
RESULT: PASS  (12 PASS, 0 WARN, 0 FAIL, 1 INFO)
```

## Cutting files, and choosing the window

These two choices matter more than anything else here, and neither needs labels.

**Files.** `od` compares each window only with the rest of its file, so:
- **Long enough:** at least ~50 gap-free windows; a few hundred is comfortable.
- **Anomalies a minority:** an anomaly that fills a large share of a file sets that file's
  normal. Cut files so that whatever you hope to find is short next to them (a 4-hour leak
  in a three-week file is 1%).
- **One way of operating per file, if you can:** a file that mixes two regimes (summer and
  winter, two products) ranks the rarer regime as unusual. Cut by season or campaign rather
  than by a fixed length when you know where the changes are.
- **Not too long:** past ~20,000 windows, split it.

**Window** (rows = duration ÷ sample step):
- **Shorter than the events of interest.** If nobody knows how long they last, that is the
  question to ask the person who knows the machine.
- **Longer than the machine's own rhythm:** a cycle, a batch, a control response. On the
  sample's compressor, which starts every 19–28 min and runs ~9 min, one hour holds two to
  three cycles.
- **Step = window** in `od` (no overlap): each stretch scored once.

## Pitfalls

- **Normalising the files.** Not needed and not wanted: `od` z-scores its features on the
  file it scores. Scaling with statistics from another period only adds a choice.
- **Interpolating across gaps.** It makes up data in exactly the places `od` looks.
- **Keeping binary channels.** Mostly constant within a window; they add noise to the
  distances, rarely signal.
- **One giant file.** Normal changes over months; `od`'s "unusual" is relative to the file.
- **A label column in the file.** Harmless to `od` if not listed in `data_columns`, but it
  invites mistakes. Keep labels and records apart, for grading.

## Local Setup

```sh
cd skills/atai-label-free-anomaly-discovery-agent-data-prep/references
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas
```

No API key, no network. Outputs go to `prepared/` (git-ignored).

## File Layout

```
skills/atai-label-free-anomaly-discovery-agent-data-prep/
├── SKILL.md
├── references/
│   ├── requirements.txt
│   ├── lf_prep.py           reading raw CSVs, the grid, frozen stretches, window counts, the report
│   ├── preflight_raw.py     step 1: the raw data, read-only
│   ├── prepare.py           step 2: grid, gaps, frozen stretches, channels, files
│   ├── preflight.py         step 3: od's rules and what LOF needs, read-only
│   └── sample_data/         nine raw days of MetroPT-3 (CC BY 4.0)
└── tests/
    └── test_lf_prep.py      synthetic data, plus the sample end to end
```

Run the tests: `python -m pytest skills/atai-label-free-anomaly-discovery-agent-data-prep/tests -v`.
Next step: [`atai-label-free-anomaly-discovery-agent`](../atai-label-free-anomaly-discovery-agent/SKILL.md),
whose sample is this skill's output on this sample.

## Data attribution

The sample is derived from the **MetroPT-3 Dataset** (Davari, Veloso, Ribeiro & Gama,
2021; UCI Machine Learning Repository; [doi:10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R)),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Details:
[`references/sample_data/README.md`](references/sample_data/README.md).
