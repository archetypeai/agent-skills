---
name: atai-rare-event-detection-agent-data-prep
description: >
  Prepare a labelled sensor recording for Archetype AI's Rare Event Detection (RED)
  agent, locally and with no API key: put it on an exact sample grid, map its labels to
  a background class and named event classes, number the incidents, check whether those
  incidents and channels can make a RED agent (incident length against window size,
  channels that go dark during events, zero-saturation, separation that does not carry
  over to held-out incidents), and write the role files the RED platform accepts (one
  training file per class piece, validation and test files around each incident with no
  row shared, delivery files with the labels held back, a manifest), then preflight them
  against every rule. Use when the user has one long recording with a few labelled
  incidents of a named fault and wants to fit, test and deploy a RED agent on it (the
  next step is `atai-rare-event-detection-agent`, Path 2), or when a RED optimization or
  eval scored oddly. Do NOT use for running the maintained RED Quick Start bundle on one
  CSV (`atai-rare-event-detection-agent`, Path 1), for normal-only data with no labelled
  incidents (`atai-anomaly-discovery-agent-data-prep`), for labelled operating states
  (`atai-operational-state-monitoring-agent-data-prep`), or for client-side embedding +
  KNN (`atai-newton-omega-model-data-prep`).
---

# RED Data Prep — a labelled recording → role files

The RED agent learns **one prototype per class** from a handful of labelled incidents
and normal operation, and flags windows that sit closer to an event's prototype than to
normal. With so few examples, a few prep decisions carry the whole result, and the ways
to get them wrong rarely error: a channel that goes dark *because* the pump failed
becomes a perfect but meaningless signature; an incident shorter than half a window is
all but undetectable at that window; two incidents close together share their normal
context between validation and test; channels chosen on held-out incidents score
optimistically. This skill makes each decision visible and checks every rule before you
upload.

```
one recording (any rate, jitter, gaps, missing readings)
  + a per-row label column     prepare.py → check_events.py → build_roles.py → preflight_roles.py → RED agent skill, Path 2
```

Everything here is **local and needs no API key**. The contract is
[`references/role-files.md`](references/role-files.md). It is the layout of the
[RED pump quickstart](https://github.com/archetypeai/rare-event-detection-agent-example-pump-quickstart),
generalised; the platform rules in it were measured on the dev deployment on 2026-10-13.

## When to Apply

- You have **a long recording** of an asset or process with a **label per row**, and the
  label marks **a few incidents of a named event** (a breakdown, a trip, a slip) against
  normal operation
- You want a RED agent that recognises **the same event** when it recurs, fitted on the
  first incidents and tested on later ones
- A RED optimization or eval **scored oddly** (an event class at 0, every window
  invalid, a suspicious false-alarm rate) and you want to check the role files

**Do not use this skill when:**
- You only want to run the maintained RED Quick Start bundle on one CSV — that's
  [`atai-rare-event-detection-agent`](../atai-rare-event-detection-agent/SKILL.md), Path 1
- You have **no labelled incidents**, only normal operation — that's anomaly discovery:
  [`atai-anomaly-discovery-agent-data-prep`](../atai-anomaly-discovery-agent-data-prep/SKILL.md)
- Your labels are **operating states** that cover the whole recording (fill / wash / spin)
  — use [`atai-operational-state-monitoring-agent-data-prep`](../atai-operational-state-monitoring-agent-data-prep/SKILL.md)

## Your input

**One recording** (CSV or Parquet): a timestamp column, the sensor channels (numeric),
and a label column.

```
,timestamp,sensor_00,sensor_01,...,sensor_51,machine_status
0,2018-04-01 00:00:00,2.465394,47.09201,...,201.3889,NORMAL
```

- **Timestamps:** epoch seconds, epoch milliseconds or ISO 8601. Jitter is snapped to the
  grid; a gap (more than 1.5 steps) starts a new segment and is never filled.
- **Labels:** any values; you map them to classes with `--classes`, background first:
  `normal=NORMAL pump_breakdown=BROKEN,RECOVERING`. Several event classes are fine
  (`trip=TRIP slip=SLIP`). An unmapped value stops the script.
- **Missing readings:** leave them. Step 2 decides per channel, and step 3 interpolates
  the short runs it keeps.

**Roles are assigned per incident**, by its number from step 1. Use the first incidents
for **training**, the next for **validation** (choosing settings), later ones for the
**test** (scored once), and deliver test incidents again, or new operation after them.
Time order matters: a deployment only ever has the past to learn from, and every choice
here is made on data from before the first held-out incident.

## The steps (on the shipped sample)

[`references/sample_data/`](references/sample_data/) holds the first four breakdowns of
the Kaggle pump recording, raw: 82,492 rows a minute apart, 15 of its 52 channels
(including ones that go dark during breakdowns, a long gap, an empty one), ISO
timestamps, `machine_status`. Everything below runs on it in about 3 s:

```bash
cd skills/atai-rare-event-detection-agent-data-prep/references
pip install -r requirements.txt
```

### 1. `prepare.py` — an exact grid, classes, numbered incidents

```bash
python prepare.py --input sample_data/pump_sample.parquet --label-col machine_status \
    --classes normal=NORMAL pump_breakdown=BROKEN,RECOVERING --out prepared
```

Sorts by time, drops duplicate timestamps, snaps jitter to the grid (the median
interval), numbers segments at gaps, maps the labels and numbers each incident (a run of
one event class inside one segment). Values are untouched. **Expect:**

```
82,492 rows, every 60 s, 15 numeric channels (1 other columns dropped: Unnamed: 0)
grid: 0 timestamps snapped, 0 rows off the grid dropped, 1 segment(s)
classes: normal 76,517, pump_breakdown 5,975
4 incidents:
  #1   pump_breakdown     rows    17,155-18,100        945 rows  from 2018-04-12 21:55
  #2   pump_breakdown     rows    24,510-27,621      3,111 rows  from 2018-04-18 00:30
  #3   pump_breakdown     rows    69,318-70,631      1,313 rows  from 2018-05-19 03:18
  #4   pump_breakdown     rows    77,790-78,396        606 rows  from 2018-05-25 00:30
```

**Why an exact grid:** the `red` blueprint checks that a window's timestamps are monotonic
and evenly spaced (each interval within 5% of the window's mean); a window failing either
comes back `INVALID_STATE`, not classified.

### 2. `check_events.py` — can these incidents and channels make a RED agent?

```bash
python check_events.py --prepared prepared --training 1,2 --validation 3 --test 4
```

On data from before the first held-out incident only:

| check | FAIL / WARN when |
|---|---|
| **incidents** | a role has none (FAIL); an event class has no training incident, so no prototype (FAIL), or is never scored (WARN) |
| **windows**, at `--windows 16,32,64,128,256` | a training class yields no window (FAIL); a scored incident cannot fill half a window (WARN): the classifier compares whole windows, so it would barely move one |
| **channels** | the triage step 3 applies: `dead`; `dropout` (missing on over 1% of rows; WARN when the missing rows sit inside events: interpolated, the sensor going dark would become the signature); `long gap` (a missing run over 16 rows, more than the platform repairs). Zero-saturation (exactly 0 in most event rows, almost never otherwise) is kept with a WARN: a real reading of a stopped machine, which makes the task "has it stopped?" |
| **separation** | a selected channel separates the held-out incidents less than half as well as the training ones (WARN) |

**Expect** `0 FAIL, 7 WARN`: `sensor_00` dropped because its dropout tracks the
breakdowns; `sensor_01`, `06` and `15` dropped; four zero-saturating channels; and

```
separation (mean |z| against build-time normal)
  sensor_04      training 10.02   held-out  6.43  <- selected
  sensor_05      training  8.82   held-out  8.59  <- selected
  ...
WARN  sensor_02: separation does not transfer (4.73 on training incidents, 2.06 held out)
WARN  sensor_03: separation does not transfer (2.87 on training incidents, 0.84 held out)
```

The warnings are for reading, not fixing: they say what the agent will key on (a stopped
pump reads zero current) and which channels are weaker evidence than they look.

### 3. `build_roles.py` — the role files

```bash
python build_roles.py --prepared prepared --out roles --training 1,2 --validation 3 --test 4 --delivery 4
```

Triage and selection as reported in step 2 (the `--top 10` channels by training
separation, or exactly `--channels ...`); short missing runs interpolated; every channel
z-scored with the build-time normal rows; then:

- **training:** `normal.csv` (the first `--normal-rows 8192` of a normal run before any
  held-out incident; a run is split at a gap, never taken across one) and one file per
  training incident, each one class, no label column. `--every-normal-run` writes
  `normal_01.csv`, `normal_02.csv`, … instead: the last `--normal-rows` of every such run,
  for a recording whose set point changes between runs (see Pitfalls);
- **validation, test:** one file per incident with `--context 4096` normal rows either
  side, cut at the midpoint where incidents are closer and at the incident's segment (a
  gap), with a `label` column;
- **delivery:** the `--delivery` test incidents with `--delivery-context 1024` rows either
  side, no label column, labels in `labels/`; or `--tail-chunk N` for the rows after the
  last scored file, in chunks (new operation: false alarms per day).

**Expect:**

```
channels (10): sensor_02 sensor_03 sensor_04 sensor_05 sensor_10 sensor_11 sensor_12 sensor_38 sensor_41 sensor_42
z-scored with normal rows before row 65,222
training    3 files    12,248 rows    4,056 event rows
validation  1 files     8,988 rows    1,313 event rows
test        1 files     8,282 rows      606 event rows
delivery    1 files     2,654 rows      606 event rows
```

The same 10 channels and row ranges as the pump quickstart's. Its z-scores differ: it took
its statistics from the first 30,000 rows, this from every normal row before the first
held-out incident.

### 4. `preflight_roles.py` — check every rule

```bash
python preflight_roles.py --roles roles --windows 16,32,64
```

Per file: the manifest's channels in order, an exact grid with no jump, finite values,
row counts. Across files: no row in two of training, validation and test; delivered
incidents inside their test files, tail files apart from everything; labels only in
validation and test, with the background and the incident's class; each training file
one class, and every class trained; delivery labels row for row; every scored incident
able to fill half of each window you will search. **Expect:** 16 `PASS`, then
`all checks pass`.

Then run the RED agent skill's Path 2 on `roles/`.

## The rules, and what breaks

Each rule, and why the platform needs it, is in
[`role-files.md`](references/role-files.md). The ones that fail silently, as the tests
here exercise:

| broken | what happens on the platform |
|---|---|
| a normal row in an incident's training file | the event's prototype moves toward normal: fewer detections, no error |
| a channel that goes dark during events, interpolated | the agent learns "sensor offline", which tells you nothing about the machine |
| validation and test sharing normal context | the test's false-alarm rate is partly a validation number |
| a time jump inside a file | every window across it is `INVALID_STATE` |
| an incident shorter than half the window | at that window it barely moves the embedding: missed, with nothing flagged |

## Pitfalls

- **The agent recognises the event's state, not its cause.** On the pump, `BROKEN` is one
  row per incident and `RECOVERING` the rest; folding them into one class detects "the
  pump is down", not the failure. Choose the label mapping for what an operator needs to
  know.
- **Choose the window from an operational requirement:** the shortest event the
  deployment must catch, from the people who run the asset. Deriving it from the test
  incidents' lengths leaks them into model selection.
- **Keep the z-score statistics** (`manifest.json` → `zscore`): any input the deployed
  agent sees must be scaled the same way.
- **More normal rows barely change a prototype** (it is a mean) but cost encoder time in
  every trial; `--normal-rows` is a cost knob more than an accuracy one.
- **Label the event, not the time before it.** CARE to Compare's wind-turbine faults come
  with a prediction window, the days before each logged fault. Labelled that way, Wind Farm
  A's six hydraulic-group faults sat about 1 σ from normal per channel, mostly through wind
  and power: validation F1 0.57 on one incident, then 0.20 on three held-out ones (chance;
  the agent skill's Path 2 pitfalls have the numbers). A prototype needs the event to change
  the readings.
- **Look for what else moved during the incidents.** On PRONTO's multiphase flow rig the
  first separation report put three separator channels at 8–11 σ: the rig's end-of-day
  shutdown sat inside one leak's label, and slow drift through the day did the rest. Two
  checks caught it before any platform time: compare each incident with the **same day's**
  normal rows (not only the pooled build-time ones), and read a held-out separation
  **larger** than the training one as a warning, not a bonus. Trim shutdowns and start-ups
  out of event labels, and drop channels that drift with the clock.
- **Normal must cover the operating points the events run at, and only those.** The
  centroid of normal is a mean: normal from a set point no event ever runs at pulls it
  away from the normal the agent will see, and normal from one set point only, beside an
  event at another, lets the agent learn the set point. Drop the first (gaps are fine;
  prep splits at them) and cover the second with `--every-normal-run`, one normal file
  per run. On PRONTO (leaks at two air/water set points) that gave three normal files.
- **A fault no sensor sees gives no prototype**, whatever the label says: PRONTO's air
  blockage and diverted flow barely move its 1 Hz process data. Leave such an event out
  rather than calling it normal or training it.

## Local Setup

```bash
cd skills/atai-rare-event-detection-agent-data-prep/references
pip install -r requirements.txt        # numpy, pandas, pyarrow
```

Tests:

```bash
pip install pytest
pytest skills/atai-rare-event-detection-agent-data-prep/tests/ -v
```

## File Layout

```
skills/atai-rare-event-detection-agent-data-prep/
├── SKILL.md
├── references/
│   ├── prepare.py            ← 1. exact grid, classes, numbered incidents
│   ├── check_events.py       ← 2. incidents, windows, channels, separation
│   ├── build_roles.py        ← 3. the role files (contract red-role-files/v1)
│   ├── preflight_roles.py    ← 4. every rule
│   ├── red_common.py         ← shared: reading, incidents, slices, channel rules
│   ├── role-files.md         ← the contract and why each rule exists
│   ├── requirements.txt
│   └── sample_data/          ← the pump's first four breakdowns, raw, with attribution
└── tests/test_red_prep.py    ← synthetic end to end, and role files broken on purpose
```

## Data attribution

The sample is derived from the Kaggle dataset **`pump_sensor_data`** by
[`nphantawee`](https://www.kaggle.com/nphantawee)
(https://www.kaggle.com/datasets/nphantawee/pump-sensor-data); credit for the underlying
data belongs to the operations team who shared it. No licence is declared on the dataset.
Full attribution and what was changed: `references/sample_data/README.md`.
