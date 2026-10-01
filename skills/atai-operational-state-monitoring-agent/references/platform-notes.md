# OSM platform behaviour worth knowing (Path 2)

Measured on the **dev** deployment on 2026-09-30, building the LARCO washing-machine
agent end to end: the [quickstart](https://github.com/archetypeai/osm-agent-example-larco-quickstart)
(19 short cycles, ~15 min) and the [full example](https://github.com/archetypeai/osm-agent-example-larco)
(199 cycles, 25 GB of role files). **Production and staging** (2026-10-01) ran the prep skill's
sample end to end with results identical to dev at every step. Each item says what you'll see if you
trip over it.

## Deployments

| deployment | Optimizations / promote / Evals | notes (2026-10-01) |
|---|---|---|
| production (`api.u1`) | ✅ | the sample reproduced dev exactly: 0.7920 / 0.8463 / 0.7431 |
| staging (`api.stage.u1`) | ✅ | the same, exactly |
| dev (`api.dev.u1`) | ✅ | where everything here was measured |
| Tokyo (`api.u2`) | ✅ | the same, exactly, once the optimizer runner and eval metrics went live on 2026-10-01 (before that, optimizations failed at job creation and evals completed with no `metrics_report`) |

**If optimizations fail the moment they're created** with `JOS job creation failed: … Pipeline
'optimizer-runner-cuda' has no active versions`, that deployment's optimizer job runner isn't
deployed: the API and blueprint are there, the runner behind them isn't. Staging, production
and Tokyo all did this until 2026-10-01. Nothing client-side fixes it; ask the platform team.

**If an eval reports `completed` with no `metrics_report`,** the scores never arrived:
`promote_and_test.py` waits 10 min for them, then stops with the eval id to report. A rerun
resumes the same eval and checks again; nothing is created twice.

Each deployment has its own `osm` blueprint id (resolve it by key) and its own API keys: a key
for one returns 401 on another.

## Data the platform accepts

The rules for role files are the prep skill's contract,
[`role-files.md`](../../atai-operational-state-monitoring-agent-data-prep/references/role-files.md).
Why they exist:

- **1 MiB per optimization config.** The whole config (every training example with
  name, file id, checksum and ground truth; pretty-printed; file lists twice) goes into
  one Kubernetes ConfigMap, capped at 1,048,576 bytes: **~636 B per training example,
  so ~1,500 examples at most**. 695 went through; 1,990 got stuck. Over the limit the
  job **never dispatches and nothing in the API says so**: the optimization stays
  `running`, every trial `pending`, for hours. The size scales with the **number** of
  training files, not their size, so pack each state's windows into **one file** as
  continuous pieces.
- **Jumps in time: training vs scored files.** A training file may contain forward jumps
  between pieces; any window across one fails the sampling-rate check and is **dropped
  silently** (not counted, not reported); the trial succeeds. In a **scored** file
  (validation, test; Optimizations and Evals alike) a single window across a jump fails
  the **whole** trial or eval: `eval-mode test data must not contain windows a
  validation node rejected`. Jumps survive there only if every one falls exactly on a
  window boundary (step = window). Keep scored files one continuous file per segment.
- **The sampling-rate tolerance is relative** to the window's mean sample interval
  (`sample_rate_interval_tolerance`, default 0.05). Loosening it doesn't rescue a real
  jump (one gap in an N-row window deviates ~N×; 10.0 wasn't enough), and the search
  space can't set the check to null or "warn". Resample onto an exact grid instead.
- **Timestamps:** fractional epoch seconds, epoch milliseconds and ISO 8601 all
  work at 200 Hz, read to the millisecond, with identical scores.
- **A state absent from a scored set counts as F1 = 0** in the platform's macro-F1,
  so one missing state caps macro-F1 at (n−1)/n. Check every scored set holds every
  state.
- **Step > window** skips the records between windows: validation is scored on a
  sample and each library piece gives fewer windows. Allowed, but do it deliberately.

## Optimizations, promote, Evals

- **Trials run one after another,** also across optimizations: a second optimization
  waits for the first. `max_trials` below the search space's size makes the platform
  sample points at random (repeats are possible); at the size, every point runs once.
- **Every trial's `created_at` is the optimization's creation time,** so time a trial
  from the previous trial's `completed_at`, not from its own `created_at` (the latter
  includes its wait in the queue).
- **`metrics_report`** (on trials and evals): `primary.value` is the objective
  (macro-F1); `targets.state.class_names` are **alphabetical**, and `per_class.f1`,
  `precision`, `recall`, `support` and `confusion_matrix` (rows = true) follow that
  order.
- **Status spelling:** a cancelled optimization is `"cancelled"`, not `"canceled"`.
- **Promote** (`POST /agents/optimizations/{opt}/trials/{trial}/promote`) makes a
  blueprint with the trial's setting and fitted kNN attached; no S3, no classifier
  artifact. Give each promotion a **unique `blueprint_key`** (ours ends in the trial
  id): a key reused across runs makes a "promote or reuse" script test an older model.
- **The Evals report is pooled over all its files.** There is no per-file breakdown:
  for a score without one file, run a second eval on that file and subtract its
  confusion matrix (confusion matrices add up).
- **Choose on validation, test once.** Selecting the best of N trials on a few cycles
  flatters the winner; the full example's +0.014 margin on its 6 search cycles was
  +0.001 on the 15 validation cycles the search never saw.

## Bundles and runs

- **A bundle from your blueprint:** `POST /agents/bundles {"blueprint": "<key>"}`,
  then `POST /agents/bundles/{id}/run` with the files as source connectors. One run =
  one agent instance, one worker taking its files in turn (~2 min per 2.2 M-row file).
- **Several runs are processed one at a time,** even though each shows `running`
  from the start: the full example's 5 runs finished 25–50 min apart.
- **Outputs don't name their inputs.** Match each output row to its file by
  `finish_timestamp`, which works when recordings don't overlap in time. Output
  timestamps drop trailing zeros (`1694181225.45`).
- **`completed` can come before the last output is fully written.** A run's status
  and its `/results` listing can be ahead of the last output file: downloaded within
  seconds, prod's last file held 326 of 997 rows, and minutes later all 997. Check
  that each file's predictions reach its end, and download again if they don't
  (`deliver.py` reports a run completed only once its outputs reach every file's end).
- **`finish_timestamp` is the window's last row**, so pair each prediction with the
  label at that row, as the Evals API does (`downsampling: last_record`).

## Timing

**It varies with platform load and job size, not a fixed cost per call:**

| | quickstart (small jobs) | full example (large jobs) |
|---|---|---|
| a trial | ~2 min (2 trials in 4 min) | ~12 min |
| an eval | 3 min (3 files) | ~40 min (1 or 23 files alike) |
| a delivery run | 3 min (5 short files) | 30–50 min (25 files) |

Upload time is separate: ~30 MB/s here. Many parallel multi-hundred-MB uploads can
saturate the uplink until **DNS lookups fail** (`nodename nor servname provided`);
retry, and keep upload concurrency at ~3.

## Results

- **Deterministic:** a fresh end-to-end rerun (new download, new uploads, new
  optimization, new blueprint) reproduced every number exactly, the confusion
  matrices included.
- **States must be visible in the sensor.** LARCO's `heating` (the heater on while the
  drum washes) is invisible to an accelerometer: a held-out-cycle classifier separated
  it from wash at 0.62 balanced accuracy (0.5 = chance), and it capped every trial's
  macro-F1 near 0.55 until it was folded into `wash` (0.72+). Check separability before
  fixing the state list; the prep skill has a check for it.
- **A balanced library over-predicts rare states.** 100 or 400 windows per state give a
  rare state (drain, 3% of the time) a quarter of the votes: high recall, low precision.
- **k matters most among the kNN settings,** and its best value depends on library
  size: k 21–31 won with 400 windows per state, k 5 with 100.
