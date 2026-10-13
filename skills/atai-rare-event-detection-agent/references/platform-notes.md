# RED on the Optimizations and Evals APIs: measured platform behaviour

What Path 2 relies on, measured on the dev deployment on 2026-10-13 while building the
[RED pump quickstart](https://github.com/archetypeai/rare-event-detection-agent-example-pump-quickstart)
and its full example. RED became optimizable on Dev with the 2026-10-12 `red` re-seed
(atai_core #7197, "Adds metrics to RED blueprint"); check another deployment before relying
on it there.

## The re-seed, and the bundles that predate it

- **Canonical blueprints are versioned.** The re-seed made a new active `red` blueprint
  (`blp_688t75…`, created 2026-10-12) and left the July one (`blp_08stra4…`) inactive.
  **A bundle stays on the blueprint version it was created on:** the RED Quick Start
  bundles still run on the July version, byte-identical to their documented output.
  `"blueprint": "red"` in a new request means the active version.
- **On the current version the head is an empty model slot,** `classification_head`, that
  an optimization fills; nothing is read from a classifier's metadata any more. So a new
  bundle with only `"artifacts": {"red-classifier": "s3://…"}` is
  `400 models.classification_head: this blueprint has no model for classification_head yet`
  at run creation. Bringing your own classifier now takes
  `"models": {"classification_head": "red-classifier"}` and pinned `window_size`,
  `metric`, `events`, `background` and columns (window 1,024 and `l1` otherwise).
  A promoted blueprint needs none of this.

## Optimize

- **The `red` blueprint's values.** Searchable as `kind: value`: `metric` (`cosine`,
  `l1`, `l2`, `dot`; default `l1`), the classifier's only fit setting, and `window_size`,
  `step_size`. The head is nearest-centroid, hardcoded (`strategy: centroid`,
  `normalize_embeddings: false`): no `k`, no threshold. **`events` and `background` are
  required** in `overrides.values`; a training class not in `events` is a 400.
- **Pin the window and step.** Defaults: window 1,024, step 256. At 1,024 a 945-row
  incident yields no training window at all.
- **No default search space,** and it can't be empty: search `metric` even at one window.
- **Training labels:** `{"from": {"constant": "<class>"}}` per file, so one class per file.
  **Validation:** `{"from": {"column": "label"}}`.
- **`primary` picks the objective:** `event.macro_f1` by default;
  `event.foreground_macro_f1` (the F1 over the events alone), `detection_recall`,
  `detection_precision`, `accuracy`.
- **Each row is scored once.** With overlapping windows, a window answers only for the
  rows since the previous window ended, collapsed by `downsampling`
  (`eval/scoring/by_time.rs`). At step 1 that is one row, so `majority_vote` scores
  exactly as `last_record` (checked on 37,623 test windows). The design rule, each window
  by the majority of its rows, has to be computed offline.
- **Trials that share window and step share one encoder pass,** computed inside the
  optimization (on `optimizer-runner-cuda`). Measured: the 4 metrics at window 64 took
  3–4 min (~21k windows); windows 16/32/64 × 4 metrics, 12 trials, 3 passes, 7 min. A
  random search over windows 16–84, 24 trials, cost a pass per trial: 41 min, and peaked
  at 0.8582 against the grid's 0.8588.
- **The platform draws distinct points at random,** unseeded: a categorical space with no
  more points than `max_trials` runs every point (a grid); a smaller budget or a range is
  a random search, and a rerun samples other points.
- **Deterministic:** the same optimization run again, from fresh uploads, gave every trial
  the same scores and confusion matrices.

## Promote, Evals

- **Promote** pins the trial's values, `events` and `background`, and points
  `models.classification_head` at an `aar_…` registry artifact (no `s3://` path, no
  download route). Promoted blueprints can't be deleted by a regular key (403): name them
  distinctively.
- **An eval takes a blueprint and no value overrides:** it runs at the trial's window and step.
- **Predictions** carry `row_index`, `example`, `predicted`, `gold`, `correct`,
  `abstained`, `outcome`: no timestamp, no probabilities. Windows tile each file from row 0
  at `step_size`; `row_index` counts across the whole eval.

## Bundles and agent runs

- **A bundle on a promoted blueprint needs only `step_size`** (bundle values override the
  blueprint's); the prototypes, window, metric, `events` and `background` come with it.
- **Agent-run speed varies a lot.** The same 2,000–2,600-window runs went at ~4 windows/s
  in the morning and ~50 in the afternoon of 2026-10-13; earlier measurements were ~2.2.
  Optimize and Evals run elsewhere and were steadier (~90 windows/s).
- **Outputs can be cut short at `completed`:** download again until the last window
  reaches the input's end. **A `failed` status can hide a finished run:** check `/results`.
- **Agent runs and Evals agree:** delivered at step 1, the same rows give the same
  predictions as the eval (the same detections, at the same minute, and the same false
  alarms).
