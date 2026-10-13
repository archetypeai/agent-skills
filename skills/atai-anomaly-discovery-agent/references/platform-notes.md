# AD on the Optimizations and Evals APIs: measured platform behaviour

What Path 2 relies on, measured on the dev deployment (2026-10-12 → 10-13) while building
the [TEP quickstart](https://github.com/archetypeai/anomaly-discovery-agent-example-tep-quickstart)
and the IMS bearings example. AD support in these APIs landed on Dev on 2026-10-12
(atai_core #7196); check another deployment before relying on it there.

## Optimize

- **The `ad` blueprint's values.** Searchable as `kind: value`: `window_size`,
  `step_size`, `n_neighbors`, `metric` (`l1`, `l2`, `cosine`), `pca_components`,
  `threshold`. `feature_names` is a list, so it can only be pinned (`overrides.values`);
  compare feature sets with one optimization each. Also pin `timestamp_column` and
  `data_columns`.
- **Not exposed, by design** (#7196): PCA is always on; `zscore_std_floor`, PCA
  whitening, the neighbour algorithm and the PCA solver are not tunable. The TEP
  detector, fitted through the old `ad-fitting` route with a floor of 0.0001, was
  reproduced without it: validation recall 0.772 vs 0.766, precision 0.990 vs 0.992.
- **Pin window, step and features.** Since the 2026-10-12 re-seed, the defaults are
  window 1,024, step 256 and seven features (`mean,std,range,min,max,slope,median`).
  At a 3-minute sampling rate a 1,024-sample window is 51 hours: on TEP's 25-48 h runs
  it yields no valid window at all.
- **No default search space yet.** A request without `search_space` is a `400 search_space
  is required`; the code says it is meant to fall back to a space the blueprint declares,
  which no blueprint publishes yet. The space can't be empty either: a 1-trial fit at
  fixed settings still searches one parameter, e.g. `n_neighbors` with one value.
- **`threshold` is a `float_range` (`{"min": …, "max": …}`) or pinned.** A categorical list
  of floats is a 422, and `low` / `high` is a 422 naming the missing `min`.
- **The platform draws distinct points at random.** A space of categoricals with no more
  points than `max_trials` runs every point (a grid); a smaller budget, or a float range,
  is a random search. The draw isn't seeded on the service, so a rerun samples different
  points; each optimization records the ones it drew.
- **Trials that share `window_size` and `step_size` share their embeddings** (atai_core's
  trial grouping): those two set a search's cost. Measured on RED (same mechanism): a
  4 × 3 grid of metric × window took 3 encoder passes and 7 min; window as an integer
  range costs a pass per distinct window drawn. A 24-trial random search over windows
  16–84 took 41 min (24 passes) and peaked at fault F1 0.8582, against the 12-point
  grid's 0.8588: on a small, cheap grid a random search buys nothing.
- **Both legs use the trial's one `step_size`.** The fit and the validation scoring share
  it, so a search that varies the step scores trials on different window sets: macro-F1
  differences under ~0.01 between strides mean little.
- **Training examples are labelled by a constant** (`{"from": {"constant": "normal"}}`);
  validation examples by the `label` column, each window by its last row
  (`"downsampling": "last_record"`). The primary metric is `anomaly.macro_f1`.
- **Validation examples are required even when nothing is searched.** A 1-trial fit at
  pinned settings still needs them; their score is then unused.
- **A time jump inside a scored file does not fail the trial or eval** (unlike OSM): the
  windows across it come back abstained. That is what lets one file hold many recordings.
- **Timing.** Queued trials wait for workers. On Dev: a pinned 1-trial fit took 92 s with
  no queue; 2 trials 6-10 min; 24 trials about 77 min (trials fan out across jobs,
  #7152). Every trial reports the optimization's `created_at`, so don't time trials by it.

## Promote, Evals

- **Promote** (`POST /agents/optimizations/{opt}/trials/{trial}/promote`, a unique
  `blueprint_key`) returns a blueprint that pins the trial's values and names its fitted
  detector as a registry artifact (`models.detection_head: aar_…`), not an `s3://` path.
  The API has no route to download it.
- **An eval takes a blueprint and no value overrides.** Whatever you set through bundle
  values (a delivery threshold, a stride) cannot be evaluated; the eval runs at the
  trial's own threshold and step.
- **Evals reports window-level macro-F1 at one threshold, and it moves with the class
  mix.** The same TEP detector scored 0.854 on validation (46% normal rows) and 0.732 on
  test (21% normal), with anomaly F1 0.870 on both.
- **Predictions carry `gold`, `predicted`, `abstained` and a `row_index` counted across
  the whole eval** — no score, no timestamp. A window's place in its file is its rank
  within its example; windows tile each file from row 0 at `step_size`. Detection delay
  and AUROC need agent runs.
- **Promoted blueprints can't be deleted by a regular key** (403, admin only), as in OSM;
  name them distinctively.

## Bundles and agent runs

- **A bundle's values override its promoted blueprint's.** `step_size: 1` scores every
  sample; `threshold` relabels at another operating point with the same detector (checked
  on every window: `predicted_label` is `anomaly` exactly when the score is over it).
- **Outputs can be cut short at `completed`**: download again until the last window
  reaches the input's end. **A `failed` status can hide a finished run**: check `/results`.
- **The slowest agent sets a delivery's time.** TEP: 5 files took 885 s with 5 agents at
  once and 886 s with 4.
- **The checkpoint abort is fixed** (runtime 1.1.9-590); `deliver.py` turns the triggers off
  anyway, which is harmless where the fix has landed.
