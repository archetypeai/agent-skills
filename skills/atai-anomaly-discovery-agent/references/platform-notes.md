# AD on the Optimizations and Evals APIs: measured platform behaviour

What Path 2 relies on, measured on the dev deployment (2026-10-12 → 10-15) while building
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
- **Choosing the key can need the admin role.** On the free trial (2026-10-29) a promote with a
  `blueprint_key` answered 403, "choosing the `blueprint_key` of a promoted blueprint requires the
  admin role"; without it the platform keyed the blueprint `ad-otr_<trial id>`, and the `name`
  (a display label, what the console's "Blueprint name" sets) was still ours. Production
  accepted chosen keys with our accounts. `promote_and_test.py` tries its own key first and falls
  back on that 403.
- **The free trial runs on CPU, at most 3 jobs per org at once, each for at most 2 hours.**
  Further jobs queue, and a queued one already reports `running`. On 2026-10-30 the sample's
  eval (trial step 8) took 29 min there, and its step-1 delivery (4,800- and 6,720-row files),
  ~13 min on production, was stopped on both files with no output: `DeadlineExceeded: stopped
  after reaching the platform's maximum runtime of 2 hours`. Through bundles there, the
  trial's step delivered 2/2 files in 28 min, and step 1 in runs of at most 1,000 windows
  took 12 runs of 24–34 min (2 h 15 min in all) with the production scores exactly. A
  package delivers at the trial's step, and `--max-windows-per-run N` still splits a long
  file into sessions. Serving at step 8 there matched batch on 805/805 windows (score within
  2e-7, the same first alarms) at a 768 ms median per query.
- **An eval takes a blueprint and no value overrides, and so does a package.** The eval and
  the delivery both run at the trial's own threshold and step; `score.py` rescores the
  delivery at other thresholds from its scores.
- **Evals reports window-level macro-F1 at one threshold, and it moves with the class
  mix.** The same TEP detector scored 0.854 on validation (46% normal rows) and 0.732 on
  test (21% normal), with anomaly F1 0.870 on both.
- **Predictions carry `example`, `expected`, `predicted`, `outcome` and `key`** — no
  score. `key.predicted` is the window's [start, end] in epoch seconds; order by it, since
  `input` is 0 on every record. An abstention is `outcome: "missed"` with a null
  `predicted`. Windows tile each file from row 0 at `step_size`. Until 2026-10-20 the
  records carried `gold`, `abstained` and a `row_index` instead. Detection delay and AUROC
  need agent runs.
- **Promoted blueprints can't be deleted by a regular key** (403, admin only), as in OSM;
  name them distinctively.

## Projects, packages, deployments and sessions (dev, 2026-10-30)

These replace bundles and agent instances; production doesn't serve deployments and sessions
yet (they answer 404 there and on staging, 2026-10-30).

- **Everything takes its project from what it is made from**: an optimization from its
  blueprint, a promoted blueprint from its optimization, a package from its blueprint, a
  deployment from its package, a session from its deployment. `POST /agents/optimizations`
  takes no `project_id`, and the canonical `ad` is outside any project, so `optimize.py`
  copies `ad` into the project first (`POST /agents/blueprints/ad/copy {"project_id", "blueprint_key"}`).
  A deployment of a package outside any project is refused. A copy's key must be unused in
  the project and by platform blueprints: `ad` itself is taken.
- **A package carries no values.** It runs the promoted blueprint as stored; its
  `blueprints[0].resolved_models` pins the encoder (`mdl_…`) and the fitted detector
  (`aar_…`). Outputs (`output_score`, `output_embeddings`) and the step are set at Optimize.
  A package was `ready` within seconds.
- **A batch deployment is ready at once; a serving one in seconds.** A batch session per
  file ran the sample's 2 delivery files in 83 s at step 8 (`session opened`, `dispatched as
  job …`, `run completed` in `/events`), against ~13 min for the bundle run at step 1.
- **A session's `status` is reliable and carries `error`.** Instances needed `/logs` to spot
  a dead pod still reading `running`; sessions have no `/logs`.
- **Result refs are root paths**: `/sessions/{id}/results` gives
  `data[0].data.ref = "/api/files/<fil_…>/content"`; resolve it against the deployment root.
  The output CSV is the same as a bundle run's.
- **`output_embeddings: true` needs its own blueprint.** A 1-trial optimization pinning the
  winner's values plus the switch, then promoted, scored the same as the winner (TEP:
  validation macro-F1 0.8547 both). One TEP delivery file's output was 327 MB with it on.
- **Listing:** `GET /agents/deployments?package_id=`, `GET /agents/sessions?deployment_id=`,
  each filterable by `status` and `project_id`.
- **Deleting:** a session (cancelled first if still going), then its deployment (stopped
  first), then the package. A promoted blueprint and its optimization deleted fine with an
  admin key; a regular key gets 403 for blueprints.
- **Outputs can be cut short at `completed`**: download again until the last window
  reaches the input's end. **The slowest run sets a delivery's time.**

## Agent serving (sessions on dev 2026-10-30; through bundles: production 2026-10-29, Dev 2026-10-15, Staging 2026-10-23)

- **A serving deployment holds a serving worker** (Dev has 3, shared by everyone) until it is
  stopped; cancelling its sessions leaves it `ready`. A query to a session that has ended, or
  whose deployment has, answers 409.
- **The promoted blueprint is served as it is.** It reads and writes CSV; deployed for
  serving, the platform supplies an `atai-records` serving source and sink in their place, so
  a package of it is all serving needs.
- **Through sessions (dev, 2026-10-30):** the sample at step 8 matched batch on 230/230
  windows (score within 1.3×10⁻⁵ relative, the same first alarms) at an 836 ms median per
  query, p95 901 ms; package, deployment and session were ready in about 10 s.
- **Staging (2026-10-23):** the TEP quickstart's replay matched batch on 1,826 / 1,826 windows
  at **~2.6 s median per query**, about three times Dev's and RED's or OSM's ~0.9 s there.
- **A query holds the whole window, and the agent keeps nothing between queries.** It numbers
  the samples 0, 1, 2… itself and answers with sample indices for `start_timestamp` and
  `finish_timestamp`; records come back with their keys in alphabetical order.
- **Served equals batch to float32 noise.** On production (2026-10-29) the sample and the TEP
  quickstart, 1,826 windows each, matched batch on every label, score (within 4×10⁻⁵
  relative) and first alarm. On Dev (2026-10-15) one window in each k-5 replay was 1.6–2% off,
  its label unchanged (most likely a neighbour swap). Served embeddings differ from batch by
  about 2×10⁻⁴ relative.
- **Latency: ~0.9 s median per one-window query on production** (sample: 907 ms, p95 1.0 s,
  worst 2.0 s; the TEP quickstart 928 ms). Free-trial accounts serve on CPU only and can be
  slower. A busy Dev stretched the sample's tail to a 3.1 s p95 and a 30 s worst query.
  16 windows per query took 11 s, 13% less per window.
- **Listing:** `GET /agents/deployments?status=ready`, then `select(.kind=="serving")`, shows
  the serving deployments holding workers.
