---
name: newton-machine-state-batch
description: >
  Run the Machine State classification pipeline as an asynchronous batch
  job on the Archetype AI Newton platform — classify millions of rows of
  time-series sensor data into n-shot example classes without holding a
  WebSocket session open. Powered by an Omega encoder (default
  `omega_1_4_base` — generic, arbitrary channel counts) plus a hosted KNN
  classifier over n-shot example files. Covers job creation, status
  polling, event log retrieval, output download, and config optimization.
  Do NOT use for real-time streaming Machine State (use newton-machine-state).
  Do NOT use for activity detection over video (use newton-activity-monitor).
  Do NOT use for file uploads (use newton-batch-upload).
---

# Newton Machine State Classification (Batch)

Create and monitor asynchronous batch jobs that classify time-series sensor data using the Machine State pipeline. Process large datasets (millions of rows) without maintaining a real-time session. The streaming counterpart is [`newton-machine-state`](../newton-machine-state/SKILL.md) — same encoder + classifier, different orchestration. For stateless per-window classification with direct access to embeddings (no Lens, no session lifecycle), see [`newton-machine-state-direct-query`](../newton-machine-state-direct-query/SKILL.md) — `/query` with the Omega encoder plus local KNN.

## When to Apply

- User wants to classify a large CSV dataset at scale
- User asks about batch processing or asynchronous inference
- User wants to run Newton on a file without a WebSocket session
- User needs to process multiple files through a pipeline
- User wants to optimize pipeline config for best accuracy

## Machine State Classification Pipeline

Classifies time-series sensor data using n-shot examples. Newton vectorizes each sensor window with an Omega encoder, then a hosted KNN classifier predicts the class label from the n-shot examples.

**Pipeline key:** `machine-state-classification`

This is the active deployment on both stage (`api.stage.u1.archetypeai.app`) and prod (`api.u1.archetypeai.app`) as of 2026-05-16 (`v1.1.1-409-e749ac0-rev5` on stage). The previously-documented `machine-state-job-pipeline` returns `Pipeline 'machine-state-job-pipeline' has no active versions` in both environments — it is not currently a usable key. The `machine-state-classification` deployment supports `omega_1_4_base` over arbitrary channel counts (verified with 4-channel vibration), so there is no channel-padding workaround required.

### Available model types

| Model | Channels | When to use |
|---|---|---|
| **`omega_1_4_base`** | **Arbitrary** (e.g. 4, 9, 52, 100+) | **Default — start here.** Generic encoder, no channel-count constraint. Suitable for almost every dataset (process plants, drilling, IoT, energy, …). Latest model. |
| `omega_1_3_surface` | Exactly 9 | Legacy domain-specific encoder for surface drilling sensors. Use only if you're working from an existing `omega_1_3_surface` n-shot library and don't want to re-shoot examples. |
| `omega_1_3_power_drive` | Exactly 9 | Legacy domain-specific encoder for downhole power-drive sensors. Same legacy guidance as above. |

For new projects pick `omega_1_4_base` — it removes the "exactly 9 sensors" constraint that made the 1.3 models painful to use on arbitrary plant/process data, and benchmarks at least as well across the domains we've tested.

### Input ports

- `worker.inference` — CSV files to classify
- `worker.n_shots` — labeled CSV example files with `metadata.class`

### Data requirements

- CSV files must include a timestamp column (Unix epoch or any monotonically increasing numeric).
- `data_columns` lists the sensor columns the encoder will see.
  - With `omega_1_4_base`, any positive count works.
  - With `omega_1_3_surface` / `omega_1_3_power_drive`, the count must be **exactly 9** — anything else throws a shape error (see *Common Errors*).
- `window_size` controls timesteps per classification window (typical: 16 / 32 / 64 / 128). Use `window_size=1` and you'll get a tensor error.
- N-shot files must produce at least `n_neighbors` windows: `(rows − window_size) / step_size ≥ n_neighbors`.
- For drilling specifically, the `omega_1_3_surface` n-shot labels should come from ACTC rig mode codes, not sensor heuristics — control-system ground truth is more reliable than thresholded readings.

### Recommended n-shot data prep (contiguous + z-scored)

The single biggest performance lever for time-series machine-state jobs is **how the n-shot files are constructed**, not which KNN config is picked. For the upstream cleanup step (gap-aware blocking + imputation on the raw source before you slice contiguous shots), use [`newton-data-prep`](../newton-data-prep/SKILL.md)'s `DataPreprocessor`. The canonical n-shot recipe itself:

1. **Each shot file is one temporally-contiguous block of a single class.** Pick the longest contiguous run of rows for that class from your raw labeled CSV (or from a single source recording / well / snapshot file) and slice the first N rows. **Never `random.sample()` rows from a class-filtered list** — the resulting shot file has non-monotonic timestamps and zero physical contiguity, and the encoder's sliding window silently glues across the gaps between sampled rows.
2. **Don't punch random per-row holes in the inference file.** The historical "inference = labeled CSV minus the n-shot rows" pattern is fine *if* the shots came from contiguous blocks (so the holes are a small number of contiguous ranges), but is catastrophic if shots were random-sampled (thousands of single-row gaps). The cleanest pattern is to draw shots from a *different* source recording than inference (e.g., 3W's shots come from `data/raw/3W/dataset/<code>/WELL-*.parquet` while inference is the labeled CSV — they don't overlap).
3. **Z-score per file (or globally) before upload.** See *Input normalization* below — this is the second half of the same recipe.

**Why it matters.** Across the seven public batch-example repos, contiguous-shot prep moved 3W's full-run accuracy from **15.3% → 41.6% (+26pp)** on the same `omega_1_4_base` model with the same default config — bigger than any move in the 96-combo hyperparameter grid. NASA Bearing, Pump Sensor, and Volve had the same bug and have all merged the same fix; see the cross-repo writeup in [`omega-model-performance`](https://github.com/archetypeai/omega-model-performance) for the empirical pattern.

**Templates to clone from**:

| Dataset | Cadence | Contiguous-prep PR | Notable |
|---|---|---|---|
| [3W](https://github.com/archetypeai/archetypeai-batch-examples-3w) | 1 Hz, 9-class | [#1](https://github.com/archetypeai/archetypeai-batch-examples-3w/pull/1) (merged) | Per-class shots from separate WELL-*.parquet files; global z-scoring |
| [NASA Bearing](https://github.com/archetypeai/archetypeai-batch-examples-nasa-bearing) | 20.48 kHz snapshots | [#2](https://github.com/archetypeai/archetypeai-batch-examples-nasa-bearing/pull/2) | Each shot = 2,000 rows from a single 1-second snapshot file (no z-score; channels already same scale) |
| [Pump Sensor](https://github.com/archetypeai/archetypeai-batch-examples-pump-sensor) | 1-minute, binary | [#2](https://github.com/archetypeai/archetypeai-batch-examples-pump-sensor/pull/2) (merged) | Split runs on `label change OR delta != 60s` (catches hidden multi-day pauses inside label-contiguous stretches); `N_SHOT` capped at 1,000 to fit the longest clean fault run |
| [Volve](https://github.com/archetypeai/archetypeai-batch-examples-volve) | ~5 sec, 14 wells | [#1](https://github.com/archetypeai/archetypeai-batch-examples-volve/pull/1) (merged) | `well_id` carried through `volve_to_csv.py` so runs split strictly per-well (avoids interleaved-well runs in the global DATE_TIME sort) |

**Validate the shot files with [`omega-1-4-preflight`](https://github.com/archetypeai/omega-1-4-preflight) before upload.** The `timestamp_monotonic` check fails specifically on `random.sample()`-style shots; the `feature_scale` check is the cue for z-scoring.

### Input normalization

The deployed pipeline runs with `normalize_input=False` and the flag is **not currently exposed in `reader_config`** — so whatever pre-encoder normalization you want has to happen *before* you upload. The encoder sees the raw values from your CSV.

When this matters:

- **Same operating condition for shots and inference** → ignore this section; absolute amplitudes are comparable already, and within-distribution accuracy stays high.
- **Different conditions** (clean vs factory-noise recordings, different loads/speeds, different equipment instances) → the encoder will read bulk-amplitude differences as class signal. Symptom: within-distribution accuracy ≥90% but cross-condition collapses to ~majority-class predictions; "normal" files under high-noise conditions get classified as the loudest-sounding fault class.

What to do about it:

- **Z-score per file before upload.** Compute per-channel mean and std on each CSV, write the standardized values back. Removes the per-file amplitude offset cleanly; cheapest fix; usually enough.
- **Or: fit a global `StandardScaler` on the n-shot pool, apply to all inputs.** Preserves cross-file amplitude relationships. The right call if the bulk amplitude *is* part of your class signal (e.g. high-energy vs low-energy operating regimes).

This is the cloud-side stand-in for omega-local's "Option B" pattern documented in [omega-local/SKILL.md](../omega-local/SKILL.md) under *Normalization Choices*. "Option A" (per-window instance normalization inside the encoder) is the local-only path; on the cloud you get whatever the deployed encoder was trained with, which currently is `normalize_input=False`.

If you're not sure whether amplitude variation matters for your dataset, run the `feature_scale` check in [omega-1-4-preflight](https://github.com/archetypeai/omega-1-4-preflight) on your shot files — a >3-decade range gap is a strong "z-score before uploading" signal.

### Step size at high sample rates

`step_size: 1` is the canonical default in the examples below, but it's not always the right one. The number of windows the pipeline produces per file is `(rows − window_size) / step_size + 1`, and the per-window inference cost is roughly constant. For a low-rate dataset (TEP at 1 Hz, drilling at 0.1 Hz) `step_size: 1` is cheap and gives you maximum temporal resolution. For high-rate vibration / accelerometer data it isn't.

Empirical example: 4-channel 100 kHz vibration, 5-second clips (500,000 rows), `window_size: 1024`.

| `step_size` | Windows per file | Compute |
|---|---|---|
| 1 | 498,977 | hours-of-GPU per file |
| 64 | 7,797 | minutes per file |
| **1024 (= window_size)** | **488 (non-overlapping)** | **~1 s per file** |

Per-file accuracy is statistically identical across all three — the discriminative signal lives in the window content, not in the sliding stride. **For accuracy-evaluation runs at sample rates ≥10 kHz, default to `step_size == window_size`** (non-overlapping windows, ~1 prediction per `window_size` rows of raw signal). Reserve `step_size: 1` for cases where you actually need per-row temporal resolution (e.g. drift detection or onset localization).

If you submit a step-1 job at 100 kHz and the events log says "Batch N for foo.csv: 240,000 success" and keeps climbing, you've made this mistake — cancel via `POST /v0.5/batch/jobs/{id}/cancel` and resubmit.

### Within-condition pilot vs cross-condition reality

The single most common pattern when promoting a config from preflight pilot to production: **pilot accuracy systematically over-estimates cross-condition accuracy by 20–45 percentage points** on time-series data with frozen encoder embeddings. The pilot tests "shot file vs held-out slice of the same shot file" — same operating condition, same equipment instance, often same recording session. Cross-condition inference is a strictly harder task.

Observed empirically:

| Dataset | Pilot accuracy | Cross-condition accuracy | Gap |
|---|---|---|---|
| TEP (per omega-1-4-preflight README) | 0.784 | 0.506–0.537 | ~25 pp |
| SSCC chain-conveyor fault (4-channel 100 kHz vibration) | 0.963 | 0.527 | 44 pp |

This isn't a tuning failure — it's the cost of using KNN over a frozen embedding space whose class regions overlap when operating conditions shift. The gap will not close by reweighting KNN, switching metrics, or expanding the n-shot library by 2–3×; we've observed multiple distinct configurations clamping at the same cross-condition number once you've moved beyond one or two operating regimes. To break past this requires changing what the encoder *sees* — fine-tuning, a different encoder, or pre-engineered features (spectrograms / wavelet bands / cyclostationary indicators) — not how the classifier votes.

**Recommended evaluation protocol:**

1. Run omega-1-4-preflight `--pilot` against your shot files. PASS verdict ≠ done.
2. Before deploying, evaluate on **at least one** inference file from an operating condition that is *not* represented in your shot library. If the per-class accuracy survives that, you have evidence beyond within-distribution.
3. If pilot says 0.95 and your one cross-condition file scores 0.30, that's the diagnosis — you're stuck inside one operating regime and the n-shot library doesn't span the production envelope yet.

### One-vs-rest (5 binary classifiers) for per-class recall

When the multi-class accuracy hits a ceiling but you need high recall on a specific class (e.g. "always catch normal", or "never miss screwdrop"), submit one binary `machine-state-classification` job per class instead of a single multi-class job:

- **Positives**: all n-shot files for class C, metadata `{"class": C}`.
- **Negatives**: ~2 shots per class from every other class, metadata `{"class": f"not_{C}"}`. Roughly balance positives to negatives.
- **Per-class config**: pick the metric / weights / library composition that works best for *that* class — they don't have to match across classifiers.

This is genuinely useful when class regions in the embedding space are mutually non-overlapping for some pairs (dry / lean) but overlapping for others (loose / screwdrop / noisy-normal). The per-class binaries can hit ≥90% recall on the well-separated classes (file-level recall of 1.000 for SSCC `dry` and `lean` in the same setup that hit 0.71 multi-class).

**Caveat: argmax combining is the weak link.** When several binaries fire simultaneously on the same input (because their positive regions overlap), `argmax(p_C)` across classifiers doesn't disambiguate cleanly. In the SSCC case, the `normal` binary scored ≥0.75 on every normal file (perfect threshold-rated recall) but lost the argmax to the loose/screwdrop binaries that scored even higher on the same files. Decision-rule fixes (margin-based voting, confidence calibration, learned combiners) are not currently exposed by the platform — if the failure mode bites, either consume the binary outputs yourself or fall back to the multi-class job.

### Anomaly-detection deployment pattern

For production monitoring, the right shape is often **not** multi-class — it's a single binary classifier that distinguishes the "normal" operating state from "anything else." Operators usually need to know *that* a fault occurred (so they can dispatch maintenance); naming the specific failure mode is downstream.

The recipe:

- **Positives**: 10–20 shots labelled `normal`, spanning the operating conditions you actually run in production (different speeds / loads / noise levels).
- **Negatives**: 5–10 shots per known fault class, all labelled `anomaly`.
- **Threshold the output**: `p(normal) < 0.5` ⇒ anomaly. Tighten the threshold (0.3, 0.2) for higher recall at the cost of false positives.

The deployment is simpler, the n-shot library is smaller, and you can extend it incrementally as new fault modes show up in the field — just add more `anomaly` shots, no schema change. Expect high recall on `normal` and the failure mode of *missing* anomalies whose embeddings sit close to the normal region (in SSCC that's `loose` and `screwdrop` under factory noise — their embeddings genuinely look "normal-ish" to omega_1_4_base).

### Class-imbalanced n-shot libraries — what to know

KNN votes are sensitive to library composition. If you ship 10 shots for `normal` and 5 each for the four fault classes, the decision boundary slides toward `normal` — every borderline window in the embedding space is more likely to find a `normal` neighbor among its top 5 nearest. We observed this directly on SSCC: imbalancing the library normal-heavy fixed the diagnosed `normal-under-noise` failure mode (recall jumped from 0.13 → 0.66 on the two affected files) but cost 12–14 pp on the loose and screwdrop classes that paid the symmetric tax.

Use this deliberately when it serves your deployment (e.g. an anomaly detector wants the decision boundary biased toward `normal`); use balanced libraries when overall classification accuracy is what you care about. `weights="distance"` partially de-biases imbalance — closer neighbours get larger votes — but doesn't eliminate it.

### Example: process-plant classification with `omega_1_4_base` (52 variables)

Classifying a chemical-plant process dataset (the public [Tennessee Eastman Process](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) benchmark — 52 process variables, 21 fault types) into binary `normal` vs `fault` from two n-shot example files. This is the canonical `omega_1_4_base` shape — the config below is what you should clone for any new project, regardless of domain.

```bash
curl -s -X POST "$BASE_URL/batch/jobs" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "tep-classification",
    "pipeline_type": "batch",
    "pipeline_key": "machine-state-classification",
    "inputs": {
      "worker.inference": [{"file_id": "tep_inference.csv"}],
      "worker.n_shots": [
        {"file_id": "tep_normal.csv", "metadata": {"class": "normal"}},
        {"file_id": "tep_fault.csv",  "metadata": {"class": "fault"}}
      ]
    },
    "parameters": {
      "worker": {
        "parallelism": 1,
        "config": {
          "model_type": "omega_1_4_base",
          "batch_size": 32,
          "reader_config": {
            "data_columns": ["xmeas_1","xmeas_2", "...", "xmv_11"],
            "timestamp_column": "timestamp",
            "window_size": 64,
            "step_size": 1
          },
          "classifier_config": {
            "n_neighbors": 5,
            "metric": "euclidean",
            "weights": "uniform",
            "normalize_embeddings": false
          },
          "flush_every_n_iteration": 150
        }
      }
    }
  }'
```

Key knobs and how they differ from the 1.3 defaults:

- `model_type: "omega_1_4_base"` (was `omega_1_3_surface` / `omega_1_3_power_drive`).
- `batch_size: 32` (was 8) — 1.4 is faster per-window, larger encoder batches don't bottleneck.
- `flush_every_n_iteration: 150` (was 1000) — write outputs more often for better mid-job visibility.
- `classifier_config.normalize_embeddings: false` — new field exposed by the 1.4 pipeline. L2-normalizing the 768-dim embeddings before KNN occasionally helps when channel scales are wildly heterogeneous; default `false` matches what we've benchmarked best on so far.

### Example: drilling classification with the legacy 1.3 encoder

Pinned to `omega_1_3_surface` because the n-shot library exists for the 9 surface channels. Full source in [archetypeai-batch-examples-volve](https://github.com/archetypeai/archetypeai-batch-examples-volve).

```bash
curl -s -X POST "$BASE_URL/batch/jobs" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "drilling-classification",
    "pipeline_type": "batch",
    "pipeline_key": "machine-state-classification",
    "inputs": {
      "worker.inference": [{"file_id": "volve_inference.csv"}],
      "worker.n_shots": [
        {"file_id": "volve_drilling.csv",     "metadata": {"class": "drilling"}},
        {"file_id": "volve_not_drilling.csv", "metadata": {"class": "not_drilling"}}
      ]
    },
    "parameters": {
      "worker": {
        "parallelism": 1,
        "config": {
          "model_type": "omega_1_3_surface",
          "batch_size": 8,
          "reader_config": {
            "data_columns": ["BPOS","DBTM","FLWI","HDTH","HKLD","ROP","RPM","SPPA","WOB"],
            "timestamp_column": "DATE_TIME",
            "window_size": 64,
            "step_size": 1
          },
          "classifier_config": {"n_neighbors": 5, "metric": "euclidean", "weights": "uniform"},
          "flush_every_n_iteration": 1000
        }
      }
    }
  }'
```

Drilling channel list (the required nine for `omega_1_3_surface`):
```
DATE_TIME, BPOS, DBTM, FLWI, HDTH, HKLD, ROP, RPM, SPPA, WOB
```

## Monitoring Jobs

### Check Status

```bash
curl -s "$BASE_URL/batch/jobs/$JOB_ID" \
  -H "Authorization: Bearer $ATAI_API_KEY"
```

Status lifecycle: `PENDING` → `RUNNING` → `COMPLETED` / `FAILED` / `CANCELLED`

### View Events/Logs

```bash
curl -s "$BASE_URL/batch/jobs/$JOB_ID/events" \
  -H "Authorization: Bearer $ATAI_API_KEY"
```

### Download Outputs

```bash
# List output artifacts (paginated, presigned S3 URLs, 1-hour expiry)
curl -s "$BASE_URL/batch/jobs/$JOB_ID/outputs?limit=50&offset=0" \
  -H "Authorization: Bearer $ATAI_API_KEY"

# Download artifact (no auth needed — signature is in the URL)
curl -s -o output.csv "$PRESIGNED_URL"
```

Machine State output format:
```csv
DATE_TIME,Prediction
1232522644,drilling
1190749684,not_drilling
```

### List All Jobs

```bash
curl -s "$BASE_URL/batch/jobs?limit=10&offset=0" \
  -H "Authorization: Bearer $ATAI_API_KEY"
```

## Config Optimization

The Machine State pipeline has several tunable hyperparameters. Use grid search with a small test dataset (200 rows) to find the best config before running on the full dataset:

| Parameter | Values to try | Description |
|-----------|--------------|-------------|
| `window_size` | 16, 32, 64, 128 | Time steps per classification window |
| `n_neighbors` | 3, 5, 7, 11 | KNN neighbors |
| `metric` | euclidean, cosine, manhattan | Distance metric |
| `weights` | uniform, distance | KNN weight function |

See [`optimize_config.py` in archetypeai-batch-examples-volve](https://github.com/archetypeai/archetypeai-batch-examples-volve/blob/main/3_batch_jobs/optimize_config.py) for a working grid-search script. The pattern transfers to `omega_1_4_base` unchanged — just swap `model_type` and `data_columns`.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v0.5/batch/jobs` | POST | Create a batch job |
| `/v0.5/batch/jobs` | GET | List all jobs |
| `/v0.5/batch/jobs/{job_id}` | GET | Get job status |
| `/v0.5/batch/jobs/{job_id}/events` | GET | Get job events/logs |
| `/v0.5/batch/jobs/{job_id}/outputs` | GET | List output artifacts (paginated, presigned URLs) |

## Fine-Tuning (TBD)

Fine-tuning via `POST /v0.5/internal/experiment/runner/jobs` is not yet available. It requires JSONL-formatted training data — see the JSONL converter in [archetypeai-batch-examples-volve](https://github.com/archetypeai/archetypeai-batch-examples-volve).

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `shape '[-1, 6912]' is invalid for input of size N` | Using `omega_1_3_surface` / `omega_1_3_power_drive` with a `data_columns` count ≠ 9 | Either supply exactly 9 columns, or switch `model_type` to `omega_1_4_base` (no channel-count constraint) |
| `n_neighbors must be <= number of samples in index` | N-shot files too small for `window_size` | Increase n-shot rows or decrease `window_size` |
| `could not parse as dtype i64` | Mixed int/float values in CSV | Ensure all sensor values are formatted as floats |
| `Failed to resolve file inputs` | File not uploaded | Upload file first via Files API ([newton-batch-upload](../newton-batch-upload/SKILL.md)) |
| `Pipeline 'X' has no active versions` | Pipeline not deployed in this environment | Verify the pipeline_key is deployed — the active key on stage and prod is `machine-state-classification` (the newer `machine-state-job-pipeline` is documented but not currently deployed in either env; if a tool defaults to it you'll need to override). |

## Best Practices

- **Default to `omega_1_4_base`** — generic, no channel-count constraint, and the model we ship as the recommended starting point. Only fall back to `omega_1_3_*` if you have an existing n-shot library tied to that encoder.
- **Start with quick tests** — use 200-row samples (contiguous, not random — see TEP `tep_quick_test_200.csv`) to verify the pipeline before launching a multi-hour full run. For optimization grids specifically, ship a *class-balanced* slice (e.g., 3W's `3w_opt_slice.csv`, NASA Bearing's `bearing_opt_slice.csv`) rather than a single-class quick test — a pure-class slice gives the grid search no F1 signal because every config that finds the class trivially scores 100%.
- **Use control-system labels for n-shots when available** — e.g. ACTC rig mode codes for drilling, plant DCS state codes for process plants. More reliable than thresholded-sensor heuristics.
- **Optimize config first** — grid search over hyperparameters with small data before committing to a multi-hour full run. Both example repos ship an `optimize_config.py`.
- **Poll status periodically** — jobs can run for minutes to hours depending on data size. Lower `flush_every_n_iteration` if you want mid-job progress visibility.
- **Check events on failure** — the `/events` endpoint provides detailed error messages including row-level context.
- **Watch n-shot window math** — `(rows − window_size) / step_size ≥ n_neighbors`. Easy to under-shoot when bumping `window_size` mid-experiment.
- **For sample rates ≥10 kHz, default to `step_size = window_size`** — non-overlapping windows give the same per-file accuracy at ~1000× less compute. Reserve `step_size: 1` for cases that need per-row temporal resolution. See *Step size at high sample rates*.
- **Always do at least one cross-condition spot-check before promoting** — pilot accuracy systematically over-estimates cross-condition accuracy by 20–45 pp on time-series data with frozen embeddings. See *Within-condition pilot vs cross-condition reality*.

## Example Code

[**archetypeai-batch-examples-volve**](https://github.com/archetypeai/archetypeai-batch-examples-volve) — full public end-to-end pipeline for the Machine State batch flow: data prep → upload → batch job → outputs → evaluation → config optimization, in Python, shell, and curl. Pinned to `omega_1_3_surface` over 9 drilling channels (Volve North Sea well data), but the orchestration code, optimize_config grid search, and evaluation utilities transfer to `omega_1_4_base` unchanged — switch `model_type` and broaden `data_columns` to match your sensor set.

Use this repo as a template even if your default model is `omega_1_4_base`. The only Volve-specific bits are the `data_columns` list and the n-shot files; everything else (upload helpers, job creation, polling loop, output download, evaluation) is reusable as-is.
