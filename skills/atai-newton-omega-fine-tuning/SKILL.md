---
name: atai-newton-omega-fine-tuning
description: >
  Fine-tune a classification head on Archetype AI's Newton Omega 1.5
  time-series encoder with the `/fine-tuning` API, run the head on new
  sensor CSVs in a `machine-state-classification` batch job, and score it.
  Use when the user has labelled sensor recordings (one class per file:
  normal vs fault, operating states) with enough data per class, and wants
  a trained classifier on frozen Omega embeddings instead of a nearest-
  neighbour vote. Covers what Omega fine-tuning is (a head only: the
  encoder's weights don't change), the CSV and `n_shot` file format, the
  create-job body and reader window, checkpoints (best = highest step; a
  single 0.33 checkpoint means the head never learned), attaching the head
  on `worker.fine_tune_checkpoint` with the still-required `worker.n_shots`,
  matching the window, the output CSV, and scoring. Do NOT use for KNN on
  Omega with no training (the batch job's own n-shot KNN, or
  `atai-newton-omega-inference-api` / `atai-newton-omega-model` client-side),
  for agents built on the Agents API (`atai-operational-state-monitoring-agent`,
  which fits its own classifier), or for Newton C (`atai-newton-c-fine-tuning`).
---

# Fine-Tuning an Omega Head

Train a small classifier ("head") on frozen Omega 1.5 embeddings of your labelled windows, then run it over new recordings in a batch job. Worked example with full results: [fine-tuning-example-newton-omega](https://github.com/archetypeai/fine-tuning-example-newton-omega) (SWaT water-treatment plant, normal vs attack).

## When to Apply

- The user has labelled sensor CSVs, one class per file, and wants a trained classifier over Omega embeddings
- A nearest-neighbour vote over a few examples isn't accurate enough, and there is enough labelled data to train on (see [How much data](#how-much-data))
- The user wants to know whether a trained head beats KNN on their data

**Do not use this skill when**:
- There are only a handful of labelled windows per class: use KNN (the batch job's own n-shot KNN, or client-side with [`atai-newton-omega-inference-api`](../atai-newton-omega-inference-api/SKILL.md))
- The user wants an agent on the Agents API: [`atai-operational-state-monitoring-agent`](../atai-operational-state-monitoring-agent/SKILL.md) fits its own classifier with the Optimizations API
- The model to fine-tune is Newton C: [`atai-newton-c-fine-tuning`](../atai-newton-c-fine-tuning/SKILL.md)

## What Omega fine-tuning is

```
window of sensor rows → [Omega 1.5 encoder, frozen] → embedding → [head] → class
                                                                   ↑ the only part trained
```

**Today, fine-tuning Omega trains a head only. The encoder's weights never change.** The head is a small neural network trained by gradient descent on the embeddings of your labelled windows; training takes minutes. The request can't choose an Omega version: `model: "omega"` resolves to the platform default, which the job reports as `base_model_tag` (`newton-omega-1-5`, `mdl_79dn74y5mk8rcsp37p950c9sfp` on production, 2026-11-23).

**Where a fine-tuned head runs:** only in a `machine-state-classification` batch job (verified on production, 2026-11-23). The Direct Query and Inference APIs serve the base Omega, and no agent node loads a fine-tuned head: the agents' classifier nodes are kNN, centroid or LOF, fitted while building the agent.

## The Run

```bash
cd skills/atai-newton-omega-fine-tuning/references
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://platform.u1.archetypeai.app/api
EOF

python3 run_omega_ft.py train --data data    # upload train/ + val/, fine-tune the head
python3 run_omega_ft.py infer --data data    # upload test/, batch job with the head
python3 run_omega_ft.py score --data data    # accuracy, confusion, per-class P/R/F1
```

Stdlib only. Every id goes to `omega-ft-state.json` as soon as it exists; `train --job <ftj_…>` and `infer --job <job_…>` re-attach after a disconnect.

## Data

```
data/train/<class>/*.csv    training files (fine-tune `training_files`)
data/val/<class>/*.csv      validation files (fine-tune `validation_files`, picks the checkpoint)
data/test/<class>/*.csv     held-out files (batch inference, scored once)
```

- Each CSV: a `timestamp` column, then numeric channels. The timestamp is read as an opaque string, so a real datetime works.
- **One class per file.** The class travels with the file, not as a column: `{"type": "n_shot", "file_id": …, "label": "<class>", "format": "csv"}` in the fine-tune, and `metadata: {"class": …}` on `n_shots` in the batch job.
- Scale every channel (z-score) with **one** set of statistics for all splits before upload. The worked example uses statistics over the whole dataset.
- Keep the splits disjoint in time: a window must not straddle two splits.
- **File ids are file names**, in an org-wide namespace, and uploading an existing name is skipped (409). Rebuilt files under an old name keep the old contents, so the helper puts a content hash in every name ([`upload_name()`](references/omega_ft.py)).
- Upload with the multipart flow (`/files/uploads/initiate` → `PUT` parts → `complete`), `file_type: "text/csv"`.

## Fine-Tune Job

```json
POST /fine-tuning/jobs
{
  "model": "omega",
  "name": "swat-head",
  "seed": 0,
  "method": {
    "type": "head",
    "head": {
      "hyperparameters": {"batch_size": 32, "n_epochs": 30},
      "reader": {"window_size": 128, "step_size": 4, "data_columns": ["FIT101", "LIT101", "…"]}
    }
  },
  "training_files":   [{"type": "n_shot", "file_id": "swat_normal_train_0.csv", "label": "normal", "format": "csv"}, …],
  "validation_files": [{"type": "n_shot", "file_id": "swat_normal_tune_0.csv",  "label": "normal", "format": "csv"}, …]
}
```

- **`reader`** cuts each file into windows: `window_size` rows (16–1024), every `step_size` rows. The window you train at is the window you must infer at.
- **Only `batch_size`, `n_epochs` and the reader are configurable.** No head architecture, no learning rate.
- **Statuses:** `PENDING` → `ADMITTED` → `RUNNING` → `COMPLETED`; accept `SUCCEEDED` too.
- **Checkpoints:** the worker saves one each time validation macro-F1 improves, so **the highest step is the best**. `GET /fine-tuning/jobs/{id}/checkpoints` → `[{id: "ckp_…", step, metrics: {macro_f1}}]`.
- **A single checkpoint at the one-class score means the head never learned.** With two balanced classes that is macro-F1 0.333 at step 1: the head predicts one class everywhere. `train` warns when it sees it ([How much data](#how-much-data)).
- **Seeds matter.** The same data gave macro-F1 0.75–0.92 across six seeds (June 2026). Pin `seed`; if a run underperforms, try another.
- **Timing (production, 2026-11-23):** 240 s for 8 files × 5,000 rows; 59 s for 2 files × 2,000 rows.
- **Checkpoints belong to the org that trained them.** There is no `fine_tuned_model` name; you attach the `ckp_…` id.

### The `knn` method

The API also accepts `method.type: "knn"` (store the training embeddings as a checkpoint). **It doesn't run on production today:** its pipeline, `knn-fine-tuning-omega`, is still on 1.1.10 from before platform 2.0.1 and its job fails with `ImagePullBackOff` (2026-11-23). For KNN on Omega, run `machine-state-classification` with your labelled files on `n_shots` and no checkpoint: the job builds the KNN itself.

## Running the Head

```json
POST /batch/jobs
{
  "name": "swat-head-eval",
  "pipeline_type": "batch",
  "pipeline_key": "machine-state-classification",
  "inputs": {
    "worker.inference":            [{"file_id": "swat_normal_eval_0.csv"}, …],
    "worker.n_shots":              [{"file_id": "swat_normal_tune_0.csv", "metadata": {"class": "normal"}}, …],
    "worker.fine_tune_checkpoint": [{"kind": "checkpoint", "checkpoint_id": "ckp_…", "metadata": {}}]
  },
  "parameters": {"worker": {"parallelism": 1, "config": {
    "model_type": "omega_1_5_base",
    "reader_config": {"window_size": 128, "step_size": 1, "timestamp_column": "timestamp", "data_columns": ["…"]},
    "batch_size": 32
  }}}
}
```

- **Attach the head on `worker.fine_tune_checkpoint`.** Without it the job builds a KNN from `n_shots` instead.
- **`worker.n_shots` is required anyway.** The pipeline's input schema marks it `required: true`; with a checkpoint attached the worker ignores it. Any labelled files will do (the skill sends the validation split).
- **`window_size` must equal the fine-tune's window.** The head only understands embeddings from the window it was trained on; a mismatch runs and quietly scores badly. `step_size` here can differ (1 scores every window).
- **Confirm the head was used from the events:** `preparing_classifier: Loading fine-tuned configuration`, then `inferencing`. A KNN run shows `train.started` on the n-shot files and `Training classifier` instead.
- `classifier_config` (KNN settings) is unused with a checkpoint; leave it out.
- **Output:** one CSV per input file, `timestamp,Prediction`, one row per window (the timestamp of the window's last row). Each output entry names its input in `input_file_id`; match classes by that, not by the output file name.
- **Timing:** ~35 s per 5,000-row file at step 1; 12 files ≈ 7 min.
- Outputs, events and checkpoints are cursor-paginated (`data`, `has_more`, `next_cursor`, newest first; next page `after=<next_cursor>`). See [`list_all()`](references/omega_ft.py).

## How much data

SWaT, normal vs attack, 51 sensors at 1 Hz (40 channels kept), window 128, held-out 54,476 windows, Omega 1.5, seed 0, production 2026-11-23:

| Classifier | Labelled data | Accuracy | Macro-F1 | Attack recall | Normal recall |
|---|---|---|---|---|---|
| Head | 2,000 rows/class (~470 windows at step 4) | 49.9% | 0.333 | 0.0% | 99.8% |
| KNN in the batch job, no fine-tune | 2,000 rows/class | 79.7% | 0.796 | 87.6% | 71.9% |
| **Head** | 20,000 rows/class | **90.3%** | **0.903** | **88.1%** | **92.5%** |

- **A head needs enough data.** On 2,000 rows per class it never learned (one checkpoint, 0.333).
- **With little data, KNN wins; with enough, the head wins.** Here the head caught the same attacks as KNN with 3.7× fewer false alarms (2,055 vs 7,659 normal windows misclassified).
- On 2026-11-23 the head and KNN numbers matched an earlier run (July 2026, platform 1.x) window for window.

## Common Pitfalls

- **Head scores ~chance.** Check the checkpoints: a single step-1 checkpoint at the one-class score means it never learned. Add data, or lower `step_size` for more (overlapping) windows.
- **Good validation, poor test.** The window at inference differs from training. Use the fine-tune's `window_size`.
- **Batch job refused.** Check that `worker.n_shots` is there: the schema requires it even with a checkpoint.
- **"Fine-tuned" run identical to KNN.** The checkpoint wasn't attached: look for `Loading fine-tuned configuration` in the events.
- **Old data after re-uploading.** The name already existed (409, skipped). Hash the content into the name.
- **`knn` fine-tune fails to start.** Not deployed on production today; see [The `knn` method](#the-knn-method).
- **Missing outputs.** Only the first page was read; follow `next_cursor`.
- **Wrong environment.** `ATAI_API_*` exported in the shell override `.env`.

## File Layout

```
skills/atai-newton-omega-fine-tuning/
├── SKILL.md                 ← this file
├── references/
│   ├── omega_ft.py          ← stdlib client: config, CSV upload, pagination, job bodies, outputs, scoring
│   ├── run_omega_ft.py      ← train · infer · score · status · cancel
│   └── .env.example
└── tests/
    └── test_references.py   ← network-free unit tests
```
