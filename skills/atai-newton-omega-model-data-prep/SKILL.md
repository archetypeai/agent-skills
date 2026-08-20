---
name: atai-newton-omega-model-data-prep
description: >
  Clean, split, and featurize multivariate time-series data before
  embedding it with the Omega model. Bundles three reusable building
  blocks: `DataPreprocessor` (diagnose timestamp regularity, gaps, and
  nulls; build continuous gap-aware temporal blocks with imputation),
  `DataSplitter` (out-of-time or random train/test split that respects
  temporal order), and `FeaturePreparer` (pivot per-sensor embeddings
  into a "joint state" (X, y, metadata) matrix with optional L2 /
  standardization / PCA). Use this skill when the user is preparing raw
  sensor CSVs for the Omega model (`atai-newton-omega-model`), when
  training data has gaps and they're unsure whether to drop / impute /
  split, or when an n-shot CSV looks noisy and they want a principled
  cleanup pipeline.
  Do NOT use this skill to run inference or produce embeddings (use
  `atai-newton-omega-model`).
  Do NOT use for video / image / text data (use `atai-newton-fusion-model`)
  — time-series only.
---

# Newton Data Prep — Clean → Split → Featurize

A pre-modeling pipeline for time-series sensor data. Three composable building blocks that take a raw multivariate dataframe and hand back the `(X, y, metadata)` arrays a downstream KNN / Isolation Forest classifier expects.

> **Origin.** The three vendored scripts originated from work by Lucas (Solutions Engineering) and have been used end-to-end in real prep pipelines. The repo copy is the source of truth — ping Lucas if you need to pull in a newer revision.

## When to Apply

- User has raw multivariate sensor CSVs (1+ sensors, irregular timestamps, NaN gaps) and is about to embed them with [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md), and needs to clean the data first.
- User asks "should I drop rows with NaNs?" / "how do I handle gaps?" / "what's a good train/test split for time-series?"
- User's classifier is suspiciously good or suspiciously bad and you suspect temporal leakage — `DataSplitter(mode='oot')` is the fix.
- User has per-sensor embeddings (from the Omega model) and needs to fold them into a single feature matrix for KNN — that's `FeaturePreparer`.
- User wants the "joint state" pattern described in [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md) in code form.

**Use the external [`omega-1-4-preflight`](https://github.com/archetypeai/omega-1-4-preflight) checks instead when**: you want a *read-only go/no-go gate* before committing to a run. Preflight makes no changes to the data. This skill makes changes — block-splitting, imputation, dimensionality reduction. The two are complementary: preflight tells you whether the dataset is salvageable; this skill cleans it up.

**Do not use this skill when**:
- The task is video, image, or text (use [`atai-newton-fusion-model`](../atai-newton-fusion-model/SKILL.md)).
- Data is already pristine (regular sampling, no NaNs, no leakage risk) — the pipeline becomes a no-op and you can hand the dataframe directly to the downstream skill.

## The Three Building Blocks

### 1. `DataPreprocessor` — diagnose + gap-aware blocking

[references/data_preprocessor.py](references/data_preprocessor.py)

A "block" is a continuous segment where all selected sensors have data, with no NaN run longer than `gap_threshold_samples`. Long-gap regions split into separate blocks because imputing across a long gap would fabricate data.

**Two-phase usage:**

```python
from data_preprocessor import DataPreprocessor

prep = DataPreprocessor(
    timestamp_col='timestamp',
    sampling_rate_minutes=1,
    gap_threshold_samples=5,        # NaN runs > 5 samples split into a new block
    imputation_method='linear',     # or 'time', 'ffill', 'bfill', 'spline' + kwargs
)

# Phase 1: diagnose (read-only). Decide gap_threshold from this.
report = prep.diagnose(df_raw, verbose=True, plot=True)

# Phase 2: build. Returns the cleaned df with added 'block_id' + 'imputed' columns.
df_clean = prep.build(df_raw)
```

**`diagnose()` returns** a structured `{'timestamp': {...}, 'nulls': {...}}` report covering: sampling-rate CV (regularity), top-N gaps with `(starts_at, ends_at, size)`, per-column null counts, and **max consecutive null run** per column — the most important number for choosing `gap_threshold_samples`. With `plot=True` it also produces a delta histogram + missingness map.

**`build()` adds two columns to the output:**
- `block_id` (int): which continuous block the row belongs to
- `imputed` (bool): True if any sensor in this row was originally NaN

Rows inside a long-gap region are dropped entirely (not imputed) because the run would have crossed the threshold.

**Important params:**
- `gap_threshold_samples`: set based on `_diagnose_nulls`'s `max_consecutive_null`. Conservative default is 5; for noisier sensors set it higher, but never high enough to bridge a real outage.
- `imputation_method`: `'linear'` is the right default. Use `'time'` if your sampling isn't perfectly regular. Use `'ffill'` / `'bfill'` only for categorical-looking sensors that shouldn't be interpolated.
- `drop_sensors`: any sensor known to be unreliable / customer-irrelevant. Dropped *before* gap detection so its NaNs don't split blocks unnecessarily.

### 2. `DataSplitter` — out-of-time train/test

[references/data_splitter.py](references/data_splitter.py)

Time-series cross-validation done wrong = false confidence. Random splits leak the future into the training set; OOT splits do not.

```python
from data_splitter import DataSplitter

splitter = DataSplitter(mode='oot', test_size=0.3, timestamp_column='timestamp')
X_train, X_test, y_train, y_test, meta_train, meta_test = splitter.split(X, y, metadata)
```

Default to `mode='oot'`. Only use `mode='random'` as a sanity check ("am I learning anything at all?") — never as the production split.

`y` may be `None` for unsupervised tasks (e.g. Isolation Forest), in which case `y_train` / `y_test` come back as `None`.

### 3. `FeaturePreparer` — joint state featurization

[references/feature_preparer.py](references/feature_preparer.py)

Converts a per-sensor-per-window embeddings dataframe into a single `(X, y, metadata)` matrix where each row is the **joint state** of all sensors at a given window (concatenated embeddings — sometimes called "philosophy 2" in Omega notes). Optionally L2-normalizes, standardizes, and/or PCA-reduces.

```python
from feature_preparer import FeaturePreparer

preparer = FeaturePreparer(
    normalize='l2',          # or 'standardize', or None
    reduce_dim=50,           # or None to skip PCA
    sensor_order=None,       # None = alphabetical, fixed across train/test
)

# df_emb is the output of an Omega embedding step
# (see atai-newton-omega-model for how to produce it).
X, y, metadata = preparer.prepare(df_emb, label_column='machine_state')
```

**Important rules:**
- `sensor_order` must be the same for train and test. If you let it default to alphabetical, it is — but pin it explicitly when you have any doubt.
- Per-window labels are required to be consistent across sensors of the same window. The class raises if it sees disagreement.
- `reduce_dim` must be < D *and* < N. For small datasets, skip PCA.
- `normalize='l2'` is the right default when downstream is KNN with cosine-like distances; `'standardize'` if downstream is a metric / linear model.

## The Full Pipeline

```python
# 1. Diagnose, then clean
prep = DataPreprocessor(timestamp_col='timestamp', sampling_rate_minutes=1, gap_threshold_samples=5)
report = prep.diagnose(df_raw, verbose=True)
df_clean = prep.build(df_raw)

# 2. Embed (see atai-newton-omega-model) — one row per (sensor, window) with an embedding column.
df_emb = embedding_generator.generate(
    df_clean, data_columns=['sensor_a', 'sensor_b'], label_columns=['machine_state'],
)

# 3. Featurize into (X, y, metadata)
preparer = FeaturePreparer(normalize='l2', reduce_dim=50)
X, y, metadata = preparer.prepare(df_emb, label_column='machine_state')

# 4. Split out-of-time
splitter = DataSplitter(mode='oot', test_size=0.3)
X_train, X_test, y_train, y_test, meta_train, meta_test = splitter.split(X, y, metadata)

# 5. Hand to your classifier of choice (KNN / IsolationForest).
```

## Composing with Other Skills

| Skill | Relationship |
|-------|--------------|
| [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md) | **Sibling + downstream** — the Omega embedding step that sits between `DataPreprocessor.build()` and `FeaturePreparer.prepare()`, then consumes the resulting `(X, y, metadata)` for per-window KNN against the `/query` Omega embedding. This skill is its on-ramp (clean) and off-ramp (featurize). |
| [`omega-1-4-preflight`](https://github.com/archetypeai/omega-1-4-preflight) | **Upstream gate (external repo)** — read-only static checks. Run *before* this skill to decide whether the dataset is worth cleaning. |

## Common Pitfalls

- **Random split on time-series.** Almost always wrong. Use `mode='oot'`. The exception: ablation experiments where you explicitly want to ignore temporal order to isolate a non-temporal effect.
- **Imputing across long gaps.** Don't raise `gap_threshold_samples` to "make more data" — you're fabricating ground truth. If the gap is long enough that interpolation isn't credible, that region belongs in a separate block (or dropped).
- **Per-window label disagreement.** If `FeaturePreparer.prepare()` raises on label inconsistency, the issue is upstream: your label propagation strategy across the embedding step (`'last'` / `'first'` / `'mode'`) doesn't match how you assigned labels to raw rows. Reconcile there, not by silencing the check.
- **Different `sensor_order` for train and test.** Silently breaks predictions because the concatenated joint-state vector is permuted. Pin `sensor_order` explicitly.
- **PCA on tiny datasets.** PCA needs both more samples than components and more dimensions than components. Skip it under ~100 samples.

## Local Setup

```bash
# In a virtualenv (system pythons are often PEP 668 "externally managed"):
python3 -m venv .venv && source .venv/bin/activate
pip install -r skills/atai-newton-omega-model-data-prep/references/requirements.txt
pip install matplotlib  # optional: only for plot=True in diagnose()
```

No Newton API key required — this skill is pure local data wrangling.

## File Layout

```
skills/atai-newton-omega-model-data-prep/
├── SKILL.md                              ← this file
├── references/
│   ├── requirements.txt                  ← runtime deps (pip install -r requirements.txt)
│   ├── data_preprocessor.py              ← gap-aware blocking + diagnostics
│   ├── data_splitter.py                  ← OOT / random train/test split
│   └── feature_preparer.py               ← joint-state featurization + PCA
└── tests/
    ├── conftest.py                       ← shared fixtures
    ├── test_data_preprocessor.py
    ├── test_data_splitter.py
    └── test_feature_preparer.py
```

## Running the Tests

```bash
pip install -r skills/atai-newton-omega-model-data-prep/references/requirements.txt pytest
pytest skills/atai-newton-omega-model-data-prep/tests/ -v
```

CI runs the suite automatically on any PR that touches `skills/atai-newton-omega-model-data-prep/**` (see `.github/workflows/test-atai-newton-omega-model-data-prep.yml`).
