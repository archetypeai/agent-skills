---
name: omega-preflight
description: >
  Predict whether a binary time-series dataset is likely to reach ~70%
  accuracy on Archetype AI's `omega_1_4_base` via the
  `machine-state-classification` pipeline — before committing to a full
  batch run. Runs 10 fast local static checks (schema, timestamp
  monotonicity, missing values, constant columns, feature-scale
  heterogeneity, n-shot support, cross-file schema match, class balance,
  window-vs-sampling translation, accuracy prior) and an optional held-out
  pilot against the real batch API. Use this skill when the user wants to
  decide *whether* to run a Machine State batch job, vet shot files before
  upload, or diagnose why a previous Machine State run underperformed.
  Do NOT use this skill to actually run the full inference job
  (use newton-machine-state-batch). Do NOT use for activity detection or
  vision pipelines. v1 is binary-only — multi-class shot files are not
  supported yet.
---

# Omega 1.4 Preflight — Vet a Dataset Before You Run It

Predict whether a dataset will clear ~70% accuracy on `omega_1_4_base` + the `machine-state-classification` pipeline **before** spending API budget on a full batch run. Repo: [archetypeai/omega-1-4-preflight](https://github.com/archetypeai/omega-1-4-preflight).

The preflight tool is the cheap upstream sibling of [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md). It looks at the shot files the user is *about* to upload and flags the issues that the cross-repo synthesis of [TEP](https://github.com/archetypeai/archetypeai-batch-examples-tep), [SWaT](https://github.com/archetypeai/archetypeai-batch-examples-swat), [3W](https://github.com/archetypeai/archetypeai-batch-examples-3w), [Pump Sensor](https://github.com/archetypeai/archetypeai-batch-examples-pump-sensor), [NASA Bearing](https://github.com/archetypeai/archetypeai-batch-examples-nasa-bearing), and [HIGGS](https://github.com/archetypeai/archetypeai-batch-examples-higgs) showed are predictive of a poor full run.

## When to Apply

- User has shot files (`normal.csv` + `fault.csv` or analogous) and is considering running them through Machine State batch
- User asks "will this dataset work with Omega 1.4?" / "what accuracy should I expect?"
- User's previous Machine State run underperformed and they want to diagnose why
- User wants a sanity check on shot files before they commit to upload + batch + evaluate
- User wants a quick local feel for the dataset without paying for a full batch run

**Use [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md) instead when**: the user has already decided to run the job. Preflight is a *go / no-go gate*, not a replacement for the real pipeline.

**Do not use this skill when**:
- The task is video / vision (use `newton-activity-monitor`).
- The task is text-in / text-out (use `newton-activity-detection-batch`).
- The user wants to fine-tune Omega (preflight diagnoses base-model readiness only).

## v1 Scope

- **Binary classification** (normal vs fault). Multi-class is on the roadmap.
- **Time-series with a numeric timestamp column.** No support yet for image / video / text input.
- **Static checks** run locally with stdlib + `requests` — no API key needed.
- **Pilot mode** (`--pilot`) optionally runs a tiny `machine-state-classification` job against your stage/prod credentials to get a real accuracy number on a held-out slice.

## Setup

```bash
git clone https://github.com/archetypeai/omega-1-4-preflight.git
cd omega-1-4-preflight

cp .env.example .env       # add ATAI_API_KEY and ATAI_API_ENDPOINT only if you plan to use --pilot

python3 -m venv myenv
source myenv/bin/activate
pip install requests       # static checks only need stdlib + requests
```

## Quick Start

### Static checks only (no API calls)

```bash
python preflight.py \
  --shots-normal data/normal_shots.csv \
  --shots-fault  data/fault_shots.csv
```

Output is a colored table of 10 checks + cross-file checks, ending in `PASS=N WARN=N FAIL=N INFO=N`. Exit code 0 = clean, 1 = at least one FAIL.

### With held-out pilot (real API call)

```bash
python preflight.py \
  --shots-normal data/normal_shots.csv \
  --shots-fault  data/fault_shots.csv \
  --pilot
```

Holds out 20% of each shot file as labeled pilot data, uploads the remaining 80% as shots, runs a real `machine-state-classification` job, scores predictions vs. known labels, and prints a verdict against the 70% bar and the majority-class baseline.

## The 10 Static Checks

| # | Check | Fails when | Why it matters |
|---|---|---|---|
| 1 | `schema` | timestamp column missing or non-numeric cells present | Pipeline requires a monotonic numeric timestamp |
| 2 | `timestamp` | non-monotonic or large gaps (>5× median delta) | Random-row shots destroy temporal structure (HIGGS-style failure mode) |
| 3 | `missing_values` | any column has missing values | NaN propagation kills embeddings |
| 4 | `constant_columns` | any feature column has zero variance | Constant channel contributes no signal and breaks z-scoring |
| 5 | `feature_scale` | ranges span >3 orders of magnitude | Euclidean distance is dominated by largest-range column — recommend z-score or `--metric cosine` |
| 6 | `nshot_support` | fewer than `--nshot-floor` rows/class (default 500) | KNN needs enough reference embeddings to discriminate |
| 7 | `schema_match` | shot files have different columns | Pipeline cannot align embedding spaces |
| 8 | `class_balance` | majority class >70% (warn) or >85% (strong warn) | Trivial-majority predictor will look "good" on accuracy |
| 9 | `window_vs_sampling` | info-only | Translates `window_size × median_delta` into natural language ("1.1 hours of process data") so the user can sanity-check window coverage |
| 10 | `accuracy_prior` | info-only | Notes that base-model accuracy varies widely (coin-flip to ~0.80) and recommends `--pilot` for a real number |

## Interpreting Verdicts

### Static-check totals

- **All PASS or PASS + WARN, no FAIL:** safe to run a real batch. WARNs that come up across multiple datasets (feature-scale, class imbalance) are usually accuracy levers worth fixing first.
- **Any FAIL:** static check refuses to recommend running. The most common FAIL is `nshot_support` (shot files too short) — re-extract longer contiguous runs from the raw labeled CSV. The second-most-common is `timestamp` non-monotonic, which is sometimes a *real* "this is not a time series" signal (HIGGS) and sometimes a shot-file curation artifact (rows shuffled across runs during prep — bumped from FAIL to a strong WARN over time).

### Pilot verdicts (`--pilot`)

| Verdict | Meaning |
|---|---|
| `PASS` | Pilot accuracy ≥ 70% AND beats majority baseline by >5pp |
| `WARN` | Pilot below 70% but above majority baseline — tuning may help |
| `FAIL` | Pilot below majority baseline — base model not useful; consider fine-tuning |

**Pilot caveat:** the held-out pilot is *within-distribution* — last 20% of the same shot files. Real cross-distribution generalization is harder. On TEP, the pilot scored 0.784 but the full-inference run on 15.3M rows from different simulations scored 0.506–0.537 (~25pp overestimate). Treat `PASS` as "worth running the full job," not as "the full job will clear 70%." Treat `WARN` / `FAIL` as strong evidence the full run will *not* clear 70% without tuning or fine-tuning.

## Key Flags

| Flag | Default | Meaning |
|---|---|---|
| `--shots-normal PATH` | required | CSV of contiguous normal-class rows |
| `--shots-fault PATH` | required | CSV of contiguous fault-class rows |
| `--timestamp-column` | `timestamp` | Override if your column is named differently (e.g. Volve uses `DATE_TIME`) |
| `--timestamp-unit` | `auto` | `seconds` / `minutes` / `hours` — enables natural-language window translation in check 9 |
| `--window-size N` | 64 | Rows per inference window |
| `--n-neighbors K` | 5 | KNN neighbors |
| `--metric` | `euclidean` | `euclidean` / `cosine` / `manhattan` — use `cosine` to neutralize feature-scale heterogeneity |
| `--weights` | `uniform` | `uniform` / `distance` |
| `--step-size N` | 1 | Sliding step |
| `--nshot-floor N` | 500 | Minimum usable rows/class |
| `--pilot` | off | Run held-out pilot via batch API |
| `--pilot-size N` | auto | Override auto split — rows/class for pilot |
| `--force` | off | Run pilot even below floor (underpowered) |
| `--env PATH` | `.env` | Env file path |

## Recommended Workflow

1. **Prep shot files** (one CSV per class, contiguous numeric timestamp, ≥1,000 rows each per `nshot_support` floor + headroom).
2. **Run static preflight.** Fix any FAILs. Triage WARNs:
   - `feature_scale` >3 decades → z-score each column before upload, or switch the downstream call to `--metric cosine` (handled by [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md)).
   - `constant_columns` → drop those columns before upload.
   - `class_balance` skew → consider whether accuracy is the right metric, or report macro F1.
3. **Optional: run `--pilot`** to get a real (within-distribution) accuracy number. Useful when the user wants a concrete prediction before paying for a multi-million-row run.
4. **If preflight clears**, hand off to [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md) for the real job.
5. **If preflight flags non-trivial WARNs and the full run still underperforms**, the WARN was the root cause — fix it and rerun. The cross-repo evidence (Pump Sensor +47pp accuracy after switching from default `w=64` to optimized `w=16, k=11, distance`) is that prep-side fixes often matter more than tuning the inference call.

## Cross-Reference: What Preflight Does NOT Catch

- **Distribution shift.** Preflight sees the shot files only. If your inference data comes from different simulations / shifts / equipment / environmental conditions than the shots, accuracy may collapse even with a clean preflight (TEP example above).
- **Multi-class structural difficulty.** v1 is binary. 3W's 9-class collapse is not modeled.
- **Class base-rate at inference.** SWaT shot files are balanced 50/50, but the real attack base-rate is 3.8% — preflight can't see that gap because it only ingests the shots.
- **Model fitness.** "Base model has no concept of your domain" is the HIGGS case — preflight's timestamp-monotonicity FAIL catches it for HIGGS-style random-event data, but a domain-mismatched but temporally-structured dataset could still pass preflight and underperform on the full run. The fix is fine-tuning, not prep.

## Common Pitfalls

- **`timestamp` column named differently.** Volve uses `DATE_TIME`. Pass `--timestamp-column DATE_TIME`.
- **Tiny shot files.** The Archetype AI batch-example repos ship 200-row "quick test" shots — well below the 500-row floor. Re-extract longer contiguous runs from the raw labeled CSV; preflight will then pass cleanly.
- **Treating `PASS` as a guarantee.** It isn't. It's "no obvious red flags in the shot files." The full run is still the source of truth.
- **Ignoring WARNs.** `feature_scale` and `class_balance` warns are the levers that moved accuracy across the sibling repos. Don't skip past them.

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Static checks clean OR pilot verdict `PASS` / `WARN` |
| 1 | Static checks failed OR missing credentials |
| 2 | Pilot verdict `FAIL` |

## See Also

- [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md) — the downstream batch job preflight prepares for
- [`newton-batch-upload`](../newton-batch-upload/SKILL.md) — required for shot files >255 MB
- [`omega-local`](../omega-local/SKILL.md) — offline counterpart for embeddings + custom downstream models
