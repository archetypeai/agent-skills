---
name: newton-models
description: >
  Catalog of model identifiers currently exposed in production on Newton
  (`api.u1.archetypeai.app`), grouped by family, with the API surfaces
  each one is reachable from (`/query`, batch pipelines, Lens sessions),
  the input modalities each accepts, the existing skills that use each,
  and recommended defaults per use case. Use this skill when the user
  asks "which model should I use?", "what models are available?", "is
  this model identifier valid?", or when picking between Newton C
  checkpoints / Omega encoder versions for a new project. Prod-scoped
  by design — staging variants and source-tree identifiers that aren't
  deployed are intentionally excluded.
  Do NOT use as a substitute for the per-capability skills (newton-machine-state-*,
  newton-query-prompting, newton-activity-monitor, etc.) — this skill is the
  registry; the others are the usage guides.
---

# Available Models (Prod)

What's currently exposed on `https://api.u1.archetypeai.app/` as of 2026-05-18. The skill is a registry, not a tutorial — for actually using these models, follow the cross-linked usage skills.

> **Heads up.** This document is hand-maintained from live probing. There is no `/v0.5/models` endpoint to read from yet (verified: `GET /v0.5/models` → `404`). When a new model ships, this skill needs updating; cross-check by running the probe in *Verification* before relying on entries here.

## At a glance

| Identifier | Family | API surfaces | Verified on prod |
|---|---|---|---|
| `Newton::c2_4_7b_251215a172f6d7` | Newton C (text reasoning, image vision) | `/query`, `activity-detection` batch pipeline | ✅ |
| `Newton::c2_5_8b_260413b723a9ab` | Newton C (text reasoning, image vision) | `/query`, `activity-detection` batch pipeline | ✅ |
| `OmegaEncoder::omega_embeddings_01` | Omega encoder (sensor → embedding) | `/query`, Machine State Lens | ✅ |
| `OmegaEncoder::omega_embeddings_1_4` | Omega encoder (sensor → embedding) | `/query`, Machine State Lens | ✅ |
| `omega_1_4_base` | Omega encoder (sensor → classification) | `machine-state-classification` batch pipeline only | ✅ |
| `omega_1_3_surface` | Omega encoder, legacy 9-channel | `machine-state-classification` batch pipeline only | ✅ (legacy) |
| `omega_1_3_power_drive` | Omega encoder, legacy 9-channel | `machine-state-classification` batch pipeline only | ✅ (legacy) |
| Activity Monitor Lens<br/>`lns-fd669361822b07e2-bc608aa3fdf8b4f9` | Newton vision (video + chart understanding) | Lens session only | ✅ |
| Machine State Lens<br/>`lns-1d519091822706e2-bc108andqxf8b4os` | Lens-wrapped Omega + KNN | Lens session only | ✅ |

## Newton C language models

Text-reasoning + image-vision multimodal LLMs.

| Identifier | Params | Build date | Use when |
|---|---|---|---|
| `Newton::c2_4_7b_251215a172f6d7` | 4.7 B | 2025-12-15 | Latency-sensitive paths that can tolerate occasional structural mistakes. ~3–6 s typical for text-only queries; cheaper to run. |
| **`Newton::c2_5_8b_260413b723a9ab`** | **8 B** | **2026-04-13** | **Default for structured-output reasoning.** Materially better JSON shape compliance and richer citations on the SWaT operator-suggestions prompt (9-of-9 valid topology-checked cards every run vs c2_4_7b's 3-of-9 average). ~13 s per call. |

Both checkpoints accept the same input modes via `/query`:

| Input | Path | Notes |
|---|---|---|
| Text | `query` field | The default; carries the prompt or state snapshot. |
| Plain-text content (logs, notes) | `file_ids` of `.txt` filename, **or** inline in `query`, **or** `data.text` event | All three inject contents into the prompt. |
| JSON content | `file_ids` of `.json` filename (upload with mime `text/plain` to dodge an upload-side validation bug), **or** inline in `query`, **or** `data.text` / `data.json` event (`contents` must be a string, not a parsed object) | All three inject contents into the prompt. |
| CSV content | inline in `query`, `data.text` event, or rename to `.txt` before upload | `file_ids` of `.csv` uploads cleanly but contents are **not** injected (likely routed to the numeric/Omega path the LLM doesn't observe). |
| Image (PNG / JPG / JPEG) | `file_ids` with the filename, **or** `data.base64_img` event | Vision pipeline activates; latency rises to ~6–8 s. **Use the `file_id` (filename) from the upload response, not the `file_uid` (`fil_…`) — the API filters file types by extension on the file_id string.** |
| Video (MP4) | `file_ids` of `.mp4` filename | API accepts; both checkpoints currently respond "I can't see videos" in ~2 s (frames don't reach the model). Use the Activity Monitor Lens instead — see [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md). |

Used by: [`newton-query-prompting`](../newton-query-prompting/SKILL.md), [`newton-activity-detection-batch`](../newton-activity-detection-batch/SKILL.md), and the [`newton-swat-demo-direct-query`](https://github.com/archetypeai/newton-swat-demo-direct-query) / [`newton-earthquake-demo`](https://github.com/archetypeai/newton-earthquake-demo) / [`newton-grid-demo`](https://github.com/archetypeai/newton-grid-demo) reference apps.

## Omega encoders

Numeric-only encoders. Input is channel-first time-series; output is per-channel 768-dim embedding vectors. Different model families across the three API surfaces — same encoder family in spirit, but the identifiers are **not interchangeable** and the underlying checkpoints differ.

### Via `/query` (Direct Query)

| Identifier | Notes |
|---|---|
| `OmegaEncoder::omega_embeddings_01` | Initial release. Wide adoption in newer demos before `1_4` became preferred. |
| **`OmegaEncoder::omega_embeddings_1_4`** | **Default for new Direct Query projects.** Replaces `_01` for most workloads. On the SWaT 6-stage KNN benchmark, swapping `_01` → `_1_4` lifts library leave-one-out accuracy on the two non-saturated stages (P1 93→98 %, P3 93→97 %) with no regressions. Note: ~2 × slower than `_01` under sustained back-to-back call load (~13 s vs ~6 s sustained on a 6-stage fan-out); isolated single-call latency is within 5 %. |

**Important:** the same input window sent to `omega_embeddings_01` and `omega_embeddings_1_4` produces **different embedding vectors** — same `[N × 768]` shape, materially different values. They're different checkpoints, not the same model under different names. Re-build any KNN library you've stored when switching identifiers.

Send these with `query: ""` and a `data.numeric_array` event carrying the window in `event_data.contents` (channel-first 2D: outer = channels, inner = window samples). Pre-normalize with a global scaler and pass `normalize_input: false` for most anomaly-detection workloads — the per-window normalization on `true` erases cross-window amplitude signal. Used by [`newton-machine-state-direct-query`](../newton-machine-state-direct-query/SKILL.md).

### Via the `machine-state-classification` batch pipeline

| Identifier | Channels | When to use |
|---|---|---|
| **`omega_1_4_base`** | Arbitrary | **Default.** Generic, no channel-count constraint. Suitable for almost every dataset. |
| `omega_1_3_surface` | Exactly 9 | Legacy. Use only if you're working from an existing `omega_1_3_surface` n-shot library. |
| `omega_1_3_power_drive` | Exactly 9 | Legacy. Same caveat. |

These names are **only valid as the `model_type` field of the batch-job parameters block**. They will fail with `400 invalid_model_version` if you try to pass them to `/query`. Used by [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md).

### Via Machine State Lens sessions

The streaming Lens accepts both `OmegaEncoder::omega_embeddings_01` and `OmegaEncoder::omega_embeddings_1_4` as the `model_version` in `model_parameters`. `omega_embeddings_01` is the prod default; `omega_embeddings_1_4` is documented as a staging fallback that also works on prod. Used by [`newton-machine-state`](../newton-machine-state/SKILL.md).

### Why three identifier conventions?

| Surface | Convention | Example |
|---|---|---|
| `/query` (Direct Query) | `OmegaEncoder::<name>` | `OmegaEncoder::omega_embeddings_1_4` |
| Batch pipeline | bare `omega_<version>_<variant>` | `omega_1_4_base` |
| Lens session | `OmegaEncoder::<name>` (same as Direct Query) | `OmegaEncoder::omega_embeddings_1_4` |

`omega_1_4_base` and `OmegaEncoder::omega_embeddings_1_4` are **probably** the same encoder weights exposed under different identifiers (the version numbers align), but this has not been verified by comparing their outputs on the same input. Don't assume bit-for-bit equivalence across surfaces.

## Vision lenses

For video + image understanding via Lens sessions (separate from Newton C's `/query` image vision). Used by [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md).

| Lens ID | Family | When to use |
|---|---|---|
| `lns-fd669361822b07e2-bc608aa3fdf8b4f9` | Newton vision (~2 B params) | Video activity detection, chart/dashboard understanding, Q&A over a video stream, any time you need video frames analyzed (Newton C `/query` doesn't process video). |

## Recommended defaults by use case

| Use case | Model | Surface | Skill |
|---|---|---|---|
| Operator suggestions / structured JSON output from text state | `Newton::c2_5_8b_260413b723a9ab` | `/query` | [`newton-query-prompting`](../newton-query-prompting/SKILL.md) |
| Stateless per-window classification (KNN + Omega) | `OmegaEncoder::omega_embeddings_1_4` | `/query` | [`newton-machine-state-direct-query`](../newton-machine-state-direct-query/SKILL.md) |
| Streaming per-window classification with a hosted KNN | `OmegaEncoder::omega_embeddings_01` (prod default) | Machine State Lens | [`newton-machine-state`](../newton-machine-state/SKILL.md) |
| Batch classification of millions of rows | `omega_1_4_base` | `machine-state-classification` batch pipeline | [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md) |
| Batch JSONL prompting (many prompts, async) | `Newton::c2_5_8b_260413b723a9ab` | `activity-detection` batch pipeline | [`newton-activity-detection-batch`](../newton-activity-detection-batch/SKILL.md) |
| Image description (single screenshot or chart) | `Newton::c2_5_8b_260413b723a9ab` | `/query` with `file_ids` or `data.base64_img` | [`newton-query-prompting`](../newton-query-prompting/SKILL.md) |
| Video activity detection / Q&A | Newton vision Lens | Lens session | [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md) |

## Verification

Re-confirm any identifier on your account with one of these probes. If you get `400 invalid_model_version`, the identifier is not exposed to you (rather than missing globally — model availability is per-account).

### `/query` probe (Newton C)

```bash
curl -s -X POST "$ATAI_API_ENDPOINT/v0.5/query" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"Reply with OK.","model":"Newton::c2_5_8b_260413b723a9ab","max_new_tokens":10,"sanitize":false}'
```

### `/query` probe (Omega encoder)

```bash
curl -s -X POST "$ATAI_API_ENDPOINT/v0.5/query" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query":"",
    "model":"OmegaEncoder::omega_embeddings_1_4",
    "normalize_input":false,
    "events":[{
      "type":"data.numeric_array",
      "event_data":{"contents":[[1,2,3,4,5,6,7,8]]}
    }]
  }'
```

`200` with a numeric array under `response.response[0]` → identifier is valid. `400 invalid_model_version` → not exposed.

### Batch pipeline probe (Omega via batch)

Submit a no-op job with a tiny inference CSV; the response surfaces the validated `model_type` field. Easier: cross-check against the [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md) skill's *Available model types* table, which is kept current.

### Compare two identifiers head-to-head

[`newton-swat-demo-direct-query`](https://github.com/archetypeai/newton-swat-demo-direct-query) ships two reusable comparison scripts:

- `scripts/compare_omega_models.py` — per-phase latency comparison (queue / load / inference / wall) between any two Omega `/query` identifiers, with random or CSV input.
- `scripts/compare-newton-models.js` — runs the SWaT operator-suggestions prompt against any two Newton C identifiers, validates JSON + topology, reports valid-card counts alongside latency.

Useful when picking between checkpoints for a new project or assessing the cost of upgrading.

## Maintenance

This skill goes stale. To refresh:

1. Run the verification probes above for each listed identifier — any that now return `400 invalid_model_version` is gone from your account.
2. Try the latest checkpoint suffix the platform team mentions (e.g. `c2_6_…` if it appears in release notes). If it `200`s on `/query`, add it to the *Newton C* table.
3. Check the `omega_embeddings_*` series for new versions (after `1_4` the next would be `1_5` or similar). Same probe pattern.
4. When a `/v0.5/models` endpoint ships, this skill should mostly become a thin pointer at it and the dynamic registry takes over.
5. Cross-reference with the existing per-capability skills — if a skill's recommended-default identifier disagrees with this catalog, one of them is out of date.

Last verified on `api.u1.archetypeai.app` on 2026-05-18.
