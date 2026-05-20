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

**Heads-up on naming:** the same model family ships under different identifier strings depending on which API surface you call. There's no single canonical string. The table below lists the **accepted-by-each-surface** identifier — pass an identifier from a different surface and you get `400 invalid_model_version` (or its equivalent) even when the underlying weights are presumably similar. The unfortunate consequence: you cannot infer the right identifier for one surface from the right identifier for another. Always cross-check.

### Newton C language + vision models

| Family | `/query` (Direct Query) | Batch pipeline (`activity-detection`) `model_variant` |
|---|---|---|
| **C 2.5 (8 B, current default)** | `Newton::c2_5_8b_260413b723a9ab` | `newton/c:2.5.1-8b-base` (default) **and** `newton/c:2.5.0-8b-base` |
| C 2.4 (4.7 B / 7 B) | `Newton::c2_4_7b_251215a172f6d7` | `newton/c:2.4.0-7b-base` |
| C 2.3 (7 B, older) | not exposed on `/query` | `newton/c:2.3.0-7b-base` |

Source: probed `/query` directly + read `/v0.5/batch/registry/pipelines/ppl_5w8x15v9n69tprdt8h9mg5cffs/schema` (the `activity-detection` pipeline's `ModelVariant` enum). The batch surface exposes the wider catalog (four versions); `/query` is more selective (only two are accepted).

Important caveats:

- **`c2_5_8b` (short, unnamespaced) and `c:2.5.0-8b-base` (semver only) are NOT valid identifiers on `/query`** — probed live, both return `400 invalid_model_version`. They're useful as conceptual handles for "the 2.5 family" but you cannot pass them to the endpoint. Use the full namespaced hash-suffixed form `Newton::c2_5_8b_260413b723a9ab`.
- **`/query`'s 2.5 build (`c2_5_8b_260413b723a9ab`, dated 2026-04-13) and batch's default `2.5.1-8b-base` may be different patch builds.** The naming doesn't line up — `/query` uses an opaque build hash, batch uses semver. Don't assume bit-for-bit equivalent outputs across surfaces.
- **None of the Newton C identifiers are accepted as a Lens `model_version`.** Newton C isn't a Lens model — there's no Lens session that wraps it. The closest Lens equivalent for vision is the Activity Monitor Lens.

### Omega numeric encoders

| Family | `/query` (Direct Query) | Batch pipeline (`machine-state-classification`) `model_type` | Lens session `model_version` |
|---|---|---|---|
| **Omega 1.4 (current default)** | `OmegaEncoder::omega_embeddings_1_4` | `omega_1_4_base` (default) **and** `omega_1_4` | `OmegaEncoder::omega_embeddings_1_4` |
| Omega initial / "01" | `OmegaEncoder::omega_embeddings_01` | not exposed | `OmegaEncoder::omega_embeddings_01` (prod default for Machine State Lens) |
| Omega 1.3 SLB | not exposed | `omega_1_3_slb` | not exposed |
| Omega 1.3 Surface | not exposed | `omega_1_3_surface` (legacy, 9-channel) | not exposed |
| Omega 1.3 Power Drive | not exposed | `omega_1_3_power_drive` (legacy, 9-channel) | not exposed |

Source: probed `/query` directly + read `/v0.5/batch/registry/pipelines/ppl_18zrcb6m1c96ds77sgqbs7cf84/schema` (the `machine-state-classification` pipeline's `ModelType` enum) + cross-referenced [`newton-machine-state`](../newton-machine-state/SKILL.md) for the Lens-side identifiers (which the Lens layer doesn't validate at session-create time — invalid model_versions accept registration and session-creation but fail later at inference, so trust the skills that verified end-to-end usage).

Important caveats:

- **`omega_1_4_base` is NOT valid on `/query`** despite being the obvious-looking identifier. Direct Query only accepts the `OmegaEncoder::omega_embeddings_*` form. Conversely, `OmegaEncoder::omega_embeddings_1_4` is NOT valid as a batch `model_type` — the batch pipeline only accepts the bare `omega_1_X_*` form.
- **`omega_1_4_base` and `OmegaEncoder::omega_embeddings_1_4` are probably the same encoder weights but this hasn't been bit-for-bit verified.** A direct comparison of their output embeddings on the same input would settle it — until then, treat them as nominally-equivalent-but-formally-different.
- **The Lens layer doesn't validate `model_version` at session-create time.** I confirmed this by registering a lens with `OmegaEncoder::nonexistent_xyz` and successfully creating a session that reached `RUNNING` status. The validation happens later in the inference path. Don't use Lens registration as a "is this identifier valid?" probe.

### Pre-configured Lens deployments

These are Lens IDs already mounted on prod. You connect to the lens_id and the lens handles model invocation internally. Each lens pins a specific model + processor combination — to switch underlying model versions, switch lens IDs.

| Lens ID | Processor | Pinned model_version | Purpose |
|---|---|---|---|
| `lns-1286e5d1d1b84a77-af311d579cc14869` | `lens_camera_processor` | `Newton::c2_5_8b_260413b723a9ab` | **Activity Monitor — C 2.5 (current).** Video activity detection / chart understanding on the newer 8B Newton C. Default `focus = "Describe the video."`, `temporal_focus = 5`, `camera_buffer_size = 5`. |
| `lns-fd669361822b07e2-bc608aa3fdf8b4f9` | `lens_camera_processor` | `Newton::c2_4_7b_251215a172f6d7` | Activity Monitor — C 2.4 (older variant). Same processor / parameter shape; uses the smaller 4.7B model. The lens currently referenced from [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md) — worth migrating to the C 2.5 lens above for newer projects. |
| `lns-1d519091822706e2-bc108andqxf8b4os` | `lens_timeseries_state_processor` | `OmegaEncoder::omega_embeddings_01` | **Machine State Lens — n-shot KNN classifier.** Real-time per-window classification with SSE output, hosted KNN over n-shot focus CSVs. Default buffer/window/step = 1024 (much larger than the per-window probes in `/query`), `normalize_input: true`. Used by [`newton-machine-state`](../newton-machine-state/SKILL.md). |
| _(lens_id not yet probed by us)_ | `lens_anomaly_detector` | `OmegaEncoder::omega_embeddings_1_4` | **Anomaly Detector Lens.** Fit-predict isolation-forest-style detector over Omega 1.4 embeddings (768-dim vectors, 100 estimators). `mode: fit_predict`, `normalize_input: true`, up to 1 GB of accumulated embedding state per session. Different processor + different model from the Machine State Lens — pick this when you want unsupervised anomaly detection rather than n-shot KNN. List via `GET /v0.5/lens/metadata` to get the lens_id. |

A few notes:

- **The Machine State Lens uses `normalize_input: true`** (per-window normalization inside the encoder). That's the *opposite* of what [`newton-machine-state-direct-query`](../newton-machine-state-direct-query/SKILL.md) recommends for `/query` workloads. The Lens layer has historically run this way (the flag isn't exposed through the Lens config to override) — see [`newton-machine-state`](../newton-machine-state/SKILL.md) and the matching batch skill for pre-normalization workarounds.
- **There are two Activity Monitor lenses on prod** — both serve the same purpose but at different model versions. The C 2.5 lens (`lns-1286e5d1d1b84a77-…`) is the one to use for newer work; the C 2.4 lens remains mounted for backwards compatibility.
- **Lens registration does NOT validate `model_version` at session-create time.** I probed: registering a lens with `OmegaEncoder::nonexistent_xyz` succeeds and the session reaches `RUNNING` status — the failure surfaces only when data streams through and inference is attempted. Do not use Lens registration as a "is this identifier valid?" probe; trust the pinned values in the table above (which come from actual mounted lenses on prod).

## Newton C input modes (when used via `/query`)

| Input | Path | Notes |
|---|---|---|
| Text | `query` field | The default; carries the prompt or state snapshot. |
| Plain-text content (logs, notes) | `file_ids` of `.txt` filename, **or** inline in `query`, **or** `data.text` event | All three inject contents into the prompt. |
| JSON content | `file_ids` of `.json` filename (upload with mime `text/plain` to dodge an upload-side validation bug), **or** inline in `query`, **or** `data.text` / `data.json` event (`contents` must be a string, not a parsed object) | All three inject contents into the prompt. |
| CSV content | inline in `query`, `data.text` event, or rename to `.txt` before upload | `file_ids` of `.csv` uploads cleanly but contents are **not** injected (likely routed to the numeric/Omega path the LLM doesn't observe). |
| Image (PNG / JPG / JPEG) | `file_ids` with the filename, **or** `data.base64_img` event | Vision pipeline activates; latency rises to ~6–8 s. **Use the `file_id` (filename) from the upload response, not the `file_uid` (`fil_…`) — the API filters file types by extension on the file_id string.** |
| Video (MP4) | `file_ids` of `.mp4` filename | API accepts; both checkpoints currently respond "I can't see videos" in ~2 s (frames don't reach the model). Use the Activity Monitor Lens instead — see [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md). |

For details on the Direct Query call shape, see [`newton-query-prompting`](../newton-query-prompting/SKILL.md). For batch Newton C usage (JSONL prompts at scale), see [`newton-activity-detection-batch`](../newton-activity-detection-batch/SKILL.md). Reference apps that exercise these patterns: [`archetypeai-swat-demo-direct-query`](https://github.com/archetypeai/archetypeai-swat-demo-direct-query), [`archetypeai-earthquake-demo`](https://github.com/archetypeai/archetypeai-earthquake-demo), [`archetypeai-grid-demo`](https://github.com/archetypeai/archetypeai-grid-demo).

## Omega input shape (when used via `/query`)

Send `query: ""` plus a `data.numeric_array` event carrying the window in `event_data.contents` (channel-first 2D: outer = channels, inner = window samples). For most anomaly-detection workloads, pre-normalize with a global scaler and pass `normalize_input: false` — the per-window normalization on `true` erases cross-window amplitude signal.

**The same input sent to `omega_embeddings_01` and `omega_embeddings_1_4` produces materially different embedding vectors** (same `[N × 768]` shape, different values). They're different checkpoints, not the same model under different names. Re-build any cached KNN library when switching identifiers. Empirical data from the SWaT 6-stage benchmark: swapping `_01` → `_1_4` lifts library leave-one-out KNN accuracy on the two non-saturated stages (P1 93→98 %, P3 93→97 %) with no regressions; per-call latency is within 5 % on isolated calls but ~2 × slower under sustained back-to-back load.

For full Omega `/query` + local KNN guidance, see [`newton-machine-state-direct-query`](../newton-machine-state-direct-query/SKILL.md).

## Recommended defaults by use case

| Use case | Model / Lens | Surface | Skill |
|---|---|---|---|
| Operator suggestions / structured JSON output from text state | `Newton::c2_5_8b_260413b723a9ab` | `/query` | [`newton-query-prompting`](../newton-query-prompting/SKILL.md) |
| Image description (single screenshot or chart) | `Newton::c2_5_8b_260413b723a9ab` | `/query` with `file_ids` or `data.base64_img` | [`newton-query-prompting`](../newton-query-prompting/SKILL.md) |
| Stateless per-window classification (KNN + Omega) | `OmegaEncoder::omega_embeddings_1_4` | `/query` | [`newton-machine-state-direct-query`](../newton-machine-state-direct-query/SKILL.md) |
| Streaming per-window classification with a hosted KNN | Machine State Lens (`lns-1d519091822706e2-…`) — pins `OmegaEncoder::omega_embeddings_01` | Lens session | [`newton-machine-state`](../newton-machine-state/SKILL.md) |
| Streaming unsupervised anomaly detection | Anomaly Detector Lens — pins `OmegaEncoder::omega_embeddings_1_4` + isolation-forest detector | Lens session | (no dedicated skill yet; new) |
| Batch classification of millions of rows | `omega_1_4_base` (`model_type`) | `machine-state-classification` batch pipeline | [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md) |
| Batch JSONL prompting (many prompts, async) | `newton/c:2.5.1-8b-base` (`model_variant`) | `activity-detection` batch pipeline | [`newton-activity-detection-batch`](../newton-activity-detection-batch/SKILL.md) |
| Video activity detection / Q&A (C 2.5, recommended) | Activity Monitor C 2.5 Lens (`lns-1286e5d1d1b84a77-…`) — pins `Newton::c2_5_8b_260413b723a9ab` | Lens session | [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md) (referenced lens_id is the C 2.4 variant; migrate to the C 2.5 one for new work) |
| Video activity detection / Q&A (legacy C 2.4) | Activity Monitor Lens (`lns-fd669361822b07e2-…`) — pins `Newton::c2_4_7b_251215a172f6d7` | Lens session | [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md) |

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

[`archetypeai-swat-demo-direct-query`](https://github.com/archetypeai/archetypeai-swat-demo-direct-query) ships two reusable comparison scripts:

- `scripts/compare_omega_models.py` — per-phase latency comparison (queue / load / inference / wall) between any two Omega `/query` identifiers, with random or CSV input.
- `scripts/compare-newton-models.js` — runs the SWaT operator-suggestions prompt against any two Newton C identifiers, validates JSON + topology, reports valid-card counts alongside latency.

Useful when picking between checkpoints for a new project or assessing the cost of upgrading.

## Maintenance

This skill goes stale. To refresh:

1. Run the verification probes above for each listed identifier — any that now return `400 invalid_model_version` is gone from your account.
2. Try the latest checkpoint suffix the platform team mentions (e.g. `c2_6_…` if it appears in release notes). If it `200`s on `/query`, add it to the *Newton C* table.
3. Pull the live batch enums via `GET /v0.5/batch/registry/pipelines/{ppl_id}/schema` for `machine-state-classification` (Omega `model_type`) and `activity-detection` (Newton C `model_variant`) — diff against the catalog tables and add anything new.
4. Check `GET /v0.5/lens/metadata` for new Lens deployments. Each entry has the lens_id and the pinned `model_version` — surface any new ones in the *Pre-configured Lens deployments* table.
5. When a `/v0.5/models` endpoint ships, this skill should mostly become a thin pointer at it and the dynamic registry takes over.
6. Cross-reference with the existing per-capability skills — if a skill's recommended-default identifier disagrees with this catalog, one of them is out of date.

### Checklist: a new C-model or Omega checkpoint is announced

When the platform team announces a new model build, this is the propagation pass across the skills registry. Roughly 30 minutes once the new identifier strings are known.

**New Newton C version (e.g. `c2_6_…`):**

| Skill | What to update | When |
|---|---|---|
| `newton-models` | Add to Newton C surface tables + Recommended-defaults; refresh the *Last verified* date | Always — this is the registry |
| `newton-query-prompting` | Model-families overview paragraph + example identifiers | If the new model is the new default for structured-output reasoning |
| `newton-activity-detection-batch` | The `newton/c:X.Y.Z-Nb-base` list in the *Tunable knobs* section | When the new variant appears in the batch pipeline schema |
| `newton-activity-monitor` | Lens IDs table + the Step 4 example's `lens_id` | When a new Activity Monitor lens is mounted that pins the new model |

**New Omega checkpoint (e.g. `omega_1_5_*`):**

| Skill | What to update | When |
|---|---|---|
| `newton-models` | Omega surface tables + Recommended-defaults | Always |
| `newton-machine-state` | Default `model_version`; any normalization caveats | When the new identifier shows up in `/lens/metadata` and is the new prod default |
| `newton-machine-state-batch` | `model_type` enum in *Available model types* section | When the new variant appears in the batch pipeline schema |
| `newton-machine-state-direct-query` | Code examples + the recommended-default callout | If a LOO comparison on a real dataset shows the new model meets or beats the previous |

**Pre-promotion verification (either type):**

1. Run the relevant probe script from `archetypeai-swat-demo-direct-query/scripts/` against the candidate identifier — `compare_omega_models.py` or `compare-newton-models.js`. The point is to confirm the new identifier returns 200 on each surface the documentation claims it's on.
2. For Omega upgrades, rebuild the SWaT KNN library with the new identifier and check that LOO accuracy doesn't regress.
3. For Newton C upgrades, run the SWaT operator-suggestion prompt and check JSON-card validity + topology compliance.

**External repos that also pin models (worth a refresh, not blocking):**

- [`archetypeai-swat-demo-direct-query`](https://github.com/archetypeai/archetypeai-swat-demo-direct-query) — pins both an Omega and a Newton C model.
- [`archetypeai-earthquake-demo`](https://github.com/archetypeai/archetypeai-earthquake-demo) — pins a Newton C model.
- [`archetypeai-grid-demo`](https://github.com/archetypeai/archetypeai-grid-demo) — pins a Newton C model.

Last verified on `api.u1.archetypeai.app` on 2026-05-18.
