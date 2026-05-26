---
name: newton-machine-state-direct-query
description: >
  Run Machine State KNN classification statelessly via Newton's /query
  endpoint with the Omega encoder, plus a local KNN classifier. Use when
  the user wants per-window anomaly / state classification but doesn't
  want a Lens session (no SSE, no setup/teardown, no warmup) or wants
  direct access to the embeddings for visualization, custom distance
  metrics, or other downstream uses. Powered by `/query` + an Omega
  model (`OmegaEncoder::omega_embeddings_1_4`) called with
  `data.numeric_array` events.
  Do NOT use for streaming Machine State with hosted KNN (use newton-machine-state).
  Do NOT use for batch classification of large CSVs (use newton-machine-state-batch).
  Do NOT use for prompt-engineering Newton's text-reasoning model (use newton-query-prompting).
  Do NOT use for activity detection over video (use newton-activity-monitor).
---

# Newton Machine State Classification (Direct Query)

Classify time-series sensor data per window by calling `/query` directly with the Omega encoder, then running KNN locally against a pre-built n-shot embedding library. The streaming counterpart is [`newton-machine-state`](../newton-machine-state/SKILL.md); the batch counterpart is [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md). Same end goal — KNN over Omega embeddings of n-shot examples — different transport: no Lens, no session lifecycle, no SSE, no warmup. Each playback window is a single round-trip to `/query`.

Reference implementation: [`archetypeai-swat-demo-direct-query`](https://github.com/archetypeai/archetypeai-swat-demo-direct-query) — six per-stage classifiers on the SWaT water-treatment dataset, with PCA-2 + UMAP-2 visualization of the live embeddings.

## When to Apply

- User wants Machine State classification without the Lens — no session warmup, no SSE plumbing, no `cleanStaleLenses` / `pagehide` cleanup, no orphaned-session risk.
- User wants to see / use the raw embeddings (for a 2D scatter, for a custom metric, for distance-based abstain, for outlier detection beyond KNN).
- User wants to run the classifier in a stateless serverless function or per-request handler.
- User wants to use a distance metric or KNN variant that the hosted classifier doesn't expose.
- User wants every prediction to be independent of the previous one (Lens streaming carries internal buffer state).

**When NOT to apply:**

- User needs long-running, low-overhead streaming over a single high-rate sensor source. The Lens path amortizes the per-window overhead better. Use [`newton-machine-state`](../newton-machine-state/SKILL.md).
- User has a million-row CSV to classify in one shot. Direct-Query is one /query call per window — for a million rows, the batch pipeline is far cheaper. Use [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md).

## Endpoint

```
POST {ATAI_API_ENDPOINT}/v0.5/query
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

Same endpoint that the text-reasoning skill uses; what changes is the `model` parameter and the `events` payload. Per-call latency is **~800 ms p50 warm**, ~1.5–2 s for a typical 6-stage fan-out done in parallel.

## Architecture

Two phases. Build is one-time, runs offline; classify runs once per playback window.

```
Build phase (one-time, offline)
   shot CSVs ─┐
              ├─▶ slide windows ─▶ /query (Omega)  ─▶ flatten embedding ─▶ KNN library
              │   (per stage)        normalize_input=false   [num_cols × 768]    {NORMAL, ATTACK}
   global scaler stats ─┘

Classify (per window, online)
   inference window ─▶ apply scaler ─▶ /query (Omega) ─▶ flatten ─▶ KNN vs library ─▶ label
```

The library *is* the model — there's no model fit beyond storing the labeled embeddings. Every runtime prediction is a distance lookup.

## Request Body (Omega)

```json
{
  "query": "",
  "model": "OmegaEncoder::omega_embeddings_1_4",
  "normalize_input": false,
  "events": [
    {
      "type": "data.numeric_array",
      "event_data": {
        "contents": [
          [/* channel 0 values: window_size floats */],
          [/* channel 1 values */],
          [/* ... */]
        ]
      }
    }
  ]
}
```

Notes on each parameter:

- **`model`** — `OmegaEncoder::omega_embeddings_1_4` is the current recommended default. `OmegaEncoder::omega_embeddings_01` is also exposed on `/query` and was the previous default; it produces materially different embedding vectors than `_1_4` for the same input (same `[N × 768]` shape, different values), so re-build any cached KNN library when switching. Empirical comparison on the SWaT 6-stage benchmark: swapping `_01` → `_1_4` lifts library leave-one-out KNN accuracy on the two non-saturated stages with no regressions. For the full prod model catalog across `/query`, Lens, and batch, see [`newton-models`](../newton-models/SKILL.md).
- **`query: ""`** — there's no prompt; you're calling the encoder, not the reasoning model.
- **`normalize_input: false`** — see *The normalize_input pitfall* below. **You almost always want this set to false** combined with global pre-normalization at the call site.
- **`events`** — `data.numeric_array` is the multivariate-window shape. A single event with `contents` set to a `[num_channels × window_size]` 2D array returns a `[num_channels × 768]` embedding matrix. Sending N separate single-channel events behaves the same but is rate-limit-wasteful — always batch into one event.

## Response Shape

```json
{
  "query_id": "...",
  "status": "completed",
  "inference_time_sec": 0.35,
  "query_response_time_sec": 0.41,
  "response": {
    "response": [
      [/* 768-dim embedding for channel 0 */],
      [/* 768-dim embedding for channel 1 */],
      [/* ... */]
    ]
  }
}
```

The nested `response.response` is the canonical extraction path. For a window with N channels, you get an `[N × 768]` matrix. For local KNN, flatten to a single `N * 768`-dim vector per window.

## The `normalize_input` pitfall

The single most consequential parameter in this skill, and the one that's easy to get wrong:

`normalize_input: true` z-scores every window **in isolation** before the encoder sees it. That's safe across channels with different units (flow vs level vs valve state) within a single window — but it erases **cross-window amplitude signal**. Two windows where `LIT401` reads `574` vs `950` look numerically identical to the encoder afterwards. If your discriminating signal is "this sensor is unusually high right now," per-window normalization deletes it before the encoder can see it.

**Symptom**: leave-one-out KNN accuracy on the library hovers between 30% and 65% per class. The 2D PCA/UMAP scatter shows fully intermingled NORMAL and ATTACK points. Adding more n-shot samples doesn't help. Adjusting `n_neighbors` or distance metric doesn't help.

**Fix**: fit a global `StandardScaler` once on the n-shot training pool, apply `(x − mean) / std` per channel to every window before the call, send to `/query` with `normalize_input: false`.

On the SWaT dataset (40 sensor channels across 6 process stages, 188 library windows per stage) this single change moved LOO accuracy from 47–89% per stage to **57–100%**:

| Stage | per-window normalize | global pre-normalize |
|---|---|---|
| P1 | 65% | **93%** |
| P2 | 89% | **100%** |
| P3 | 60% | **93%** |
| P4 | 62% | **100%** |
| P5 | 66% | **100%** |
| P6 | 47% | **57%** (only 2 sensors, mostly idle — thin signal) |

The matching scope discussion lives in the Lens skills under [Input normalization](../newton-machine-state-batch/SKILL.md#input-normalization) and in [omega-local/SKILL.md](../omega-local/SKILL.md#normalization-choices) (Option B). The cloud Lens deployment runs with `normalize_input=False` always and doesn't expose the flag — you have to pre-normalize. The Direct-Query path *does* expose the flag and defaults to `true`, which is the wrong default for most real anomaly-detection workloads. **Set it to `false` and pre-normalize.**

## Workflow

### Step 1: Compute the global scaler

```js
// scripts/build-scaler.js — runs once
import { readFileSync, writeFileSync } from 'fs';

function readCsv(path) {
  const lines = readFileSync(path, 'utf-8').split(/\r?\n/).filter((l) => l.trim());
  const headers = lines[0].split(',').map((h) => h.trim());
  const rows = lines.slice(1).map((line) => line.split(','));
  return { headers, rows };
}

const normal = readCsv('data/shots_normal.csv');
const attack = readCsv('data/shots_attack.csv');
const columns = normal.headers.filter((h) => h !== 'timestamp');

const stats = Object.fromEntries(columns.map((c) => [c, { sum: 0, sumsq: 0, n: 0 }]));
for (const { headers, rows } of [normal, attack]) {
  const colIdx = columns.map((c) => headers.indexOf(c));
  for (const row of rows) {
    for (let i = 0; i < columns.length; i++) {
      const v = parseFloat(row[colIdx[i]]);
      if (Number.isNaN(v)) continue;
      const s = stats[columns[i]];
      s.sum += v;
      s.sumsq += v * v;
      s.n += 1;
    }
  }
}

const mean = {}, std = {};
for (const col of columns) {
  const { sum, sumsq, n } = stats[col];
  const m = sum / n;
  const variance = Math.max(0, sumsq / n - m * m);
  mean[col] = m;
  std[col] = Math.sqrt(variance) > 1e-9 ? Math.sqrt(variance) : 1;
}

writeFileSync('data/scaler.json', JSON.stringify({ columns, mean, std }, null, 2));
```

This file is small (a few KB for 40 channels) — commit it. The build script for the library and the runtime classifier both load it.

### Step 2: Build the n-shot KNN library

**Each shot CSV must be a temporally-contiguous block of a single class.** Sliding `window_size`-row windows over a CSV that was built with `random.sample()` (rows scattered across the timeline) gives you embeddings of *fake* signals — adjacent rows in the file are not adjacent in time, so each window concatenates physically distant moments into one input. The Direct-Query path is even more sensitive to this than the batch pipeline because you can produce useless embeddings *silently*; the API has no way to know your rows aren't contiguous. Pick the longest contiguous run per class from your raw labeled CSV before windowing — same recipe documented in [`newton-machine-state-batch/SKILL.md`](../newton-machine-state-batch/SKILL.md#recommended-n-shot-data-prep-contiguous--z-scored).

For each stage / class, slide windows over the shot CSV, apply the scaler, send to `/query`, store the flattened embedding with its label.

```js
function applyScaler(channelFirstWindow, columns, scaler) {
  return channelFirstWindow.map((channel, c) => {
    const col = columns[c];
    const m = scaler.mean[col] ?? 0;
    const s = scaler.std[col] ?? 1;
    return channel.map((v) => (v - m) / s);
  });
}

async function embedWindow(endpoint, apiKey, channelFirstWindow) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: '',
      model: 'OmegaEncoder::omega_embeddings_1_4',
      normalize_input: false,
      events: [{ type: 'data.numeric_array', event_data: { contents: channelFirstWindow } }]
    })
  });
  if (!res.ok) throw new Error(`/query failed: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.response.response; // [num_channels × 768]
}

function flatten2D(matrix) {
  const out = [];
  for (const row of matrix) for (const v of row) out.push(v);
  return out;
}
```

Step size matters: `step_size = window_size` (non-overlapping) is the cheapest and gives ~15 windows per class from a 2,000-row shot file. Tightening to `step_size ≈ window_size / 6` (overlapping) yields ~94 windows per class and noticeably better LOO. Bigger libraries also help PCA/UMAP visualizations look less noisy.

Cost: for 6 stages × 2 classes × 94 windows = **1,128 `/query` calls** per build. At ~1 s each, ~20 minutes total. Run it once; cache `data/knn-library.json`.

### Step 3: Classify per window at runtime

```js
function euclideanSq(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

function classify(libraryEmbeddings, queryEmbedding, k = 3) {
  const dists = libraryEmbeddings.map((e) => ({
    d: euclideanSq(e.vec, queryEmbedding),
    label: e.label
  }));
  dists.sort((a, b) => a.d - b.d);
  const votes = {};
  for (const t of dists.slice(0, k)) votes[t.label] = (votes[t.label] || 0) + 1;
  return Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
}

// Per playback window:
const scaled = applyScaler(window, stageColumns, scaler);
const matrix = await embedWindow(endpoint, apiKey, scaled);
const queryEmb = flatten2D(matrix);
const label = classify(library, queryEmb, 3);
```

End-to-end latency for a typical 6-stage parallel classification: 1.5–2 s.

### Step 4: Validate with leave-one-out KNN

Don't trust the 2D visualization to tell you whether the classifier works — PCA/UMAP throw away 95%+ of the variance for high-dim embeddings. The right diagnostic is leave-one-out KNN accuracy on the library itself, in the *full* embedding space:

```js
function looAccuracy(libraryEmbeddings, k = 3) {
  let correct = 0;
  const n = libraryEmbeddings.length;
  for (let i = 0; i < n; i++) {
    const target = libraryEmbeddings[i];
    const rest = libraryEmbeddings.filter((_, j) => j !== i);
    const pred = classify(rest, target.vec, k);
    if (pred === target.label) correct++;
  }
  return correct / n;
}
```

Useful thresholds:

| LOO | What it means |
|---|---|
| **≥80%** | The classifier works for this stage. 2D visualizations may still look messy — that's a visualization-fidelity issue, not a classifier issue. |
| **65–80%** | Borderline. Library may be too small, or the classes genuinely overlap for these sensors. |
| **<65%** | Classifier is unreliable for this stage. The library can't separate its own classes — adding more shots from different operating regimes is the most common fix. |
| **~50% (random)** | Something is wrong. First suspect: `normalize_input: true`. Second suspect: shot files contaminated (e.g., "attack" shots that include non-attack rows). |

## Trade-offs vs Lens (streaming and batch)

| | Lens streaming | Lens batch | Direct Query (this skill) |
|---|---|---|---|
| Setup overhead | session create + n-shot upload (~30–60s) | n-shot upload + job submit (~10s) | none beyond scaler + library build, both one-time |
| Per-window latency | <100 ms steady-state (SSE) | N/A (asynchronous) | ~800 ms (round-trip /query) |
| Embeddings exposed | no | no | **yes** — returned in /query response |
| Custom distance / classifier | no | no | **yes** — KNN runs in your code |
| Failure mode if hosted classifier breaks | downtime | job stuck | unaffected (only encoder is hosted) |
| Cost per million rows | low | lowest | high (one /query per window) |
| Right for | live dashboards on a single source | large historical CSVs | per-request handlers, viz panels, custom classifiers |

The archetypeai-swat-demo-direct-query reference cut a ~250-line Lens orchestration layer (session lifecycle, stale-session cleanup, SSE parsing, `pagehide` handlers, `localStorage` orphan tracking) down to a single synchronous POST handler. If your app doesn't actually benefit from the streaming optimization, the simplification is real.

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `unexpected Omega response shape` | Forgot the inner `response.response` extraction; some code paths return `response` as a string or top-level array | Defensively check `data.response?.response` first, fall back to `data.response`, then `data.text` |
| LOO accuracy ~50% (no signal) | `normalize_input: true` while expecting cross-window amplitude to matter | Set `normalize_input: false`, pre-normalize with a global scaler — see *The normalize_input pitfall* |
| LOO accuracy 60–70% no matter what | Library too small (e.g. `step=window` gave only 15 windows per class) | Tighten `step_size` to ~`window_size / 5` for ~5x more windows; or broaden shot coverage to more operating modes |
| 2D scatter shows intermingled clusters but LOO ≥90% | This is normal — PCA-2 captures ~5–15% of variance for 3K–9K-dim embeddings. The classifier works in the full space; the 2D picture is a lossy summary | Show the LOO badge in the UI; don't over-interpret the 2D picture |
| 400 `missing_parameter: contents` | Used `data.json` with `sensor_data` (Lens-shaped payload) instead of `data.numeric_array` with `contents` | Switch the event type to `data.numeric_array`; the Lens session-payload shape doesn't work on /query |

## Best Practices

- **Always pre-normalize and pass `normalize_input: false`.** The default value of the flag is wrong for most real anomaly-detection workloads.
- **Validate with LOO before trusting the live classifier.** A 2D scatter that looks good doesn't mean classification works (PCA throws away most of the signal). LOO is the honest diagnostic.
- **One event per call, multivariate `contents`.** Don't loop per channel and send N separate events — same throughput, more rate-limit pressure, more latency.
- **Use `step_size < window_size` for the library build.** Overlapping windows give a ~6x larger library at the same shot-file cost. Within-window overlap doesn't matter for KNN distances.
- **Commit `scaler.json`; don't commit `knn-library.json`.** The scaler is small (a few KB). The library is large (~100 MB for the SWaT-scale reference; will exceed GitHub's 100 MB file limit). Regenerate with a script and document the command.
- **Cache the library file in server memory at boot.** It's read-only and only needs to be loaded once. Reload on next deploy.
- **Pre-warm at server boot, not at first request.** Lazy-loading a 100 MB library on the first /api/classify call adds a ~1 s spike to whoever hits the endpoint first.
- **For viz panels, show the live embedding's PCA / UMAP coords as a cursor over the static library scatter.** PCA can be applied to new embeddings linearly; for UMAP, fit `umap-js` at boot and call `.transform()` per query (~20–50 ms). t-SNE doesn't have a `transform()` and won't work for live cursors.

## Example Code

[**archetypeai/archetypeai-swat-demo-direct-query**](https://github.com/archetypeai/archetypeai-swat-demo-direct-query) — full end-to-end Svelte/SvelteKit reference: six per-stage classifiers on the SWaT water-treatment dataset, with the global-scaler + Direct-Query pattern, server-side PCA-2 / UMAP-2 fits, and a live embedding-viz panel that overlays the playback window onto the static library scatter. The build scripts (`scripts/build-scaler.js`, `scripts/build-knn-library.js`, `scripts/build-inference-sample.js`) and the in-process classifier (`src/lib/server/newton.js`, `src/lib/server/projections.js`) are the cleanest reusable parts of this pattern. Includes leave-one-out badges in the UI as the per-stage health diagnostic.
