---
name: newton-machine-state
description: >
  Build n-shot classification pipelines using Newton's Machine State Lens.
  Use when the user wants to classify sensor data into categories (e.g.,
  stressed vs. relaxed, normal vs. anomaly, idle vs. active), perform
  anomaly detection on time-series data, or implement n-shot learning
  with CSV sensor data. This skill covers session lifecycle, focus CSV
  uploads, sliding window configuration, and SSE result parsing.
  Do NOT use for vision-based analysis or chart reading (use newton-activity-monitor).
  Do NOT use for initial API setup (use newton-setup).
---

# Newton Machine State Lens

Classify time-series sensor data using n-shot learning. The Machine State Lens uses labeled CSV examples (focus files) to classify new data streams via KNN over sliding windows. The batch counterpart is [`newton-machine-state-batch`](../newton-machine-state-batch/SKILL.md) — same encoder + classifier, asynchronous job. For stateless per-window classification with direct access to embeddings (no Lens, no session lifecycle), see [`newton-machine-state-direct-query`](../newton-machine-state-direct-query/SKILL.md) — `/query` with the Omega encoder plus local KNN.

## When to Apply

- User wants to classify sensor data into discrete states
- User asks about anomaly detection or state monitoring
- User wants n-shot classification without training a model
- User is building a real-time monitoring dashboard with status indicators
- User references HRV stress detection or OBD2 vehicle health patterns

## Lens ID

```
lns-1d519091822706e2-bc108andqxf8b4os
```

## Core Concepts

### N-Shot Classification

Provide labeled CSV examples (focus files) — one per class — and Newton classifies new data against those examples. No model training required.

### Sliding Windows

Newton processes CSV data in overlapping windows:
- **window_size**: Number of rows per window (default: 16)
- **step_size**: Rows to advance between windows (default: 8)
- Expected windows = `floor((dataPoints - windowSize) / stepSize) + 1`

## Workflow

### Step 1: Prepare Focus CSVs

Create one CSV per class with labeled sensor data. Each CSV should have:
- Consistent column names matching your data stream
- A representative sample of that class (50-200 rows recommended)
- No header row mismatches between focus files and query data

**Normalize before uploading if your sensors aren't already on comparable scales.** The deployed encoder runs with `normalize_input=False` and the flag is not currently exposed in the Lens config — so what you upload is what the encoder sees. If focus files and query data come from different operating conditions with different bulk amplitudes (e.g. noisy vs clean recordings, low-load vs high-load runs), the encoder reads the amplitude offset as a class signal and cross-condition accuracy collapses even when within-condition is ≥90%. Either z-score each CSV per-channel before upload, or fit a global `StandardScaler` on your focus pool and apply it to both focus and query data. See the matching [Input normalization](../newton-machine-state-batch/SKILL.md#input-normalization) section in the batch skill for the longer discussion; the local-only `normalize_input` flag and the two-option framing are documented in [omega-local/SKILL.md](../omega-local/SKILL.md#normalization-choices).

**Example: HRV Stress Detection**
```
# focus_relaxed.csv
rmssd,sdnn,mean_hr,pnn50,sd1
42.5,45.2,68.3,22.1,30.1
...

# focus_stressed.csv
rmssd,sdnn,mean_hr,pnn50,sd1
18.2,22.1,95.7,8.3,12.9
...
```

**Example: Vehicle Health Monitoring**
```
# focus_normal.csv
rpm,speed,coolant_temp,iat,engine_load,throttle,map
800,0,90,35,18,12,34
...

# focus_attention.csv
rpm,speed,coolant_temp,iat,engine_load,throttle,map
850,0,108,52,45,18,42
...
```

### Step 2: Upload Focus Files (One-Time)

```typescript
async function uploadFile(filePath: string, content: string): Promise<string> {
  const formData = new FormData();
  formData.append("file", new Blob([content], { type: "text/csv" }), filePath);

  const response = await newtonFetch("/files", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  return data.file_id;
}
```

Cache the returned `file_id` values — focus files only need to be uploaded once per session.

### Step 3: Create a Session

```typescript
const response = await newtonFetch("/lens/sessions/create", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ lens_id: "lns-1d519091822706e2-bc108andqxf8b4os" }),
});
const { session_id } = await response.json();
```

### Step 4: Configure Session with Focus Files

Send a `session.modify` event with n-shot focus file IDs and CSV config:

```typescript
await newtonFetch("/lens/sessions/events/process", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id,
    events: [{
      kind: "session.modify",
      payload: {
        input_n_shot: [
          { label: "relaxed", file_id: relaxedFileId },
          { label: "stressed", file_id: stressedFileId },
        ],
        csv_configs: {
          window_size: 16,
          step_size: 8,
        },
      },
    }],
  }),
});
```

### Step 5: Connect SSE Consumer (Before Input!)

**Critical:** Always connect the SSE consumer before setting the input stream to avoid missing results.

```typescript
const sseResponse = await newtonFetch(
  `/lens/sessions/consumer/${sessionId}`,
  { headers: { Accept: "text/event-stream" } }
);
```

### Step 6: Set Input Stream

Upload query CSV and set as input:

```typescript
const queryFileId = await uploadFile("query.csv", csvContent);

await newtonFetch("/lens/sessions/events/process", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id: sessionId,
    events: [{
      kind: "input_stream.set",
      payload: {
        streams: [{
          id: "input-0",
          source: { type: "csv_file_reader", file_id: queryFileId },
        }],
      },
    }],
  }),
});
```

### Step 7: Parse SSE Results

Each `inference.result` event contains a classification for one window:

```typescript
// SSE event data shape
{
  "kind": "inference.result",
  "payload": {
    "result": ["stressed", { "stressed": 0.85, "relaxed": 0.15 }]
  }
}
```

Aggregate across windows — majority vote or average scores:

```typescript
function aggregateResults(results: Array<[string, Record<string, number>]>) {
  const totals: Record<string, number> = {};
  for (const [label, scores] of results) {
    for (const [key, score] of Object.entries(scores)) {
      totals[key] = (totals[key] || 0) + score;
    }
  }
  const count = results.length;
  return Object.fromEntries(
    Object.entries(totals).map(([k, v]) => [k, v / count])
  );
}
```

### Step 8: Early SSE Termination (Optimization)

Calculate expected window count and close the stream early instead of waiting for the idle timeout (60-80s):

```typescript
const expectedWindows = Math.floor((dataPoints - windowSize) / stepSize) + 1;
let receivedWindows = 0;

// In SSE parse loop:
if (++receivedWindows >= expectedWindows) {
  reader.cancel(); // Close early
}
```

## Session Lifecycle

```
Upload Focus CSVs (once) → Create Session (once) → [Query Loop: Upload CSV → Set Input → Read SSE Results] → Destroy Session
```

Sessions are reusable — create once, query many times. Only recreate on error.

### Cleanup

```typescript
// Destroy session
await newtonFetch("/lens/sessions/destroy", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id: sessionId }),
});

// Delete uploaded files
await newtonFetch(`/files/delete/${fileId}`, { method: "DELETE" });
```

## NewtonStreamManager Singleton Pattern

For real-time applications, implement a server-side singleton that:

1. Buffers incoming sensor data (max ~300 points)
2. Runs periodic queries every 15 seconds
3. Manages session lifecycle (create on first query, reuse, recreate on error)
4. Pushes results to frontend via SSE

See [references/stream-manager-pattern.md](references/stream-manager-pattern.md) for the full implementation pattern.

## Parallel Per-Subsystem Sessions

A single shared session produces one verdict for the whole system. If you need to know *which* subsystem saw the anomaly — to route alerts, rank severity, or drive per-component UI — run **N parallel sessions, one per subsystem, each over a column subset of the same wide CSV**.

Canonical shape: the subsystems share the same n-shot focus files (`swat_normal.csv` / `swat_attack.csv`) but each lens registers with a different `data_columns` filter — P1's session sees `FIT101/LIT101/MV101/P101`, P3's session sees its nine UF columns, and so on. Six classifiers, one problem, six per-subsystem verdicts per window.

```js
// Per-stage lens register — same n-shot files, different column subset
{
  lens_name: `swat-stage-${stageId}-${ts}`,
  model_pipeline: [{ processor_name: 'lens_timeseries_state_processor' }],
  model_parameters: {
    model_name: 'OmegaEncoder',
    model_version: 'OmegaEncoder::omega_embeddings_1_4',
    buffer_size: 30,
    input_n_shot: { NORMAL: normalFileId, ATTACK: attackFileId },
    csv_configs: {
      data_columns: STAGE_COLUMNS[stageId],  // <-- only this stage's sensors
      window_size: 30,
      step_size: 30
    },
    knn_configs: { n_neighbors: 3, metric: 'euclidean' }
  }
}
```

`omega_embeddings_1_4` is the current default encoder version for Machine State. Other Omega encoder versions may be available on your account (newer builds, domain-specialized variants) — check with your Newton API contact for the current list, then swap the `model_version` string accordingly.

> **Note on lens-pinned vs override-able defaults.** The platform-mounted Machine State Lens (`lns-1d519091822706e2-…`) is pinned to `OmegaEncoder::omega_embeddings_01` in its base configuration — that's what `GET /v0.5/lens/metadata` reports as the lens's default. When you register a *child* lens via `POST /v0.5/lens/register` with your own `model_parameters` (as in the snippet above), the `model_version` you pass overrides the inherited default. `omega_embeddings_1_4` is what this skill recommends and what most current demos use; the `_01` value in the platform-mounted base is what older sessions inherit if no override is supplied. See [`newton-models`](../newton-models/SKILL.md) for the full catalog of identifiers exposed across `/query`, batch, and Lens surfaces.

Streaming per window fans out too — transpose the current window to channel-first (`[[col1 values], [col2 values], ...]`, not row-major) and `POST /lens/sessions/events/process` to each session in parallel. On the consumer side the browser opens N concurrent `EventSource` connections (one per session) and bucket-sorts incoming `inference.result` events by session ID to update per-subsystem state.

**When to prefer N parallel sessions over 1 shared session:**
- You need component-level verdicts (to color a plant schematic, page a specific oncall rotation, surface per-area suggestions)
- Sensors are naturally grouped by subsystem, with limited cross-group interaction
- 6× the session-create calls at startup is acceptable (`Promise.all` makes it ~1.5s instead of 9s sequential)

**When the singleton pattern is better:**
- You only care about a single "is the system OK" verdict
- Subsystems are tightly coupled and the anomaly is usually observable from any sensor
- Session count matters for cost (most Newton pricing bills by inference count, not session count — but verify for your account)

See [references/parallel-subsystem-pattern.md](references/parallel-subsystem-pattern.md) for the full pattern, including browser-side cleanup on tab close.

## Multi-Sensor N-Shot (Single Lens, 4 Variates)

Different from parallel-subsystem (N lenses, one per column subset). Here you have **N sibling channels** all watching the same subsystem during the same incident, and you want to use up to 4 of their primary measurements as the **4 variates of a single lens**. Common in benchmarks where each "channel" is one sensor's `.npy` file rather than one column of a wide CSV — e.g. NASA telemanom, where SMAP-E means 13 separate `.npy` files (E-1 through E-13) sampled during the same electrical incident.

Pattern (reference: [`archetypeai-nasa-jpl-telemanom-demo`](https://github.com/archetypeai/archetypeai-nasa-jpl-telemanom-demo)):

1. **Pick up to 4 sibling channels** (per-docs hard cap on variates). Rank by anomaly coverage or by mutual information against the union label.
2. **Truncate to the shortest channel** — sibling channels are typically extracted with slightly different lengths. Use min-length as the common timeline.
3. **Assume row-index alignment** — within a sibling group, anomaly start indices cluster tightly (e.g. SMAP-E channels all flag in rows 5000–5800), strongly suggesting wall-clock alignment. We don't have absolute timestamps to verify, so this is best-effort.
4. **Build the ground truth as the union of per-sensor anomaly ranges.** A row is "anomaly" if any sibling sensor flagged it. Intersection sounds cleaner but is wrong here — different sensors respond on different timescales (voltage drops first, temperature climbs later), so intersection produces an unrealistically narrow, often-empty GT.
5. **Send the 4 channels as 4 variates** in one lens — `csv_configs.data_columns: ["c0", "c1", "c2", "c3"]` with each `cN` being one sibling's `c0` (primary measurement).

**When this beats single-channel + mode flags:**
- Sibling channels diverge during normal operation but synchronize during anomaly (multi-sensor agreement is the signal)
- The anomaly is observable on multiple sensors with different physical principles
- You have substantial held-out normal rows for honest precision (>100)

**When it doesn't:**
- Fewer than 3 truly correlated siblings available (thin KNN library)
- Sibling channels span multiple unrelated incidents — mixing patterns dilutes the n-shot signal
- Anomaly covers >50% of the channel (no held-out normal → degenerate precision)

For groups with >4 sibling channels, combine with the parallel-subsystem pattern: run 3 lenses with disjoint 4-channel subsets and merge their predictions (majority vote or union).

## Honest Held-out Evaluation

When you have labeled ground truth and want to measure precision/recall/F1 — not just produce classifications — the n-shot pattern requires careful splitting so the model never sees the rows it's evaluated on.

### Half-and-half split per anomaly region

For each labeled anomaly range `[a, b]`:
- **First half** `[a, mid)` → feeds the anomaly focus CSV
- **Second half** `[mid, b)` → held out for evaluation

Newton sees the first halves as anomaly examples; the second halves are unseen. Precision/recall on the second halves is the honest number. (Don't hold out whole anomaly regions unless the channel has multiple — you'd lose all training signal for that anomaly *type*.)

### Held-out must exclude *every* row Newton was shown

Common bug: defining `held_out = test_rows MINUS training_ranges` (where training_ranges = first halves of anomalies). This treats the **normal focus source rows** as if they were held out. They're not — Newton saw them. The correct definition:

```
seen_rows       = normal_source_ranges ∪ anomaly_training_ranges
held_out_rows   = test_rows ∖ seen_rows
```

Then `held_out_anomaly_rows = held_out_rows ∩ all_GT_ranges` and `held_out_normal_rows = held_out_rows ∖ all_GT_ranges`. F1 is computed only on `held_out_rows`.

### Multi-segment normal focus

Don't sample normal only from pre-first-anomaly. For long channels with multiple anomaly regions, also sample the **first half of each inter-anomaly gap** and the **first half of the post-last-anomaly tail**. The second halves of those gaps stay in held-out, so evaluation remains honest.

The reason: a channel may have 5,000 rows of normal data, but if your normal focus is only the first 1,000 rows, the encoder learns "normal = early-mission pattern." Later mission phases look different and get false-alarmed even though they're normal. Multi-segment coverage attacks this directly.

```python
def multisegment_normal_ranges(seqs, n_rows, fraction=0.5, min_segment=128):
    ranges = []
    sorted_seqs = sorted(seqs)
    # Pre-anomaly
    if sorted_seqs[0][0] >= min_segment:
        ranges.append([0, sorted_seqs[0][0]])
    # First halves of inter-anomaly gaps
    for i in range(len(sorted_seqs) - 1):
        gap_start, gap_end = sorted_seqs[i][1], sorted_seqs[i + 1][0]
        if gap_end - gap_start >= min_segment:
            ranges.append([gap_start, gap_start + int((gap_end - gap_start) * fraction)])
    # First half of post-last-anomaly tail
    last_end = sorted_seqs[-1][1]
    if n_rows - last_end >= min_segment:
        ranges.append([last_end, last_end + int((n_rows - last_end) * fraction)])
    return ranges
```

## Adaptive Window Sizing

**The window must fit inside the smallest n-shot training chunk.** If `window > min(training_chunk_lengths)`, no embedding in the focus library is "pure anomaly" — every window straddles the chunk boundary into surrounding normal/noise. The result is catastrophic: Newton can't distinguish the classes (we've seen F1=0% on channels where this constraint was violated).

Heuristic:
```python
def adaptive_window(seqs, training_ranges, total_rows):
    min_chunk = min(b - a for a, b in training_ranges)
    # Largest power-of-2 that fits inside the smallest training chunk
    for w in [512, 256, 128, 64, 32]:
        if w <= min_chunk:
            window = w
            break
    # step = window / 8 for 8x overlap; bump step (halve overlap) if predictions
    # would exceed ~500 (runtime cap, since each prediction is 0.5-1s on staging)
    step = max(1, window // 8)
    while ((total_rows - window) // step + 1) > 500 and step < window // 2:
        step *= 2
    return window, step
```

**Why step = window / 8.** Consecutive windows share `window − step` rows. At step = window/8 (87.5% overlap), the embeddings are *meaningfully different* but you get 8 predictions covering each row. Step = 1 (max overlap) costs 8× more inference for no real resolution gain — adjacent windows produce nearly identical embeddings.

## MI-Picked Variates with Constant-Column Filter

When using telemetry + N mode flags as variates (single-channel mode), pick mode flags by **mutual information between flag state and the anomaly label**, computed only over rows Newton will see (no held-out leakage).

### Required filter: drop columns constant in *both* focus files

A column can have nonzero MI by accident — the y-label and x-value happen to covary across the train/normal boundary — even though within each focus file it's constant. Such a column appears as a dead axis in the KNN embedding (no information to discriminate inside either class). Preflight's `constant_columns` check flags these, and you'll lose 10–20pp F1 if you leave them in.

```python
VAR_FLOOR = 1e-6
def is_dead_in_focus(col_values, normal_mask, training_mask):
    v_n = col_values[normal_mask].var()
    v_t = col_values[training_mask].var()
    return v_n < VAR_FLOOR and v_t < VAR_FLOOR
```

**Keep informative asymmetries.** If a column is constant in *one* focus class but varies in the *other*, KEEP it — that asymmetry is exactly what KNN exploits ("this flag fires only during anomaly"). Filtering on "constant in either" is too aggressive and drops your most predictive features.

## Staging Gotchas

### `omega_embeddings_01` may not be available on staging

The production-default model version `OmegaEncoder::omega_embeddings_01` is not always exposed on `api.stage.u1.archetypeai.app`. Symptom: lens registration and session creation both succeed, the session reaches `LensSessionStatus.SESSION_STATUS_RUNNING`, but every inference emits:

```json
{"type": "inference.error",
 "event_data": {"error_messages": ["query_id: session-modify-qry-XXX failed!"]}}
```

The `session-modify-qry-` prefix is Newton's internal naming for *all* lens queries — it does NOT mean a `session.modify` event is broken. Switch the `model_version` to `OmegaEncoder::omega_embeddings_1_4` and it works.

### KNN ranking is non-deterministic under load

Same focus files, same query CSV, same lens config — F1 fluctuates ±10–15pp between runs on staging. Tie-breaking in the KNN library appears load-dependent. Report median over 3+ runs, or move to prod for stable metrics.

### SSE consumer drops mid-stream

The streaming response from `GET /lens/sessions/consumer/{session_id}` sometimes closes before all `inference.result` events are delivered (`httpx.RemoteProtocolError: peer closed connection`). Wrap the consumer loop in `try/except (httpx.RemoteProtocolError, ReadError, ConnectError, ReadTimeout)` and return whatever predictions arrived as a *partial* result — better than losing the full run on a network blip.

### Validate focus CSVs with `omega-1-4-preflight` before paying for inference

The [omega-1-4-preflight](https://github.com/archetypeai/omega-1-4-preflight) static checks (schema / timestamp / constant_columns / feature_scale / n-shot-support / schema_match / class_balance / window-vs-sampling) run in milliseconds against your focus CSVs, no API call. Run them before every classify if you're iterating on focus selection — catches most "0 predictions" mysteries before you spend 90s on a doomed Newton run.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 16 | Rows per sliding window |
| `step_size` | 8 | Row stride between windows |
| `MAX_BUFFER_SIZE` | 300 | Max sensor readings to buffer |
| `MIN_DATA_POINTS` | 20-32 | Minimum points before first query |
| `QUERY_INTERVAL` | 15000ms | Time between periodic queries |

## Best Practices

- **Focus CSV quality matters more than quantity** — 50-200 well-labeled rows per class is sufficient
- **Column names must match** between focus files and query data exactly
- **Reuse sessions** — creating sessions has overhead; create once and query repeatedly
- **Always connect SSE before setting input** — otherwise results may be missed
- **Implement early termination** — avoids 60-80s idle wait per query
- **Graceful degradation** — apps should work without Newton; check availability first
- **Clean up on tab close** (browser-side sessions) — fire `DELETE /lens/sessions/destroy` from a `pagehide` handler with `fetch(..., { keepalive: true })` so orphaned sessions don't accumulate on refresh/close. `navigator.sendBeacon` works too but only sends POSTs.
- **Clean stale lenses on startup** — before registering a new lens, `GET /lens/metadata` and delete any old ones matching your name prefix. Lens registrations persist across sessions and accumulate.
- **Channel-first transpose** — Machine State Lens expects streams as `[[col1 values], [col2 values], ...]`, not row-major. If classifications look random, check this first.

## Building a frontend on top of this

If you're wrapping the Machine State Lens in a React/Svelte/etc. UI — per-stage anomaly dashboard, real-time classification monitor, n-shot replay tool — **read [`DESIGN.md`](../../DESIGN.md) at the root of this repo before writing any CSS**. The Archetype design system (Tailwind v4 + `@archetypeai/ds-lib-tokens` + Geist sans/mono + OKLCH palette + dark-first) is the expected visual language for these demos. [`archetypeai-swat-demo`](https://github.com/archetypeai/archetypeai-swat-demo) is the canonical reference implementation for Machine State (6 parallel SSE sessions, per-stage cards with `good` / `warning` / `critical` Badge variants, mono numeric readouts, sharp 2px radii); [`archetypeai-wifi-demo`](https://github.com/archetypeai/archetypeai-wifi-demo) shows the same patterns applied to a different domain; [`archetypeai-nasa-jpl-telemanom-demo`](https://github.com/archetypeai/archetypeai-nasa-jpl-telemanom-demo) demonstrates honest held-out evaluation on the NASA telemanom benchmark with single-channel (telemetry + MI-picked mode flags) and subsystem (multi-sensor variates with union GT) modes side-by-side, including adaptive window sizing keyed to the smallest n-shot training chunk. Setting this up at the start is much cheaper than retrofitting later.
