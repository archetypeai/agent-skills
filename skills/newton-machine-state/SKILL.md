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
- **Rows are a single temporally-contiguous block** — not `random.sample(...)` rows scattered across the recording. The Lens uses a sliding window over the focus CSV during library construction, so non-contiguous rows produce embeddings of fake signals (each window concatenates physically distant moments). Pick the longest contiguous run of that class from your raw labeled data and slice the first N rows. Same recipe as the batch pipeline — see [Recommended n-shot data prep](../newton-machine-state-batch/SKILL.md#recommended-n-shot-data-prep-contiguous--z-scored).

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

### Step 6: Push windows via `session.update` (channel-first)

**Stream data in via `session.update` events with channel-first numeric arrays — not via `csv_file_reader`.** Both modes were canonical at different points, but on the current platform-mounted Machine State Lens (`lns-1d519091822706e2-…`), pointing `input_stream.set` at a pre-uploaded CSV is accepted (`is_valid: true` for `session.modify` / `input_stream.set` / `output_stream.set`), the session reaches `SESSION_STATUS_RUNNING`, but **zero `inference.result` events ever fire** — only `sse.stream.heartbeat` until the idle timeout (~58s) closes the SSE. Reproduced with a 13,104-row CSV and `window_size=512, step_size=512` against a child lens pinned to `omega_embeddings_1_4`. The same data, same focus files, same lens — switched to `session.update`-push mode — produced 91 of 102 predictions cleanly.

```typescript
// Channel-first: outer array = channels, inner array = window samples.
// For a 4-channel × 128-sample window, sensor_data is [[a1...], [a2...], [a3...], [a4...]].
const windowRows = data.slice(windowStart, windowStart + WINDOW_SIZE);
const sensorData = COLUMNS.map((col) => windowRows.map((r) => parseFloat(r[col])));

await newtonFetch("/lens/sessions/events/process", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id: sessionId,
    event: {
      type: "session.update",
      event_data: {
        type: "data.json",
        event_data: {
          sensor_data: sensorData,
          sensor_metadata: {
            sensor_timestamp: Date.now() / 1000,
            sensor_id: `window_${windowIndex}`,
          },
        },
      },
    },
  }),
});
```

The full pattern (per-stage parallel sessions + channel-first transpose) lives in [references/parallel-subsystem-pattern.md](references/parallel-subsystem-pattern.md).

**Throttle pushes — Newton's lens runner drains ~1 inference/s/session and the input buffer is ~20 windows deep.** Push 102 windows back-to-back and you get *exactly* 20 predictions before the runner goes silent (no error, no `sse.stream.end` — just heartbeats until idle timeout). Cap pushes at one per second per session and you get all 102. The buffer-depth limit is per-session; running N parallel sessions multiplies your effective rate by N at the cost of N runner slots.

```typescript
const MIN_PUSH_INTERVAL_MS = 1000;
let lastPushTime = 0;
async function pushPaced(windowIndex) {
  const wait = MIN_PUSH_INTERVAL_MS - (Date.now() - lastPushTime);
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  await pushWindow(windowIndex);
  lastPushTime = Date.now();
}
```

**Warmup also requires a minimum-queued-depth — pushing just 1 window doesn't kick the inference pipeline.** `preWarmSessions(count = 1)` in the swat-demo works for that app (6 parallel sessions), but on 2-session setups we've observed Newton sit idle indefinitely after a single `session.update`. Reproduced with two sessions on the same child lens: push 1 window each → 0 `inference.result` events in 60s+ (no error, no stream end, just heartbeats). Push 5 each → same silence. Push all available windows → first prediction at t≈37s, predictions stream at ~1/s/session thereafter. We didn't isolate the exact threshold between 5 and 102 — until we do, queue *all* windows up front during warmup and let the per-session 1s throttle drain them in the background. The cost is bounded by the existing rate-limit, so over-priming has no downside.

```typescript
// Warmup: push every window through the throttled queue immediately.
// NewtonSession's internal 1-pop-per-second cadence keeps the lens runner
// fed without overflowing the ~20-window buffer.
async function warmup(session) {
  for (let i = 0; i < session.nWindows; i++) {
    enqueue(session, i);  // returns immediately; thread drains at 1/s
  }
  // Wait for first non-"unknown" inference.result before showing UI.
  await session.firstRealVerdict;
}
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

### One lens per stream — even when the column set is identical

The "share one lens across N sessions" shortcut looks tempting when every stream watches the **same** columns (e.g., a fleet of identical wind turbines, all classified against the same 4 SCADA channels). Resist it. Reproduced against `api.stage.u1.archetypeai.app` on a Penmanshiel turbine demo: one child lens, two `POST /lens/sessions/create` calls a few seconds apart, both reach `SESSION_STATUS_RUNNING`, both consume the same `session.update` push pace — but **only one of the two sessions ever emits `inference.result` events**. The other reaches RUNNING and stays silent for the entire 5-minute run. The lens-runner pool appears to either bind a runner per-lens (not per-session) or some shared state inside the lens serializes inference.

Registering one child lens per turbine (same `model_parameters` apart from `lens_name`) eliminated the silence — both sessions emitted predictions at parallel cadence (33 predictions each in 80s, with first verdicts arriving at t≈46s for both).

Concretely: the parallel-subsystem snippet above already gives you the right shape — keep the `STAGE_COLUMNS` loop even when every stage's columns happen to be the same, and register N lenses with N distinct `lens_name`s. The cost is N extra `lens/register` calls at startup (~2s each) and N cleanup deletions on teardown; the benefit is N independent runners with no cross-session contention.

See [references/parallel-subsystem-pattern.md](references/parallel-subsystem-pattern.md) for the full pattern, including browser-side cleanup on tab close.

### Account runner quota — when even one-lens-per-stream fails

Even with one lens per stream, N parallel sessions assume your account has ≥ N concurrent runner slots in the lens-runner pool. Some staging accounts are provisioned with **just 1 slot**, which means the second `POST /lens/sessions/create` fails immediately with `"Failed to allocate lens runner - try stopping an older session!"` regardless of how long you wait — there is no temporal cooldown that fixes it because the slot stays occupied for the entire lifetime of the first session.

Reproduced on the Penmanshiel turbine demo: `GET /lens/sessions/metadata` showed exactly one active session across the whole account (our own, just created), but the next create-session call still returned the allocation failure. Stretching the inter-session delay from 0s → 5s → 15s changed nothing.

When that happens, route both/all streams **through a single shared session** and route results by FIFO push-tag order:

1. **One lens, one session.** Register a single child lens with the shared `data_columns` / focus files / etc., create a single session on it.
2. **Interleave pushes.** Round-robin window pushes from each stream into the same `session.update` queue. Tag each push with `sensor_metadata.sensor_id = f"{stream_id}_{window_index}"` for traceability (Newton doesn't echo this back today, but it's good practice and survives future-API changes).
3. **Route by FIFO tag order.** Keep an in-process FIFO of `(stream_id, window_index)` tags, one entry per push. Pop the front on each `inference.result` to determine which stream this result belongs to. Newton processes pushes serially (~1/s) and SSE events arrive in processing order, so the tag-order ↔ result-order correspondence holds as long as you keep per-stream FIFO at the producer (don't push window N+1 of stream A before window N of stream A lands).

```python
class MultiplexNewtonSession:
    def __init__(self, streams):
        self._frames = {sid: load_frame(wt_id) for sid, wt_id in streams}
        self._pending_tags = queue.Queue()  # (stream_id, window_index) in push order
        ...

    def push_next_window(self, stream_id):
        w_idx = self._pushed[stream_id]
        self._pushed[stream_id] += 1
        self._push_queue.put((stream_id, w_idx))

    # In the SSE consumer:
    def on_inference_result(self, event):
        stream_id, w_idx = self._pending_tags.get_nowait()
        emit({"kind": "newton_prediction", "stream_id": stream_id, "window_index": w_idx, ...})

    # In the push loop:
    def push_loop(self, session_id):
        while not self._stop.is_set():
            stream_id, w_idx = self._pushes.get(timeout=1.0)
            self._pending_tags.put((stream_id, w_idx))
            client.lens.sessions.process_event(session_id, build_event(stream_id, w_idx))
            time.sleep(MIN_PUSH_INTERVAL_SEC)
```

Verified on the same Penmanshiel demo that previously hit the allocation error: switched from two-lens / two-session to one-lens / one-session-with-multiplex, both turbines now stream interleaved verdicts (WT01: 34 predictions, WT09: 33 predictions in 80s) with zero allocation errors. The per-stream prediction throughput halves (since both streams share one ~1/s runner), but for visualization workloads that's typically fine.

**Prefer one-lens-per-stream when your runner pool allows it** — independent runners mean per-stream throughput doesn't degrade as N grows. Multiplexing is the fallback when the pool is undersized; treat it as a quota-shape adapter, not the default.

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

### Silent inference: the four observed failure modes

The Machine State Lens has at least four ways to **accept a session, reach `SESSION_STATUS_RUNNING`, and then emit zero `inference.result` events** — no error, just SSE heartbeats until the ~58s idle timeout. Diagnosing this in isolation is painful because the SDK calls all return `is_valid: true` and the session metadata shows `RUNNING` the whole time. The fixes:

1. **Use `omega_embeddings_1_4`, not `_01`.** Per [`newton-models`](../newton-models/SKILL.md), the platform-mounted Machine State Lens pins `OmegaEncoder::omega_embeddings_01`. On `api.stage.u1.archetypeai.app` this version has been observed to emit `inference.error` events (and sometimes nothing at all) instead of `inference.result`:
   ```json
   {"type": "inference.error",
    "event_data": {"error_messages": ["query_id: session-modify-qry-XXX failed!"]}}
   ```
   The `session-modify-qry-` prefix is Newton's internal naming for *all* lens queries — it does NOT mean a `session.modify` event is broken. Register a child lens with `model_version: "OmegaEncoder::omega_embeddings_1_4"` to fix.

2. **Push via `session.update`, not `csv_file_reader`.** See [Step 6](#step-6-push-windows-via-sessionupdate-channel-first) — file-reader input mode is silently broken on staging today even with `_1_4`.

3. **Throttle pushes to ≤1/s/session.** See Step 6's throttling sub-section. Pushing the SDK's full pre-uploaded CSV in one burst (via `csv_file_reader`) compounds with mode #2; pushing 102 channel-first windows in one second via `session.update` triggers the same buffer-overflow → silent-skip. The runner emits exactly 20 predictions then goes quiet for the rest of the session.

4. **Set config at lens-register time, not in `session.modify`.** Putting `input_n_shot`, `csv_configs`, and `knn_configs` in the `model_parameters` block of `lens/register` (per the [parallel-subsystem pattern](references/parallel-subsystem-pattern.md)) is the route observed-to-work end-to-end. Splitting the same fields between `lens/register` and `session.modify` has produced ambiguous silent runs in the wild — defaults can leak through.

The four fixes compose: working production code uses all of them. Failing in only one mode tends to mask the others.

### Orphan sessions hold lens runners — clean on every request that creates sessions

Killing a Python process mid-run (Ctrl-C, `pkill`, container restart) doesn't tear down the platform-side lens session — `auto_destroy` only fires through the SDK's normal exit path. Each orphan holds a `lens_service:runner:atai-platform-lens-node-worker-...` slot from a finite pool. Once all slots are occupied, the next `POST /lens/sessions/create` fails:

```json
{"errors": ["Failed to allocate lens runner - try stopping an older session!"]}
```

**Process-startup cleanup is necessary but not sufficient.** Long-running servers (Flask, FastAPI, Svelte hot-reload, etc.) reach the same failure state without ever restarting, because *abandoned client connections* leak runners just as readily as killed processes:

- Browser tab closes mid-stream → SSE generator exits abruptly, `auto_destroy` doesn't fire.
- `curl --max-time` (or any client-side deadline) terminates the SSE before the server-side teardown runs.
- Hot-reload re-binds the route handler before the previous handler's `finally` block completes.

A single 30-minute Flask session with 5–10 of these abandonments accumulates enough orphans to starve the next `/lens/sessions/create` call. Move cleanup to the top of every request handler that creates sessions:

```python
def cleanup_orphan_sessions(client, lens_name_prefix):
    sessions = client.lens.sessions.get_metadata()
    my_lens_prefix = "lns-" + lens_name_prefix[:16]  # lens_id encodes first 16 chars of lens_name
    for s in sessions:
        if (s.get("lens_id") or "").startswith(my_lens_prefix):
            try:
                client.lens.sessions.destroy(s["session_id"])
            except Exception:
                pass  # already destroyed, race with another handler, etc.

@app.route("/api/replay")
def replay():
    cleanup_orphan_sessions(client, "my-app")  # before NewtonSession(...).start()
    ...
```

Also wrap the SSE generator body in `try/finally` so your own session-close path always runs on client disconnect:

```python
@stream_with_context
def _stream():
    sessions = {wt: NewtonSession(wt) for wt in turbines}
    for s in sessions.values(): s.start()
    try:
        yield from _stream_body(sessions)
    finally:
        for s in sessions.values():
            s.close()  # auto_destroy → DELETE /lens/sessions/destroy
```

`client.lens.sessions.get_metadata()` returns a list of currently-active sessions across your account (not just yours), each with `session_id`, `lens_id`, `session_status`, `session_duration_sec`. Filter by your child-lens prefix before destroying so you don't tear down a coworker's session.

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
