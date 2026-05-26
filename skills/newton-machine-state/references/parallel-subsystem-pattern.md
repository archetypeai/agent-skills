# Parallel Per-Subsystem Sessions Pattern

When a single lens session gives you one verdict for the whole system but you need to know *which* subsystem/component/stage is anomalous, run N parallel sessions — one per subsystem — each filtered to its own column subset of the same wide CSV.

This is the pattern used by [archetypeai-swat-demo](https://github.com/archetypeai/archetypeai-swat-demo) (6-stage water treatment plant: P1–P6) and applies anywhere sensor data is naturally grouped by subsystem: industrial processes with staged flow, multi-zone HVAC, multi-well oil fields, multi-rack data centers.

## When to Apply

- Wide CSV with sensors from multiple subsystems mixed together
- You need per-subsystem verdicts, not a plant-wide verdict
- Subsystems have enough independent behavior that cross-contamination (one subsystem's anomaly triggering a neighbor's session) is a real risk
- Same two-class problem (normal vs anomaly) across all subsystems

## Architecture

```
                          ┌─────────────────┐
swat_normal.csv ─────────▶│ Upload once     │──── normalFileId
swat_attack.csv ─────────▶│ (shared n-shot) │──── attackFileId
                          └─────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
       register lens P1    register lens P2    … register lens P6
       data_columns:       data_columns:         data_columns:
       [FIT101, LIT101,    [AIT201, AIT202,      [FIT601, P601,
        MV101, P101]        AIT203, FIT201,        P602]
                            MV201, MV202, ...]
              │                   │                   │
              ▼                   ▼                   ▼
       create session P1   create session P2    … create session P6
              │                   │                   │
              ▼                   ▼                   ▼
            EventSource         EventSource         EventSource
            (one SSE            (one SSE            (one SSE
             per session)        per session)        per session)
              │                   │                   │
              └───────────────────┴───────────────────┘
                                  │
                                  ▼
                     per-subsystem stageStatuses
                     { P1: 'attack', P2: 'normal', ... }
```

## Step-by-Step

### 1. One n-shot upload, many lens registers

Upload `normal.csv` and `attack.csv` once — they describe the two-class problem, not any single subsystem. The per-subsystem specialization happens at lens-register time via `data_columns`.

```js
const STAGE_COLUMNS = {
  P1: ['FIT101', 'LIT101', 'MV101', 'P101'],
  P2: ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'MV202', 'P201', 'P203', 'P205'],
  P3: ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302'],
  P4: ['AIT401', 'AIT402', 'FIT401', 'LIT401', 'P402', 'UV401'],
  P5: ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504',
       'P501', 'PIT501', 'PIT502', 'PIT503'],
  P6: ['FIT601', 'P601', 'P602']
};

const normalFileId = await uploadFile('normal.csv', normalCsv);
const attackFileId = await uploadFile('attack.csv', attackCsv);

// Register all six lenses in parallel
const lenses = await Promise.all(
  Object.entries(STAGE_COLUMNS).map(([stageId, cols]) =>
    newtonFetch('/lens/register', {
      method: 'POST',
      body: JSON.stringify({
        lens_name: `myapp-stage-${stageId}-${Date.now()}`,
        model_pipeline: [{ processor_name: 'lens_timeseries_state_processor' }],
        model_parameters: {
          model_name: 'OmegaEncoder',
          model_version: 'OmegaEncoder::omega_embeddings_1_4',
          buffer_size: 30,
          input_n_shot: { NORMAL: normalFileId, ATTACK: attackFileId },
          csv_configs: { data_columns: cols, window_size: 30, step_size: 30 },
          knn_configs: { n_neighbors: 3, metric: 'euclidean' }
        }
      })
    }).then((r) => r.json().then((d) => ({ stageId, lens_id: d.lens_id })))
  )
);
```

### 2. One session per lens

```js
const sessions = await Promise.all(
  lenses.map(({ stageId, lens_id }) =>
    newtonFetch('/lens/sessions/create', {
      method: 'POST',
      body: JSON.stringify({ lens_id })
    }).then((r) => r.json().then((d) => ({ stageId, session_id: d.session_id })))
  )
);
```

### 3. Fan out each window to all sessions in parallel

Each sensor window (e.g., 30 rows) is fired to all N sessions. Each session sees only its configured columns even though the source window contains all sensors — the lens does the column projection.

**Critical:** The payload must be channel-first (`[[col1 values], [col2 values], ...]`), not row-major. Newton expects columns, not rows.

```js
function transposeToChannelFirst(rows, columns) {
  // rows: [{FIT101: 2.5, LIT101: 540, ...}, ...]
  // columns: ['FIT101', 'LIT101', ...]
  return columns.map((col) => rows.map((r) => parseFloat(r[col])));
}

async function streamWindow(windowRows) {
  await Promise.allSettled(
    sessions.map(({ stageId, session_id }) => {
      const channels = transposeToChannelFirst(windowRows, STAGE_COLUMNS[stageId]);
      return newtonFetch('/lens/sessions/events/process', {
        method: 'POST',
        body: JSON.stringify({
          session_id,
          events: [{
            kind: 'input_stream.push',
            payload: { streams: [{ id: 'input-0', data: channels }] }
          }]
        })
      });
    })
  );
}
```

### 4. N parallel `EventSource` connections on the browser

Each session has its own SSE URL. The browser opens N connections and routes incoming events by the session they arrived on:

```js
const stageStatuses = $state({});  // P1: 'normal', P2: 'attack', ...

for (const { stageId, session_id } of sessions) {
  const es = new EventSource(`/api/sse-proxy?session=${session_id}`);
  es.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.kind === 'inference.result') {
      stageStatuses[stageId] = msg.payload.result[0].toLowerCase();
    }
  };
}
```

## Session Cleanup on Browser Unload

Browser sessions orphan easily — tab close, refresh, and navigation all drop `EventSource` without signaling the server. Lens sessions on Newton's side persist until explicitly destroyed, and over time that balloons your session count.

Use `pagehide` (works on tab close AND navigation, unlike `beforeunload`) with `fetch(..., { keepalive: true })`:

```js
window.addEventListener('pagehide', () => {
  for (const { session_id } of sessions) {
    fetch('/api/session/destroy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id }),
      keepalive: true  // <-- lets the request survive unload
    });
  }
});
```

`keepalive` is limited to ~64KB total per page and sometimes silently drops requests — pair it with a server-side janitor that reaps stale sessions older than N minutes.

## Stale Lens Cleanup on Startup

Lens registrations (not sessions) also persist across app restarts. If your app re-registers on every startup without cleaning up, you'll accumulate thousands of orphaned lenses.

Before registering, list and delete old ones that match your naming prefix:

```js
const metadata = await newtonFetch('/lens/metadata').then((r) => r.json());
const stale = (metadata.lenses ?? []).filter((l) =>
  l.lens_name?.startsWith('myapp-stage-')
);
await Promise.all(
  stale.map((l) =>
    newtonFetch(`/lens/delete/${l.lens_id}`, { method: 'DELETE' })
  )
);
```

## Gotchas

- **Channel-first transpose is non-negotiable.** If per-subsystem verdicts look random or all identical, check the shape of what you're POSTing. Row-major (`[[row1], [row2], ...]`) will classify but produce garbage.
- **Parallel session-create is ~6× faster than sequential** — use `Promise.all`. SWaT's startup went from ~9s sequential to ~1.5s parallel for 6 sessions.
- **Silent skip on a single session doesn't fail the batch.** `Promise.allSettled` (not `Promise.all`) for per-window fanouts so one stage's hiccup doesn't crash the whole stream.
- **Same-window guarantees on SSE are weak.** Two sessions processing the same window can emit `inference.result` seconds apart. If the UI needs "all N verdicts for window K" coherently, buffer by window ID until you have all N.
- **Classifications may flap** on weakly-supported windows. If downstream logic (e.g., operator suggestions) keys off the anomaly set, debounce with a minimum stable duration (~1–2s) or track hysteresis.

## Related

- [stream-manager-pattern.md](stream-manager-pattern.md) — the single-session singleton variant, for cases where you don't need per-subsystem verdicts.
- [archetypeai-swat-demo](https://github.com/archetypeai/archetypeai-swat-demo) — full working implementation of this pattern with 6 parallel P1–P6 sessions.
