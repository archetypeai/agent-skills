---
name: atai-rare-event-detection-agent
description: >
  Run Archetype AI's managed Rare Event Detection (RED) agent over the Agent
  API — upload a sensor CSV, create a bundle from the canonical `red`
  blueprint pinning a fitted nearest-prototype classifier, run it (one agent
  per input file), poll status + audit events, and download per-window event
  predictions. Use this skill when the user has a small number of labelled
  examples of a *named* rare fault (as few as one incident), plus normal-
  operation data, and wants fully-managed server-side detection of that
  fault recurring — equipment breakdowns, process excursions, undesirable
  well events. Covers the bundle request shape (the `rad-classifier`
  artifact slot, `step_size`), the fit path via the `red-fitting` blueprint
  with `strategy: centroid`, the run/poll/results lifecycle, the output CSV
  schema (`timestamp, predicted_state, invalid, p_<class>…`), and why
  incident-level detection must be scored separately from window-level
  accuracy. Do NOT use for classifying every operating regime from a full
  labelled library (that's `atai-operational-state-monitoring-agent`), for
  unnamed/uncharacterized anomalies against normal-only data, for
  client-side embedding + KNN over `/query` (`atai-newton-omega-model`), or
  for cleaning and windowing raw CSVs
  (`atai-newton-omega-model-data-prep`).
---

# RED Agent — Managed Rare Event Detection via the Agent API

The RED agent detects **named, rare events** from a handful of labelled
examples. It is the few-shot middle path between two siblings: OSM needs a full
labelled library across every regime you care about, anomaly detection needs
only normal data but flags *everything* unusual. RED takes a small named catalog
— "pump breakdown", "severe slugging" — with a few labelled examples each plus
normal data, and detects those specific signatures recurring.

Algorithmically it is two modules: the **Omega encoder** turns windowed
multivariate sensor data into embeddings, and a **nearest-prototype classifier**
assigns each window to the closest class prototype. One prototype per class —
the arithmetic mean of that class's shot embeddings — which is what makes it
tolerate the severe imbalance between abundant normal data and one or two fault
incidents.

The platform runs the whole graph server-side:

```
source → interpolate → window → windowInterpolate → samplingRate
       → limitValues → encoder (omega:1.5) → classifier → sink
```

One run = one agent instance = one input file. You upload the CSV, create a
**bundle** from the canonical `red` **blueprint** (pinning your classifier
artifact), run the bundle, poll until terminal, and download one output CSV of
per-window predictions.

## When to Apply

- Detect a **named** fault recurring, given only one or two labelled incidents
  of it plus normal-operation data
- Deploy a detector as a repeatable batch job with **no client-side ML** — the
  platform embeds and classifies every window
- Score a few-shot detector honestly, where standard accuracy is meaningless
  because the positive class is under 1% of windows

**Do not use this skill when:**
- The user has a full labelled library across all regimes and wants "which state
  is the asset in?" — use
  [`atai-operational-state-monitoring-agent`](../atai-operational-state-monitoring-agent/SKILL.md)
- The faults are unnamed or uncharacterized and only normal data exists — RED
  needs a named catalog; flagging everything unusual is a different problem
- You want per-window embeddings to do ML client-side — use
  [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)
- The raw CSV still needs cleaning, gap-aware segmentation or normalization —
  see [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md).
  RED assumes prepared, z-scored input with a regular sampling rate

## Endpoints

```
Files API   POST {ATAI_API_ENDPOINT}/v0.5/files                  (multipart upload)
            GET  {ATAI_API_ENDPOINT}/v0.5/files/download/{name}  (download)
Agent API   {ATAI_API_ENDPOINT}/agents/...                       (versionless!)
Authorization: Bearer <API_KEY> on every call
```

The Agent API is **versionless** — `/agents`, not `/v0.5/agents`. If
`ATAI_API_ENDPOINT` carries a `/vX.Y` suffix, strip it before appending
`/agents`. Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required; there is no
default endpoint.

> **⚠️ Dev-only for now.** Everything here is verified against the **Dev**
> deployment (`https://api.dev.u1.archetypeai.app`) — the canonical `red`
> blueprint, the `red-fitting` blueprint, the Agent API surface, and the runtime
> numbers below.

## The five-step lifecycle

### 1. Upload the input CSV

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@prepared_slice.csv;type=text/csv" \
  "$ATAI_API_ENDPOINT/v0.5/files"
```

Returns both `file_id` (the filename, what connectors take) and `file_uid`
(a `fil_…` handle). **Use `file_id` in the connector**, not `file_uid`.

Input requirements the blueprint enforces: a fixed, regular sampling rate;
one numeric column per variate; a timestamp column (ISO 8601 or Unix); and
z-normalized values per variate. Irregular timestamps come back as
`INVALID_STATE` rather than predictions.

### 2. Create a bundle from the `red` blueprint

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundle" -d '{
    "blueprint": "red",
    "name": "pump breakdown detector",
    "values": {"step_size": 1},
    "artifacts": {"rad-classifier": "s3://bucket/path/fit-classifier.safetensors"}
  }'
```

**The artifact key is `rad-classifier`.** That string is the model name the
`red` blueprint declares (`models.classifier: "rad-classifier"`), so it is what
the `artifacts` map must be keyed by — `red-classifier` fails. "rad" is an
internal name for the same agent; the blueprints' source files are `rad.yaml`
and `rad_fitting.yaml`.

`step_size` is normally the only value worth setting. Everything else —
`window_size`, `data_columns`, `timestamp_column`, `encoder_model` — is
inherited from the classifier's own `parameters` metadata via
`${models.classifier.parameters.window_size:1024}`, which is why the payload is
so small. Omit `step_size` and you inherit the stride the classifier was fit
with.

### 3. Run the bundle — one agent per input file

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundle/$BUNDLE_ID/run" -d '{
    "connectors": {"source": [{"type": "file", "id": "prepared_slice.csv"}]}
  }'
```

Returns a new `agt_…` instance. Reuse the same bundle for further files — the
classifier does not change.

### 4. Poll until terminal

```sh
curl -H "Authorization: Bearer $ATAI_API_KEY" "$ATAI_API_ENDPOINT/agents/instances/$AGENT_ID"
curl -H "Authorization: Bearer $ATAI_API_KEY" "$ATAI_API_ENDPOINT/agents/instances/$AGENT_ID/events"
```

`status` goes `running` → `completed` / `failed` / `canceled`. The events stream
is the audit log and announces the JOS job id at dispatch.

### 5. Download the results

```sh
curl -H "Authorization: Bearer $ATAI_API_KEY" "$ATAI_API_ENDPOINT/agents/instances/$AGENT_ID/results"
```

Each result nests its fields under an inner `data` object: `filename`,
`num_bytes`, and `ref`. For a **run output** the `ref` is a *relative*
`/files/download/{name}` path that resolves under `/v0.5` and needs the bearer
token; `expires_at` is `null`, so run outputs do not expire. For a **fitted
classifier artifact** the `ref` is an absolute presigned S3 URL that expires in
~20 minutes — derive a durable `s3://` path instead (see below).

Output is one row per window, keyed to the window-end timestamp:

```
timestamp,predicted_state,invalid,p_normal,p_pump_breakdown
1530962520.0,normal,false,1,0
```

`predicted_state` is the class name or `INVALID_STATE`; `p_<class>` columns are
per-class probabilities when `output_probabilities` is on (the default).

## Fitting the classifier — the `red-fitting` blueprint

RED's classifier is fitted by a second blueprint whose sink uses
**`strategy: centroid`** — one mean prototype per class, which is the design's
nearest-prototype head. The kNN knobs (`k`, `weights`) are absent by design:
with one candidate per class there is nothing to vote on.

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundle" -d '{
    "blueprint": "red-fitting",
    "name": "pump breakdown fit",
    "values": {"window_size": 64, "step_size": 1, "metric": "L2",
               "strategy": "centroid", "timestamp_column": "timestamp",
               "data_columns": ["sensor_02", "sensor_04"],
               "states": ["normal", "pump_breakdown"]}
  }'
```

Then run it over one CSV per class. Four things about this are not discoverable
from the blueprint's API response:

- **The class label comes from the FILENAME.** Each input file is matched to the
  class whose name appears in it (case-insensitively, longest match wins), so
  `shots_pump_breakdown_inc01.csv` → `pump_breakdown`. A renamed file silently
  mislabels every window in it. Each shot file must therefore contain **only**
  rows of its own class.
- **`states` is not a blueprint value.** The runner strips it and uses it as the
  class vocabulary for that filename matching.
- **`output` does not name the file.** The blueprint defaults to
  `rad-classifier.safetensors`, but the runner writes
  **`fit-classifier.safetensors`** regardless. Read the real filename off
  `/results`; building an `s3://` URI from the requested name points at nothing
  and the inference run then fails at artifact load.
- **`red-fitting` is not canonical** (`is_canonical: false`), so it is org-scoped
  and an org **admin** must register it once before anyone can fit. The `red`
  inference blueprint *is* canonical and needs no setup.

The artifact's durable location follows the JOS convention:

```
s3://<bucket>/jos/jobs/<job_id>/agent/worker-0/outputs/output/fit-classifier.safetensors
```

Use that raw `s3://` form as the bundle artifact — the `/results` `ref` for a
fitted artifact is presigned and expires.

The result is tiny and worth reading in full:

```
tensors   ids [K] U64 · vectors [K, D] F32 · weights [K] F32
metadata  parameters = {window_size, step_size, data_columns, encoder_model,
                        timestamp_column}     <- inference inherits these
          classifier = {config: {strategy: "centroid", metric, …},
                        labels: ["normal", "pump_breakdown"]}
          index      = {backend: "ball_tree", metric, dim, n: K}
```

K rows for K classes. A two-class, 10-channel classifier is **62 KB** —
`vectors [2, 7680]`, since Omega emits 768 dims per variate and concatenates.

## Scoring: window-level accuracy is misleading here

A rare-event detector must be scored three ways, because the positive class is
often under 1% of windows.

**Window level** — precision / recall / F1 per class. Label each window by the
design's majority rule: a window's truth is the label holding the majority of
its rows, ties going to the rare event.

**Incident level** — for each contiguous ground-truth episode, was it detected
at all, and how long after onset? This is the number an operator cares about,
and it is *not* derivable from window averages.

**False-alarm rate** — fault predictions on windows containing **zero** fault
rows. Unlike precision this stays comparable across window sizes.

Why all three: a slice can score **0.9941 accuracy while missing its incident
entirely**. With 42 fault rows against 8,192 normal ones, predicting `normal`
everywhere is 99.4% correct and 100% useless. Nothing in the window-level table
catches that.

Two structural facts follow from majority labelling:

- **`window_size` is an upper bound on detectable event duration.** An event
  occupying less than half a window can never be labelled as the rare event, so
  `window_size <= 2 × (shortest event you must catch)`. Choose it from an
  operational requirement, not from the evaluation set.
- **Incident detection is an OR across windows** — it needs only one window to
  fire, which is a different quantity from per-window accuracy. A coarse
  `step_size` therefore costs detections outright, not just temporal resolution.

## Verified platform behavior

- **Artifacts must be `s3://` URIs.** The files API's MIME allowlist rejects
  `.safetensors`, and `ClassifierNode` resolves artifact strings as
  filesystem/S3 paths only — a platform `file_id` or an `https://` URL fails
  with ENOENT without attempting a fetch.
- **All-`INVALID_STATE` output means input validation, not a model problem.**
  The blueprint defaults to `validate_monotonic_timestamps: true` and
  `sample_rate_interval_tolerance: 0.05`. A common cause is a timestamp bug in
  prep: with pandas 2.x+, `astype("int64") // 10**9` divides by 1000× too much
  when parsing yields microsecond resolution, collapsing a 1-minute cadence so
  every timestamp is equal. Use `astype("datetime64[s]").astype("int64")`.
- **Two NaN-interpolation nodes exist** that OSM's graph lacks:
  `ValueInterpolationNode` and `InWindowsInterpolationNode` repair short NaN
  runs server-side (`window_interpolation_max_gap` defaults to 16).
- **Output timestamps are floats.** The platform emits `1530962520.0` where the
  input carried integer seconds — join on the numeric value, not the string.
- **A `failed` status can hide a successful run.** We have seen
  `repeated failures polling JOS job` on a job that completed. Always check
  `/results` before re-running.
- **Encoder throughput is ~2.2 windows/s and roughly independent of window
  size** (measured 1.85–2.34 win/s across `w=32`…`w=256`, an 8× range in samples
  per window). Budget by *window count*: a 12,059-window fit takes ~90 minutes,
  and inference is ~2.1–2.3 win/s. A local reference encoder on one M-series GPU
  runs ~15× faster, which suggests per-window overhead rather than compute.
- **Run agents sequentially.** Concurrent runs share dev's GPU workers; five at
  once ran roughly 5× slower each. Total GPU work is unchanged, so what
  concurrency costs is feedback, not throughput.
- **Cancel with `POST /agents/instances/{id}/cancel`.** Killing a local client
  does not stop the job — `DELETE` returns 409 while running.

## References

- `references/run_red_agent.py` — stdlib-only end-to-end runner: upload →
  bundle → run → poll → download → score, with the three scoring views above.
- `references/.env.example` — the two required environment variables.
- `references/sample_data/` — prepared pump slices: three single-class shot
  files for fitting and a held-out slice with a ground-truth sidecar for running
  and scoring. See its README for full data attribution.

The full seven-stage build (raw CSV → preflight → prep → grid search → fit →
run → evaluate), including the channel-leakage audits this data requires, lives
in the
[RED example repo](https://github.com/archetypeai/rare-event-detection-agent-example).

## Data attribution

The sample data derives from the **`pump_sensor_data`** dataset published by the
Kaggle user [`nphantawee`](https://www.kaggle.com/nphantawee):
**https://www.kaggle.com/datasets/nphantawee/pump-sensor-data** — 220,320 rows,
52 sensor channels, 1-minute cadence, seven breakdown episodes. Credit for the
underlying data belongs to the operations team who shared it. Kaggle declares no
licence for it, so treat it as research/study/development use. Full attribution
and provenance in `references/sample_data/README.md`.
