---
name: atai-rare-event-detection-agent
description: >
  Run Archetype AI's managed Rare Event Detection (RED) agent over the Agent
  API — upload a sensor CSV, resolve the pre-packaged "RED Quick Start"
  bundle by name (portable across dev/staging/prod; a fitted
  nearest-prototype classifier and its windowing are already pinned), run it
  (one agent per input file), poll status + audit events, and download
  per-window event predictions. Use this skill when the user wants
  fully-managed server-side detection of a *named* rare fault recurring —
  equipment breakdowns, process excursions — from a
  handful of labelled examples. Covers bundle resolution by name (?query=),
  the run/poll/results lifecycle, the output CSV schema
  (`finish_timestamp, start_timestamp, predicted_state, invalid,
  p_<class>…`), the Embeddings bundle variant (per-row Omega embeddings,
  ~2,000× larger output), bring-your-own-classifier via the `red` and
  `red-fitting` blueprints, and why incident-level detection must be scored
  separately from window-level accuracy. Do NOT use for classifying every
  operating regime from a full labelled library
  (`atai-operational-state-monitoring-agent`), for unnamed anomalies against
  normal-only data, for client-side embedding + KNN over `/query`
  (`atai-newton-omega-model`), or for cleaning and windowing raw CSVs
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

You don't have to fit or host anything: the platform ships **canonical "RED
Quick Start" bundles** with a pump-breakdown classifier and its windowing
(`window_size=64, step_size=1`) already pinned. One run = one agent instance =
one input file. You upload the CSV, **resolve the pre-packaged bundle by
name**, run it, poll until terminal, and download one output CSV of per-window
predictions. (Detecting *your own* fault catalog means fitting your own
classifier and creating your own bundle — see "Bring your own classifier"
below.)

## When to Apply

- Run managed rare-event detection **without fitting anything** — the
  pre-packaged Quick Start bundle pins a classifier and its windowing; upload
  a prepared CSV and run
- Detect a **named** fault of your own recurring, given only one or two
  labelled incidents of it plus normal-operation data (bring your own
  classifier, below)
- Deploy a detector as a repeatable batch job with **no client-side ML** — the
  platform embeds and classifies every window
- Score a few-shot detector honestly, where standard accuracy is meaningless
  because the positive class is under 1% of windows

> **Your own data?** As of today, this skill runs the **pre-packaged RED
> Quick Start bundles**, whose classifier is fit to the Kaggle pump-breakdown
> data. To detect faults in your own data, contact
> **support@archetypeai.dev** — Archetype AI will work with you to create
> agent bundles tailored to your data and fault catalog. (The "Bring your own
> classifier" section below documents the underlying mechanics.)

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

> **⚠️ The bundle API is plural everywhere** as of 2026-08-11:
> `GET /agents/bundles` (list/search), `GET /agents/bundles/{id}` (fetch),
> `POST /agents/bundles` (create), `POST /agents/bundles/{id}/run` (run). The
> singular forms (`POST /agents/bundle`, `POST /agents/bundle/{id}/run`,
> `GET /agents/bundle/{id}`) now return **404** — earlier docs (and the OSM
> sibling skill) describing a singular/plural split predate this migration.

> **⚠️ Dev-only for now.** Everything here is verified against the **Dev**
> deployment (`https://api.dev.u1.archetypeai.app`) — the pre-packaged Quick
> Start bundles, the canonical `red` blueprint, the `red-fitting` blueprint,
> the Agent API surface, and the runtime numbers below. If name resolution
> reports `no bundle named … found`, the pre-packaged bundle isn't published
> in that environment yet (Staging/Prod rollout pending) — point at Dev, or
> pass a known `--bundle-id` for that environment. Please contact
> support@archetypeai.dev.

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

### 2. Resolve the pre-packaged bundle by name

The Quick Start bundles are canonical (platform-published) and identified by a
**stable name**; their `bnd_…` **id differs per environment** (dev/staging/
prod), so resolve by name for portability:

```sh
curl -H "Authorization: Bearer $ATAI_API_KEY" \
  "$ATAI_API_ENDPOINT/agents/bundles?query=RED%20Quick%20Start"
```

`?query=` does a case-insensitive **substring** search over name and id
(`?name=` and `?search=` are silently ignored). Match the name **exactly**
client-side and prefer `is_canonical: true`. Two bundles are published:

| Name | What you get |
|------|--------------|
| `RED Quick Start (Pump Breakdown)` | Per-window predictions from a pump-breakdown classifier fit to two labelled Kaggle pump-sensor incidents |
| `RED Quick Start (Pump Breakdown, Embeddings)` | The same, plus the **Newton Omega encoder embedding for each window** — one `embedding_{variate}` column per sensor channel, each a 768-d vector (the same embeddings `atai-newton-omega-model` gets from `/query`, here computed server-side as part of the run; `output_embeddings: true`) |

Both pin the classifier artifact (`red-classifier` slot) and its windowing
(`window_size=64, step_size=1`), so there is **nothing to create and no
classifier URI to supply**. (For reference, on Dev these currently resolve to
`bnd_47c7pesmwx8yct495bwtm9f05z` and `bnd_01cr02sex781592xa2xhcvby8z` — pin ids
only as a last resort, since they differ in staging/prod.)

### 3. Run the bundle — one agent per input file

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundles/$BUNDLE_ID/run" -d '{
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

Output is one row per window. The row carries the window's **span** —
`finish_timestamp` (the window end, the value to join ground truth on) and
`start_timestamp`:

```
finish_timestamp,start_timestamp,predicted_state,invalid,p_normal,p_pump_breakdown
1526966220.0,1526962440.0,normal,false,1,0
```

(Older blueprint versions emitted a single `timestamp` column — the window
end. The runner's scorer accepts either.) `predicted_state` is the class name
or `INVALID_STATE`; `p_<class>` columns are per-class probabilities when
`output_probabilities` is on (the default).

**The Embeddings bundle adds the Newton Omega embedding for each window** —
one `embedding_{variate}` column per sensor channel, each a 768-d vector —
and the size cost is dramatic: on the same 537-window slice, **45.5 MB vs
23 KB** (~85 KB/row with 10 channels, ~2,000× the plain output). Use it when
you want the vectors alongside the predictions — client-side similarity,
drift monitoring, projections, or downstream ML per
[`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)'s patterns —
without paying one `/query` call per window.

## Bring your own classifier (advanced)

The pre-packaged bundles run Archetype AI's pump-breakdown classifier. To
detect **your own** fault catalog, fit your own classifier (below) and create
your own bundle from the canonical `red` blueprint, then run it as in Step 3:

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundles" -d '{
    "blueprint": "red",
    "name": "my fault detector",
    "values": {"step_size": 1},
    "artifacts": {"red-classifier": "s3://bucket/path/fit-classifier.safetensors"}
  }'
```

**The artifact key is `red-classifier`.** That string is the model name the
`red` blueprint declares (`models.classifier: "red-classifier"`), so it is what
the `artifacts` map must be keyed by.

> ⚠️ Until **2026-07-28** this key was `rad-classifier`, a typo since fixed. The
> old key still returns **HTTP 201 `status: ready`** at bundle creation and only
> fails ~30 s into the run with `repeated failures polling JOS job` — with no
> mention of the artifact. Bundles created before the fix must be recreated.
> `red-fitting`'s `output` value still defaults to the old spelling, which is
> harmless only because the runner ignores it (below).

`step_size` is normally the only value worth setting. Everything else —
`window_size`, `data_columns`, `timestamp_column`, `encoder_model` — is
inherited from the classifier's own `parameters` metadata via
`${models.classifier.parameters.window_size:1024}`, which is why the payload is
so small. Omit `step_size` and you inherit the stride the classifier was fit
with.

### Fitting the classifier — the `red-fitting` blueprint

RED's classifier is fitted by a second blueprint whose sink uses
**`strategy: centroid`** — one mean prototype per class, which is the design's
nearest-prototype head. The kNN knobs (`k`, `weights`) are absent by design:
with one candidate per class there is nothing to vote on.

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundles" -d '{
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
- **The fitted artifact is always `fit-classifier.safetensors`.** The `output`
  value is accepted and then ignored, so no artifact is ever named what you asked
  for. (The blueprint's own default is the stale `rad-classifier.safetensors`,
  which is harmless precisely because it never reaches disk.) Read the real
  filename off `/results`; building an `s3://` URI from the requested name points
  at nothing, and the inference run then fails at artifact load with
  `repeated failures polling JOS job` — naming the poller, not the missing file.
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
  concurrency costs is feedback, not throughput. Contention is also **highly
  variable**: two concurrent 537-window Quick Start runs on a busy Dev took
  ~2.5 h wall-clock (~0.1 win/s effective) against the ~2.2 win/s solo rate —
  budget generously and treat the events stream, not the clock, as the signal.
- **Cancel with `POST /agents/instances/{id}/cancel`.** Killing a local client
  does not stop the job — `DELETE` returns 409 while running.

## References

- `references/run_red_agent.py` — stdlib-only end-to-end runner: upload →
  resolve the Quick Start bundle by name → run → poll → download → score, with
  the three scoring views above. `--embeddings` switches to the Embeddings
  bundle; `--bundle-name`/`--bundle-id` run any other bundle.
- `references/.env.example` — the two required environment variables.
- `references/sample_data/` — prepared pump slices: three single-class shot
  files for fitting and a held-out slice with a ground-truth sidecar for running
  and scoring. See its README for full data attribution.

The full seven-stage build behind the pre-packaged classifier (raw CSV →
preflight → prep → grid search → fit → run → evaluate, including the
channel-leakage audits this data requires) is Archetype AI-internal — for a
detector fitted and packaged for your own data, contact
support@archetypeai.dev.

## Data attribution

The sample data derives from the **`pump_sensor_data`** dataset published by the
Kaggle user [`nphantawee`](https://www.kaggle.com/nphantawee):
**https://www.kaggle.com/datasets/nphantawee/pump-sensor-data** — 220,320 rows,
52 sensor channels, 1-minute cadence, seven breakdown episodes. Credit for the
underlying data belongs to the operations team who shared it. Kaggle declares no
licence for it, so treat it as research/study/development use. Full attribution
and provenance in `references/sample_data/README.md`.
