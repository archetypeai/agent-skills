---
name: atai-rare-event-detection-agent
description: >
  Run Archetype AI's managed Rare Event Detection (RED) agent over the Agent
  API — upload a sensor CSV, resolve the pre-packaged "RED Quick Start"
  bundle by name (its nearest-prototype classifier and windowing are already
  pinned), run it, poll status + audit events, download per-window event
  predictions. Use when the user wants fully-managed, server-side detection
  of a *named* rare fault recurring — equipment breakdowns, process
  excursions — from a handful of labelled examples. Covers name-based bundle
  resolution, the run/poll/results lifecycle, the output schema, the
  Embeddings variant, bring-your-own-classifier via the `red` blueprint, and
  incident-level vs window-level scoring. Do NOT use for classifying every
  operating regime (`atai-operational-state-monitoring-agent`), for unnamed
  anomalies with no labelled examples, for client-side embedding + KNN over
  `/query` (`atai-newton-omega-model`), or for cleaning and windowing raw
  CSVs (`atai-newton-omega-model-data-prep`).
---

# RED Agent — Managed Rare Event Detection via the Agents API

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
predictions. (Detecting *your own* fault catalog means a classifier fitted
for your data — Archetype AI does that with you; see "Bring your own
classifier" below.)

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
Agents API   {ATAI_API_ENDPOINT}/agents/...                       (versionless!)
Authorization: Bearer <API_KEY> on every call
```

The Agents API is **versionless** — `/agents`, not `/v0.5/agents`. If
`ATAI_API_ENDPOINT` carries a `/vX.Y` suffix, strip it before appending
`/agents`. Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required; there is no
default endpoint.

> **⚠️ The bundle API is plural everywhere** as of 2026-08-11:
> `GET /agents/bundles` (list/search), `GET /agents/bundles/{id}` (fetch),
> `POST /agents/bundles` (create), `POST /agents/bundles/{id}/run` (run). The
> singular forms (`POST /agents/bundle`, `POST /agents/bundle/{id}/run`,
> `GET /agents/bundle/{id}`) now return **404** — earlier docs (and the OSM
> sibling skill) describing a singular/plural split predate this migration.

> **Availability.** The pre-packaged "RED Quick Start" bundles are published
> on the production deployment (`https://api.u1.archetypeai.app`) — set
> `ATAI_API_ENDPOINT` to it and the full upload → run → score cycle works as
> documented here. If name resolution reports `no bundle named … found`, the
> bundle isn't published in the deployment you're pointed at: resolving by
> name is portable, so pass a known `--bundle-id` meanwhile, or contact
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
**stable name**; their `bnd_…` **id is deployment-specific**, so resolve by
name for portability:

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
| `RED Quick Start (Pump Breakdown, Embeddings)` | The same, plus the **Newton Omega encoder embedding for each window** — one `embedding_{variate}` column per sensor channel, each a 768-d vector (the same embeddings `atai-newton-omega-model` gets from `/query`, here computed server-side as part of the run; `output_embeddings: true`). **The output file gets dramatically larger**: 740 MB vs 381 KB on the sample slice, ~1,900× |

Both pin the classifier artifact (`red-classifier` slot) and its windowing
(`window_size=64, step_size=1`), so there is **nothing to create and no
classifier URI to supply**. (For reference, these currently resolve to
`bnd_0ykawrhhd795kv20cvr18ak618` and `bnd_7xye786cph98xb8ch2yn2q56ey` in
production; the ids are deployment-specific, which is why you resolve by
name — pin ids only as a last resort.)

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

Like the other list endpoints, `/results` **pages**: `data`, `has_more`,
`next_cursor`, with `limit` (default 100, max 1000) and `after`/`before`
cursors. **The cursor is opaque** — pass `next_cursor` back verbatim and never
derive it from `data[last].id`; a fabricated value is rejected with
`400 invalid cursor`. One run through a quick-start bundle emits one output, so
the first page is the whole answer — a bundle with several sink ports is where
paging starts to matter.

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
and the size cost is dramatic: **740 MB vs 381 KB** on the full 8,735-window
sample slice (~85 KB/row with 10 channels, ~2,000× the plain output —
verified, with predictions identical to the plain bundle's). Use it when
you want the vectors alongside the predictions — client-side similarity,
drift monitoring, projections, or downstream ML per
[`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)'s patterns —
without paying one `/query` call per window.

## Bring your own classifier (advanced)

The pre-packaged bundles run Archetype AI's pump-breakdown classifier. To
detect **your own** fault catalog you need a classifier fitted on your data.
The fitting pipeline is not accessible to external users yet — contact
support@archetypeai.dev and Archetype AI will fit one with you and hand back
the classifier artifact. With that artifact you create your own bundle from
the canonical `red` blueprint, then run it as in Step 3:

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

`step_size` is normally the only value worth setting. Everything else —
`window_size`, `data_columns`, `timestamp_column`, `encoder_model` — is
inherited from the classifier's own `parameters` metadata via
`${models.classifier.parameters.window_size:1024}`, which is why the payload is
so small. Omit `step_size` and you inherit the stride the classifier was fit
with.

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

A verified quick-start run on the shipped slice (8,735 windows) shows the
three views in action: accuracy **0.9883**, `pump_breakdown` precision 0.8602
/ recall 0.9934 / F1 0.9220, false-alarm rate **0.0076**, and the incident
**DETECTED 21 samples after onset** — one run, three different-looking
numbers, each answering a different operational question.

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

- **Bundle artifacts must be `s3://` URIs** (bring-your-own-classifier path
  only). Pass the URI Archetype AI hands you **verbatim**: the platform
  resolves artifact strings as S3/filesystem paths only, so a platform
  `file_id` or an `https://` URL is accepted at bundle creation and then
  fails at run time with ENOENT, without attempting a fetch. There is no
  upload route either — the files API rejects `.safetensors`.
- **All-`INVALID_STATE` output means input validation, not a model problem.**
  The blueprint defaults to `validate_monotonic_timestamps: true` and
  `sample_rate_interval_tolerance: 0.05`. A common cause is a timestamp bug in
  prep: with pandas 2.x+, `astype("int64") // 10**9` divides by 1000× too much
  when parsing yields microsecond resolution, collapsing a 1-minute cadence so
  every timestamp is equal. Use `astype("datetime64[s]").astype("int64")`.
- **Short NaN runs are repaired server-side** (unlike the OSM sibling's
  graph): gaps up to `window_interpolation_max_gap` samples (default 16) are
  interpolated inside the run, so your prep only needs to handle longer
  outages.
- **Output timestamps are floats.** The platform emits `1530962520.0` where the
  input carried integer seconds — join on the numeric value, not the string.
- **Runs are reproducible.** The same input through the same-named bundle
  produces a **byte-identical output** run to run and across deployments
  (`cmp`-verified on both the 381 KB base output and the 740 MB embeddings
  output). The model is not re-fit per run.
- **A `failed` status can hide a successful run.** We have seen
  `repeated failures polling JOS job` on a job that completed. Always check
  `/results` before re-running.
- **Runtime is dominated by worker contention, not window count.** With a
  clear queue, the full 8,735-window sample slice completes end-to-end in
  **84–109 s** (~100 win/s; both bundle variants verified, the Embeddings
  runs including their 740 MB downloads). Under load, the same platform has run
  at ~2.2 win/s — a 12,059-window run once took ~90 min, and two contended
  537-window runs took ~2.5 h. Other tenants' jobs aren't visible to you, so
  those historical per-window-count timings are contention artifacts, not
  intrinsic rates. Treat the **audit events, not the clock**, as the signal.
- **Prefer sequential runs.** Whether concurrent runs queue depends on what
  else is running on the deployment at that moment: they queue when other
  workloads hold the workers, and run as concurrent jobs when they don't.
  Under load, five concurrent runs ran ~5× slower each; with a clear queue,
  concurrent runs completed at full speed. Other tenants' workloads aren't
  visible to you, so there is no serialization to rely on and no parallelism
  to count on — sequential stays the predictable default.
- **Cancel with `POST /agents/instances/{id}/cancel`.** Killing a local client
  does not stop the job — `DELETE` returns 409 while running.

## References

- `references/run_red_agent.py` — stdlib-only end-to-end runner: upload →
  resolve the Quick Start bundle by name → run → poll → download → score, with
  the three scoring views above. `--embeddings` switches to the Embeddings
  bundle; `--bundle-name`/`--bundle-id` run any other bundle.
- `references/.env.example` — the two required environment variables.
- `references/sample_data/` — a prepared, held-out pump slice with a
  ground-truth sidecar for running and scoring. See its README for full data
  attribution.

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
