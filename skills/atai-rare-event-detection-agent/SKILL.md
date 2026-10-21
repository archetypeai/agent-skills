---
name: atai-rare-event-detection-agent
description: >
  Run Archetype AI's Rare Event Detection (RED) agent over the Agents API, on two
  paths. Path 1 — run the maintained "RED Quick Start" bundle (a pump-breakdown
  nearest-prototype classifier and its windowing already pinned): upload a
  prepared CSV, resolve the bundle by exact name, run, poll, download per-window
  event predictions. Path 2 — build a RED agent on YOUR OWN data: from role files
  (a few labelled incidents of a named event for training, the next ones for
  validation and test, a delivery) fit the class prototypes and choose the window
  and distance with the Optimizations API, promote the best trial to a blueprint,
  test it once with the Evals API, deliver it as a bundle, and score each incident
  (detected, how soon) and the false alarms. Use when the user wants server-side
  detection of a *named* rare event recurring — equipment breakdowns, process
  excursions — from a handful of labelled examples. Covers the output schema,
  the Embeddings variant, what the `red` blueprint exposes since its 2026-10-12
  re-seed, bringing your own classifier, and incident-level vs window-level
  scoring. Do NOT use for classifying every operating regime
  (`atai-operational-state-monitoring-agent`), for unnamed anomalies with no
  labelled examples (`atai-anomaly-discovery-agent`), for client-side embedding
  + KNN over `/query` (`atai-newton-omega-model`), or for preparing a labelled
  recording (`atai-rare-event-detection-agent-data-prep`).
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

There are two ways in. **Path 1** runs the canonical "RED Quick Start" bundles,
whose pump-breakdown classifier and windowing (`window_size=64, step_size=1`) are
already pinned: nothing to fit or host. **Path 2** builds a RED agent on your own
data with the Optimizations and Evals APIs: the platform fits the prototypes on
your labelled incidents, you promote the best setting to a blueprint, test it once
and deliver it. Either way, one run = one agent instance = one input file, and the
output is one CSV of per-window predictions.

## When to Apply

- Run managed rare-event detection **without fitting anything** — the
  pre-packaged Quick Start bundle pins a classifier and its windowing; upload
  a prepared CSV and run (Path 1)
- Detect a **named** fault of your own recurring, given only one or two
  labelled incidents of it plus normal-operation data (Path 2)
- Deploy a detector as a repeatable batch job with **no client-side ML** — the
  platform embeds and classifies every window
- Score a few-shot detector honestly, where standard accuracy is meaningless
  because the positive class is under 1% of windows


**Do not use this skill when:**
- The user has a full labelled library across all regimes and wants "which state
  is the asset in?" — use
  [`atai-operational-state-monitoring-agent`](../atai-operational-state-monitoring-agent/SKILL.md)
- The faults are unnamed or uncharacterized and only normal data exists — RED
  needs a named catalog; flagging everything unusual is
  [`atai-anomaly-discovery-agent`](../atai-anomaly-discovery-agent/SKILL.md)
- You want per-window embeddings to do ML client-side — use
  [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)
- Your recording still needs gridding, a label mapping, channel triage or role
  files — use [`atai-rare-event-detection-agent-data-prep`](../atai-rare-event-detection-agent-data-prep/SKILL.md)
  first; for generic cleaning of messy raw data,
  [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md).
  RED assumes prepared, z-scored input with a regular sampling rate

## Choose a path

| | **Path 1 — run a maintained bundle** | **Path 2 — build a RED agent on your data** |
|---|---|---|
| you have | a prepared CSV (the pump sample, or data like it) | a labelled recording of **your** asset with a few incidents of a named event |
| you get | per-window predictions from a classifier fitted on **two pump breakdowns** | your own blueprint and bundle: prototypes fitted on your incidents, window and distance chosen on the next ones, a test score and a delivery score |
| APIs | files, bundles, instances | files, **Optimizations**, **promote**, **Evals**, bundles, instances |
| scripts | [`references/run_red_agent.py`](references/run_red_agent.py), on the official `archetypeai` client | [`references/red_lifecycle/`](references/red_lifecycle/), a small stdlib HTTP helper (the client doesn't cover Optimizations / Evals / promote yet) |
| time | ~1.5 min on a clear queue | ~6 min on the sample; longer with load |
| status | verified on production; re-verified on dev 2026-10-13 | **verified on dev, 2026-10-13** (RED became optimizable there on 2026-10-12) |

The Quick Start classifier holds one pump's two breakdowns, so running your own asset
through it is a transfer test, not a deployment: for your own asset, take Path 2. Path
2's last step *is* Path 1 run on your own bundle, so the output schema, polling and
scoring below apply to both. Worked example:
[rare-event-detection-agent-example-pump-quickstart](https://github.com/archetypeai/rare-event-detection-agent-example-pump-quickstart)
(the pump, about 10 minutes end to end).

---

# Path 1 — Run a maintained bundle

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
> on the production deployment (`https://platform.u1.archetypeai.app`) — set
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
(`?name=` and `?search=` are silently ignored), and results come back
newest-first. Match the name **exactly** client-side and prefer
`is_canonical: true` — measured on Prod:

```
query 'RED Quick Start'                    -> 2   Embeddings first
query 'RED Quick Start (Pump Breakdown)'   -> 1   the closing paren excludes it
```

Taking `data[0]` on a prefix runs the Embeddings bundle: 740 MB where you
expected 381 KB. Two bundles are published:

| Name | What you get |
|------|--------------|
| `RED Quick Start (Pump Breakdown)` | Per-window predictions from a pump-breakdown classifier fit to two labelled Kaggle pump-sensor incidents |
| `RED Quick Start (Pump Breakdown, Embeddings)` | The same, plus the **Newton Omega encoder embedding for each window** — one `embedding_{variate}` column per sensor channel, each a 768-d vector (the same embeddings `atai-newton-omega-model` gets from `/query`, here computed server-side as part of the run; `output_embeddings: true`). **The output file gets dramatically larger**: 740 MB vs 381 KB on the sample slice, ~1,900× |

Both pin the classifier artifact (`red-classifier` slot) and its windowing
(`window_size=64, step_size=1`), so there is **nothing to create and no
classifier URI to supply**. They are pinned to the `red` blueprint version they were
created on (July 2026), which reads its window, metric and columns from the classifier;
the 2026-10-12 re-seed made a new active version and left theirs in place, so they run
exactly as before (re-verified on dev 2026-10-13: byte-identical output). (For reference, these currently resolve to
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

**A `completed` run can still be writing its output** (a known platform issue): a download
within seconds can return a partial CSV, its rows cut off before the input's end. The runner
counts the output complete only once its last window's `finish_timestamp` reaches the input's
last timestamp (within one window span and a minute), downloading again every 15 s until it
does, for up to 10 min.

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

## Bring your own classifier (legacy route)

**Prefer Path 2,** which fits the classifier on the platform and promotes it to a
blueprint that needs no artifact path. This route is for a classifier you already hold as
an `s3://` safetensors file (from the pre-Optimize `red-fitting` pipeline, or one Archetype
AI fitted with you). On the current `red` blueprint (since the 2026-10-12 re-seed) the
bundle has to fill the head's slot and pin everything the classifier was fitted with:

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundles" -d '{
    "blueprint": "red",
    "name": "my fault detector",
    "values": {"step_size": 1, "window_size": 64, "metric": "l2",
               "events": ["normal", "pump_breakdown"], "background": "normal",
               "timestamp_column": "timestamp", "data_columns": ["sensor_02", "...", "sensor_42"]},
    "models": {"classification_head": "red-classifier"},
    "artifacts": {"red-classifier": "s3://bucket/path/fit-classifier.safetensors"}
  }'
```

- **`models.classification_head`** — the head's model slot is empty on the current
  blueprint (an optimization fills it). Pointing it at a key of the bundle's `artifacts`
  binds your file. Without it, the run is
  `400 models.classification_head: this blueprint has no model for classification_head yet`.
- **`window_size`, `metric`, the columns** — what the classifier was fitted with. The
  current blueprint reads nothing from the classifier's metadata and falls back to window
  1,024 and `l1`, which would silently change every prediction. `events` and
  `background` are required.
- Verified on dev 2026-10-13: the Quick Start's own classifier, through this body on the
  current blueprint, gives an output **byte-identical** to the Quick Start bundle's on the
  sample slice (`references/red-output.csv`).

On the July blueprint version (the Quick Start bundles' own), `"artifacts":
{"red-classifier": …}` and `step_size` were enough: window, columns and encoder came from
the classifier's `parameters` metadata. The artifact key was `rad-classifier` until a
typo fix on 2026-07-28; the old key still returns HTTP 201 at bundle creation and fails
~30 s into the run with `repeated failures polling JOS job`.

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

Why all three: a slice can score **0.9947 accuracy while missing its incident
entirely**. With 42 fault rows against 8,192 normal ones, predicting `normal`
everywhere is 99.4% correct and 100% useless. Nothing in the window-level table
catches that.

**The platform's own score labels windows differently.** Optimize and Evals score each
row once: a window answers only for the rows since the previous window ended, so at step
1 its label is its last row's. Path 2's scripts report both; on the pump the two event F1s
differ by about 0.005.

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
- **Canonical blueprints are versioned, and a bundle keeps its version.** A re-seed makes
  a new active version (`red`: 2026-10-12) and leaves the old one inactive; bundles created
  before it keep running on the old one, and `"blueprint": "red"` in a new request means
  the active one.
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

## Local Setup (Path 1)

```bash
cd skills/atai-rare-event-detection-agent/references

# One dependency: the official Archetype AI client. Note the -r.
pip install -r requirements.txt

# Create the .env IN THIS DIRECTORY — the runner reads ./.env from where it
# runs (the file is gitignored). BOTH variables required, no default endpoint.
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://platform.u1.archetypeai.app
EOF

python3 run_red_agent.py                       # bundled sample slice
python3 run_red_agent.py --embeddings          # + Omega embedding per window
python3 run_red_agent.py --csv my_slice.csv    # your own prepared CSV
python3 run_red_agent.py --score-only out.csv  # re-score a downloaded output
```

Either endpoint form works — the runner reduces it to a root and mounts each API
surface where the platform serves it: files under `/v0.5/files`, the agents API
versionless at `/agents`. No single `api_endpoint` serves both, so the runner
builds one client per surface; handing the client a single endpoint 404s half the
calls and reports a bodyless `ApiError: {}`. The model skills in this repo ship
`ATAI_API_ENDPOINT` with `/v0.5`; one `.env` serves both families.

## References

- `references/run_red_agent.py` — end-to-end runner on the official [`archetypeai` client](https://github.com/archetypeai/python-client): upload →
  resolve the Quick Start bundle by name → run → poll → download → score, with
  the three scoring views above. `--embeddings` switches to the Embeddings
  bundle; `--bundle-name`/`--bundle-id` run any other bundle.
- `references/.env.example` — the two required environment variables.
- `references/requirements.txt` — the one dependency, the official `archetypeai` client.
- `references/sample_data/` — a prepared, held-out pump slice with a
  ground-truth sidecar for running and scoring. See its README for full data
  attribution.

For a classifier fitted on your own data, take Path 2 below; the
[pump quickstart](https://github.com/archetypeai/rare-event-detection-agent-example-pump-quickstart)
runs the whole build, including the channel-leakage audits this data requires.

---

# Path 2 — Build a RED agent on your own data

> **Status: verified end to end on dev (2026-10-13)**, where the `red` blueprint became
> optimizable on 2026-10-12; check another deployment before relying on it there. Built
> on the Kaggle pump:
> [rare-event-detection-agent-example-pump-quickstart](https://github.com/archetypeai/rare-event-detection-agent-example-pump-quickstart)
> (the whole lifecycle in about 10 minutes, reproduced byte for byte by a second run from
> scratch). The scripts use a small stdlib HTTP helper
> ([`atai_http.py`](references/red_lifecycle/atai_http.py), shared with the OSM and AD
> skills' Path 2) because the official client doesn't cover these APIs yet.

```
role files ──► 1 Optimize ──► 2 promote ──► 2 test once (Evals) ──► 3 deliver (bundle + runs) ──► 4 score
  (prep skill)   prototypes from   best trial    one-shot, with           a prediction per row        incidents caught,
                 training, scored  → blueprint   predictions checked                                  delay, false alarms
                 on validation
```

Measured platform behaviour, with the numbers: [`references/platform-notes.md`](references/platform-notes.md).

## Step 0 — Role files

Path 2 starts from **role files**, built and checked by
[`atai-rare-event-detection-agent-data-prep`](../atai-rare-event-detection-agent-data-prep/SKILL.md)
to its contract, [`role-files.md`](../atai-rare-event-detection-agent-data-prep/references/role-files.md):

| role | what | used in |
|---|---|---|
| `training/` | normal operation, and one file per training incident: each one class, no label column | every trial: one prototype per class |
| `validation/` | one file per incident with normal context, `label` column | scoring each trial |
| `test/` | the same, later incidents | scored **once**, after promotion |
| `delivery/` + `labels/` | test incidents again, or new operation after them, **without** `label`; labels held back | the deployed run, scored afterwards |

plus `manifest.json` (classes, channels, z-score statistics, every file). Build the prep
skill's sample (the pump's first four breakdowns) in about 2 s:

```sh
cd skills/atai-rare-event-detection-agent-data-prep/references
pip install -r requirements.txt
python prepare.py --input sample_data/pump_sample.parquet --label-col machine_status \
    --classes normal=NORMAL pump_breakdown=BROKEN,RECOVERING --out prepared
python check_events.py --prepared prepared --training 1,2 --validation 3 --test 4
python build_roles.py --prepared prepared --out roles --training 1,2 --validation 3 --test 4 --delivery 4
python preflight_roles.py --roles roles          # all checks pass
```

## Step 1 — Optimize: prototypes from training, scored on validation

```sh
cd skills/atai-rare-event-detection-agent/references/red_lifecycle
pip install -r requirements.txt
# ATAI_API_KEY + ATAI_API_ENDPOINT: the nearest .env from here upwards is used
R=../../../atai-rare-event-detection-agent-data-prep/references/roles
python optimize.py --roles $R --dry-run                                     # the plan: no uploads, no jobs
python optimize.py --roles $R --out out --name red-sample --background      # tail -f out/optimize.log
```

`POST /agents/optimizations` with the `red` blueprint's id, each training file as a
training example labelled by its class (`{"from": {"constant": "pump_breakdown"}}`), the
validation files as validation examples (`{"from": {"column": "label"}}`),
`"primary": "event.foreground_macro_f1"`, a `budget.max_trials`, `overrides.values` with
the columns, `events` (`["normal", "pump_breakdown"]`), `background` (`"normal"`) and
every single-valued option, and a search space:

```json
{"parameters": {"metric": {"kind": "value", "spec": {"type": "categorical", "values": ["cosine", "l1", "l2", "dot"]}}}}
```

with `window_size` 64 and `step_size` 1 pinned.

- **`metric` is the classifier's only fit setting** (the distance to the prototypes; the
  blueprint defaults to `l1`). There is no `k` and no threshold: one prototype per class,
  the mean of its training windows.
- **Pin the window and step.** The blueprint's defaults are 1,024 and 256. Choose the
  window so the shortest event you must catch fills at least half of it (the prep skill's
  `check_events.py` shows which incidents can, per window).
- **`events` and `background` are required,** and every training class must be in `events`.
- **Search more:** `--windows 16 32 64` (a list) or `--windows 16:84` (a range),
  `--steps`, `--metrics`. Trials that share window and step share one encoder pass, so
  metrics are nearly free and each window costs a pass. A grid of up to 12 points runs in
  full; a bigger one or a range gets a 12-trial random search (`--max-trials` to change).
  On the pump, a 24-trial random search over windows 16–84 took six times as long as the
  12-point grid and did not beat it.

**Expect, on the sample** (4 trials, 2.5 min on dev):

```
otr_…  completed {"metric": "dot"}     event F1 0.8677  macro-F1 0.9215  recall 0.924  precision 0.818  windows 8,925
otr_…  completed {"metric": "cosine"}  event F1 0.8653  macro-F1 0.9201  recall 0.925  precision 0.813  windows 8,925
otr_…  completed {"metric": "l2"}      event F1 0.8625  macro-F1 0.9184  recall 0.925  precision 0.808  windows 8,925
otr_…  completed {"metric": "l1"}      event F1 0.8539  macro-F1 0.9131  recall 0.924  precision 0.794  windows 8,925
```

`dot` wins on validation (breakdown #3), as on the full pump data; `l1`, the blueprint's
default, is last.

## Step 2 — Promote, then test once

```sh
python promote_and_test.py --roles $R --out out --name red-sample
```

1. **Promote** the best trial across `optimize.py`'s runs (a tie goes to the optimization
   that tried more points; `--trial otr_…` picks one): the blueprint pins the trial's
   values, `events` and `background`, and carries the prototypes as a registry artifact
   (`aar_…`).
2. **Test once** with the Evals API on every test file, with predictions. An eval takes no
   value overrides: it runs at the trial's window and step.
3. **Check** every prediction's gold label against the file's `label` column (the
   platform's rule: each row scored once, so at step 1 a window's last row) and the
   offline confusion matrix against the platform's, then score each incident as the
   design does: each window by the majority of its rows, detected or missed, how soon,
   and false alarms on windows with no event row.

**Expect, on the sample** (2 min on dev):

```
test, 1 files (window 64, step 1, dot; 0 abstained)
  platform (each row once): event F1 0.9493  macro-F1 0.9726  recall 0.9736  precision 0.9262
  design rule (each window by majority): event F1 0.9727  precision 0.9498  recall 0.9967
  false alarms 10 / 7,550 windows with no event row (0.13%)
    inc04.csv      pump_breakdown    606 rows  +16 rows     false alarms   10 / 7,550
```

Breakdown #4, never seen, caught 16 minutes after onset, from two labelled breakdowns.

Once you have seen the test number, the setting is frozen: tuning it further would make
the test number optimistic.

## Step 3 — Deliver

```sh
python deliver.py --roles $R --out out --name red-sample
```

A bundle on the promoted blueprint adding only `step_size: 1` (one prediction per row;
the prototypes, window, metric, `events` and `background` come with the blueprint), then
one agent run per delivery file, `--parallel 3` at a time. Each output is downloaded again
until its last window reaches the input's end (a run can report `completed` while its
output is still being written).

## Step 4 — Score the delivery

```sh
python score.py --roles $R --out out
```

Offline, against `labels/`: per file, event precision and recall (each window by the
majority of its rows), each incident detected or missed and the delay in rows, false
alarms on windows with no event row (and, on a tail file of new operation, per day); then
pooled. **Expect, on the sample** (1 min of agent run on dev):

```
delivery/inc04.csv: 2,591 windows scored (0 invalid)
  event  precision 0.9498  recall 0.9967  F1 0.9727   tp 605 fp 32 fn 2 tn 1952
  false alarms 10 / 1,922 = 0.0052
  pump_breakdown incident, 606 rows: detected, +16 rows after onset
```

The agent run and the eval agree: the same detection minute and the same 10 false
alarms, all within 1,024 rows of the breakdown.

## Path 2 pitfalls

- **A window of 1,024.** The `red` blueprint's default; an incident shorter than half a
  window is all but undetectable, and a 945-row training incident yields no window. Pin it.
- **The platform's F1 and the design rule's F1 differ:** each row once (at step 1, a
  window's last row) against each window by the majority of its rows. Report which.
- **The agent recognises the event's state.** On the pump, `RECOVERING` rows make up the
  class: it detects "the pump is down", not the failure. Choose the label mapping in prep
  for what an operator needs.
- **Short incidents get missed with no false alarm to show for it:** on the pump, a
  42-minute breakdown at window 64 was missed by every configuration tried. Score incidents,
  not just windows.
- **Don't trust `completed` for the output, or `failed` for the run**: `deliver.py`
  downloads until the output reaches the input's end, and a run reported failed can still
  have one.
- **A key exported in your shell beats `.env`**; keys are per deployment. The helper checks
  the key with one GET first and says where it came from (showing only its last 4
  characters).

## Cleanup (both paths)

The pre-packaged Quick Start bundles are canonical and shared — **do not delete them**.
Delete your own agent instances, and any bundle you created:

| object | delete | notes |
|---|---|---|
| agent instances (`agt_…`) | ✅ `DELETE /agents/instances/{id}` | cancel a running one first (`POST …/cancel`) |
| your bundles (`bnd_…`) | ✅ `DELETE /agents/bundles/{id}` | delete the instances first |
| promoted blueprints (`blp_…`) | ❌ 403 for a regular key | ask an org admin; use a distinctive `--name` |
| optimizations (`opt_…`), evals (`evl_…`) | — | records of what was fitted and scored; keep them if your docs cite them |

## File Layout

```
skills/atai-rare-event-detection-agent/
├── SKILL.md                  ← this file
├── references/
│   ├── run_red_agent.py      ← Path 1: the managed flow on the official client (upload → resolve → run → poll → download → score)
│   ├── requirements.txt      ← Path 1: archetypeai
│   ├── .env.example          ← copy to .env and fill in (both paths)
│   ├── red-output.csv        ← Path 1: the Quick Start bundle's output on the sample slice
│   ├── sample_data/          ← Path 1: a prepared, held-out pump slice, its labels sidecar, attribution
│   ├── platform-notes.md     ← Path 2: measured platform behaviour
│   └── red_lifecycle/        ← Path 2 (stdlib HTTP until the client covers these APIs)
│       ├── atai_http.py        requests with retries, uploads with a content-keyed cache, paging, polling
│       ├── common.py           the role-file manifest, examples, pinned values
│       ├── background.py       --background: nohup (+ caffeinate on macOS), output to a log
│       ├── optimize.py         Step 1: search space → Optimizations API → trials
│       ├── promote_and_test.py Step 2: best trial → blueprint → one eval, predictions checked
│       ├── deliver.py          Step 3: bundle from the blueprint → one run per file → complete outputs
│       ├── score.py            Step 4: incidents, delay, false alarms (and the shared scorer)
│       └── requirements.txt    numpy, pandas
└── tests/
    ├── test_references.py    ← Path 1: network-free unit tests
    └── test_lifecycle.py     ← Path 2: network-free unit tests
```

Run both: `python -m pytest skills/atai-rare-event-detection-agent/tests`.

## Data attribution

The sample data derives from the **`pump_sensor_data`** dataset published by the
Kaggle user [`nphantawee`](https://www.kaggle.com/nphantawee):
**https://www.kaggle.com/datasets/nphantawee/pump-sensor-data** — 220,320 rows,
52 sensor channels, 1-minute cadence, seven breakdown episodes. Credit for the
underlying data belongs to the operations team who shared it. Kaggle declares no
licence for it, so treat it as research/study/development use. Full attribution
and provenance in `references/sample_data/README.md`.
