---
name: atai-operational-state-monitoring-agent
description: >
  Run Archetype AI's managed Operational State Monitoring (OSM) agent over
  the Agent API — upload a sensor CSV, resolve a maintained pre-packaged
  "OSM Quick Start" bundle (classifier + windowing already pinned), run it
  (one agent per input file), poll status + audit events, and download the
  per-window state predictions. Use this skill when the user wants
  fully-managed, server-side classification of operational states over a CSV
  of sensor records — drilling states, machine modes, process phases —
  without fitting or hosting a classifier themselves. Covers resolving the
  pre-packaged bundle by name (portable across dev/staging/prod), the
  run/poll/results lifecycle, the output CSV schema (`finish_timestamp,
  predicted_state, invalid, p_<state>…`), and scoring against a ground-truth
  sidecar. Do NOT use for client-side embedding + KNN over `/query` (that's
  `atai-newton-omega-model`), for cleaning / windowing raw CSVs
  (`atai-newton-omega-model-data-prep`), or for fitting a classifier
  artifact yourself (see the OSM example repo).
---

# OSM Agent — Managed State Classification via the Agent API

The OSM agent is the **fully-managed counterpart** to the client-side embed-and-KNN pattern in [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md): instead of fanning out `/query` embedding calls and classifying locally, you hand the platform a CSV and the platform runs the whole graph server-side:

```
source → interpolate → window → windowInterpolate → samplingRate → limitValues → encoder → classifier → sink
```

You don't build or host anything: the platform ships **canonical "OSM Quick Start" bundles** with the six-state Volve classifier and its windowing already pinned. One run = one agent instance = one input file. You upload the CSV, **resolve the pre-packaged bundle by name**, run it, poll until terminal, and download one output CSV of per-window predictions.

## When to Apply

- Classify operational states over a full CSV of sensor records with **no client-side ML and no classifier of your own** — the maintained bundle embeds and classifies every window
- Demo or evaluate the managed OSM path as a **deployed, repeatable batch job**
- Score the managed predictions on held-out slices against a ground-truth sidecar

**Do not use this skill when:**
- You want interactive, per-window embeddings to do ML client-side — use [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)
- The raw CSV still needs cleaning / gap-aware segmentation / normalization — see [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md); the OSM agent assumes prepared, z-scored input
- You want to run **your own** fitted classifier rather than the pre-packaged one — that means creating your own bundle (blueprint `osm` + a `fit-classifier` S3 artifact); see the [OSM example repo](https://github.com/archetypeai/operational-state-monitoring-agent-example) (Stages 5–6) and the "Bring your own classifier" note below

## Endpoints

Two API surfaces are involved, mounted differently:

```
Files API   POST {ATAI_API_ENDPOINT}/v0.5/files                  (multipart upload)
            GET  {ATAI_API_ENDPOINT}/v0.5/files/download/{name}  (download)
Agent API   {ATAI_API_ENDPOINT}/agents/...                       (versionless!)
Authorization: Bearer <API_KEY> on every call
```

The Agent API is **versionless** — it lives at `/agents`, not `/v0.5/agents`. If your `ATAI_API_ENDPOINT` carries a `/vX.Y` suffix, strip it before appending `/agents`. **Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required** — there is no default endpoint.

> **⚠️ Singular vs plural — the bundle API is split.** Reads are **plural**
> (`GET /agents/bundles` to list/search, `GET /agents/bundles/{id}` to fetch
> one); create and run are **singular** (`POST /agents/bundle`,
> `POST /agents/bundle/{id}/run`). Using `GET /agents/bundle` (singular)
> 404s. This skill only needs the plural read (to resolve by name) and the
> singular run.

The full Agent API surface is specified in [`references/openapi.yaml`](references/openapi.yaml); [`references/agent-cli`](references/agent-cli) wraps every endpoint for interactive use.

## Step 1 — Upload the input CSV

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@sensor_slice.csv;type=text/csv" \
  "$ATAI_API_ENDPOINT/v0.5/files"
```

The response carries two identifiers — `file_id` (the filename) and `file_uid` (`fil_…`). **Source connectors reference the `file_id`, not the `fil_` uid.**

## Step 2 — Resolve the pre-packaged bundle by name

The maintained bundles are canonical (org-wide) and identified by a **stable name**. Their **id changes per environment** (dev/staging/prod), so resolve by name for portability — the plural read endpoint does a case-insensitive substring search over name and id:

```sh
curl -G -H "Authorization: Bearer $ATAI_API_KEY" \
  "$ATAI_API_ENDPOINT/agents/bundles" \
  --data-urlencode "query=OSM Quick Start (Volve Six State)" --data-urlencode "limit=20"
```

Two bundles are published:

| Name | Emits |
|---|---|
| `OSM Quick Start (Volve Six State)` | per-window state predictions |
| `OSM Quick Start (Volve Six State, Embeddings)` | the above **plus** `embedding_{variate}` columns (the encoder embedding per prediction) |

**Select the EXACT name match, not the first result** — the base name is a substring of the embeddings name, so a `query=` for the base name returns *both*. Pick `b["name"] == "OSM Quick Start (Volve Six State)"` and take its `id`.

The bundle already pins the classifier and its windowing (`window_size=16, step_size=1`, plus the Volve-sized validation tolerances), so there is **nothing to create and no classifier URI to supply**. (For reference, on Dev these currently resolve to `bnd_7877k9t49g9sdbgy9q92dakv3e` and `bnd_7vs5a4v58c8s3v14qpatx1y1eh` — but pin ids only as a last resort, since they differ in staging/prod.)

## Step 3 — Run the bundle

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundle/$BUNDLE_ID/run" -d '{
    "connectors": {"source": [{"type": "file", "id": "sensor_slice.csv"}]}
  }'
```

Each run creates a **new agent instance** (`agt_…`) — one agent per input file. With no sink configured, the runner writes one output per input, named after the input file. **Run agents sequentially**: concurrent runs share the deployment's GPU workers, and three at once run ~3× slower each than a solo run.

## Step 4 — Poll until terminal

```sh
GET $ATAI_API_ENDPOINT/agents/instances/{agent_id}          # status: running | paused | completed | failed | ...
GET $ATAI_API_ENDPOINT/agents/instances/{agent_id}/events   # audit log (level, created_at, message)
```

Poll every ~15 s, echoing new audit events (`run started`, `dispatched to JOS as job_…`, `JOS job completed`). Runtime scales with window count — see the table below.

**A `failed` status can hide a successful run.** Observed live: a job whose container logged "terminated successfully" (output present) surfaced as `status=failed` with `error: repeated failures polling JOS job` — the service's job poller flaked, not the job. Before re-running a "failed" agent, check `/results`; if the output is there, the run succeeded.

## Step 5 — Fetch results and download

```sh
GET $ATAI_API_ENDPOINT/agents/instances/{agent_id}/results
GET $ATAI_API_ENDPOINT/v0.5/files/download/{filename}
```

`/results` lists output refs (`data.filename`, `data.num_bytes`); download each via the files API. Run outputs are owned by the user who launched the run and do not expire.

[`references/run_osm_agent.py`](references/run_osm_agent.py) scripts the whole flow (upload → resolve bundle → run → poll → download, stdlib-only) and — if a `<input>_labels.csv` ground-truth sidecar sits next to the input — scores the run automatically: accuracy (all-windows and steady-state cuts), per-class precision/recall/F1, macro-F1.

## Output CSV — one row per window

```
finish_timestamp, start_timestamp, predicted_state, invalid, p_<state>, p_<state>, ...
```

- **`finish_timestamp` is the window-END timestamp** — a prediction means "the state *now*, given the last `window_size` samples"; `start_timestamp` is the window's first sample. Score each window against the ground-truth label of its final row (end-row labeling on `finish_timestamp`).
- **`p_<state>` columns are emitted in alphabetical state order**, not library order.
- **`predicted_state=INVALID_STATE`** marks windows straddling a timestamp seam (backward jump between concatenated segments) — the platform validates timestamp monotonicity strictly. Exclude these when scoring; count them.
- The **Embeddings** bundle adds `embedding_{variate}` columns (~76 KB/row extra); the base bundle omits them.
- Runs are **deterministic**: a fresh agent on the same input + bundle reproduces predictions exactly.

## Runtime

Runtime scales with window count (dev deployment, solo runs, `step_size=1`):

| Windows | Observed |
|--------:|---------:|
| ~1,750  | ~12 min  |
| ~3,940  | ~22 min  |
| ~4,185  | ~27 min  |

Classifier load is ~30 s of that; the rest is mostly Omega-encoding the windows. Concurrent runs divide the same GPU workers — sequential is faster per run and barely slower in total. Dev workers can be heavily contended; a run occasionally takes far longer than the table.

## Common Pitfalls

- **Resolve by name, not by id.** The pre-packaged bundle's id changes across dev/staging/prod; only the name is stable. And **match the name exactly** — the base name is a substring of the `…, Embeddings)` name, so a substring `query=` returns both.
- **Reads are plural, run is singular.** `GET /agents/bundles` (list/search) and `GET /agents/bundles/{id}` (fetch); `POST /agents/bundle/{id}/run` (run). `GET /agents/bundle/…` (singular) 404s.
- **Source connectors take the `file_id` (filename), not the `fil_` uid.** Both come back from the upload; using the uid fails to resolve.
- **The Agent API is versionless.** `POST {endpoint}/v0.5/agents/…` 404s; strip any `/vX.Y` suffix and use `/agents/…`. The files API keeps its `/v0.5`.
- **`failed` ≠ failed until you check `/results`.** The job poller can flake after a successful job; output present ⇒ the run succeeded.
- **Don't run agents concurrently.** Shared GPU workers make N parallel runs ~N× slower each.
- **Sampling-rate warnings are expected on irregular data.** The bundle loosens the tolerance for Volve's irregular sampling (Δt 1–27 s); expect warnings, not failures.
- **Score with end-row labeling on `finish_timestamp` and exclude `INVALID_STATE`.** Predictions are keyed to the window-end timestamp; seam windows are invalidated.

## Bring your own classifier (advanced)

The pre-packaged bundle runs Archetype AI's six-state Volve classifier. To run **your own** fitted classifier instead, create your own bundle from the `osm` blueprint and skip Step 2:

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundle" -d '{
    "blueprint": "osm",
    "name": "my OSM run",
    "values": {"window_size": 16, "step_size": 1,
               "sample_rate_interval_tolerance": 10.0},
    "artifacts": {"fit-classifier": "s3://<bucket>/<prefix>/my-classifier.safetensors"}
  }'
```

Then run the returned bundle id as in Step 3. Notes: the `fit-classifier` artifact **must be an `s3://` URI** (platform file ids and `https://` URLs fail with ENOENT; the files API's MIME allowlist rejects safetensors anyway); it must be in the **platform schema** (`ids`/`vectors`/`weights` tensors + `index`/`classifier` manifests), and `values.window_size` **must match the window the classifier was fitted with** (a mismatch silently degrades accuracy instead of erroring). Fitting the artifact is out of scope here — see the [OSM example repo](https://github.com/archetypeai/operational-state-monitoring-agent-example) (Stages 4–5, plus its `run_osm_fit.py` for fitting on the platform without local weights).

## Cleanup

Each run leaves an agent instance behind; the pre-packaged bundle is canonical and shared — **do not delete it**. Delete your own agent instances (and any bundle you created yourself):

```sh
python3 references/agent-cli agents delete agt_...    # your run's instance
python3 references/agent-cli bundles delete bnd_...   # only a bundle YOU created (cascades)
```

## Local Setup

```bash
# No third-party deps — references/run_osm_agent.py is stdlib-only.

# Drop a .env next to where you run (BOTH variables required, no default endpoint;
# note: NO /v0.5 suffix — the script mounts /agents and /v0.5/files itself):
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://api.dev.u1.archetypeai.app
EOF

cd skills/atai-operational-state-monitoring-agent/references
python3 run_osm_agent.py                        # default sample slice, base bundle
python3 run_osm_agent.py --embeddings           # variant that also emits embeddings
python3 run_osm_agent.py --csv my_slice.csv     # your own prepared CSV
```

Expect ~20–25 min for the ~4,185 step-1 windows of the sample slice; the script resolves the bundle by name, streams audit events while it polls, and self-scores against the `_labels.csv` sidecar at the end.

## File Layout

```
skills/atai-operational-state-monitoring-agent/
├── SKILL.md                  ← this file
├── references/
│   ├── run_osm_agent.py      ← the whole managed flow, stdlib-only (upload → resolve pre-packaged bundle → run → poll → download → score)
│   ├── agent-cli             ← hand-rolled CLI covering every Agent API endpoint (incl. delete cascades)
│   ├── openapi.yaml          ← Agent API spec (under development)
│   ├── .env.example          ← copy to .env and fill in
│   └── sample_data/
│       ├── volve_states_opt_slice_04.csv         ← 4,200-row six-state eval slice (prepared + z-scored)
│       ├── volve_states_opt_slice_04_labels.csv  ← ground-truth sidecar (DATE_TIME,label), scoring only
│       └── README.md                             ← dataset attribution (Equinor Volve) + prep provenance
└── tests/
    └── test_references.py    ← network-free unit tests (python -m unittest)
```
