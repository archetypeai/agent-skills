---
name: atai-operational-state-monitoring-agent
description: >
  Run Archetype AI's managed Operational State Monitoring (OSM) agent over
  the Agent API — upload a sensor CSV, create a bundle from the canonical
  `osm` blueprint pinning a fitted classifier artifact, run it (one agent
  per input file), poll status + audit events, and download the per-window
  state predictions. Use this skill when the user has a fitted OSM
  classifier (safetensors on S3) and wants fully-managed, server-side
  classification of operational states over a CSV of sensor records —
  drilling states, machine modes, process phases. Covers the bundle
  request shape (`values` windowing overrides, the `fit-classifier`
  artifact slot), the run/poll/results lifecycle, the output CSV schema
  (`timestamp, predicted_state, invalid, p_<state>…`), and scoring against
  a ground-truth sidecar. Do NOT use for client-side embedding + KNN over
  `/query` (that's `atai-newton-omega-model`), for cleaning / windowing
  raw CSVs (`atai-newton-omega-model-data-prep`), or for fitting the
  classifier artifact itself (see the OSM example repo's fit stages).
---

# OSM Agent — Managed State Classification via the Agent API

The OSM agent is the **fully-managed counterpart** to the client-side embed-and-KNN pattern in [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md): instead of fanning out `/query` embedding calls and classifying locally, you hand the platform a CSV and a fitted classifier, and it runs the whole graph server-side:

```
source → window → samplingRate → limitValues → encoder (omega:1.5) → classifier → sink
```

One run = one agent instance = one input file. You upload the CSV, create a **bundle** from the canonical `osm` **blueprint** (pinning your classifier artifact and windowing), run the bundle, poll until terminal, and download one output CSV of per-window predictions.

## When to Apply

- Classify operational states over a full CSV of sensor records with **no client-side ML** — the platform embeds and classifies every window
- Run a classifier that was fitted elsewhere (grid search over local Omega embeddings, or the reference fit tools) as a **deployed, repeatable batch job**
- Evaluate a fitted classifier on held-out slices with the platform's own inference path (parity with local KNN is ~100% on valid windows)

**Do not use this skill when:**
- You want interactive, per-window embeddings to do ML on client-side — use [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)
- The raw CSV still needs cleaning / gap-aware segmentation / normalization — see [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md); the OSM agent assumes prepared, z-scored input
- You don't have a fitted classifier artifact yet — fitting is out of scope here (see the [OSM example repo](https://github.com/archetypeai/operational-state-monitoring-agent-example), Stages 4–5, for the grid search and artifact build)

## Endpoints

Two API surfaces are involved, mounted differently:

```
Files API   POST {ATAI_API_ENDPOINT}/v0.5/files                  (multipart upload)
            GET  {ATAI_API_ENDPOINT}/v0.5/files/download/{name}  (download)
Agent API   {ATAI_API_ENDPOINT}/agents/...                       (versionless!)
Authorization: Bearer <API_KEY> on every call
```

The Agent API is **versionless** — it lives at `/agents`, not `/v0.5/agents`. If your `ATAI_API_ENDPOINT` carries a `/vX.Y` suffix, strip it before appending `/agents`. **Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required** — there is no default endpoint.

> **⚠️ Dev-only for now.** Everything in this skill is verified against the
> **Dev** deployment (`https://api.dev.u1.archetypeai.app`) — the canonical
> `osm` blueprint, the Agent API surface, the published default classifier
> artifact, and the runtime numbers below. The underlying
> [OSM example](https://github.com/archetypeai/operational-state-monitoring-agent-example)
> has not yet been validated on Staging or Prod; until it is, don't point
> this skill at those deployments and expect the blueprint or the pinned
> `s3://` classifier to resolve.

The full Agent API surface (blueprints, bundles, instances, events, logs, results, pause/resume/cancel, node registry) is specified in [`references/openapi.yaml`](references/openapi.yaml); [`references/agent-cli`](references/agent-cli) wraps every endpoint for interactive use.

## Step 1 — Upload the input CSV

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@sensor_slice.csv;type=text/csv" \
  "$ATAI_API_ENDPOINT/v0.5/files"
```

The response carries two identifiers — `file_id` (the filename) and `file_uid` (`fil_…`). **Source connectors reference the `file_id`, not the `fil_` uid.**

## Step 2 — Create a bundle from the `osm` blueprint

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$AGENTS/bundle" -d '{
    "blueprint": "osm",
    "name": "OSM six-state run",
    "values": {"window_size": 16, "step_size": 1},
    "artifacts": {"fit-classifier": "s3://<bucket>/<prefix>/my-classifier-20260710T063913Z.safetensors"}
  }'
```

Two fields do all the work:

- **`values`** — per-bundle overrides of the blueprint's node config. `window_size` **must match the window the classifier was fitted with**: the embedding dimension is identical for every window size, so a mismatch doesn't error — it silently degrades accuracy. `step_size: 1` scores every row (window count ≈ row count); larger steps trade coverage for runtime.
- **`artifacts.fit-classifier`** — the fitted classifier, **as an `s3://` URI**. Platform file ids and `https://` URLs fail with ENOENT (the `ClassifierNode` resolves artifact strings as filesystem/S3 paths only, no HTTP fetch), and the files API's MIME allowlist rejects safetensors uploads anyway. Upload the artifact to S3 with a timestamped name so every deployment is unique and rollback is trivial.

The classifier artifact must be in the **platform schema** (`ids`/`vectors`/`weights` tensors + `index` and `classifier` JSON manifests) — a reference-tools artifact fails with `checkpoint missing 'index' manifest`. The platform honors the KNN hyperparameters (k, metric `l1`/`l2`, weights) from the artifact's `classifier` manifest, so those travel with the artifact, not the bundle.

## Step 3 — Run the bundle

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$AGENTS/bundle/$BUNDLE_ID/run" -d '{
    "connectors": {"source": [{"type": "file", "id": "sensor_slice.csv"}]}
  }'
```

Each run creates a **new agent instance** (`agt_…`) — one agent per input file. With no sink configured, the runner writes one output per input, named after the input file. **Run agents sequentially**: concurrent runs share the deployment's GPU workers, and three at once run ~3× slower each than a solo run.

## Step 4 — Poll until terminal

```sh
GET $AGENTS/instances/{agent_id}          # status: running | paused | completed | failed | ...
GET $AGENTS/instances/{agent_id}/events   # audit log (level, created_at, message)
```

Poll every ~15 s, echoing new audit events (`run started`, `dispatched to JOS as job_…`, `JOS job completed`). Runtime scales with window count — see the table below.

**A `failed` status can hide a successful run.** Observed live: a job whose container logged "terminated successfully" (output present) surfaced as `status=failed` with `error: repeated failures polling JOS job` — the service's job poller flaked, not the job. Before re-running a "failed" agent, check `/results`; if the output is there, the run succeeded.

## Step 5 — Fetch results and download

```sh
GET $AGENTS/instances/{agent_id}/results
GET $ATAI_API_ENDPOINT/v0.5/files/download/{filename}
```

`/results` lists output refs (`data.filename`, `data.num_bytes`); download each via the files API.

[`references/run_osm_agent.py`](references/run_osm_agent.py) scripts the whole flow (upload → bundle → run → poll → download, stdlib-only) and — if a `<input>_labels.csv` ground-truth sidecar sits next to the input — scores the run automatically: accuracy (all-windows and steady-state cuts), per-class precision/recall/F1, macro-F1.

## Output CSV — one row per window

```
timestamp, predicted_state, invalid, p_<state>, p_<state>, ...
```

- **`timestamp` is the window-END timestamp** — a prediction means "the state *now*, given the last `window_size` samples". Score each window against the ground-truth label of its final row (end-row labeling).
- **`p_<state>` columns are emitted in alphabetical state order**, not library order.
- **`predicted_state=INVALID_STATE`** marks windows straddling a timestamp seam (backward jump between concatenated segments) — the platform validates timestamp monotonicity strictly, where the reference tools only invalidate on NaN/Inf. Exclude these when scoring; count them.
- Runs are **deterministic**: a fresh agent on the same input + artifact reproduces predictions exactly, and platform-vs-local KNN parity on the same embeddings measured 0.9998–1.0000 across evaluations.

## Runtime

Runtime scales with window count (dev deployment, solo runs, `step_size=1`):

| Windows | Observed |
|--------:|---------:|
| ~1,750  | ~12 min  |
| ~3,940  | ~22 min  |
| ~4,185  | ~27 min  |

Classifier load is ~30 s of that; the rest is mostly Omega-encoding the windows. Concurrent runs divide the same GPU workers — sequential is faster per run and barely slower in total.

## Common Pitfalls

- **Artifacts must be `s3://` URIs.** Platform file ids and full `https://` download URLs fail with instant ENOENT — no fetch is attempted. The files API can't host safetensors either (MIME allowlist: images, mp4, CSV/TSV/JSON/JSONL, plain text).
- **`window_size` mismatch is silent.** Every window size produces the same embedding dimension, so running a w=16-fitted classifier at w=64 doesn't error — accuracy just quietly drops. Pin the bundle's `values.window_size` to the fit config.
- **Source connectors take the `file_id` (filename), not the `fil_` uid.** Both come back from the upload; using the uid fails to resolve.
- **The Agent API is versionless.** `POST {endpoint}/v0.5/agents/bundle` 404s; strip any `/vX.Y` suffix and use `/agents/...`. The files API keeps its `/v0.5`.
- **`failed` ≠ failed until you check `/results`.** The job poller can flake after a successful job; output present ⇒ the run succeeded.
- **Don't run agents concurrently.** Shared GPU workers make N parallel runs ~N× slower each.
- **Sampling-rate warnings are expected on irregular data.** The platform's sampling-rate validation is warn-only (tolerance 0.5); irregularly-sampled telemetry (e.g. Δt 1–27 s) produces warnings, not failures. (The reference validator makes this a hard error — platform and local disagree here by design.)
- **Score with end-row labeling and exclude `INVALID_STATE`.** Window predictions are keyed to the window-end timestamp; seam windows are invalidated on the platform but not by the reference tools — excluding them is what makes platform-vs-local parity exact.

## Cleanup

Deletion cascades blueprint ← bundle ← agent; each run leaves an agent instance behind.

```sh
python3 references/agent-cli bundles delete bnd_...   # cancels + cascades, with a prompt
python3 references/agent-cli agents delete agt_...    # single instance
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
python3 run_osm_agent.py \
  --csv sample_data/volve_states_opt_slice_04.csv \
  --classifier s3://<bucket>/<prefix>/<your-classifier>.safetensors \
  --window-size 16 --step-size 1
```

Expect ~20–25 min for the ~4,185 step-1 windows of the sample slice; the script streams audit events while it polls and self-scores against the `_labels.csv` sidecar at the end.

## File Layout

```
skills/atai-operational-state-monitoring-agent/
├── SKILL.md                  ← this file
├── references/
│   ├── run_osm_agent.py      ← the whole Stage-6 flow, stdlib-only (upload → bundle → run → poll → download → score)
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
