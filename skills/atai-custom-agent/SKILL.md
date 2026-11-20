---
name: atai-custom-agent
description: >
  Build and run a custom agent on Archetype AI's platform: write your own
  blueprint YAML that wires the platform's existing agent nodes (Omega encoder,
  windowing and validation, channel features, anomaly / classifier heads,
  Newton C 2.6 / Fusion, Whisper, video readers, prompt builders, parsers,
  sources and sinks) into a new graph, then create it in a project, pin its
  values on a copy, package it, deploy it in batch or for serving, and run
  sessions on Agent Lifecycle 2.0; and, when the graph has a head to fit and
  you have labelled data, optimize it, promote the best trial and eval it once.
  Admin only on production: blueprint creation needs the admin role
  (developers get 403). Covers reading the node registry
  (`GET /agents/nodes/registry`: keys, ports, types, config schemas), the
  blueprint layout (`values`, `models`, `connectors`, `nodes`, `graph.edges`,
  `metrics`, `${values.x}`, `${required}`), a local pre-check of keys and
  ports, what the platform refuses at create and why, batch vs serving (a
  `TextSink` graph fails in batch: end with a parser and a `RecordsSink`), the
  serving request shapes for sensor windows and for video, reading failures
  from job events, and the optimize / promote / eval calls for a custom
  trainable head. Worked examples: Omega embeddings + channel statistics, a
  Newton C 2.6 video summary, and a trainable Omega anomaly detector fitted on
  Tennessee Eastman. Do NOT use to run a canonical agent as published
  (`atai-manual-generation-agent`, `atai-task-verification-agent`,
  `atai-omega-encoder-agent`, the OSM / RED / AD / label-free AD skills), to
  fit a canonical agent on your data (the OSM / RED / AD skills, Path 2), to
  prepare role files (the `*-data-prep` skills), or to call a model directly
  without an agent (`atai-newton-omega-model`, `atai-newton-fusion-model`).
---

# Custom Agent — your own blueprint from the platform's agent nodes

A canonical agent (MGA, TVA, OSM, …) is a **blueprint**: a YAML graph of the platform's agent nodes with some settings exposed as `values`. A **custom agent** is a blueprint you write yourself from the same nodes: a different graph, different models, different outputs. Once created it runs through exactly the same lifecycle as a canonical one:

```
blueprint.yaml ─► check ─► create (in a project) ─► copy with values ─► package ─► deployment ─► session ─► results
                  local     admin only     │          (packages carry      batch | serving
                                           │           no values)              ▲
                                           └─► optimize ─► promote ─► eval ─────┘   (trainable graphs only)
```

> **Admin only on production.** `POST /agents/blueprints` with a developer key returns
> **403** `forbidden`, "blueprint creation requires the admin role", even in the developer's
> own project. Everything after create (copy, package, deploy, sessions) is ordinary.
> This skill will gain a developer path when that restriction is lifted.

> **Verified on production** (`https://platform.u1.archetypeai.app/api`), **2026-11-20**,
> with `references/custom_agent.py` and an admin key:
>
> | Example | Batch | Serving |
> |---|---|---|
> | `omega-embed-stats.yaml` on `synthetic_2ch_100hz.csv` | 15 windows in 46 s; identical to the SE Sandbox and staging (features equal, embedding cosine ≥ 0.9999996) | deployment ready in 6 s, 15/15 windows match batch, median 0.24 s / query |
> | `omega-ad-custom.yaml` on the TEP roles: optimize → promote → eval | `delivery_01.csv`: 835 windows in 46 s, labels identical to staging | validation macro-F1 0.8417, test 0.7462, same as staging (Step 6) |
> | `newton-c-video-summary(-batch).yaml` on `three_steps.mp4` | `-batch` variant: 368 s (C 2.6 download 2 min 36 s, load 44 s, on GPU), the same 3 timestamped steps as staging and the SE Sandbox | plain variant: deployment ready in 197 s, same 3 steps, 15.2 s / query |

## Quick start

```sh
cd references
cp .env.example .env            # ATAI_API_ENDPOINT + an admin ATAI_API_KEY

python custom_agent.py registry                         # what you can build with
python custom_agent.py registry OmegaEncoderNode        # one node: ports, config schema, defaults
python custom_agent.py check blueprints/omega-embed-stats.yaml

python custom_agent.py create blueprints/omega-embed-stats.yaml --key my-omega-embed-stats \
    --values '{"timestamp_column": "timestamp", "data_columns": ["sensor_a", "sensor_b"]}'
python custom_agent.py batch --input sample_data/synthetic_2ch_100hz.csv --out results/
python custom_agent.py serve-records --input sample_data/synthetic_2ch_100hz.csv \
    --columns sensor_a,sensor_b --window 256 --step 128
python custom_agent.py cleanup
```

`create` makes a project the first time (or takes `--project prj_…`), and every step writes its IDs to `custom_agent_state.json`, so the next step picks them up. Use `--state other.json` per agent.

## Step 1 — Know what you can build with: the node registry

`GET /agents/nodes/registry` lists every node and connector on the platform you're on. Each entry has a `key`, a `description`, a `contract` (named input and output **ports**, each with a **type**), a `config_schema` and a `default_config`. **Registries differ between environments**, so build against the one you'll deploy to; `references/node-registry.prod.json` is production's on 2026-11-20 (34 nodes, 8 connectors).

| Group | Production keys (in → out types) |
|---|---|
| Sources | `RecordsSource` (→ Record), `MultiModalSource` (→ `text`: Text, `video`: VideoRef), `TextSource`, `ConversationSource` (serving chat) |
| Sinks | `RecordsSink` (Record), `TextSink` (Text), `ConversationSink`, `MultiheadClassifierSink` |
| Sensor prep | `ValueInterpolationNode` (Record → Record), `WindowingNode` (Record → Window), `InWindowsInterpolationNode`, `SamplingRateValidationNode`, `LimitValuesValidationNode` (Window → Window), `FlattenWindowsNode` (Window → Record) |
| Sensor models / features | `OmegaEncoderNode` (Window → Record, 768-d per channel), `ChannelFeaturesNode` (Window → Record) |
| Classical heads | `ClassifierNode`, `MultiheadClassifierNode`, `ClassifierFittingNode`, `AnomalyDetectorNode`, `OutlierDetectorNode`, `AnomalyEventsFormatNode`, `InvalidSplitterNode` |
| Video / audio | `VideoReaderNode` (VideoRef → VideoFrame), `VideoPreprocessorNode` (→ VideoPatch), `ImageReaderNode`, `AudioReaderNode` (VideoRef → Audio), `WhisperModelNode` (→ AudioTranscript) |
| Prompts | `ManualGenerationPromptGeneratorNode`, `TaskVerificationPromptGeneratorNode`, `ChatPromptGeneratorNode` (→ MultiModalPrompt), `PrepareSOPNode` |
| LLMs | `NewtonCModelNode` (C 2.5 / 2.6), `NewtonFusionModelNode` (MultiModalPrompt → Text) |
| Parsers | `ManualGenerationResultsParserNode`, `TaskVerificationResultsParserNode` (Text → Record) |
| Plumbing | `TeeNode` (copy to N named ports), `ConcatColumnsNode` (row-zip N ports), `SplitterNode` (left/right), `RouteNode`, `MergeNode`, `InterleaveNode` (generic `'a`) |

`python custom_agent.py registry` prints this table live; `registry <Key>` prints one node in full, which is how you find its config fields (e.g. `WindowingNode`: `window_size`, `step_size`, `compaction_threshold`).

## Step 2 — Write the blueprint

A blueprint has six parts. `references/blueprints/omega-embed-stats.yaml` in full:

```
values:       # the knobs: everything a copy may override; null = must be pinned before it runs
models:       # model names, referenced as ${models.x}
connectors:   # source (one) and sink (one): where data comes from and goes
nodes:        # id -> { key: <registry key>, config: {...} }
graph.edges:  # { from: <id>.<output port>, to: <id>.<input port> }
```

```
source ─► interpolate ─► window ─► windowInterpolate ─► samplingRate ─► limitValues ─► tee ─┬─ emb ─► encoder (Omega) ──┐
                                                                                              └─ feat ─► features ───────┴─► fuse ─► sink (jsonl)
```

Rules that matter:

1. **Exactly one source and one sink**, named `source` and `sink` under `connectors`. Two sources or two sinks register but cannot be deployed: a batch session with two sources is refused, and a serving deployment answers "needs exactly one source and one sink".
2. **Port names come from the contract**: `source.out`, `window.in`, `MultiModalSource`'s `text` and `video`, `SplitterNode`'s `left` and `right`. `TeeNode` and `ConcatColumnsNode` take their port names from `config.ports` (`ports: [emb, feat]` gives `tee.emb`, `tee.feat`, `fuse.emb`, `fuse.feat`).
3. **Types must match along each edge**: Window into a Window input, Record into a Record input. The platform checks this at create and refuses mismatches, e.g. feeding a classifier's records into an LLM: `incompatible contracts on edge classifier.out -> newton_c.in: MultiModalPrompt cannot accept values of Record`. There is no records → prompt node on production, so an "Omega verdict → LLM explanation" agent can't be built as one graph today; do the second half client-side.
4. **`values` are the interface.** Reference them as `${values.x}` anywhere in config. Leave inputs you can't know in advance as `null` (`timestamp_column`, `data_columns`) and pin them on a copy (Step 3).
5. **Models by name** in `models:` (`newton-omega-1-5-safetensors`, `newton-c-2-6-8b`, `newton-fusion-1-0`, `whisper-base` on production), referenced as `${models.x}`.

Pre-check locally before you create; it catches unknown keys, unknown ports and the source/sink count, offline against the live registry:

```sh
python custom_agent.py check my-agent.yaml
#   ✗ node `features`: `ChannelStatsNode` is not in this platform's registry
#   ✗ edge fuse.out -> sink.input: `RecordsSink` has no input `input` (has in)
```

The platform's own validation at create is the authority (it also checks types and config schemas); the pre-check just saves round trips.

## Step 3 — Create, pin values, package

```sh
A=https://platform.u1.archetypeai.app/api
H=(-H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json")

PRJ=$(curl -s -X POST "$A/projects" "${H[@]}" -d '{"name":"Custom Agents"}' | jq -r .id)

jq -n --arg y "$(cat my-agent.yaml)" --arg p "$PRJ" \
   '{blueprint_key:"my-agent-template", project_id:$p, yaml_document:$y}' > bp.json
BP=$(curl -s -X POST "$A/agents/blueprints" "${H[@]}" --data @bp.json | jq -r .id)

# A package carries no values: pin them on a copy, then package the copy
CP=$(curl -s -X POST "$A/agents/blueprints/$BP/copy" "${H[@]}" -d '{"project_id":"'$PRJ'","blueprint_key":"my-agent",
     "overrides":{"values":{"timestamp_column":"timestamp","data_columns":["sensor_a","sensor_b"]}}}' | jq -r .id)
PKG=$(curl -s -X POST "$A/agents/packages" "${H[@]}" \
     -d '{"blueprint_ids":["'$CP'"],"target":{"kind":"platform"},"name":"my-agent"}' | jq -r .id)
```

| Refusal | Meaning |
|---|---|
| 403 `forbidden` "blueprint creation requires the admin role" | not an admin key |
| 400 `project_required` | no `project_id`: every blueprint lives in a project |
| 400 "incompatible contracts on edge …" | a type mismatch (rule 3) |
| 409 | `blueprint_key` already used in the org: keys are unique per org, so version them (`my-agent-v2`) |

Reading a copy back can show `values: null` even though the override applies at run time (verified: the pinned columns were used). Check behaviour by running it, not by the GET.

One template serves many inputs: copy it again with other columns or another window for each dataset, and package each copy.

## Step 4 — Run it in batch

```sh
python custom_agent.py batch --input data.csv --out results/        # CSV: RecordsSource
python custom_agent.py batch --input clip.mp4 --out results/        # video: MultiModalSource
```

What it does, and the details that bite:

1. **Batch deployment** `POST /agents/deployments {"package_id", "kind": "batch"}`: holds no worker; reuse it for every file.
2. **Upload.** CSV: `POST /files/uploads/initiate` → `PUT` the presigned URL → `POST /files/uploads/{id}/complete`; the session references the file by **name**. Video: multipart `POST /files`; reference the returned `file_id`. File names are unique per org, so add a timestamp.
3. **Session** `POST /agents/sessions {"deployment_id", "connectors": {"source": [{"type": "file", "id": <name or file_id>, "format": "csv" | "mp4"}]}}`.
4. **Watch the job, not just the session.** A batch session can stay `running` with `error: null` after its pod failed. Read `GET /batch/jobs/{job_id}/events`: any `ERROR` event means it's over (the script then cancels the session and prints the events).
5. **Results** `GET /agents/sessions/{id}/results`: each `data.ref` is a root path (`/api/…`); fetch it from the host root, not from `…/api`.
6. **`completed` doesn't mean the windows are valid.** Check the `invalid` column of the output. A job whose every window failed validation still completes, with `predicted_label: INVALID_STATE` and a `NaN` score on every row. The usual cause is **timestamps in epoch milliseconds**: the validation nodes read timestamps as seconds, so `sampling_rate` and `max_temporal_gap` never match. Write them as seconds (`1577883600.004`, not `1577883600004`). Seen on 2-lead ECG at 250 Hz: 3597 of 3597 windows invalid with milliseconds, 0 with seconds. The script refuses a CSV whose first timestamp looks like milliseconds, and warns when more than half the windows come back invalid.

**Batch needs a sink that writes per-request files.** A graph ending in `TextSink` fails in batch right after the model loads ("internal error", pod exit 1) on every environment we tried. For batch, end text-producing graphs the way the canonical MGA blueprint does: a results parser (`ManualGenerationResultsParserNode`) → `RecordsSink` with `format: json/per-request`. That's the only difference between `newton-c-video-summary.yaml` (serving) and `newton-c-video-summary-batch.yaml` (batch). The parser expects MGA's numbered-step format, so the batch variant keeps MGA's default system prompt.

## Step 5 — Serve it

`POST /agents/deployments {"package_id", "kind": "serving"}` → wait for `ready` → `POST /agents/sessions {"deployment_id"}` → wait for `running` → `POST /agents/sessions/{id}/query` → cancel and delete the session, stop and delete the deployment. A serving deployment holds a GPU worker: always remove it.

Sensor windows (a `RecordsSource` graph) — one window per query, channel-first, no timestamps:

```json
{"query": "", "events": [{"type": "data.numeric_array",
  "event_data": {"contents": [[...256 values of sensor_a...], [...256 values of sensor_b...]]}}]}
```

The answer's `response.data` holds the records the sink would have written for that window.

Video (a `MultiModalSource` graph) — the instruction and the clip inline:

```json
{"query": "Summarize what happens in this video as a short numbered list of steps.",
 "file_ids": ["base64:<the .mp4, base64-encoded>"]}
```

The text is at `response.response.response[0]` (Fusion prefixes it with an empty `<think>\n\n</think>` block).

```sh
python custom_agent.py serve-video --input clip.mp4 --prompt "Summarize what happens in this video."
```

Timing: an Omega serving deployment was ready in 6 s; a video deployment downloads its model the first time (C 2.6 ≈ 17.5 GB, Fusion ≈ 36.6 GB), so allow minutes to a quarter of an hour, and run one GPU serving deployment at a time if capacity is tight.

## Step 6 — Trainable agents: optimize, promote, eval (when it makes sense)

Optimize only applies when the graph has **something to fit** and you have **labelled data to score it on**. The Omega embed-stats and video summary agents have neither (they are zero-shot: run them as they are). An agent is trainable when its blueprint has:

1. **A head with fittable state**: `AnomalyDetectorNode` (LOF on normal data), `ClassifierNode` / `MultiheadClassifierNode` (prototypes per class), `OutlierDetectorNode`. Its `model` is a placeholder (`models: { detection_head: }` left empty): the optimizer fits it per trial, and promote fills it in.
2. **A `metrics:` section** naming what to score:

```yaml
metrics:
  targets:
    - name: anomaly                     # a target; ground truth is supplied under this name
      type: binary
      predicted: predicted_label        # the output column holding the prediction
      labels: [${values.anomaly_label}, ${values.normal_label}]
      positive: ${values.anomaly_label}
      pairing: by_time                  # match predictions to labels by timestamp
      objectives: [macro_f1, accuracy, foreground_macro_f1, detection_recall, detection_precision]
  primary: anomaly.macro_f1             # what optimize maximizes
```

3. **Values the search can set**, and `${required}` for one that must come from the copy or the search (`threshold: ${required}` in `omega-ad-custom.yaml`).

The data comes as role files: `library/` (fit), `validation/` (choose settings), `test/` (score once), each CSV with a `label` column where it's scored. Preparing them is what the AD / OSM / RED data-prep skills do; the worked example uses the Tennessee Eastman roles from [archetypeai/custom-blueprint](https://github.com/archetypeai/custom-blueprint) (`tep/roles/`).

```sh
git clone https://github.com/archetypeai/custom-blueprint && ROLES=custom-blueprint/tep/roles

python custom_agent.py --state ad.json create blueprints/omega-ad-custom.yaml --key my-ad --no-package
python custom_agent.py --state ad.json optimize --roles $ROLES --spec optimize/omega-ad-tep.json
python custom_agent.py --state ad.json promote --key my-ad-tep-best        # best trial -> blueprint + package
python custom_agent.py --state ad.json eval --roles $ROLES --spec optimize/omega-ad-tep.json
python custom_agent.py --state ad.json batch --input $ROLES/delivery/delivery_01.csv --out results/
```

The spec (`references/optimize/omega-ad-tep.json`) holds what's specific to the agent and the data:

| Field | What |
|---|---|
| `primary` | `<target>.<objective>` to maximize, from the blueprint's `metrics` |
| `values` | pinned for every trial (columns come from `manifest.json` in the roles dir when absent) |
| `search_space` | `{name: {"kind": "value", "spec": {"type": "categorical", "values": [...]}}}`; categorical values must be ints or strings (no booleans) |
| `max_trials` | budget |
| `ground_truth.training` / `.scored` | per target: `{"from": {"constant": "normal"}}` for a normal-only library, `{"from": {"column": "label"}}` for scored files; `"downsampling": "last_record"` labels a window by its last row |

What the calls are:

1. **Optimize** `POST /agents/optimizations` with the **template** `blueprint_id` (not a package), `overrides.values`, `search_space`, `budget`, `training_examples` and `validation_examples` (each `{"name", "inputs": [{"type": "file", "id": <file_id>, "format": "csv"}], "ground_truth": {...}}`). Files are uploaded with multipart `POST /files`. Trials run as jobs; `GET /agents/optimizations/{id}/trials` gives each trial's values, `objective_value` and `metrics_report`.
2. **Promote** `POST /agents/optimizations/{id}/trials/{trial}/promote {"blueprint_key", "name"}`: a new blueprint with the trial's values pinned and the fitted head as a registry artifact in `models`. Choosing the key needs the admin role (you have it). Package it and deploy it like any other.
3. **Eval** `POST /agents/evals {"blueprint_id": <promoted>, "examples": [test files], "emit_predictions": true}`: scores the promoted agent once, at its own pinned values (no overrides).

Verified on production, 2026-11-20, with the commands above on the TEP roles (52 channels, 10 library files):

| Step | Result |
|---|---|
| Optimize (4 trials, ~3 min) | best validation macro-F1 **0.8417** at `n_neighbors` 5 (10: 0.8336); `interpolation_method` made no difference (tied pairs) |
| Promote | `skill-omega-ad-tep-best`, packaged |
| Eval (~1 min) | test macro-F1 **0.7462**, accuracy 0.7542, detection recall 0.7017, precision 1.0 |
| Batch on `delivery_01.csv` | 835 windows in 46 s, 835/835 labels identical to staging and the SE Sandbox |

All three numbers match staging and the SE Sandbox exactly. When trials tie, as here, the search dimension doesn't matter for this data: drop it and spend the budget on one that does (window, threshold, PCA).

## Worked examples (`references/blueprints/`)

| File | Graph | Values to pin | Output |
|---|---|---|---|
| `omega-embed-stats.yaml` | CSV → interpolate → window → validate → tee → (Omega, channel features) → concat → jsonl | `timestamp_column`, `data_columns`; optional `window_size` (256), `step_size` (128), `feature_names` | per window: `start_timestamp`, `finish_timestamp`, `<channel>` (768-d), `<channel>_features` (`mean`, `std`, `slope`) |
| `newton-c-video-summary.yaml` | video + instruction → (frames → patches, audio → Whisper) → prompt → Newton C 2.6 → `TextSink` | optional `prompt`, `system_message`, `max_frames` (32), `max_new_tokens` | serving: the model's text answer |
| `newton-c-video-summary-batch.yaml` | same, then MGA parser → `RecordsSink` `json/per-request` | optional `prompt`, `max_frames` | batch: `{"id", "results": [{step, instruction, timestamp_start/end, frame_start/end}]}` |
| `omega-ad-custom.yaml` (trainable) | the embed-stats graph → `AnomalyDetectorNode` (LOF, PCA) → jsonl; `metrics` on `anomaly` | `timestamp_column`, `data_columns`, `threshold` (`${required}`); searched: e.g. `n_neighbors`, `interpolation_method` | per window: `predicted_label` (`normal` / `anomaly`), score |

For Fusion, swap the model node and model name (`NewtonFusionModelNode`, `newton-fusion-1-0`) and drop C 2.6's frame rules (`index_rule`, `timestamp_rule`) from the reader and preprocessor.

## Cleanup

```sh
python custom_agent.py cleanup              # cancel open sessions, stop the batch deployment
python custom_agent.py cleanup --packages   # and delete the package
```

`DELETE /agents/deployments/{id}` on a stopped batch deployment can answer 404; a stopped batch deployment holds no worker, so it's harmless. Blueprints and projects stay until you delete them.

## Checklist

- [ ] Admin key; `GET /agents/sessions?limit=1` answers 200 (Lifecycle 2.0)
- [ ] Every node key and port is in the registry of the environment you deploy to (`check`)
- [ ] One source, one sink; batch graphs end in a `RecordsSink`
- [ ] Inputs left `null` in the template are pinned on the copy you package
- [ ] Blueprint keys and uploaded file names are unique
- [ ] Batch failures read from job events, not session status
- [ ] Timestamps in seconds; the output's `invalid` share checked after each batch
- [ ] Serving deployments stopped and deleted after use
