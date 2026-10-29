# Archetype AI Agent Skills

[![skills.sh](https://www.skills.sh/b/archetypeai/agent-skills)](https://www.skills.sh/archetypeai/agent-skills)

Agent skills for building applications with [Archetype AI's Newton](https://www.archetypeai.io/) — a real-time sensor intelligence platform that understands physical world data through foundation models.

Inspired by [mongodb/agent-skills](https://github.com/mongodb/agent-skills).

## Skills

Four groups: **Agents** run a maintained pipeline server-side over the Agents API, **Models** call Newton directly on `/api/lens/query` and leave the orchestration to you, **Integrations** pair a Newton model with a third-party model, and **Design** covers the demo front-end.

### Agents

Managed end-to-end pipelines — upload an input, run a pre-packaged bundle or canonical blueprint, poll, download the output. No model to host. The three sensor agents differ by **what you already have labelled**: every regime (OSM), a handful of examples of one named fault (RED), or nothing but normal operation (AD). OSM and AD can also be **fitted to your own data** on the platform (Optimizations → promote → Evals → bundle), each with its data-prep skill in front.

| Skill | Description |
|-------|-------------|
| [atai-operational-state-monitoring-agent](skills/atai-operational-state-monitoring-agent/) | Run the OSM agent over the Agents API, two paths. **Path 1:** resolve the pre-packaged "OSM Quick Start" bundle by name (classifier + windowing already pinned), run one agent per input CSV, poll, and download per-window state predictions. **Path 2:** build an agent on your own labelled data — fit and score settings with the Optimizations API, promote the best trial to a blueprint, test it once with the Evals API, deliver it as a bundle, and score the delivery. Worked example: the [LARCO washing-machine quickstart](https://github.com/archetypeai/operational-state-monitoring-agent-example-larco-quickstart) |
| [atai-operational-state-monitoring-agent-data-prep](skills/atai-operational-state-monitoring-agent-data-prep/) | Turn labelled sensor recordings into the role files the OSM platform accepts — exact sample grid split at gaps, a check that the sensor can tell the states apart, a group-aware split, library-only z-score, one training file per state, continuous scored files, held-back delivery labels — and preflight every platform rule before upload. Local, no API key |
| [atai-rare-event-detection-agent](skills/atai-rare-event-detection-agent/) | Run the Rare Event Detection agent over the Agents API, two paths. **Path 1:** resolve the pre-packaged "RED Quick Start" bundle by name (nearest-prototype classifier + windowing already pinned), run one agent per input CSV, poll, and download per-window rare-event predictions. **Path 2:** build an agent on your own labelled incidents — fit one prototype per class and choose the window and distance with the Optimizations API, promote the best trial, test it once with the Evals API, deliver it as a bundle, and score each incident (caught, how soon) and the false alarms. Worked example: the [pump quickstart](https://github.com/archetypeai/rare-event-detection-agent-example-pump-quickstart) |
| [atai-rare-event-detection-agent-data-prep](skills/atai-rare-event-detection-agent-data-prep/) | Turn one labelled recording into the role files the RED platform accepts — exact sample grid, labels mapped to a background and named event classes, numbered incidents, a check that the incidents and channels can make an agent (incident length against window size, channels that go dark during events, zero-saturation, separation that carries over to held-out incidents), build-time-only z-score, one training file per class piece, scored files around each incident with no row shared, held-back delivery labels — and preflight every rule before upload. Local, no API key |
| [atai-anomaly-discovery-agent](skills/atai-anomaly-discovery-agent/) | Run the Anomaly Discovery agent over the Agents API, two paths. **Path 1:** resolve the pre-packaged "AD Quick Start" bundle by name (fitted LOF detector + threshold already pinned), run one agent per input CSV, poll, and download a per-window anomaly score. **Path 2:** build a detector on your own data — fit it on normal-only recordings and choose its settings with the Optimizations API, promote the best trial, test it once with the Evals API, deliver it as a bundle, score detection, delay and false alarms, and serve it a window per query over `/query` (agent serving), checked against the delivery. For assets with **no fault history**: fitted on normal data only, so everything it flags is something it was never shown. Worked example: the [Tennessee Eastman quickstart](https://github.com/archetypeai/anomaly-discovery-agent-example-tep-quickstart) |
| [atai-anomaly-discovery-agent-data-prep](skills/atai-anomaly-discovery-agent-data-prep/) | Turn normal-operation recordings (plus a few labelled incidents) into the role files the AD platform accepts — exact sample grid split at gaps, a check that the normal-only library is a trustworthy baseline (purity, stability, agreement, enough windows for the detector), library-only z-score, scored files that join recordings with a gap the platform rejects, held-back delivery labels, a calibration file — and preflight every rule before upload. Local, no API key |
| [atai-manual-generation-agent](skills/atai-manual-generation-agent/) | Run the managed Manual Generation (MGA) agent over the Agents API — upload a procedure video, run the canonical `mga` blueprint, and get back an ordered, timestamped manual with each step traceable to a time range in the source |
| [atai-task-verification-agent](skills/atai-task-verification-agent/) | Run the managed Task Verification (TVA) agent over the Agents API — upload a recording **and** a reference procedure (an SOP), and get back a per-step PASSED / FAILED / MISSING verdict with a timestamp and a reason for each step. The SOP is a runtime input, so one bundle serves every procedure |

### Models

Direct Query API — one stateless `POST /api/lens/query` per request, for when you want control over the pipeline or the raw vectors to build on.

| Skill | Description |
|-------|-------------|
| [atai-newton-omega-model-data-prep](skills/atai-newton-omega-model-data-prep/) | Clean, split, and featurize multivariate time-series data before the Omega model — gap-aware blocking + imputation, out-of-time train/test split, and joint-state (X, y) featurization |
| [atai-newton-omega-model](skills/atai-newton-omega-model/) | Get time-series embeddings from the Omega encoder (`OmegaEncoder::omega_embeddings_1_4`) over `/api/lens/query` — one stateless call per channel, fanned out in parallel — for client-side KNN classification, anomaly scoring, and similarity search |
| [atai-newton-fusion-model](skills/atai-newton-fusion-model/) | Call the Newton C 2.6 fusion model on `/api/lens/query` with text, image, or video in one stateless POST — the first C checkpoint to reason over video frames via the Direct Query API |

### Integrations

Newton paired with a third-party model: Newton does the sensing, the other model reasons over what Newton found.

| Skill | Description |
|-------|-------------|
| [atai-newton-omega-nvidia-nemotron](skills/atai-newton-omega-nvidia-nemotron/) | Omega senses, NVIDIA Nemotron explains — Omega embeddings + client-side KNN decide the state of a sensor window, and Nemotron (NVIDIA hosted API) turns the verdict and window statistics into a validated fault brief or operator actions. Covers NVIDIA key types, model retirement, turning Nemotron 3's reasoning off, JSON-only prompts, and caching |

### Design

| Skill | Description |
|-------|-------------|
| [atai-design-system](skills/atai-design-system/) | Build a Newton demo front-end with the Archetype AI Design System — scaffold via the `ds` CLI (`@archetypeai/ds-cli`) and compose the published Svelte 5 primitives + OKLCH tokens (`@archetypeai/ds-{lib-tokens,ui-svelte-console,ui-svelte-labs}`) instead of hand-rolling UI |

More skills are in review and will be added as they land.

## Building with the Design System

When starting a new demo front-end, don't hand-roll the UI — scaffold with the [atai-design-system](skills/atai-design-system/) skill and build on top of what it generates. The `ds` CLI stands up a SvelteKit + Tailwind v4 project wired to the published Archetype AI Design System packages (branded tokens, fonts, and Svelte 5 primitives), and writes an agent config (`CLAUDE.md`/`AGENTS.md` + `ds-manifest.json`) that documents every available component, its usage recipes, and variant axes.

Use that scaffolded project as the baseline for your demo: the tokens, brand styling, and component library are already in place, so you can focus on the Newton integration (fusion or Omega calls via the model skills) and the demo-specific views.

## Example apps

End-to-end demos built on these skills, on the Direct Query API (SvelteKit unless noted):

| Demo | Skills | What it demonstrates |
|------|--------|----------------------|
| [Traffic Monitor](https://github.com/archetypeai/archetypeai-traffic-demo) | `atai-newton-fusion-model` | Live Caltrans CCTV feed → C 2.6 vision. Samples a short burst of frames per interval and sends it as one multi-frame Direct Query API clip to reason over traffic flow, incidents, and conditions. |
| [Wildfire Watch](https://github.com/archetypeai/archetypeai-wildfire-demo) | `atai-newton-fusion-model` | 1,200+ ALERTCalifornia cameras → C 2.6 vision. Per-camera smoke/fire/haze detection, a zone scan that batches frames per query, and text-only zone Q&A. |
| [Earthquake monitor](https://github.com/archetypeai/archetypeai-earthquake-demo) | `atai-newton-fusion-model` | Live USGS feed → C 2.6 **text reasoning**. Formats the current quakes as structured text and asks Newton (one stateless query) to surface aftershock sequences, spatial clustering, and ranked regional risk — alongside an interactive world map. |
| [SWaT water treatment](https://github.com/archetypeai/archetypeai-swat-demo-direct-query) | `atai-newton-omega-model` + `atai-newton-fusion-model` | Six-stage plant anomaly detection: Omega per-channel embeddings + client-side KNN, plus C 2.6 text reasoning for operator action suggestions. |
| [Wind turbine monitor](https://github.com/archetypeai/archetypeai-wind-turbine-demo) (Python/Flask) | `atai-newton-omega-model` + `atai-newton-omega-model-data-prep` | Penmanshiel wind-farm SCADA anomaly detection: Omega per-channel embeddings + local KNN against a leakage-free n-shot library precomputed offline. Replays 3 months of telemetry and detects a real frequency-converter fault on one turbine against its healthy peer. |
| [SWaT × NVIDIA Nemotron](https://github.com/archetypeai/archetypeai-swat-demo-nemotron) | `atai-newton-omega-model` + `atai-newton-omega-nvidia-nemotron` | The SWaT demo with NVIDIA Nemotron in place of C 2.6 for operator suggestions: Nemotron returns upstream / local / downstream action cards as JSON, checked in code against the plant topology and each stage's equipment, cached per anomaly set. |
| [Wind turbine × NVIDIA Nemotron](https://github.com/archetypeai/archetypeai-wind-turbine-demo-nemotron) (Python/Flask) | `atai-newton-omega-model` + `atai-newton-omega-nvidia-nemotron` | The Penmanshiel demo plus NVIDIA Nemotron fault briefs: when Omega flags a turbine, Nemotron compares the window with the healthy peer over the same hours and returns what changed, the likely cause and what to check. |
| [Drilling state monitor](https://github.com/archetypeai/archetypeai-drilling-demo) | `atai-newton-omega-model` + `atai-newton-omega-model-data-prep` | Equinor Volve North Sea well SCADA: per-channel Omega embeddings + local KNN classify each window as drilling / not-drilling, against a leakage-free n-shot library precomputed offline from held-out reference wells. Replays real well telemetry with live accuracy vs ACTC ground truth. |
| [Grid monitor](https://github.com/archetypeai/archetypeai-grid-demo) | `atai-newton-fusion-model` | Live CAISO power-grid feed → C 2.6 **text reasoning**. Formats 5-minute demand/supply data as structured text and asks Newton (one stateless query) about duck-curve dynamics, evening ramp, renewable share, and grid-stress risk, with supply/demand charts. |
| [WiFi occupancy monitor](https://github.com/archetypeai/archetypeai-wifi-demo) | `atai-newton-fusion-model` | Residential gateway WiFi telemetry → C 2.6 **text reasoning**. Sends an anonymized 15-minute per-device flow/byte/protocol snapshot as JSON via one stateless query; Newton infers home occupancy (OCCUPIED … EMPTY) from traffic patterns alone — no device-type labels, no online flag. |

## Agent examples

The agent lifecycle end to end on real data, over the Agents API (Python scripts, one per stage):

| Example | Skills | What it demonstrates |
|---------|--------|----------------------|
| [LARCO washing-machine quickstart](https://github.com/archetypeai/operational-state-monitoring-agent-example-larco-quickstart) | `atai-operational-state-monitoring-agent` + `atai-operational-state-monitoring-agent-data-prep` | LARCO washing-machine vibration (9 channels, 200 Hz) → an OSM agent fitted on the platform: an Optimizations search over windowing and kNN settings, the best trial promoted to a blueprint, one Evals test, then a bundle delivered to a second machine and scored against its held-back labels (fill / wash / spin / drain). Test macro-F1 0.86, delivery 0.77. |

## Quick Start

### Any coding agent (recommended)

Install via the [skills.sh](https://www.skills.sh/) CLI — it detects your coding agents (Claude Code, Cursor, Codex, Copilot, and 20+ others) and installs to the right location for each:

```bash
# Install into the current project
npx skills add archetypeai/agent-skills

# Or install globally (available in all projects)
npx skills add archetypeai/agent-skills -g
```

### Manual (Claude Code)

```bash
# Add as global skills (available in all projects)
cp -r skills/* ~/.claude/skills/

# Or add to a specific project
cp -r skills/* your-project/.claude/skills/
```

### Invoke a Skill

```
# Agents — managed pipelines over the Agents API
/atai-operational-state-monitoring-agent            # Every operating regime, from a full labelled library
/atai-operational-state-monitoring-agent-data-prep  # Labelled recordings -> OSM role files, preflighted
/atai-rare-event-detection-agent                    # One named fault, from a handful of labelled examples
/atai-rare-event-detection-agent-data-prep          # A labelled recording -> RED role files, preflighted
/atai-anomaly-discovery-agent                       # Normal-only fit; per-window anomaly score, no fault history needed
/atai-manual-generation-agent                       # Procedure video -> ordered, timestamped manual
/atai-task-verification-agent                       # Recording + SOP -> per-step PASSED / FAILED / MISSING

# Models — Direct Query API
/atai-newton-omega-model-data-prep                  # Clean / split / featurize time-series before the Omega model
/atai-newton-omega-model                            # Omega time-series embeddings + client-side KNN via /query
/atai-newton-fusion-model                           # Multimodal (text/image/video) queries on the C 2.6 fusion model

# Integrations
/atai-newton-omega-nvidia-nemotron                  # Omega + KNN verdicts explained by NVIDIA Nemotron (fault briefs, operator actions)

# Design
/atai-design-system                                 # Scaffold + build a Newton demo front-end with the Design System
```

## Architecture

Two API surfaces, one per skill family.

**Models — the Direct Query API.** One stateless POST to `/api/lens/query` per request: no session lifecycle, no batch jobs, no SSE plumbing.

```
Text:          POST /api/lens/query ───────────────────────────────► response
Image(s):      Upload file(s) → POST /api/lens/query (file_ids) ────► response
Video (.mp4):  Upload file → POST /api/lens/query + max_frames ─────► response
Video frames:  POST /api/lens/query (frames + query_metadata) ──────► response
```

**Integrations** reuse the Direct Query API for the Newton side and call the partner model on its own API — for Nemotron, NVIDIA's hosted `https://integrate.api.nvidia.com/v1/chat/completions`, with its own `NVIDIA_API_KEY`:

```
Omega:     POST /api/lens/query per channel ──► embeddings ──► local KNN ──► verdict
Nemotron:  verdict + window stats ──► POST chat/completions ──► validated brief
```

**Agents — the Agents API.** You upload an input, run a pre-packaged bundle or a canonical blueprint, poll, and download the output; the model runs server-side.

```
Upload:   POST /api/files ───────────────────────────────────► file_id
Resolve:  GET  /api/agents/bundles?query=<name> ──────────────► bnd_…
          (or GET /api/agents/blueprints/<key> → POST /api/agents/bundles)
Run:      POST /api/agents/bundles/<bnd_…>/run ──────────────► agt_…
Poll:     GET  /api/agents/instances/<agt_…>/logs | /events
Collect:  GET  /api/agents/instances/<agt_…>/results ────────► output file

Fit (OSM Path 2: your own labelled data):
          POST /api/agents/optimizations ────────────────────► opt_… (trials, scored)
          POST /api/agents/optimizations/<opt_…>/trials/<otr_…>/promote ► blp_…
          POST /api/agents/evals ────────────────────────────► evl_… (the test score)
```

A run can report `completed` before its last output file is fully written (a known platform
issue): check that an output reaches its input's end before using it, as the OSM runners do.

Every reference script that calls the platform is built on the [official Archetype AI python client](https://github.com/archetypeai/python-client) (`pip install archetypeai`), which owns auth, retries and endpoint mounting. Each skill declares it in `references/requirements.txt`. **One exception:** the OSM skill's Path 2 (`references/osm_lifecycle/`) drives the Optimizations, Evals and promote endpoints through a small stdlib HTTP helper of its own, rather than the client's `agents.optimizations` and `agents.evals` namespaces.

## API Base URL

```
https://platform.u1.archetypeai.app
```

**Every surface lives under one versionless `/api` prefix** — `/api/agents`, `/api/files`, `/api/lens/query` — so a single client serves them all. Set `ATAI_API_ENDPOINT` to the bare root above; the reference scripts normalise it and append `/api`, and accept a `/v0.5` or `/api` suffix too, so one `.env` serves every skill here.

**A wrong prefix is the most common way to lose a run, because it says nothing:** a 404 comes back with an empty body, which the client raises as `ApiError: {}` — no status, no URL, no message. To tell a wrong path from a wrong key, repeat the call **without the auth header**: 404 means the route isn't registered, 401 means it exists and your key or scope is the problem.

Note that `/api/query` is not a route. The Direct Query API keeps its `lens/` namespace at `/api/lens/query` — that is the current path, not a deprecated one. The python client deprecates the `client.lens` *accessor* in favour of `client.agents` for agent workflows; the HTTP route is unaffected.

## License

Apache-2.0
