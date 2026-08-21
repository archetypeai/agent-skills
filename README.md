# Archetype AI Agent Skills

[![skills.sh](https://www.skills.sh/b/archetypeai/agent-skills)](https://www.skills.sh/archetypeai/agent-skills)

Agent skills for building applications with [Archetype AI's Newton](https://www.archetypeai.io/) — a real-time sensor intelligence platform that understands physical world data through foundation models.

Inspired by [mongodb/agent-skills](https://github.com/mongodb/agent-skills).

## Skills

Three groups: **Agents** run a maintained pipeline server-side over the Agents API, **Models** call Newton directly on `/query` and leave the orchestration to you, and **Design** covers the demo front-end.

### Agents

Managed end-to-end pipelines — upload an input, run a pre-packaged bundle or canonical blueprint, poll, download the output. Nothing to fit, no model to host. The three sensor agents differ by **what you already have labelled**: every regime (OSM), a handful of examples of one named fault (RED), or nothing but normal operation (AD).

| Skill | Description |
|-------|-------------|
| [atai-operational-state-monitoring-agent](skills/atai-operational-state-monitoring-agent/) | Run the managed OSM agent over the Agents API — resolve the pre-packaged "OSM Quick Start" bundle by name (classifier + windowing already pinned), run one agent per input CSV, poll, and download per-window state predictions |
| [atai-rare-event-detection-agent](skills/atai-rare-event-detection-agent/) | Run the managed RED agent over the Agents API — resolve the pre-packaged "RED Quick Start" bundle by name (nearest-prototype classifier + windowing already pinned), run one agent per input CSV, poll, and download per-window rare-event predictions |
| [atai-anomaly-discovery-agent](skills/atai-anomaly-discovery-agent/) | Run the managed Anomaly Discovery agent over the Agents API — resolve the pre-packaged "AD Quick Start" bundle by name (fitted LOF detector + threshold already pinned), run one agent per input CSV, poll, and download a per-window anomaly score. For assets with **no fault history**: fitted on normal-only data, so everything it flags is something it was never shown |
| [atai-manual-generation-agent](skills/atai-manual-generation-agent/) | Run the managed Manual Generation (MGA) agent over the Agents API — upload a procedure video, run the canonical `mga` blueprint, and get back an ordered, timestamped manual with each step traceable to a time range in the source |
| [atai-task-verification-agent](skills/atai-task-verification-agent/) | Run the managed Task Verification (TVA) agent over the Agents API — upload a recording **and** a reference procedure (an SOP), and get back a per-step PASSED / FAILED / MISSING verdict with a timestamp and a reason for each step. The SOP is a runtime input, so one bundle serves every procedure |

### Models

Direct Query API — one stateless `POST /query` per request, for when you want control over the pipeline or the raw vectors to build on.

| Skill | Description |
|-------|-------------|
| [atai-newton-omega-model-data-prep](skills/atai-newton-omega-model-data-prep/) | Clean, split, and featurize multivariate time-series data before the Omega model — gap-aware blocking + imputation, out-of-time train/test split, and joint-state (X, y) featurization |
| [atai-newton-omega-model](skills/atai-newton-omega-model/) | Get time-series embeddings from the Omega encoder (`OmegaEncoder::omega_embeddings_1_4`) over `/query` — one stateless call per channel, fanned out in parallel — for client-side KNN classification, anomaly scoring, and similarity search |
| [atai-newton-fusion-model](skills/atai-newton-fusion-model/) | Call the Newton C 2.6 fusion model on `/query` with text, image, or video in one stateless POST — the first C checkpoint to reason over video frames via `/query` |

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
| [Traffic Monitor](https://github.com/archetypeai/archetypeai-traffic-demo) | `atai-newton-fusion-model` | Live Caltrans CCTV feed → C 2.6 vision. Samples a short burst of frames per interval and sends it as one multi-frame `/query` clip to reason over traffic flow, incidents, and conditions. |
| [Wildfire Watch](https://github.com/archetypeai/archetypeai-wildfire-demo) | `atai-newton-fusion-model` | 1,200+ ALERTCalifornia cameras → C 2.6 vision. Per-camera smoke/fire/haze detection, a zone scan that batches frames per `/query`, and text-only zone Q&A. |
| [Earthquake monitor](https://github.com/archetypeai/archetypeai-earthquake-demo) | `atai-newton-fusion-model` | Live USGS feed → C 2.6 **text reasoning**. Formats the current quakes as structured text and asks Newton (stateless `/query`) to surface aftershock sequences, spatial clustering, and ranked regional risk — alongside an interactive world map. |
| [SWaT water treatment](https://github.com/archetypeai/archetypeai-swat-demo-direct-query) | `atai-newton-omega-model` + `atai-newton-fusion-model` | Six-stage plant anomaly detection: Omega per-channel embeddings + client-side KNN, plus C 2.6 text reasoning for operator action suggestions. |
| [Wind turbine monitor](https://github.com/archetypeai/archetypeai-wind-turbine-demo) (Python/Flask) | `atai-newton-omega-model` + `atai-newton-omega-model-data-prep` | Penmanshiel wind-farm SCADA anomaly detection: Omega per-channel embeddings + local KNN against a leakage-free n-shot library precomputed offline. Replays 3 months of telemetry and detects a real frequency-converter fault on one turbine against its healthy peer. |
| [Drilling state monitor](https://github.com/archetypeai/archetypeai-drilling-demo) | `atai-newton-omega-model` + `atai-newton-omega-model-data-prep` | Equinor Volve North Sea well SCADA: per-channel Omega embeddings + local KNN classify each window as drilling / not-drilling, against a leakage-free n-shot library precomputed offline from held-out reference wells. Replays real well telemetry with live accuracy vs ACTC ground truth. |
| [Grid monitor](https://github.com/archetypeai/archetypeai-grid-demo) | `atai-newton-fusion-model` | Live CAISO power-grid feed → C 2.6 **text reasoning**. Formats 5-minute demand/supply data as structured text and asks Newton (stateless `/query`) about duck-curve dynamics, evening ramp, renewable share, and grid-stress risk, with supply/demand charts. |
| [WiFi occupancy monitor](https://github.com/archetypeai/archetypeai-wifi-demo) | `atai-newton-fusion-model` | Residential gateway WiFi telemetry → C 2.6 **text reasoning**. Sends an anonymized 15-minute per-device flow/byte/protocol snapshot as JSON via stateless `/query`; Newton infers home occupancy (OCCUPIED … EMPTY) from traffic patterns alone — no device-type labels, no online flag. |

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
/atai-operational-state-monitoring-agent # Every operating regime, from a full labelled library
/atai-rare-event-detection-agent   # One named fault, from a handful of labelled examples
/atai-anomaly-discovery-agent      # Normal-only fit; per-window anomaly score, no fault history needed
/atai-manual-generation-agent      # Procedure video -> ordered, timestamped manual
/atai-task-verification-agent      # Recording + SOP -> per-step PASSED / FAILED / MISSING

# Models — Direct Query API
/atai-newton-omega-model-data-prep # Clean / split / featurize time-series before the Omega model
/atai-newton-omega-model           # Omega time-series embeddings + client-side KNN via /query
/atai-newton-fusion-model          # Multimodal (text/image/video) queries on the C 2.6 fusion model

# Design
/atai-design-system                # Scaffold + build a Newton demo front-end with the Design System
```

## Architecture

Two API surfaces, one per skill family.

**Models — the Direct Query API.** One stateless POST to `/query` per request: no session lifecycle, no batch jobs, no SSE plumbing.

```
Text:          POST /query ──────────────────────────────────► response
Image(s):      Upload file(s) → POST /query (file_ids) ───────► response
Video (.mp4):  Upload file → POST /query + max_frames ────────► response
Video frames:  POST /query (frames + query_metadata) ─────────► response
```

**Agents — the Agents API.** You upload an input, run a pre-packaged bundle or a canonical blueprint, poll, and download the output; the model runs server-side.

```
Upload:   POST /v0.5/files ───────────────────────────────────► file_id
Resolve:  GET  /agents/bundles?query=<name> ──────────────────► bnd_…
          (or GET /agents/blueprints/<key> → POST /agents/bundles)
Run:      POST /agents/bundles/<bnd_…>/run ───────────────────► agt_…
Poll:     GET  /agents/instances/<agt_…>/logs | /events
Collect:  GET  /agents/instances/<agt_…>/results ─────────────► output file
```

Every reference script is built on the [official Archetype AI python client](https://github.com/archetypeai/python-client) (`pip install archetypeai`), which owns auth, retries and endpoint mounting. Each skill declares it in `references/requirements.txt`.

## API Base URL

```
https://api.u1.archetypeai.app
```

**The two surfaces are mounted differently, and this is the most common way to lose a run:** the files API is versioned (`/v0.5/files`) while the Agents API is **versionless** (`/agents`, never `/v0.5/agents`). The client handles both from one value — pass the endpoint *with* the `/v0.5` suffix and it strips the version for `/agents` itself. Pass a bare root and uploads fail with an empty `ApiError: {}` while bundle calls keep working, which points at nothing. The agent runners normalise either form, so one `.env` serves every skill here.

## License

Apache-2.0
