# Archetype AI Agent Skills

Agent skills for building applications with [Archetype AI's Newton](https://www.archetypeai.io/) — a real-time sensor intelligence platform that understands physical world data through foundation models.

Inspired by [mongodb/agent-skills](https://github.com/mongodb/agent-skills).

## Skills

| Skill | Description |
|-------|-------------|
| [atai-newton-fusion-model](skills/atai-newton-fusion-model/) | Call the Newton C 2.6 fusion model on `/query` with text, image, or video in one stateless POST — the first C checkpoint to reason over video frames via `/query` |
| [atai-newton-omega-model](skills/atai-newton-omega-model/) | Get time-series embeddings from the Omega encoder (`OmegaEncoder::omega_embeddings_1_4`) over `/query` — one stateless call per channel, fanned out in parallel — for client-side KNN classification, anomaly scoring, and similarity search |
| [atai-newton-omega-model-data-prep](skills/atai-newton-omega-model-data-prep/) | Clean, split, and featurize multivariate time-series data before the Omega model — gap-aware blocking + imputation, out-of-time train/test split, and joint-state (X, y) featurization |
| [atai-operational-state-monitoring-agent](skills/atai-operational-state-monitoring-agent/) | Run the managed OSM agent over the Agent API — bundle the canonical `osm` blueprint with a fitted classifier artifact, run one agent per input CSV, poll, and download per-window state predictions |

More skills are in review and will be added to this table as they land.

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

### Claude Code

```bash
# Add as global skills (available in all projects)
cp -r skills/* ~/.claude/skills/

# Or add to a specific project
cp -r skills/* your-project/.claude/skills/
```

### Invoke a Skill

```
/atai-newton-fusion-model          # Multimodal (text/image/video) queries on the C 2.6 fusion model
/atai-newton-omega-model           # Omega time-series embeddings + client-side KNN via /query
/atai-newton-omega-model-data-prep # Clean / split / featurize time-series before the Omega model
```

## Architecture

The skills here target Newton's **Direct Query API** — one stateless POST to `/query` per request, no session lifecycle, no batch jobs, no SSE plumbing:

```
Text:          POST /query ──────────────────────────────────► response
Image(s):      Upload file(s) → POST /query (file_ids) ───────► response
Video (.mp4):  Upload file → POST /query + max_frames ────────► response
Video frames:  POST /query (frames + query_metadata) ─────────► response
```

The reference scripts are built on the [official Archetype AI python client](https://github.com/archetypeai/python-client) (`pip install archetypeai`).

## API Base URL

```
https://api.u1.archetypeai.app/v0.5
```

## License

Apache-2.0
