# Archetype AI Agent Skills

Agent skills for building applications with [Archetype AI's Newton](https://www.archetypeai.io/) — a real-time sensor intelligence platform that understands physical world data through foundation models.

Inspired by [mongodb/agent-skills](https://github.com/mongodb/agent-skills).

## Skills

| Skill | Description |
|-------|-------------|
| [atai-newton-fusion-model](skills/atai-newton-fusion-model/) | Call the Newton C 2.6 fusion model on `/query` with text, image, or video in one stateless POST — the first C checkpoint to reason over video frames via `/query` |
| [atai-newton-omega-model](skills/atai-newton-omega-model/) | Get time-series embeddings from the Omega encoder (`OmegaEncoder::omega_embeddings_1_4`) over `/query` — one stateless call per channel, fanned out in parallel — for client-side KNN classification, anomaly scoring, and similarity search |
| [atai-newton-omega-model-data-prep](skills/atai-newton-omega-model-data-prep/) | Clean, split, and featurize multivariate time-series data before the Omega model — gap-aware blocking + imputation, out-of-time train/test split, and joint-state (X, y) featurization |
| [atai-design-system](skills/atai-design-system/) | Build a Newton demo front-end with the Archetype AI Design System — scaffold via the `ds` CLI (`@archetypeai/ds-cli`) and compose the published Svelte 5 primitives + OKLCH tokens (`@archetypeai/ds-{lib-tokens,ui-svelte-console,ui-svelte-labs}`) instead of hand-rolling UI |

More skills are in review and will be added to this table as they land.

## Example apps

End-to-end demos built on these skills, on the Direct Query API (SvelteKit unless noted):

| Demo | Skills | What it demonstrates |
|------|--------|----------------------|
| [Traffic Monitor](https://github.com/archetypeai/archetypeai-traffic-demo) | `atai-newton-fusion-model` | Live Caltrans CCTV feed → C 2.6 vision. Samples a short burst of frames per interval and sends it as one multi-frame `/query` clip to reason over traffic flow, incidents, and conditions. |
| [Wildfire Watch](https://github.com/archetypeai/archetypeai-wildfire-demo) | `atai-newton-fusion-model` | 1,200+ ALERTCalifornia cameras → C 2.6 vision. Per-camera smoke/fire/haze detection, a zone scan that batches frames per `/query`, and text-only zone Q&A. |
| [SWaT water treatment](https://github.com/archetypeai/archetypeai-swat-demo-direct-query) | `atai-newton-omega-model` + `atai-newton-fusion-model` | Six-stage plant anomaly detection: Omega per-channel embeddings + client-side KNN, plus C 2.6 text reasoning for operator action suggestions. |
| [Wind turbine monitor](https://github.com/archetypeai/archetypeai-wind-turbine-demo) (Python/Flask) | `atai-newton-omega-model` + `atai-newton-omega-model-data-prep` | Penmanshiel wind-farm SCADA anomaly detection: Omega per-channel embeddings + local KNN against a leakage-free n-shot library precomputed offline. Replays 3 months of telemetry and detects a real frequency-converter fault on one turbine against its healthy peer. |

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
/atai-design-system                # Scaffold + build a Newton demo front-end with the Archetype AI Design System
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
