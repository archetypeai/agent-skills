# Archetype AI Agent Skills

Agent skills for building applications with [Archetype AI's Newton](https://www.archetypeai.io/) — a real-time sensor intelligence platform that understands physical world data through foundation models.

Inspired by [mongodb/agent-skills](https://github.com/mongodb/agent-skills).

## Skills

| Skill | Description |
|-------|-------------|
| [newton-setup](skills/newton-setup/) | Configure Newton API access, environment setup, and SDK initialization |
| [newton-query-prompting](skills/newton-query-prompting/) | Prompt-engineering patterns for the `/query` text-reasoning endpoint (structured output, topology validation, contamination avoidance) |
| [newton-activity-monitor](skills/newton-activity-monitor/) | Vision-based analysis and Q&A using the Activity Monitor Lens |
| [newton-sensor-streaming](skills/newton-sensor-streaming/) | Real-time sensor data ingestion patterns (BLE, OBD2, serial, etc.) |
| [newton-batch-upload](skills/newton-batch-upload/) | Upload large files (> 255 MB) via multipart presigned URLs |
| [newton-activity-detection-batch](skills/newton-activity-detection-batch/) | Run text-in / text-out batch jobs on the C model via the `activity-detection` pipeline — narratives over large CSV/log datasets, with MapReduce / hierarchical-reduce patterns and quality-cliff guidance |
| [atai-newton-fusion-model](skills/atai-newton-fusion-model/) | Call the Newton C 2.6 fusion model on `/query` with text, image, or video in one stateless POST — the first C checkpoint to reason over video frames via `/query` |

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
/newton-setup             # Set up API access
/newton-query-prompting   # Engineer /query prompts for structured output
/newton-activity-monitor  # Analyze visual data
/newton-sensor-streaming  # Connect hardware sensors
/newton-batch-upload      # Upload large files (> 255 MB)
/newton-activity-detection-batch  # Batch text generation on the C model (narratives over CSV/log data)
/atai-newton-fusion-model # Multimodal (text/image/video) queries on the C 2.6 fusion model
```

## Architecture

Newton operates through **Lens Sessions** — persistent inference pipelines that process sensor data in real-time:

```
Real-time:  Sensor Data → Upload File → Create Session → Set Input Stream → SSE Results
                                              ↑
                                      Focus/N-shot Examples

Batch:      Upload Files → Create Batch Job → Poll Status → View Results
```

### Two Core Lenses

| Lens | ID | Input | Output |
|------|----|-------|--------|
| **Machine State** | `lns-1d519091822706e2-bc108andqxf8b4os` | CSV time-series | Classification labels + confidence scores |
| **Activity Monitor** | `lns-fd669361822b07e2-bc608aa3fdf8b4f9` | Video | Natural language text |

- **Machine State Lens** — Classifies time-series sensor data using n-shot learning (e.g., stressed vs. relaxed, normal vs. anomaly). Provide labeled CSV examples and it classifies new data via KNN over sliding windows.
- **Activity Monitor Lens** — Analyzes visual data (charts, dashboards, camera feeds) and answers natural language questions using a 2B-parameter vision model.

## Example Projects

These projects demonstrate the patterns covered by these skills:

- [corsense-hrv](https://github.com/NathanNam/corsense-hrv) — Real-time HRV stress detection using BLE heart rate monitors + Newton Machine State & Activity Monitor
- [obd2-scanner](https://github.com/NathanNam/obd2-scanner) — Browser-based vehicle diagnostics via OBD2/ELM327 + Newton health classification & chat
- [archetypeai-traffic-demo](https://github.com/archetypeai/archetypeai-traffic-demo) — Live traffic monitoring via Caltrans HLS camera + Newton vision (lens session + model.query)
- [archetypeai-wildfire-demo](https://github.com/archetypeai/archetypeai-wildfire-demo) — Wildfire detection across 1,200+ ALERTCalifornia cameras + Newton vision
- [archetypeai-earthquake-demo](https://github.com/archetypeai/archetypeai-earthquake-demo) — Real-time USGS earthquake analysis + Newton text reasoning (direct query API)
- [archetypeai-grid-demo](https://github.com/archetypeai/archetypeai-grid-demo) — California power grid monitoring via CAISO supply/demand data + Newton text reasoning
- [archetypeai-drilling-demo](https://github.com/archetypeai/archetypeai-drilling-demo) — Drilling state classification from 14 North Sea wells + Newton Machine State Lens (SSE streaming)
- [archetypeai-swat-demo](https://github.com/archetypeai/archetypeai-swat-demo) — 6-stage water treatment plant anomaly dashboard with parallel per-stage Machine State Lens sessions + `/query`-generated operator suggestions (reference implementation for both `newton-machine-state` parallel-subsystem pattern and `newton-query-prompting`)
- [archetypeai-nasa-jpl-telemanom-demo](https://github.com/archetypeai/archetypeai-nasa-jpl-telemanom-demo) — NASA SMAP/MSL spacecraft telemetry anomaly explorer (Hundman et al., KDD 2018) + Newton Machine State Lens. Single-channel mode (telemetry + 3 MI-picked mode flags) vs. subsystem mode (4 sibling-channel sensors with union-of-flags GT), with honest held-out F1/Precision/Recall, multi-segment normal focus, adaptive window sizing, and vendored `omega-1-4-preflight` static checks.
- [archetypeai-batch-examples-volve](https://github.com/archetypeai/archetypeai-batch-examples-volve) — Batch upload, inference, and evaluation with Volve drilling data (Machine State + Activity Detection)
- [archetypeai-wind-turbine-demo](https://github.com/archetypeai/archetypeai-wind-turbine-demo) — Side-by-side wind turbine anomaly monitor (Penmanshiel SCADA / Cubico / Zenodo). Flask + Jinja + vanilla-CSS port of the design system; reference implementation for the `MultiplexNewtonSession` pattern (one shared session, FIFO push-tag routing) when the account's lens-runner pool is quota = 1
- [archetypeai-batch-examples-ghost-iot](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot) — 1 GB WiFi flow CSV folded into 9 daily narratives via six MapReduce stages on the `activity-detection` batch pipeline (reference implementation for `newton-activity-detection-batch` — cliff sweep, hierarchical reduce, N-way positional split, content-key joins)

## API Base URL

```
https://api.u1.archetypeai.app/v0.5
```

## License

Apache-2.0
