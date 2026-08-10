---
name: atai-manual-generation-agent
description: >
  Run Archetype AI's managed Manual Generation (MGA) agent over the Agent
  API — upload a procedure video, create a bundle from the canonical `mga`
  blueprint, run it (one agent per video), poll status + audit logs, and
  download an ordered, timestamped manual. Use this skill when the user
  has a recording of a procedure (a repair, an assembly, a workflow) and
  wants the platform to turn it into step-by-step instructions traceable
  back to the video. Covers video suitability (the ~5-minute ceiling, the
  audio-track requirement), the bundle request shape (`max_frames`, and
  the `max_new_tokens`/`prompt` values the active blueprint does NOT
  expose), the run/poll/results lifecycle, the output JSONL schema
  (`step, instruction, frame_start/end, timestamp_start/end`), and scoring
  against reference step annotations. Do NOT use for verifying that a task
  was performed correctly against a reference procedure (that is the `tva`
  blueprint), for one-shot multimodal questions over a clip (that is
  `atai-newton-fusion-model` via `/query`), or for time-series state
  classification (`atai-operational-state-monitoring-agent`).
---

# MGA Agent — Managed Manual Generation via the Agent API

The MGA agent turns a procedure video into an ordered manual with timestamps. You hand the platform an `.mp4`; it samples frames, transcribes the audio, fuses both in one pass, and returns steps you can trace back to the recording:

```
video ─► sample frames ─┐
                        ├─► newton-fusion f1-0 ─► parse ─► steps + timestamps
        whisper ASR ────┘
```

MGA V1 is **zero-shot**. The `mga` blueprint pins its own models (`newton-fusion:1.0` and `whisper:large-v3`), so unlike `osm`/`red` there is **no classifier to fit and no `artifacts` map to pass**. A run is upload → bundle → run → download.

## When to Apply

**Use when** the user wants a written procedure extracted from a recording — a maintenance manual from a repair video, an SOP from a screen capture, work instructions from an assembly clip — and wants each step tied to a time range in the source.

**Do NOT use when:**

| Need | Use instead |
|---|---|
| Was this task done correctly, against a known procedure? | the `tva` (Task Verification) blueprint |
| One question about a clip, stateless | `atai-newton-fusion-model` (`/query`) |
| Classify sensor windows into operational states | `atai-operational-state-monitoring-agent` |
| The video is longer than ~5 minutes | chunk it first — see Step 1 |

## Endpoints

The Agent API is mounted **without a version prefix**; the files API is under `/v0.5`.

```
POST   {endpoint}/v0.5/files                          upload the video
POST   {endpoint}/agents/bundles                      create a bundle   ← PLURAL
POST   {endpoint}/agents/bundles/{bundle_id}/run      start a run       ← returns 202
GET    {endpoint}/agents/instances/{agent_id}/logs    the real log stream
GET    {endpoint}/agents/instances/{agent_id}/results output refs
POST   {endpoint}/agents/instances/{agent_id}/cancel  stop a run
```

`POST /agents/bundle` (singular) returns **404**. Some older clients still use it.

## ⚠️ The output-token cap — read before anything else

The active `mga` blueprint caps generation at a hardcoded **`max_new_tokens: 256`** and does not expose the value. On a 173-second video with 11 documented steps that yields a **truncated manual**: 6 steps covering 16–85 s, with the last one severed mid-clause.

| | active blueprint | `max_new_tokens: 2048` | + a coverage prompt |
|---|---|---|---|
| Steps emitted | 6 | 10 | 19 |
| Coverage | 16–85 s of 173 s | 16–120 s | 11–139 s |
| Reference steps found | **4 / 11** | **10 / 11** | 10 / 11 |
| Final step | cut mid-clause | clean | clean |

**Setting `max_new_tokens` on the active blueprint fails silently.** The bundle returns **HTTP 201** and echoes the value back in `values`, then ignores it — the fusion node's config is `{model}` only and never references `${values.max_new_tokens}`. Three runs with and without it produced byte-identical output.

A superseded version of the same blueprint (`blp_76kyqm4vjp9pt8tvfz8tks7x6t`, `is_active: false`) does wire it, plus seven other generation parameters and `${values.prompt}`. It still accepts bundles and runs, so it is usable **for diagnosis**, but nothing built on an inactive blueprint should ship. `references/run_mga_agent.py` targets it by default and says so on every run.

**Check before you spend a run.** A value is honoured only if it is declared in the blueprint's `values` *and* referenced as `${values.<key>}` by some node or connector:

```python
doc = GET(f"{endpoint}/agents/blueprints/{blueprint}")["document"]
wired = json.dumps({"nodes": doc["nodes"], "connectors": doc["connectors"]})
inert = [k for k in my_values
         if k not in doc["values"] or f"${{values.{k}}}" not in wired]
```

## Step 1 — Choose a video

The platform validates none of this for you, and getting it wrong costs ~15 minutes per attempt.

| Requirement | Why |
|---|---|
| **Under ~5 minutes** | Longer videos fail outright. A 563 s clip ran 5 min 13 s then died: `pipeline execution failed: Newton generation event stream closed without a terminal event`, exit 1. |
| **Has an audio track** | `AudioReaderNode` → whisper → `prompt.audio` is half the pipeline. Spoken-only content (safety cautions) shows up in the output, so a silent video loses real information. |
| **Reference steps, to score it** | Otherwise there is nothing to evaluate against. |

The ~5-minute ceiling is **not** a context limit: at `max_frames: 16` the video occupies 392 visual tokens, ~0.15% of F1-0's 256K window.

Source resolution barely matters — every frame is resized to `size × size` (default 224), so 360p is no worse than 1080p, only smaller and faster to upload.

## Step 2 — Upload the video

```python
POST {endpoint}/v0.5/files          # multipart/form-data, field name "file"
→ {"is_valid": true, "file_id": "clip.mp4", "file_uid": "fil_…"}
```

The **declared** `Content-Type` is enforced against a MIME allowlist, not the bytes — send `video/mp4`. Use `file_id` (the filename) in the run payload, not `file_uid`.

## Step 3 — Create a bundle

**No `artifacts` map** — the blueprint pins its own models. This is the main shape difference from `osm`/`red`.

```json
POST {endpoint}/agents/bundles
{
  "blueprint": "mga",
  "name": "manual generation run",
  "values": {"max_frames": 64}
}
→ 201 {"id": "bnd_…", "status": "ready", "is_canonical": false}
```

| Value | Default | Notes |
|---|---|---|
| `max_frames` | 16 | Frames uniformly sampled across the whole video. 64 is the reader/preprocessor batch size. **No validation at all** — 16, 512, 1024 and `-1` are all accepted as `ready`. The arithmetic ceiling from `preprocessor_max_pixels` (24 Mi at `size: 224`) is 501 frames. |
| `size` | 224 | Each frame resized to a square. |
| `parser_compute_stats` | false | Attaches template-conformance stats — useful for diagnosing a parser mismatch. |
| `max_new_tokens` | **256, not exposed** | See the cap section above. |
| `prompt` | **not exposed** | See below. |

### On `prompt`

The instruction is `connectors.source.config.default_text`, hardcoded on the active blueprint, with `text_extensions: []` disabling text inputs entirely. It was exposed once and **rolled back**, for a good reason: `ManualGenerationResultsParserNode` owns the output template, so **a prompt that specifies its own output format makes the parser return zero steps.**

That is about format, not about custom prompts generally. A prompt saying what to *cover* parses cleanly:

> Generate a concise, ordered list of every distinct step performed in this video, covering the procedure from the first action to the last. Include brief steps and steps that are repeated. Use up to 20 steps. Keep each step to at most 15 words. Use both what is shown and what is said.

A prompt saying how to *lay the output out* (markdown headings, a `## Steps` section) does not. If you ever set `prompt`, **never specify an output format.**

## Step 4 — Run the bundle

```json
POST {endpoint}/agents/bundles/{bundle_id}/run
{"connectors": {"source": [{"type": "file", "id": "clip.mp4", "format": "mp4"}]}}
→ 202 {"id": "agt_…", "status": "running"}
```

**202, not 201.** A client that treats only 201 as success reports a failure while the agent runs unattended on the GPU with nothing collecting its output.

One agent per video. Dev serializes these jobs — four concurrent submissions produced 799–817 s queue waits, not parallelism.

## Step 5 — Poll until terminal

Poll **`/logs`**, not `/events`. `/events` returns only `run started` and `dispatched to JOS`; `/logs` is what the console shows — model load timings, per-input progress, and the actual error.

**Do not trust the `status` field. It lies in both directions**: it stayed `running` for 20+ minutes after a pod terminated with `exit=1`, and `running` after a job had completed. Judge terminality from the log stream:

```python
TERMINAL = ("pod.terminated", "job.completed", "job.failed", "job.canceled")
rows = GET(f"{endpoint}/agents/instances/{agent_id}/logs?limit=500")["data"]
rows.sort(key=lambda r: r["created_at"])
done   = rows and rows[-1]["event_type"] in TERMINAL
failed = any(r["level"] == "ERROR" for r in rows)
```

Between `input_started` and the terminal event there are **zero log rows** across 5–7 minutes — no frame counts, no transcript length, no per-node timing. A failure in that window cannot be attributed from the logs.

## Step 6 — Fetch results

```python
GET {endpoint}/agents/instances/{agent_id}/results
→ {"data": [{"data": {"filename": "…jsonl", "ref": "/files/download/…"}}]}
```

The `ref` is a **relative** platform path that resolves under `/v0.5` and needs the bearer token. Run outputs do **not** expire.

## Output JSONL — one record per video

```json
{"id":"clip","results":[
  {"step":0,"instruction":"Set up a warning triangle and turn on the hazard lights.",
   "frame_start":400.0,"frame_end":625.0,
   "timestamp_start":16.0,"timestamp_end":25.0}, …]}
```

Four properties the blueprint does not advertise:

- **Timestamps in seconds come free.** The blueprint exposes only `parser_output_frame_indices`, but `timestamp_start`/`timestamp_end` are emitted anyway — no frames→seconds conversion needed, and no second model pass to recover `[MM:SS]`.
- **Frame indices are SOURCE frames, not sampled frames.** `frame = timestamp × source_fps` exactly. Sampled space would cap at `max_frames - 1`.
- **`step` is 0-based.** A 10-step manual reports `step` 0–9. If your reference steps are 1-based, a naive join is off by one and produces plausible-looking wrong alignments rather than an error.
- **The output tiles the timeline contiguously.** Every step's `timestamp_end` is exactly the next step's `timestamp_start`, on integer-second boundaries. **MGA does segmentation, not event detection** — there is no "nothing happening here" span, so every moment is labelled. That is why a spoken caution gets its own numbered step: it has nowhere else to go, and there is no `kind: action | caution` field.

## Runtime

| Phase | Cost |
|---|---|
| Queue | ~12 s solo; **799–817 s** if other runs are in flight |
| whisper download + load | ~35–40 s + ~4 s |
| **newton-fusion download + load** | **~5 min + ~1 min 40 s** |
| Processing | ~2.5× realtime (173 s video → ~7 min) |
| **Total** | **~15 min for a 3-minute video** |

**Cold start is ~7.5 min on every run and is never cached.** Budget accordingly: a sweep of six configurations is ~45 minutes of pure model loading before any inference.

## Common Pitfalls

| Symptom | Cause |
|---|---|
| `404` on bundle creation | `/agents/bundle` is singular; use `/agents/bundles` |
| Client reports failure, run proceeds anyway | `/run` returns **202**, not 201 |
| `max_new_tokens` has no effect | Not wired on the active blueprint. Accepted (201) and ignored |
| Parser returns **0 steps** | A custom `prompt` specified an output format; the parser owns the template |
| Run "hangs" at `running` forever | The `status` field lags; check `/logs` for `pod.terminated` |
| `Newton generation event stream closed without a terminal event` | Usually a video longer than ~5 minutes |
| Manual stops halfway through the video | The 256-token cap — check whether the last step ends mid-clause |
| Blueprint key resolves to a version without the values you expect | `GET /agents/blueprints` returns **only `is_active: true`** blueprints; superseded versions still exist and still run, reachable by `blueprint_id` |

## Cleanup

Dev has **one GPU** and serializes jobs, so an abandoned run blocks everyone:

```python
POST {endpoint}/agents/instances/{agent_id}/cancel
```

**Do not send `DELETE` to an instance URL expecting a no-op** — it returns 204 and removes the run. Bundles are cheap to leave; runs are not.

## Local Setup

```sh
# No third-party deps — references/run_mga_agent.py is stdlib-only.
# yt-dlp is optional, only to fetch the sample video from YouTube.

# Drop a .env next to where you run (BOTH variables required, no default endpoint;
# note: NO /v0.5 suffix — the script mounts /agents and /v0.5/files itself):
#   ATAI_API_KEY=<dev API key>
#   ATAI_API_ENDPOINT=https://api.dev.u1.archetypeai.app

python3 references/run_mga_agent.py --video my_procedure.mp4
python3 references/run_mga_agent.py --video my_procedure.mp4 --blueprint mga  # the active one
python3 references/run_mga_agent.py --score references/sample_data/mga-output-max_new_tokens2048-coverage-prompt.jsonl \
    --reference references/sample_data/40567_i2JWkDyg26A_reference_steps.csv
```

## File Layout

```
skills/atai-manual-generation-agent/
├── SKILL.md
├── references/
│   ├── .env.example
│   ├── run_mga_agent.py          stdlib-only runner: upload → bundle → run → logs → results → score
│   └── sample_data/
│       ├── README.md             attribution, and why no video ships here
│       ├── 40567_i2JWkDyg26A_reference_steps.csv
│       ├── mga-output-truncated-active-blueprint.jsonl
│       ├── mga-output-max_new_tokens2048.jsonl
│       └── mga-output-max_new_tokens2048-coverage-prompt.jsonl
└── tests/
    └── test_references.py        network-free
```

Worked end-to-end example, including the video-selection stage and an offline scorer:
[manual-generation-agent-example](https://github.com/archetypeai/manual-generation-agent-example).
