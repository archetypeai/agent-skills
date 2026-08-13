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
  audio-track requirement), the bundle request shape (`max_frames`,
  `max_new_tokens`, and the `prompt` the active blueprint does NOT
  expose), the run/poll/results lifecycle, the output JSON schema
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

## ⚠️ Check which values are real — read before anything else

Twice in five days the one setting that decided whether the manual was usable was a setting the blueprint did not expose. Both are exposed again today. Neither fix is a reason to stop checking — the point is the frequency, and that the failure is silent every time.

| | status today | what it cost while unreachable |
|---|---|---|
| `max_new_tokens` | ✅ wired, default **16384** | hardcoded at 256: a 173 s video truncated to 6 steps, the last cut mid-clause |
| `prompt` | ✅ wired (**PLDEV-1730**, canonical `blp_6va7xx…` from 2026-08-12) | hardcoded to "10 steps or less": 10 steps instead of 18, two spoken safety cautions dropped |

**A value is honoured only if it is declared in the blueprint's `values` *and* referenced as `${values.<key>}` by some node or connector.** Nothing in the response tells you: setting an unwired value returns **HTTP 201** and echoes it straight back in `values`. Preflight it — `references/run_mga_agent.py` does this on every run and prints a warning:

```python
doc = GET(f"{endpoint}/agents/blueprints/{blueprint}")["document"]
wired = json.dumps({"nodes": doc["nodes"], "connectors": doc["connectors"]})
inert = [k for k in my_values
         if k not in doc["values"] or f"${{values.{k}}}" not in wired]
```

### `max_new_tokens` is a FLOOR, not a ceiling

The model **reasons before it answers**, and pays for the reasoning out of this same budget. Set it too low and generation ends inside the reasoning block having never emitted an answer. Every signal says success: `job.completed`, no ERROR row, every HTTP call 2xx, and a 38-byte output file containing `{"id": "…", "results": []}`.

Measured on one 173 s video, one variable, 2026-08-13:

| `max_new_tokens` | 2048 | 4096 | **16384** | 32768 | 65536 |
|---|---|---|---|---|---|
| steps | 0 | 0 | **18** | 18 | 18 |

The last three are **byte-identical** — generation is deterministic at temperature 0 — so the threshold is between 4k and 16k and nothing above 16384 changes the answer. **Lowering this does not give you a shorter manual, it gives you no manual.** Send the blueprint's default and leave it alone.

The platform logs a warning for this case, and it is the only thing that makes it diagnosable:

```
WARN parser.running  ManualGenerationResultsParserNode: generation ended inside the
  model's reasoning block, so it never produced an answer; dropping 0 rehearsed
  row(s). Raise `max_new_tokens` if the reasoning was truncated.
```

### What a caller-supplied `prompt` is worth

Same clip, same `max_frames`, only the instruction differing — the blueprint's hardcoded "10 steps or less" versus a coverage instruction:

| | hardcoded | coverage prompt |
|---|---|---|
| steps | 10 | **18** |
| ref steps found at IoU ≥ 0.5 | 4/11 | **7/11** |
| spoken-only cautions | dropped | present |

**Judge a manual by whether it can be followed.** The clearest signal is how many
actions get bundled into one step — an operator cannot tick off "apply the parking
brake" separately from "switch off the ignition", and each step carries one frame
thumbnail however many actions it contains. Scoring the same clip both ways:

| | hardcoded prompt | coverage prompt |
|---|---|---|
| actions per step | 3.2 avg, 7 max | **1.7 avg, 4 max** |
| ref steps at IoU ≥ 0.5 | 4/11 | **7/11** |
| ref steps at IoU ≥ 0.3 | **10/11** | 9/11 |

The gain is in temporal precision, not verbosity. Note the trap in scoring this: at a *loose* threshold the 10-step version scores **higher** (10/11 vs 9/11 at IoU ≥ 0.3), because MGA tiles the timeline contiguously and fewer steps means longer ones that overlap a reference interval more easily. Coarser segmentation flatters loose-threshold recall while being less useful as a manual.

Where the remaining error sits is worth checking per step rather than in aggregate: on the current run, every score below 0.4 came from **two pairs of reference steps collapsing onto one prediction each** (remove-the-nuts with remove-the-wheel; tighten with put-things-back). Every other step scored 0.5–0.69. That is a prompt-shaped problem, not a model ceiling.

## Step 1 — Choose a video

The platform validates none of this for you, and getting it wrong costs ~15 minutes per attempt.

| Requirement | Why |
|---|---|
| **Under ~5 minutes** | Longer videos fail outright. A 563 s clip ran 5 min 13 s then died: `pipeline execution failed: Newton generation event stream closed without a terminal event`, exit 1. |
| **Has an audio track** | `AudioReaderNode` → whisper → `prompt.audio` is half the pipeline. Spoken-only content (safety cautions) shows up in the output, so a silent video loses real information. |
| **Reference steps, to score it** | Otherwise there is nothing to evaluate against. |

The ~5-minute ceiling is **not** a context limit: at `max_frames: 16` the video occupies 392 visual tokens, ~0.15% of F1-0's 256K window.

Source resolution barely matters — every frame is resized to `size × size` (default 224), so 360p is no worse than 1080p, only smaller and faster to upload.

### Longer videos: chunk them

Chunking is also the only way to get more than ~10 steps out of a long procedure, since the step ceiling is per-job (see above). What has worked in production:

- **90 s chunks with 15 s overlap.** 90 s keeps each job well inside the failure band; the overlap is there because a step that straddles a cut is otherwise described from half its evidence in both chunks.
- **Cut on keyframes.** `ffmpeg -ss <t> -i in.mp4 -c copy` rewinds to the previous keyframe and silently shifts the segment — measured **−7.57 s** of drift on a real file. Read the keyframe timestamps first (`ffprobe -select_streams v -skip_frame nokey -show_entries frame=pts_time`) and snap each cut to one, or re-encode.
- **Step from each chunk's actual start, not from `i × stride`.** Snapping to keyframes moves the boundaries; a fixed grid then eats the overlap (15 s → 10 s on our first attempt).
- **One run per chunk, or one run with `source` as an array.** The array form loads the model once instead of per chunk, which is the dominant cost — but results come back **out of order** and every timestamp is **chunk-local**. Tag each input and add the chunk's start offset before merging, or the manual is silently scrambled.
- **Dedupe the overlap conservatively.** Two steps may be merged only if they came from *different* chunks *and* their time spans actually overlap. A similarity-only rule deletes real repeated actions — three genuine "Click on the OK button" steps collapsed into one before we added the time test.

A `pipeline execution failed: Newton generation event stream closed without a terminal event` at 138–175 s of runtime, on inputs that are well under the ceiling, is a **platform-side** failure and not something to fix by shortening further — retry the chunk.

## Step 2 — Upload the video

```python
POST {endpoint}/v0.5/files          # multipart/form-data, field name "file"
→ {"is_valid": true, "file_id": "clip.mp4", "file_uid": "fil_…"}
```

The **declared** `Content-Type` is enforced against a MIME allowlist, not the bytes — send `video/mp4`. Use `file_id` (the filename) in the run payload, not `file_uid`.

## Step 3 — Create a bundle

**No `artifacts` map** — the blueprint pins its own models. This is the main shape difference from `osm`/`red`.

**Send a `prompt`.** Omit it and the blueprint's own instruction applies, which on
current versions has read *"…with 10 steps or less"* — halving the manual for a
reason nothing in the response reveals.

```json
POST {endpoint}/agents/bundles
{
  "blueprint": "mga",
  "name": "manual generation run",
  "values": {
    "max_frames": 64,
    "max_new_tokens": 16384,
    "prompt": "Generate a concise, ordered list of every distinct step performed in this video, covering the procedure from the first action to the last. Include brief steps and steps that are repeated. Use up to 20 steps. Keep each step to at most 15 words. Use both what is shown and what is said."
  }
}
→ 201 {"id": "bnd_…", "status": "ready", "is_canonical": false}
```

| Value | Default | Notes |
|---|---|---|
| `max_frames` | 16 | Frames uniformly sampled across the whole video. 64 is the reader/preprocessor batch size. **No validation at all** — 16, 512, 1024 and `-1` are all accepted as `ready`. The arithmetic ceiling from `preprocessor_max_pixels` (24 Mi at `size: 224`) is 501 frames. |
| `size` | 224 | Each frame resized to a square. |
| `parser_compute_stats` | false | Attaches template-conformance stats — useful for diagnosing a parser mismatch. |
| `max_new_tokens` | 16384 | Exposed and honoured. A **floor**, not a ceiling — 2048 and 4096 both returned an EMPTY manual on a 173 s video. Send the default; above it nothing changes. |
| `prompt` | (blueprint's own) | Exposed again since 2026-08-12 (PLDEV-1730). Preflight it — it has been hardcoded before. See below. |

### On `prompt`

Wired as `${values.prompt}` today, feeding `connectors.source.config.default_text`. It has been hardcoded twice before, so **preflight it rather than assuming** — and note `text_extensions: []`, which means a text input cannot be used as a back door when it is hardcoded.

It was exposed once and **rolled back**, for a reason worth keeping: `ManualGenerationResultsParserNode` owns the output template, so **a prompt that specifies its own output format makes the parser return zero steps.** That is about format, not about custom prompts generally. A prompt saying what to *cover* parses cleanly, and gives 18 steps on the video where the hardcoded instruction gives 10:

> Generate a concise, ordered list of every distinct step performed in this video, covering the procedure from the first action to the last. Include brief steps and steps that are repeated. Use up to 20 steps. Keep each step to at most 15 words. Use both what is shown and what is said.

A prompt saying how to *lay the output out* (markdown headings, a `## Steps` section) does not. **Never specify an output format.**

What a prompt buys, measured on the same clip: **3.2 actions bundled into the average step falls to 1.7**, and spoken-only content becomes its own step at all — with the instruction hardcoded, none of the narrated safety cautions appear. What it did *not* buy: every caution. An earlier run found three, the current one finds one, and since the two differ in blueprint, model build and token budget as well as prompt, nothing isolates the cause.

## Step 4 — Run the bundle

```json
POST {endpoint}/agents/bundles/{bundle_id}/run
{"connectors": {"source": [{"type": "file",
                            "id": "clip-20260813T154418Z-410f.mp4", "format": "mp4"}]}}
→ 202 {"id": "agt_…", "status": "running"}
```

The `id` is whatever `file_id` the upload returned — suffix it, see below.

**202, not 201.** A client that treats only 201 as success reports a failure while the agent runs unattended on the GPU with nothing collecting its output.

One agent per video. **Do not assume dev serializes these jobs.** Four concurrent submissions once produced 799–817 s queue waits, which looked like serialization; three concurrent runs on 2026-08-13 instead came up as three concurrent pods and one was SIGKILLed mid-load (`pod.terminated  Error (exit=137)`). Run them one at a time unless you are deliberately testing this.

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
→ {"data": [{"data": {"filename": "…_0001.json", "file_extension": "json",
                      "ref": "/files/download/…"}}]}
```

The `ref` is a **relative** platform path that resolves under `/v0.5` and needs the bearer token. Run outputs do **not** expire.

## Output JSON — one document per run

**It is JSON, not JSONL.** The results metadata says so (`file_extension: "json"`,
filename ending `.json`) and the body parses as a single object. A reader that
assumes one-record-per-line happens to work on a single-video run and breaks the
moment it does not.

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
| A custom `prompt` has no effect | It is not wired on that blueprint. Accepted (201), echoed back, ignored. Preflight it |
| Manual has ~10 steps no matter how long the video is | A blueprint whose prompt is hardcoded to "10 steps or less". Check `default_text` |
| `job.completed`, no ERROR, and **`results: []`** | `max_new_tokens` below the reasoning threshold. Use the blueprint default (16384) |
| Run dies with `S3 object not found` minutes in | Something re-uploaded your `file_id`. Suffix every upload — see below |
| Parser returns **0 steps** | A custom `prompt` specified an output format; the parser owns the template. Only reachable on a blueprint that wires `prompt` |
| Run "hangs" at `running` forever | The `status` field lags; check `/logs` for `pod.terminated` |
| `Newton generation event stream closed without a terminal event` | Usually a video longer than ~5 minutes |
| Manual stops halfway through the video | Output-token exhaustion — check whether the last step ends mid-clause, then raise `max_new_tokens` |
| Run dies ~1 s in: `resolving blueprint: invalid config for 1 node(s)` | A pinned `blueprint_id` that has been superseded. Target the **key** — see below |

## `file_id` is a mutable, org-wide pointer

`file_id` **is the filename**. Uploading the same name again repoints it to a fresh object and orphans the previous one — same `file_id`, new `file_uid`:

```
upload 1: {"file_id": "clip.mp4", "file_uid": "fil_58shkhqztj9tnbye4dmphzxa2r"}
upload 2: {"file_id": "clip.mp4", "file_uid": "fil_6zmefmn6mz84jb5zvsevfzan4x"}
```

That namespace is shared across the org, and the timing makes it lethal: a run resolves its inputs at **submit** time but does not fetch the bytes until after ~7.5 min of model loading. Anything that re-uploads the same name inside that window kills a run that already looked healthy:

```
ERROR source.running  skipping video "clip.mp4": downloading video
  "s3://…/files_service/archetypeai/c03580dd-…" from S3 failed: S3 object not found
```

Nothing in that message implicates the upload that overwrote it, so it reads as a storage fault. **Suffix every upload** — `<stem>-<UTC timestamp>-<4 random hex>.mp4`. The stem alone is not enough: two runs of the same video collide, and so does a colleague running the same file. If the output `id` matters to you, rewrite it back to the plain stem on save, since it is derived from the uploaded name.

## Target the blueprint KEY, not an id

Use `"blueprint": "mga"`. A key resolves to whatever is canonical and active; a pinned id does not survive republication, and republication is frequent — three `mga` versions shipped in about two hours on 2026-08-11.

A superseded id fails **late and quietly**. `GET /agents/blueprints/{id}` still returns the document, `POST /agents/bundles` still returns `201 ready`, and `POST .../run` still returns `202 running` — then the pod dies one second after it starts:

```
pod.terminated  resolving blueprint: invalid config for 1 node(s):
                - `fusion`: config validation failed: Additional properties are not allowed
                  ('repetition_penalty', 'seed', 'stop', 'temperature', 'top_k', 'top_p' were unexpected)
```

The blueprint that once wired `prompt` and the seven sampling parameters (`blp_76kyqm4vjp9pt8tvfz8tks7x6t`) is dead in exactly this way: its fusion node carries config the node schema no longer accepts. **An earlier version of this skill recommended pinning that id to get `prompt`. Do not** — nothing built on it runs, and there is currently no supported way to set `prompt`.

`GET /agents/blueprints` lists only `is_active: true` versions, so a superseded id will not appear there even while it still reads back individually. Pin an id only to reproduce one specific past run, and expect it to stop working.

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

python3 references/run_mga_agent.py --video my_procedure.mp4          # key `mga`, 16384 tokens
python3 references/run_mga_agent.py --video my_procedure.mp4 --max-frames 32
python3 references/run_mga_agent.py --score references/sample_data/mga-output-max_new_tokens2048-coverage-prompt.json \
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
│       ├── mga-output-truncated-active-blueprint.json      historic: the 256-token cap
│       ├── mga-output-max_new_tokens2048.json               historic: cap lifted
│       └── mga-output-max_new_tokens2048-coverage-prompt.json  historic: `prompt` honoured
└── tests/
    └── test_references.py        network-free
```

Worked end-to-end example, including the video-selection stage and an offline scorer:
[manual-generation-agent-example](https://github.com/archetypeai/manual-generation-agent-example).
