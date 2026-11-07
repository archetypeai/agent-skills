---
name: atai-manual-generation-agent
description: >
  Run Archetype AI's managed Manual Generation (MGA) agent over the Agents
  API — upload a procedure video, copy the canonical `mga` blueprint into a
  project with your values, package it, deploy it, run a session, poll,
  download an ordered, timestamped manual; or serve it and send one `/query`
  per video. Use when the user has a recording of a procedure (a repair, an
  assembly, a workflow) and wants the platform to turn it into step-by-step
  instructions traceable back to the video. Covers video suitability (the
  ~5-minute ceiling, the audio-track requirement), the values set on the copy
  (`max_frames`, `max_new_tokens`, and a `prompt` that must say what to cover
  and never how to format), the project → copy → package → deployment →
  session lifecycle, the output JSON schema (`step, instruction,
  frame_start/end, timestamp_start/end`), and scoring against reference
  annotations. Do NOT use for verifying a task was performed
  correctly (the `tva` blueprint), for one-shot multimodal questions over a
  clip (`atai-newton-fusion-model`), or for time-series state classification
  (`atai-operational-state-monitoring-agent`).
---

# MGA Agent — Managed Manual Generation via the Agents API

The MGA agent turns a procedure video into an ordered manual with timestamps. You hand the platform an `.mp4`; it samples frames, transcribes the audio, fuses both in one pass, and returns steps you can trace back to the recording:

```
video ─► sample frames ─┐
                        ├─► newton-fusion f1-0 ─► parse ─► steps + timestamps
        whisper ASR ────┘
```

MGA V1 is **zero-shot**. The `mga` blueprint pins its own models (`newton-fusion-1-0` and `whisper-base` on production as of 2026-10-27; older versions used `newton-fusion:1.0` and `whisper:large-v3`), so unlike `osm`/`red` there is **no classifier to fit, no optimization and no `artifacts` map to pass**. A run is upload → copy (with your values) → package → deploy → session → download.

> **Draft: Agent Lifecycle 2.0.** This skill now runs on projects, packages,
> deployments and sessions. Bundles and agent instances are being removed: only
> Archetype AI's dev deployment serves the new calls so far, and production,
> staging and the free trial keep bundles and instances until their next platform
> release. Check yours first: `GET {endpoint}/agents/sessions?limit=1` answers
> 200, not 404. Numbers below say which they were measured with.

> **Verified on the lifecycle (dev).** Batch, **2026-11-05**: one session on the
> 173 s tire video (`i2JWkDyg26A`) returned production's reference manual,
> `sample_data/mga-output-current-16384.json`, **exactly** — 16 steps over
> 14–138 s, content md5 `866909a9315bce0e493db7ab1627749d` (of `[document]` as
> sorted-key compact JSON, `id` set to the video's stem), so it scores the same: 11/11 reference
> steps at IoU ≥ 0.3, 7/11 at IoU ≥ 0.5, mean IoU 0.56. **10.6 min** from
> opening the session to `completed`. Serving, **2026-11-06**: a manual
> byte-identical to the one the same video got from a serving bundle (15 steps,
> 9/11 at IoU ≥ 0.3, 6/11 at IoU ≥ 0.5); the serving deployment took **881 s**
> to start and the query **27.8 s**, 911 s in all.
>
> **With bundles, on production.** The canonical `mga` blueprint resolves by key
> on production (`https://platform.u1.archetypeai.app/api`). Re-verified
> **2026-10-27** on `blp_7vc935…`: four runs on the same 173 s video took
> **610–618 s** each — ~13 s queued, whisper 2.9 s download + 1.3 s load,
> newton-fusion **~5 min download + ~1 min 48 s load**, then ~3 min 15 s to
> process the video (~1.1× realtime). Three runs with the same values returned
> **byte-identical** output. If the blueprint key doesn't resolve, contact
> support@archetypeai.dev.

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

Every surface lives under one versionless `/api` prefix: `{endpoint}` below is the
deployment root plus `/api`, e.g. `https://platform.u1.archetypeai.app/api`.

```
POST   {endpoint}/files                                    upload the video
GET    {endpoint}/projects                                 find your project by name
POST   {endpoint}/projects                                 {name, description}
POST   {endpoint}/agents/blueprints/mga/copy               a copy with your values, in the project
POST   {endpoint}/agents/packages                          package the copy; poll until "ready"
POST   {endpoint}/agents/deployments                       deploy it: "batch" or "serving"
POST   {endpoint}/agents/sessions                          a run: one video (batch), or a serving client
GET    {endpoint}/agents/sessions/{id}                     status, error
GET    {endpoint}/agents/sessions/{id}/events              what the platform did, newest first
GET    {endpoint}/agents/sessions/{id}/results             output refs
POST   {endpoint}/agents/sessions/{id}/query               serving only
POST   {endpoint}/agents/sessions/{id}/cancel              stop a session
POST   {endpoint}/agents/deployments/{id}/stop             release a deployment (and its GPU)
```

Statuses: package `building → ready | failed`; deployment `deploying → ready →
stopping → stopped | failed`; session `pending → running → completed | failed |
cancelled`. A session has **no `/logs`**.

| With bundles | On the lifecycle |
|---|---|
| `POST /agents/bundles` `{blueprint: "mga", values}` | `POST /agents/blueprints/mga/copy` `{project_id, blueprint_key, name, overrides: {values}}`, then `POST /agents/packages`; poll until `ready` |
| `POST /agents/bundles/{id}/run` `{connectors}` | `POST /agents/deployments` `{package_id, kind: "batch"}`, then `POST /agents/sessions` `{deployment_id, connectors}` |
| `POST /agents/bundles/{id}/run` `{mode: "serving"}` | `POST /agents/deployments` `{package_id, kind: "serving"}`; poll until `ready`; `POST /agents/sessions` `{deployment_id}`; poll until `running` |
| `GET /agents/instances/{id}/logs` | `GET /agents/sessions/{id}` and `/events` |
| `GET /agents/instances/{id}/results`, `/query`, `/cancel` | the same under `/agents/sessions/{id}` |
| `DELETE /agents/bundles/{id}` | `POST /agents/deployments/{id}/stop`, `DELETE /agents/deployments/{id}`, then `DELETE /agents/packages/{id}` |

## ⚠️ Two values decide whether you get a manual — read before anything else

Both are settable today, on your copy of the blueprint (Step 3). Both fail silently when they are wrong.

| Set | Why |
|---|---|
| `max_new_tokens: 16384`, **set explicitly** | The blueprint now defaults to **8192**, not 16384 — leave it off the copy and you get 8192. Shared with the model's reasoning block. Run out inside it and you get an empty manual, reported as success — check the manual is non-empty. No budget in 2048–65536 changed the output on 2026-08-20, so set 16384 unless you have a reason not to. |
| a `prompt` saying what to **cover** | Leave it off and the blueprint's own instruction applies, halving the manual. Never specify an output *format* — the parser owns the template and returns zero steps. |

And **check that they are honoured**, rather than assuming. A value is real only if
it is declared in the blueprint's `values` *and* referenced as `${values.<key>}` by
some node or connector. A copy refuses an **undeclared** key with a `400`, but a key
that is declared and wired to nothing is accepted and has no effect — nothing in the
response tells you.

```python
doc = GET(f"{endpoint}/agents/blueprints/{blueprint}")["document"]
wired = json.dumps({"nodes": doc["nodes"], "connectors": doc["connectors"]})
inert = [k for k in my_values
         if k not in doc["values"] or f"${{values.{k}}}" not in wired]
```

`references/run_mga_agent.py` runs that against the blueprint it copies, before every run, and prints both halves, so
silence never has to be read as success:

```
blueprint blp_7vc935235b964bw5b6819dq065 (key=mga, active=True)
  honoured: ['max_frames', 'max_new_tokens', 'prompt']
```

Run the check every time. The `mga` blueprint is republished often — three versions
in one week — and each republication can change which values are wired. When one of
these two stops being honoured, the run still succeeds and the manual quietly gets
worse, so the preflight is the only thing standing between you and a plausible bad
result.

### `max_new_tokens` and the reasoning block

The model **reasons before it answers**, and pays for the reasoning out of this same
budget. If generation ends inside the reasoning block it never emits an answer, and every
signal says success: the session ends `completed` with no `error`, every HTTP call 2xx,
and a 38-byte output containing `{"id": "…", "results": []}`. **Check that the manual is
non-empty before trusting a run** — that check is the one thing here that never goes stale.

**What the budget actually changed, measured 2026-08-20 on Prod.** One 173 s video
(`tire_i2JWkDyg26A`, the source of the reference outputs in `sample_data/`), five
budgets, everything else held constant:

| `max_new_tokens` | 2048 | 4096 | 16384 | 32768 | 65536 |
|---|---|---|---|---|---|
| steps | **18** | **18** | **18** | **18** | **18** |

**Nothing about the budget mattered.** Every run produced the same 18-step manual —
**18/18 instructions and 18/18 timestamps identical** to the 16384 run at every budget,
across a 32× range. No empty manual, no reasoning-block WARN. Reproduce any cell with
`--max-new-tokens <n>`.

Two caveats on that row. The five outputs are the same length (3214 bytes) but had
**five different md5s** that day, so compare content, not bytes — even though three
runs with identical values on 2026-10-27 were byte-identical. And a 173 s video with an 18-step manual is one point in a space:
reasoning cost scales with how much there is to describe, so a longer or busier video may
behave differently. Send 16384 unless you have a reason not to (it was the blueprint
default when this was measured; production now defaults to 8192); the evidence for
preferring any particular value is gone.

#### `results: []` — plan for it, do not predict it

It is a real outcome and you should handle it, but **there is no threshold to memorise.**
Whether a run lands in it depends on how much reasoning the model does before answering —
an interaction of the video, the prompt, and the budget, not the budget alone. It was
observed at 2048 and 4096 in Aug 2026 on this very video; on 2026-08-20 the same two
budgets produced full manuals. Assume it can happen on any combination you have not
tried, including one that worked before.

Detect it explicitly: **count the steps**. With bundles, a run's `/logs` carried a WARN
for exactly this case:

```
WARN parser.running  ManualGenerationResultsParserNode: generation ended inside the
  model's reasoning block, so it never produced an answer; dropping 0 rehearsed
  row(s). Raise `max_new_tokens` if the reasoning was truncated.
```

A session has **no `/logs`**, only lifecycle `/events` (opened, dispatched, failed with a
reason), so that WARN is no longer visible from outside: an empty manual from too small a
budget and one from a format-specifying prompt look the same. Rule out the prompt first
(see [On `prompt`](#on-prompt)), then raise the budget. Don't treat raising it as a sure
fix either: the `atai-task-verification-agent` sibling has observed the opposite
direction, where a larger budget produced the empty output and a smaller one worked.

<details>
<summary><b>Earlier measurement (2026-08-13) — dated, and did not reproduce</b></summary>

This skill previously measured a hard floor on the same 173 s video and prescribed the
blueprint default:

| `max_new_tokens` | 2048 | 4096 | **16384** | 32768 | 65536 |
|---|---|---|---|---|---|
| steps | 0 | 0 | **18** | 18 | 18 |

`sample_data/mga-output-current-4096-EMPTY.json` is a captured artifact of that failure
at 4096 — kept because nothing else shows what an empty manual looks like, not because
it reproduces. At 4096 today, the same video returned the full 18-step manual.

The `atai-task-verification-agent` sibling documents the **opposite** rule — that 5760
works and 8192/16384 return nothing — and that did not reproduce either. Two blueprints,
opposite prescriptions, neither observable now; the shared reasoning behaviour is the
likely variable rather than anything specific to either blueprint.

**Non-reproduction is not retirement.** The failure was observed; only the numbers are
unreliable.

</details>

### What a caller-supplied `prompt` is worth

Measured on dev in Aug 2026: same clip, same `max_frames`, only the instruction differing — the blueprint's hardcoded "10 steps or less" against a coverage instruction (this skill's default prompt until 2026-10-27; see [On `prompt`](#on-prompt) for why it was replaced):

| | hardcoded | coverage prompt |
|---|---|---|
| steps | 10 | **18** |
| actions bundled per step | 3.2 avg, 7 max | **1.7 avg, 4 max** |
| ref steps at IoU ≥ 0.5 | 4/11 | **7/11** |
| ref steps at IoU ≥ 0.3 | **10/11** | 9/11 |
| spoken cautions as their own step | 0 | 1 |

**Judge a manual by whether it can be followed**, and note that the last two rows disagree with that judgement. At the *loose* threshold the 10-step version scores **higher**, because MGA tiles the timeline contiguously so fewer steps means longer ones that overlap a reference interval more easily. Coarser segmentation flatters loose-threshold recall while being worse to work from: an operator cannot tick off "apply the parking brake" separately from "switch off the ignition", and each step carries one frame thumbnail however many actions it contains.

Where the remaining error sits is worth checking per step rather than in aggregate. On that run the four scores below 0.4 have **two different causes, pulling in opposite directions**:

- **Under-segmentation** — two reference steps collapsing onto one prediction: remove-the-nuts with remove-the-wheel (steps 5–6, IoU 0.27 and 0.26) and tighten with put-things-back (step 10, 0.34).
- **Over-segmentation** — the opening reference step (14.8–28.9 s, "stop the car") spread across three predictions: stop / warning triangle / parking brake, so no single prediction covers more than a third of it (step 1, 0.35).

Every other step scored 0.52–0.69. Both are prompt-shaped rather than a model ceiling, but they want **opposite** instructions — asking for the collapsed actions as separate steps risks splitting the opening further, and asking for coarser opening steps risks re-collapsing the middle. Untested, and worth testing as one change rather than two.

### Score by text, not only by time

Temporal IoU credits whichever predicted step **overlaps** a reference interval,
whatever that step says. A manual whose texts sit one step early in the timeline
still scores well. Measured on production, 2026-10-27, same 173 s video, same values,
two prompts:

| | recall @ IoU ≥ 0.5, by time | … by the step whose **text** names the action |
|---|---|---|
| previous default prompt (20 steps, `mga-output-prod-old-prompt-SHIFTED.json`) | 7/11 | **0/11** — every text ~5 s early: *"Stop your car safely"* at 7–14 s, inside a music intro that runs to 14 s |
| **current default prompt** (16 steps, `mga-output-current-16384.json`) | 7/11 | 6/11 |
| previous default prompt on dev, Aug 2026 (`mga-output-dev-old-prompt-18steps.json`) | 7/11 | 7/11 |

The time-only score could not tell the first row from the second. The same prompt
was not shifted on dev, so the 2026-10-15 blueprint (`whisper-base`) may be part of
it; one run each, so not separated. Either way, **check matched texts, not only IoU**:
`--keywords` takes `{"<step>": ["phrase", …]}` and reports the text-matched score
next to the temporal one, flagging every time-only match whose text names a different
action. `sample_data/40567_i2JWkDyg26A_step_keywords.json` is the set for the tire video.

## Step 1 — Choose a video

The platform validates none of this for you, and getting it wrong costs ~10 minutes per attempt.

| Requirement | Why |
|---|---|
| **Under ~5 minutes** | Longer videos fail outright. A 563 s clip ran 5 min 13 s then died: `pipeline execution failed: Newton generation event stream closed without a terminal event`, exit 1. |
| **Has an audio track** | `AudioReaderNode` → whisper → `prompt.audio` is half the pipeline. Spoken-only content (safety cautions) shows up in the output, so a silent video loses real information. Step boundaries come from it too: on a silent 80 s clip every step came back exactly 5 s long (production, 2026-10-27, both prompts), so those timestamps say nothing about the video. |
| **Reference steps, to score it** | Otherwise there is nothing to evaluate against. |

The ~5-minute ceiling is **not** a context limit: at `max_frames: 16` the video occupies 392 visual tokens, ~0.15% of F1-0's 256K window.

Source resolution barely matters — every frame is resized to `size × size` (default 224), so 360p is no worse than 1080p, only smaller and faster to upload.

### Longer videos: chunk them

Chunking also bounds how much each job has to describe: a 20-step manual over 5 minutes leaves each step longer and coarser than over 90 s. What has worked in production:

- **90 s chunks with 15 s overlap.** 90 s keeps each job well inside the failure band; the overlap is there because a step that straddles a cut is otherwise described from half its evidence in both chunks.
- **Cut on keyframes.** `ffmpeg -ss <t> -i in.mp4 -c copy` rewinds to the previous keyframe and silently shifts the segment — measured **−7.57 s** of drift on a real file. Read the keyframe timestamps first (`ffprobe -select_streams v -skip_frame nokey -show_entries frame=pts_time`) and snap each cut to one, or re-encode.
- **Step from each chunk's actual start, not from `i × stride`.** Snapping to keyframes moves the boundaries; a fixed grid then eats the overlap (15 s → 10 s on our first attempt).
- **One session per chunk, or one with `source` as an array.** The array form loads the model once instead of per chunk, which is the dominant cost — but results came back **out of order** (measured with bundles) and every timestamp is **chunk-local**. Tag each input and add the chunk's start offset before merging, or the manual is silently scrambled. [Agent serving](#agent-serving--load-once-query-many) is the other way to load once.
- **Dedupe the overlap conservatively.** Two steps may be merged only if they came from *different* chunks *and* their time spans actually overlap. A similarity-only rule deletes real repeated actions — three genuine "Click on the OK button" steps collapsed into one before we added the time test.
- **Tell every chunk after the first that it comes from the middle — and don't tell the first.** Asked as if it were a whole video, a later chunk invents an introduction and a conclusion (*"Wait for the logo to appear on the screen"*). Asked as if it came from the middle, the *first* chunk loses the clock: on two of 11 videos (agent serving, dev, 2026-10-27) every step came back at 0.0–0.0 s, which empties the first 75 s of the manual. What worked, prepended to the prompt for chunks 2 onward only:

  ```
  This video is one segment taken from the middle of a longer recording of a single
  continuous procedure. List only the steps performed within this segment. Do not
  summarise, introduce, or conclude the overall procedure, and do not describe the
  segment as beginning or ending anything — the steps before and after it are covered
  elsewhere. Work at a fine grain: a single button press, acknowledging a dialog,
  reading a value, or watching an indicator each count as their own step, and steps
  only seconds apart are expected.
  ```

  On the same 11 opening chunks, the plain prompt also dropped intro padding (22 *"View the sugar"*, *"View the ladle"*… steps on one cooking video) and lifted the 173 s tire video from 2/11 to 5/11 reference steps at IoU ≥ 0.5. The exception was a **silent** opening chunk: with neither narration nor the preamble to anchor it, it wrote a generic procedure at an even 8 s per step. One example; watch for it.
- **Fold a short tail into the chunk before it.** A last chunk that adds less new footage than the overlap is mostly an end card, and the model describes it as steps — *"Press the 'Enter' key."* eleven times on a 20 s tail.
- **Drop a chunk that loops or has lost its clock, and a step past its chunk's end.** A chunk with 6+ steps of which under a third are distinct (*"Cut a piece of wood using a table saw."* ×17), or whose every step is zero-length, is noise: drop it whole and report the hole rather than keep its steps. Drop, don't clamp, a step that ends more than ~5 s past its chunk: the model has lost track of the video.

A `pipeline execution failed: Newton generation event stream closed without a terminal event` at 138–175 s of runtime, on inputs that are well under the ceiling, is a **platform-side** failure and not something to fix by shortening further — retry the chunk.

## Step 2 — Upload the video

```python
POST {endpoint}/files               # multipart/form-data, field name "file"
→ {"is_valid": true, "file_id": "clip.mp4", "file_uid": "fil_…"}
```

The **declared** `Content-Type` is enforced against a MIME allowlist, not the bytes — send `video/mp4`. Use `file_id` (the filename) in the session's connectors, not `file_uid`.

## Step 3 — Copy `mga` into a project, with your values

Everything lives in a **project**: find yours by name (`GET {endpoint}/projects`) or
create it (`POST {endpoint}/projects {"name": …, "description": …}`). A copy of a
blueprint lands in the project you name, and its package, deployments and sessions take
their project from it.

**The values go on the copy.** A package, a deployment and a batch session take **no
values** (a session given `values` answers `422`), so the copy is the one place left to
set `max_frames`, `max_new_tokens` and the instruction. Make **one copy per setting**,
under a key that names it, and reuse it for every later run at that setting:

```json
POST {endpoint}/agents/blueprints/mga/copy
{
  "project_id": "prj_…",
  "blueprint_key": "mga-f64-t16384-pc647bb",
  "name": "MGA mga-f64-t16384-pc647bb",
  "overrides": {"values": {
    "max_frames": 64,
    "max_new_tokens": 16384,
    "prompt": "Generate an ordered list of every distinct step performed in this video, in order. Include brief steps and steps that are repeated. Write each step as a direct, actionable instruction, as if telling the reader what to do. Do not narrate or describe what is happening in the video. Capture the details needed to repeat the action exactly: control and part names, printed and on-screen text quoted word for word, part numbers, model and serial numbers, tool types and sizes, quantities, and which way things move. For every component being handled, state where it is using grid terms such as top-left, top-right, bottom-left, bottom-right, center-left, or upper-center. Where several similar actions are performed, such as four screws on one panel, treat each as its own step and say what distinguishes it. Where the video shows how to tell the step worked, say that. Base every step on what is visible; use the spoken narration where there is any, and rely on the picture alone where the video is silent. Each step must describe a different action from the one before it — where two moments look alike, state what changed between them. Do not invent steps, parts, or details that are not shown or stated in the video."
  }}
}
→ 201 {"id": "blp_…", …}
```

The key is `mga-f<max_frames>-t<max_new_tokens>-p<first 6 hex of md5(prompt)>`; a key
already in use in the project is a `409`, so look it up first
(`GET {endpoint}/agents/blueprints?project_id=…&key=…`).

**Set all three, every time.** A plain copy runs at the blueprint's own defaults, which on
dev in 2026-11 were `max_new_tokens` **8192** and the instruction *"Generate a concise,
ordered list of distinct steps with 10 steps or less."* — the ~10-step ceiling that halves
the manual (see [What a caller-supplied `prompt` is worth](#what-a-caller-supplied-prompt-is-worth)).

**No `artifacts` map** — the blueprint pins its own models. This is the main shape difference from `osm`/`red`.

| Value | Default | Notes |
|---|---|---|
| `max_frames` | 64 | Frames uniformly sampled across the whole video. **`1..=64`** — values outside that range are rejected when the blueprint is resolved (earlier versions accepted anything, including 512 and `-1`). 64 is also the reader/preprocessor batch size. |
| `size` | 224 | Each frame resized to a `size × size` square. **`224..=2048`**, or `null` to keep the source resolution; either way no frame is decoded with its longest side above 2048. |
| `min_temporal_similarity_threshold` | 0.0 | New on 2026-10-15: collapses runs of near-identical frames before the model sees them. 0.0 keeps every sampled frame. Not measured here. |
| `parser_compute_stats` | false | Attaches template-conformance stats — useful for diagnosing a parser mismatch. |
| `max_new_tokens` | **8192** | Exposed and honoured, and shared with the reasoning block. The default was 16384 until the 2026-10-15 blueprint. 2048 and 4096 returned an EMPTY manual in Aug 2026 but the full manual on 2026-08-20; nothing in 2048–65536 changed the output that day. **Set 16384** — see "`max_new_tokens` and the reasoning block". |
| `prompt` | (blueprint's own) | **Set one.** Omitting it halves the manual. Preflight it — it has been unwired before. See below. |

### On `prompt`

Wired as `${values.prompt}`, feeding `connectors.source.config.default_text`. Preflight it — it has been unwired before, and `text_extensions: []` means there is no text-input back door when it is. A new prompt is a new copy (its md5 is in the key).

**The one hard rule: say what to COVER, never how to FORMAT.** `ManualGenerationResultsParserNode` owns the output template, so a prompt specifying markdown headings or its own step layout makes the parser return **zero steps** — with no error. The runner's default parses cleanly:

> Generate an ordered list of every distinct step performed in this video, in order. Include brief steps and steps that are repeated. Write each step as a direct, actionable instruction, as if telling the reader what to do. Do not narrate or describe what is happening in the video. Capture the details needed to repeat the action exactly: control and part names, printed and on-screen text quoted word for word, part numbers, model and serial numbers, tool types and sizes, quantities, and which way things move. For every component being handled, state where it is using grid terms such as top-left, top-right, bottom-left, bottom-right, center-left, or upper-center. Where several similar actions are performed, such as four screws on one panel, treat each as its own step and say what distinguishes it. Where the video shows how to tell the step worked, say that. Base every step on what is visible; use the spoken narration where there is any, and rely on the picture alone where the video is silent. Each step must describe a different action from the one before it — where two moments look alike, state what changed between them. Do not invent steps, parts, or details that are not shown or stated in the video.

It replaced a shorter prompt on 2026-10-27 — *"Generate a concise, ordered list of every distinct step performed in this video, covering the procedure from the first action to the last. Include brief steps and steps that are repeated. Use up to 20 steps. Keep each step to at most 15 words. Use both what is shown and what is said."* — after both were run on production with the same values:

| | previous prompt | **current prompt** |
|---|---|---|
| silent 80 s mouse-assembly clip | 16 steps, **2 distinct texts** — *"Take the battery out of the mouse."* 15 times in a row, on a wired mouse | **17 steps, 17 distinct** |
| 173 s tire video, by text at IoU ≥ 0.5 | 0/11 (texts ~5 s early) | **6/11** |

The loop is the decoder re-emitting its most likely sentence: *"concise"* plus a word cap
drove the steps short and nothing required consecutive steps to differ. What stops it is
naming the **kinds of detail** wanted (part names, quoted text, tool sizes, direction of
movement), which makes a repeated string unlikely, and the explicit rule that each step
must differ from the one before. Both pairs of outputs are in `sample_data/`.

Useful things to put in it: the kinds of detail to capture, whether to use narration as well as what is on screen, a rule that consecutive steps differ, and an instruction not to invent steps. Avoid a per-step word cap and "concise" — together they produced the loop. Things that will cost you the run: any mention of output structure.

What it does **not** fix: on the silent clip every step is still exactly 5 s long and
the last one ends at 85 s of an 80 s video — the timestamps are a grid, not a reading of
the footage. A prompt cannot supply the narration a silent video lacks; check silent
results against the video before trusting them.

If a run comes back with **zero steps**, suspect the prompt's wording before the budget — with bundles, a format-shaped instruction was the one failure that produced silence rather than the reasoning-block warning, and a session shows neither.

## Step 4 — Package and deploy

```json
POST {endpoint}/agents/packages
{"blueprint_ids": ["blp_…"], "target": {"kind": "platform"}, "name": "MGA mga-f64-t16384-pc647bb"}
→ 201 {"id": "pkg_…", "status": "building"}          poll GET /agents/packages/{id} until "ready"

POST {endpoint}/agents/deployments
{"package_id": "pkg_…", "kind": "batch"}
→ 201 {"id": "dpl_…", "status": "deploying"}         poll until "ready" (a batch one is ready at once)
```

A package pins the copy and the models it resolves to, once; a batch deployment holds no
worker. Build both once per setting and reuse them. **Write each id down before you wait
on it**: a package that fails to build, or a client that dies while it builds, otherwise
leaves something nothing can find to delete. `run_mga_agent.py` keeps them in
`mga-lifecycle.json`, with every session's.

## Step 5 — Run a session, and poll it

```json
POST {endpoint}/agents/sessions
{"deployment_id": "dpl_…",
 "connectors": {"source": [{"type": "file",
                            "id": "clip-20260813T154418Z-410f.mp4", "format": "mp4"}]}}
→ 201 {"id": "ses_…", "status": "pending"}
```

The `id` is whatever `file_id` the upload returned — suffix it, see below. One session per
video, the video in `connectors.source` exactly as a bundle run took it; MGA takes no text
input, so the instruction is the copy's `prompt`.

**Run them one at a time.** Whether concurrent runs queue depends on what else is running on the deployment at that moment — they queue when other workloads hold the workers, and come up as concurrent pods when they don't. Both have been observed with bundles: four at once produced 799–817 s waits (which looked like serialization), while three at once came up as three concurrent pods and one was SIGKILLed mid-load (`exit=137`). Other tenants' workloads aren't visible to you, so neither outcome is predictable from your side and neither is safe to design around.

Poll `GET /agents/sessions/{id}` until `status` is `completed`, `failed` or `cancelled`
(it was reliable on dev, unlike an instance's), and read `GET /agents/sessions/{id}/events`
for what happened. Events come **newest first**, with UTC `created_at`; reverse them, and
print them in local time so they line up with your own log lines:

```python
ses = GET(f"{endpoint}/agents/sessions/{session_id}")
events = GET(f"{endpoint}/agents/sessions/{session_id}/events?limit=100")["data"]
for ev in reversed(events):
    print(local(ev["created_at"]), ev.get("level"), ev["message"])
done = ses["status"] in ("completed", "failed", "cancelled")
```

Events say what the platform did — opened, dispatched, failed with a reason — and on a
failure `error` carries the reason. The model-load timings, per-node progress and parser
warnings that a bundle run's `/logs` carried are **not** there, and a session has no
`/logs`. A session that produced **no steps still ends `completed`**: count them.

## Step 6 — Fetch results

```python
GET {endpoint}/agents/sessions/{session_id}/results
→ {"data": [{"data": {"filename": "…_0001.json", "file_extension": "json",
                      "ref": "/api/files/fil_…/content"}}]}
```

The `ref` is a **root path**: it already includes `/api`, so resolve it against the
deployment root, not against `{endpoint}` — appended to `…/api` it 404s as `/api/api/…`.
It needs the bearer token. (An absolute, presigned `ref` must not get one.)

**A finished run can still be writing its output** (a known platform issue, seen with
bundles): a download within seconds can return a partial file. A partly written JSON
document doesn't parse, so the runner downloads again every 15 s until it does, for up to
10 min. (A whole document that the model cut short, like
`mga-output-truncated-active-blueprint.json`, parses: that's a different failure.)

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
- **The output (almost always) tiles the timeline contiguously.** Every step's `timestamp_end` is the next step's `timestamp_start`, on integer-second boundaries — with one exception seen so far: a 2026-10-27 production run left a 6 s gap (24–30 s) between two steps, so don't assume no gaps in code. **MGA does segmentation, not event detection** — there is no "nothing happening here" span, so every moment is labelled. That is why a spoken caution gets its own numbered step: it has nowhere else to go, and there is no `kind: action | caution` field.

## Runtime

Measured with bundles on production, 2026-10-27, four runs of a 173 s video:

| Phase | Cost |
|---|---|
| Queue | ~12 s solo; **799–817 s** if other runs are in flight (Aug 2026) |
| whisper-base download + load | ~3 s + ~1.3 s |
| **newton-fusion download + load** | **~5 min + ~1 min 48 s** |
| Processing | ~1.1× realtime (173 s video → ~3 min 15 s) |
| **Total** | **~10 min for a 3-minute video** (610–618 s) |

On the lifecycle (dev, 2026-11-05) the same video took **10.6 min** from opening the
session to `completed`; a package and a batch deployment are built once per setting and
add nothing to later runs.

**Cold start is ~7 min on every batch session and is never cached.** Budget accordingly: a sweep of six configurations is ~40 minutes of pure model loading before any inference. (In Aug 2026, with `whisper:large-v3`, a run took ~15 min and processing ran ~2.5× realtime.) [Agent serving](#agent-serving--load-once-query-many) pays the load once for many videos.

## Agent serving — load once, query many

> **Preview.** Serving deployments run only where the lifecycle does (see the draft
> note at the top). The lifecycle numbers are from dev, 2026-11-06; the rest were
> measured on dev with serving bundles on 2026-10-27 and -28, and will change.

A batch session downloads and loads the model for **every** video. A **serving
deployment** of the same package loads it once onto a GPU worker and holds it there; a
session on it then answers one request per video in seconds:

```
POST {endpoint}/agents/deployments     {"package_id": "pkg_…", "kind": "serving"}   → {"id": "dpl_…", "status": "deploying"}
GET  {endpoint}/agents/deployments/{id}                         poll until "ready" (881 s on 2026-11-06)
POST {endpoint}/agents/sessions        {"deployment_id": "dpl_…"}                    → {"id": "ses_…"}
GET  {endpoint}/agents/sessions/{id}                            poll until "running"
POST {endpoint}/agents/sessions/{id}/query
     {"model": "mga", "query": "<prompt>", "file_ids": ["base64:<the mp4's bytes>"]}
→ 200 {"response": {"data": [{"step": 0, "instruction": "…", "frame_start": …,
                              "frame_end": …, "timestamp_start": …, "timestamp_end": …}, …]}}
POST   {endpoint}/agents/sessions/{id}/cancel,     DELETE {endpoint}/agents/sessions/{id}
POST   {endpoint}/agents/deployments/{id}/stop,    DELETE {endpoint}/agents/deployments/{id}
```

The package is the same one batch uses, built from the same copy (Step 3), so
`max_frames` and `max_new_tokens` come from the copy; the session takes no values. No
upload and no connectors: the video travels inline in the request, and `query` carries
the prompt for that one request — as it did with bundles — which is how each chunk gets
its own (see [chunking](#longer-videos-chunk-them)). The rows are batch's rows, but under
`response.data`, not `{"id", "results"}`.

**The deployment holds the GPU, not the session.** Stop and delete the serving deployment
when you are done — on every way out, Ctrl-C and `kill` included — and keep its id on disk
until then. Cancelling the session alone leaves the worker held. `run_mga_agent.py
--serve` does this in a `finally`.

**What it costs.** On the lifecycle (dev, 2026-11-06), one 173 s video: **881 s** for the
serving deployment to become ready (it can also fail outright with *"there was no
capacity to start the deployment's agent"*: retry later), the session running at once,
the query **27.8 s**, **911 s** in all against 768 s through a serving bundle. With
bundles, thirteen videos (80 s to 10 min, 46 chunk queries of 90 s):

| | wall clock |
|---|---|
| load, once per agent | **664–760 s** across nine agents (10 min 48 s – 12 min 7 s of it loading newton-fusion, none downloading) |
| one query | **3–11 s** typically; a few 20–28 s |
| a video under 4 min (1–3 chunks) | 7–21 s |
| a 10-minute video (8 chunks) | 45–57 s |
| all thirteen on one agent | **~16–18 min**, against ~2.2 h as thirteen batch runs |

**Limits, measured with bundles (2026-10-28) unless noted:**

- **One GPU worker at a time.** A second fails within a minute with *"no worker is free"*. Release yours when done, Ctrl-C included.
- **The whole request is capped at 2 MiB** — about 1.57 MB of video after base64. Re-encode to fit; frames are resized to 224 px anyway, so give up resolution and frame rate first (360p at 8–15 fps, or 240p at 5 fps, CRF 30–32). **Keep the audio** (mono AAC, 24–32 kbit/s): whisper's transcript is half the input.
- **Each query has a 58 s budget**, then `504 query_timeout`. Retry it: 2 of ~150 chunk queries timed out and both answered on the retry. Also set a client-side timeout (~120 s): one query in a run on an unstable day hung for 857 s with no answer at all.
- **A session with no query for a while is closed** — 20 minutes with bundles; 30 by default on the lifecycle (dev, 2026-11) — and every later query gets a `409`. Stop on a 409 and open a new session. A hung request is not activity, which is the other reason for the client timeout.
- **A bad video fails only its own query** — `400 … probe failed: no video stream` for an audio-only mp4, `400 … decode failed` for garbage — and the session answers the next one. A 35-frame clip is answered.
- **The load is not visible as it happens.** A serving bundle's `/logs` showed *"Still loading newton-fusion-1-0 model after 4min0s"* once a minute; a serving deployment shows only its `status`, so expect ~15 min of `deploying` with nothing to say how far along it is.
- **A fresh agent can answer differently, at temperature 0.** Two queries to one agent gave identical steps, and a rerun on a new agent reproduced six videos byte for byte — but four of seven others came back different, two of them in step count (an opening chunk went from 3 steps to 12). Treat one serving run's step counts as a sample. (On the lifecycle, the tire video reproduced the bundle-era answer byte for byte.)

## Common Pitfalls

| Symptom | Cause |
|---|---|
| `404` on `/agents/bundles` or `/agents/instances` | Bundles and instances are gone on that deployment: use the lifecycle |
| `404` on `/agents/sessions` | The deployment doesn't serve the lifecycle yet: use bundles there until its next release |
| `422` opening a session | It was given `values`. Values go on the copy (Step 3) |
| `400` on the copy | A value the blueprint does not declare |
| `409` on the copy | The key is already in use in the project: look it up and reuse it |
| A custom `prompt` has no effect | It is declared but not wired on that blueprint: accepted, ignored. Preflight it |
| Manual has ~10 steps no matter how long the video is | The copy was made without a `prompt`, or with a blueprint whose prompt is hardcoded to "10 steps or less". Check the copy's values and `default_text` |
| `completed`, no `error`, and **`results: []`** | `max_new_tokens` below the reasoning threshold, or a format-specifying prompt. Set 16384 — the blueprint's own default is now 8192 |
| The same step text repeated over and over | The decoder looping. Seen with a "concise… at most 15 words" prompt on a silent clip. Name the kinds of detail wanted and require each step to differ from the one before — see [On `prompt`](#on-prompt) |
| Run dies with `S3 object not found` minutes in | Something re-uploaded your `file_id`. Suffix every upload — see below |
| Parser returns **0 steps** | A custom `prompt` specified an output format; the parser owns the template. Only reachable on a blueprint that wires `prompt` |
| `404` downloading a result | The root-path `ref` (`/api/files/…`) was appended to `…/api`. Resolve it against the root |
| A serving deployment `failed`: *"no capacity to start the deployment's agent"* | No GPU worker free. Delete it and retry later |
| `Newton generation event stream closed without a terminal event` | Usually a video longer than ~5 minutes |
| Manual stops halfway through the video | Output-token exhaustion — check whether the last step ends mid-clause, then raise `max_new_tokens` |
| Run dies ~1 s in: `resolving blueprint: invalid config for 1 node(s)` | A pinned `blueprint_id` that has been superseded. Copy from the **key** — see below |

## `file_id` is a mutable, org-wide pointer

`file_id` **is the filename**. Uploading the same name again repoints it to a fresh object and orphans the previous one — same `file_id`, new `file_uid`:

```
upload 1: {"file_id": "clip.mp4", "file_uid": "fil_58shkhqztj9tnbye4dmphzxa2r"}
upload 2: {"file_id": "clip.mp4", "file_uid": "fil_6zmefmn6mz84jb5zvsevfzan4x"}
```

That namespace is shared across the org, and the timing makes it lethal: a session resolves its inputs when it is **opened** but does not fetch the bytes until after ~7.5 min of model loading. Anything that re-uploads the same name inside that window kills a run that already looked healthy:

```
ERROR source.running  skipping video "clip.mp4": downloading video
  "s3://…/files_service/archetypeai/c03580dd-…" from S3 failed: S3 object not found
```

Nothing in that message implicates the upload that overwrote it, so it reads as a storage fault. **Suffix every upload** — `<stem>-<UTC timestamp>-<4 random hex>.mp4`. The stem alone is not enough: two runs of the same video collide, and so does a colleague running the same file. If the output `id` matters to you, rewrite it back to the plain stem on save, since it is derived from the uploaded name.

## Target the blueprint KEY, not an id

Copy from `POST /agents/blueprints/mga/copy`. A key resolves to whatever is canonical and active; a pinned id does not survive republication, and republication is frequent — three `mga` versions shipped in about two hours on 2026-08-11.

A superseded id fails **late and quietly**. With bundles, `GET /agents/blueprints/{id}` still returned the document, `POST /agents/bundles` `201 ready`, and `POST .../run` `202 running` — then the pod died one second after it started:

```
pod.terminated  resolving blueprint: invalid config for 1 node(s):
                - `fusion`: config validation failed: Additional properties are not allowed
                  ('repetition_penalty', 'seed', 'stop', 'temperature', 'top_k', 'top_p' were unexpected)
```

That signature — a run accepted, then dead about a second later — means a stale id, not
a bad video or a bad value. Switch to the key and re-run. A copy is taken from the blueprint
as it stands, so to pick up a republished `mga`, copy it again (delete the old copy first,
or use a new project).

`GET /agents/blueprints` lists only `is_active: true` versions, so a superseded id
will not appear there even while it still reads back individually. **Reading a
blueprint successfully tells you nothing about whether a run against it will
resolve.** Pin an id only to reproduce one specific past run, and expect it to stop
working.

## Cleanup

Each object holds the next, so delete in this order: **sessions, then deployments, then
packages**.

```python
POST   {endpoint}/agents/sessions/{id}/cancel          # if it is still going
DELETE {endpoint}/agents/sessions/{id}
POST   {endpoint}/agents/deployments/{id}/stop         # if it is still up
DELETE {endpoint}/agents/deployments/{id}
DELETE {endpoint}/agents/packages/{id}
```

`GET /agents/deployments?package_id=…` and `GET /agents/sessions?deployment_id=…` find
what a package or a deployment holds, including ones your client never recorded; listings
page with `limit` (max 1000) and an opaque `next_cursor` passed back as `after`.
**A serving deployment holds a GPU worker until it is stopped**, and an abandoned batch
session holds one until it ends; on a deployment with few of them that delays everyone
else's. **`DELETE` on a session is not a no-op** — it removes the session and its results,
so download first. The project and its copies are cheap to leave; the next run at a
setting packages its copy again. `run_mga_agent.py --cleanup` lists what it would delete
from its state file, and `--cleanup --yes` deletes it.

## Local Setup

`references/run_mga_agent.py` is **stdlib only** — nothing to install. It talks plain
HTTP because the official `archetypeai` client does not cover deployments and sessions
yet. It needs an `.env` beside wherever you run it, with **both** variables; there is no
default endpoint. Any form works — with or without `/api` or a `/vX.Y` suffix, quoted or
not: the script reduces it to the deployment root and appends `/api` itself:

```
ATAI_API_KEY=<your API key>
ATAI_API_ENDPOINT=https://platform.u1.archetypeai.app/api
```

```sh
# See exactly what would be sent — no key, no API calls, no GPU, safe to run first.
python3 references/run_mga_agent.py --video my_procedure.mp4 --dry-run

# Batch: preflight, project, copy, package, deployment (once per setting), then
# upload, session, poll status and /events, download, print. ~10 minutes, most of it
# model loading. Defaults are the ones you want. Several videos run one after another.
python3 references/run_mga_agent.py --video my_procedure.mp4

# Serving: one load (~15 min), then a /query per video (each under the 2 MiB cap);
# the serving deployment is stopped and deleted on exit.
python3 references/run_mga_agent.py --serve --video a.mp4 b.mp4

# What is still on the platform from mga-lifecycle.json; --yes deletes it.
python3 references/run_mga_agent.py --cleanup

# Offline, instant, no key: print a manual, or score one against reference steps.
python3 references/run_mga_agent.py --show references/sample_data/mga-output-current-16384.json
python3 references/run_mga_agent.py --score references/sample_data/mga-output-current-16384.json \
    --reference references/sample_data/40567_i2JWkDyg26A_reference_steps.csv \
    --keywords references/sample_data/40567_i2JWkDyg26A_step_keywords.json
```

**Start with `--dry-run`.** It resolves nothing remotely and prints the exact project,
copy, package, deployment and session bodies, which is the cheapest way to confirm you
are setting a prompt and a sufficient budget before spending 10 minutes of shared GPU.

`mga-lifecycle.json` (`--state`) keeps the project, each setting's copy, package and
deployments, and every session the moment it is opened, so a rerun builds nothing new and
a client that died mid-run leaves nothing `--cleanup` can't find. It records the endpoint
too: pointed at another deployment, the script stops rather than use ids that don't exist
there.

## File Layout

```
skills/atai-manual-generation-agent/
├── SKILL.md
├── references/
│   ├── .env.example
│   ├── run_mga_agent.py          stdlib runner: project → copy → package → deployment → session → results → score
│   └── sample_data/
│       ├── README.md             attribution, and why no video ships here
│       ├── 40567_i2JWkDyg26A_reference_steps.csv
│       ├── 40567_i2JWkDyg26A_step_keywords.json            phrases naming each reference step, for --keywords
│       ├── mga-output-current-16384.json                   what the defaults produce (production 2026-10-27; lifecycle, dev 2026-11-05)
│       ├── mga-output-prod-old-prompt-SHIFTED.json          previous prompt, same day: texts ~5 s early
│       ├── mga-output-mouse-current-prompt.json             silent clip, current prompt: 17 distinct steps
│       ├── mga-output-mouse-old-prompt-LOOP.json            silent clip, previous prompt: 2 distinct texts
│       ├── mga-output-dev-old-prompt-18steps.json           previous prompt on dev, Aug 2026
│       ├── mga-output-current-4096-EMPTY.json               captured empty manual (Aug 2026); does not reproduce
│       └── three older outputs, kept as scoring fixtures
└── tests/
    └── test_references.py        network-free
```
