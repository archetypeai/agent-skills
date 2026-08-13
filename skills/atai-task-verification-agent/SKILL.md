---
name: atai-task-verification-agent
description: >
  Run Archetype AI's managed Task Verification (TVA) agent over the Agent
  API — upload a recording AND a reference procedure (an SOP), create a
  bundle from the `tva` blueprint, run it, poll the audit log, and download
  a per-step PASSED / FAILED / MISSING verdict with timestamps and a reason
  for each step. Use this skill when the user has a recording of work that
  was supposed to follow a known procedure and wants to know whether each
  step was actually performed — assembly QA, maintenance sign-off, training
  assessment, SOP compliance. Covers the runtime SOP input (`source.text ->
  PrepareSOPNode`, so one bundle serves every SOP), SOP authoring as the
  only tuning lever (there is no `prompt` value), the two silent failures
  that cost the most (a sink format with no registered connector, and the
  output budget being consumed by the model's `<think>` block so a run
  returns `results: []` while reporting `job.completed`), the output schema
  (`step, status, timestamp_start/end, reason`), and scoring against
  labelled clips. Do NOT use for generating a procedure from a video where
  none exists (that is `atai-manual-generation-agent`), for one-shot
  multimodal questions over a clip (`atai-newton-fusion-model` via
  `/query`), or for time-series state classification
  (`atai-operational-state-monitoring-agent`).
---

# TVA Agent — Managed Task Verification via the Agent API

The TVA agent checks a recording against a procedure you supply. You hand the platform an `.mp4` **and** an SOP as a `.txt`; it samples frames, transcribes the audio, fuses both with your procedure in one pass, and returns one verdict per SOP step:

```
video ──► sample frames ──► preprocess ─┐
                                        ├─► newton-fusion f1-0 ─► parse ─► per-step verdicts
SOP (.txt) ──► PrepareSOPNode ──────────┤
         whisper ASR ───────────────────┘
```

TVA V1 is **zero-shot**. The `tva` blueprint pins its own models (`newton-fusion:1.0` and `whisper:large-v3`), so like `mga` — and unlike `osm`/`red` — there is **no classifier to fit and no `artifacts` map to pass**.

## When to Apply

**Use when** the procedure is already known and the question is whether it was followed: did the operator install the o-ring, torque the fitting, apply the tape? The output is auditable — each verdict carries a time range and a reason citing what was on screen.

**Do NOT use when:**

| Need | Use instead |
|---|---|
| No procedure exists yet — extract one from the video | `atai-manual-generation-agent` (`mga`) |
| One question about a clip, stateless | `atai-newton-fusion-model` (`/query`) |
| Classify sensor windows into operational states | `atai-operational-state-monitoring-agent` |
| Detect a rare event in a long recording | `atai-rare-event-detection-agent` |

**TVA vs MGA in one line:** MGA *writes* the procedure, TVA *checks against* one. That difference is visible in the blueprints — and it is the reason TVA accepts a runtime input MGA does not:

```yaml
# mga                                       # tva
source:                                     source:
  key: MultiModalSource                       key: MultiModalSource
  config:                                     config: {}      # accepts a .txt
    default_text: ${values.prompt}
    text_extensions: []   # NO text input
```

MGA takes its instruction from a bundle *value*, fixed for the life of a bundle. TVA routes a `.txt` through `source.text -> prepare.in`, so **every run carries its own procedure and one bundle serves every SOP.**

## Endpoints

The Agent API is mounted **without a version prefix**; the files API is under `/v0.5`.

```
POST   {endpoint}/v0.5/files                          upload the video, then the SOP
POST   {endpoint}/agents/bundles                      create a bundle   ← PLURAL
POST   {endpoint}/agents/bundles/{bundle_id}/run      start a run       ← returns 202
GET    {endpoint}/agents/instances/{agent_id}/logs    the real log stream
GET    {endpoint}/agents/instances/{agent_id}/results output refs
POST   {endpoint}/agents/instances/{agent_id}/cancel  stop a run
```

`POST /agents/bundle` (singular) returns **404**. The `/agents` API is **dev-only** — a prod endpoint 404s on every `/agents` path, which reads like an unregistered blueprint rather than the wrong host.

## ⚠️ Read before spending a run

Three things will cost you a run or, worse, hand you a confident wrong answer. All
report success at the HTTP layer.

| | symptom | catch it with |
|---|---|---|
| **A false PASS on a skipped step** | three verdicts, all `PASSED`, fluent reasons | you cannot, from the output — see below |
| **Reasoning overflow** | `job.completed`, no ERROR, `results: []` | count the verdicts; read `dropping N` in the WARN |
| **Sink not instantiable** | `pod.terminated exit=1`, `/results` empty | read `connectors.sink.config.format` — but do not hard-code a denylist |

### 1. It verifies object PRESENCE, not performance — so `reason` is not evidence

Measured over 12 clips of a 3-step assembly with deliberate omissions: **0 of 6** on an
omitted o-ring, **3 of 3** on an omitted wrench, **1 of 1** on a prop present in no
clip. The rule that fits all of it: the model reliably answers *"is this thing in the
video at all?"* and, when it is, **assumes the action happened.** A workstation has its
parts laid out by definition, so the omissions you deployed the agent to catch are the
ones in the blind spot.

`reason` reads like an observation and is not one. Change a single adjective in a SOP
step and that adjective appears in the model's account of what it saw:

| the SOP writes | the reason writes |
|---|---|
| the **small black rubber** O-ring | "the O-ring" |
| the **black** O-ring | "the **black** O-ring" |
| "attaches, places, or aligns" | "**attaches** it to" — the first verb offered |

Frames for one false pass show the parts never moving during the interval the agent
itself reported. **Do not surface `reason` to a user as an audit trail, and do not use
its confidence as a signal** — a fabricated observation is word-for-word as assured as
a real one.

**Rewriting the SOP does not fix this.** Seven variants were tried — plain,
attention-directed, scene-anchored, interrogative, conditional, and a faithful port of a
prompt that reportedly scored 98% on the same clips via a direct query — and every one
returned the identical wrong verdict or no verdict at all. Budget for human review of
`PASSED` verdicts; do not budget for prompt engineering.

### 2. The output budget is shared with the reasoning block, and bigger is worse

f1-0 emits `<think>…</think>` before its answer and that reasoning spends
`max_new_tokens`. Run out inside it and the parser has nothing to parse — the job still
reports `job.completed` with no ERROR row and `results: []`.

The instinct, and the platform's own WARN, is to raise the budget. **That is wrong
here.** The window is not the constraint — the served `max_seq_len` is at least 18,754
and the blueprint defaults to 16,384 — and the model expands its reasoning to fill
whatever room it is given. On a clip with two skipped steps:

| `max_new_tokens` | result |
|---|---|
| **5760** | **21 rows, all verdicts correct** |
| 8192 | `results: []` |
| 16384 | `results: []`, identical |

Send **5760**, not the ceiling. The diagnostic that tells you which failure you have is
the `dropping N rehearsed row(s)` count in the WARN:

| N | what happened | what to do |
|---|---|---|
| large (37) | the verdict block was emitted ~12× — repetition | **lower** the budget |
| small (3) | drafted once, cut at `</think>` | budget is not the lever |
| `0` | never converged | budget is not the lever |

`N > 0` means the answer was formed and thrown away, so no extra budget was needed at
all — which is worth saying out loud when you report the failure upstream.

### 3. Read the sink format, but never hard-code which one is broken

`tva` shipped for ~18 hours with a sink format that had no registered connector, and
every run died at graph instantiation after the full ~7-minute model load. It was fixed
on 2026-08-12.

The lesson is not "avoid `json/per-request`" — that is the working format today. An
earlier version of `run_tva_agent.py` encoded it as a denylist and then **refused every
run against the fixed blueprint while sounding certain.** Warn on an unrecognised
format, never refuse, and re-test before trusting a constant you measured. Neither the
key nor a pinned id tells you whether the graph will instantiate: **read the document.**

## Step 1 — Write the SOP

**One step per line, in a `.txt`.** `PrepareSOPNode` accumulates the lines into a single prepared instruction, so **the line breaks are the step boundaries and nothing else is** — a wrapped line silently becomes two steps.

```
Step 1: Install the rubber O-ring into the blue block-off cap, ensuring it is properly seated in the groove.
Step 2: Thread the blue cap onto the black manifold by hand until snug, without cross-threading.
Step 3: Using a wrench, tighten the blue cap to the specified torque.
```

**There is no `prompt` value on this blueprint.** The system message lives inside `TaskVerificationPromptGeneratorNode` and is not exposed, so the SOP wording is your *entire* influence over the model. That is a real simplification rather than a gap: MGA's `prompt` is exposed and fragile — a prompt specifying its own output format makes its parser return zero steps — and TVA gives you no way to make that mistake.

What is worth putting in the wording:

| choice | why |
|---|---|
| **Explicit `Step N:` prefixes** | Redundant to the parser (`step` is positional and 0-based regardless), but they tie each verdict to a numbered step rather than to line order. |
| **Name colours and materials** | "the cap" is ambiguous on a 224×224 frame when the bench holds several similar fittings. Worth knowing the model volunteered *"the blue cap"* **without** being told the colour, so this is insurance, not a fix. |
| **Behavioural, not numeric** | "tighten to the specified torque", not "to 12 Nm". Nothing in a handheld video can evidence a number, so asking for one invites a guess. |
| **One action per line** | A line containing two actions gets one verdict, and you lose the ability to say which half failed. |

## Step 2 — Upload both inputs

```python
POST {endpoint}/v0.5/files          # multipart/form-data, field name "file"
→ {"is_valid": true, "file_id": "clip.mp4", "file_uid": "fil_…"}
```

Twice: once for the video, once for the SOP. The **declared** `Content-Type` is enforced against a MIME allowlist, not the bytes — `video/mp4` and `text/plain`. Use `file_id` (the basename) in the run payload, **not** `file_uid`; the uid fails at source resolution after the pod has started.

### Name the pair so the stems match — and so nothing can overwrite it

A run's inputs arrive as **one flat list of file ids**, with nothing saying which text belongs to which video, so the pipeline pairs them **by matching stems**. Upload as `<clip>-<sop>.mp4` and `<clip>-<sop>.txt`:

```
1_pass_2_pass_3_pass_A-oring-numbered.mp4
1_pass_2_pass_3_pass_A-oring-numbered.txt
```

The clip's stem alone would satisfy the matching, but then **every SOP variant collides on one name per clip.** That matters more than it sounds:

**`file_id` IS the basename, so re-uploading REPLACES the object a queued run is going to read.** A run pins its inputs at *input-resolution* time and dev can queue for an hour, so uploading the same name in that window kills whatever is already waiting — it surfaces minutes later, inside the run, as `S3 object not found` with `job.completed` on the job. Three runs were lost to this before the mechanism was clear.

`run_tva_agent.py` does two things about it: `pair_names()` derives the ids from both the clip and the SOP, and `upload()` compares local bytes against `GET /v0.5/files/download/{file_id}` and **skips when they match**. Keep the SOP in version control too — that, not the platform's file list, is the record of what a past run was checked against.

## Step 3 — Create a bundle (one, for every clip)

**No `artifacts` map** — the blueprint pins its own models.

```json
POST {endpoint}/agents/bundles
{"blueprint": "tva", "name": "task verification run",
 "values": {"max_frames": 64, "max_new_tokens": 5760}}
→ 201 {"id": "bnd_…", "status": "ready"}
```

| Value | Default | Notes |
|---|---|---|
| `max_new_tokens` | **5760** | **Shared with the `<think>` block, and bigger is worse.** The blueprint default has moved 2048 → 8192 → 16384; set it explicitly, and set it BELOW the ceiling — 8192 and 16384 both returned nothing on a clip where 5760 returned correct verdicts. |
| `max_frames` | 16 | Uniform across the whole video. On a 30 s clip, 16 is one frame every 1.9 s. 64 is the reader/preprocessor batch size. |
| `size` | 224 | Each frame resized to a square, so 1080p is no better than 480p — only slower to upload. |
| `parser_compute_stats` | false | Attaches template-conformance stats; useful when the parser returns nothing. |
| `parser_output_frame_indices` | true | Emits `frame_start`/`frame_end`. |
| `prompt` | **does not exist** | Not a defect to work around — absent. The SOP replaces it. |
| `min_temporal_similarity_threshold` | **not exposed** | The design doc's temporal-compression control. See Divergences. |

**One bundle is enough for every clip and every SOP.** Nothing that varies per run lives in the bundle: `values` are identical across clips, and both the video and the SOP are source connector inputs supplied at run time. If you find yourself creating a bundle per input, something is in `values` that belongs in a connector.

A value is honoured only if it is declared in the blueprint's `values` **and** referenced as `${values.<key>}` by some node or connector. Setting an unwired value returns **201** and echoes it back:

```python
wired = json.dumps({"nodes": doc["nodes"], "connectors": doc["connectors"]})
inert = [k for k in my_values
         if k not in doc["values"] or f"${{values.{k}}}" not in wired]
```

## Step 4 — Run it, with BOTH inputs

```json
POST {endpoint}/agents/bundles/{bundle_id}/run
{"connectors": {"source": [
  {"type": "file", "id": "1_pass_2_pass_3_pass_A-oring-numbered.mp4", "format": "mp4"},
  {"type": "file", "id": "1_pass_2_pass_3_pass_A-oring-numbered.txt", "format": "txt"}]}}
→ 202 {"id": "agt_…", "status": "running"}
```

**`MultiModalSource` routes by the `format` field, not by position** — order does not matter. Omit the text input and the run still starts, then the prompt generator waits on an instruction that never arrives.

**202, not 201.** A client treating only 201 as success reports a failure while the agent runs unattended on the GPU with nothing collecting its output.

`Resolved 2 inputs` in the log is the confirmation that the SOP arrived.

## Step 5 — Poll until terminal

Poll **`/logs`**, not `/events`. **Do not trust the `status` field** — it has read `running` 20+ minutes after a pod terminated with `exit=1`, and `running` after a job completed.

```python
TERMINAL = ("pod.terminated", "job.completed", "job.failed", "job.canceled")
rows = sorted(GET(f"{endpoint}/agents/instances/{agent_id}/logs?limit=500")["data"],
              key=lambda r: r["created_at"])
done   = rows and rows[-1]["event_type"] in TERMINAL
failed = any(r["level"] == "ERROR" for r in rows)
# and a WARN mentioning the reasoning block means results:[] — see above
```

**`job.admitted`'s queue figure is not the real wait.** One run reported `Job scheduled after 12s in queue` and then did not start a pod for **7 minutes**; another reported 32 s and idled **11+ minutes** with the instance reading `running` and no job of ours holding the GPU. `GET /agents/instances` is scoped to your org, so another org's job can block you with nothing in your view explaining it. Judge from `job.started`.

Between `input_started` and the terminal event there are **zero log rows** across ~5 minutes — no frame count, no transcript length, no per-node timing.

## Step 6 — Fetch results

```python
GET {endpoint}/agents/instances/{agent_id}/results
→ {"data": [{"data": {"filename": "agt_…__output_clip.json", "ref": "/files/download/…"}}]}
```

The `ref` is a **relative** platform path resolving under `/v0.5`, and needs the bearer token. **Run outputs do not expire**, so a client that dies — closed laptop, dropped network, Ctrl-C — has abandoned nothing. Re-fetch by agent id rather than re-running; the output filename embeds the input name (`…__output_<clip>.json`), which is enough to reassemble a batch after the fact.

## Output JSON — one record per video

```json
{"id":"1_pass_2_pass_3_pass_A","results":[
  {"step":0,"status":"PASSED","timestamp_start":2.0,"timestamp_end":9.0,
   "frame_start":60.0,"frame_end":270.0,
   "reason":"The video clearly shows the person picking up the blue cap and O-ring, then installing the O-ring onto the cap."}, …]}
```

| Field | Notes |
|---|---|
| `step` | **0-BASED.** A 3-step SOP returns 0, 1, 2. |
| `status` | `PASSED` · `FAILED` · `MISSING` |
| `timestamp_start/end` | Seconds. **Absent on `MISSING`** — there is no interval for a step that never happened, so handle nulls. |
| `frame_start/end` | **SOURCE frames**, not sampled: `frame = timestamp × source_fps` (810 = 27.0 × 30). |
| `reason` | Free text, and the most useful field — it cites what was on screen. |

Four properties the blueprint does not advertise:

- **`step` is 0-based, and an off-by-one fails QUIETLY.** Human-authored SOPs are 1-based. A naive join credits step 2's verdict to step 1 and reports the last step missing — which reads as a model that gave up, not a join bug. A perfect 3-of-3 run misread this way:

  ```
  step 1: expected PASS  got PASSED        [OK]
  step 2: expected PASS  got PASSED        [OK]
  step 3: expected PASS  got NOT REPORTED  [MISS]   ← the verdict existed, under step 2
  ```

- **`FAILED` and `MISSING` are different claims.** `FAILED` = attempted and done wrong; `MISSING` = never happened. Both mean "not done correctly", so a binary pass/fail label cannot distinguish them — report the split rather than collapsing it, because which one appears tells you whether the model saw an attempt.
- **The output does NOT tile the timeline.** MGA segments contiguously, so every second must be labelled; TVA emits one row per SOP step and rows may leave gaps. **TVA is verification, not segmentation.**
- **Generation is deterministic.** The same clip and SOP produced byte-identical output across two API keys in two orgs, 25 minutes apart (678 bytes, md5 `7eb028ac7d39938b261be17eaaf3e2bb`). A matching digest means an exact reproduction; a differing one means something really changed.

## Scoring — and the trap in it

A step the run **never reported** is a missing answer, **never a correct FAIL**. Credit it and an empty output scores 100% on every clip where a step was skipped — which is precisely what the reasoning overflow produces:

```python
ok = row is not None and (row["status"] == "PASSED") == expected_pass
```

And **an all-pass clip cannot validate a detector.** Three `PASSED` verdicts on a clip where everything was done correctly is exactly what a model that always answers `PASSED` would return. Always score clips containing a skipped step; `references/run_tva_agent.py --score` prints this warning when the labels are all `pass`.

## Runtime

| Phase | Cost |
|---|---|
| Queue | ~15–32 s solo; **796 s** behind another run, and can idle 11+ min invisibly |
| whisper download + load | ~35–40 s + ~3 s — **even with no audio track** |
| **newton-fusion download + load** | **~5 min + ~1 min 40 s** |
| Generation, 30 s clip | ~5 min |
| **Total** | **754–1614 s of job time.** Budget **20–30 min** wall clock — the queue is on top |

**Cold start is ~7 min on every run and is never cached** — two back-to-back runs on the same GPU each downloaded newton-fusion from scratch. So the marginal cost of one more clip is much lower than the first: batch them.

**Dev serializes these jobs.** One pod started **3 seconds** after the previous finished, having waited 7 minutes. Submitting concurrently buys queue position, not parallelism.

## Common Pitfalls

| Symptom | Cause |
|---|---|
| `404` on bundle creation | `/agents/bundle` is singular; use `/agents/bundles` |
| `404` on every `/agents` path | Prod endpoint. The Agent API is dev-only |
| `401` on dev with a working key | That key is for prod |
| Client reports failure, run proceeds | `/run` returns **202**, not 201 |
| `pod.terminated exit=1`, `instantiating graph: no connector registered` | The blueprint's sink format. Read the document first |
| `job.completed` but **`results: []`** | Reasoning filled the budget. Read `dropping N` in the WARN — then **lower** `max_new_tokens` toward 5760, do not raise it |
| Verdicts for some steps only | The SOP had a wrapped line, or the budget truncated mid-answer |
| Last `reason` cut mid-sentence | Raise `max_new_tokens` — this is the one case where raising helps |
| Every step scores correct on fail clips | A scorer crediting NOT REPORTED as a correct FAIL |
| Step 3's verdict looks like step 2's | `step` is 0-based; your SOP is 1-based |
| `MISSING` rows crash the consumer | They carry no `timestamp_start/end` |
| Run "hangs" at `running` | The `status` field lags; read `/logs` |
| Pod not started long after `job.admitted` | Another org's job. Invisible from your instance list |
| Source resolution didn't help | Every frame is resized to `size × size` (224) |

## Divergences from the design doc

Observations for the algorithm author, not defects.

- **`min_temporal_similarity_threshold` is not exposed.** The doc devotes its longest section to the temporal compression algorithm it controls — four stages, MAD-based adaptive segmentation, a documented *"70% compression with a 25% performance improvement"*. The blueprint wires only `model` and `max_new_tokens` into the fusion node, so **the headline algorithmic contribution of TVA V1 is not reachable from a bundle and cannot be evaluated through this API.**
- **Audio is in V1.** The doc says F1-0 adds audio in a later version; the blueprint runs `whisper:large-v3` into `prompt.audio` today. It also loads whisper for videos with **no audio track**, costing ~40 s per run for nothing.
- **`step` is 0-based and undocumented.** The doc specifies `step: <int>` without saying where it starts.
- **`MISSING` rows carrying no timestamps** matches the doc's own format (`step N: MISSING, reason: <reason>`), but belongs in the output schema, since consumers must handle nulls.
- **The model is fp8** — `f1_0_35b_a3b_fp8_base_040625b4d0750b`, confirming the 35B/3B-active MoE description and adding quantization the doc does not mention.

## Cleanup

Dev has **one GPU** and serializes jobs, so an abandoned run blocks everyone:

```python
POST {endpoint}/agents/instances/{agent_id}/cancel
```

**Do not send `DELETE` to an instance URL expecting a no-op** — it returns 204 and removes the run. Bundles are cheap to leave; runs are not.

## Local Setup

```sh
# No third-party deps — references/run_tva_agent.py is stdlib-only.
# Drop a .env next to where you run it (BOTH required, NO /v0.5 suffix:
# the script mounts /agents and /v0.5/files itself):
#   ATAI_API_KEY=<dev API key>
#   ATAI_API_ENDPOINT=https://api.dev.u1.archetypeai.app

python3 references/run_tva_agent.py \
    --video references/sample_data/1_pass_2_pass_3_pass_A.mp4 \
    --sop   references/sample_data/oring-numbered.txt

# --sop is OPTIONAL. It resolves to sop/oring-numbered.txt if your project has one,
# else to the SOP bundled in references/sample_data/. The script always prints which
# is in force, so the procedure a run was checked against is never a guess.
python3 references/run_tva_agent.py --video clip.mp4
python3 references/run_tva_agent.py --video clip.mp4 --sop my-sop.txt --dry-run

# offline: read and score a committed output, no key needed
python3 references/run_tva_agent.py --score references/sample_data/tva-output-1_pass_2_pass_3_pass_A.json
python3 references/run_tva_agent.py --score references/sample_data/tva-output-1_fail_2_pass_3_pass_A-FALSE-PASS.json   # a PASSED verdict that is false
python3 references/run_tva_agent.py --score references/sample_data/tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.json
```

## File Layout

```
skills/atai-task-verification-agent/
├── SKILL.md
├── references/
│   ├── .env.example
│   ├── run_tva_agent.py          stdlib-only: upload video+SOP → bundle → run → logs → results → score
│   └── sample_data/
│       ├── README.md                          what these are, and how to read them together
│       ├── 1_pass_2_pass_3_pass_A.mp4         all three steps performed
│       ├── 1_pass_2_pass_3_fail_A.mp4         step 3 skipped — the wrench never appears
│       ├── 1_fail_2_pass_3_pass_A.mp4         step 1 skipped — the o-ring is never fitted
│       ├── oring-numbered.txt                 the 3-step SOP both clips were run against
│       ├── tva-output-1_pass_2_pass_3_pass_A.json               3 PASSED, with reasons
│       ├── tva-output-sealant-CORRECT-MISSING.json              correct MISSING on an absent prop
│       ├── tva-output-1_pass_2_pass_3_fail_A-CORRECT-MISSING.json  correct MISSING on a skipped step
│       ├── tva-output-1_fail_2_pass_3_pass_A-FALSE-PASS.json    PASSED on a step never performed
│       └── tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.json the reasoning overflow, 44 bytes
└── tests/
    └── test_references.py        network-free
```

Worked end-to-end example, with twelve labelled clips, a batch sweep and an offline scorer:
[task-verification-agent-example](https://github.com/archetypeai/task-verification-agent-example).
