---
name: atai-task-verification-agent
description: >
  Run Archetype AI's managed Task Verification (TVA) agent over the Agents
  API — upload a recording AND a reference procedure (an SOP), create a
  bundle from the `tva` blueprint, run it, poll, download a per-step
  PASSED / FAILED / MISSING verdict per step. Use when
  the user has a recording of work that should have followed a known
  procedure and wants to know whether each step was performed — assembly
  QA, maintenance sign-off, training assessment, SOP compliance. Covers the
  runtime SOP input (one bundle serves every SOP), SOP authoring as the only
  tuning lever, the output schema, the measured reliability limit — a
  skipped step whose tool or part is visible comes back PASSED with an
  invented reason — and the empty-`results` failure that still reports
  success. Do NOT use for generating a procedure where none exists
  (`atai-manual-generation-agent`), for one-shot questions over a clip
  (`atai-newton-fusion-model`), or for time-series state classification
  (`atai-operational-state-monitoring-agent`).
---

# TVA Agent — Managed Task Verification via the Agents API

The TVA agent checks a recording against a procedure you supply. You hand the platform an `.mp4` **and** an SOP as a `.txt`; it samples frames, transcribes the audio, fuses both with your procedure in one pass, and returns one verdict per SOP step:

```
video ──► sample frames ──► preprocess ─┐
                                        ├─► newton-fusion f1-0 ─► parse ─► per-step verdicts
SOP (.txt) ──► PrepareSOPNode ──────────┤
         whisper ASR ───────────────────┘
```

TVA V1 is **zero-shot**. The `tva` blueprint pins its own models (`newton-fusion:1.0` and `whisper:large-v3`), so like `mga` — and unlike `osm`/`red` — there is **no classifier to fit and no `artifacts` map to pass**.

## When to Apply

**Use when** the procedure is already known and the question is whether it was followed: did the operator install the o-ring, torque the fitting, apply the tape? Each verdict carries a status, a time range and a reason.

**Do not treat that output as an audit trail.** A `PASSED` verdict is not evidence the step happened, and the `reason` is a restatement of your SOP rather than a report of the video — measured, with the frames to prove it, in §1 below. **Use this to triage which recordings a human should watch, not to sign work off unreviewed.**

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

The Agents API is mounted **without a version prefix**; the files API is under `/v0.5`.

```
POST   {endpoint}/v0.5/files                          upload the video, then the SOP
POST   {endpoint}/agents/bundles                      create a bundle   ← PLURAL
POST   {endpoint}/agents/bundles/{bundle_id}/run      start a run       ← returns 202
GET    {endpoint}/agents/instances/{agent_id}/logs    the real log stream
GET    {endpoint}/agents/instances/{agent_id}/results output refs
POST   {endpoint}/agents/instances/{agent_id}/cancel  stop a run
```

`POST /agents/bundle` (singular) returns **404** — the plural `/agents/bundles` is the endpoint. The Agents API is live on the production deployment (`https://api.u1.archetypeai.app`), where this skill is verified end-to-end.

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

### 2. The output budget is shared with the reasoning block

f1-0 emits `<think>…</think>` before its answer and that reasoning spends
`max_new_tokens`. Run out inside it and the parser has nothing to parse — the job still
reports `job.completed` with no ERROR row and `results: []`. **Check `/results` is
non-empty before trusting any run.** That check is the one thing in this section that
never goes stale.

**What the budget actually changed, measured 2026-08-20 on Prod.** Every shipped clip
against the shipped 3-step SOP, at four budgets — 12 runs, 36 verdicts:

| clip | 2048 | 5760 | 8192 | 16384 |
|---|---|---|---|---|
| `1_pass_2_pass_3_fail_A` (wrench never enters frame) | 3/3 | 3/3 | 3/3 | 3/3 |
| `1_fail_2_pass_3_pass_A` (O-ring never fitted, rings visible) | 2/3 | 2/3 | 2/3 | 2/3 |
| `1_pass_2_pass_3_pass_A` (control, all steps performed) | 3/3 | 3/3 | 3/3 | 3/3 |

**Nothing about the budget mattered.** Every cell returned three well-formed verdicts:
no `results: []`, no WARN, no truncation, across an 8× range. Each clip scored
identically at every budget, and per clip the verdicts and reason text were the same
word for word — only the md5 varied. Reproduce any cell with
`--max-new-tokens <n>`; all three clips and the SOP ship in `references/sample_data/`.

What decided correctness was **what is absent from the frame**, not the budget:

- An absent **object** — the wrench, never on screen — is caught: `MISSING`, at every budget.
- An absent **action on a visible object** — the O-ring, sitting on the mat the whole
  time — is not: falsely `PASSED`, at every budget, with the same invented reason each
  time. **Raising or lowering `max_new_tokens` does not turn a wrong verdict into a
  right one**, which is worth knowing because the budget dial is the first thing you
  will reach for. See §1.

So for a short SOP, send whatever keeps your runs comparable and spend your attention on
reviewing `PASSED` verdicts instead.

#### `results: []` — plan for it, do not predict it

It is a real outcome and you should handle it, but **there is no threshold to memorise.**
Whether a run lands in it depends on how much reasoning the model does before answering,
which is an interaction of the video, the SOP's step count, and the budget — not the
budget alone. It has been observed at 2048 on a clip with a skipped step, and at 8192 and
16384 on a 21-step SOP; on 2026-08-20 none of the twelve runs above hit it at any budget.
Assume it can happen on any combination you have not tried, including one that worked
yesterday.

What it looks like — every signal says success:

```
HTTP           every call 2xx
job status     job.completed, no ERROR row in /logs
output         {"id": "…", "results": []}     ~40 bytes
```

So detect it explicitly, in this order:

1. **`/results` must be non-empty.** This is the check; nothing else catches it.
2. **`invalid` must not be `"true"` on every row** — a populated output can still be
   entirely invalid windows.
3. **Read `/logs` for the WARN**, which names the cause and the direction to move in
   (table below). Absence of the WARN with an empty output means something other than
   the reasoning block consumed the run.

Then respond by what the WARN says rather than by instinct: `dropping N` large means the
answer was produced and discarded, so **lower** the budget; small or `0` means the budget
is not the lever and more of it will not help. If you change the budget, change one
thing and re-run — generation is deterministic for a given budget, so a re-run at the
same value returns the same outcome.

Two things not to conclude. **Empty does not mean "too small"** — that is the instinct,
and half the observed cases went the other way. And **a clean clip passing at some budget
does not mean a defect clip will** — the runner's own note once hypothesised that clean
clips fit where defect clips do not; the grid above did not bear that out, with both
defect clips returning full verdicts at the lowest budget tried.

<details>
<summary><b>Earlier measurement (2026-08-11/12) — dated, and did not reproduce</b></summary>

On a clip with two skipped steps against a **21-step** SOP, this skill previously
measured a hard ceiling and prescribed 5760:

| `max_new_tokens` | result |
|---|---|
| **5760** | 21 rows, all verdicts correct |
| 8192 | `results: []` |
| 16384 | `results: []`, identical |

`sample_data/tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.json` is a captured
artifact of that failure at 2048 — kept because nothing else shows what it looks like,
not because it reproduces.

None of it held on 2026-08-20, including at the exact clip and budget of that fixture.
The `atai-manual-generation-agent` sibling documents the **opposite** rule — 16384 as a
floor, with 2048 and 4096 returning an empty manual — and that did not reproduce
either: both returned the full 18-step manual, instruction-for-instruction and
timestamp-for-timestamp identical to the 16384 run. Two blueprints, opposite
prescriptions, neither observable now; the shared reasoning behaviour is the likely
variable rather than anything specific to either blueprint.

**Non-reproduction is not retirement.** The failure was observed, and the platform still
emits a WARN for exactly this case. Two things follow. First, keep the `/results` check
above. Second, **do not infer a safe budget from a short SOP** — reasoning cost scales
with the number of steps to adjudicate, and the 21-step case is not reproducible from
this repo, so the grid above says nothing about a 20-step procedure.

</details>

When the WARN does fire, the `dropping N rehearsed row(s)` count tells you which failure
you have:

| N | what happened | what to do |
|---|---|---|
| large (37) | the verdict block was emitted ~12× — repetition | **lower** the budget |
| small (3) | drafted once, cut at `</think>` | budget is not the lever |
| `0` | never converged | budget is not the lever |

`N > 0` means the answer was formed and thrown away, so no extra budget was needed at
all — which is worth saying out loud when you report the failure upstream.

### 3. Read the sink format from the blueprint document

A sink with no registered connector dies at graph instantiation — **after** the full
~7-minute model load, with `/results` empty. `check_sink()` reads
`connectors.sink.config.format` in about a second and **warns rather than refuses**: a
format observed broken once may be fixed by the time you run. Neither the key nor a
pinned id tells you whether the graph will instantiate.

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

### Name the pair so it says what it is, and so nothing can overwrite it

A run's inputs arrive as **one flat list of file ids** with nothing saying which text belongs to which video, so the pipeline works it out from the names. Matching is **substring, not equality** — a `<stem>.txt` pairs with any video whose name *contains* `<stem>`:

| you pass | what happens |
|---|---|
| **1 video + 1 SOP** | that procedure applies **whether or not the names match** |
| 1 SOP + N videos | the same procedure applies to every video, stems irrelevant |
| N SOPs + N videos | each video takes the SOP whose stem it contains; **longest match wins**, with a warning |
| a video no SOP matches | falls back to `default_text`, and is **skipped** if that is unset |

So for a 1:1 run — what this script does — naming cannot break pairing. It still matters for two other reasons: **substring matching is loose** (`file-1.txt` pairs with `cam-file-1-run.mp4`, and short or overlapping stems mis-pair in multi-input runs), and an id is the only record of which procedure a run used. Upload as:

```
1_pass_2_pass_3_pass_A-oring-numbered-20260813T193901Z-56ce.mp4
1_pass_2_pass_3_pass_A-oring-numbered-20260813T193901Z-56ce.txt
```

| part | why |
|---|---|
| the clip stem | recognisable in an org-wide file list |
| the SOP stem | an id says WHICH procedure a run was checked against |
| `<UTC timestamp>` | sorts and greps; tells you which upload was yours and when |
| `<4 hex>` | what actually guarantees uniqueness — **two people starting in the same second is exactly the case being fixed** |

**Both halves must share the suffix** — the pipeline matches them by stem, so a per-file suffix would break the pairing it exists to protect. `run_tva_agent.py` generates one per run with `run_suffix()`.

The last two parts are not decoration. **An org shares ONE flat file namespace**, so without them two people verifying the same clip write to the same object:

**`file_id` IS the basename, so re-uploading REPLACES the object a queued run is going to read.** A run pins its inputs at *input-resolution* time and a run can queue for an hour, so uploading the same name in that window kills whatever is already waiting — it surfaces minutes later, inside the run, as `S3 object not found` with `job.completed` on the job.

A unique suffix makes that **impossible by construction** rather than something to defend against. `upload()` also still compares local bytes against `GET /v0.5/files/download/{file_id}` and skips when they match, which matters if you pass explicit names.

Two consequences: unique ids mean the skip never fires on the default path, so each run re-uploads its video (~0.8 s for 8 MB against a ~1000 s run — noise), and **uploads accumulate**, because this API has no file-cleanup endpoint. Keep the SOP in version control too: that, not the platform's file list, is the record of what a past run was checked against.

## Step 3 — Create a bundle (one, for every clip)

**No `artifacts` map** — the blueprint pins its own models.

```json
POST {endpoint}/agents/bundles
{"blueprint": "tva", "name": "tva 1_pass_2_pass_3_fail_A mnt5760",
 "values": {"max_frames": 64, "max_new_tokens": 5760}}
→ 201 {"id": "bnd_…", "status": "ready"}
```

> **Name the bundle after its inputs.** A bundle's `name` is set at creation and
> **cannot be changed** — `PATCH` and `PUT` on `/agents/bundles/{id}` both return
> **405** — and it is the only thing distinguishing your runs from each other in
> the console. Reuse one constant name across a batch and every row looks
> identical; you then have to recover which bundle was which from your own local
> records. `run_tva_agent.py` defaults `--name` to `tva <clip> mnt<budget>` for
> this reason: the clip and the token budget are what actually differ between
> runs. Whatever convention you pick, encode the values you varied.

| Value | Default | Notes |
|---|---|---|
| `max_new_tokens` | **5760** | Shared with the `<think>` block. The blueprint default has moved 2048 → 8192 → 16384, so set it explicitly to keep runs comparable. 5760 is what this runner sends; on a 21-step SOP in Aug 2026 it was the only budget that returned rows, though that no longer reproduces — see "The output budget is shared with the reasoning block". |
| `max_frames` | 16 | Uniform across the whole video. On a 30 s clip, 16 is one frame every 1.9 s. 64 is the reader/preprocessor batch size. |
| `size` | 224 | Each frame resized to a square, so 1080p is no better than 480p — only slower to upload. |
| `parser_compute_stats` | false | Attaches template-conformance stats; useful when the parser returns nothing. |
| `parser_output_frame_indices` | true | Emits `frame_start`/`frame_end`. |
| `prompt` | **does not exist** | Not a defect to work around — absent. The SOP replaces it. |
| `min_temporal_similarity_threshold` | **not exposed** | The design doc's temporal-compression control — specified there, unreachable from a bundle. |

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
  {"type": "file", "id": "1_pass_2_pass_3_pass_A-oring-numbered-20260813T193901Z-56ce.mp4", "format": "mp4"},
  {"type": "file", "id": "1_pass_2_pass_3_pass_A-oring-numbered-20260813T193901Z-56ce.txt", "format": "txt"}]}}
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
| `reason` | Free text that reads like a report of the video. **It is a restatement of your SOP step — see §1. Do not use it as evidence or as a confidence signal.** |

Five properties the blueprint does not advertise:

- **`step` is 0-based, and an off-by-one fails QUIETLY.** Human-authored SOPs are 1-based. A naive join credits step 2's verdict to step 1 and reports the last step missing — which reads as a model that gave up, not a join bug. A perfect 3-of-3 run misread this way:

  ```
  step 1: expected PASS  got PASSED        [OK]
  step 2: expected PASS  got PASSED        [OK]
  step 3: expected PASS  got NOT REPORTED  [MISS]   ← the verdict existed, under step 2
  ```

- **`FAILED` and `MISSING` are different claims.** `FAILED` = attempted and done wrong; `MISSING` = never happened. Both mean "not done correctly", so a binary pass/fail label cannot distinguish them — report the split rather than collapsing it, because which one appears tells you whether the model saw an attempt.
- **The output does NOT tile the timeline.** MGA segments contiguously, so every second must be labelled; TVA emits one row per SOP step and rows may leave gaps. **TVA is verification, not segmentation.**
- **`len(results)` is NOT the step count.** The model repeats itself and the parser keeps every repetition — one 3-step SOP returned **21 rows**. Collapse by `step` before scoring, and prefer the informative row: a placeholder (`reason: "..."`, no timestamps) can arrive *first*. Nothing in the pipeline compares the row count to the SOP's step count, so a self-evidently wrong result passes through as data.
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

**Do not count on these jobs being serialized.** Whether concurrent submissions queue depends on what else is running on the deployment at that moment: they queue when other workloads hold the workers, and come up as concurrent jobs when they don't. Queuing has been observed here — one pod waited 7 minutes and started **3 seconds** after the previous finished — but other tenants' workloads aren't visible to you, so that is an observation, not a guarantee. Submit one at a time.

## Common Pitfalls

| Symptom | Cause |
|---|---|
| `404` on bundle creation | `/agents/bundle` is singular; use `/agents/bundles` |
| `404` on every `/agents` path | A `/vX.Y` prefix on the endpoint. The Agents API is versionless — `/agents`, never `/v0.5/agents` |
| `401` with a key that works elsewhere | API keys are **deployment-scoped** — each deployment needs its own (verified: a key issued for one 401s on another) |
| Client reports failure, run proceeds | `/run` returns **202**, not 201 |
| `pod.terminated exit=1`, `instantiating graph: no connector registered` | The blueprint's sink format. Read the document first |
| `job.completed` but **`results: []`** | Generation ended inside the reasoning block. Read `dropping N` in the WARN: large N means the answer was formed and dropped, so **lower** the budget; small or `0` means the budget is not the lever |
| **A step that was skipped came back `PASSED`** | **Expected, and not fixable from the SOP.** The part was on screen, so the action was assumed — see §1. Human-review every `PASSED` |
| `S3 object not found`, inside a run that had already started | Something re-uploaded an input name while this run sat in the queue — often a colleague on the same org. Give every upload a `<UTC>-<hex>` tail |
| Verdicts for some steps only | The SOP had a wrapped line, or the budget truncated mid-answer |
| Last `reason` cut mid-sentence | Raise `max_new_tokens` — this is the one case where raising helps |
| Every step scores correct on fail clips | A scorer crediting NOT REPORTED as a correct FAIL |
| Step 3's verdict looks like step 2's | `step` is 0-based; your SOP is 1-based |
| `MISSING` rows crash the consumer | They carry no `timestamp_start/end` |
| Run "hangs" at `running` | The `status` field lags; read `/logs` |
| Pod not started long after `job.admitted` | Another org's job. Invisible from your instance list |
| Source resolution didn't help | Every frame is resized to `size × size` (224), so 1080p carries no more detail than 480p |
| Raising `size` returned nothing | Visual tokens grow quadratically with it, starving generation. One run at 448 had to drop to 32 frames and produced no verdicts |

## Cleanup

A deployment with a single GPU serializes these jobs, so an abandoned run can
block everyone else's:

```python
POST {endpoint}/agents/instances/{agent_id}/cancel
```

**Do not send `DELETE` to an instance URL expecting a no-op** — it returns 204 and removes the run. Bundles are cheap to leave; runs are not.

## Local Setup

```sh
# No third-party deps — references/run_tva_agent.py is stdlib-only.
# Drop a .env next to where you run it (BOTH required, NO /v0.5 suffix:
# the script mounts /agents and /v0.5/files itself):
#   ATAI_API_KEY=<your API key for that environment>
#   ATAI_API_ENDPOINT=https://api.u1.archetypeai.app

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
│       ├── tva-output-1_pass_2_pass_3_fail_A-CORRECT-MISSING.json  correct MISSING on a skipped step
│       ├── tva-output-1_fail_2_pass_3_pass_A-FALSE-PASS.json    PASSED on a step never performed
│       └── tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.json the reasoning overflow, 44 bytes
└── tests/
    └── test_references.py        network-free
```

Worked end-to-end example, with twelve labelled clips, a batch sweep and an offline scorer:
[task-verification-agent-example](https://github.com/archetypeai/task-verification-agent-example).
