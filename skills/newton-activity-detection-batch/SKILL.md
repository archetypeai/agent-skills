---
name: newton-activity-detection-batch
description: >
  Run asynchronous batch jobs on Archetype AI Newton via the
  `activity-detection` pipeline — the Newton C language model invoked
  over a JSONL prompt file. Each line is an InferenceRecord with
  `inputs[]` carrying text, image, or video evidence (inline base64).
  Use when the user wants to summarize, narrate, or reason over large
  structured datasets (CSV logs, sensor flows, event streams) at
  offline scale, or to run per-record vision Q&A across many short
  videos / image batches without holding a `/query` session open.
  Covers job creation, polling, output download, cancellation,
  per-input-type schemas, the empirical ~4K-token quality cliff for
  text-heavy inputs, MapReduce / hierarchical reduce patterns for
  GB-scale inputs, N-way positional split for batch concurrency, and
  the silent join bugs (content-key vs. position-based join,
  `line_index` collisions after concat) that bite when chaining reduce
  stages.
  Do NOT use for time-series classification (use newton-machine-state-batch).
  Do NOT use for live streaming video Q&A (use newton-activity-monitor —
  it streams via Lens session with low latency, where this skill is
  async batch).
  Do NOT use for synchronous text reasoning with a fixed-size state
  snapshot (use newton-query-prompting + the `/query` endpoint).
  Do NOT use for file uploads (use newton-batch-upload).
---

# Newton Activity Detection (Batch Text Generation)

Create and monitor asynchronous batch jobs that run the **Newton C language model** against a JSONL prompt file. Each line in the input is an `InferenceRecord` (`system` + `instruction` + `prompt` + optional `inputs` array); each line in the output is `{"line_index": N, "prediction": "..."}`. The pipeline is named `activity-detection` on the platform.

The streaming/synchronous counterpart is [`newton-query-prompting`](../newton-query-prompting/SKILL.md) — same C model, different orchestration. Use this skill when you need to fold thousands-to-millions of prompts asynchronously; use `/query` when you have one state snapshot and need an answer in 3–6 s.

Worked end-to-end example, including the full MapReduce code: [**archetypeai-batch-examples-ghost-iot**](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot) — 1 GB of synthetic WiFi flow CSV (5.8 M flows across 24 devices × 24 hours) folded into 9 daily narratives via six reduce stages. The numbers and pattern depths quoted below come from that pipeline; the patterns are model- and dataset-independent — only the depth of the hierarchy changes with the cliff value.

## When to Apply

- User wants Newton to write per-row / per-record narratives over a large dataset (logs, flows, events) where the input is too big or too numerous for `/query`.
- User wants to fold many small inputs into one summary via MapReduce / hierarchical reduce.
- User is hitting silent quality degradation on long prompts and isn't sure why ("status COMPLETED, predictions are garbage").
- User is chaining reduce stages and getting cross-scope leakage in the final narratives.

## Pipeline shape

```
Pipeline key:   activity-detection
Input port:     worker.data        (one JSONL file; each line = InferenceRecord)
Output:         outputs/pred_*.jsonl
                  one JSON per line: {"line_index": N, "prediction": "..."}
```

### Input record schema

```json
{
  "system": "You are a network security analyst reviewing smart-home WiFi traffic...",
  "instruction": "Analyze the attached flow log and describe what happened on this day. Summarize activity level, dominant protocols, temporal patterns, most-active device, and flag anything unusual.",
  "prompt": "Date: 2019-10-19 UTC. Scope: home wlan0 interface. Flow count: 78.",
  "inputs": [
    {
      "type": "text",
      "format": "plain",
      "data": "Flow log fields (pipe-separated): time_utc|mac_a|mac_b|prot|tran|port_a|port_b|bytes_a|bytes_b|pkts_a|pkts_b. Transport: 6=TCP, 17=UDP, 1=ICMP.\n\n15:55:11|ebd1a7fa8544|e323b826aa71|DHCP|17|67|68|0|900|0|3\n15:55:14|e323b826aa71|13d35af5c06b|DHCP|17|68|67|900|0|3|0\n..."
    }
  ]
}
```

The **question** goes in `prompt`, the **evidence** goes in `inputs[0].data`. The `inputs` array is the InferenceRecord schema's contract for extra context. Full schema: [Nano Inference input format](https://github.com/archetypeai/atai_core/tree/main/services/jos_service/nano_inference#input-format).

### Input types supported by `inputs[]`

Per the platform's validation error message when the wrong shape is submitted:

```
Expected JSONL with fields: system?, instruction?, prompt?,
  inputs?: [{type: text|image|video, format: base64|plain, data, extra?}]
```

So each item in `inputs[]` is `{type, format, data}`. Note that **`file_id`
is NOT accepted here** — bytes must be embedded inline.

| `type`  | `format` | `data` | Notes |
|---|---|---|---|
| `text`  | `plain`  | Raw text (CSV rows, flow logs, prose context) | Default for structured-data MapReduce — see the ghost-iot example below. |
| `image` | `base64` | Base64-encoded JPEG/PNG bytes | One or more frames sampled from a video, screen captures, dashboard renders. |
| `video` | `base64` | Base64-encoded MP4 bytes | The pipeline samples up to `max_video_frames` (default 32) frames internally. Verified against `newton/c:2.5.1-8b-base` (vision-only, no audio); model variants with audio may behave differently. |

Vision example record (one 90 s MP4 chunk per record):

```json
{
  "system": "You are analyzing a procedural video. Produce a written step-by-step guide based only on what is visible.",
  "instruction": "Create a step-by-step procedural guide for the work shown in this video segment.",
  "prompt": "Source: machine_repair.mp4, segment 0-90s (chunk_index=0).",
  "inputs": [
    {"type": "video", "format": "base64", "data": "<base64-encoded MP4 bytes ~25–35 MB per 90 s @ 1080p>"}
  ]
}
```

Image-mode equivalent (N stills, no video decode):

```json
{
  "inputs": [
    {"type": "image", "format": "base64", "data": "<base64 JPEG>"},
    {"type": "image", "format": "base64", "data": "<base64 JPEG>"},
    {"type": "image", "format": "base64", "data": "<base64 JPEG>"}
  ]
}
```

### Uploading the JSONL for vision-heavy records

A multi-record JSONL with base64-embedded videos quickly grows past 100 MB
(eight 90 s 1080p chunks ≈ 240 MB). The simple `POST /files` endpoint
**rejects `application/x-ndjson`** at its multipart-form validator (despite
listing it in the supported-types error message — known platform
inconsistency). Use the 3-step **presigned multipart flow** for the JSONL:

```
POST /v0.5/files/uploads/initiate  { filename, file_type: "application/x-ndjson", num_bytes }
PUT   <presigned-url>              <part-bytes>   (one PUT per part)
POST /v0.5/files/uploads/{upload_id}/complete  { parts: [...] }
```

See [`newton-batch-upload`](../newton-batch-upload/SKILL.md) for the full
flow. Frame uploads to `/files` (image/jpeg, video/mp4) work fine via the
simple endpoint — only `application/x-ndjson` needs the multipart path.

## Creating a Batch Job

```bash
curl -s -X POST "$BASE_URL/batch/jobs" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-narrative-job",
    "pipeline_type": "batch",
    "pipeline_key": "activity-detection",
    "inputs": {
      "worker.data": [{"file_id": "my_prompts.jsonl"}]
    },
    "parameters": {
      "worker": {
        "parallelism": 1,
        "config": {
          "generation": {
            "do_sample": true,
            "max_new_tokens": 1024,
            "repetition_penalty": 1,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8
          }
        }
      }
    }
  }'
```

Response: `{"id": "job_...", "status": "PENDING", ...}`. The standard lifecycle is `PENDING → RUNNING → COMPLETED / FAILED / CANCELLED`.

### Generation knobs

| Knob | Typical | Notes |
|---|---|---|
| `max_new_tokens` | 512–1024 (text), 2048–4096 (vision) | Newton truncates mid-sentence at the cap. Size for your longest expected narrative + ~20 % headroom. |
| `temperature` | 0.5–0.8 | Lower = more consistent across reduce passes (useful when you'll concatenate predictions back). |
| `do_sample` | `true` | `false` gives greedy decoding — sometimes useful for deterministic fold stages. |
| `top_k` / `top_p` | 20 / 0.8 | Defaults match `/query`. Rarely worth tuning unless you see repetition. |

### Worker-config knobs (siblings of `generation`)

Visible in the platform Batch Manager UI; confirmed from the GET
`/batch/jobs/{id}` response. All live under `parameters.worker.config`:

| Knob | Default | Notes |
|---|---|---|
| `model_variant` | `newton/c:2.5.1-8b-base` (today) | Override to pin a specific model build. Whether other variants are exposed depends on your org. |
| `max_video_frames` | `32` | How many frames the pipeline samples from a `type: "video"` input. Higher = denser visual coverage but pushes you toward the per-record token budget; lower = leaner records, faster inference, less faithful procedure recall. |
| `batch_size` | auto-escalates to 16–24 | **Pin to `1` for media-heavy records.** The engine auto-escalates and `OOMKilled`s the pod when records (~30 MB base64 video) exceed VRAM. Same lesson the [ghost-iot CSV cliff](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot#reason-2-csv-heavy-quality-cliff-no-warning-silent-garbage) flagged in a different shape. |
| `parallelism` | `1` | Multi-worker not currently supported. Get parallelism by submitting N independent jobs. |

## Monitoring & Outputs

```bash
# Poll status
curl -s "$BASE_URL/batch/jobs/$JOB_ID" -H "Authorization: Bearer $ATAI_API_KEY"

# Events / logs
curl -s "$BASE_URL/batch/jobs/$JOB_ID/events" -H "Authorization: Bearer $ATAI_API_KEY"

# List output artifacts (paginated, presigned S3 URLs, 1-hour expiry)
curl -s "$BASE_URL/batch/jobs/$JOB_ID/outputs?limit=50&offset=0" \
  -H "Authorization: Bearer $ATAI_API_KEY"

# Download an artifact (no auth — signature is in the URL)
curl -s -o pred_0.jsonl "$PRESIGNED_URL"
```

Output line:

```json
{"line_index": 0, "prediction": "On 2019-10-19 the home wlan0 interface saw 78 flows..."}
```

On per-record failure:

```json
{"line_index": 5, "prediction": null, "error": "parse error"}
```

`"parse error"` almost always means **the input wasn't JSONL** (raw CSV submitted by mistake, or a stray blank line). The job itself still returns `COMPLETED`.

## Cancelling a Job

A submitted batch job can be cancelled while it's in `PENDING` or `RUNNING`. The platform stops compute and the job's terminal status flips to `CANCELLED`.

```bash
curl -s -X POST "$BASE_URL/batch/jobs/$JOB_ID/cancel" \
  -H "Authorization: Bearer $ATAI_API_KEY"
```

Response (200 OK):

```json
{
  "id": "job_2abc3def4ghi5jkl6mno7pqr",
  "status": "CANCELLED",
  "cancelled_at": "2026-04-14T10:10:00Z"
}
```

Response status codes:

| Code | Meaning |
|---|---|
| `200` | Cancel accepted. Job will transition to `CANCELLED` (may not be instantaneous; the platform tears down the worker first). |
| `404` | Job id not recognized. |
| `409` | Job is already in a terminal state (`COMPLETED`, `FAILED`, `CANCELLED`). Nothing to cancel. |

**Best treated as idempotent + best-effort**: if you're cancelling defensively in client code, treat `409` as success — the job is already done one way or another, so there's nothing more to stop.

```python
# Pattern: don't bubble 409 to the user — it just means "already gone"
resp = requests.post(f"{base}/v0.5/batch/jobs/{job_id}/cancel", headers=headers)
if resp.ok or resp.status_code == 409:
    pass  # cancelled or already terminal — same outcome
else:
    raise RuntimeError(f"cancel failed: {resp.status_code} {resp.text}")
```

### Caveats worth knowing

- **No per-record cancellation.** Cancel is a job-level operation. If you have 50 records mid-batch and cancel, the platform stops scheduling new ones; whatever record was actively running may complete on its hot pod before the worker tears down.
- **No per-`/query` cancellation.** The synchronous `/query` endpoint (covered by [`newton-query-prompting`](../newton-query-prompting/SKILL.md)) has no documented cancel. Aborting the HTTP request closes the client connection but server-side compute may still run to completion.
- **Cancel only releases *future* work.** Anything that already completed and was billed remains billed. If you're seeing partial output via `/outputs` already, cancelling won't refund it.
- **The `cancelled_at` timestamp is set when the platform accepts the cancel**, not when the worker actually terminates. Subsequent `GET /batch/jobs/$JOB_ID` calls will show `status: "CANCELLED"` immediately, but `input_progress` may still show a record `processing` for a few seconds.

## The Quality Cliff (read this before designing anything)

The Newton C model has a **nominal context budget of 16,384 tokens** but a much tighter **practical quality cliff at ~4K tokens (~16 KB of pipe-separated structured rows)**. Above the cliff the model silently degrades into table-completion mode and returns 1–3-character fragments like `'00'`, `'|153|'`, or empty strings.

**The failure is silent.** No `WARN` event, no `FAILED` status. The job reports `Status: COMPLETED`, `0 failed`. The only signal is reading the predictions themselves.

Empirical cliff sweep on CSV-heavy `inputs[0].data` (from the [ghost-iot worked example](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot), §10.6 of the README):

| `inputs[0].data` size | ~Tokens | Output |
|---|---|---|
| 10.4 KB | ~2.5K | Real narratives ✓ |
| **16.5 KB** | **~4K** | **Real narratives ✓ (last good)** |
| 20.6 KB | ~5K | ✗ Garbage fragments |
| 41.1 KB | ~10K | ✗ Garbage + truncation warning fires |

The cliff and the nominal context window are **independent failure modes**. The model can ingest the tokens and emit plausible-sized output — it just can't synthesize narrative from that much structured data at once.

### Cliff sweep methodology

Before designing a multi-day pipeline, spend 30 minutes establishing the cliff for *your* model + *your* input shape:

1. Take 5 representative records.
2. Build 4 variants per record, capped at 10 KB / 16 KB / 20 KB / 40 KB of `inputs[0].data`.
3. Submit all 20 as a single batch job.
4. Read every prediction. The cliff is the largest cap where all 5 still produce real narratives.

**Revalidate per model and per input shape.** A model with a larger nominal context window doesn't automatically have a higher cliff — CSV-heavy structured input degrades earlier than prose. Newton 2.5.1, Gemma, Claude, GPT-4o will all have different cliffs.

## MapReduce / Hierarchical Reduce

Once you know the cliff, every reduce stage in a pipeline answers the same question:

> Does the concatenation of N inputs at this stage fit under the cliff?

- **Yes** → single-pass reduce works.
- **No** → insert a hierarchical pre-fold (group N inputs into packs of K, reduce each pack into a super-input, then fold super-inputs).

### Worked shape — folding 1 GB of CSV into 9 narratives

From the [ghost-iot worked example](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot), §8 of the README — observed across one end-to-end run on the C 2.4 model (1 GB of WiFi flow CSV, 5.8 M flows across 24 devices × 24 hours, multi-tenant topology of 3 homes × 2 humans × 4 devices/human):

```
chunks (≤16 KB each, every flow preserved) → bucket reduce
  ├─ obvious: fold all chunks of a (device, hour) bucket → ✗ overflows for ~all buckets
  └─ hierarchical:
       Stage A:   chunks → partials       (groups of K=3 chunks per partial)
       Stage A₂:  partials → super-partials (groups of K=4 partials per super) — needed because Stage A's
                                              busy buckets had up to 44 partials = ~44 KB concat
       Stage B:   super-partials → 1 narrative per (device, hour) ✓

hourly narratives → device-day reduce
  ├─ obvious: fold 24 hours per device → ✗ overflows for the busiest device (17.8 KB)
  └─ hierarchical:
       Stage A:   hours → super-hours (groups of K=4 hours per super-hour)
       Stage B:   super-hours → 1 narrative per device ✓

device-day narratives → user-day / house-day reduce
  └─ single-pass:  3 / 8 inputs per group, fits → ✓
```

Six reduce passes total, folding 23,249 chunks down to 9 final narratives. The hierarchical layers exist because the cliff is ~16 KB; **a model with a higher cliff lets you collapse layers**, not change the pattern.

### The shape, restated

```python
# Pseudo-code for one hierarchical reduce pass.
# Inputs: list of N narratives (from a prior stage)
# Output: list of ceil(N/K) super-narratives

# Stage A: pack the inputs into groups of K and submit as a batch job
groups = [inputs[i:i+K] for i in range(0, len(inputs), K)]
for group_index, group in enumerate(groups):
    record = {
        "system":      SYSTEM_PROMPT,
        "instruction": "Summarize the following K narratives into one super-summary.",
        "prompt":      f"super-group {group_index}, K={len(group)}",
        "inputs":      [{"type": "text", "format": "plain",
                         "data":  "\n\n---\n\n".join(group)}],
    }
    emit_jsonl(record)

# Validate the cliff: max(len(record["inputs"][0]["data"])) must be < CLIFF_BYTES
# If it isn't, decrease K and re-pack.
```

Pick `K` so the worst-case packed size sits comfortably below the cliff. The ghost-iot pipeline settled at `K=3` for chunks→partials and `K=4` for the higher layers.

## N-way Batch Split (concurrency, optional)

Most orgs have batch concurrency ≥ 1. If yours has N > 1 slots, every stage can run N batch jobs in parallel, cutting wall-clock by ~N× **with no quality cost**. The recipe:

1. Build the per-stage JSONL as a single ordered file (`combined.jsonl`). **Do not shuffle.**
2. Split it positionally into N contiguous halves: `combined_p1.jsonl`, ..., `combined_pN.jsonl`.
3. Upload all N (see [`newton-batch-upload`](../newton-batch-upload/SKILL.md) if any half exceeds 255 MB).
4. Submit N batch jobs in parallel, one per half.
5. After all complete, concatenate the prediction files **in the same positional order** and renumber `line_index` sequentially (see next section).

Skip the split if N=1 (single-slot org). Use whatever N your org allows — the pattern is the same.

**Why positional, not random:** the join from predictions back to the source manifest depends on either index or content. Random shuffle destroys index alignment and complicates content-key reconstruction (see §"Two silent join bugs" below).

## Two silent join bugs when chaining reduce stages

Both bugs produce zero errors and zero warnings — `COMPLETED` status, plausible-looking but wrong narratives. **Always inspect at least one final narrative against its manifest scope** before trusting the pipeline.

### Bug 1 — Position-based join after shuffle

If you `shuf` the master JSONL before splitting, each output prediction's `line_index` is relative to the *shuffled* input, not the original manifest. A position-based join `predictions[i] → manifest[i]` quietly maps every prediction to the wrong scope (wrong device, wrong hour, wrong home).

**Symptom:** cross-scope leakage in final narratives — Home A's narrative cites devices from Home B.

**Fix — content-key join.** For every reduce stage that needs to know what each prediction is *about*, encode the identifying tuple inside the prompt text and extract it back from the prediction record:

```python
# At prep time, embed the key in the prompt:
prompt = (f"Scope: device={device_id}, hour={hour}, n_flows={n_flows}, "
          f"ts_lo={ts_lo}, ts_hi={ts_hi}, n_bytes={n_bytes}")

# After predictions arrive, extract the key from the original prompt text
# and join to the manifest by (device_id, hour, n_flows, ts_lo, ts_hi, n_bytes).
# This is bulletproof under shuffle, split, retry, or any reordering.
```

A content key is essentially free to compute and survives every reordering. Use it.

### Bug 2 — `line_index` collisions after concat-of-splits

Each split batch job emits predictions with `line_index` relative to *its own* input file (`0..N/2-1` for a half-split). Cat'ing the halves produces **duplicate** `line_index` values.

If any downstream prep script loads predictions into a `dict[line_index] → prediction` and looks up by `line_index`, dict insertion order means the last-loaded value wins:

- Manifest entries with `line_index 0..N/2-1` → silently get the *second half's* predictions (wrong scope).
- Manifest entries with `line_index N/2..N-1` → silently get **no prediction at all** (empty string).

**Symptom:** the next reduce stage describes "no activity" for scopes that should be busy.

**Fix — renumber `line_index` sequentially to `0..N-1` after every concat**, before passing predictions to the next stage's prep script. Five lines of Python:

```python
import json, sys
for new_idx, line in enumerate(sys.stdin):
    rec = json.loads(line)
    rec["line_index"] = new_idx
    print(json.dumps(rec))
```

Run this after every `cat split_*.jsonl`. Skipping it silently breaks the next reduce stage.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v0.5/batch/jobs` | POST | Create a batch job |
| `/v0.5/batch/jobs` | GET | List all jobs |
| `/v0.5/batch/jobs/{job_id}` | GET | Get job status |
| `/v0.5/batch/jobs/{job_id}/events` | GET | Get job events / logs |
| `/v0.5/batch/jobs/{job_id}/outputs` | GET | List output artifacts (paginated presigned URLs) |
| `/v0.5/batch/jobs/{job_id}/cancel` | POST | Cancel a `PENDING` / `RUNNING` job; 409 if already terminal. See [Cancelling a Job](#cancelling-a-job). Docs: [archetypeai/docs](https://docs.archetypeai.app/api-reference/batch/jobs/cancel-job). |

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| Every prediction is `null` with `"error": "parse error"` | Input is not JSONL (raw CSV, blank lines, trailing comma) | Re-emit the file with one `json.dumps(record)` per line and no blank lines |
| `COMPLETED`, predictions are 1–3-char fragments like `'00'`, `'|153|'`, `''` | `inputs[0].data` is above the model's quality cliff (~4K tokens / ~16 KB for Newton C 2.4) | Run a cliff sweep, then insert a hierarchical pre-fold so every record stays under the cliff |
| `COMPLETED`, final narratives reference the wrong scope | Position-based join after shuffle/split (Bug 1) | Switch to content-key join (encode identifying tuple in `prompt`) |
| `COMPLETED`, half the manifest reports "no activity" | `line_index` collisions after concat-of-splits (Bug 2) | Renumber `line_index` sequentially after every concat |
| `Failed to resolve file inputs` | File not uploaded | Upload first via [`newton-batch-upload`](../newton-batch-upload/SKILL.md) |
| `Pipeline 'X' has no active versions` | Pipeline key wrong for this environment | Verify `pipeline_key` is `activity-detection` and that your org has it deployed |
| `validation errors for InferenceRecord — inputs.0.format Field required` | An `inputs[]` item used `file_id` instead of inline bytes | Use `{type, format: "base64", data: <b64-bytes>}` — `inputs[]` does not accept file_id references |
| Upload returns `Unsupported file type: application/x-ndjson` | JSONL submitted via simple `POST /files` multipart-form | Switch to the 3-step presigned flow (`/files/uploads/initiate` → S3 `PUT` → `/complete`). See [`newton-batch-upload`](../newton-batch-upload/SKILL.md) |
| `WARN inference.truncation … 1 items exceed token_budget=16384` on video records | Per-record token budget exceeded (e.g. 32 video frames + procedure prompt) | Often coped with — read predictions before reacting. To eliminate: reduce `max_video_frames`, shorten the prompt, or split the video into smaller chunks |
| `POST /cancel` returns 409 | Job is already in a terminal state (`COMPLETED` / `FAILED` / `CANCELLED`) | Treat as success — nothing left to stop. See [Cancelling a Job](#cancelling-a-job). |

## Best Practices

- **Run a cliff sweep first.** 30 minutes of testing 5 records × 4 size caps pays for itself many times over — the whole architecture follows from that single number.
- **Default to hierarchical reduce when N inputs × per-input size approaches the cliff.** Same shape at every level (group K → super → fold).
- **Content-key joins, always.** Position-based joins look fine until the first shuffle or retry and then break silently. Encode `(scope_id, count, time_range, ...)` in `prompt` and parse it back.
- **Renumber `line_index` after every concat.** Five-line Python loop. Skipping it silently breaks the next stage.
- **Inspect at least one final narrative against its manifest scope.** Events log is not enough — both silent failure modes (cliff, joins) produce `COMPLETED` with bad content.
- **Use lower `temperature` (0.5–0.6) for intermediate fold stages.** Variance compounds across reduce passes; tighter sampling at the bottom keeps the final narrative coherent.
- **Split positionally for concurrency, never randomly.** Random shuffle complicates content-key reconstruction and gains nothing.
- **The hierarchy depth follows from the cliff, not the nominal context window.** A model with a higher cliff lets you collapse layers; same pattern, less depth.

## Example Code

### Text MapReduce — `archetypeai-batch-examples-ghost-iot`

[**archetypeai-batch-examples-ghost-iot**](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot) — full public end-to-end pipeline for the `activity-detection` batch flow: data prep → upload → batch jobs → outputs → join → view, in Python, shell, and curl. Covers both the **simple single-record pattern** (§2–6 of the README — one prompt summarizing a whole filtered day) and the **multi-home MapReduce pattern at GB scale** (§8 — 23 K chunks → six reduce stages → 9 daily narratives). Has the cliff-sweep methodology (§8.2 / §10.6), content-key join (§10.7), `line_index` renumbering (§10.8), and observed wall-clock across all stages.

The orchestration code is pipeline-agnostic — `upload_multipart.py`, the polling loop, and the paginated outputs download transfer unchanged to any other `pipeline_key`. Only the prep scripts (which build the per-stage JSONL) are activity-detection-specific.

See also [`EXPERIENCE.md`](https://github.com/archetypeai/archetypeai-batch-examples-ghost-iot/blob/main/EXPERIENCE.md) in that repo — long-form writeup of *why* the design ended up needing six MapReduce stages instead of one fold.

### Video + image batch — `manual-gen-amat-jos`

[**manual-gen-amat-jos**](https://github.com/archetypeai/manual-gen-amat-jos) — production-style app that runs the `activity-detection` pipeline with `type: "video"` inputs to extract procedural step-by-step manuals from instructional MP4s. End-to-end:

- Chunks a source MP4 into 90 s segments (ffmpeg)
- Builds an N-record JSONL with each segment base64-embedded in `inputs[0].data`
- Uploads the JSONL via the multipart-presigned flow (required for `application/x-ndjson`)
- Submits one batch job with `batch_size: 1`, `max_video_frames: 32`
- Polls, downloads per-segment predictions, parses `[MM:SS]`-anchored steps, dedups, exports to PDF

Has standalone probes (`probe_batch_video.py`, `concurrency_probe.py`) that document the schema corrections this SKILL incorporates — useful as a reference for any new app integrating the batch flow with vision inputs. The repo's [`ARCHETYPE_INTEGRATION.md`](https://github.com/archetypeai/manual-gen-amat-jos/blob/main/ARCHETYPE_INTEGRATION.md) §4 walks the platform error messages → schema corrections journey.
