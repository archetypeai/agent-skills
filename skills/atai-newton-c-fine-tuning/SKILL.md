---
name: atai-newton-c-fine-tuning
description: >
  LoRA-fine-tune Archetype AI's Newton C model (`newton/c:2.5.1-8b-base`,
  the only Newton model fine-tunable on production) on labelled video
  clips with the `/fine-tuning` API, serve the checkpoint through a JOS
  batch job (`activity-detection`), and measure the fine-tuned model
  against the base model on the same held-out clips. Use when the user
  wants to adapt Newton C to their own video classes or vocabulary, check
  whether fine-tuning helps, or debug a fine-tune whose "after" looks
  identical to its "before". Covers which models can be fine-tuned, the
  two data formats (chat `messages` for training, `InferenceRecord` for
  inference), multipart JSONL upload, the create-job body, job statuses
  and checkpoints, attaching the checkpoint on the `worker.checkpoint`
  port (`config.checkpoint_id` is silently ignored), cursor pagination,
  closed-set and LLM-judged scoring, and what to expect (~3 h, fixed
  1,024 steps). Works with a developer key. Do NOT use for one-off
  multimodal questions (`atai-newton-fusion-inference-api`,
  `atai-newton-fusion-model`), for fine-tuning C 2.6 or Fusion (not
  possible today), or for time-series embeddings (`atai-newton-omega-*`).
---

# Fine-Tuning Newton C

Baseline the base model, fine-tune it with LoRA, serve the checkpoint, and compare — all over the API, no GPU or weights. Worked example with full results: [fine-tuning-example-newton-c](https://github.com/archetypeai/fine-tuning-example-newton-c) (UCF101, 10 classes).

## When to Apply

- The user has labelled video clips (or frames) and wants Newton C to name their classes, in their vocabulary
- The base model is close but confuses fine-grained classes (in the example: `BasketballDunk` called `Basketball`)
- The user wants a before/after number, not just a trained checkpoint
- A fine-tuned run scores exactly like the base model and the user wants to know why

**Do not use this skill when**:
- The user wants to fine-tune **C 2.6 or Fusion** — they can't be fine-tuned on production today (see [Which Model](#which-model)); serve them with [`atai-newton-fusion-inference-api`](../atai-newton-fusion-inference-api/SKILL.md)
- The task is per-call multimodal Q&A with no training
- The data is numeric time-series — use the Omega skills

## Which Model

As of 2026-11-23 on production (platform 2.0.1), checked against the batch pipeline registry, `/models` and a live job:

| Model (`/models` tag) | Fine-tunable | Servable in `activity-detection` |
|---|---|---|
| **C 2.5.1-8b** (`newton-c-2-5-1-8b`) | ✅ the only option | ✅ |
| C 2.5.1-2b (`newton-c-2-5-1-2b`) | ⚠️ a pipeline preset exists, but the API can't select it | ❌ |
| C 2.6-8b / 2.6-30b-a3b | ❌ | ✅ |
| Fusion 1.0 | ❌ no fine-tuning pipeline | via the Inference API |

The only training pipelines are `fine-tuning-newton` (C 2.5.1) and the Omega ones. The create body takes `model: "newton"` and has **no variant selector**; a `base_model_tag` in the body is ignored. Run the baseline on the same base (`newton/c:2.5.1-8b-base`) so the comparison is fair.

**Access:** a **developer** key is enough for every call here (upload, create, checkpoints, batch jobs) — verified on production, 2026-11-23. The free trial (`try.archetypeai.app`) has no fine-tuning.

## Two Surfaces

| Step | API | Endpoint |
|---|---|---|
| Upload | Files | `POST /files/uploads/initiate` → `PUT` presigned parts → `POST /files/uploads/{id}/complete` |
| Fine-tune | Fine-tuning (OpenAI-style; the Console uses it) | `POST /fine-tuning/jobs` |
| Inference, base **and** fine-tuned | JOS batch | `POST /batch/jobs`, `pipeline_key: "activity-detection"` |

All paths are under `{root}/api` (`https://platform.u1.archetypeai.app/api` on production). The 1.x hosts and `/v0.5`, `/v0.6` prefixes are gone. Send `X-Atai-Client: agent_skills`.

## The Run

```bash
cd skills/atai-newton-c-fine-tuning/references
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://platform.u1.archetypeai.app/api
EOF

python3 build_dataset.py --src clips --out dataset          # clips/{train,val,test}/<Class>/*.mp4
python3 run_fine_tune.py infer --baseline --mode closed      # before (~5–10 min)
python3 run_fine_tune.py infer --baseline --mode open
python3 run_fine_tune.py train                               # ~3 h
python3 run_fine_tune.py infer --mode closed                 # after
python3 run_fine_tune.py infer --mode open
python3 run_fine_tune.py score                               # closed-set, per class
```

**Baseline first.** It takes minutes and tells you whether there is anything to fix. If the base model is already near 100% closed-set, a fine-tune has little room to show.

Every id is written to `ft-state.json` as soon as it exists; `train --job <ftj_…>` and `infer --job <job_…>` re-attach after a disconnect. Stdlib only, plus `ffmpeg`/`ffprobe` for frames.

## Data

Two formats, one per API. Both carry the same N evenly spaced JPEG frames per clip (8 frames, longest side 448 px in the example).

**Training** (`training_files`, `validation_files`) — one OpenAI chat record per clip, the label as the assistant turn:

```json
{"messages": [
  {"role": "system", "content": "You are a video action-recognition assistant."},
  {"role": "user", "content": [
    {"type": "text", "text": "What action is the person performing in this video? Answer with the action name."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]},
  {"role": "assistant", "content": "BasketballDunk"}
]}
```

**Inference** (`worker.data`) — one `InferenceRecord` per clip:

```json
{"system": "...", "prompt": "... Answer with exactly one of these labels: A, B, C.",
 "inputs": [{"type": "image", "format": "base64", "data": "..."}]}
```

Output rows come back as `{"line_index": N, "prediction": "..."}`; sort by `line_index` to line them up with your labels.

**Splits.** `train/` trains, `val/` goes in `validation_files` and picks the checkpoint (lowest `eval_loss`), `test/` is scored once, for both models. Never put test clips in `val/`.

**The assistant turn is what the model learns to say.** Terse class names as targets make terse answers — good for closed-set, and it can cost open-ended credit (see [What to Expect](#what-to-expect)). If you want descriptive open-ended answers, write descriptive targets.

### Upload

- **JSONL must use the multipart flow.** The simple `POST /files` rejects `application/x-ndjson`. Initiate with `{filename, file_type: "application/x-ndjson", num_bytes}`, `PUT` each part to its presigned URL (no auth header), complete with `{parts: [{part_number, part_token: <ETag>}]}`.
- **`file_id` is the file name**, in an org-wide namespace, and initiating a name that exists returns **409**. Reusing a fixed name after rebuilding a file silently serves the old contents. Put a content hash in the name ([`upload_name()`](references/atai_ft.py)).
- Inline base64 frames grow fast: 300 clips × 8 frames at 448 px is ~47 MB.

## Fine-Tune Job

```json
POST /fine-tuning/jobs
{
  "model": "newton",
  "name": "newton-c-ucf101",
  "method": {"type": "lora", "lora": {"hyperparameters": {
    "n_epochs": "auto", "batch_size": "auto", "learning_rate_multiplier": "auto"}}},
  "training_files":   [{"type": "plain", "file_id": "<train>.jsonl", "format": "jsonl"}],
  "validation_files": [{"type": "plain", "file_id": "<val>.jsonl",   "format": "jsonl"}]
}
```

- **`name` and `validation_files` are required** — without them, `422 ... did not match any variant of untagged enum CreateFineTuningJobRequest`.
- **File ids aren't validated on create.** A typo fails later, inside the job. Check your ids before you create.
- **Creating is not idempotent.** A retried create is a second ~3 h job. Send it once; cancel strays with `POST /fine-tuning/jobs/{id}/cancel`.
- **Statuses:** `PENDING` → `ADMITTED` → `RUNNING` → `COMPLETED`. Success can also be reported as `SUCCEEDED`: accept both before reading checkpoints.
- **Steps are fixed at 1,024**, whatever the dataset size. `n_epochs` is accepted; there is no `max_steps`. On 300 clips that is ~56 epochs.
- **Checkpoints:** `GET /fine-tuning/jobs/{id}/checkpoints` → `[{id: "ckp_…", step, metrics: {eval_loss, epoch, …}}]`. Keep the lowest `eval_loss`. Both runs of the example produced a single checkpoint, at step 1024.
- **`fine_tuned_model` stays `null`.** There is no served model name; you serve the `ckp_…` id (next section).
- **Budget ~3 h.** Measured: 10,460 s (2 h 54 min) on production, 2026-11-23, 300 training clips. The Console shows live loss and steps.

## Serving the Checkpoint

Run the **base** variant and attach the checkpoint on the **`worker.checkpoint` input port**; the worker merges the LoRA adapter onto the base:

```json
POST /batch/jobs
{
  "name": "newton-c-finetuned",
  "pipeline_type": "batch",
  "pipeline_key": "activity-detection",
  "inputs": {
    "worker.data":       [{"kind": "file",       "file_id": "<test>.jsonl"}],
    "worker.checkpoint": [{"kind": "checkpoint", "checkpoint_id": "ckp_…"}]
  },
  "parameters": {"worker": {"parallelism": 1, "config": {
    "model_variant": "newton/c:2.5.1-8b-base",
    "generation": {"do_sample": false, "max_new_tokens": 64}
  }}}
}
```

The baseline is the same body **without** `worker.checkpoint`.

> **⚠️ `config.checkpoint_id` is accepted and silently ignored** — the job runs the base model and the before/after comes out identical. A checkpoint id as `model_variant` is rejected. **If your fine-tuned run matches the baseline clip for clip, check the port first.**

- `kind` is required on every input.
- Outputs: `GET /batch/jobs/{id}/outputs` → entries with `data.ref = "/api/files/fil_…/content"`. The ref is a root path: strip `/api` before joining it to an endpoint that already ends in `/api`, or you get `/api/api/…` and a 404.
- Success for a batch job is `COMPLETED`. Score only after it: a `RUNNING` job returns partial or empty outputs.
- ~5–10 min for 75 clips, closed or open.

## Pagination

Every list on platform 2.0 — jobs, events, checkpoints, batch outputs — is `{data, has_more, next_cursor, prev_cursor}`, **newest first**. Fetch the next (older) page with `after=<next_cursor>`. `offset` is ignored and there is no `total`. Reading only the first page truncates outputs and makes a polling loop reprint the same events. See [`list_all()`](references/atai_ft.py).

## Scoring

Run each model twice: **closed-set** (the prompt lists the labels) and **open-ended** (no labels). Closed-set is the headline; the gap to open-ended shows how much of the score is vocabulary rather than recognition.

**Closed-set: take the longest class name in the answer, then compare it to the label.** Plain substring matching is wrong when labels nest: "BasketballDunk" contains "Basketball", so a dunk answer would count as correct for a `Basketball` clip. In the example this inflated the baseline from 68 to 70 of 75. See [`predicted_class()`](references/run_fine_tune.py).

**Open-ended: grade with an LLM judge, and give it the label set.** Free-text answers ("applying eye makeup", "dunk") need semantic matching. Without the label set the judge passes vague answers: "playing basketball" for `BasketballDunk`, "applying makeup" for `ApplyEyeMakeup` when `ApplyLipstick` is also a class. Rubric that worked (Claude Opus 5.5, structured `{correct, reason}` output):

> A paraphrase or a more specific description is correct. A more general answer is correct only if it still points to the ground-truth class: if it would fit another class in the label set just as well, it is incorrect. A different action is incorrect.

Judge both models with the **same** judge model and rubric. The example's [`llm_judge.py`](https://github.com/archetypeai/fine-tuning-example-newton-c/blob/main/5_evaluate/llm_judge.py) is a working implementation.

## What to Expect

The example, 75 held-out clips, 10 classes, production 2026-11-23:

| | Base | Fine-tuned |
|---|---|---|
| Closed-set | 68/75 (90.7%) | 69/75 (92.0%) |
| Open-ended, LLM-judged | 65/75 (86.7%) | 64/75 (85.3%) |

- **The targeted confusion got fixed.** `BasketballDunk` went 1/6 → 2/6 closed-set and 4/6 → 6/6 open-ended. An earlier run (June 2026) showed the same fix (1/6 → 3/6, 4/6 → 6/6).
- **Answers became terse**, like the training targets: "Dunk", "Applying makeup". That cost `ApplyEyeMakeup` open-ended credit (4/6 → 2/6), because "applying makeup" also fits `ApplyLipstick`.
- **Errors the training data doesn't address stay.** `BalanceBeam` was answered "vault" open-ended on every clip, before and after.
- **On 75 clips, ±1–2 clips is noise.** Report the per-class change, not just the total. Fine-tunes aren't bit-reproducible (8B LoRA on GPUs), and the base model's open-ended answers drifted between platform versions.

## Common Pitfalls

- **Identical before/after.** The checkpoint went in `config.checkpoint_id`. Use the `worker.checkpoint` port.
- **`422` on create.** Missing `name` or `validation_files`.
- **Stale training data.** A re-uploaded file with the same name kept its old contents (409 → reused). Hash the name.
- **`POST /files` rejects the JSONL.** Use the multipart flow.
- **Script stops at `SUCCEEDED`.** It only accepted `COMPLETED`. Accept both.
- **Missing outputs or repeated events.** Only the first page was read. Follow `next_cursor`.
- **404 fetching outputs.** `/api` doubled when joining `data.ref`.
- **Wrong environment.** `ATAI_API_*` exported in the shell override `.env`. Unset them.
- **A vague open-ended answer counted as correct.** The judge didn't see the label set.
- **A nested label counted as correct.** Closed-set used plain substring matching.

## File Layout

```
skills/atai-newton-c-fine-tuning/
├── SKILL.md                 ← this file
├── references/
│   ├── atai_ft.py           ← stdlib client: config, multipart upload, pagination, job bodies, outputs
│   ├── build_dataset.py     ← clips → train/val `messages` JSONL + closed/open InferenceRecord JSONL (ffmpeg)
│   ├── run_fine_tune.py     ← infer (baseline / fine-tuned) · train · score · status · cancel
│   └── .env.example
└── tests/
    └── test_references.py   ← network-free unit tests
```
