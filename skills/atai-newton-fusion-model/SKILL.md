---
name: atai-newton-fusion-model
description: >
  Call Newton's C 2.6 fusion model (`Newton::c2_6_8b_fp8_260424d7a55d5e`)
  on the prod `/query` endpoint with text, image, and video inputs in a
  single stateless request — no session, no batch job, no SSE plumbing.
  Use this skill when the user wants per-call multimodal reasoning
  (describe an image, summarize a clip, classify a state from sensor +
  visual context) without managing session lifecycle, or when they need
  the Newton C checkpoint that reasons over video frames via `/query`.
  Covers the request shape per modality, the two image-attachment paths
  (file_ids vs base64), multi-image mode, both video paths (`.mp4` +
  `max_frames`, and client-sampled frames + `query_metadata`), generation
  parameters, JSON-output prompting, latency budgets, and the C 2.6
  identifier gotcha.
  Do NOT use for streaming / session-based activity monitoring, large
  multi-file batch jobs, or time-series embedding (KNN / anomaly)
  classification.
---

# Newton C 2.6 Fusion Model — Multimodal `/query` in One Hop

Single stateless POST to `/query` for text, image, or video reasoning against the C 2.6 fusion checkpoint. No Lens session, no batch pipeline, no SSE. One call → one response.

## When to Apply

- User wants to describe / classify / extract structure from an image with one API call
- User wants the same for a short video in a single `/query` call
- User wants stateless multimodal reasoning (no per-session warmup, no orphan cleanup)
- User wants JSON output from a multimodal input (the prompt is the schema)
- User is building a serverless / per-request handler where a streaming Lens session would be the wrong shape

**Do not use this skill when**:
- The workload is live, streaming video → text from a long-running camera feed — a session-based activity monitor is the right shape, not stateless `/query`
- The workload is a large multi-file batch job
- The task is time-series classification via embeddings (KNN / anomaly detection)

## The Model Identifier

```
Newton::c2_6_8b_fp8_260424d7a55d5e
```

The full `Newton::` prefix **and** the `_fp8_` segment are required. Variants that omit either are rejected with `400 invalid_model_version`:

```text
c2_6_8b_fp8_260424d7a55d5e            ❌ missing Newton:: prefix
Newton::c2_6_8b_260424d7a55d5e        ❌ missing _fp8_
c2_6_8b_260424d7a55d5e                ❌ both missing
```

This was probed end-to-end against `https://api.u1.archetypeai.app/v0.5/query` on prod.

## Endpoint

```
POST {ATAI_API_ENDPOINT}/v0.5/query
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

**Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required** — there is no default endpoint, so a wrong-deployment mistake fails loudly at startup instead of silently at query time. Prod is `https://api.u1.archetypeai.app/v0.5`. The same endpoint serves all three modalities below — what changes is the request body.

## Wire Shapes

### Text + JSON

```json
{
  "query": "Classify the operational state. Respond with ONLY the JSON object.",
  "instruction_prompt": "You output {\"state\": \"...\", \"confidence\": <0..1>}.",
  "file_ids": [],
  "model": "Newton::c2_6_8b_fp8_260424d7a55d5e",
  "max_new_tokens": 300
}
```

- Put the system turn in `instruction_prompt`. That is the only field C 2.6 honors — a directive sent in the legacy `system_prompt` field alone is silently ignored (verified against `Newton::c2_6_8b_fp8_260424d7a55d5e`: the same directive is obeyed in `instruction_prompt` and dropped in `system_prompt`), so omit `system_prompt` entirely.
- JSON output: put the schema in `instruction_prompt`, the input in `query`, and **explicitly tell the model not to wrap in markdown fences**. The model otherwise often returns <code>```json ... ```</code>.

For reasoning over an attached text file: upload it as **`text/plain`** (a `.txt` file) via `/v0.5/files`, then put the **filename** in `file_ids`. The model then sees the file contents in its prompt context. **Critical:** this only works for `text/plain`. A file uploaded as `text/csv` (a `.csv`) is accepted and stored byte-identically, but its contents never reach the model on `/query` — it silently confabulates from priors (see the gotcha below and [`csv_vs_txt_proof.py`](references/csv_vs_txt_proof.py)). `application/json` uploads are rejected outright. So put tabular data in a `.txt`, or inline it in `query`.

See [`text_query.py`](references/text_query.py) for runnable demos of all three patterns. Its examples are framed as smart-home WiFi flow analysis; the attached flow log in example 3 is real data from the [GHOST-IoT dataset](https://github.com/gspathoulas/ghost-iot-dataset) (Anagnostopoulos et al., *Sensors* 2020, [DOI:10.3390/s20226600](https://doi.org/10.3390/s20226600)) — see the script's docstring for full attribution.

### Image

Two attachment paths, both reach the same fusion model — pick by use case — plus a flag for sending several images at once:

**(a) Upload + reference by filename in `file_ids`** — best when the same image is reused across queries (one upload, many `/query` calls):

```bash
curl -X POST $ATAI_API_ENDPOINT/v0.5/files \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@dashboard.png;type=image/png"
```

```json
{
  "query": "Describe this image in three bullets.",
  "instruction_prompt": "...",
  "file_ids": ["dashboard.png"],
  "model": "Newton::c2_6_8b_fp8_260424d7a55d5e",
  "max_new_tokens": 300
}
```

**(b) Inline base64 in a `data.base64_img` event** — best for one-shot calls where the upload roundtrip is wasted:

```json
{
  "query": "What is in this image? One sentence.",
  "instruction_prompt": "...",
  "file_ids": [],
  "model": "Newton::c2_6_8b_fp8_260424d7a55d5e",
  "max_new_tokens": 200,
  "events": [
    {
      "type": "data.base64_img",
      "event_data": {
        "contents": "<base64 string>",
        "mime_type": "image/png"
      }
    }
  ]
}
```

**(c) Multiple images — set `multi_image: true`** for *multi-image mode*: each attachment is an **independent** image (before/after, multi-view, "is the same person in these images?"), not frames of a video. Platform limit: **16 images max** per request. Without `multi_image: true`, an image list means "frames of one video" and requires `query_metadata` (see the Video section); lacking both, the request fails with `400 query_failed`:

```json
{
  "query": "Image 1 is the bench BEFORE the task; image 2 is DURING. What changed?",
  "instruction_prompt": "...",
  "file_ids": ["assembly_before.png", "assembly_after.png"],
  "model": "Newton::c2_6_8b_fp8_260424d7a55d5e",
  "max_new_tokens": 400,
  "multi_image": true
}
```

See [`image_query.py`](references/image_query.py) for runnable demos of all three paths.

### Video

C 2.6 reasons over video on `/query` two ways. This distinguishes it from the C 2.4 / 2.5 text checkpoints, which accept an `.mp4` but ignore the frames and reply *"I can't see videos."*

**(a) `.mp4` + `max_frames`** — upload the clip, and GPQ decodes and uniformly samples it server-side before the model sees the frames. No client-side video tooling needed:

```bash
curl -X POST $ATAI_API_ENDPOINT/v0.5/files \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@clip.mp4;type=video/mp4"
```

```json
{
  "query": "Evaluate the assembly steps shown in the attached video.",
  "instruction_prompt": "<task / inspection prompt>",
  "file_ids": ["clip.mp4"],
  "model": "Newton::c2_6_8b_fp8_260424d7a55d5e",
  "max_new_tokens": 500,
  "max_frames": 32
}
```

**(b) Client-sampled frames + `query_metadata`** — send the frames yourself (one `data.base64_img` event each, or image `file_ids`), set `multi_image: false`, and include a `query_metadata` block with the video-metadata triple. **The `query_metadata` block is what makes this work** — without it the request fails with `400 query_failed`:

```json
{
  "query": "This is a short video. Describe the sequence of actions in order.",
  "instruction_prompt": "...",
  "model": "Newton::c2_6_8b_fp8_260424d7a55d5e",
  "multi_image": false,
  "max_new_tokens": 150,
  "events": [
    {"type": "data.base64_img", "event_data": {"contents": "<frame_0 base64>"}},
    {"type": "data.base64_img", "event_data": {"contents": "<frame_1 base64>"}}
  ],
  "query_metadata": {"raw_fps": 1.0, "frames_indices": [0, 1], "total_num_frames": 2}
}
```

Both shapes verified live (the `file_ids` variant of (b) works too). Notes:

- **`max_frames`** — frames GPQ samples uniformly from the video before feeding the model. Default 32, **platform max 64**. Increase for longer / higher-detail clips; decrease to cap latency. **Values above 64 fail silently**: the request returns 200, but the video is dropped and the model confabulates a scene from priors (verified: `max_frames: 80` on the assembly clip described "a person holding a smartphone"). Stay at ≤ 64.
- **Timeout** — set the client request timeout to **at least 600s**. Video latency scales with `max_frames` and clip length; default 30–60s timeouts will trip on non-trivial clips.
- **`.mp4`-direct on C 2.6 is a recent platform change.** Earlier the `.mp4` path silently dropped the frames for *every* model (a known GPQ bug), so older notes say "video doesn't work on `/query`." On C 2.6 today it does, via `max_frames`.
- **`multi_image` selects between video and multi-image mode.** `multi_image: false` (with `query_metadata`) = the attached images are frames of **one video**. `multi_image: true` = *multi-image mode* — the images are **independent** (before/after, multi-view), and `query_metadata` is not needed.

See [`video_query.py`](references/video_query.py) for runnable demos of both paths (mp4 direct, a `max_frames` tradeoff, and a frame-list demo) using a worker-assembly PASS/FAIL inspection prompt.

## Generation Parameters

All C-model generations via `/query` accept sampling controls in the request body (verified accepted + honored on prod with C 2.6):

| Param | Default | Notes |
|-------|---------|-------|
| `max_new_tokens` | 256 | Hard ceiling on generated **tokens** (~4 chars each) — stops mid-sentence at the cap |
| `temperature` | — | 0.0 = greedy/deterministic-ish; 0.7–1.0 for creative output. Capped at 2.0 |
| `do_sample` | `true` | Master switch: `false` forces greedy decoding (temperature → 0, `top_p`/`top_k` stripped) — use for reproducibility |
| `repetition_penalty` | 1.0 | Multiplicative logit penalty on already-emitted tokens; try 1.1–1.3 if the model loops on a phrase or repeats numbers |
| `presence_penalty` | 0.0 | Additive penalty on any token that has appeared at least once (frequency-independent). Range ≈ −2..+2. **C 2.6 only** |
| `top_p` | 0.8 | Nucleus sampling — keep the smallest token set with cumulative probability ≥ `top_p` |
| `top_k` | — | Hard cap on candidate tokens per step (applied before `top_p`); 0 disables |

For "extract X"-style tasks, `do_sample: false` (or `temperature: 0.0`) plus a tight `max_new_tokens` is the right starting point.

**Reproducibility caveat (verified):** the sampling knobs are honored — on the same prompt, 5 runs gave 2 distinct outputs greedy (`do_sample: false`), 3 distinct at defaults, 5 distinct at `temperature: 1.9` — but greedy is **not bit-deterministic** on this deployment (fp8 numerics / batching sit outside the decoder), contrary to the platform note that `do_sample: false` output "is deterministic given the prompt". On borderline judgments the residual wobble is enough to flip the answer: the same greedy assembly-inspection body returned all-PASS on one run and two FAILs on another. If a verdict matters, aggregate over a few calls rather than trusting one.

## Latency Budgets

Observed against prod with the sample assets in `references/sample_assets/`:

| Modality | Body | p50 | Notes |
|----------|------|----:|-------|
| Text only | `{"query": "...", "file_ids": []}` | **~500 ms** | Confirmed during model-identifier probing |
| Text + JSON output | instruction prompt has schema | ~600 ms | Same shape, slightly longer with verbose schema |
| Image via `file_ids` | one 1.9 MB PNG | **~1.3–1.9 s** | observed (wind-turbines.png) |
| Image via inline base64 | same | **~0.8–2.5 s** | base64 inflates the request; scales with image size |
| Two images, `multi_image: true` | 2 × ~400 KB PNG | **~2.9 s** | before/after comparison (assembly frames) |
| Video, mp4 + `max_frames=32` | ~31 s assembly clip | **~2–6 s** | server-side decode/sample |
| Video, 2 frames + `query_metadata` | 2 × ~400 KB PNG | **~1.9 s** | client-sampled frame-list path |

Numbers are smoke-test order-of-magnitude. Production data may be 5–10× heavier and proportionally slower.

## Response Shape

```json
{
  "query_id": "...",
  "status": "completed",
  "inference_time_sec": 0.42,
  "query_response_time_sec": 0.49,
  "response": {
    "response": ["<the model output as a string>"]
  }
}
```

Walk `payload["response"]["response"][0]` as the canonical extraction path. The reference scripts' `extract_text()` helper handles every shape variant we've seen.

## Common Pitfalls

- **Wrong model identifier.** `c2_6_8b_fp8_260424d7a55d5e` without `Newton::` returns `400 invalid_model_version`. So does the variant without `_fp8_`. Always use the full `Newton::c2_6_8b_fp8_260424d7a55d5e`.
- **JSON wrapped in markdown fences.** The model sometimes returns <code>```json ... ```</code> even when explicitly told not to. Either parse defensively (`text.strip().removeprefix("```json").removesuffix("```")`) or restate the no-fences rule in the user query as well as `instruction_prompt`.
- **Video timeout.** Default `requests.post` timeout of 30–60s will fail on anything but trivial clips. Set timeout ≥ 600s.
- **Multiple images without `multi_image: true` need the `query_metadata` triple.** A list of images defaults to "frames of one video" — but that path requires `query_metadata: {"raw_fps": ..., "frames_indices": [...], "total_num_frames": N}` in the body; without it the request fails with `400 query_failed` (the error doesn't hint at the missing field). For *independent* images (before/after, multi-view), set `multi_image: true` instead — max 16 images, no `query_metadata` needed.
- **`max_frames` above 64 fails silently.** The platform max is 64. `max_frames: 80` still returns 200, but the video never reaches the model — it confabulates a scene from priors (same failure mode as the CSV gotcha below). Keep `max_frames` ≤ 64.
- **"Video doesn't work on `/query`" is stale.** Older notes say C 2.6 (and 2.4/2.5) can't see video. That was a GPQ bug that silently dropped `.mp4` frames before any model. C 2.6 reads video now via `.mp4` + `max_frames` — verified. The 2.4/2.5 *text* checkpoints still ignore frames.
- **`file_ids` takes the filename, not the `fil_...` uid.** On upload the API returns both `{"file_id": "<filename>", "file_uid": "fil_..."}`. `/query` resolves `file_ids` by the **filename** (the `file_id` value, e.g. `"dashboard.png"`), not the `fil_...` `file_uid`. Passing the `file_uid` fails with a **misleading** `400 unsupported_file_type` ("Found unsupported file type with file: fil_…") — it looks like a content-type problem but is really the wrong identifier. Verified: `file_ids=["wind-turbines.png"]` → 200; `file_ids=["fil_…"]` → 400.
- **Text attachments must be `text/plain`, not `text/csv`.** A `.csv` upload (`text/csv`) is accepted, stored byte-for-byte, and returns `status: completed` — but its contents **never reach the model** on `/query`; the model silently confabulates from priors (e.g. answers "TCP"/"port 80" for an all-SSH file). The **identical bytes** uploaded as `.txt` (`text/plain`) are read faithfully. There is **no warning** — the failure is silent, so you can only catch it by validating the answer. Upload tabular/text data as `.txt`, or inline it in `query`. Verify yourself with [`csv_vs_txt_proof.py`](references/csv_vs_txt_proof.py) (`.txt` reads 5/5, `.csv` 0/5 on identical content). `application/json` uploads are rejected with `400 invalid_file_type`.
- **Mixed-modality calls.** You can attach both an image and a video file_id, but the fusion model's attention is finite. Stick to one modality per call for predictable results.

## Local Setup

```bash
# In a virtualenv (system pythons are often PEP 668 "externally managed"):
python3 -m venv .venv && source .venv/bin/activate
pip install -r skills/atai-newton-fusion-model/references/requirements.txt

# Drop a .env at the repo root (or anywhere up the tree from the run dir).
# BOTH variables are required — there is no default endpoint:
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://api.u1.archetypeai.app/v0.5
EOF

python skills/atai-newton-fusion-model/references/text_query.py
python skills/atai-newton-fusion-model/references/image_query.py
python skills/atai-newton-fusion-model/references/video_query.py
```

The scripts are built on the [official Archetype AI python client](https://github.com/archetypeai/python-client) (`pip install archetypeai`): uploads go through `client.files.local.upload(...)`, base64 encoding through `archetypeai.utils.base64_encode(...)` (utf-8), and each script creates one client and passes it into the helpers. They auto-load `.env` if `python-dotenv` is installed — `find_dotenv()` walks up from cwd so the file works at the repo root, the project root, or inside `references/`. Without `python-dotenv`, the scripts fall back to plain environment variables (`export ATAI_API_KEY=... ATAI_API_ENDPOINT=...`).

## File Layout

```
skills/atai-newton-fusion-model/
├── SKILL.md                  ← this file
├── references/
│   ├── _common.py            ← shared helpers on the official archetypeai client
│   ├── text_query.py         ← 3 text patterns (plain, JSON output, .txt flow-log attachment)
│   ├── image_query.py        ← 4 image patterns (file_ids, base64, JSON extraction, multi-image)
│   ├── video_query.py        ← 3 video patterns (mp4 direct, max_frames tradeoff, frame list + query_metadata)
│   ├── csv_vs_txt_proof.py   ← diagnostic: proves text/csv attachments aren't read, text/plain are
│   ├── requirements.txt      ← runtime deps (pip install -r requirements.txt)
│   ├── .env.example          ← copy to .env and fill in
│   └── sample_assets/
│       ├── wind-turbines.png ← AI-generated photo (default for image_query.py)
│       ├── assembly_before.png / assembly_after.png ← frames from the clip below (multi-image demo)
│       ├── 1_pass_2_pass_3_pass_B.mp4 ← assembly-inspection clip (default for video_query.py)
│       └── README.md         ← provenance of the samples
└── tests/
    └── test_references.py    ← network-free unit tests (python -m unittest)
```
