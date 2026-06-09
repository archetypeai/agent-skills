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
  (file_ids vs base64), the two video paths (`.mp4` + `max_frames`, or
  client-sampled frames + `multi_image`), JSON-output prompting,
  latency budgets, and the C 2.6 identifier gotcha.
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

Default `ATAI_API_ENDPOINT` is `https://api.u1.archetypeai.app/v0.5` (prod). The same endpoint serves all three modalities below — what changes is the request body.

## Wire Shapes

### Text + JSON

```json
{
  "query": "Classify the operational state. Respond with ONLY the JSON object.",
  "instruction_prompt": "You output {\"state\": \"...\", \"confidence\": <0..1>}.",
  "file_ids": [],
  "model": "Newton::c2_6_8b_fp8_260424d7a55d5e",
  "max_new_tokens": 300,
  "sanitize": false
}
```

- Put the system turn in `instruction_prompt`. That is the only field C 2.6 honors — a directive sent in the legacy `system_prompt` field alone is silently ignored (verified against `Newton::c2_6_8b_fp8_260424d7a55d5e`: the same directive is obeyed in `instruction_prompt` and dropped in `system_prompt`), so omit `system_prompt` entirely.
- `sanitize: false` — leave the model's output as-is. `true` strips some content the C model is already trained to avoid; usually unnecessary here.
- JSON output: put the schema in `instruction_prompt`, the input in `query`, and **explicitly tell the model not to wrap in markdown fences**. The model otherwise often returns <code>```json ... ```</code>.

For reasoning over an attached text file: upload it as **`text/plain`** (a `.txt` file) via `/v0.5/files`, then put the **filename** in `file_ids`. The model then sees the file contents in its prompt context. **Critical:** this only works for `text/plain`. A file uploaded as `text/csv` (a `.csv`) is accepted and stored byte-identically, but its contents never reach the model on `/query` — it silently confabulates from priors (see the gotcha below and [`csv_vs_txt_proof.py`](references/csv_vs_txt_proof.py)). `application/json` uploads are rejected outright. So put tabular data in a `.txt`, or inline it in `query`.

See [`text_query.py`](references/text_query.py) for runnable demos of all three patterns. Its examples are framed as smart-home WiFi flow analysis; the attached flow log in example 3 is real data from the [GHOST-IoT dataset](https://github.com/gspathoulas/ghost-iot-dataset) (Anagnostopoulos et al., *Sensors* 2020, [DOI:10.3390/s20226600](https://doi.org/10.3390/s20226600)) — see the script's docstring for full attribution.

### Image

Two paths, both reach the same fusion model. Pick by use case:

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
  "max_new_tokens": 300,
  "sanitize": false
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
  "sanitize": false,
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

See [`image_query.py`](references/image_query.py) for runnable demos of both paths.

### Video

C 2.6 genuinely reasons over video on `/query` — it uses the sampled frames. This is the capability that distinguishes it from the C 2.4 / 2.5 text checkpoints, which accept an `.mp4` but ignore the frames and reply *"I can't see videos."* Two working paths, both on `c2_6_8b_fp8`:

**(A) Pass the `.mp4` directly — recommended.** Upload the mp4, reference it by filename in `file_ids`, set `max_frames`. GPQ decodes and samples the video server-side; no client-side video tooling needed.

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
  "max_frames": 32,
  "sanitize": false
}
```

**(B) Client-sampled frames.** Sample frames yourself (e.g. ffmpeg), send each as a `data.base64_img` event, and set **`multi_image: true`**. Use when you want control over which/how many frames. `multi_image: true` is **required** — without it, multiple image inputs return `400 query_failed`.

```json
{
  "query": "Evaluate the assembly steps shown in the attached video.",
  "instruction_prompt": "<task / inspection prompt>",
  "file_ids": [],
  "model": "Newton::c2_6_8b_fp8_260424d7a55d5e",
  "max_new_tokens": 500,
  "multi_image": true,
  "sanitize": false,
  "events": [
    { "type": "data.base64_img", "event_data": { "contents": "<frame 1 base64>", "mime_type": "image/png" } },
    { "type": "data.base64_img", "event_data": { "contents": "<frame 2 base64>", "mime_type": "image/png" } }
  ]
}
```

- **`max_frames`** (Path A) — frames GPQ samples uniformly from the video before feeding the model. Default 32. Increase for longer/higher-detail clips; decrease to cap latency.
- **Timeout** — set the client request timeout to **at least 600s**. Video latency scales with `max_frames` and clip length; default 30–60s timeouts will trip on non-trivial clips.
- **`.mp4`-direct on C 2.6 is a recent platform change.** Earlier, the `.mp4` path silently dropped the frames for *every* model (a known GPQ bug), so older notes/skills say "video doesn't work on `/query`." On C 2.6 today it does, via both paths above.

See [`video_query.py`](references/video_query.py) for runnable demos of both paths plus a `max_frames` tradeoff (using a worker-assembly PASS/FAIL inspection prompt).

## Latency Budgets

Observed against prod with the sample assets in `references/sample_assets/`:

| Modality | Body | p50 | Notes |
|----------|------|----:|-------|
| Text only | `{"query": "...", "file_ids": []}` | **~500 ms** | Confirmed during model-identifier probing |
| Text + JSON output | instruction prompt has schema | ~600 ms | Same shape, slightly longer with verbose schema |
| Image via `file_ids` | one 1.9 MB PNG | **~1.3–1.9 s** | observed (wind-turbines.png) |
| Image via inline base64 | same | **~0.8–2.5 s** | base64 inflates the request; scales with image size |
| Video, mp4 + `max_frames=32` | ~31 s assembly clip | **~2–6 s** | server-side decode/sample (Path A) |
| Video, client frames + `multi_image` | 8 frames | **~5 s** | one `/query`, no upload; needs client-side sampling (Path B) |

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
- **Multi-frame needs `multi_image: true`.** Sending several image inputs (client-sampled video frames) as multiple `data.base64_img` events or `file_ids` returns `400 query_failed` **unless** you set `multi_image: true` — that flag tells the model to treat them as one multi-frame input rather than independent queries. Not needed for the `.mp4`-direct path (Path A), only for client-sampled frames (Path B).
- **"Video doesn't work on `/query`" is stale.** Older notes say C 2.6 (and 2.4/2.5) can't see video. That was a GPQ bug that silently dropped `.mp4` frames before any model. C 2.6 reads video now via Path A (mp4 + `max_frames`) or Path B (`multi_image` frames) — verified. The 2.4/2.5 *text* checkpoints still ignore frames.
- **`file_ids` takes the filename, not the `fil_...` uid.** On upload the API returns both `{"file_id": "<filename>", "file_uid": "fil_..."}`. `/query` resolves `file_ids` by the **filename** (the `file_id` value, e.g. `"dashboard.png"`), not the `fil_...` `file_uid`. Passing the `file_uid` fails with a **misleading** `400 unsupported_file_type` ("Found unsupported file type with file: fil_…") — it looks like a content-type problem but is really the wrong identifier. Verified: `file_ids=["wind-turbines.png"]` → 200; `file_ids=["fil_…"]` → 400.
- **Text attachments must be `text/plain`, not `text/csv`.** A `.csv` upload (`text/csv`) is accepted, stored byte-for-byte, and returns `status: completed` — but its contents **never reach the model** on `/query`; the model silently confabulates from priors (e.g. answers "TCP"/"port 80" for an all-SSH file). The **identical bytes** uploaded as `.txt` (`text/plain`) are read faithfully. There is **no warning** — the failure is silent, so you can only catch it by validating the answer. Upload tabular/text data as `.txt`, or inline it in `query`. Verify yourself with [`csv_vs_txt_proof.py`](references/csv_vs_txt_proof.py) (`.txt` reads 5/5, `.csv` 0/5 on identical content). `application/json` uploads are rejected with `400 invalid_file_type`.
- **Mixed-modality calls.** You can attach both an image and a video file_id, but the fusion model's attention is finite. Stick to one modality per call for predictable results.

## Local Setup

```bash
pip install requests python-dotenv

# Drop a .env at the repo root (or anywhere up the tree from the run dir):
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://api.u1.archetypeai.app/v0.5
EOF

python skills/atai-newton-fusion-model/references/text_query.py
python skills/atai-newton-fusion-model/references/image_query.py
python skills/atai-newton-fusion-model/references/video_query.py
```

The scripts auto-load `.env` if `python-dotenv` is installed — `find_dotenv()` walks up from cwd so the file works at the repo root, the project root, or inside `references/`. Without `python-dotenv`, the scripts fall back to plain environment variables (`export ATAI_API_KEY=...`).

## File Layout

```
skills/atai-newton-fusion-model/
├── SKILL.md                  ← this file
├── references/
│   ├── _common.py            ← shared auth / upload / extract helpers
│   ├── text_query.py         ← 3 text patterns (plain, JSON output, .txt flow-log attachment)
│   ├── image_query.py        ← 3 image patterns (file_ids, base64, JSON extraction)
│   ├── video_query.py        ← 3 video patterns (mp4 direct, max_frames tradeoff, client-sampled frames)
│   ├── csv_vs_txt_proof.py   ← diagnostic: proves text/csv attachments aren't read, text/plain are
│   ├── .env.example          ← copy to .env and fill in
│   └── sample_assets/
│       ├── wind-turbines.png ← AI-generated photo (default for image_query.py)
│       ├── 1_pass_2_pass_3_pass_B.mp4 ← assembly-inspection clip (default for video_query.py)
│       └── README.md         ← provenance of the samples
└── tests/
    └── test_references.py    ← network-free unit tests (python -m unittest)
```
