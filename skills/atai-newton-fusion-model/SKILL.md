---
name: atai-newton-fusion-model
description: >
  Call Newton's multimodal models on Model Serving — Newton-C 2.6
  (`newton-c-2.6`, the default) and Newton-Fusion 1.0
  (`newton-fusion-1.0`) — with text, image, and video in one stateless
  request, on any of three APIs under `/api/inference`: Direct Query
  (`/query`), OpenAI Chat Completions, and OpenAI Responses. Use this skill
  when the user wants per-call multimodal reasoning (describe an image,
  summarize or inspect a clip, classify a state, extract JSON) without a
  session or batch job, or wants to choose between C 2.6 and Fusion.
  Covers the request shape per modality and API, uploads (`fil_` uids) vs
  inline data, which fields Model Serving refuses, Fusion's reasoning
  (`<think>`, `/no_think`, token budget), generation parameters, latency,
  and how to pick a model with the bundled comparison script.
  Do NOT use for streaming / session-based activity monitoring, large
  multi-file batch jobs, or time-series embeddings (that's
  `atai-newton-omega-model`).
---

# Newton C 2.6 and Fusion 1.0 — Multimodal Model Serving in One Hop

One stateless POST per request, for text, image, or video, against Newton-C 2.6 or Newton-Fusion 1.0. No session, no batch pipeline, no SSE. One call → one response.

> **Model Serving is on Dev for now** (`https://platform.dev.u1.archetypeai.app`). Every number below was measured there on 2026-11-07.

## When to Apply

- Describe / classify / extract structure from an image or a short video in one API call
- Stateless multimodal Q&A in a serverless or per-request handler
- JSON output from a multimodal input (the prompt is the schema)
- Choosing between C 2.6 and Fusion for a workload
- Calling Newton through an OpenAI SDK (Chat Completions or Responses)

**Do not use this skill when**:
- The workload is live, streaming video → text from a long-running camera feed (a session-based activity monitor is the right shape)
- The workload is a large multi-file batch job
- The task is time-series classification via embeddings (`atai-newton-omega-model`)

## Which Model

| | `newton-c-2.6` | `newton-fusion-1.0` |
|---|---|---|
| Size | 8B, dense (every parameter used for every token) | 35B mixture-of-experts, ~3B used per token |
| Reasoning step | None | Yes: replies open with a `<think>` block (see [Fusion Reasoning](#fusion-reasoning)) |
| Text-only latency | ~1–6 s | ~15–40 s (it always reasons on text) |
| Image / video latency | ~1–9 s | ~1–8 s |
| Extra handling | None | Strip `<think>`; larger budget when it reasons |

1. **Default to C 2.6.** It is smaller and predictable, needs no reasoning handling, and is the closest match to the Direct Query API's C 2.6, so switching code over is simplest.
2. **Try Fusion when the task needs judgement over several steps** — counting, comparing, following a procedure, explaining why — **and C 2.6's answers are wrong on your data.** Use its reasoning mode when accuracy matters more than speed.
3. **Decide on your own inputs, not on this table.** On simple descriptions the two give similar answers. Run [`compare_models.py`](references/compare_models.py) on 10–20 inputs that look like your workload and read the answers side by side.

What we saw on the bundled samples: both describe the images equally well. The assembly clip's step-by-step PASS/FAIL check (31 s, ground truth: all three steps PASS) separates them. Verdicts were identical on each of 3 runs:

| Steps correct | Full clip (16 frames over 31 s) | Cropped to the work area | Split into one piece per step |
|---|---|---|---|
| `newton-c-2.6` | 1/3 | 1/3 | 1/3 |
| `newton-fusion-1.0` | 0/3 | 2/3 | **3/3** |
| Fusion + reasoning | 1/3 | 2/3 | **3/3** |

- **Fusion was limited by its input:** zooming in helped, and giving each step its own 16 frames solved it.
- **C 2.6 was limited by perception, not frames.** On an 8 s piece showing only the wrench step it answered that the wrench was used "to loosen the blue cap, not tighten it"; on the manifold piece, that the cap was "being held separately".
- So for fine-grained procedure checks, use Fusion and [split the clip](#video) so each part of the action gets the full frame budget. One clip is not a benchmark — confirm on your own.

## Endpoints

```
POST {root}/api/inference/query               Direct Query shape
POST {root}/api/inference/chat/completions    OpenAI Chat Completions shape
POST {root}/api/inference/responses           OpenAI Responses shape
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

The body's `model` names the model: `newton-c-2.6` or `newton-fusion-1.0`. All three APIs reach the same models with the same inputs — pick by the client you already have. Streaming (`"stream": true`) is refused.

**Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required** — there is no default endpoint, so a wrong-deployment mistake fails loudly at startup. The scripts accept the bare root, `/api`, or `/api/inference`.

**OpenAI SDK:** point `base_url` at `{root}/api/inference` and use your Archetype key:

```python
from openai import OpenAI
client = OpenAI(base_url=f"{root}/api/inference", api_key=ATAI_API_KEY)
reply = client.chat.completions.create(
    model="newton-c-2.6",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "What is in this image?"},
    ]}],
    max_tokens=200,
)
```

## Wire Shapes

### Text + JSON

```json
{
  "model": "newton-c-2.6",
  "query": "Classify the operational state. Respond with ONLY the JSON object.",
  "system_prompt": "You output {\"state\": \"...\", \"confidence\": <0..1>}.",
  "file_ids": [],
  "max_new_tokens": 300
}
```

- **The system turn goes in `system_prompt`.** On Model Serving it is the real system message. `instruction_prompt` is only appended to the query text (the reverse of the Direct Query API, where only `instruction_prompt` worked).
- JSON output: put the schema in `system_prompt`, the input in `query`, and tell the model not to wrap it in markdown fences.
- **Text files can't be attached.** A `.txt` is refused by every route (`fil_` uid, filename, `base64:`). Put the text in `query` — the 14-row flow log in [`text_query.py`](references/text_query.py) goes inline.

Chat Completions and Responses take the same text as a `text` / `input_text` part; the system turn is a `system` message / `instructions`. [`text_query.py`](references/text_query.py) runs one question on all three APIs. Its examples are framed as smart-home WiFi flow analysis; the flow log in example 3 is real data from the [GHOST-IoT dataset](https://github.com/gspathoulas/ghost-iot-dataset) (Anagnostopoulos et al., *Sensors* 2020, [DOI:10.3390/s20226600](https://doi.org/10.3390/s20226600)).

### Image

| API | How the image goes in |
|---|---|
| `/query` | Upload via `/api/files`, put the returned **`file_uid`** (`fil_...`) in `file_ids` |
| `/chat/completions` | `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` |
| `/responses` | `{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}` |

```bash
curl -X POST $ROOT/api/files -H "Authorization: Bearer $ATAI_API_KEY" -F "file=@photo.png;type=image/png"
# → {"file_id": "photo.png", "file_uid": "fil_33fd6ct3...", "is_valid": true}
```

```json
{
  "model": "newton-c-2.6",
  "query": "Describe this image in three bullets.",
  "system_prompt": "...",
  "file_ids": ["fil_33fd6ct3..."],
  "max_new_tokens": 300
}
```

**Several independent images** (before/after, multi-view): put several `image_url` parts in one chat message. `multi_image` on `/query` is refused.

**Refused for images** (each a bare 400): inline `data.base64_img` events, `base64:<image>` in `file_ids`, the upload's **filename** in `file_ids`, `multi_image: true`, and a `fil_` uid as a chat `image_url`.

See [`image_query.py`](references/image_query.py): upload on `/query`, inline on `/chat/completions`, JSON extraction on `/responses`, and a before/after pair.

### Video

| API | How the clip goes in |
|---|---|
| `/query` | `file_ids: ["fil_..."]` (uploaded) or `file_ids: ["base64:<mp4 bytes>"]` (inline) |
| `/chat/completions` | `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` |
| `/responses` | `{"type": "input_video", "video_url": "data:video/mp4;base64,..."}` |

```json
{
  "model": "newton-c-2.6",
  "query": "Evaluate the assembly steps shown in the attached video.",
  "system_prompt": "<task / inspection prompt>",
  "file_ids": ["fil_7fmsm4xmjy..."],
  "max_new_tokens": 500
}
```

- **The frame budget is fixed at 16 frames × 224 px.** Both models' blueprints set `max_frames: 16` and `size: 224`: the video reader takes `min(frames in clip, 16)` evenly spaced frames across the whole clip and resizes each to a **224 × 224 square** (a 16:9 clip is squashed, not cropped). On a 31 s clip that is one frame every ~2 s. `max_frames` in the request is refused, and a session can't change it either (422 `` `max_frames` reaches node `video_reader` outside its session fields``). **To see a brief action in detail, split or trim the clip** so the 16 frames land on it — e.g. one piece per procedure step, or fixed ~8–10 s windows whose verdicts you combine. On the assembly sample this took Fusion from 0/3 to 3/3 steps correct (see [Which Model](#which-model)). This is a cut from the Direct Query API, where C 2.6 took up to 64 frames.
- **No frame lists.** `data.base64_img` frame events, `multi_image: false` and `query_metadata` are gone. If you have frames, mux them into an `.mp4` first: `ffmpeg -framerate 2 -i f%03d.jpg -c:v libx264 -pix_fmt yuv420p clip.mp4`.
- **A clip with no audio track is fine** (the inline samples have none). The models' blueprints also transcribe a clip's audio for the model; that path wasn't measured here.
- **Refused for video:** the upload's filename in `file_ids`, a `fil_` uid as a chat `video_url`, `max_frames`.

See [`video_query.py`](references/video_query.py): the full 7.8 MB clip uploaded on `/query`, and a 470 KB re-encode inline on all three APIs.

### Inline vs upload

Inline data (base64, data URIs) counts toward the request body. **Keep it under ~1.5 MB as base64:** two 400 KB PNGs (~1.1 MB) worked, a 1.9 MB PNG (2.6 MB as base64) got `413`, and a 7.8 MB clip (~10 MB) hung until the client gave up. Upload anything bigger and send its `fil_` uid on `/query`. The helpers refuse oversized inline payloads before sending.

## Fusion Reasoning

Fusion is a reasoning model. Every reply opens with `<think>…</think>` — empty when it didn't reason. Strip it before showing the answer ([`split_thinking()`](references/_common.py)).

| You want | Send | What happens |
|---|---|---|
| No reasoning, with an image or video | append ` /no_think` to the user text | Empty `<think>` block, fast answer |
| No reasoning, text only | — | **Not possible:** `/no_think` is ignored without media, Fusion reasons anyway |
| Reasoning | no switch | Fusion decides; on non-trivial prompts it reasons |

- `/think` does **not** force reasoning — the block stays empty. Send no switch instead.
- **When Fusion reasons it can outrun the token budget**, and then there is no answer — only an unclosed `<think>`. A 14-row flow log took ~9,000 characters of reasoning (~37 s); a dense night-time video looped while counting cars one by one. `repetition_penalty: 1.1` fixed both in every test, and 4,096 tokens was enough for the bundled examples. The helpers apply both automatically whenever Fusion reasons (reasoning mode, or any text-only request).
- Treat an unclosed `<think>` as a failure, not as an answer. `split_thinking()` raises on it.

## Generation Parameters

| Param | Notes |
|-------|-------|
| `max_new_tokens` (`max_tokens` / `max_output_tokens` on the OpenAI APIs) | Hard ceiling on generated tokens. Fusion's limit is 65,536 |
| `temperature` | 0.0 = greedy |
| `do_sample` | `false` = greedy decoding |
| `top_p`, `top_k` | Nucleus / top-k sampling |
| `repetition_penalty` | Try 1.1 if the model loops — and always for Fusion reasoning |
| `presence_penalty` | **Refused** (400) |

With the defaults, repeated calls on the same input returned identical answers in our runs (3/3 on the assembly clip for each model, 2/2 on text). Don't rely on it for borderline verdicts: aggregate over a few calls if a verdict matters.

## Latency

Measured on Dev, 2026-11-07, warm replica, with the bundled scripts:

| Input | `newton-c-2.6` | `newton-fusion-1.0` | Fusion + reasoning |
|---|---:|---:|---:|
| Short text question | 2.4 s | 16 s | 22 s |
| Text → JSON | 3.3 s | 19 s | 2.1 s |
| 14-row flow log (query / chat / responses) | 6.2 / 2.0 / 2.1 s | 29 / 40 / 40 s | 37 / 33 / 33 s |
| Image, uploaded `fil_` on `/query` | 2.5 s | 2.7 s | 3.0 s |
| Image inline (chat / responses) | 0.9 / 1.6 s | 1.9 / 1.5 s | 0.8 / 1.5 s |
| Before/after image pair (chat) | 3.4 s | 1.6 s | 1.5 s |
| 7.8 MB clip, uploaded, PASS/FAIL prompt | 8.8 s | 8.2 s | 7.4 s |
| 470 KB clip inline (query / chat / responses) | 1.9 / 1.8 / 1.9 s | 1.7 / 1.4 / 1.4 s | 1.9 / 1.5 / 1.6 s |

- **Cold start:** the first call after a model has been idle took ~40 s. Set client timeouts well above that; the scripts use 180 s.
- **Server budget:** a request still running at ~60 s comes back as `504`. Fusion reasoning on long text gets close.
- Uploads of the 7.8 MB clip took ~2.4 s.

## Response Shapes

```json
// /inference/query
{"query_id": "...", "status": "completed", "response": {"response": ["<text>"], "success": true}}
// /inference/chat/completions
{"choices": [{"message": {"role": "assistant", "content": "<text>"}}]}
// /inference/responses
{"output": [{"type": "message", "content": [{"type": "output_text", "text": "<text>"}]}]}
```

On `/query`, `response.success: false` is a failure even with HTTP 200. [`extract_text()`](references/_common.py) handles all three shapes.

## Sessions

`POST /api/inference/sessions` opens a session on these models, but it gives them nothing: a session doesn't remember earlier questions (each request carries its own conversation), and it can't change `max_frames` or `size`. Use the stateless calls above.

## Coming from the Direct Query API?

| Direct Query API (`/api/lens/query`) | Model Serving (`/api/inference/*`) |
|---|---|
| `"model": "Newton::c2_6_8b_fp8_260424d7a55d5e"` | `"model": "newton-c-2.6"` (or `"newton-fusion-1.0"`) |
| System turn in `instruction_prompt` | System turn in `system_prompt` |
| `file_ids` takes the upload's filename | `file_ids` takes the upload's `fil_` uid; the filename is refused |
| Inline image as a `data.base64_img` event | Data URI on `/chat/completions` or `/responses` |
| `multi_image: true` for independent images | Several `image_url` parts in one chat message |
| Frame list + `query_metadata` | Mux the frames into an `.mp4` |
| `max_frames` up to 64 | Fixed 16 frames × 224 px; `max_frames` refused |
| `.txt` attachments | Refused; inline the text |
| `presence_penalty` | Refused |

Every refusal is a bare `400` with no reason, so check this table first when old code starts failing.

## Common Pitfalls

- **Refusals don't say why.** Every refused field returns `400 {"code": "bad_request", "message": "model '<model>' refused this request"}`. Compare your body against [Coming from the Direct Query API?](#coming-from-the-direct-query-api).
- **Filename instead of `fil_` uid in `file_ids`.** The upload returns both; Model Serving needs `file_uid`.
- **Inline payload too big.** Over ~1.5 MB of base64 → `413` or a hang. Upload instead.
- **Fusion answer is empty or cut off.** It ran out of tokens while reasoning. Raise the budget (≥ 4,096) and set `repetition_penalty: 1.1`, or add ` /no_think` when the request has media.
- **Fusion is slow on plain text.** It always reasons without media. Use C 2.6 for text-only work unless Fusion's answers are measurably better.
- **Showing `<think>` to users.** Strip it from every Fusion reply.
- **Expecting more than 16 frames.** Long clips are sampled down to 16 frames (the Direct Query API allowed 64). Split long videos into shorter clips when the action is brief or the check is fine-grained.
- **JSON wrapped in markdown fences.** Parse defensively or restate the no-fences rule in both `system_prompt` and the query.
- **Shell variables beat `.env`.** `ATAI_API_*` exported in your shell override `.env` (`load_dotenv(override=False)`). Unset them if the scripts hit the wrong environment.

## Local Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r skills/atai-newton-fusion-model/references/requirements.txt

cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://platform.dev.u1.archetypeai.app
EOF

cd skills/atai-newton-fusion-model/references
python text_query.py                   # C 2.6 (default)
python image_query.py --model fusion   # Fusion, no reasoning
python video_query.py --model fusion --reasoning
python compare_models.py --prompt "Describe what happens." sample_assets/1_pass_2_pass_3_pass_B.mp4
```

The scripts use the [official Archetype AI python client](https://github.com/archetypeai/python-client) (`pip install archetypeai`): uploads go through `client.files.local.upload(...)`, requests through `client.requests_post(...)`. They auto-load `.env` if `python-dotenv` is installed.

## File Layout

```
skills/atai-newton-fusion-model/
├── SKILL.md                  ← this file
├── references/
│   ├── _common.py            ← client, upload, query / chat / respond, Fusion reasoning handling
│   ├── text_query.py         ← plain, JSON, inline flow log; the same question on all three APIs
│   ├── image_query.py        ← upload on /query, inline chat, JSON on responses, before/after pair
│   ├── video_query.py        ← uploaded clip on /query; inline clip on all three APIs
│   ├── compare_models.py     ← same inputs through C 2.6, Fusion, Fusion + reasoning, side by side
│   ├── requirements.txt
│   ├── .env.example
│   └── sample_assets/        ← images and clips (full size for uploads, small for inline); see README.md
└── tests/
    └── test_references.py    ← network-free unit tests (python -m unittest)
```
