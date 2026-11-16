---
name: atai-newton-fusion-inference-api
description: >
  Call Newton's multimodal models on the Inference API (also called Model
  Serving) — Newton C 2.6 (`newton-c-2.6`, the default) and Newton Fusion 1.0
  (`newton-fusion-1.0`) — with text, an image, or a short video in one
  stateless request, on any of three APIs under `/api/inference`: Direct Query
  shape (`/query`), OpenAI Chat Completions, and OpenAI Responses. Use this
  skill when the user wants per-call multimodal reasoning on the Inference
  API, wants Newton Fusion 1.0, or wants to call Newton through an OpenAI
  SDK. Covers the request shape per modality and API, inline media (uploads
  are refused), the 224 px / 16-frame input, the ~60 s query limit, Fusion's
  reasoning (`<think>`, `/no_think`, token budget), and how to pick a model.
  Do NOT use when the user asks for the Direct Query API, or needs uploaded
  files, full-resolution images, more than 16 video frames, or long answers —
  that's `atai-newton-fusion-model`. Do NOT use for time-series embeddings
  (`atai-newton-omega-inference-api`).
---

# Newton C 2.6 and Fusion 1.0 on the Inference API

One stateless POST per request, for text, an image, or a short video, against Newton C 2.6 or Newton Fusion 1.0. No session, no batch pipeline, no SSE.

## Which Fusion skill?

During the move from the Direct Query API to the Inference API, there are two skills for Newton's multimodal models:

| | This skill | [`atai-newton-fusion-model`](../atai-newton-fusion-model/SKILL.md) |
|---|---|---|
| API | Inference API (`/api/inference/*`) | Direct Query API (`/api/lens/query`) |
| Models | Newton C 2.6 (`newton-c-2.6`), Newton Fusion 1.0 (`newton-fusion-1.0`) | Newton C 2.6 (`Newton::c2_6_8b_fp8_260424d7a55d5e`) |
| Images | Inline data URIs on Chat Completions / Responses; resized to 224 × 224 | Uploaded files or inline events, at full resolution |
| Video | Inline clip under ~1.4 MB; 16 frames at 224 px | Uploaded clip; `max_frames` up to 64 |
| Uploaded files | Refused | Yes |
| Long answers | Stopped at ~60 s | No limit seen |
| Use when | The user asks for the Inference API, Model Serving, Fusion 1.0 or an OpenAI SDK, and the inputs fit the limits above | The user asks for the Direct Query API, or needs any of the inputs this skill can't take |

These are the **current limitations as of November 15, 2026, on production**; see [Migrating from the Direct Query API](https://docs.archetypeai.app/api-reference/inference/migrate-from-direct-query#current-limitations) for the up-to-date list.

**Accounts:** the free trial (`try.archetypeai.app`) serves neither this API's models nor the Direct Query API; both skills need a production account.

## When to Apply

- Describe / classify / extract structure from an image or a short video in one API call
- Stateless multimodal Q&A in a serverless or per-request handler
- JSON output from a multimodal input (the prompt is the schema)
- Choosing between C 2.6 and Fusion for a workload
- Calling Newton through an OpenAI SDK (Chat Completions or Responses)

**Do not use this skill when**:
- The user asks for the Direct Query API, or the inputs need uploads, full resolution, more than 16 frames or long answers — use [`atai-newton-fusion-model`](../atai-newton-fusion-model/SKILL.md)
- The workload is live, streaming video → text from a long-running camera feed (a session-based activity monitor is the right shape)
- The workload is a large multi-file batch job
- The task is time-series classification via embeddings ([`atai-newton-omega-inference-api`](../atai-newton-omega-inference-api/SKILL.md))

## Which Model

| | `newton-c-2.6` | `newton-fusion-1.0` |
|---|---|---|
| Size | 8B, dense (every parameter used for every token) | 35B mixture-of-experts, ~3B used per token |
| Reasoning step | None | Yes: replies open with a `<think>` block (see [Fusion Reasoning](#fusion-reasoning)) |
| Text-only latency | ~2–4 s | ~16–40 s (it always reasons on text) |
| Image / video latency | ~1–4 s | ~1–3 s without reasoning |
| Extra handling | None | Strip `<think>`; larger budget when it reasons |

1. **Default to C 2.6.** It is smaller and predictable, needs no reasoning handling, and is the same model as the Direct Query API's C 2.6, so switching code over is simplest.
2. **Try Fusion when the task needs judgement over several steps** — counting, comparing, following a procedure, explaining why — **and C 2.6's answers are wrong on your data.** Use its reasoning mode when accuracy matters more than speed.
3. **Decide on your own inputs, not on this table.** Run [`compare_models.py`](references/compare_models.py) on 10–20 inputs that look like your workload and read the answers side by side.

What we saw on the assembly clip's step-by-step PASS/FAIL check (31 s, ground truth: all three steps PASS; measured on 2026-11-07, from the same 16 frames at 224 px the inline clip gives). Verdicts were identical on each of 3 runs:

| Steps correct | Full clip | Cropped to the work area | Split into one piece per step |
|---|---|---|---|
| `newton-c-2.6` | 1/3 | 1/3 | 1/3 |
| `newton-fusion-1.0` | 0/3 | 2/3 | **3/3** |
| Fusion + reasoning | 1/3 | 2/3 | **3/3** |

- **Fusion was limited by its input:** zooming in helped, and giving each step its own 16 frames solved it.
- **C 2.6 was limited by perception, not frames.** On an 8 s piece showing only the wrench step it answered that the wrench was used "to loosen the blue cap, not tighten it".
- So for fine-grained procedure checks, use Fusion and [split the clip](#video) so each part of the action gets the full frame budget. One clip is not a benchmark — confirm on your own.

## Endpoints

```
POST {ATAI_API_ENDPOINT}/inference/query               Direct Query shape (text, video)
POST {ATAI_API_ENDPOINT}/inference/chat/completions    OpenAI Chat Completions shape (text, image, video)
POST {ATAI_API_ENDPOINT}/inference/responses           OpenAI Responses shape (text, image, video)
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

`ATAI_API_ENDPOINT` is `https://platform.u1.archetypeai.app/api` on production. The body's `model` names the model: `newton-c-2.6` or `newton-fusion-1.0`. Streaming (`"stream": true`) is refused.

**Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required** — there is no default endpoint, so a wrong-deployment mistake fails loudly at startup. The scripts accept the bare root, `/api`, or `/api/inference`.

**OpenAI SDK:** point `base_url` at `{ATAI_API_ENDPOINT}/inference` and use your Archetype key:

```python
from openai import OpenAI
client = OpenAI(base_url="https://platform.u1.archetypeai.app/api/inference", api_key=ATAI_API_KEY)
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
  "max_new_tokens": 300
}
```

- **The system turn goes in `system_prompt`.** On the Inference API it is the real system message. `instruction_prompt` is only appended to the query text (the reverse of the Direct Query API, where only `instruction_prompt` worked).
- JSON output: put the schema in `system_prompt`, the input in `query`, and tell the model not to wrap it in markdown fences.
- **Text files can't be attached.** A `.txt` is refused by every route. Put the text in `query` — the 14-row flow log in [`text_query.py`](references/text_query.py) goes inline. `data.text` and `data.json` events are refused too.

Chat Completions and Responses take the same text as a `text` / `input_text` part; the system turn is a `system` message / `instructions`. [`text_query.py`](references/text_query.py) runs one question on all three APIs. Its flow log in example 3 is real data from the [GHOST-IoT dataset](https://github.com/gspathoulas/ghost-iot-dataset) (Anagnostopoulos et al., *Sensors* 2020, [DOI:10.3390/s20226600](https://doi.org/10.3390/s20226600)).

### Image

| API | How the image goes in |
|---|---|
| `/chat/completions` | `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` |
| `/responses` | `{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}` |
| `/query` | **Refused**, in every form |

**`/query` refuses still images** on production, each with a bare 400: a `data.base64_img` event, `base64:<image>` in `file_ids`, an uploaded file by `fil_` uid or by filename, and `multi_image: true`. Send images to Chat Completions or Responses.

**Every image reaches the model as a 224 × 224 square**, whatever size you send, on every endpoint; `detail: "high"` makes no difference. Measured (2026-11-07): on a 1024 px image, both models read a 110 px code but not 14 px or 10 px ones — exactly what they did with the same image pre-shrunk to 224 px — and read the small codes once those lines were cropped and enlarged. **For fine detail (labels, gauges, small parts, text), crop to the region of interest before sending**, or send several crops as separate images. For full-resolution reading, use the Direct Query skill.

**Several independent images** (before/after, multi-view): put several `image_url` parts in one chat message.

See [`image_query.py`](references/image_query.py): inline on `/chat/completions`, JSON extraction on `/responses`, and a before/after pair.

### Video

| API | How the clip goes in |
|---|---|
| `/query` | `file_ids: ["base64:<mp4 bytes>"]` |
| `/chat/completions` | `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` |
| `/responses` | `{"type": "input_video", "video_url": "data:video/mp4;base64,..."}` |

```json
{
  "model": "newton-c-2.6",
  "query": "Evaluate the assembly steps shown in the attached video.",
  "system_prompt": "<task / inspection prompt>",
  "file_ids": ["base64:AAAAIGZ0eXBpc29t..."],
  "max_new_tokens": 500
}
```

- **The frame budget is fixed at 16 frames × 224 px.** The video reader takes `min(frames in clip, 16)` evenly spaced frames across the whole clip and resizes each to a **224 × 224 square** (a 16:9 clip is squashed, not cropped). On a 31 s clip that is one frame every ~2 s. `max_frames` in the request is refused, and a session can't change it either. **To see a brief action in detail, split or trim the clip** so the 16 frames land on it — one piece per procedure step, or fixed ~8–10 s windows whose verdicts you combine.
- **No frame lists.** `data.base64_img` frame events, `multi_image` and `query_metadata` are refused. If you have frames, mux them into an `.mp4` first: `ffmpeg -framerate 2 -i f%03d.jpg -c:v libx264 -pix_fmt yuv420p clip.mp4`.
- **A clip with no audio track is fine** (the sample has none).

See [`video_query.py`](references/video_query.py): the 470 KB assembly clip inline on all three APIs, with a PASS/FAIL prompt on `/query`.

### Inline only

**Uploaded files can't be used.** `/query` refuses a file uploaded through `/api/files`, by `fil_` uid and by filename alike, for images and video. Every image and clip goes inline, and inline data counts toward the request body, which is capped at **2 MiB**: a 1.4 MiB body worked, a 2.1 MiB body hung with no answer, and 3.2 MiB got `413` (production, 2026-11-15). Keep each clip under about 1.4 MB. Because the model sees only 16 frames at 224 px, shrinking loses nothing:

```bash
ffmpeg -i input.mp4 -vf "scale=448:-2,fps=4" -c:v libx264 -crf 28 -an small.mp4
```

The helpers refuse oversized inline payloads before sending. For larger files, use the Direct Query skill.

## Long Answers and the Time Limit

A query is stopped after about **60 seconds** and returns `504 query_timeout`. Generation is slower than on the Direct Query API — Newton C 2.6 wrote about 40 tokens (100–140 characters) a second, against about 210 characters on the Direct Query API — so **answers over about 2,000 tokens time out**. Open-ended analyses of a long text input are the usual case: in an earthquake-monitoring app, one question timed out every time at `max_new_tokens: 3072`.

- Ask for a bounded answer in the prompt ("Keep the answer under 300 words") and set `max_new_tokens` to about 1,536.
- When listing numbers, C 2.6 can repeat itself ("M1.8, M1.9, M1.8, …") until it runs out of tokens or time. `repetition_penalty: 1.1` stops it — **on `/query` only**: Chat Completions and Responses accept the field and ignore it (production, 2026-11-15).
- Fusion on text is the slowest case: its reasoning took 16–40 s per answer on the bundled examples, close to the limit.
- If an answer genuinely needs to be long, split the request, or use the Direct Query skill.

## Fusion Reasoning

Fusion is a reasoning model. Every reply opens with `<think>…</think>` — empty when it didn't reason. Strip it before showing the answer ([`split_thinking()`](references/_common.py)).

| You want | Send | What happens |
|---|---|---|
| No reasoning, with an image or video | append ` /no_think` to the user text | Empty `<think>` block, fast answer |
| No reasoning, text only | — | **Not possible:** `/no_think` is ignored without media, Fusion reasons anyway |
| Reasoning | no switch | Fusion decides; on non-trivial prompts it reasons |

- `/think` does **not** force reasoning — the block stays empty. Send no switch instead.
- **When Fusion reasons it can outrun the token budget**, and then there is no answer — only an unclosed `<think>`. A 14-row flow log took ~9,000 characters of reasoning; a dense night-time video looped while counting cars one by one. `repetition_penalty: 1.1` fixed both, and 4,096 tokens was enough for the bundled examples. The helpers apply both whenever Fusion reasons (reasoning mode, or any text-only request).
- **Images can't get the penalty:** they go through Chat Completions or Responses, which ignore it. For images, prefer `/no_think` (or C 2.6) unless you've checked that reasoning finishes on your inputs.
- **Reasoning counts against the ~60 s limit.** Long reasoning on long text can hit `504`.
- Treat an unclosed `<think>` as a failure, not as an answer. `split_thinking()` raises on it.

## Generation Parameters

| Param | Notes |
|-------|-------|
| `max_new_tokens` (`max_tokens` / `max_output_tokens` on the OpenAI APIs) | Hard ceiling on generated tokens; keep answers under ~2,000 tokens (see the time limit) |
| `temperature` | 0.0 = greedy |
| `do_sample` | `false` = greedy decoding |
| `top_p`, `top_k` | Nucleus / top-k sampling |
| `repetition_penalty` | Try 1.1 if the model loops — applied on `/query` only |
| `presence_penalty`, `frequency_penalty` | **Refused** (400) |

With the defaults, repeated calls on the same input returned identical answers in our runs. Don't rely on it for borderline verdicts: aggregate over a few calls if a verdict matters.

## Latency

Measured on production, 2026-11-15, warm, with the bundled scripts (one run each):

| Input | `newton-c-2.6` | `newton-fusion-1.0` | Fusion + reasoning |
|---|---:|---:|---:|
| Short text question | 2.2 s | 16 s | 22 s |
| Text → JSON | 3.7 s | 19 s | 2.9 s |
| 14-row flow log (query / chat / responses) | 2.2 / 2.1 / 2.1 s | 29 / 40 / 40 s | 38 / 33 / 33 s |
| Image inline (chat / responses) | 1.4 / 1.4 s | 1.5 / 1.5 s | 1.4 / 1.6 s |
| Before/after image pair (chat) | 3.2 s | 1.6 s | 1.4 s |
| 470 KB clip inline, PASS/FAIL on `/query` | 3.8 s | 3.0 s | 4.0 s |
| 470 KB clip inline, describe (chat / responses) | 1.8 / 1.7 s | 1.6 / 1.3 s | 11.9 / 5.5 s |

- **Cold start:** the first call after a model has been idle can take ~40 s. Set client timeouts well above that.
- **Server budget:** a request still running at ~60 s comes back as `504`.

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

`POST /api/inference/sessions` opens a session on these models, but it gives them nothing: a session doesn't remember earlier questions (each request carries its own conversation), and it can't change `max_frames` or the image size. Use the stateless calls above.

## Coming from the Direct Query API?

| Direct Query API (`/api/lens/query`) | Inference API (`/api/inference/*`) |
|---|---|
| `"model": "Newton::c2_6_8b_fp8_260424d7a55d5e"` | `"model": "newton-c-2.6"` (or `"newton-fusion-1.0"`); the old identifier returns 404 |
| System turn in `instruction_prompt` | System turn in `system_prompt` |
| Uploaded files in `file_ids` (by filename) | **Refused** (by filename or `fil_` uid); send media inline |
| Inline image as a `data.base64_img` event | Data URI on `/chat/completions` or `/responses`; refused on `/query` |
| `multi_image: true` for independent images | Several `image_url` parts in one chat message |
| Frame list + `query_metadata` | Mux the frames into an `.mp4` |
| `max_frames` up to 64 | Fixed 16 frames × 224 px; `max_frames` refused |
| Full-resolution images | 224 × 224 |
| Long answers | ~60 s per query |
| `.txt` attachments, `data.text` events | Refused; inline the text |
| `presence_penalty` | Refused |

Every refusal is a bare `400` with no reason, so check this table first when old code starts failing. If a row matters to your app, keep that call on the Direct Query API ([`atai-newton-fusion-model`](../atai-newton-fusion-model/SKILL.md)) for now.

## Common Pitfalls

- **Refusals don't say why.** Every refused field returns `400 {"code": "bad_request", "message": "model '<model>' refused this request"}`. Compare your body against [Coming from the Direct Query API?](#coming-from-the-direct-query-api).
- **Images on `/query`.** Refused in every form. Use `/chat/completions` or `/responses`.
- **Uploaded files.** Refused. Send media inline.
- **Inline payload too big.** Over 2 MiB of request body → `413` or a hang. Shrink the clip, or use the Direct Query skill.
- **Answer times out.** Over ~2,000 tokens, or long Fusion reasoning, hits the ~60 s limit (`504`). Bound the answer.
- **Fusion answer is empty or cut off.** It ran out of tokens while reasoning. Raise the budget (≥ 4,096) and set `repetition_penalty: 1.1` (on `/query`), or add ` /no_think` when the request has media.
- **Fusion is slow on plain text.** It always reasons without media. Use C 2.6 for text-only work unless Fusion's answers are measurably better.
- **Showing `<think>` to users.** Strip it from every Fusion reply.
- **Expecting fine detail from a large image.** Images and video frames are squashed to 224 × 224 before the model sees them. Crop to what matters. When it couldn't read small text, C 2.6 answered "?", but **Fusion invented plausible digits** — validate anything Fusion reads from small print.
- **Expecting more than 16 frames.** Long clips are sampled down to 16 frames. Split long videos into shorter clips when the action is brief or the check is fine-grained.
- **JSON wrapped in markdown fences.** Parse defensively or restate the no-fences rule in both `system_prompt` and the query.
- **Shell variables beat `.env`.** `ATAI_API_*` exported in your shell override `.env` (`load_dotenv(override=False)`). Unset them if the scripts hit the wrong environment.

## Local Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r skills/atai-newton-fusion-inference-api/references/requirements.txt

cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://platform.u1.archetypeai.app/api
EOF

cd skills/atai-newton-fusion-inference-api/references
python text_query.py                   # C 2.6 (default)
python image_query.py --model fusion   # Fusion, no reasoning
python video_query.py --model fusion --reasoning
python compare_models.py --prompt "Describe what happens." \
  ../../atai-newton-fusion-model/references/sample_assets/1_pass_2_pass_3_pass_B_small.mp4
```

The scripts use the [official Archetype AI python client](https://github.com/archetypeai/python-client) (`pip install archetypeai`): requests go through `client.requests_post(...)`. They auto-load `.env` if `python-dotenv` is installed.

## File Layout

```
skills/atai-newton-fusion-inference-api/
├── SKILL.md                  ← this file
├── references/
│   ├── _common.py            ← client, query / chat / respond, inline-size guard, Fusion reasoning handling
│   ├── text_query.py         ← plain, JSON, inline flow log; the same question on all three APIs
│   ├── image_query.py        ← inline on chat, JSON on responses, before/after pair
│   ├── video_query.py        ← inline clip on all three APIs, PASS/FAIL on /query
│   ├── compare_models.py     ← same inputs through C 2.6, Fusion, Fusion + reasoning, side by side
│   ├── requirements.txt
│   └── .env.example
└── tests/
    └── test_references.py    ← network-free unit tests (python -m unittest)
```

The sample images and clips are shared with the Direct Query skill rather than copied: the scripts read them from [`../atai-newton-fusion-model/references/sample_assets/`](../atai-newton-fusion-model/references/sample_assets/README.md) (the small inline copies are `wind-turbines-small.jpg` and `1_pass_2_pass_3_pass_B_small.mp4`; the README there has their provenance).
