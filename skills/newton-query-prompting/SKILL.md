---
name: newton-query-prompting
description: >
  Prompt-engineering patterns for Newton's /query text-reasoning endpoint
  across text, structured data (CSV / JSON), and image inputs. Use when the
  user wants Newton to reason over plant state, log data, tabular data,
  configuration files, or screenshots and return JSON / structured output
  — e.g. operator suggestions from sensor readings, categorized findings
  from logs, routing decisions from event streams, or natural-language
  descriptions of dashboards and charts. Covers structured-output
  enforcement, contamination-avoidance, constraint tables, server-side
  pre-picking of salient inputs, topology validation on parsed responses,
  and the right path for embedding text / structured / image data.
  Do NOT use for classification of raw sensor windows into n-shot classes
  (use newton-machine-state, newton-machine-state-batch, or
  newton-machine-state-direct-query).
  Do NOT use for video analysis — the Newton text checkpoints (c2_4_7b,
  c2_5_8b) accept .mp4 file_ids but return polite refusals; use
  newton-activity-monitor for video.
  Do NOT use for initial API setup (use newton-setup).
---

# Newton /query Prompt Engineering

Newton's `/query` endpoint is a multi-modal reasoning path: the model sees a system prompt, a query string, and optionally attached files or inline data events (text, CSV / JSON, base64 images). It returns a text response shaped by your prompt — there's no fixed output schema, the prompt is the schema. This skill covers the prompt patterns that make `/query` reliable in production, plus how to attach data of each supported input type without hitting the gotchas.

Reference implementations for the text + structured-state pattern (all three demos use the same wire shape — text in `query`, empty `file_ids`, no `events`, `sanitize: false`, duplicated `system_prompt`/`instruction_prompt`):

- [`archetypeai-swat-demo-direct-query`](https://github.com/archetypeai/archetypeai-swat-demo-direct-query) — water treatment plant; per-stage NORMAL/ATTACK status with pre-picked z-scored sensor citations.
- [`archetypeai-earthquake-demo`](https://github.com/archetypeai/archetypeai-earthquake-demo) — seismic event reasoning.
- [`archetypeai-grid-demo`](https://github.com/archetypeai/archetypeai-grid-demo) — power-grid status reasoning.
- Predecessor: [`archetypeai-swat-demo`](https://github.com/archetypeai/archetypeai-swat-demo) — same prompt patterns, Lens-based classification underneath.

Each ships a near-identical `src/lib/server/newton.js::queryNewton(query)` helper — the only domain-specific bits are the `SYSTEM_PROMPT` content and what gets serialized into `query`.

## When to Apply

- User wants Newton to output structured JSON (arrays, objects, typed fields) rather than prose
- User needs Newton to route outputs by topology (source→target, categories, zones)
- User wants Newton to reason over a CSV / JSON document or describe an image
- User observes Newton picking familiar-looking identifiers over genuinely salient ones
- User observes Newton copying example values verbatim instead of substituting its own
- User observes Newton returning prose/markdown when they asked for JSON
- User's current prompt works sometimes but fails under specific states

## Endpoint

```
POST {ATAI_API_ENDPOINT}/v0.5/query
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

Latency is typically **3–6 s p50** for text-only queries with a Newton text checkpoint, **6–8 s** when an image is attached (vision pipeline engages), and **~1–2 s warmup** on the first call. Can be called directly from the browser if the server proxy adds latency (see "Direct Browser Calls" below).

## Input Modes

The same `/query` endpoint accepts four genuinely different input shapes. Picking the right one matters — not all combinations work, and some look like they should but don't (see *Inputs that look supported but aren't* below).

| Input | Recommended path | Also accepted | Avoid |
|---|---|---|---|
| Text (state snapshot, prompt context) | `query` field directly | `data.text` event | — |
| JSON content for the model to read | `file_ids` with the `.json` filename | inline in `query`, or `data.text` / `data.json` event with the JSON as a **string** | passing parsed objects to `data.json` (`contents` must be a string, not a dict) |
| Plain-text content (logs, notes) | `file_ids` with the `.txt` filename | inline in `query`, or `data.text` event | — |
| **CSV content** | inline in `query`, **or** `data.text` event with the CSV as a string, **or** upload with a `.txt` filename | — | `file_ids` of a `.csv` file — uploads successfully but the Newton text model does **not** see the contents (almost certainly routed to the numeric/Omega ingestion path, not the LLM prompt). Renaming the same bytes to `.txt` before upload fixes it. |
| Image (screenshot, chart, photo) | `file_ids` with the filename (e.g. `"dashboard.png"`) | `data.base64_img` event with inline base64 | `file_ids` with the `fil_...` UID — rejected as `unsupported_file_type` because the API filters by extension |
| Video (.mp4) | None reliable on Newton text checkpoints today | API accepts the file but `c2_4_7b` / `c2_5_8b` respond "I can't see videos" | Use [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md) instead |

### Request body — text + state

The canonical operator-suggestion shape:

```json
{
  "query": "<per-request state snapshot in natural language>",
  "system_prompt": "<SYSTEM_PROMPT with rules + output shape>",
  "instruction_prompt": "<same as system_prompt>",
  "file_ids": [],
  "model": "Newton::c2_5_8b_260413b723a9ab",
  "max_new_tokens": 700,
  "sanitize": false
}
```

### Request body — JSON / TXT content the model should read

JSON and TXT files **do** get their contents injected into the Newton text model's prompt context via `file_ids`. Three working paths in order of convenience:

**(a) Upload the JSON/TXT file and reference by filename in `file_ids`** (cleanest if the content is reused across queries):

```bash
# 1) Upload
curl -s -X POST "$ATAI_API_ENDPOINT/v0.5/files" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@plant_state.json;type=text/plain"
# Note: upload with mime=text/plain — application/json mime is rejected
# by a server-side validation bug despite being listed as supported.
# → { "is_valid": true, "file_id": "plant_state.json", "file_uid": "fil_..." }
```

```json
{
  "query": "What is the status field and which sensor has the highest value?",
  "file_ids": ["plant_state.json"],
  "model": "Newton::c2_5_8b_260413b723a9ab",
  "max_new_tokens": 250,
  "sanitize": false
}
```

Verified: Newton reads obscure keys correctly (e.g. `alpha_zorlon_cannon: 42.7` from a randomly-named JSON came back exactly).

**(b) Inline the JSON / text directly into `query`** (simplest for one-shot queries):

```json
{
  "query": "What is the status field?\n\n{\"plant\":\"P3\",\"status\":\"attack\",\"sensors\":{\"LIT301\":850}}",
  "model": "Newton::c2_5_8b_260413b723a9ab",
  "max_new_tokens": 200,
  "sanitize": false
}
```

**(c) Use a `data.text` event** (cleanest when you don't want to re-escape the JSON into the query string):

```json
{
  "query": "What is the status field?",
  "events": [
    { "type": "data.text", "event_data": { "contents": "<full JSON or text>" } }
  ],
  "model": "Newton::c2_5_8b_260413b723a9ab",
  "max_new_tokens": 200,
  "sanitize": false
}
```

There's also a typed `data.json` event variant — same effect, but `contents` must be a **string** (a serialized JSON document), not a parsed object:

```json
{
  "events": [
    {
      "type": "data.json",
      "event_data": { "contents": "{\"plant\":\"P3\",\"status\":\"attack\"}" }
    }
  ],
  "...": "..."
}
```

Passing a dict here returns `400: Parameter 'contents' should be a <class 'str'>. Got: <class 'dict'>`. Bound by `max_query_size_mb` server-side (default 0.04 MB combined prompt size); split into multiple queries if you exceed it.

### Request body — CSV content the model should read

`.csv` is the one extension where `file_ids` upload does **not** inject contents into the Newton text model's prompt. The file uploads cleanly (200 OK with a `file_id`), `/query` accepts the file_id reference, but the model responds as if no file is attached ("please upload the CSV file"). Verified by uploading the same bytes with `.txt` extension instead — `.txt` Newton reads correctly. The likely explanation is that `.csv` extension routes downstream to the numeric / Omega ingestion path the LLM doesn't observe.

Three workarounds that all work:

**(a) Rename to `.txt` before upload** (cheapest if you're not changing your storage layout):

```bash
curl -s -X POST "$ATAI_API_ENDPOINT/v0.5/files" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@plant_history.txt;type=text/plain"
# Same bytes that wouldn't be injected as .csv are injected as .txt.
```

**(b) Inline content directly into `query`** (simplest for small payloads):

```json
{
  "query": "Identify the attack rows and report their values:\n\ntimestamp,sensor,value,status\n1000,temp,72.1,normal\n1020,temp,98.7,attack\n1030,temp,99.1,attack",
  "model": "Newton::c2_5_8b_260413b723a9ab",
  "max_new_tokens": 250,
  "sanitize": false
}
```

**(c) Use a `data.text` event** with the CSV as a string:

```json
{
  "query": "Identify the attack rows and report their values.",
  "events": [
    { "type": "data.text", "event_data": { "contents": "timestamp,sensor,value,status\n1000,...\n1020,...,attack\n..." } }
  ],
  "model": "Newton::c2_5_8b_260413b723a9ab",
  "max_new_tokens": 250,
  "sanitize": false
}
```

### Request body — image input

Newton text checkpoints **do have working image vision**. Two paths:

**(a) Upload via `/v0.5/files`, then reference by filename in `file_ids`:**

```bash
# 1) Upload
curl -s -X POST "$ATAI_API_ENDPOINT/v0.5/files" \
  -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@dashboard.png;type=image/png"
# → { "is_valid": true, "file_id": "dashboard.png", "file_uid": "fil_..." }
```

```json
{
  "query": "Describe this dashboard. Identify any stages flagged as anomalous.",
  "file_ids": ["dashboard.png"],
  "model": "Newton::c2_5_8b_260413b723a9ab",
  "max_new_tokens": 400,
  "sanitize": false
}
```

**Use the `file_id` (filename), not the `file_uid` (`fil_...`).** The API validates file types by extension on the file_id string — `dashboard.png` matches `.png`, `fil_xxxxx` matches nothing and is rejected as `unsupported_file_type`.

**(b) Inline base64 via `data.base64_img` event** (when you don't want a separate upload):

```json
{
  "query": "Describe this.",
  "events": [
    {
      "type": "data.base64_img",
      "event_data": { "contents": "<base64 string>", "mime_type": "image/png" }
    }
  ],
  "model": "Newton::c2_5_8b_260413b723a9ab",
  "max_new_tokens": 300,
  "sanitize": false
}
```

Both paths produce comparable description quality. Use `file_ids` if the image is reused across many queries (one upload, many `/query` references); use `data.base64_img` for one-shot queries where the upload roundtrip is wasted.

When an image is attached, end-to-end latency jumps from text-only's ~3 s to ~6–8 s — that's the vision pipeline activating. The same model returns to text-only latency immediately when no image is attached.

### Parameter notes

- **`model`** — current Newton checkpoint ID for your account. The two text checkpoints we've tested are `Newton::c2_4_7b_251215a172f6d7` and `Newton::c2_5_8b_260413b723a9ab`; both accept images.
- **`max_new_tokens`** — size for your expected output, plus ~20% headroom. Newton truncates mid-sentence when it runs out of budget. Observed sane defaults: 300 for short structured output or image descriptions, 700 for a dozen JSON cards, 1500+ for long-form prose.
- **`max_frames`** — only relevant for `.mp4` `file_ids` (default 32). Documented in the API but Newton text checkpoints don't currently use the sampled frames.
- **`sanitize: false`** — **required for structured output**. `sanitize: true` rewrites punctuation (smart quotes, dashes) and breaks `JSON.parse` downstream.
- **`system_prompt` + `instruction_prompt`** — Newton's chat template uses `instruction_prompt` as the authoritative system turn; `system_prompt` is a legacy alias. Send the same string to both — cheapest insurance.
- **`file_ids: []`** — leave empty when the prompt is fully self-contained. Populate for images or for retrieval-style reference docs (uploaded text/CSV/JSON files are accepted but not injected into the prompt — see the *Inputs that look supported but aren't* section).
- **`multi_image: false`** — set to `true` when you want multiple `file_ids` or image events treated as one multi-image input rather than independent inputs (e.g. before/after pair, multi-view of the same object).
- **`events: []`** — inline data events as an alternative to file uploads. Valid types: `data.text`, `data.json`, `data.numeric_array`, `data.base64_img`, `data.base64_img_array`.

### Inputs that look supported but aren't (or have gotchas)

| What | What happens | What to do instead |
|---|---|---|
| `file_ids` with `.csv` filename | Upload returns 200 with a `file_id`. `/query` accepts the file_id. Newton responds "please upload the CSV" — contents are not injected. Same bytes uploaded as `.txt` are read correctly, so the discriminator is the `.csv` extension itself (probably routes to the numeric / Omega ingestion path). | Rename to `.txt` before upload, inline in `query`, or use a `data.text` event. |
| `file_ids` with the `fil_...` UID instead of the filename | `400 unsupported_file_type` — the API filters by extension on the file_id string, and the UID has no extension. | Use the `file_id` returned from the upload (i.e. the filename), not the `file_uid`. |
| Upload with `Content-Type: application/json` | `400 invalid_file_type: Unsupported file type: application/json` even though the same error's `suggestion` field lists `application/json` as a supported type. Server-side validation bug. | Upload with `Content-Type: text/plain` (the file extension is what the rest of the pipeline checks). |
| `data.json` event with `contents` set to a parsed object | `400: Parameter 'contents' should be a <class 'str'>. Got: <class 'dict'>` | Pass `contents` as a serialized JSON string, or just use `data.text` with the JSON text. |
| `file_ids` with `.mp4` | API accepts; `c2_4_7b` and `c2_5_8b` respond "I'm sorry, but as an AI language model, I don't have the capability to view videos." in ~2 s (consistent with frames never reaching the model). | Use [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md). |
| `data.base64_img_array` event with `contents: [<base64>, <base64>, ...]` | Schema accepts `contents` as a list but returns `400 event_payload_error: "The data could not be converted"`. Exact accepted payload shape is undocumented. | Send each image as a separate `data.base64_img` event (or upload them via `file_ids`) and set `multi_image: true` if you want them treated as one input. |

## The Five Prompt Patterns

### 1. Structured-output enforcement

Newton *will* wrap responses in markdown fences (` ```json ... ``` `), add preambles ("Here are the actions you should take:"), and sometimes explain itself afterwards. Three mitigations, in order of leverage:

**(a) Say it loudly in the system prompt:**

```
Return ONLY a JSON array. No prose, no markdown code fences, no explanation
— just the JSON. Shape: [{"field":"..."}]
```

**(b) Specify the exact shape with a type-like signature:**

```
Shape: [{"origin":"Pn","target":"Pm","direction":"upstream|local|downstream","text":"..."}]
```

Enumerated string unions (`"upstream|local|downstream"`) are honored reliably; free-form strings get creative.

**(c) Parse defensively.** Always strip fences and extract the outermost JSON before parsing:

```js
function parseStructured(text) {
  const cleaned = text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/i, '')
    .trim();
  const start = cleaned.indexOf('[');
  const end = cleaned.lastIndexOf(']');
  if (start === -1 || end === -1 || end <= start) return null;
  try {
    return JSON.parse(cleaned.slice(start, end + 1));
  } catch {
    return null;
  }
}
```

### 2. Constraint tables for routing / topology

If outputs need to map to a fixed graph (stage → neighbor, category → handler, zone → oncall), inline the full mapping in the system prompt as a table and repeat the rule:

```
TOPOLOGY — the "target" field MUST EXACTLY match these pairings.
Never substitute or reassign:
- P1 anomaly: upstream=none, local=P1, downstream=P2
- P2 anomaly: upstream=P1, local=P2, downstream=P3
- P3 anomaly: upstream=P2, local=P3, downstream=P4
- P4 anomaly: upstream=P3, local=P4, downstream=P5
- P5 anomaly: upstream=P4, local=P5, downstream=P6
- P6 anomaly: upstream=P5, local=P6, downstream=none

If a direction maps to "none" for a given origin, DO NOT emit that direction.
```

Then **validate server-side** after parsing. Newton occasionally routes cards wrong (e.g. P3 upstream → P1 instead of P2) no matter how forcefully you state the rule. Drop violators:

```js
const TOPOLOGY = {
  P1: { upstream: null, local: 'P1', downstream: 'P2' },
  /* ... */
};

const valid = parsed.filter((s) => {
  const topo = TOPOLOGY[s.origin];
  if (!topo) return false;
  const expected = topo[s.direction];
  if (expected === null) return false;
  return s.target === expected;
});
```

### 3. Avoid example contamination

Concrete identifiers in your prompt get copied verbatim by Newton. If your example says:

```
Example: {"sensor":"FIT101","value":0.00,"action":"check valve MV101"}
```

...Newton will cite `FIT101` and suggest checking `MV101` even when the actual anomaly is on `AIT402`. Two fixes:

**(a) Use placeholder names for shape examples:**

```
Example shape (use sensor name from state, not this placeholder):
{"sensor":"ZZZ000","value":0.00,"action":"check valve"}
```

**(b) Use generic Pn/XXn placeholders for category examples:**

```
{"origin":"PX","target":"PY","direction":"upstream",
 "text":"<sensor>=<value> — reduce feed from PY"}
```

Newton recognizes these as structural placeholders and doesn't paste them into output.

### 4. Pre-pick salient inputs server-side

Newton has strong priors toward familiar-looking identifiers. If your state has 40 sensors and you hand them all over with "pick the most anomalous," Newton will often cite whichever one has the most common-looking name (e.g., `FIT101`, `LIT101`) rather than the one actually deviating.

Pre-pick server-side. Rank by whatever signal defines "salient" in your domain (z-score against baseline, rate of change, rule score) and give Newton a single choice per group with an emphatic "cite this" instruction:

```js
function pickTopDeviation(sensors, baselines) {
  let best = null;
  for (const [col, val] of Object.entries(sensors)) {
    const b = baselines[col];
    if (!b) continue;
    const z = Math.abs((val - b.mean) / (b.std || 0.0001));
    if (!best || z > best.z) best = { col, val, z };
  }
  return best;
}

// In query body:
const top = pickTopDeviation(stageSensors, baselines);
line += `\n    cite this sensor: ${top.col}=${top.val.toFixed(2)} ` +
        `z=${top.z.toFixed(1)} (strong deviation)`;
```

And in the system prompt:

```
CRITICAL: use the sensor name+value specified after "cite this sensor:"
for that stage. Do NOT substitute a different sensor even if another
looks more familiar.
```

This single pattern was the biggest quality win in the SWaT demo — from "Newton cites FIT101 regardless of actual anomaly" to "Newton cites the top-z sensor every time."

### 5. Multi-part output with explicit separators

If each output item needs two or more distinct sub-fields (e.g., `citation + action`, `finding + severity + owner`), specify a separator and both halves:

```
Each "text" field is a full instruction with TWO parts joined by " — ":
  part 1: the EXACT sensor citation from that stage's "cite this sensor:" line
          (copy sensor name and value verbatim; drop the z annotation)
  part 2: an imperative operator action with a concrete verb
          (reduce, check, isolate, alert, hold, bypass)
```

Without the separator, Newton picks one half and drops the other. Concrete verb lists keep the "action" half from devolving into passive-voice observations.

## Response Shape

Newton's response body:

```json
{ "response": { "response": ["<string>"] } }
```

The string is your raw output. Defensive unwrapping (shape varies across model versions):

```js
let raw = '';
if (data.response?.response && Array.isArray(data.response.response)) {
  raw = data.response.response[0] || '';
} else if (Array.isArray(data.response)) {
  raw = data.response[0] || '';
} else if (typeof data.response === 'string') {
  raw = data.response;
} else if (data.text) {
  raw = data.text;
}
```

## Direct Browser Calls

If your server proxy adds serialization overhead or wedges on long-running `/query` calls, call Newton directly from the browser. Steps:

1. Ship a server-side endpoint that returns the API key + endpoint URL (the SWaT demo uses `GET /api/baselines`, which also returns precomputed per-sensor baselines for the prompt). This keeps the key out of the client bundle while letting the browser POST directly to Newton.
2. Fetch and cache the credentials on mount.
3. POST to `{endpoint}/v0.5/query` with `Authorization: Bearer <key>` from the browser directly. Fall back to the server route if credentials haven't loaded yet.

Observed in SWaT demo: server proxy path was hanging on `/query` calls for 90–150 s while a direct probe (Node → Newton, same payload) responded in 3.5 s. Going direct from the browser restored the fast path.

The same pattern works for image queries — the browser can attach `file_ids` (from a prior `/v0.5/files` upload) or `data.base64_img` events directly to Newton without any server proxying.

See [references/suggested-actions-prompt.md](references/suggested-actions-prompt.md) for a full worked example including the direct-call implementation.

## Debounce & Caching

`/query` calls are not free (latency + billing). If your query input is a function of upstream state that flaps frequently:

- **Hash the input** (e.g., sort the anomalous set into a signature string `"P1,P3,P4"`). Skip the call if it matches the last-queried signature.
- **Debounce** by 1–2s so rapid flapping collapses into a single call. Track in-flight state to avoid restarting the debounce while a previous call is still running.
- **Refetch on signature drift after the in-flight call settles**, not while it's still pending — preserves consistency if the anomaly set moves during a long-running query.

## Testing Prompts In Isolation

Before wiring up the full app, test your prompt with a standalone Node probe:

```js
// probe-query.js
import { readFileSync } from 'fs';

const env = Object.fromEntries(
  readFileSync('.env', 'utf-8')
    .split('\n')
    .map((l) => l.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/))
    .filter(Boolean)
    .map((m) => [m[1], m[2]])
);

const res = await fetch(`${env.ATAI_API_ENDPOINT}/v0.5/query`, {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${env.ATAI_API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: '...', system_prompt: '...', instruction_prompt: '...',
    file_ids: [], model: 'Newton::...', max_new_tokens: 700, sanitize: false
  })
});
console.log(await res.text());
```

Iterate on prompt text here, not in the browser, until you're getting the shape you want.

To probe image input, upload first and attach by filename:

```js
// Upload (multipart)
const buffer = readFileSync('dashboard.png');
const fd = new FormData();
fd.append('file', new Blob([buffer], { type: 'image/png' }), 'dashboard.png');
const up = await fetch(`${env.ATAI_API_ENDPOINT}/v0.5/files`, {
  method: 'POST',
  headers: { Authorization: `Bearer ${env.ATAI_API_KEY}` },
  body: fd
}).then((r) => r.json());

// Query (file_id = filename, NOT file_uid)
const res = await fetch(`${env.ATAI_API_ENDPOINT}/v0.5/query`, {
  method: 'POST',
  headers: { Authorization: `Bearer ${env.ATAI_API_KEY}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'Describe this dashboard.',
    file_ids: [up.file_id],
    model: 'Newton::c2_5_8b_260413b723a9ab',
    max_new_tokens: 400,
    sanitize: false
  })
});
```

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Output wrapped in ` ```json ... ``` ` | Newton defaulted to markdown | Fence-stripping in the parser (always do this regardless) |
| Cites a familiar-looking sensor instead of the actual anomaly | Newton followed priors over state | Pre-pick server-side, emphatic "cite this" rule |
| Copies your example values verbatim | Concrete identifiers in the prompt | Use placeholder names (`ZZZ000`, `PX`) in examples |
| Drops half the expected fields | No separator or missing multi-part spec | `part 1 — part 2` format with verb list |
| "Please upload the CSV file" when a `.csv` is attached via `file_ids` | `.csv` extension is the one text-ish file type that doesn't inject contents into the LLM prompt (likely routed to a numeric pipeline). `.json` and `.txt` via `file_ids` both work | Rename to `.txt` before upload, inline content in `query`, or use a `data.text` event — see *Input Modes* |
| `400 unsupported_file_type` when referencing an uploaded file by `fil_...` UID | API checks file extension on the file_id string | Use the `file_id` (filename) returned from the upload, not the `file_uid` |
| `400 invalid_file_type: Unsupported file type: application/json` | Upload-side mime validation bug — `application/json` is rejected even though the error message lists it as supported | Upload with `Content-Type: text/plain`; the rest of the pipeline keys off the file extension |
| "I can't view videos" on `.mp4` attachment | Newton text checkpoints (`c2_4_7b`, `c2_5_8b`) don't process video frames | Use [`newton-activity-monitor`](../newton-activity-monitor/SKILL.md) instead |
| Routes outputs to wrong target (e.g. upstream → P1 when it should be P2) | Topology implicit, not stated | Inline topology table + server-side validation |
| `JSON.parse` crashes on punctuation | `sanitize: true` rewrote quotes/dashes | Set `sanitize: false` |
| Output truncated mid-sentence | `max_new_tokens` too low | Bump to 1.2× your observed p95 output length |
| First call slow (5–10s), subsequent fast | Model warmup | Accept; pre-warm at startup with a no-op query if cold-start matters |

## Minimal Skeleton

```js
const SYSTEM_PROMPT = `<your rules + shape + constraints + examples with placeholders>`;

function buildQuery(state) {
  // Render `state` into structured text. Include "cite this" hints if pre-picking.
  return `Current state:\n${formatState(state)}\n\n<restate output instruction>`;
}

async function callNewton(state) {
  const res = await fetch(`${endpoint}/v0.5/query`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: buildQuery(state),
      system_prompt: SYSTEM_PROMPT,
      instruction_prompt: SYSTEM_PROMPT,
      file_ids: [],
      model: 'Newton::<your_checkpoint>',
      max_new_tokens: 700,
      sanitize: false
    })
  });
  const data = await res.json();
  const raw = data.response?.response?.[0] ?? '';
  const parsed = parseStructured(raw);
  if (!parsed) throw new Error(`Unparseable: ${raw.slice(0, 200)}`);
  return validateTopology(parsed);  // your server-side validator
}
```

## Building a frontend on top of this

If you're wrapping `/query` output in a React/Svelte/etc. UI — operator suggestion panel, structured-decision dashboard, AI-reasoning sidebar — **read [`DESIGN.md`](../../DESIGN.md) at the root of this repo before writing any CSS**. The Archetype design system (Tailwind v4 + `@archetypeai/ds-lib-tokens` + Geist sans/mono + OKLCH palette + dark-first) is the expected visual language for these demos. The [`SuggestedActions` panel in `archetypeai-swat-demo`](https://github.com/archetypeai/archetypeai-swat-demo) is the canonical reference for surfacing structured `/query` JSON in an operator-facing UI (per-action card with `good`/`warning`/`critical` Badge, mono numeric thresholds, sharp 2px radii); [`archetypeai-wifi-demo`](https://github.com/archetypeai/archetypeai-wifi-demo) shows the per-window verdict + reason pattern. Setting this up at the start is much cheaper than retrofitting later.
