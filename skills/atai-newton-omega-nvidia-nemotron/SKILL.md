---
name: atai-newton-omega-nvidia-nemotron
description: >
  Pair Archetype AI's Omega encoder (`OmegaEncoder::omega_embeddings_1_4`
  over `/query`) with NVIDIA Nemotron on NVIDIA's hosted API
  (`integrate.api.nvidia.com`) — Omega + client-side KNN decide what state a
  sensor window is in, Nemotron explains it in operator language. Use this
  skill when the user wants an LLM to turn Omega/KNN verdicts into fault
  briefs, suggested operator actions, or alert summaries with NVIDIA
  Nemotron. Covers the "Omega senses, Nemotron explains" split (Nemotron gets
  verdicts + window statistics, never embeddings), NVIDIA key types and
  model retirement, turning Nemotron 3's reasoning off, JSON-only prompts
  with validation, peer/baseline comparison, caching, and two worked demos
  (SWaT operator suggestions, wind-turbine fault briefs). For the Omega call
  itself see `atai-newton-omega-model`; for data prep see
  `atai-newton-omega-model-data-prep`. Do NOT use for Newton's own text
  reasoning (that's `atai-newton-fusion-model`).
---

# Newton Omega × NVIDIA Nemotron — Omega senses, Nemotron explains

Omega turns sensor windows into embeddings; a client-side KNN turns those into a verdict (`healthy` / `fault`, `normal` / `attack`, …). That verdict is accurate but mute. This skill adds **NVIDIA Nemotron** as the explanation layer: when Omega flags something, Nemotron reads the verdict and plain window statistics and writes what an operator needs — what changed, the likely cause, what to check.

```
sensor window ──► scale ──► Omega /query (one per channel) ──► KNN vs n-shot library ──► verdict + votes
                                                                                            │
                         window stats (+ healthy baseline or peer) ─────────────────────────┤
                                                                                            ▼
                                                          NVIDIA Nemotron (chat/completions, JSON only)
                                                                                            │
                                                                     validate ──► cache ──► UI
```

## When to Apply

- An Omega + KNN pipeline already produces verdicts and you want **operator-facing text**: fault briefs, suggested actions, alert summaries.
- The user specifically wants **NVIDIA Nemotron** (hosted on build.nvidia.com) as the reasoning model.

**Do not use this skill when:**
- You need the Omega call, windowing or the KNN library — that's [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md) (and [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md) before it). This skill assumes those work.
- You want Newton's own reasoning model on `/query` — that's [`atai-newton-fusion-model`](../atai-newton-fusion-model/SKILL.md).

## The split: what Nemotron sees

**Never send embeddings to Nemotron.** A 768-d vector means nothing to a language model. Send what Omega and KNN *concluded*, plus numbers a person could read off a dashboard:

| Send | Example |
|---|---|
| The verdict and the KNN vote | `{"verdict": "fault", "knn_votes": {"fault": 5}}` |
| Statistics for the flagged window, in real units | mean / min / max power, wind, pitch; RMS, peak, kurtosis per accelerometer |
| A reference to compare against | a healthy **baseline** (the healthy n-shot pool) or a healthy **peer** over the same hours |
| Asset context in one or two sentences | "Senvion MM82 turbines, 2,050 kW rated" |

The reference is what makes the brief specific. "Power is 0 kW" is ambiguous; "0 kW vs the peer's 904 kW in the same 9.5 m/s wind" is a fault.

## NVIDIA hosted API

```
POST https://integrate.api.nvidia.com/v1/chat/completions
Authorization: Bearer $NVIDIA_API_KEY
Content-Type: application/json
```

OpenAI-compatible, so any OpenAI client works with `base_url=https://integrate.api.nvidia.com/v1`. The reference scripts use only the standard library.

```json
{
  "model": "nvidia/nemotron-3-super-120b-a12b",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "temperature": 0.2,
  "max_tokens": 800,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

- **Key:** create one at https://build.nvidia.com/settings/api-keys (free NVIDIA Developer Program). An **NGC key** — the one used to pull NVIDIA containers — also starts with `nvapi-` but returns **403** on every hosted model.
- **Model:** `nvidia/nemotron-3-super-120b-a12b` is the default here (~0.4 s for a short reply with reasoning off, ~1–2 s for a brief). Override with `NEMOTRON_MODEL`.
- **Reasoning off:** Nemotron 3 switches its reasoning trace off through `chat_template_kwargs: {"enable_thinking": false}`. A `/no_think` line in the system prompt is **ignored** — measured: 42 reasoning tokens with `/no_think`, 0 with the template flag.
- **Models get retired.** `nvidia/nvidia-nemotron-nano-9b-v2` returns **410 Gone** since 2026-08-26. Some listed models also return **404** for a given account. Run `check_nemotron.py` before you build.
- **Free tier:** about 40 requests/minute, for development and prototyping. Cache (below) and you'll stay far under it.
- **Diagnostics that don't prove anything:** `GET /v1/models` needs no key, and a retired model returns 410 before the key is checked — neither tells you the key works. Only a successful chat call does.

| Status | Usual meaning |
|---|---|
| 401 / 403 | Wrong or wrong-type key (NGC key) |
| 404 | Model not available to your account |
| 410 | Model retired — pick a current one |
| 429 | Rate limited — cache and back off |

## Prompting for operator text

Patterns that held up across both demos:

1. **JSON only, with a fixed shape.** `Return ONLY a JSON object, no prose or code fences: {"observation": "...", "likely_cause": "...", "checks": [...]}`. Parse the first `{…}` / `[…]` out of the reply anyway (`parse_json`) — models occasionally add fences.
2. **Cite numbers.** Require the observation to quote values from the input, and say "round numbers for reading at a glance" — otherwise you get `904.49 kW`.
3. **Hypotheses, not diagnoses.** `likely_cause` must be phrased as "consistent with…" / "suggests…", and the UI should label it as a suggestion.
4. **Push past restating.** Without "don't just restate that it isn't generating", the cause comes back as *"turbine failed to generate"*.
5. **Domain reading hints are fair; answers aren't.** General hints such as *"pitch near 90° = blades feathered; normal grid frequency points away from a grid outage"* sharpened turbine briefs from "shutdown or failure" to "turbine-side trip (pitch, converter or safety chain)". Don't hint at the specific fault you expect.
6. **Hand it only the names that exist.** List the channels or equipment it may mention; then check its output against that list (below).

## Validate before anything reaches the UI

Nemotron output is a suggestion; treat it as untrusted input.

- **Shape:** every required field present and non-empty, list lengths capped (`validate_brief`).
- **Domain rules, in code:** in the SWaT demo, each card's `(origin, direction)` must map to the expected target stage, and an upstream/downstream card may cite the anomalous stage's readings but must not tell the operator to operate *its* valves or pumps — those belong on the local card. A one-line regex check caught the case the prompt alone didn't.
- **On failure:** show "Nemotron unavailable" for that item and keep the Omega verdict visible. The verdict never depends on Nemotron.

## Caching and latency

- **Cache per anomaly signature** — the set of flagged stages (`"P2,P3"`) or `(turbine, window_start)`. A repeat returns instantly, and the free-tier rate limit stops mattering.
- **Don't block the sensing loop.** Submit the Nemotron call to a thread pool / async task and attach the result to the alert when it arrives (the turbine demo streams a separate `nemotron_brief` SSE event keyed by turbine + window).
- **Server-side only.** Never ship `NVIDIA_API_KEY` (or `ATAI_API_KEY`) to the browser.

## Worked example — bearing sample

[`explain_knn.py`](references/explain_knn.py) runs the whole pattern on the bundled NASA IMS bearing data: 14-window KNN library (56 Omega calls), two held-out windows, a brief for the degraded one. Measured output:

```
window rows 0–255: healthy  votes={'healthy': 2, 'degraded': 1}
  healthy — no brief requested

window rows 256–511: degraded  votes={'degraded': 3}
  Nemotron brief (1.2 s):
  {
    "observation": "Bearing 1 shows elevated std (0.133 vs 0.074), rms (0.134 vs 0.074), peak (0.471 vs 0.361)…",
    "likely_cause": "Consistent with early-stage bearing degradation in bearing 1, indicated by increased vibration energy and impulsiveness.",
    "checks": ["Inspect bearing 1 for spalling or surface damage.", "Check lubrication and mounting of bearing 1.", "Verify accelerometer coupling on bearing 1."]
  }
```

Bearing 1 is the one that failed in IMS Set 2 (outer race) — Nemotron picked it from the statistics alone. Note the slip, too: it says "impulsiveness" although the crest factor *fell*. That is why the cause is framed as a hypothesis and the numbers are shown next to it.

## Worked demos

Two full apps built with this skill. Start from the key files — they're short and self-contained.

| Demo | Omega side | Nemotron side | Key files |
|---|---|---|---|
| [SWaT × NVIDIA Nemotron](https://github.com/archetypeai/archetypeai-swat-demo-nemotron) (SvelteKit) | Per-stage KNN: normal / attack for six plant stages | Upstream / local / downstream **operator action cards** as a JSON array; topology + equipment checks in code; cached per anomaly set | [`nemotron.js`](https://github.com/archetypeai/archetypeai-swat-demo-nemotron/blob/main/src/lib/server/nemotron.js) (client) · [`api/suggestions/+server.js`](https://github.com/archetypeai/archetypeai-swat-demo-nemotron/blob/main/src/routes/api/suggestions/+server.js) (prompt, validation, cache) |
| [Wind turbine × NVIDIA Nemotron](https://github.com/archetypeai/archetypeai-wind-turbine-demo-nemotron) (Python/Flask) | Per-window KNN: healthy / fault, WT01 vs WT09 | **Fault brief** comparing the flagged turbine with its healthy peer over the same hours; streamed as an SSE event. Correctly pointed to the pitch system for WT09's real 23 h pitch-circuit stop, without seeing the status log | [`nemotron_client.py`](https://github.com/archetypeai/archetypeai-wind-turbine-demo-nemotron/blob/main/nemotron_client.py) (stats, prompt, validation) · [`app.py`](https://github.com/archetypeai/archetypeai-wind-turbine-demo-nemotron/blob/main/app.py) (async brief jobs, SSE) |

Which pattern to copy:

- **Action cards (SWaT)** when the asset has a known structure (stages, lines, a topology) and the operator needs *what to do where*. Validate every card against that structure in code.
- **Fault brief (wind turbine)** when there's a healthy peer or baseline to compare with and the operator needs *what happened and what to check*.

## Local Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r skills/atai-newton-omega-nvidia-nemotron/references/requirements.txt

cp skills/atai-newton-omega-nvidia-nemotron/references/.env.example .env
# fill in ATAI_API_KEY, ATAI_API_ENDPOINT, NVIDIA_API_KEY

cd skills/atai-newton-omega-nvidia-nemotron/references
python check_nemotron.py   # key + model preflight (lists models, one call, reasoning tokens)
python explain_knn.py      # Omega + KNN + Nemotron brief on the bearing sample (~25 s)
```

## Common Pitfalls

- **Sending embeddings to the LLM.** It can't read them. Send verdicts and statistics.
- **NGC key instead of a build.nvidia.com key.** Both start with `nvapi-`; only the latter calls hosted models (403 otherwise).
- **`/no_think` on Nemotron 3.** Ignored — use `chat_template_kwargs: {"enable_thinking": false}`.
- **Hard-coding a model that later retires.** Keep the id in `NEMOTRON_MODEL` and run `check_nemotron.py` when calls start returning 410.
- **No reference to compare against.** Without a baseline or peer, briefs restate the window ("power is low") instead of explaining it.
- **Rendering unvalidated output.** Parse, check fields and domain rules, then render; show "unavailable" on failure.
- **Blocking the sensing loop on the LLM.** Run Nemotron off the hot path and attach results when they arrive.

## File Layout

```
skills/atai-newton-omega-nvidia-nemotron/
├── SKILL.md                  ← this file
├── references/
│   ├── _common.py            ← Omega embed helpers + Nemotron chat (stdlib), parse/validate, window stats, brief
│   ├── check_nemotron.py     ← preflight: list models, one call, latency + reasoning tokens, error hints
│   ├── explain_knn.py        ← Omega + KNN + Nemotron brief on the bearing sample
│   ├── requirements.txt
│   ├── .env.example
│   └── sample_data/
│       ├── bearing_healthy.csv     ← n-shot library: healthy
│       ├── bearing_degraded.csv    ← n-shot library: degraded
│       ├── bearing_incoming.csv    ← two held-out windows (healthy, degraded)
│       └── README.md               ← attribution + layout
└── tests/
    └── test_references.py    ← network-free unit tests (python -m pytest)
```
