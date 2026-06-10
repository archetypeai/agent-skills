# Worked Example: SWaT Suggested Actions

Full `/query` prompt from the [archetypeai-swat-demo](https://github.com/archetypeai/archetypeai-swat-demo) Suggested Actions panel. Turns per-stage anomaly flags into concrete upstream/local/downstream operator actions.

Every rule in this prompt came from a specific failure during development — the inline comments explain *why* each line is there.

## System prompt

```text
You are an operator assistant for the SWaT six-stage water treatment plant. Flow: P1 raw intake → P2 chemical dosing → P3 ultrafiltration → P4 UV dechlorination → P5 reverse osmosis → treated water. P6 backwash recycles P5 reject to clean P3 UF membranes.

For each stage marked ATTACK, suggest specific operator actions in three directions:
- upstream: reduce or hold feed from the stage immediately before the anomalous one
- local: check / isolate equipment on the anomalous stage itself
- downstream: alert or protect the stage immediately after the anomalous one from cascading effects

TOPOLOGY — the "target" field MUST EXACTLY match these pairings. Never substitute or reassign:
- P1 anomaly: upstream=none, local=P1, downstream=P2
- P2 anomaly: upstream=P1, local=P2, downstream=P3
- P3 anomaly: upstream=P2, local=P3, downstream=P4
- P4 anomaly: upstream=P3, local=P4, downstream=P5
- P5 anomaly: upstream=P4, local=P5, downstream=P6
- P6 anomaly: upstream=P5, local=P6, downstream=none

If a direction maps to "none" for a given origin, DO NOT emit that direction.

Return ONLY a JSON array. No prose, no markdown code fences, no explanation — just the JSON. Shape:
[{"origin":"Pn","target":"Pm","direction":"upstream|local|downstream","text":"..."}]

Rules:
- Only generate suggestions for stages marked ATTACK. Skip NORMAL, STANDBY, CLASSIFYING.
- For EVERY ATTACK stage emit all three direction cards (or two cards if topology says "none" for one direction). Do not skip a stage.
- Each "text" field is a full instruction with TWO parts joined by " — ":
    part 1: the EXACT sensor citation from that stage's "cite this sensor:" line (copy sensor name and value verbatim; drop the z annotation)
    part 2: an imperative operator action with a concrete verb (reduce, check, isolate, alert, hold, bypass)
- Example shape (do not copy sensor name — use the one from "cite this sensor:" for each stage):
    {"origin":"PX","target":"PX","direction":"local","text":"<sensor>=<value> — check valve and surrounding piping"}
    {"origin":"PX","target":"PY","direction":"upstream","text":"<sensor>=<value> — reduce feed from PY to stabilise intake"}
    {"origin":"PX","target":"PZ","direction":"downstream","text":"<sensor>=<value> — alert PZ of contamination risk"}
- CRITICAL: use the sensor name+value specified after "cite this sensor:" for that stage. Do NOT substitute a different sensor even if another looks more familiar.
- For upstream/downstream cards, the action text must name the TARGET stage by its Pn code.
- Keep each "text" field under 140 characters.
```

## Per-request query

Fresh per anomaly-set change. Pre-picks the top-z sensor per ATTACK stage:

```text
Current plant state. Each attack stage specifies the exact sensor to cite — do not substitute:
- P1 raw intake: ATTACK
    cite this sensor: FIT101=0.00 z=84.3 (strong deviation)
- P2 chemical dosing: NORMAL
- P3 ultrafiltration: ATTACK
    cite this sensor: LIT301=815.40 z=0.5 (weak signal)
- P4 UV dechlorination: NORMAL
- P5 reverse osmosis: NORMAL
- P6 backwash: STANDBY

Generate suggestions for ATTACK stages only. Emit three direction cards per attack stage (or two if topology says "none" for one direction).
```

## Request body

```json
{
  "query": "<query text above>",
  "system_prompt": "<system prompt above>",
  "instruction_prompt": "<same system prompt>",
  "file_ids": [],
  "model": "Newton::c2_4_7b_251215a172f6d7",
  "max_new_tokens": 700,
  "sanitize": false
}
```

## Example output

Raw response string (after unwrapping `data.response.response[0]`):

```json
[
  {"origin":"P1","target":"P1","direction":"local","text":"FIT101=0.00 — check MV101 intake valve, likely shut"},
  {"origin":"P1","target":"P2","direction":"downstream","text":"FIT101=0.00 — alert P2 dosing: no feed, hold chemical pumps"},
  {"origin":"P3","target":"P2","direction":"upstream","text":"LIT301=815.40 — hold feed from P2 while UF tank level settles"},
  {"origin":"P3","target":"P3","direction":"local","text":"LIT301=815.40 — inspect UF tank level transmitter"},
  {"origin":"P3","target":"P4","direction":"downstream","text":"LIT301=815.40 — alert P4 of reduced clarified flow"}
]
```

5 cards for 2 attack stages (P1 has no upstream by topology, so 2 cards; P3 has all three directions, so 3 cards).

## Why each rule is in the prompt

Every bullet in the system prompt came from a specific failure during development:

| Rule | Failure it fixes |
|---|---|
| Explicit TOPOLOGY table | Newton routed upstream cards to wrong target (P3 → P1 instead of P3 → P2) |
| `"none"` directions omitted | Phantom upstream cards for P1 and downstream cards for P6 with invented targets |
| "Return ONLY a JSON array" | Responses wrapped in ` ```json ``` ` fences + preamble prose |
| "TWO parts joined by ` — `" | Output was either pure citation (`LIT101=648.46`) or pure prose with no numbers |
| "cite this sensor:" pre-selection | Newton cited familiar-looking sensors (FIT101, LIT101) regardless of actual anomaly |
| Placeholder `PX/PY/PZ` in examples | Earlier concrete example (`FIT101=0.00 — isolate MV101`) got copy-pasted verbatim |
| `sanitize: false` | `sanitize: true` was rewriting the ` — ` em-dash and breaking the two-part format |
| 140-char cap on `text` | Without it, cards wrap ugly in the UI |
| `max_new_tokens: 700` (up from 500) | Final card of a 4-stage response was getting truncated mid-sentence |

## Topology validation (server-side)

Even with the explicit table, Newton occasionally routes cards wrong. Validate after parsing:

```js
const TOPOLOGY = {
  P1: { upstream: null, local: 'P1', downstream: 'P2' },
  P2: { upstream: 'P1', local: 'P2', downstream: 'P3' },
  P3: { upstream: 'P2', local: 'P3', downstream: 'P4' },
  P4: { upstream: 'P3', local: 'P4', downstream: 'P5' },
  P5: { upstream: 'P4', local: 'P5', downstream: 'P6' },
  P6: { upstream: 'P5', local: 'P6', downstream: null }
};

function validate(parsed) {
  return parsed.filter((s) => {
    if (!['upstream', 'local', 'downstream'].includes(s.direction)) return false;
    const topo = TOPOLOGY[s.origin];
    if (!topo) return false;
    const expected = topo[s.direction];
    if (expected === null) return false;
    return s.target === expected;
  });
}
```

Typical drop rate: 0–1 cards out of 10 on well-formed runs, 2–3 if the model is warming up.

## Full reference implementation

[src/lib/suggestions-direct.js](https://github.com/archetypeai/archetypeai-swat-demo/blob/main/src/lib/suggestions-direct.js) in the SWaT demo — includes the z-score ranking, direct browser call, response unwrapping, and topology validation in ~200 lines.
