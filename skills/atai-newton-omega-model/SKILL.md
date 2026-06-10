---
name: atai-newton-omega-model
description: >
  Get time-series embeddings from Archetype AI's Omega encoder
  (`OmegaEncoder::omega_embeddings_1_4`) over the prod `/query` endpoint —
  send a window of sensor readings, get back a fixed-size vector per
  channel, no batch job and no session. Use this skill when the user wants
  to embed multivariate sensor windows (vibration, flow, pressure,
  network, etc.) for lightweight downstream ML — KNN
  classification, anomaly scoring, similarity search, or 2D projection —
  done client-side over the embeddings. Covers the request shape (`data.numeric_array`,
  channel-first window), the per-channel 768-d output, the 1024-length
  native window and zero-padding behavior, `normalize_input`, and the
  "joint multi-channel state" + KNN pattern. For cleaning / splitting /
  windowing raw sensor CSVs first, see `newton-data-prep`.
  Do NOT use for text / image / video reasoning (that's the Newton fusion
  model on `/query`).
---

# Newton Omega Encoder — Time-Series Embeddings via `/query`

Omega is a **time-series encoder**: feed it a window of sensor readings, get back a fixed-size embedding you can do ML on. This skill calls the cloud Omega model on the same `/query` endpoint as the Newton fusion model — one stateless POST per window, embeddings back, no batch job or session lifecycle.

## When to Apply

- Embed multivariate sensor windows (vibration, pressure, flow, network, …) into vectors
- Build lightweight downstream ML over those vectors **client-side**: KNN classification, anomaly scoring, similarity search, PCA/UMAP projection
- Prototype classification without standing up the managed batch pipeline

**Do not use this skill when:**
- The input is text, an image, or a video — that's the Newton fusion model (`/query` with `Newton::c2_6_8b_fp8_...`)
- You need fully-managed, server-side classification over millions of rows

For preparing the raw sensor CSVs (timestamp regularity, gap-aware blocks, temporal-order train/test split, the joint-state feature matrix), see [`newton-data-prep`](../newton-data-prep/SKILL.md).

## The Model

```
OmegaEncoder::omega_embeddings_1_4
```

Note the `OmegaEncoder::` prefix (not `Newton::`). Output is a **768-dimensional embedding per channel**.

## Endpoint

```
POST {ATAI_API_ENDPOINT}/v0.5/query
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

Same `/query` endpoint as the fusion model; the model id selects the Omega encoder. **Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required** — there is no default endpoint, so a wrong-deployment mistake fails loudly at startup. Prod is `https://api.u1.archetypeai.app/v0.5`.

## Request Shape

The window goes in a **`data.numeric_array` event** as a **channel-first** matrix — `contents[channel][t]`, a list of `channels` lists each `window` long. No `file_ids`, no prompt.

```json
{
  "query": "",
  "model": "OmegaEncoder::omega_embeddings_1_4",
  "normalize_input": false,
  "events": [
    { "type": "data.numeric_array", "event_data": { "contents": [[/* channel 1: w floats */], [/* channel 2 */]] } }
  ]
}
```

## Response — one 768-d vector per channel

`response.response` is `[channels x 768]` — one embedding per input channel. A 4-channel window in → four 768-d vectors out.

```json
{
  "response": {
    "response": [[0.45, -0.32, ...768...], [...], [...], [...]],
    "warning_messages": []
  }
}
```

- **Window length = 1024.** That's the encoder's native receptive field. Feed 1024 timesteps per window. Shorter inputs still work but are **zero-padded server-side**, and you get a `warning_messages` entry per channel: *"Data length 256 is less than 1024, padding with zeros."* Padding dilutes the signal — prefer real 1024-length windows.
- **`normalize_input`** — when `true`, the encoder z-normalizes **each window independently**. That's usually *not* what you want for comparing windows: it erases cross-window amplitude (a low-flow and a high-flow window can look identical). For classification / anomaly / similarity, fit **one** per-channel scaler on your training pool, apply it to every window, and call with `normalize_input=false` (see the downstream pattern below). Reserve `normalize_input=true` for one-off single-window encodes where only within-window shape matters.
- **Per-channel embeddings.** Omega embeds each channel independently. To get one fingerprint for a multi-channel window, concatenate the per-channel vectors into a **joint multi-channel state** (e.g. 4 channels → a 3072-d feature), optionally L2-normalized. That single vector is what you feed to KNN / anomaly / projection.

See [`embed_query.py`](references/embed_query.py) for the basic call, the padding behavior, and `normalize_input`.

## Downstream pattern — embeddings → KNN classification

This is what the managed batch pipeline does, reproduced client-side over `/query` embeddings:

0. **Prep + normalize once** — fit a per-channel scaler (mean/std) on the n-shot training pool and apply it to every window, calling `/query` with `normalize_input=false`. This is the data-prep / preflight convention; vet the shot files first with [`omega-1-4-preflight`](https://github.com/archetypeai/omega-1-4-preflight) and prep raw CSVs with [`newton-data-prep`](../newton-data-prep/SKILL.md).
1. **Build an n-shot library** — embed several labelled windows per class (e.g. `healthy` / `degraded`), fold each window's per-channel vectors into one joint feature, L2-normalize.
2. **Classify** a new window the same way and take a majority vote over the *k* nearest library features (euclidean).

A few dozen labelled embeddings make a serviceable classifier — no training, no batch job. See [`classify_knn.py`](references/classify_knn.py): it builds the library from the labelled shot files and runs a **genuine held-out evaluation** against `bearing_inference.csv` (sensors only — **no `label` column**, and timestamps disjoint from the shot files, so leakage is impossible two ways over), scoring predictions against `bearing_labels.csv`.

It reports a full binary metrics report (`degraded` = positive). On **1000 held-out windows** (500 healthy + 500 degraded), the sample run scored:

```
accuracy : 0.946  (946/1000)
precision: 0.903   recall: 1.000   f1: 0.949
```

i.e. it caught every degraded window (recall 1.0) at the cost of 54 healthy windows flagged as degraded — a realistic operating point for an n-shot vibration classifier. The default run embeds ~1000 windows (~6–7 min, 8-way parallel); use `--max-windows 50` for a ~30 s check.

## Latency

| Operation | Observed |
|-----------|---------:|
| One window embed (4 channels × 1024) | **~1.1 s** |
| n-shot library (8 windows) | ~8 embeds, a few seconds |
| Held-out eval (1000 windows, 8-way parallel) | **~6–7 min** (~1000 embeds) |

Each embed is an independent stateless call, so `classify_knn.py` fans them out with `--workers`; raise it (or lower `--max-windows`) to trade throughput for budget/time.

## Common Pitfalls

- **Channel-first, not row-major.** `contents` is `[channels x timesteps]` (one list per sensor), not `[timesteps x channels]`. Transposing gives garbage embeddings.
- **Drop the timestamp column.** A `timestamp` parses as a number, so naive "use all numeric columns" includes it as a fake channel — and if your class files have disjoint time ranges, a downstream classifier can cheat on it. Exclude time columns (the helper drops `timestamp`/`time`/… by header name).
- **Temporal contiguity matters.** Omega reads a window as an ordered series; randomly-sampled or gap-spanning rows produce meaningless embeddings. Sample contiguous blocks. (See `newton-data-prep` for gap-aware windowing.)
- **`OmegaEncoder::` prefix.** The model id is `OmegaEncoder::omega_embeddings_1_4`, not `Newton::...`.
- **Sub-1024 windows are padded, not rejected.** Check `warning_messages` if you're surprised by weak embeddings on short inputs.

## Local Setup

```bash
pip install archetypeai python-dotenv numpy

# Drop a .env at the repo root (or anywhere up the tree from the run dir).
# BOTH variables are required — there is no default endpoint:
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://api.u1.archetypeai.app/v0.5
EOF

cd skills/atai-newton-omega-model/references
python embed_query.py     # embedding basics
python classify_knn.py    # n-shot KNN classification
```

The scripts are built on the [official Archetype AI python client](https://github.com/archetypeai/python-client) (`pip install archetypeai`): each script creates one client via `make_client()` and passes it into the helpers, and the `/query` POST goes through the client's retrying transport. They auto-load `.env` if `python-dotenv` is installed (`find_dotenv()` walks up from cwd); otherwise export `ATAI_API_KEY` and `ATAI_API_ENDPOINT`. `numpy` is required for `classify_knn.py`.

## File Layout

```
skills/atai-newton-omega-model/
├── SKILL.md                  ← this file
├── references/
│   ├── _common.py            ← official-client setup, the Omega embed call, CSV/window helpers
│   ├── embed_query.py        ← embedding basics (call, padding, normalize_input)
│   ├── classify_knn.py       ← embeddings → n-shot KNN classification (held-out eval)
│   ├── .env.example          ← copy to .env and fill in
│   └── sample_data/
│       ├── bearing_healthy.csv          ← n-shot library: healthy (4 channels)
│       ├── bearing_degraded.csv         ← n-shot library: degraded
│       ├── bearing_inference.csv        ← held-out test input, sensors only (~1000 windows, NO label column)
│       ├── bearing_labels.csv           ← ground-truth labels (timestamp,label), scoring only
│       └── README.md                    ← dataset attribution + held-out layout
└── tests/
    └── test_references.py    ← network-free unit tests (python -m unittest)
```
