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
  channel-first window), the per-channel 768-d output, the supported
  16–1024 window-length range, `normalize_input`, and the
  "joint multi-channel state" + KNN pattern. For cleaning / splitting /
  windowing raw sensor CSVs first, see `atai-newton-omega-model-data-prep`.
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

For preparing the raw sensor CSVs (timestamp regularity, gap-aware blocks, temporal-order train/test split, the joint-state feature matrix), see [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md).

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

**Recommended: one request per channel, fanned out in parallel.** Each request carries a **`data.numeric_array` event** with a single channel — `contents` is `[[/* one channel: w floats */]]`. No `file_ids`, no prompt. Per-channel requests cap the request/response payload size (a single request carrying many channels eventually exceeds REST payload limits and corrupts in transit), and a thread-pool / async fan-out keeps wall-clock close to a single call — see `embed()` in [`_common.py`](references/_common.py).

```json
{
  "query": "",
  "model": "OmegaEncoder::omega_embeddings_1_4",
  "normalize_input": false,
  "events": [
    { "type": "data.numeric_array", "event_data": { "contents": [[/* one channel: w floats */]] } }
  ]
}
```

(The API also accepts all channels in one request — `contents[channel][t]` — but don't, for the payload reason above. And **don't mix the two conventions in one downstream model**: per-channel and all-in-one embeddings of the same data differ slightly, ~2e-2 per coordinate measured, so a KNN library built one way can't score windows embedded the other way.)

## Response — one 768-d vector per channel

A single-channel request returns its 768-d vector **flat** in `response.response`; reassemble per-channel results in channel order client-side (a multi-channel request returns `[channels x 768]` instead — another shape difference between the two conventions).

```json
{
  "response": {
    "response": [0.45, -0.32, ...768...],
    "warning_messages": []
  }
}
```

- **Window length: 16–1024 timesteps.** The encoder is trained to handle signal lengths from 16 to 1024. Under the hood, sub-1024 windows are padded **and masked** — the mask tells the model which time samples are real and which are padding, so padding is ignored rather than read as signal. **Don't default to 1024**: pick the window length that matches your signal's dynamics — shorter windows are sometimes more appropriate and more performant. The edges (verified live): any length < 1024 returns an informational `warning_messages` entry per channel (*"Data length N is less than 1024, padding with zeros"* — not a quality alarm within the 16–1024 range); below 16 you get *"the model may not perform well"* (below the trained range); above 1024 the input is **truncated to the last 1024 points** (with a warning).
- **`normalize_input`** — when `true`, the encoder z-normalizes **each window independently**. That's usually *not* what you want for comparing windows: it erases cross-window amplitude (a low-flow and a high-flow window can look identical). For classification / anomaly / similarity, fit **one** per-channel scaler on your training pool, apply it to every window, and call with `normalize_input=false` (see the downstream pattern below). Reserve `normalize_input=true` for one-off single-window encodes where only within-window shape matters.
- **Per-channel embeddings.** Omega embeds each channel independently. To get one fingerprint for a multi-channel window, concatenate the per-channel vectors into a **joint multi-channel state** (e.g. 4 channels → a 3072-d feature), optionally L2-normalized. That single vector is what you feed to KNN / anomaly / projection.

See [`embed_query.py`](references/embed_query.py) for the basic call, window-length behavior across the 16–1024 range, and `normalize_input`.

## Downstream pattern — embeddings → KNN classification

This is what the managed batch pipeline does, reproduced client-side over `/query` embeddings:

0. **Prep + normalize once** — fit a per-channel scaler (mean/std) on the n-shot training pool and apply it to every window, calling `/query` with `normalize_input=false`. This is the data-prep / preflight convention; vet the shot files first with [`omega-1-4-preflight`](https://github.com/archetypeai/omega-1-4-preflight) and prep raw CSVs with [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md).
1. **Build an n-shot library** — embed several labelled windows per class (e.g. `healthy` / `degraded`), fold each window's per-channel vectors into one joint feature, L2-normalize.
2. **Classify** a new window the same way and take a majority vote over the *k* nearest library features (euclidean).

A few dozen labelled embeddings make a serviceable classifier — no training, no batch job. See [`classify_knn.py`](references/classify_knn.py): it builds the library from the labelled shot files and runs a **genuine held-out evaluation** against `bearing_inference.csv` (sensors only — **no `label` column**, and timestamps disjoint from the shot files, so leakage is impossible two ways over), scoring predictions against `bearing_labels.csv`.

It reports a full binary metrics report (`degraded` = positive), and `--window-size` accepts one or more lengths — each runs as a separate full eval (library and test windows re-windowed together at that length), with a comparison table at the end. On **1000 held-out windows** (500 healthy + 500 degraded) per length, the full sweep `--window-size 16 256 1024` scored:

```
 window  library  windows  accuracy  precision  recall     f1
-------------------------------------------------------------
     16       16     1000     0.671      0.858   0.410  0.555
    256       14     1000     0.973      0.949   1.000  0.974
   1024        8     1000     0.944      0.899   1.000  0.947
```

The shape of that table is the lesson. At 16, the window is shorter than the bearing defect's impulse signature, so recall collapses — most degraded windows genuinely contain no evidence at that zoom. At 256 the signature fits: every degraded window caught, and the fewest false alarms (27 healthy windows flagged). At 1024 recall holds but precision drops (56 false alarms) — the extra context dilutes the defect rather than sharpening it. **256 — a quarter of the data per window — beats 1024 on this dataset**, which is the empirical case for the window-length guidance above: don't default to 1024; sweep a few lengths and let held-out F1 pick.

Each leg embeds ~1000 windows — ~4000 per-channel API calls, ~26 min at the default 8-way window fan-out; use `--max-windows 50` for a fast preview per length.

## Latency

| Operation | Observed |
|-----------|---------:|
| One window embed (4 channels × 1024, per-channel parallel) | **~2 s** |
| n-shot library (8 windows) | ~8 embeds, a few seconds |
| Held-out eval (1000 windows, 8-way parallel) | **~26 min** (~4000 per-channel calls) |

Each embed is an independent stateless call, so `classify_knn.py` fans them out with `--workers`; raise it (or lower `--max-windows`) to trade throughput for budget/time.

## Common Pitfalls

- **Channel-first, not row-major.** `contents` is `[channels x timesteps]` (one list per sensor), not `[timesteps x channels]`. Transposing gives garbage embeddings.
- **Drop the timestamp column.** A `timestamp` parses as a number, so naive "use all numeric columns" includes it as a fake channel — and if your class files have disjoint time ranges, a downstream classifier can cheat on it. Exclude time columns (the helper drops `timestamp`/`time`/… by header name; pass your own `time_columns` set if your data names them differently).
- **Temporal contiguity matters.** Omega reads a window as an ordered series; randomly-sampled or gap-spanning rows produce meaningless embeddings. Sample contiguous blocks. (See `atai-newton-omega-model-data-prep` for gap-aware windowing.)
- **`OmegaEncoder::` prefix.** The model id is `OmegaEncoder::omega_embeddings_1_4`, not `Newton::...`.
- **One embedding convention per model.** Per-channel requests (recommended) and all-channels-in-one-request produce slightly different vectors (~2e-2 per coordinate) for the same data. Build the n-shot library and embed inference windows the same way, or KNN distances are subtly wrong with no error anywhere.
- **The "padding with zeros" warning is informational, not an error.** Any window in the trained 16–1024 range is handled natively (padding + mask). The warnings to act on are *"less than 16"* (below the trained range) and *"greater than 1024, truncating to the last 1024 points"* (you silently lose everything before the final 1024 samples — window the data yourself instead).

## Local Setup

```bash
# In a virtualenv (system pythons are often PEP 668 "externally managed"):
python3 -m venv .venv && source .venv/bin/activate
pip install -r skills/atai-newton-omega-model/references/requirements.txt

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
│   ├── embed_query.py        ← embedding basics (call, window lengths, normalize_input)
│   ├── classify_knn.py       ← embeddings → n-shot KNN classification (held-out eval)
│   ├── requirements.txt      ← runtime deps (pip install -r requirements.txt)
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
