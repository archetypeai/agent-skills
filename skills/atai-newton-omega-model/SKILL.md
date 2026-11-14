---
name: atai-newton-omega-model
description: >
  Get time-series embeddings from Archetype AI's Omega encoder (Newton Omega
  1.5, model `omega-embedding`) on Model Serving (`/api/inference/*`) — send a
  window of sensor readings, get back a 768-d vector per channel, no batch
  job. Use this skill when the user wants to embed multivariate sensor windows
  (vibration, flow, pressure, network, etc.) for lightweight downstream ML —
  KNN classification, anomaly scoring, similarity search, or 2D projection —
  done client-side over the embeddings. Covers sessions and `window_size`
  (16–1024), the request shape (`data.numeric_array`, channel-first, all
  channels in one query), the `[windows, channels x 768]` response, the
  shared-scaler convention, the request-size limit, and the "joint
  multi-channel state" + KNN pattern. For cleaning / splitting / windowing raw
  sensor CSVs first, see `atai-newton-omega-model-data-prep`. Do NOT use for
  text / image / video reasoning (that's the Newton fusion model on `/query`).
---

# Newton Omega Encoder — Time-Series Embeddings via Model Serving

Omega is a **time-series encoder**: feed it a window of sensor readings, get back a fixed-size embedding you can do ML on. This skill calls Newton Omega 1.5 on **Model Serving**: open a session with your window length, send windows to it, get embeddings back. No batch job.

## When to Apply

- Embed multivariate sensor windows (vibration, pressure, flow, network, …) into vectors
- Build lightweight downstream ML over those vectors **client-side**: KNN classification, anomaly scoring, similarity search, PCA/UMAP projection
- Prototype classification without standing up the managed batch pipeline

**Do not use this skill when:**
- The input is text, an image, or a video — that's the Newton fusion model (`/query` with `Newton::c2_6_8b_fp8_...`)
- You need fully-managed, server-side classification over millions of rows
- You want Omega embeddings from a deployed agent — batch sessions over many files, or a serving deployment that matches them — with the window set on the platform: that's [`atai-omega-encoder-agent`](../atai-omega-encoder-agent/SKILL.md)

For preparing the raw sensor CSVs (timestamp regularity, gap-aware blocks, temporal-order train/test split, the joint-state feature matrix), see [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md).

## The Model

```
omega-embedding
```

Newton Omega 1.5. Output is a **768-dimensional embedding per channel**.

**Coming from Omega 1.4** (`OmegaEncoder::omega_embeddings_1_4` on `/api/lens/query`)? The two models' embeddings are not comparable (cosine similarity about 0.03 for the same window), so rebuild every n-shot library with 1.5 — never mix vectors from the two.

## Endpoints

All under `{ATAI_API_ENDPOINT}` (e.g. `https://platform.u1.archetypeai.app/api`), with `Authorization: Bearer <API_KEY>` and `Content-Type: application/json`:

| Call | Request | Purpose |
|---|---|---|
| Open a session | `POST /inference/sessions` with `{"model": "omega-embedding", "values": {"window_size": N, "step_size": N}}` | Fix the window length (16–1024). Returns `{"id": ...}` |
| Query the session | `POST /inference/sessions/{id}/query` | Embed windows of length N |
| Cancel the session | `POST /inference/sessions/{id}/cancel` (empty body) | Free its place under the org's session limit |
| Query without a session | `POST /inference/query` | Window fixed at 1024 |

**Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required** — there is no default endpoint, so a wrong-deployment mistake fails loudly at startup.

## Request Shape — all channels in one query

```json
{
  "model": "omega-embedding",
  "query": "",
  "events": [
    { "type": "data.numeric_array", "event_data": { "contents": [[/* channel 1: N floats */], [/* channel 2 */]] } }
  ]
}
```

`contents` is **channel-first**: `[channels x timesteps]`. No `file_ids`, no prompt. The session query needs `model` in the body too.

**Send all of a window's channels in one query.** Omega 1.5 encodes each channel independently, so the vectors are the same whether channels travel together or apart (measured: largest difference 0.0000) — one query is simply fewer calls. The exception is very wide data: request bodies are capped at **2 MiB** (HTTP 413 above it), about 95 channels × 1024 timesteps at full float precision. Split wider windows into groups of channels and join the vectors; [`_common.py`](references/_common.py) does this at 64 channels per request.

## Response — `[windows, channels x 768]`

```json
{
  "response": {
    "success": true,
    "response": [[/* channel 1: 768 floats, then channel 2: 768 floats, ... */]]
  }
}
```

Each **row** is one window: the channels' 768-d vectors concatenated in channel order. Split a row into `channels` slices of 768.

## Window length — sessions

The session fixes the window. Every query sent to it is cut into `window_size`-timestep windows, advancing `step_size`, and each full window returns one row:

- **Exactly `window_size` timesteps** → one row. This is the normal case: slice your data yourself and send one window per query.
- **Longer** → back-to-back windows, one row each; timesteps past the last full window are **dropped**.
- **Shorter** → `success: false` with an empty response and **no error message**. Check `success`.
- **Without a session** (`/inference/query`), the window is always 1024.

The encoder is trained for **16–1024 timesteps**. Windows shorter than 1024 are padded **and masked** inside the model, so padding is ignored rather than read as signal. Pick the length that matches your signal's dynamics and **compare a few on held-out data** — the best length depends on the data *and* on how many channels you embed (see the table below).

**Session lifecycle.** A session ends when you cancel it, when it sits idle long enough, or when the server holding it is replaced; a query then returns HTTP 409 with code `session_not_running`. Open a new one and retry — `OmegaSession` in [`_common.py`](references/_common.py) does this. Each org can hold a limited number of open sessions per model (16 at the time of writing), so cancel sessions when you're done (`with OmegaSession(...)` does). Asking for an unknown session value returns HTTP 422 `invalid_session_values`.

## Normalization — one shared scaler

Omega 1.5 z-normalizes each channel **inside every window** and passes the window's overall level (mean and spread) to the model separately, so amplitude differences between windows survive. There is **no `normalize_input` option** to set.

Still fit **one** per-channel scaler (mean/std) on your n-shot pool and apply it to every window — library and inference alike. It keeps the level information on one reference frame. On a single bearing channel it added 7–10 points of held-out accuracy at windows of 64 and 256 (no difference at 1024).

**Per-channel embeddings.** To get one fingerprint for a multi-channel window, concatenate the per-channel vectors into a **joint multi-channel state** (e.g. 4 channels → a 3072-d feature), optionally L2-normalized. That single vector is what you feed to KNN / anomaly / projection.

See [`embed_query.py`](references/embed_query.py): the call without a session, sessions at 16 / 256 / 1024, what a session does with longer and shorter input, and the channel-independence check.

## Downstream pattern — embeddings → KNN classification

This is what the managed batch pipeline does, reproduced client-side:

0. **Prep + normalize once** — fit a per-channel scaler (mean/std) on the n-shot training pool and apply it to every window. Vet the shot files first with [`omega-1-5-preflight`](https://github.com/archetypeai/omega-1-5-preflight) and prep raw CSVs with [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md).
1. **Open a session** with the window length you chose, and use it for every window below.
2. **Build an n-shot library** — embed several labelled windows per class (e.g. `healthy` / `degraded`), fold each window's per-channel vectors into one joint feature, L2-normalize.
3. **Classify** a new window the same way and take a majority vote over the *k* nearest library features (euclidean).

A few dozen labelled embeddings make a serviceable classifier — no training, no batch job. See [`classify_knn.py`](references/classify_knn.py): it builds the library from the labelled shot files and runs a **genuine held-out evaluation** against `bearing_inference.csv` (sensors only — **no `label` column**, and timestamps disjoint from the shot files, so leakage is impossible two ways over), scoring predictions against `bearing_labels.csv`.

It reports a full binary metrics report (`degraded` = positive), and `--window-size` accepts one or more lengths — each runs as a separate full eval with its own session (library and test windows at that length), with a comparison table at the end. On **1000 held-out windows** per length, the sweep `--window-size 16 64 256 1024` scored:

```
 window  library  windows  accuracy  precision  recall     f1
-------------------------------------------------------------
     16       16     1000     0.734      0.794   0.632  0.704
     64       16     1000     0.954      0.987   0.920  0.952
    256       14     1000     0.970      0.947   0.996  0.971
   1024        8     1000     0.998      0.996   1.000  0.998
```

With all four channels in the joint feature, accuracy rises with the window: at 16 the window is shorter than the defect's impulse signature and recall collapses; at 1024 only 2 of 1000 windows are wrong. The answer is not universal, though — embedding **one** channel (`bearing_1` only, as in the platform quickstart) reverses it: 64 scored 94%, 256 85% and 1024 82% on 100 held-out windows, because the long single-channel windows flag healthy windows as degraded. Sweep a few lengths on *your* setup and let held-out F1 pick; the default here is 1024 because it wins on this one.

## Latency

Measured per window (p50, sequential), all channels in one query, 1024 timesteps:

| Channels | Latency | Channel embeddings/s with 8 queries in flight |
|---:|---:|---:|
| 1 | ~0.3 s | ~18 |
| 4 | ~0.5 s | ~35 |
| 16 | ~0.9 s | ~45 |

| Operation | Observed |
|-----------|---------:|
| n-shot library (8–16 windows) | a few seconds |
| Held-out eval (1000 windows × 4 channels, 8 in flight) | **~2 min** per window length |

Queries to one session are independent, so `classify_knn.py` sends several at once with `--workers`.

## Common Pitfalls

- **Channel-first, not row-major.** `contents` is `[channels x timesteps]` (one list per sensor), not `[timesteps x channels]`. Transposing gives garbage embeddings.
- **Drop the timestamp column.** A `timestamp` parses as a number, so naive "use all numeric columns" includes it as a fake channel — and if your class files have disjoint time ranges, a downstream classifier can cheat on it. Exclude time columns (the helper drops `timestamp`/`time`/… by header name; pass your own `time_columns` set if your data names them differently).
- **Temporal contiguity matters.** Omega reads a window as an ordered series; randomly-sampled or gap-spanning rows produce meaningless embeddings. Sample contiguous blocks. (See `atai-newton-omega-model-data-prep` for gap-aware windowing.)
- **Input shorter than the window fails silently.** The response is `success: false` with an empty list and no message. Check `success`, and slice every query to exactly `window_size` timesteps.
- **One window length per model.** The library and the inference windows must use the same session `window_size`; mixing lengths invalidates KNN distances.
- **No mixing of model versions.** Omega 1.4 and 1.5 vectors are not comparable. Rebuild the library when you switch.
- **Ended sessions return HTTP 409.** Reopen and retry rather than failing the run; cancel sessions you no longer need.
- **Very wide windows hit HTTP 413.** Above about 95 channels × 1024 timesteps, split channels across requests (the vectors are the same).

## Local Setup

```bash
# In a virtualenv (system pythons are often PEP 668 "externally managed"):
python3 -m venv .venv && source .venv/bin/activate
pip install -r skills/atai-newton-omega-model/references/requirements.txt

# Drop a .env at the repo root (or anywhere up the tree from the run dir).
# BOTH variables are required — there is no default endpoint:
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://platform.u1.archetypeai.app
EOF

cd skills/atai-newton-omega-model/references
python embed_query.py     # embedding basics
python classify_knn.py    # n-shot KNN classification
```

The scripts are built on the [official Archetype AI python client](https://github.com/archetypeai/python-client) (`pip install archetypeai`): each script creates one client via `make_client()`, opens an `OmegaSession`, and the POSTs go through the client's retrying transport. They auto-load `.env` if `python-dotenv` is installed (`find_dotenv()` walks up from cwd); otherwise export `ATAI_API_KEY` and `ATAI_API_ENDPOINT`. `numpy` is required for `classify_knn.py`.

## File Layout

```
skills/atai-newton-omega-model/
├── SKILL.md                  ← this file
├── references/
│   ├── _common.py            ← official-client setup, OmegaSession, embed_once, CSV/window helpers
│   ├── embed_query.py        ← embedding basics (no session, window lengths, session input rules, channels)
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
