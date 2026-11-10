---
name: atai-omega-encoder-agent
description: >
  Get Omega 1.5 time-series embeddings from Archetype AI's Omega Encoder agent (the
  canonical `omega-embeddings` blueprint) through Agent Lifecycle 2.0 (project, blueprint,
  package, deployment, session) with a developer key, in batch and in serving, and build your
  own anomaly detector on them. Batch: copy the blueprint with the window and step set as
  copy overrides, package it, deploy it as batch, open a session per role over its files,
  follow the paged results and match each window back to its file. Serving: deploy a package
  as serving, open a session whose `values` set the window, send one segment per `/query`,
  and check the answers against batch. Detector: client-side kNN on per-channel joint states,
  with the window size, channels, PCA, k and window matching chosen on validation, a threshold
  at a false-alarm rate on held-out normal data, one test, a delivery. Use when the user wants
  Omega embeddings over the Agents API (not the legacy Direct Query API), wants to choose the
  window, or wants a detector of their own on normal-only references with a few labelled
  faults. Covers the output schemas (batch: channels by name; serving: `channel_<i>`), the
  verified limits (batch sessions refuse `values`, a query shorter than its window is a 503,
  results paged at 100, partial results right after `completed`), and cleanup order. Do NOT
  use for the Direct Query API (`atai-newton-omega-model`), for platform-fitted detectors
  and classifiers (`atai-anomaly-discovery-agent`, `atai-operational-state-monitoring-agent`,
  `atai-rare-event-detection-agent`), or for turning recordings into role files
  (`atai-omega-encoder-agent-data-prep`).
---

# Omega Encoder Agent: Omega Embeddings over Agent Lifecycle 2.0

> **Preview: dev and staging** (verified with a developer key on `platform.dev.u1.archetypeai.app`,
> 2026-11-06, and `platform.stage.u1.archetypeai.app`, 2026-11-09, with identical results).
> Agent Lifecycle 2.0 and the `omega-embeddings` blueprint aren't on production yet.

The **Omega Encoder agent** is the canonical `omega-embeddings` blueprint: a records source,
a windowing node, the Omega 1.5 encoder and a JSONL sink. It answers one thing: for each
window of each channel, a **768-dimensional embedding**. It scores, classifies and checks
nothing. What you build on the embeddings is yours: here, an anomaly detector fitted on
normal operation only, the setting the [AD agent](../atai-anomaly-discovery-agent/SKILL.md)
covers on the platform, but with every choice (window, channels, distance, threshold) in your
hands and on your side of the API.

```
role files ─► batch: project → blueprint copy (window W) → package → batch deployment → session per role ─► embeddings
                                                                                                             │
                                          detect.py: kNN on joint states, chosen on validation, frozen ◄──────┘
                                                                                                             │
live segments ─► serving: package → serving deployment → session {values: window W} → /query per segment ─► same detector
```

It's the replacement path for the legacy Direct Query API
([`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md), Omega 1.4 on
`/api/lens/query`): the same kind of embeddings, with the window under your control and a
serving endpoint that matches batch.

Worked example, the whole way, with a live app:
[AURSAD screwdriving quickstart](https://github.com/archetypeai/omega-encoder-example-aursad-quickstart).

## When to Apply

- You want **Omega embeddings over the Agents API**: many files at once (batch), or one
  segment per request (serving).
- You want to **choose the window**: 1,024 is the blueprint's default; any size that divides
  your segments works, set per blueprint copy (batch) or per session (serving).
- You want **your own detector**, fitted on normal operation, with its settings chosen on a
  few labelled faults and its threshold set on held-out normal data.

**Do not use this skill when:**
- The platform should fit and serve the detector for you, with Optimizations, promote and
  Evals: [`atai-anomaly-discovery-agent`](../atai-anomaly-discovery-agent/SKILL.md) (normal-only),
  [`atai-operational-state-monitoring-agent`](../atai-operational-state-monitoring-agent/SKILL.md)
  (labelled states), [`atai-rare-event-detection-agent`](../atai-rare-event-detection-agent/SKILL.md)
  (a named fault).
- Your recordings aren't role files yet:
  [`atai-omega-encoder-agent-data-prep`](../atai-omega-encoder-agent-data-prep/SKILL.md) first.

## What a developer key can do (dev, measured 2026-11-02 and 11-06; staging, 2026-11-09)

| | |
|---|---|
| **Copy** `omega-embeddings` into a project | yes, with `overrides.values` (window, step, columns) |
| **Register** a new blueprint | no: admin only (403) |
| **Package, batch deployment, batch sessions** | yes. A batch session **refuses** `values` (`422 values_not_supported`): set the window on the copy |
| **Serving deployment, serving sessions, `/query`** | yes. A serving session **takes** `values`: `window_size`, `step_size` (and the source's columns) for that session only |
| **Optimizations, Evals** on `omega-embeddings` | no: it has no trainable node or metrics ("no metrics schema"); choose settings client-side, as below |

## Endpoints

Every call: `Authorization: Bearer <ATAI_API_KEY>`, under `{root}/api` (`ATAI_API_ENDPOINT` may
carry `/api` or not; `atai_api.api_base()` normalises it).

| step | call |
|---|---|
| project | `POST /api/projects` `{name}` (or find it in `GET /api/projects`) |
| blueprint copy | `POST /api/agents/blueprints/omega-embeddings/copy` `{project_id, blueprint_key, overrides: {values: {window_size, step_size}}}` |
| package | `POST /api/agents/packages` `{blueprint_ids: [id], target: {kind: "platform"}, name}`; poll until `ready` |
| deployment | `POST /api/agents/deployments` `{package_id, kind: "batch" \| "serving", sessions?: {max, idle_timeout_secs}}`; poll until `ready` |
| upload | `POST /api/files` (multipart); the file id is the file name |
| batch session | `POST /api/agents/sessions` `{deployment_id, connectors: {source: [{type: "file", id, format: "csv"}]}}`; poll until `completed` |
| results | `GET /api/agents/sessions/{id}/results?limit=…&after=…`: **paged** (100 by default, one output per input file); follow `next_cursor` |
| serving session | `POST /api/agents/sessions` `{deployment_id, values: {window_size, step_size}}`; poll until `running` |
| query | `POST /api/agents/sessions/{id}/query` `{query: "", events: [{type: "data.numeric_array", event_data: {contents: [[ch0…], [ch1…], …]}}]}` |
| end | `POST …/sessions/{id}/cancel`, `DELETE …/sessions/{id}`; `POST …/deployments/{id}/stop`, `DELETE …/deployments/{id}` |

## Output schemas

**Batch:** one JSONL row per window, channels **by column name**:

```json
{"start_timestamp": 21200.36, "finish_timestamp": 21202.91, "torque_current": [768 floats], "tcp_force_z": [768 floats], …}
```

The first column of the input file is the timestamp; every other numeric column is embedded.
Windows are matched back to files by `start_timestamp`: window k of a file starts k × W rows
after its first timestamp. That only works when no two files' time ranges overlap, which the
data-prep skill guarantees.

**Serving:** `{"response": {"data": [record, …]}}`, one record per window, channels **by
position**, `channel_0 … channel_N` in the order the lists were sent, 768 floats each. The
agent numbers the samples itself; a query carries no timestamps. Map `channel_i` back with
your channel list.

**Batch and serving agree:** on the same segments, cosine ≥ 0.9999 per window and channel,
and the same detector's alarms identical on every segment (AURSAD: 90/90).

## The pipeline

Role files come from the data-prep skill: `roles/manifest.json` (channels, step, rows per
segment, candidate windows, files) and `roles/labels/<role>.json`. Every script runs from a
directory with a `.env` (`ATAI_API_KEY`, `ATAI_API_ENDPOINT`), which **overrides** anything
exported in the shell; each prints the endpoint it uses first.

### 1. Library and validation, in batch, at each window size

```bash
python references/embed_batch.py --roles roles --role library validation --window 1024 512 256
```

For each window size: a copy `omega-encoder-w<W>` (window and step set at copy time), a
package, a batch deployment, then one session per role. Files are uploaded once and shared by
every window size. `embed_batch.py` follows the result pages, re-fetches until every window is
there, maps windows to files, and writes `out/embed/w<W>/<role>.npz`
(`emb [segments, windows, channels, 768]`).

**Expect** (the data-prep skill's sample: 88 AURSAD tightenings, 1,024 rows each; dev,
2026-11-06, and the same on staging, 2026-11-09), about a minute a session:

```
library @ 1024: 32 windows (32, 1, 12, 768) in 73s
validation @ 1024: 19 windows (19, 1, 12, 768) in 47s
library @ 512: 64 windows (32, 2, 12, 768) in 52s
…
validation @ 256: 76 windows (19, 4, 12, 768) in 49s
```

### 2. Choose the detector on validation, freeze it

```bash
python references/detect.py select --roles roles --groups groups.json
```

- **Joint state of a window:** each channel's 768-d embedding, L2-normalised; a channel
  group's channels concatenated; optionally PCA, fitted on the library's windows.
- **Window score:** mean distance to the k nearest library windows, either any of them
  (`match any`) or only those at the same position in their segment (`match position`,
  meaningful when segments are aligned on an event). **Segment score:** its worst window.
- **Selection:** every embedded window size × channel group (`--groups`, a JSON
  `{"name": [channels]}`; default one group of every channel) × PCA {16, 32, 64, none} ×
  k {1, 3, 5, 10} × match, by validation AUROC (normal against every fault).
- **Threshold:** 5% false alarms on **validation's normal segments**, data the library has
  not seen. Thresholds on the library's own leave-one-out scores come out too high.
- Frozen in `eval_out/settings.json`; nothing after this changes it.

**Expect** (the sample, with a `groups.json` of `screwdriver`, `screwdriver+tool` and `all`):

```
library 32 segments, validation 19; windows [1024, 512, 256]; 192 settings tried
  window  1024  screwdriver+tool     best AUROC 0.795  (PCA 16, k 1)
  window   256  screwdriver          best AUROC 0.807  (PCA 16, k 1, match position)
chosen and frozen: window 256, channels screwdriver, PCA 16, k 1, match position (validation AUROC 0.807); threshold 1.0486 = 5% false alarms on validation's normal segments
```

With 11 normal and 8 faulty validation segments, these numbers show the mechanics, not the
detector: a 5% false-alarm threshold on 11 normals is set by one segment. For real numbers
(105 validation segments, 443 in all), see the
[AURSAD quickstart](https://github.com/archetypeai/omega-encoder-example-aursad-quickstart#results-and-they-reproduce).

### 3. Test once, deliver, in batch at the frozen window

```bash
python references/embed_batch.py --roles roles --role test delivery --window frozen
python references/detect.py test --roles roles
python references/detect.py deliver --roles roles
```

**Expect** (the sample): test 0 of 8 faults caught, AUROC 0.64, no false alarms; delivery 1
of 5 caught, AUROC 0.92. Small samples; in the quickstart, missing screws are caught 93–100%
and damaged screws and extra components mostly missed.

### 4. Serve it, one segment per query

```bash
python references/serve.py --roles roles             # the delivery, in time order
python references/detect.py deliver --roles roles --source serving
```

A serving deployment of the frozen window's package, a session with `values` set to the
frozen window, and one `/query` per segment, scored at once and checked against the batch
embeddings of the same segments. The session and deployment are ended however the run ends.

**Expect** (the sample's delivery, 17 segments at window 256):

```
serving deployment dpl_… of pkg_… ready                          (~7 s)
serving session ses_… with values window_size=256, step_size=256: running
17 delivery segments served in 17 queries, window 256 (4 per segment); 1 alarm(s)
  latency per query: median 1551 ms, p95 1576 ms, max 1579 ms
  against batch: embeddings cosine min 0.999992 over 816 window-channels; alarm identical in 17/17; score max relative difference 1.6e-04
```

`detect.py deliver --source serving` then prints the batch delivery's table, number for number.

### 5. Clean up

```bash
python references/cleanup.py           # what would be deleted, and any serving deployment still up
python references/cleanup.py --yes
```

Sessions → deployments → packages → blueprints → project: a deployment must go before its
package and a package before its blueprint, or the delete is a 409. Uploaded files are kept.

## Verified platform behavior

| behavior | what to do |
|---|---|
| A batch session's `/results` is **paged at 100** items, one per input file | follow `next_cursor` (pass it back as `after`) |
| Right after `completed`, `/results` can briefly be **partial** | re-fetch until the window count is what the files imply |
| A batch session **refuses `values`** (422) | set window and step on the blueprint copy (`overrides.values`) |
| A serving session **takes `values`** for itself only | one serving deployment serves any window: a session per window size |
| A query **shorter than the session's window** answers **503** "deployment could not be reached" | send whole windows; make segments a multiple of every window you'll try |
| Rows past the last whole window are **dropped** | the same: whole windows |
| Serving channels are **`channel_<i>`**, in the order sent | map them back with your channel list |
| A serving session **closes after 30 idle minutes** (default `idle_timeout_secs: 1800`); on dev an idle serving deployment is **stopped after 20** | reopen a session, or a deployment; end both when done |
| Serving deployment ready in **~5 s**; a 1,024-row, 12-channel query answered in **~0.6 s** at window 1,024, **~1.3–1.5 s** at window 256 (4 windows) | |
| Batch: **~2 min** a session of 100–150 files (dev and staging); ~1 min for 20–30 | |
| A file id **is its file name**; uploading a name again replaces it | distinct names (the data-prep skill's are) |

## Pitfalls

- **Choosing on validation, then re-choosing after the test.** The test number is only honest
  once. The AURSAD quickstart's chosen setting (window 256, k 1, no PCA) won validation but its
  threshold let 13% false alarms through on test against the 5% target: reported, not re-tuned.
- **A flexible setting on a small validation set** (k 1, no PCA, 45 anomalies) wins partly by
  luck. Prefer more validation data to a wider grid.
- **Matching batch windows by start time across overlapping files** silently swaps windows
  between files. The data-prep preflight checks it; don't skip it.
- **Leaving a serving deployment up.** It holds a worker. `serve.py` ends it in a `finally`;
  `cleanup.py` names any still up.

## Local Setup

```bash
cd skills/atai-omega-encoder-agent/references
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas
cp .env.example .env                     # ATAI_API_KEY, ATAI_API_ENDPOINT (dev or staging)
```

Run the tests (offline: no key, no network):

```bash
pip install pytest
pytest skills/atai-omega-encoder-agent/tests/ -v
```

They check the kNN against brute force, that `select` picks the separating window and sets a
threshold on validation's normals, that batch outputs map back to files by start time (and fail
loudly when a window is missing), the serving query and answer shapes, and the endpoint
normalisation.

## File Layout

```
atai-omega-encoder-agent/
├── SKILL.md
├── references/
│   ├── atai_api.py          # stdlib client: requests, uploads, packages, deployments, sessions, polling
│   ├── embed_batch.py       # Steps 1 and 3: batch embeddings per role and window
│   ├── detect.py            # Steps 2-4: the kNN detector; select, test, deliver
│   ├── serve.py             # Step 4: serving, one segment per query, checked against batch
│   ├── cleanup.py           # Step 5
│   ├── requirements.txt
│   └── .env.example
└── tests/
    └── test_references.py
```
