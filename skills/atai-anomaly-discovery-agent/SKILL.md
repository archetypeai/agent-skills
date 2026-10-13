---
name: atai-anomaly-discovery-agent
description: >
  Run Archetype AI's Anomaly Discovery (AD) agent over the Agents API, on two
  paths. Path 1 — run the maintained "AD Quick Start" bundle (a fitted LOF
  detector and its threshold already pinned): upload a prepared sensor CSV,
  resolve the bundle by exact name, run, poll, download a per-window anomaly
  score. Path 2 — build a detector on YOUR OWN data: from role files (a
  normal-only library, validation and test with labelled incidents, a delivery)
  fit and score settings with the Optimizations API, promote the best trial to a
  blueprint, test it once with the Evals API, deliver it as a bundle and score
  detection, delay and false alarms. Use when the user wants to flag "this no
  longer looks like normal operation" from **normal-only reference data** — no
  fault library, no labelled examples of the thing being detected. Covers the
  output schema, the validation settings that silently invalidate every window
  above 1 kHz, what the `ad` blueprint exposes, thresholds as operating points,
  and scoring by detection, delay and false-alarm rate. Do NOT use for
  labelled operating regimes (`atai-operational-state-monitoring-agent`), a
  *named* recurring fault (`atai-rare-event-detection-agent`), client-side
  embeddings over `/query` (`atai-newton-omega-model`), or raw-CSV prep
  (`atai-newton-omega-model-data-prep`).
---

# AD Agent — Managed Anomaly Discovery via the Agents API

The AD agent answers one question: **does this window still look like normal
operation?** It is fitted on normal data only — a reference period from a
healthy asset — and everything it flags afterwards is, by construction,
something it was never shown.

That premise is the whole reason the agent exists. Its two siblings both need
examples of what you are looking for: OSM needs a labelled library of every
state, RED needs a handful of shots of a named fault. Neither is available when
a machine has never failed, which is the normal condition of most industrial
assets — and the condition under which a monitoring system is most valuable.

The graph is the first canonical blueprint with a **forked** topology:

```
source → interpolate → window → windowInterpolate → samplingRate → limitValues
       → tee ─┬→ encoder (omega:1.5) ─────→ fuse.emb ─┐
              └→ features (ChannelFeatures) → fuse.feat ─┴→ detector (LOF) → sink
```

Both branches see the same window: the Omega encoder produces a 768-dimensional
embedding, `ChannelFeatures` produces a handful of per-channel statistics, and
`ConcatColumnsNode` fuses them before the detector head. That fork has
consequences documented under **Verified platform behavior** — it is why long
inputs abort and why only one feature mode is reachable.

## When to Apply

- Flag departures from normal on an asset with **no fault history** — nothing
  has broken yet, so no labelled example of the failure can exist
- Monitor an asset whose failure modes are **unknown or unenumerable**, where a
  named catalog would be a guess
- Get a **continuous score** rather than a class — "how far from normal", not
  "which of these six states"
- Deploy per-asset detectors as repeatable batch jobs with **no client-side ML**
- Score a detector honestly when the data has no per-window ground truth, which
  is the usual case for run-to-failure data

**Do not use this skill when:**
- The user has a labelled library across all regimes and wants "which state is
  the asset in?" — use
  [`atai-operational-state-monitoring-agent`](../atai-operational-state-monitoring-agent/SKILL.md)
- The fault is **named** and a couple of labelled incidents exist — use
  [`atai-rare-event-detection-agent`](../atai-rare-event-detection-agent/SKILL.md).
  AD will flag it, but it cannot tell you *which* fault it is
- You want per-window embeddings to do ML client-side — use
  [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)
- Your recordings still need gridding, labelling or splitting into role files — use
  [`atai-anomaly-discovery-agent-data-prep`](../atai-anomaly-discovery-agent-data-prep/SKILL.md)
  first; for generic cleaning of messy raw data,
  [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md).
  AD assumes prepared input with a documented sampling rate

## Choose a path

| | **Path 1 — run a maintained bundle** | **Path 2 — build a detector on your data** |
|---|---|---|
| you have | a prepared CSV (the bearing sample, or data like it) | normal-operation recordings of **your** asset, and a few labelled incidents |
| you get | per-window scores from a detector fitted on **one bearing's** healthy baseline | your own blueprint and bundle, fitted on your normal data and chosen on your incidents, with a test score and a delivery score |
| APIs | files, bundles, instances | files, **Optimizations**, **promote**, **Evals**, bundles, instances |
| scripts | [`references/run_ad_agent.py`](references/run_ad_agent.py), on the official `archetypeai` client | [`references/ad_lifecycle/`](references/ad_lifecycle/), a small stdlib HTTP helper (the client doesn't cover Optimizations / Evals / promote yet) |
| time | ~30 s–1 min on a clear queue | ~30 min on the sample; longer with the dataset and the platform's load |
| status | verified on production; re-verified on dev 2026-10-13 | **verified on dev, 2026-10-13** (AD support in these APIs landed there on 2026-10-12) |

The Quick Start detector encodes one bearing's notion of normal, so running your own
asset through it is a transfer test, not a deployment: for your own asset, take Path 2.
Path 2's last step *is* Path 1 run on your own bundle, so the output schema, polling and
pitfalls below apply to both. Worked example:
[anomaly-discovery-agent-example-tep-quickstart](https://github.com/archetypeai/anomaly-discovery-agent-example-tep-quickstart)
(the Tennessee Eastman Process, about 30 minutes end to end).

---

# Path 1 — Run a maintained bundle

## Endpoints (both paths)

```
Files API   POST {ATAI_API_ENDPOINT}/v0.5/files                  (multipart upload)
            GET  {ATAI_API_ENDPOINT}/v0.5/files/download/{name}  (download)
Agents API   {ATAI_API_ENDPOINT}/agents/...                       (versionless!)
Authorization: Bearer <API_KEY> on every call
```

The Agents API is **versionless** — `/agents`, not `/v0.5/agents`. If
`ATAI_API_ENDPOINT` carries a `/vX.Y` suffix, strip it before appending
`/agents`. Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required; there is no
default endpoint.

> **⚠️ The bundle API is plural everywhere** as of 2026-08-11:
> `GET /agents/bundles`, `GET /agents/bundles/{id}`, `POST /agents/bundles`,
> `POST /agents/bundles/{id}/run`. The singular forms return **404**.

> **Finding a bundle's runs.** There is no `/agents/bundles/{id}/runs` (it
> 404s). Use the **List Agents** endpoint, `GET /agents/instances`, which
> filters **server-side**: `bundle_id` and `blueprint_id` are exact-match,
> `status` restricts to one lifecycle state, and `query` is a case-insensitive
> substring search over agent name, agent id, bundle id, and blueprint key.
> It pages with `limit` (default 100, max 1000) and `after`/`before` cursors,
> returning `data`, `has_more`, `next_cursor`.
> **The cursor is opaque** — pass `next_cursor` back verbatim; never derive it
> from `data[last].id`. Capturing the `agt_…` id from the run response is
> still the cheapest path when you control the run.

> **Availability.** The pre-packaged "AD Quick Start" bundles are published
> on the production deployment (`https://api.u1.archetypeai.app`) — set
> `ATAI_API_ENDPOINT` to it and the full upload → run → score cycle works as
> documented here. If name resolution reports no match, the bundle isn't
> published in the deployment you're pointed at: resolving by name is
> portable, so pass a known `--bundle-id` meanwhile, or contact
> support@archetypeai.dev.

## The five-step lifecycle

### 1. Upload the input CSV

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@bearing_eval.csv;type=text/csv" \
  "$ATAI_API_ENDPOINT/v0.5/files"
```

The response carries a `file_id`, which the platform sets to the **filename**.

> **⚠️ Re-uploading a filename can kill a run already using it.** File ids
> *are* filenames, so a second upload of the same name replaces the record and
> orphans the object an in-flight run already resolved. That run then dies with
> an error naming a UUID and nothing else:
> `Object not found: files_service/archetypeai/2d579dbe-…`. Upload under
> timestamped names when anyone else might be running.

### 2. Resolve the pre-packaged bundle by name

Two canonical bundles are published. **Names are the stable handles** — the
`bnd_…` ids are deployment-specific, so resolve by name:

| Name | What you get |
|---|---|
| `AD Quick Start (Bearing Breakdown)` | per-window `anomaly_score` + `predicted_label` |
| `AD Quick Start (Bearing Breakdown, Embeddings)` | the above **plus** the **Newton Omega encoder embedding for each window** — one `embedding_{variate}` column per channel and one `embedding_{variate}_features` column, 768-d vectors (the same embeddings [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md) gets from `/query`, computed server-side as part of the run). **The output file gets much larger**: 2.0 MB vs 10 KB on the bundled slice, ~200× with one channel |

Both pin the same detector — fitted on one bearing's healthy baseline, with
its validated threshold (1.762) and the input-validation settings the data
needs (`validate_monotonic_timestamps: false`,
`sample_rate_interval_tolerance: null`) already set. Nothing to configure.

```sh
curl -H "Authorization: Bearer $ATAI_API_KEY" \
  "$ATAI_API_ENDPOINT/agents/bundles?query=AD%20Quick%20Start&limit=100"
```

`?query=` is a case-insensitive **substring** search over name *and* id
(`?name=`/`?search=` are silently ignored), and the base name is a substring
of the Embeddings name — so a query for the base name returns *both*.
**Select the exact name match**, preferring `is_canonical: true`, and take its
`id`. Measured on Prod: `query 'AD Quick Start'` returns **2** with Embeddings
first (results are newest-first), while the full name including its closing
paren returns **1**. Taking `data[0]` on a prefix runs the Embeddings bundle.

### 3. Run the bundle — one agent per input file

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundles/$BUNDLE_ID/run" -d '{
    "connectors": {"source": [{"type": "file", "id": "bearing_eval.csv"}]}
  }'
```

Each run is a new agent (`agt_…`). **Capture that id** — it saves a lookup
later (see "Finding a bundle's runs" above). Reuse one bundle across every
input of the same asset; the detector does not change between files.

### 4. Poll until terminal — and do not trust `status`

```sh
curl -H "Authorization: Bearer $ATAI_API_KEY" \
  "$ATAI_API_ENDPOINT/agents/instances/$AGENT_ID"
curl -H "Authorization: Bearer $ATAI_API_KEY" \
  "$ATAI_API_ENDPOINT/agents/instances/$AGENT_ID/logs"
```

> **⚠️ `status` is not a reliable terminal signal — `/logs` is.** Pods have
> terminated with `Error (exit=1)` while `status` still read `running`, hours
> later. Treat an `error`-level or `pod.terminated` log event as terminal
> regardless of status, or a client polls to its timeout. Note **`/logs`**, not
> `/events` — the latter carries only lifecycle info.

### 5. Download the results

```sh
curl -H "Authorization: Bearer $ATAI_API_KEY" \
  "$ATAI_API_ENDPOINT/agents/instances/$AGENT_ID/results"
```

**A `completed` run can still be writing its output** (a known platform issue): a download
within seconds can return a partial CSV, its rows cut off before the input's end. The runner
counts the output complete only once its last window's `finish_timestamp` reaches the input's
last timestamp (within one window span and a minute), downloading again every 15 s until it
does, for up to 10 min.

`/results` pages like the listing endpoints — `data`, `has_more`,
`next_cursor`, with `limit` (default 100, max 1000) and `after`/`before`, and
the **same opaque-cursor rule**: pass `next_cursor` back verbatim, never
derive it from `data[last].id`. One run through a quick-start bundle produces
one output, so the first page is the whole answer; a bundle with several sink
ports is where paging starts to matter.

Each entry nests its fields under an inner `data` object — `filename`,
`num_bytes`, `ref` — not at the top level.

One row per window:

```
finish_timestamp,start_timestamp,predicted_label,invalid,anomaly_score
0.05,0.0,normal,false,1.0234712
```

- **`anomaly_score`** is the LOF score. **1.0 means "as dense as its
  neighbours"** — see below. Present when `output_score: true`.
- **`predicted_label`** is `anomaly_label` when the score exceeds the threshold,
  else `normal_label`. It is threshold-derived, so it adds no information the
  score lacks.
- **`invalid`** arrives as the **string** `"false"`, not a boolean.
- With `output_embeddings: true` (the Embeddings quick-start bundle) there is
  one `embedding_{variate}` column per channel at 768-d, **plus** an
  `embedding_{variate}_features` column carrying the `ChannelFeatures` output —
  one per branch of the `tee`. Filter by name and check the length; matching
  any key beginning `embedding` will hand you the short feature vector as if it
  were an embedding. Verified on the bundled slice: **2.0 MB vs the base
  bundle's 10 KB** (~200× with one channel), **240/240 label agreement** with
  the base bundle and a max score difference of **4.0×10⁻⁵** — at the
  platform's noise floor, i.e. the two bundles score the same detector.

**Verify the run before trusting it:** `/results` must be non-empty, and
`invalid` must not be `"true"` on every row. Both failure modes report
`completed`.

## How the score behaves, and what the threshold is

The head is **Local Outlier Factor** in novelty mode. LOF asks whether a point
sits in a sparser neighbourhood than its neighbours do — it is a **ratio of
densities**, not a distance:

```
LOF(p) = mean( local density of p's k nearest reference points ) / local density of p
```

which is why **1.0 is the neutral value** and why real scores hover just above
it. Measured on one fitted detector, scoring its own reference set: median
**1.043**, 99th percentile **1.397**, worst reference window **2.156**. A
threshold of 1.762 was crossed by **0.09%** of the reference data, and a
late-life failure reached **17**.

Two consequences for anyone reading the output:

- **The useful range is narrow and the tail is long.** Plot it on a log axis;
  a linear one puts every value that matters in the bottom tenth.
- **The threshold is an operating point, not part of the detector.** The score
  does not depend on it; only `predicted_label` does. A bundle's `threshold` value
  overrides its blueprint's: the same detector then labels at that threshold,
  verified on every window (the IMS example delivered one detector at 1.979 per
  installation where it was trained at 1.568). Choose it on held-out normal data,
  e.g. a target false-alarm rate, and report it with the results; Path 2's scorer
  reports at the trial's threshold and at a 1% false-alarm one.

LOF suits this problem because it assumes nothing about the shape of normal — no
Gaussian, no single cluster — and needs no anomalies to fit.

## One detector, many assets: what carries over

A detector encodes **the normal it was fitted on**. Running it on another asset
or installation is a transfer test, and the scores are not comparable to the
ones it was validated on: on the IMS bearings, healthy bearings of another
installation scored a median 1.23–1.38 under the first installation's detector
(against about 1.05 under their own), close to its 1.568 threshold, and raised
667 false crossings where their own detector raised 214.

Two fixes, both measured there: **fit a detector per installation** on its own
first normal hours (Path 2's Optimize with every setting pinned, then promote),
or **keep one detector and set each installation's threshold** from the highest
score over its first normal hours (a bundle value). The first is the robust one;
the second needs no fit but only works while that threshold clears the new
installation's healthy floor. Where the assets share one normal (one plant, many
runs, as in the TEP examples), one detector is the right design. Pooling
genuinely different machines into one library inflates "normal" until little
looks abnormal.

## Bring your own detector (legacy route)

**Path 2 replaces this:** Optimize fits the detector on your data and promote
turns it into a blueprint, with no artifact to handle. For completeness: a bundle
can also be created from the canonical `ad` blueprint around an externally fitted
detector artifact, then run as in step 3:

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" \
  -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundles" -d '{
    "blueprint": "ad",
    "name": "AD bearing outer race",
    "values": {"threshold": 1.762,
               "validate_monotonic_timestamps": false,
               "sample_rate_interval_tolerance": null,
               "max_temporal_gap": 60.0,
               "output_score": true},
    "artifacts": {"ad-detector": "s3://…/fit-detector.safetensors"}
  }'
```

Three values matter more than the rest:

- **`ad-detector` must be an `s3://` URI — pass the one you receive
  verbatim.** The platform resolves artifact strings as filesystem/S3 paths
  only, so a platform file id or an `https://` URL fails with ENOENT without
  attempting a fetch (and there is no upload route — the files API rejects
  safetensors by MIME type). **A wrong artifact key is accepted at creation**
  (HTTP 201, `status: ready`) and only fails ~30 s into the run, with an error
  naming the job poller rather than the artifact. The key is `ad-detector`; a
  second blueprint `ada` exists with key `ada-detector`.
- **`validate_monotonic_timestamps: false` is required above 1 kHz.**
  Timestamps are stored as `u64` **milliseconds**, so above 1 kHz consecutive
  samples share a millisecond and no window is ever strictly increasing. Left at
  the blueprint default of `true`, **every window is marked invalid, and the run
  still reports `completed`** — the tell is an empty `/results`, not the
  runtime.
- **`sample_rate_interval_tolerance: null`** for burst-sampled data. If the
  within-burst interval and the between-burst interval differ by orders of
  magnitude, no single tolerance describes both.

**Pin `window_size`, `step_size` and `feature_names` to the values the detector
was fitted with.** Since the 2026-10-12 `ad` re-seed the blueprint's own defaults
are window 1024, step 256 and seven features. On Dev that day, a bundle around the
IMS bearings detector that pinned none of them died once data flowed, with only
"This agent run failed because of an internal error" in its logs, and ran
correctly once they were pinned. The Quick Start bundles, which pin none of them,
ran correctly on Dev on 2026-10-13, so the detector's saved parameters may still
apply in some cases; pinning the fit's values is harmless either way.

## Scoring: lead time and false-alarm rate, not precision and recall

Run-to-failure data has **no per-window ground truth**. The binary labels in
common circulation are a cut someone placed by hand partway along a gradual
degradation curve, so precision and recall computed against them largely measure
where that line was drawn.

Score these four things instead:

1. **Detected or missed**, per asset, with **lead time** — operating hours from
   detection to the end of the recording, on an operating-time axis that
   excludes periods when the machine was stopped. Wall-clock time is not machine
   life: on one experiment here it overstated bearing life by 59%.
2. **False alarms** on assets that never failed. No true positive is possible
   there, so every crossing is a false alarm and the denominator is unambiguous.
   **Exclude the reference region** — scoring a detector on its own fit window
   flatters it.
3. **Per failure mode**, marking which modes were unseen when the configuration
   was chosen. A mode that informed the hyperparameter search is not held out,
   even though no fault data ever entered the fit.
4. **Window-level precision/recall across a *range* of candidate onset cuts**,
   reported as a range and never as a single number.

> **Require a sustained crossing, and know that survivors trip it too.** A
> single window over the threshold is noise; N consecutive observations is a
> detection. But on the reference dataset here, **six of eight assets that never
> failed** eventually tripped a three-observation rule — five of them only in
> their final hours, when the failing asset beside them was shaking the whole
> rig. One healthy asset tripped it 24.3 hours out while a real failure tripped
> at 4.8. Report the crossing **rate** over windows, which stays defensible
> because survivor crossings are few and late; a per-asset hit rate does not.

## Verified platform behavior

Verified 2026-08-11 → 2026-08-20, updated 2026-10-13. Path 2's own measurements are in
[`references/platform-notes.md`](references/platform-notes.md).

- **Only the fused feature mode is reachable.** The design admits three —
  embeddings, hand-crafted features, or both — but the `ad` blueprint exposes
  only `feature_names`, and the graph always routes the encoder into `fuse`.
  An empty `feature_names` is accepted at bundle creation and then fails at
  graph instantiation with `ChannelFeaturesNode requires at least one feature`.
- **Inputs over ~50 MiB used to abort in `ConcatColumnsNode`:**
  `checkpoint boundary reached with unequal buffered rows per port ([0, 2]); the
  branches are not row-aligned`. The runner checkpoints every 50 MiB of input or
  10 minutes, and the forked graph's barrier copies misaligned at `fuse`.
  **Fixed in runtime `1.1.9-590-194e878`** (atai_core #6317, Dev since
  2026-08-17): full 102 MB and 299 MB inputs then completed with every window
  valid. On a runtime without the fix, split inputs under 50 MiB, or turn the
  triggers off in the bundle's values:
  `"runner_config": {"checkpointing": {"trigger_after_input_byte_size": 0,
  "trigger_after_delay_ms": 0}}` (Path 2's `deliver.py` does). (The *edge*
  runtime reports the graph as `{enabled: false, supported: false}` and declines
  to checkpoint instead of aborting.)
- **Runtime is dominated by worker contention, not window count.** Budget by
  the audit events, not the clock. The bundled 240-window slice, run through
  the two quick-start bundles minutes apart: **33 s end-to-end** (job time
  18 s) with a clear queue vs **~9 min wall-clock** for the identical input
  when scheduling was contended. Other tenants' jobs aren't visible to you, so
  the run's own audit events are the only queue-state signal. Do not read a
  fast run as a broken one — see the `/results` check above.
- **Prefer sequential runs.** Whether concurrent runs queue depends on what
  else is running on the deployment at that moment: they queue when other
  workloads hold the workers, and run as concurrent jobs when they don't.
  Other tenants' workloads aren't visible to you, so there is no
  serialization to rely on and no parallelism to count on — submit one at a
  time unless you are deliberately testing this.
- **Re-runs agree to within a small noise floor, sometimes exactly.** Two runs
  of the same 840-window bearing slice with the same detector: max absolute score
  difference **4.2×10⁻⁵**, **100%** label agreement; across deployments on the
  bundled slice, 240/240 labels, max difference 2.9×10⁻⁵. On the TEP quickstart
  (Dev, 2026-10-13) two complete runs from scratch, each with its own
  optimization, blueprint and delivery bundle, gave **byte-identical** outputs.
  Differences above ~10⁻⁴ are real.

## End-to-end verification

`python3 run_ad_agent.py` with no arguments — upload → resolve
`AD Quick Start (Bearing Breakdown)` by name → run → poll → download → score,
on the bundled 120-snapshot transition slice:

```
windows           240   invalid 0
snapshots         120
score median/max  1.708 / 2.687
crossings         105  (43.75% of windows)
DETECTED          snapshot 47  ->  56.0 operating hours before end of record
```

Re-run on **dev on 2026-10-13**, after the `ad` blueprint's re-seed: the same output, to the
snapshot (240 windows, 0 invalid, median 1.708, detected at snapshot 47, 56.0 h).

`--embeddings` resolves the Embeddings bundle and scored **identically**
(240/240 labels, scores within the noise floor), completing in **33 s
end-to-end** with a clear queue (job time 18 s: ~10 s queued, Omega download
3.8 s + load 6.4 s, detector load 3 ms) and downloading the 2.0 MB output.
The base run of the identical input minutes earlier took ~9 min wall-clock
under contention — the runtime bullet above, demonstrated back-to-back.

**Cross-checked against an independent path.** The same bearing scored as a
full 984-snapshot lifetime through an independent offline scorer (Archetype
AI-internal) detects at snapshot **650** with a **55.5 h** lead. This slice
starts at snapshot 600, so snapshot 47 here is snapshot 647 there — three
snapshots and half an hour apart, from a different runner, a different input
length and a different scorer. That agreement is the check worth having; a
single number reproduced by the code that produced it is not.

Two bugs this run caught, which no unit test would have:

- The runner first reported every successful run as **failed**, because it
  treated any log message containing "terminated" as terminal failure — and the
  happy path logs `Agent execution terminated successfully`.
- The runner read the results ref from the wrong level of the envelope (see
  step 5 for the shape), which yields an empty filename and a 404 on a bare
  `/files/download/` URL.

## Local Setup (Path 1)

```bash
cd skills/atai-anomaly-discovery-agent/references

# One dependency: the official Archetype AI client. Note the -r.
pip install -r requirements.txt

# Create the .env IN THIS DIRECTORY — the runner reads ./.env from where it
# runs (the file is gitignored). BOTH variables required, no default endpoint;
# note: NO /v0.5 suffix — the runner mounts /agents and /v0.5/files itself:
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://api.u1.archetypeai.app
EOF

python3 run_ad_agent.py                          # bundled sample slice, base bundle
python3 run_ad_agent.py --embeddings             # + Omega embedding per window
python3 run_ad_agent.py --csv my_slice.csv       # your own prepared CSV
python3 run_ad_agent.py --detector s3://...      # your own fitted detector
python3 run_ad_agent.py --score-only out.csv     # re-score a downloaded output
```

The `--csv` default resolves next to the script, so the runner works from any
directory as long as `.env` is in the one you run from. Expect **~30–50 s**
end-to-end for the bundled 240-window slice with a clear queue, and minutes
when workers are contended — the run's own log events are the only queue-state
signal. The runner resolves the bundle by name, streams `/logs` while it polls,
and self-scores against the `_labels.csv` sidecar at the end.

---

# Path 2 — Build an AD agent on your own data

> **Status: verified end to end on dev (2026-10-13)**, where AD support in the
> Optimizations and Evals APIs landed on 2026-10-12; check another deployment before relying
> on it there. Built on the Tennessee Eastman Process:
> [anomaly-discovery-agent-example-tep-quickstart](https://github.com/archetypeai/anomaly-discovery-agent-example-tep-quickstart)
> (the whole lifecycle in about 30 minutes, reproduced exactly by a second run from
> scratch). The scripts use a small stdlib HTTP helper
> ([`atai_http.py`](references/ad_lifecycle/atai_http.py), shared with the OSM skill's
> Path 2) because the official client doesn't cover these APIs yet.

```
role files ──► 1 Optimize ──► 2 promote ──► 2 test once (Evals) ──► 3 deliver (bundle + runs) ──► 4 score
  (prep skill)   fit on normal,  best trial     one-shot, with           a score per window          detection, delay,
                 score on        → blueprint    predictions checked                                   false alarms
                 validation
```

Measured platform behaviour, with the numbers: [`references/platform-notes.md`](references/platform-notes.md).

## Step 0 — Role files

Path 2 starts from **role files**, built and checked by
[`atai-anomaly-discovery-agent-data-prep`](../atai-anomaly-discovery-agent-data-prep/SKILL.md)
to its contract, [`role-files.md`](../atai-anomaly-discovery-agent-data-prep/references/role-files.md):

| role | what | used in |
|---|---|---|
| `library/` | one file per normal recording, no label column | training, every trial: every row `normal` |
| `validation/` | normal-only and incident recordings, joined with a rejected gap, with `label` | scoring each trial |
| `test/` | the same | scored **once**, after promotion |
| `delivery/` + `labels/` | the delivery recordings and a `calibration.csv` of known-normal data, **without** `label`; labels held back | the deployed run, scored afterwards |

plus `manifest.json` (channels, step, every file and its recordings). Nothing in the scripts
is specific to one dataset. Build the prep skill's sample in about 3 s:

```sh
cd skills/atai-anomaly-discovery-agent-data-prep/references
pip install -r requirements.txt
S=sample_data/recordings.csv
python prepare.py --index $S --out prepared --step-ms 180000
python check_baseline.py --index $S --prepared prepared --window 48 --step 8
python build_roles.py --index $S --prepared prepared --out roles
python preflight_roles.py --roles roles --prepared prepared --index $S --window 48 --step 8   # RESULT: PASS
```

## Step 1 — Optimize: fit on normal, score on incidents

```sh
cd references/ad_lifecycle
pip install -r requirements.txt
# ATAI_API_KEY + ATAI_API_ENDPOINT: the nearest .env from here upwards is used
R=../../../atai-anomaly-discovery-agent-data-prep/references/roles
python optimize.py --roles $R --dry-run                                   # the plan: no uploads, no jobs
python optimize.py --roles $R --out out --name ad-sample --background     # tail -f out/optimize.log
```

`POST /agents/optimizations` with the `ad` blueprint's id, the library files as
`training_examples` (ground truth `{"from": {"constant": "normal"}}`), the validation files
as `validation_examples` (`{"from": {"column": "label"}, "downsampling": "last_record"}`),
`"primary": "anomaly.macro_f1"`, a `budget.max_trials`, `overrides.values` with everything
pinned (`timestamp_column`, `data_columns`, and every single-valued option) and a search
space of `kind: value` parameters:

```json
{"parameters": {"n_neighbors": {"kind": "value", "spec": {"type": "categorical", "values": [5, 10]}}}}
```

with `window_size` 48, `step_size` 8, `metric` `l2`, `pca_components` 16, `threshold` 1.215
and `feature_names` `std,range,slope,median` pinned: the TEP full example's 24-trial winner.
Give an option several values to search it; `--threshold LOW:HIGH` searches a float range.

- **Pin the window, step and features.** The blueprint's defaults (window 1,024, step 256,
  seven features) apply otherwise, and a 1,024-sample window can exceed your recordings.
- **`feature_names` can only be pinned**; compare feature sets with one optimization each.
- **The space can't be empty**, so `n_neighbors` is always in it, even with one value.
- **Both legs use the one `step_size`**, so validation is scored at the fit stride.

**Expect, on the sample** (2 trials, ~10 min on a busy queue):

```
  otr_…  completed {"n_neighbors": 5}   macro-F1 0.8417  anomaly F1 0.8285  recall 0.717  precision 0.982  windows 1,265
  otr_…  completed {"n_neighbors": 10}  macro-F1 0.8336  anomaly F1 0.8161  recall 0.692  precision 0.995  windows 1,265
```

## Step 2 — Promote, then test once

```sh
python promote_and_test.py --roles $R --out out --name ad-sample
```

1. **Promote** the best trial: `POST /agents/optimizations/{opt}/trials/{trial}/promote` with a
   `blueprint_key` unique per trial (`<name>-<last 6 of the trial id>`). The blueprint pins the
   trial's values and carries the fitted detector as a registry artifact (`aar_…`).
2. **Test once:** `POST /agents/evals` with `blueprint_id`, every test file and
   `emit_predictions: true`. An eval takes no value overrides: it runs at the trial's own
   threshold and step.
3. **Check before reporting:** every prediction's gold label equals the sidecar's at the
   window's last row (windows tile each file from row 0 at `step_size`; `row_index` counts
   across the whole eval, so a window's place is its rank within its file), and the offline
   confusion matrix equals the platform's.

**Expect, on the sample** (~3–8 min for the eval):

```
promoted otr_… -> ad-sample-<6 chars> (blp_…)
test, 2 files: macro-F1 0.7462  anomaly F1 0.8247  recall 0.702  precision 1.000  windows 805 (25 abstained)
  false alarms on normal windows: 0 / 205 (0.00%)
  idv01          window detection   100 /   100  1.000
  idv10          window detection    28 /   100  0.280
  idv15          window detection     0 /   100  0.000
```

The 25 abstained windows cross the gaps between joined recordings. Macro-F1 moves with the
class mix (test here is 29% normal rows); anomaly F1, recall and precision are the numbers
to read. Once you have seen the test, the setting is frozen.

## Step 3 — Deliver

```sh
python deliver.py --roles $R --out out --name ad-sample --background     # tail -f out/deliver.log
```

`POST /agents/bundles {"blueprint": "<your key>", "values": {"step_size": 1, "output_score": true,
"runner_config": {…checkpointing off…}}}`: the bundle's values override the blueprint's, so
every sample gets a score. `--threshold` relabels at another operating point with the same
detector. Then one run per delivery file (`POST /agents/bundles/{id}/run`), a few at a time;
each output is downloaded again until its last window reaches the input's end, and a
network outage is waited out instead of failing every file. **Expect, on the sample**
(~13 min): `2 of 2 delivery files have complete outputs`.

## Step 4 — Score the delivery

```sh
python score.py --roles $R --out out
```

Each window is paired with the held-back label of its last row; windows the platform marked
invalid are left out and counted. At the trial's threshold, and at one set to a 1%
false-alarm rate on the calibration file's scores, per incident type: **window detection**,
**detected** (3 consecutive windows over the threshold at or after onset), **delay** to that
detection, and **AUROC** against every normal window; overall, the **false-alarm rate**.

```
delivery: 1 files, 6,391 windows scored; left out: 470 invalid, 0 with no label at finish time
calibration: 4,565 known-normal windows; 1.51% over the trial's threshold; threshold for 1% false alarms = 1.234
=== calibrated to 1% false alarms on known-normal data: threshold 1.234, sustain 3 ===
  type           window det  detected  delay h   AUROC
  idv01               99.0%    1/1        0.50   0.996
  idv02               98.0%    1/1        0.90   0.989
  idv05               99.9%    1/1        0.15   1.000
  idv10               43.6%    1/1        2.85   0.910
  idv13               94.4%    1/1        2.35   0.976
  idv15                1.8%    1/1       34.80   0.597
  false alarms on normal windows: 0.38% (6 / 1,591)
  mean window detection 72.8%   macro AUROC 0.911
```

- **Read "detected" with the window rate and the delay.** IDV 15 counts as detected, 34.8 h
  after onset, with 1.8% of its windows flagged and AUROC 0.597: a false alarm that happened
  to fall after onset. It is one of the three TEP faults the literature calls undetectable.
- **Step changes are caught within an hour, slow faults in hours** (IDV 13's drift 2.35 h).
- The quickstart, on 4 delivery files of 21 runs each, reads 78.0% detection at 1.41% false
  alarms, AUROC 0.901, with all 17 detectable fault types caught in every run.

## Path 2 pitfalls

- **A window of 1,024 on slow data.** The `ad` blueprint's default window at a 3-minute rate
  is 51 hours; recordings shorter than a window produce no valid window and a meaningless
  score. Pin the window from your data's time scale.
- **No default search space.** Leaving `search_space` out is a `400`; give at least one parameter.
- **`threshold` as a list of floats is a 422**; use one value or `{"type": "float_range", "min", "max"}`.
- **The eval number and the delivery number measure different things.** Evals scores windows
  at the trial's threshold and stride; the delivery is scored per incident, with a delay, at
  stride 1 and a calibrated threshold. Report both.
- **Don't trust `completed` for the output, or `failed` for the run**: `deliver.py` downloads
  until the output reaches the input's end, and a run reported failed can still have one.
- **A key exported in your shell beats `.env`**; keys are per deployment. The helper checks the
  key with one GET first and says where it came from (showing only its last 4 characters).

## Cleanup (both paths)

The pre-packaged Quick Start bundles are canonical and shared — **do not delete them**. Delete
your own agent instances, and any bundle you created:

| object | delete | notes |
|---|---|---|
| agent instances (`agt_…`) | ✅ `DELETE /agents/instances/{id}` | cancel a running one first (`POST …/cancel`) |
| your delivery bundles (`bnd_…`) | ✅ `DELETE /agents/bundles/{id}` | delete the instances first |
| promoted blueprints (`blp_…`) | ❌ 403 for a regular key | ask an org admin; use a distinctive `--name` |
| optimizations (`opt_…`), evals (`evl_…`) | — | records of what was fitted and scored; keep them if your docs cite them |

## File Layout

```
skills/atai-anomaly-discovery-agent/
├── SKILL.md                  ← this file
├── references/
│   ├── run_ad_agent.py       ← Path 1: the managed flow on the official client (upload → resolve → run → poll → download → score)
│   ├── requirements.txt      ← Path 1: archetypeai
│   ├── .env.example          ← copy to .env and fill in (both paths)
│   ├── sample_data/          ← Path 1: a bearing slice spanning normal → anomalous, its labels sidecar, attribution
│   ├── platform-notes.md     ← Path 2: measured platform behaviour
│   └── ad_lifecycle/         ← Path 2 (stdlib HTTP until the client covers these APIs)
│       ├── atai_http.py        requests with retries, uploads with a cache, paging, polling
│       ├── common.py           the role-file manifest, examples, pinned columns
│       ├── background.py       --background: nohup (+ caffeinate on macOS), output to a log
│       ├── optimize.py         Step 1: search space → Optimizations API → trials
│       ├── promote_and_test.py Step 2: best trial → blueprint → one eval, predictions checked
│       ├── deliver.py          Step 3: bundle from the blueprint → one run per file → complete outputs
│       ├── score.py            Step 4: detection, delay, AUROC, false alarms, at two thresholds
│       └── requirements.txt    numpy, pandas (score.py, promote_and_test.py)
└── tests/
    ├── test_references.py    ← Path 1: network-free unit tests
    └── test_lifecycle.py     ← Path 2: network-free unit tests
```

Run both: `python -m pytest skills/atai-anomaly-discovery-agent/tests`.

## Data attribution

The sample data is derived from the **IMS Bearing Data Set**, generated by the
NSF I/UCRC Center for Intelligent Maintenance Systems (University of Cincinnati)
with support from Rexnord Corp., and distributed by NASA's Prognostics Center of
Excellence.

> Qiu, H., Lee, J., Lin, J., and Yu, G. (2006). "Wavelet Filter-based Weak
> Signature Detection Method and its Application on Rolling Element Bearing
> Prognostics." *Journal of Sound and Vibration* 289, 1066–1090.

Publicly available from NASA's Prognostics Data Repository for research
purposes.
