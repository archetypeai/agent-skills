---
name: atai-anomaly-discovery-agent
description: >
  Run Archetype AI's managed Anomaly Discovery (AD) agent over the Agent API —
  upload a prepared sensor CSV, resolve the pre-packaged "AD Quick Start"
  bundle by name (a fitted LOF detector and its threshold are already pinned),
  run it, poll, download a per-window anomaly score. Use when the user wants
  to flag "this no longer looks like normal operation" from **normal-only
  reference data** — no fault library, no labelled examples of the thing being
  detected, because none exist. Covers the resolve → run → poll → results
  lifecycle, the output schema, the per-asset framing, the validation settings
  that silently invalidate every window above 1 kHz, the 50 MiB checkpoint
  limit, and scoring by lead time and false-alarm rate. Do NOT use for
  labelled operating regimes (`atai-operational-state-monitoring-agent`), a
  *named* recurring fault (`atai-rare-event-detection-agent`), client-side
  embeddings over `/query` (`atai-newton-omega-model`), or raw-CSV prep
  (`atai-newton-omega-model-data-prep`).
---

# AD Agent — Managed Anomaly Discovery via the Agent API

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

> **Your own data?** The pre-packaged **"AD Quick Start" bundles** pin a
> detector fitted on **one specific bearing's** healthy baseline (the bundled
> sample slice is from that same bearing). A detector is asset-specific by
> construction — it encodes one machine's notion of normal — so running your
> own asset through the quick-start bundle is a transfer test, not a
> deployment. For your own data, contact **support@archetypeai.dev**:
> Archetype AI will fit a detector with you, and you create a bundle from the
> canonical `ad` blueprint around it ("Bring your own detector", below).

**Do not use this skill when:**
- The user has a labelled library across all regimes and wants "which state is
  the asset in?" — use
  [`atai-operational-state-monitoring-agent`](../atai-operational-state-monitoring-agent/SKILL.md)
- The fault is **named** and a couple of labelled incidents exist — use
  [`atai-rare-event-detection-agent`](../atai-rare-event-detection-agent/SKILL.md).
  AD will flag it, but it cannot tell you *which* fault it is
- You want per-window embeddings to do ML client-side — use
  [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)
- The raw CSV still needs cleaning, gap-aware segmentation or normalization —
  see [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md).
  AD assumes prepared input with a documented sampling rate

## Endpoints

```
Files API   POST {ATAI_API_ENDPOINT}/v0.5/files                  (multipart upload)
            GET  {ATAI_API_ENDPOINT}/v0.5/files/download/{name}  (download)
Agent API   {ATAI_API_ENDPOINT}/agents/...                       (versionless!)
Authorization: Bearer <API_KEY> on every call
```

The Agent API is **versionless** — `/agents`, not `/v0.5/agents`. If
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
`id`.

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

## How the score behaves, and why the threshold is in the artifact

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
- **The threshold is chosen at fit time and stored in the artifact.** Overriding
  `threshold` in bundle `values` is scoring a different detector than the one
  that was validated. The blueprint defaults it to
  `${models.detector.parameters.threshold}` for exactly that reason.

LOF suits this problem because it assumes nothing about the shape of normal — no
Gaussian, no single cluster — and needs no anomalies to fit.

## One detector per asset

A detector encodes **one machine's** notion of normal, fitted on that machine's
own baseline. Running it against a different asset is a transfer test, not a
deployment, and the scores are not comparable to the ones it was validated on.

This is not a limitation to work around; it is the framing that makes the
false-alarm rate meaningful. Pooling assets into one detector means the
reference set spans several machines' normals, and the manifold inflates until
everything looks normal.

## Bring your own detector (advanced)

The quick-start detector encodes one bearing's notion of normal. For your own
asset you need a detector fitted on **its** healthy baseline. The fitting
pipeline is not accessible to external users yet — contact
**support@archetypeai.dev** and Archetype AI will fit one with you and hand
back the detector artifact. With that artifact you create your own bundle from
the canonical `ad` blueprint, then run it as in step 3:

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

Everything else — `window_size`, `step_size`, `data_columns`, `feature_names`,
`encoder_model`, `threshold` — is read from the detector's own `parameters`
metadata, which is why the quick-start bundles needed no window pinning and why
overriding `threshold` means scoring a different detector than the one that was
validated.

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

Verified 2026-08-11 → 2026-08-20.

- **Only the fused feature mode is reachable.** The design admits three —
  embeddings, hand-crafted features, or both — but the `ad` blueprint exposes
  only `feature_names`, and the graph always routes the encoder into `fuse`.
  An empty `feature_names` is accepted at bundle creation and then fails at
  graph instantiation with `ChannelFeaturesNode requires at least one feature`.
- **Inputs over ~50 MiB abort in `ConcatColumnsNode`:**
  `checkpoint boundary reached with unequal buffered rows per port ([0, 2]); the
  branches are not row-aligned`. The runner checkpoints at a byte threshold, and
  the forked graph cannot be snapshotted while rows sit inside the encoder's
  GPU batch — the signal reaches `fuse` ahead of its own data. Deterministic,
  not contention: the same file failed alone and eight-way parallel, and passed
  when truncated. **`runner_config.checkpointing` is accepted by the bundle API
  and never reaches the runner**, so it cannot be disabled. Split inputs under
  the threshold and reassemble the outputs. (The *edge* runtime reports the
  graph as `{enabled: false, supported: false}` and declines to checkpoint
  instead of aborting.)
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
- **Re-running the same input is not bit-identical — within a deployment or
  across them.** Two runs of the same 840-window slice with the same detector:
  max absolute score difference **4.2×10⁻⁵**, median relative **0.0002%**,
  **100%** label agreement. Across deployments on the bundled slice (both
  bundle variants): 240/240 label agreement, max score difference 2.9×10⁻⁵.
  That is the platform's noise floor; differences larger than that are real.
  (The OSM and RED siblings *are* byte-identical across deployments — the
  difference is this graph's forked GPU batching.)

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

## Local Setup

```bash
# No third-party deps — references/run_ad_agent.py is stdlib-only.

cd skills/atai-anomaly-discovery-agent/references

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

## References

- [`references/run_ad_agent.py`](references/run_ad_agent.py) — stdlib-only
  runner: upload → resolve the Quick Start bundle by name → run → poll (status
  **and** logs) → download → score against a ground-truth sidecar (lead time,
  crossing rate, sustained-crossing rule). Runs the bundled sample with no
  arguments; `--embeddings` switches to the Embeddings bundle;
  `--detector s3://…` creates a bundle around your own detector instead.
- [`references/sample_data/`](references/sample_data/) — a prepared bearing
  slice spanning the transition from normal to anomalous, plus its ground-truth
  sidecar. See the README there.
- [`references/.env.example`](references/.env.example) — the two required
  variables.

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
