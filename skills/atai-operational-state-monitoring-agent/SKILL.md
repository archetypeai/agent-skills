---
name: atai-operational-state-monitoring-agent
description: >
  Run Archetype AI's Operational State Monitoring (OSM) agent over the Agents API,
  on two paths. Path 1 — run a maintained, pre-packaged "OSM Quick Start" bundle
  (the Volve six-state classifier + windowing already pinned; plain or with per-window
  Omega embeddings): upload a sensor CSV, resolve the bundle by exact name, run, poll,
  download per-window state predictions, score against a ground-truth sidecar.
  Path 2 — build an OSM agent on YOUR OWN labelled data: from role files
  (library / validation / test / delivery) fit and score settings with the
  Optimizations API, promote the best trial to a blueprint, test it once with the
  Evals API, then deliver it as a bundle over new recordings and score the delivery.
  Covers the search space, promote, Evals, batch runs, the output CSV schema
  (`finish_timestamp, predicted_state, invalid, p_<state>…`), last-record scoring,
  and the platform's limits (1 MiB config, time jumps, absent states score 0).
  Path 2 is verified end to end on production, staging, Tokyo and dev. Do NOT use for
  client-side embedding + KNN over `/query` (that's `atai-newton-omega-model`), for
  turning raw sensor recordings into role files (that's
  `atai-operational-state-monitoring-agent-data-prep`), or for generic time-series
  cleaning (`atai-newton-omega-model-data-prep`).
---

# OSM Agent — Managed State Classification via the Agents API

The OSM agent labels every window of a multichannel sensor recording with an **operational
state** (drilling activities, machine modes, process phases) server-side. The platform runs the
whole graph:

```
source → interpolate → window → windowInterpolate → samplingRate → limitValues → encoder → classifier → sink
```

The encoder is Newton Omega; the classifier is a kNN over Omega embeddings. You never host
either.

## Choose a path

| | **Path 1 — run a maintained bundle** | **Path 2 — build an agent on your data** |
|---|---|---|
| you have | a prepared CSV (the Volve sample, or data like it) | labelled recordings of **your** machine |
| you get | per-window predictions from Archetype AI's six-state Volve classifier | your own blueprint and bundle, fitted and chosen on your data, with a test score and a delivery score |
| APIs | files, bundles, instances | files, **Optimizations**, **promote**, **Evals**, bundles, instances |
| scripts | [`references/run_osm_agent.py`](references/run_osm_agent.py), on the official `archetypeai` client | [`references/osm_lifecycle/`](references/osm_lifecycle/), a small stdlib HTTP helper (the client doesn't cover Optimizations / Evals / promote yet) |
| time | ~1–2 min on a clear queue | minutes to hours, with the dataset's size and the platform's load |
| status | verified on production | **verified on production, staging (2026-10-01) and dev (2026-09-30), identical results** |

Path 2's last step *is* Path 1, run on your own bundle instead of the maintained one, so the
output schema, polling, pitfalls and cleanup below apply to both.

**Tailored work** (a classifier Archetype AI fits and packages for your data, or help with a
hard dataset): support@archetypeai.dev.

## When to Apply

- Classify operational states over a CSV with **no client-side ML** — Path 1 for the
  maintained Volve bundle, Path 2 for your own states
- Build, test and deploy an OSM agent for **your** machine from labelled recordings — Path 2
- Demo or evaluate the managed OSM path as a **deployed, repeatable batch job**
- Score managed predictions against held-out ground truth

**Do not use this skill when:**
- You want interactive, per-window embeddings to do ML client-side — use [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)
- Your recordings still need resampling, labelling or splitting into role files — use [`atai-operational-state-monitoring-agent-data-prep`](../atai-operational-state-monitoring-agent-data-prep/SKILL.md) first; for generic cleaning of messy raw data, [`atai-newton-omega-model-data-prep`](../atai-newton-omega-model-data-prep/SKILL.md)

---

# Path 1 — Run a maintained bundle

The platform ships **canonical "OSM Quick Start" bundles** with the six-state Volve classifier and
its windowing already pinned. One run = one agent instance = one input file. You upload the CSV,
**resolve the pre-packaged bundle by name**, run it, poll until terminal, and download one output CSV
of per-window predictions. The classifier is fit to Volve drilling data; for your own states, see
Path 2.

## Endpoints (both paths)

Two API surfaces are involved, mounted differently:

```
Files API   POST {ATAI_API_ENDPOINT}/v0.5/files                  (multipart upload)
            GET  {ATAI_API_ENDPOINT}/v0.5/files/download/{name}  (download)
Agents API   {ATAI_API_ENDPOINT}/agents/...                       (versionless!)
Authorization: Bearer <API_KEY> on every call
```

The Agents API is **versionless** — it lives at `/agents`, not `/v0.5/agents`. If your `ATAI_API_ENDPOINT` carries a `/vX.Y` suffix, strip it before appending `/agents`. **Both `ATAI_API_KEY` and `ATAI_API_ENDPOINT` are required** — there is no default endpoint.

> **⚠️ The bundle API is plural everywhere** as of 2026-08-11:
> `GET /agents/bundles` (list/search), `GET /agents/bundles/{id}` (fetch),
> `POST /agents/bundles` (create), `POST /agents/bundles/{id}/run` (run). The
> singular forms (`POST /agents/bundle`, `POST /agents/bundle/{id}/run`,
> `GET /agents/bundle/{id}`) now return **404** — earlier revisions of this
> skill described a singular/plural split that predates this migration.

> **Availability.** The pre-packaged "OSM Quick Start" bundles are published
> on the production deployment (`https://api.u1.archetypeai.app`) — set
> `ATAI_API_ENDPOINT` to it and the full upload → run → score cycle works as
> documented here. If name resolution returns `no bundle named … found`, the
> bundle isn't published in the deployment you're pointed at: resolving by
> name is portable, so pass a known `--bundle-id` meanwhile, or contact
> support@archetypeai.dev.

## Step 1 — Upload the input CSV

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" \
  -F "file=@sensor_slice.csv;type=text/csv" \
  "$ATAI_API_ENDPOINT/v0.5/files"
```

The response carries two identifiers — `file_id` (the filename) and `file_uid` (`fil_…`). **Source connectors reference the `file_id`, not the `fil_` uid.**

## Step 2 — Resolve the pre-packaged bundle by name

The maintained bundles are canonical (org-wide) and identified by a **stable name**. Their **id is deployment-specific**, so resolve by name for portability — the plural read endpoint does a case-insensitive substring search over name and id:

```sh
curl -G -H "Authorization: Bearer $ATAI_API_KEY" \
  "$ATAI_API_ENDPOINT/agents/bundles" \
  --data-urlencode "query=OSM Quick Start (Volve Six State)" --data-urlencode "limit=20"
```

Two bundles are published:

| Name | Emits |
|---|---|
| `OSM Quick Start (Volve Six State)` | per-window state predictions |
| `OSM Quick Start (Volve Six State, Embeddings)` | the above **plus** the **Newton Omega encoder embedding for each window** — one `embedding_{variate}` column per sensor channel, each a 768-d vector (the same embeddings `atai-newton-omega-model` gets from `/query`, here computed server-side as part of the run). **The output file gets dramatically larger**: 314 MB vs 221 KB on the sample slice, ~1,400× |

**Select the EXACT name match, not the first result.** `query=` is a substring match and results come back newest-first, so a *prefix* returns both variants with Embeddings first — verified on Prod:

```
query 'OSM Quick Start'                    -> 2   Embeddings first
query 'OSM Quick Start (Volve Six State'   -> 2   Embeddings first
query 'OSM Quick Start (Volve Six State)'  -> 1   the closing paren excludes the variant
```

Taking `data[0]` on either of the first two silently runs the Embeddings bundle — a 314 MB output where you expected 221 KB. Pick `b["name"] == "OSM Quick Start (Volve Six State)"` and take its `id`, and prefer `is_canonical` if two bundles share a name.

The bundle already pins the classifier and its windowing (`window_size=16, step_size=1`, plus the Volve-sized validation tolerances), so there is **nothing to create and no classifier URI to supply**. (For reference, these currently resolve to `bnd_02yp01y1er80vv1s5egaeb7p74` and `bnd_1hd3aymx308tct952dn5nwram9` in production; the ids are deployment-specific, which is why you resolve by name — pin ids only as a last resort.)

## Step 3 — Run the bundle

```sh
curl -X POST -H "Authorization: Bearer $ATAI_API_KEY" -H "Content-Type: application/json" \
  "$ATAI_API_ENDPOINT/agents/bundles/$BUNDLE_ID/run" -d '{
    "connectors": {"source": [{"type": "file", "id": "sensor_slice.csv"}]}
  }'
```

Each run creates a **new agent instance** (`agt_…`) — one agent per input file. With no sink configured, the runner writes one output per input, named after the input file. **Run agents sequentially.** Whether concurrent runs queue depends on what else is running on the deployment at that moment: they queue when other workloads hold the workers, and run as concurrent jobs when they don't. Workers are shared either way, so three at once have run ~3× slower each than one at a time. Other tenants' workloads aren't visible to you, so neither outcome is predictable from your side.

## Step 4 — Poll until terminal

```sh
GET $ATAI_API_ENDPOINT/agents/instances/{agent_id}          # status: running | paused | completed | failed | ...
GET $ATAI_API_ENDPOINT/agents/instances/{agent_id}/events   # audit log (level, created_at, message)
```

Poll every ~15 s, echoing new audit events (`run started`, `dispatched to JOS as job_…`, `JOS job completed`). Runtime is dominated by worker contention, not window count — see the Runtime section below.

**A `completed` status can come before the output is fully written** (a known platform issue). The status, and the `/results` listing, can be ahead of the output file: a download within seconds of `completed` can return a partial CSV, and minutes later the full one (prod: 326 of 997 rows at first, all 997 later). Treat a run as complete only once its output is: the output's last window ends at the input's last row (`finish_timestamp` = the input's last timestamp, seams and all). Until then, download again every ~15 s.

**A `failed` status can hide a successful run.** Observed live: a job whose container logged "terminated successfully" (output present) surfaced as `status=failed` with `error: repeated failures polling JOS job` — the service's job poller flaked, not the job. Before re-running a "failed" agent, check `/results`; if the output is there, the run succeeded.

## Step 5 — Fetch results and download

```sh
GET $ATAI_API_ENDPOINT/agents/instances/{agent_id}/results
GET $ATAI_API_ENDPOINT/v0.5/files/download/{filename}
```

`/results` lists output refs, each nesting its fields under an inner `data` object (`data[].data.filename`, `data[].data.num_bytes`, `data[].data.ref`) — not at the top level. Download each via the files API. Run outputs are owned by the user who launched the run and do not expire.

Like the other list endpoints, `/results` **pages**: `data`, `has_more`, `next_cursor`, with `limit` (default 100, max 1000) and `after`/`before` cursors. **The cursor is opaque** — pass `next_cursor` back verbatim and never derive it from `data[last].id`; a fabricated value is rejected with `400 invalid cursor`. One run through a quick-start bundle emits one output, so the first page is the whole answer — a bundle with several sink ports is where paging starts to matter.

[`references/run_osm_agent.py`](references/run_osm_agent.py) scripts the whole flow (upload → resolve bundle → run → poll → download, again until the output reaches the input's end) on the official [`archetypeai` python client](https://github.com/archetypeai/python-client) and — if a `<input>_labels.csv` ground-truth sidecar sits next to the input — scores the run automatically: accuracy (all-windows and steady-state cuts), per-class precision/recall/F1, macro-F1.

## Output CSV — one row per window

```
finish_timestamp, start_timestamp, predicted_state, invalid, p_<state>, p_<state>, ...
```

- **`finish_timestamp` is the window-END timestamp** — a prediction means "the state *now*, given the last `window_size` samples"; `start_timestamp` is the window's first sample. Score each window against the ground-truth label of its final row (end-row labeling on `finish_timestamp`).
- **`p_<state>` columns are emitted in alphabetical state order**, not library order.
- **`predicted_state=INVALID_STATE`** marks windows straddling a timestamp seam (backward jump between concatenated segments) — the platform validates timestamp monotonicity strictly. Exclude these when scoring; count them.
- The **Embeddings** bundle adds the **Newton Omega embedding for each window**: one `embedding_{variate}` column per sensor channel, each a 768-d vector (~76 KB/row extra with 9 channels — verified: **314 MB vs the base bundle's 221 KB** on the 4,185-window sample slice, ~1,400×, with predictions identical); the base bundle omits them. Use it when you want the vectors alongside the predictions — client-side similarity, drift monitoring, projections, or downstream ML per [`atai-newton-omega-model`](../atai-newton-omega-model/SKILL.md)'s patterns — without paying one `/query` call per window.
- Runs are **reproducible**: the same input through the same-named bundle produces a **byte-identical output** run to run, `cmp`-verified on the 221 KB base file across repeat runs and across deployments (the model is not re-fit per run). The embeddings variant's ten prediction columns are byte-identical to the base output — the embedding columns are strictly additive, never a change in prediction.

## Runtime

**Runtime is dominated by worker contention, not window count.** The same ~4,185-window sample slice, measured end-to-end across verified runs:

| Queue state | End-to-end | Notes |
|---|---:|---|
| Empty (verified) | **~1–2 min** | 63–107 s for the base bundle, 70–75 s for embeddings including the 314 MB download. One run's job breakdown: 41 s total — ~16 s queued, ~14 s model loading, **~26 s to encode+classify** ≈ 160 win/s |
| Busy (verified) | 21m53s – 27m18s | same slice, same bundle — ~2.2 win/s effective, ~17× slower |

Budget by the **audit events, not the clock** — you cannot see other tenants' jobs, so the queue state is only observable from your run's own event timing. Any past "runtime per window count" figure measured without knowing the queue state is a contention artifact, not an intrinsic rate. Concurrent runs of your own divide the same pool (N parallel ran ~N× slower each under load); whether they queue or run side by side depends on what else holds the workers, so sequential remains the predictable default.

## Common Pitfalls

- **Resolve by name, not by id.** The pre-packaged bundle's id is deployment-specific; only the name is stable. And **match the name exactly** — a prefix of the base name also matches the `…, Embeddings)` variant, which sorts first, so `data[0]` runs the wrong bundle.
- **The bundle API is plural everywhere** (as of 2026-08-11). `GET /agents/bundles` (list/search), `GET /agents/bundles/{id}` (fetch), `POST /agents/bundles` (create), `POST /agents/bundles/{id}/run` (run). Every singular form (`/agents/bundle/…`) 404s.
- **Source connectors take the `file_id` (filename), not the `fil_` uid.** Both come back from the upload; using the uid fails to resolve.
- **The Agents API is versionless.** `POST {endpoint}/v0.5/agents/…` 404s; strip any `/vX.Y` suffix and use `/agents/…`. The files API keeps its `/v0.5`.
- **`completed` ≠ complete until the output reaches the input's end.** The status can arrive before the output file is fully written; check the last window's `finish_timestamp` against the input's last row, and download again if it ends early.
- **`failed` ≠ failed until you check `/results`.** The job poller can flake after a successful job; output present ⇒ the run succeeded.
- **Prefer sequential runs.** Whether concurrent runs queue depends on what else is running on the deployment at that moment: they queue when other workloads hold the workers, and run as concurrent jobs when they don't. Under load, N parallel runs ran ~N× slower each; with an empty queue, concurrent runs completed at full speed. There is no serialization to rely on and no parallelism to count on — sequential stays the predictable default.
- **Sampling-rate warnings are expected on irregular data.** The bundle loosens the tolerance for Volve's irregular sampling (Δt 1–27 s); expect warnings, not failures.
- **Score with end-row labeling on `finish_timestamp` and exclude `INVALID_STATE`.** Predictions are keyed to the window-end timestamp; seam windows are invalidated.

## Local Setup (Path 1)

```bash
cd skills/atai-operational-state-monitoring-agent/references

# One dependency: the official Archetype AI client. Note the -r.
pip install -r requirements.txt

# A .env with BOTH variables (no default endpoint). The scripts use the nearest
# .env from where they run, upwards: one at the repo root serves every skill, and
# a closer one (e.g. here) wins. Variables exported in your shell beat any .env.
# Every .env is gitignored.
cat > .env <<EOF
ATAI_API_KEY=sk_...
ATAI_API_ENDPOINT=https://api.u1.archetypeai.app
EOF

# Either endpoint form works: the runner normalises. The client itself wants the
# /v0.5 suffix (it strips the version for the versionless /agents and keeps it for
# /v0.5/files), so a bare root passed straight to it breaks uploads with an empty
# `ApiError: {}` while bundle calls keep working. The model skills in this repo ship
# ATAI_API_ENDPOINT with /v0.5; one .env now serves both families.

python3 run_osm_agent.py                        # default sample slice, base bundle
python3 run_osm_agent.py --embeddings           # + Newton Omega embedding per window
python3 run_osm_agent.py --csv my_slice.csv     # your own prepared CSV
```

Expect **~1–2 min** for the ~4,185 step-1 windows of the sample slice when the worker queue is clear (verified: 107 s base, 70 s Embeddings, end-to-end) — and **~22–27 min** when workers are contended (also verified, same slice). The queue state isn't visible to you; the run's own audit events are the signal. The script resolves the bundle by name, streams audit events while it polls, and self-scores against the `_labels.csv` sidecar at the end.


---

# Path 2 — Build an OSM agent on your own data

> **Status: verified end to end on production, staging, Tokyo (2026-10-01) and dev (2026-09-30)**,
> with identical results on all four: the prep skill's sample below scored the same at every step
> (validation 0.7920, test 0.8463, delivery 0.7431). Built on the LARCO washing-machine data:
> [operational-state-monitoring-agent-example-larco-quickstart](https://github.com/archetypeai/operational-state-monitoring-agent-example-larco-quickstart)
> (19 short cycles) and the
> [full example](https://github.com/archetypeai/operational-state-monitoring-agent-example-larco) (199 cycles, the full study). A deployment without the optimizer job runner fails every optimization at
> once with `Pipeline 'optimizer-runner-cuda' has no active versions` (staging, prod and Tokyo
> did, until it was deployed on 2026-10-01).
> The scripts use a small stdlib HTTP helper ([`atai_http.py`](references/osm_lifecycle/atai_http.py))
> because the official `archetypeai` client doesn't cover these APIs yet; they'll move onto the
> client when it does.

```
role files ──► 1 Optimize ──► 2 promote ──► 2 test once (Evals) ──► 3 deliver (bundle + runs) ──► 4 score
  (prep skill)   fit + score     best trial     one-shot, pooled          predictions per file       held-back labels
                 on validation   → blueprint
```

Measured platform behaviour, with the numbers: [`references/platform-notes.md`](references/platform-notes.md).

## Step 0 — Role files

Path 2 starts from **role files**, built and checked by
[`atai-operational-state-monitoring-agent-data-prep`](../atai-operational-state-monitoring-agent-data-prep/SKILL.md)
to its contract, [`role-files.md`](../atai-operational-state-monitoring-agent-data-prep/references/role-files.md):

| role | what | used in |
|---|---|---|
| `library/<state>__library.csv` | **one file per state**, its windows as continuous pieces | training, every trial |
| `validation/<recording>__seg<N>.csv` | whole recordings, with `label` | scoring each trial (the `search_validation` subset) |
| `test/…` | whole recordings, with `label` | scored **once**, after promotion |
| `delivery/…` + `delivery_labels/…` | whole recordings **without** `label`; labels held back row for row | the deployed run, scored afterwards |

plus `manifest.json` (states, channels, every file and piece). The scripts read the states from
the manifest; nothing is hard-coded to one dataset.

Three contract rules decide whether a run works at all:
- **One library file per state:** an optimization's config must fit in **1 MiB** (~1,500 training
  files at most); over it the job stays `pending` forever with no error.
- **No time jumps in scored files:** one scored window across a jump fails the whole trial or eval.
- **Every state in every scored set:** an absent state scores F1 = 0.

**To try Path 2 on the prep skill's sample** (8 short LARCO washing-machine cycles,
already labelled and assigned to roles), build its role files first, locally, in a few
seconds:

```sh
cd skills/atai-operational-state-monitoring-agent-data-prep/references
pip install -r requirements.txt
S=sample_data/recordings.csv
python prepare.py --index $S --out prepared --hz 200 --abs-max 2.2
python check_states.py --index $S --prepared prepared --window 512
python build_roles.py --index $S --prepared prepared --out roles --window 1024 --per-state 200
python preflight_roles.py --roles roles --prepared prepared --index $S    # expect RESULT: PASS
```

What each step does and why is in the prep skill; for your own recordings, start there.

## Step 1 — Optimize: fit and score settings

Every step below runs as-is on the prep skill's sample (role files built as in Step 0):

```sh
cd references/osm_lifecycle
pip install -r requirements.txt
# ATAI_API_KEY + ATAI_API_ENDPOINT: the nearest .env from here upwards is used (e.g. one at the repo root)
R=../../../atai-operational-state-monitoring-agent-data-prep/references/roles
python optimize.py --roles $R --dry-run                                    # the plan: no uploads, no jobs
python optimize.py --roles $R --out out --name osm-sample --background     # tail -f out/optimize.log
```

Use your own `--name`: it prefixes every platform object (blueprints, bundles), so they're
easy to find and clean up.

`POST /agents/optimizations` with the `osm` blueprint's id (resolved from its key), the library
files as `training_examples` (ground truth `{"from": {"constant": "<state>"}}`, the state from the
filename), the search-validation files as `validation_examples` (ground truth
`{"from": {"column": "label"}, "downsampling": "last_record"}`: each window gets the label of its
last row), `"objective": "macro_f1"`, a `budget.max_trials`, and a search space:

```json
{"parameters": {
  "window_size": {"kind": "value",   "spec": {"type": "categorical", "values": [512]}},
  "step_size":   {"kind": "value",   "spec": {"type": "categorical", "values": [512]}},
  "k_neighbors": {"kind": "fitting", "spec": {"type": "categorical", "values": [5, 31]}},
  "metric":      {"kind": "fitting", "spec": {"type": "categorical", "values": ["cosine"]}},
  "weights":     {"kind": "fitting", "spec": {"type": "categorical", "values": ["uniform"]}}}}
```

- **`value`** parameters shape the data (windowing); **`fitting`** parameters shape the kNN.
- **The space is the product** of the values. `max_trials` equal to its size runs every point
  once (the script's default); below it, the platform **samples at random**, repeats possible.
- **Step > window** skips records; the script refuses it unless `--allow-gaps`.
- **Trials run one after another,** also across optimizations. Poll
  `GET /agents/optimizations/{id}` (status + `progress`), then list
  `GET /agents/optimizations/{id}/trials` (paged; pass `next_cursor` back verbatim).
- **Each completed trial carries a `metrics_report`:** `primary.value` = macro-F1 (also
  `objective_value`), and under `targets.state`: `class_names` (**alphabetical**),
  `per_class.{f1, precision, recall, support}` in that order, and the `confusion_matrix`
  (rows = true).

Writes `out/optimize_<id>.json`. **Expect, on the sample** (2 trials, ~3 min plus the 85 MB
upload; reproduced exactly by two runs on dev):

```
  #1   w=512   step=512   k=5   cosine uniform  completed macro-F1 0.7920  drain 0.87  fill 1.00  spin 0.57  wash 0.73  windows 386
  #2   w=512   step=512   k=31  cosine uniform  completed macro-F1 0.7496  drain 0.81  fill 0.99  spin 0.51  wash 0.68  windows 386
```

## Step 2 — Promote, then test once

```sh
python promote_and_test.py --roles $R --out out --name osm-sample --background   # tail -f out/test.log
```

1. **Promote** the best completed trial (by `objective_value`; `--trial otr_…` for another):
   `POST /agents/optimizations/{opt}/trials/{trial}/promote` with a `blueprint_key`, `name` and
   `description` returns a **blueprint with the setting and the fitted kNN attached**: a reusable
   model with no S3 artifact to build. Keys must be **unique per trial**: the script uses
   `<name>-w512-s512-cosine-k5-uniform-<last 6 of the trial id>`.
2. **Test once:** `POST /agents/evals` with `blueprint_id`, `name`, `emit_predictions: false` and
   one example per test file (ground truth as in validation). Poll `GET /agents/evals/{id}`. The
   `metrics_report` has the same shape as a trial's, **pooled over all files** (no per-file
   breakdown).

Every id is saved in `out/test_state.json`, so a rerun resumes instead of promoting or testing
again. Once you've seen the test number, the setting is frozen: tuning after it makes the test
optimistic. **Expect, on the sample** (~1 min for the eval):

```
promoted otr_… -> osm-sample-w512-s512-cosine-k5-uniform-<6 chars> (blp_…)
test, 1 files: macro-F1 0.8463  drain 0.87  fill 0.98  spin 0.60  wash 0.93  windows 1,076
```

## Step 3 — Deliver: your bundle, run in batches

```sh
python deliver.py --roles $R --out out --name osm-sample --background   # tail -f out/deliver.log
```

`POST /agents/bundles {"blueprint": "<your key>", "name": …}` creates the deployable agent; then
`POST /agents/bundles/{id}/run` with the delivery files as source connectors (`file_id`, not the
`fil_` uid), as in Path 1's Step 3. `--files-per-run N` splits the files into several runs (a
failed run then loses less); the default is one run. The platform processes a run's files one at
a time, and several runs one at a time too.

**Outputs don't name their inputs:** the script matches every output row to its delivery file by
`finish_timestamp` (recordings must not overlap in time) and writes `out/delivery/<file>.csv`.
Run ids are saved to `out/delivery/runs.json` as each run starts; `--resume` collects them.

**A run counts as completed only once its outputs are.** The platform can report a run
`completed` while its last output file is still being written (Path 1's Step 4), so the
script checks that each file's predictions reach that file's last timestamp (within 60 s),
downloading again every 15 s until they do. The log keeps the platform's status and the
script's apart, and `runs.json` records `completed` only after the check:

```
11:53:37   agt_… platform: completed; outputs 4 of 5 complete (…sport_40_2__seg0.csv still being written)
11:53:52   agt_… completed: all 5 outputs complete
```

Outputs still short 10 min after the platform said `completed` are logged as a warning and
recorded on the run (`error: outputs incomplete: …`); `--resume` downloads them again.
**Expect, on the sample** (~1 min): `1 of 1 files have predictions: 367 windows (0 invalid); 0
rows matched no file`.

## Step 4 — Score the delivery

```sh
python score.py --roles $R --out out
```

Each prediction is paired with the held-back label at its window's last row (the
`delivery_labels/` row whose timestamp equals `finish_timestamp`); `invalid` windows are left out
and counted. Macro-F1 is platform-style: over every state in the manifest, an absent state
counting 0. Reported pooled and per recording, next to the test number. **Expect, on the
sample:**

```
delivery: 1 recordings, 1 files, 367 windows scored; left out {'invalid': 0, 'no label at finish time': 0}
  all                                      macro-F1 0.7431  drain 0.80  fill 0.84  spin 0.71  wash 0.62  windows 367
next to the test: macro-F1 0.8463
```

The second machine scores lower than the test, as in the LARCO examples (quickstart: delivery
0.7707 against test 0.8617 over 5 and 3 cycles).

## Path 2 pitfalls

- **A stuck-`pending` optimization with no error** is almost always the 1 MiB config limit: too
  many training files. Pack one file per state.
- **A trial or eval that fails with `eval-mode test data must not contain windows a validation
  node rejected`** means a scored file has a time jump inside it.
- **Suspiciously low macro-F1 with one state at 0** usually means that state is missing from the
  scored set, not that the model can't see it.
- **Don't time trials by their `created_at`:** every trial gets the optimization's creation time.
- **Don't download outputs the moment a run says `completed`.** The last output file can still
  be being written: the quickstart's prod delivery came back with 2,507 of 3,178 windows, one
  file cut at 326 of 997. `deliver.py` waits for every file's predictions to reach its end; a
  hand-rolled client should do the same.
- **A key exported in your shell beats `.env`.** Keys are per deployment, so a staging key left
  exported while `.env` points at prod makes every upload fail with "Broken pipe": the server
  rejects the request before the body is sent. The scripts now check the key with one GET
  first, and say whether it came from the shell or `.env`; `unset ATAI_API_KEY
  ATAI_API_ENDPOINT` lets `.env` decide.
- **Keep upload concurrency low (~3):** parallel large uploads can saturate the uplink until DNS
  lookups fail. The helper retries uploads and GETs; it never retries a POST that may have
  reached the server.
- **A state the sensor can't see caps every score.** Check separability in data prep before
  fixing the state list (LARCO's `heating` had to be folded into `wash`).

## Cleanup (both paths)

Each run leaves an agent instance behind; the pre-packaged bundle is canonical and shared — **do not delete it**. Delete your own agent instances (and any bundle you created yourself, e.g. Path 2's delivery bundle):

```sh
curl -X DELETE -H "Authorization: Bearer $ATAI_API_KEY" \
  "$ATAI_API_ENDPOINT/agents/instances/$AGENT_ID"       # your run's instance
curl -X DELETE -H "Authorization: Bearer $ATAI_API_KEY" \
  "$ATAI_API_ENDPOINT/agents/bundles/$BUNDLE_ID"        # only a bundle YOU created
```

(`DELETE` on a running instance returns 409 — cancel first with
`POST /agents/instances/{id}/cancel`.)

**What Path 2 leaves behind, and what you can delete** (checked on dev and staging, 2026-10-01,
with a regular user key):

| object | delete | notes |
|---|---|---|
| agent instances (`agt_…`) | ✅ `DELETE /agents/instances/{id}` → 204 | one per delivery run |
| your delivery bundles (`bnd_…`) | ✅ `DELETE /agents/bundles/{id}` → 204 | delete the instances first |
| promoted blueprints (`blp_…`) | ❌ 403 `blueprint deletion requires the admin role` | ask an org admin; use a distinctive `--name` so they're easy to find |
| optimizations (`opt_…`), evals (`evl_…`) | not attempted | records of what was fitted and scored, small; keep them if your docs cite them |
| uploaded files | — | uploaded once per `--out` folder and reused through its `uploads.json` |


## Bring your own classifier artifact (legacy route)

Path 2's promote replaces this. For completeness: a bundle can also be created from the `osm`
blueprint with an externally fitted classifier, `"artifacts": {"fit-classifier":
"s3://<bucket>/<prefix>/my-classifier.safetensors"}` plus matching `values`
(`window_size`, `step_size`, `sample_rate_interval_tolerance`). The artifact must be an `s3://`
URI in the platform schema (`ids`/`vectors`/`weights` tensors + `index`/`classifier`
manifests), and `values.window_size` must match the window it was fitted with (a mismatch
silently degrades accuracy). Fitting it is out of scope; contact support@archetypeai.dev.

## File Layout

```
skills/atai-operational-state-monitoring-agent/
├── SKILL.md                  ← this file
├── references/
│   ├── run_osm_agent.py      ← Path 1: the managed flow on the official client (upload → resolve bundle → run → poll → download → score)
│   ├── requirements.txt      ← Path 1: archetypeai
│   ├── .env.example          ← copy to .env and fill in (both paths)
│   ├── sample_data/          ← Path 1: the Volve six-state eval slice + labels sidecar + attribution
│   ├── platform-notes.md     ← Path 2: measured platform behaviour, per deployment (prod, staging, dev)
│   └── osm_lifecycle/        ← Path 2 (stdlib HTTP until the client covers these APIs)
│       ├── atai_http.py        requests with retries, uploads with a cache, paging, polling
│       ├── common.py           the role-file manifest, examples, logging
│       ├── background.py       --background: nohup (+ caffeinate on macOS), output to a log
│       ├── optimize.py         Step 1: search space → Optimizations API → trials
│       ├── promote_and_test.py Step 2: best trial → blueprint → one eval
│       ├── deliver.py          Step 3: bundle from the blueprint → runs → outputs matched by time
│       ├── score.py            Step 4: last-record pairing against the held-back labels
│       └── requirements.txt    numpy, pandas, pyarrow (score.py)
└── tests/
    ├── test_references.py    ← Path 1: network-free unit tests (python -m unittest)
    └── test_lifecycle.py     ← Path 2: network-free unit tests
```

Run both: `python -m unittest discover -s skills/atai-operational-state-monitoring-agent/tests`
(`test_references.py` needs `archetypeai` installed; `test_lifecycle.py`'s scoring tests need pandas).
