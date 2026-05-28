---
name: omega-local
description: >
  Generate time-series embeddings locally using a vendored Omega encoder
  checkpoint — no Newton API, no cloud roundtrip. Use when the user wants
  to run the Omega 1.3 encoder offline (PyTorch + a `.pt` checkpoint),
  build a custom classification or anomaly-detection pipeline on top of
  the embeddings (KNN, Isolation Forest, t-SNE/UMAP), or experiment with
  windowing/normalization strategies that the cloud Lens API doesn't expose.
  This skill covers checkpoint loading, single-channel input shape,
  windowing + padding, normalization choices, the "joint state" pattern
  for multi-sensor data, and common pitfalls.
  Do NOT use for the cloud Newton API (use newton-setup, newton-machine-state,
  etc.). Do NOT use for vision tasks (no image encoder is exposed locally).
---

# Omega Local Inference

Run the Omega 1.3 encoder directly in PyTorch from a checkpoint file. This is the offline counterpart to the cloud Machine State Lens — same encoder, but you control windowing, normalization, and the downstream classifier instead of using Newton's hosted KNN.

## When to Apply

- User has an Omega checkpoint (`.pt`) and wants to generate embeddings without hitting the API
- User is prototyping a custom downstream model (KNN with custom metric, Isolation Forest, neural classifier, etc.) on top of Omega embeddings
- User wants to inspect raw embeddings (PCA / t-SNE / UMAP) before deciding on a classifier
- User needs reproducible offline experiments (paper, benchmark, air-gapped environment)
- User is migrating an existing notebook from another encoder to Omega

**Use Newton Machine State Lens instead when**: the user just wants classification results, doesn't need to control the model, and is happy with a hosted KNN. The local path adds significant code complexity in exchange for control.

## What You Need

| Thing | Notes |
|-------|-------|
| Omega checkpoint | `.pt` file, e.g. `omega1_3_encoder_forecast_decoder.pt`. Place under `omega_core/weights/` (or wherever convenient). |
| `omega_core/` SDK | Vendored package containing `core/`, `embedding_lens.py`, `model_loader.py`, `utils.py`. Treat `omega_core/core/` as upstream code — do not edit it locally; sync changes through the source of truth instead. |
| Python 3.11 or 3.12 | 3.9 is EOL and several deps (numpy 2.x, pandas 3.x) require 3.10+. |
| Deps | `torch>=2.0`, `numpy`, `pandas`, `scikit-learn`, `einops`, `umap-learn`, `plotly` |

## Encoder Quick Facts

- **Embedding dim**: 768 (CLS token at position `[:, -1, :]` of encoder output)
- **Target series length**: 1024 timesteps
- **Channels**: **1** — Omega is single-channel. Multi-sensor data is processed per-sensor; embeddings are concatenated downstream (the "joint state" pattern below).
- **Encoder input format**: a dict `{"timeseries": tensor(B, 1, T), "timepoint_mask": tensor(B, T)}`

## Loading the Encoder

```python
from omega_core.embedding_lens import EmbeddingLens

lens = EmbeddingLens(
    checkpoint_path="omega_core/weights/omega1_3_encoder_forecast_decoder.pt",
    device=None,            # see device note below
    normalize_input=True,   # see "Normalization choices" below
)

embedding = lens.embed(timeseries)   # shape (1, 1024) → (768,)
```

`get_encoder_from_checkpoint(path)` (in `omega_core/model_loader.py`) is the lower-level loader if you need direct access to the `nn.Module`.

**Device note**: as shipped, `EmbeddingLens.__init__` only autodetects CUDA vs CPU — it doesn't pick MPS. On Apple Silicon, pass it explicitly. One-liner that works portably across CUDA / MPS / CPU machines without a wrapper class:

```python
import torch
device = "mps" if torch.backends.mps.is_available() else None
lens = EmbeddingLens(checkpoint_path=..., device=device, ...)
```

Passing `device=None` falls through to the lens's CUDA/CPU autodetect, so the same line works on a CUDA box, an Apple Silicon laptop, and a CPU-only CI runner.

## Single-Channel Constraint

Omega's `in_channels=1`. **Do not** stack sensors along the channel axis. The supported pattern is:

1. Treat each sensor column as an independent univariate series.
2. Run the encoder per-sensor per-window → one 768-dim embedding per (sensor, window).
3. Concatenate embeddings across sensors at the **window** level for the downstream model (joint state). The canonical code form of this pattern — joint-state concat + optional L2 / standardization / PCA into `(X, y, metadata)` arrays — lives in [`newton-data-prep`](../newton-data-prep/SKILL.md)'s `FeaturePreparer`.

Trying to feed `(B, n_sensors, T)` raises `ValueError: Model only supports single-channel input`.

## Sensor selection for directional time-series

If your data comes from a moving object (vehicle, drone, robot, satellite), be careful which kinematic channels you feed in. Raw world-frame `vx`, `vy`, `ax`, `ay` encode **heading** (which way the object is pointing) along with **behavior** (what it's doing). The encoder will happily learn the heading — meaning two windows of "hard right turn at 5 m/s" land far apart in embedding space if one was heading north and the other east. That's almost never what you want; downstream clustering then splits along compass direction instead of along behavior.

The fix is to choose **heading-invariant scalars only**:

| Channel | Invariant? | Why |
|---|---|---|
| `speed = hypot(vx, vy)` | ✓ | Magnitude of velocity; doesn't care about direction |
| `yaw_rate` (rad/s, angular velocity about vertical axis) | ✓ | Rate of heading change, frame-independent |
| `ax_body` (longitudinal accel in body frame) | ✓ | "How hard accelerating / braking" — relative to vehicle, not world |
| `ay_body` (lateral accel in body frame) | ✓ | "Cornering force" — relative to vehicle |
| `vx`, `vy`, `ax`, `ay` (world frame) | ✗ | Direction-dependent |
| `yaw`, `x`, `y` (absolute pose) | ✗ | Translate or rotate the whole log and you get different embeddings |

If you have world-frame accelerations and yaw, the body-frame projection is two lines:

```python
import numpy as np

cy, sy = np.cos(yaw), np.sin(yaw)
ax_body =  ax * cy + ay * sy   # longitudinal: + accelerating, − braking
ay_body = -ax * sy + ay * cy   # lateral: signed (left positive)
```

For clustering and cluster-mean summaries downstream, take the **absolute value** of signed turning channels (`|yaw_rate|`, `|ay_body|`) when computing per-cluster averages — "turning hard left" and "turning hard right" should land in the same cluster, distinguished from "going straight," not from each other. (Keep the signed series as the encoder *input*; only fold the sign at summary time.)

This isn't Omega-specific — it applies to any time-series encoder you'd substitute in — but it bites particularly hard on multi-session datasets that span different headings (e.g. cars driven in different cities, drones flown in different orientations).

## Windowing

For `T = window_size` rows of a single sensor:

- `window_size == 1024` → fast path, `lens.embed(window)` runs as-is.
- `window_size < 1024` → **left-pad** with zeros to 1024 and pass a `timepoint_mask` (1.0 for valid points, 0.0 for padding). Padding goes on the **left** (start of the sequence), valid points are the most recent. Instance normalization (when enabled) is computed only over valid points.
- `window_size > 1024` → not supported. Resample, downsample, or split.

⚠️ **`EmbeddingLens.embed()` does not accept a mask** — it builds an all-ones mask internally and the encoder will raise `ValueError: Make sure the time series length matches what the model expects. Expected 1024 but got <N>` for any other length. For short windows, bypass `lens.embed()` and call the encoder directly:

```python
import torch
import numpy as np

@torch.no_grad()
def embed_short(lens, windows, normalize=True):
    """Embed variable-length 1D windows by left-padding to 1024 + masking padding.

    `windows` is an iterable of 1D arrays, each of length <= 1024.
    Returns a (B, 768) numpy array.
    """
    SERIES_LEN = 1024
    windows = list(windows)
    B = len(windows)
    x = torch.zeros(B, 1, SERIES_LEN, dtype=torch.float32)
    mask = torch.zeros(B, SERIES_LEN, dtype=torch.float32)
    for i, w in enumerate(windows):
        a = np.nan_to_num(np.asarray(w, dtype=np.float32).reshape(-1), nan=0.0)
        n = min(a.shape[0], SERIES_LEN)
        x[i, 0, -n:] = torch.from_numpy(a[-n:])
        mask[i, -n:] = 1.0
    if normalize:
        valid = mask.unsqueeze(1)                                    # (B, 1, T)
        n = valid.sum(dim=2, keepdim=True).clamp(min=1.0)
        mean = (x * valid).sum(dim=2, keepdim=True) / n
        var  = ((x - mean) ** 2 * valid).sum(dim=2, keepdim=True) / n
        x = (x - mean) / (var.sqrt() + 1e-5)
        x = x * valid                                                # zero padding after normalize
    x, mask = x.to(lens.device), mask.to(lens.device)
    out = lens.encoder({"timeseries": x, "timepoint_mask": mask})
    return out[:, -1, :].cpu().numpy()                               # CLS token → (B, 768)
```

Then the per-session loop:

```python
read_indexes = list(range(0, num_rows - window_size + 1, step_size))
batch = [sensor_data[i : i + window_size] for i in read_indexes]
embs = embed_short(lens, batch, normalize=True)                       # (n_windows, 768)
```

A common starting point: `window_size=1024, step_size=512` (50% overlap). For short datasets (a few hundred rows), use smaller windows (e.g. `window_size=60, step_size=10`) with heavy overlap. Batching all windows of a session into one encoder call (as above) is also significantly faster than calling `lens.embed()` per window.

**Input rows must be temporally contiguous.** The encoder treats `sensor_data[i : i + window_size]` as `window_size` consecutive samples at the sensor's native cadence. If the source array was built by row-shuffling or `random.sample` (e.g. concatenating "the 2,000 fault-class rows" pulled from across a multi-day recording), each window concatenates physically distant moments into one input — the embedding describes a signal that doesn't exist. For n-shot library construction specifically, draw each class's rows as a single contiguous block from the raw recording; the cloud-side version of this recipe and the cross-repo evidence (+26pp on 3W from prep alone) is in [`newton-machine-state-batch/SKILL.md`](../newton-machine-state-batch/SKILL.md#recommended-n-shot-data-prep-contiguous--z-scored).

## Normalization Choices

The encoder was trained with instance-normalized inputs (per-window mean 0, std 1). You have two options:

### Option A — let the encoder normalize (`normalize_input=True`)

Each window is independently centered and scaled inside `EmbeddingLens.embed()`. Simple, matches training distribution. **Downside**: every window looks "self-similar" — a window with values 0–1 and a window with values 100–200 produce nearly identical embeddings. You lose cross-window context.

### Option B — normalize externally, set `normalize_input=False`

Z-score each sensor across the **whole dataset** with `StandardScaler`, then disable instance norm. The encoder sees globally normalized values, so a low-amplitude window stays "low" relative to the dataset.

```python
from sklearn.preprocessing import StandardScaler

df_norm = df.copy()
df_norm[sensor_cols] = StandardScaler().fit_transform(df[sensor_cols])

lens = EmbeddingLens(checkpoint_path=..., normalize_input=False)
```

This is usually the right choice for anomaly detection and any task where the absolute level of a signal matters. Fit the scaler **on train only** and reuse it for test/inference — fitting on the whole dataset leaks information.

## The "Joint State" Pattern

Per-window output from the encoder is one 768-dim vector per sensor. To get a single feature vector representing the whole machine at that window, **concatenate across sensors in a stable order**:

```python
import pandas as pd
import numpy as np

# df_emb: long format, one row per (sensor, window)
# columns: sensor, read_index, embedding (np.ndarray of length 768), ...
sensor_order = sorted(df_emb["sensor"].unique())   # alphabetical or explicit list
pivot = df_emb.pivot(index="read_index", columns="sensor", values="embedding")[sensor_order]
X = np.stack([np.concatenate(row) for row in pivot.values])
# X.shape == (n_windows, n_sensors * 768)
```

Two requirements:
- Same `sensor_order` at train, test, and inference time (or your KNN distances will be meaningless).
- Every window must contain every sensor — drop windows where a sensor is missing.

## Cross-session comparison: fit the projection jointly

A common analyst question is "do these two sessions look similar to Omega?" — e.g. two driving sessions, two production runs, two patient recordings. The naive approach is to project each session's joint-state with its own UMAP/t-SNE/PCA fit, then plot both. **This doesn't answer the question**: each fit is independent, so the (x, y) coordinates aren't comparable across sessions — a window that's "similar" in embedding space could land at completely different 2D coordinates because UMAP fits its own neighborhood graph each time.

Fit the projection on the **concatenated** joint-state matrix, then split the resulting coordinates back per session:

```python
# joint_a shape: (n_a, n_sensors * 768)
# joint_b shape: (n_b, n_sensors * 768)  — must have the same column count

import umap
combined = np.vstack([joint_a, joint_b])
reducer = umap.UMAP(n_components=2, random_state=0)
coords = reducer.fit_transform(combined)              # (n_a + n_b, 2)
coords_a, coords_b = coords[:joint_a.shape[0]], coords[joint_a.shape[0]:]
```

Both sets of coordinates now live in the same axis system, so spatial overlap means semantic similarity. Same idea for t-SNE and PCA (PCA's case is easier — `pca.fit(combined)` then `pca.transform(joint_a)` etc. — but the concat-and-split form works for all three).

If the two sessions have different sensor sets, intersect first and slice the joint matrix accordingly: `n_sensors * 768` must match between `joint_a` and `joint_b` or the concat will fail. Pin a deterministic `common_sensors = sorted(set(sensors_a) & set(sensors_b))` and reshape `joint = joint.reshape(n_windows, n_sensors, 768)[:, idx_of_common, :].reshape(n_windows, -1)`.

## Wide vs long format: getting your data into the right shape

Most of the patterns above assume **wide format** — one row per timestamp, one column per sensor. If your data arrives in **long format** (one row per `(entity, timestamp)` reading — common for meter readings, multi-tenant time-series databases, etc.), pivot first:

```python
# Long → wide: each meter/sensor becomes a column
df = df_long.pivot(index='timestamp', columns='meter_id', values='reading')
df = df.reset_index().sort_values('timestamp').reset_index(drop=True)
```

Three things to watch for:
1. **Pivoting introduces NaN** where some entities lack data at certain timestamps. Drop entities with high NaN coverage (>5–10%) before windowing — every window must contain every sensor or the joint-state pivot fails.
2. **The pipeline treats each entity as a sensor.** Joint state = `n_entities × 768`. For a building with 21 sub-meters, that's 16,128 dims — fine for IsolationForest/KNN, slow for RandomForest with `n_estimators=200+`.
3. **Loading directly from compressed CSVs**: `pd.read_csv('archive.zip', usecols=['needed', 'cols'])` works when the zip contains exactly one CSV. `usecols` filters at the parser level — much faster than dropping columns post-load on a multi-GB file. Add `low_memory=False` to suppress the mixed-dtype warning.

If your data is **per-block** (e.g. multiple jobs/recordings concatenated, where each block is a separate continuous time series), iterate over blocks and offset `read_index` so the joint-state pivot's read_index is globally unique. See `gripper_anomaly.ipynb` for the full pattern.

## Preflight: data-quality checks before embedding

The Omega encoder works best on **monotonically increasing timestamps with a roughly constant sampling rate**. Deviations don't always break the pipeline, but they're worth flagging early — and the same checks catch unrelated data issues that *do* break things downstream.

Pattern adapted from [`archetypeai/omega-1-4-preflight`](https://github.com/archetypeai/omega-1-4-preflight) (that repo targets the cloud `machine-state-job-pipeline` for `omega_1_4_base`; the static checks transfer cleanly to local 1.3 inference). For the *active cleanup* counterpart — same diagnostics but with gap-aware blocking + imputation that produces a cleaned dataframe instead of just a report — see [`newton-data-prep`](../newton-data-prep/SKILL.md)'s `DataPreprocessor`. Run once after loading, before any transformations:

| Check | What it catches | Severity |
|---|---|---|
| `schema` | Time column missing, non-numeric features | FAIL |
| `timestamp` | Non-monotonic, duplicates, gaps where `max(Δt) > N × median(Δt)` | FAIL/WARN |
| `missing_values` | NaN columns (worst-offender first) | WARN |
| `constant_columns` | Zero-variance sensors that should be dropped | WARN |
| `feature_scale` | Range gap > N decades — flags need for StandardScaler | WARN |
| `class_balance` | Severe (>85%) or moderate (>70%) imbalance, with majority baseline | WARN |
| `window_vs_sampling` | Translates `window_size × median(Δt)` into human time so you can sanity-check it covers a typical fault duration | INFO |

Sketch:

```python
def preflight(df, time_col, sensor_cols=None, label_col=None, block_col=None,
              window_size=None, gap_threshold_x=5.0, scale_decade_warn=3.0):
    """Returns list of (level, name, message). Use block_col for per-block
    timestamp checks (e.g. when each job is a separate recording)."""
    results = []

    # 1. schema (time col, numeric sensors)
    # 2. timestamp per block: monotonic, unique, max(Δt)/median(Δt) ≤ gap_threshold_x
    # 3. missing values per column
    # 4. constant columns: std < 1e-12
    # 5. feature-scale heterogeneity: log10(range) span > scale_decade_warn
    # 6. class balance with majority baseline
    # 7. window_size × median(Δt) → human-readable duration

    return results
```

Run with `block_col='job_id'` (or whatever splits your recordings) when the dataset has multiple independent blocks — windowing must not cross block boundaries, and per-block timestamp checks catch issues a global check would hide.

## Pipeline Skeleton

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from omega_core.embedding_lens import EmbeddingLens

# 1. Load + clean
df = pd.read_csv("data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# 2. Drop constant sensors (std=0 → no information; can also poison normalization)
sensors = [c for c in df.columns if c not in ("timestamp", "label")]
sensors = [c for c in sensors if df[c].std() > 0]

# 3. Normalize (fit on train only in production)
scaler = StandardScaler().fit(df[sensors])
df[sensors] = scaler.transform(df[sensors])

# 4. Embeddings, per sensor per window
lens = EmbeddingLens(checkpoint_path="weights/omega.pt", normalize_input=False)
W, S = 60, 30   # window_size, step_size — pick to fit your dataset

records = []
for col in sensors:
    series = df[col].values.astype(np.float32)
    for read_idx in range(0, len(series) - W + 1, S):
        window = series[read_idx:read_idx + W]                    # shape (W,)
        emb = lens.embed(window.reshape(1, -1))                   # → (768,)
        label = df["label"].iloc[read_idx + W - 1]                # label of last point
        records.append({"sensor": col, "read_index": read_idx,
                        "embedding": emb, "label": label})

df_emb = pd.DataFrame(records)

# 5. Joint state
pivot = df_emb.pivot(index="read_index", columns="sensor", values="embedding")[sensors]
X = np.stack([np.concatenate(row) for row in pivot.values])
y = df_emb.groupby("read_index")["label"].first().reindex(pivot.index).values

# 6. Out-of-time split (no shuffling — preserve temporal order)
n_train = int(0.7 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# 7. Downstream model
clf = KNeighborsClassifier(n_neighbors=3, metric="manhattan", weights="distance")
clf.fit(X_train, y_train)
print("test acc:", clf.score(X_test, y_test))
```

For anomaly detection, swap step 7 for `IsolationForest(contamination="auto").fit(X_train)` and don't pass `y_train` — labels are only used for evaluation. **Caveat**: Isolation Forest assumes anomalies are *rare and structurally different*. If your "anomaly" class is 30%+ of data and shares structure with normal (e.g. same machine, different operating mode), IF will collapse to predicting "all normal." That's a classification problem, not an anomaly-detection one — stick with the supervised baseline.

## Unsupervised cluster discovery: HDBSCAN + auto-labels

Once you have a joint-state matrix, clustering surfaces "what behaviors did the encoder find?" without supervision. **HDBSCAN** is the right default — it finds variable-shape groups, marks borderline points as **noise** instead of forcing them into clusters, and doesn't require choosing `k`.

### Cluster on PCA-50, not raw or UMAP-2D

Three obvious options for clustering input, only one is right:

| Input | Dim | What happens |
|---|---|---|
| Raw joint state | `n_sensors × 768` (e.g. 3072) | Density too sparse — HDBSCAN finds 1–2 clusters and dumps everything else as noise |
| **PCA-50 of joint state** | 50 | **Sweet spot** — preserves ~95–98% variance, density is well-defined |
| UMAP-2D coords | 2 | Over-collapsed — HDBSCAN finds many tiny "clusters" that are artifacts of UMAP's local-neighborhood objective |

PCA-50 (or wherever ~95–98% explained variance lands) is the natural default. It's also exactly what you'd hand IsolationForest, so caching once and reusing is natural.

```python
from sklearn.cluster import HDBSCAN
hdb = HDBSCAN(min_cluster_size=20, min_samples=5).fit(stacked_pca50)
labels = hdb.labels_                       # cluster_id per window, -1 = noise
```

`min_cluster_size=20` is a starting point — bump it up for noisier data, down if you have <500 windows total.

### Auto-label clusters from the *original signals*, not the embeddings

HDBSCAN gives you anonymous cluster IDs (`0, 1, 2, …`). To make them human-readable, compute per-cluster averages on the **original windowed sensor data** (embeddings are uninterpretable for this) and run a small rule table:

```python
def label_from_stats(speed, yaw_abs, ax, ay_abs) -> str:
    """First match wins — ordering matters."""
    if speed < 0.7:        return "stopped"
    if yaw_abs > 0.18:     return "sharp turn"
    if yaw_abs > 0.08:     return "turning"
    if ax < -1.0:          return "braking"
    if ax > 1.0:           return "accelerating"
    if speed > 11.0:       return "fast cruising"
    if speed > 6.0:        return "cruising"
    return "creeping"

# stacked_stats: (n_windows, n_summary_features) — e.g. [speed, |yaw_rate|, ax_body, |ay_body|]
labels_text = []
for cid in sorted(set(labels) - {-1}):
    means = stacked_stats[labels == cid].mean(axis=0)
    labels_text.append(label_from_stats(*means))
```

Tailor the rule table to the domain (vehicles → kinematic states; chillers → on/off/transient; meters → idle/peak/ramp). Keep it short and ordered — it's a labeling heuristic, not a classifier.

### Disambiguate duplicate labels with A/B/C suffixes

The rule table is intentionally coarse — multiple clusters often land on the same label. Disambiguate with a suffix in cluster_id order:

```python
seen, counts = {}, {l: labels_text.count(l) for l in set(labels_text)}
for i, lbl in enumerate(labels_text):
    if counts[lbl] > 1:
        idx = seen.get(lbl, 0)
        labels_text[i] = f"{lbl} {chr(ord('A') + idx)}"
        seen[lbl] = idx + 1
```

### Why duplicate labels are a feature, not a bug

The rule table summarizes each cluster by ~4 averaged numbers. The encoder saw the full window — much more nuance. So you'll often see e.g. three `stopped` clusters that look identical on the summary stats (speed=0, yaw=0, accel≈0) but live in three distinct regions of latent space. **That's the encoder telling you it found texture inside the window that your summary flattens** — brake-release pattern, micro-yaw of settling, approach style, …

A common finding: same-label clusters frequently align with a **categorical attribute the encoder was never told about** — one city per `stopped` variant, one operator per `idle` variant, one production line per `peak` variant. Cross-tabulate cluster_id against the categorical to check:

```python
import pandas as pd
df = pd.DataFrame({"category": category_per_window, "cluster": labels})
pd.crosstab(df["cluster"], df["category"])  # one column dominates per row → fingerprint
```

That's often the headline result — "the encoder learned to distinguish how each `<entity>` does `<common behavior>`, without ever being told what entity it was."

### Treat noise as honest signal

HDBSCAN typically labels 30–70% of windows as `-1` (noise), especially on smooth-manifold data where most windows are transitions between states. **This is not a parameter problem to tune away.** The encoder formed tight identities for a handful of behaviors and a smooth manifold for the rest; the rest doesn't have well-defined cluster membership, and saying so honestly is more useful than forcing it. Render noise in transparent gray on the scatter and move on.

## Caching: separate the joint state from the projection

Encoder forward passes are the expensive step (~10s per session on MPS for a few hundred windows × 25 sensors). UMAP/t-SNE/PCA on the joint-state matrix is comparatively cheap but still seconds. If your app lets users switch projections or compare different session pairs, cache the two stages separately:

- `joint_state_<session>_<window>_<step>.pkl` — the (n_windows, n_sensors × 768) matrix + per-window metadata (timestamps, raw signals, etc.). Heavy. Computed once per (session, window, step).
- `embedding_<session>_<window>_<step>_<projection>.pkl` — the 2D/3D coords. Light. Computed once per (session, window, step, projection).
- `compare_<primary>_<overlay>_<window>_<step>_<projection>.pkl` — both sets of coords from a joint fit. Computed once per session pair × params; reuses the two joint-state caches.

This way, switching projection on the same session only re-runs the reduction (a few seconds), not the encoder. And comparing session A overlaid on B reuses the joint-state caches built for single-session views of A and B.

Bump a `CACHE_VERSION` integer baked into the cache key whenever you change the shape of what you're persisting — invalidates old caches automatically without manual `rm -rf cache/`.

## Building an embedding-viewer frontend

If you're wrapping Omega embeddings in a React/Svelte/etc. UI — playback dashboard, trajectory plot, anomaly browser — **read [`DESIGN.md`](../../DESIGN.md) at the root of this repo before writing any CSS**. The Archetype design system (Tailwind v4 + `@archetypeai/ds-lib-tokens` + Geist sans/mono + OKLCH palette + dark-first) is the expected visual language for these demos. `archetypeai-swat-demo` and `archetypeai-wifi-demo` are the reference implementations — pattern-match them for layout (Menubar, Card, Badge with `good`/`warning`/`critical` variants, mono numbers, sharp 2px radii). Setting this up at the start is much cheaper than retrofitting later.

### Click-to-replay anomaly windows

If IsolationForest is in the pipeline, the most useful interaction in an embedding viewer is **click an anomaly point → auto-replay that 8-second window**. Concretely:

1. Each anomaly point on the embedding scatter gets a click handler. If the demo also has a map / route / signal track, mirror the anomaly markers there with the same click behavior so the user can drive from either side.
2. The click jumps playback to the **start** of the window (not the centre — users want the lead-in).
3. Playback runs through to the end of the window at the user's current speed setting (1×, 5×, 50×, …), then **auto-pauses**.
4. One click → one anomaly replayed in full, no scrubbing.

The window-aware auto-pause is the bit that makes this feel like a tool rather than a toy — without it, users have to mash spacebar to stop with no way to know where the window ended.

Give `is_anomaly` points a distinct color (e.g. red) and a slightly larger marker on the scatter so they're easily targetable, and render them **above** the normal cluster coloring so they don't disappear into a dense group. Their hover tooltip is a good place to show `anomaly_score` ("higher = more normal" in sklearn's IsolationForest, so sorting ascending gives the most-anomalous-first).

## Distribution without weights: precompute → static JSON

The Omega checkpoint is not always redistributable. To share an embedding-viewer demo with collaborators who don't have the `.pt`, run the full pipeline once on a privileged machine and emit JSON:

```python
# scripts/precompute.py — run once, writes one JSON per (session × projection × overlay)
def main():
    embedder = OmegaEmbedder()
    for session in sessions:
        for proj in ("umap", "tsne", "pca"):
            r = compute_embedding(embedder, session, window_size=60, step_size=10,
                                  projection=proj)
            write_json(out_dir / f"{session}_w60_s10_{proj}.json", to_payload(r))
    # plus: one sessions.json index, one cmp_<a>_<b>_<...>.json per ordered pair
```

Two design choices that make this nice:

1. **Deterministic filenames built from UI state**: the frontend never needs a per-load index call — it constructs the URL from current selections (`/embeddings/<session>_w60_s10_<projection>.json` or `/embeddings/cmp_<primary>_<overlay>_w60_s10_<projection>.json`). One `sessions.json` lists what's available.

2. **Match the live API's response shape exactly**: the precomputed JSON files should be structurally identical to whatever your FastAPI/Flask endpoint returns. Then the frontend code path is one `fetch()` regardless of whether the backend is live or fully static, and the dev-time live path stays useful for iteration.

Total payload for a typical "viewer of N sessions" demo: ~100KB per (session, projection) JSON × 3 projections × N sessions (×2 if you also precompute every ordered pair for compare mode). For N=2 that's ~1.5MB — fits comfortably in a static-host repo and loads instantly. Beyond ~10 sessions, consider on-demand fetch from object storage rather than committing the JSONs to git.

The frontend then runs as a pure static site (Vercel/Netlify/GitHub Pages); the encoder + checkpoint never leave the precompute machine.

## Beyond a baseline: threshold tuning + model selection

The default KNN baseline often underperforms on the rare/expensive class — even when overall accuracy looks fine. Two leverage points before reaching for a different architecture:

### 1. Decision-threshold tuning

sklearn defaults to threshold 0.5 on `prob_1`. For binary fault detection, a lower threshold catches more faults at the cost of more false alarms. Sweep and pick:

```python
prob_fault = clf.predict_proba(X_test)[:, 1]
for t in np.linspace(0.05, 0.95, 19):
    y_pred = (prob_fault >= t).astype(int)
    # compute precision, recall, F-beta — pick the t that fits your cost ratio
```

### 2. F-beta scoring

When recall on the rare class matters more than precision, score with F-beta instead of F1:

```python
def fbeta(precision, recall, beta=2.0):
    if precision + recall == 0: return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)
```

`beta=1` is F1 (balanced). `beta=2` weights recall 2× as much as precision — appropriate when missing a fault is much more costly than a false alarm. `beta=4` heavily favors recall; `beta=0.5` favors precision.

### 3. Multi-model auto-select

Sweep across model variants, pick the F-beta-optimal threshold per model, then pick the F-beta-best overall:

```python
candidates = {
    'KNN distance':    KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance'),
    'KNN uniform':     KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='uniform'),
    'LogReg balanced': LogisticRegression(class_weight='balanced', max_iter=2000),
    'RF balanced':     RandomForestClassifier(class_weight='balanced', n_estimators=200,
                                              random_state=42, n_jobs=-1),
}
# For each: fit, sweep thresholds, record F-beta-best. Pick winner across models.
```

`class_weight='balanced'` (LogReg, RF) automatically reweights losses inversely to class frequency. KNN doesn't support `class_weight` natively — `weights='uniform'` vs `'distance'` is the closest knob, and threshold sweeping does more for KNN than weight choice.

Cross-cutting observation: on hard datasets, **RF balanced + low threshold** consistently dominates F2. KNN distance is a strong default for clean datasets where threshold ≈ 0.5 already works.

## Visualizing temporal structure: trajectory plots

The default visualization for embeddings is t-SNE/UMAP colored by **class** (fault vs normal) or a **categorical metadata column** (month, day-of-week). That answers "do classes separate?" but hides temporal dynamics.

A **trajectory plot** colors the same 2D scatter by **chronological position** (viridis: purple for earliest, yellow for latest). It surfaces patterns that categorical coloring can't:

| Pattern | What it looks like | Implication |
|---|---|---|
| Closed loop | Year cycles back, December points sit near January's | Steady-state seasonal behavior; year is repeatable |
| One-way drift | Trajectory goes purple → yellow without returning | Capacity additions, sensor calibration drift, system wear |
| Distinct phases | Sharp color boundaries between regions of the scatter | Phase transitions (normal → degrading → fault) |
| Tangled center, smooth edges | Most points cluster, a few sweep around the perimeter | Anomalies are temporal outliers, not just point outliers |

Pattern adapted from [`archetypeai/omega_tools/embedding_tools/plot_utils.py`](https://github.com/archetypeai/omega_tools) (matplotlib `plot_trajectory`). For Plotly:

```python
def plot_trajectory(df_reduced, sort_by=None, connect=False, point_size=6):
    """Color-by-time scatter on the output of a 2D reduction."""
    import plotly.graph_objects as go
    if sort_by is None:
        sort_by = 'timestamp' if 'timestamp' in df_reduced.columns else 'read_index'
    df = df_reduced.sort_values(sort_by).reset_index(drop=True)
    df['t_step'] = range(len(df))
    fig = go.Figure()
    if connect:
        fig.add_trace(go.Scatter(x=df['dim1'], y=df['dim2'], mode='lines',
                                  line=dict(color='lightgray', width=0.5), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=df['dim1'], y=df['dim2'], mode='markers',
        marker=dict(color=df['t_step'], colorscale='Viridis', size=point_size,
                    showscale=True, colorbar=dict(title='time step')),
        showlegend=False,
    ))
    return fig
```

**Per-block trajectories**: when the dataset has multiple independent recordings (jobs, sessions, runs), facet the trajectory plot — one per block, each viridis-colored from start to end of that block. Reveals whether **each individual run** evolves consistently. Inside a `for` loop, you have to call `display(fig)` explicitly — Jupyter only auto-displays the *last* expression of a cell, not loop returns.

**`connect=True` caveat**: a polyline through consecutive points helps for ≤200 points (you can see the path); for thousands of points it becomes a tangled mess that obscures the scatter. Default to `False` for whole-year datasets, `True` for short recordings.

## Common Pitfalls

### 1. Constant-zero or all-NaN sensor columns

Sensors like `refVelocityBack` that are 0.0 across the whole dataset have `std=0`. Two failure modes:

- With `normalize_input=True`, the encoder's instance norm divides by `std + 1e-5`, gives zero output, and produces a degenerate constant embedding. Not strictly an error, but adds zero signal and inflates feature dimensionality.
- `StandardScaler` is robust (it returns zeros for constant features), but **manual `(x - mean) / std`** is not — that gives `0 / 0 = NaN`, which propagates into the encoder and out into the embeddings, then into t-SNE/KNN/IsolationForest, all of which raise `ValueError: Input X contains NaN`.

Real OBD-II / SCADA / lab data also routinely contains **all-NaN "lookalike twin" columns** — e.g. an OBD-II logger emits both `hv_system_voltage` (PID not supported by the vehicle, 100% NaN) and `hv_voltage` (PID supported, real readings). Both end up in the wide-format CSV. Picking the wrong column means a flat zero/NaN strip in your visualization and degenerate embeddings for that sensor.

**Fix**: filter on both std *and* NaN-fraction up front, and sanity-check `.describe()` on a couple of representative sessions before pinning a sensor list:

```python
def numeric_sensor_columns(df, max_nan_frac=0.5):
    cols = []
    for c in df.columns:
        if c in metadata_cols: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().mean() > max_nan_frac: continue        # all-/mostly-NaN → drop
        std = s.std(skipna=True)
        if std is None or pd.isna(std) or std <= 0: continue # constant → drop
        cols.append(c)
    return cols
```

### 2. NaN propagation is silent until the classifier

The encoder happily produces NaN embeddings for NaN input. Downstream visualizers and sklearn estimators reject NaN with a long traceback **far from the source**. Sanity-check after the embedding step:

```python
n_nan = sum(1 for e in df_emb["embedding"] if np.isnan(e).any())
assert n_nan == 0, f"{n_nan} NaN embeddings — check for NaN columns in df"
```

If you see exactly `n_windows` NaN embeddings, exactly one sensor column is bad — find it with `df.isna().sum()`.

### 3. Mixed-up sensor sets across runs

The joint state vector's dimensionality is `n_sensors * 768`. If you train a classifier on `X.shape == (N, 14*768)` and at inference time the data has 13 sensors (or 15), the classifier will fail with a shape mismatch — or worse, silently give garbage results if dimensions match by coincidence.

Pin `sensor_order` explicitly and assert `len(sensor_order)` at every stage. A common mistake is filtering sensors based on `df.columns` after a DataFrame mutation that added a column (e.g. `block_id`, `imputed`, or a leftover `label_num` from a copy-pasted notebook cell).

### 4. Notebook state corruption

Embeddings are expensive enough that people resist re-running cells. The result is `df_emb` from a previous configuration sitting in the kernel while downstream cells use new parameters. **When debugging unexpected NaN or shape errors, restart the kernel and Run All before assuming the code is wrong.**

### 5. window_size > 1024

Not supported — the encoder's positional embeddings are fixed at 1024. Resample (`df.resample("5min").mean()`) or split into multiple windows and aggregate embeddings.

### 6. Channel-axis confusion

If you call `lens.embed(np.random.randn(2, 1024))` (2 channels), you get `ValueError: Model only supports single-channel input`. Reshape to `(1, 1024)` per call, or pass a list of `(1, 1024)` arrays for batched inference.

### 7. Padding direction

Short windows are **left-padded**, not right-padded. The valid points are the *end* of the sequence. If you implement custom padding, match this convention or your embeddings won't be comparable to the rest of the pipeline.

### 8. OOT split breaks on single-transition datasets

Out-of-time (chronological) splitting assumes both classes appear throughout time. When the fault region is one contiguous block (e.g. rows 0–93 are fault, 94–246 are normal), a 70/30 split puts all of one class in train and all of the other in test:

- `y_test` has only one unique value — precision/recall on the missing class are undefined.
- Macro F1 may report 1.0 if the model trivially predicts the only present class — looks great but is uninformative.

The deceptive symptom: the pipeline runs without error and reports a strong-looking score. Always check `np.unique(y_train, return_counts=True)` and `np.unique(y_test, return_counts=True)` after splitting.

**Fix**: use `mode='random'` (with `random_state` for reproducibility) when the dataset has a single fault episode. OOT remains correct for multi-block datasets where each block is internally mixed (e.g. one job containing both fault and normal samples) — the issue is *temporal class structure*, not OOT vs random in general.

### 9. Tz-aware timestamps lose their tz through `.values`

If your source data has tz-aware timestamps (`+05:30` from ISO 8601, or UTC from `pd.to_datetime(..., utc=True)`), and you build a metadata DataFrame using `.values` on a tz-aware Series:

```python
metadata['timestamp'] = some_series.values   # ⚠ silently strips tz
```

`numpy datetime64` has no timezone support, so `.values` drops the tz. Downstream tz-aware comparisons then fail with:

```
TypeError: Cannot compare tz-naive and tz-aware datetime-like objects
```

The symptom is deceptive: the embedding flow works, the anomaly detection runs, then a "zoom into the raw signal around this anomaly" cell crashes because it does `df['Timestamp'] >= some_metadata_timestamp` and the two sides have incompatible tz-awareness.

**Fix**: keep timestamps as a pandas Series. Use `.reset_index(drop=True)` for positional alignment with sibling array columns instead of `.values` for stripping:

```python
metadata['timestamp'] = some_series.reset_index(drop=True)
```

`pd.DataFrame` accepts mixed Series/array values when their lengths match. Or, convert both sides to a consistent tz at the comparison point.

## Suggested Module Layout

A clean way to organize a project around these patterns is one class per file, each owning a single stage of the pipeline:

| Class | Responsibility |
|-------|----------------|
| `DataPreprocessor` | Splits a multivariate series with NaN gaps into continuous "blocks", imputes short gaps |
| `EmbeddingGenerator` | Wraps `EmbeddingLens`, handles per-sensor windowing, padding+mask for short windows, label propagation |
| `FeaturePreparer` | Pivots to joint-state matrix, normalization (L2/standardize), optional PCA |
| `DataSplitter` | Out-of-time or random train/test split |
| `MachineState` | Unified KNN (classify) / Isolation Forest (anomaly) with consistent prediction DataFrame |
| `Visualizer` | PCA/t-SNE/UMAP scatter, raw signals, predictions-over-time with label-shaded background |

Each class consumes the output of the previous one. The whole pipeline fits in ~1000 lines and is easy to swap individual stages on (e.g. trying a different windowing strategy without touching the classifier).

## Comparing Local vs Newton Machine State Lens

| Concern | Local Omega | Newton Machine State Lens |
|---------|-------------|---------------------------|
| Where the encoder runs | Your machine (CPU/MPS/CUDA) | Newton cloud |
| Auth | None — checkpoint file | Bearer token, see `newton-setup` |
| Classifier | Your choice — KNN, IF, MLP, anything | Hosted KNN over n-shot focus files |
| Window/step config | Full control via your loop | `csv_configs.window_size` / `step_size` |
| Normalization control | Pre-encoder StandardScaler or instance norm | Instance norm only |
| Multi-sensor handling | Joint-state concat (your code) | `data_columns` + per-stage sessions, see `newton-machine-state` |
| Cost | Free (compute only) | Per-inference billing |
| Real-time streaming | You build it | SSE consumer built-in |
| Reproducibility | Pinned by checkpoint hash + seed | Tied to API version |

If you need both — e.g. local prototyping followed by cloud deployment — the same `data_columns`, `window_size`, and `step_size` settings transfer cleanly. The encoder is the same on both sides; only the classifier and orchestration differ.
