# Sample data

A small subset of the **NASA IMS Bearing Dataset** (Set 2: 4 accelerometer
channels from a bearing run-to-failure experiment), used to make the Omega
embedding examples runnable out-of-the-box.

| File | Rows | Role | Timestamp range |
|------|-----:|------|-----------------|
| `bearing_healthy.csv`  | 2000 | n-shot library — healthy | ts 2,027,520–2,029,519 |
| `bearing_degraded.csv` | 2000 | n-shot library — degraded | ts 19,435,520–19,437,519 |
| `bearing_inference_subset.csv` | 7000 | **held-out test input** (no labels) | ts 5,000,000–5,003,499 + 18,500,000–18,503,499 |
| `bearing_labels_subset.csv` | 7000 | ground-truth labels for the test input (`timestamp,label`) | same as above |

Sensor rows are `timestamp,bearing_1,bearing_2,bearing_3,bearing_4`; the scripts
drop `timestamp` and embed the four vibration channels.

**Held-out, no leakage.** The test block timestamps (~5.0M healthy, ~18.5M
degraded) are disjoint from the n-shot shot files (~2.0M, ~19.4M), so
`classify_knn.py` evaluates on data the library never saw. The two test blocks
are each internally contiguous (one timestamp gap between them, which the
windower skips). The test input comes from the dataset's `bearing_inference.csv`
and its labels from `bearing_raw_labeled.csv` (a single clean transition:
healthy for ts < 18,432,000, degraded after).

## Attribution

- Dataset: [NASA IMS Bearing Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/),
  NASA Prognostics Center of Excellence — accelerometer data from a bearing
  run-to-failure experiment at the University of Cincinnati Center for
  Intelligent Maintenance Systems (IMS).
- Citation: Qiu, H., Lee, J., Lin, J., & Yu, G. (2006). "Wavelet Filter-based
  Weak Signature Detection Method and its Application on Rolling Element
  Bearing Prognostics." *Journal of Sound and Vibration* 289, 1066–1090.
- Publicly available from NASA's Prognostics Data Repository for research.

These two files are a curated subset of that dataset (a healthy window and a
degraded window). Replace them with your own sensor CSV — every script accepts
a positional path and auto-detects numeric channels.
