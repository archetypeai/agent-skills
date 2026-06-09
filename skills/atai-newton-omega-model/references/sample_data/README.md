# Sample data

A small subset of the **NASA IMS Bearing Dataset** (Set 2: 4 accelerometer
channels from a bearing run-to-failure experiment), used to make the Omega
embedding examples runnable out-of-the-box.

| File | Rows | Role | Timestamp range |
|------|-----:|------|-----------------|
| `bearing_healthy.csv`  | 2,000 | n-shot library — healthy | ts 2,027,520–2,029,519 |
| `bearing_degraded.csv` | 2,000 | n-shot library — degraded | ts 19,435,520–19,437,519 |
| `bearing_inference.csv` | 1,040,000 | **held-out test input — sensors only, NO `label` column** | ts 3,000,000–3,519,999 + 18,500,000–19,019,999 |
| `bearing_labels.csv` | 1,040,000 | ground-truth labels (`timestamp,label`), used only for scoring | same as above |

Shot/inference rows are `timestamp,bearing_1,bearing_2,bearing_3,bearing_4`; the
scripts drop `timestamp` and embed the four vibration channels.

**Held-out, no leakage — two ways.**
1. **Disjoint timestamps.** The test blocks (~3.0M healthy, ~18.5M degraded) are
   disjoint from the n-shot shot files (~2.0M, ~19.4M) — verified by set
   intersection (0 shared timestamps), so the library never saw the test data.
2. **No label in the input.** `bearing_inference.csv` has no `label` column at
   all, so the label cannot leak into what gets embedded; ground truth lives
   only in `bearing_labels.csv` and is read separately for scoring.

`bearing_inference.csv` holds ~1000 non-overlapping 1024-step windows (500
healthy + 500 degraded). Its sensor values are identical to the dataset's
`bearing_inference.csv` for the same timestamps; labels follow
`bearing_raw_labeled.csv` (a single clean transition: healthy for
ts < 18,432,000, degraded after).

## Attribution

These files are a small curated subset of the **NASA IMS Bearing Dataset**
(Set 2: a run-to-failure experiment with 4 accelerometer channels), produced by
the University of Cincinnati Center for Intelligent Maintenance Systems (IMS)
and distributed via the NASA Prognostics Data Repository.

**Dataset citation** (cite when you use this data):

> J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007).
> *Bearing Data Set*, IMS, University of Cincinnati. NASA Prognostics Data
> Repository, NASA Ames Research Center, Moffett Field, CA.
> https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

**Methodology paper:**

> Qiu, H., Lee, J., Lin, J., & Yu, G. (2006). "Wavelet Filter-based Weak
> Signature Detection Method and its Application on Rolling Element Bearing
> Prognostics." *Journal of Sound and Vibration* 289, 1066–1090.
> https://doi.org/10.1016/j.jsv.2005.03.007

**Terms:** NASA Prognostics Data Repository datasets are publicly available;
NASA requests that the dataset citation above be included in any publication or
redistribution. (US-government-produced data; no additional license required.)

The two shot files are a healthy and a degraded window; the inference/labels
subsets are held-out slices of the same experiment. Replace any of them with
your own sensor CSV — every script accepts a positional path and auto-detects
numeric channels.
