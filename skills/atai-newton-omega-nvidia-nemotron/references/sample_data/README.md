# Sample data

A small subset of the **NASA IMS Bearing Dataset** (Set 2: 4 accelerometer
channels from a bearing run-to-failure experiment), the same data the
[`atai-newton-omega-model`](../../../atai-newton-omega-model/) skill ships.

| File | Rows | Role |
|------|-----:|------|
| `bearing_healthy.csv`  | 2,000 | n-shot library — healthy (ts 2,027,520–2,029,519) |
| `bearing_degraded.csv` | 2,000 | n-shot library — degraded (ts 19,435,520–19,437,519) |
| `bearing_incoming.csv` | 512 | two held-out 256-row windows to classify and explain: rows 0–255 healthy (ts 3,000,000–3,000,255), rows 256–511 degraded (ts 18,500,000–18,500,255) |

`bearing_incoming.csv` is cut from the Omega skill's held-out `bearing_inference.csv`,
so its timestamps are disjoint from both shot files. Rows are
`timestamp,bearing_1,bearing_2,bearing_3,bearing_4`; the scripts drop `timestamp`.

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

**Terms:** NASA Prognostics Data Repository datasets are publicly available;
NASA requests that the dataset citation above be included in any publication or
redistribution. (US-government-produced data; no additional license required.)
