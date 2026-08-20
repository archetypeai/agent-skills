# Sample data — Volve six-state eval slice

- `volve_states_opt_slice_04.csv` — 4,200 rows of prepared drilling telemetry
  (9 sensor channels, globally z-scored), one contiguous segment per state:
  `drilling · reaming · off_bottom · in_slips · trip_in_slips · shut_in`.
- `volve_states_opt_slice_04_labels.csv` — ground-truth sidecar (the same
  columns plus a `label` column; scoring reads only `DATE_TIME` + `label`),
  used only for scoring, never uploaded.

The slice is **held out**: it was never used to fit or tune the classifier
pinned by the pre-packaged "OSM Quick Start" bundles (`window_size=16,
step_size=1`). Expect macro-F1 ≈ 0.83 on this slice (0.8341 on a verified
end-to-end run) — lower than the classifier's 0.92 on other held-out slices,
because two low-activity states (`off_bottom` / `shut_in`) blur together on
segments from a different period of the field's life. Concatenated segments
contain timestamp seams; the platform marks windows straddling them
`INVALID_STATE` — exactly 75 here (5 seams between the 6 segments × 15
straddling windows at `window_size=16`).

Produced by Archetype AI's OSM data-prep pipeline (label by ACTC code →
segment on gaps/collisions → global per-channel z-score → cut slices; no row
is used twice across slices and shots).

## Attribution

The telemetry comes from the **Volve field dataset**, generously disclosed by
**Equinor** and its Volve licence partners (ExxonMobil E&P Norway and
Bayerngas Norge) for research, study, and development purposes:
[equinor.com/energy/volve-data-sharing](https://www.equinor.com/energy/volve-data-sharing).
Used under the terms of the Equinor Open Data Licence.
