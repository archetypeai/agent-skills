# Sample data — Volve six-state eval slice

- `volve_states_opt_slice_04.csv` — 4,200 rows of prepared drilling telemetry
  (9 sensor channels, globally z-scored), one contiguous segment per state:
  `drilling · reaming · off_bottom · in_slips · trip_in_slips · shut_in`.
- `volve_states_opt_slice_04_labels.csv` — ground-truth sidecar
  (`DATE_TIME,label`), used only for scoring, never uploaded.

The slice is **held out**: it was never used to fit or tune the published
classifier the skill's default run pins (`window_size=16`, k=3, manhattan,
uniform). Expect macro-F1 ≈ 0.83 on this slice — lower than the classifier's
0.92 on other held-out slices, because two low-activity states
(`off_bottom` / `shut_in`) blur together on segments from a different period
of the field's life. Concatenated segments contain timestamp seams; the
platform marks windows straddling them `INVALID_STATE` (~45 windows).

Produced by the prep stage of
[operational-state-monitoring-agent-example](https://github.com/archetypeai/operational-state-monitoring-agent-example)
(label by ACTC code → segment on gaps/collisions → global per-channel
z-score → cut slices; no row is used twice across slices and shots).

## Attribution

The telemetry comes from the **Volve field dataset**, generously disclosed by
**Equinor** and its Volve licence partners (ExxonMobil E&P Norway and
Bayerngas Norge) for research, study, and development purposes:
[equinor.com/energy/volve-data-sharing](https://www.equinor.com/energy/volve-data-sharing).
Used under the terms of the Equinor Open Data Licence.
