# Sample data — Volve drilling log, five states

- `drilling_volve_f14.csv` — 2,750 rows of drilling telemetry, 9 sensor channels in their
  recorded units (not z-scored), on an exact 5 s grid: the input the platform's demo
  blueprint `playground-osm-drilling` is published with. Columns: `timestamp, BPOS, DBTM,
  FLWI, HDTH, HKLD, ROP, RPM, SPPA, WOB`.
- `drilling_volve_f14_labels.csv` — ground-truth sidecar (`timestamp, label`), used only
  for scoring, never uploaded.

| rows | state |
|---|---|
| 0–549 | `shut_in` |
| 550–1099 | `off_bottom` |
| 1100–1374 | `drilling` |
| 1375–1924 | `reaming` |
| 1925–2199 | `drilling` |
| 2200–2749 | `trip_in_slips` |

The classifier knows six states; `in_slips` is not in this log.

**How the labels were made.** Every row of the log is a row of the Volve real-time drilling
data, all nine channels identical, with its timestamp rewritten onto the 5 s grid (no
value is interpolated). The source rows carry the rig's activity code (`ACTC`), mapped to
states as 1 `drilling`, 2 `reaming`, 3 `off_bottom`, 4 `in_slips`, 8 `trip_in_slips`,
9 `shut_in`. Each block above is one stretch, or a few stitched stretches, of a single
code; no row's own code disagrees with its block's.

**Why an exact grid.** `playground-osm-drilling` keeps the blueprint default
`sample_rate_interval_tolerance: 0.05`: a window whose row gaps differ by more than 5% is
`INVALID_STATE`. Volve is recorded every 3–5 s, irregularly, and three of its states have
no 40-row stretch at one constant gap anywhere in the field's record, so at the recorded
cadence most windows are invalid. On this log every window is valid.

The blocks are joined end to end, so the timeline is continuous where the recording
wasn't: a window that straddles a block boundary mixes two states. The scorer's
steady-state view leaves those out.

Which rows the demo classifier was fitted on isn't published, so this log may not be held
out from it.

## Attribution

The telemetry comes from the **Volve field dataset**, generously disclosed by
**Equinor** and its Volve licence partners (ExxonMobil E&P Norway and
Bayerngas Norge) for research, study, and development purposes:
[equinor.com/energy/volve-data-sharing](https://www.equinor.com/energy/volve-data-sharing).
Used under the terms of the Equinor Open Data Licence.
