# AD role files: the contract (`ad-role-files/v1`)

What `build_roles.py` writes, what `preflight_roles.py` checks, and what the AD agent
skill's Path 2 reads. The layout is the
[TEP quickstart's](https://github.com/archetypeai/anomaly-discovery-agent-example-tep-quickstart),
generalised. Platform behaviour behind the rules was measured on the dev deployment
(2026-10-12 → 10-13) building the IMS bearings and Tennessee Eastman examples.

## Layout

```
roles/
├── manifest.json                 every file below, the channels and settings
├── zscore_stats.json             per-channel mean / std, from the library only (per condition with --scale-by)
├── library/<recording>__seg<N>.csv          one per recording segment, all normal, no label column
├── validation/validation_normal.csv         the role's normal-only recordings, joined, with `label`
├── validation/validation_incidents_NN.csv   its incident recordings, joined, with `label`
├── test/test_normal.csv, test_incidents_NN.csv   the same
├── delivery/delivery_NN.csv                 the role's recordings as they'd arrive, no label column
├── delivery/calibration.csv                 validation's normal-only file again, no label column
└── labels/<file>_labels.csv                 per scored file: timestamp, label, recording, type
```

Every CSV: `timestamp` first (epoch seconds, exactly 3 decimals), then the channels in
the manifest's order (z-scored, `decimals` places), then `label` (`normal` / `anomaly`) in
validation and test only.

## Rules, and why

| rule | why |
|---|---|
| **The library is normal only, with no label column** | Optimize labels every training row `normal` by a constant; the detector is fitted on the library and nothing else, so anything in it is "normal" by definition |
| **One library file per recording segment** | joined, the windows across each boundary would be lost (about 5 per boundary at window 48, step 8); separate files keep each recording identifiable |
| **Enough library windows** for `n_neighbors` and `pca_components` | a thin library does not error: the platform silently clamps both to the number of windows minus one |
| **Joined recordings have a gap between them well over the step** (default one day) | the platform marks every window across it invalid and abstains on it, instead of mixing two recordings. A gap within 5% of the step would pass the sampling-rate check and splice them |
| **An exact grid inside each recording segment** | the `ad` blueprint checks each window's sampling rate against its own mean (`sample_rate_interval_tolerance`, default 0.05); jitter and gaps invalidate windows. The segments of one recording that `prepare.py` split at a gap are joined with the same `seam_s` gap as two recordings, so no window spans the gap |
| **Both classes in validation and in test** | Optimize and Evals score macro-F1, and a missing class scores 0 |
| **Delivery files carry no label; the labels sit in `labels/`, row for row** | the deployed run must not see them; the scorer pairs each window with the label of its last row |
| **A calibration file in delivery** | known-normal data scored by the same deployed agent sets a false-alarm-rate threshold. It repeats validation's normal recordings, so it is the one file whose recordings are in two roles, and it is never scored as delivery |
| **z-score from the library only; a recording (or group) in one role** | otherwise validation and test scores come out optimistic |
| **Choose on validation, test once** | re-choosing after seeing the test number makes it optimistic |

Unlike OSM, a time jump inside a scored AD file does not fail the trial or eval: the
windows across it come back invalid (abstained), which is what lets one file hold many
recordings.

## `manifest.json`

```json
{
  "contract": "ad-role-files/v1",
  "step_ms": 180000,
  "channels": ["xmeas_1", "…", "xmv_11"],
  "timestamp_column": "timestamp",
  "label_column": "label",
  "labels": ["normal", "anomaly"],
  "timestamp_format": "epoch seconds, 3 decimals",
  "decimals": 4,
  "seam_s": 86400,
  "zscore_stats": "zscore_stats.json",
  "scale_by": null,
  "library":    [{"file": "library/normal_25h_run01__seg0.csv", "recording": "normal_25h_run01", "segment": 0, "rows": 500}],
  "validation": [{"file": "validation/validation_normal.csv", "labels": "labels/validation_normal_labels.csv",
                  "recordings": ["normal_48h_run41", "…"], "rows": 4800, "normal": 4800, "anomaly": 0}],
  "test":       ["… same shape"],
  "delivery":   [{"file": "delivery/delivery_01.csv", "labels": "labels/delivery_01_labels.csv", "recordings": ["…"],
                  "rows": 6720, "normal": 1920, "anomaly": 4800},
                 {"file": "delivery/calibration.csv", "labels": "…", "recordings": ["…"], "calibration": true}]
}
```

`labels/<file>_labels.csv` columns: `timestamp` (as in the file), `label`, `recording` (the
source recording's id), `type` (the index's `type`, empty if none). The scorer reports per
`type`, else per recording.
