# RED role files: the contract (`red-role-files/v1`)

What `build_roles.py` writes, what `preflight_roles.py` checks, and what the RED agent
skill's Path 2 reads. The layout is the
[RED pump quickstart's](https://github.com/archetypeai/rare-event-detection-agent-example-pump-quickstart),
generalised to any background and event classes. Platform behaviour behind the rules was
measured on the dev deployment on 2026-10-13, building the pump examples.

## Layout

```
roles/
├── manifest.json                      every file below, the classes, channels and z-score stats
├── training/<background>.csv          normal operation, no label column
├── training/<class>_incNN.csv         one per training incident, that class only, no label column
├── validation/incNN.csv               one per incident, with normal context, and `label`
├── test/incNN.csv                     the same
├── delivery/incNN.csv                 a test incident again, shorter context, no label column
├── delivery/tail_NN.csv               (optional) rows after the last scored file, no label column
└── labels/<file>_labels.csv           per delivery file: timestamp, label, row for row
```

Every CSV: `timestamp` first (as prepared: integer epoch seconds when the grid allows),
then the channels in the manifest's order, z-scored with 6 decimals, then `label` (one of
the classes) in validation and test only.

## Rules, and why

| rule | why |
|---|---|
| **Training files are each one class, with no label column** | Optimize labels each training example by a constant (`{"from": {"constant": "<class>"}}`) and averages its windows into that class's prototype: a normal row in an incident file pulls the prototype toward normal |
| **Every class has a training file**, and every class is in `events` | a class with no training windows has no prototype; a training class missing from the request's `events` is a 400 at creation |
| **No row in two of training, validation and test** | incidents close together share their normal context otherwise (the pump's #3 and #4 did by 1,033 rows); slices are cut at the midpoint between two incidents |
| **Delivered incidents lie inside their test files; a tail shares no row with anything** | delivery re-runs a test incident as an operator would; a tail is new operation |
| **An exact grid, no jump, inside each file** | the `red` blueprint validates monotonic timestamps and each interval within 5% of the window's mean: a jump makes every window across it `INVALID_STATE` |
| **Validation and test hold the background and their incident's class** | Optimize and Evals score the classes present; a file with one class says nothing about telling them apart |
| **Every scored incident can fill half a window** of the sizes you search | the classifier compares whole windows to the prototypes: an incident shorter than half a window barely moves one, so at that window it is all but undetectable |
| **Channels and z-score statistics from build-time data only** (rows before the first held-out incident's file) | chosen on held-out data, the scores come out optimistic |
| **Delivery files carry no label; the labels sit in `labels/`, row for row** | the deployed run must not see them; the scorer reads them afterwards |

## Scoring rules the files assume

- **The platform scores each row once.** With overlapping windows, a window answers only
  for the rows since the previous window ended, collapsed by `downsampling`: at
  `step_size: 1` that is one row, so `majority_vote` and `last_record` score alike.
- **The design rule labels a whole window by the majority of its rows**, ties to the
  event. The RED agent skill's scorer reports both.

## `manifest.json`

```json
{
  "contract": "red-role-files/v1",
  "classes": ["normal", "pump_breakdown"],
  "background": "normal",
  "timestamp_column": "timestamp",
  "channels": ["sensor_02", "…", "sensor_42"],
  "step_s": 60.0,
  "source": "pump_sample.parquet",
  "build_end_row": 65222,
  "zscore": {"sensor_02": {"mean": 51.9, "std": 2.1}, "…": {}},
  "training":   [{"file": "training/normal.csv", "rows": [0, 8192], "n_rows": 8192, "event_rows": 0, "class": "normal"},
                 {"file": "training/pump_breakdown_inc01.csv", "rows": [17155, 18100], "n_rows": 945,
                  "event_rows": 945, "class": "pump_breakdown", "incident": 1}],
  "validation": [{"file": "validation/inc03.csv", "rows": [65222, 74210], "n_rows": 8988, "event_rows": 1313,
                  "incident": 3, "class": "pump_breakdown", "incident_rows": [69318, 70631]}],
  "test":       [{"file": "test/inc04.csv", "…": "…"}],
  "delivery":   [{"file": "delivery/inc04.csv", "rows": [76766, 79420], "…": "…", "labels": "labels/inc04_labels.csv"}]
}
```

`rows` are half-open row ranges of the prepared recording; `incident_rows` the incident's
own rows. Keep the z-score statistics with the deployed agent: any later input must be
scaled the same way.
