# Sample data

Prepared slices from the Kaggle pump-sensor dataset (see **Data attribution**
below), z-scored per variate on a strictly regular 1-minute cadence — the shape
the `red` blueprint requires. Ten channels, selected for fault separation using
build-region data only.

**For fitting** (`red-fitting`) — single-class files. The fitting runner takes
each file's class from its **filename**, so these contain no label column and no
rows of the other class:

| File | Rows | Class |
|---|---|---|
| `pump_nshot_normal.csv` | 8,192 | `normal` |
| `pump_nshot_pump_breakdown_inc01.csv` | 945 | `pump_breakdown` (incident #1) |
| `pump_nshot_pump_breakdown_inc02.csv` | 3,111 | `pump_breakdown` (incident #2) |

**For running and scoring** (`red`) — a held-out slice with a row-aligned
ground-truth sidecar:

| File | Rows | Contents |
|---|---|---|
| `pump_eval_inc04.csv` | 8,798 | 4,096 normal → 606 fault → 4,096 normal |
| `pump_eval_inc04_labels.csv` | 8,798 | `timestamp,label` |

Incident #4 is a genuinely held-out event — the prototypes come from incidents
#1–#2 only — and at 606 rows it is the shortest incident this configuration
reliably catches. Its 8,192 rows of normal context are what make the false-alarm
rate meaningful: it is measured over ~8,000 windows containing zero fault rows.

The fault class is **0.5–7% of windows** depending on the slice, which is the
point: standard accuracy is close to useless here. See the SKILL's scoring
section.

## Regenerating, or using other slices

These come from Stage 2 of the
[RED example repo](https://github.com/archetypeai/rare-event-detection-agent-example):

```sh
# with data/raw/sensor.csv downloaded from Kaggle
python3 prep/prepare_pump_events.py
```

That also produces eval slices for incidents #3, #5, #6 and #7. Do not skip the
prep: the raw data contains two distinct leakage mechanisms — five channels that
go NaN *only* inside fault episodes, and several that read a saturated zero when
the pump stops — and the prep script audits both. Interpolating the first group
would teach a model "sensor offline = fault" rather than any pump behaviour.

## Data attribution

This data is derived from the **`pump_sensor_data`** dataset, published by the
Kaggle user **[`nphantawee`](https://www.kaggle.com/nphantawee)** on 2019-03-04:

> **pump_sensor_data** — "Pump sensor data for predictive maintenance"
> **https://www.kaggle.com/datasets/nphantawee/pump-sensor-data**

The source is 220,320 rows, 52 raw sensor channels at a 1-minute cadence, with a
`machine_status` label covering seven breakdown episodes. The uploader describes
its provenance in the dataset's own words:

> "I have a friend who working in a small team that taking care of water pump of
> a small area far from big town, there are 7 system failure in last year. Those
> failure cause huge problem to many people and also lead to some serious living
> problem of some family. The team can't see any pattern in the data when the
> system goes down, so they are not sure where to put more attention. […] The
> data are from all available sensor, all of them are raw value. Total sensor are
> 52 unit."
>
> *Acknowledgements: "Thanks to my friend and his team for sharing this data."*

**Credit for the underlying data belongs to that unnamed operations team**, who
shared it so the failures could be understood. The "7 system failure" in the
description corresponds exactly to the seven `BROKEN`/`RECOVERING` episodes these
files are cut from.

The files here are **derived**, not the original: a 10-channel subset, z-scored
per variate, cut into contiguous single-incident slices. The original remains
available at the Kaggle link above.

**Licence.** Kaggle reports no declared licence for this dataset — the uploader
selected none — so treat it as available for **research, study and development**
use, as the uploader intended in asking for help finding the failure pattern.
There is no explicit grant of commercial use. If you need a dataset with an
explicit licence for the same rare-event shape, the Petrobras **3W** dataset
([petrobras/3W](https://github.com/petrobras/3W), Apache-2.0 as declared on the
repository) provides named undesirable well events and a genuine multi-class
catalog.
