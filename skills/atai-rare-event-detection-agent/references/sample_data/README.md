# Sample data

`make_sample_data.py` generates a small synthetic set so the RED lifecycle is
runnable without any download:

```sh
python3 make_sample_data.py
```

It writes four files — two pure-class shot files for fitting, one inference
slice, and a ground-truth sidecar:

| File | Rows | Purpose |
|---|---|---|
| `red_sample_shots_normal.csv` | 4,096 | normal prototype |
| `red_sample_shots_excursion.csv` | 512 | fault prototype (pure fault rows) |
| `red_sample_inference.csv` | 4,608 | normal → excursion → normal |
| `red_sample_inference_labels.csv` | 4,608 | row-aligned truth for scoring |

The signal is deliberately simple: five z-scored channels of correlated noise,
with the `excursion` class shifting the mean of three of them and damping a
fourth. It exercises the API contract, the filename-labelling rule, and the
scoring views. **It is not a benchmark** — real rare-event data is harder in
ways synthetic data cannot imitate, particularly the channel-dropout and
sensor-saturation effects that dominate the pump dataset below.

## Using the real dataset instead

The [RED example repo](https://github.com/archetypeai/rare-event-detection-agent-example)
builds its classifier from the **Kaggle pump-sensor dataset**, and its Stage 2
prep script produces exactly the file shapes above (n-shot files per class plus
eval slices with label sidecars):

```sh
# in the example repo, with data/raw/sensor.csv downloaded from Kaggle
python3 prep/prepare_pump_events.py
```

That is the recommended path for anything you intend to draw conclusions from.
Do not skip the prep: the raw data contains two distinct leakage mechanisms —
five channels that go NaN *only* inside fault episodes, and several that read a
saturated zero when the pump stops — and the prep script audits both.

## Data attribution

The pump-sensor data referenced above is the **`pump_sensor_data`** dataset,
published by the Kaggle user **[`nphantawee`](https://www.kaggle.com/nphantawee)**
on 2019-03-04:

> **pump_sensor_data** — "Pump sensor data for predictive maintenance"
> https://www.kaggle.com/datasets/nphantawee/pump-sensor-data

220,320 rows, 52 raw sensor channels, 1-minute cadence, with a `machine_status`
label covering seven breakdown episodes. The uploader describes its provenance
in the dataset's own words:

> "I have a friend who working in a small team that taking care of water pump of
> a small area far from big town, there are 7 system failure in last year. […]
> The data are from all available sensor, all of them are raw value. Total sensor
> are 52 unit."
>
> *Acknowledgements: "Thanks to my friend and his team for sharing this data."*

Credit for the underlying data belongs to that unnamed operations team, who
shared it so the failures could be understood.

**Licence: Unknown.** Kaggle reports no declared licence for this dataset — the
uploader selected none — so there is **no explicit grant of redistribution or
commercial use**. That is why this directory ships a synthetic generator rather
than pump slices: download the data yourself from Kaggle, under Kaggle's terms
and the uploader's, and treat it as research/educational use. This is materially
different from a dataset like Volve, which is redistributable under Equinor's
stated Open Data Licence.

If you need a permissively licensed dataset with the same rare-event shape, the
Petrobras **3W** dataset ([petrobras/3W](https://github.com/petrobras/3W),
Apache-2.0 as declared on the repository) provides named undesirable well events
and, unlike the pump data, a genuine multi-class catalog.
