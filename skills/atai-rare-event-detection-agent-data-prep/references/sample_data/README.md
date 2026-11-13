# Sample data: one water pump, four breakdowns

`pump_sample.parquet` (4.5 MB): the first 82,492 rows of the Kaggle pump-sensor recording
(2018-04-01 to 2018-05-27, one row a minute) with 15 of its 52 sensor channels, the
`timestamp` (ISO 8601 text) and `machine_status` (`NORMAL`, `BROKEN`, `RECOVERING`),
**as raw as the source**: unscaled values, missing readings left missing, and the
source's unnamed row-index column. Four breakdowns:

| # | starts | length | role in the walkthrough |
|---|---|---|---|
| 1 | 2018-04-12 21:55 | 945 rows | training |
| 2 | 2018-04-18 00:30 | 3,111 rows | training |
| 3 | 2018-05-19 03:18 | 1,313 rows | validation |
| 4 | 2018-05-25 00:30 | 606 rows | test, and delivered again |

The channels were picked so every rule in the skill has something to act on:

| channels | what the steps do with them |
|---|---|
| `sensor_02`–`05`, `10`–`12`, `38`, `41`, `42` | the 10 the pump examples select |
| `sensor_00`, `sensor_06` | go dark during breakdowns: 4,114 of `sensor_00`'s 4,122 missing readings and 238 of `sensor_06`'s 246 are inside one. Dropped (`sensor_00` for dropout, `sensor_06` for a 194-reading gap) |
| `sensor_01` | a missing run of 30 readings, longer than the platform repairs: dropped |
| `sensor_15` | empty: dropped |
| `sensor_13` | reads exactly 0 in 96% of breakdown rows (as do selected `sensor_05`, `11` and `12`, in 60–92%): kept, with a warning; not selected |

## Data attribution

Derived from the **`pump_sensor_data`** dataset, published by the Kaggle user
**[`nphantawee`](https://www.kaggle.com/nphantawee)** on 2019-03-04:

> **pump_sensor_data** — "Pump sensor data for predictive maintenance"
> **https://www.kaggle.com/datasets/nphantawee/pump-sensor-data**

The source is 220,320 rows, 52 raw sensor channels at a 1-minute cadence, with a
`machine_status` label covering seven breakdown episodes. The uploader describes its
provenance: a friend's small team looking after the water pump of an area far from a big
town, which failed seven times in a year, and could not see a pattern in the data. *"Thanks
to my friend and his team for sharing this data."* **Credit for the underlying data
belongs to that operations team.**

**What was changed:** rows 0–82,491 kept (the first four breakdowns and 4,096 rows after
the fourth); 15 of the 52 sensor channels kept; nothing else. The original remains
available at the Kaggle link above.

**Licence:** none is declared on the Kaggle dataset page.
