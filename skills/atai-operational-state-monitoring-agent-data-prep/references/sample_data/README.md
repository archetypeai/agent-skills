# Sample data — five short LARCO washing-machine cycles

Raw-ish recordings for trying the whole prep in about 15 seconds:

| recording | role (in `recordings.csv`) | raw rows | length |
|---|---|---|---|
| `becken_warm_15-min_40_2` | library | 188,820 | ~16 min |
| `becken_warm_sport_40_2` | library | 561,851 | ~47 min |
| `becken_warm_15-min_40_0` | validation | 198,476 | ~17 min |
| `becken_warm_fast-45_40_0` | test | 551,813 | ~46 min |
| `becken-flt_warm_fast-15_0` | delivery | 204,798 | ~17 min |

As in the LARCO examples, library, validation and test come from the healthy machine
(`becken`), and delivery from the second unit of the same model (`becken-flt`, marked
faulty in the dataset). Each cycle is in the role the examples' split gave it.

Each Parquet file has `timestamp` (epoch seconds, the sensor's own jittered ~200 Hz
timestamps), the 9 accelerometer channels in g (`back.x` … `top.z`: three triaxial sensors,
back, side and top), and `label`: one of `fill`, `wash`, `spin`, `drain` per row. Every
recording contains all four states.

**How the labels were made** (the dataset-specific step, done once): LARCO labels each
second from the machine's own measurements. A second is `spin` if its centrifuge label is
on, else `fill` if water flows in, else `drain` if water flows out (or in and out at once),
else `wash`; the heater label is not a state (vibration can't tell it from wash). Each
vibration sample takes the label of the second it falls in; samples outside any labelled
second were dropped. Nothing else was changed: no resampling, no scaling, no glitch removal
— that's what the prep skill does.

## Attribution

The data comes from the **LARCO** dataset, "Household **La**undry Appliance **R**esource
**C**onsumption and **O**peration Dataset", by Žygimantas Jasiūnas, João Alexandre Braz
Ferreira, Tiago Julião, José Cecílio, Guilherme Carrilho da Graça and Pedro M. Ferreira:
Zenodo, 2026, [doi:10.5281/zenodo.19666168](https://doi.org/10.5281/zenodo.19666168);
paper: *Scientific Data*, 2026, [doi:10.1038/s41597-026-07361-6](https://doi.org/10.1038/s41597-026-07361-6).
Used under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

**Changes made:** 5 of the dataset's cycles selected; the per-second state labels derived
as above and attached to each vibration sample; samples outside a labelled second dropped;
the vibration and labels combined into one Parquet file per cycle. The authors are not
involved in this skill and do not endorse it.
