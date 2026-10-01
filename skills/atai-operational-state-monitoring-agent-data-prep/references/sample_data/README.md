# Sample data — eight short LARCO washing-machine cycles

Raw-ish recordings for trying the whole prep in about 20 seconds:

| recording | role (in `recordings.csv`) | raw rows | length |
|---|---|---|---|
| `becken_warm_15-min_40_2` | library | 188,820 | ~16 min |
| `becken_warm_sport_40_2` | library | 561,851 | ~47 min |
| `becken_warm_delicate_30_0` | library | 574,741 | ~48 min |
| `becken_warm_20-deg_20_0` | library | 692,681 | ~58 min |
| `becken_warm_mix_40_2` | library | 743,093 | ~62 min |
| `becken_warm_15-min_40_0` | validation | 198,476 | ~17 min |
| `becken_warm_fast-45_40_0` | test | 551,813 | ~46 min |
| `becken-flt_warm_fast-15_2` | delivery | 186,976 | ~16 min |

As in the LARCO examples, library, validation and test come from the healthy machine
(`becken`), and delivery from the second unit of the same model (`becken-flt`, marked
faulty in the dataset). Each cycle is in the role the examples' split gave it.

**Why five library programs:** each washing programme spins at its own speed, so spin looks
different in each, and a short cycle has little spin to learn from. With only two library
programs, the platform scored spin 0.05 on the validation cycle; with these five, 0.57 (see
SKILL.md, "Cover the modes you'll score").

**Why this delivery cycle:** becken-flt's `warm_fast-15_0` was the first choice, but its
spin is unusually hard for every model we ran on it (spin F1 0.00–0.32 across three runs,
where the LARCO quickstart's other short becken-flt cycles scored 0.53–0.72), which makes a
first run look broken. `warm_fast-15_2` (spin 0.72 in the quickstart, the same length) was
chosen instead. It's a choice made for the sample, not a result.

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

**Changes made:** 8 of the dataset's cycles selected; the per-second state labels derived
as above and attached to each vibration sample; samples outside a labelled second dropped;
the vibration and labels combined into one Parquet file per cycle. The authors are not
involved in this skill and do not endorse it.
