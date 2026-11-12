# Sample data: nine days of a metro train's air compressor (MetroPT-3)

- `metropt3_2020-07-08_2020-07-16.csv` (3.1 MB): what the agent reads. The output of the
  data-prep skill's `prepare.py` on its sample: `timestamp` in epoch seconds every 10 s
  exactly, then the 7 analog channels (`TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`,
  `Oil_temperature`, `Motor_current`) in their original units. 56,449 rows in 19 continuous
  pieces (the train is sometimes out of service). No labels.
- `known_events.csv`: the one failure the dataset's maintenance reports list in this span,
  an air leak on 2020-07-15 from 14:30 to 19:00 (the dataset's clock). For `grade.py` only;
  the agent never sees it.

Times are the dataset's own clock, written as seconds since 1970 with no time zone shift.

## Attribution

Derived from the **MetroPT-3 Dataset**: Davari, N., Veloso, B., Ribeiro, R., & Gama, J.
(2021), UCI Machine Learning Repository, [doi:10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R)
([dataset page](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)). Introductory
paper: Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2021),
*Predictive maintenance based on anomaly detection using deep learning for air production
unit in the railway industry*, IEEE DSAA 2021. The failure in `known_events.csv` is quoted
from the dataset page's failure table.

Licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/),
like the original; not covered by this repository's licence. The creators and their
institutions are not involved in this sample and do not endorse it.

**What was changed:** rows from 2020-07-08 to 2020-07-16 only; snapped to an exact 10 s
grid (holes up to 50 s filled with the last value, longer gaps kept); the 8 digital
channels left out; timestamps written as epoch seconds.
