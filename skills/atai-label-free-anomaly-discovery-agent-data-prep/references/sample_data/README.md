# Sample data: nine days of a metro train's air compressor (MetroPT-3)

`metropt3_2020-07-08_2020-07-16.csv.gz` (1.1 MB): the raw rows of the MetroPT-3 dataset
from 2020-07-08 to 2020-07-16, in the dataset's own format: an unnamed index column, a
`timestamp` column of text, the 7 analog channels (`TP2`, `TP3`, `H1`, `DV_pressure`,
`Reservoirs`, `Oil_temperature`, `Motor_current`) and the 8 digital ones (valve and switch
states), about one row every 10 s. 56,944 rows; the data starts at 15:20 on Jul 8, after a
gap. No labels.

The dataset's maintenance reports list one failure in this span, an air leak on
2020-07-15 from 14:30 to 19:00. It is in `known_events.csv` of the agent skill's sample,
for grading only; nothing in these scripts reads it.

## Attribution

Derived from the **MetroPT-3 Dataset**: Davari, N., Veloso, B., Ribeiro, R., & Gama, J.
(2021), UCI Machine Learning Repository, [doi:10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R)
([dataset page](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)). Introductory
paper: Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2021),
*Predictive maintenance based on anomaly detection using deep learning for air production
unit in the railway industry*, IEEE DSAA 2021.

Licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/),
like the original; not covered by this repository's licence. The creators and their
institutions are not involved in this sample and do not endorse it.

**What was changed:** only the rows from 2020-07-08 to 2020-07-16 were kept, and the file
was compressed. Values and columns are as published.
