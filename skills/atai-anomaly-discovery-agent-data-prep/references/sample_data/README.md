# Sample data: 35 Tennessee Eastman Process runs

35 simulation runs of the Tennessee Eastman Process, one Parquet file each (4.8 MB), and
`recordings.csv`, the index with roles, groups and types filled in. The same data and
split as the [TEP quickstart](https://github.com/archetypeai/anomaly-discovery-agent-example-tep-quickstart),
smaller.

| role | recordings | rows |
|---|---|---|
| library | 10 fault-free 25-hour runs (`normal_25h_run01`–`10`) | 500 each, all `normal` |
| validation | 5 fault-free 48-hour runs (41, 44–47) + run 1 of six faults | 960 each |
| test | 1 fault-free 48-hour run (42) + run 2 of the six faults | 960 each |
| delivery | 1 fault-free 48-hour run (43) + run 3 of the six faults | 960 each |

The six faults (`type`): IDV 1 and 2 (step changes), IDV 5 (a step the controllers
absorb), IDV 10 (random variation), IDV 13 (a slow drift) and IDV 15 (a sticking valve,
one of the three the literature calls undetectable). Every faulty run is 8 hours of
normal operation, then the fault: `label` is `anomaly` from sample 161 on.

Each file: `timestamp` (epoch seconds, every 180 s; each run starts three days after the
previous one), the 33 continuous variables (`xmeas_1`–`22` measurements, `xmv_1`–`11`
controller outputs; the 19 laboratory analysers are left out), and `label`.

## Attribution

Derived from **Additional Tennessee Eastman Process Simulation Data for Anomaly
Detection Evaluation**: Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017),
Harvard Dataverse, [doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1). Original
process: Downs, J. J., & Vogel, E. F. (1993), *A plant-wide industrial process control
problem*, Computers & Chemical Engineering 17(3), 245–255.

The dataset is dedicated to the public domain: its user agreement allows copying,
modifying and distributing the work without permission. Neither the authors nor Pacific
Science & Engineering Group, Inc. are involved in this sample or endorse it.

**What was changed:** 35 runs were selected; 33 of the 52 variables kept; timestamps
added; a per-row label derived from the documented fault onset.
