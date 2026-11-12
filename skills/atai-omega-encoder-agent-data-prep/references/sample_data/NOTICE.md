# Sample data: AURSAD (CC BY 4.0)

100 screwdriving tightenings from **AURSAD: Universal Robot Screwdriving Anomaly Detection
Dataset**, by Błażej Leporowski, Daniella Tola, Casper Hansen and Alexandros Iosifidis
(Aarhus University; Technicon ApS), [doi:10.5281/zenodo.4487073](https://doi.org/10.5281/zenodo.4487073),
[arXiv:2102.01409](https://arxiv.org/abs/2102.01409), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## What was changed

- **A subset:** 100 of the 4,094 samples, chosen evenly by sample number within each role and
  class of the [AURSAD quickstart](https://github.com/archetypeai/omega-encoder-example-aursad-quickstart)
  (library: 32 normal; validation and test: 12 normal + 4 of each fault; delivery: 12 normal + 4
  damaged + 4 missing screws).
- **12 of 134 columns,** renamed: `output_double_register_25` → `torque_current`,
  `output_double_register_24` → `torque_achieved`, `actual_current_0..5` → `joint_current_0..5`,
  `actual_TCP_force_2` → `tcp_force_z`, `actual_tool_accelerometer_0..2` → `tool_acc_x/y/z`; plus
  `timestamp`, as recorded (seconds; it restarts in every recording session).
- **Each file is one whole sample** (one tightening) as recorded: its own length, its timestamps'
  jitter, and the robot's motion before and after the screwdriver acts. The channels are stored as
  float32, in Parquet.
- **`index.csv`** adds the role, the recording session (`group`) and the class (`label`): the
  dataset's labels 0-3 named `normal`, `damaged_screw`, `extra_component`, `missing_screw`.

Taken from `AURSAD.h5` (the pandas HDF5 file, read with h5py); the `.pkl` was never unpickled.
