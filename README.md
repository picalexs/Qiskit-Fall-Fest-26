# Quantum Genome Detectives

The project compares methods for genome phasing using classical algorithms and
QAOA.

- [all_together.ipynb](all_together.ipynb) is the self-contained team notebook.
- [backend_comparison.ipynb](backend_comparison.ipynb) runs or loads backend
  comparisons. Its current `RUN_REAL` setting submits real hardware jobs when
  run; set it to `False` for a local study, or use load mode to inspect a saved
  run.
- [experiments/](experiments/) contains the original benchmark, objective and
  hardware notebooks and their supporting code. Run the benchmark notebooks with
  `experiments/bench/` as the notebook working directory so
  `%run qgd_bench.ipynb` works.
- [results/](results/) contains saved benchmark data and future study outputs.
  Each backend comparison has a timestamped folder; `20261107_225405_604619` is
  a partial screened run and `20261107_234749_341927` is a completed real-device
  run. The manifest records the run configuration and status.
- [figures/](figures/) contains the saved study figures and the latest
  backend-comparison export, including the plot data tables and cached job
  metrics.
- [docs/](docs/) contains the challenge material and research notes.
  [Hardware](experiments/hardware/README.md) and
  [objective](experiments/objectives/README_Objective_Experiments.md) experiment
  guides describe their studies.
- [scripts/plots/](scripts/plots/) contains the backend plot generator and
  optional job-metrics fetcher.

Install the packages in `requirements.txt` in a Python environment and select it
as the notebook kernel. Credentials for IBM modes are read from the environment
or an ignored root `.env` file.

To regenerate figures from the latest complete saved run, without a quantum
backend:

```bash
python scripts/plots/make_plots.py
```

Outputs go to `figures/backend_comparison/<run>/`; use `--help` to select a run
or output directory. The presentation includes `03b_mitigation_resources`,
which compares each technique's measured qubits, two-qubit gates and circuit
depth with its matching raw reference. It shows the ZNE factor-3 circuit and
excludes calibration circuits; scheduled delays count toward circuit depth.
The plot generator requires version 2 run data; older runs
retain their original plots. Runs and figures are included in Git, including
historical circuit and calibration files. The saved data is large (about 580
MB).
