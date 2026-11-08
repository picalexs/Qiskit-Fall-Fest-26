# Our investigation

## Question

How do QAOA method and depth affect the chance of finding the full pairwise
optimum on real hardware?

**Working hypothesis:** More flexible circuits could find better solutions, but
deeper circuits would also use more gates and could be more affected by noise.
This states the motivation for the comparison; it is not a recorded pre-run
prediction.

**Change:** Alongside standard QAOA, we tested warm-start and multi-angle
circuits at depths 1–3. We used the baseline pairwise evidence and a
maximum-weight spanning forest to reduce the circuit while preserving the active
graph's connectivity.

**Comparison:** All methods used the same five-SNP reads, graph and exact
full-model reference. Angles were trained locally and frozen before execution.
On each of `ibm_berlin` and `ibm_aachen`, we ran three repeats of 2,048 shots
per circuit. Success means sampling either full pairwise minimum (`01101` or
`10010`), not recovering verified biological truth. The standard and multi-angle
training limits were different (100 and 500 optimizer evaluations per start),
and warm-start also used classical search; their training costs are not equal.

| Depth 2 method | Berlin optimum rate | Aachen optimum rate |
| -------------- | ------------------: | ------------------: |
| Standard       |               27.9% |               27.9% |
| Warm-start     |               56.5% |               56.7% |
| Multi-angle    |           **92.0%** |           **91.7%** |

These are means of the raw hardware repeats. Multi-angle at depth 3 reached
90.4% and 90.5%, respectively, while increasing the transpiled two-qubit gate
count from 16 to 24. For this instance, depth 2 gave the best measured result
among the tested circuits; more depth did not improve it. This is one small
problem and one hardware run, not a general ranking or a quantum speedup claim.

[Notebook](backend_comparison.ipynb) ·
[Results and circuit resources](results/backend_comparison/20261107_234749_341927/real_device_summary.md)
·
[Figure](figures/backend_comparison/20261107_234749_341927/png/01_multiangle_devices.png)
·
[Configuration and job IDs](results/backend_comparison/20261107_234749_341927/manifest.json)

The real-device jobs were `dbnq1nsnqjds73evj030`, `dbnq1o00rnvc73974go0`,
`dbnq1o80rnvc73974gp0` on Berlin and `dbnq1oq4udtc73c409hg`,
`dbnq1p24udtc73c409ig`, `dbnq1p80rnvc73974gr0` on Aachen.

## Noise suppression and mitigation

For the depth-2 multi-angle circuit, we also tested dynamical decoupling,
Pauli twirling, readout correction, TREX-style correction and zero-noise
extrapolation on both devices. Each technique used a 4,096-shot total budget
per repeat and was compared with a matching raw reference.

Across three repeats, dynamical decoupling and twirling did not lower mean
cost on either device. Readout correction moved the mean cost from 1.704 to
1.324 on Berlin and from 1.865 to 1.622 on Aachen, closer to the ideal value
of 1.000. These are corrected cost estimates, not corrected bitstrings or
improved measured success rates. The small sample and single circuit limit
what we can conclude. See the [technique results](results/backend_comparison/20261107_234749_341927/techniques.csv)
and [comparison figure](figures/backend_comparison/20261107_234749_341927/png/03_technique_effect.png).

## Related team work

- **Evidence and objective:** We compared pairwise read-weighting rules and
  full-read minimum error correction (MEC), which counts the read calls that
  must change to fit either copy. These objectives can rank candidates
  differently; neither establishes biological truth. See the
  [objective guide](experiments/objectives/README_Objective_Experiments.md).
- **Pruning and information:** In the saved five-SNP synthetic sweep, removing
  weak edges could disconnect the graph and lose relative phase information. A
  maximum-weight spanning forest preserved active-graph connectivity, but did
  not guarantee the full model's optimum. See the
  [pruning study](docs/Pruning_Track%20.md).
- **Classical and hardware comparisons:** We included exact enumeration, local
  search and annealing, and tested compilation and error-mitigation methods in
  local noise simulations. Those simulator pilots are reported separately from
  real-device results in the [hardware guide](experiments/hardware/README.md).

## Disclosure and submission

The supplied lab's pairwise cost, exact reference, Ising mapping and standard
QAOA are the foundation. The [combined notebook](all_together.ipynb) includes
the organizer's fallback definitions under a `fallback_` prefix and checks them
against the team's implementations on the shared input. The team extended that
foundation with the objective, pruning, search and hardware studies above, and
brought them together with common references. The objective study and its
documentation disclose AI assistance.

Our contribution is the design, implementation and comparison of those
extensions, including the negative pruning result and the measured
hardware/resource trade-off. The notebook and saved results contain the
evidence; the presentation should show the question, comparison, results and
limitations. Judging uses the presentation and submitted evidence, with manual
or AI-assisted review and no hidden leaderboard.
