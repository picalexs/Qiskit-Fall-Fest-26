# Larger-instance objective comparison

Source: `experiments/bench/qgd_bench.ipynb::make_instance`. The realistic error/missing settings are loaded directly from `experiments/bench/tudor_classical_vs_quantum.ipynb::REAL`. The 12-SNP seed-0 input reproduces that notebook's larger demo.

Settings: {'error': 0.05, 'missing': 0.1}; generator defaults: coverage=6, read_len=(2,4), gaps=0. The generator repairs uncovered adjacent boundaries with extra two-site reads; these reads can also contain errors. This differs from Alexandra's original generator.

All five methods receive identical observations per instance. Every native objective, including full-read MEC, is solved exactly. One representative per complement pair is enumerated in chunks, with a hard cap of 20 SNPs. Pairwise minima are checked against the team's banded dynamic program. MEC is not substituted into a pairwise Hamiltonian.

Truth is used only for evaluation. Metrics average over ALL exact minimizers; no winner is selected using hidden truth. The selected candidate is the first lexicographic representative. Switch errors use the FULL original read graph's linked boundaries for all methods. Full pairwise and full-read MEC are common reference costs in results.csv; differently scaled native costs should not be compared across variants.

| SNPs | Seed | Reads | Method | Recovery | Mean Hamming | Mean switches | Optimal phasings |
|---:|---:|---:|---|---:|---:|---:|---:|
| 12 | 0 | 25 | A baseline | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 0 | 25 | B1 per-read | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 0 | 25 | B2 per-site | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 0 | 25 | C deduplicated | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 0 | 25 | D MEC | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 1 | 24 | A baseline | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 1 | 24 | B1 per-read | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 1 | 24 | B2 per-site | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 1 | 24 | C deduplicated | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 1 | 24 | D MEC | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 2 | 25 | A baseline | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 2 | 25 | B1 per-read | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 2 | 25 | B2 per-site | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 2 | 25 | C deduplicated | 100.0% | 0.000 | 0.000 | 1 |
| 12 | 2 | 25 | D MEC | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 0 | 32 | A baseline | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 0 | 32 | B1 per-read | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 0 | 32 | B2 per-site | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 0 | 32 | C deduplicated | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 0 | 32 | D MEC | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 1 | 32 | A baseline | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 1 | 32 | B1 per-read | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 1 | 32 | B2 per-site | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 1 | 32 | C deduplicated | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 1 | 32 | D MEC | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 2 | 32 | A baseline | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 2 | 32 | B1 per-read | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 2 | 32 | B2 per-site | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 2 | 32 | C deduplicated | 100.0% | 0.000 | 0.000 | 1 |
| 16 | 2 | 32 | D MEC | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 0 | 40 | A baseline | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 0 | 40 | B1 per-read | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 0 | 40 | B2 per-site | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 0 | 40 | C deduplicated | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 0 | 40 | D MEC | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 1 | 40 | A baseline | 0.0% | 7.000 | 1.000 | 1 |
| 20 | 1 | 40 | B1 per-read | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 1 | 40 | B2 per-site | 50.0% | 3.500 | 0.500 | 2 |
| 20 | 1 | 40 | C deduplicated | 0.0% | 7.000 | 1.000 | 1 |
| 20 | 1 | 40 | D MEC | 50.0% | 3.500 | 0.500 | 2 |
| 20 | 2 | 40 | A baseline | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 2 | 40 | B1 per-read | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 2 | 40 | B2 per-site | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 2 | 40 | C deduplicated | 100.0% | 0.000 | 0.000 | 1 |
| 20 | 2 | 40 | D MEC | 100.0% | 0.000 | 0.000 | 1 |

## Observed results

- At 12 SNPs, all five methods uniquely recover truth on all 3 tested seeds.
- At 16 SNPs, all five methods uniquely recover truth on all 3 tested seeds.
- At 20 SNPs, seed 1: A baseline: recovery=0%, Hamming=7, switches=1; B1 per-read: recovery=100%, Hamming=0, switches=0; B2 per-site: recovery=50%, Hamming=3.5, switches=0.5; C deduplicated: recovery=0%, Hamming=7, switches=1; D MEC: recovery=50%, Hamming=3.5, switches=0.5.

An individual case where B1 helps does not overturn the original distribution-level results or establish that B1 is universally better. It shows that the choice of objective can matter on the team generator too.

## What to inspect

- Compare recovery, Hamming and switch errors on the same instance.
- Compare full_mean and mec_mean under common references, not native minima.
- Equal native optimum cost does not imply identical minimizing sets.
- Each size uses only a few seeds: this is a descriptive pilot, not evidence of universal superiority or a timing/speedup benchmark.

## Exact-run wall time (all five methods together)

- n12-s0: 0.02 seconds.
- n12-s1: 0.02 seconds.
- n12-s2: 0.02 seconds.
- n16-s0: 0.10 seconds.
- n16-s1: 0.10 seconds.
- n16-s2: 0.10 seconds.
- n20-s0: 2.21 seconds.
- n20-s1: 2.14 seconds.
- n20-s2: 2.16 seconds.

Reproduce: `python experiments/objectives/alexandra_larger_objectives.py --sizes 12 16 20 --seeds 0 1 2`.
Outputs include inputs.npz, manifest.json, results.csv, comparison.png and report.md. The module and original notebooks are unchanged. No QAOA/hardware run or quantum-advantage claim. Prepared with AI assistance.
