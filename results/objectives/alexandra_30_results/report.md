# Exact comparison at 30 SNPs

Same team generator, 5% per-call error, 10% missing calls, interval lengths 2-4, coverage 6, seeds 0/1/2. Short local factors permit exact dynamic programming for all five objectives, INCLUDING full-read MEC. This is not a new pairwise mapping of MEC. Every minimizing phasing is counted; truth is used only for evaluation. DP metrics were checked against enumeration up to 20 SNPs, and pairwise minima against the shared banded DP.

| Seed | Reads | Method | Recovery | Mean Hamming | Mean switches | Optimal phasings |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 60 | A baseline | 0.0% | 9.500 | 1.500 | 2 |
| 0 | 60 | B1 per-read | 25.0% | 6.500 | 1.000 | 4 |
| 0 | 60 | B2 per-site | 25.0% | 6.500 | 1.000 | 4 |
| 0 | 60 | C deduplicated | 0.0% | 9.500 | 1.500 | 2 |
| 0 | 60 | D MEC | 25.0% | 6.500 | 1.000 | 4 |
| 1 | 60 | A baseline | 100.0% | 0.000 | 0.000 | 1 |
| 1 | 60 | B1 per-read | 0.0% | 1.000 | 1.000 | 1 |
| 1 | 60 | B2 per-site | 50.0% | 0.500 | 0.500 | 2 |
| 1 | 60 | C deduplicated | 100.0% | 0.000 | 0.000 | 1 |
| 1 | 60 | D MEC | 50.0% | 0.500 | 0.500 | 2 |
| 2 | 60 | A baseline | 50.0% | 0.500 | 1.000 | 2 |
| 2 | 60 | B1 per-read | 0.0% | 14.500 | 2.000 | 2 |
| 2 | 60 | B2 per-site | 25.0% | 7.500 | 1.500 | 4 |
| 2 | 60 | C deduplicated | 50.0% | 0.500 | 1.000 | 2 |
| 2 | 60 | D MEC | 25.0% | 7.500 | 1.500 | 4 |

## Observed results

- A baseline: mean recovery across the three seeds 50.00%.
- B1 per-read: mean recovery across the three seeds 8.33%.
- B2 per-site: mean recovery across the three seeds 33.33%.
- C deduplicated: mean recovery across the three seeds 50.00%.
- D MEC: mean recovery across the three seeds 33.33%.

B1 helps on seed 0 but loses to baseline on seeds 1 and 2. The earlier 20-SNP case is therefore an example of conditional benefit, not evidence of universal superiority. Percentages are uniform-tie recovery probabilities, not measured QAOA shot frequencies.

Three instances are a descriptive pilot, not a universal ranking. No QAOA, hardware or quantum advantage tested. Original 12/16/20 results are unchanged. Fields requiring counts of non-optimal cost levels (e.g. truth rank) are intentionally not reported by this DP. Common-reference mean costs are retained.

Reproduce: `python experiments/objectives/alexandra_30_objectives.py`.
