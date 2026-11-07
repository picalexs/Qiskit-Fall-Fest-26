# Length-dependent error experiment

**Question:** Does normalization help when longer reads have higher per-call error rates?

10 settings, 1000 paired replicates per setting. Five SNPs, 12 reads, uniform interval lengths 2-5, no missing calls. All methods see identical observations. The expected error per observed call is held at 10% or 15%; only its allocation across lengths changes. Actual error fractions fluctuate.

p(L) = base + strength * base / (27/7 - 2) * (L - 27/7). The center 27/7 is E[L^2]/E[L], ensuring equal expected per-call error budget. At strength=0, errors are IID. At strength=1, length-2 reads are error-free, while length-5 errors are about 1.615 times base.

Recovery uses uniform tie breaking over exact minimizers. Differences below are paired per dataset; intervals are approximate 95% mean CIs. Wins/ties/losses compare replicate recovery probabilities, not individual quantum shots.

| Base error | Strength | Method | Recovery | Delta vs A (pp) | 95% delta CI (pp) | Wins/ties/losses |
|---|---|---|---:|---:|---|---|
| 10% | 0.00 | A baseline | 93.5% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 10% | 0.00 | B1 per-read | 90.5% | -3.00 | [-4.59, -1.41] | 25/921/54 |
| 10% | 0.00 | B2 per-site | 93.4% | -0.15 | [-1.20, +0.90] | 25/951/24 |
| 10% | 0.25 | A baseline | 92.5% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 10% | 0.25 | B1 per-read | 91.4% | -1.10 | [-2.66, +0.46] | 35/920/45 |
| 10% | 0.25 | B2 per-site | 93.8% | +1.35 | [+0.24, +2.46] | 36/943/21 |
| 10% | 0.50 | A baseline | 91.9% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 10% | 0.50 | B1 per-read | 93.3% | +1.45 | [-0.15, +3.05] | 52/916/32 |
| 10% | 0.50 | B2 per-site | 94.3% | +2.45 | [+1.32, +3.58] | 46/940/14 |
| 10% | 0.75 | A baseline | 91.2% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 10% | 0.75 | B1 per-read | 94.8% | +3.55 | [+1.98, +5.12] | 62/917/21 |
| 10% | 0.75 | B2 per-site | 94.8% | +3.65 | [+2.44, +4.86] | 55/936/9 |
| 10% | 1.00 | A baseline | 90.8% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 10% | 1.00 | B1 per-read | 96.1% | +5.35 | [+3.85, +6.85] | 69/923/8 |
| 10% | 1.00 | B2 per-site | 95.0% | +4.30 | [+3.09, +5.51] | 60/936/4 |
| 15% | 0.00 | A baseline | 84.3% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 15% | 0.00 | B1 per-read | 77.8% | -6.58 | [-8.92, -4.25] | 57/821/122 |
| 15% | 0.00 | B2 per-site | 84.6% | +0.23 | [-1.29, +1.76] | 55/897/48 |
| 15% | 0.25 | A baseline | 81.6% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 15% | 0.25 | B1 per-read | 80.3% | -1.25 | [-3.45, +0.95] | 76/842/82 |
| 15% | 0.25 | B2 per-site | 84.7% | +3.10 | [+1.59, +4.61] | 78/890/32 |
| 15% | 0.50 | A baseline | 79.6% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 15% | 0.50 | B1 per-read | 82.9% | +3.30 | [+1.11, +5.49] | 105/835/60 |
| 15% | 0.50 | B2 per-site | 84.8% | +5.23 | [+3.64, +6.82] | 98/875/27 |
| 15% | 0.75 | A baseline | 78.0% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 15% | 0.75 | B1 per-read | 86.1% | +8.08 | [+5.89, +10.27] | 143/819/38 |
| 15% | 0.75 | B2 per-site | 85.8% | +7.82 | [+6.09, +9.54] | 128/851/21 |
| 15% | 1.00 | A baseline | 75.3% | +0.00 | [+0.00, +0.00] | 0/1000/0 |
| 15% | 1.00 | B1 per-read | 88.8% | +13.45 | [+11.25, +15.65] | 184/796/20 |
| 15% | 1.00 | B2 per-site | 86.3% | +10.98 | [+9.15, +12.82] | 160/826/14 |

## Observed findings

- Base 10%, strength 0: A recovery 93.55%, B1 90.55%; paired difference -3.00 percentage points.
- Base 10%, strength 1: A recovery 90.75%, B1 96.10%; paired difference +5.35 percentage points.
- Base 15%, strength 0: A recovery 84.33%, B1 77.75%; paired difference -6.58 percentage points.
- Base 15%, strength 1: A recovery 75.30%, B1 88.75%; paired difference +13.45 percentage points.

B1 changes from lower recovery in the IID controls to higher recovery under strong length-dependent errors in this simulation. This supports a conditional benefit, not a universal benefit or a claim about real sequencing. Quality can act as a confounder in the comparison of read lengths.

## Interpretation

Read the paired deltas: positive means higher recovery than baseline in that setting; negative means lower. An interval crossing zero does not establish a directional difference. Exploratory intervals are not adjusted for multiple comparisons. No outcome was used to tune the predeclared settings.

This is a synthetic sensitivity study, not proof that real long reads are less reliable. No quality-aware optimal weighting or QAOA/hardware advantage is tested. Normalization depends only on length, so benefit is not guaranteed.

Original results in objective_results are unchanged. Reproduce with `python experiments/objectives/length_error_experiment.py --repeats 1000`. Prepared with AI assistance.
