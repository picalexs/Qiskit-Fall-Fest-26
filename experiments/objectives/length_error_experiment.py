"""Controlled extension: are normalized objectives helped by length-dependent errors?

Run from project root:
  .venv-objectives/Scripts/python.exe experiments/objectives/length_error_experiment.py
Outputs are separate from the original objective_results study.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from objective_study import (
    VARIANTS, all_states, mean_ci, objective_scores, score_metrics, write_csv,
)

OUT = Path(__file__).resolve().parents[2] / 'results' / 'objectives' / 'length_error_results'
N_READS = 12
# For uniform L in {2,3,4,5}, E[L^2]/E[L] = 27/7.
# Centering here keeps expected error probability per observed call at base_error.
CALL_WEIGHTED_LENGTH = 27 / 7


def error_probability(length, base_error, strength):
    """strength 0 -> IID control; 1 -> p(error|L=2)=0, no clipping needed."""
    slope = strength * base_error / (CALL_WEIGHTED_LENGTH - 2)
    return base_error + slope * (np.asarray(length) - CALL_WEIGHTED_LENGTH)


def latent_dataset(replicate):
    """Same truth, intervals, copy origins and error uniforms across settings."""
    truth_seed, read_seed = 310_000 + replicate, 410_000 + replicate
    truth = np.random.default_rng(truth_seed).integers(2, size=5)
    rng = np.random.default_rng(read_seed)
    clean = np.full((N_READS, 5), -1, dtype=int)
    lengths = rng.integers(2, 6, size=N_READS)
    for row, length in zip(clean, lengths):
        start = int(rng.integers(6 - length))
        copy = truth if rng.integers(2) == 0 else 1 - truth
        row[start:start+length] = copy[start:start+length]
    uniforms = rng.random(clean.shape)
    return truth, clean, lengths, uniforms


def observed_reads(clean, lengths, uniforms, base_error, strength):
    probabilities = error_probability(lengths, base_error, strength)
    flipped = (clean != -1) & (uniforms < probabilities[:, None])
    reads = clean.copy()
    reads[flipped] = 1 - reads[flipped]
    return reads, flipped


def validate():
    lengths = np.arange(2, 6)
    for base in (0.1, 0.15):
        for strength in (0, 0.25, 0.5, 0.75, 1):
            p = error_probability(lengths, base, strength)
            assert np.all(p >= -1e-12) and np.all(p <= 1)
            assert np.isclose(np.average(p, weights=lengths), base)
            assert np.all(np.diff(p) >= -1e-12)
            if strength == 0:
                assert np.allclose(p, base)
    truth, clean, lengths, uniforms = latent_dataset(0)
    reads, flips = observed_reads(clean, lengths, uniforms, 0, 0)
    assert np.array_equal(reads, clean) and not flips.any()
    assert np.array_equal(clean != -1, observed_reads(clean, lengths, uniforms, 0.15, 1)[0] != -1)
    assert np.array_equal(latent_dataset(0)[1], clean)
    states = all_states(5)
    for costs in objective_scores(clean, states)[0].values():
        assert score_metrics(states, costs, truth)['truth_is_optimal']


def run(repeats):
    OUT.mkdir(parents=True, exist_ok=True)
    settings = [dict(base_error=base, strength=strength)
                for base in (0.1, 0.15) for strength in (0, 0.25, 0.5, 0.75, 1)]
    manifest = dict(repeats=repeats, n_snps=5, n_reads=N_READS,
                    length_distribution='Uniform L=2,3,4,5; contiguous intervals; no missing calls',
                    settings=settings, truth_seeds=list(range(310_000, 310_000+repeats)),
                    read_seeds=list(range(410_000, 410_000+repeats)),
                    error_rule='p(L)=base + strength*base/(27/7-2)*(L-27/7)',
                    control='Expected error per observed call fixed at base; strength=0 is IID',
                    pairing='Shared truth, intervals, origins and error uniforms across settings',
                    tie_policy='Uniform over all exact minimizers, modulo complement',
                    limitation='Artificial reliability model; not evidence about actual sequencing')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    states = all_states(5)
    rows, datasets, summary = [], {}, []
    for setting_id, setting in enumerate(settings):
        setting_rows, saved_reads, saved_truth = [], [], []
        for replicate in range(repeats):
            truth, clean, lengths, uniforms = latent_dataset(replicate)
            reads, flips = observed_reads(clean, lengths, uniforms, **setting)
            saved_reads.append(reads)
            saved_truth.append(truth)
            scores, _ = objective_scores(reads, states)
            for variant, costs in scores.items():
                row = dict(setting_id=setting_id, replicate=replicate, **setting,
                           variant=variant, actual_error_rate=float(flips.sum()/(clean != -1).sum()),
                           **score_metrics(states, costs, truth))
                rows.append(row)
                setting_rows.append(row)
        datasets[f'reads_{setting_id}'] = np.array(saved_reads)
        datasets[f'truth_{setting_id}'] = np.array(saved_truth)
        baseline = [r for r in setting_rows if r['variant'] == VARIANTS[0]]
        for variant in VARIANTS:
            group = [r for r in setting_rows if r['variant'] == variant]
            delta = np.array([r['recovery']-b['recovery'] for r, b in zip(group, baseline)])
            mean, low, high = mean_ci(delta)
            recovery, rlow, rhigh = mean_ci([r['recovery'] for r in group])
            summary.append(dict(setting_id=setting_id, **setting, variant=variant,
                                replicates=repeats, recovery=recovery,
                                recovery_ci_low=max(0, rlow), recovery_ci_high=min(1, rhigh),
                                paired_delta=mean, delta_ci_low=low, delta_ci_high=high,
                                wins=int(np.sum(delta>1e-9)), ties=int(np.sum(np.abs(delta)<=1e-9)),
                                losses=int(np.sum(delta < -1e-9)),
                                mean_hamming=float(np.mean([r['mean_hamming'] for r in group])),
                                actual_error_rate=float(np.mean([r['actual_error_rate'] for r in group]))))
        print(f'Setting {setting_id+1}/{len(settings)} complete: {setting}', flush=True)
    write_csv(OUT/'replicate_metrics.csv', rows)
    write_csv(OUT/'paired_summary.csv', summary)
    np.savez_compressed(OUT/'datasets.npz', **datasets)
    plot(summary)
    report(summary, repeats)


def plot(summary):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2]/'.venv-objectives'/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(VARIANTS, ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple')))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, base in enumerate((0.1, 0.15)):
        for variant in VARIANTS:
            points = [r for r in summary if r['base_error']==base and r['variant']==variant]
            x = [r['strength'] for r in points]
            axes[0,col].plot(x, [r['recovery'] for r in points], marker='o', label=variant,
                             color=colors[variant])
            if variant in VARIANTS[1:3]:
                axes[1,col].errorbar(x, [100*r['paired_delta'] for r in points],
                    yerr=[[100*(r['paired_delta']-r['delta_ci_low']) for r in points],
                          [100*(r['delta_ci_high']-r['paired_delta']) for r in points]],
                    marker='o', capsize=3, label=variant, color=colors[variant])
        axes[0,col].set(title=f'Expected per-call error fixed at {base:.0%}', ylim=(0,1.02), ylabel='Exact recovery')
        axes[1,col].axhline(0, color='gray', linestyle='--')
        axes[1,col].set(xlabel='Length-error dependence (0 = IID control)', ylabel='Recovery minus baseline (percentage points)')
        for ax in axes[:,col]:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    fig.suptitle('5 SNPs; 12 reads; mixed lengths; paired replicates; approximate 95% delta CIs')
    fig.tight_layout()
    fig.savefig(OUT/'length_error_comparison.png', dpi=160)
    plt.close(fig)


def report(summary, repeats):
    lines = ['# Length-dependent error experiment', '',
             '**Question:** Does normalization help when longer reads have higher per-call error rates?', '',
             f'10 settings, {repeats} paired replicates per setting. Five SNPs, 12 reads, '
             'uniform interval lengths 2-5, no missing calls. All methods see identical observations. '
             'The expected error per observed call is held at 10% or 15%; only its allocation '
             'across lengths changes. Actual error fractions fluctuate.', '',
             'p(L) = base + strength * base / (27/7 - 2) * (L - 27/7). '
             'The center 27/7 is E[L^2]/E[L], ensuring equal expected per-call error budget. '
             'At strength=0, errors are IID. At strength=1, length-2 reads are error-free, '
             'while length-5 errors are about 1.615 times base.', '',
             'Recovery uses uniform tie breaking over exact minimizers. Differences below '
             'are paired per dataset; intervals are approximate 95% mean CIs. Wins/ties/losses '
             'compare replicate recovery probabilities, not individual quantum shots.', '',
             '| Base error | Strength | Method | Recovery | Delta vs A (pp) | 95% delta CI (pp) | Wins/ties/losses |',
             '|---|---|---|---:|---:|---|---|']
    for r in summary:
        if r['variant'] in VARIANTS[:3]:
            lines.append(f'| {r["base_error"]:.0%} | {r["strength"]:.2f} | {r["variant"]} | '
                         f'{r["recovery"]:.1%} | {100*r["paired_delta"]:+.2f} | '
                         f'[{100*r["delta_ci_low"]:+.2f}, {100*r["delta_ci_high"]:+.2f}] | '
                         f'{r["wins"]}/{r["ties"]}/{r["losses"]} |')
    lines += ['', '## Observed findings', '']
    for base in (0.1, 0.15):
        for strength in (0, 1):
            a = next(r for r in summary if r['base_error']==base and
                     r['strength']==strength and r['variant']==VARIANTS[0])
            b = next(r for r in summary if r['base_error']==base and
                     r['strength']==strength and r['variant']==VARIANTS[1])
            lines.append(f'- Base {base:.0%}, strength {strength}: A recovery '
                         f'{a["recovery"]:.2%}, B1 {b["recovery"]:.2%}; '
                         f'paired difference {100*b["paired_delta"]:+.2f} percentage points.')
    lines += ['', 'B1 changes from lower recovery in the IID controls to higher recovery '
              'under strong length-dependent errors in this simulation. This supports a '
              'conditional benefit, not a universal benefit or a claim about real sequencing. '
              'Quality can act as a confounder in the comparison of read lengths.', '',
              '## Interpretation', '',
              'Read the paired deltas: positive means higher recovery than baseline in that '
              'setting; negative means lower. An interval crossing zero does not establish '
              'a directional difference. Exploratory intervals are not adjusted for multiple '
              'comparisons. No outcome was used to tune the predeclared settings.', '',
              'This is a synthetic sensitivity study, not proof that real long reads are '
              'less reliable. No quality-aware optimal weighting or QAOA/hardware advantage '
              'is tested. Normalization depends only on length, so benefit is not guaranteed.', '',
              'Original results in objective_results are unchanged. Reproduce with '
              '`python experiments/objectives/length_error_experiment.py --repeats 1000`. '
              'Prepared with AI assistance.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=1000)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('Use at least two replicates for uncertainty estimates.')
    validate()
    print('Generator and fixed-error-budget checks passed.', flush=True)
    run(args.repeats)
