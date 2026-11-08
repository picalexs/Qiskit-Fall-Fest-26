"""Fixed observed read length 3 vs 5, with two observation-budget controls.

python experiments/objectives/fixed_length_experiment.py --repeats 1000
No missing calls: otherwise observed lengths would no longer be fixed.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from objective_study import (
    all_states, build_graph, mean_ci, optimal_mask, pairwise_scores,
    per_read_weight, per_site_weight, score_metrics, write_csv,
)

OUT = Path(__file__).resolve().parents[2] / 'results' / 'objectives' / 'fixed_length_results'


def generate_fixed(truth, length, n_reads, error_rate, seed):
    rng = np.random.default_rng(seed)
    origins = rng.integers(2, size=n_reads)
    starts = np.floor(rng.random(n_reads) * (6-length)).astype(int)
    error_uniforms = rng.random((n_reads, 5))
    reads = np.full((n_reads, 5), -1, dtype=int)
    for i, (origin, start) in enumerate(zip(origins, starts)):
        positions = np.arange(start, start+length)
        copy = truth if origin == 0 else 1-truth
        reads[i, positions] = copy[positions] ^ (error_uniforms[i, positions] < error_rate)
    return reads


def run(repeats):
    OUT.mkdir(parents=True, exist_ok=True)
    states = all_states(5)
    settings = []
    for budget in ('12 reads', '60 observed calls'):
        for length in (3, 5):
            for error in (0, 0.05, 0.1, 0.15):
                settings.append(dict(budget=budget, length=length,
                                     n_reads=12 if budget=='12 reads' else 60//length,
                                     error_rate=error))
    manifest = dict(repeats=repeats, n_snps=5, missing_rate=0, settings=settings,
                    truth_seeds=list(range(510_000, 510_000+repeats)),
                    read_seeds=list(range(610_000, 610_000+repeats)),
                    model='Contiguous fixed-length reads; IID per-call errors; random copy',
                    tie_policy='Uniform over all exact minimizing labels; modulo complement',
                    controls='Same read count vs same total observed-call count',
                    note='Overlapping settings across protocols are deliberately identical; not independent evidence')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows, summary, datasets = [], [], {}
    for sid, setting in enumerate(settings):
        group, saved_reads, saved_truths = [], [], []
        for rep in range(repeats):
            truth = np.random.default_rng(510_000+rep).integers(2, size=5)
            reads = generate_fixed(truth, setting['length'], setting['n_reads'],
                                   setting['error_rate'], 610_000+rep)
            length = setting['length']
            assert np.all(np.sum(reads != -1, axis=1) == length)
            a, d, _, _ = build_graph(reads)
            costs = pairwise_scores(states, a, d)
            for weight in (per_read_weight, per_site_weight):
                aw, dw, _, _ = build_graph(reads, weight)
                weighted = pairwise_scores(states, aw, dw)
                assert np.allclose(weighted, costs * weight(length))
                assert np.array_equal(optimal_mask(costs), optimal_mask(weighted))
            if setting['error_rate'] == 0:
                assert score_metrics(states, costs, truth)['truth_is_optimal']
            row = dict(setting_id=sid, replicate=rep, **setting,
                       observed_calls=setting['n_reads']*length,
                       variant='A = B1 = B2 (verified)',
                       **score_metrics(states, costs, truth))
            rows.append(row)
            group.append(row)
            saved_reads.append(reads)
            saved_truths.append(truth)
        datasets[f'reads_{sid}'] = np.array(saved_reads)
        datasets[f'truth_{sid}'] = np.array(saved_truths)
        result = dict(setting_id=sid, **setting, observed_calls=setting['n_reads']*setting['length'],
                      replicates=repeats)
        for metric in ('recovery', 'mean_hamming', 'truth_is_optimal', 'tied_phasings'):
            mean, low, high = mean_ci([r[metric] for r in group])
            if metric in ('recovery', 'truth_is_optimal'):
                low, high = max(0, low), min(1, high)
            result.update({metric:mean, metric+'_ci_low':low, metric+'_ci_high':high})
        summary.append(result)
        print(f'Setting {sid+1}/{len(settings)}: {setting}', flush=True)
    write_csv(OUT/'replicate_metrics.csv', rows)
    write_csv(OUT/'summary.csv', summary)
    np.savez_compressed(OUT/'datasets.npz', **datasets)
    plot(summary, repeats)
    report(summary, repeats)


def plot(summary, repeats):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2]/'.venv-objectives'/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, budget in zip(axes, ('12 reads', '60 observed calls')):
        for length in (3, 5):
            points = [r for r in summary if r['budget']==budget and r['length']==length]
            ax.errorbar([r['error_rate'] for r in points], [r['recovery'] for r in points],
                        yerr=[[r['recovery']-r['recovery_ci_low'] for r in points],
                              [r['recovery_ci_high']-r['recovery'] for r in points]],
                        label=f'L={length}; {points[0]["n_reads"]} reads', marker='o', capsize=3)
        ax.set(title=f'Fixed budget: {budget}', xlabel='Per-call error probability', ylim=(0,1.03))
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('Exact recovery (uniform tie policy)')
    fig.suptitle(f'5 SNPs; no missing calls; {repeats} replicates; A, B1, B2 have identical optima')
    fig.tight_layout()
    fig.savefig(OUT/'length_3_vs_5.png', dpi=160)
    plt.close(fig)


def report(summary, repeats):
    text = ['# Fixed length 3 versus 5', '',
            f'16 settings x {repeats} replicates. Five SNPs, contiguous reads, IID errors, '
            'no missing calls. Shared replicate seeds across conditions; repeated settings '
            'between budget protocols are identical, not independent evidence.', '',
            '## Exact property verified', '',
            'For every dataset and all 32 candidates: B1 cost = A cost * 2/[L(L-1)] '
            'and B2 cost = A cost/(L-1). Positive scaling preserves every minimizer. '
            'Recovery therefore coincides for A, B1, B2. A single row per dataset '
            'records their common metrics rather than triplicating identical results.', '',
            '## Budget controls', '',
            '- Same read count: 12 reads of L=3 give 36 observed calls; L=5 gives 60.',
            '- Same observed-call count: 20 reads of L=3 and 12 of L=5 both give 60 calls.',
            '- Equal call count does not equalize linkage/connectivity: L=5 reads span '
            'the entire five-site problem. Do not generalize to larger genomes.', '',
            '| Budget | L | Reads | Error | Recovery | Approx. 95% CI |',
            '|---|---:|---:|---:|---:|---|']
    for r in summary:
        text.append(f'| {r["budget"]} | {r["length"]} | {r["n_reads"]} | {r["error_rate"]:.0%} | '
                    f'{r["recovery"]:.1%} | [{r["recovery_ci_low"]:.1%}, {r["recovery_ci_high"]:.1%}] |')
    text += ['', '## Meaning', '',
             'This tests information available at different read lengths. It cannot '
             'distinguish normalization performance when all observed lengths are equal. '
             'Mixed lengths are required for A/B1/B2 to change candidate rankings. '
             'No QAOA run or quantum advantage is tested. Intervals are approximate '
             'replicate-level mean intervals; they are not multiple-comparison-adjusted. '
             'Original experiments remain unchanged.', '',
             'Reproduce: `python experiments/objectives/fixed_length_experiment.py --repeats 1000`.']
    (OUT/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=1000)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('At least two replicates are required.')
    run(args.repeats)
