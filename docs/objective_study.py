"""Person 1: exact objective comparisons on five-SNP synthetic instances.

No Qiskit required. Shared interfaces: build_graph(reads, weight_fn),
evaluate(candidate, truth), and generate_reads(...).
"""

import csv
import itertools
from pathlib import Path

import numpy as np


VARIANTS = ('A baseline', 'B1 per-read', 'B2 per-site', 'C deduplicated', 'D MEC')
LENGTHS = {'short': (0.7, 0.3, 0.0, 0.0),
           'mixed': (0.25, 0.25, 0.25, 0.25),
           'long': (0.0, 0.0, 0.3, 0.7)}


def baseline_weight(length):
    return 1.0


def per_read_weight(length):
    return 2.0 / (length * (length - 1)) if length >= 2 else 0.0


def per_site_weight(length):
    return 1.0 / (length - 1) if length >= 2 else 0.0


def build_graph(reads, weight_fn=baseline_weight):
    """Return upper-triangular A, D, J and constant offset; preserve floats."""
    reads = np.asarray(reads)
    a = np.zeros((reads.shape[1], reads.shape[1]), dtype=float)
    d = np.zeros_like(a)
    for read in reads:
        observed = np.flatnonzero(read != -1)
        weight = weight_fn(len(observed)) if len(observed) >= 2 else 0.0
        for i, j in itertools.combinations(observed, 2):
            if read[i] == read[j]:
                a[i, j] += weight
            else:
                d[i, j] += weight
    return a, d, (d - a) / 2, float((a + d).sum() / 2)


def all_states(n):
    return np.array(list(itertools.product((0, 1), repeat=n)), dtype=int)


def pairwise_scores(states, a, d):
    scores = np.zeros(len(states), dtype=float)
    for i, j in itertools.combinations(range(states.shape[1]), 2):
        scores += np.where(states[:, i] == states[:, j], d[i, j], a[i, j])
    return scores


def mec_scores(states, reads):
    """Per read, choose its closest copy. Ignore missing calls, including empty reads."""
    scores = np.zeros(len(states), dtype=float)
    for read in reads:
        observed = read != -1
        mismatches = np.sum(states[:, observed] != read[observed], axis=1)
        scores += np.minimum(mismatches, observed.sum() - mismatches)
    return scores


def objective_scores(reads, states):
    graphs = {
        VARIANTS[0]: build_graph(reads),
        VARIANTS[1]: build_graph(reads, per_read_weight),
        VARIANTS[2]: build_graph(reads, per_site_weight),
        VARIANTS[3]: build_graph(np.unique(reads, axis=0)),
    }
    scores = {name: pairwise_scores(states, graph[0], graph[1])
              for name, graph in graphs.items()}
    scores[VARIANTS[4]] = mec_scores(states, reads)
    return scores, graphs


def evaluate(candidate, truth):
    distance = int(np.count_nonzero(np.asarray(candidate) != truth))
    errors = min(distance, len(truth) - distance)
    return {'hamming_errors': errors, 'exact_match': int(errors == 0)}


def optimal_mask(scores):
    return np.isclose(scores, np.min(scores), rtol=0, atol=1e-9)


def score_metrics(states, scores, truth):
    """Uniform tie policy: never select a minimizer using hidden truth.

    Average metrics over ALL minimizing labels, equivalent to uniform tie breaking.
    Complementary labels count as one biological phasing.
    """
    winners = optimal_mask(scores)
    distances = np.count_nonzero(states != truth, axis=1)
    errors = np.minimum(distances, len(truth) - distances)
    truth_index = np.flatnonzero(np.all(states == truth, axis=1))[0]
    truth_cost = scores[truth_index]
    return {
        'mean_hamming': float(errors[winners].mean()),
        'recovery': float(np.mean(errors[winners] == 0)),
        'truth_is_optimal': int(winners[truth_index]),
        'unique_truth': int(winners[truth_index] and winners.sum() == 2),
        'tied_labels': int(winners.sum()),
        'tied_phasings': int(winners.sum() // 2),
        'truth_rank': 1 + int(np.sum(scores < truth_cost - 1e-9)) // 2,
        'minimum': float(scores.min()),
    }


def generate_reads(truth, n_reads, length_mode, error_rate, missing_rate, seed):
    """IID reads: random copy, contiguous interval length 2..5, then missingness/errors.

    Length mode refers to interval length BEFORE independent missing calls.
    Identical rows arise naturally; no PCR-duplication model is assumed.
    """
    rng = np.random.default_rng(seed)
    n = len(truth)
    if n != 5:
        raise ValueError('This experiment defines length distributions for exactly 5 SNPs.')
    reads = np.full((n_reads, n), -1, dtype=int)
    for row in reads:
        length = int(rng.choice([2, 3, 4, 5], p=LENGTHS[length_mode]))
        start = int(rng.integers(n - length + 1))
        positions = np.arange(start, start + length)
        copy = truth if rng.integers(2) == 0 else 1 - truth
        calls = copy[positions].copy()
        calls ^= (rng.random(length) < error_rate).astype(int)
        calls[rng.random(length) < missing_rate] = -1
        row[positions] = calls
    return reads


def make_manifest(repeats=200):
    settings = [dict(length_mode=mode, error_rate=error, missing_rate=missing,
                     n_reads=count)
                for mode in LENGTHS for error in (0.0, 0.02, 0.05, 0.1)
                for missing in (0.0, 0.25) for count in (6, 12, 24)]
    return {'n_snps': 5, 'repeats': repeats,
            'truth_seed_base': 100_000, 'read_seed_base': 200_000,
            'seeds': [dict(replicate=i, truth_seed=100_000+i, read_seed=200_000+i)
                      for i in range(repeats)],
            'seed_policy': 'Seed bases + replicate index; same seeds across settings.',
            'settings': settings,
            'tie_policy': 'Uniform over all exact minimizing labels; no truth selection.',
            'generator': 'Contiguous intervals; IID errors and missing calls; random copy.',
            'metrics': 'Truth rank and ties count complementary labels as one phasing.'}


def run_study(manifest, output_dir=None):
    states = all_states(manifest['n_snps'])
    rows, cases, datasets = [], [], {}
    seeds = manifest['seeds']
    if len(seeds) != manifest['repeats']:
        raise ValueError('Seed list must match the requested number of replicates.')
    for setting_id, setting in enumerate(manifest['settings']):
        setting_reads, setting_truths = [], []
        for replicate, seed_pair in enumerate(seeds):
            truth_seed, read_seed = seed_pair['truth_seed'], seed_pair['read_seed']
            truth = np.random.default_rng(truth_seed).integers(2, size=manifest['n_snps'])
            reads = generate_reads(truth, seed=read_seed, **setting)
            setting_reads.append(reads)
            setting_truths.append(truth)
            scores, _ = objective_scores(reads, states)
            baseline_winners = optimal_mask(scores[VARIANTS[0]])
            lengths = np.sum(reads != -1, axis=1)
            for name, costs in scores.items():
                row = dict(setting_id=setting_id, replicate=replicate,
                           truth_seed=truth_seed, read_seed=read_seed, **setting,
                           variant=name, observed_mean_length=float(lengths.mean()),
                           **score_metrics(states, costs, truth))
                row['disagrees_baseline'] = int(not np.array_equal(
                    baseline_winners, optimal_mask(costs)))
                rows.append(row)
            if len(cases) < 12 and not np.array_equal(
                    baseline_winners, optimal_mask(scores[VARIANTS[1]])):
                cases.append(dict(setting_id=setting_id, replicate=replicate,
                                  **setting, truth=truth.tolist(), reads=reads.tolist(),
                                  candidates=states.tolist(),
                                  scores={k: v.tolist() for k, v in scores.items()},
                                  optima={k: states[optimal_mask(v)].tolist()
                                          for k, v in scores.items()}))
        datasets[f'reads_{setting_id}'] = np.array(setting_reads)
        datasets[f'truth_{setting_id}'] = np.array(setting_truths)
    if output_dir is not None:
        np.savez_compressed(Path(output_dir) / 'datasets.npz', **datasets)
    return rows, cases


def mean_ci(values):
    """Approximate 95% normal CI for a mean of independent replicate-level metrics."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, mean - 1.96 * se, mean + 1.96 * se


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['setting_id'], row['variant']), []).append(row)
    summaries = []
    for (_, variant), group in groups.items():
        first = group[0]
        out = {key: first[key] for key in
               ('setting_id', 'length_mode', 'error_rate', 'missing_rate', 'n_reads')}
        out.update(variant=variant, replicates=len(group))
        for metric in ('recovery', 'mean_hamming', 'truth_is_optimal', 'unique_truth',
                       'tied_phasings', 'truth_rank', 'observed_mean_length',
                       'disagrees_baseline'):
            mean, low, high = mean_ci([r[metric] for r in group])
            if metric == 'recovery':
                low, high = max(0.0, low), min(1.0, high)
            elif metric in ('truth_is_optimal', 'unique_truth', 'disagrees_baseline'):
                # Wilson interval for binary indicators; recovery can be fractional.
                z, n = 1.96, len(group)
                center = (mean + z*z/(2*n))/(1 + z*z/n)
                half = z * np.sqrt(mean*(1-mean)/n + z*z/(4*n*n))/(1+z*z/n)
                low, high = center-half, center+half
            out[metric] = mean
            out[metric + '_ci_low'], out[metric + '_ci_high'] = low, high
        summaries.append(out)
    return summaries


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary, output_dir):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    # Hold missingness and coverage fixed; never pool different conditions.
    fixed = [r for r in summary if r['missing_rate'] == 0 and r['n_reads'] == 12]
    repeats = fixed[0]['replicates']
    plots = [
        ('length', 'error_rate', (0, 0.02, 0.05, 0.1),
         'observed_mean_length', 'Mean observed read length'),
        ('error', 'length_mode', tuple(LENGTHS),
         'error_rate', 'Per-call error probability'),
    ]
    for filename, panel_key, panels, x_key, xlabel in plots:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.5*len(panels), 4), sharey=True)
        for ax, panel in zip(axes, panels):
            for name in VARIANTS:
                points = sorted([r for r in fixed if r[panel_key] == panel and
                                 r['variant'] == name], key=lambda r: r[x_key])
                ax.errorbar([r[x_key] for r in points], [r['recovery'] for r in points],
                            yerr=[[r['recovery']-r['recovery_ci_low'] for r in points],
                                  [r['recovery_ci_high']-r['recovery'] for r in points]],
                            marker='o', capsize=2, label=name)
            title = f'Error rate {panel:.0%}' if panel_key == 'error_rate' else panel
            ax.set(title=title, xlabel=xlabel, ylim=(-0.03, 1.03))
        axes[0].set_ylabel('Exact recovery (uniform tie policy)')
        axes[-1].legend(fontsize=8)
        fig.suptitle(f'12 reads; no missing calls; {repeats} replicates per setting; 95% mean CIs')
        fig.tight_layout()
        fig.savefig(output_dir / f'accuracy_vs_{filename}.png', dpi=160)
        plt.close(fig)
