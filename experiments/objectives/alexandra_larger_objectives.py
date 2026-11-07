"""Compare A/B1/B2/C/MEC on the team's larger synthetic instances.

Run from the repository root:
  .venv-objectives/Scripts/python.exe experiments/objectives/alexandra_larger_objectives.py
Loads only selected definitions from the shared notebook, never its experiments.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments'/'objectives'))
from objective_study import (
    VARIANTS, build_graph, mec_scores, optimal_mask, pairwise_scores,
    per_read_weight, per_site_weight, write_csv,
)

OUT = ROOT/'results'/'objectives'/'alexandra_larger_results'


def load_shared():
    """Use actual team generator/reference functions without importing Qiskit."""
    path = ROOT/'experiments'/'bench'/'qgd_bench.ipynb'
    wanted = {'Instance', 'make_instance', 'pairwise_evidence', 'bandwidth',
              'banded_exact', 'linked_boundaries', 'switch_errors'}
    module = types.ModuleType('_alexandra_shared_definitions')
    module.__dict__.update(np=np, math=math, dataclass=dataclass, field=field)
    sys.modules[module.__name__] = module  # dataclasses resolve their defining module
    found = set()
    for cell in json.loads(path.read_text(encoding='utf-8'))['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
                if isinstance(node, ast.FunctionDef):
                    # Registry decorators belong to the benchmark plug-in loader;
                    # the evidence function body is reused unchanged here.
                    node.decorator_list = []
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'),
                     module.__dict__)
                found.add(node.name)
    if found != wanted:
        raise ValueError(f'Missing shared definitions: {wanted-found}')
    # Read, rather than retype, Tudor's realistic generator settings.
    real = None
    experiment = ROOT/'experiments'/'bench'/'tudor_classical_vs_quantum.ipynb'
    for cell in json.loads(experiment.read_text(encoding='utf-8'))['cells']:
        for line in ''.join(cell['source']).splitlines():
            if line.startswith('REAL = '):
                assignment = ast.parse(line).body[0]
                call = assignment.value
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != 'dict':
                    raise ValueError('Unsupported REAL declaration; inspect shared settings.')
                real = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    if real is None:
        raise ValueError('Shared REAL settings not found.')
    return module, real


def decode_representatives(indices, n):
    """Exactly one label per complement pair, with SNP 1 fixed to zero."""
    indices = np.asarray(indices, dtype=np.uint64)
    states = np.zeros((len(indices), n), dtype=np.int8)
    states[:, 1:] = (indices[:, None] >> np.arange(n-2, -1, -1, dtype=np.uint64)) & 1
    return states


def graphs_for(reads):
    return {
        VARIANTS[0]: build_graph(reads),
        VARIANTS[1]: build_graph(reads, per_read_weight),
        VARIANTS[2]: build_graph(reads, per_site_weight),
        VARIANTS[3]: build_graph(np.unique(reads, axis=0)),
    }


def score(states, reads, graphs):
    values = {name: pairwise_scores(states, g[0], g[1]) for name, g in graphs.items()}
    values[VARIANTS[4]] = mec_scores(states, reads)
    return values


def compare_instance(instance, engine, chunk_size=16384):
    n, reads, truth = instance.n, instance.reads, instance.truth
    if not 2 <= n <= 20:
        raise ValueError('Exact five-objective study is deliberately capped at 20 SNPs.')
    start = time.perf_counter()
    graphs = graphs_for(reads)
    full_a, full_d = engine.pairwise_evidence(reads)
    assert np.allclose((full_a, full_d), graphs[VARIANTS[0]][:2])
    phases = 2**(n-1)
    costs = {name: np.empty(phases) for name in VARIANTS}
    for first in range(0, phases, chunk_size):
        states = decode_representatives(np.arange(first, min(first+chunk_size, phases)), n)
        for name, values in score(states, reads, graphs).items():
            costs[name][first:first+len(states)] = values
    # Validate against the team's independent banded DP for pairwise variants.
    for name, (_, _, j, offset) in graphs.items():
        reference = engine.banded_exact(j, offset)
        assert reference is not None and np.isclose(costs[name].min(), reference)
    truth_scores = score(truth[None, :], reads, graphs)
    linked = engine.linked_boundaries(full_a, full_d)
    full, mec = costs[VARIANTS[0]], costs[VARIANTS[4]]
    truth_rep = truth if truth[0] == 0 else 1-truth
    truth_index = int(''.join(map(str, truth_rep[1:])), 2)
    rows, winners = [], {}
    for name, native in costs.items():
        mask = optimal_mask(native)
        indices = np.flatnonzero(mask)
        winners[name] = indices
        hamming_sum = switch_sum = 0
        for first in range(0, len(indices), chunk_size):
            states = decode_representatives(indices[first:first+chunk_size], n)
            distance = np.sum(states != truth, axis=1)
            hamming_sum += np.minimum(distance, n-distance).sum()
            difference = states ^ truth
            switch_sum += np.sum((difference[:, 1:] != difference[:, :-1]) & linked)
        assert np.isclose(native[truth_index], truth_scores[name][0])
        chosen = decode_representatives(indices[:1], n)[0]
        total = len(indices)
        rows.append(dict(instance=instance.name, n_snps=n, seed=instance.params['seed'],
                         n_reads=len(reads), unique_reads=len(np.unique(reads, axis=0)),
                         observed_mean_length=float(np.mean(np.sum(reads != -1, axis=1))),
                         variant=name, native_minimum=float(native.min()),
                         native_truth_cost=float(truth_scores[name][0]),
                         truth_is_optimal=int(mask[truth_index]),
                         recovery=float(mask[truth_index]/total),
                         mean_hamming=float(hamming_sum/total),
                         hamming_fraction=float(hamming_sum/total/n),
                         mean_switch_errors=float(switch_sum/total),
                         linked_boundaries=int(linked.sum()),
                         tied_phasings=total, tied_labels=2*total,
                         truth_rank=1+int(np.sum(native < truth_scores[name][0]-1e-9)),
                         selected_candidate=''.join(map(str, chosen)),
                         # Common-reference evaluation, NOT raw native-score comparison.
                         full_mean=float(full[mask].mean()),
                         full_optimum_rate=float(optimal_mask(full)[mask].mean()),
                         mec_mean=float(mec[mask].mean()),
                         mec_optimum_rate=float(optimal_mask(mec)[mask].mean()),
                         disagrees_A=int(not np.array_equal(mask, optimal_mask(full)))))
    return rows, costs, winners, time.perf_counter()-start


def checks(engine, real):
    from objective_study import all_states, objective_scores, score_metrics
    toy = engine.make_instance(6, seed=81, **real)
    rows, _, _, _ = compare_instance(toy, engine, chunk_size=7)
    states = all_states(6)
    scores, _ = objective_scores(toy.reads, states)
    for row in rows:
        expected = score_metrics(states, scores[row['variant']], toy.truth)
        for key in ('recovery', 'mean_hamming', 'truth_is_optimal', 'tied_labels',
                    'tied_phasings', 'truth_rank'):
            assert np.isclose(row[key], expected[key]), (row['variant'], key)
        x = states[optimal_mask(scores[row['variant']])]
        a, d = engine.pairwise_evidence(toy.reads)
        linked = engine.linked_boundaries(a, d)
        assert np.isclose(row['mean_switch_errors'], np.mean([
            engine.switch_errors(candidate, toy.truth, linked) for candidate in x]))
    assert np.array_equal(decode_representatives(np.arange(32), 6), states[:32])
    print('Checks passed: chunking/complement reduction vs full enumeration; shared DP; switch metrics.', flush=True)


def plot(rows):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.venv-objectives'/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sizes = sorted({r['n_snps'] for r in rows})
    for name in VARIANTS:
        groups = [[r for r in rows if r['n_snps']==n and r['variant']==name] for n in sizes]
        for ax, metric in zip(axes, ('recovery', 'hamming_fraction')):
            ax.plot(sizes, [np.mean([r[metric] for r in group]) for group in groups],
                    marker='o', label=name)
            ax.set_xticks(sizes)
            ax.grid(alpha=.2)
            ax.set_xlabel('Number of SNPs')
    axes[0].set_ylabel('Mean exact recovery (uniform tie policy)')
    axes[0].set_ylim(-.02, 1.02)
    axes[1].set_ylabel('Mean fraction of wrong SNPs, modulo complement')
    axes[1].legend(fontsize=8)
    seeds = len({r['seed'] for r in rows})
    fig.suptitle(f'Shared team generator; {seeds} seeds per size; descriptive pilot, not a significance test')
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png', dpi=160)
    plt.close(fig)


def report(rows, real, timings):
    lines = ['# Larger-instance objective comparison', '',
             'Source: `experiments/bench/qgd_bench.ipynb::make_instance`. The realistic error/missing '
             'settings are loaded directly from `experiments/bench/tudor_classical_vs_quantum.ipynb::REAL`. '
             'The 12-SNP seed-0 input reproduces that notebook\'s larger demo.', '',
             f'Settings: {real}; generator defaults: coverage=6, read_len=(2,4), gaps=0. '
             'The generator repairs uncovered adjacent boundaries with extra two-site reads; '
             'these reads can also contain errors. This differs from Alexandra\'s original generator.', '',
             'All five methods receive identical observations per instance. Every native '
             'objective, including full-read MEC, is solved exactly. One representative per '
             'complement pair is enumerated in chunks, with a hard cap of 20 SNPs. '
             'Pairwise minima are checked against the team\'s banded dynamic program. '
             'MEC is not substituted into a pairwise Hamiltonian.', '',
             'Truth is used only for evaluation. Metrics average over ALL exact minimizers; '
             'no winner is selected using hidden truth. The selected candidate is the first '
             'lexicographic representative. Switch errors use the FULL original read graph\'s '
             'linked boundaries for all methods. Full pairwise and full-read MEC are common '
             'reference costs in results.csv; differently scaled native costs should not be '
             'compared across variants.', '',
             '| SNPs | Seed | Reads | Method | Recovery | Mean Hamming | Mean switches | Optimal phasings |',
             '|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["n_snps"]} | {r["seed"]} | {r["n_reads"]} | {r["variant"]} | '
                     f'{r["recovery"]:.1%} | {r["mean_hamming"]:.3f} | '
                     f'{r["mean_switch_errors"]:.3f} | {r["tied_phasings"]} |')
    lines += ['', '## Observed results', '']
    for n in sorted({r['n_snps'] for r in rows}):
        group = [r for r in rows if r['n_snps']==n]
        if all(r['recovery']==1 for r in group):
            lines.append(f'- At {n} SNPs, all five methods uniquely recover truth on all '
                         f'{len({r["seed"] for r in group})} tested seeds.')
        for seed in sorted({r['seed'] for r in group}):
            case = [r for r in group if r['seed']==seed]
            if len({r['recovery'] for r in case}) > 1:
                detail = '; '.join(f'{r["variant"]}: recovery={r["recovery"]:.0%}, '
                                   f'Hamming={r["mean_hamming"]:g}, '
                                   f'switches={r["mean_switch_errors"]:g}' for r in case)
                lines.append(f'- At {n} SNPs, seed {seed}: {detail}.')
    lines += ['', 'An individual case where B1 helps does not overturn the original '
              'distribution-level results or establish that B1 is universally better. '
              'It shows that the choice of objective can matter on the team generator too.', '',
              '## What to inspect', '',
              '- Compare recovery, Hamming and switch errors on the same instance.',
              '- Compare full_mean and mec_mean under common references, not native minima.',
              '- Equal native optimum cost does not imply identical minimizing sets.',
              '- Each size uses only a few seeds: this is a descriptive pilot, not evidence '
              'of universal superiority or a timing/speedup benchmark.', '',
              '## Exact-run wall time (all five methods together)', '']
    lines += [f'- {name}: {seconds:.2f} seconds.' for name, seconds in timings.items()]
    lines += ['', 'Reproduce: `python experiments/objectives/alexandra_larger_objectives.py --sizes 12 16 20 --seeds 0 1 2`.',
              'Outputs include inputs.npz, manifest.json, results.csv, comparison.png and report.md. '
              'The module and original notebooks are unchanged. No QAOA/hardware run or '
              'quantum-advantage claim. Prepared with AI assistance.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[12, 16, 20])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if any(n < 2 or n > 20 for n in args.sizes):
        parser.error('Use 2-20 SNPs; larger exact enumeration is intentionally disabled.')
    engine, real = load_shared()
    checks(engine, real)
    OUT.mkdir(exist_ok=True)
    manifest = dict(sizes=args.sizes, seeds=args.seeds, realistic_settings=real,
                    generator='experiments/bench/qgd_bench.ipynb::make_instance',
                    generator_notebook_sha256=hashlib.sha256((ROOT/'experiments'/'bench'/'qgd_bench.ipynb').read_bytes()).hexdigest(),
                    defaults=dict(coverage=6, read_len=[2,4], gaps=0),
                    enumeration='First SNP fixed to 0; exact over complementary phasing pairs',
                    tie_policy='Uniform over ALL exact minimizers; truth only for evaluation')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows, inputs, timings = [], {}, {}
    for n in args.sizes:
        for seed in args.seeds:
            instance = engine.make_instance(n, seed=seed, **real)
            print(f'Running {instance.name}: {len(instance.reads)} reads; {2**(n-1):,} complementary pairs.', flush=True)
            result, _, _, elapsed = compare_instance(instance, engine)
            rows.extend(result)
            timings[instance.name] = elapsed
            inputs[instance.name+'_reads'] = instance.reads
            inputs[instance.name+'_truth'] = instance.truth
            print(f'Completed {instance.name} in {elapsed:.2f}s.', flush=True)
    write_csv(OUT/'results.csv', rows)
    np.savez_compressed(OUT/'inputs.npz', **inputs)
    plot(rows)
    report(rows, real, timings)
    print(f'Saved results in {OUT}', flush=True)


if __name__ == '__main__':
    main()
