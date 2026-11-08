#!/usr/bin/env python
"""Plot saved QAOA backend-comparison results without a quantum backend.

Run from anywhere: python bench/plots/make_plots.py [--run RUN] [--depth 2]
Use --help for output and format options. See README.md for interpretation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pandas as pd

METHODS = ('standard', 'warm-start', 'multi-angle')
CYAN, GOLD, GREY, INK, TIE = '#6ACCE6', '#F7BD69', '#888888', '#14243A', '#A78BFA'
PALE_CYAN, PALE_GOLD, PANEL, GRID = '#CDEBF4', '#FBE9CC', '#EEF2F6', '#E7ECF2'
COLORS = {'standard': GREY, 'warm-start': GOLD, 'multi-angle': CYAN}
SEQUENTIAL = LinearSegmentedColormap.from_list('lab_sequential', ['#F4F7FA', PALE_CYAN, CYAN, INK])
DIVERGING = LinearSegmentedColormap.from_list('lab_diverging', [CYAN, '#F4F7FA', GOLD])
READS_CMAP = ListedColormap([GREY, PALE_CYAN, PALE_GOLD])
MARKERS = {'standard': 'o', 'warm-start': 's', 'multi-angle': 'D'}
LABELS = {'standard': 'Standard QAOA', 'warm-start': 'Warm-start QAOA', 'multi-angle': 'Multi-angle QAOA'}
TECHNIQUES = {'dd': 'X–X DD', 'twirling': 'Pauli twirling', 'readout': 'Readout correction', 'trex': 'TREX-style (local response)', 'zne': 'ZNE'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.grid': True,
                     'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '#9AA7B8',
                     'xtick.color': INK, 'ytick.color': INK, 'grid.color': GRID,
                     'axes.prop_cycle': matplotlib.cycler(color=[GREY, GOLD, CYAN, TIE, INK]),
                     'grid.alpha': 1, 'grid.linestyle': ':', 'legend.frameon': False,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42})


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def latest_run(root):
    """Choose the most recent complete run with results, ignoring partial folders."""
    base = root / 'backend_comparison'
    for path in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
        try:
            manifest = read_json(path / 'manifest.json')
            if manifest.get('status') == 'complete' and (path / 'results.csv').is_file():
                return path
        except (OSError, ValueError):
            continue
    raise FileNotFoundError(f'No completed run with results.csv in {base}')


def default_output_path(run):
    """Keep each run's generated plots together beside this plotting script."""
    return Path(__file__).resolve().parent / 'generated' / Path(run).resolve().name


def output_file(folder, name):
    """Group generated figures and their source tables by file format."""
    target = Path(folder) / Path(name).suffix.lstrip('.').lower() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def load_run(path):
    data = {'path': path, 'manifest': read_json(path / 'manifest.json')}
    for name in ('results', 'techniques'):
        file = path / f'{name}.csv'
        if file.exists():
            data[name] = pd.read_csv(file)
    for name in ('problem', 'training', 'compiled_stats', 'ideal_probabilities'):
        file = path / f'{name}.json'
        if file.exists():
            data[name] = read_json(file)
    # Records are the source of truth. Rebuild the table when the notebook has
    # just saved measurements and has not previously generated any plots.
    frequencies, records = frequencies_from_records(path, data.get('problem', {}))
    data['records'] = records
    if not frequencies.empty:
        data['frequencies'] = frequencies
    else:
        file = path / 'plots' / 'outcome_frequencies.csv'
        if file.exists():
            data['frequencies'] = pd.read_csv(file, dtype={'state_snp_first': str})
    return data


def frequencies_from_records(path, problem):
    """Reconstruct all raw outcome frequencies from saved, per-PUB counts.

    Qiskit displays the highest classical bit first; saved state labels use
    SNP 1 first, so reverse each measurement string as in the notebook.
    """
    files = sorted((path / 'records').glob('*.json'))
    records = [read_json(file) for file in files]
    reads = problem.get('reads')
    if not reads:
        return pd.DataFrame(), records
    n = len(reads[0])
    same = np.asarray(problem['same'])
    opposite = np.asarray(problem['opposite'])
    states = [''.join(bits) for bits in itertools.product('01', repeat=n)]
    costs = {
        state: float(sum(opposite[i, j] if state[i] == state[j] else same[i, j]
                         for i, j in itertools.combinations(range(n), 2)))
        for state in states
    }
    rows = []
    for record in records:
        if record.get('stage') not in ('ideal', 'noisy', 'real') or record.get('technique') != 'raw':
            continue
        counts = record.get('counts') or {}
        total = sum(counts.values())
        if total <= 0:
            raise ValueError(f'No measurements for record {record.get("key", "unknown")}')
        measured = {}
        for label, count in counts.items():
            state = label.replace(' ', '')[::-1]
            if state not in costs:
                raise ValueError(f'Unexpected count bitstring: {label}')
            measured[state] = measured.get(state, 0) + count
        for state in states:
            rows.append({key: record[key] for key in ('stage', 'backend', 'method', 'depth', 'repeat')}
                        | {'state_snp_first': state, 'probability': measured.get(state, 0) / total,
                           'full_cost': costs[state], 'shots': total})
    return pd.DataFrame(rows), records


def evaluation_rows(results):
    """Keep independent raw evaluation only; never mix in champion screening."""
    rows = results[results['stage'].isin(['ideal', 'noisy', 'real'])].copy()
    if 'technique' in rows:
        rows = rows[rows['technique'].eq('raw')]
    return rows


def environments(frame):
    env = []
    if (frame.stage == 'ideal').any():
        env.append(('ideal', 'ideal'))
    for backend in sorted(frame.loc[frame.stage.isin(['noisy', 'real']), 'backend'].dropna().unique()):
        for stage in ('noisy', 'real'):
            if ((frame.stage == stage) & (frame.backend == backend)).any():
                env.append((stage, backend))
    return env


def env_title(stage, backend):
    if stage == 'ideal':
        return 'Ideal simulation'
    label = backend.replace('ibm_', '').replace('_', ' ').title()
    return f'{label} · ' + ('calibrated noise' if stage == 'noisy' else 'real device')


def select_env(frame, stage, backend):
    return frame[(frame.stage == stage) & (frame.backend == backend)]


def save(fig, out, name, formats, dpi):
    for fmt in formats:
        fig.savefig(output_file(out, f'{name}.{fmt}'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_metric(data, out, metric, title, ylabel, name, formats, dpi, probability=False):
    rows = evaluation_rows(data['results'])
    if metric not in rows:
        return
    measured = rows.dropna(subset=[metric]).copy()
    if measured.empty:
        return
    # Preserve the individual measured repeats, even though the cells show means.
    measured[['stage', 'backend', 'method', 'depth', 'repeat', metric]].to_csv(
        output_file(out, f'{name}_data.csv'), index=False)

    config = data.get('manifest', {}).get('config', {})
    method_names = set(rows.method.dropna()) | set(config.get('methods', []))
    methods = [m for m in METHODS if m in method_names] + sorted(method_names - set(METHODS))
    depth_values = set(rows.depth.dropna()) | set(config.get('depths', []))
    depths = sorted(depth_values)
    backend_names = set(rows.loc[rows.stage.isin(['noisy', 'real']), 'backend'].dropna())
    backend_names.update(config.get('backends', []))
    backend_names.discard('ideal')
    envs = [('ideal', 'ideal')] + [(stage, backend) for backend in sorted(backend_names)
                                   for stage in ('noisy', 'real')]
    means = measured.groupby(['method', 'stage', 'backend', 'depth'])[metric].mean()
    values = np.full((len(methods) * len(envs), len(depths)), np.nan)
    labels = []
    for i, method in enumerate(methods):
        method_label = LABELS.get(method, method).replace(' QAOA', '')
        for j, (stage, backend) in enumerate(envs):
            env = 'Ideal · sampled' if stage == 'ideal' else (
                backend.replace('ibm_', '').replace('_', ' ').title() +
                (' · noisy' if stage == 'noisy' else ' · hardware'))
            labels.append(f'{method_label}  |  {env}')
            for k, depth in enumerate(depths):
                values[i * len(envs) + j, k] = means.get((method, stage, backend, depth), np.nan)

    fig, ax = plt.subplots(figsize=(9.5, max(5, .37 * len(labels) + 2.2)))
    cmap = (SEQUENTIAL if probability else SEQUENTIAL.reversed()).with_extremes(bad=PANEL)
    bounds = {'vmin': 0, 'vmax': 1} if probability or metric == 'total_variation_from_ideal' else {}
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect='auto',
                      interpolation='nearest', **bounds)
    ax.grid(False)
    ax.set_xticks(range(len(depths)), [str(int(depth)) for depth in depths])
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel('QAOA depth $p$')
    for i in range(1, len(methods)):
        ax.axhline(i * len(envs) - .5, color='white', lw=2)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                ax.text(j, i, '—', ha='center', va='center', color='#888888', fontsize=10)
                continue
            color = image.cmap(image.norm(value))
            brightness = .2126 * color[0] + .7152 * color[1] + .0722 * color[2]
            text_color = '#14243A' if brightness > .55 else 'white'
            label = f'{100 * value:.1f}%' if probability else f'{value:.2f}'
            ax.text(j, i, label, ha='center', va='center', color=text_color, fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, pad=.03, shrink=.92)
    colorbar.set_label(ylabel)
    if probability:
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle(title + ' · repeat means', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, .04, 1, .96))
    save(fig, out, name, formats, dpi)


def exact_reference_metrics(data):
    """Score saved exact distributions in the notebook's SNP-first state order."""
    problem = data.get('problem', {})
    probabilities = data.get('ideal_probabilities', {})
    if not probabilities or not {'reads', 'same', 'opposite'} <= problem.keys():
        return pd.DataFrame()
    reads = np.asarray(problem['reads'], dtype=int)
    n = reads.shape[1]
    states = np.asarray(list(itertools.product((0, 1), repeat=n)), dtype=int)
    same, opposite = np.asarray(problem['same']), np.asarray(problem['opposite'])
    costs = np.zeros(len(states))
    for i, j in itertools.combinations(range(n), 2):
        costs += np.where(states[:, i] == states[:, j], opposite[i, j], same[i, j])
    mec = np.zeros(len(states))
    for read in reads:
        seen = read != -1
        distance = (states[:, seen] != read[seen]).sum(axis=1)
        mec += np.minimum(distance, seen.sum() - distance)
    rows = []
    for key, values in probabilities.items():
        method, depth = key.rsplit('|', 1)
        vector = np.asarray(values, dtype=float)
        if len(vector) != len(states) or not np.isfinite(vector).all() or not np.isclose(vector.sum(), 1):
            raise ValueError(f'Invalid exact ideal probability vector for {key}')
        rows.append({'method': method, 'depth': int(depth),
                     'ideal_full_optimum_probability': float(vector[np.isclose(costs, costs.min())].sum()),
                     'ideal_mean_full_cost': float(vector @ costs), 'ideal_mean_mec': float(vector @ mec)})
    return pd.DataFrame(rows)


def noise_comparison_data(data):
    """Join each independent raw repeat to its own method/depth exact ideal."""
    ideal = exact_reference_metrics(data)
    if ideal.empty:
        return pd.DataFrame()
    raw = evaluation_rows(data['results'])
    raw = raw[raw.stage.isin(['noisy', 'real'])]
    merged = raw.merge(ideal, on=['method', 'depth'], how='inner', validate='many_to_one')
    for metric in ('full_optimum_probability', 'mean_full_cost', 'mean_mec'):
        if metric in merged:
            merged[f'delta_{metric}'] = merged[metric] - merged[f'ideal_{metric}']
    return merged


def plot_noise_comparisons(data, out, depth, formats, dpi):
    compared = noise_comparison_data(data)
    if compared.empty:
        print('  Noise comparisons skipped: exact ideal or raw noisy/real results are unavailable.')
        return
    compared.to_csv(output_file(out, 'noise_vs_exact_ideal_data.csv'), index=False)
    backends = presentation_backends(data)
    envs = [(stage, backend) for backend in backends for stage in ('noisy', 'real')]
    raw = evaluation_rows(data['results'])
    raw = raw[raw.stage.isin(['noisy', 'real'])]
    present_methods = set(raw.method.dropna())
    methods = ([method for method in METHODS if method in present_methods] +
               sorted(present_methods - set(METHODS)))
    rows = [(stage, backend, method) for method in methods for stage, backend in envs]
    labels = [f"{LABELS.get(method, method).replace(' QAOA', '')} · "
              f"{backend.replace('ibm_', '').title()} · {'noise' if stage == 'noisy' else 'device'}"
              for stage, backend, method in rows]
    metrics = [
        ('full_optimum_probability', 'Full-optimum frequency', 'Change in optimum frequency', True),
        ('mean_full_cost', 'Mean full pairwise cost', 'Change in mean full cost', False),
        ('mean_mec', 'Mean full-read MEC', 'Change in mean MEC', False),
    ]
    for metric, title, delta_label, is_probability in metrics:
        delta = f'delta_{metric}'
        if delta not in compared:
            continue
        depths = sorted(raw.depth.dropna().unique())
        matrix = np.full((len(rows), len(depths)), np.nan)
        for i, (stage, backend, method) in enumerate(rows):
            part = compared[(compared.stage == stage) & (compared.backend == backend) &
                            (compared.method == method)]
            means = part.groupby('depth')[delta].mean()
            for j, p in enumerate(depths):
                if p in means.index:
                    matrix[i, j] = means[p] * (100 if is_probability else 1)
        finite = matrix[np.isfinite(matrix)]
        if not len(finite):
            continue
        limit = max(float(np.abs(finite).max()), 1e-12)
        fig, ax = plt.subplots(figsize=(8.8, max(4.5, .39 * len(rows) + 2.5)))
        cmap = (DIVERGING.reversed() if is_probability else DIVERGING).with_extremes(bad=PANEL)
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-limit, vmax=limit,
                          aspect='auto', interpolation='nearest')
        for i in range(len(rows)):
            for j in range(len(depths)):
                value = matrix[i, j]
                text = (f'{value:+.1f}' if is_probability else f'{value:+.3f}') if np.isfinite(value) else '—'
                color = INK
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.set_xticks(range(len(depths)), [str(int(p)) for p in depths])
        ax.set_xlabel('QAOA depth $p$')
        ax.set_title(title + ': change from exact ideal', fontsize=14, fontweight='bold', pad=14)
        ax.grid(False)
        ax.tick_params(length=0)
        colorbar_label = delta_label + (' (percentage points)' if is_probability else '')
        fig.colorbar(image, ax=ax, label=colorbar_label, fraction=.045, pad=.025)
        fig.tight_layout(rect=(0, .04, 1, 1))
        save(fig, out, 'noise_delta_' + metric, formats, dpi)

        selected = compared[compared.depth == depth]
        if selected.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.4, max(4.5, .39 * len(rows) + 2.5)))
        for i, (stage, backend, method) in enumerate(rows):
            part = selected[(selected.stage == stage) & (selected.backend == backend) &
                            (selected.method == method)].dropna(subset=[metric, f'ideal_{metric}'])
            if part.empty:
                ax.text(.02, i, '—', transform=ax.get_yaxis_transform(), color='#888888', va='center')
                continue
            ideal = float(part[f'ideal_{metric}'].iloc[0])
            mean = float(part[metric].mean())
            style = device_style(backend)
            ax.plot([ideal, mean], [i, i], color=style['color'], linestyle=style['linestyle'],
                    lw=1.5, zorder=1)
            ax.scatter([ideal], [i], marker='o', s=46, facecolors='white',
                       edgecolors=CYAN, lw=1.3, zorder=3, clip_on=False)
            ax.scatter([mean], [i], marker='s' if stage == 'noisy' else 'D', s=42,
                       color=style['color'], zorder=3)
            offsets = np.linspace(-.12, .12, len(part)) if len(part) > 1 else [0]
            ax.scatter(part[metric], i + np.asarray(offsets), color=style['color'],
                       alpha=.38, s=14, zorder=2)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.set_title(f'{title}: ideal versus sampled · $p={depth}$', fontsize=14, fontweight='bold', pad=14)
        if is_probability:
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        handles = [Line2D([0], [0], marker='o', color=CYAN, markerfacecolor='white', lw=0, label='Exact ideal'),
                   Line2D([0], [0], marker='s', color='#14243A', lw=0, label='Noise mean'),
                   Line2D([0], [0], marker='D', color='#14243A', lw=0, label='Device mean')]
        ax.legend(handles=handles, loc='best', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        save(fig, out, f'noise_vs_ideal_{metric}_p{depth}', formats, dpi)


def plot_resources(data, out, formats, dpi):
    rows = []
    for key, info in data.get('compiled_stats', {}).items():
        parts = key.split('|')
        if len(parts) != 3:
            continue
        backend, method, depth = parts
        rows.append({'backend': backend, 'method': method, 'depth': int(depth),
                     'total_depth': info.get('depth'), 'two_qubit_depth': info.get('two_qubit_depth'),
                     'two_qubit_gates': info.get('two_qubit_gates')})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(output_file(out, 'circuit_resources_data.csv'), index=False)
    config = data.get('manifest', {}).get('config', {})
    backends = sorted(df.backend.unique())
    methods = [m for m in METHODS if m in set(df.method)]
    methods += sorted(set(df.method) - set(METHODS))
    depths = sorted(set(df.depth.unique()) | set(config.get('depths', [])))
    missing_backends = sorted(set(config.get('backends', [])) - set(backends))
    missing_methods = sorted(set(config.get('methods', [])) - set(methods))
    for metric, title in [('total_depth', 'Total compiled depth'),
                          ('two_qubit_depth', 'Two-qubit depth'),
                          ('two_qubit_gates', 'Two-qubit gate count')]:
        if df[metric].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        ax.set_axisbelow(True)
        ax.grid(axis='x', visible=False)
        # If every recorded circuit has the same value at each depth, a single
        # trace states this clearly instead of stacking six indistinguishable lines.
        expected = len(backends) * len(methods)
        common = all(len(group) == expected and group[metric].notna().all() and
                     group[metric].nunique() == 1
                     for group in (df[df.depth == depth] for depth in depths))
        if common:
            series = df.groupby('depth')[metric].first().reindex(depths)
            ax.plot(series.index, series, color='#14243A', marker='o', lw=2, ms=6,
                    label='All recorded methods and devices')
            ax.legend(loc='upper left', fontsize=9)
        else:
            combinations = [(method, backend) for method in methods for backend in backends]
            offsets = np.linspace(-.075, .075, len(combinations)) if len(combinations) > 1 else [0]
            for (method, backend), offset in zip(combinations, offsets):
                sub = df[(df.backend == backend) & (df.method == method)]
                if sub.empty:
                    continue
                # Reindex before plotting: never connect over a missing depth.
                series = sub.set_index('depth')[metric].reindex(depths)
                style = device_style(backend)
                color = COLORS.get(method, '#888888')
                ax.plot(np.asarray(depths) + offset, series, marker=MARKERS.get(method, '^'),
                        color=color, linestyle=style['linestyle'], lw=2, ms=6,
                        markerfacecolor=color if backend == 'ibm_aachen' else 'white',
                        markeredgewidth=1.3, label=f'{method} | {backend}')
            method_handles = [Line2D([0], [0], color=COLORS.get(m, '#888888'),
                                     marker=MARKERS.get(m, '^'), lw=2, label=LABELS.get(m, m))
                              for m in methods]
            device_handles = [Line2D([0], [0], color='#14243A',
                                     linestyle=device_style(b)['linestyle'], lw=2,
                                     label=device_style(b)['label']) for b in backends]
            fig.legend(handles=method_handles + device_handles, loc='upper center',
                       bbox_to_anchor=(.5, .90), ncol=min(5, len(method_handles) + len(device_handles)),
                       fontsize=9)
        if missing_backends or missing_methods:
            missing = [device_style(b)['label'] for b in missing_backends] + missing_methods
            ax.text(.98, .03, 'No compiled data: ' + ', '.join(missing),
                    transform=ax.transAxes, ha='right', fontsize=8, color='#14243A')
        maximum = df[metric].max()
        if pd.notna(maximum):
            ax.set_ylim(0, max(1, maximum * 1.08))
        ax.set_xticks(depths)
        ax.set_xlabel('QAOA depth $p$' + (' (markers offset for visibility)' if not common else ''))
        ax.set_ylabel(title)
        fig.suptitle(title + ' by method and target', fontsize=15, fontweight='bold', y=.98)
        fig.tight_layout(rect=(0, .04, 1, .83 if not common else .92))
        save(fig, out, 'resource_' + metric, formats, dpi)


def plot_training(data,out,formats,dpi):
    training=data.get('training',{}); depths=sorted({int(k.rsplit('_p',1)[1]) for k in training if '_p' in k})
    if not depths: return
    fig,ax=plt.subplots(figsize=(9,5)); rows=[]
    styles=dict(zip(depths,itertools.cycle(['-','--',':','-.'])))
    for depth in depths:
        for method in METHODS:
            info=training.get(f'{method}_p{depth}',{}); starts=info.get('starts',[])
            best=next((s for s in starts if s.get('start') == info.get('best_start')),None)
            if not best or not best.get('trace'): continue
            trace=pd.DataFrame(best['trace']).sort_values('evaluation')
            trace['best_so_far']=trace.native_cost.cummin(); trace['method']=method; trace['depth']=depth
            trace['start']=best['start']; rows.append(trace[['method','depth','start','evaluation','native_cost','best_so_far']])
            ax.plot(trace.evaluation,trace.best_so_far,color=COLORS[method],linestyle=styles[depth],lw=1.8)
    if not rows: plt.close(fig); return
    ax.set(xlabel='Objective evaluations in selected start',ylabel='Best native expected cost so far')
    fig.suptitle('Training convergence · unequal budgets',fontsize=15,fontweight='bold',y=.99)
    handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=LABELS[m]) for m in METHODS]
    handles += [Line2D([0],[0],color='#14243A',linestyle=styles[p],lw=2,label=f'p={p}') for p in depths]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.79)); pd.concat(rows).to_csv(output_file(out, 'training_convergence_data.csv'),index=False)
    save(fig,out,'training_convergence',formats,dpi)


def paired_techniques(frame):
    keys=['stage','backend','method','depth','repeat']
    shot_column=['total_shots'] if 'total_shots' in frame else []
    raw=frame[frame.technique=='raw_reference'][keys+['mean_full_cost']+shot_column].rename(
        columns={'mean_full_cost':'raw_cost','total_shots':'raw_total_shots'})
    arms=frame[frame.technique.isin(TECHNIQUES)].copy()
    paired=arms.merge(raw,on=keys,how='inner',validate='many_to_one')
    if shot_column:
        paired=paired[paired.total_shots.eq(paired.raw_total_shots)].copy()
    return paired.assign(delta=lambda x:x.mean_full_cost-x.raw_cost)


def plot_techniques(data,out,formats,dpi):
    tech=data.get('techniques')
    if tech is None or tech.empty: return
    pairs=paired_techniques(tech).dropna(subset=['delta'])
    if pairs.empty: return
    # The ideal vector and the full-cost table share SNP-first state ordering.
    frequency=data.get('frequencies')
    exact=data.get('ideal_probabilities',{})
    pairs['ideal_cost']=np.nan
    if frequency is not None and 'full_cost' in frequency:
        state_cost=frequency.groupby('state_snp_first').full_cost.first().sort_index()
        for (method,depth),group in pairs.groupby(['method','depth']):
            probabilities=exact.get(f'{method}|{depth}')
            if probabilities is not None and len(probabilities)==len(state_cost):
                pairs.loc[group.index,'ideal_cost']=np.dot(probabilities,state_cost.to_numpy())
    pairs['delta_absolute_error']=(pairs.mean_full_cost-pairs.ideal_cost).abs()-(pairs.raw_cost-pairs.ideal_cost).abs()
    pairs.to_csv(output_file(out, 'technique_paired_data.csv'),index=False)
    pairs=pairs[pairs.stage.isin(['noisy','real'])]
    if pairs.empty: return
    envs=environments(pairs)
    names=[x for x in TECHNIQUES if x in set(pairs.technique)]
    for metric,filename,xlabel,title in [
        ('delta','technique_paired_cost','Cost difference from raw reference','Technique cost versus raw'),
        ('delta_absolute_error','technique_error_from_ideal','Change in absolute error from ideal vs raw','Technique error versus exact ideal')]:
        comparable=pairs.dropna(subset=[metric])
        if comparable.empty: continue
        fig,ax=plt.subplots(figsize=(9,5.6));ax.axvline(0,color='#888888',lw=1,ls='--')
        offsets=np.linspace(-.29,.29,len(envs)) if len(envs)>1 else [0]
        handles=[]
        for (stage,backend),offset in zip(envs,offsets):
            style=device_style(backend);color=style['color'];marker='o' if stage=='noisy' else 'D'
            handles.append(Line2D([0],[0],color=color,marker=marker,lw=0,label=env_title(stage,backend)))
            sub=select_env(comparable,stage,backend)
            for i,arm in enumerate(names):
                values=sub.loc[sub.technique==arm,metric].to_numpy()
                if not len(values): continue
                y=i+offset;jitter=np.linspace(-.035,.035,len(values)) if len(values)>1 else [0]
                ax.scatter(values,y+np.asarray(jitter),color=color,marker=marker,alpha=.28,s=20)
                ax.scatter([values.mean()],[y],color=color,marker=marker,s=48,zorder=3)
        ax.set_yticks(range(len(names)),[TECHNIQUES[n] for n in names]);ax.invert_yaxis()
        ax.set_xlabel(xlabel);ax.grid(axis='y',visible=False)
        fig.suptitle(title,fontsize=14,fontweight='bold',y=.99)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.8));save(fig,out,filename,formats,dpi)


def complete_repeat_categories(frame, category):
    """Missing outcomes in a saved repeat mean zero, not missing observations."""
    keys = ['stage', 'backend', 'method', 'repeat']
    groups = frame[keys].drop_duplicates()
    categories = pd.DataFrame({category: sorted(frame[category].unique())})
    return groups.merge(categories, how='cross').merge(frame, on=keys+[category], how='left').fillna({'probability': 0})


def distribution_rows(frame, backend, category, categories, value, include_ideal):
    """Build one matrix with a separate row for each method and execution mode."""
    matrix, labels = [], []
    stages = [('ideal','ideal'),('noisy',backend),('real',backend)] if include_ideal else [('noisy',backend),('real',backend)]
    for method in METHODS:
        for stage, name in stages:
            part=frame[(frame.method==method)&(frame.stage==stage)&(frame.backend==name)]
            matrix.append(part.set_index(category)[value].reindex(categories,fill_value=0).to_numpy()
                          if not part.empty else np.full(len(categories),np.nan))
            labels.append(f"{LABELS[method].replace(' QAOA','')} · " +
                          {'ideal':'ideal sampled','noisy':'noisy','real':'hardware'}[stage])
    return np.asarray(matrix),labels


def plot_distribution_changes(data, out, depth, formats, dpi):
    freq = data.get('frequencies')
    exact = data.get('ideal_probabilities', {})
    reads = data.get('problem', {}).get('reads')
    if freq is None or not exact or not reads:
        return
    n = len(reads[0])
    states = [''.join(map(str, bits)) for bits in itertools.product((0, 1), repeat=n)]
    noisy = freq[(freq.depth == depth) & freq.stage.isin(['noisy', 'real'])].copy()
    if noisy.empty:
        return
    rows = []
    for (stage, backend, method), group in noisy.groupby(['stage', 'backend', 'method']):
        probabilities = exact.get(f'{method}|{depth}')
        if probabilities is None:
            continue
        if len(probabilities) != len(states):
            raise ValueError(f'Ideal distribution length does not match state count for {method}|{depth}')
        repeats = sorted(group.repeat.unique())
        matrix = group.pivot_table(index='repeat', columns='state_snp_first', values='probability', aggfunc='sum', fill_value=0)
        matrix = matrix.reindex(index=repeats, columns=states, fill_value=0)
        observed = matrix.mean(axis=0).to_numpy()
        for state, ideal, sampled in zip(states, probabilities, observed):
            rows.append({'stage': stage, 'backend': backend, 'method': method, 'depth': depth,
                         'state_snp_first': state, 'ideal_probability': ideal,
                         'sampled_probability': sampled, 'delta_probability': sampled - ideal,
                         'repeats': len(repeats)})
    if not rows:
        return
    changes = pd.DataFrame(rows)
    changes.to_csv(output_file(out, f'noise_distribution_changes_p{depth}_data.csv'), index=False)
    if len(states) > 32:
        selected = changes.groupby('state_snp_first')[['ideal_probability','sampled_probability']].max().max(axis=1).nlargest(31).index
        changes['display_state'] = changes.state_snp_first.where(changes.state_snp_first.isin(selected), 'Other')
    else:
        changes['display_state'] = changes.state_snp_first
    display = (changes.groupby(['stage','backend','method','display_state'], as_index=False)
               [['ideal_probability','sampled_probability','delta_probability']].sum())
    labels = sorted(display.display_state.unique(), key=lambda state: (state == 'Other', state))
    limit = max(abs(display.delta_probability.min()), abs(display.delta_probability.max()), 1e-12)
    for backend in sorted(display.backend.unique()):
        matrix,rows=distribution_rows(display,backend,'display_state',labels,'delta_probability',False)
        fig,ax=plt.subplots(figsize=(max(11,len(labels)*.36),5.2))
        cmap=DIVERGING.with_extremes(bad=PANEL)
        image=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap=cmap,vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_yticks(range(len(rows)),rows,fontsize=9)
        ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=7)
        ax.set_xlabel('Bitstring in SNP-first order (Other combines remaining outcomes)')
        ax.grid(False);ax.tick_params(axis='y',length=0)
        for i in range(1,len(METHODS)): ax.axhline(2*i-.5,color='white',lw=2)
        for i,row in enumerate(matrix):
            if np.isnan(row).all(): ax.text((len(labels)-1)/2,i,'Not recorded',ha='center',va='center',fontsize=8)
        fig.suptitle(f'Sampled frequency minus exact ideal · {backend.replace("ibm_", "").title()} · $p={depth}$', fontsize=14, fontweight='bold')
        fig.colorbar(image,ax=ax,label='Change (percentage points)',format=FuncFormatter(lambda value,_:f'{100*value:.1f}'))
        fig.tight_layout(rect=(0,.04,1,.95))
        save(fig, out, f'noise_distribution_changes_{backend}_p{depth}', formats, dpi)


def plot_distributions(data,out,depth,formats,dpi):
    freq=data.get('frequencies')
    if freq is None or freq.empty: return
    freq=freq[(freq.depth==depth)&freq.stage.isin(['ideal','noisy','real'])].copy()
    if freq.empty: return
    # Keep labels as strings: SNP-first ordering is deliberately not a number.
    states=sorted(freq.state_snp_first.dropna().unique()); max_states=32
    if len(states)>max_states:
        ranked=freq.groupby('state_snp_first').probability.mean().nlargest(max_states-1).index
        freq['display_state']=freq.state_snp_first.where(freq.state_snp_first.isin(ranked),'Other')
    else: freq['display_state']=freq.state_snp_first
    # Sum outcomes within each repeat before averaging; missing states represent zero frequency.
    per_repeat=freq.groupby(['stage','backend','method','repeat','display_state'],as_index=False).probability.sum()
    per_repeat=complete_repeat_categories(per_repeat,'display_state')
    means=per_repeat.groupby(['stage','backend','method','display_state'],as_index=False).probability.mean()
    means.to_csv(output_file(out, f'outcome_frequencies_p{depth}_data.csv'),index=False)
    display_states=sorted(means.display_state.unique(),key=lambda s:(s=='Other',s))
    backends=sorted(freq.loc[freq.stage.isin(['noisy','real']),'backend'].unique())
    for backend in backends:
        vmax=max(means[(means.backend.isin(['ideal',backend]))].probability.max(),.01)
        matrix,labels=distribution_rows(means,backend,'display_state',display_states,'probability',True)
        fig,ax=plt.subplots(figsize=(max(11,len(display_states)*.36),5.7))
        cmap=SEQUENTIAL.with_extremes(bad=PANEL)
        image=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap=cmap,norm=PowerNorm(gamma=.5,vmin=0,vmax=vmax),interpolation='nearest')
        ax.set_yticks(range(len(labels)),labels,fontsize=9)
        ax.set_xticks(range(len(display_states)),display_states,rotation=90,fontsize=7)
        ax.set_xlabel('Measured bitstring (SNP-first order; Other combines remaining outcomes)')
        ax.grid(False);ax.tick_params(axis='y',length=0)
        for i in range(1,len(METHODS)): ax.axhline(3*i-.5,color='white',lw=2)
        for i,row in enumerate(matrix):
            if np.isnan(row).all(): ax.text((len(display_states)-1)/2,i,'Not recorded',ha='center',va='center',fontsize=8)
        fig.suptitle(f'Final outcome frequencies · {backend.replace("ibm_", "").title()} · $p={depth}$',fontweight='bold',fontsize=15)
        fig.colorbar(image,ax=ax,label='Fraction of shots',format=PercentFormatter(1))
        fig.tight_layout(rect=(0,.04,1,.95))
        save(fig,out,f'outcomes_{backend}_p{depth}',formats,dpi)
        # Cost distributions are aggregated per repeat, then averaged, not pooled over depths.
        costs=freq[(freq.backend.isin(['ideal',backend]))].groupby(['stage','backend','method','repeat','full_cost'],as_index=False).probability.sum()
        costs=complete_repeat_categories(costs,'full_cost')
        costs=costs.groupby(['stage','backend','method','full_cost'],as_index=False).probability.mean()
        costs.to_csv(output_file(out, f'cost_distribution_{backend}_p{depth}_data.csv'),index=False)
        cost_values=sorted(costs.full_cost.unique())
        matrix,labels=distribution_rows(costs,backend,'full_cost',cost_values,'probability',True)
        fig,ax=plt.subplots(figsize=(10,5.7));cmap=SEQUENTIAL.with_extremes(bad=PANEL)
        image=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap=cmap,
                        norm=PowerNorm(gamma=.5,vmin=0,vmax=max(float(costs.probability.max()),.01)),interpolation='nearest')
        ax.set_yticks(range(len(labels)),labels,fontsize=9)
        ax.set_xticks(range(len(cost_values)),[f'{value:g}' for value in cost_values]);ax.set_xlabel('Full pairwise cost')
        ax.grid(False);ax.tick_params(axis='y',length=0)
        for i in range(1,len(METHODS)): ax.axhline(3*i-.5,color='white',lw=2)
        for i,row in enumerate(matrix):
            if np.isnan(row).all(): ax.text((len(cost_values)-1)/2,i,'Not recorded',ha='center',va='center',fontsize=8)
        fig.colorbar(image,ax=ax,label='Fraction of shots',format=PercentFormatter(1))
        fig.suptitle(f'Cost distribution · {backend.replace("ibm_", "").title()} · $p={depth}$',fontsize=15,fontweight='bold')
        fig.tight_layout(rect=(0,.04,1,.95)); save(fig,out,f'cost_distribution_{backend}_p{depth}',formats,dpi)


def plot_problem(data, out, formats, dpi):
    problem = data.get('problem', {})
    if not problem.get('reads'):
        return
    reads = np.asarray(problem['reads'])
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    im = ax.imshow(reads, cmap=READS_CMAP, vmin=-1, vmax=1, aspect='auto')
    ax.set(xlabel='SNP', ylabel='Read', title='Shared input reads',
           xticks=np.arange(reads.shape[1]), xticklabels=np.arange(1, reads.shape[1]+1))
    ax.grid(False)
    fig.colorbar(im, ax=ax, label='−1 missing · 0 / 1 observed')
    fig.tight_layout(); save(fig, out, 'input_reads', formats, dpi)
    if not all(key in problem for key in ('same', 'opposite', 'native_j')):
        return
    same, opposite, native = (np.asarray(problem[key]) for key in ('same', 'opposite', 'native_j'))
    n = len(native)
    theta = np.linspace(np.pi/2, np.pi/2+2*np.pi, n, endpoint=False)
    xy = np.column_stack((np.cos(theta), np.sin(theta)))
    fig, ax = plt.subplots(figsize=(5, 5))
    edges = []
    for i, j in itertools.combinations(range(n), 2):
        if same[i, j] + opposite[i, j] == 0:
            continue
        retained = bool(native[i, j])
        edges.append({'snp_1': i+1, 'snp_2': j+1, 'retained': retained,
                      'same_evidence': same[i, j], 'opposite_evidence': opposite[i, j]})
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=CYAN if retained else GREY,
                ls='-' if retained else '--', lw=2 if retained else 1.2, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=650, color=PALE_CYAN, edgecolor=CYAN, zorder=2)
    for i, (x, y) in enumerate(xy):
        ax.text(x, y, str(i+1), ha='center', va='center', zorder=3)
    ax.legend(handles=[Line2D([], [], color=CYAN, label='Retained'),
                       Line2D([], [], color=GREY, ls='--', label='Removed')],
              loc='upper center', bbox_to_anchor=(.5, -.02), ncol=2)
    ax.set_title('Pairwise evidence graph'); ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout(); save(fig, out, 'input_graph', formats, dpi)
    pd.DataFrame(edges).to_csv(output_file(out, 'input_graph_data.csv'), index=False)


def plot_zne(data, out, formats, dpi):
    tech = data.get('techniques')
    if tech is None or tech.empty or not {'factor_1_cost', 'factor_3_cost'}.issubset(tech):
        return
    zne = tech[(tech.technique == 'zne') & tech.stage.isin(['noisy', 'real'])].copy()
    zne = zne.dropna(subset=['factor_1_cost', 'factor_3_cost', 'mean_full_cost'], how='all')
    if zne.empty:
        return
    references = exact_reference_metrics(data)
    zne.to_csv(output_file(out, 'zne_factors_data.csv'), index=False)
    keys = ['stage', 'backend', 'method', 'depth']
    grouped = (zne.groupby(keys, as_index=False)
               .agg(factor_0=('mean_full_cost', 'mean'), n_0=('mean_full_cost', 'count'),
                    factor_1=('factor_1_cost', 'mean'), n_1=('factor_1_cost', 'count'),
                    factor_3=('factor_3_cost', 'mean'), n_3=('factor_3_cost', 'count')))
    if references.empty:
        grouped['ideal_mean_full_cost'] = np.nan
    else:
        grouped = grouped.merge(references[['method', 'depth', 'ideal_mean_full_cost']],
                                on=['method', 'depth'], how='left', validate='many_to_one')
    grouped = grouped.sort_values(['method', 'depth', 'backend', 'stage'])
    values = grouped[['ideal_mean_full_cost', 'factor_0', 'factor_1', 'factor_3']].to_numpy(dtype=float)
    differences = values - values[:, :1]
    finite = differences[np.isfinite(differences)]
    limit = max(float(np.abs(finite).max()), 1e-12) if len(finite) else 1.
    fig, ax = plt.subplots(figsize=(9.4, max(4.2, .57 * len(grouped) + 2.3)))
    cmap = DIVERGING.with_extremes(bad=PANEL)
    image = ax.imshow(np.ma.masked_invalid(differences), cmap=cmap, vmin=-limit, vmax=limit,
                      aspect='auto', interpolation='nearest')
    labels = []
    for i, row in enumerate(grouped.itertuples()):
        labels.append(f"{LABELS.get(row.method, row.method).replace(' QAOA', '')} · p={row.depth} · "
                      f"{row.backend.replace('ibm_', '').title()} · {'noise' if row.stage == 'noisy' else 'device'}")
        counts = [None, row.n_0, row.n_1, row.n_3]
        for j, value in enumerate(values[i]):
            text = f'{value:.3f}' if np.isfinite(value) else '—'
            if j and np.isfinite(value):
                text += f'\n(n={int(counts[j])})'
            difference = differences[i, j]
            color = INK
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xticks(range(4), ['Ideal\nexact', 'Factor 0\nextrapolated',
                             'Factor 1\nsampled', 'Factor 3\nsampled'])
    ax.set_title('ZNE: cost at each noise factor', fontsize=14, fontweight='bold', pad=14)
    ax.grid(False)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, label='Cost minus exact ideal', fraction=.045, pad=.025)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, out, 'zne_factors', formats, dpi)


def batch_runtime_data(records, job_metrics):
    """Keep simulator elapsed time and hardware QPU usage as distinct measures."""
    columns = ('stage', 'backend', 'repeat', 'batch_wall_seconds', 'batch_circuits', 'job_id')
    timing = pd.DataFrame([{key: record.get(key) for key in columns} for record in records]).drop_duplicates()
    timing = timing[timing.stage.isin(['ideal', 'noisy', 'real'])].copy()
    timing['runtime_seconds'] = timing.apply(
        lambda row: (job_metrics.get(row.job_id, {}).get('quantum_seconds')
                     if row.stage == 'real' else row.batch_wall_seconds), axis=1)
    timing['time_kind'] = timing.stage.map({'ideal': 'simulator elapsed', 'noisy': 'simulator elapsed',
                                             'real': 'QPU usage'})
    return timing


def plot_run_diagnostics(data, out, formats, dpi):
    """Export and plot preparation times and resources from saved records."""
    rows = []
    for key, info in data.get('training', {}).items():
        if '_p' in key and info.get('wall_seconds') is not None:
            method, depth = key.rsplit('_p', 1)
            rows.append({'method': method, 'depth': int(depth), 'seconds': info['wall_seconds']})
    if rows:
        times = pd.DataFrame(rows); times.to_csv(output_file(out, 'training_time_data.csv'), index=False)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for method in METHODS:
            part = times[times.method == method].sort_values('depth')
            if not part.empty:
                ax.plot(part.depth, part.seconds, color=COLORS[method], marker=MARKERS[method], label=LABELS[method])
        ax.set(xlabel='QAOA depth p', ylabel='Wall seconds, all starts', title='Training time'); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.legend(fontsize=8)
        fig.tight_layout(); save(fig, out, 'training_time', formats, dpi)
    rows = []
    for key, info in data.get('compiled_stats', {}).items():
        if info.get('compile_wall_seconds') is not None:
            backend, method, depth = key.split('|')
            rows.append({'backend': backend, 'method': method, 'depth': int(depth), 'seconds': info['compile_wall_seconds']})
    if rows:
        times = pd.DataFrame(rows); times.to_csv(output_file(out, 'compilation_time_data.csv'), index=False)
        backends = sorted(times.backend.unique()); fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for backend in backends:
            style = device_style(backend)
            for method in METHODS:
                part = times[(times.backend == backend) & (times.method == method)].sort_values('depth')
                if not part.empty:
                    ax.plot(part.depth, part.seconds, color=COLORS[method], marker=MARKERS[method],
                            linestyle=style['linestyle'])
        ax.set(xlabel='QAOA depth p', ylabel='Compilation wall seconds', title='Compilation time')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], label=LABELS[m]) for m in METHODS]
        handles += [Line2D([0], [0], color='#14243A', linestyle=device_style(b)['linestyle'],
                           label=device_style(b)['label']) for b in backends]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .96), ncol=3, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, .82)); save(fig, out, 'compilation_time', formats, dpi)
    records = data.get('records', [])
    if not records:
        return
    metrics_path = Path(out) / 'job_metrics.json'
    job_metrics = read_json(metrics_path) if metrics_path.exists() else {}
    timing = batch_runtime_data(records, job_metrics)
    timing.to_csv(output_file(out, 'batch_times.csv'), index=False)
    if not timing.empty:
        envs = environments(timing); fig, ax = plt.subplots(figsize=(8, max(3.4, .55*len(envs)+1.8)))
        for i, (stage, backend) in enumerate(envs):
            part = select_env(timing, stage, backend)
            available = part.dropna(subset=['runtime_seconds'])
            offsets=np.linspace(-.12,.12,len(available)) if len(available)>1 else [0]
            color=CYAN if stage=='ideal' else device_style(backend)['color']
            if not available.empty:
                ax.scatter(available.runtime_seconds,i+np.asarray(offsets),color=color,alpha=.65,s=30)
            if len(available) < len(part):
                ax.text(.98, i, 'QPU usage not recorded', transform=ax.get_yaxis_transform(),
                        ha='right', va='center', fontsize=8, color=GREY)
        ax.set_yticks(range(len(envs)),[env_title(stage,backend) for stage,backend in envs]);ax.invert_yaxis()
        ax.set_xlabel('Seconds · simulator elapsed / hardware QPU usage');ax.grid(axis='y',visible=False)
        values=timing.runtime_seconds.dropna()
        if len(values) and (values > 0).all() and values.max()/values.min()>=100:
            ax.set_xscale('log');ax.set_xlabel('Seconds · simulator elapsed / hardware QPU usage (log scale)')
        fig.suptitle('Execution time by batch · excludes hardware queue', fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, .91)); save(fig, out, 'batch_time', formats, dpi)
    resource_rows = []
    for record in records:
        if record.get('stage') not in ('noisy', 'real') or not record.get('circuit_resources'):
            continue
        technique, variant = record.get('technique'), record.get('variant')
        if ((technique in ('raw_reference', 'readout', 'dd') and variant != 'target') or
                (technique == 'zne' and variant != 'factor_3') or
                (technique == 'trex' and not str(variant).startswith('target_mask_')) or
                technique not in ('raw_reference', 'readout', 'dd', 'twirling', 'trex', 'zne')):
            continue
        resource_rows.append({key: record.get(key) for key in ('stage', 'backend', 'technique', 'variant', 'repeat')} | record['circuit_resources'])
    if resource_rows:
        resources = pd.DataFrame(resource_rows); resources.to_csv(output_file(out, 'technique_resources.csv'), index=False)
        envs = environments(resources)
        names=[name for name in ('raw_reference',*TECHNIQUES) if name in set(resources.technique)]
        for metric, label in [('depth', 'Total depth'), ('two_qubit_depth', 'Two-qubit depth'), ('two_qubit_gates', 'Two-qubit gates')]:
            fig, ax = plt.subplots(figsize=(9,5.5))
            offsets=np.linspace(-.28,.28,len(envs)) if len(envs)>1 else [0]
            handles=[]
            for (stage, backend), offset in zip(envs, offsets):
                part = select_env(resources, stage, backend)
                values = part.groupby('technique')[metric].median()
                color=device_style(backend)['color'];marker='o' if stage=='noisy' else 'D'
                handles.append(Line2D([0],[0],color=color,marker=marker,lw=0,label=env_title(stage,backend)))
                for i,name in enumerate(names):
                    if name in values and pd.notna(values[name]):
                        ax.scatter([values[name]],[i+offset],color=color,marker=marker,s=40,zorder=3)
            ax.set_yticks(range(len(names)),[TECHNIQUES.get(name,'Raw reference') for name in names]);ax.invert_yaxis()
            ax.set_xlabel(label);ax.grid(axis='y',visible=False)
            fig.suptitle('Median target-circuit resources · ZNE factor 3', fontweight='bold',y=.99)
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,fontsize=8)
            fig.tight_layout(rect=(0,.04,1,.8)); save(fig, out, f'technique_{metric}', formats, dpi)



ENV_STYLE = {
    'ideal': {'color': CYAN, 'marker': 'o', 'label': 'Ideal · exact'},
    'noisy': {'color': GOLD, 'marker': 's', 'label': 'Noisy simulation'},
    'real': {'color': TIE, 'marker': 'D', 'label': 'Real device'},
}


def presentation_backends(data):
    """Use configured backends, including a device whose measurements are missing."""
    names = set(data.get('manifest', {}).get('config', {}).get('backends', []))
    for key in ('results', 'techniques'):
        frame = data.get(key)
        if frame is not None and 'backend' in frame:
            names.update(frame.loc[frame.backend.ne('ideal'), 'backend'].dropna().unique())
    preferred = ['ibm_aachen', 'ibm_berlin']
    return [name for name in preferred if name in names] + sorted(names - set(preferred))


def presentation_multiangle_data(data):
    """Exact ideal plus recorded raw evaluation repeats, with no fabricated modes."""
    references = exact_reference_metrics(data)
    if references.empty:
        return pd.DataFrame()
    reference = references[references.method == 'multi-angle']
    rows = []
    for backend in presentation_backends(data):
        for row in reference.itertuples():
            rows.append({'stage': 'ideal', 'backend': backend, 'method': row.method,
                         'depth': row.depth, 'repeat': np.nan,
                         'full_optimum_probability': row.ideal_full_optimum_probability,
                         'mean_full_cost': row.ideal_mean_full_cost})
    frame = evaluation_rows(data['results'])
    frame = frame[frame.method.eq('multi-angle') & frame.stage.isin(['noisy', 'real'])]
    columns = ['stage', 'backend', 'method', 'depth', 'repeat', 'full_optimum_probability', 'mean_full_cost']
    return pd.concat([pd.DataFrame(rows, columns=columns), frame[columns]], ignore_index=True)


def presentation_technique_data(data):
    """Cost-error changes vs a raw reference, joined on the full experimental key."""
    frame = data.get('techniques')
    references = exact_reference_metrics(data)
    if frame is None or frame.empty or references.empty or 'total_shots' not in frame:
        return pd.DataFrame()
    keys = ['stage', 'backend', 'method', 'depth', 'repeat']
    raw = frame[frame.technique == 'raw_reference'][keys + ['mean_full_cost', 'total_shots']].rename(
        columns={'mean_full_cost': 'raw_cost', 'total_shots': 'raw_total_shots'})
    arms = frame[frame.technique.isin(TECHNIQUES) & frame.stage.isin(['noisy', 'real'])]
    pairs = arms.merge(raw, on=keys, how='inner', validate='many_to_one')
    pairs = pairs.merge(references[['method', 'depth', 'ideal_mean_full_cost']],
                        on=['method', 'depth'], how='inner', validate='many_to_one')
    pairs = pairs.rename(columns={'ideal_mean_full_cost': 'ideal_cost'})
    pairs = pairs[(pairs.total_shots == pairs.raw_total_shots) &
                  pairs.mean_full_cost.notna() & pairs.raw_cost.notna()].copy()
    pairs['delta_abs_error'] = ((pairs.mean_full_cost-pairs.ideal_cost).abs() -
                                (pairs.raw_cost-pairs.ideal_cost).abs())
    return pairs


def _presentation_legend(stage_names):
    return [Line2D([0], [0], color=ENV_STYLE[s]['color'], marker=ENV_STYLE[s]['marker'],
                   lw=1.7, ms=5, label=ENV_STYLE[s]['label']) for s in stage_names]


def _presentation_axes(ax):
    ax.grid(False)
    ax.grid(axis='y', alpha=.14, linestyle='-', linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


DEVICE_STYLES = {
    'ibm_aachen': {'color': INK, 'linestyle': '-', 'marker': 'o', 'label': 'Aachen'},
    'ibm_berlin': {'color': TIE, 'linestyle': '--', 'marker': 's', 'label': 'Berlin'},
}


def device_style(backend):
    return DEVICE_STYLES.get(backend, {'color': GREY, 'linestyle': ':', 'marker': '^',
                                       'label': backend.replace('ibm_', '').title()})


def save_slide(fig, out, name, formats, dpi):
    """Write a true 16:9 canvas for PowerPoint, in raster and editable formats."""
    with plt.rc_context({'savefig.bbox': None}):
        for fmt in formats:
            fig.savefig(output_file(out, f'{name}.{fmt}'), dpi=dpi, bbox_inches=None)
    plt.close(fig)


def slide_axes():
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.12, right=.96, bottom=.14, top=.76)
    _presentation_axes(ax)
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(14); ax.yaxis.label.set_size(14)
    return fig, ax


def plot_multiangle_zoom(fig, main, table, zne, measured, depth, backends, out):
    """Show measured and estimated success against each equal-shot raw reference."""
    rows = [(stage, backend) for backend in backends for stage in ('noisy', 'real')]
    labels = [f"{device_style(backend)['label']}\n{'noise' if stage == 'noisy' else 'device'}"
              for stage, backend in rows]
    success = main.inset_axes([.58, .13, .39, .34])
    success.set_title(f'p={depth} · ○ raw  △ DD  ■ twirling  ★ ZNE estimate', fontsize=8, pad=5, loc='left', x=.04)
    success.spines['right'].set_visible(True)
    success.set_ylabel('Success frequency', fontsize=8)
    success.yaxis.set_major_formatter(PercentFormatter(1))
    success.tick_params(labelsize=7)
    success.grid(axis='x', visible=False)
    success.set_xticks(range(len(rows)), labels)
    success.set_xlim(-.5, len(rows)-.5)
    combined = []
    if not zne.empty:
        combined.append(zne.assign(technique='zne', mitigated_success=zne.zne_success_estimate,
                                   value_type='extrapolated estimate'))
    if not measured.empty:
        combined.append(measured.assign(mitigated_success=measured.optimum_probability,
                                        value_type='measured frequency'))
    points = pd.concat(combined, ignore_index=True)
    points.to_csv(output_file(out, '01_mitigation_success_data.csv'), index=False)
    for x, (stage, backend) in enumerate(rows):
        color = GOLD if stage == 'noisy' else TIE
        for technique, offset, marker in [('dd', -.22, '^'), ('twirling', 0, 's'), ('zne', .22, '*')]:
            part = points[(points.stage == stage) & (points.backend == backend) &
                          (points.technique == technique)]
            if part.empty:
                continue
            raw, value = part.raw_success.mean(), part.mitigated_success.mean()
            success.plot([x+offset, x+offset], [raw, value], color=color, linewidth=1.4, zorder=2)
            success.scatter([x+offset], [raw], marker='o', facecolors='white', edgecolors=color, s=24, zorder=3)
            success.scatter([x+offset], [value], marker=marker, color=color, edgecolors=INK,
                            linewidths=.4, s=48, zorder=3)
    selected = table[(table.depth == depth) & table.stage.isin(['noisy', 'real'])]
    if not selected.empty:
        values = selected.full_optimum_probability.dropna().to_numpy()
        if len(values):
            low, high = float(values.min())-.01, float(values.max())+.01
            main.add_patch(Rectangle((depth-.17, low), .34, high-low, fill=False,
                                     edgecolor=INK, linewidth=1, linestyle=':'))
            for x, target in [(depth-.17, 0), (depth+.17, 1)]:
                fig.add_artist(ConnectionPatch(xyA=(x, low), coordsA=main.transData,
                                                xyB=(target, 1), coordsB=success.transAxes,
                                                color=GREY, linewidth=.7, linestyle=':', zorder=6))


def plot_presentation_multiangle(data, out, formats, dpi):
    table=presentation_multiangle_data(data); backends=presentation_backends(data)
    if table.empty or not backends: return
    table.to_csv(output_file(out, '01_multiangle_data.csv'),index=False)
    fig,ax=slide_axes()
    champion = data.get('manifest', {}).get('champion') or {}
    zne = (_selected_champion_rows(data, presentation_zne_success_data(data))
           if champion.get('method') == 'multi-angle' else pd.DataFrame())
    measured = (measured_mitigation_data(data)
                if champion.get('method') == 'multi-angle' else pd.DataFrame())
    has_zoom = not zne.empty or not measured.empty
    reference=table[(table.stage=='ideal') & (table.backend==backends[0])].sort_values('depth')
    depths=sorted(reference.depth.unique())
    ax.plot(reference.depth,reference.full_optimum_probability,color=CYAN,marker='D',ms=7,lw=2,
            label='Ideal · exact',zorder=4)
    for stage,color,label in [('noisy',GOLD,'Noisy'),('real',TIE,'Hardware')]:
        for backend in backends:
            device=device_style(backend)
            values=table[(table.stage==stage)&(table.backend==backend)].dropna(subset=['full_optimum_probability'])
            if values.empty: continue
            means=values.groupby('depth').full_optimum_probability.mean().reindex(depths)
            for depth,group in values.groupby('depth'):
                jitter=np.linspace(-.035,.035,len(group)) if len(group)>1 else [0]
                ax.scatter(depth+np.asarray(jitter),group.full_optimum_probability,s=22,color=color,alpha=.24,zorder=2)
            ax.plot(means.index,means,color=color,linestyle=device['linestyle'],marker=device['marker'],
                    markerfacecolor=color if stage=='noisy' else 'white',markeredgewidth=1.5,
                    ms=7,lw=2,label=f"{label} · {device['label']}",zorder=3)
    observed=table.full_optimum_probability.dropna()
    lower=max(0.0, np.floor((float(observed.min())-.02)*10)/10) if not observed.empty else 0.0
    upper=max(1.02, float(observed.max())+.02) if not observed.empty else 1.02
    if lower < .8 < upper:
        # Expand the range where hardware and noisy curves are close, including 100%.
        ax.set_yscale('function', functions=(
            lambda value: np.where(value <= .8, value, .8 + 4 * (value - .8)),
            lambda value: np.where(value <= .8, value, .8 + (value - .8) / 4)))
        ticks = np.r_[lower, np.arange(np.floor(lower * 5 + 1) / 5, .8, .2),
                      [.8, .85, .9, .95, 1.0]]
        ax.set_yticks(np.unique(ticks[(ticks >= lower) & (ticks <= upper)]))
        ylabel = (f'Optimal frequency\n(above 80% expanded; from {lower:.0%})' if has_zoom else
                  f'Optimal-solution frequency\n(above 80% expanded; from {lower:.0%})')
    else:
        ylabel = f'Optimal-solution frequency (from {lower:.0%})'
    ax.set(xlabel='QAOA depth $p$',ylabel=ylabel,ylim=(lower,upper))
    ax.set_xticks(depths); ax.yaxis.set_major_formatter(PercentFormatter(1))
    if has_zoom:
        plot_multiangle_zoom(fig, ax, table, zne, measured, champion['depth'], backends, out)
    fig.suptitle('Multi-angle QAOA: ideal, noisy and real',fontsize=21,fontweight='bold',y=.97)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,fontsize=10)
    missing=[]
    for stage,label in [('noisy','Noisy'),('real','Hardware')]:
        absent=[device_style(b)['label'] for b in backends
                if not ((table.stage==stage)&(table.backend==b)).any()]
        if absent: missing.append(f"{label}: not recorded for {', '.join(absent)}")
    if missing:
        ax.text(.98,.05,'\n'.join(missing),transform=ax.transAxes,ha='right',fontsize=10,color='#888888')
    save_slide(fig,out,'01_multiangle_devices',formats,dpi)


def plot_presentation_resources(data, out, formats, dpi):
    rows=[]
    for key,info in data.get('compiled_stats',{}).items():
        parts=key.split('|')
        if len(parts)!=3: continue
        backend,method,depth=parts
        used=info.get('physical_qubits_used') if not info.get('operations',{}).get('delay') else None
        rows.append({'backend':backend,'method':method,'depth':int(depth),'two_qubit_gates':info.get('two_qubit_gates'),
                     'two_qubit_depth':info.get('two_qubit_depth'),'total_depth':info.get('depth'),'physical_qubits_used':used})
    if not rows: return
    resources=pd.DataFrame(rows);resources.to_csv(output_file(out, '02_resources_data.csv'),index=False)
    results=evaluation_rows(data['results']);results=results[results.stage.isin(['noisy','real'])]
    results.to_csv(output_file(out, '02_quality_data.csv'),index=False)
    stage='real' if (results.stage=='real').any() else 'noisy'
    selected=results[results.stage==stage]; backends=presentation_backends(data)
    if selected.empty: return
    fig,ax=slide_axes()
    depths=sorted(resources.depth.unique())
    for method in METHODS:
        for backend in backends:
            sub=selected[(selected.method==method)&(selected.backend==backend)]
            if sub.empty: continue
            style=device_style(backend);means=sub.groupby('depth').full_optimum_probability.mean().reindex(depths)
            for depth,group in sub.groupby('depth'):
                jitter=np.linspace(-.03,.03,len(group)) if len(group)>1 else [0]
                ax.scatter(depth+np.asarray(jitter),group.full_optimum_probability,s=20,color=COLORS[method],alpha=.20)
            ax.plot(means.index,means,color=COLORS[method],marker=MARKERS[method],linestyle=style['linestyle'],
                    markerfacecolor=COLORS[method] if backend=='ibm_aachen' else 'white',markeredgewidth=1.5,lw=2,ms=7)
    n=len(data.get('problem',{}).get('reads',[[0]])[0]); gates=[]; two_depths=[]
    for depth in depths:
        at_depth=resources.loc[resources.depth==depth]
        values=at_depth.two_qubit_gates.dropna().unique()
        if len(values)==1 and at_depth.two_qubit_gates.notna().all(): gates.append(str(int(values[0])))
        values=at_depth.two_qubit_depth.dropna().unique()
        if len(values)==1 and at_depth.two_qubit_depth.notna().all(): two_depths.append(str(int(values[0])))
    subtitle=f"{n} logical qubits"
    if len(gates)==len(depths):
        subtitle+=f" · two-qubit gates (p={'/'.join(map(str,depths))}): {' / '.join(gates)}"
    if len(two_depths)==len(depths):
        subtitle+=f" · two-qubit depth: {' / '.join(two_depths)}"
    ax.set(xlabel='QAOA depth $p$',ylabel='Optimal-solution frequency',ylim=(0,1.04));ax.set_xticks(depths)
    if depths:
        last=depths[-1]
        for method in METHODS:
            endpoint=selected[(selected.method==method)&(selected.depth==last)].groupby('backend').full_optimum_probability.mean().dropna()
            if endpoint.empty: continue
            low,high=100*endpoint.min(),100*endpoint.max()
            label=f'{low:.1f}–{high:.1f}%' if high-low >= .05 else f'{high:.1f}%'
            ax.annotate(label,(last,float(endpoint.mean())),xytext=(12,0),textcoords='offset points',
                        ha='left',va='center',fontsize=10,color=COLORS[method],fontweight='bold')
        ax.set_xlim(min(depths)-.15,last+.6)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    title='Hardware' if stage=='real' else 'Noisy simulation'
    fig.suptitle(f'{title}: compare QAOA methods',fontsize=21,fontweight='bold',y=.97)
    fig.text(.5,.905,subtitle,ha='center',fontsize=11,color='#14243A')
    handles=[Line2D([0],[0],color=COLORS[m],marker=MARKERS[m],lw=2,label=LABELS[m]) for m in METHODS]
    handles += [Line2D([0],[0],color='#14243A',linestyle=device_style(b)['linestyle'],lw=2,label=device_style(b)['label']) for b in backends]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.86),ncol=5,fontsize=10)
    absent=[device_style(b)['label'] for b in backends if not (selected.backend==b).any()]
    if absent:
        ax.text(.98,.05,f"{title} not recorded for {', '.join(absent)}",transform=ax.transAxes,
                ha='right',fontsize=10,color='#888888')
    save_slide(fig,out,'02_quality_and_resources',formats,dpi)


def presentation_relative_error_data(data):
    """Percentage change in absolute expected-cost error; never clip regressions."""
    table=presentation_technique_data(data)
    if table.empty: return table
    champion=data.get('manifest',{}).get('champion') or {}
    if champion.get('method') is not None and champion.get('depth') is not None:
        table=table[(table.method==champion['method'])&(table.depth==champion['depth'])].copy()
    baseline=(table.raw_cost-table.ideal_cost).abs()
    tolerance=1e-9*np.maximum(1,table.ideal_cost.abs())
    table['error_reduction_percent']=np.divide(-100*table.delta_abs_error,baseline,
        out=np.full(len(table),np.nan),where=baseline>tolerance)
    return table


def plot_presentation_techniques(data, out, formats, dpi):
    table=presentation_relative_error_data(data);backends=presentation_backends(data)
    if table.empty or not backends: return
    table.to_csv(output_file(out, '03_technique_effect_data.csv'),index=False)
    stage='real' if (table.stage=='real').any() else 'noisy';selected=table[table.stage==stage]
    present=set(selected.technique)|set(data.get('manifest',{}).get('config',{}).get('techniques',[]))
    names=[name for name in TECHNIQUES if name in present]
    if not names: return
    fig,ax=slide_axes();fig.subplots_adjust(left=.28);ax.grid(False);ax.axvline(0,color='#888888',ls=':',lw=1.2)
    offsets=np.linspace(-.18,.18,len(backends)) if len(backends)>1 else [0]
    for backend,offset in zip(backends,offsets):
        style=device_style(backend);sub=selected[selected.backend==backend]
        for i,name in enumerate(names):
            values=sub.loc[sub.technique==name,'error_reduction_percent'].dropna().to_numpy()
            if not len(values): continue
            center=i+offset;mean=float(values.mean());jitter=np.linspace(-.055,.055,len(values)) if len(values)>1 else [0]
            ax.plot([0,mean],[center,center],color=style['color'],linestyle=style['linestyle'],lw=2)
            ax.scatter(values,center+np.asarray(jitter),color=style['color'],alpha=.30,s=25,zorder=2)
            ax.scatter([mean],[center],color=style['color'],marker=style['marker'],s=70,zorder=3)
            ax.annotate(f'{mean:+.1f}%',(mean,center),xytext=(0,12),textcoords='offset points',
                        ha='center',va='bottom',color=INK,fontsize=9,fontweight='bold')
    ax.set_yticks(range(len(names)),[TECHNIQUES[name] for name in names]);ax.invert_yaxis()
    ax.set_xlabel('Reduction in cost error from ideal vs raw (%)');ax.tick_params(axis='y',labelsize=12)
    valid=selected.error_reduction_percent.dropna()
    if not valid.empty:
        lo=min(float(valid.min()),0);hi=max(float(valid.max()),0);pad=max((hi-lo)*.14,5)
        ax.set_xlim(lo-pad,hi+pad)
    label='Hardware' if stage=='real' else 'Noisy simulation'
    fig.suptitle(f'{label}: did mitigation help?',fontsize=21,fontweight='bold',y=.97)
    handles=[Line2D([0],[0],color=device_style(b)['color'],linestyle=device_style(b)['linestyle'],marker=device_style(b)['marker'],lw=2,label=device_style(b)['label']) for b in backends]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.89),ncol=len(handles),fontsize=12)
    absent=[device_style(b)['label'] for b in backends
            if selected.loc[selected.backend==b,'error_reduction_percent'].dropna().empty]
    if absent:
        ax.text(.98,.04,f"No comparable data: {', '.join(absent)}",transform=ax.transAxes,
                ha='right',fontsize=10,color='#888888')
    save_slide(fig,out,'03_technique_effect',formats,dpi)


def presentation_zne_success_data(data):
    """Extrapolate an optimum-projector expectation from saved ZNE counts."""
    optima=set(data.get('problem',{}).get('full_minimizers',[]))
    if not optima: return pd.DataFrame()
    grouped={}
    for record in data.get('records',[]):
        if record.get('stage') not in ('noisy','real') or record.get('technique') not in ('zne','raw_reference'): continue
        variant=record.get('variant') if record['technique']=='zne' else 'raw'
        if variant not in ('raw','factor_1','factor_3'): continue
        key=tuple(record.get(k) for k in ('stage','backend','method','depth','repeat'))
        counts=record.get('counts') or {};shots=sum(counts.values())
        if shots<=0 or shots!=record.get('shots'): continue
        successes=sum(count for keystr,count in counts.items() if keystr.replace(' ','')[::-1] in optima)
        grouped.setdefault(key,{})[variant]=(successes/shots,shots)
    rows=[]
    for key,items in grouped.items():
        if not {'raw','factor_1','factor_3'} <= items.keys(): continue
        if items['raw'][1] != items['factor_1'][1]+items['factor_3'][1]: continue
        raw=items['raw'][0];one=items['factor_1'][0];three=items['factor_3'][0]
        estimate=(3*one-three)/2
        rows.append(dict(zip(('stage','backend','method','depth','repeat'),key)) |
                    {'raw_success':raw,'factor_1_success':one,'factor_3_success':three,
                     'zne_success_estimate':estimate,'change_pp':100*(estimate-raw),
                     'raw_shots':items['raw'][1],'factor_1_shots':items['factor_1'][1],'factor_3_shots':items['factor_3'][1],
                     'outside_probability_range':not 0<=estimate<=1})
    return pd.DataFrame(rows)


def plot_presentation_zne(data, out, formats, dpi):
    table=presentation_zne_success_data(data);backends=presentation_backends(data)
    champion=data.get('manifest',{}).get('champion') or {}
    if not table.empty and champion.get('method') is not None and champion.get('depth') is not None:
        table=table[(table.method==champion['method'])&(table.depth==champion['depth'])]
    if table.empty or not backends:
        # Older or incomplete runs may lack the factor counts needed for this observable.
        return
    table.to_csv(output_file(out, '04_zne_success_data.csv'),index=False)
    fig,ax=slide_axes();fig.subplots_adjust(left=.21);ax.grid(False)
    references=exact_reference_metrics(data)
    reference=references[(references.method==champion.get('method'))&(references.depth==champion.get('depth'))]
    if not reference.empty:
        ax.axvline(float(reference.ideal_full_optimum_probability.iloc[0]),color=CYAN,ls=':',lw=1.5,label='Ideal · exact')
    stages=['noisy','real'];positions={'noisy':1,'real':0}
    offsets=np.linspace(-.17,.17,len(backends)) if len(backends)>1 else [0]
    all_values=[]
    for stage in stages:
        for backend,offset in zip(backends,offsets):
            sub=table[(table.stage==stage)&(table.backend==backend)]
            if sub.empty: continue
            style=device_style(backend);y=positions[stage]+offset
            raw=float(sub.raw_success.mean());zne=float(sub.zne_success_estimate.mean())
            all_values.extend(sub.raw_success.tolist()+sub.zne_success_estimate.tolist())
            jitter=np.linspace(-.045,.045,len(sub)) if len(sub)>1 else [0]
            ax.scatter(sub.raw_success,y+np.asarray(jitter),color=style['color'],s=24,alpha=.28,zorder=2)
            estimates=sub.zne_success_estimate.to_numpy()
            in_range=estimates<=1
            ax.scatter(estimates[in_range],(y+np.asarray(jitter))[in_range],color=style['color'],s=24,alpha=.28,zorder=2)
            if (~in_range).any():
                ax.scatter(np.ones((~in_range).sum()),(y+np.asarray(jitter))[~in_range],color=style['color'],
                           marker='>',s=35,alpha=.6,zorder=4,clip_on=False)
                if zne<=1:
                    text=', '.join(f'{value:.1%}' for value in estimates[~in_range])
                    ax.annotate(f'Off-scale repeats: {text}',(1,y),xytext=(-8,12 if offset>=0 else -19),
                                textcoords='offset points',ha='right',color=style['color'],fontsize=9)
            shown=min(zne,1.)
            ax.plot([raw,shown],[y,y],color=style['color'],linestyle=style['linestyle'],lw=2.5,zorder=3)
            ax.scatter([raw],[y],color=style['color'],marker='o',s=75,zorder=4)
            ax.scatter([shown],[y],facecolor='white',edgecolor=style['color'],marker='>' if zne>1 else 'D',
                       s=75,lw=2,zorder=4,clip_on=False)
            label=f'{(zne-raw)*100:+.1f} pp' + (f' · {zne:.1%} estimate' if zne>1 else '')
            right=shown>=.98
            ax.annotate(label,(shown,y),xytext=(-8 if right else 8,8 if offset>=0 else -15),
                        textcoords='offset points',ha='right' if right else 'left',color=style['color'],fontsize=11,fontweight='bold')
    ax.set_yticks([1,0],['Noisy simulation','Real device']);ax.set_ylim(-.55,1.55)
    ax.tick_params(axis='y',labelsize=13);ax.set_xlabel('Optimal-solution probability')
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    if all_values:
        if not reference.empty: all_values.append(float(reference.ideal_full_optimum_probability.iloc[0]))
        low=min(all_values);high=max(all_values);pad=max((high-low)*.25,.01);ax.set_xlim(min(low-pad,.95),1.0)
    handles=[Line2D([0],[0],color=device_style(b)['color'],linestyle=device_style(b)['linestyle'],lw=2,label=device_style(b)['label']) for b in backends]
    for stage in stages:
        absent=[device_style(b)['label'] for b in backends
                if not ((table.stage==stage)&(table.backend==b)).any()]
        if absent:
            ax.text(.98,positions[stage],f"Not recorded: {', '.join(absent)}",
                    transform=ax.get_yaxis_transform(),ha='right',fontsize=10,color='#888888')
    handles += [Line2D([0],[0],color='#14243A',marker='o',lw=0,label='Raw · measured'),
                Line2D([0],[0],color='#14243A',marker='D',markerfacecolor='white',lw=0,label='ZNE · estimated')]
    if not reference.empty:
        handles.append(Line2D([0],[0],color=CYAN,ls=':',lw=1.5,label='Ideal · exact'))
    if (table.zne_success_estimate>1).any():
        handles.append(Line2D([0],[0],color='#14243A',marker='>',lw=0,label='Estimate >100%'))
    fig.suptitle('ZNE: estimated change in success probability',fontsize=21,fontweight='bold',y=.97)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.89),ncol=5,fontsize=10)
    save_slide(fig,out,'04_zne',formats,dpi)


def plot_presentation(data, output, formats, dpi):
    output.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.size':10, 'axes.titlesize':11, 'axes.labelsize':9,
                         'axes.spines.top':False, 'axes.spines.right':False}):
        plot_presentation_multiangle(data,output,formats,dpi)
        plot_presentation_resources(data,output,formats,dpi)
        plot_presentation_techniques(data,output,formats,dpi)
        plot_presentation_zne(data,output,formats,dpi)


def _selected_champion_rows(data, rows):
    """Use one logical circuit so references and arm comparisons stay meaningful."""
    if rows.empty:
        return rows
    champion = data.get('manifest', {}).get('champion') or {}
    if champion.get('method') is not None and champion.get('depth') is not None:
        return rows[(rows.method == champion['method']) & (rows.depth == champion['depth'])].copy()
    return rows if len(rows[['method', 'depth']].drop_duplicates()) == 1 else rows.iloc[0:0]


def readout_comparison_data(data):
    """Cost estimates paired with their same-repeat, equal-shot raw reference."""
    table = presentation_technique_data(data)
    if table.empty:
        return table
    return _selected_champion_rows(data, table[table.technique.isin(['readout', 'trex'])])


def plot_readout_comparison(data, out, formats, dpi):
    table = readout_comparison_data(data)
    backends = presentation_backends(data)
    if table.empty or not backends:
        return
    table.to_csv(output_file(out, '05_readout_estimates_data.csv'), index=False)
    fig, ax = slide_axes()
    fig.subplots_adjust(left=.33)
    ax.grid(False)
    ideal = float(table.ideal_cost.iloc[0])
    ax.axvline(ideal, color=CYAN, linestyle=':', lw=1.5)
    rows = [(stage, backend, technique) for stage in ('real', 'noisy') for backend in backends
            for technique in ('readout', 'trex')]
    all_values = [ideal]
    for index, (stage, backend, technique) in enumerate(rows):
        part = table[(table.stage == stage) & (table.backend == backend) & (table.technique == technique)]
        y = len(rows) - index - 1
        if part.empty:
            ax.text(.98, y, 'No comparable data', transform=ax.get_yaxis_transform(), ha='right',
                    va='center', fontsize=9, color='#888888')
            continue
        style = device_style(backend)
        raw, estimate = float(part.raw_cost.mean()), float(part.mean_full_cost.mean())
        all_values.extend(part.raw_cost.tolist() + part.mean_full_cost.tolist())
        jitter = np.linspace(-.11, .11, len(part)) if len(part) > 1 else [0]
        ax.scatter(part.raw_cost, y + np.asarray(jitter), color=style['color'], s=19, alpha=.26)
        ax.scatter(part.mean_full_cost, y + np.asarray(jitter), color=style['color'], s=19, alpha=.26)
        ax.plot([raw, estimate], [y, y], color=style['color'], linestyle=style['linestyle'], lw=2)
        ax.scatter([raw], [y], color=style['color'], marker='o', s=62, zorder=3)
        ax.scatter([estimate], [y], facecolor='white', edgecolor=style['color'],
                   marker='D', s=65, lw=1.8, zorder=4)
    labels = [f"{'Hardware' if stage == 'real' else 'Simulation'} · {device_style(backend)['label']} · "
              f"{'Readout' if technique == 'readout' else 'TREX-style'}"
              for stage, backend, technique in rows]
    ax.set_yticks(range(len(rows)), labels[::-1])
    ax.set_ylim(-.6, len(rows)-.4)
    ax.set_xlabel('Expected full pairwise cost')
    lo, hi = min(all_values), max(all_values)
    pad = max((hi-lo)*.10, .02)
    ax.set_xlim(lo-pad, hi+pad)
    handles = [Line2D([0], [0], color=device_style(b)['color'], linestyle=device_style(b)['linestyle'],
                      lw=2, label=device_style(b)['label']) for b in backends]
    handles += [Line2D([0], [0], color='#14243A', marker='o', lw=0, label='Raw · measured'),
                Line2D([0], [0], color='#14243A', marker='D', markerfacecolor='white', lw=0,
                       label='Corrected · estimated'),
                Line2D([0], [0], color=CYAN, linestyle=':', lw=1.5, label='Ideal · exact')]
    fig.suptitle('Readout correction: before and after', fontsize=21, fontweight='bold', y=.97)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .89), ncol=len(handles), fontsize=10)
    save_slide(fig, out, '05_readout_estimates', formats, dpi)


def readout_calibration_data(data):
    """Derive per-qubit calibration errors, retaining physical measurement mappings.

    Qiskit count strings are reversed into classical/SNP order. TREX-style
    calibration masks are already stored in SNP order. State preparation can
    contribute to these errors; they are not an independent gate-noise metric.
    """
    reads = data.get('problem', {}).get('reads')
    if not reads:
        return pd.DataFrame()
    n = len(reads[0])
    keys = ('stage', 'backend', 'method', 'depth', 'repeat', 'technique')
    groups = {}
    for record in data.get('records', []):
        if record.get('stage') not in ('noisy', 'real') or record.get('technique') not in ('readout', 'trex'):
            continue
        key = tuple(record.get(name) for name in keys)
        variant = record.get('variant')
        if not isinstance(variant, str):
            continue
        group = groups.setdefault(key, {})
        if variant in group:
            group['_duplicate'] = True
        group[variant] = record
    rows = []
    for key, group in groups.items():
        technique = key[-1]
        if '_duplicate' in group:
            continue
        if technique == 'readout':
            if not {'target', 'cal0', 'cal1'} <= group.keys():
                continue
            calibrations = [(group['cal0'], '0'*n), (group['cal1'], '1'*n)]
        else:
            cal_masks = {v[len('cal_mask_'):] for v in group if v.startswith('cal_mask_')}
            target_masks = {v[len('target_mask_'):] for v in group if v.startswith('target_mask_')}
            if (len(cal_masks) != min(16, 2**n) or cal_masks != target_masks or
                    any(len(mask) != n or set(mask) - {'0', '1'} for mask in cal_masks)):
                continue
            calibrations = [(group['cal_mask_'+mask], mask) for mask in sorted(cal_masks)]
        # response[qubit, prepared, measured] contains shot counts.
        response = np.zeros((n, 2, 2), dtype=np.int64)
        mappings = []
        valid = True
        for record, prepared in calibrations:
            counts = record.get('counts') or {}
            shots = sum(counts.values())
            if shots <= 0 or shots != record.get('shots'):
                valid = False
                break
            mapping = record.get('circuit_resources', {}).get('measurement_map') or {}
            mappings.append({str(k): v for k, v in mapping.items()})
            for label, count in counts.items():
                measured = label.replace(' ', '')[::-1]
                if len(measured) != n or set(measured) - {'0', '1'} or count < 0:
                    valid = False
                    break
                for bit in range(n):
                    response[bit, int(prepared[bit]), int(measured[bit])] += count
            if not valid:
                break
        if not valid or any(mapping != mappings[0] for mapping in mappings):
            continue
        for bit in range(n):
            totals = response[bit].sum(axis=1)
            if (totals == 0).any():
                continue
            p01 = response[bit, 0, 1] / totals[0]
            p10 = response[bit, 1, 0] / totals[1]
            rows.append(dict(zip(keys, key)) | {
                'snp': bit + 1, 'physical_qubit': mappings[0].get(str(bit)),
                'zero_preparation_shots': int(totals[0]), 'one_preparation_shots': int(totals[1]),
                'zero_to_one_errors': int(response[bit, 0, 1]), 'one_to_zero_errors': int(response[bit, 1, 0]),
                'zero_to_one_rate': p01, 'one_to_zero_rate': p10, 'mean_calibration_error': (p01+p10)/2,
            })
    table = pd.DataFrame(rows)
    return _selected_champion_rows(data, table) if not table.empty else table


def plot_readout_calibration(data, out, formats, dpi):
    table = readout_calibration_data(data)
    backends = presentation_backends(data)
    if table.empty or not backends:
        return
    table.to_csv(output_file(out, '06_readout_calibration_data.csv'), index=False)
    stage = 'real' if (table.stage == 'real').any() else 'noisy'
    selected = table[table.stage == stage]
    n = len(data['problem']['reads'][0])
    styles = {'readout': (CYAN, 'o', 'Readout'), 'trex': (TIE, 'D', 'TREX-style')}
    fig, ax = slide_axes()
    missing = []
    for technique, (color, marker, label) in styles.items():
        for backend in backends:
            part = selected[(selected.technique == technique) & (selected.backend == backend)]
            if part.empty:
                missing.append(f"{device_style(backend)['label']} {label}")
                continue
            means = part.groupby('snp').mean_calibration_error.mean().reindex(range(1, n+1))
            for snp, samples in part.groupby('snp'):
                jitter = np.linspace(-.035, .035, len(samples)) if len(samples) > 1 else [0]
                ax.scatter(snp + np.asarray(jitter), samples.mean_calibration_error,
                           color=color, s=20, alpha=.22)
            ax.plot(means.index, means, color=color, linestyle=device_style(backend)['linestyle'],
                    marker=marker, lw=2, ms=7)
    ax.set_xticks(range(1, n+1))
    ax.set_xlabel('Logical qubit / SNP')
    ax.set_ylabel('Mean calibration error (includes state preparation)')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles = [Line2D([0], [0], color=color, marker=marker, lw=2, label=label)
               for color, marker, label in styles.values()]
    handles += [Line2D([0], [0], color='#14243A', linestyle=device_style(b)['linestyle'], lw=2,
                       label=device_style(b)['label']) for b in backends]
    title = 'Hardware' if stage == 'real' else 'Noisy simulation'
    fig.suptitle(f'{title}: readout calibration by qubit', fontsize=21, fontweight='bold', y=.97)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .89), ncol=len(handles), fontsize=11)
    if missing:
        ax.text(.98, .96, 'Not recorded: '+', '.join(missing), transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='#888888')
    save_slide(fig, out, '06_readout_calibration', formats, dpi)


def measured_mitigation_data(data):
    """DD and twirling success frequencies matched to same-repeat raw shots."""
    frame = data.get('techniques')
    if frame is None or frame.empty or not {'total_shots', 'optimum_probability'} <= set(frame):
        return pd.DataFrame()
    keys = ['stage', 'backend', 'method', 'depth', 'repeat']
    raw = frame[frame.technique == 'raw_reference'][keys+['total_shots', 'optimum_probability']].rename(
        columns={'total_shots': 'raw_total_shots', 'optimum_probability': 'raw_success'})
    arms = frame[frame.stage.isin(['noisy', 'real']) & frame.technique.isin(['dd', 'twirling'])]
    table = arms.merge(raw, on=keys, how='inner', validate='many_to_one')
    table = table[(table.total_shots == table.raw_total_shots) & table.optimum_probability.notna() &
                  table.raw_success.notna()].copy()
    table['change_pp'] = 100 * (table.optimum_probability - table.raw_success)
    return _selected_champion_rows(data, table)


def plot_measured_mitigation(data, out, formats, dpi):
    table = measured_mitigation_data(data)
    backends = presentation_backends(data)
    if table.empty or not backends:
        return
    table.to_csv(output_file(out, '07_measured_mitigation_data.csv'), index=False)
    fig, ax = slide_axes()
    fig.subplots_adjust(left=.26)
    ax.grid(False)
    ax.axvline(0, color='#888888', ls=':', lw=1.2)
    rows = [(stage, technique) for stage in ('real', 'noisy') for technique in ('dd', 'twirling')]
    offsets = np.linspace(-.17, .17, len(backends)) if len(backends) > 1 else [0]
    values = []
    for i, (stage, technique) in enumerate(rows):
        base_y = len(rows)-1-i
        for backend, offset in zip(backends, offsets):
            part = table[(table.stage == stage) & (table.backend == backend) & (table.technique == technique)]
            style = device_style(backend)
            y = base_y + offset
            if part.empty:
                ax.text(.98, y, f"{style['label']}: no comparable data", transform=ax.get_yaxis_transform(),
                        ha='right', va='center', fontsize=9, color=style['color'])
                continue
            changes = part.change_pp.to_numpy()
            values.extend(changes)
            mean = float(changes.mean())
            jitter = np.linspace(-.05, .05, len(changes)) if len(changes) > 1 else [0]
            ax.plot([0, mean], [y, y], color=style['color'], linestyle=style['linestyle'], lw=2)
            ax.scatter(changes, y + np.asarray(jitter), color=style['color'], s=24, alpha=.30)
            ax.scatter([mean], [y], color=style['color'], marker=style['marker'], s=65, zorder=3)
    ax.set_yticks(range(len(rows)), [f"{'Hardware' if stage == 'real' else 'Simulation'} · "
                                   f"{'X–X DD' if technique == 'dd' else 'Pauli twirling'}"
                                   for stage, technique in rows][::-1])
    ax.set_ylim(-.6, len(rows)-.4)
    if values:
        lo, hi = min(0, min(values)), max(0, max(values))
        pad = max((hi-lo)*.16, .15)
        ax.set_xlim(lo-pad, hi+pad)
    ax.set_xlabel('Measured optimum frequency change vs raw (percentage points)')
    handles = [Line2D([0], [0], color=device_style(b)['color'], linestyle=device_style(b)['linestyle'],
                      marker=device_style(b)['marker'], lw=2, label=device_style(b)['label']) for b in backends]
    fig.suptitle('DD and twirling: did success improve?', fontsize=21, fontweight='bold', y=.97)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .89), ncol=len(handles), fontsize=12)
    save_slide(fig, out, '07_measured_mitigation', formats, dpi)


def plot_mitigation_details(data, out, formats, dpi):
    plot_readout_comparison(data, out, formats, dpi)
    plot_readout_calibration(data, out, formats, dpi)
    plot_measured_mitigation(data, out, formats, dpi)


def problem(reads):
    reads = np.asarray(reads)
    if reads.ndim != 2 or not 1 <= reads.shape[1] <= 12 or not len(reads) or not np.isin(reads, [-1, 0, 1]).all():
        raise ValueError('Reads must be a nonempty matrix of -1/0/1 with 1–12 SNPs.')
    n = reads.shape[1]
    a, d = np.zeros((n,n)), np.zeros((n,n))
    for row in reads:
        for i,j in itertools.combinations(np.flatnonzero(row != -1), 2):
            (a if row[i] == row[j] else d)[i,j] += 1
    states = np.array(list(itertools.product([0,1], repeat=n)), dtype=int)
    full = sum((np.where(states[:,i] == states[:,j], d[i,j], a[i,j])
                for i,j in itertools.combinations(range(n),2)), np.zeros(len(states)))
    mec = np.zeros(len(states))
    for row in reads:
        seen = row != -1
        dist = (states[:,seen] != row[seen]).sum(axis=1)
        mec += np.minimum(dist, seen.sum()-dist)
    return reads, a, d, states, full, mec


MODE_COLORS = {'ideal': CYAN, 'noisy': GOLD, 'real': TIE}
MODE_LABELS = {'ideal': 'Ideal sample', 'noisy': 'Calibrated noise', 'real': 'Real device'}


def _subplots(*args, figsize=None, constrained_layout=False, **kwargs):
    fig = Figure(figsize=figsize, layout='constrained' if constrained_layout else None)
    FigureCanvasAgg(fig)
    return fig, fig.subplots(*args, **kwargs)


def _name(name):
    return name.replace('standard', 'Standard').replace('multi', 'Multi-angle').replace('_full_', ' · full · ').replace('_forest_', ' · forest · ')


def _groups(frame, backends, methods):
    return [(backend, method) for backend in backends for method in methods
            if ((frame.backend == backend) & (frame.method == method)).any()]


def _bars(ax, frame, metric, groups, modes, *, probability=False, errors=False, baseline=None):
    width = 0.75 / max(len(modes), 1)
    for k, mode in enumerate(modes):
        y, vals, err_low, err_high = [], [], [], []
        for i, (backend, method) in enumerate(groups):
            match = frame[(frame.backend == backend) & (frame.method == method) & (frame['mode'] == mode)]
            if match.empty or metric not in match or pd.isna(match.iloc[0][metric]):
                continue
            row = match.iloc[0]; y.append(i + (k-(len(modes)-1)/2)*width)
            vals.append(float(row[metric]))
            if errors:
                # Binomial shot uncertainty only; this does not include model/calibration uncertainty.
                p, shots = float(row[metric]), int(row.shots)
                z = 1.96
                denominator = 1 + z*z/shots
                center = (p + z*z/(2*shots))/denominator
                half = z*np.sqrt(p*(1-p)/shots + z*z/(4*shots*shots))/denominator
                err_low.append(max(0., p-(center-half)))
                err_high.append(max(0., (center+half)-p))
        if vals:
            ax.barh(y, vals, height=width*.9, color=MODE_COLORS[mode], label=MODE_LABELS[mode],
                    xerr=np.array([err_low,err_high]) if errors else None, error_kw={'capsize': 2, 'linewidth': .8})
    if baseline is not None:
        ax.axvline(baseline, color='#14243A', linestyle='--', linewidth=1, label='Exact minimum')
    ax.set_yticks(range(len(groups)), [f'{b.replace("ibm_", "").title()}  |  {_name(m)}' for b,m in groups])
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    if probability:
        ax.set_xlim(0,1.05); ax.xaxis.set_major_formatter(PercentFormatter(1))


def _save(fig, folder, name, paths):
    target = output_file(folder, name)
    fig.savefig(target, dpi=150, bbox_inches='tight', facecolor='white')
    fig.clear(); paths.append(target)


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def plot_legacy_run(run_dir, output=None):
    """Create figures and frequency tables using already saved data; no backend access."""
    # Import lazily to reuse the study's validated scoring and bit-order conventions.

    run = Path(run_dir)
    manifest = _read(run/'manifest.json')
    reference = _read(run/'references.json')
    counts = _read(run/'counts.json')
    training = _read(run/'training.json')
    results = pd.read_csv(run/'results.csv')
    screening = pd.read_csv(run/'screening.csv')
    _, _, _, states, full, mec = problem(manifest['reads'])
    state_labels = [''.join(map(str, s)) for s in states]
    state_index = {state: i for i, state in enumerate(state_labels)}
    backends = list(dict.fromkeys(results.backend))
    methods = manifest.get('selected_methods') or list(dict.fromkeys(results.method))
    methods = [m for m in methods if m in set(results.method)]
    modes = [m for m in ('ideal', 'noisy', 'real') if m in set(results['mode'])]
    groups = _groups(results, backends, methods)
    out = Path(output) if output else default_output_path(run); out.mkdir(parents=True, exist_ok=True)
    paths = []

    # The screening plot covers every candidate; final evaluation may contain only a subset.
    candidate_order = list(dict.fromkeys(screening.method))
    screen_groups = _groups(screening, backends, candidate_order)
    screen = screening.copy(); screen['mode'] = 'noisy'
    fig, axes = _subplots(1, 2, figsize=(14, max(5, .36*len(screen_groups)+2)), constrained_layout=True)
    for ax, col, title, prob in zip(axes, ['full_optimum_probability','mean_full_cost'],
                                     ['Full-optimum frequency', 'Mean full pairwise cost'], [True, False]):
        _bars(ax, screen, col, screen_groups, ['noisy'], probability=prob, errors=prob,
              baseline=None if prob else reference['full_min'])
        ax.set_title(title); ax.set_xlabel('Proportion of shots' if prob else 'Cost')
    axes[1].set_yticklabels([])
    fig.suptitle('Screening: all candidates under each device noise model', fontsize=14, fontweight='bold')
    _save(fig, out, '01_screening.png', paths)

    fig, axes = _subplots(2, 2, figsize=(15, max(7, .42*len(groups)+4)), constrained_layout=True)
    panels = [('full_optimum_probability','Full pairwise optimum frequency',True,None),
              ('mec_optimum_probability','MEC optimum frequency',True,None),
              ('mean_full_cost','Mean full pairwise cost',False,reference['full_min']),
              ('mean_mec','Mean MEC',False,reference['mec_min'])]
    for ax, (col,title,prob,baseline) in zip(axes.flat, panels):
        _bars(ax, results, col, groups, modes, probability=prob, errors=prob, baseline=baseline)
        ax.set_title(title); ax.set_xlabel('Proportion of shots' if prob else 'Cost')
    for ax in axes[:,1]: ax.set_yticklabels([])
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=len(modes),frameon=False)
    fig.suptitle('Final evaluation · whiskers show 95% Wilson shot intervals', fontsize=14, fontweight='bold')
    _save(fig, out, '02_quality.png', paths)

    # Each resource describes the compiled circuit, independent of execution mode.
    resources = results.drop_duplicates(['backend','method']).copy(); resources['mode'] = 'noisy'
    fig, axes = _subplots(1, 3, figsize=(15,max(4,.35*len(groups)+2)), constrained_layout=True)
    for ax, (col,title) in zip(axes, [('depth','Total depth'),('two_qubit_depth','Two-qubit depth'),('two_qubit_gates','Two-qubit gates')]):
        _bars(ax, resources, col, groups, ['noisy']); ax.set_title(title); ax.set_xlabel('Compiled circuit')
    for ax in axes[1:]: ax.set_yticklabels([])
    fig.suptitle('Hardware-compiled circuit resources · one circuit per device and method', fontsize=14, fontweight='bold')
    _save(fig, out, '03_resources.png', paths)

    fig, axes = _subplots(1, 2, figsize=(14,max(4,.38*len(groups)+2)), constrained_layout=True)
    _bars(axes[0], results, 'total_variation_from_ideal', groups, modes, probability=True)
    axes[0].set_title('Distance from exact ideal distribution'); axes[0].set_xlabel('Total variation distance')
    raw_corrected = results[results['mode'].isin(['noisy','real'])].copy()
    # Use color for raw versus corrected; label includes device and mode.
    quality_groups = [(r.backend,r.method,r['mode']) for _,r in raw_corrected.iterrows()]
    for i, (backend,method,mode) in enumerate(quality_groups):
        row = raw_corrected[(raw_corrected.backend==backend)&(raw_corrected.method==method)&(raw_corrected['mode']==mode)].iloc[0]
        axes[1].barh(i-.18, row.mean_full_cost, height=.34, color=GREY, label='Raw' if i == 0 else None)
        if pd.notna(row.readout_corrected_mean_full_cost):
            axes[1].barh(i+.18, row.readout_corrected_mean_full_cost, height=.34, color=CYAN, label='Readout corrected' if i == 0 else None)
    axes[1].set_yticks(range(len(quality_groups)), [f'{b.replace("ibm_", "").title()} | {_name(m)} | {mode}' for b,m,mode in quality_groups], fontsize=8)
    axes[1].invert_yaxis(); axes[1].axvline(reference['full_min'],color='#14243A',linestyle='--',label='Exact minimum')
    axes[1].set_title('Readout correction estimates mean cost'); axes[1].set_xlabel('Full pairwise cost'); axes[1].grid(axis='x',alpha=.18)
    axes[1].legend(fontsize=8)
    fig.suptitle('Distribution error and readout correction', fontsize=14, fontweight='bold')
    _save(fig, out, '04_noise_and_readout.png', paths)

    # Seconds are reported separately: each mode has different execution and calibration overhead.
    fig, axes = _subplots(1, 2, figsize=(14,max(4,.38*len(groups)+2)), constrained_layout=True)
    methods_with_time = [m for m in methods if training.get(m,{}).get('wall_seconds') is not None]
    axes[0].barh([_name(m) for m in methods_with_time], [training[m]['wall_seconds'] for m in methods_with_time], color=INK)
    axes[0].invert_yaxis(); axes[0].set_title('Local training'); axes[0].set_xlabel('Wall seconds · all starts')
    if not methods_with_time:
        axes[0].text(.5,.5,'Not recorded in this run',ha='center',transform=axes[0].transAxes)
    _bars(axes[1], resources, 'compile_wall_seconds', groups, ['noisy'])
    if 'compile_wall_seconds' not in resources or resources['compile_wall_seconds'].isna().all():
        axes[1].text(.5,.5,'Not recorded in this run',ha='center',transform=axes[1].transAxes)
    axes[1].set_title('Compilation per device'); axes[1].set_xlabel('Wall seconds')
    fig.suptitle('Classical preparation time', fontsize=14, fontweight='bold')
    _save(fig, out, '05_preparation_time.png', paths)
    if 'execution_wall_seconds' in results and results.execution_wall_seconds.notna().any():
        fig, axes = _subplots(1,len(modes),figsize=(6*len(modes),max(4,.38*len(groups)+2)),squeeze=False,constrained_layout=True)
        for ax,mode in zip(axes[0],modes):
            _bars(ax, results, 'execution_wall_seconds', groups, [mode])
            ax.set_title(MODE_LABELS[mode]); ax.set_xlabel('Wall seconds')
        fig.suptitle('Execution wall time · noise/real include two calibrations; real includes queue', fontsize=13, fontweight='bold')
        _save(fig, out, '06_execution_time.png', paths)

    # Full frequency tables use SNP-first labels. Qiskit count strings are reversed.
    frequencies = []; cost_frequencies = []
    exact = dict(reference.get('ideal_probabilities', {}))
    for method in methods:
        # Older runs stored the frozen circuit but not the exact probabilities.
        circuit_file = run/f'{method}_logical.qpy'
        if method not in exact and circuit_file.exists():
            from qiskit import qpy
            from qiskit.quantum_info import Statevector
            with circuit_file.open('rb') as stream:
                circuit = qpy.load(stream)[0]
            vector = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities()
            exact[method] = {label: float(vector[sum(int(bit) << i for i, bit in enumerate(state))])
                             for label,state in zip(state_labels,states)}
        distributions = []
        if method in exact:
            distributions.append(('Exact ideal', np.array([float(exact[method].get(s,0)) for s in state_labels])))
        for backend in backends:
            for mode in modes:
                entry = counts.get(f'{backend}:{method}:{mode}')
                if not entry: continue
                samples = entry['counts']; total = sum(samples.values())
                if total <= 0: raise ValueError('A saved result has no shots.')
                probs = np.zeros(len(states))
                for key,count in samples.items():
                    label = key.replace(' ','')[::-1]
                    if label not in state_index: raise ValueError(f'Unexpected count bitstring: {key}')
                    probs[state_index[label]] += count/total
                distributions.append((f'{backend.replace("ibm_", "").title()} · {MODE_LABELS[mode]}',probs))
                for label,prob in zip(state_labels,probs):
                    frequencies.append({'backend':backend,'method':method,'mode':mode,'state_snp_first':label,'probability':prob,'shots':total})
        if method in exact:
            for label,prob in zip(state_labels,distributions[0][1]):
                frequencies.append({'backend':'','method':method,'mode':'ideal_exact','state_snp_first':label,'probability':prob,'shots':None})
        if not distributions: continue
        matrix=np.stack([x[1] for x in distributions])
        if len(state_labels) <= 32:
            chosen=np.arange(len(state_labels)); display_labels=state_labels; shown=matrix
        else:
            chosen=np.argsort(-matrix.max(axis=0),kind='stable')[:31]
            display_labels=[state_labels[i] for i in chosen]+['Other']
            shown=np.column_stack([matrix[:,chosen], np.maximum(0, 1-matrix[:,chosen].sum(axis=1))])
        fig, ax = _subplots(figsize=(max(10,.40*len(display_labels)+3),max(3,.55*len(distributions)+1.8)),constrained_layout=True)
        im=ax.imshow(shown,aspect='auto',cmap=SEQUENTIAL,norm=PowerNorm(gamma=.45,vmin=0,vmax=max(float(shown.max()),.01)))
        ax.set_xticks(range(len(display_labels)), [s+'*' if s in reference['full_minimizers'] else s for s in display_labels], rotation=90,fontsize=8)
        ax.set_yticks(range(len(distributions)), [name for name,_ in distributions])
        ax.set_xlabel('Output state · SNP 1 first · * full optimum')
        ax.set_title(f'Final output frequencies · {_name(method)}',fontweight='bold')
        color=fig.colorbar(im,ax=ax,shrink=.8); color.ax.yaxis.set_major_formatter(PercentFormatter(1)); color.set_label('Fraction of shots')
        _save(fig,out,f'07_outcomes_{method}.png',paths)

        # Aggregate every output, including tail states, by its full pairwise cost.
        cost_values=np.unique(full)
        if len(cost_values)>30:
            edges=np.linspace(float(full.min()),float(full.max())+1e-9,21)
            bin_index=np.clip(np.digitize(full,edges)-1,0,len(edges)-2)
            x=(edges[:-1]+edges[1:])/2; lower=edges[:-1]; upper=edges[1:]; xlabel='Full pairwise cost (20 bins)'
        else:
            bin_index=np.searchsorted(cost_values,full); x=cost_values; lower=upper=cost_values; xlabel='Full pairwise cost'
        fig, ax=_subplots(figsize=(max(9,.65*len(x)+3),4.5),constrained_layout=True)
        width=.8/len(distributions)
        for index,(label,probs) in enumerate(distributions):
            values=np.bincount(bin_index,weights=probs,minlength=len(x))
            positions=np.arange(len(x))+(index-(len(distributions)-1)/2)*width
            ax.bar(positions,values,width=width,label=label)
            for cost,low,high,prob in zip(x,lower,upper,values):
                cost_frequencies.append({'method':method,'series':label,'cost_or_bin_midpoint':float(cost),
                                         'bin_lower':float(low),'bin_upper':float(high),'probability':float(prob)})
        ax.set_xticks(range(len(x)),[f'{v:g}' for v in x],rotation=45 if len(x)>12 else 0)
        ax.set(title=f'Whole cost distribution · {_name(method)}',xlabel=xlabel,ylabel='Fraction of shots')
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y',alpha=.2); ax.legend(fontsize=8,ncol=2)
        _save(fig,out,f'08_cost_distribution_{method}.png',paths)
    pd.DataFrame(frequencies).to_csv(output_file(out, 'outcome_frequencies.csv'),index=False)
    pd.DataFrame(cost_frequencies).to_csv(output_file(out, 'cost_frequencies.csv'),index=False)
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',help='Run folder name or path; default: latest completed run')
    parser.add_argument('--results-root',default='bench/results',help='Results root relative to project, or absolute path')
    parser.add_argument('--output',help='Output folder; default: bench/plots/generated/<run-name>')
    parser.add_argument('--depth',type=int,help='Distribution depth; default: selected champion depth')
    parser.add_argument('--all-depths',action='store_true',help='Create distribution plots for every recorded depth')
    parser.add_argument('--presentation-only',action='store_true',help='Generate the compact presentation and mitigation figures')
    parser.add_argument('--dpi',type=int,default=200)
    parser.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf'])
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[2]; root=Path(args.results_root)
    if not root.is_absolute(): root=project/root
    if args.run:
        candidate=Path(args.run)
        run=candidate if candidate.is_absolute() or candidate.exists() else root/'backend_comparison'/candidate
    else: run=latest_run(root)
    out=Path(args.output) if args.output else default_output_path(run); out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(run/'manifest.json')
    if manifest.get('schema_version') != 2:
        if args.presentation_only:
            raise ValueError('Presentation figures require version 2 saved results.')
        paths=plot_legacy_run(run, out)
        print(f'Legacy run: {run}\nOutput: {out}\nSaved {len(paths)} plots. Missing historical measurements are not recorded.')
        return
    data=load_run(run)
    if 'results' not in data: raise FileNotFoundError(f'No results.csv in {run}')
    if manifest.get('status') != 'complete':
        print('Warning: this run is not complete; saved result tables may omit later measurements.',file=sys.stderr)
    if args.dpi<=0: parser.error('--dpi must be positive')
    depth=args.depth if args.depth is not None else data['manifest'].get('champion',{}).get('depth',2)
    if depth <= 0: parser.error('--depth must be positive')
    print(f'Run: {run}\nOutput: {out}\nDistribution depth: {depth}')
    plot_presentation(data,out,args.formats,args.dpi)
    plot_mitigation_details(data,out,args.formats,args.dpi)
    if args.presentation_only:
        print(f'Presentation figures and tables are grouped by format in: {out}')
        return
    if 'frequencies' in data:
        data['frequencies'].to_csv(output_file(out, 'outcome_frequencies.csv'),index=False)
    data['results'].to_csv(output_file(out, 'metric_plot_data.csv'),index=False)
    if 'techniques' in data:
        data['techniques'].to_csv(output_file(out, 'technique_plot_data.csv'),index=False)
    plot_problem(data,out,args.formats,args.dpi)
    for metric,title,ylabel,name,probability in [
        ('full_optimum_probability','Full-optimum frequency','Fraction of shots at a full optimum','optimum_vs_depth',True),
        ('mean_full_cost','Full pairwise cost','Mean full pairwise cost','cost_vs_depth',False),
        ('mean_mec','Full-read MEC','Mean MEC','mec_vs_depth',False),
        ('total_variation_from_ideal','Distribution distance from ideal','Total variation distance','distance_vs_depth',False)]:
        plot_metric(data,out,metric,title,ylabel,name,args.formats,args.dpi,probability)
    plot_noise_comparisons(data,out,depth,args.formats,args.dpi)
    plot_resources(data,out,args.formats,args.dpi)
    plot_training(data,out,args.formats,args.dpi)
    plot_techniques(data,out,args.formats,args.dpi)
    plot_zne(data,out,args.formats,args.dpi)
    plot_run_diagnostics(data,out,args.formats,args.dpi)
    depths = sorted(data['results'].depth.dropna().astype(int).unique()) if args.all_depths else [depth]
    for selected_depth in depths:
        plot_distributions(data,out,selected_depth,args.formats,args.dpi)
        plot_distribution_changes(data,out,selected_depth,args.formats,args.dpi)
    print('Finished. Real-device results appear only when recorded data exists.')


if __name__=='__main__':
    try: main()
    except (FileNotFoundError,ValueError,KeyError) as exc:
        print(f'Error: {exc}',file=sys.stderr); sys.exit(1)
