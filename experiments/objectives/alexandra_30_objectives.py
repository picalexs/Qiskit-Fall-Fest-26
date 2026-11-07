"""Exact five-objective comparison at 30 SNPs using local-factor dynamic programming.

python experiments/objectives/alexandra_30_objectives.py
Short reads make the factor span small. No 2^30 enumeration or QAOA is used.
"""

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from alexandra_larger_objectives import ROOT, load_shared, compare_instance, graphs_for
from objective_study import VARIANTS, write_csv

OUT = ROOT/'results'/'objectives'/'alexandra_30_results'


def factors(reads):
    result = []
    for read in reads:
        positions = np.flatnonzero(read != -1)
        if len(positions) >= 2:
            result.append((positions, read[positions]))
    return result


def solve(reads, truth, variant, linked):
    """Exact minima and uniform-over-minimizers metrics, including ALL ties.

    Integer-scaled factors avoid floating-point tie decisions. Truth only updates
    reporting accumulators; optimization and lexicographic selection are truth-blind.
    """
    original = factors(reads)
    native = factors(np.unique(reads, axis=0)) if variant == VARIANTS[3] else original
    spans = [int(p[-1]-p[0]) for p, _ in original]
    width = max(spans, default=1)
    if width > 12:
        raise ValueError('DP factor width exceeds the deliberate small-window limit.')
    denominators = [len(p)*(len(p)-1)//2 if variant==VARIANTS[1]
                    else len(p)-1 if variant==VARIANTS[2] else 1 for p, _ in native]
    scale = math.lcm(*denominators) if denominators else 1
    n = reads.shape[1]
    mask = (1 << width)-1
    original_by_end = [[] for _ in range(n)]
    native_by_end = [[] for _ in range(n)]
    for p, calls in original:
        original_by_end[int(p[-1])].append((p,calls))
    for p, calls in native:
        native_by_end[int(p[-1])].append((p,calls))

    def mismatches(p, calls, i, state, bit):
        return sum((bit if j==i else (state >> (i-int(j)-1)) & 1) != int(call)
                   for j, call in zip(p,calls))

    def increment(group, i, state, bit, mode):
        total = 0
        for p, calls in group:
            length = len(p)
            d = mismatches(p,calls,i,state,bit)
            if mode == VARIANTS[4]:
                total += min(d,length-d)
            else:
                denominator = length*(length-1)//2 if mode==VARIANTS[1] else length-1 if mode==VARIANTS[2] else 1
                total += d*(length-d)*(scale//denominator if mode==variant else 1)
        return int(total)

    # state -> (minimum cost, count, Hamming histogram, switch sum,
    #           full pair cost sum, full MEC sum, lexicographic representative)
    records = {0:(0,1,[1]+[0]*n,0,0,0,'')}
    for i in range(n):
        new = {}
        for state, (cost,count,hist,switch_sum,full_sum,mec_sum,path) in records.items():
            for bit in ((0,) if i==0 else (0,1)):
                delta = increment(native_by_end[i],i,state,bit,variant)
                full_delta = sum(mismatches(p,c,i,state,bit)*(len(p)-mismatches(p,c,i,state,bit))
                                 for p,c in original_by_end[i])
                mec_delta = sum(min(mismatches(p,c,i,state,bit),len(p)-mismatches(p,c,i,state,bit))
                                for p,c in original_by_end[i])
                error = int(bit != int(truth[i]))
                new_hist = ([0]+hist[:-1]) if error else hist.copy()
                switch = int(i>0 and linked[i-1] and
                             ((state & 1)^int(truth[i-1])) != (bit^int(truth[i])))
                key = ((state << 1)|bit)&mask
                candidate = (cost+delta,count,new_hist,switch_sum+count*switch,
                             full_sum+count*full_delta,mec_sum+count*mec_delta,path+str(bit))
                old = new.get(key)
                if old is None or candidate[0] < old[0]:
                    new[key] = candidate
                elif candidate[0] == old[0]:
                    new[key] = (old[0],old[1]+count,
                                [a+b for a,b in zip(old[2],new_hist)],
                                old[3]+candidate[3],old[4]+candidate[4],old[5]+candidate[5],
                                min(old[6],candidate[6]))
        records = new
    minimum = min(r[0] for r in records.values())
    winners = [r for r in records.values() if r[0]==minimum]
    total = sum(r[1] for r in winners)
    hist = [sum(r[2][d] for r in winners) for d in range(n+1)]
    truth_cost = 0
    for p,calls in native:
        length = len(p)
        d = int(np.sum(truth[p] != calls))
        denom = length*(length-1)//2 if variant==VARIANTS[1] else length-1 if variant==VARIANTS[2] else 1
        truth_cost += min(d,length-d) if variant==VARIANTS[4] else d*(length-d)*(scale//denom)
    correct = int(truth_cost==minimum)
    hamming = sum(min(d,n-d)*value for d,value in enumerate(hist))/total
    return dict(variant=variant,native_minimum=minimum/scale,native_truth_cost=truth_cost/scale,
                truth_is_optimal=correct,recovery=correct/total,mean_hamming=hamming,
                hamming_fraction=hamming/n,mean_switch_errors=sum(r[3] for r in winners)/total,
                tied_phasings=total,tied_labels=2*total,
                selected_candidate=min(r[6] for r in winners),
                full_mean=sum(r[4] for r in winners)/total,
                mec_mean=sum(r[5] for r in winners)/total,factor_width=width)


def checks(engine, real):
    for n, seed in ((6,0),(6,3),(12,0),(20,1)):
        inst = engine.make_instance(n,seed=seed,**real)
        exact,_,_,_ = compare_instance(inst,engine)
        a,d = engine.pairwise_evidence(inst.reads)
        linked = engine.linked_boundaries(a,d)
        for expected in exact:
            actual = solve(inst.reads,inst.truth,expected['variant'],linked)
            for key in actual:
                if key in expected:
                    assert actual[key]==expected[key] if isinstance(actual[key],str) else np.isclose(actual[key],expected[key]), (n,seed,key)
    # Many minimizers: isolated SNPs must be counted without enumerating them.
    empty = np.full((1,30),-1)
    for name in VARIANTS:
        result = solve(empty,np.zeros(30,dtype=int),name,np.zeros(29,dtype=bool))
        assert result['tied_phasings']==2**29 and np.isclose(result['recovery'],1/2**29,rtol=1e-12,atol=0)
        assert result['native_minimum']==0
    print('DP verified against enumeration through 20 SNPs, all five objectives and ties.',flush=True)


def main():
    engine,real = load_shared()
    checks(engine,real)
    OUT.mkdir(exist_ok=True)
    rows,inputs = [],{}
    for seed in (0,1,2):
        inst = engine.make_instance(30,seed=seed,**real)
        a,d = engine.pairwise_evidence(inst.reads)
        linked = engine.linked_boundaries(a,d)
        graphs = graphs_for(inst.reads)
        for name in VARIANTS:
            start = time.perf_counter()
            result = solve(inst.reads,inst.truth,name,linked)
            if name in graphs:
                _,_,j,offset = graphs[name]
                assert np.isclose(result['native_minimum'],engine.banded_exact(j,offset))
            row = dict(instance=inst.name,n_snps=30,seed=seed,n_reads=len(inst.reads),
                       unique_reads=len(np.unique(inst.reads,axis=0)),
                       linked_boundaries=int(linked.sum()),**result,
                       wall_seconds=time.perf_counter()-start)
            rows.append(row)
            print(f'{inst.name} | {name} | recovery={result["recovery"]:.1%} | Hamming={result["mean_hamming"]:.3f}',flush=True)
        inputs[inst.name+'_reads']=inst.reads
        inputs[inst.name+'_truth']=inst.truth
    write_csv(OUT/'results.csv',rows)
    np.savez_compressed(OUT/'inputs.npz',**inputs)
    (OUT/'manifest.json').write_text(json.dumps(dict(n_snps=30,seeds=[0,1,2],settings=real,
        generator='experiments/bench/qgd_bench.ipynb::make_instance',coverage=6,read_len=[2,4],gaps=0,
        solver='Exact integer-scaled local-factor DP, first SNP fixed to zero',
        tie_policy='All minimizers counted; exact histogram for Hamming modulo complement'),indent=2)+'\n')
    lines=['# Exact comparison at 30 SNPs','',
        'Same team generator, 5% per-call error, 10% missing calls, interval lengths 2-4, coverage 6, seeds 0/1/2. '
        'Short local factors permit exact dynamic programming for all five objectives, INCLUDING full-read MEC. '
        'This is not a new pairwise mapping of MEC. Every minimizing phasing is counted; truth is used only for evaluation. '
        'DP metrics were checked against enumeration up to 20 SNPs, and pairwise minima against the shared banded DP.','',
        '| Seed | Reads | Method | Recovery | Mean Hamming | Mean switches | Optimal phasings |',
        '|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["n_reads"]} | {r["variant"]} | {r["recovery"]:.1%} | '
                     f'{r["mean_hamming"]:.3f} | {r["mean_switch_errors"]:.3f} | {r["tied_phasings"]} |')
    lines += ['', '## Observed results', '']
    for name in VARIANTS:
        group=[r for r in rows if r['variant']==name]
        lines.append(f'- {name}: mean recovery across the three seeds '
                     f'{np.mean([r["recovery"] for r in group]):.2%}.')
    lines += ['', 'B1 helps on seed 0 but loses to baseline on seeds 1 and 2. '
              'The earlier 20-SNP case is therefore an example of conditional benefit, '
              'not evidence of universal superiority. Percentages are uniform-tie '
              'recovery probabilities, not measured QAOA shot frequencies.']
    lines += ['', 'Three instances are a descriptive pilot, not a universal ranking. No QAOA, hardware or quantum advantage tested. '
        'Original 12/16/20 results are unchanged. Fields requiring counts of non-optimal cost levels '
        '(e.g. truth rank) are intentionally not reported by this DP. Common-reference mean costs are retained.','',
        'Reproduce: `python experiments/objectives/alexandra_30_objectives.py`.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    # Compatible plot fields allow a combined view without overwriting old outputs.
    from alexandra_larger_objectives import plot
    import alexandra_larger_objectives as larger
    with (ROOT/'results'/'objectives'/'alexandra_larger_results'/'results.csv').open(newline='') as f:
        old=list(csv.DictReader(f))
    for r in old:
        for key in ('n_snps','seed'):r[key]=int(r[key])
        for key in ('recovery','hamming_fraction'):r[key]=float(r[key])
    larger.OUT=OUT
    plot(old+rows)


if __name__=='__main__':
    main()
