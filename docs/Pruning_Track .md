# Pruning Track

**Question:** Does dropping weak or suspicious edges improve haplotype phasing,
or does it destroy information?

**Short answer:** It does not help. The clear, repeatable effect is harm — once
pruning severs a coupling and disconnects the graph, phase relationships become
unrecoverable. Judged honestly (over *all* datasets, not just the ones that
survived connected), weak-edge pruning is strongly negative at every error rate.

This investigation reuses the lab's exact phasing model (signed graph → pairwise
cost) and adds several pruning rules, active-graph connected-component analysis,
and a paired full-vs-pruned sweep averaged over several seeds.

---

## Where it lives

The entire investigation is **Section 7 of
`docs/Quantum_Genome_Detective_Lab.ipynb`** (self-contained, CPU-only, runs top
to bottom). There is no separate package to install or run — the notebook defines
`edge_table`, `components`, `n_blocks`, `prune` (including `mstforest`),
`evaluate`, and the full seed-averaged sweep in its own cells. This file is a
standalone summary of that section; the notebook is the source of truth.

---

## Experiment settings

| Setting | Value |
|---|---|
| SNPs per instance (n) | 5 |
| Error rates swept | 2%, 5%, 10%, 15%, 20% |
| Strata (read length, coverage) | (3, 3.0), (2, 4.0), (4, 2.5), (5, 8.0), (3, 5.0) |
| Seeds averaged | 1, 2, 3, 4, 5 |
| Datasets per (error × stratum) cell | 20 |

**Pruning rules** (one at a time):

| Rule | Drops edges where... | Thresholds swept |
|---|---|---|
| `net` | `|A − D| < t` (weak net preference) | t = 0, 1, 2, 3 |
| `support` | `A + D < m` (few total votes) | m = 1, 2, 3, 4 |
| `tie` | `A == D` (no information, constant in cost) | — |
| `topk` | keep only the k strongest edges | k = 0 … 10 |
| `mstforest` | keep a max-weight spanning forest on `|A − D|` (connectivity-preserving) | — |
| `bridge` | edge whose removal splits a component | — |

**Controls:** `random` (drop the same number of edges at random) and `strong`
(drop the strongest edges). They confirm that *which* edges you drop matters.

**Scoring discipline (important):**
- Connectivity is computed on the **active (net ≠ 0) coupling graph**. A tie edge
  (`A == D`) has Ising weight `(D − A)/2 = 0`, so it neither connects nor
  disconnects. Counting ties would make `net` t = 0 and t = 1 look different
  (t = 1 merely drops ties) even though the minima are identical.
- Every pruned candidate is rescored on the **full** model — we record whether a
  pruned minimum is still a minimum of the full model (`pruned_min_is_full_min`),
  never judging it by its own smaller cost.
- Tied minima are scored by the honest random-pick accuracy `1/|set|`, never as
  free wins.
- The **headline** gain is the *unconditioned* paired difference (pruned − full)
  over every dataset; the connected-only number is reported only as a diagnostic,
  because conditioning on "stayed connected" is selection bias.

### The motivating case

Pruning the weak edge 3-4 (one vote) at net threshold t = 2 removes the only
bridge, splitting the graph into {1,2,3} and {4,5}. The model can no longer say
whether SNPs 3 and 4 share a copy.

---

## Conclusions (averaged over seeds 1–5)

### 1. Where pruning hurts — robust

Raising the `net` threshold cuts bridges, the graph splits, and accuracy
collapses exactly where the component count climbs:

| t | accuracy | mean components k | fraction disconnected | pruned min is full min |
|---|----------|-------------------|-----------------------|------------------------|
| 0 | 0.696 | 1.28 | 0.24 | 1.00 |
| 1 | 0.696 | 1.28 | 0.24 | 1.00 |
| 2 | **0.547** | **2.01** | 0.64 | 0.52 |
| 3 | 0.359 | 2.98 | 0.82 | 0.22 |

Note that **t = 0 and t = 1 are identical**: t = 1 drops only tie edges, which
carry no coupling, so neither the component count nor the accuracy moves. Harm
begins at **t = 2**, exactly where `k` jumps and where the pruned minimum stops
agreeing with the full model (full-min preservation falls from 1.00 to 0.52).
Per-seed spread is small (±0.007–0.013), so this is a property of the model, not
one unlucky draw.

### 2. Where pruning "helps" — it doesn't, once judged honestly

`support` pruning targets the thin edges a single wrong read creates. The
**unconditioned** seed-averaged gain (over every dataset) is strongly negative at
every error rate — it clears its own std, so the harm is real, not noise:

| error rate | support gain, unconditioned (headline) | connected-only (diagnostic) |
|---|---|---|
| 2% | −0.567 ± 0.017 | +0.000 ± 0.000 |
| 5% | −0.502 ± 0.038 | +0.000 ± 0.000 |
| 10% | −0.363 ± 0.015 | −0.025 ± 0.056 |
| 15% | −0.261 ± 0.023 | −0.108 ± 0.328 |
| 20% | −0.186 ± 0.028 | −0.026 ± 0.066 |

The connected-only column (near zero) is exactly the selection bias this track
warns against: it only describes the minority of cases pruning failed to
disconnect, and it hides the harm. The `net` rule never helps either: a single
error rarely flips an edge's sign, so it only disconnects.

### 3. A connectivity-preserving alternative — `mstforest`

If the goal is a smaller circuit rather than better accuracy, the right tool is a
maximum-weight spanning forest on `|A − D|`, not a blunt threshold. It keeps at
most n − 1 edges yet preserves exactly the connectivity of the active graph:

| rule | mean edges kept | mean components k | fraction disconnected | accuracy | pruned min is full min |
|---|---|---|---|---|---|
| `mstforest` | 3.72 | 1.28 | 0.24 | 0.688 | 0.98 |
| `topk` (k = 4) | 3.95 | 1.81 | 0.78 | 0.488 | 0.49 |

At essentially the same edge budget, `mstforest` keeps the graph as connected as
the full model (disconnected fraction 0.24, same as `net` t = 0), holds accuracy
near baseline (0.688 vs the 0.696 ceiling), and keeps the pruned minimum aligned
with the full model 98% of the time — whereas `topk` at the same size disconnects
78% of instances and halves accuracy. Connectivity, not edge count, is what
matters.

### 4. The circuit-size trade-off (hand-off to Person 4)

Keeping only the k strongest edges (`topk`) is a direct circuit-size budget —
each kept edge is one two-qubit gate. Accuracy climbs as edges are added until the
graph connects (~4–5 edges here), then flattens: beyond that, extra edges cost
gates without buying accuracy. `mstforest` reaches the connected regime with the
fewest edges.

### 5. Seed robustness

| quantity | mean | std across seeds | range |
|---|---|---|---|
| baseline accuracy (full model) | 0.696 | 0.013 | [0.682, 0.714] |
| accuracy after net t = 2 | 0.547 | 0.007 | [0.538, 0.557] |

Both stable to ~0.01, far below the ~0.15 drop pruning causes by t = 2. The harm
holds regardless of seed.

### 6. Ambiguity that isn't disconnection

Even among **connected** pruned graphs, ~6.8% still had more than one tied
minimum — ambiguity from contradictory votes or symmetry, not a severed graph.
Components and tied minima are different things: a connected graph can still tie;
a disconnected one always does (at least `2^(k−1)` tied orientations).

---

## Limitations

n = 5 toy instances with a fixed set of strata and short synthetic reads; the
pairwise cost is not biological truth; tied-set size is reported, and accuracy
within a tie is `1/|set|`, never a free win. Longer reads, larger n, or real
sequencing-error profiles could behave differently and would need their own sweep.
