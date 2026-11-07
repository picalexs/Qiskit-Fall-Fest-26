# Evidence and Objective — presentation reference

## What was investigated?

**Does changing the influence of sequencing reads improve recovery of a known phasing pair?**

All experiments use five SNPs and classical exact enumeration. They study the quality of an objective, not QAOA optimization performance. A minimizer of a chosen cost is not necessarily biological truth.

## Notebook-first integration

Alexandra Enachescu's objective notebook and all three study scripts are retained. The large generated result archives are intentionally omitted from this integration; the scripts/notebook regenerate the directories listed below. Historical numerical findings in this guide are the author's standalone study, not new combined-benchmark measurements.

[The combined notebook](../bench/all_together.ipynb) embeds `objective_study.py`, both study scripts and this notebook's checks as self-contained cells, alongside Carla's pruning/circuit definitions, Tudor's engine and Alex's hardware methods; the grids run only when `RUN_FULL_STUDIES` is enabled. These files remain the standalone originals. Use repository `requirements.txt` for the combined notebook, or `requirements-objectives.txt` for this classical study alone.

All circuit candidates are rescored on full unpruned baseline pairwise evidence and full-read MEC. MEC remains a reference/evaluation objective, not an invented pairwise circuit cost. The combined smoke run does not rerun the full objective grid.

Three separate studies are available:

| Study | Question | Script / notebook | Results |
|---|---|---|---|
| Main grid | Do A, B1, B2, C and MEC differ under IID errors? | `Alexandra_Objective_Variants.ipynb`, `objective_study.py` | `objective_results/` |
| Reliability sensitivity | Does normalization help when longer reads are less reliable? | `length_error_experiment.py` | `length_error_results/` |
| Fixed length 3 vs 5 | How does available information change with fixed read length? | `fixed_length_experiment.py` | `fixed_length_results/` |

## 1. Notation

- A read is one row; a SNP is one column. `-1` means unobserved.
- $x_i\in\{0,1\}$ is a candidate allele at SNP $i$. The proposed pair is $(x,1-x)$.
- $L_r$ is the number of observed SNPs in read $r$, excluding `-1`.
- $A_{ij}$ and $D_{ij}$ are weighted same/opposite votes for $i<j$.
- $[P]$ is 1 if statement $P$ is true, otherwise 0.

## 2. Number of pairs and weighting formulas

$$m_r=\binom{L_r}{2}=\frac{L_r(L_r-1)}2.$$

Reads with fewer than two observed sites cast no pair votes.

| Variant | Weight per observed pair | Total edge weight per read |
|---|---|---|
| A baseline | $w_r=1$ | $L_r(L_r-1)/2$ |
| B1 per-read | $w_r=2/[L_r(L_r-1)]$ | $1$ |
| B2 per-site | $w_r=1/(L_r-1)$ | $L_r/2$ |
| C deduplicated | Remove exactly identical rows, then use baseline | Depends on retained reads |

B2 gives each site incident weight $(L_r-1)/(L_r-1)=1$. Total edge weight is $L_r/2$ because each edge has two endpoints.

For `011`, the pairs are 1–2 opposite, 1–3 opposite, and 2–3 same. Their weights are 1 under A, 1/3 under B1, and 1/2 under B2. Relations do not change; weights do.

## 3. Graph and pairwise cost

For reads observing both sites:

$$A_{ij}=\sum_r w_r[r_i=r_j],\qquad D_{ij}=\sum_r w_r[r_i\ne r_j].$$

$$C(x)=\sum_{i<j}\left(A_{ij}[x_i\ne x_j]+D_{ij}[x_i=x_j]\right).$$

Equal candidate bits contradict opposite votes; different bits contradict same votes. Use only the upper triangle and preserve fractional costs as floats.

**Fixed-length identity:** if every informative read has the same observed length $L$,

$$C_{B1}(x)=\frac{2}{L(L-1)}C_A(x),\qquad C_{B2}(x)=\frac1{L-1}C_A(x).$$

Positive scaling preserves the full set of minimizers. At $L=3$: B1=A/3 and B2=A/2. At $L=5$: B1=A/10 and B2=A/4. This equivalence concerns exact minima; finite-budget QAOA training need not behave identically without angle/scaling adjustments. Missing calls can break fixed observed length.

## 4. MEC versus pairwise cost

For read $r$, let $d_r(x)$ count mismatches to $x$ at its $L_r$ observed sites.

$$\operatorname{MEC}(x)=\sum_r\min(d_r(x),L_r-d_r(x)).$$

Under baseline pair voting, the same read contributes:

$$C_r(x)=d_r(x)(L_r-d_r(x)).$$

Each mismatch forms a contradicted pair with each matching call. MEC instead counts individual changes to match the closer copy. MEC minimum is not distance to hidden truth. MEC stays classical in this project; it is not directly substituted into the pairwise circuit.

## 5. Ising and the connection to QAOA

$$z_i=1-2x_i,\quad J_{ij}=\frac{D_{ij}-A_{ij}}2,\quad c_0=\sum_{i<j}\frac{A_{ij}+D_{ij}}2.$$

$$E(x)=c_0+\sum_{i<j}J_{ij}z_i z_j=C(x).$$

$z_i z_j=+1$ for equal bits and $-1$ for different bits. Negative $J$ favors equal bits; positive $J$ favors opposite bits.

The starter circuit uses:

$$RZZ(\theta)=e^{-i\theta ZZ/2},\quad\theta=2\gamma J_{ij}/s,\quad s=\max(1,\max|J|).$$

Mixer angle is $2\beta$ in RX. The constant offset affects reported energy but not sampling probabilities. Normalization changes couplings; it does not automatically remove edges or gates. Recompute exact reference and retrain after changing graphs. No quantum advantage is measured here.

## 6. Exact reference and truth metrics

Five SNPs give $2^5=32$ labels, or 16 complementary phasing pairs. Define the exact minimizing set $S=\arg\min_x C(x)$.

$$h(x,t)=\min\left(\sum_i[x_i\ne t_i],\;5-\sum_i[x_i\ne t_i]\right).$$

$$\text{recovery}=\frac1{|S|}\sum_{x\in S}[h(x,t)=0],\qquad
\text{mean Hamming}=\frac1{|S|}\sum_{x\in S}h(x,t).$$

- `truth_is_optimal`: truth belongs to $S$.
- `unique_truth`: $S$ contains only truth and its complement.
- `tied_phasings`: $|S|/2$.
- `truth_rank`: 1 plus the number of complementary pairs with strictly lower cost than truth.

Example: two optimal pairs, one real, give recovery 1/2, truth-is-optimal 1 and unique-truth 0. Do not choose a minimizer using hidden truth. Recovery here assumes uniform tie breaking, not a measured QAOA distribution.

## 7. Replicates and confidence intervals

For replicate-level metric values $v_1,\ldots,v_R$:

$$\bar v=\frac1R\sum_r v_r,\qquad \operatorname{SE}=s_v/\sqrt R,\qquad
\text{approximate 95% mean CI}=\bar v\pm1.96\operatorname{SE}.$$

Recovery bounds are clipped to [0,1]. Main-study binary indicators use Wilson intervals:

$$\text{center}=\frac{\hat p+z^2/(2R)}{1+z^2/R},\quad
\text{half-width}=\frac{z\sqrt{\hat p(1-\hat p)/R+z^2/(4R^2)}}{1+z^2/R},\quad z=1.96.$$

For a paired comparison, compute $\Delta_r=v_r(B)-v_r(A)$ first, then its mean and CI. Positive recovery delta favors B. Wins/ties/losses compare replicate recovery probabilities, including ties. Confidence intervals are approximate; exploratory comparisons are not multiple-comparison-adjusted significance claims.

## 8. Experiment designs and main findings

### Main grid

3 length distributions × 4 error rates × 2 missing rates × 3 read counts × 200 replicates = **14,400 datasets**, each scored by five methods. The same 200 replicate seeds are reused between settings.

Short: 70% L=2, 30% L=3. Mixed: uniform L=2,3,4,5. Long: 30% L=4, 70% L=5. Length distributions apply before missing calls; weights use observed length.

Normalization did not consistently improve recovery. A/B1 minimizing sets differed on 700 datasets, including changes in ties. At mixed lengths, 10% errors, no missing calls and 12 reads: A recovery 94.3%, B1 89.0%, B2 92.5%, C 92.2%, MEC 94.0%. This is one condition, not a pooled average or a universal ranking.

### Length-dependent reliability

Keep five SNPs, 12 reads, uniform L=2,3,4,5 and no missing calls. The observed-call-weighted mean length is:

$$\mu=E[L^2]/E[L]=27/7.$$

$$p(L)=p_0+\lambda\frac{p_0}{\mu-2}(L-\mu),\quad\lambda\in\{0,0.25,0.5,0.75,1\}.$$

Centering at $\mu$ keeps $E[Lp(L)]/E[L]=p_0$: expected per-call error budget is unchanged. At lambda=0 all lengths have equal error probability; at lambda=1 length-2 reads are error-free and length-5 reads have probability about 1.615*p0. This is an artificial sensitivity model, not evidence about actual sequencing reliability.

10 conditions × 1,000 paired replicates = **10,000 datasets**. At p0=15%, IID errors gave A=84.3%, B1=77.8%; strong length-dependent errors gave A=75.3%, B1=88.8%. The latter paired B1 gain is 13.45 percentage points, approximate 95% CI [11.25,15.65]. Benefit depends on the reliability model.

### Fixed length 3 versus 5

Two controls, four error rates, two lengths, 1,000 replicates per setting; no missing calls:

- Fixed read count: 12 reads at L=3 give 36 calls; L=5 gives 60.
- Fixed observed-call budget: 20 reads at L=3 and 12 reads at L=5 both give 60.

All A/B1/B2 minimizers are verified identical per dataset. This comparison measures available information, not a normalization advantage. Equal call count does not equalize linkage: L=5 spans the entire toy chromosome. Repeated settings across the two protocols are deliberately identical, not additional independent evidence. Numerical results are in `fixed_length_results/report.md`.

At 10% per-call errors, 12 reads gave recovery of about **85.0% at L=3** versus **99.7% at L=5**. With 60 observed calls for both lengths, recovery was **95.6% at L=3** (20 reads) versus **99.7% at L=5** (12 reads). These results describe this five-site contiguous-read model. Perfect empirical recovery in a finite sample is not proof of perfect population recovery; a zero-width normal interval at the boundary is an approximation limitation.

## 9. What to say on a slide

> We separated objective quality from search quality using exact enumeration. Normalization is not universally beneficial: it can suppress useful evidence under IID errors, but can improve recovery when long reads are systematically less reliable. With equal observed read lengths, A/B1/B2 have mathematically identical minimizers. Choice of weights should reflect the observation model, not an assumed universal benefit.

Use one paired-comparison plot and one explicit disagreement case. Report conditions and uncertainty alongside percentages. Do not equate model minimum with hidden truth.

## 10. Reproduce and reuse

From project root, using the local environment:

```powershell
.\.venv-objectives\Scripts\python.exe docs/length_error_experiment.py --repeats 1000
.\.venv-objectives\Scripts\python.exe docs/fixed_length_experiment.py --repeats 1000
```

Run the main study through its notebook. Each results folder keeps its own manifest, metrics and saved datasets. Scripts use separate output folders; original results are not overwritten by the extensions.

For teammates, with `docs` on the Python import path:

```python
from objective_study import build_graph, per_read_weight
A, D, J, c0 = build_graph(reads, per_read_weight)
```

Raw metrics are for verification; summaries and plots are for presentation. All results concern synthetic five-SNP instances. Supplied baseline logic originates in the team starter; extensions and documentation were prepared with AI assistance.
