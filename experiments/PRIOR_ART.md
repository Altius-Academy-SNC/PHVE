# Prior-art check for the results R1–R4

Recorded before the proofs were written, as required by the work plan.
Searches were run in August 2026. Each entry states what the existing work
already establishes and what, if anything, is left to contribute.

---

## R1 — anisotropy-corrected bound and per-axis orders

**What already exists.**

- **Hamilton & Rau-Chaplin, "Compact Hilbert indices: space-filling curves
  for domains with unequal side lengths", *Inf. Process. Lett.* 105 (2008)
  155–163**, and the companion technical report *Compact Hilbert indices for
  multi-dimensional data* (Dalhousie CS). This is the reference construction
  for a Hilbert index on a grid
  $\prod_i \{0,\dots,2^{m_i}-1\}$ with **different per-axis resolutions
  $m_i$**. It removes exactly the padding waste that a cubic order incurs on
  a non-cubic box.
  → **The construction of a per-axis-resolution Hilbert order is not new.**
  Any claim of novelty for it would be wrong.

- The literature on **SFC partitioning** (Zoltan's HSFC geometric
  partitioner; p4est; the tetrahedral SFC of Burstedde–Holke) all works on
  non-cubic domains and is aware of the resolution question.

**What is left.**

1. Hamilton & Rau-Chaplin give the *index*; they do not give a **bandwidth
   or index-gap bound for a finite-element matrix**, and in particular no
   dependence on the box aspect ratio $\kappa$.
2. They do not give a **rule for choosing the $m_i$** from the geometry of
   the reference box under a fixed code-length budget.

So R1 is stated in this paper as: (i) the correction of the published bound
by the factor $\kappa$ (which the v4 statement silently set to 1), and
(ii) the selection rule $2^{p_i}\propto L_i$ with its integer-rounding loss.
The underlying encoding is credited to Hamilton & Rau-Chaplin.

**Not verified:** whether the restricted-cube order used in our
implementation coincides *as an index* with their compact Hilbert index. We
only use, and only prove, that the two induce the same total order is not
claimed; what we prove is the rank-compression lemma, which is all the
bandwidth argument needs. See `UNVERIFIED.md`.

---

## R2 — per-time-step cost including renumbering

**What already exists.**

- The claim that **SFC-based (re)partitioning is cheaper than graph-based
  partitioning, and therefore preferable when the mesh is re-adapted
  frequently**, is standard and well documented: Zoltan's user guide states
  it for HSFC vs. ParMETIS; p4est is built on it; Sasidharan & Snir,
  *Space-filling curves for partitioning adaptively refined meshes*, make it
  explicit; the survey "Revisiting the space-filling curves for storage,
  reordering and partitioning mesh based data" reports the same trade-off
  (SFC faster, graph partitioners give lower communication volume).
- **Cache/preconditioner benefit of SFC vertex- and element-reordering** in
  finite elements: Ozturan et al., "Improved cache utilization and
  preconditioner efficiency through use of a space-filling curve mesh
  element- and vertex-reordering technique", *Engineering with Computers*
  31 (2015).

**What is left.**

The qualitative statement is prior art. What we contribute is a **closed
operation count for one semi-implicit step**, with the renumbering cost
carried explicitly and amortised over the remeshing period $m$, and the
resulting **crossover condition** in $m$. The numbers are measured, not
asserted; and where the measurement contradicts the expected sign we say so.

---

## R3 — the constant $C_3$

**What already exists.**

- **Moon, Jagadish, Faloutsos & Saltz**, *IEEE TKDE* 13 (2001) 124–141, is
  the standard reference for the constant $6$ in dimension two. Their
  clustering analysis concerns the **inverse (Hölder) direction**.
- **Gotsman & Lindenbaum**, *IEEE TIP* 5 (1996) 794–797, prove lower bounds
  on the metric distortion of *any* discrete space-filling curve.
- **Niedermeier, Reinhardt & Sanders**, *Discrete Appl. Math.* 117 (2002),
  give asymptotic optimality of Hilbert among mesh indexings.
- **Hamilton & Rau-Chaplin (2008)** state finiteness in higher dimension
  without an explicit value.

**What is left.**

~~No explicit numerical value of the three-dimensional constant was found in
the literature. We compute it exhaustively at small orders and extrapolate.~~

### CORRECTION (second search pass, August 2026) — this entry was wrong

A second search, run specifically against the computational-geometry
literature rather than the database/indexing literature, found that the
quantity we call $C'_3$ is studied there under the name **worst-case $L_2$
dilation, $WL_2$**, defined exactly as
$\max \|H^{-1}(i)-H^{-1}(j)\|_2^{\,d} / |i-j|$ — the same quantity, and the
$d=2$ value $6$ of Moon et al. is its two-dimensional case.

Explicit three-dimensional values are published:

- **Gotsman & Lindenbaum (1996)** prove $WL_2 \le 48\sqrt6 \approx 117.6$ for
  a three-dimensional Hilbert curve, and report $WL_2 \lesssim 23$ from
  computer simulation.
- **Haverkort**, *An inventory of three-dimensional Hilbert space-filling
  curves* (arXiv:1109.2323), and Bos & Haverkort, *Hyperorthogonal
  well-folded Hilbert curves* (arXiv:1508.02517), identify the curve within
  the scope of that simulation as `A26.0010 1011.1011 0011` and give
  $WL_2 = 22.9$.
- Haverkort, *How many three-dimensional Hilbert curves are there?*
  (arXiv:1610.00155), shows there are $10\,694\,807$ structurally distinct
  three-dimensional Hilbert curves, and tabulates locality measures for
  named families. Dilation is computed there by **exact automata-based
  methods**, not by extrapolation.

**Consequences, all of them against us.**

1. The sentence "no explicit numerical value of the three-dimensional
   constant was found in the literature" is **false** and must be removed
   from the paper wherever it appears.
2. Our $C'_3 \approx 29.5$ is not "the" three-dimensional constant. It is
   the value for **one particular variant** — the Skilling (2004)
   construction our kernel implements. It is *worse* than the best published
   3D Hilbert variant ($22.9$). Any statement must name the variant.
3. A geometric extrapolation from $p = 4,5,6$ will not survive refereeing in
   a community that computes this quantity exactly. Either the value is
   obtained by their automaton method, or it is not claimed.

What may remain, and only if verified: an explicit $WL_2$ for the Skilling
variant specifically — the variant almost all software uses — placed inside
Haverkort's taxonomy, with the observation that the widely-implemented
variant is not the locality-optimal one. That is a footnote, not a result.

### Follow-up (exp13, exp14): the footnote is now supported

`exp13` establishes that Skilling's 3D curve and H&RC's are genuinely
distinct — no match among the 96 cube symmetries (with or without traversal
reversal) for $p\ge2$ — and that H&RC's is better at every order tested, in
both the forward and inverse measures. So $29.5$ and $22.9$ were never
values of the same object.

`exp14` then uses the published 3D value as a *calibration* rather than a
competitor. The same extrapolation code, run on the H&RC series, returns
$23.778$ against the published $22.9$: a $3.83\%$ overshoot. Combined with
$0.02\%$ in $d=2$, this is the first honest error bar the procedure has
had in the dimension where it is used.

Revised claim, which we believe is defensible:

> The Skilling (2004) construction, which is what most software
> implements, has $WL_2 \approx 29.5$ in $d=3$ (extrapolated, $\sim4\%$
> method error), against $22.9$ for the best published three-dimensional
> Hilbert curve — a ratio of $1.24$. Both figures are produced by the same
> code on the same grids, so the ratio is more robust than either limit.

This is a remark about a widely-used implementation, not a new constant of
the Hilbert curve, and must be phrased that way.

We also found that the *forward* constant quoted in v4 of this paper does
not exist (it diverges); this is a correction to our own earlier statement,
not a claim against the cited works, which concern the inverse direction.
Note that the qualitative fact — no space-filling curve has bounded
*forward* dilation — is itself standard in the same literature; what is
ours is only the exact closed form of the maxima, and only its lower bound
is proved.

---

## R4 — regularity transfer along the Hilbert parametrisation

**What already exists.**

- The $1/d$-Hölder continuity of the continuous Hilbert parametrisation is
  classical (Sagan, *Space-Filling Curves*, Springer 1994, Ch. 2).
- That composing an $\alpha$-Hölder function with a $1/d$-Hölder
  parametrisation yields an $(\alpha/d)$-Hölder function is the elementary
  composition rule for Hölder classes.
- Hilbert-scan DPCM/lossless compression of medical volumes: Chen & Mudge,
  *Proc. IEEE EMBS* (2005).

**What is left.**

The composition rule is elementary; what is not in the literature we found
is the **criterion that predicts the sign of the DPCM gain** from a
quantity computable in $O(N)$ before compression (the ratio of the
raster wrap-around residual energy to the bulk residual energy), together
with its validation on volumes where the gain is present and on volumes
where it is absent. We claim only the criterion and its validation.

---

## Consequence for the claims of the paper

| Result | Construction | Analysis |
|---|---|---|
| R1 | Hamilton & Rau-Chaplin 2008 | new: $\kappa$-correction + selection rule |
| R2 | folklore (Zoltan, p4est) | new: explicit per-step count + crossover |
| R3 | — | new: numerical value in $d=3$; correction of the forward statement |
| R4 | classical composition rule | new: predictive DPCM criterion |
