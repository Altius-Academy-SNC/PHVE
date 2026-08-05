# What is not verified

Maintained alongside the paper. Each entry says what is missing and what it
would take to close it. Nothing here is a hidden caveat: every item is also
stated in the paper (§ "What Remains Unverified").

---

## U1 — The timing comparison is implementation-bound

The incomplete factorisation is more than 90 % of every total in
`exp04_comparative.py`. It is SuperLU's threshold ILU, reached through
`scipy.sparse.linalg.spilu`, with `permc_spec="NATURAL"` so that it does
not reorder the matrix behind our back.

For a symmetric positive definite system that is the wrong factorisation:
an incomplete Cholesky with a fixed pattern (IC(0)) or a thresholded
variant (ICT) is what a production code would use. Its fill is fixed by the
pattern, so the 36 % fill penalty we measured for PHVE would become a
*quality* penalty instead, showing up in the iteration count rather than in
the factorisation time.

**Consequence.** The conclusion "no crossover in m" (paper,
Cor. "No crossover on the model problem") is sound for the solver we ran
and is *not* established for a fixed-pattern IC. The fill ratio and the
iteration counts are implementation-independent; the wall times are not.

### Update: IC(0) implemented; the negative result changes its reason

`phve/ic0.c` implements the factorisation and the two triangular solves
with the pattern fixed to `tril(A)`; verified `max |LLᵀ − A| = 7e-12` on
the pattern. `exp04_comparative.py` was rerun against it, and
`exp10_equivariant_precond.py` sweeps it over N.

What changed:

- **fill is now identical for all three orderings** (×0.53 of nnz(A)), by
  construction. The entire basis of the v5 negative result — a 36 % denser
  factorisation for PHVE — was an artefact of SuperLU's *threshold* ILU
  choosing its own fill. It is gone.
- the negative result survives, for a different and more interesting
  reason: not fill but the **quality** of the factorisation. IC(0)-PCG
  iteration counts, PHVE/RCM: 1.15, 1.56, 1.14, 1.74, 2.03 across
  N = 3 864 … 63 265. Hilbert's long-range seam couplings degrade IC(0),
  and RCM wins it at every size tested.
- the contrast with a *permutation-equivariant* preconditioner is now the
  paper's positive result (see U-note in `DIRECTION.md` §2.5): with
  Chebyshev the iteration counts are **exactly** equal across orderings.

**Still open:** ICT (thresholded incomplete Cholesky) was not implemented.
IC(0) alone was enough to separate fill from quality, which was the point;
ICT would say whether the quality penalty persists when the factoriser is
allowed to choose fill under a fixed budget.

---

## U2 — Hardware cache counters were unavailable

`kernel.perf_event_paranoid = 4` on the machine used, and no privileged
access. All cache figures come from `phve/cachesim.c`, an exact
set-associative LRU simulation of the SpMV access trace (32 KiB, 8-way,
64-byte lines).

The simulation is deterministic and reproducible on any machine, which is
an advantage for a paper, but it is a model. It ignores hardware
prefetching, the L2/L3 hierarchy, and the sequential reads of the CSR
`indptr`/`indices` arrays (which are identical for all orderings and were
deliberately excluded).

**To close it:** rerun on a machine with `perf_event_paranoid <= 2` and
compare `perf stat -e L1-dcache-load-misses` with the simulation.

---

## U3 — Anisotropic domain — CLOSED

*Closed by `exp16_anisotropic_domain.py`.* The domain is the MNI152 brain
with its **affine scaled along one axis**: same non-convex mask, same data,
same mesh generator and operator, only the aspect ratio moves. A synthetic
ellipsoid would have discarded the non-convexity that forces the mesh to be
unstructured in the first place.

R1, now measured against a real operator (N ≈ 39 000, seed 1, budget
Σpᵢ = 30):

| κ | R1 orders | mean gap cubic → aniso | factor | max gap cubic → aniso |
|---|---|---|---|---|
| 1.24 | (10,10,10) | 817.5 → 817.5 | ×1.00 | 38 595 → 38 595 |
| 1.87 | (10,10,10) | 740.2 → 740.2 | ×1.00 | 38 468 → 38 468 |
| 3.74 | (11,10,9) | 725.0 → 551.8 | **×1.31** | 38 344 → 23 265 |
| 7.48 | (12,9,9) | 797.6 → 522.7 | **×1.53** | 38 132 → 23 687 |

**R1 works, and the improvement grows with κ** — consistent with the
×1.16/×1.41/×2.31 of `exp05` on point clouds.

**A limitation of the rule, not previously stated.** Integer rounding makes
the rule *inert* below κ ≈ 2.5: at κ = 1.24 and 1.87 it returns the cubic
orders, so `phve_aniso` is bit-for-bit `phve_cubic`. R1 has no effect on
the brain domain the rest of the paper uses.

**An effect that contradicts our own earlier statement, and needs care.**
`exp05` reported that the maximum gap "barely moves, as the seam theorem
requires". Here it drops by ~40 % (×1.65 and ×1.61). The seam theorem is
not contradicted — the maximum is still Θ(N), 59 % of N instead of 98 % —
but the constant moves a lot.

The likely cause is the *restricted-cube* construction: with orders
(12,9,9) the points occupy a 4096 × 512 × 512 slab of the 4096³ cube, and
the cube's principal seam need not lie inside that slab. Since `exp12`
established that this construction is **not** the compact Hilbert index in
d = 3, this reduction may be an artefact of the embedding rather than a
property of anisotropic Hilbert ordering. **Not to be claimed until checked
against the true compact index.**

**Does it reach the solver?** Jacobi spread across all four orderings: 1,
2, 0, 1 iterations — equivariance holds on anisotropic domains too. IC(0):
at κ = 3.74, `phve_aniso` takes 24 iterations against RCM's 30 and cubic's
45 — the first configuration where a PHVE order beats RCM. At κ = 7.48 it
does not (33 against 26). One seed, one size: **this is an observation, not
a result**, and needs the full sweep before it can be stated.

---

## U4 — No full-resolution clinical volume

DPCM uses MNI152 at 2 mm and 1 mm plus synthetic fractional Brownian
volumes. IXI and ACDC, announced as future work in v4, are still untested.
The v4 claim that "validation on individual clinical scans is reported in
[Chen & Mudge 2005] with a 5–15 % gain consistent with our theorem" was
taken from the literature and has **not** been reproduced here.

---

## U5 — The compact Hilbert index — CLOSED, and the answer is negative

*Closed by `exp12_compact_hilbert.py`, which implements Hamilton &
Rau-Chaplin's Algorithms 1 and 2 as a reference. Both self-tests pass: the
compact index with all mⱼ = M reproduces their standard index exactly
(d ∈ {2,3}, M ≤ 4), and is a bijection onto {0,…,2^{Σmⱼ}−1} in all nine
anisotropic cases.*

The question was split in three so that a failure could be localised.

**(Q1) Which variant do we implement?** In **d = 2** the Skilling index of
`phve/hilbert.py` is *identical* to H&RC's standard index (M ≤ 4, all
points). In **d = 3** it is **not** — different indices *and* different
total orders, already at p = 1. They are two different space-filling
curves. This is not an error in either: Haverkort has shown there are
10 694 807 structurally distinct three-dimensional Hilbert curves.

**(Q2) Is compactness order-preserving?** **Yes**, in every case tested
(nine per-axis bit vectors, d ∈ {2,3}): within H&RC's own variant, the
restricted-cube order and the compact index induce the *same total order*.
This is the content the rank-compression lemma asserts, and it survives.

**(Q3) The claim as written.** `phve_order_aniso` agrees with the compact
Hilbert index in d = 2 and **disagrees in d = 3** — the dimension the paper
is about.

**Consequences.**

1. The docstring of `phve/hilbert.py` asserted order-equivalence as a fact.
   It was **false in d = 3** and has been corrected in place.
2. The bandwidth argument is unaffected: it uses only the rank-compression
   lemma, which Q2 confirms.
3. Credit must be stated more carefully. What is Hamilton & Rau-Chaplin's
   is the idea and construction of a per-axis-resolution Hilbert index. Our
   anisotropic order is a *different* order in d = 3, obtained by a
   different (restricted-cube) construction on a different variant.
4. This resolves the apparent conflict in U6: our WL₂ ≈ 29.5 and the
   published 22.9 are values for **different curves**, so they were never
   comparable. Every locality constant measured in this work belongs to the
   Skilling variant and must be labelled as such.

**Still open, and cheap:** whether the *cubic* Skilling order and the
*cubic* H&RC order differ by a symmetry of the cube (which would make the
gap statistics identical) or genuinely differ in locality. All locality
claims in d = 3 depend on this and it is not yet tested.

---

## U6 — Closed forms and extrapolations are not proofs

- `A_2(p) = (5·4^p − 2)/6` and `A_3(p) = (13·8^p − 6)/14`.
  **Verification extended** by `exp17_seam_closed_forms.py`, which
  enumerates *all* adjacent pairs (there are only d(n−1)n^{d−1} of them, so
  the cost is linear in the grid, not quadratic as for the inverse
  constant):

  | | was | now | grid at the new limit |
  |---|---|---|---|
  | d = 2 | exhaustive to p = 6 | **exhaustive to p = 12** | 16 777 216 points |
  | d = 3 | exhaustive to p = 4 | **exhaustive to p = 7** | 2 097 152 points |

  Both hold at every order, with A(p)/n^d converging to 5/6 = 0.833333 and
  13/14 = 0.928571 to six digits. They remain **conjectures** beyond those
  orders; what is *proved* is the lower bound (1 − 2^{1−d})·n^d (paper,
  Thm. "Seam obstruction"), and the matching upper bound is still open.

  Incidentally the same run gives the H&RC variant's maxima in d = 3 —
  7, 59, 471, 3767 — which depart from the Skilling closed form from p = 3
  onwards, an independent confirmation that the two curves differ.
- `C'_3` (= the worst-case L₂ dilation WL₂ of the *Skilling* variant) is a
  geometric extrapolation from p = 4, 5, 6. **The uncertainty estimate has
  been redone properly** (`exp14_wl2_extrapolation.py`).

  Previously it was calibrated only in d = 2, where the answer 6 is known.
  That is a weak calibration: it says nothing about how the procedure
  behaves in the dimension where it is actually used. Since `exp13`
  established that Skilling's and H&RC's 3D curves are different, and
  Haverkort publishes WL₂ = 22.9 for the latter, the same procedure could
  be run on a *three-dimensional* curve whose answer is known:

  | series | orders used | extrapolated | published | error |
  |---|---|---|---|---|
  | Skilling, d = 2 | 6,7,8 | 5.999 | 6.0 | 0.02 % |
  | H&RC, d = 2 | 6,7,8 | 5.999 | 6.0 | 0.02 % |
  | **H&RC, d = 3** | **4,5,6** | **23.778** | **22.9** | **3.83 %** |
  | Skilling, d = 3 | 4,5,6 | 29.480 | — | — |

  So in three dimensions the procedure **overshoots by about 4 %** on the
  one case where truth is known. The honest statement is therefore
  `WL₂(Skilling, d=3) = 29.5` with a **4 % method error**, i.e. roughly
  28–30, and likely on the high side — not `± 0.5`. It remains an
  extrapolation and no closed form is proposed.

  What this does buy: the ratio WL₂(Skilling)/WL₂(H&RC) = **1.24**. The
  variant implemented in most software has about 24 % worse worst-case
  locality than the best published three-dimensional Hilbert curve. Both
  numbers come from the same code on the same grids, so the ratio is far
  more robust than either limit.

  Caveat on the sweep: at p = 6 the argmax index gap sits at 0.0636 N for
  H&RC against 0.0382 N for Skilling, both inside the swept 0.08 N, but the
  H&RC margin is thinner and its p = 6 value is more likely to be a slight
  underestimate.
- The p = 6 value in d = 3 comes from a gap sweep restricted to
  g ≤ 0.08 N. The argmax gap is 0.0382 N at every order p ≥ 3, comfortably
  inside, but this is evidence, not a certificate.

---

## U7 — Numerical-analysis properties of the scheme — now measured

Established: unique solvability at each step (Lax–Milgram / SPD),
unconditional L² stability, exact mass conservation.

**Not** established: a discrete maximum principle (which for P1 needs a
non-obtuse mesh condition that our adaptive Delaunay meshes do not
enforce), and convergence to a solution of the continuous quasi-linear
problem as h, τ → 0.

### What `exp18_scheme_properties.py` measures

The two gaps stay open — neither can be closed by computation — but they
are no longer only asserted.

**The non-obtuse condition fails, steadily and by a lot.** The fraction of
*positive* off-diagonal stiffness entries is **31.96 %–33.78 %** across
20 steps and two seeds, with the positive entries carrying about a quarter
of the weight of the negative ones (0.21–0.27). Roughly a third of the
couplings have the sign that breaks the sufficient condition, and refining
does not help: the fraction drifts down only from 33.8 % to 32.0 % as N
goes 12 000 → 58 685.

**But it rarely bites.** Checking at every step whether the update creates
a new extremum, `min(uⁿ) ≤ uⁿ⁺¹ ≤ max(uⁿ)`: violated on **1 of 10 steps**
for each seed, with worst overshoot 1.29×10⁻³ (seed 1) and 3.99×10⁻³
(seed 2) relative to the range of u. Undershoot never occurred.

So the sufficient condition fails on a third of the mesh while the
conclusion holds on 90 % of the steps and is off by a few parts per
thousand when it does not. That is worth stating precisely, because
"no discrete maximum principle" without a number invites the reader to
imagine something much worse.

**Self-convergence is inconclusive, as expected.** Solving on meshes of
4 000–32 000 vertices and comparing against the finest by nearest-neighbour
transfer gives an observed order of **0.60** in the median edge length.
This does **not** establish convergence and must not be quoted as
supporting it: the nearest-neighbour transfer is itself first-order at
best, so it contaminates the estimate and caps it, and the reference mesh
is only a factor 2 in N (1.26 in h) finer than the finest test mesh. The
number is recorded for completeness and as evidence that nothing is
grossly wrong, not as a convergence study.

This limitation remains deliberate. The scheme is used as a *generator of
realistic sparse SPD systems on a changing unstructured mesh*, which is all
the renumbering study needs. It should not be cited as a validated solver
for
Perona–Malik diffusion.

---

## U8 — Constants in the mean-gap theorem — CLOSED, and (A3) is broken

*Closed by `exp11_meangap_constants.py`.* The exponent was already
confirmed by `exp03` (0.643 measured against 2/3, R² = 0.9998). The
prefactor was not, and it does not survive measurement.

**Hypothesis (A3) cannot hold with an N-independent constant.** As
published it requires *every* dyadic cell of *every* level ℓ to contain at
most c₁N2^{−dℓ} vertices. As soon as 2^{dℓ} > N a non-empty cell still
holds at least one vertex while the bound is below 1, so necessarily
c₁ ≥ 2^{dℓ}/N. Measured, c₁ is *exactly* pinned there:

| N | c₁ measured | 2^{dp}/N (p = 10) | ratio |
|---|---|---|---|
| 4 000 | 268 435.5 | 268 435.5 | 1.00 |
| 8 000 | 134 217.7 | 134 217.7 | 1.00 |
| 16 000 | 67 108.9 | 67 108.9 | 1.00 |
| 32 000 | 67 108.9 | 33 554.4 | 2.00 |
| 64 000 | 33 554.4 | 16 777.2 | 2.00 |

The ratio is the maximum occupancy at the finest level, i.e. 1 or 2. The
constant carries no information about the mesh at all. Consequently the
bound is vacuous: **bound/measured runs from 5.0×10⁷ to 1.9×10⁸.**

**The repair.** Restrict (A3) to levels ℓ ≤ ℓ* with 2^{dℓ*} ≤ N, and
truncate the sum in the proof at ℓ*. This is legal: two points sharing a
level-ℓ cell with ℓ > ℓ* also share their level-ℓ* cell, so the level-ℓ*
estimate already covers them, and the geometric sum is unchanged in form.
The restricted constant is then a genuine mesh quantity, and is bounded:

| N | 4 000 | 8 000 | 16 000 | 32 000 | 64 000 |
|---|---|---|---|---|---|
| ℓ* | 3 | 4 | 4 | 4 | 5 |
| c₁* | 3.84 | 6.14 | 5.38 | 4.10 | 7.17 |

**What is still not good.** With c₁* the bound improves by a factor
1.7×10⁴, but bound/measured is still **2 773 to 10 717**. The theorem has
the right exponent and a constant three to four orders of magnitude
pessimistic. The main culprits are visible in the statement: h is the
*maximum* edge length (44–66 mm on a domain ~150 mm across, i.e. rare long
Delaunay edges near the non-convex mask boundary dominate it) and D is the
*maximum* degree (31–38, against a mean degree near 15).

The other two constants are well behaved: c₂ ∈ [2.46, 2.86] and
D ∈ [31, 38] over the whole sequence.

**Consequence for the paper.** (A3) must be restated with the level
restriction — as written it is unsatisfiable. And the theorem must be
presented for what it is: a statement about the exponent. Any sentence
suggesting the bound is quantitatively useful on these meshes has to go.

---

## U9 — Nothing here is about parallel partitioning

The strongest argument in the literature for space-filling curves is cheap
*incremental repartitioning* across processes (Zoltan HSFC, p4est,
Sasidharan & Snir). We tested none of it.

Our negative result concerns **sequential renumbering for a sequential
solve**. It must not be read as a statement about partitioning, and the
paper says so.

---

## U10 — Results carried over from v4 — CLOSED

*Closed by `exp15_v4_carryover.py`.*

**Bijectivity on MNI152 — the test had never been run.** Reading
`simulations/bijectivity_mni152.py` shows it checks bijectivity on 512
points at p = 3, then encodes seven landmarks and draws a figure. The
number 235 375 in its output is the size of the brain mask, not the result
of a collision test. The claim "0 collisions at p = 8" therefore had no
computation behind it.

Run properly now, on the same mask (`nilearn load_mni152_brain_mask`,
resolution 2 mm, 235 375 voxels, all inside the CR box):

| p | cell (mm) | colliding voxels | rate | max multiplicity |
|---|---|---|---|---|
| 6 | 4.69 × 3.91 × 3.91 | 207 171 | 88.02 % | 12 |
| 7 | 2.34 × 1.95 × 1.95 | 33 183 | 14.10 % | 2 |
| 8 | 1.17 × 0.98 × 0.98 | **0** | 0.00 % | 1 |
| 9, 10 | — | 0 | 0.00 % | 1 |

**The v4 claim is correct**, and p = 8 is exactly the threshold, matching
the elementary criterion (cell smaller than the 2 mm voxel spacing along
every axis). A first attempt at this test used `data > 0` instead of the
brain mask and did not separate voxels clipped at the box boundary; it
produced a spurious 0.96 % residual. That was our protocol error, recorded
here so the corrected run is not mistaken for a retraction.

**"Inter-patient stability" — the number reproduces, the interpretation
does not.** `simulations/refuted/interpatient_stability.py` uses no inter-patient
data. It perturbs seven fixed landmarks with Gaussian noise of standard
deviation σ and counts unchanged codes. Re-run at p = 6, σ = 0.5 mm: 43.4 %
identical, against the 49 % quoted in v4 (protocol details differ; same
order of magnitude).

Compared against the elementary prediction that the point simply has to
stay inside its own quantisation cell along each axis independently, over
p ∈ {5,…,8} and σ ∈ {0.25,…,4} mm, the largest deviation is **0.05**, and
below 0.01 for σ ≥ 2 mm. The experiment measures the probability that a
jitter of size σ leaves a point in its cell — a property of the grid. It
says nothing about anatomy, about registration, or about patients, and
must not be presented as a corollary about inter-patient agreement.

Re-run: surface morphing (22.9× confirmed, `surface_morphing.py`).
