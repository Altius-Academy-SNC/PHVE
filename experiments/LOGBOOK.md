# Laboratory notebook — PHVE v5

Chronological record of what was run, what came out, and what it forced us
to change in the paper. Negative results are kept.

Machine for every timing below: AMD Ryzen 7 7840HS, 8 cores / 16 threads,
L1d 32 KiB per core, L2 1 MiB per core, L3 16 MiB shared; Linux 6.17;
CPython 3.12.3, NumPy 2.4.2, SciPy 1.17.1. `kernel.perf_event_paranoid = 4`,
so no hardware counters (see `UNVERIFIED.md` U2).

---

## 0. Prior art (before any proof was written)

Recorded in `PRIOR_ART.md`. Two findings changed what we could claim:

- Hamilton & Rau-Chaplin (2008) already give the Hilbert index on a grid
  with unequal per-axis resolutions. **R1's construction is not new**; only
  the κ-correction of the bound and the selection rule for the pᵢ are.
- "SFC repartitioning is cheaper than graph partitioning when the mesh
  changes often" is folklore in the partitioning literature (Zoltan, p4est,
  Sasidharan & Snir). **R2 must be an explicit operation count and a
  measured crossover, not a qualitative claim.**

---

## 1. Kernel validation

`phve/hilbert.py`, vectorised Skilling.

- Bijective, exact round trip, unit-step adjacency for d ∈ {2,3}, p ≤ 5.
- Agrees with the repository's scalar `codec3d.xyz2d` on all 4096 points at
  d = 3, p = 4.

## 2. FEM validation

`phve/fem.py`, P1 tetrahedra.

- `sum(M) = 1.76507e6` vs. mesh volume `1.765e6` mm³ (partition of unity).
- `‖K₀·1‖∞ = 5.2e-12` (patch test: constants in the kernel).
- `‖K − Kᵀ‖∞ = 1.8e-12`.
- Mass drift over one step: `-7e-7` absolute on a total of 1.7e6.
- Frozen diffusivity range `g ∈ [0.0188, 0.99992]`, so `g_min > 0` as the
  Lax–Milgram argument requires.

---

## 3. exp01 — Equivalence lemma  ✔ as predicted

N = 2500. Eigenvalues agree to 6.19e-15 relative; κ₂ agrees to 12
significant digits; one direct solve differs by 15 ε; after 20 steps,
1.04e4 ε. Unpreconditioned CG iteration counts differ by at most 4 (they
are equal in exact arithmetic).

**Consequence for the paper:** the scope statement in the introduction —
the encoding is analytically inert — is now measured, not asserted.

---

## 4. exp02 — Hypotheses of the v4 theorems  ✘ two violated

(a) **`N ≤ n^d` is not sufficient.** On the finest graded mesh
(N = 84 044) the counting condition first holds at p = 6, where **40.5 % of
nodes still collide** (up to 30 nodes in one cell). Zero collisions only at
p = 10, four orders later.
→ Disambiguation rule adopted: stable sort, ties broken by insertion order.
Documented in the paper's definition of the PHVE numbering.

(b) **The regime `(h/Δ)^d ≪ N` is never entered.** At p = 8 the ratio
`(h_max/Δ)³/N` runs 228 → 118 → 61 → 27.7 → 14.2 → 7.6 across the six
refinement levels. It never drops below 1.
→ The v4 bandwidth bound is not merely pessimistic, it is inapplicable on
this problem.

(c) **Anisotropy: `r_grid_max/(h/Δ_max) = κ` to within 2 %** over
κ ∈ [1, 20] (measured 1.17, 2.34, 3.12, 5.08, 8.20, 13.28, 20.31).
→ Confirms R1's correction factor.

(d) Round trip exact at p ∈ {6,8,10,12}; quantisation error equals the
theoretical bound to all digits (the max is attained).

---

## 5. exp03 — Scaling  ✘ the v4 bandwidth theorem is false

Two seeds, N from 3.9e3 to 1.27e5, adaptive graded meshes.

| metric | natural | RCM | PHVE | predicted for PHVE |
|---|---|---|---|---|
| max \|i−j\| | 1.001 | 0.689 | **0.998** | 1 (seam theorem) |
| mean \|i−j\| | 0.977 | 0.716 | **0.643** | 2/3 (mean-gap theorem) |
| median \|i−j\| | 0.955 | 0.528 | **0.064** | — |

At N = 126 920: max 126 767 / 8 971 / 124 272; mean 34 406 / 2 988 / 1 760;
N^{2/3} = 2 525.

**This is the pivotal experiment.** It forced:
- the seam theorem (proved: adjacent points with index gap > (1−2^{1−d})n^d),
- the exhaustive computation of the exact adjacent-pair maxima,
- the replacement of the bandwidth theorem by the mean-gap theorem,
- the erratum about v4's experimental table, whose "bandwidth 1112/49/47"
  are mean index gaps (confirmed by re-running v4's own
  `fem_bandwidth.py`, whose variable is literally named `bw_avg`).

Note the median: PHVE's is essentially **constant in N** (11–12 across two
decades). Almost all couplings sit next to the diagonal; a thin tail of
seam couplings does not.

---

## 6. exp04 — Comparative protocol  ✘ PHVE loses on total time

12 steps, m ∈ {1,2,4,12}, 3 % refined per remesh.

First run crashed: `RuntimeError: CG failed`. Diagnosis — **not** a bad
matrix (unpreconditioned CG converged in 121 iterations). Two real bugs in
the harness:
1. `spilu` defaults to `permc_spec="COLAMD"`, i.e. **it reorders the matrix
   itself**, so the experiment would have measured COLAMD, not our
   ordering. Fixed to `"NATURAL"`.
2. The resulting factor is not SPD, which breaks CG. Fixed with
   `SymmetricMode=True`, `diag_pivot_thresh=0`.
   The natural ordering still diverges (12/12 steps) — that is a genuine
   result, not a bug, and is reported as such.

Totals at m = 1, N = 41 091: RCM 27.43 s, PHVE 29.91 s, natural 792.78 s.
RCM wins at every m.

Decomposition, per step:
- renumbering saving for PHVE: **0.0503 s** (0.0293 vs 0.0796 s per
  remesh — PHVE is 2.7× faster, exactly the "no adjacency graph" benefit),
- ILU penalty for PHVE: **0.257 s** (fill ×5.78 vs ×4.26),
- ratio 5.1 → **the crossover m\* = 0.20 < 1, so no remeshing frequency
  can make PHVE win.** Reported as a negative result.

What PHVE does win: mean gap 482 vs 745, simulated cache misses
7.73e4 vs 1.02e5 (−24 %) and vs 2.56e6 for natural (÷33). ILU-CG iteration
counts are a tie (48 vs 49).

Control: unpreconditioned CG totals 4486 / 4475 / 4471 — agree to 0.3 %,
as the equivalence lemma requires.

Caveat kept in `UNVERIFIED.md` U1: the ILU dominates everything and is not
the factorisation a production SPD solver would use.

---

## 7. exp05 — R1 anisotropy  ✔ direction confirmed, factor not attained

Ellipsoid in a box, 20 000 vertices, budget Σpᵢ = 30.

| κ | ρ | rule p | mean gap cubic → rule | measured | achievable bound |
|---|---|---|---|---|---|
| 2 | 1.41 | (11,10,9) | 569 → 493 | ×1.16 | ×1.00 |
| 5 | 2.24 | (11,10,9) | 571 → 404 | ×1.41 | ×2.00 |
| 20 | 4.47 | (12,10,8) | 709 → 307 | **×2.31** | ×4.00 |

Monotone in κ, but roughly half the bound — expected, the bound is an upper
bound and is not tight. The **max** gap barely moves, as the seam theorem
requires.

FB box 2000×600×400: κ = 5, ρ = 1.957, but integer rounding gives
p = (11,10,9) and an achieved Δ_min ratio of exactly **1.50**, so the
d-th-power bound improves by 3.38, not by ρ³ = 7.50. We report the achieved
figure, not the continuum one.

---

## 8. exp06 — R3 constant

**Forward constant diverges.** Exhaustive maxima over adjacent pairs:

- d = 2: 3, 13, 53, 213, 853, 3413 → `A_2(p) = 4A_2(p−1)+1 = (5·4^p−2)/6`
- d = 3: 7, 59, 475, 3803, 30427, 243419 → `A_3(p) = 8A_3(p−1)+3 = (13·8^p−6)/14`

i.e. ~0.833 n² and ~0.929 n³. So no forward bound exists, and the constant
"6" cited in v4 cannot belong to that direction.

**Inverse constant.** Recomputed as
`max ‖H⁻¹(i)−H⁻¹(j)‖₂^d / |i−j|`:

- d = 2: 1.000, 2.500, 3.625, 4.481, 5.233, 5.620, 5.811, 5.906
  → geometric extrapolation **5.999 ≈ 6**. This is Moon et al.'s constant,
  and it validates the whole method.
- d = 3: 1.732, 9.000, 18.053, 22.959, 25.960, 27.580
  → **29.48**, i.e. `C'_3 = 29.5 ± 0.5`.

Argmax index gap is a stable 0.038 N for d = 3 and 0.042 N for d = 2, so
the gap sweep at 0.08 N is safe.

---

## 9. exp07 — R4 regularity transfer and DPCM

**Exponent transfer** verified on 8 volumes: measured β exceeds α/3 by
6–21 %, with the right ordering and the right monotone dependence on α.

**A v4 bug found.** `dpcm_compression.py` calls `np.round(x)` on a template
whose values lie in [0, 0.988] → the signal becomes **binary** before any
residual is taken. The reported "original entropy 0.462 bits" is the
entropy of a two-level image.

Corrected to 8-bit:

| quantisation | levels | H_orig | raster | Hilbert | gain |
|---|---|---|---|---|---|
| as published, round(x) | 2 | 0.4617 | 0.4802 | 0.5638 | −0.0836 b (−17.4 %) |
| corrected, round(255x) | 191 | 7.0232 | 6.1716 | 6.3394 | −0.1678 b (−2.72 %) |

**The sign does not change: Hilbert still loses on that protocol.**

Second defect: at p = 7 the CR box gives ~2 mm cells and **14.1 % of voxels
collide**. Sweeping p with correct quantisation: 88.0 % collisions at p=6
(loss 0.294 b), 14.1 % at p=7 (0.168 b), 0 % at p≥8 (0.105 b). Removing
collisions recovers a third of the loss; the rest is genuine.

**Why it is genuine, and predicted.** The criterion decomposes each
traversal into bulk and wrap pairs:
- full-volume raster of MNI152: W/B = 0.011 → no gain predicted, none
  observed (row-ends sit in the zero background);
- centred 64³ cube: W/B = 21.0 → gain predicted, observed
  (M2 ratio 0.70, −0.068 bits);
- mask-compacted raster (the v4 protocol): W/B = 4.26 with wrap fraction
  0.0230 → raster M2 = 1.075·B. But the Hilbert traversal restricted to the
  mask has only 97.06 % unit steps, so its M2 = 1.246·B. The **full**
  criterion 1.075 < 1.246 predicts the loss; the reduced criterion W > B
  does not apply here and would predict the wrong sign.

The reduced criterion is right on 8/8 cube volumes. Its scope is full-cube
traversals, and the paper says so.

**Artefact caught during this experiment:** the first synthetic fBm fields
were generated by FFT on an n³ torus, hence periodic, so the raster wrap
pairs degenerated into short steps and W/B collapsed to ~1.7 regardless of
H. Fixed by synthesising on (2n)³ and cropping to n³; W/B then ranges 6.6
to 142 as H goes 0.20 to 0.80.

---

## 10. exp12 — Compact Hilbert index  ✘ a false claim found in our own code

`phve/hilbert.py` stated in its docstring that `phve_order_aniso` is
"order-equivalent to the compact Hilbert index of Hamilton & Rau-Chaplin
(2008)". Nothing verified it. Their Algorithms 1 and 2 were implemented as
a scalar reference (`phve/compact_hilbert.py`) and self-tested first: the
compact index with all mⱼ = M reproduces their standard index exactly
(d ∈ {2,3}, M ≤ 4), and is a bijection onto {0,…,2^{Σmⱼ}−1} in all nine
anisotropic cases.

Three questions, kept separate so a failure could be localised:

| | question | d = 2 | d = 3 |
|---|---|---|---|
| Q1 | Skilling index = H&RC index? | **identical** | **different** |
| Q2 | H&RC restricted-cube order = H&RC compact order? | yes | **yes** |
| Q3 | our `phve_order_aniso` = compact index? | yes | **no** |

So the docstring was **false in d = 3**, the dimension the paper is about,
and has been corrected in place. The cause is isolated: not compactness
(Q2 holds everywhere, which is exactly the rank-compression lemma the
bandwidth argument uses), but the *variant*.

## 11. exp13 — The two variants are different curves  ✔ and it rescues R3

Searching all 48 signed axis permutations of the cube, and all 48 again
with the traversal reversed: **zero** matches for p ≥ 2 (two spurious
matches at p = 1, where 8 points admit many symmetries). Skilling's 3D
curve and H&RC's are genuinely distinct, not one curve in two frames.

Computed identically for both:

| p | A_Skilling | A_H&RC | WL₂ Skilling | WL₂ H&RC |
|---|---|---|---|---|
| 1 | 7 | 7 | 1.7321 | 1.7321 |
| 2 | 59 | 59 | 9.0000 | 7.9057 |
| 3 | 475 | 471 | 18.0526 | 13.8036 |
| 4 | 3803 | 3767 | 22.9592 | 18.2406 |

(p = 4 by gap sweep ≤ 327, so those two are lower bounds.)

**H&RC's variant is strictly better at every order tested**, in both the
forward and the inverse measure. This explains the conflict recorded in
`PRIOR_ART.md`: our WL₂ ≈ 29.5 and the published 22.9 are values of
*different curves* and were never comparable.

It also turns the objection into a test, run as exp14: if the same
extrapolation procedure applied to the H&RC variant returns ≈ 22.9, the
procedure is calibrated against two independent published values in two
dimensions.

## 12. exp11 — Constants of the mean-gap theorem  ✘ (A3) is unsatisfiable

Hypothesis (A3) requires every dyadic cell of every level ℓ to hold at most
c₁N2^{−dℓ} vertices. Once 2^{dℓ} > N that bound is below 1 while a
non-empty cell holds at least one vertex, so c₁ ≥ 2^{dℓ}/N is forced.
Measured, c₁ equals 2^{dp}/N *exactly* (ratio 1.00 or 2.00 = the maximum
occupancy at the finest level), so it carries no information about the
mesh. Bound/measured: **5.0×10⁷ to 1.9×10⁸** — the theorem is vacuous as
stated.

Repair: restrict (A3) to levels ℓ ≤ ℓ* with 2^{dℓ*} ≤ N and truncate the
sum in the proof there (legal: two points sharing a level-ℓ cell with
ℓ > ℓ* also share their level-ℓ* cell). The restricted constant is then
genuinely bounded — c₁* ∈ [3.84, 7.17] over N = 4 000…64 000 — and the
bound improves by ×1.7×10⁴.

It is still **2 773–10 717× above the measured mean gap**. The exponent is
right (exp03: 0.643 vs 2/3); the constant is three to four orders
pessimistic, driven by h = *max* edge length (44–66 mm on a domain 150 mm
across) and D = *max* degree (31–38 against a mean near 15).
c₂ ∈ [2.46, 2.86] is well behaved.

## 13. exp15 — The two v4 carry-overs  ✔ one confirmed, one reinterpreted

**Bijectivity on MNI152: the test had never been run.**
`simulations/bijectivity_mni152.py` checks 512 points at p = 3, then
encodes seven landmarks; the 235 375 in its output is the mask size, not a
collision count. Run properly on the same mask: 88.02 % colliding at p = 6,
14.10 % at p = 7, and **0 at p = 8, 9, 10** — the v4 claim is correct, and
p = 8 is exactly where the cell first fits inside the 2 mm voxel.

A first attempt used `data > 0` instead of the brain mask and did not
separate voxels clipped at the box boundary, producing a spurious 0.96 %
residual. Our error, recorded so the corrected run is not misread as a
retraction.

**"Inter-patient stability" measures quantisation, not anatomy.** The
script uses no inter-patient data: it jitters seven fixed landmarks by
Gaussian noise. Re-run at p = 6, σ = 0.5 mm gives 43.4 % (v4 quoted 49 %).
Against the elementary prediction that the point merely has to stay in its
own cell along each axis, over p ∈ {5..8} and σ ∈ {0.25..4} mm, the largest
deviation is 0.05 and below 0.01 for σ ≥ 2 mm. It is a property of the
grid. It must not be stated as a corollary about patients.

## 14. exp14 — WL₂ recalibrated  ✔ the R3 objection becomes a test

Since exp13 showed the two variants differ and Haverkort publishes
WL₂ = 22.9 for H&RC's, the extrapolation could be run on a *three-
dimensional* curve with a known answer. Same code, imported from exp06 so
it cannot be tuned:

| series | orders | extrapolated | published | error |
|---|---|---|---|---|
| Skilling, d = 2 | 6,7,8 | 5.999 | 6.0 | 0.02 % |
| H&RC, d = 2 | 6,7,8 | 5.999 | 6.0 | 0.02 % |
| H&RC, d = 3 | 4,5,6 | 23.778 | 22.9 | **3.83 %** |
| Skilling, d = 3 | 4,5,6 | 29.480 | — | — |

So the procedure overshoots by ~4 % in the dimension where it is used. The
honest figure is WL₂(Skilling, d=3) ≈ 29.5 with a 4 % method error, not
± 0.5. The robust quantity is the **ratio 1.24**: the variant most software
implements has about 24 % worse worst-case locality than the best
published 3D Hilbert curve. Both come from the same code on the same grids.

## 15. exp17 — Closed forms verified far further  ✔

Enumerating adjacent pairs costs O(d n^d), not O(n^{2d}), so the check goes
much further than exp06's sweep:

| | was | now | grid |
|---|---|---|---|
| d = 2 | p ≤ 6 | **p ≤ 12** | 16 777 216 points |
| d = 3 | p ≤ 4 | **p ≤ 7** | 2 097 152 points |

A(p)/n^d → 0.833333 and 0.928571 to six digits. Still conjectures beyond;
the proved statement remains the lower bound.

H&RC's d = 3 maxima are 7, 59, 471, 3767 — departing from Skilling's closed
form at p = 3, an independent confirmation that the curves differ.

## 16. exp16 — Anisotropic domain  ✔ R1 confirmed against an operator

MNI152 with its affine scaled along one axis: same mask, same generator,
only the aspect ratio moves.

| κ | R1 orders | mean gap cubic → aniso | factor |
|---|---|---|---|
| 1.24 | (10,10,10) | 817.5 → 817.5 | ×1.00 |
| 1.87 | (10,10,10) | 740.2 → 740.2 | ×1.00 |
| 3.74 | (11,10,9) | 725.0 → 551.8 | ×1.31 |
| 7.48 | (12,9,9) | 797.6 → 522.7 | ×1.53 |

Two things not previously stated. **The rule is inert below κ ≈ 2.5**
(integer rounding returns the cubic orders), so it does nothing on the
brain domain the rest of the paper uses. And the **maximum** gap drops by
~40 %, contradicting exp05's "the max barely moves" — probably because the
restricted-cube embedding puts the points in a slab that misses the cube's
principal seam. Since exp12 showed that embedding is not the compact index
in d = 3, this must not be claimed until checked against the real thing.

Jacobi spread across all four orderings: 1, 2, 0, 1 — equivariance holds on
anisotropic domains as well.

## 17. exp18 — Properties of the scheme  ✔ the limitation is now quantified

A first run was discarded: `DiffusionRun`'s default `refine_fraction = 0.15`
inserts ~0.9 N vertices per remesh, so the mesh nearly doubles each step and
10 steps are unusable. Rerun at 0.03, as `exp04` uses.

**Non-obtuse condition:** positive off-diagonal stiffness entries are
**31.96 %–33.78 %** over 20 steps and two seeds, carrying 0.21–0.27 of the
weight of the negative ones. Refinement barely helps (33.8 % → 32.0 % as
N goes 12 000 → 58 685).

**Does it bite?** The update creates a new extremum on **1 of 10 steps**
per seed, with worst overshoot 1.29e-3 and 3.99e-3 relative to the range of
u; undershoot never. So a third of the couplings break the sufficient
condition while the conclusion still holds 90 % of the time.

**Self-convergence:** observed order 0.60 in the median edge length. Not
usable as evidence of convergence — the nearest-neighbour transfer is
first-order at best and caps the estimate, and the reference mesh is only
1.26× finer in h than the finest test mesh. Recorded as a sanity check
only.

## 18. exp10 — Equivariant preconditioners, both seeds  ✔ and ✘

The table in `DIRECTION.md` §2.5 was produced by no script in the
repository. Redone properly, over two seeds and five sizes, with four
preconditioners.

**Robust, and stronger than claimed.** Relative spread of the iteration
count across the three orderings: Chebyshev degree 4 gives **0.00 % at
every size on both seeds**; Jacobi at most 2.67 %; the unpreconditioned
control at most 1.96 %; IC(0) up to 50.7 %. Equivariance is not an
approximation, it is exact, and the round-off that shows up in Jacobi is
the same order as in the control.

**Robust.** L1 miss ratio PHVE/RCM: 1.214/1.181 at N ≈ 7 770, then
1.066/0.819, 0.643/0.691, 0.629/0.585, 0.566/0.608. Crossover between
N ≈ 8 000 and 32 000, advantage 35–43 % above it, both seeds. Ordering
cost ratio 1.48–4.27, growing with N, both seeds.

**Corrected.** The cache advantage is **L1 only**: L2 gives 1.000 at every
size except the largest (1.028, 1.029 — a slight disadvantage), L3 gives
1.000 throughout. "PHVE strictly dominates RCM" is withdrawn.

**Retracted.** §2.1 claimed "RCM is consistently better" for IC(0), from
one seed. RCM wins 5 of 5 on seed 1 and **2 of 5 on seed 2** — at
N ≈ 31 530 PHVE takes 28 iterations against RCM's 34, and at N ≈ 126 830,
49 against 54. The effect is smaller than the seed-to-seed variation.
Nothing can be claimed about which ordering gives the better IC(0), only
that IC(0) is not equivariant.

This is the single most useful thing the verification campaign produced:
a claim that looked systematic on one seed, and would have gone into the
paper, is noise.

## 19. exp20 + exp21 — The transposition programme is closed  ✘

**exp20 — the master theorem's hypothesis fails on our own meshes.**
Reconstructing the dyadic argument shows the bound is N-independent exactly
when `h N^{1/d} = O(1)`, i.e. quasi-uniformity: it is the mechanism, not a
convenience. Measured slopes of `log(h N^{1/d})` against `log N`:

| | h_max | h_p99 | h_median | grading ratio |
|---|---|---|---|---|
| initial meshes | **+0.186** | −0.087 | −0.004 | +0.586 |
| after refinement | **+0.182** | −0.053 | −0.017 | +0.576 |

The *median* edge is quasi-uniform; the *maximum* is not and diverges like
N^0.18. Shape regularity is poor: ρ_min = 1.5e-9, 1st percentile 0.0105.
This is the direct cause of the 2 773–10 717 slack measured in U8.

**exp21 — nonlocal transposition does not work either.** With γ being
1/d-Hölder, the exponent arithmetic predicts that a d-dimensional kernel
‖x−y‖^{−(d+2s)} transposes to a 1-D fractional kernel |a−b|^{−1−2s/d}. The
median ratio is indeed ≈ 1, so the arithmetic is right. But:

| d | p | s | ratio q05 | q50 | q95 | pairs off by >2× | **row-sum mass in seam** |
|---|---|---|---|---|---|---|---|
| 2 | 4 | 0.25 | 0.298 | 0.848 | 6.83 | 45.7 % | **0.505** |
| 2 | 5 | 0.25 | 0.312 | 1.035 | 11.96 | 50.2 % | **0.540** |
| 2 | 5 | 0.50 | 0.247 | 1.042 | 19.65 | 58.1 % | **0.551** |
| 3 | 3 | 0.50 | 0.136 | 0.784 | 12.36 | 61.3 % | **0.635** |

The deviating pairs carry **half to two thirds of the row-sum mass, and the
fraction grows with p**. The seam is the main term, not a perturbation.

**Conclusion, recorded so it is not revisited.** Three attempts, one
answer:

1. *local, discrete* (FEM renumbering) — the equivalence lemma: a
   permutation changes nothing analytically;
2. *local, continuous* (PDE dimension reduction) — impossible: γ is
   nowhere differentiable, so no 1-D differential operator exists;
3. *nonlocal* (fractional) — measured above: the seam carries the mass.

The obstruction is structural, not a defect of the Hilbert variant: a
space-filling curve transposition is a **measure-preserving isometry**, so
it cannot reduce anything, and the only thing it could contribute —
locality — requires γ^{-1} to be Hölder, which no space-filling curve is.

Transposition by space-filling curve as a route to dimension reduction is
**closed**. No further simulation will reopen it.

## 20. Net effect on the paper

Corrected: 8 statements (see the errata table in the paper).
Proved: seam theorem, mean-gap theorem, rank-compression lemma,
equivalence lemma, anisotropy correction, optimal per-axis orders, per-step
cost accounting, exponent transfer, DPCM criterion.
Negative results kept: no crossover in m; DPCM loss on the masked protocol;
R1's achieved factor below the continuum prediction.
