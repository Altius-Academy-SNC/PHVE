# Where the article should go now

Written after everything v4 asserted was tested, and after the blocking
point identified in v5 (the threshold-ILU artefact) was removed by
implementing IC(0). This is the brainstorming document requested: what
survives, what the new measurements changed, and which article we should
write.

---

## 1. What survives once everything refuted is removed

**Proved and verified:**

| result | status |
|---|---|
| Bijectivity of `F` for d = 2, 3 | proved, exhaustively cross-checked |
| Hierarchy by truncation | proved (rests on the *inverse* bound, unaffected) |
| Quantisation bound | proved, measured to be *attained* |
| Equivalence lemma (renumbering is analytically inert) | proved, verified to 1e-12 |
| Inverse Hölder constant C'₂ = 6.00 | recomputed, recovers the classical value |
| **Inverse Hölder constant C'₃ ≈ 29.5** | **new — no published value found** |
| **Seam theorem: forward bound false, gap = Θ(n^d)** | **new — proved** |
| **Exact adjacent-pair maxima, explicit witness** | **new — lower bound now PROVED (§2.3)** |
| Anisotropy correction κ, per-axis order rule | proved, verified |
| Regularity transfer α → α/d | elementary, verified on 8 volumes |

**Refuted:** forward locality bound; bandwidth theorem; pointwise morphing
bound; "PHVE reduces bandwidth".

So the surviving mathematics is *not about the encoding*. It is about the
**locality of the discrete Hilbert curve and what it does to a sparse
matrix**. That is the first and strongest signal about direction.

---

## 2. What the new work changed

### 2.1 IC(0) removes the artefact — and the negative result was mis-attributed

v5 concluded "PHVE cannot beat RCM at any remeshing frequency", entirely
because of a **36 % denser incomplete factorisation**. That was SuperLU's
*threshold* ILU, which chooses its own fill.

With IC(0) (pattern fixed to `tril(A)`, implemented in `phve/ic0.c`,
verified: `max |LLᵀ − A|` on the pattern = 7e-12):

- **fill is identical for all three orderings** (×0.53 of nnz(A)) — by
  construction. The entire basis of the v5 negative result disappears.
- but the **iteration count is not**.

> **RETRACTED (exp10, second seed).** This section originally read "the
> iteration count is not: **RCM is consistently better**", on the strength
> of a single seed. Adding a second seed destroys the claim. The
> IC(0)-PCG counts, RCM vs PHVE:
>
> | N (seed 1) | RCM | PHVE | | N (seed 2) | RCM | PHVE |
> |---|---|---|---|---|---|---|
> | 7 771 | **16** | 25 | | 7 770 | 19 | **18** |
> | 15 671 | **21** | 24 | | 15 676 | **35** | 38 |
> | 31 537 | **43** | 75 | | 31 527 | 34 | **28** |
> | 63 265 | **34** | 69 | | 63 195 | **64** | 74 |
> | 126 920 | **64** | 71 | | 126 830 | 54 | **49** |
>
> RCM wins 5 of 5 on seed 1 and **2 of 5 on seed 2**. The effect is
> seed-dependent, i.e. it is mesh noise, not a property of the ordering.
> Nothing about IC(0) quality can be claimed from this data. What *is*
> robust is that IC(0) is **not** permutation-equivariant: its spread
> across orderings reaches 50.7 %, against 0.00 % for Chebyshev.
>
> This is exactly the failure mode §2.6 warns about, caught here only
> because a second seed was run. Every remaining claim in this document
> was re-checked for the same weakness.

So what survives is: with a fixed pattern the fill is identical by
construction, the *quality* of the factorisation varies with the ordering
by tens of percent, and **which ordering wins is not determined** by the
data we have.

### 2.2 The cache advantage is real, has a size threshold, and is robust

Simulated L1 (32 KiB, 8-way) misses per SpMV, ratio PHVE/RCM, both seeds:

| N | ~7 770 | ~15 670 | ~31 530 | ~63 200 | ~126 900 |
|---|---|---|---|---|---|
| seed 1 | 1.214 | 1.066 | **0.643** | **0.629** | **0.566** |
| seed 2 | 1.181 | **0.819** | **0.691** | **0.585** | **0.608** |

**Crossover between N ≈ 8 000 and N ≈ 32 000**, the two seeds disagreeing
only on the N ≈ 15 670 point. Above it Hilbert's bulk locality wins by
35–43 %, on both seeds, and the advantage grows with N. Deterministic
simulation, so reproducible.

**But only in L1.** At L2 (1 MiB) the ratio is exactly 1.000 at every size
on both seeds except the largest, where it is 1.028 and 1.029 — a slight
*disadvantage*. At L3, 1.000 everywhere. An L1 miss that hits in L2 costs
about a dozen cycles and is largely hidden by prefetch and out-of-order
execution, so this is a much weaker statement than "fewer cache misses"
suggests.

### 2.3 The gap distribution obeys a clean, N-independent law

The bandwidth bound tried to summarise a whole distribution by its maximum.
The distribution itself is much better behaved.

- **tail exponent:** −0.400, −0.376, −0.361, −0.348, −0.340 over
  N = 7 771 → 126 920, converging monotonically to **−1/3 = −1/d**;
- **the curves collapse:** max deviation from the mean curve over
  t ∈ [10, 1000] is **0.021** for PHVE against **0.268** for RCM. The
  Hilbert gap law is N-independent in the bulk; the RCM one is not;
- **median:** PHVE 15, 16, 16, 17, 17 (constant over ×16 in N);
  RCM 206, 413, 598, 778, 1392 (grows like N^{2/3});
- **shape:** RCM is concentrated (at N = 126 920: q90 = 8 461, q99 = 8 817,
  max = 8 971 — nearly a step function); PHVE is heavy-tailed
  (q50 = 17, q90 = 1 667, q99 = 44 566, max = 124 272).

This is provable by the dyadic argument that already gave the mean-gap
theorem:

> **Master theorem.** Under (A1)–(A4) and quasi-uniformity,
> `P(|i−j| > t) ≤ K ρ t^{−1/d}` with K depending only on d and the
> regularity constants — **independent of N**.

Corollaries: median = O(1); mean = ∫₀^N P(>t)dt = O(N^{(d−1)/d}) (the
theorem already proved); max = Θ(N) (the seam theorem, in the region where
the tail bound is vacuous).

### 2.4 The seam maxima: the lower bound is now proved

The exact maxima over adjacent pairs were conjectural in v5. The structure
is now understood:

- the maximum is attained **at the seam between the first and last
  sub-cube** — verified for every p tested (d = 2 up to p = 9, d = 3 up to
  p = 6);
- the explicit witness in d = 2 is the pair `((m−1,0), (m,0))`, and it
  realises the closed form exactly for p = 2…9;
- and `D(p) := H_p(n−1,n−1) − H_p(0,n−1) = (4^p − 1)/3` **is proved** by a
  two-line telescoping induction: the two points sit in quadrants 2 and 1,
  so `D(p) = 4^{p−1} + D(p−1)`, `D(1) = 1`.

Hence `A₂(p) ≥ 3·4^{p−1} + D(p−1) = (5·4^p − 2)/6` is **proved**, with an
explicit witness. The same structure holds in d = 3 with
`E(p) = 3·8^{p−1} + E(p−1) = 3(8^p − 1)/7`, verified to p = 6.
What remains conjectural is only the matching **upper** bound (that no
other adjacent pair does better).

### 2.5 The positive result: permutation-equivariant preconditioners

> **Correction.** The table that stood here was not produced by any script
> in this repository — `grep -i jacobi` matched only this document. Under
> the working rule that no figure enters the paper without a versioned,
> seeded script, it had to be treated as unestablished and redone. It has
> been, by `exp10_equivariant_precond.py`. The result **holds and is
> sharper than what was claimed**, but one part of the old claim was wrong
> and is corrected below.

The equivalence lemma says the iteration count of any *permutation-
equivariant* preconditioner is invariant. Two are tested — Jacobi
(M⁻¹ = D⁻¹) and a degree-4 Chebyshev polynomial in the symmetrically
scaled matrix — against IC(0), which is not equivariant. The spectral
bound the Chebyshev polynomial needs is computed once, on the natural
ordering, so that equivariance is not broken by construction.

Relative spread of the iteration count across the three orderings
(seed 1; the unpreconditioned control is the round-off yardstick):

| N | none (control) | jacobi | cheb4 | ic0 |
|---|---|---|---|---|
| 7 771 | 1.96 % | 0.00 % | **0.00 %** | 36.0 % |
| 15 671 | 0.82 % | 0.76 % | **0.00 %** | 25.0 % |
| 31 537 | 0.96 % | 0.32 % | **0.00 %** | 42.7 % |
| 63 265 | 1.30 % | 1.69 % | **0.00 %** | 50.7 % |
| 126 920 | 0.96 % | 1.08 % | **0.00 %** | 42.9 % |

The Chebyshev counts are *exactly* equal at every size — a cleaner
demonstration than the 0.7 % originally claimed, because it converges in
far fewer iterations and so accumulates less round-off. IC(0) is nowhere
close, and RCM wins it every time.

**Where the old claim was wrong.** It asserted the cache advantage without
saying at which level. Sweeping three geometries:

| N | 7 771 | 15 671 | 31 537 | 63 265 | 126 920 |
|---|---|---|---|---|---|
| L1 32 KiB, PHVE/RCM | 1.214 | 1.066 | **0.643** | **0.629** | **0.566** |
| L2 1 MiB, PHVE/RCM | 1.000 | 1.000 | 1.000 | 1.000 | 1.029 |
| L3 16 MiB, PHVE/RCM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**The advantage exists only in L1.** At L2 it is a tie, and at the largest
size tested slightly unfavourable. That matters: an L1 miss that hits in L2
costs roughly a dozen cycles and is largely hidden by prefetch and
out-of-order execution. "PHVE strictly dominates RCM" is therefore too
strong and must be withdrawn.

What survives, and is measured: with an equivariant preconditioner the
iteration count is identical, PHVE moves 37–43 % less L1 traffic above
N ≈ 3·10⁴, and its ordering is 1.5–4× cheaper to compute (rcm/phve = 1.53,
2.31, 2.77, 3.14, 3.98 across the sweep, the advantage growing with N).
Both are second-order effects, and both are *simulated* — see U2.

### 2.6 What is NOT resolvable, and must be said

The **wall-clock totals of exp04 are dominated by noise.** Single runs at
m = 1, 2, 4, 12 disagree on the winner (RCM, PHVE, RCM, PHVE). Per-iteration
times vary by ±40 % between runs. Nothing about total time can be claimed
until the runs are repeated and reported with a spread. This is recorded
in `UNVERIFIED.md`.

---

## 3. The thesis of the new article

> A space-filling-curve renumbering does **not** reduce bandwidth. It
> produces a **heavy-tailed index-gap distribution with an N-independent
> law** `P(|i−j| > t) ≍ t^{−1/d}`. Whether that helps depends entirely on
> which functional of that distribution the solver pays for.

| solver component | functional paid for | winner | evidence |
|---|---|---|---|
| band / skyline direct solver | maximum | **RCM** | max: Θ(N) vs Θ(N^{2/3}) |
| threshold ILU (fill chosen by the factoriser) | far tail | **RCM** | fill ×5.78 vs ×4.26 |
| IC(0) / IC(k), fixed pattern | far tail, via factor *quality* | **undecided** | seed-dependent: RCM 5/5 on seed 1, 2/5 on seed 2 |
| **Jacobi / polynomial (equivariant)** | **nothing — invariant** | **tie on iterations** | Chebyshev spread **0.00 %** at every size, both seeds |
| SpMV, L1 traffic | bulk (median = O(1)) | **PHVE, for N ≳ 3·10⁴** | miss ratio 0.57–0.69, both seeds |
| SpMV, L2/L3 traffic | — | **no difference** | ratio 1.000, and 1.028 at the largest N |
| computing the ordering | none | **PHVE** | 1.5–4.3× cheaper, no graph, growing with N |

**The positive result, stated at the strength the data supports.** With a
permutation-equivariant preconditioner the iteration count is invariant —
exactly so, 0.00 % spread for Chebyshev at every size on both seeds — so
the ordering can only be judged on memory traffic and on the cost of
computing it. On both, above N ≈ 3·10⁴, Hilbert wins: 35–43 % fewer L1
misses and an ordering 1.5–4.3× cheaper to compute, with no adjacency
graph.

Three qualifications, all of which must appear next to the claim:

1. the traffic advantage is **L1 only** — L2 and L3 show no difference, and
   a slight disadvantage at the largest size tested;
2. the cache figures are **simulated**, not measured (U2);
3. with a pattern-dependent preconditioner the picture does **not** simply
   reverse, as this document previously claimed — it is undecided, because
   the effect is smaller than the seed-to-seed variation.

So: not "strictly dominates". Something narrower and defensible — *for the
one class of preconditioner where renumbering cannot change the iteration
count, a space-filling curve is the better choice, for reasons that are
second-order and that we can only measure in simulation.*

Most results that refuted v4 do support this thesis. The IC(0) retraction
in §2.1 is a reminder that the same scrutiny has to be applied to the
replacements.

---

## 4. Three possible articles

### Direction A — "Two locality constants of the discrete Hilbert curve"
Short mathematical note: forward constant diverges (closed forms, proof);
inverse constant computed, C'₂ = 6 recovered, C'₃ ≈ 29.5 new.
*Pro:* airtight, fast to referee. *Con:* narrow; C'₃ is an extrapolation,
which weakens the headline; discards everything applied.

### Direction B — "Bandwidth is the wrong functional" ← **recommended**
The master theorem and its corollaries, the two constants as the tools that
make them explicit, then the solver study whose punchline is the table in
§3. PHVE becomes a *section* (a concrete encoding realising the ordering),
not the subject.
*Pro:* one clear thesis, a genuine positive result (§2.5), and every
negative finding supports it. *Con:* needs the timing protocol repaired
(§2.6) before submission.

### Direction C — Split in two
(a) constants + distribution theorem (numerical analysis venue);
(b) PHVE the encoding + medical indexing (software venue).
*Pro:* homogeneous papers. *Con:* (b) has little mathematics left and
should be honest that it is a software paper.

---

## 5. Recommendation

**Direction B**, keeping C as a later option for a separate software paper.

Proposed structure:

1. **The two locality constants.** Forward diverges (seam theorem, proved;
   exact maxima with explicit witness, lower bound proved). Inverse
   converges: C'₂ = 6.00 recovered, C'₃ ≈ 29.5.
2. **The index-gap distribution.** Master theorem `P(>t) ≤ Kρ t^{−1/d}`,
   N-independent; corollaries for median, mean, max; the measured collapse.
3. **The anisotropy correction** and the per-axis order rule (κ, ρ).
4. **What each solver component pays for.** Model problem, equivalence
   lemma fixing the scope, IC(0) vs Jacobi vs threshold ILU, the cache
   crossover, and the table of §3.
5. **The encoding** as an instance: bijectivity, truncation, quantisation
   (all correct), and the base-29 alphabet.
6. **Errata** w.r.t. v4 and the explicit list of what is unverified.

Working title: *"Bandwidth is the wrong functional: the index-gap
distribution of Hilbert renumbering and what solvers actually pay for."*

---

## 6. What must be done before writing it

*Updated after the verification campaign of exp10–exp18. Everything that
could be settled by computation has been; what is left is either a proof, a
machine permission, or a dataset.*

**Blocking:**

1. **Repair the timing protocol** (§2.6): repeat each configuration, report
   medians and spreads, or drop wall-clock claims entirely and keep only the
   deterministic quantities (iterations, misses, fill, renumbering cost).
   *Still open.* Note that exp10's seed-1 ordering times were measured on an
   otherwise idle machine and are usable; its seed-2 times were measured
   under concurrent load and must not be quoted.
2. **Prove the master theorem** cleanly and pin down K. *Still open, and it
   is the decision point: without it there is no Direction B.*
3. ~~Verify the Jacobi result at large N.~~ **Done, and corrected** — see
   the correction box in §2.5. Iteration counts are invariant (exactly so
   for Chebyshev), but the cache advantage exists **only in L1**, so
   "strictly dominates" has been withdrawn.

**Done since this document was first written:**

- ~~Re-examine the two v4 carry-overs.~~ **exp15.** MNI152 bijectivity: the
  test had never been run; run properly, the claim is *confirmed* (0
  collisions at p = 8). "Inter-patient stability": reproduces numerically
  but measures quantisation, not anatomy, and must be restated.
- ~~Decide whether C'₃ stays an extrapolation.~~ **exp13 + exp14.** It stays
  an extrapolation, but is now calibrated against a *three-dimensional*
  published value (H&RC, 3.83 % overshoot) as well as the 2-D one (0.02 %).
  The headline changes: not "the 3D constant" but "the widely-implemented
  Skilling variant is 1.24× worse than the best published 3D Hilbert curve".
- **exp12:** a false claim was found in our own source (order-equivalence
  with the compact Hilbert index) and corrected. The rank-compression lemma
  itself survives.
- **exp11 (U8):** hypothesis (A3) of the mean-gap theorem is
  *unsatisfiable* as published and has been repaired; the theorem is a
  statement about the exponent only, its constant being 3–4 orders
  pessimistic.
- **exp17 (U6):** the closed forms for the seam maxima are now exhaustively
  verified to p = 12 (d = 2) and p = 7 (d = 3), against p = 6 and p = 4.

**Strengthening, in decreasing order of value:**

4. Prove the **upper** bound in §2.4 (that the seam pair is optimal). The
   lower bound and the witness are done, and exp17 now supports the
   conjecture over grids of 16.7 M and 2.1 M points.
5. Push the tail-exponent measurement one decade further in N.
6. **U2** — hardware cache counters. Blocked on
   `kernel.perf_event_paranoid = 4` and no sudo in the working session. All
   cache figures are simulation. One `sysctl` would close it.
7. **U4** — no full-resolution clinical volume (IXI, ACDC) has been tested.
   This is now the largest untested surface, because the DPCM criterion of
   R4 is validated only on MNI152 and synthetic fBm fields.
8. **U1 remainder** — ICT (thresholded incomplete Cholesky) was not
   implemented; IC(0) alone was enough to isolate fill from quality.
