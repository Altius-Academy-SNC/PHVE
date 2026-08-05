# PHVE experiment notebook

Everything that produces a number in the v5 paper
(`altius_code_arxiv_v5.tex`). Nothing in the paper is asserted without a
script here.

Read `LOGBOOK.md` first: it is the chronological record, including the
runs that failed and the two bugs found in the v4 scripts.
`PRIOR_ART.md` records what the literature already establishes.
`UNVERIFIED.md` lists what this work does *not* settle.

## Install

```bash
python3 -m venv venv && ./venv/bin/pip install -r ../requirements.txt
./venv/bin/pip install nilearn nibabel psutil
cc -O2 -shared -fPIC -o phve/cachesim.so phve/cachesim.c
```

The C file is the cache simulator; hardware performance counters were
unavailable on the machine used (`kernel.perf_event_paranoid = 4`), so
cache behaviour is measured by exact deterministic simulation instead.
`cachesim.so` is required by `exp04`.

## Run

Every script writes a JSON file to `results/` and prints a summary. All are
seeded; rerunning reproduces the numbers bit for bit on the same machine
(timings excepted).

| script | paper section | wall time |
|---|---|---|
| `exp01_equivalence.py` | Equivalence lemma, verified | ~2 min |
| `exp02_hypotheses.py --n-target 3000 --remesh-steps 5` | Hypotheses of the published theorems | ~10 min |
| `exp03_ordering_scaling.py --seeds 1 2 --sizes 2000 4000 8000 16000 32000 64000` | Scaling of the index-gap functionals | ~25 min |
| `exp04_comparative.py --n-target 6000 --steps 12 --remesh-every 1 2 4 12 --refine-fraction 0.03` | The comparative protocol | ~20 min |
| `exp05_R1_anisotropy.py` | R1 measured | ~3 min |
| `exp06_R3_C3.py --inv-p2 1 2 3 4 5 6 7 8 --inv-p3 1 2 3 4 5 6 --gap-fraction 0.08` | R3, the locality constants | ~20 min |
| `exp07_R4_dpcm.py` | R4, regularity transfer and DPCM | ~5 min |
| `exp08_gap_distribution.py` | The index-gap distribution and its tail | ~15 min |
| `exp10_equivariant_precond.py` | Equivariant preconditioners; the cache crossover | ~60 min |
| `exp11_meangap_constants.py` | Constants of the mean-gap theorem (U8) | ~10 min |
| `exp12_compact_hilbert.py` | Compact Hilbert index comparison (U5) | ~1 min |
| `exp13_variant_locality.py` | Skilling vs H&RC: symmetry search and locality | ~10 min |
| `exp14_wl2_extrapolation.py` | WL₂ extrapolation, calibrated in d = 2 and d = 3 | ~15 min |
| `exp15_v4_carryover.py` | The two v4 carry-overs (U10) | ~3 min |
| `exp16_anisotropic_domain.py` | The protocol on a stretched domain (U3, R1) | ~20 min |
| `exp17_seam_closed_forms.py` | Exhaustive check of A₂(p), A₃(p) (U6) | ~20 min |
| `exp18_scheme_properties.py` | Maximum principle and self-convergence (U7) | ~15 min |

`exp09_cache_and_ic0.py` is **superseded by `exp10`**, which measures the
same IC(0) iteration counts and cache misses -- identical values, both
being deterministic -- and adds two more cache geometries, two more
preconditioners and two larger sizes. It is kept only for the record.

Everything, in order:

```bash
for s in exp01_equivalence exp02_hypotheses exp03_ordering_scaling \
         exp04_comparative exp05_R1_anisotropy exp06_R3_C3 exp07_R4_dpcm \
         exp08_gap_distribution exp10_equivariant_precond \
         exp11_meangap_constants exp12_compact_hilbert \
         exp13_variant_locality exp14_wl2_extrapolation exp15_v4_carryover \
         exp16_anisotropic_domain exp17_seam_closed_forms \
         exp18_scheme_properties; do
  ./venv/bin/python -u $s.py | tee results/$s.log
done
```

## Layout

```
phve/hilbert.py    vectorised Skilling kernel; base-29 encoding;
                   per-axis-order (anisotropic) PHVE ordering; the R1 rule
phve/mesh.py       unstructured adaptive tetrahedral meshes
                   (jittered sampling + Delaunay + mask filter + refinement)
phve/fem.py        P1 assembly, mesh-based mollification, semi-implicit step
phve/metrics.py    orderings (natural / RCM / PHVE), bandwidth, profile,
                   ILU fill, cache simulation, timers
phve/model.py      the model problem: MNI152 domain, u0, adaptivity
phve/cachesim.c    set-associative LRU cache simulator for CSR SpMV
phve/ic0.c         IC(0) factorisation and triangular solves, fixed pattern
phve/compact_hilbert.py
                   reference implementation of Hamilton & Rau-Chaplin's
                   standard and compact Hilbert indices, used to test our
                   own claims about them rather than assert them
common.py          hardware record, JSON I/O
```

## Two things to know before reading the results

**The mesh is never a lattice.** On a structured grid the lexicographic
ordering already gives a bandwidth Θ(n^{d-1}), which is optimal in order,
so no space-filling curve can win and a negative result there proves
nothing. Every mesh here comes from jittered rejection sampling of a
non-convex domain followed by a Delaunay triangulation, refined by
inserting centroids and retriangulating.

**RCM timings include building the adjacency graph.** That is the honest
accounting: reverse Cuthill–McKee cannot run without it, and not needing it
is the whole claim being tested for PHVE. See `metrics.order_rcm`.

## Reproducibility notes

- Seeds are explicit arguments; the default is 1.
- Hardware and library versions are recorded in every JSON file under
  `"hardware"`.
- `scipy.sparse.linalg.spilu` is called with `permc_spec="NATURAL"`. With
  the default (`COLAMD`) the factorisation reorders the matrix itself and
  the experiment would measure COLAMD instead of the ordering under test.
  This mattered: see `LOGBOOK.md` §6.
