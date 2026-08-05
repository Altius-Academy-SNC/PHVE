# Refuted claims

These four scripts backed statements in an earlier version of this work
(`arXiv v4`/`v5`, not published in this repository). The verification campaign
in `../../experiments/` refuted each of them. They are kept because their
outputs are cited in those refutations, not because their conclusions hold.

**Nothing here should be used as evidence for anything.** Two of them contain
bugs that the campaign found, noted below.

They still run — a `sys.path` line was added when they were moved here so that
`codec3d` is still importable — but running them reproduces the *incorrect*
figures, which is the point of keeping them.

---

## `fem_bandwidth.py` — the bandwidth theorem is false

Claimed that renumbering finite-element nodes by their Hilbert index reduces
the bandwidth of the stiffness matrix to $C_d(h/\Delta)^d + O(n^{d-1})$.

**Refuted by `experiments/exp03_ordering_scaling.py`.** The maximum index gap
under the Hilbert ordering is $\Theta(N)$ — measured ratio 0.998 against 1.001
for the natural ordering, i.e. no reduction at all, and far worse than reverse
Cuthill–McKee. The seam theorem (`experiments/exp17_seam_closed_forms.py`)
proves the obstruction: adjacent cells can differ in index by
$\tfrac56 n^2$ in $d=2$ and $\tfrac{13}{14}n^3$ in $d=3$, verified exhaustively
to $p=12$ and $p=7$.

**Bug in this script.** Its output labelled "bandwidth" is computed by a
variable literally named `bw_avg`: it is the *mean* index gap, not the
bandwidth. The published table "1112/49/47" therefore reported a different
quantity from the one it claimed. Only the mean gap is genuinely reduced.

Also relevant: `experiments/exp02_hypotheses.py` shows the regime
$(h/\Delta)^d \ll N$ assumed by the bound is never entered on these meshes.

## `dpcm_compression.py` — the compression gain does not exist

Claimed an entropy advantage of $\tfrac12\log_2 n$ bits per sample for the
Hilbert traversal over a raster scan.

**Refuted by `experiments/exp07_R4_dpcm.py`.** On MNI152 the Hilbert traversal
*loses*: −0.168 bits, −2.72 %.

**Bug in this script.** It calls `np.round(x)` on a signal whose values lie in
$[0, 0.988]$, so the signal is reduced to **two levels** before any residual is
taken. The "original entropy 0.462 bits" it reported is the entropy of a binary
image. Corrected to 8-bit quantisation the sign of the result does not change.

A second defect: at $p=7$ the reference box gives ~2 mm cells and 14.1 % of
voxels collide, which accounts for about a third of the loss; the rest is
genuine.

## `interpatient_stability.py` — measures quantisation, not anatomy

Presented as showing that "the same anatomical landmark receives an identical
code across patients after MNI registration".

**Reinterpreted by `experiments/exp15_v4_carryover.py`.** The script uses no
inter-patient data whatsoever. It perturbs seven fixed landmark coordinates
with Gaussian noise and counts unchanged codes. Over $p \in \{5,\dots,8\}$ and
$\sigma \in \{0.25,\dots,4\}$ mm the measured rate agrees with the elementary
prediction that a point merely has to stay inside its own quantisation cell —
largest deviation 0.05, and below 0.01 for $\sigma \ge 2$ mm.

It is a property of the grid. Whether codes agree across registered subjects is
an open question that this repository does not answer.

## `surface_morphing.py` — the measurement stands, the theorem does not

This one needs a distinction. The *numerical* result — a deformation applied
along the Hilbert parameter is 22.9× smoother than under a random reordering —
was re-run and **confirmed**.

What was refuted is the *pointwise bound* the script was presented as
illustrating (Prop. 10.4 of the earlier paper). The script is here because the
statement it supported is gone, not because its measurement is wrong.
