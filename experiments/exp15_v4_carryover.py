#!/usr/bin/env python3
"""
EXPERIMENT 15 -- Re-examination of the two results carried over from v4
without verification (closes U10).

U10 listed two claims inherited from v4 and never re-run:

  (i)  "Bijectivity on MNI152: 235 375 voxels, 0 collisions at p = 8";
  (ii) "Inter-patient stability: 49 % identical codes at p = 6 under
        simulated 0.5 mm jitter".

Re-reading `simulations/bijectivity_mni152.py` shows that **it never
performs the test in (i)**.  It checks bijectivity on 512 points at p = 3,
then encodes seven anatomical landmarks at p = 8 and draws a figure.  The
number 235 375 in its output is the count of brain voxels in the mask, not
the outcome of a collision test.  Claim (i) therefore had no computation
behind it at all.

This script performs the missing test properly and characterises (ii).

(A) Collision test.  Encode the centre of *every* voxel of the MNI152
    brain mask, in the reference volume CR, at p = 4..10, and count
    genuine collisions -- distinct voxels receiving the same code.  This is
    the quantity the claim is about.  The theoretical cell size is reported
    alongside so the outcome is predictable rather than surprising: the
    encoding is injective on the voxel grid exactly when the cell is
    smaller than the voxel spacing along every axis.

(B) The "inter-patient" protocol, re-run and described for what it is.
    `simulations/interpatient_stability.py` does not use inter-patient
    data.  It perturbs seven fixed landmark coordinates with Gaussian noise
    of standard deviation sigma and counts how often the code is unchanged.
    That measures the probability that a jitter of size sigma keeps a point
    inside its own quantisation cell -- a property of the grid, not of
    anatomy or of registration across subjects.  It is re-run here over a
    range of sigma and p, and the measured rate is compared with the
    elementary prediction for a cell of edge Delta_i,

        P(code unchanged) = prod_i E[ 1{ x_i + sigma Z_i stays in its cell } ],

    estimated by the same Monte-Carlo draw.  If the two agree, the
    experiment is confirmed to be a quantisation measurement and nothing
    more, and the paper must not present it as evidence about patients.

Output: results/exp15_v4_carryover.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "simulations"))

from phve import model                                            # noqa: E402
from common import hardware_record, save_json                     # noqa: E402
import codec3d                                                     # noqa: E402

LANDMARKS = {
    "Brain centre": (0, 0, 0),
    "Frontal cortex": (0, 50, 30),
    "Occipital cortex": (0, -90, 0),
    "L. hippocampus": (-25, -20, -15),
    "R. hippocampus": (25, -20, -15),
    "Cerebellum": (0, -60, -35),
    "Brain stem": (0, -30, -30),
}


def mm_to_grid(xyz_mm, p, volume="CR"):
    """Integer cell of the PHVE grid, vectorised (mirrors codec3d.encode)."""
    dims = np.asarray(codec3d.VOLUMES[volume]["dims"], dtype=float)
    n = 1 << p
    shifted = np.asarray(xyz_mm, dtype=float) + dims / 2.0
    g = np.floor(shifted / dims * n).astype(np.int64)
    return np.clip(g, 0, n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=str, default="CR")
    ap.add_argument("--orders", type=int, nargs="+",
                    default=[4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--jitter-orders", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--n-patients", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = {"experiment": "exp15_v4_carryover", "params": vars(args),
           "hardware": hardware_record()}

    # ------------------------------------------------------------------
    # (A) The collision test that was never run
    # ------------------------------------------------------------------
    print("=== (A) MNI152 collision test on the brain mask ===")
    from nilearn.datasets import load_mni152_brain_mask
    vol = model.load_mni152(2)
    affine = vol.affine
    mask_img = load_mni152_brain_mask(resolution=2)
    mask = np.asarray(mask_img.dataobj).astype(bool)
    idx = np.argwhere(mask)
    vox_mm = idx @ affine[:3, :3].T + affine[:3, 3]
    n_vox = int(idx.shape[0])
    spacing = np.abs(np.diag(affine)[:3])
    dims = np.asarray(codec3d.VOLUMES[args.volume]["dims"], dtype=float)
    print("   brain-mask voxels: %d   voxel spacing: %s mm   volume %s dims %s mm"
          % (n_vox, spacing.tolist(), args.volume, dims.tolist()))

    # Voxels outside the reference box are *clipped* by codec3d.encode, which
    # makes them collide for reasons that have nothing to do with the
    # encoding's injectivity.  They must be separated out, not counted as
    # collisions.
    lo = -dims / 2.0
    hi = dims / 2.0
    inside = np.all((vox_mm >= lo) & (vox_mm < hi), axis=1)
    n_out = int((~inside).sum())
    print("   voxels outside the %s box (clipped by encode): %d (%.2f %%)"
          % (args.volume, n_out, 100.0 * n_out / n_vox))
    if n_out:
        oob = vox_mm[~inside]
        print("   their mm range: x %s  y %s  z %s"
              % (np.round([oob[:, 0].min(), oob[:, 0].max()], 1).tolist(),
                 np.round([oob[:, 1].min(), oob[:, 1].max()], 1).tolist(),
                 np.round([oob[:, 2].min(), oob[:, 2].max()], 1).tolist()))

    rows = []
    for p in args.orders:
        cell = dims / (1 << p)
        r = {"p": p, "cell_mm": cell.tolist(),
             "cell_smaller_than_voxel": bool(np.all(cell < spacing))}
        for label, sel in (("all", np.ones(n_vox, bool)), ("inside_box", inside)):
            g = mm_to_grid(vox_mm[sel], p, args.volume)
            key = (g[:, 0].astype(np.int64) << (2 * p)) \
                | (g[:, 1].astype(np.int64) << p) | g[:, 2].astype(np.int64)
            uniq, counts = np.unique(key, return_counts=True)
            m = int(sel.sum())
            r[label] = {"n_voxels": m, "n_distinct_codes": int(uniq.size),
                        "n_colliding_voxels": int(m - uniq.size),
                        "collision_rate": (m - uniq.size) / max(m, 1),
                        "max_multiplicity": int(counts.max())}
        rows.append(r)
        print("   p=%2d  cell %-24s | all: %6d colliding (%.2f %%) | inside box:"
              " %6d colliding (%.2f %%)  max mult %d"
              % (p, str(np.round(cell, 3).tolist()),
                 r["all"]["n_colliding_voxels"],
                 100 * r["all"]["collision_rate"],
                 r["inside_box"]["n_colliding_voxels"],
                 100 * r["inside_box"]["collision_rate"],
                 r["inside_box"]["max_multiplicity"]))
        sys.stdout.flush()
    out["mni152_collisions"] = {
        "rows": rows, "voxel_spacing_mm": spacing.tolist(),
        "volume": args.volume, "dims_mm": dims.tolist(),
        "n_voxels": n_vox, "n_outside_box": n_out,
        "mask": "nilearn load_mni152_brain_mask(resolution=2)"}

    zero = [r["p"] for r in rows if r["inside_box"]["n_colliding_voxels"] == 0]
    print("   -> zero collisions among in-box voxels from p = %s onwards"
          % (min(zero) if zero else "never in the tested range"))
    out["mni152_collisions"]["first_p_without_collision_inside_box"] = \
        min(zero) if zero else None

    # ------------------------------------------------------------------
    # (B) What the "inter-patient" experiment actually measures
    # ------------------------------------------------------------------
    print("\n=== (B) The jitter protocol, re-run and characterised ===")
    rng = np.random.default_rng(args.seed)
    jrows = []
    for p in args.jitter_orders:
        cell = dims / (1 << p)
        for sigma in args.sigmas:
            same_total = 0
            n_total = 0
            per_landmark = {}
            for name, xyz in LANDMARKS.items():
                base = np.asarray(xyz, dtype=float)
                ref = mm_to_grid(base[None, :], p, args.volume)[0]
                noise = rng.normal(0.0, sigma, size=(args.n_patients, 3))
                g = mm_to_grid(base[None, :] + noise, p, args.volume)
                same = int(np.all(g == ref[None, :], axis=1).sum())
                per_landmark[name] = same / args.n_patients
                same_total += same
                n_total += args.n_patients
            rate = same_total / n_total
            # elementary prediction: the point must stay in its own cell
            # along every axis, independently
            pred = 1.0
            for i in range(3):
                # position of the landmark inside its cell along axis i
                lo = np.array([mm_to_grid(np.asarray(v, float)[None, :], p,
                                          args.volume)[0][i]
                               for v in LANDMARKS.values()], dtype=float)
                offs = []
                for v in LANDMARKS.values():
                    x = float(v[i]) + dims[i] / 2.0
                    c = np.floor(x / dims[i] * (1 << p))
                    offs.append(x - c * cell[i])
                offs = np.asarray(offs)
                z = rng.normal(0.0, sigma, size=(args.n_patients, offs.size))
                inside = ((offs[None, :] + z >= 0) & (offs[None, :] + z < cell[i]))
                pred *= float(inside.mean())
            jrows.append({"p": p, "sigma_mm": sigma, "cell_mm": cell.tolist(),
                          "rate_identical": rate,
                          "independent_axis_prediction": pred,
                          "per_landmark": per_landmark})
            print("   p=%d sigma=%.2f mm  cell %s  identical %.3f  "
                  "axis-independent prediction %.3f"
                  % (p, sigma, np.round(cell, 2).tolist(), rate, pred))
            sys.stdout.flush()
    out["jitter_protocol"] = jrows

    dev = max(abs(r["rate_identical"] - r["independent_axis_prediction"])
              for r in jrows)
    print("   -> largest deviation from the pure-quantisation prediction: %.4f"
          % dev)
    out["jitter_max_deviation_from_quantisation_model"] = dev

    save_json(out, "exp15_v4_carryover.json")


if __name__ == "__main__":
    main()
