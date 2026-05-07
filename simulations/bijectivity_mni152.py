"""
PHVE 3D bijectivity on MNI152 (Theorem 4.3 of the paper)

Loads the MNI152 T1-weighted template via nilearn, applies the PHVE
encoding map F_p^{(3),alpha} to every brain voxel, and produces
visualisations of:
  1. The MRI volume with overlaid Hilbert index Hil_p^{(3)}.
  2. The truncation hierarchy (multi-scale zoom).
  3. Prefix queries (anatomical region selection).
  4. Locality preservation (nearby voxels => nearby codes).

Author: Paul Guindo, Altius Academy SNC.
"""

import os
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from codec3d import (
    VOLUMES, xyz2d, d2xyz, encode, decode, truncate,
    _code_length_3d, _int_to_base29, _canonical_precision_3d,
    verify_bijectivity,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_mni152():
    """Load the MNI152 T1w template via nilearn."""
    from nilearn.datasets import load_mni152_template
    print("  Loading MNI152 T1w (2 mm)...", end=" ", flush=True)
    img = load_mni152_template(resolution=2)
    data = np.asarray(img.dataobj)
    affine = img.affine
    print(f"OK -- shape={data.shape}, voxel={img.header.get_zooms()[:3]} mm")
    return data, affine, img


def load_mni152_brain_mask():
    """Load the MNI152 brain mask."""
    from nilearn.datasets import load_mni152_brain_mask
    print("  Loading brain mask...", end=" ", flush=True)
    mask_img = load_mni152_brain_mask(resolution=2)
    mask = np.asarray(mask_img.dataobj).astype(bool)
    print(f"OK -- {mask.sum()} brain voxels")
    return mask


def voxel_ijk_to_mm(i, j, k, affine):
    """Voxel indices -> MNI mm coordinates."""
    ijk1 = np.array([i, j, k, 1.0])
    return (affine @ ijk1)[:3]


def mm_to_phve(x_mm, y_mm, z_mm, p=6, volume="CR"):
    """MNI coordinates (mm) -> PHVE 3D code.

    The CR cranial volume is centred on the MNI origin; we shift to [0, dims].
    """
    dims = VOLUMES[volume]["dims"]
    x = x_mm + dims[0] / 2
    y = y_mm + dims[1] / 2
    z = z_mm + dims[2] / 2
    return encode(x, y, z, p=p, volume=volume)


def compute_hilbert_index_volume(shape, affine, p=6, volume="CR"):
    """Hilbert index Hil_p^{(3)} for every voxel, normalised to [0, 1]."""
    dims = VOLUMES[volume]["dims"]
    n = 1 << p
    max_d = 8 ** p - 1

    hilbert_vol = np.zeros(shape, dtype=np.float32)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                xyz = voxel_ijk_to_mm(i, j, k, affine)
                ix = max(0, min(n - 1, int((xyz[0] + dims[0]/2) / dims[0] * n)))
                iy = max(0, min(n - 1, int((xyz[1] + dims[1]/2) / dims[1] * n)))
                iz = max(0, min(n - 1, int((xyz[2] + dims[2]/2) / dims[2] * n)))
                d = xyz2d(p, ix, iy, iz)
                hilbert_vol[i, j, k] = d / max_d

    return hilbert_vol


def demo_encode_decode(affine):
    """Encode/decode a handful of well-known anatomical landmarks."""
    print("\n  Encoding anatomical landmarks (MNI space):")

    landmarks = {
        "Brain centre":        (0, 0, 0),
        "Frontal cortex":      (0, 50, 30),
        "Occipital cortex":    (0, -90, 0),
        "Left hippocampus":    (-25, -20, -15),
        "Right hippocampus":   (25, -20, -15),
        "Cerebellum":          (0, -60, -35),
        "Brain stem":          (0, -30, -30),
    }

    results = {}
    for name, (x, y, z) in landmarks.items():
        code = mm_to_phve(x, y, z, p=8, volume="CR")
        decoded = decode(code)
        results[name] = code
        print(f"    {name:25s} ({x:+4d}, {y:+4d}, {z:+4d}) mm -> {code}")

    code_l = results["Left hippocampus"]
    code_r = results["Right hippocampus"]
    raw_l = code_l.split(":")[1].replace("-", "")
    raw_r = code_r.split(":")[1].replace("-", "")
    common = 0
    for a, b in zip(raw_l, raw_r):
        if a == b:
            common += 1
        else:
            break
    print(f"\n    Left/right hippocampi: common prefix = {common}/{len(raw_l)} chars")
    print( "    Note: the two hippocampi sit on opposite sides of the median")
    print( "    plane and therefore land in different top-level Hilbert octants.")
    print( "    Cor. 5.3 (prefix => proximity) is one-directional: nearby points")
    print( "    do NOT necessarily share a long prefix when they straddle a")
    print( "    high-level subdivision boundary.")

    return results


def demo_hierarchy(affine):
    """Truncation hierarchy = multi-scale zoom (Theorem 6.1)."""
    print("\n  Truncation hierarchy (frontal cortex):")

    code = mm_to_phve(0, 50, 30, p=8, volume="CR")
    decoded = decode(code)
    print(f"    Full code  : {code}  (resolution {decoded['resolution_mm']:.1f} mm)")

    for i in range(1, 5):
        t = truncate(code, i)
        if ":" in t and t.split(":")[1]:
            raw = t.split(":")[1].replace("-", "")
            p_trunc = _canonical_precision_3d(len(raw))
            n_trunc = 1 << p_trunc
            dims = VOLUMES["CR"]["dims"]
            res = max(dims[0]/n_trunc, dims[1]/n_trunc, dims[2]/n_trunc)
            print(f"    Truncated -{i}: {t:20s}  (resolution ~{res:.0f} mm)")
        else:
            print(f"    Truncated -{i}: {t:20s}  (whole volume)")


def demo_prefix_search(data, affine, mask):
    """Prefix search = anatomical region selection."""
    print("\n  Prefix search (region selection):")

    target_code = mm_to_phve(0, 0, 0, p=6, volume="CR")
    target_raw = target_code.split(":")[1].replace("-", "")
    prefix = target_raw[:2]
    print(f"    Target code   : {target_code}")
    print(f"    Search prefix : CR:{prefix}...")

    dims = VOLUMES["CR"]["dims"]
    n = 1 << 6
    k = _code_length_3d(6)
    match_count = 0
    total_brain = 0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k_idx in range(data.shape[2]):
                if not mask[i, j, k_idx]:
                    continue
                total_brain += 1
                xyz = voxel_ijk_to_mm(i, j, k_idx, affine)
                ix = max(0, min(n-1, int((xyz[0] + dims[0]/2) / dims[0] * n)))
                iy = max(0, min(n-1, int((xyz[1] + dims[1]/2) / dims[1] * n)))
                iz = max(0, min(n-1, int((xyz[2] + dims[2]/2) / dims[2] * n)))
                d = xyz2d(6, ix, iy, iz)
                code = _int_to_base29(d, _code_length_3d(6))
                if code.startswith(prefix):
                    match_count += 1

    pct = match_count / total_brain * 100
    print(f"    Matching voxels : {match_count}/{total_brain} ({pct:.1f}%)")
    print(f"    -> SQL query    : WHERE code LIKE 'CR:{prefix}%'")


def plot_results(data, affine, mask):
    """Render the figure."""
    print("\n[+] Generating figure...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("PHVE 3D encoding on MNI152 -- bijectivity and locality (Thm. 4.3, Cor. 5.3)",
                 fontsize=14, fontweight="bold")

    mid_x = data.shape[0] // 2
    mid_y = data.shape[1] // 2
    mid_z = data.shape[2] // 2

    ax = axes[0, 0]
    ax.imshow(np.rot90(data[mid_x, :, :]), cmap="gray", aspect="auto")
    ax.set_title("MNI152 T1w -- sagittal slice (2 mm)")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")

    ax = axes[0, 1]
    p = 5
    dims = VOLUMES["CR"]["dims"]
    n = 1 << p
    max_d = 8 ** p - 1

    hilbert_slice = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            xyz = voxel_ijk_to_mm(i, j, mid_z, affine)
            ix = max(0, min(n-1, int((xyz[0] + dims[0]/2) / dims[0] * n)))
            iy = max(0, min(n-1, int((xyz[1] + dims[1]/2) / dims[1] * n)))
            iz = max(0, min(n-1, int((xyz[2] + dims[2]/2) / dims[2] * n)))
            d = xyz2d(p, ix, iy, iz)
            hilbert_slice[i, j] = d / max_d

    masked_hilbert = np.ma.masked_where(~mask[:, :, mid_z], hilbert_slice)
    ax.imshow(np.rot90(data[:, :, mid_z]), cmap="gray", alpha=0.3, aspect="auto")
    im = ax.imshow(np.rot90(masked_hilbert), cmap="hsv", alpha=0.7, aspect="auto")
    ax.set_title(f"Hilbert index $\\mathrm{{Hil}}_p^{{(3)}}$ ($p={p}$)\naxial slice -- colour = position along the curve")
    plt.colorbar(im, ax=ax, label="normalised Hilbert index")

    ax = axes[0, 2]
    ax.axis("off")
    headers = ["Level", "Code", "Resolution"]
    code_full = mm_to_phve(0, 50, 30, p=8, volume="CR")
    rows = []
    for i in range(5):
        if i == 0:
            c = code_full
        else:
            c = truncate(code_full, i)
        raw = c.split(":")[1].replace("-", "") if ":" in c and c.split(":")[1] else ""
        if raw:
            p_t = _canonical_precision_3d(len(raw))
            n_t = 1 << p_t
            res = max(dims[0]/n_t, dims[1]/n_t, dims[2]/n_t)
            rows.append([f"p={p_t}", c, f"~{res:.0f} mm"])
        else:
            rows.append(["p=0", c, "whole volume"])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#3498db")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Truncation hierarchy\n(frontal cortex)", fontsize=11, fontweight="bold", pad=20)

    ax = axes[1, 0]
    ax.imshow(np.rot90(data[:, mid_y, :]), cmap="gray", aspect="auto")
    ax.set_title("MNI152 T1w -- coronal slice")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")

    ax = axes[1, 1]
    prefix_len = 2
    region_map = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
    k_code = _code_length_3d(p)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not mask[i, j, mid_z]:
                continue
            xyz = voxel_ijk_to_mm(i, j, mid_z, affine)
            ix = max(0, min(n-1, int((xyz[0] + dims[0]/2) / dims[0] * n)))
            iy = max(0, min(n-1, int((xyz[1] + dims[1]/2) / dims[1] * n)))
            iz = max(0, min(n-1, int((xyz[2] + dims[2]/2) / dims[2] * n)))
            d = xyz2d(p, ix, iy, iz)
            code = _int_to_base29(d, k_code)
            prefix_val = hash(code[:prefix_len]) % 256
            region_map[i, j] = prefix_val

    masked_regions = np.ma.masked_where(~mask[:, :, mid_z], region_map)
    ax.imshow(np.rot90(data[:, :, mid_z]), cmap="gray", alpha=0.3, aspect="auto")
    ax.imshow(np.rot90(masked_regions), cmap="tab20", alpha=0.6, aspect="auto")
    ax.set_title(f"Regions by prefix ({prefix_len} chars)\nsame colour = same prefix = same region")

    ax = axes[1, 2]
    ax.axis("off")
    landmarks = {
        "Brain centre":      (0, 0, 0),
        "Frontal cortex":    (0, 50, 30),
        "Occipital cortex":  (0, -90, 0),
        "L. hippocampus":    (-25, -20, -15),
        "R. hippocampus":    (25, -20, -15),
        "Cerebellum":        (0, -60, -35),
        "Brain stem":        (0, -30, -30),
    }
    headers = ["Landmark", "MNI (mm)", "PHVE code"]
    rows = []
    for name, (x, y, z) in landmarks.items():
        code = mm_to_phve(x, y, z, p=8, volume="CR")
        rows.append([name, f"({x:+d},{y:+d},{z:+d})", code])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.6)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2ecc71")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Encoded anatomical landmarks\n(p=8, volume CR)", fontsize=11,
                 fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bijectivity_mni152.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


def main():
    print("=" * 65)
    print("  PHVE 3D bijectivity on MNI152 (Theorem 4.3)")
    print("  Altius Academy SNC")
    print("=" * 65)

    print("\n[1/5] Bijectivity check (p=3)...", end=" ", flush=True)
    ok, msg = verify_bijectivity(3)
    print(msg)
    if not ok:
        print("  ERROR: bijectivity failed, aborting.")
        return

    print("\n[2/5] Loading MNI152 template...")
    data, affine, img = load_mni152()
    mask = load_mni152_brain_mask()

    print("\n[3/5] Encode/decode demonstration...")
    demo_encode_decode(affine)

    print("\n[4/5] Hierarchy demonstration...")
    demo_hierarchy(affine)

    print("\n[5/5] Visualisations...")
    plot_results(data, affine, mask)

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"\n  MRI volume    : MNI152 T1w, {data.shape}, 2 mm isotropic")
    print(f"  Brain voxels  : {mask.sum()}")
    print(f"  Codec         : Hilbert (Skilling 2004), bijective")
    print(f"  Alphabet      : base-29, unambiguous (29 characters)")
    print(f"  Volumes       : {', '.join(VOLUMES.keys())}")
    print(f"\n  The same brain voxel is identified by:")
    print(f"    - MNI coords  : (0.0, 50.0, 30.0) mm")
    code = mm_to_phve(0, 50, 30, p=8, volume="CR")
    print(f"    - PHVE code   : {code}")
    print(f"    - SQL query   : WHERE code LIKE 'CR:{code.split(':')[1][:3]}%'")

    print(f"\n{'=' * 65}")
    print(f"  bijectivity_mni152 done.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
