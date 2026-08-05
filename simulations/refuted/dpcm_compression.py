"""
DPCM compression along the Hilbert traversal (Theorem 8.2 of the paper)

Benchmarks differential pulse-code modulation (DPCM) on the MNI152 T1w
template. Measures the Shannon entropy of the residual signal r^gamma
for three traversals gamma:
  1. Raster (row-major, C order)              -- gamma_R
  2. Z-order (Morton curve)                   -- gamma_M
  3. Hilbert curve (Skilling 3D)              -- gamma_H

Theorem 8.2 predicts that the empirical second moment M_2(r^{gamma_H}) is
bounded by L^2 for L-Lipschitz signals (independent of grid size n),
whereas the raster bound grows as L^2 * n. Remark 8.4 from the paper
expects the same trend for the empirical variance and hence for the
Shannon entropy of the residual signal under a Gaussian model.

Author: Paul Guindo, Altius Academy SNC.
"""

import os
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from collections import Counter

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from codec3d import VOLUMES, xyz2d, d2xyz

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ===================================================================
# Helpers
# ===================================================================

def shannon_entropy(signal):
    """Shannon entropy (bits) of a discrete signal."""
    counts = Counter(signal)
    total = len(signal)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def dpcm_encode(signal):
    """DPCM encoding: consecutive differences r_i = f(gamma(i)) - f(gamma(i-1))."""
    return np.diff(signal.astype(np.int32))


def morton_index_3d(x, y, z):
    """Z-order (Morton) index for (x, y, z)."""
    d = 0
    for i in range(16):
        d |= ((x >> i) & 1) << (3 * i + 2)
        d |= ((y >> i) & 1) << (3 * i + 1)
        d |= ((z >> i) & 1) << (3 * i)
    return d


# ===================================================================
# MNI152 loading
# ===================================================================

def load_mni152(resolution=2):
    """Load the MNI152 T1w template via nilearn."""
    from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
    print(f"  Loading MNI152 T1w ({resolution} mm)...", end=" ", flush=True)
    img = load_mni152_template(resolution=resolution)
    data = np.asarray(img.dataobj)
    affine = img.affine
    print(f"OK -- shape={data.shape}")

    print(f"  Loading brain mask...", end=" ", flush=True)
    mask_img = load_mni152_brain_mask(resolution=resolution)
    mask = np.asarray(mask_img.dataobj).astype(bool)
    print(f"OK -- {mask.sum()} brain voxels")

    return data, mask, affine


# ===================================================================
# 1D signals along each traversal
# ===================================================================

def extract_raster(data, mask=None):
    """Intensities in raster (row-major) order."""
    if mask is not None:
        return data[mask].astype(np.float64)
    return data.flatten().astype(np.float64)


def extract_hilbert(data, mask, affine, p):
    """Intensities of brain voxels in Hilbert order Hil_p^{(3)}.

    For each masked voxel, compute the Hilbert index, then sort
    intensities by ascending index.
    """
    dims = VOLUMES["CR"]["dims"]
    n = 1 << p
    brain_coords = np.argwhere(mask)
    total = len(brain_coords)
    print(f"  Computing Hilbert indices for {total} voxels (p={p})...")

    hilbert_indices = np.zeros(total, dtype=np.int64)
    intensities = np.zeros(total, dtype=np.float64)

    t0 = time.time()
    for idx, (i, j, k) in enumerate(brain_coords):
        ijk1 = np.array([i, j, k, 1.0])
        xyz_mm = affine @ ijk1

        ix = max(0, min(n - 1, int((xyz_mm[0] + dims[0] / 2) / dims[0] * n)))
        iy = max(0, min(n - 1, int((xyz_mm[1] + dims[1] / 2) / dims[1] * n)))
        iz = max(0, min(n - 1, int((xyz_mm[2] + dims[2] / 2) / dims[2] * n)))

        hilbert_indices[idx] = xyz2d(p, ix, iy, iz)
        intensities[idx] = data[i, j, k]

        if (idx + 1) % 50000 == 0:
            elapsed = time.time() - t0
            pct = (idx + 1) / total * 100
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"    {idx+1}/{total} ({pct:.0f}%) -- ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s")

    order = np.argsort(hilbert_indices)
    return intensities[order]


def extract_morton(data, mask, affine, p):
    """Intensities of brain voxels in Z-order (Morton)."""
    dims = VOLUMES["CR"]["dims"]
    n = 1 << p
    brain_coords = np.argwhere(mask)
    total = len(brain_coords)
    print(f"  Computing Morton indices for {total} voxels...")

    morton_indices = np.zeros(total, dtype=np.int64)
    intensities = np.zeros(total, dtype=np.float64)

    t0 = time.time()
    for idx, (i, j, k) in enumerate(brain_coords):
        ijk1 = np.array([i, j, k, 1.0])
        xyz_mm = affine @ ijk1

        ix = max(0, min(n - 1, int((xyz_mm[0] + dims[0] / 2) / dims[0] * n)))
        iy = max(0, min(n - 1, int((xyz_mm[1] + dims[1] / 2) / dims[1] * n)))
        iz = max(0, min(n - 1, int((xyz_mm[2] + dims[2] / 2) / dims[2] * n)))

        morton_indices[idx] = morton_index_3d(ix, iy, iz)
        intensities[idx] = data[i, j, k]

    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s")

    order = np.argsort(morton_indices)
    return intensities[order]


def extract_full_volume_hilbert(data, p):
    """Whole-volume intensities in Hilbert order."""
    shape = data.shape
    n = 1 << p

    print(f"  Hilbert traversal of full volume ({shape} on grid {n}^3)...")
    indices = []
    intensities = []

    t0 = time.time()
    count = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                ix = min(n - 1, int(i / shape[0] * n))
                iy = min(n - 1, int(j / shape[1] * n))
                iz = min(n - 1, int(k / shape[2] * n))
                h = xyz2d(p, ix, iy, iz)
                indices.append(h)
                intensities.append(data[i, j, k])
                count += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            pct = (i + 1) / shape[0] * 100
            eta = elapsed / (i + 1) * (shape[0] - i - 1)
            print(f"    slice {i+1}/{shape[0]} ({pct:.0f}%) -- ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"    {count} voxels in {elapsed:.1f}s")

    indices = np.array(indices, dtype=np.int64)
    intensities = np.array(intensities, dtype=np.float64)
    order = np.argsort(indices)
    return intensities[order]


def extract_full_volume_morton(data, p):
    """Whole-volume intensities in Morton order."""
    shape = data.shape
    n = 1 << p

    print(f"  Morton traversal of full volume ({shape})...")
    indices = []
    intensities = []

    t0 = time.time()
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                ix = min(n - 1, int(i / shape[0] * n))
                iy = min(n - 1, int(j / shape[1] * n))
                iz = min(n - 1, int(k / shape[2] * n))
                m = morton_index_3d(ix, iy, iz)
                indices.append(m)
                intensities.append(data[i, j, k])

    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s")

    indices = np.array(indices, dtype=np.int64)
    intensities = np.array(intensities, dtype=np.float64)
    order = np.argsort(indices)
    return intensities[order]


# ===================================================================
# Benchmark
# ===================================================================

def run_benchmark(label, raster_signal, hilbert_signal, morton_signal):
    """Run the DPCM benchmark on the three ordered signals."""
    print(f"\n  {'='*55}")
    print(f"  DPCM benchmark -- {label}")
    print(f"  {'='*55}")

    raster_q = np.round(raster_signal).astype(np.int32)
    hilbert_q = np.round(hilbert_signal).astype(np.int32)
    morton_q = np.round(morton_signal).astype(np.int32)

    h_orig = shannon_entropy(raster_q)
    print(f"\n  Original signal: {len(raster_q)} samples, entropy = {h_orig:.3f} bits")

    dpcm_raster = dpcm_encode(raster_q)
    dpcm_hilbert = dpcm_encode(hilbert_q)
    dpcm_morton = dpcm_encode(morton_q)

    h_raster = shannon_entropy(dpcm_raster)
    h_hilbert = shannon_entropy(dpcm_hilbert)
    h_morton = shannon_entropy(dpcm_morton)

    red_hilbert_vs_raster = (1 - h_hilbert / h_raster) * 100
    red_morton_vs_raster = (1 - h_morton / h_raster) * 100
    red_hilbert_vs_morton = (1 - h_hilbert / h_morton) * 100

    print(f"\n  DPCM entropy:")
    print(f"    Raster  : {h_raster:.4f} bits")
    print(f"    Morton  : {h_morton:.4f} bits  ({red_morton_vs_raster:+.2f}% vs raster)")
    print(f"    Hilbert : {h_hilbert:.4f} bits  ({red_hilbert_vs_raster:+.2f}% vs raster)")
    print(f"    Hilbert vs Morton : {red_hilbert_vs_morton:+.2f}%")

    print(f"\n  DPCM residuals (|diff|):")
    for name, d in [("Raster", dpcm_raster), ("Morton", dpcm_morton), ("Hilbert", dpcm_hilbert)]:
        ad = np.abs(d)
        print(f"    {name:8s} : mean={ad.mean():.2f}, std={ad.std():.2f}, "
              f"median={np.median(ad):.0f}, max={ad.max()}, "
              f"zeros={np.sum(d==0)/len(d)*100:.1f}%")

    window = 64
    var_raster = np.array([dpcm_raster[i:i+window].var()
                           for i in range(0, len(dpcm_raster) - window, window)])
    var_hilbert = np.array([dpcm_hilbert[i:i+window].var()
                            for i in range(0, len(dpcm_hilbert) - window, window)])
    var_morton = np.array([dpcm_morton[i:i+window].var()
                           for i in range(0, len(dpcm_morton) - window, window)])

    print(f"\n  Local variance (window={window}):")
    print(f"    Raster  : mean={var_raster.mean():.1f}")
    print(f"    Morton  : mean={var_morton.mean():.1f}")
    print(f"    Hilbert : mean={var_hilbert.mean():.1f}")

    return {
        "label": label,
        "n_voxels": len(raster_q),
        "h_orig": h_orig,
        "h_raster": h_raster,
        "h_morton": h_morton,
        "h_hilbert": h_hilbert,
        "red_hilbert_vs_raster": red_hilbert_vs_raster,
        "red_morton_vs_raster": red_morton_vs_raster,
        "red_hilbert_vs_morton": red_hilbert_vs_morton,
        "dpcm_raster": dpcm_raster,
        "dpcm_hilbert": dpcm_hilbert,
        "dpcm_morton": dpcm_morton,
        "var_raster": var_raster,
        "var_hilbert": var_hilbert,
        "var_morton": var_morton,
    }


# ===================================================================
# Figure
# ===================================================================

def plot_results(results_brain, results_full):
    """Render the DPCM benchmark figure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("DPCM compression: Hilbert vs raster vs Morton (Thm. 8.2)\n"
                 "MNI152 T1w (2 mm)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    r = results_brain
    bins = np.arange(-200, 201, 5)
    ax.hist(r["dpcm_raster"], bins=bins, alpha=0.5, label="Raster $\\gamma_R$", density=True, color="C0")
    ax.hist(r["dpcm_hilbert"], bins=bins, alpha=0.5, label="Hilbert $\\gamma_H$", density=True, color="C2")
    ax.hist(r["dpcm_morton"], bins=bins, alpha=0.3, label="Morton $\\gamma_M$", density=True, color="C1")
    ax.set_xlabel("DPCM residual $r^\\gamma_i$")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of DPCM residuals\n(brain voxels)")
    ax.legend()
    ax.set_xlim(-200, 200)

    ax = axes[0, 1]
    methods = ["Raster", "Morton", "Hilbert"]
    entropies_brain = [r["h_raster"], r["h_morton"], r["h_hilbert"]]
    colors = ["#3498db", "#e67e22", "#2ecc71"]
    bars = ax.bar(methods, entropies_brain, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, entropies_brain):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title(f"DPCM entropy -- brain voxels\n"
                 f"Hilbert: {r['red_hilbert_vs_raster']:+.1f}% vs raster")
    ax.set_ylim(0, max(entropies_brain) * 1.15)

    ax = axes[0, 2]
    if results_full is not None:
        rf = results_full
        entropies_full = [rf["h_raster"], rf["h_morton"], rf["h_hilbert"]]
        bars = ax.bar(methods, entropies_full, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, entropies_full):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Shannon entropy (bits)")
        ax.set_title(f"DPCM entropy -- full volume\n"
                     f"Hilbert: {rf['red_hilbert_vs_raster']:+.1f}% vs raster")
        ax.set_ylim(0, max(entropies_full) * 1.15)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Full volume\n(skipped)", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)

    ax = axes[1, 0]
    x_axis = np.arange(min(len(r["var_raster"]), 2000))
    ax.plot(x_axis, r["var_raster"][:len(x_axis)], alpha=0.6, label="Raster", color="C0", linewidth=0.5)
    ax.plot(x_axis, r["var_morton"][:len(x_axis)], alpha=0.6, label="Morton", color="C1", linewidth=0.5)
    ax.plot(x_axis, r["var_hilbert"][:len(x_axis)], alpha=0.6, label="Hilbert", color="C2", linewidth=0.5)
    ax.set_xlabel("Position (windows of 64 samples)")
    ax.set_ylabel("Local variance of $r^\\gamma$")
    ax.set_title("Local variance of DPCM residuals")
    ax.legend()
    ax.set_yscale("log")

    ax = axes[1, 1]
    n_show = 2000
    ax.plot(r["dpcm_raster"][:n_show], alpha=0.6, label="Raster", color="C0", linewidth=0.3)
    ax.plot(r["dpcm_hilbert"][:n_show], alpha=0.6, label="Hilbert", color="C2", linewidth=0.3)
    ax.set_xlabel("Sample index $i$")
    ax.set_ylabel("DPCM residual $r^\\gamma_i$")
    ax.set_title(f"DPCM signal (first {n_show} samples)")
    ax.legend()

    ax = axes[1, 2]
    ax.axis("off")
    headers = ["Metric", "Raster", "Morton", "Hilbert"]
    rows = [
        ["Voxels", f"{r['n_voxels']:,}", f"{r['n_voxels']:,}", f"{r['n_voxels']:,}"],
        ["H(original)", f"{r['h_orig']:.3f}", f"{r['h_orig']:.3f}", f"{r['h_orig']:.3f}"],
        ["H(DPCM)", f"{r['h_raster']:.3f}", f"{r['h_morton']:.3f}", f"{r['h_hilbert']:.3f}"],
        ["Reduction vs raster", "--",
         f"{r['red_morton_vs_raster']:+.2f}%",
         f"{r['red_hilbert_vs_raster']:+.2f}%"],
        ["Mean |residual|",
         f"{np.abs(r['dpcm_raster']).mean():.1f}",
         f"{np.abs(r['dpcm_morton']).mean():.1f}",
         f"{np.abs(r['dpcm_hilbert']).mean():.1f}"],
        ["% zero residuals",
         f"{np.sum(r['dpcm_raster']==0)/len(r['dpcm_raster'])*100:.1f}%",
         f"{np.sum(r['dpcm_morton']==0)/len(r['dpcm_morton'])*100:.1f}%",
         f"{np.sum(r['dpcm_hilbert']==0)/len(r['dpcm_hilbert'])*100:.1f}%"],
    ]
    if results_full is not None:
        rf = results_full
        rows.append(["H(DPCM) full volume",
                      f"{rf['h_raster']:.3f}", f"{rf['h_morton']:.3f}", f"{rf['h_hilbert']:.3f}"])
        rows.append(["Reduction full vol.", "--",
                      f"{rf['red_morton_vs_raster']:+.2f}%",
                      f"{rf['red_hilbert_vs_raster']:+.2f}%"])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#8e44ad")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        if rows[i-1][0].startswith("Reduction"):
            table[i, 3].set_facecolor("#d5f5e3")
    ax.set_title("DPCM benchmark summary", fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "dpcm_compression.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  -> figure saved : {path}")
    return path


# ===================================================================
# Report
# ===================================================================

def write_report(results_brain, results_full):
    """Write the markdown report."""
    r = results_brain
    path = os.path.join(OUTPUT_DIR, "dpcm_compression_report.md")

    lines = [
        "# DPCM compression along the Hilbert traversal",
        "",
        "## Context",
        "",
        "Theorem 8.2 of the paper bounds the empirical second moment of the",
        "DPCM residual sequence along the Hilbert traversal by L^2",
        "(independent of grid size n), against L^2 * n for the raster",
        "traversal. Under a zero-mean Gaussian residual model (Section 10.2),",
        "this translates into a Shannon-entropy advantage of (1/2) log_2(n)",
        "bits per sample asymptotically.",
        "",
        "This script benchmarks DPCM compression on the MNI152 T1w (2 mm)",
        "template and reports a deliberately disclosed negative result on",
        "this near-binary atlas (see Remark 8.4).",
        "",
        "## Method",
        "",
        "1. Load MNI152 T1w (2 mm, 91x109x91 voxels).",
        "2. For each traversal gamma in {raster, Morton, Hilbert}:",
        "   - extract the intensities along gamma;",
        "   - compute the DPCM residual r^gamma = diff(signal);",
        "   - measure the Shannon entropy of r^gamma.",
        "3. Compare entropies.",
        "",
        f"- **Hilbert** : p=7 (grid 128^3), Skilling kernel (Algorithm 1).",
        f"- **Morton**  : Z-order (bit-interleaving).",
        f"- **Raster**  : row-major (C order).",
        "",
        "## Results -- brain voxels",
        "",
        f"| Metric | Raster | Morton | Hilbert |",
        f"|--------|--------|--------|---------|",
        f"| Voxels | {r['n_voxels']:,} | {r['n_voxels']:,} | {r['n_voxels']:,} |",
        f"| H(original) | {r['h_orig']:.3f} bits | {r['h_orig']:.3f} bits | {r['h_orig']:.3f} bits |",
        f"| **H(DPCM)** | **{r['h_raster']:.3f} bits** | **{r['h_morton']:.3f} bits** | **{r['h_hilbert']:.3f} bits** |",
        f"| Reduction vs raster | -- | {r['red_morton_vs_raster']:+.2f}% | **{r['red_hilbert_vs_raster']:+.2f}%** |",
        f"| Mean |residual| | {np.abs(r['dpcm_raster']).mean():.1f} | {np.abs(r['dpcm_morton']).mean():.1f} | {np.abs(r['dpcm_hilbert']).mean():.1f} |",
        f"| Zero residuals | {np.sum(r['dpcm_raster']==0)/len(r['dpcm_raster'])*100:.1f}% | {np.sum(r['dpcm_morton']==0)/len(r['dpcm_morton'])*100:.1f}% | {np.sum(r['dpcm_hilbert']==0)/len(r['dpcm_hilbert'])*100:.1f}% |",
        "",
    ]

    if results_full is not None:
        rf = results_full
        lines += [
            "## Results -- full volume",
            "",
            f"| Metric | Raster | Morton | Hilbert |",
            f"|--------|--------|--------|---------|",
            f"| Voxels | {rf['n_voxels']:,} | {rf['n_voxels']:,} | {rf['n_voxels']:,} |",
            f"| **H(DPCM)** | **{rf['h_raster']:.3f} bits** | **{rf['h_morton']:.3f} bits** | **{rf['h_hilbert']:.3f} bits** |",
            f"| Reduction vs raster | -- | {rf['red_morton_vs_raster']:+.2f}% | **{rf['red_hilbert_vs_raster']:+.2f}%** |",
            "",
        ]

    lines += [
        "## Interpretation",
        "",
        "- The Hilbert curve preserves spatial locality better than raster",
        "  and Z-order, producing smaller consecutive differences and a",
        "  lower DPCM entropy in well-behaved volumes.",
        "- On the heavily averaged MNI152 atlas, however, the original",
        "  signal is near-binary (H ~ 0.46 bits) and the high-magnitude",
        "  raster boundary jumps land in near-constant regions, so they do",
        "  not penalise raster as Theorem 8.2 would predict in the worst",
        "  case. This negative result is documented in Remark 8.4 of the",
        "  paper; reproducing the predicted gain on individual clinical",
        "  scans (full 256^3, 12-bit dynamic range, IXI/ACDC datasets) is",
        "  left as future work.",
        "",
        "## Conclusion",
        "",
        f"On the MNI152 atlas, Hilbert yields a {r['red_hilbert_vs_raster']:+.2f}% change in",
        f"DPCM entropy versus raster ({r['h_raster']:.3f} -> {r['h_hilbert']:.3f} bits).",
        f"The result is consistent with the empirical caveat noted in Remark 8.4.",
        "",
        "## Files",
        "",
        "- Script : `dpcm_compression.py`",
        "- Figure : `results/dpcm_compression.png`",
        "- Report : `results/dpcm_compression_report.md`",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> report saved : {path}")
    return path


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 65)
    print("  DPCM compression along the Hilbert traversal (Thm. 8.2)")
    print("  Altius Academy SNC")
    print("=" * 65)

    p = 7

    print("\n[1/6] Loading MNI152...")
    data, mask, affine = load_mni152(resolution=2)

    print("\n[2/6] Extracting raster signal (brain)...")
    raster_brain = extract_raster(data, mask)
    print(f"  {len(raster_brain)} brain voxels extracted")

    print("\n[3/6] Extracting Hilbert signal (brain)...")
    hilbert_brain = extract_hilbert(data, mask, affine, p)

    print("\n[4/6] Extracting Morton signal (brain)...")
    morton_brain = extract_morton(data, mask, affine, p)

    results_brain = run_benchmark("Brain voxels (MNI152, 2 mm)",
                                   raster_brain, hilbert_brain, morton_brain)

    print("\n[5/6] Full-volume benchmark...")
    raster_full = extract_raster(data, mask=None)
    hilbert_full = extract_full_volume_hilbert(data, p)
    morton_full = extract_full_volume_morton(data, p)
    results_full = run_benchmark("Full volume (91x109x91)",
                                  raster_full, hilbert_full, morton_full)

    print("\n[6/6] Generating figure and report...")
    plot_results(results_brain, results_full)
    write_report(results_brain, results_full)

    r = results_brain
    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"\n  Volume        : MNI152 T1w, 2 mm, {data.shape}")
    print(f"  Hilbert order : p={p} (grid {1<<p}^3)")
    print(f"  Brain voxels  : {r['n_voxels']:,}")
    print(f"\n  DPCM entropy (brain):")
    print(f"    Raster  = {r['h_raster']:.4f} bits")
    print(f"    Morton  = {r['h_morton']:.4f} bits ({r['red_morton_vs_raster']:+.2f}%)")
    print(f"    Hilbert = {r['h_hilbert']:.4f} bits ({r['red_hilbert_vs_raster']:+.2f}%)")
    if results_full:
        rf = results_full
        print(f"\n  DPCM entropy (full volume):")
        print(f"    Raster  = {rf['h_raster']:.4f} bits")
        print(f"    Hilbert = {rf['h_hilbert']:.4f} bits ({rf['red_hilbert_vs_raster']:+.2f}%)")
    print(f"\n{'=' * 65}")
    print(f"  dpcm_compression done.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
