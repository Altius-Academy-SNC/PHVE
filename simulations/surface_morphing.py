"""
Surface morphing via Hilbert reparameterisation (Proposition 10.4)

Demonstrates that a deformation defined in the 1D Hilbert parameter
space produces a spatially smooth 3D animation, thanks to locality
preservation of the Hilbert curve. By contrast, the same deformation
applied along a random reordering of the vertices produces spatial
noise.

Author: Paul Guindo, Altius Academy SNC.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

from codec3d import VOLUMES, xyz2d

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ===================================================================
# Surface extraction
# ===================================================================

def extract_brain_surface():
    """Marching-cubes brain surface from the MNI152 mask.

    Returns:
        verts_mm: (V, 3) vertices in MNI mm.
        faces  : (F, 3) triangle faces.
        normals: (V, 3) vertex normals.
    """
    from nilearn.datasets import load_mni152_brain_mask
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter

    print("  Loading brain mask...", end=" ", flush=True)
    mask_img = load_mni152_brain_mask(resolution=2)
    mask = np.asarray(mask_img.dataobj).astype(np.float32)
    affine = mask_img.affine
    print(f"OK")

    smooth = gaussian_filter(mask, sigma=1.5)
    verts_vox, faces, normals, _ = marching_cubes(smooth, level=0.5)

    ones = np.ones((verts_vox.shape[0], 1))
    verts_hom = np.hstack([verts_vox, ones])
    verts_mm = (verts_hom @ affine.T)[:, :3]

    print(f"  Surface : {verts_mm.shape[0]:,} vertices, {faces.shape[0]:,} faces")
    return verts_mm, faces, normals


def compute_vertex_hilbert_indices(verts_mm, p=5, volume="CR"):
    """Compute the Hilbert index Hil_p^{(3)} for every vertex.

    Returns:
        hilbert_indices    : (V,) raw integer index.
        hilbert_normalized : (V,) index normalised to [0, 1].
    """
    dims = VOLUMES[volume]["dims"]
    n = 1 << p
    max_d = 8 ** p - 1
    N = len(verts_mm)

    hilbert_indices = np.zeros(N, dtype=np.int64)

    print(f"  Computing Hilbert indices ({N:,} vertices, p={p})...", end=" ", flush=True)
    t0 = time.time()
    for i in range(N):
        x, y, z = verts_mm[i]
        vx = x + dims[0] / 2
        vy = y + dims[1] / 2
        vz = z + dims[2] / 2
        ix = max(0, min(n - 1, int(vx / dims[0] * n)))
        iy = max(0, min(n - 1, int(vy / dims[1] * n)))
        iz = max(0, min(n - 1, int(vz / dims[2] * n)))
        hilbert_indices[i] = xyz2d(p, ix, iy, iz)

    elapsed = time.time() - t0
    print(f"{elapsed:.1f}s")

    hilbert_normalized = hilbert_indices.astype(np.float64) / max_d
    return hilbert_indices, hilbert_normalized


# ===================================================================
# Deformation
# ===================================================================

def apply_hilbert_deformation(verts_mm, normals, hilbert_normalized,
                                amplitude=3.0, frequency=4.0, phase=0.0):
    """Sinusoidal deformation indexed by the Hilbert parameter h(v).

    Each vertex is displaced along its outward normal by
        d(v) = A * sin(2*pi*f * h(v) + phi),
    where h(v) is the normalised Hilbert index. Locality preservation
    of Hil_p^{(3)} (Theorem 5.2) ensures that nearby vertices receive
    nearby h-values, hence nearly identical displacements.
    """
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit_normals = normals / norms

    displacements = amplitude * np.sin(2 * np.pi * frequency * hilbert_normalized + phase)

    deformed = verts_mm + unit_normals * displacements[:, np.newaxis]
    return deformed, displacements


def apply_random_deformation(verts_mm, normals, amplitude=3.0, frequency=4.0,
                               phase=0.0, seed=42):
    """Same deformation as apply_hilbert_deformation but with a random index.

    The vertices are reindexed by a uniform random permutation, breaking
    locality. The resulting per-vertex displacement is spatial noise.
    """
    rng = np.random.RandomState(seed)
    N = len(verts_mm)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit_normals = normals / norms

    random_normalized = rng.permutation(N).astype(np.float64) / N

    displacements = amplitude * np.sin(2 * np.pi * frequency * random_normalized + phase)
    deformed = verts_mm + unit_normals * displacements[:, np.newaxis]
    return deformed, displacements


# ===================================================================
# Smoothness metric
# ===================================================================

def compute_smoothness(verts_mm, displacements, faces):
    """Per-edge displacement difference on the mesh.

    For each undirected edge (v1, v2) of the mesh, compute
    |d(v1) - d(v2)|. A smooth deformation has small edge differences.

    Returns:
        mean_diff : average difference (mm).
        max_diff  : worst-case difference (mm).
        std_diff  : standard deviation of differences.
        diffs     : (E,) array of all per-edge differences.
    """
    edges = set()
    for f in faces:
        for i in range(3):
            e = tuple(sorted([f[i], f[(i + 1) % 3]]))
            edges.add(e)

    diffs = []
    for v1, v2 in edges:
        diffs.append(abs(displacements[v1] - displacements[v2]))

    diffs = np.array(diffs)
    return diffs.mean(), diffs.max(), diffs.std(), diffs


# ===================================================================
# Figure
# ===================================================================

def plot_results(verts_mm, faces, normals, hilbert_normalized,
                 deformed_h, disp_h, deformed_r, disp_r,
                 smooth_h, smooth_r):
    """Render the 6-panel figure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Surface morphing via Hilbert reparameterisation (Prop. 10.4)\n"
                 "sinusoidal deformation: Hilbert (smooth) vs random (noise)",
                 fontsize=14, fontweight="bold")

    step = max(1, len(verts_mm) // 8000)
    idx = np.arange(0, len(verts_mm), step)

    ax = axes[0, 0]
    sc = ax.scatter(verts_mm[idx, 0], verts_mm[idx, 1],
                     c=hilbert_normalized[idx], cmap="hsv", s=1, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="normalised Hilbert index $h(v)$")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Original surface\ncoloured by $h(v) = \\mathrm{Hil}_p^{(3)}/(n^d-1)$")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    sc = ax.scatter(deformed_h[idx, 0], deformed_h[idx, 1],
                     c=disp_h[idx], cmap="RdBu", s=1, alpha=0.6,
                     vmin=-3, vmax=3)
    plt.colorbar(sc, ax=ax, label="displacement (mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Hilbert deformation\n(spatially smooth)")
    ax.set_aspect("equal")

    ax = axes[0, 2]
    sc = ax.scatter(deformed_r[idx, 0], deformed_r[idx, 1],
                     c=disp_r[idx], cmap="RdBu", s=1, alpha=0.6,
                     vmin=-3, vmax=3)
    plt.colorbar(sc, ax=ax, label="displacement (mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Random deformation\n(spatial noise)")
    ax.set_aspect("equal")

    ax = axes[1, 0]
    bins = np.linspace(0, 6, 60)
    ax.hist(smooth_h[3], bins=bins, alpha=0.6,
            label=f"Hilbert (mean={smooth_h[0]:.3f})", color="green")
    ax.hist(smooth_r[3], bins=bins, alpha=0.4,
            label=f"Random  (mean={smooth_r[0]:.3f})", color="red")
    ax.set_xlabel("Per-edge displacement difference (mm)")
    ax.set_ylabel("Edge count")
    ax.set_title("Mesh-edge smoothness of the deformation")
    ax.legend()

    ax = axes[1, 1]
    methods = ["Hilbert", "Random"]
    means = [smooth_h[0], smooth_r[0]]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(methods, means, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Mean neighbour difference (mm)")
    ratio = smooth_r[0] / smooth_h[0]
    ax.set_title(f"Deformation quality\nHilbert is {ratio:.1f}x smoother than random")

    ax = axes[1, 2]
    ax.axis("off")
    headers = ["Metric", "Hilbert", "Random", "Ratio"]
    rows = [
        ["Mean neighbour diff.", f"{smooth_h[0]:.4f} mm", f"{smooth_r[0]:.4f} mm",
         f"{smooth_r[0]/smooth_h[0]:.1f}x"],
        ["Max neighbour diff.", f"{smooth_h[1]:.3f} mm", f"{smooth_r[1]:.3f} mm",
         f"{smooth_r[1]/smooth_h[1]:.1f}x"],
        ["Std of diffs", f"{smooth_h[2]:.4f}", f"{smooth_r[2]:.4f}",
         f"{smooth_r[2]/smooth_h[2]:.1f}x"],
        ["Vertices", f"{len(verts_mm):,}", f"{len(verts_mm):,}", "--"],
        ["Faces", f"{len(faces):,}", f"{len(faces):,}", "--"],
    ]
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#9b59b6")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, 4):
        table[i, 1].set_facecolor("#d5f5e3")
    ax.set_title("Morphing summary", fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "surface_morphing.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  -> figure saved : {path}")
    return path


# ===================================================================
# Report
# ===================================================================

def write_report(verts_mm, faces, smooth_h, smooth_r):
    """Write the markdown report."""
    path = os.path.join(OUTPUT_DIR, "surface_morphing_report.md")

    lines = [
        "# Surface morphing via Hilbert reparameterisation (Proposition 10.4)",
        "",
        "## Context",
        "",
        "Proposition 10.4 of the paper bounds the per-edge displacement",
        "difference of a Lipschitz scalar deformation D applied along the",
        "Hilbert reparameterisation h(v) = Hil_p^{(d)}(v) / (n^d - 1):",
        "",
        "    |D(h_i) - D(h_j)| <= L_D * C_d / n^d * (l/Delta)^d",
        "",
        "for any two vertices v_i, v_j connected by an edge of length l on a",
        "grid of spacing Delta. By contrast, applying D along a uniform",
        "random reordering yields E[|D(r_i) - D(r_j)|] <= L_D / 3,",
        "*independent* of n and l.",
        "",
        "## Method",
        "",
        "1. Extract the brain surface from MNI152 by marching cubes.",
        f"2. Compute the Hilbert index for each vertex ({len(verts_mm):,} vertices).",
        "3. Apply a sinusoidal scalar deformation along the outward normal:",
        "   - **Hilbert** : d(v) = A * sin(2*pi*f * h(v) + phi).",
        "   - **Random**  : same deformation but with h(v) replaced by a uniform random index.",
        "4. Measure smoothness as the per-edge displacement difference on the mesh.",
        "",
        "## Results",
        "",
        "| Metric | Hilbert | Random | Ratio |",
        "|--------|---------|--------|-------|",
        f"| **Mean neighbour difference** | **{smooth_h[0]:.4f} mm** | {smooth_r[0]:.4f} mm | {smooth_r[0]/smooth_h[0]:.1f}x |",
        f"| Max neighbour difference | {smooth_h[1]:.3f} mm | {smooth_r[1]:.3f} mm | {smooth_r[1]/smooth_h[1]:.1f}x |",
        f"| Std of differences | {smooth_h[2]:.4f} | {smooth_r[2]:.4f} | {smooth_r[2]/smooth_h[2]:.1f}x |",
        "",
        "## Interpretation",
        "",
        "- A deformation defined in the 1D Hilbert parameter space produces",
        "  a spatially smooth animation: neighbouring mesh vertices receive",
        "  almost identical displacements.",
        "- The same deformation applied along a random reordering yields",
        "  spatial noise: neighbouring vertices have very different displacements.",
        f"- Hilbert is **{smooth_r[0]/smooth_h[0]:.1f}x smoother** than the random ordering.",
        "",
        "## Conclusion",
        "",
        "The Hilbert traversal is a natural 1D parameterisation for surface",
        "animation. Any continuous 1D function lifts to a spatially coherent",
        "3D deformation thanks to locality preservation of Hil_p^{(3)}.",
        "",
        "## Files",
        "",
        "- Script : `surface_morphing.py`",
        "- Figure : `results/surface_morphing.png`",
        "- Report : `results/surface_morphing_report.md`",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> report saved : {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 65)
    print("  Surface morphing via Hilbert reparameterisation (Prop. 10.4)")
    print("  Altius Academy SNC")
    print("=" * 65)

    p = 5

    print("\n[1/5] Extracting brain surface...")
    verts_mm, faces, normals = extract_brain_surface()

    print("\n[2/5] Computing Hilbert indices...")
    hilbert_indices, hilbert_normalized = compute_vertex_hilbert_indices(verts_mm, p=p)

    print("\n[3/5] Applying deformations...")
    deformed_h, disp_h = apply_hilbert_deformation(
        verts_mm, normals, hilbert_normalized,
        amplitude=3.0, frequency=4.0, phase=0.0
    )
    deformed_r, disp_r = apply_random_deformation(
        verts_mm, normals,
        amplitude=3.0, frequency=4.0, phase=0.0
    )
    print(f"  Hilbert : displacement [{disp_h.min():.2f}, {disp_h.max():.2f}] mm")
    print(f"  Random  : displacement [{disp_r.min():.2f}, {disp_r.max():.2f}] mm")

    print("\n[4/5] Measuring smoothness...")
    smooth_h = compute_smoothness(verts_mm, disp_h, faces)
    smooth_r = compute_smoothness(verts_mm, disp_r, faces)
    print(f"  Hilbert : mean neighbour diff. = {smooth_h[0]:.4f} mm")
    print(f"  Random  : mean neighbour diff. = {smooth_r[0]:.4f} mm")
    print(f"  Ratio   : {smooth_r[0]/smooth_h[0]:.1f}x smoother with Hilbert")

    print("\n[5/5] Generating figure and report...")
    plot_results(verts_mm, faces, normals, hilbert_normalized,
                 deformed_h, disp_h, deformed_r, disp_r,
                 smooth_h, smooth_r)
    write_report(verts_mm, faces, smooth_h, smooth_r)

    print(f"\n{'=' * 65}")
    print(f"  surface_morphing done.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
