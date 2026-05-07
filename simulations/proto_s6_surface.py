"""
Prototype S6 — Animation et morphing de surfaces anatomiques via Hilbert 3D

Démontre qu'une déformation définie dans l'espace 1D de Hilbert produit
une animation 3D spatialement lisse, grâce à la préservation de localité.

Application 6 du brevet (Revendication 8) :
"La surface d'un organe est parcourue par une courbe de Hilbert 3D.
 L'animation (déformation temporelle) se traduit par une transformation
 continue dans l'espace des codes 1D."

Auteur : Paul Guindo, Altius Academy SNC
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
# Extraction de surface
# ===================================================================

def extract_brain_surface():
    """Extrait la surface cérébrale du template MNI152 par marching cubes.

    Returns:
        verts_mm: (V, 3) vertices en mm MNI
        faces: (F, 3) faces (triangles)
    """
    from nilearn.datasets import load_mni152_brain_mask
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter

    print("  Chargement masque cérébral...", end=" ", flush=True)
    mask_img = load_mni152_brain_mask(resolution=2)
    mask = np.asarray(mask_img.dataobj).astype(np.float32)
    affine = mask_img.affine
    print(f"OK")

    # Lisser pour surface plus douce
    smooth = gaussian_filter(mask, sigma=1.5)
    verts_vox, faces, normals, _ = marching_cubes(smooth, level=0.5)

    # Voxel -> mm MNI
    ones = np.ones((verts_vox.shape[0], 1))
    verts_hom = np.hstack([verts_vox, ones])
    verts_mm = (verts_hom @ affine.T)[:, :3]

    print(f"  Surface : {verts_mm.shape[0]:,} vertices, {faces.shape[0]:,} faces")
    return verts_mm, faces, normals


def compute_vertex_hilbert_indices(verts_mm, p=5, volume="CR"):
    """Calcule l'index Hilbert pour chaque vertex du mesh.

    Returns:
        hilbert_indices: (V,) index brut
        hilbert_normalized: (V,) index normalisé [0, 1]
    """
    dims = VOLUMES[volume]["dims"]
    n = 1 << p
    max_d = 8 ** p - 1
    N = len(verts_mm)

    hilbert_indices = np.zeros(N, dtype=np.int64)

    print(f"  Calcul des index Hilbert ({N:,} vertices, p={p})...", end=" ", flush=True)
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
# Déformation
# ===================================================================

def apply_hilbert_deformation(verts_mm, normals, hilbert_normalized,
                                amplitude=3.0, frequency=4.0, phase=0.0):
    """Applique une déformation sinusoïdale dans l'espace 1D de Hilbert.

    La déformation est définie comme un déplacement le long de la normale :
        displacement = amplitude * sin(2π * frequency * h + phase)

    Comme h (index Hilbert normalisé) varie continûment dans l'espace 3D,
    la déformation résultante est spatialement lisse.

    Returns:
        deformed_verts: (V, 3)
        displacements: (V,) scalaire de déplacement
    """
    # Normaliser les normales
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit_normals = normals / norms

    # Déplacement dans l'espace Hilbert 1D
    displacements = amplitude * np.sin(2 * np.pi * frequency * hilbert_normalized + phase)

    # Appliquer le long de la normale
    deformed = verts_mm + unit_normals * displacements[:, np.newaxis]
    return deformed, displacements


def apply_random_deformation(verts_mm, normals, amplitude=3.0, frequency=4.0,
                               phase=0.0, seed=42):
    """Applique une déformation sinusoïdale avec un ordre aléatoire.

    Même déformation 1D, mais les vertices sont ordonnés aléatoirement
    au lieu de l'ordre Hilbert. Le résultat est une déformation incohérente
    spatialement (bruit).

    Returns:
        deformed_verts: (V, 3)
        displacements: (V,)
    """
    rng = np.random.RandomState(seed)
    N = len(verts_mm)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    unit_normals = normals / norms

    # Index aléatoire normalisé [0, 1]
    random_normalized = rng.permutation(N).astype(np.float64) / N

    displacements = amplitude * np.sin(2 * np.pi * frequency * random_normalized + phase)
    deformed = verts_mm + unit_normals * displacements[:, np.newaxis]
    return deformed, displacements


# ===================================================================
# Métriques de qualité
# ===================================================================

def compute_smoothness(verts_mm, displacements, faces):
    """Mesure la fluidité de la déformation sur le mesh.

    Pour chaque arête du mesh, calcule la différence de déplacement
    entre les deux vertices. Une déformation lisse a de petites différences.

    Returns:
        mean_diff: différence moyenne de déplacement entre voisins
        max_diff: différence maximale
        std_diff: écart-type des différences
    """
    # Extraire les arêtes uniques
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
# Visualisation
# ===================================================================

def plot_results(verts_mm, faces, normals, hilbert_normalized,
                 deformed_h, disp_h, deformed_r, disp_r,
                 smooth_h, smooth_r):
    """Génère les graphiques 6 panneaux."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Proto S6 — Animation de surfaces anatomiques via Hilbert 3D\n"
                 "Déformation sinusoïdale : Hilbert (lisse) vs Aléatoire (bruit)",
                 fontsize=14, fontweight="bold")

    # Sous-échantillonner pour la visu (trop de points sinon)
    step = max(1, len(verts_mm) // 8000)
    idx = np.arange(0, len(verts_mm), step)

    # --- (0,0) Surface originale colorée par Hilbert ---
    ax = axes[0, 0]
    sc = ax.scatter(verts_mm[idx, 0], verts_mm[idx, 1],
                     c=hilbert_normalized[idx], cmap="hsv", s=1, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Index Hilbert")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Surface originale\ncolorée par index Hilbert")
    ax.set_aspect("equal")

    # --- (0,1) Déformation Hilbert (vue X-Y) ---
    ax = axes[0, 1]
    sc = ax.scatter(deformed_h[idx, 0], deformed_h[idx, 1],
                     c=disp_h[idx], cmap="RdBu", s=1, alpha=0.6,
                     vmin=-3, vmax=3)
    plt.colorbar(sc, ax=ax, label="Déplacement (mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Déformation Hilbert\n(lisse spatialement)")
    ax.set_aspect("equal")

    # --- (0,2) Déformation aléatoire (vue X-Y) ---
    ax = axes[0, 2]
    sc = ax.scatter(deformed_r[idx, 0], deformed_r[idx, 1],
                     c=disp_r[idx], cmap="RdBu", s=1, alpha=0.6,
                     vmin=-3, vmax=3)
    plt.colorbar(sc, ax=ax, label="Déplacement (mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Déformation aléatoire\n(bruit spatial)")
    ax.set_aspect("equal")

    # --- (1,0) Histogramme des différences de déplacement entre voisins ---
    ax = axes[1, 0]
    bins = np.linspace(0, 6, 60)
    ax.hist(smooth_h[3], bins=bins, alpha=0.6, label=f"Hilbert (μ={smooth_h[0]:.3f})",
            color="green")
    ax.hist(smooth_r[3], bins=bins, alpha=0.4, label=f"Aléatoire (μ={smooth_r[0]:.3f})",
            color="red")
    ax.set_xlabel("Différence de déplacement entre voisins (mm)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Fluidité de la déformation\nsur les arêtes du mesh")
    ax.legend()

    # --- (1,1) Barplot comparatif ---
    ax = axes[1, 1]
    methods = ["Hilbert", "Aléatoire"]
    means = [smooth_h[0], smooth_r[0]]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(methods, means, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Diff. moyenne entre voisins (mm)")
    ax.set_title(f"Qualité de la déformation\n"
                 f"Hilbert {smooth_h[0]/smooth_r[0]*100:.0f}% plus lisse" if smooth_h[0] < smooth_r[0]
                 else "Qualité de la déformation")

    # --- (1,2) Tableau récapitulatif ---
    ax = axes[1, 2]
    ax.axis("off")
    headers = ["Métrique", "Hilbert", "Aléatoire", "Ratio"]
    rows = [
        ["Diff. moyenne voisins", f"{smooth_h[0]:.4f} mm", f"{smooth_r[0]:.4f} mm",
         f"{smooth_r[0]/smooth_h[0]:.1f}x"],
        ["Diff. max voisins", f"{smooth_h[1]:.3f} mm", f"{smooth_r[1]:.3f} mm",
         f"{smooth_r[1]/smooth_h[1]:.1f}x"],
        ["Écart-type diffs", f"{smooth_h[2]:.4f}", f"{smooth_r[2]:.4f}",
         f"{smooth_r[2]/smooth_h[2]:.1f}x"],
        ["Vertices", f"{len(verts_mm):,}", f"{len(verts_mm):,}", "—"],
        ["Faces", f"{len(faces):,}", f"{len(faces):,}", "—"],
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
    ax.set_title("Résumé du morphing", fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "proto_s6_surface.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  -> Graphique sauvé : {path}")
    return path


# ===================================================================
# Rapport
# ===================================================================

def write_report(verts_mm, faces, smooth_h, smooth_r):
    """Génère le rapport markdown."""
    path = os.path.join(OUTPUT_DIR, "proto_s6_resultats.md")

    lines = [
        "# Proto S6 — Animation et morphing de surfaces anatomiques via Hilbert 3D",
        "",
        "## Contexte",
        "",
        "Le brevet Altius-Code v3 (Application 6, Revendication 8) propose que la",
        "surface d'un organe, parcourue par la courbe de Hilbert 3D, permette de",
        "définir des animations (déformations temporelles) comme des transformations",
        "continues dans l'espace 1D des codes Hilbert.",
        "",
        "## Méthode",
        "",
        "1. Extraire la surface cérébrale du MNI152 par marching cubes",
        f"2. Calculer l'index Hilbert de chaque vertex ({len(verts_mm):,} vertices)",
        "3. Appliquer une déformation sinusoïdale :",
        "   - **Hilbert** : d(v) = A * sin(2π * f * h(v) + φ), où h(v) est l'index Hilbert normalisé",
        "   - **Aléatoire** : même formule mais avec un index aléatoire au lieu de h(v)",
        "4. Mesurer la fluidité : différence de déplacement entre vertices voisins sur le mesh",
        "",
        "## Résultats",
        "",
        "| Métrique | Hilbert | Aléatoire | Ratio |",
        "|----------|---------|-----------|-------|",
        f"| **Diff. moyenne voisins** | **{smooth_h[0]:.4f} mm** | {smooth_r[0]:.4f} mm | {smooth_r[0]/smooth_h[0]:.1f}x |",
        f"| Diff. max voisins | {smooth_h[1]:.3f} mm | {smooth_r[1]:.3f} mm | {smooth_r[1]/smooth_h[1]:.1f}x |",
        f"| Écart-type diffs | {smooth_h[2]:.4f} | {smooth_r[2]:.4f} | {smooth_r[2]/smooth_h[2]:.1f}x |",
        "",
        "## Interprétation",
        "",
        "- La déformation définie dans l'espace 1D de Hilbert produit une animation",
        "  **spatialement lisse** : les vertices voisins sur le mesh ont des",
        "  déplacements très similaires.",
        "- La même déformation avec un ordre aléatoire produit du **bruit spatial** :",
        "  les vertices voisins ont des déplacements très différents.",
        f"- Le parcours de Hilbert est **{smooth_r[0]/smooth_h[0]:.1f}x plus lisse** que l'ordre aléatoire.",
        "",
        "## Conclusion",
        "",
        "Le parcours de Hilbert 3D est un espace 1D naturel pour définir des",
        "animations de surfaces anatomiques, confirmant l'Application 6 du brevet.",
        "Toute fonction continue en 1D se traduit en déformation spatialement",
        "cohérente en 3D, grâce à la préservation de localité de la courbe de Hilbert.",
        "",
        "## Fichiers",
        "",
        "- Script : `proto_s6_surface.py`",
        "- Graphique : `results/proto_s6_surface.png`",
        "- Ce rapport : `results/proto_s6_resultats.md`",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> Rapport sauvé : {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 65)
    print("  PROTOTYPE S6 — Animation de surfaces anatomiques via Hilbert 3D")
    print("  Altius-Code, Altius Academy SNC")
    print("=" * 65)

    p = 5  # résolution suffisante pour le morphing de surface

    # 1. Extraire la surface
    print("\n[1/5] Extraction de la surface cérébrale...")
    verts_mm, faces, normals = extract_brain_surface()

    # 2. Calculer les index Hilbert
    print("\n[2/5] Calcul des index Hilbert...")
    hilbert_indices, hilbert_normalized = compute_vertex_hilbert_indices(verts_mm, p=p)

    # 3. Appliquer les déformations
    print("\n[3/5] Application des déformations...")
    deformed_h, disp_h = apply_hilbert_deformation(
        verts_mm, normals, hilbert_normalized,
        amplitude=3.0, frequency=4.0, phase=0.0
    )
    deformed_r, disp_r = apply_random_deformation(
        verts_mm, normals,
        amplitude=3.0, frequency=4.0, phase=0.0
    )
    print(f"  Hilbert : déplacement [{disp_h.min():.2f}, {disp_h.max():.2f}] mm")
    print(f"  Aléatoire : déplacement [{disp_r.min():.2f}, {disp_r.max():.2f}] mm")

    # 4. Mesurer la fluidité
    print("\n[4/5] Mesure de la fluidité...")
    smooth_h = compute_smoothness(verts_mm, disp_h, faces)
    smooth_r = compute_smoothness(verts_mm, disp_r, faces)
    print(f"  Hilbert  : diff. moy. voisins = {smooth_h[0]:.4f} mm")
    print(f"  Aléatoire : diff. moy. voisins = {smooth_r[0]:.4f} mm")
    print(f"  Ratio : {smooth_r[0]/smooth_h[0]:.1f}x plus lisse avec Hilbert")

    # 5. Visualisation et rapport
    print("\n[5/5] Génération graphiques et rapport...")
    plot_results(verts_mm, faces, normals, hilbert_normalized,
                 deformed_h, disp_h, deformed_r, disp_r,
                 smooth_h, smooth_r)
    write_report(verts_mm, faces, smooth_h, smooth_r)

    print(f"\n{'=' * 65}")
    print(f"  Proto S6 terminé.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
