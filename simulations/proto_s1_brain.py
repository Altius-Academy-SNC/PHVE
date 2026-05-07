"""
Prototype S1 — Codage Altius-Code 3D sur cerveau IRM (MNI152)

Charge le template MNI152 via nilearn, applique le codage Altius-Code 3D,
et génère des visualisations montrant :
1. Le volume IRM avec le codage Altius-Code superposé
2. La hiérarchie par troncature (zoom multi-échelle)
3. La recherche par préfixe (régions anatomiques)
4. La préservation de localité (voxels proches = codes proches)

Auteur : Paul Guindo, Altius Academy SNC
"""

import os
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Importer notre codec 3D
from codec3d import (
    VOLUMES, xyz2d, d2xyz, encode, decode, truncate,
    _code_length_3d, _int_to_base29, _canonical_precision_3d,
    verify_bijectivity,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_mni152():
    """Charge le template MNI152 T1w via nilearn."""
    from nilearn.datasets import load_mni152_template
    print("  Chargement MNI152 T1w (2mm)...", end=" ", flush=True)
    img = load_mni152_template(resolution=2)
    data = np.asarray(img.dataobj)
    affine = img.affine
    print(f"OK — shape={data.shape}, voxel={img.header.get_zooms()[:3]} mm")
    return data, affine, img


def load_mni152_brain_mask():
    """Charge le masque cérébral MNI152."""
    from nilearn.datasets import load_mni152_brain_mask
    print("  Chargement masque cérébral...", end=" ", flush=True)
    mask_img = load_mni152_brain_mask(resolution=2)
    mask = np.asarray(mask_img.dataobj).astype(bool)
    print(f"OK — {mask.sum()} voxels cérébraux")
    return mask


def voxel_ijk_to_mm(i, j, k, affine):
    """Convertit des indices voxel en coordonnées mm (espace MNI)."""
    ijk1 = np.array([i, j, k, 1.0])
    return (affine @ ijk1)[:3]


def mm_to_altius(x_mm, y_mm, z_mm, p=6, volume="CR"):
    """Coordonnées MNI (mm) -> code Altius-Code 3D.

    Le volume crânien CR est centré sur l'origine MNI.
    On décale les coordonnées pour les mettre dans [0, dims].
    """
    dims = VOLUMES[volume]["dims"]
    x = x_mm + dims[0] / 2
    y = y_mm + dims[1] / 2
    z = z_mm + dims[2] / 2
    return encode(x, y, z, p=p, volume=volume)


def compute_hilbert_index_volume(shape, affine, p=6, volume="CR"):
    """Calcule l'index Hilbert pour chaque voxel du volume.

    Retourne un volume de même shape contenant l'index Hilbert normalisé [0,1].
    """
    dims = VOLUMES[volume]["dims"]
    n = 1 << p
    max_d = 8 ** p - 1

    hilbert_vol = np.zeros(shape, dtype=np.float32)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                xyz = voxel_ijk_to_mm(i, j, k, affine)
                # Normaliser dans [0, n-1]
                ix = max(0, min(n - 1, int((xyz[0] + dims[0]/2) / dims[0] * n)))
                iy = max(0, min(n - 1, int((xyz[1] + dims[1]/2) / dims[1] * n)))
                iz = max(0, min(n - 1, int((xyz[2] + dims[2]/2) / dims[2] * n)))
                d = xyz2d(p, ix, iy, iz)
                hilbert_vol[i, j, k] = d / max_d

    return hilbert_vol


def demo_encode_decode(affine):
    """Démo basique : encoder/décoder des points anatomiques connus."""
    print("\n  Encodage de points anatomiques connus (espace MNI) :")

    landmarks = {
        "Centre du cerveau":   (0, 0, 0),
        "Cortex frontal":      (0, 50, 30),
        "Cortex occipital":    (0, -90, 0),
        "Hippocampe gauche":   (-25, -20, -15),
        "Hippocampe droit":    (25, -20, -15),
        "Cervelet":            (0, -60, -35),
        "Tronc cérébral":     (0, -30, -30),
    }

    results = {}
    for name, (x, y, z) in landmarks.items():
        code = mm_to_altius(x, y, z, p=8, volume="CR")
        decoded = decode(code)
        results[name] = code
        print(f"    {name:25s} ({x:+4d}, {y:+4d}, {z:+4d}) mm -> {code}")

    # Démontrer la localité : hippocampes gauche et droit
    code_l = results["Hippocampe gauche"]
    code_r = results["Hippocampe droit"]
    # Trouver le préfixe commun
    raw_l = code_l.split(":")[1].replace("-", "")
    raw_r = code_r.split(":")[1].replace("-", "")
    common = 0
    for a, b in zip(raw_l, raw_r):
        if a == b:
            common += 1
        else:
            break
    print(f"\n    Hippocampes G/D : préfixe commun = {common}/{len(raw_l)} caractères")
    print(f"    -> Les hippocampes partagent {common} caractères car ils sont proches spatialement")

    return results


def demo_hierarchy(affine):
    """Démo : hiérarchie par troncature = zoom multi-échelle."""
    print("\n  Hiérarchie par troncature (cortex frontal) :")

    code = mm_to_altius(0, 50, 30, p=8, volume="CR")
    decoded = decode(code)
    print(f"    Code complet : {code}  (résolution {decoded['resolution_mm']:.1f} mm)")

    for i in range(1, 5):
        t = truncate(code, i)
        if ":" in t and t.split(":")[1]:
            raw = t.split(":")[1].replace("-", "")
            p_trunc = _canonical_precision_3d(len(raw))
            n_trunc = 1 << p_trunc
            dims = VOLUMES["CR"]["dims"]
            res = max(dims[0]/n_trunc, dims[1]/n_trunc, dims[2]/n_trunc)
            print(f"    Tronqué -{i}   : {t:20s}  (résolution ~{res:.0f} mm)")
        else:
            print(f"    Tronqué -{i}   : {t:20s}  (volume entier)")


def demo_prefix_search(data, affine, mask):
    """Démo : recherche par préfixe = sélection de région anatomique."""
    print("\n  Recherche par préfixe (sélection de région) :")

    # Encoder un point au centre du cerveau
    target_code = mm_to_altius(0, 0, 0, p=6, volume="CR")
    target_raw = target_code.split(":")[1].replace("-", "")
    prefix = target_raw[:2]  # 2 premiers caractères
    print(f"    Code cible : {target_code}")
    print(f"    Préfixe recherché : CR:{prefix}...")

    # Compter les voxels qui partagent ce préfixe
    dims = VOLUMES["CR"]["dims"]
    n = 1 << 6  # p=6
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
    print(f"    Voxels correspondants : {match_count}/{total_brain} ({pct:.1f}%)")
    print(f"    -> Requête SQL : WHERE code LIKE 'CR:{prefix}%'")


def plot_results(data, affine, mask):
    """Génère les visualisations."""
    print("\n[+] Génération des graphiques...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Proto S1 — Codage Altius-Code 3D sur cerveau IRM (MNI152)",
                 fontsize=14, fontweight="bold")

    # Coupes centrales
    mid_x = data.shape[0] // 2
    mid_y = data.shape[1] // 2
    mid_z = data.shape[2] // 2

    # --- (0,0) Coupe sagittale avec IRM ---
    ax = axes[0, 0]
    ax.imshow(np.rot90(data[mid_x, :, :]), cmap="gray", aspect="auto")
    ax.set_title("IRM T1w — Coupe sagittale\n(MNI152, 2mm)")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")

    # --- (0,1) Coupe axiale avec index Hilbert ---
    ax = axes[0, 1]
    p = 5  # p=5 pour visualisation rapide
    dims = VOLUMES["CR"]["dims"]
    n = 1 << p
    max_d = 8 ** p - 1

    # Calculer l'index Hilbert sur la coupe axiale (z = mid)
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
    ax.set_title(f"Index Hilbert 3D (p={p})\nCoupe axiale — couleur = position sur la courbe")
    plt.colorbar(im, ax=ax, label="Index Hilbert normalisé")

    # --- (0,2) Hiérarchie par troncature ---
    ax = axes[0, 2]
    ax.axis("off")
    headers = ["Niveau", "Code", "Résolution"]
    code_full = mm_to_altius(0, 50, 30, p=8, volume="CR")
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
            rows.append(["p=0", c, "Volume entier"])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#3498db")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Hiérarchie par troncature\n(cortex frontal)", fontsize=11, fontweight="bold", pad=20)

    # --- (1,0) Coupe coronale IRM ---
    ax = axes[1, 0]
    ax.imshow(np.rot90(data[:, mid_y, :]), cmap="gray", aspect="auto")
    ax.set_title("IRM T1w — Coupe coronale")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")

    # --- (1,1) Préfixe commun = localité ---
    ax = axes[1, 1]
    # Encoder tous les voxels de la coupe axiale avec p=5
    prefix_len = 2  # montrer les régions par préfixe de 2 chars
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
            # Hash du préfixe pour colorer par région
            prefix_val = hash(code[:prefix_len]) % 256
            region_map[i, j] = prefix_val

    masked_regions = np.ma.masked_where(~mask[:, :, mid_z], region_map)
    ax.imshow(np.rot90(data[:, :, mid_z]), cmap="gray", alpha=0.3, aspect="auto")
    ax.imshow(np.rot90(masked_regions), cmap="tab20", alpha=0.6, aspect="auto")
    ax.set_title(f"Régions par préfixe ({prefix_len} chars)\n"
                 "Même couleur = même préfixe = même région")

    # --- (1,2) Points anatomiques encodés ---
    ax = axes[1, 2]
    ax.axis("off")
    landmarks = {
        "Centre cerveau":    (0, 0, 0),
        "Cortex frontal":    (0, 50, 30),
        "Cortex occipital":  (0, -90, 0),
        "Hippocampe G":      (-25, -20, -15),
        "Hippocampe D":      (25, -20, -15),
        "Cervelet":          (0, -60, -35),
        "Tronc cérébral":   (0, -30, -30),
    }
    headers = ["Point", "MNI (mm)", "Code Altius"]
    rows = []
    for name, (x, y, z) in landmarks.items():
        code = mm_to_altius(x, y, z, p=8, volume="CR")
        rows.append([name, f"({x:+d},{y:+d},{z:+d})", code])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.6)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2ecc71")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Points anatomiques encodés\n(p=8, volume CR)", fontsize=11,
                 fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "proto_s1_brain.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


def main():
    print("=" * 65)
    print("  PROTOTYPE S1 — Codage Altius-Code 3D sur cerveau IRM")
    print("  Altius-Code, Altius Academy SNC")
    print("=" * 65)

    # Vérification bijectivité rapide
    print("\n[1/5] Vérification bijectivité (p=3)...", end=" ", flush=True)
    ok, msg = verify_bijectivity(3)
    print(msg)
    if not ok:
        print("  ERREUR: bijectivité échouée, arrêt.")
        return

    # Charger MNI152
    print("\n[2/5] Chargement du template MNI152...")
    data, affine, img = load_mni152()
    mask = load_mni152_brain_mask()

    # Démo encode/decode
    print("\n[3/5] Démonstration encode/decode...")
    demo_encode_decode(affine)

    # Hiérarchie
    print("\n[4/5] Démonstration hiérarchie...")
    demo_hierarchy(affine)

    # Visualisations
    print("\n[5/5] Visualisations...")
    plot_results(data, affine, mask)

    # Résumé
    print(f"\n{'=' * 65}")
    print(f"  RÉSUMÉ")
    print(f"{'=' * 65}")
    print(f"\n  Volume IRM     : MNI152 T1w, {data.shape}, 2mm isotropique")
    print(f"  Voxels cerveau : {mask.sum()}")
    print(f"  Codec          : Hilbert 3D (Skilling 2004), bijectif")
    print(f"  Alphabet       : base-29, non-ambigu (29 caractères)")
    print(f"  Volumes        : {', '.join(VOLUMES.keys())}")
    print(f"\n  Le même voxel cérébral est identifié par :")
    print(f"    - Coordonnées MNI : (0.0, 50.0, 30.0) mm")
    code = mm_to_altius(0, 50, 30, p=8, volume="CR")
    print(f"    - Code Altius     : {code}")
    print(f"    - Recherche SQL   : WHERE code LIKE 'CR:{code.split(':')[1][:3]}%'")

    print(f"\n{'=' * 65}")
    print(f"  Proto S1 terminé.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
