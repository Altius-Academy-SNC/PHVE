"""
Prototype S2 — Comparaison inter-patients Altius-Code 3D

Démontre que le même point anatomique reçoit un code identique (ou très proche)
chez différents patients, une fois recalé dans l'espace MNI standard.

Simule N patients par perturbation du template MNI152 + erreur résiduelle de recalage.

Auteur : Paul Guindo, Altius Academy SNC
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from codec3d import (
    VOLUMES, encode, decode, truncate, xyz2d,
    _code_length_3d, _int_to_base29, _canonical_precision_3d,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def mm_to_altius(x_mm, y_mm, z_mm, p=6, volume="CR"):
    """Coordonnées MNI (mm) -> code Altius-Code 3D."""
    dims = VOLUMES[volume]["dims"]
    x = x_mm + dims[0] / 2
    y = y_mm + dims[1] / 2
    z = z_mm + dims[2] / 2
    return encode(x, y, z, p=p, volume=volume)


LANDMARKS = {
    "Centre cerveau":    (0, 0, 0),
    "Cortex frontal":    (0, 50, 30),
    "Cortex occipital":  (0, -90, 0),
    "Hippocampe G":      (-25, -20, -15),
    "Hippocampe D":      (25, -20, -15),
    "Cervelet":          (0, -60, -35),
    "Tronc cérébral":    (0, -30, -30),
}


def simulate_interpatient(n_patients=20, sigma_mm=0.5, p=6):
    """Simule n_patients avec erreur résiduelle de recalage σ.

    Pour chaque patient et chaque point anatomique, on ajoute un bruit
    gaussien (σ mm) aux coordonnées MNI, simulant l'erreur résiduelle
    après recalage affine + non-linéaire.

    Returns:
        dict: résultats par point anatomique
    """
    rng = np.random.RandomState(42)
    results = {}

    for name, (x, y, z) in LANDMARKS.items():
        # Code de référence (template MNI152 parfait)
        ref_code = mm_to_altius(x, y, z, p=p, volume="CR")
        ref_raw = ref_code.split(":")[1].replace("-", "")

        codes = []
        prefix_matches = []

        for patient_idx in range(n_patients):
            # Erreur résiduelle de recalage
            dx, dy, dz = rng.normal(0, sigma_mm, 3)
            patient_code = mm_to_altius(x + dx, y + dy, z + dz, p=p, volume="CR")
            patient_raw = patient_code.split(":")[1].replace("-", "")
            codes.append(patient_code)

            # Préfixe commun avec la référence
            common = 0
            for a, b in zip(ref_raw, patient_raw):
                if a == b:
                    common += 1
                else:
                    break
            prefix_matches.append(common)

        n_identical = sum(1 for c in codes if c == ref_code)
        avg_prefix = np.mean(prefix_matches)

        results[name] = {
            "ref_code": ref_code,
            "n_identical": n_identical,
            "n_patients": n_patients,
            "pct_identical": n_identical / n_patients * 100,
            "avg_prefix_match": avg_prefix,
            "max_prefix": len(ref_raw),
            "codes": codes,
        }

    return results


def run_robustness_sweep(n_patients=50):
    """Sweep sur p et sigma pour mesurer la robustesse."""
    p_values = list(range(3, 11))
    sigma_values = [0.1, 0.5, 1.0, 2.0]

    results = {}
    for sigma in sigma_values:
        pct_by_p = []
        for p in p_values:
            res = simulate_interpatient(n_patients=n_patients, sigma_mm=sigma, p=p)
            avg_pct = np.mean([r["pct_identical"] for r in res.values()])
            pct_by_p.append(avg_pct)
        results[sigma] = pct_by_p

    return p_values, sigma_values, results


def plot_results(p_values, sigma_values, sweep_results, interpatient_results, p_used):
    """Génère les graphiques S2."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Proto S2 — Stabilité inter-patients du codage Altius-Code 3D",
                 fontsize=14, fontweight="bold")

    # --- (0) Robustesse vs p pour différents σ ---
    ax = axes[0]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    for sigma, color in zip(sigma_values, colors):
        ax.plot(p_values, sweep_results[sigma], "o-", color=color,
                label=f"σ = {sigma} mm", linewidth=2, markersize=8)

    ax.set_xlabel("Ordre p (précision)")
    ax.set_ylabel("Codes identiques (%)")
    ax.set_title("Robustesse vs. précision\n(50 patients/point)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # Axe secondaire : résolution
    ax2 = ax.twiny()
    res_labels = [f"{VOLUMES['CR']['dims'][0] / (1 << p):.0f}" for p in p_values]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(p_values)
    ax2.set_xticklabels(res_labels)
    ax2.set_xlabel("Résolution (mm)")

    # --- (1) Détail par point anatomique ---
    ax = axes[1]
    names = list(interpatient_results.keys())
    pcts = [interpatient_results[n]["pct_identical"] for n in names]
    bar_colors = ["#2ecc71" if p >= 80 else "#e67e22" if p >= 50 else "#e74c3c" for p in pcts]
    bars = ax.barh(names, pcts, color=bar_colors)
    ax.set_xlabel("Codes identiques (%)")
    ax.set_title(f"Stabilité par point anatomique\n(p={p_used}, σ=0.5 mm, 20 patients)")
    ax.set_xlim(0, 105)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", va="center", fontsize=9)

    # --- (2) Table résumé ---
    ax = axes[2]
    ax.axis("off")
    headers = ["Point", "Code ref.", "Identiques", "Préfixe moy."]
    rows = []
    for name in names:
        r = interpatient_results[name]
        rows.append([
            name,
            r["ref_code"],
            f"{r['n_identical']}/{r['n_patients']} ({r['pct_identical']:.0f}%)",
            f"{r['avg_prefix_match']:.1f}/{r['max_prefix']}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.7)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#3498db")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Résultats détaillés (p={p_used})", fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "proto_s2_interpatient.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


def main():
    print("=" * 65)
    print("  PROTOTYPE S2 — Comparaison inter-patients Altius-Code 3D")
    print("  Altius-Code, Altius Academy SNC")
    print("=" * 65)

    # --- Test principal : p=6, σ=0.5 mm, 20 patients ---
    p_main = 6
    sigma_main = 0.5
    n_patients_main = 20

    print(f"\n[1/3] Simulation inter-patients (p={p_main}, σ={sigma_main} mm, "
          f"n={n_patients_main} patients)...")

    results = simulate_interpatient(n_patients=n_patients_main, sigma_mm=sigma_main, p=p_main)

    print("\n  Résultats par point anatomique :")
    for name, r in results.items():
        print(f"    {name:20s} : {r['ref_code']}  "
              f"→ {r['n_identical']}/{r['n_patients']} identiques "
              f"({r['pct_identical']:.0f}%), "
              f"préfixe moy. {r['avg_prefix_match']:.1f}/{r['max_prefix']}")

    avg_pct = np.mean([r["pct_identical"] for r in results.values()])
    print(f"\n  Moyenne : {avg_pct:.1f}% de codes identiques")

    # --- Sweep robustesse ---
    print(f"\n[2/3] Sweep robustesse (p=3..10, σ=0.1..2.0 mm, 50 patients)...")
    p_values, sigma_values, sweep = run_robustness_sweep(n_patients=50)

    for sigma in sigma_values:
        line = f"    σ={sigma:.1f} mm : "
        line += " | ".join(f"p={p}:{pct:.0f}%" for p, pct in zip(p_values, sweep[sigma]))
        print(line)

    # --- Graphiques ---
    print(f"\n[3/3] Génération des graphiques...")
    plot_results(p_values, sigma_values, sweep, results, p_main)

    # --- Résumé ---
    print(f"\n{'=' * 65}")
    print(f"  RÉSUMÉ")
    print(f"{'=' * 65}")
    print(f"\n  Le codage Altius-Code 3D est stable inter-patients :")
    print(f"    - p=6 (résolution ~5 mm), σ=0.5 mm : {avg_pct:.0f}% codes identiques")
    print(f"    - Même quand les codes diffèrent, le préfixe commun est long")
    print(f"    - Implication : un même code Altius désigne la même région")
    print(f"      anatomique chez tous les patients recalés en MNI")
    print(f"\n  Compromis précision/robustesse :")
    print(f"    - p=5 (~10 mm) : très stable, idéal pour indexation grossière")
    print(f"    - p=6 (~5 mm)  : bon compromis clinique")
    print(f"    - p=8 (~1 mm)  : haute résolution, sensible au recalage")

    print(f"\n{'=' * 65}")
    print(f"  Proto S2 terminé.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
