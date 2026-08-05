"""
Inter-patient stability of PHVE codes (Corollary 5.3 of the paper)

Demonstrates that the same anatomical landmark receives an identical (or
very close) code across patients, once they are registered to the MNI
template. We simulate N patients by perturbing landmark coordinates with
a residual registration error sigma in mm.

Author: Paul Guindo, Altius Academy SNC.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from codec3d import (
    VOLUMES, encode, decode, truncate, xyz2d,
    _code_length_3d, _int_to_base29, _canonical_precision_3d,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def mm_to_phve(x_mm, y_mm, z_mm, p=6, volume="CR"):
    """MNI mm coordinates -> PHVE 3D code."""
    dims = VOLUMES[volume]["dims"]
    x = x_mm + dims[0] / 2
    y = y_mm + dims[1] / 2
    z = z_mm + dims[2] / 2
    return encode(x, y, z, p=p, volume=volume)


LANDMARKS = {
    "Brain centre":      (0, 0, 0),
    "Frontal cortex":    (0, 50, 30),
    "Occipital cortex":  (0, -90, 0),
    "L. hippocampus":    (-25, -20, -15),
    "R. hippocampus":    (25, -20, -15),
    "Cerebellum":        (0, -60, -35),
    "Brain stem":        (0, -30, -30),
}


def simulate_interpatient(n_patients=20, sigma_mm=0.5, p=6):
    """Simulate n_patients with a residual registration error sigma (mm).

    For each patient and landmark, we add Gaussian noise (sigma in mm) to
    the MNI coordinates to model the residual error of an affine +
    non-linear registration.
    """
    rng = np.random.RandomState(42)
    results = {}

    for name, (x, y, z) in LANDMARKS.items():
        ref_code = mm_to_phve(x, y, z, p=p, volume="CR")
        ref_raw = ref_code.split(":")[1].replace("-", "")

        codes = []
        prefix_matches = []

        for patient_idx in range(n_patients):
            dx, dy, dz = rng.normal(0, sigma_mm, 3)
            patient_code = mm_to_phve(x + dx, y + dy, z + dz, p=p, volume="CR")
            patient_raw = patient_code.split(":")[1].replace("-", "")
            codes.append(patient_code)

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
    """Sweep over p and sigma to chart robustness."""
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
    """Render the inter-patient stability figure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Inter-patient stability of PHVE codes (Cor. 5.3)",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    for sigma, color in zip(sigma_values, colors):
        ax.plot(p_values, sweep_results[sigma], "o-", color=color,
                label=f"$\\sigma$ = {sigma} mm", linewidth=2, markersize=8)

    ax.set_xlabel("Order $p$ (precision)")
    ax.set_ylabel("Identical codes (%)")
    ax.set_title("Robustness vs.\\ precision\n(50 patients/landmark)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    ax2 = ax.twiny()
    res_labels = [f"{VOLUMES['CR']['dims'][0] / (1 << p):.0f}" for p in p_values]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(p_values)
    ax2.set_xticklabels(res_labels)
    ax2.set_xlabel("Resolution (mm)")

    ax = axes[1]
    names = list(interpatient_results.keys())
    pcts = [interpatient_results[n]["pct_identical"] for n in names]
    bar_colors = ["#2ecc71" if p >= 80 else "#e67e22" if p >= 50 else "#e74c3c" for p in pcts]
    bars = ax.barh(names, pcts, color=bar_colors)
    ax.set_xlabel("Identical codes (%)")
    ax.set_title(f"Stability per landmark\n($p={p_used}$, $\\sigma=0.5$ mm, 20 patients)")
    ax.set_xlim(0, 105)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", va="center", fontsize=9)

    ax = axes[2]
    ax.axis("off")
    headers = ["Landmark", "Reference code", "Identical", "Mean prefix"]
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
    ax.set_title(f"Detailed results ($p={p_used}$)", fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "interpatient_stability.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")


def main():
    print("=" * 65)
    print("  Inter-patient stability of PHVE codes (Cor. 5.3)")
    print("  Altius Academy SNC")
    print("=" * 65)

    p_main = 6
    sigma_main = 0.5
    n_patients_main = 20

    print(f"\n[1/3] Inter-patient simulation (p={p_main}, sigma={sigma_main} mm, "
          f"n={n_patients_main} patients)...")

    results = simulate_interpatient(n_patients=n_patients_main, sigma_mm=sigma_main, p=p_main)

    print("\n  Per-landmark results:")
    for name, r in results.items():
        print(f"    {name:20s} : {r['ref_code']}  "
              f"-> {r['n_identical']}/{r['n_patients']} identical "
              f"({r['pct_identical']:.0f}%), "
              f"mean prefix {r['avg_prefix_match']:.1f}/{r['max_prefix']}")

    avg_pct = np.mean([r["pct_identical"] for r in results.values()])
    print(f"\n  Mean: {avg_pct:.1f}% identical codes")

    print(f"\n[2/3] Robustness sweep (p=3..10, sigma=0.1..2.0 mm, 50 patients)...")
    p_values, sigma_values, sweep = run_robustness_sweep(n_patients=50)

    for sigma in sigma_values:
        line = f"    sigma={sigma:.1f} mm : "
        line += " | ".join(f"p={p}:{pct:.0f}%" for p, pct in zip(p_values, sweep[sigma]))
        print(line)

    print(f"\n[3/3] Generating figure...")
    plot_results(p_values, sigma_values, sweep, results, p_main)

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"\n  PHVE 3D codes are stable across patients:")
    print(f"    - p=6 (resolution ~5 mm), sigma=0.5 mm : {avg_pct:.0f}% identical codes")
    print(f"    - Even when codes differ, the common prefix is long")
    print(f"    - Consequence: the same PHVE code labels the same anatomical")
    print(f"      region across all MNI-registered patients")
    print(f"\n  Precision/robustness trade-off:")
    print(f"    - p=5 (~10 mm) : very stable, suitable for coarse indexing")
    print(f"    - p=6 (~5 mm)  : good clinical compromise")
    print(f"    - p=8 (~1 mm)  : high resolution, sensitive to registration error")

    print(f"\n{'=' * 65}")
    print(f"  interpatient_stability done.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
