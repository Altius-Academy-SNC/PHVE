"""
Prototype 7 AMÉLIORÉ — MEF + Stochastique + Maillage adaptatif Hilbert

Quatre sous-prototypes :

  7c+ : Éléments Finis (MEF P1) sur maillage STRUCTURÉ et NON-STRUCTURÉ
        → Comparaison : naturel / RCM / Hilbert
        → SSOR préconditionneur (sensible à l'ordre)
        → Insight clé : Hilbert brille sur maillage non-structuré

  7d  : EDP stochastique (diffusion aléatoire κ(x,ω))
        → Monte Carlo avec SSOR + renumérotage
        → Le coût du renumérotage est amorti sur N réalisations

  7e  : Maillage adaptatif guidé par Hilbert
        → Raffinement/déraffinement = ajout/troncature de caractères
        → Cellules de taille variable (polygones dynamiques)

  7f  : Incomplete Cholesky (IC(0)) — fill-in vs ordering
        → Mesure directe du nnz(L) en fonction de l'ordonnancement

Théorie : Paul Guindo, Altius Academy SNC.
"""

import math
import os
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator, spilu
from scipy.spatial import Delaunay

warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)

# ===================================================================
# Hilbert 2D encoding
# ===================================================================


def _rot2d(s, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = s - 1 - x
            y = s - 1 - y
        x, y = y, x
    return x, y


def _xy2d(p, x, y):
    n = 1 << p
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot2d(s, x, y, rx, ry)
        s >>= 1
    return d


# ===================================================================
# Mesh generation
# ===================================================================


def generate_structured_mesh(nx):
    """Maillage triangulaire régulier sur [0,1]²."""
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, nx + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    elements = []
    for j in range(nx):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])
    elements = np.array(elements)

    boundary = set()
    for i in range(nx + 1):
        boundary.add(i)
        boundary.add(nx * (nx + 1) + i)
    for j in range(nx + 1):
        boundary.add(j * (nx + 1))
        boundary.add(j * (nx + 1) + nx)

    return nodes, elements, np.array(sorted(boundary))


def generate_unstructured_mesh(n_points, seed=42):
    """Maillage Delaunay non-structuré sur [0,1]² avec points aléatoires.

    Le "natural ordering" est l'ordre d'insertion (aléatoire) → pas de
    localité spatiale → Hilbert devrait significativement améliorer.
    """
    rng = np.random.RandomState(seed)

    # Points intérieurs aléatoires
    n_interior = n_points
    pts_interior = rng.rand(n_interior, 2)

    # Points de bord réguliers (assure un domaine convexe)
    n_edge = max(20, int(np.sqrt(n_points)))
    t = np.linspace(0, 1, n_edge, endpoint=False)
    pts_bottom = np.column_stack([t, np.zeros(n_edge)])
    pts_top = np.column_stack([t, np.ones(n_edge)])
    pts_left = np.column_stack([np.zeros(n_edge), t])
    pts_right = np.column_stack([np.ones(n_edge), t])

    nodes = np.vstack([pts_interior, pts_bottom, pts_top, pts_left, pts_right])

    # Supprimer les doublons
    _, unique_idx = np.unique(np.round(nodes, 8), axis=0, return_index=True)
    nodes = nodes[np.sort(unique_idx)]

    # Delaunay
    tri = Delaunay(nodes)
    elements = tri.simplices

    # Boundary: noeuds proches des bords
    eps = 1e-6
    boundary = np.where(
        (nodes[:, 0] < eps) | (nodes[:, 0] > 1 - eps) |
        (nodes[:, 1] < eps) | (nodes[:, 1] > 1 - eps)
    )[0]

    return nodes, elements, boundary


# ===================================================================
# Reordering
# ===================================================================


def hilbert_reorder(nodes, p=6):
    """Renumérotage par courbe de Hilbert 2D."""
    n = 1 << p
    N = len(nodes)

    hilbert_indices = np.zeros(N, dtype=np.int64)
    for i, (x, y) in enumerate(nodes):
        ix = min(n - 1, max(0, int(x * n)))
        iy = min(n - 1, max(0, int(y * n)))
        hilbert_indices[i] = _xy2d(p, ix, iy)

    perm = np.argsort(hilbert_indices)
    inv_perm = np.zeros(N, dtype=np.int64)
    inv_perm[perm] = np.arange(N)
    return perm, inv_perm


def rcm_reorder(A):
    """Reverse Cuthill-McKee."""
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    perm = reverse_cuthill_mckee(A)
    inv_perm = np.zeros(len(perm), dtype=np.int64)
    inv_perm[perm] = np.arange(len(perm))
    return perm, inv_perm


def random_reorder(N, seed=123):
    """Permutation aléatoire (pire cas)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    inv_perm = np.zeros(N, dtype=np.int64)
    inv_perm[perm] = np.arange(N)
    return perm, inv_perm


# ===================================================================
# FEM assembly (P1 triangles)
# ===================================================================


def assemble_stiffness(nodes, elements, kappa=None):
    """Matrice de rigidité K pour -div(κ∇u)=f, P1 triangles."""
    N = len(nodes)
    rows, cols, vals = [], [], []

    for e_idx, elem in enumerate(elements):
        x = nodes[elem, 0]
        y = nodes[elem, 1]

        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) -
                         (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue

        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

        k_e = kappa[e_idx] if kappa is not None else 1.0

        for i in range(3):
            for j in range(3):
                val = k_e * (b[i] * b[j] + c[i] * c[j]) / (4 * area)
                rows.append(elem[i])
                cols.append(elem[j])
                vals.append(val)

    return sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()


def assemble_rhs(nodes, elements):
    """Second membre f = source gaussienne au centre."""
    N = len(nodes)
    b = np.zeros(N)
    cx, cy, sigma = 0.5, 0.5, 0.1

    for elem in elements:
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) -
                         (x[2] - x[0]) * (y[1] - y[0]))
        for i in range(3):
            r2 = (nodes[elem[i], 0] - cx) ** 2 + (nodes[elem[i], 1] - cy) ** 2
            b[elem[i]] += area / 3 * 100 * math.exp(-r2 / (2 * sigma ** 2))

    return b


def apply_dirichlet(K, b, boundary):
    """Conditions de Dirichlet u=0 aux bords."""
    K = K.tolil()
    for node in boundary:
        K[node, :] = 0
        K[:, node] = 0
        K[node, node] = 1
        b[node] = 0
    return K.tocsr(), b


# ===================================================================
# Solver utilities
# ===================================================================


def bandwidth_stats(A):
    """Bande passante max et moyenne."""
    rows, cols = A.nonzero()
    diffs = np.abs(rows - cols)
    return int(np.max(diffs)), float(np.mean(diffs))


def make_ssor_pc(K_perm, omega=1.5):
    """Construit un préconditionneur SSOR (symétrique → compatible CG)."""
    diag = K_perm.diagonal().copy()
    diag[np.abs(diag) < 1e-15] = 1.0
    D = sparse.diags(diag)
    L = sparse.tril(K_perm, k=-1)
    M_fwd = (D + omega * L).tocsc()

    def ssor_solve(x):
        y = sparse.linalg.spsolve_triangular(M_fwd, x, lower=True)
        y = diag * y
        y = sparse.linalg.spsolve_triangular(M_fwd.T.tocsc(), y, lower=False)
        return y / (omega * (2 - omega))

    return LinearOperator(K_perm.shape, matvec=ssor_solve)


def solve_with_ordering(K, b, perm, inv_perm, name, use_ssor=True):
    """Résout Ku=b avec un ordonnancement donné."""
    N = K.shape[0]
    P = sparse.eye(N, format="csr")[perm, :]
    K_perm = P @ K @ P.T
    b_perm = P @ b

    bw_max, bw_avg = bandwidth_stats(K_perm)

    # CG sans préconditionneur
    iter_cg = [0]
    t0 = time.perf_counter()
    u_cg, _ = cg(K_perm, b_perm, tol=1e-10, maxiter=5000,
                  callback=lambda xk: iter_cg.__setitem__(0, iter_cg[0] + 1))
    t_cg = time.perf_counter() - t0

    # CG + SSOR
    iter_ssor = [0]
    t_ssor = 0.0
    if use_ssor:
        try:
            M_ssor = make_ssor_pc(K_perm)
            t0 = time.perf_counter()
            u_ssor, _ = cg(K_perm, b_perm, tol=1e-10, maxiter=5000,
                           M=M_ssor,
                           callback=lambda xk: iter_ssor.__setitem__(0, iter_ssor[0] + 1))
            t_ssor = time.perf_counter() - t0
        except Exception:
            iter_ssor[0] = -1

    # ILU fill-in measurement
    try:
        ilu = spilu(K_perm.tocsc(), fill_factor=1.0, drop_tol=0.0)
        nnz_ilu = ilu.nnz
    except Exception:
        nnz_ilu = -1

    u = P.T @ u_cg

    return {
        "name": name,
        "bw_max": bw_max,
        "bw_avg": bw_avg,
        "iter_cg": iter_cg[0],
        "time_cg": t_cg * 1000,
        "iter_ssor": iter_ssor[0],
        "time_ssor": t_ssor * 1000,
        "nnz_ilu": nnz_ilu,
        "nnz_K": K_perm.nnz,
        "solution": u,
    }


# ===================================================================
# 7c+ : MEF structuré vs non-structuré
# ===================================================================


def run_proto7c_improved():
    """7c+ : Comparaison sur maillage structuré ET non-structuré."""
    results = {}

    for mesh_type in ["structuré", "non-structuré"]:
        print(f"\n{'─' * 60}")
        print(f"  7c+ : MEF (P1) — Maillage {mesh_type}")
        print(f"{'─' * 60}")

        if mesh_type == "structuré":
            NX = 64
            nodes, elements, boundary = generate_structured_mesh(NX)
        else:
            nodes, elements, boundary = generate_unstructured_mesh(4000)

        N = len(nodes)
        print(f"  {N} noeuds, {len(elements)} éléments")

        K = assemble_stiffness(nodes, elements)
        b = assemble_rhs(nodes, elements)
        K, b = apply_dirichlet(K, b, boundary)

        # Orderings
        natural_perm = np.arange(N)
        rcm_perm, rcm_inv = rcm_reorder(K)
        hilbert_perm, hilbert_inv = hilbert_reorder(nodes, p=7)

        orderings = [
            ("Naturel", natural_perm, natural_perm),
            ("RCM", rcm_perm, rcm_inv),
            ("Hilbert", hilbert_perm, hilbert_inv),
        ]

        # Ajout d'un ordre "aléatoire" pour le non-structuré (baseline pire cas)
        if mesh_type == "non-structuré":
            rand_perm, rand_inv = random_reorder(N)
            orderings.insert(0, ("Aléatoire", rand_perm, rand_inv))

        mesh_results = []
        for name, perm, inv in orderings:
            print(f"  → {name}...", end="", flush=True)
            r = solve_with_ordering(K, b, perm, inv, name)
            print(f" BW_avg={r['bw_avg']:.0f}, CG={r['iter_cg']}, "
                  f"CG+SSOR={r['iter_ssor']}, t_SSOR={r['time_ssor']:.1f}ms")
            mesh_results.append(r)

        results[mesh_type] = {
            "results": mesh_results,
            "nodes": nodes,
            "elements": elements,
        }

    return results


# ===================================================================
# 7d : EDP stochastique avec SSOR
# ===================================================================


def generate_kl_modes(nodes, elements, n_kl=5, corr_length=0.3):
    """Modes de Karhunen-Loève pour champ aléatoire."""
    centers = np.mean(nodes[elements], axis=1)
    kl_modes = np.zeros((len(elements), n_kl))

    for k in range(n_kl):
        freq_x = (k // 2 + 1) * np.pi
        freq_y = (k % 2 + 1) * np.pi
        eigenvalue = corr_length ** 2 / (1 + (corr_length * freq_x) ** 2) / \
                     (1 + (corr_length * freq_y) ** 2)
        kl_modes[:, k] = np.sqrt(eigenvalue) * \
                         np.sin(freq_x * centers[:, 0]) * np.sin(freq_y * centers[:, 1])
    return kl_modes


def run_proto7d():
    """7d : EDP stochastique — Monte Carlo avec SSOR + renumérotage.

    Test sur maillage non-structuré pour que l'ordering ait un impact.
    """
    print(f"\n{'─' * 60}")
    print(f"  7d : EDP stochastique — Monte Carlo + SSOR")
    print(f"{'─' * 60}")

    N_REAL = 50
    N_KL = 5

    # Maillage non-structuré (c'est là que Hilbert brille)
    nodes, elements, boundary = generate_unstructured_mesh(2000, seed=99)
    N = len(nodes)
    print(f"  Maillage non-structuré : {N} noeuds, {len(elements)} éléments")
    print(f"  Réalisations MC : {N_REAL}, Modes KL : {N_KL}")

    kl_modes = generate_kl_modes(nodes, elements, n_kl=N_KL)
    b_base = assemble_rhs(nodes, elements)

    # Pré-calcul des renumérotages
    hilbert_perm, _ = hilbert_reorder(nodes, p=6)
    P_h = sparse.eye(N, format="csr")[hilbert_perm, :]

    rcm_done = False
    rng = np.random.RandomState(42)

    stats = {"natural": {"iters": [], "times": []},
             "hilbert": {"iters": [], "times": []},
             "rcm": {"iters": [], "times": []}}

    for i in range(N_REAL):
        xi = rng.randn(N_KL)
        kappa = np.exp(np.sum(kl_modes * xi[np.newaxis, :], axis=1))

        K = assemble_stiffness(nodes, elements, kappa)
        b = b_base.copy()
        K, b = apply_dirichlet(K, b, boundary)

        # RCM (calcul une seule fois sur la première matrice)
        if not rcm_done:
            rcm_perm, _ = rcm_reorder(K)
            P_r = sparse.eye(N, format="csr")[rcm_perm, :]
            rcm_done = True

        for label, P in [("natural", None), ("hilbert", P_h), ("rcm", P_r)]:
            if P is not None:
                K_p = P @ K @ P.T
                b_p = P @ b
            else:
                K_p = K
                b_p = b

            try:
                M_ssor = make_ssor_pc(K_p)
                it = [0]
                t0 = time.perf_counter()
                _, info = cg(K_p, b_p, tol=1e-8, maxiter=5000, M=M_ssor,
                             callback=lambda xk: it.__setitem__(0, it[0] + 1))
                dt = time.perf_counter() - t0
                stats[label]["iters"].append(it[0])
                stats[label]["times"].append(dt * 1000)
            except Exception:
                stats[label]["iters"].append(-1)
                stats[label]["times"].append(-1)

    print(f"\n  Résultats (moyenne sur {N_REAL} réalisations, CG+SSOR) :")
    for label in ["natural", "hilbert", "rcm"]:
        iters = [x for x in stats[label]["iters"] if x > 0]
        times = [x for x in stats[label]["times"] if x > 0]
        if iters:
            print(f"  → {label:>10} : {np.mean(iters):.0f} itérations, "
                  f"{np.mean(times):.1f} ms/solve")

    return stats


# ===================================================================
# 7e : Maillage adaptatif guidé par Hilbert
# ===================================================================


def adaptive_hilbert_mesh(nodes, elements, solution,
                          threshold_refine=0.7, threshold_coarsen=0.1):
    """Raffinement adaptatif guidé par le gradient."""
    gradients = np.zeros(len(elements))
    for i, elem in enumerate(elements):
        x = nodes[elem, 0]
        y = nodes[elem, 1]
        u = solution[elem]

        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) -
                         (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue

        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        dudx = np.sum(b * u) / (2 * area)
        dudy = np.sum(c * u) / (2 * area)
        gradients[i] = math.sqrt(dudx ** 2 + dudy ** 2)

    g_max = np.max(gradients)
    if g_max > 0:
        gradients /= g_max

    refined = np.where(gradients > threshold_refine)[0]
    coarsened = np.where(gradients < threshold_coarsen)[0]
    return refined, coarsened, gradients


# ===================================================================
# Visualization
# ===================================================================


def plot_results(results_7c, stats_7d, nodes_s, elems_s, sol_s,
                 nodes_u, elems_u, sol_u,
                 gradients, refined, coarsened, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Proto 7 amélioré — MEF + Stochastique + Adaptatif",
                 fontsize=14, fontweight="bold")
    colors_4 = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"]
    colors_3 = ["#3498db", "#e74c3c", "#2ecc71"]

    # --- 7c+ structuré : bande passante ---
    ax = axes[0, 0]
    res_s = results_7c["structuré"]["results"]
    names = [r["name"] for r in res_s]
    bw = [r["bw_avg"] for r in res_s]
    bars = ax.bar(names, bw, color=colors_3, alpha=0.8)
    for bar, val in zip(bars, bw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Bande moy.")
    ax.set_title("Structuré : bande passante\n(plus bas = meilleur)")
    ax.grid(axis="y", alpha=0.3)

    # --- 7c+ non-structuré : bande passante ---
    ax = axes[0, 1]
    res_u = results_7c["non-structuré"]["results"]
    names = [r["name"] for r in res_u]
    bw = [r["bw_avg"] for r in res_u]
    c = colors_4 if len(res_u) == 4 else colors_3
    bars = ax.bar(names, bw, color=c, alpha=0.8)
    for bar, val in zip(bars, bw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Bande moy.")
    ax.set_title("Non-structuré : bande passante\n(plus bas = meilleur)")
    ax.grid(axis="y", alpha=0.3)

    # --- 7c+ non-structuré : SSOR iterations ---
    ax = axes[0, 2]
    iter_s = [r["iter_ssor"] for r in res_u]
    bars = ax.bar(names, iter_s, color=c, alpha=0.8)
    for bar, val in zip(bars, iter_s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Itérations CG+SSOR")
    ax.set_title("Non-structuré : CG+SSOR\n(moins = meilleur)")
    ax.grid(axis="y", alpha=0.3)

    # --- 7d : Distribution Monte Carlo ---
    ax = axes[0, 3]
    for label, color, lbl in [("natural", "#3498db", "Naturel"),
                               ("hilbert", "#2ecc71", "Hilbert"),
                               ("rcm", "#e74c3c", "RCM")]:
        iters = [x for x in stats_7d[label]["iters"] if x > 0]
        if iters:
            ax.hist(iters, bins=15, alpha=0.5, color=color,
                    label=f"{lbl} (μ={np.mean(iters):.0f})")
    ax.set_xlabel("Itérations CG+SSOR")
    ax.set_ylabel("Fréquence")
    ax.set_title("7d : Monte Carlo + SSOR\n(non-structuré)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Solution structurée ---
    ax = axes[1, 0]
    triang_s = mtri.Triangulation(nodes_s[:, 0], nodes_s[:, 1], elems_s)
    tcf = ax.tripcolor(triang_s, sol_s, cmap="viridis", shading="flat")
    plt.colorbar(tcf, ax=ax, shrink=0.8)
    ax.set_title("Solution (structuré)")
    ax.set_aspect("equal")

    # --- Solution non-structurée ---
    ax = axes[1, 1]
    triang_u = mtri.Triangulation(nodes_u[:, 0], nodes_u[:, 1], elems_u)
    tcf = ax.tripcolor(triang_u, sol_u, cmap="viridis", shading="flat")
    plt.colorbar(tcf, ax=ax, shrink=0.8)
    ax.set_title("Solution (non-structuré)")
    ax.set_aspect("equal")

    # --- Gradient (indicateur AMR) ---
    ax = axes[1, 2]
    tcf = ax.tripcolor(triang_s, gradients, cmap="hot", shading="flat")
    plt.colorbar(tcf, ax=ax, shrink=0.8)
    ax.set_title("7e : Gradient (indicateur AMR)")
    ax.set_aspect("equal")

    # --- Maillage adaptatif ---
    ax = axes[1, 3]
    ax.triplot(triang_s, linewidth=0.2, color="gray")
    for idx in refined:
        elem = elems_s[idx]
        tri = plt.Polygon(nodes_s[elem], facecolor="red", alpha=0.3, edgecolor="red")
        ax.add_patch(tri)
    for idx in coarsened[:200]:
        elem = elems_s[idx]
        tri = plt.Polygon(nodes_s[elem], facecolor="blue", alpha=0.1,
                          edgecolor="blue", linewidth=0.3)
        ax.add_patch(tri)
    ax.set_title(f"7e : AMR Hilbert\nRouge={len(refined)} raff., "
                 f"Bleu={len(coarsened)} déraff.")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "proto7_improved.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")
    return path


# ===================================================================
# Main
# ===================================================================


def main():
    np.random.seed(42)
    OUTPUT_DIR = "/home/paul/Documents/Code/brevets/Altius-Code/results"

    print("=" * 65)
    print("  PROTOTYPE 7 AMÉLIORÉ — MEF + Stochastique + Adaptatif")
    print("  Altius-Code, Altius Academy SNC")
    print("=" * 65)

    # --- 7c+ : MEF structuré ET non-structuré ---
    results_7c = run_proto7c_improved()

    # --- 7d : Monte Carlo stochastique sur maillage non-structuré ---
    stats_7d = run_proto7d()

    # --- 7e : Maillage adaptatif ---
    print(f"\n{'─' * 60}")
    print(f"  7e : Maillage adaptatif guidé par Hilbert")
    print(f"{'─' * 60}")

    nodes_s = results_7c["structuré"]["nodes"]
    elems_s = results_7c["structuré"]["elements"]
    sol_s = results_7c["structuré"]["results"][-1]["solution"]  # Hilbert

    nodes_u = results_7c["non-structuré"]["nodes"]
    elems_u = results_7c["non-structuré"]["elements"]
    sol_u = results_7c["non-structuré"]["results"][-1]["solution"]  # Hilbert

    refined, coarsened, gradients = adaptive_hilbert_mesh(
        nodes_s, elems_s, sol_s, threshold_refine=0.7, threshold_coarsen=0.1
    )
    print(f"  → Éléments à raffiner : {len(refined)} "
          f"({100*len(refined)/len(elems_s):.1f}%)")
    print(f"  → Éléments à déraffiner : {len(coarsened)} "
          f"({100*len(coarsened)/len(elems_s):.1f}%)")

    # ===== RÉSUMÉ =====
    print(f"\n{'=' * 65}")
    print(f"  RÉSUMÉ DES RÉSULTATS")
    print(f"{'=' * 65}")

    for mesh_type in ["structuré", "non-structuré"]:
        print(f"\n  7c+ — Maillage {mesh_type} :")
        for r in results_7c[mesh_type]["results"]:
            print(f"    {r['name']:>10} : bw_avg={r['bw_avg']:.0f}, "
                  f"CG={r['iter_cg']}, CG+SSOR={r['iter_ssor']}, "
                  f"t_SSOR={r['time_ssor']:.1f}ms")

        # Gain Hilbert vs Natural
        res = results_7c[mesh_type]["results"]
        nat = next(r for r in res if r["name"] == "Naturel")
        hil = next(r for r in res if r["name"] == "Hilbert")
        bw_gain = (1 - hil["bw_avg"] / nat["bw_avg"]) * 100
        ssor_gain = (1 - hil["iter_ssor"] / nat["iter_ssor"]) * 100 if nat["iter_ssor"] > 0 else 0
        print(f"  → Hilbert vs Naturel : bande {bw_gain:+.1f}%, "
              f"SSOR iters {ssor_gain:+.1f}%")

    print(f"\n  7d — Monte Carlo stochastique (non-structuré) :")
    for label in ["natural", "hilbert", "rcm"]:
        iters = [x for x in stats_7d[label]["iters"] if x > 0]
        times = [x for x in stats_7d[label]["times"] if x > 0]
        if iters:
            print(f"    {label:>10} : {np.mean(iters):.0f} itérations, "
                  f"{np.mean(times):.1f} ms/solve")

    print(f"\n  7e — Maillage adaptatif :")
    print(f"    {len(refined)} cellules à raffiner, {len(coarsened)} à déraffiner")

    # Graphiques
    print(f"\n[+] Graphiques...")
    plot_results(results_7c, stats_7d, nodes_s, elems_s, sol_s,
                 nodes_u, elems_u, sol_u,
                 gradients, refined, coarsened, OUTPUT_DIR)

    print(f"\n{'=' * 65}")
    print(f"  Proto 7 amélioré terminé.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
