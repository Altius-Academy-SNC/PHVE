"""
FEM stiffness-matrix bandwidth reduction (Theorem 10.2 of the paper)

Four sub-experiments:

  (a) FEM (P1) on structured AND unstructured triangular meshes:
      compare natural / RCM / Hilbert orderings, report bandwidth(K),
      preconditioned CG iteration counts, and SSOR wall-time.
      Hilbert shines on unstructured meshes (the paper's headline).

  (b) Stochastic PDE (random diffusion kappa(x, omega)): Monte Carlo
      with SSOR + reordering. The reordering cost is amortised over
      many realisations.

  (c) Adaptive mesh refinement guided by Hilbert: refine/coarsen =
      append/truncate code characters; cells of variable size become
      dynamic polygons (Hierarchy by truncation, Theorem 6.1).

Author: Paul Guindo, Altius Academy SNC.
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
# 2D Hilbert encoding
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
    """Regular triangular mesh on [0, 1]^2."""
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
    """Unstructured Delaunay mesh on [0, 1]^2 from random interior points.

    The "natural" ordering is the random insertion order, which has no
    spatial locality -- exactly the regime where Hilbert reordering
    delivers the largest bandwidth reduction (Theorem 10.2(c)).
    """
    rng = np.random.RandomState(seed)

    n_interior = n_points
    pts_interior = rng.rand(n_interior, 2)

    n_edge = max(20, int(np.sqrt(n_points)))
    t = np.linspace(0, 1, n_edge, endpoint=False)
    pts_bottom = np.column_stack([t, np.zeros(n_edge)])
    pts_top = np.column_stack([t, np.ones(n_edge)])
    pts_left = np.column_stack([np.zeros(n_edge), t])
    pts_right = np.column_stack([np.ones(n_edge), t])

    nodes = np.vstack([pts_interior, pts_bottom, pts_top, pts_left, pts_right])

    _, unique_idx = np.unique(np.round(nodes, 8), axis=0, return_index=True)
    nodes = nodes[np.sort(unique_idx)]

    tri = Delaunay(nodes)
    elements = tri.simplices

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
    """2D Hilbert reordering of the mesh nodes."""
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
    """Random permutation (worst-case baseline)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    inv_perm = np.zeros(N, dtype=np.int64)
    inv_perm[perm] = np.arange(N)
    return perm, inv_perm


# ===================================================================
# FEM assembly (P1 triangles)
# ===================================================================


def assemble_stiffness(nodes, elements, kappa=None):
    """Stiffness matrix K for -div(kappa * grad u) = f, P1 triangles."""
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
    """Right-hand side: Gaussian source centred at (0.5, 0.5)."""
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
    """Homogeneous Dirichlet conditions u = 0 on the boundary."""
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
    """Max and mean bandwidth of a sparse matrix."""
    rows, cols = A.nonzero()
    diffs = np.abs(rows - cols)
    return int(np.max(diffs)), float(np.mean(diffs))


def make_ssor_pc(K_perm, omega=1.5):
    """SSOR preconditioner (symmetric => CG-compatible)."""
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
    """Solve Ku = b under the given ordering."""
    N = K.shape[0]
    P = sparse.eye(N, format="csr")[perm, :]
    K_perm = P @ K @ P.T
    b_perm = P @ b

    bw_max, bw_avg = bandwidth_stats(K_perm)

    iter_cg = [0]
    t0 = time.perf_counter()
    u_cg, _ = cg(K_perm, b_perm, rtol=1e-10, maxiter=5000,
                  callback=lambda xk: iter_cg.__setitem__(0, iter_cg[0] + 1))
    t_cg = time.perf_counter() - t0

    iter_ssor = [0]
    t_ssor = 0.0
    if use_ssor:
        try:
            M_ssor = make_ssor_pc(K_perm)
            t0 = time.perf_counter()
            u_ssor, _ = cg(K_perm, b_perm, rtol=1e-10, maxiter=5000,
                           M=M_ssor,
                           callback=lambda xk: iter_ssor.__setitem__(0, iter_ssor[0] + 1))
            t_ssor = time.perf_counter() - t0
        except Exception:
            iter_ssor[0] = -1

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
# (a) FEM on structured vs unstructured meshes
# ===================================================================


def run_fem_meshes():
    """FEM bandwidth on structured AND unstructured meshes."""
    results = {}

    for mesh_type in ["structured", "unstructured"]:
        print(f"\n{'-' * 60}")
        print(f"  (a) FEM (P1) -- {mesh_type} mesh")
        print(f"{'-' * 60}")

        if mesh_type == "structured":
            NX = 64
            nodes, elements, boundary = generate_structured_mesh(NX)
        else:
            nodes, elements, boundary = generate_unstructured_mesh(4000)

        N = len(nodes)
        print(f"  {N} nodes, {len(elements)} elements")

        K = assemble_stiffness(nodes, elements)
        b = assemble_rhs(nodes, elements)
        K, b = apply_dirichlet(K, b, boundary)

        natural_perm = np.arange(N)
        rcm_perm, rcm_inv = rcm_reorder(K)
        hilbert_perm, hilbert_inv = hilbert_reorder(nodes, p=7)

        orderings = [
            ("Natural", natural_perm, natural_perm),
            ("RCM", rcm_perm, rcm_inv),
            ("Hilbert", hilbert_perm, hilbert_inv),
        ]

        if mesh_type == "unstructured":
            rand_perm, rand_inv = random_reorder(N)
            orderings.insert(0, ("Random", rand_perm, rand_inv))

        mesh_results = []
        for name, perm, inv in orderings:
            print(f"  -> {name}...", end="", flush=True)
            r = solve_with_ordering(K, b, perm, inv, name)
            print(f" BW_avg={r['bw_avg']:.0f}, CG={r['iter_cg']}, "
                  f"CG+SSOR={r['iter_ssor']}, t_SSOR={r['time_ssor']:.1f} ms")
            mesh_results.append(r)

        results[mesh_type] = {
            "results": mesh_results,
            "nodes": nodes,
            "elements": elements,
        }

    return results


# ===================================================================
# (b) Stochastic PDE with SSOR
# ===================================================================


def generate_kl_modes(nodes, elements, n_kl=5, corr_length=0.3):
    """Karhunen-Loeve modes for the random diffusion field."""
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


def run_stochastic_pde():
    """(b) Stochastic PDE: Monte Carlo with SSOR + reordering.

    Run on an unstructured mesh, where ordering matters most.
    """
    print(f"\n{'-' * 60}")
    print(f"  (b) Stochastic PDE -- Monte Carlo + SSOR")
    print(f"{'-' * 60}")

    N_REAL = 50
    N_KL = 5

    nodes, elements, boundary = generate_unstructured_mesh(2000, seed=99)
    N = len(nodes)
    print(f"  Unstructured mesh: {N} nodes, {len(elements)} elements")
    print(f"  MC realisations: {N_REAL}, KL modes: {N_KL}")

    kl_modes = generate_kl_modes(nodes, elements, n_kl=N_KL)
    b_base = assemble_rhs(nodes, elements)

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
                _, info = cg(K_p, b_p, rtol=1e-8, maxiter=5000, M=M_ssor,
                             callback=lambda xk: it.__setitem__(0, it[0] + 1))
                dt = time.perf_counter() - t0
                stats[label]["iters"].append(it[0])
                stats[label]["times"].append(dt * 1000)
            except Exception:
                stats[label]["iters"].append(-1)
                stats[label]["times"].append(-1)

    print(f"\n  Results (mean over {N_REAL} realisations, CG+SSOR):")
    for label in ["natural", "hilbert", "rcm"]:
        iters = [x for x in stats[label]["iters"] if x > 0]
        times = [x for x in stats[label]["times"] if x > 0]
        if iters:
            print(f"  -> {label:>10} : {np.mean(iters):.0f} iterations, "
                  f"{np.mean(times):.1f} ms/solve")

    return stats


# ===================================================================
# (c) Hilbert-guided adaptive mesh refinement
# ===================================================================


def adaptive_hilbert_mesh(nodes, elements, solution,
                          threshold_refine=0.7, threshold_coarsen=0.1):
    """Gradient-driven AMR with Hilbert tagging."""
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
# Figure
# ===================================================================


def plot_results(results_7c, stats_7d, nodes_s, elems_s, sol_s,
                 nodes_u, elems_u, sol_u,
                 gradients, refined, coarsened, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("FEM bandwidth reduction by Hilbert reordering (Thm. 10.2)",
                 fontsize=14, fontweight="bold")
    colors_4 = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"]
    colors_3 = ["#3498db", "#e74c3c", "#2ecc71"]

    ax = axes[0, 0]
    res_s = results_7c["structured"]["results"]
    names = [r["name"] for r in res_s]
    bw = [r["bw_avg"] for r in res_s]
    bars = ax.bar(names, bw, color=colors_3, alpha=0.8)
    for bar, val in zip(bars, bw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean bandwidth $\\overline{\\mathrm{BW}}(\\mathbf{K})$")
    ax.set_title("Structured mesh: bandwidth\n(lower = better)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    res_u = results_7c["unstructured"]["results"]
    names = [r["name"] for r in res_u]
    bw = [r["bw_avg"] for r in res_u]
    c = colors_4 if len(res_u) == 4 else colors_3
    bars = ax.bar(names, bw, color=c, alpha=0.8)
    for bar, val in zip(bars, bw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean bandwidth $\\overline{\\mathrm{BW}}(\\mathbf{K})$")
    ax.set_title("Unstructured mesh: bandwidth\n(lower = better)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 2]
    iter_s = [r["iter_ssor"] for r in res_u]
    bars = ax.bar(names, iter_s, color=c, alpha=0.8)
    for bar, val in zip(bars, iter_s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CG+SSOR iterations")
    ax.set_title("Unstructured mesh: CG+SSOR\n(lower = better)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 3]
    for label, color, lbl in [("natural", "#3498db", "Natural"),
                               ("hilbert", "#2ecc71", "Hilbert"),
                               ("rcm", "#e74c3c", "RCM")]:
        iters = [x for x in stats_7d[label]["iters"] if x > 0]
        if iters:
            ax.hist(iters, bins=15, alpha=0.5, color=color,
                    label=f"{lbl} (mean={np.mean(iters):.0f})")
    ax.set_xlabel("CG+SSOR iterations")
    ax.set_ylabel("MC realisations")
    ax.set_title("(b) Stochastic PDE Monte Carlo + SSOR\n(unstructured mesh)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    triang_s = mtri.Triangulation(nodes_s[:, 0], nodes_s[:, 1], elems_s)
    tcf = ax.tripcolor(triang_s, sol_s, cmap="viridis", shading="flat")
    plt.colorbar(tcf, ax=ax, shrink=0.8)
    ax.set_title("Solution $u$ (structured mesh)")
    ax.set_aspect("equal")

    ax = axes[1, 1]
    triang_u = mtri.Triangulation(nodes_u[:, 0], nodes_u[:, 1], elems_u)
    tcf = ax.tripcolor(triang_u, sol_u, cmap="viridis", shading="flat")
    plt.colorbar(tcf, ax=ax, shrink=0.8)
    ax.set_title("Solution $u$ (unstructured mesh)")
    ax.set_aspect("equal")

    ax = axes[1, 2]
    tcf = ax.tripcolor(triang_s, gradients, cmap="hot", shading="flat")
    plt.colorbar(tcf, ax=ax, shrink=0.8)
    ax.set_title("(c) $|\\nabla u|$ -- AMR indicator")
    ax.set_aspect("equal")

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
    ax.set_title(f"(c) Hilbert-guided AMR\nred = {len(refined)} refined, "
                 f"blue = {len(coarsened)} coarsened")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "fem_bandwidth.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")
    return path


# ===================================================================
# Main
# ===================================================================


def main():
    np.random.seed(42)
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    print("=" * 65)
    print("  FEM bandwidth reduction by Hilbert reordering (Thm. 10.2)")
    print("  Altius Academy SNC")
    print("=" * 65)

    results_7c = run_fem_meshes()

    stats_7d = run_stochastic_pde()

    print(f"\n{'-' * 60}")
    print(f"  (c) Hilbert-guided adaptive mesh refinement")
    print(f"{'-' * 60}")

    nodes_s = results_7c["structured"]["nodes"]
    elems_s = results_7c["structured"]["elements"]
    sol_s = results_7c["structured"]["results"][-1]["solution"]

    nodes_u = results_7c["unstructured"]["nodes"]
    elems_u = results_7c["unstructured"]["elements"]
    sol_u = results_7c["unstructured"]["results"][-1]["solution"]

    refined, coarsened, gradients = adaptive_hilbert_mesh(
        nodes_s, elems_s, sol_s, threshold_refine=0.7, threshold_coarsen=0.1
    )
    print(f"  -> elements to refine  : {len(refined)} "
          f"({100*len(refined)/len(elems_s):.1f}%)")
    print(f"  -> elements to coarsen : {len(coarsened)} "
          f"({100*len(coarsened)/len(elems_s):.1f}%)")

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")

    for mesh_type in ["structured", "unstructured"]:
        print(f"\n  (a) FEM -- {mesh_type} mesh:")
        for r in results_7c[mesh_type]["results"]:
            print(f"    {r['name']:>10} : bw_avg={r['bw_avg']:.0f}, "
                  f"CG={r['iter_cg']}, CG+SSOR={r['iter_ssor']}, "
                  f"t_SSOR={r['time_ssor']:.1f} ms")

        res = results_7c[mesh_type]["results"]
        nat = next(r for r in res if r["name"] == "Natural")
        hil = next(r for r in res if r["name"] == "Hilbert")
        bw_gain = (1 - hil["bw_avg"] / nat["bw_avg"]) * 100
        ssor_gain = (1 - hil["iter_ssor"] / nat["iter_ssor"]) * 100 if nat["iter_ssor"] > 0 else 0
        print(f"  -> Hilbert vs Natural : bandwidth {bw_gain:+.1f}%, "
              f"SSOR iters {ssor_gain:+.1f}%")

    print(f"\n  (b) Stochastic PDE Monte Carlo (unstructured mesh):")
    for label in ["natural", "hilbert", "rcm"]:
        iters = [x for x in stats_7d[label]["iters"] if x > 0]
        times = [x for x in stats_7d[label]["times"] if x > 0]
        if iters:
            print(f"    {label:>10} : {np.mean(iters):.0f} iterations, "
                  f"{np.mean(times):.1f} ms/solve")

    print(f"\n  (c) Adaptive mesh:")
    print(f"    {len(refined)} elements to refine, {len(coarsened)} to coarsen")

    print(f"\n[+] Generating figure...")
    plot_results(results_7c, stats_7d, nodes_s, elems_s, sol_s,
                 nodes_u, elems_u, sol_u,
                 gradients, refined, coarsened, OUTPUT_DIR)

    print(f"\n{'=' * 65}")
    print(f"  fem_bandwidth done.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
