"""
P1 finite elements on tetrahedra for the quasi-linear anisotropic diffusion

    d_t u = div( g(|grad u_sigma|^2) grad u ),   d_nu u = 0 on the boundary,
    g(s) = 1 / (1 + s/lambda^2).

Semi-implicit Euler with g frozen at u^n:

    (M + tau K(u^n)) u^{n+1} = M u^n.

The Neumann condition is natural for the P1 weak form, so no boundary
degrees of freedom are eliminated.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "element_gradients",
    "assemble_mass",
    "assemble_stiffness",
    "mollify",
    "edge_diffusivity",
    "step",
]


def element_gradients(points, tets):
    """Return (volumes, grads) with grads of shape (T, 4, 3).

    grads[t, a, :] is the gradient of the P1 basis function of local
    vertex ``a`` on tetrahedron ``t`` (constant on the element).
    """
    p = points[tets]                       # (T, 4, 3)
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    e3 = p[:, 3] - p[:, 0]
    J = np.stack([e1, e2, e3], axis=1)     # (T, 3, 3) rows = e_i
    det = np.linalg.det(J)
    vol = np.abs(det) / 6.0
    Jinv = np.linalg.inv(J)                # (T, 3, 3)
    # grad(lambda_a) for a = 1,2,3 are the columns of J^{-1}
    g123 = np.transpose(Jinv, (0, 2, 1))   # (T, 3, 3): row a-1 = grad lambda_a
    g0 = -g123.sum(axis=1, keepdims=True)
    grads = np.concatenate([g0, g123], axis=1)   # (T, 4, 3)
    return vol, grads


def _assemble(tets, Ke, n):
    """Scatter (T, 4, 4) element matrices into a CSR matrix of size n."""
    T = tets.shape[0]
    rows = np.repeat(tets, 4, axis=1).ravel()
    cols = np.tile(tets, (1, 4)).ravel()
    A = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
    return A.tocsr()


def assemble_mass(points, tets, lumped=False):
    vol, _ = element_gradients(points, tets)
    n = points.shape[0]
    if lumped:
        Me = np.zeros((tets.shape[0], 4, 4))
        idx = np.arange(4)
        Me[:, idx, idx] = (vol / 4.0)[:, None]
    else:
        base = (np.ones((4, 4)) + np.eye(4)) / 20.0
        Me = vol[:, None, None] * base[None, :, :]
    return _assemble(tets, Me, n)


def assemble_stiffness(points, tets, g=None):
    """Assemble K = sum_e g_e * vol_e * G_e G_e^T (G_e = (4,3) gradients)."""
    vol, grads = element_gradients(points, tets)
    coef = vol if g is None else vol * np.asarray(g, dtype=float)
    Ke = coef[:, None, None] * np.einsum("tad,tbd->tab", grads, grads)
    return _assemble(tets, Ke, points.shape[0])


def mollify(u, M, K0, sigma):
    """Discrete Gaussian mollification: one implicit heat step of
    variance sigma^2, i.e. solve (M + sigma^2/2 K0) u_sigma = M u.

    This is the mesh-based counterpart of the convolution with a Gaussian
    of standard deviation ``sigma``; it agrees with it to first order in
    sigma^2 and, unlike a voxel convolution, is intrinsic to the mesh.
    """
    A = (M + 0.5 * sigma ** 2 * K0).tocsc()
    return spla.spsolve(A, M @ u)


def edge_diffusivity(points, tets, u_sigma, lam):
    """g(|grad u_sigma|^2) evaluated per element, plus |grad u_sigma|."""
    _, grads = element_gradients(points, tets)
    gu = np.einsum("tad,ta->td", grads, u_sigma[tets])
    s = np.einsum("td,td->t", gu, gu)
    g = 1.0 / (1.0 + s / lam ** 2)
    return g, np.sqrt(s)


def step(M, K, tau, u, rtol=1e-10, maxiter=2000, precond=None, callback=None):
    """One semi-implicit Euler step; returns (u_next, n_iterations)."""
    A = (M + tau * K).tocsr()
    b = M @ u
    it = {"n": 0}

    def cb(xk):
        it["n"] += 1
        if callback is not None:
            callback(xk)

    x, info = spla.cg(A, b, x0=u, rtol=rtol, atol=0.0, maxiter=maxiter,
                      M=precond, callback=cb)
    if info != 0:
        raise RuntimeError(f"CG failed to converge (info={info})")
    return x, it["n"]
