"""
The model problem: quasi-linear anisotropic diffusion on an MRI volume,
discretised by P1 finite elements on an unstructured adaptive tetrahedral
mesh.

    d_t u = div( g(|grad u_sigma|^2) grad u ),  u(0) = u_0,  d_nu u = 0,
    g(s) = 1/(1 + s/lambda^2).

The domain Omega is the *brain region* of the template, not its bounding
box: the mask is genuinely non-convex, which is what forces the mesh to be
unstructured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import fem, mesh as meshmod

__all__ = ["Volume", "load_mni152", "load_volume", "DiffusionRun"]


@dataclass
class Volume:
    data: np.ndarray            # (I, J, K) float64, normalised to [0, 1]
    affine: np.ndarray          # (4, 4) voxel -> mm
    name: str

    @property
    def inv_affine(self):
        return np.linalg.inv(self.affine)

    def to_voxel(self, xyz):
        inv = self.inv_affine
        return xyz @ inv[:3, :3].T + inv[:3, 3]

    def bounding_box(self, threshold):
        idx = np.argwhere(self.data > threshold)
        lo = idx.min(axis=0).astype(float)
        hi = idx.max(axis=0).astype(float)
        corners = np.array(np.meshgrid(*[[lo[i], hi[i]] for i in range(3)],
                                       indexing="ij")).reshape(3, -1).T
        mm = corners @ self.affine[:3, :3].T + self.affine[:3, 3]
        return np.stack([mm.min(axis=0), mm.max(axis=0)], axis=1)

    def sample(self, xyz, order=1):
        """Trilinear (order=1) or nearest (order=0) sampling in mm."""
        v = self.to_voxel(np.atleast_2d(xyz))
        if order == 0:
            ijk = np.rint(v).astype(int)
            ok = np.all((ijk >= 0) & (ijk < np.array(self.data.shape)), axis=1)
            out = np.zeros(len(ijk))
            ijk = np.clip(ijk, 0, np.array(self.data.shape) - 1)
            out[ok] = self.data[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
            return out
        f = np.floor(v).astype(int)
        t = v - f
        shape = np.array(self.data.shape)
        out = np.zeros(len(v))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = f + np.array([dx, dy, dz])
                    ok = np.all((idx >= 0) & (idx < shape), axis=1)
                    idc = np.clip(idx, 0, shape - 1)
                    w = ((1 - t[:, 0]) if dx == 0 else t[:, 0]) * \
                        ((1 - t[:, 1]) if dy == 0 else t[:, 1]) * \
                        ((1 - t[:, 2]) if dz == 0 else t[:, 2])
                    out += np.where(ok, w * self.data[idc[:, 0], idc[:, 1], idc[:, 2]], 0.0)
        return out


def load_mni152(resolution=2):
    from nilearn import datasets, image
    img = datasets.load_mni152_template(resolution=resolution)
    data = np.asarray(img.get_fdata(), dtype=float)
    m = data.max()
    if m > 0:
        data = data / m
    return Volume(data=data, affine=np.asarray(img.affine, dtype=float),
                  name=f"MNI152-T1-{resolution}mm")


def load_volume(path):
    import nibabel as nib
    img = nib.load(path)
    data = np.asarray(img.get_fdata(), dtype=float)
    m = data.max()
    if m > 0:
        data = data / m
    return Volume(data=data, affine=np.asarray(img.affine, dtype=float),
                  name=path)


# ----------------------------------------------------------------------

class DiffusionRun:
    """Drives the semi-implicit scheme with periodic adaptive remeshing."""

    def __init__(self, volume: Volume, n_target=6000, threshold=0.15,
                 lam=0.05, sigma=3.0, tau=1.0, seed=0, refine_fraction=0.15):
        self.vol = volume
        self.threshold = threshold
        self.lam = lam
        self.sigma = sigma
        self.tau = tau
        self.seed = seed
        self.refine_fraction = refine_fraction
        self.rng = np.random.default_rng(seed + 9973)

        self.box = volume.bounding_box(threshold)
        self.mesh = meshmod.build_mesh(self.mask_fn, self.box, n_target, seed=seed)
        self.u = self.vol.sample(self.mesh.points)
        self.step_index = 0

    # -- domain ---------------------------------------------------------
    def mask_fn(self, xyz):
        return self.vol.sample(xyz, order=0) > self.threshold

    # -- operators ------------------------------------------------------
    def operators(self):
        pts, tets = self.mesh.points, self.mesh.tets
        M = fem.assemble_mass(pts, tets)
        K0 = fem.assemble_stiffness(pts, tets)
        u_sig = fem.mollify(self.u, M, K0, self.sigma)
        g, gradnorm = fem.edge_diffusivity(pts, tets, u_sig, self.lam)
        K = fem.assemble_stiffness(pts, tets, g)
        return M, K, K0, g, gradnorm

    # -- adaptivity -----------------------------------------------------
    def mark(self, gradnorm):
        """Mark the top ``refine_fraction`` of elements by |grad u_sigma|
        weighted by the element size (a standard gradient indicator)."""
        vol = self.mesh.volumes()
        eta = gradnorm * vol ** (1.0 / 3.0)
        k = max(1, int(self.refine_fraction * eta.size))
        return np.argsort(-eta)[:k]

    def remesh(self, gradnorm, min_edge=None):
        marked = self.mark(gradnorm)
        old_u, old_pts = self.u, self.mesh.points
        new_mesh, n_new = meshmod.refine_mesh(
            self.mesh, marked, self.mask_fn, self.rng, min_edge=min_edge)
        if n_new == 0:
            return 0
        # transfer the solution by nearest-neighbour on the old vertices,
        # exact on the retained ones
        from scipy.spatial import cKDTree
        tree = cKDTree(old_pts)
        _, nn = tree.query(new_mesh.points, k=1)
        self.u = old_u[nn]
        self.mesh = new_mesh
        return n_new
