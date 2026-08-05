"""
Unstructured, adaptive tetrahedral meshes for the PHVE experiment notebook.

The mesh is *never* a regular lattice.  Vertices are produced by a jittered
Poisson-type sampling of the physical domain, tetrahedralised by a Delaunay
triangulation, and the tetrahedra whose centroid falls outside the domain
mask are discarded.  Refinement inserts the centroids of the marked
tetrahedra (plus a deterministic jitter) and re-triangulates, which is a
genuine remeshing step and produces graded meshes.

Everything is driven by a single numpy Generator so that a fixed seed
reproduces the mesh bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay

__all__ = ["TetMesh", "sample_domain", "build_mesh", "refine_mesh",
           "unique_edges"]


def unique_edges(tets: np.ndarray, n_vertices: int) -> np.ndarray:
    """Unique undirected edges of a tetrahedral mesh, as (E, 2) with i < j.

    Uses a single int64 key instead of ``np.unique(..., axis=0)``, which is
    an order of magnitude faster on meshes with millions of tetrahedra.
    """
    t = tets.astype(np.int64)
    pairs = np.concatenate([
        t[:, [0, 1]], t[:, [0, 2]], t[:, [0, 3]],
        t[:, [1, 2]], t[:, [1, 3]], t[:, [2, 3]],
    ], axis=0)
    lo = pairs.min(axis=1)
    hi = pairs.max(axis=1)
    key = lo * np.int64(n_vertices) + hi
    key = np.unique(key)
    return np.stack([key // n_vertices, key % n_vertices], axis=1)


@dataclass
class TetMesh:
    """A P1 tetrahedral mesh.

    Attributes
    ----------
    points : (N, 3) float64
        Vertex coordinates, in insertion order.
    tets : (T, 4) int32
        Vertex indices of each tetrahedron.
    box : (3, 2) float64
        Axis-aligned bounding box actually used for the PHVE normalisation.
    """

    points: np.ndarray
    tets: np.ndarray
    box: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tets.shape[0])

    def edges(self) -> np.ndarray:
        """Unique undirected edges as a (E, 2) array with i < j."""
        return unique_edges(self.tets, self.n_vertices)

    def edge_lengths(self) -> np.ndarray:
        e = self.edges()
        return np.linalg.norm(self.points[e[:, 0]] - self.points[e[:, 1]], axis=1)

    def volumes(self) -> np.ndarray:
        p = self.points[self.tets]
        a = p[:, 1] - p[:, 0]
        b = p[:, 2] - p[:, 0]
        c = p[:, 3] - p[:, 0]
        return np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0


# ----------------------------------------------------------------------
# Domain sampling
# ----------------------------------------------------------------------

def sample_domain(mask_fn, box, n_target, rng, jitter=0.75, batch=8):
    """Jittered sampling of ``n_target`` points inside ``mask_fn``.

    A background lattice of spacing ``s`` is used only to organise the
    rejection sampling; every retained point is displaced by a uniform
    jitter of amplitude ``jitter * s``, so the resulting cloud carries no
    lattice structure.  ``batch`` oversamples to compensate for rejection.
    """
    box = np.asarray(box, dtype=float)
    L = box[:, 1] - box[:, 0]
    vol = float(np.prod(L))
    pts = []
    got = 0
    s = (vol / max(n_target, 1)) ** (1.0 / 3.0)
    while got < n_target:
        m = int(batch * (n_target - got)) + 64
        cand = box[:, 0] + rng.random((m, 3)) * L
        cand = cand + jitter * s * (rng.random((m, 3)) - 0.5)
        cand = np.clip(cand, box[:, 0], box[:, 1])
        keep = mask_fn(cand)
        cand = cand[keep]
        if cand.shape[0] == 0:
            batch *= 2
            if batch > 4096:
                raise RuntimeError("domain mask appears empty")
            continue
        pts.append(cand)
        got += cand.shape[0]
    P = np.vstack(pts)[:n_target]
    return P


def _corner_points(box):
    box = np.asarray(box, dtype=float)
    g = np.array(np.meshgrid(*[box[i] for i in range(3)], indexing="ij"))
    return g.reshape(3, -1).T.copy()


def _tetrahedralise(points, mask_fn):
    tri = Delaunay(points, qhull_options="Qbb Qc Qz Q12")
    tets = tri.simplices.astype(np.int32)
    p = points[tets]
    centroid = p.mean(axis=1)
    keep = mask_fn(centroid)
    tets = tets[keep]
    # drop degenerate (near-zero volume) tetrahedra produced by Qhull
    a = p[keep, 1] - p[keep, 0]
    b = p[keep, 2] - p[keep, 0]
    c = p[keep, 3] - p[keep, 0]
    vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    tets = tets[vol > 1e-12 * max(vol.max(), 1e-30)]
    return tets


def _prune_unused(points, tets):
    used = np.unique(tets)
    remap = -np.ones(points.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return points[used], remap[tets].astype(np.int32), used


def build_mesh(mask_fn, box, n_target, seed=0):
    """Initial unstructured mesh: jittered cloud + Delaunay + mask filter."""
    rng = np.random.default_rng(seed)
    P = sample_domain(mask_fn, box, n_target, rng)
    tets = _tetrahedralise(P, mask_fn)
    P, tets, _ = _prune_unused(P, tets)
    return TetMesh(points=P, tets=tets, box=np.asarray(box, float),
                   meta={"seed": seed, "level": 0})


def refine_mesh(mesh: TetMesh, marked: np.ndarray, mask_fn, rng,
                jitter=0.15, min_edge=None):
    """Insert the centroids of the marked tetrahedra and re-triangulate.

    New vertices are appended after the existing ones, so the *natural*
    (insertion) ordering of the refined mesh is the realistic one: old
    vertices keep their labels, new vertices get the largest labels.
    """
    p = mesh.points[mesh.tets[marked]]
    new = p.mean(axis=1)
    # deterministic jitter proportional to the local element size
    h_loc = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    new = new + jitter * h_loc[:, None] * (rng.random(new.shape) - 0.5)
    keep = mask_fn(new)
    new = new[keep]
    if min_edge is not None and new.shape[0] > 0:
        # reject insertions that would create edges shorter than min_edge
        from scipy.spatial import cKDTree
        tree = cKDTree(mesh.points)
        dist, _ = tree.query(new, k=1)
        new = new[dist > min_edge]
    if new.shape[0] == 0:
        return mesh, 0
    P = np.vstack([mesh.points, new])
    tets = _tetrahedralise(P, mask_fn)
    # keep insertion order but drop vertices Qhull left unused
    P2, tets2, used = _prune_unused(P, tets)
    return (TetMesh(points=P2, tets=tets2, box=mesh.box,
                    meta={**mesh.meta, "level": mesh.meta.get("level", 0) + 1}),
            int(new.shape[0]))
