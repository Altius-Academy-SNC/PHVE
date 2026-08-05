#!/usr/bin/env python3
"""
Build-time pre-computation of the brain surface used by the demonstrator.

Why this exists
---------------
The demonstrator's first render was measured, inside the deployed container,
at 3.58 s. Almost all of it is import cost and data loading, not computation:

    imports          2.43 s   of which nilearn.datasets 1.57 s
                              and scipy.ndimage         0.63 s
    load_brain       0.80 s
    marching_cubes   0.05 s   (31 996 vertices, 63 976 triangles)
    vertex Hilbert   0.30 s

Every one of those imports and both of those steps exist only to turn the
MNI152 template into a triangulated surface, which never changes. Doing it
once when the image is built, and shipping the result, removes nilearn,
scikit-image and scipy.ndimage from the runtime path of the two pages the
demonstrator actually serves.

Running this script is optional: `app_demo.py` falls back to computing the
surface at runtime when the artefact is absent, so a checkout without a build
step still works.

Usage:
    python simulations/prewarm.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "precomputed")
OUT = os.path.join(OUT_DIR, "brain_mesh.npz")


def build():
    from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter

    t0 = time.perf_counter()
    img = load_mni152_template(resolution=2)
    mask_img = load_mni152_brain_mask(resolution=2)
    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = np.asarray(mask_img.dataobj).astype(bool)
    affine = np.asarray(img.affine, dtype=np.float64)
    t1 = time.perf_counter()

    # Same construction as extract_brain_mesh() in app_demo.py; kept in step
    # with it deliberately, since the artefact stands in for its output.
    smooth_mask = gaussian_filter(mask.astype(np.float32), sigma=1.0)
    verts_vox, faces, normals, _ = marching_cubes(smooth_mask, level=0.5)
    ones = np.ones((verts_vox.shape[0], 1))
    verts_mm = (np.hstack([verts_vox, ones]) @ affine.T)[:, :3]
    vi = np.clip(np.round(verts_vox[:, 0]).astype(int), 0, data.shape[0] - 1)
    vj = np.clip(np.round(verts_vox[:, 1]).astype(int), 0, data.shape[1] - 1)
    vk = np.clip(np.round(verts_vox[:, 2]).astype(int), 0, data.shape[2] - 1)
    intensity = data[vi, vj, vk]
    t2 = time.perf_counter()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        OUT,
        verts_mm=verts_mm.astype(np.float32),
        faces=faces.astype(np.int32),
        normals=normals.astype(np.float32),
        intensity=intensity.astype(np.float32),
        affine=affine,
        data_shape=np.asarray(data.shape, dtype=np.int64),
        n_brain_voxels=np.int64(mask.sum()),
    )
    t3 = time.perf_counter()

    size = os.path.getsize(OUT) / 2 ** 20
    print("  load        : %5.2f s" % (t1 - t0))
    print("  mesh        : %5.2f s  (%d vertices, %d triangles)"
          % (t2 - t1, verts_mm.shape[0], faces.shape[0]))
    print("  write       : %5.2f s" % (t3 - t2))
    print("  artefact    : %s (%.2f MiB)" % (OUT, size))
    return 0


if __name__ == "__main__":
    sys.exit(build())
