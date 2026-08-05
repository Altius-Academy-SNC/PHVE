"""
PHVE -- interactive demonstrator (Streamlit)

Two tabs, matching the two claims of the software paper that survived
verification:
  - Bijectivity      : the order at which the code separates MNI152 voxels.
  - Prefix search    : a dyadic-region query as a range scan on a sorted column.

Four further tabs (inter-patient stability, DPCM compression, FEM bandwidth,
surface morphing) were removed from the navigation on 2026-08-05: the
statements they displayed were refuted by the experimental campaign in
`experiments/`, and `experiments/LOGBOOK.md` records each refutation with the
script that produced it. Their page functions are kept below, unreferenced, so
that the history is not lost.

Usage:
    streamlit run app_demo.py

Author: Paul Guindo, Altius Academy SNC.
"""

import math
import os
import time
from collections import Counter

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.spatial import Delaunay

from codec3d import (
    VOLUMES, xyz2d, encode, decode, truncate,
    _code_length_3d, _int_to_base29, _canonical_precision_3d,
    verify_bijectivity,
)

# ===================================================================
# Page configuration
# ===================================================================

st.set_page_config(
    page_title="PHVE",
    page_icon="🧠",
    layout="wide",
)


# ===================================================================
# Cached loaders
# ===================================================================

@st.cache_data
def load_brain():
    """Load MNI152 T1w + brain mask via nilearn."""
    from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
    img = load_mni152_template(resolution=2)
    mask_img = load_mni152_brain_mask(resolution=2)
    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = np.asarray(mask_img.dataobj).astype(bool)
    return data, img.affine, mask


@st.cache_data
def extract_brain_mesh(_data, _affine, _mask):
    """Marching-cubes brain surface with per-vertex T1w intensity."""
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter
    smooth_mask = gaussian_filter(_mask.astype(np.float32), sigma=1.0)
    verts_vox, faces, normals, _ = marching_cubes(smooth_mask, level=0.5)
    ones = np.ones((verts_vox.shape[0], 1))
    verts_hom = np.hstack([verts_vox, ones])
    verts_mm = (verts_hom @ _affine.T)[:, :3]
    vi = np.clip(np.round(verts_vox[:, 0]).astype(int), 0, _data.shape[0] - 1)
    vj = np.clip(np.round(verts_vox[:, 1]).astype(int), 0, _data.shape[1] - 1)
    vk = np.clip(np.round(verts_vox[:, 2]).astype(int), 0, _data.shape[2] - 1)
    intensity = _data[vi, vj, vk]
    return verts_mm, faces, normals, intensity


_PRECOMPUTED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "precomputed", "brain_mesh.npz")


@st.cache_data
def brain_summary():
    """(volume shape, brain-voxel count, affine).

    Read from the build-time artefact when it exists, so that the two pages
    the demonstrator serves never import nilearn -- which alone costs 1.57 s
    of the 3.58 s first render measured in the container.
    """
    if os.path.exists(_PRECOMPUTED):
        with np.load(_PRECOMPUTED) as z:
            return (tuple(int(v) for v in z["data_shape"]),
                    int(z["n_brain_voxels"]), z["affine"])
    data, affine, mask = load_brain()
    return data.shape, int(mask.sum()), affine


@st.cache_data
def brain_mesh():
    """verts_mm, faces, normals, intensity of the brain surface.

    Falls back to computing the surface when the artefact is absent, so a
    plain checkout without a build step still runs.
    """
    if os.path.exists(_PRECOMPUTED):
        with np.load(_PRECOMPUTED) as z:
            return (z["verts_mm"].astype(np.float64), z["faces"],
                    z["normals"], z["intensity"])
    data, affine, mask = load_brain()
    return extract_brain_mesh(data, affine, mask)


@st.cache_data
def compute_vertex_hilbert(verts_mm, p=5, volume="CR"):
    """Hilbert index Hil_p^{(3)} for every vertex, normalised to [0, 1]."""
    dims = VOLUMES[volume]["dims"]
    n = 1 << p
    max_d = 8 ** p - 1
    k = _code_length_3d(p)

    hilbert_vals = np.zeros(verts_mm.shape[0], dtype=np.float64)
    codes = []
    for idx in range(verts_mm.shape[0]):
        x, y, z = verts_mm[idx]
        vx, vy, vz = x + dims[0]/2, y + dims[1]/2, z + dims[2]/2
        ix = max(0, min(n - 1, int(vx / dims[0] * n)))
        iy = max(0, min(n - 1, int(vy / dims[1] * n)))
        iz = max(0, min(n - 1, int(vz / dims[2] * n)))
        d = xyz2d(p, ix, iy, iz)
        hilbert_vals[idx] = d / max_d
        if idx < 10000:
            codes.append(_int_to_base29(d, k))
    return hilbert_vals, codes


def voxel_to_mm(i, j, k, affine):
    return (affine @ np.array([i, j, k, 1.0]))[:3]


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


# ===================================================================
# Tab 1: Bijectivity (Proposition 1 of the software paper)
# ===================================================================

def page_bijectivity():
    st.header("Bijectivity of $\\mathcal{F}_p^{(3),\\alpha}$ on MNI152")
    st.caption("Proposition 1: the encoding map is injective on the cell-centre grid. On the MNI152 brain mask it separates every voxel from $p = 8$ onwards; $14.1\\%$ still collide at $p = 7$.")

    shape, n_brain_voxels, affine = brain_summary()
    st.caption(f"MNI152 T1w | shape {shape} | 2 mm isotropic | {n_brain_voxels:,} brain voxels")

    col_ctrl, col_3d = st.columns([1, 3])

    with col_ctrl:
        p = st.slider("Order $p$", 3, 7, 5,
                       help="Order of the Hilbert curve used to colour the surface.")
        color_mode = st.radio("Colour by", ["Hilbert index", "T1w intensity", "Prefix region"])
        opacity = st.slider("Opacity", 0.3, 1.0, 0.8, 0.05)
        show_landmarks = st.checkbox("Show anatomical landmarks", value=True)

    with st.spinner("Loading brain surface..."):
        verts_mm, faces, normals, intensity = brain_mesh()

    st.sidebar.metric("Vertices", f"{verts_mm.shape[0]:,}")
    st.sidebar.metric("Triangles", f"{faces.shape[0]:,}")

    if color_mode == "Hilbert index":
        with st.spinner(f"Computing Hil_p^{{(3)}} (p={p})..."):
            hilbert_vals, _ = compute_vertex_hilbert(verts_mm, p=p, volume="CR")
        vertex_colors = hilbert_vals
        colorscale = "HSV"
        colorbar_title = "Hilbert index"
    elif color_mode == "T1w intensity":
        vertex_colors = intensity
        colorscale = "Gray"
        colorbar_title = "T1w intensity"
    else:
        with st.spinner(f"Computing prefix regions (p={p})..."):
            hilbert_vals, _ = compute_vertex_hilbert(verts_mm, p=p, volume="CR")
        dims = VOLUMES["CR"]["dims"]
        n = 1 << p
        k = _code_length_3d(p)
        region_ids = np.zeros(verts_mm.shape[0])
        for idx in range(verts_mm.shape[0]):
            x, y, z = verts_mm[idx]
            vx, vy, vz = x + dims[0]/2, y + dims[1]/2, z + dims[2]/2
            ix = max(0, min(n-1, int(vx / dims[0] * n)))
            iy = max(0, min(n-1, int(vy / dims[1] * n)))
            iz = max(0, min(n-1, int(vz / dims[2] * n)))
            d = xyz2d(p, ix, iy, iz)
            code = _int_to_base29(d, k)
            region_ids[idx] = hash(code[:2]) % 20
        vertex_colors = region_ids
        colorscale = "Rainbow"
        colorbar_title = "Region (2-char prefix)"

    with col_3d:
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(
            x=verts_mm[:, 0], y=verts_mm[:, 1], z=verts_mm[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=vertex_colors,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, len=0.6),
            opacity=opacity,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=300),
            hovertemplate="x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>z: %{z:.1f} mm<extra></extra>",
        ))

        if show_landmarks:
            lm_x, lm_y, lm_z, lm_names, lm_codes = [], [], [], [], []
            for name, (x, y, z) in LANDMARKS.items():
                code = mm_to_phve(x, y, z, p=p, volume="CR")
                lm_x.append(x); lm_y.append(y); lm_z.append(z)
                lm_names.append(name); lm_codes.append(code)
            fig.add_trace(go.Scatter3d(
                x=lm_x, y=lm_y, z=lm_z,
                mode="markers+text",
                marker=dict(size=6, color="red", symbol="diamond"),
                text=lm_codes, textposition="top center",
                textfont=dict(size=9, color="white"),
                hovertext=[f"{n}<br>{c}" for n, c in zip(lm_names, lm_codes)],
                hoverinfo="text", name="Anatomical landmarks",
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.8, y=0.8, z=0.6)),
            ),
            height=700, margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Encoder / decoder")
    col_enc, col_dec = st.columns(2)

    with col_enc:
        st.markdown("**MNI coordinates -> PHVE code**")
        c1, c2, c3 = st.columns(3)
        x_in = c1.number_input("X (mm)", value=0.0, step=5.0, key="enc_x")
        y_in = c2.number_input("Y (mm)", value=0.0, step=5.0, key="enc_y")
        z_in = c3.number_input("Z (mm)", value=0.0, step=5.0, key="enc_z")
        p_enc = st.slider("Order $p$", 3, 10, 8, key="enc_p")
        code = mm_to_phve(x_in, y_in, z_in, p=p_enc, volume="CR")
        decoded = decode(code)
        st.code(code, language=None)
        st.caption(f"Resolution: {decoded['resolution_mm']:.1f} mm")
        with st.expander("Truncation hierarchy (Proposition 2)"):
            raw_full = code.split(":")[1].replace("-", "")
            vol = code.split(":")[0]
            dims = VOLUMES[vol]["dims"]
            for i in range(len(raw_full)):
                c = code if i == 0 else truncate(code, i)
                raw = c.split(":")[1].replace("-", "") if c.split(":")[1] else ""
                if raw:
                    p_t = _canonical_precision_3d(len(raw))
                    n_t = 1 << p_t
                    res = max(dims[0]/n_t, dims[1]/n_t, dims[2]/n_t)
                    st.text(f"  {c:20s}  ~{res:.0f} mm")

    with col_dec:
        st.markdown("**PHVE code -> MNI coordinates**")
        code_input = st.text_input("Code", placeholder="CR:GSX-7X", key="dec_code")
        if code_input:
            try:
                result = decode(code_input)
                d = VOLUMES[result["volume"]]["dims"]
                x_mni = result["x_mm"] - d[0] / 2
                y_mni = result["y_mm"] - d[1] / 2
                z_mni = result["z_mm"] - d[2] / 2
                st.success(f"**({x_mni:.1f}, {y_mni:.1f}, {z_mni:.1f})** mm MNI")
                st.caption(f"Resolution: {result['resolution_mm']:.1f} mm | Volume: {result['volume']}")
            except ValueError as e:
                st.error(str(e))


# ===================================================================
# Tab 2: Prefix search (Proposition 2 of the software paper)
# ===================================================================

def page_prefix_search():
    st.header("Prefix search as an ordered-index range scan (Proposition 2)")
    st.caption("Codes sharing a prefix form a contiguous interval, so a region query is two binary searches on an ordinary ordered column -- no spatial index. Measured against a $k$-d tree in `experiments/exp22_prefix_index.py`: identical result sets, and a cost independent of how many points are returned.")

    col1, col2 = st.columns([1, 3])

    with col1:
        p = st.slider("Order $p$", 3, 6, 4, key="pf_p")
        example = mm_to_phve(0, 0, 0, p=p, volume="CR")
        example_raw = example.split(":")[1].replace("-", "")
        st.caption(f"Brain centre = `{example}` (raw: `{example_raw}`)")
        prefix = st.text_input("Prefix", value=example_raw[:1],
                               help="Leading characters of the code (without the 'CR:' volume tag).")

    with col2:
        verts_mm, faces, normals, intensity = brain_mesh()
        dims = VOLUMES["CR"]["dims"]
        n = 1 << p
        k = _code_length_3d(p)
        match_mask = np.zeros(verts_mm.shape[0], dtype=bool)
        for idx in range(verts_mm.shape[0]):
            x, y, z = verts_mm[idx]
            vx, vy, vz = x + dims[0]/2, y + dims[1]/2, z + dims[2]/2
            ix = max(0, min(n-1, int(vx / dims[0] * n)))
            iy = max(0, min(n-1, int(vy / dims[1] * n)))
            iz = max(0, min(n-1, int(vz / dims[2] * n)))
            d = xyz2d(p, ix, iy, iz)
            code = _int_to_base29(d, k)
            if code.startswith(prefix.upper()):
                match_mask[idx] = True
        vertex_colors = np.where(match_mask, 1.0, 0.2)

        fig = go.Figure()
        fig.add_trace(go.Mesh3d(
            x=verts_mm[:, 0], y=verts_mm[:, 1], z=verts_mm[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=vertex_colors,
            colorscale=[[0, "rgb(80,80,80)"], [1, "rgb(0,255,100)"]],
            showscale=False, opacity=0.8,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
            lightposition=dict(x=100, y=200, z=300),
        ))
        n_match = int(match_mask.sum())
        n_total = verts_mm.shape[0]
        fig.update_layout(
            scene=dict(
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.8, y=0.8, z=0.6)),
            ),
            title=f"Prefix 'CR:{prefix}...' -- {n_match:,}/{n_total:,} vertices ({n_match/max(n_total,1)*100:.1f}%)",
            height=600, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width="stretch")
        st.code(f"SELECT * FROM voxels WHERE phve_code LIKE 'CR:{prefix}%';", language="sql")


# ===================================================================
# Tab 3: Inter-patient stability (Corollary 5.3)
# ===================================================================

def page_interpatient():
    st.header("Inter-patient stability (Corollary 5.3)")
    st.caption("After MNI registration, the same anatomical landmark receives an identical or near-identical PHVE code across patients.")

    data, affine, mask = load_brain()
    p = st.slider("Order $p$", 3, 10, 6, key="s2_p",
                  help="$p=6$ corresponds to a $\\sim$5 mm grid.")
    sigma = st.slider("Residual registration error $\\sigma$ (mm)", 0.1, 2.0, 0.5, 0.1, key="s2_sigma")

    rows = []
    matches = 0
    for name, (x, y, z) in LANDMARKS.items():
        code_a = mm_to_phve(x, y, z, p=p, volume="CR")
        rng_b = np.random.RandomState(hash(name) % 2**31)
        residual = rng_b.normal(0, sigma, size=3)
        code_b = mm_to_phve(x + residual[0], y + residual[1], z + residual[2],
                              p=p, volume="CR")
        raw_a = code_a.split(":")[1].replace("-", "")
        raw_b = code_b.split(":")[1].replace("-", "")
        common = sum(1 for a, b in zip(raw_a, raw_b) if a == b)
        is_match = code_a == code_b
        if is_match:
            matches += 1
        rows.append({
            "Landmark": name,
            "MNI (mm)": f"({x:+d}, {y:+d}, {z:+d})",
            "Patient A": code_a,
            "Patient B": code_b,
            "Match": "Identical" if is_match else f"Common prefix: {common}/{len(raw_a)}",
        })

    st.dataframe(rows, width="stretch", hide_index=True)
    pct = matches / len(LANDMARKS) * 100
    st.metric("Identical codes", f"{matches}/{len(LANDMARKS)} ({pct:.0f}%)")

    st.subheader("Robustness vs. order $p$")
    p_values = list(range(3, 11))
    match_pcts = []
    for p_test in p_values:
        m = 0
        for name, (x, y, z) in LANDMARKS.items():
            code_a = mm_to_phve(x, y, z, p=p_test, volume="CR")
            rng_b = np.random.RandomState(hash(name) % 2**31)
            residual = rng_b.normal(0, sigma, size=3)
            code_b = mm_to_phve(x + residual[0], y + residual[1], z + residual[2],
                                  p=p_test, volume="CR")
            if code_a == code_b:
                m += 1
        match_pcts.append(m / len(LANDMARKS) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_values, y=match_pcts, mode="lines+markers",
                              line=dict(color="#2ecc71", width=3), marker=dict(size=10)))
    fig.update_layout(xaxis_title="Order $p$", yaxis_title="Identical codes (%)",
                       height=350, yaxis=dict(range=[0, 105]))
    st.plotly_chart(fig, width="stretch")


# ===================================================================
# Tab 4: DPCM compression (Theorem 8.2)
# ===================================================================

def _shannon_entropy(signal):
    counts = Counter(signal)
    total = len(signal)
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * np.log2(p)
    return h


def _morton_index_3d(x, y, z):
    d = 0
    for i in range(16):
        d |= ((x >> i) & 1) << (3 * i + 2)
        d |= ((y >> i) & 1) << (3 * i + 1)
        d |= ((z >> i) & 1) << (3 * i)
    return d


@st.cache_data
def compute_dpcm_benchmark(_data, _mask, _affine, p):
    """DPCM benchmark on the brain mask, three traversals."""
    dims = VOLUMES["CR"]["dims"]
    n = 1 << p
    brain_coords = np.argwhere(_mask)
    total = len(brain_coords)

    raster_signal = _data[_mask].astype(np.int32)

    hilbert_indices = np.zeros(total, dtype=np.int64)
    morton_indices = np.zeros(total, dtype=np.int64)
    intensities = np.zeros(total, dtype=np.float64)

    for idx, (i, j, k) in enumerate(brain_coords):
        ijk1 = np.array([i, j, k, 1.0])
        xyz_mm = _affine @ ijk1
        ix = max(0, min(n - 1, int((xyz_mm[0] + dims[0] / 2) / dims[0] * n)))
        iy = max(0, min(n - 1, int((xyz_mm[1] + dims[1] / 2) / dims[1] * n)))
        iz = max(0, min(n - 1, int((xyz_mm[2] + dims[2] / 2) / dims[2] * n)))
        hilbert_indices[idx] = xyz2d(p, ix, iy, iz)
        morton_indices[idx] = _morton_index_3d(ix, iy, iz)
        intensities[idx] = _data[i, j, k]

    hilbert_order = np.argsort(hilbert_indices)
    morton_order = np.argsort(morton_indices)

    hilbert_signal = intensities[hilbert_order].astype(np.int32)
    morton_signal = intensities[morton_order].astype(np.int32)

    dpcm_raster = np.diff(raster_signal)
    dpcm_hilbert = np.diff(hilbert_signal)
    dpcm_morton = np.diff(morton_signal)

    h_orig = _shannon_entropy(raster_signal)
    h_raster = _shannon_entropy(dpcm_raster)
    h_hilbert = _shannon_entropy(dpcm_hilbert)
    h_morton = _shannon_entropy(dpcm_morton)

    return {
        "n_voxels": total,
        "h_orig": h_orig,
        "h_raster": h_raster,
        "h_hilbert": h_hilbert,
        "h_morton": h_morton,
        "red_hilbert_vs_raster": (1 - h_hilbert / h_raster) * 100,
        "red_morton_vs_raster": (1 - h_morton / h_raster) * 100,
        "dpcm_raster": dpcm_raster,
        "dpcm_hilbert": dpcm_hilbert,
        "dpcm_morton": dpcm_morton,
    }


def page_dpcm():
    st.header("DPCM compression along the Hilbert traversal (Theorem 8.2)")
    st.markdown(r"""
The empirical second moment of the DPCM residual along the Hilbert traversal
$\gamma_H$ is bounded by $L^2$ for $L$-Lipschitz signals (independent of grid
size $n$), versus $L^2 \cdot n$ for the raster traversal $\gamma_R$. Under a
zero-mean Gaussian residual model (Section 10.2), this yields a Shannon-entropy
advantage of $\tfrac{1}{2}\log_2 n$ bits per sample asymptotically.

The MNI152 atlas is a near-binary averaged volume and falls outside the
worst-case regime: see Remark 8.4 of the paper.
""")

    data, affine, mask = load_brain()
    p = st.slider("Hilbert order $p$", 4, 7, 6, key="s4_p",
                   help="$p=6$: $64^3$ grid (~5 mm); $p=7$: $128^3$ (~2.5 mm).")

    with st.spinner(f"Computing DPCM (p={p}, {mask.sum():,} voxels)... ~1 min"):
        r = compute_dpcm_benchmark(data, mask, affine, p)

    col1, col2, col3 = st.columns(3)
    col1.metric("Raster entropy", f"{r['h_raster']:.3f} bits")
    col2.metric("Morton entropy", f"{r['h_morton']:.3f} bits",
                delta=f"{r['red_morton_vs_raster']:+.2f}%")
    col3.metric("Hilbert entropy", f"{r['h_hilbert']:.3f} bits",
                delta=f"{r['red_hilbert_vs_raster']:+.2f}%")

    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure()
        methods = ["Raster", "Morton", "Hilbert"]
        vals = [r['h_raster'], r['h_morton'], r['h_hilbert']]
        colors = ["#3498db", "#e67e22", "#2ecc71"]
        fig.add_trace(go.Bar(x=methods, y=vals, marker_color=colors,
                             text=[f"{v:.3f}" for v in vals], textposition="outside"))
        fig.update_layout(
            title="DPCM entropy (bits) -- lower is better",
            yaxis_title="Shannon entropy (bits)",
            height=400, yaxis=dict(range=[0, max(vals) * 1.15]),
        )
        st.plotly_chart(fig, width="stretch")

    with col_b:
        fig = go.Figure()
        bins_spec = dict(start=-200, end=200, size=5)
        fig.add_trace(go.Histogram(x=r['dpcm_raster'], name="Raster",
                                    opacity=0.5, xbins=bins_spec, histnorm="probability density"))
        fig.add_trace(go.Histogram(x=r['dpcm_hilbert'], name="Hilbert",
                                    opacity=0.5, xbins=bins_spec, histnorm="probability density"))
        fig.update_layout(
            title="DPCM residual distribution",
            xaxis_title="residual $r^\\gamma_i$",
            yaxis_title="density",
            barmode="overlay", height=400,
            xaxis=dict(range=[-200, 200]),
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Summary")
    st.dataframe([
        {"Metric": "Brain voxels", "Raster": f"{r['n_voxels']:,}",
         "Morton": f"{r['n_voxels']:,}", "Hilbert": f"{r['n_voxels']:,}"},
        {"Metric": "H(original)", "Raster": f"{r['h_orig']:.3f}",
         "Morton": f"{r['h_orig']:.3f}", "Hilbert": f"{r['h_orig']:.3f}"},
        {"Metric": "H(DPCM)", "Raster": f"{r['h_raster']:.3f}",
         "Morton": f"{r['h_morton']:.3f}", "Hilbert": f"{r['h_hilbert']:.3f}"},
        {"Metric": "Reduction vs raster", "Raster": "--",
         "Morton": f"{r['red_morton_vs_raster']:+.2f}%",
         "Hilbert": f"{r['red_hilbert_vs_raster']:+.2f}%"},
        {"Metric": "Mean |residual|",
         "Raster": f"{np.abs(r['dpcm_raster']).mean():.1f}",
         "Morton": f"{np.abs(r['dpcm_morton']).mean():.1f}",
         "Hilbert": f"{np.abs(r['dpcm_hilbert']).mean():.1f}"},
        {"Metric": "Zero residuals",
         "Raster": f"{np.sum(r['dpcm_raster']==0)/len(r['dpcm_raster'])*100:.1f}%",
         "Morton": f"{np.sum(r['dpcm_morton']==0)/len(r['dpcm_morton'])*100:.1f}%",
         "Hilbert": f"{np.sum(r['dpcm_hilbert']==0)/len(r['dpcm_hilbert'])*100:.1f}%"},
    ], width="stretch", hide_index=True)


# ===================================================================
# Tab 5: FEM bandwidth (Theorem 10.2)
# ===================================================================

def _xy2d_2d(p, x, y):
    """2D Hilbert encoder for the FEM tab."""
    n = 1 << p
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s >>= 1
    return d


@st.cache_data
def build_fem_problem(n_points, mesh_type, seed=42):
    """Generate mesh + assemble stiffness matrix and rhs for the FEM tab."""
    rng = np.random.RandomState(seed)
    if mesh_type == "structured":
        nx = max(8, int(np.sqrt(n_points)))
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
        boundary = np.array(sorted(boundary))
    else:
        pts_int = rng.rand(n_points, 2)
        n_edge = max(20, int(np.sqrt(n_points)))
        t = np.linspace(0, 1, n_edge, endpoint=False)
        pts_b = np.column_stack([t, np.zeros(n_edge)])
        pts_t = np.column_stack([t, np.ones(n_edge)])
        pts_l = np.column_stack([np.zeros(n_edge), t])
        pts_r = np.column_stack([np.ones(n_edge), t])
        nodes = np.vstack([pts_int, pts_b, pts_t, pts_l, pts_r])
        _, uidx = np.unique(np.round(nodes, 8), axis=0, return_index=True)
        nodes = nodes[np.sort(uidx)]
        tri = Delaunay(nodes)
        elements = tri.simplices
        eps = 1e-6
        boundary = np.where(
            (nodes[:, 0] < eps) | (nodes[:, 0] > 1 - eps) |
            (nodes[:, 1] < eps) | (nodes[:, 1] > 1 - eps)
        )[0]

    # Assembly
    N = len(nodes)
    rows, cols, vals = [], [], []
    for elem in elements:
        x = nodes[elem, 0]; y = nodes[elem, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) -
                         (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-15:
            continue
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        for i in range(3):
            for j in range(3):
                rows.append(elem[i]); cols.append(elem[j])
                vals.append((b[i] * b[j] + c[i] * c[j]) / (4 * area))
    K = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    # rhs: Gaussian source at (0.5, 0.5)
    rhs = np.zeros(N)
    cx, cy, sigma = 0.5, 0.5, 0.1
    for elem in elements:
        x = nodes[elem, 0]; y = nodes[elem, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) -
                         (x[2] - x[0]) * (y[1] - y[0]))
        for i in range(3):
            r2 = (nodes[elem[i], 0] - cx) ** 2 + (nodes[elem[i], 1] - cy) ** 2
            rhs[elem[i]] += area / 3 * 100 * math.exp(-r2 / (2 * sigma ** 2))

    # Dirichlet
    K = K.tolil()
    for node in boundary:
        K[node, :] = 0
        K[:, node] = 0
        K[node, node] = 1
        rhs[node] = 0
    K = K.tocsr()

    return nodes, elements, boundary, K, rhs


def _solve_one_ordering(K, rhs, perm):
    P = sparse.eye(K.shape[0], format="csr")[perm, :]
    K_p = (P @ K @ P.T).tocsr()
    rhs_p = P @ rhs
    rows, cols = K_p.nonzero()
    bw_avg = float(np.mean(np.abs(rows - cols))) if len(rows) > 0 else 0.0
    bw_max = int(np.max(np.abs(rows - cols))) if len(rows) > 0 else 0

    iter_count = [0]
    t0 = time.perf_counter()
    u_p, info = cg(K_p, rhs_p, rtol=1e-8, maxiter=2000,
                    callback=lambda xk: iter_count.__setitem__(0, iter_count[0] + 1))
    dt = (time.perf_counter() - t0) * 1000
    return {
        "K_p": K_p,
        "u": P.T @ u_p,
        "bw_avg": bw_avg,
        "bw_max": bw_max,
        "iter_cg": iter_count[0],
        "time_ms": dt,
    }


def page_fem():
    st.header("FEM stiffness-matrix bandwidth (Theorem 10.2)")
    st.markdown(r"""
Reordering the nodes of an unstructured mesh by their 2D Hilbert index
$\mathrm{Hil}_p^{(2)}$ reduces the bandwidth of the assembled stiffness matrix
$\mathbf{K}$ to $C_d (h/\Delta)^d + O(n^{d-1})$ (Theorem 10.2(a)), against
$N - 1$ in the worst-case natural ordering (10.2(b)).
""")

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        mesh_type = st.radio("Mesh type", ["unstructured", "structured"], key="fem_mesh")
        n_points = st.slider("Number of nodes", 200, 3000, 1000, 100, key="fem_n")
        run = st.button("Run", type="primary", key="fem_run")

    if not run:
        st.info("Click **Run** to assemble $\\mathbf{K}$ and benchmark the four orderings.")
        return

    with st.spinner("Building mesh and assembling K..."):
        nodes, elements, boundary, K, rhs = build_fem_problem(n_points, mesh_type)

    N = K.shape[0]
    st.caption(f"{N} nodes, {len(elements)} elements")

    natural_perm = np.arange(N)

    rng = np.random.RandomState(123)
    random_perm = rng.permutation(N)

    rcm_perm = np.array(reverse_cuthill_mckee(K))

    p_h = 7
    n_h = 1 << p_h
    h_idx = np.zeros(N, dtype=np.int64)
    for i, (x, y) in enumerate(nodes):
        ix = min(n_h - 1, max(0, int(x * n_h)))
        iy = min(n_h - 1, max(0, int(y * n_h)))
        h_idx[i] = _xy2d_2d(p_h, ix, iy)
    hilbert_perm = np.argsort(h_idx)

    orderings = [("Random", random_perm), ("Natural", natural_perm),
                 ("RCM", rcm_perm), ("Hilbert", hilbert_perm)]
    if mesh_type == "structured":
        orderings = orderings[1:]

    with st.spinner("Solving with each ordering..."):
        results = {name: _solve_one_ordering(K, rhs, perm) for name, perm in orderings}

    metrics_cols = st.columns(len(orderings))
    for col, (name, _) in zip(metrics_cols, orderings):
        r = results[name]
        col.metric(f"{name} -- bandwidth (mean)", f"{r['bw_avg']:.0f}",
                   delta=f"CG iters: {r['iter_cg']}")

    st.subheader("Sparsity pattern of $\\mathbf{K}$ under each ordering")
    n_cols = len(orderings)
    fig = make_subplots(rows=1, cols=n_cols,
                        subplot_titles=[name for name, _ in orderings],
                        horizontal_spacing=0.04)
    for col_idx, (name, _) in enumerate(orderings, 1):
        K_p = results[name]["K_p"].tocoo()
        fig.add_trace(go.Scattergl(
            x=K_p.col, y=K_p.row, mode="markers",
            marker=dict(size=2, color="#2c3e50"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col_idx)
        fig.update_xaxes(title_text="column $j$", row=1, col=col_idx,
                         showgrid=False, scaleanchor=f"y{col_idx if col_idx > 1 else ''}",
                         scaleratio=1)
        fig.update_yaxes(title_text="row $i$" if col_idx == 1 else "",
                         row=1, col=col_idx, autorange="reversed", showgrid=False)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Summary")
    rows_summary = []
    for name, _ in orderings:
        r = results[name]
        rows_summary.append({
            "Ordering": name,
            "BW(K) max": r["bw_max"],
            "BW(K) mean": f"{r['bw_avg']:.1f}",
            "CG iterations": r["iter_cg"],
            "Solve time (ms)": f"{r['time_ms']:.1f}",
        })
    st.dataframe(rows_summary, width="stretch", hide_index=True)

    if "Hilbert" in results and "Natural" in results:
        bw_gain = (1 - results["Hilbert"]["bw_avg"] / results["Natural"]["bw_avg"]) * 100
        cg_gain = (1 - results["Hilbert"]["iter_cg"] / results["Natural"]["iter_cg"]) * 100
        st.success(f"**Hilbert vs Natural** -- mean bandwidth: {bw_gain:+.1f}%, CG iterations: {cg_gain:+.1f}%")

    st.subheader("FEM solution $u$")
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    u = results["Hilbert" if "Hilbert" in results else "Natural"]["u"]
    fig = go.Figure(data=go.Contour(
        x=nodes[:, 0], y=nodes[:, 1], z=u,
        colorscale="Viridis",
        contours=dict(showlines=False),
    ))
    fig.update_layout(height=400, xaxis_title="x", yaxis_title="y",
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig, width="stretch")


# ===================================================================
# Tab 6: Surface morphing (Proposition 10.4)
# ===================================================================

@st.cache_data
def compute_surface_hilbert(_verts_mm, p=5):
    """Hilbert index normalised to [0, 1] for each surface vertex."""
    dims = VOLUMES["CR"]["dims"]
    n = 1 << p
    max_d = 8 ** p - 1
    N = len(_verts_mm)
    h_norm = np.zeros(N, dtype=np.float64)
    for i in range(N):
        x, y, z = _verts_mm[i]
        ix = max(0, min(n-1, int((x + dims[0]/2) / dims[0] * n)))
        iy = max(0, min(n-1, int((y + dims[1]/2) / dims[1] * n)))
        iz = max(0, min(n-1, int((z + dims[2]/2) / dims[2] * n)))
        h_norm[i] = xyz2d(p, ix, iy, iz) / max_d
    return h_norm


def _edge_smoothness(displacements, faces):
    """Mean per-edge displacement difference."""
    edges = set()
    for f in faces:
        for i in range(3):
            edges.add(tuple(sorted([f[i], f[(i+1) % 3]])))
    diffs = np.array([abs(displacements[v1] - displacements[v2]) for v1, v2 in edges])
    return diffs.mean(), diffs


def page_morphing():
    st.header("Surface morphing via Hilbert reparameterisation (Prop. 10.4)")
    st.markdown(r"""
A scalar deformation $D$ applied along the Hilbert-normalised parameter
$h(v) = \mathrm{Hil}_p^{(3)}(v) / (n^d - 1)$ produces per-edge displacement
differences bounded by $L_D \cdot C_d \cdot n^{-d} \cdot (\ell/\Delta)^d$,
whereas a random reordering yields $\mathbb{E}[|D(r_i) - D(r_j)|] \le L_D / 3$
independently of $n$ and $\ell$.
""")

    data, affine, mask = load_brain()

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        amplitude = st.slider("Amplitude (mm)", 0.5, 8.0, 3.0, 0.5, key="s6_a")
        frequency = st.slider("Frequency", 1.0, 10.0, 4.0, 0.5, key="s6_f")
        phase = st.slider("Phase", 0.0, 6.28, 0.0, 0.1, key="s6_ph")
        p_morph = st.slider("Order $p$", 3, 6, 5, key="s6_p")
        show_mode = st.radio("Display", ["Hilbert (smooth)", "Random (noise)", "Side by side"],
                              key="s6_mode")

    with st.spinner("Loading brain surface..."):
        verts_mm, faces, normals, intensity = brain_mesh()

    with st.spinner(f"Computing Hilbert indices (p={p_morph})..."):
        h_norm = compute_surface_hilbert(verts_mm, p=p_morph)

    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_len[norms_len == 0] = 1
    unit_normals = normals / norms_len

    disp_h = amplitude * np.sin(2 * np.pi * frequency * h_norm + phase)
    deformed_h = verts_mm + unit_normals * disp_h[:, np.newaxis]

    rng = np.random.RandomState(42)
    rand_norm = rng.permutation(len(verts_mm)).astype(np.float64) / len(verts_mm)
    disp_r = amplitude * np.sin(2 * np.pi * frequency * rand_norm + phase)
    deformed_r = verts_mm + unit_normals * disp_r[:, np.newaxis]

    sm_h, _ = _edge_smoothness(disp_h, faces)
    sm_r, _ = _edge_smoothness(disp_r, faces)

    st.sidebar.metric("Hilbert smoothness", f"{sm_h:.4f} mm")
    st.sidebar.metric("Random smoothness", f"{sm_r:.4f} mm")
    st.sidebar.metric("Ratio", f"{sm_r/sm_h:.1f}x")

    with col_main:
        def _make_mesh(verts, disp, title, colorscale="RdBu"):
            return go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=disp, colorscale=colorscale,
                colorbar=dict(title="displ. (mm)", len=0.6),
                cmin=-amplitude, cmax=amplitude,
                opacity=0.85,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
                lightposition=dict(x=100, y=200, z=300),
                name=title,
            )

        scene = dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                     aspectmode="data",
                     camera=dict(eye=dict(x=1.8, y=0.8, z=0.6)))

        if show_mode == "Side by side":
            fig = make_subplots(rows=1, cols=2,
                                specs=[[{"type": "scene"}, {"type": "scene"}]],
                                subplot_titles=[
                                    f"Hilbert (smoothness: {sm_h:.4f})",
                                    f"Random  (smoothness: {sm_r:.4f})"])
            fig.add_trace(_make_mesh(deformed_h, disp_h, "Hilbert"), row=1, col=1)
            fig.add_trace(_make_mesh(deformed_r, disp_r, "Random"), row=1, col=2)
            fig.update_layout(height=600, scene=scene, scene2=scene,
                              margin=dict(l=0, r=0, t=40, b=0))
        else:
            if show_mode == "Hilbert (smooth)":
                fig = go.Figure(data=[_make_mesh(deformed_h, disp_h, "Hilbert")])
                title = f"Hilbert deformation -- smoothness: {sm_h:.4f} mm"
            else:
                fig = go.Figure(data=[_make_mesh(deformed_r, disp_r, "Random")])
                title = f"Random deformation -- smoothness: {sm_r:.4f} mm"
            fig.update_layout(height=600, scene=scene, title=title,
                              margin=dict(l=0, r=0, t=40, b=0))

        st.plotly_chart(fig, width="stretch")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Hilbert", "Random"], y=[sm_h, sm_r],
                             marker_color=["#2ecc71", "#e74c3c"],
                             text=[f"{sm_h:.4f}", f"{sm_r:.4f}"],
                             textposition="outside"))
        fig.update_layout(title="Mean neighbour difference (mm) -- lower is smoother",
                           yaxis_title="mm", height=350)
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown(f"""
        ### Results

        | | Hilbert | Random |
        |---|---|---|
        | Mean neighbour diff. | **{sm_h:.4f}** mm | {sm_r:.4f} mm |
        | Ratio | **{sm_r/sm_h:.1f}x smoother** | -- |
        | Vertices | {len(verts_mm):,} | {len(verts_mm):,} |
        | Faces | {len(faces):,} | {len(faces):,} |

        The Hilbert deformation is **{sm_r/sm_h:.0f}x smoother** because the
        Hilbert curve preserves locality: neighbouring vertices have nearby
        Hilbert indices, hence nearly identical displacements.
        """)


# ===================================================================
# About
# ===================================================================

def page_about():
    st.header("About PHVE")
    st.markdown(r"""
**PHVE** -- Parametric Hilbert Volumetric Encoding -- is a bijective encoding
map $\mathcal{F}_p^{(d),\alpha}$ from points of a $d$-dimensional bounded
volume ($d \in \{2, 3\}$) to fixed-length strings over a 29-symbol unambiguous
alphabet, built from the Hilbert space-filling curve.

| Property | Description |
|---|---|
| **Bijective** | Each cell-centre maps to a unique code. Verified exhaustively on the MNI152 brain mask: injective from $p = 8$ onwards, $14.1\%$ of voxels colliding at $p = 7$. |
| **Compact** | 5 characters at $p = 8$ cover a $\sim$1 mm cell in a $300 \times 250 \times 250$ mm cranium. |
| **Hierarchical** | Truncating a code yields the code of the enclosing dyadic cell, so a prefix denotes a region. |
| **Prefix implies proximity** | Two points sharing a long prefix are close in space, with an explicit constant. |
| **Indexable** | A dyadic-cell query is two binary searches on an ordered integer column: no spatial index needed. |
| **Unambiguous** | base-29 alphabet without confusable characters (no 0/O, 1/I/L). |

### One property PHVE does **not** have

Proximity does *not* imply a shared prefix. Two spatially adjacent points can
receive codes whose indices differ by $\Theta(n^d)$ -- almost the whole range.
This is not a defect of the implementation: it is a theorem, and it holds for
every space-filling curve. The maximum index gap between adjacent cells is
$\tfrac{5}{6}n^2$ in $d = 2$ and $\tfrac{13}{14}n^3$ in $d = 3$, verified
exhaustively up to grids of $16\,777\,216$ and $2\,097\,152$ points.

Only the *inverse* direction holds. Its constant is the worst-case $L_2$
dilation: $6$ in $d = 2$, and about $29.5$ in $d = 3$ for the Skilling variant
implemented here -- against $22.9$ for the best published three-dimensional
Hilbert curve, so this variant is roughly $24\%$ less tight.

### Demonstrations in this application

| Tab | What it shows | Script |
|---|---|---|
| Bijectivity | the order at which the code separates voxels | `bijectivity_mni152.py` |
| Prefix search | a region query as a range scan | `experiments/exp22_prefix_index.py` |

Four further demonstrations were removed on 2026-08-05 because the statements
they displayed were refuted by the experimental campaign in `experiments/`:
inter-patient stability (measures quantisation, not anatomy), DPCM compression
(the Hilbert traversal loses to a raster scan), FEM bandwidth (the bandwidth
theorem is false; the maximum index gap is $\Theta(n^d)$), and surface morphing.
The laboratory notebook `experiments/LOGBOOK.md` records each refutation with
the script that produced it.

### Reference anatomical volumes
""")

    vol_rows = []
    for k, v in VOLUMES.items():
        vol_rows.append({
            "Code": k,
            "Name": v["name"],
            "Dimensions (mm)": f"{v['dims'][0]} x {v['dims'][1]} x {v['dims'][2]}",
        })
    st.dataframe(vol_rows, width="stretch", hide_index=True)

    st.markdown("""
---
**Author**: Paul Guindo, Altius Academy SNC, Echallens (Vaud), Switzerland.

**Repository**: <https://github.com/Altius-Academy-SNC/PHVE>

**Live geolocation demos** (2D variant of $\\mathcal{F}_p^{(2),\\alpha}$):
[Yoro Maps](https://altius-academy-snc.github.io/yoro-maps/) ·
[Yoro](https://altius-academy-snc.github.io/yoro/)

**License**: MIT (code).
""")

    st.subheader("Bijectivity self-test")
    p_test = st.slider("Order $p$", 1, 4, 3, key="bj_p",
                        help="$p=4$ tests $4096$ points exhaustively.")
    if st.button("Run", key="bj_run"):
        with st.spinner(f"Verifying {(1 << p_test)**3} points..."):
            ok, msg = verify_bijectivity(p_test)
        if ok:
            st.success(f"PASS -- {msg}")
        else:
            st.error(f"FAIL -- {msg}")


# ===================================================================
# Navigation
# ===================================================================

# Four demonstrations were removed on 2026-08-05 because the statements they
# displayed were refuted by the experimental campaign of `experiments/`:
#
#   Inter-patient stability  the experiment uses no inter-patient data; it
#                            jitters fixed landmarks, and the measured rate
#                            agrees with pure quantisation (exp15).
#   DPCM compression         the Hilbert traversal *loses* to a raster scan on
#                            MNI152; the earlier gain came from a script that
#                            binarised the signal with round() (exp07).
#   FEM bandwidth            the bandwidth theorem is false: the maximum index
#                            gap is Theta(n^d), essentially unchanged from the
#                            natural ordering (exp03, seam theorem).
#   Surface morphing         the pointwise morphing bound is refuted.
#
# The page functions are kept in the file, unreferenced, so the history is not
# lost; they are not reachable from the navigation.
PAGES = {
    "Bijectivity (Prop. 1)": page_bijectivity,
    "Prefix search (Prop. 2)": page_prefix_search,
    "---": None,
    "About": page_about,
}

st.sidebar.title("PHVE")
st.sidebar.caption("Altius Academy SNC")

page_names = list(PAGES.keys())
page = st.sidebar.radio("Navigation", page_names,
                         format_func=lambda x: x if x.strip() and x != "---" else "─" * 20)

if PAGES.get(page):
    PAGES[page]()
