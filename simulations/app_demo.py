"""
Altius-Code 3D — Interface interactive (Streamlit)

Dashboard complet avec toutes les applications du brevet :
- Cerveau 3D : surface cerebrale coloree par code Altius (rotatif)
- Explorateur : coupes IRM + encodage/decodage
- Inter-patients (S2) : comparaison de codes entre patients
- Recherche par prefixe : selection de region anatomique
- Compression DPCM (S4) : benchmark Hilbert vs raster vs Morton
- Simulation cardiaque (S5) : propagation d'onde via Hilbert 3D
- Morphing surfaces (S6) : deformation lisse via Hilbert 1D

Usage : streamlit run app_demo.py

Auteur : Paul Guindo, Altius Academy SNC
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from collections import Counter
from scipy.stats import spearmanr

from codec3d import (
    VOLUMES, xyz2d, encode, decode, truncate,
    _code_length_3d, _int_to_base29, _canonical_precision_3d,
    verify_bijectivity,
)

# ===================================================================
# Configuration
# ===================================================================

st.set_page_config(
    page_title="Altius-Code 3D",
    page_icon="🧠",
    layout="wide",
)


# ===================================================================
# Chargement des donnees (cached)
# ===================================================================

@st.cache_data
def load_brain():
    """Charge MNI152 + masque cerebral."""
    from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
    img = load_mni152_template(resolution=2)
    mask_img = load_mni152_brain_mask(resolution=2)
    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = np.asarray(mask_img.dataobj).astype(bool)
    return data, img.affine, mask


@st.cache_data
def extract_brain_mesh(_data, _affine, _mask):
    """Extrait la surface du cerveau par marching cubes."""
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


@st.cache_data
def compute_vertex_hilbert(verts_mm, p=5, volume="CR"):
    """Calcule l'index Hilbert pour chaque vertex du mesh."""
    dims = VOLUMES[volume]["dims"]
    n = 1 << p
    max_d = 8 ** p - 1
    k = _code_length_3d(p)

    hilbert_vals = np.zeros(verts_mm.shape[0], dtype=np.float64)
    codes = []

    for idx in range(verts_mm.shape[0]):
        x, y, z = verts_mm[idx]
        vx = x + dims[0] / 2
        vy = y + dims[1] / 2
        vz = z + dims[2] / 2
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


def mm_to_voxel(x, y, z, affine):
    inv = np.linalg.inv(affine)
    ijk = inv @ np.array([x, y, z, 1.0])
    return int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))


def mm_to_altius(x_mm, y_mm, z_mm, p=6, volume="CR"):
    dims = VOLUMES[volume]["dims"]
    x = x_mm + dims[0] / 2
    y = y_mm + dims[1] / 2
    z = z_mm + dims[2] / 2
    return encode(x, y, z, p=p, volume=volume)


LANDMARKS = {
    "Centre cerveau": (0, 0, 0),
    "Cortex frontal": (0, 50, 30),
    "Cortex occipital": (0, -90, 0),
    "Hippocampe G": (-25, -20, -15),
    "Hippocampe D": (25, -20, -15),
    "Cervelet": (0, -60, -35),
    "Tronc cerebral": (0, -30, -30),
}


# ===================================================================
# Page : Cerveau 3D
# ===================================================================

def page_brain_3d():
    st.header("Cerveau 3D — Codage Altius-Code")

    data, affine, mask = load_brain()
    st.caption(f"MNI152 T1w | {data.shape} | 2 mm | {mask.sum():,} voxels cerebraux")

    col_ctrl, col_3d = st.columns([1, 3])

    with col_ctrl:
        p = st.slider("Precision (p)", 3, 7, 5,
                       help="Ordre de la courbe de Hilbert pour la coloration")
        color_mode = st.radio("Coloration", ["Index Hilbert", "IRM (intensite)", "Regions (prefixe)"])
        opacity = st.slider("Opacite", 0.3, 1.0, 0.8, 0.05)
        show_landmarks = st.checkbox("Afficher points anatomiques", value=True)

    with st.spinner("Extraction de la surface cerebrale..."):
        verts_mm, faces, normals, intensity = extract_brain_mesh(data, affine, mask)

    st.sidebar.metric("Vertices", f"{verts_mm.shape[0]:,}")
    st.sidebar.metric("Triangles", f"{faces.shape[0]:,}")

    if color_mode == "Index Hilbert":
        with st.spinner(f"Calcul des index Hilbert (p={p})..."):
            hilbert_vals, _ = compute_vertex_hilbert(verts_mm, p=p, volume="CR")
        vertex_colors = hilbert_vals
        colorscale = "HSV"
        colorbar_title = "Index Hilbert"
    elif color_mode == "IRM (intensite)":
        vertex_colors = intensity
        colorscale = "Gray"
        colorbar_title = "Intensite T1w"
    else:
        with st.spinner(f"Calcul des regions (p={p})..."):
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
        colorbar_title = "Region (prefixe 2 chars)"

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
                code = mm_to_altius(x, y, z, p=p, volume="CR")
                lm_x.append(x); lm_y.append(y); lm_z.append(z)
                lm_names.append(name); lm_codes.append(code)
            fig.add_trace(go.Scatter3d(
                x=lm_x, y=lm_y, z=lm_z,
                mode="markers+text",
                marker=dict(size=6, color="red", symbol="diamond"),
                text=lm_codes, textposition="top center",
                textfont=dict(size=9, color="white"),
                hovertext=[f"{n}<br>{c}" for n, c in zip(lm_names, lm_codes)],
                hoverinfo="text", name="Points anatomiques",
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.8, y=0.8, z=0.6)),
            ),
            height=700, margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Encodeur / Decodeur")
    col_enc, col_dec = st.columns(2)

    with col_enc:
        st.markdown("**Coordonnees MNI -> Code Altius**")
        c1, c2, c3 = st.columns(3)
        x_in = c1.number_input("X (mm)", value=0.0, step=5.0, key="enc_x")
        y_in = c2.number_input("Y (mm)", value=0.0, step=5.0, key="enc_y")
        z_in = c3.number_input("Z (mm)", value=0.0, step=5.0, key="enc_z")
        p_enc = st.slider("Precision", 3, 10, 8, key="enc_p")
        code = mm_to_altius(x_in, y_in, z_in, p=p_enc, volume="CR")
        decoded = decode(code)
        st.code(code, language=None)
        st.caption(f"Resolution : {decoded['resolution_mm']:.1f} mm")
        with st.expander("Hierarchie par troncature"):
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
        st.markdown("**Code Altius -> Coordonnees MNI**")
        code_input = st.text_input("Code", placeholder="CR:GSX-7X", key="dec_code")
        if code_input:
            try:
                result = decode(code_input)
                d = VOLUMES[result["volume"]]["dims"]
                x_mni = result["x_mm"] - d[0] / 2
                y_mni = result["y_mm"] - d[1] / 2
                z_mni = result["z_mm"] - d[2] / 2
                st.success(f"**({x_mni:.1f}, {y_mni:.1f}, {z_mni:.1f})** mm MNI")
                st.caption(f"Resolution : {result['resolution_mm']:.1f} mm | Volume : {result['volume']}")
            except ValueError as e:
                st.error(str(e))


# ===================================================================
# Page : Explorateur coupes IRM
# ===================================================================

def page_explorer():
    st.header("Explorateur IRM — Coupes orthogonales")
    data, affine, mask = load_brain()

    col1, col2, col3 = st.columns(3)
    with col1:
        si = st.slider("Sagittale (i)", 0, data.shape[0] - 1, data.shape[0] // 2)
    with col2:
        sj = st.slider("Coronale (j)", 0, data.shape[1] - 1, data.shape[1] // 2)
    with col3:
        sk = st.slider("Axiale (k)", 0, data.shape[2] - 1, data.shape[2] // 2)

    xyz = voxel_to_mm(si, sj, sk, affine)
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Sagittale", "Coronale", "Axiale"],
                        horizontal_spacing=0.05)
    for idx, s in enumerate([
        np.rot90(data[si, :, :]),
        np.rot90(data[:, sj, :]),
        np.rot90(data[:, :, sk]),
    ]):
        fig.add_trace(go.Heatmap(z=s, colorscale="Gray", showscale=False),
                       row=1, col=idx + 1)
    for col_idx, (hx, hy) in enumerate([
        (sj, data.shape[2] - sk - 1),
        (si, data.shape[2] - sk - 1),
        (si, data.shape[1] - sj - 1),
    ], 1):
        fig.add_hline(y=hy, line_dash="dot", line_color="cyan", line_width=1, row=1, col=col_idx)
        fig.add_vline(x=hx, line_dash="dot", line_color="cyan", line_width=1, row=1, col=col_idx)
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

    p = st.slider("Precision (p)", 3, 10, 8, key="expl_p")
    code = mm_to_altius(xyz[0], xyz[1], xyz[2], p=p, volume="CR")
    decoded = decode(code)
    st.markdown(f"""
    | | Valeur |
    |---|---|
    | **Coordonnees MNI** | ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm |
    | **Voxel** | ({si}, {sj}, {sk}) |
    | **Code Altius** | `{code}` |
    | **Resolution** | {decoded['resolution_mm']:.1f} mm |
    | **Recherche SQL** | `WHERE code LIKE '{code.split(":")[1][:3]}%'` |
    """)


# ===================================================================
# Page S2 : Comparaison inter-patients
# ===================================================================

def page_interpatient():
    st.header("S2 — Comparaison inter-patients")
    st.markdown("""
    **Objectif** : Montrer que le meme point anatomique recoit un code identique
    (ou tres proche) chez differents patients, une fois recale dans l'espace MNI.
    """)

    data, affine, mask = load_brain()
    p = st.slider("Precision (p)", 3, 10, 6, key="s2_p")

    rows = []
    matches = 0
    for name, (x, y, z) in LANDMARKS.items():
        code_a = mm_to_altius(x, y, z, p=p, volume="CR")
        rng_b = np.random.RandomState(hash(name) % 2**31)
        residual = rng_b.normal(0, 0.5, size=3)
        code_b = mm_to_altius(x + residual[0], y + residual[1], z + residual[2],
                              p=p, volume="CR")
        raw_a = code_a.split(":")[1].replace("-", "")
        raw_b = code_b.split(":")[1].replace("-", "")
        common = sum(1 for a, b in zip(raw_a, raw_b) if a == b)
        is_match = code_a == code_b
        if is_match:
            matches += 1
        rows.append({
            "Point": name,
            "MNI (mm)": f"({x:+d}, {y:+d}, {z:+d})",
            "Patient A": code_a,
            "Patient B": code_b,
            "Match": "Identique" if is_match else f"Prefixe: {common}/{len(raw_a)}",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)
    pct = matches / len(LANDMARKS) * 100
    st.metric("Codes identiques", f"{matches}/{len(LANDMARKS)} ({pct:.0f}%)")

    st.subheader("Robustesse vs. precision")
    p_values = list(range(3, 11))
    match_pcts = []
    for p_test in p_values:
        m = 0
        for name, (x, y, z) in LANDMARKS.items():
            code_a = mm_to_altius(x, y, z, p=p_test, volume="CR")
            rng_b = np.random.RandomState(hash(name) % 2**31)
            residual = rng_b.normal(0, 0.5, size=3)
            code_b = mm_to_altius(x + residual[0], y + residual[1], z + residual[2],
                                  p=p_test, volume="CR")
            if code_a == code_b:
                m += 1
        match_pcts.append(m / len(LANDMARKS) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_values, y=match_pcts, mode="lines+markers",
                              line=dict(color="#2ecc71", width=3), marker=dict(size=10)))
    fig.update_layout(xaxis_title="Ordre p", yaxis_title="Codes identiques (%)",
                       height=350, yaxis=dict(range=[0, 105]))
    st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Page : Recherche par prefixe
# ===================================================================

def page_prefix_search():
    st.header("Recherche par prefixe")
    st.markdown("Selectionnez un prefixe pour visualiser la region 3D correspondante.")

    data, affine, mask = load_brain()
    col1, col2 = st.columns([1, 3])

    with col1:
        p = st.slider("Precision (p)", 3, 6, 4, key="pf_p")
        example = mm_to_altius(0, 0, 0, p=p, volume="CR")
        example_raw = example.split(":")[1].replace("-", "")
        st.caption(f"Centre cerveau = `{example}` (raw: `{example_raw}`)")
        prefix = st.text_input("Prefixe", value=example_raw[:1],
                               help="Premiers caracteres du code (sans 'CR:')")

    with col2:
        verts_mm, faces, normals, intensity = extract_brain_mesh(data, affine, mask)
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
        n_match = match_mask.sum()
        n_total = verts_mm.shape[0]
        fig.update_layout(
            scene=dict(
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.8, y=0.8, z=0.6)),
            ),
            title=f"Prefixe 'CR:{prefix}...' — {n_match:,}/{n_total:,} vertices ({n_match/max(n_total,1)*100:.1f}%)",
            height=600, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.code(f"SELECT * FROM voxels WHERE altius_code LIKE 'CR:{prefix}%';", language="sql")


# ===================================================================
# Page S4 : Compression DPCM
# ===================================================================

def _shannon_entropy(signal):
    """Entropie de Shannon (bits)."""
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
    """Calcule le benchmark DPCM pour les 3 ordres de parcours."""
    dims = VOLUMES["CR"]["dims"]
    n = 1 << p
    brain_coords = np.argwhere(_mask)
    total = len(brain_coords)

    # Raster
    raster_signal = _data[_mask].astype(np.int32)

    # Hilbert & Morton
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

    # DPCM
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
    st.header("S4 — Compression DPCM : Hilbert vs Raster vs Morton")
    st.markdown("""
    **Revendication 9** : Le parcours de Hilbert 3D reduit l'entropie DPCM
    de **5 a 15%** par rapport au raster sur des volumes cliniques.

    La compression DPCM (prediction differentielle) encode la difference entre
    valeurs consecutives. Un parcours preservant la localite spatiale produit
    des differences plus petites = entropie plus basse = meilleure compression.
    """)

    data, affine, mask = load_brain()
    p = st.slider("Ordre Hilbert (p)", 4, 7, 6, key="s4_p",
                   help="p=6: 64^3 grid (~5mm), p=7: 128^3 (~2.5mm)")

    with st.spinner(f"Calcul DPCM (p={p}, {mask.sum():,} voxels)... peut prendre ~1 min"):
        r = compute_dpcm_benchmark(data, mask, affine, p)

    # Metriques
    col1, col2, col3 = st.columns(3)
    col1.metric("Entropie Raster", f"{r['h_raster']:.3f} bits")
    col2.metric("Entropie Morton", f"{r['h_morton']:.3f} bits",
                delta=f"{r['red_morton_vs_raster']:+.2f}%")
    col3.metric("Entropie Hilbert", f"{r['h_hilbert']:.3f} bits",
                delta=f"{r['red_hilbert_vs_raster']:+.2f}%")

    # Graphiques
    col_a, col_b = st.columns(2)

    with col_a:
        # Barplot
        fig = go.Figure()
        methods = ["Raster", "Morton", "Hilbert"]
        vals = [r['h_raster'], r['h_morton'], r['h_hilbert']]
        colors = ["#3498db", "#e67e22", "#2ecc71"]
        fig.add_trace(go.Bar(x=methods, y=vals, marker_color=colors,
                             text=[f"{v:.3f}" for v in vals], textposition="outside"))
        fig.update_layout(
            title="Entropie DPCM (bits) — plus bas = meilleur",
            yaxis_title="Entropie de Shannon (bits)",
            height=400, yaxis=dict(range=[0, max(vals) * 1.15]),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Histogramme
        fig = go.Figure()
        bins_spec = dict(start=-200, end=200, size=5)
        fig.add_trace(go.Histogram(x=r['dpcm_raster'], name="Raster",
                                    opacity=0.5, xbins=bins_spec, histnorm="probability density"))
        fig.add_trace(go.Histogram(x=r['dpcm_hilbert'], name="Hilbert",
                                    opacity=0.5, xbins=bins_spec, histnorm="probability density"))
        fig.update_layout(
            title="Distribution des residus DPCM",
            xaxis_title="Residu (diff)", yaxis_title="Densite",
            barmode="overlay", height=400,
            xaxis=dict(range=[-200, 200]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tableau
    st.subheader("Resume")
    st.dataframe([
        {"Metrique": "Voxels cerebraux", "Raster": f"{r['n_voxels']:,}",
         "Morton": f"{r['n_voxels']:,}", "Hilbert": f"{r['n_voxels']:,}"},
        {"Metrique": "H(original)", "Raster": f"{r['h_orig']:.3f}",
         "Morton": f"{r['h_orig']:.3f}", "Hilbert": f"{r['h_orig']:.3f}"},
        {"Metrique": "H(DPCM)", "Raster": f"{r['h_raster']:.3f}",
         "Morton": f"{r['h_morton']:.3f}", "Hilbert": f"{r['h_hilbert']:.3f}"},
        {"Metrique": "Reduction vs raster", "Raster": "—",
         "Morton": f"{r['red_morton_vs_raster']:+.2f}%",
         "Hilbert": f"{r['red_hilbert_vs_raster']:+.2f}%"},
        {"Metrique": "|residu| moyen", "Raster": f"{np.abs(r['dpcm_raster']).mean():.1f}",
         "Morton": f"{np.abs(r['dpcm_morton']).mean():.1f}",
         "Hilbert": f"{np.abs(r['dpcm_hilbert']).mean():.1f}"},
        {"Metrique": "% residus = 0",
         "Raster": f"{np.sum(r['dpcm_raster']==0)/len(r['dpcm_raster'])*100:.1f}%",
         "Morton": f"{np.sum(r['dpcm_morton']==0)/len(r['dpcm_morton'])*100:.1f}%",
         "Hilbert": f"{np.sum(r['dpcm_hilbert']==0)/len(r['dpcm_hilbert'])*100:.1f}%"},
    ], use_container_width=True, hide_index=True)


# ===================================================================
# Page S5 : Simulation cardiaque
# ===================================================================

CA_DIMS = VOLUMES["CA"]["dims"]
A_OUT, B_OUT, C_OUT = 55, 40, 35
A_IN, B_IN, C_IN = 35, 25, 20
CX, CY, CZ = CA_DIMS[0] / 2, CA_DIMS[1] / 2, CA_DIMS[2] / 2


@st.cache_data
def create_heart_model(spacing_mm=3.0, sa_offset=(30, 25, 20), speed=0.8, p=6):
    """Cree le modele cardiaque et calcule tous les indices."""
    nx = int(CA_DIMS[0] / spacing_mm)
    ny = int(CA_DIMS[1] / spacing_mm)
    nz = int(CA_DIMS[2] / spacing_mm)
    n = 1 << p

    coords, hilbert_idx, raster_idx, morton_idx = [], [], [], []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = (i + 0.5) * spacing_mm
                y = (j + 0.5) * spacing_mm
                z = (k + 0.5) * spacing_mm
                dx = (x - CX) / A_OUT
                dy = (y - CY) / B_OUT
                dz = (z - CZ) / C_OUT
                if dx**2 + dy**2 + dz**2 > 1.0:
                    continue
                dxi = (x - CX) / A_IN
                dyi = (y - CY) / B_IN
                dzi = (z - CZ) / C_IN
                if dxi**2 + dyi**2 + dzi**2 < 1.0:
                    continue
                coords.append([x, y, z])
                ix = max(0, min(n-1, int(x / CA_DIMS[0] * n)))
                iy = max(0, min(n-1, int(y / CA_DIMS[1] * n)))
                iz = max(0, min(n-1, int(z / CA_DIMS[2] * n)))
                hilbert_idx.append(xyz2d(p, ix, iy, iz))
                raster_idx.append(ix * n * n + iy * n + iz)
                m = 0
                for b in range(16):
                    m |= ((ix >> b) & 1) << (3*b+2)
                    m |= ((iy >> b) & 1) << (3*b+1)
                    m |= ((iz >> b) & 1) << (3*b)
                morton_idx.append(m)

    coords = np.array(coords)
    hilbert_idx = np.array(hilbert_idx, dtype=np.int64)
    raster_idx = np.array(raster_idx, dtype=np.int64)
    morton_idx = np.array(morton_idx, dtype=np.int64)

    sa = np.array([CX + sa_offset[0], CY + sa_offset[1], CZ + sa_offset[2]])
    arrival = np.linalg.norm(coords - sa, axis=1) / speed

    return coords, arrival, hilbert_idx, raster_idx, morton_idx, sa


def _traversal_stats(arrival, indices, coords, threshold=5.0):
    order = np.argsort(indices)
    sorted_t = arrival[order]
    jumps = np.abs(np.diff(sorted_t))
    rho, _ = spearmanr(np.arange(len(arrival)), sorted_t)
    sorted_c = coords[order]
    dists = np.linalg.norm(np.diff(sorted_c, axis=0), axis=1)
    cache = np.mean(dists < threshold)
    return {"rho": rho, "jump_mean": jumps.mean(), "cache": cache * 100,
            "dist_mean": dists.mean(), "jumps": jumps}


def page_cardiac():
    st.header("S5 — Simulation cardiaque : propagation d'onde via Hilbert 3D")
    st.markdown("""
    **Revendication 8** : Le parcours de Hilbert 3D suit naturellement la propagation
    physique de l'onde de depolarisation dans le myocarde (de proche en proche).

    Modele : myocarde ellipsoidal dans le volume **CA** (150x120x100 mm),
    onde depuis le noeud sino-atrial (SA).
    """)

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        spacing = st.slider("Espacement (mm)", 2.0, 5.0, 3.0, 0.5, key="s5_sp")
        speed = st.slider("Vitesse (mm/ms)", 0.3, 2.0, 0.8, 0.1, key="s5_v")
        sa_x = st.slider("SA offset X", 0, 50, 30, key="s5_sx")
        sa_y = st.slider("SA offset Y", 0, 40, 25, key="s5_sy")
        sa_z = st.slider("SA offset Z", 0, 30, 20, key="s5_sz")

    with st.spinner("Calcul du modele cardiaque..."):
        coords, arrival, h_idx, r_idx, m_idx, sa = create_heart_model(
            spacing, (sa_x, sa_y, sa_z), speed)

    st.sidebar.metric("Voxels myocardiques", f"{len(coords):,}")

    stats_h = _traversal_stats(arrival, h_idx, coords)
    stats_m = _traversal_stats(arrival, m_idx, coords)
    stats_r = _traversal_stats(arrival, r_idx, coords)

    c1, c2, c3 = st.columns(3)
    c1.metric("Coherence Raster", f"{stats_r['cache']:.1f}%")
    c2.metric("Coherence Morton", f"{stats_m['cache']:.1f}%")
    c3.metric("Coherence Hilbert", f"{stats_h['cache']:.1f}%")

    with col_main:
        tab1, tab2, tab3 = st.tabs(["Coeur 3D", "Correlation", "Statistiques"])

        with tab1:
            color_by = st.radio("Colorer par", ["Temps d'arrivee", "Index Hilbert"], key="s5_c",
                                 horizontal=True)
            if color_by == "Temps d'arrivee":
                cvals = arrival
                cscale = "Hot"
                ctitle = "Temps (ms)"
            else:
                cvals = h_idx.astype(float) / h_idx.max()
                cscale = "HSV"
                ctitle = "Index Hilbert"

            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode="markers",
                marker=dict(size=2, color=cvals, colorscale=cscale,
                            colorbar=dict(title=ctitle, len=0.6), opacity=0.7),
                hovertemplate="(%{x:.0f}, %{y:.0f}, %{z:.0f}) mm<extra></extra>",
            ))
            fig.add_trace(go.Scatter3d(
                x=[sa[0]], y=[sa[1]], z=[sa[2]],
                mode="markers+text", text=["SA"],
                marker=dict(size=8, color="cyan", symbol="diamond"),
                name="Noeud SA",
            ))
            fig.update_layout(
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                           aspectmode="data"),
                height=550, margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Hilbert vs onde", "Raster vs onde"])
            h_rank = np.argsort(np.argsort(h_idx))
            r_rank = np.argsort(np.argsort(r_idx))
            step = max(1, len(arrival) // 3000)
            fig.add_trace(go.Scattergl(
                x=h_rank[::step], y=arrival[::step], mode="markers",
                marker=dict(size=2, color="#2ecc71", opacity=0.4),
                name=f"Hilbert (rho={stats_h['rho']:.3f})"), row=1, col=1)
            fig.add_trace(go.Scattergl(
                x=r_rank[::step], y=arrival[::step], mode="markers",
                marker=dict(size=2, color="#3498db", opacity=0.4),
                name=f"Raster (rho={stats_r['rho']:.3f})"), row=1, col=2)
            fig.update_xaxes(title_text="Rang de parcours")
            fig.update_yaxes(title_text="Temps d'arrivee (ms)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.dataframe([
                {"Metrique": "Correlation (Spearman rho)",
                 "Raster": f"{stats_r['rho']:.3f}", "Morton": f"{stats_m['rho']:.3f}",
                 "Hilbert": f"{stats_h['rho']:.3f}"},
                {"Metrique": "Saut moyen (ms)",
                 "Raster": f"{stats_r['jump_mean']:.2f}", "Morton": f"{stats_m['jump_mean']:.2f}",
                 "Hilbert": f"{stats_h['jump_mean']:.2f}"},
                {"Metrique": "Coherence cache (<5mm)",
                 "Raster": f"{stats_r['cache']:.1f}%", "Morton": f"{stats_m['cache']:.1f}%",
                 "Hilbert": f"{stats_h['cache']:.1f}%"},
                {"Metrique": "Distance moy. consecutifs",
                 "Raster": f"{stats_r['dist_mean']:.1f} mm", "Morton": f"{stats_m['dist_mean']:.1f} mm",
                 "Hilbert": f"{stats_h['dist_mean']:.1f} mm"},
            ], use_container_width=True, hide_index=True)

            # Histogramme des sauts
            fig = go.Figure()
            bins_spec = dict(start=0, end=30, size=0.5)
            fig.add_trace(go.Histogram(x=stats_h['jumps'], name="Hilbert",
                                        opacity=0.6, xbins=bins_spec))
            fig.add_trace(go.Histogram(x=stats_r['jumps'], name="Raster",
                                        opacity=0.3, xbins=bins_spec))
            fig.update_layout(title="Sauts temporels entre voxels consecutifs",
                               xaxis_title="Saut (ms)", barmode="overlay", height=350)
            st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Page S6 : Morphing de surfaces
# ===================================================================

@st.cache_data
def compute_surface_hilbert(_verts_mm, p=5):
    """Calcule l'index Hilbert normalise pour chaque vertex."""
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
    """Diff. moyenne de deplacement entre voisins sur le mesh."""
    edges = set()
    for f in faces:
        for i in range(3):
            edges.add(tuple(sorted([f[i], f[(i+1) % 3]])))
    diffs = np.array([abs(displacements[v1] - displacements[v2]) for v1, v2 in edges])
    return diffs.mean(), diffs


def page_morphing():
    st.header("S6 — Animation et morphing de surfaces via Hilbert 3D")
    st.markdown("""
    **Revendication 8** : La surface d'un organe, parcourue par la courbe de Hilbert 3D,
    permet de definir des animations comme des transformations continues dans l'espace 1D.

    Une deformation sinusoidale dans l'espace Hilbert 1D produit une onde **spatialement
    lisse** en 3D. La meme deformation avec un ordre aleatoire produit du **bruit spatial**.
    """)

    data, affine, mask = load_brain()

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        amplitude = st.slider("Amplitude (mm)", 0.5, 8.0, 3.0, 0.5, key="s6_a")
        frequency = st.slider("Frequence", 1.0, 10.0, 4.0, 0.5, key="s6_f")
        phase = st.slider("Phase", 0.0, 6.28, 0.0, 0.1, key="s6_ph")
        p_morph = st.slider("Precision (p)", 3, 6, 5, key="s6_p")
        show_mode = st.radio("Afficher", ["Hilbert (lisse)", "Aleatoire (bruit)", "Cote a cote"],
                              key="s6_mode")

    with st.spinner("Extraction de la surface..."):
        verts_mm, faces, normals, intensity = extract_brain_mesh(data, affine, mask)

    with st.spinner(f"Calcul des index Hilbert (p={p_morph})..."):
        h_norm = compute_surface_hilbert(verts_mm, p=p_morph)

    # Normales unitaires
    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_len[norms_len == 0] = 1
    unit_normals = normals / norms_len

    # Deformation Hilbert
    disp_h = amplitude * np.sin(2 * np.pi * frequency * h_norm + phase)
    deformed_h = verts_mm + unit_normals * disp_h[:, np.newaxis]

    # Deformation aleatoire
    rng = np.random.RandomState(42)
    rand_norm = rng.permutation(len(verts_mm)).astype(np.float64) / len(verts_mm)
    disp_r = amplitude * np.sin(2 * np.pi * frequency * rand_norm + phase)
    deformed_r = verts_mm + unit_normals * disp_r[:, np.newaxis]

    # Smoothness
    sm_h, _ = _edge_smoothness(disp_h, faces)
    sm_r, _ = _edge_smoothness(disp_r, faces)

    st.sidebar.metric("Fluidite Hilbert", f"{sm_h:.4f} mm")
    st.sidebar.metric("Fluidite Aleatoire", f"{sm_r:.4f} mm")
    st.sidebar.metric("Ratio", f"{sm_r/sm_h:.1f}x")

    with col_main:
        def _make_mesh(verts, disp, title, colorscale="RdBu"):
            return go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=disp, colorscale=colorscale,
                colorbar=dict(title="Depl. (mm)", len=0.6),
                cmin=-amplitude, cmax=amplitude,
                opacity=0.85,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
                lightposition=dict(x=100, y=200, z=300),
                name=title,
            )

        scene = dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                     aspectmode="data",
                     camera=dict(eye=dict(x=1.8, y=0.8, z=0.6)))

        if show_mode == "Cote a cote":
            fig = make_subplots(rows=1, cols=2,
                                specs=[[{"type": "scene"}, {"type": "scene"}]],
                                subplot_titles=[
                                    f"Hilbert (fluidite: {sm_h:.4f})",
                                    f"Aleatoire (fluidite: {sm_r:.4f})"])
            fig.add_trace(_make_mesh(deformed_h, disp_h, "Hilbert"), row=1, col=1)
            fig.add_trace(_make_mesh(deformed_r, disp_r, "Aleatoire"), row=1, col=2)
            fig.update_layout(height=600, scene=scene, scene2=scene,
                              margin=dict(l=0, r=0, t=40, b=0))
        else:
            if show_mode == "Hilbert (lisse)":
                fig = go.Figure(data=[_make_mesh(deformed_h, disp_h, "Hilbert")])
                title = f"Deformation Hilbert — fluidite: {sm_h:.4f} mm"
            else:
                fig = go.Figure(data=[_make_mesh(deformed_r, disp_r, "Aleatoire")])
                title = f"Deformation Aleatoire — fluidite: {sm_r:.4f} mm"
            fig.update_layout(height=600, scene=scene, title=title,
                              margin=dict(l=0, r=0, t=40, b=0))

        st.plotly_chart(fig, use_container_width=True)

    # Resume
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Hilbert", "Aleatoire"], y=[sm_h, sm_r],
                             marker_color=["#2ecc71", "#e74c3c"],
                             text=[f"{sm_h:.4f}", f"{sm_r:.4f}"],
                             textposition="outside"))
        fig.update_layout(title="Diff. moyenne entre voisins (mm) — plus bas = plus lisse",
                           yaxis_title="mm", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(f"""
        ### Resultats

        | | Hilbert | Aleatoire |
        |---|---|---|
        | Diff. voisins | **{sm_h:.4f}** mm | {sm_r:.4f} mm |
        | Ratio | **{sm_r/sm_h:.1f}x plus lisse** | — |
        | Vertices | {len(verts_mm):,} | {len(verts_mm):,} |
        | Faces | {len(faces):,} | {len(faces):,} |

        La deformation Hilbert est **{sm_r/sm_h:.0f}x plus lisse** car la courbe
        de Hilbert preserve la localite : des points voisins en 3D ont des
        indices Hilbert proches, donc des deplacements similaires.
        """)


# ===================================================================
# Page : A propos
# ===================================================================

def page_info():
    st.header("A propos d'Altius-Code 3D")
    st.markdown("""
    ### Qu'est-ce qu'Altius-Code ?

    Altius-Code est un systeme d'encodage spatial bijectif qui transforme des coordonnees
    3D (x, y, z) en codes alphanumeriques compacts via la **courbe de Hilbert 3D**.

    | Propriete | Description |
    |-----------|-------------|
    | **Bijectif** | Chaque point 3D correspond a exactement un code, et inversement |
    | **Compact** | 5-8 caracteres pour identifier un voxel a 1mm de resolution |
    | **Hierarchique** | Tronquer un caractere = zoomer en arriere (~29x le volume) |
    | **Locality-preserving** | Points proches = codes proches (prefixe commun) |
    | **Indexable** | Compatible B-tree, recherche par prefixe en O(log n) |
    | **Non-ambigu** | Alphabet base-29 sans caracteres confusables (0/O, 1/I/L) |

    ### Applications du brevet (v3)

    | # | Application | Revendication | Prototype |
    |---|-------------|---------------|-----------|
    | 1 | Indexation IRM/Scanner | Rev. 6 | S1 |
    | 2 | Comparaison inter-patients | Rev. 7 | S2 |
    | 3 | Compression DPCM | Rev. 9 | **S4** |
    | 4 | Simulation cardiaque | Rev. 8 | **S5** |
    | 5 | Renumerotage MEF | Rev. 10 | Proto 7 |
    | 6 | Animation surfaces | Rev. 8 | **S6** |

    ### Volumes anatomiques
    """)

    vol_rows = []
    for k, v in VOLUMES.items():
        vol_rows.append({
            "Code": k,
            "Nom": v["name"],
            "Dimensions (mm)": f"{v['dims'][0]} x {v['dims'][1]} x {v['dims'][2]}",
        })
    st.dataframe(vol_rows, use_container_width=True, hide_index=True)

    st.markdown("""
    ---
    **Auteur** : Paul Guindo, Altius Academy SNC

    **Licence** : Open-source (code) / Service commercial (API, SaaS)

    **Modele d'utilite** : IPI (Institut Federal de la Propriete Intellectuelle), Suisse
    """)

    st.subheader("Test de bijectivite")
    p_test = st.slider("Ordre p", 1, 4, 3, key="bj_p",
                        help="p=4 teste 4096 points")
    if st.button("Lancer le test"):
        with st.spinner(f"Verification de {(1 << p_test)**3} points..."):
            ok, msg = verify_bijectivity(p_test)
        if ok:
            st.success(f"PASS — {msg}")
        else:
            st.error(f"FAIL — {msg}")


# ===================================================================
# Main — Navigation
# ===================================================================

PAGES = {
    "Cerveau 3D": page_brain_3d,
    "Coupes IRM": page_explorer,
    "Inter-patients (S2)": page_interpatient,
    "Recherche par prefixe": page_prefix_search,
    "---": None,
    "Compression DPCM (S4)": page_dpcm,
    "Simulation cardiaque (S5)": page_cardiac,
    "Morphing surfaces (S6)": page_morphing,
    " ": None,
    "A propos": page_info,
}

st.sidebar.title("Altius-Code 3D")
st.sidebar.caption("Altius Academy SNC")

page_names = list(PAGES.keys())
page = st.sidebar.radio("Navigation", page_names,
                         format_func=lambda x: x if x.strip() and x != "---" else "─" * 20)

if PAGES.get(page):
    PAGES[page]()
