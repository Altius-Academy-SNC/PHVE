# PHVE — Parametric Hilbert Volumetric Encoding

Reference implementation and reproducibility kit for the paper

> **Parametric Hilbert Volumetric Encoding for Geometric Data with Locality, Hierarchy, and Variance Bounds**
> Paul Guindo, Altius Academy SNC, 2026.

PHVE is a bijective encoding map $\mathcal{F}_p^{(d),\alpha}$ from
points of a $d$-dimensional bounded volume ($d \in \{2, 3\}$) to
fixed-length strings over a 29-symbol unambiguous alphabet, built from
the Hilbert space-filling curve. The construction yields:

- exact bijectivity at every order $p$,
- locality preservation with explicit $C_d$ constants,
- a clean prefix hierarchy on the codes,
- a Lipschitz second-moment bound for DPCM compression along the
  Hilbert traversal.

Live geolocation demonstrators (2D variant of $\mathcal{F}_p^{(2),\alpha}$):

- **Yoro Maps** — interactive encode/decode on the WGS84 globe:
  <https://altius-academy-snc.github.io/yoro-maps/>
- **Yoro** — lightweight address-lookup:
  <https://altius-academy-snc.github.io/yoro/>

## Repository layout

```
PHVE/
├── paper/                   # arXiv source
│   ├── altius_code_arxiv_v4.tex
│   └── figures/             # all TikZ figures included by the paper
├── simulations/             # reproducibility scripts
│   ├── codec.py             # 2D Skilling kernel + base-29 codec (geolocation)
│   ├── codec3d.py           # 3D Skilling kernel (anatomical encoding)
│   ├── constants.py         # geographic Reference Family (DOMAINS dict)
│   ├── proto_s1_brain.py    # Theorem 4.3 (3D bijectivity) on MNI152
│   ├── proto_s2_interpatient.py  # Corollary 5.3 (prefix => proximity)
│   ├── proto_s4_dpcm.py     # Theorem 8.2 (DPCM second-moment bound)
│   ├── proto7_improved.py   # Theorem 10.2 (FEM bandwidth reduction)
│   ├── proto_s6_surface.py  # Proposition 10.4 (surface morphing)
│   └── app_demo.py          # Streamlit demonstrator (8 tabs)
├── requirements.txt
├── LICENSE                  # MIT
└── README.md
```

## Reproducing the paper experiments

```bash
git clone https://github.com/Altius-Academy-SNC/PHVE.git
cd PHVE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd simulations
```

Then run any single experiment:

| Experiment | Script | Theorem |
|---|---|---|
| 3D bijectivity, MNI152, $p=8$ (235 375 voxels, 0 collisions) | `python proto_s1_brain.py` | 4.3 |
| Inter-patient prefix stability, $p=6$ | `python proto_s2_interpatient.py` | 5.3 |
| DPCM raster-vs-Hilbert on MNI152, $p=7$ | `python proto_s4_dpcm.py` | 8.2 |
| FEM stiffness-matrix bandwidth, 4 251-node Delaunay mesh | `python proto7_improved.py` | 10.2 |
| Surface morphing on a 31 328-vertex brain mesh | `python proto_s6_surface.py` | 10.4 |

Or launch the interactive demonstrator:

```bash
streamlit run app_demo.py
```

The MNI152 templates are downloaded automatically by `nilearn` on
first use (~50 MB); they are cached under `~/nilearn_data/`.

## Building the paper

Requires a TeX Live distribution with `pdflatex`, `tikz`, `algorithmic`,
`amsmath`, `amssymb`, `cleveref`, `booktabs`, and `microtype`.

```bash
cd paper
pdflatex altius_code_arxiv_v4.tex
pdflatex altius_code_arxiv_v4.tex   # second pass for cross-references
```

## Citation

```bibtex
@misc{guindo2026phve,
  author = {Paul Guindo},
  title  = {Parametric Hilbert Volumetric Encoding for Geometric Data
            with Locality, Hierarchy, and Variance Bounds},
  year   = {2026},
  note   = {Altius Academy SNC},
  url    = {https://github.com/Altius-Academy-SNC/PHVE}
}
```

## Status

This repository is **work in progress**: the paper is being revised
toward arXiv submission, and the simulation scripts are under active
audit. Issues and pull requests are welcome.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Paul Guindo, Altius Academy SNC, Échallens (Vaud), Switzerland.
<paulguindo@altius-group.ch>
