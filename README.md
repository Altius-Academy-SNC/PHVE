# PHVE — Parametric Hilbert Volumetric Encoding

Reference implementation and reproducibility kit for

> **PHVE: a bijective, prefix-searchable spatial code for anatomical volumes**
> Paul Guindo, Altius Academy SNC, 2026. — `paper/phve_software.tex`

PHVE is a bijective encoding map $\mathcal{F}_p^{(d),\alpha}$ from points of a
$d$-dimensional bounded volume ($d \in \{2, 3\}$) to fixed-length strings over
a 29-symbol unambiguous alphabet, built from the Hilbert space-filling curve.
None of the ingredients is new; what this repository provides is a verified,
reproducible instantiation for anatomy.

What is established, each by a seeded script in `experiments/`:

- **bijectivity** — injective on the MNI152 brain mask from $p = 8$ onwards
  ($14.1\%$ of voxels collide at $p = 7$);
- **a prefix hierarchy** — truncating a code gives the enclosing dyadic cell,
  so a region query is a range scan on an ordinary ordered column, at a cost
  independent of the number of results;
- **prefix implies proximity**, with an explicit constant (the worst-case
  $L_2$ dilation: $6$ in $d = 2$, $\approx 29.5$ in $d = 3$ for the Skilling
  variant used here, against $22.9$ for the best published 3D Hilbert curve).

### What was refuted

An earlier version of this work (`arXiv v4`/`v5`, not included here) claimed
more. A systematic campaign — `experiments/`, 22 seeded scripts — refuted the
following, and each refutation is recorded in `experiments/LOGBOOK.md` with
the script that produced it:

| claim | outcome |
|---|---|
| bandwidth reduction for FEM matrices | **false** — the maximum index gap is $\Theta(n^d)$, essentially the natural ordering; the seam theorem proves it |
| forward locality (proximity $\Rightarrow$ shared prefix) | **false for every space-filling curve** |
| DPCM compression gain | **false** — the Hilbert traversal loses to a raster scan on MNI152 |
| inter-patient stability | **misattributed** — the experiment measures quantisation, not anatomy |
| pointwise surface-morphing bound | **false** |
| equivalence with the compact Hilbert index | **false in $d = 3$** |

`experiments/UNVERIFIED.md` lists what remains untested.

Live geolocation demonstrators (2D variant of $\mathcal{F}_p^{(2),\alpha}$):

- **Yoro Maps** — interactive encode/decode on the WGS84 globe:
  <https://altius-academy-snc.github.io/yoro-maps/>
- **Yoro** — lightweight address-lookup:
  <https://altius-academy-snc.github.io/yoro/>

## Repository layout

```
PHVE/
├── paper/
│   └── phve_software.tex    # the software paper
├── experiments/             # the verification campaign — read LOGBOOK.md first
│   ├── LOGBOOK.md           # chronological record, including the failures
│   ├── UNVERIFIED.md        # what this work does NOT settle
│   ├── PRIOR_ART.md         # what the literature already establishes
│   ├── phve/                # kernels: Hilbert, mesh, FEM, IC(0), cache sim
│   ├── exp01..exp22         # one script per claim; each writes a JSON record
│   └── results/             # those JSON records
├── simulations/             # the codec and the demonstrator
│   ├── codec.py             # 2D Skilling kernel + base-29 codec (geolocation)
│   ├── codec3d.py           # 3D Skilling kernel (anatomical encoding)
│   ├── constants.py         # geographic Reference Family (DOMAINS dict)
│   ├── bijectivity_mni152.py     # 3D bijectivity on MNI152
│   ├── app_demo.py          # Streamlit demonstrator (2 tabs)
│   └── …                    # scripts of the refuted claims, kept for the record
├── deploy/                  # Docker + nginx + certbot for the demonstrator
├── requirements.txt
├── LICENSE                  # MIT
└── README.md
```

The scripts `interpatient_stability.py`, `dpcm_compression.py`,
`fem_bandwidth.py` and `surface_morphing.py` are kept because their outputs are
cited in the refutations, not because their conclusions hold. Two of them
contain bugs that the campaign found; see `experiments/LOGBOOK.md`.

## Reproducing the paper experiments

```bash
git clone https://github.com/Altius-Academy-SNC/PHVE.git
cd PHVE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd simulations
```

Then run any single experiment. The two that back the paper:

| Experiment | Script |
|---|---|
| Collision test on the MNI152 brain mask, $p = 4\ldots10$ | `python experiments/exp15_v4_carryover.py` |
| Dyadic-cell queries vs. a $k$-d tree | `python experiments/exp22_prefix_index.py` |

`experiments/README.md` lists all 22 scripts with their wall times. Build the
C extensions first:

```bash
cc -O2 -shared -fPIC -o experiments/phve/cachesim.so experiments/phve/cachesim.c
cc -O2 -shared -fPIC -o experiments/phve/libic0.so   experiments/phve/ic0.c -lm
```

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
