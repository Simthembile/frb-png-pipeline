# FRB DM × Timing: Primordial Non-Gaussianity Fisher Pipeline

> **Constraining Primordial Non-Gaussianity with Fast Radio Burst Dispersion Measure and Timing Cross-Spectra**

This pipeline forecasts constraints on the local primordial non-Gaussianity parameter **fNL** using the joint angular power spectra of Fast Radio Burst (FRB) Dispersion Measure (DM) and timing delay fields. The key result is that the DM × timing *cross-spectrum* self-calibrates the electron bias parameter `b_e`, dramatically improving fNL constraints.

---

## Physics Overview

FRBs probe the large-scale structure of the Universe through two observables:

| Observable | Symbol | Tracer of |
|---|---|---|
| Dispersion Measure | DM | Free electron density along l.o.s. |
| Timing delay | t | Gravitational potential (Shapiro delay) |

The cross-spectrum C^{Dt}(ℓ) breaks the degeneracy between fNL and the electron bias b_e, enabling self-calibrated constraints without external bias priors.

### Four Analysis Cases

| Case | Data Vector | Parameters | Notes |
|---|---|---|---|
| A | C^DD only | fNL (b₀e, zfb fixed) | Reischke et al. 2020 benchmark |
| B | C^DD only | {fNL, b₀e, zfb} marginalised | Bias-marginalised DM only |
| C | {C^DD, C^tt, C^Dt} | {fNL, b₀e, zfb} marginalised | **New result: cross-spectrum self-calibration** |
| D | C^tt only | fNL (bias-free) | Reference: timing alone cannot constrain fNL |

---

## Repository Structure

```
frb-png-pipeline/
├── cosmology.py      # CAMB P_mm, chi(z) interpolators, bias & astrophysical functions
├── spectra.py        # Limber angular power spectra C^DD, C^tt, C^Dt
├── fisher.py         # Fisher matrix forecasts (single-bin, sweeps)
├── tomo.py           # Tomographic Fisher (n_tomo redshift bins)
├── pipeline.py       # Master pipeline: runs all 4 stages end-to-end
├── run_all.py        # Computation script → results.pkl
├── plot_all.py       # Generates all 5 publication figures
└── results.pkl       # Pre-computed results (skip recomputation)
```

---

## Installation

### Requirements

- Python ≥ 3.9
- [CAMB](https://camb.readthedocs.io/) (for matter power spectrum)
- NumPy, SciPy, Matplotlib

```bash
pip install camb numpy scipy matplotlib
```

### Clone

```bash
git clone https://github.com/Simthembile/frb-png-pipeline.git
cd frb-png-pipeline
```

---

## Usage

### Option 1 — Full pipeline (all stages)

```bash
python pipeline.py
```

Runs cosmology setup → Fisher forecasts → tomographic Fisher → figure generation. Estimated runtime: **5–10 minutes** on a modern laptop.

```bash
python pipeline.py --skip-plots   # skip figure generation
python pipeline.py --outdir ./out # custom output directory
```

### Option 2 — Separate computation + plotting

```bash
# Step 1: run all Fisher analyses (saves results.pkl)
python run_all.py

# Step 2: generate all 5 figures from results.pkl
python plot_all.py
```

---

## Outputs

| File | Description |
|---|---|
| `results.pkl` | All Fisher results (single-bin + tomographic) |
| `fig1_spectra.pdf` | C^DD, C^tt, C^Dt vs ℓ |
| `fig2_sigma_be.pdf` | σ(b₀e) vs N_FRB |
| `fig3_sigma_fNL_bar.pdf` | Bar chart of σ(fNL) for all cases |
| `fig4_sigma_fNL_vs_N.pdf` | σ(fNL) vs N_FRB |
| `fig5_lmin_vs_N.pdf` | Minimum baseline vs N_FRB |

---

## Survey Configurations

| Survey | Spectral index α | Description |
|---|---|---|
| Shallow | 3.5 | Bright/nearby FRB population |
| Deep | 2.0 | Faint/high-z FRB population |

Baselines explored: **100 AU** and **500 AU** (pulsar timing / VLBI-style).

---

## Key Results

At N_FRB = 10⁴ and l = 100 AU:

| Survey | σ(fNL) Case A | σ(fNL) Case B | σ(fNL) Case C | Improvement B→C |
|---|---|---|---|---|
| Shallow | — | — | — | see fig3 |
| Deep | — | — | — | see fig3 |

*(Run the pipeline to populate this table.)*

---

## Cosmology

Planck 2018 ΛCDM:

- H₀ = 67.4 km/s/Mpc
- Ωm = 0.315
- Ωb = 0.049
- All distances in Mpc; times in seconds.

---

## Citation

If you use this pipeline, please cite:

```
Dlamini, Simthembile. (2026). Fast Radio Bursts and Primordial Non-Gaussianity 
Cosmology Pipeline: Constraints via Dispersion Measure and Timing Cross-Spectra. 
Zenodo. https://doi.org/10.5281/zenodo.19440686
```

And the benchmark it extends:

```
Reischke et al. 2020 — Probing cosmology with Fast Radio Bursts
```

---

## License

MIT License — see `LICENSE` for details.
