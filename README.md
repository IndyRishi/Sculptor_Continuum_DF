# Sculptor & Fornax Dark-Matter Inner-Slope Pipeline

A chemo-dynamical pipeline that measures the **dark-matter inner density slope** of the
Sculptor (and Fornax) dwarf spheroidal galaxies across several independent modelling
frameworks, and tests a central methodological question:

> Is the standard **two-population split** of a dwarf's stars (metal-rich vs metal-poor)
> justified, or is the galaxy a **continuous metallicity–kinematics sequence** whose
> arbitrary splitting biases the inferred DM slope?

The pipeline reproduces the **Walker & Peñarrubia (2011)** mass-slope method and the
**Arroyo-Polonio et al. (2025, "AP25")** action-based analysis on real spectroscopic data,
and introduces a novel **continuous `f(J, [Fe/H])`** action-distribution-function model in
which metallicity is a coordinate inside a single DF rather than a label for two populations.

---

## Key results (real Sculptor data, Tolstoy et al. 2023; 1339 members)

| Framework | Method | Inner slope |
|---|---|---|
| Spherical Jeans (`--dm5`) | gNFW, σ_los only | γ = 0.78 (+0.29 / −0.39) |
| GravSphere (`--gravsphere`) | Jeans + Virial Shape Parameters + free β(r) | γ = 0.48 (+0.41 / −0.29) |
| Walker & Peñarrubia 2011 (`--wp11`) | Two-subcomponent mass slope | Γ = 2.50 (+0.23 / −0.20); NFW excluded 99.6% |
| Continuous `f(J,[Fe/H])` (`--continuous`) | Novel single-DF, metallicity as coordinate | validated on mocks (γ, k_J recovered); real-data run in progress |

All frameworks point to a **core-like** (shallow) inner slope, consistent within
uncertainties with AP25's published γ = 0.39 (+0.23 / −0.26).

### Evidence that Sculptor is one continuous population

- **Metallicity is statistically unimodal** by three independent tests: bimodality
  coefficient BC = 0.30 (< 0.555), ΔBIC = −8 (favours a single Gaussian), and Hartigan's
  dip test p = 0.79 (consistent with unimodality) — despite a visual impression of structure.
- **Kinematics vary smoothly** with any imposed metallicity split (no plateau).
- **σ_los alone is degenerate** in γ (a core-through-cusp range fits equally well), which is
  why point estimates from different methods scatter — posteriors, not MLEs, are reported.
- **On mocks, a two-population split biases γ high** (+0.32 to +0.42), while the continuous
  treatment is far closer to truth (+0.05 to +0.14).

---

## Installation

Requires Python 3.11+ and a scientific stack. The heavy dynamical modelling uses
[AGAMA](https://github.com/GalacticDynamics-Oxford/Agama).

```bash
# core dependencies
pip install numpy scipy pandas matplotlib emcee corner astropy astroquery h5py

# optional but recommended
pip install diptest          # enables the Hartigan dip test on the [Fe/H] distribution

# AGAMA (needs a C++ toolchain and GSL)
#   Linux:  apt-get install -y libgsl-dev
pip install agama --no-build-isolation
```

Real-data commands query [VizieR](https://vizier.cds.unistra.fr/) for the spectroscopic
catalogs, so they require network access. Mock/validation commands run fully offline.

---

## Usage

All analyses are exposed as flags on the single entry-point script. Add `--galaxy fornax`
to any real-data command to switch targets (outputs are automatically prefixed `fornax_`).

### Evidence figures (fast, minutes)
```bash
python sculptor_full_pipeline.py --overview     # data overview + [Fe/H] unimodality stats
python sculptor_full_pipeline.py --slide        # sliding-threshold σ_los + γ-degeneracy curve
python sculptor_full_pipeline.py --biasgate     # two-population split biases γ (mocks, 40 realizations)
python sculptor_full_pipeline.py --biasconv     # mean bias vs number of realizations
```

### Dark-matter slope measurements (MCMC; resume by re-running)
```bash
python sculptor_full_pipeline.py --dm5          # spherical-Jeans gNFW inner slope
python sculptor_full_pipeline.py --gravsphere   # GravSphere: Jeans + VSPs + free β(r)
python sculptor_full_pipeline.py --wp11         # Walker & Peñarrubia 2011 mass slope (reproduces their Fig. 10)
python sculptor_full_pipeline.py --continuous   # novel continuous f(J,[Fe/H]) model
python sculptor_full_pipeline.py --chain        # full 25-parameter AP25 action-DF (cluster-scale)
```

### Comparison figures
```bash
python sculptor_full_pipeline.py --compare      # γ posteriors across frameworks
python sculptor_full_pipeline.py --fig4all      # AP25 Fig. 4: DM density ρ(r) from all chains + AP25's curve
```

### Validation / cross-checks
```bash
python sculptor_full_pipeline.py --wp11 --mock          # WP11 recovery on a known-Γ mock
python sculptor_full_pipeline.py --continuous --mock    # continuous method recovers known γ, k_J
python sculptor_full_pipeline.py --crosscheck --repo <path/to/gravsphere>   # vs reference GravSphere
```

### Common flags
`--steps N` (MCMC steps to add), `--walkers N`, `--nproc N` (parallel processes),
`--nsub N` (fit a random subsample for tractability), `--backend FILE.h5` (checkpoint),
`--no-resume`, `--galaxy {sculptor,fornax}`, `--mock`.

**MCMC runs checkpoint every step** to an HDF5 backend and **resume** on re-launch — run the
same command across multiple sessions until the convergence report stops reporting
`NOT CONVERGED`. Check progress from a second terminal:
```bash
python -c "import emcee; b=emcee.backends.HDFBackend('wp11.h5'); print('steps:', b.iteration)"
```

---

## Outputs

### Chains (`.npy`, posterior samples)
`dm5_chain.npy`, `gravsphere_chain.npy`, `wp11_chain.npy`, `cont_chain.npy`,
`ap25_chain.npy` (Fornax runs prefix `fornax_`).

### Figures
| File | Content |
|---|---|
| `figure_data_overview.png` | Sample overview + [Fe/H] unimodality (BC, ΔBIC, dip test) |
| `figure_sliding_metallicity.png` | σ_los continuum + σ_los γ-degeneracy curve |
| `figure_bias_gate.png` | Two-population split biases γ (mocks) |
| `figure_bias_vs_realizations.png` | Bias convergence vs number of realizations |
| `figure_dm5_corner.png` | Jeans gNFW posterior |
| `figure_gravsphere_corner.png`, `figure_gravsphere_beta.png` | GravSphere posterior + anisotropy β(r) |
| `figure_wp11.png`, `figure_wp11_corner.png` | WP11 Fig. 10 reproduction (Γ, NFW exclusion) |
| `figure_continuous_corner.png` | Continuous `f(J,[Fe/H])` posterior |
| `figure_ap25_fig4.png` | DM density/mass profile + AP25's published curve |
| `figure_fig4_all_chains.png` | DM density ρ(r) from **all** frameworks + AP25's curve |
| `figure_framework_comparison.png` | γ posteriors across frameworks |

---

## Methods

**Data.** Sculptor: Tolstoy et al. (2023) VizieR catalog `J/A+A/675/A49`, members with
reliable [Fe/H], in the galaxy rest frame, on elliptical (semi-major-axis) radii (Muñoz+2018
centre/ellipticity/PA, D = 84 kpc). Fornax: Walker et al. (2009) MMFS (`J/AJ/137/3100`),
using the Mg spectral index W′ as the metallicity separator (as WP11 did), D = 147 kpc.

**Frameworks.**
- *Spherical Jeans* — gNFW halo (α=1, η=3), Osipkov–Merritt anisotropy, fit to the binned
  σ_los(R) of the two metallicity halves.
- *GravSphere* — a from-scratch implementation of Read & Steger (2017): spherical Jeans with
  two Virial Shape Parameters and a free Baes–van Hese anisotropy profile β(r), cross-checked
  against the reference `gravsphere` code (σ_los to 0.3%, VSPs to 0.14%).
- *Walker & Peñarrubia (2011)* — two chemo-dynamically distinct stellar subcomponents, each a
  Plummer sphere with Gaussian velocity and metallicity distributions; the mass estimator
  M(r_h) = 5 r_h σ² / (2G) at two half-light radii gives the slope Γ ≡ Δlog M / Δlog r.
  Γ > 2 excludes an NFW cusp.
- *AP25 action-DF* — the full 25-parameter model with the paper's Eq. 4 gNFW-with-cutoff
  potential.
- *Continuous `f(J,[Fe/H])`* (this work) — a single DoublePowerLaw DF whose scale action
  varies smoothly with metallicity, `log₁₀ J₀(z) = logJ₀ + k_J·(z − ⟨z⟩)`; each star's
  likelihood marginalises over its true metallicity. A detected `k_J < 0` means the
  metal-rich stars are centrally concentrated — a gradient, captured without splitting.

---

## Scope & caveats

- Reported slopes are **MCMC posteriors, never MLE point estimates**, because the σ_los-only
  likelihood is degenerate in γ.
- The full 25-parameter AP25 action-DF (`--chain`) is **cluster-scale**; on a laptop it is run
  on subsamples with honest wide error bars, and the reduced frameworks (`--dm5`,
  `--gravsphere`, `--wp11`) serve as the discrete baselines.
- WP11's absolute masses differ from their Table 4 by a constant estimator offset (~0.16 dex)
  that **cancels in Γ**; the slope reproduces their value exactly.
- The AP25 curve overplotted in the Fig. 4 figures is computed from their **published
  parameters via their Eq. 4** (not by re-running their method), anchored at r_s for an
  inner-slope comparison.
- **Fornax is "verify-then-run":** confirm the Walker+2009 catalog column names in a networked
  session before trusting Fornax chains (start with `--overview --galaxy fornax`).

---

## Repository layout

- `continuous_agama.py` — the complete pipeline (all frameworks and figures).

---

## References

- Tolstoy et al. (2023), *A&A* 675, A49
- Walker & Peñarrubia (2011), *ApJ* 742, 20
- Read & Steger (2017), *MNRAS* 471, 4541
- Arroyo-Polonio et al. (2025), *A&A* (Sculptor II)
- Muñoz et al. (2018); Walker et al. (2009)
- AGAMA: Vasiliev (2019), *MNRAS* 482, 1525

*This pipeline was developed as part of a research project on dwarf-spheroidal dark
matter. Results involving the continuous method are preliminary pending final convergence of
the real-data chains.*
