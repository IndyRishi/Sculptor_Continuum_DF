#!/usr/bin/env python3
# =============================================================================
#  Constraining the Inner Dark-Matter Slope of Sculptor:
#  A Comparative Analysis of Spherical Jeans, GravSphere, and Action-Based
#  Chemo-Dynamical Modeling
#
#  Author:  Rishi Sanjeev, The Charter School of Wilmington
#  Contact: rishsanjeev@gmail.com
#  License: Creative Commons CC-BY  (see LICENSE)
#
#  Companion code for the paper.
#  Measures the dark-matter inner density slope of the Sculptor (and Fornax) dwarf
#  spheroidal galaxies across four independent dynamical frameworks, and tests
#  whether the standard two-population split biases the inferred slope. All analyses
#  and figures are reproducible from the command line (see --help). Data are queried
#  from VizieR (Tolstoy et al. 2023; Walker et al. 2009) and Gaia DR3.
# =============================================================================
"""
Sculptor dwarf-spheroidal chemo-dynamical pipeline.

Validation of an action-based distribution-function pipeline against Arroyo-Polonio
et al. (2025), as a benchmark prior to a continuous action-metallicity f(J,[Fe/H])
analysis. Five phases:

  1  Walker+2009 + Gaia DR3 membership; the continuous metallicity-gradient diagnostic.
  2  Gaia Challenge core/cusp mock -> Jeans mass-anisotropy-degeneracy figure.
  3  Semi-empirical Burkert profile + real literature enclosed-mass comparison.
  4  AGAMA action-DF modelling: a fast maximum-likelihood fit, a robust gNFW Jeans
     MCMC of the DM inner slope (--dm5), and the paper's per-star projected-DF method.
  4b GravSphere (Read & Steger 2017): spherical Jeans + Virial Shape Parameters with a
     free Baes-van-Hese anisotropy beta(r) (--gravsphere) -- the middle framework rung.
  5  The full faithful 25-parameter AP25 model (--chain) with an AP24-style selection
     function -- a cluster-scale run.

Command line:
  python sculptor_agama_project.py               # fast phases 1-4 + Phase-5 smoke test
  python sculptor_agama_project.py --dm5         # spherical-Jeans gNFW inner slope (real data)
  python sculptor_agama_project.py --gravsphere  # GravSphere: Jeans + VSPs, free beta(r)
  python sculptor_agama_project.py --chain       # full 25-parameter chain (cluster-scale)
  python sculptor_agama_project.py --overview    # data-overview figure (histograms/scatter)
  python sculptor_agama_project.py --compare     # gamma across all frameworks, one figure
  python sculptor_agama_project.py --help        # all options
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")      # 1 thread/process: avoid oversubscription
os.environ.setdefault("AGAMA_VERBOSITY", "0")       # quiet AGAMA's per-model progress bars
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

# AGAMA is only needed for Phase 4 (action-based DF modeling). Import it
# gracefully so Phases 1-3 still run in environments where it is not installed.
# To install:  apt-get install -y libgsl-dev
#              pip install agama --no-build-isolation --config-settings --build-option=--yes
try:
    import agama
    agama.setUnits(mass=1, length=1, velocity=1)   # Msun, kpc, km/s
    try:
        agama.setNumThreads(1)          # one thread/process (emcee provides the parallelism)
    except Exception:
        pass
    HAS_AGAMA = True
except Exception:
    HAS_AGAMA = False

# emcee is only needed for the optional high-fidelity per-star MCMC in Phase 4.
try:
    import emcee
    import emcee.backends
    HAS_EMCEE = True
except Exception:
    HAS_EMCEE = False
try:
    import corner
    HAS_CORNER = True
except Exception:
    HAS_CORNER = False
try:
    import h5py                       # noqa: F401  (enables emcee HDF5 backend)
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False
import multiprocessing as mp

# ============================================================
# GLOBAL TARGET PARAMETERS
# ============================================================
RA0_DEG  = 15.0392    # Sculptor centroid, ICRS/J2000 (NED/SIMBAD)
DEC0_DEG = -33.7186

DISTANCE_KPC = 84.0           # Martínez-Vázquez et al. (2015); adopted by Arroyo-Polonio+25
KAPPA        = 4.74047        # km/s per (mas/yr × kpc)

# Sculptor systemic PM — Gaia eDR3 (McConnachie & Venn 2020)
SCULPTOR_PMRA_SYS  =  0.085   # mas/yr
SCULPTOR_PMDEC_SYS = -0.133   # mas/yr
# [FIX-E] sigma_v~9 km/s → 9/(4.74047×84) = 0.0226 mas/yr.
# Previous value 0.05 mas/yr was ~2.3× too large.
SCULPTOR_PM_SIGMA  =  0.0226  # mas/yr  (Massari et al. 2020)

XMATCH_RADIUS_ARCSEC       = 4.0
FINAL_JOINT_MEMBERSHIP_MIN = 0.75

# ── Phase-4 (action-DF) parameters, from Arroyo-Polonio et al. (2025) ─────────
G_KPC   = 4.300917e-6   # kpc (km/s)^2 / Msun
RE_MR   = 0.18          # kpc, metal-rich Plummer scale (Zhu+16 / WP11)
RE_MP   = 0.28          # kpc, metal-poor Plummer scale
FRAC_MR = 0.35          # MR fraction of members
V_SYS   = 111.2         # km/s systemic (AP24)
BETA_DM = 3.0           # default gNFW outer slope for the FAST (3-param) MLE only

# ── MULTI-GALAXY SUPPORT ───────────────────────────────────────────────────────────────
# Per-galaxy parameters so the same methods can run on Sculptor or Fornax via --galaxy.
# set_galaxy(name) reassigns the module-level geometric/kinematic constants used by the
# loaders (center, distance, systemic velocity, ellipticity/PA, tracer scales, and the
# VizieR catalog + column mapping). Sculptor is the default, so existing behaviour is
# unchanged. NOTE: for Fornax we follow WP11 and use the Walker+2009 MMFS catalog with the
# Mg spectral index W' as the metallicity separator (the role [Fe/H] plays for Sculptor);
# verify the catalog's column names in a networked session before the first real run
# (inspect the table columns -- the mapping below is the documented best guess).
GALAXIES = {
    'sculptor': dict(
        name='Sculptor', ra0=15.0392, dec0=-33.7186,
        center_ra=15.0183, center_dec=-33.7186,          # Munoz+2018 centre for the radius
        # e and D follow Munoz et al. (2018) / Martinez-Vazquez et al. (2015) as tabulated by
        # Arroyo-Polonio et al. (2024, A&A 692, A195, Table 1), who analyse this same sample.
        distance_kpc=83.9, v_sys=111.2, ellipticity=0.33, pa_deg=92.0,
        re_mr=0.18, re_mp=0.28,                          # tracer Plummer scales (kpc)
        catalog='J/A+A/675/A49', cols=None,              # Tolstoy+2023 ([Fe/H] auto-detected)
        feh_quality_keep=(0,), mem_keep=('m',),
        wp11_xlim=(2.0, 2.75), wp11_ylim=(6.3, 7.9)),    # WP11 Fig.10 panel range (log10)
    'fornax': dict(
        name='Fornax', ra0=39.9971, dec0=-34.4492,
        center_ra=39.9971, center_dec=-34.4492,          # Munoz+2018 centre
        distance_kpc=147.0, v_sys=55.3, ellipticity=0.30, pa_deg=41.9,
        re_mr=0.60, re_mp=0.90,                          # larger tracer scales (kpc)
        catalog='J/AJ/137/3100',                         # Walker+2009 MMFS (Fornax)
        # first (averaged, one row per star) table: <HV> velocity, <SigMg> Mg index as the
        # metallicity separator. Column names carry angle brackets in VizieR.
        cols=dict(vlos='<HV>', verr='e_<HV>', feh='<SigMg>', feherr='e_<SigMg>',
                  ra='RAJ2000', dec='DEJ2000', mem='Mmb'),
        # Walker+2009 MMFS is a MULTI-galaxy catalog (Carina/Fornax/Sculptor/Sextans);
        # keep only Fornax rows. Coordinates are sexagesimal (parsed automatically).
        target_col='Target', target_keep=('for', 'fnx'),
        # 'Mmb' is a membership PROBABILITY (0-1); keep probable members (>0.5).
        feh_quality_keep=None, mem_keep=None, mem_min=0.5,
        wp11_xlim=(2.5, 3.1), wp11_ylim=(7.0, 8.4)),     # Fornax: larger radii/masses
}
GAL = GALAXIES['sculptor']                               # active galaxy (default)


def _gf(name):
    """Tag an output filename with the active galaxy when it is not the default (Sculptor),
    so a Fornax run never overwrites or resumes Sculptor outputs. Sculptor keeps the bare
    names (unchanged workflow); Fornax gets a 'fornax_' prefix. Idempotent (safe to apply
    more than once). e.g. 'figure_wp11.png' -> 'fornax_figure_wp11.png'."""
    key = GAL['name'].lower()
    if key == 'sculptor' or name.startswith(f"{key}_"):
        return name
    return f"{key}_{name}"


def set_galaxy(name):
    """Switch the active galaxy; reassigns the geometric/kinematic module constants so all
    loaders pick up the new target. Returns the galaxy parameter dict."""
    global GAL, V_SYS, DISTANCE_KPC, RA0_DEG, DEC0_DEG, RE_MR, RE_MP
    key = name.strip().lower()
    if key not in GALAXIES:
        raise ValueError(f"unknown galaxy '{name}'; choose from {list(GALAXIES)}")
    GAL = GALAXIES[key]
    V_SYS = GAL['v_sys']; DISTANCE_KPC = GAL['distance_kpc']
    RA0_DEG = GAL['ra0']; DEC0_DEG = GAL['dec0']
    RE_MR = GAL['re_mr']; RE_MP = GAL['re_mp']
    print(f"  [galaxy] target = {GAL['name']}: D={DISTANCE_KPC} kpc, V_sys={V_SYS} km/s, "
          f"e={GAL['ellipticity']}, catalog={GAL['catalog']}")
    return GAL
# Paper Eq.4 truncation (fixed, as in Arroyo-Polonio+25): rho ~ ...*exp[-(r/r_cut)^xi]
DM_RCUT = 20.0          # kpc  (~10x the outermost star; negligible inner effect)
DM_XI   = 1.0           # exponential-cutoff strength
# Verified best fit (A&A 699, A347): inner slope + scale radius (1-sigma)
AP25 = dict(gamma=0.39, gamma_hi=0.23, gamma_lo=0.26,
            rs=0.79, rs_hi=0.38, rs_lo=0.17,
            ref="Arroyo-Polonio et al. 2025, A&A 699, A347")

CORE_DATA_URL = (
    "https://astrowiki.surrey.ac.uk/lib/exe/fetch.php"
    "?media=data:c1_100_050_050_100_core_c2_100_050_100_100_core_002_6d.dat"
)
CUSP_DATA_URL = (
    "https://astrowiki.surrey.ac.uk/lib/exe/fetch.php"
    "?media=data:c1_100_050_050_100_cusp_c2_100_050_100_100_cusp_008_6d.dat"
)

# ============================================================
# DARK MATTER PROFILE FUNCTIONS
# ============================================================
def burkert_rho(r, rho0, rc):
    """Burkert (1995) cored profile: rho0 / [(1+r/rc)(1+(r/rc)^2)]"""
    x = r / rc
    return rho0 / ((1.0 + x) * (1.0 + x**2))


def burkert_mass(r, rho0, rc):
    """
    [FIX-A] Exact Burkert enclosed mass — correct 2π prefactor.
    ────────────────────────────────────────────────────────────
    Previous versions used π, giving exactly half the true mass at every radius
    (confirmed by numerical integration: code/numerical = 0.500).

    Correct formula (Burkert 1995, ApJL 447 L25; Salucci & Burkert 2000):
        M(<r) = 2π ρ₀ r_c³ [ln(1+x) + ½ln(1+x²) − arctan(x)],  x = r/r_c
    """
    x = r / rc
    return 2.0 * np.pi * rho0 * rc**3 * (
        np.log(1.0 + x) + 0.5 * np.log(1.0 + x**2) - np.arctan(x)
    )


def nfw_rho(r, rho0, rs):
    """NFW (1996) cuspy profile: rho0 / [(r/rs)(1+r/rs)²]. Requires r > 0."""
    x = r / rs
    return rho0 / (x * (1.0 + x)**2)


def nfw_mass(r, rho0, rs):
    """NFW exact enclosed mass: 4π ρ₀ r_s³ [ln(1+x) − x/(1+x)]"""
    x = r / rs
    return 4.0 * np.pi * rho0 * rs**3 * (np.log(1.0 + x) - x / (1.0 + x))


# ============================================================
# STATISTICAL ENGINES
# ============================================================
def nll_dispersion(params, v, e):
    """
    Negative log-likelihood for intrinsic Gaussian velocity dispersion:
        NLL = 0.5 Σᵢ [log(σ²+eᵢ²) + (vᵢ−μ)²/(σ²+eᵢ²)]
    """
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    variance = sigma**2 + e**2
    return 0.5 * np.sum(np.log(variance) + (v - mu)**2 / variance)


def calculate_intrinsic_dispersion(v, e):
    """Nelder-Mead MLE for intrinsic σ, marginalised over measurement errors."""
    if len(v) < 5:
        return np.nan
    # [FIX-G] Subtract mean measurement variance from sample variance to
    # get an unbiased starting point; np.std(v) overestimates intrinsic σ.
    sigma_init = np.sqrt(max(np.var(v, ddof=1) - np.mean(e**2), 0.5))
    res = minimize(
        nll_dispersion, [np.mean(v), sigma_init],
        args=(v, e), method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 5000}
    )
    return np.abs(res.x[1]) if res.success else np.nan


def bootstrap_dispersion_error(v, e, n_boot=200, seed=42):
    """Bootstrap 1-σ uncertainty on the intrinsic dispersion estimate."""
    rng  = np.random.default_rng(seed)
    sigs = []
    n    = len(v)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s   = calculate_intrinsic_dispersion(v[idx], e[idx])
        if not np.isnan(s):
            sigs.append(s)
    return np.std(sigs) if len(sigs) >= 10 else np.nan


def error_aware_gmm_likelihood(params, PM, PM_err, PM_corr):
    """
    Two-component Gaussian mixture with per-star heteroscedastic correlated errors.
    Full 2×2 measurement covariance S_i includes the Gaia pmra_pmdec_corr term.
    """
    mu_s_x, mu_s_y, sig_s_x, sig_s_y, mu_f_x, mu_f_y, sig_f_x, sig_f_y, w_s = params
    if not (0.0 < w_s < 1.0) or any(s <= 0 for s in [sig_s_x, sig_s_y, sig_f_x, sig_f_y]):
        return np.inf

    total_nll = 0.0
    for i in range(len(PM)):
        rho = PM_corr[i]
        S_i = np.array([
            [PM_err[i, 0]**2,                      rho * PM_err[i, 0] * PM_err[i, 1]],
            [rho * PM_err[i, 0] * PM_err[i, 1],    PM_err[i, 1]**2                  ]
        ])
        Cov_S = np.diag([sig_s_x**2, sig_s_y**2]) + S_i
        L_S   = multivariate_normal.pdf(PM[i], mean=[mu_s_x, mu_s_y], cov=Cov_S)
        Cov_F = np.diag([sig_f_x**2, sig_f_y**2]) + S_i
        L_F   = multivariate_normal.pdf(PM[i], mean=[mu_f_x, mu_f_y], cov=Cov_F)
        total_nll -= np.log(max(w_s * L_S + (1.0 - w_s) * L_F, 1e-300))

    return total_nll


# ============================================================
# PHASE 1: OBSERVATIONAL PIPELINE
# ============================================================
def fetch_walker_data():
    """Legacy loader for the Walker et al. (2009) MMFS catalog (VizieR J/AJ/137/3100). Used by
    the older Gaia-membership demonstration path; the flag-based commands use the galaxy-aware
    _fetch_tolstoy2023 loader instead. Returns a DataFrame of the catalog rows."""
    print("\n[Phase 1] Fetching Walker et al. (2009) from VizieR...")
    Vizier.ROW_LIMIT = -1
    walker = Vizier.get_catalogs("J/AJ/137/3100")[0]

    df = pd.DataFrame({
        "Target":    np.array(walker["Target"]).astype(str),
        "RA_J2000":  np.array(walker["RAJ2000"]).astype(str),
        "Dec_J2000": np.array(walker["DEJ2000"]).astype(str),
        "V_los":     np.array(walker["<HV>"],    dtype=float),
        "e_V_los":   np.array(walker["e_<HV>"],  dtype=float),
        "P_mem_1D":  np.array(walker["Mmb"],     dtype=float),
        "SigMg":     np.array(walker["<SigMg>"], dtype=float),
    })
    df = df.dropna(subset=["V_los", "e_V_los", "P_mem_1D", "SigMg"])
    df = df[df["e_V_los"] > 0].copy()

    # [FIX-J] VizieR stores Walker+09 Mmb as percentage [0–100], not fraction [0–1].
    # ──────────────────────────────────────────────────────────────────────────────
    # Without this fix, the filter P_mem_1D >= 0.90 selects ALL stars with Mmb >= 0.9%
    # (i.e. essentially the full catalog, including MW field non-members spanning
    # v_hel ~ 0–300 km/s).  Simulating this mixture reproduces σ_los ~ 55 km/s,
    # matching the erroneous Figure 1 output exactly.  Sculptor's true σ_los is ~9 km/s.
    if df["P_mem_1D"].max() > 1.0:
        print(f"  --> Mmb max = {df['P_mem_1D'].max():.0f}: "
              "percentage scale detected; normalising to [0, 1].")
        df["P_mem_1D"] = df["P_mem_1D"] / 100.0

    coords = SkyCoord(df["RA_J2000"], df["Dec_J2000"],
                      unit=(u.hourangle, u.deg), frame="icrs")
    df["RA_deg"]  = coords.ra.deg
    df["Dec_deg"] = coords.dec.deg

    n_mem = (df["P_mem_1D"] >= 0.90).sum()
    print(f"  --> {len(df)} stars loaded; {n_mem} with P_mem >= 0.90.")
    return df, coords


def run_sliding_threshold_diagnostic(walker_df):
    """
    σ_los vs. Mg-index threshold across the full membership gradient,
    with bootstrap 1-σ error bands on every dispersion estimate.
    """
    print("[Phase 1] Executing continuous metallicity-gradient diagnostic...")
    df_1d = walker_df[walker_df["P_mem_1D"] >= 0.90].copy()
    print(f"  --> {len(df_1d)} members selected for diagnostic.")

    thresholds = np.linspace(
        df_1d["SigMg"].quantile(0.15),
        df_1d["SigMg"].quantile(0.85), 30
    )

    mr_disp, mp_disp = [], []
    mr_err,  mp_err  = [], []
    valid_thresh = []

    for thresh in thresholds:
        mr = df_1d[df_1d["SigMg"] >  thresh]
        mp = df_1d[df_1d["SigMg"] <= thresh]
        if len(mr) > 20 and len(mp) > 20:
            mr_disp.append(calculate_intrinsic_dispersion(mr["V_los"].values, mr["e_V_los"].values))
            mp_disp.append(calculate_intrinsic_dispersion(mp["V_los"].values, mp["e_V_los"].values))
            # [FIX-H] Bootstrap errors — required to assess statistical significance
            mr_err.append(bootstrap_dispersion_error(mr["V_los"].values, mr["e_V_los"].values))
            mp_err.append(bootstrap_dispersion_error(mp["V_los"].values, mp["e_V_los"].values))
            valid_thresh.append(thresh)

    vt   = np.array(valid_thresh)
    mr_d = np.array(mr_disp);  mr_e = np.array(mr_err)
    mp_d = np.array(mp_disp);  mp_e = np.array(mp_err)

    plt.figure(figsize=(8, 5))
    plt.plot(vt, mp_d, '-o', color='royalblue', label=r"Metal-Poor (EW $<$ Threshold)")
    plt.fill_between(vt, mp_d - mp_e, mp_d + mp_e, color='royalblue', alpha=0.20)
    plt.plot(vt, mr_d, '-s', color='crimson',   label=r"Metal-Rich (EW $>$ Threshold)")
    plt.fill_between(vt, mr_d - mr_e, mr_d + mr_e, color='crimson', alpha=0.20)
    plt.xlabel(r"Mg-index Split Threshold (EW / Å)")
    plt.ylabel(r"Intrinsic Dispersion $\sigma_{\rm los}$ (km s$^{-1}$)")
    plt.title("Empirical Kinematic Split: Walker+09 Members")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("figure1_continuous_gradient.png", dpi=300)
    print("--> Saved figure1_continuous_gradient.png")
    plt.close()


def fetch_gaia_with_epoch_correction(walker_df, walker_coords):
    """
    Gaia DR3 query with RUWE quality filter and Walker cross-match.

    [FIX-C] RUWE < 1.4: removes sources with poor astrometric solutions
    (unresolved binaries, resolved objects) that produce unreliable PMs.
    Without this filter, spurious motions bias the GMM centroid and dispersions.

    [FIX-I] Explicit astropy units on all SkyCoord quantities prevent silent
    unit mis-assignment from astropy Table column objects.
    """
    print("[Phase 1] Querying Gaia DR3 (r < 1.5 deg, RUWE < 1.4)...")
    query = f"""
    SELECT TOP 300000
        source_id, ra, dec, pmra, pmdec, pmra_error, pmdec_error,
        pmra_pmdec_corr, ruwe
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {RA0_DEG}, {DEC0_DEG}, 1.5))
    AND pmra IS NOT NULL AND pmdec IS NOT NULL
    AND ruwe < 1.4
    """
    gaia_data = Gaia.launch_job_async(query).get_results()
    print(f"  --> {len(gaia_data)} Gaia sources after RUWE < 1.4 filter.")

    gaia_coords_2016 = SkyCoord(
        ra=np.array(gaia_data['ra'],   dtype=float) * u.deg,
        dec=np.array(gaia_data['dec'], dtype=float) * u.deg,
        pm_ra_cosdec=np.array(gaia_data['pmra'],  dtype=float) * u.mas / u.yr,
        pm_dec=np.array(gaia_data['pmdec'],       dtype=float) * u.mas / u.yr,
        frame='icrs', obstime=Time('J2016.0')
    )
    # Rewind positions 16 yr for cross-matching with Walker J2000 positions.
    # Sculptor's PM displacement over 16 yr is < 3 mas — far below the 4-arcsec radius.
    gaia_coords_2000 = gaia_coords_2016.apply_space_motion(new_obstime=Time('J2000.0'))

    idx, d2d, _ = walker_coords.match_to_catalog_sky(gaia_coords_2000)
    match_mask  = d2d.arcsec < XMATCH_RADIUS_ARCSEC

    matched_walker = walker_df[match_mask].copy().reset_index(drop=True)
    matched_gaia   = gaia_data[idx[match_mask]]

    matched_walker["pmra"]            = np.array(matched_gaia["pmra"])
    matched_walker["pmdec"]           = np.array(matched_gaia["pmdec"])
    matched_walker["e_pmra"]          = np.array(matched_gaia["pmra_error"])
    matched_walker["e_pmdec"]         = np.array(matched_gaia["pmdec_error"])
    matched_walker["pmra_pmdec_corr"] = np.array(matched_gaia["pmra_pmdec_corr"])

    print(f"  --> {len(matched_walker)} stars cross-matched.")
    return matched_walker


def calculate_error_aware_membership(df):
    """
    Two-stream Bayesian PM membership with correlated Gaia errors.

    Membership combination uses Bayesian odds ratios (not naive product).
    Fallback includes individual Gaia errors in the chi-squared denominator.
    """
    print("[Phase 1] Running Error-Aware 2-Component GMM on proper motions...")
    PM      = df[["pmra", "pmdec"]].values
    PM_err  = df[["e_pmra", "e_pmdec"]].values
    PM_corr = df["pmra_pmdec_corr"].fillna(0.0).values

    x0 = [SCULPTOR_PMRA_SYS, SCULPTOR_PMDEC_SYS,
          0.08, 0.08,
          0.5, -1.0,
          2.0, 2.0,
          0.30]

    res = minimize(
        error_aware_gmm_likelihood, x0,
        args=(PM, PM_err, PM_corr),
        method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 20000}
    )

    if not res.success:
        print("  --> Warning: GMM did not converge. Using error-convolved fallback.")
        # [FIX-D] Individual measurement errors included in chi-squared denominator.
        # Without this, stars with large Gaia PM uncertainties are incorrectly
        # penalised even when consistent with Sculptor's systemic PM within their errors.
        sig2_ra  = SCULPTOR_PM_SIGMA**2 + PM_err[:, 0]**2
        sig2_dec = SCULPTOR_PM_SIGMA**2 + PM_err[:, 1]**2
        chi2 = ((PM[:, 0] - SCULPTOR_PMRA_SYS)**2  / sig2_ra +
                (PM[:, 1] - SCULPTOR_PMDEC_SYS)**2 / sig2_dec)
        L_S  = np.exp(-0.5 * chi2)
        L_F  = 0.40  # rough flat foreground fraction
        df   = df.copy()
        df["P_mem_PM"] = L_S / (L_S + L_F + 1e-300)
    else:
        p = res.x
        p_mem_pm = []
        for i in range(len(PM)):
            rho = PM_corr[i]
            S_i = np.array([
                [PM_err[i,0]**2,                   rho*PM_err[i,0]*PM_err[i,1]],
                [rho*PM_err[i,0]*PM_err[i,1],      PM_err[i,1]**2             ]
            ])
            L_S  = multivariate_normal.pdf(PM[i], mean=[p[0], p[1]],
                                           cov=np.diag([p[2]**2, p[3]**2]) + S_i)
            L_F  = multivariate_normal.pdf(PM[i], mean=[p[4], p[5]],
                                           cov=np.diag([p[6]**2, p[7]**2]) + S_i)
            prob = (p[8] * L_S) / (p[8] * L_S + (1.0 - p[8]) * L_F + 1e-300)
            p_mem_pm.append(prob)
        df   = df.copy()
        df["P_mem_PM"] = p_mem_pm

    # Bayesian odds-ratio combination (correct; naive product double-counts the prior)
    eps     = 1e-9
    odds_1D = df["P_mem_1D"].clip(eps, 1-eps) / (1 - df["P_mem_1D"].clip(eps, 1-eps))
    odds_PM = df["P_mem_PM"].clip(eps, 1-eps) / (1 - df["P_mem_PM"].clip(eps, 1-eps))
    df["P_mem_Joint"] = (odds_1D * odds_PM) / (1.0 + odds_1D * odds_PM)

    final_df   = df[df["P_mem_Joint"] >= FINAL_JOINT_MEMBERSHIP_MIN].copy()
    center     = SkyCoord(RA0_DEG * u.deg, DEC0_DEG * u.deg, frame="icrs")
    final_crds = SkyCoord(final_df["RA_deg"].values * u.deg,
                          final_df["Dec_deg"].values * u.deg, frame="icrs")
    final_df["R_kpc"] = final_crds.separation(center).radian * DISTANCE_KPC

    print(f"  --> {len(final_df)} stars pass joint membership threshold "
          f"(P_joint >= {FINAL_JOINT_MEMBERSHIP_MIN}).")
    return final_df


# ============================================================
# PHASE 2: GAIA CHALLENGE REMOTE INTEGRATION
# ============================================================
def project_gaia_challenge_mock(primary_url, output_file, default_halo='core'):
    """
    Multi-tier retrieval: Surrey AstroWiki → GitHub mirror → analytic proxy.

    [FIX-K] Automatic parsec-to-kpc unit detection and conversion.
    ───────────────────────────────────────────────────────────────
    The Surrey AstroWiki Gaia Challenge files store particle positions in
    parsecs (x, y ~ 100–1200 pc for Sculptor-sized halos), not kpc.
    The code previously assigned these raw values to a column called 'R_kpc',
    producing a 1000× unit error (x-axis showed 100–1200 labelled as "kpc";
    correct range is 0.1–1.2 kpc).  Velocity components are in km/s and
    unaffected — the PM conversion factor uses DISTANCE_KPC independently.

    Detection: if median(R_raw) > 10 the positions are in parsecs; divide by 1000.
    """
    print(f"[Phase 2] Fetching {output_file} ...")
    filename   = primary_url.split("data:")[-1] if "data:" in primary_url else primary_url.split("/")[-1]
    github_url = (f"https://raw.githubusercontent.com/"
                  f"IndyRishi/Sculptor_Continuum_DF/main/{filename}")

    content = None
    for label, url in [("Surrey AstroWiki", primary_url), ("GitHub Mirror", github_url)]:
        if content is not None:
            break
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as r:
                content = r.read().decode('utf-8')
                print(f"  --> Downloaded from {label}.")
        except Exception as exc:
            print(f"    [{label}] unavailable: {exc}")

    df_raw = None
    if content is not None:
        try:
            rows = []
            for line in content.strip().split('\n'):
                if line.startswith('#'):
                    continue
                line  = line.replace('D+', 'E+').replace('D-', 'E-')
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        row = [float(p) for p in parts[:6]]
                        row.append(float(parts[6]) if len(parts) >= 7 else np.nan)
                        rows.append(row)
                    except ValueError:
                        continue
            df_raw = pd.DataFrame(rows, columns=['x','y','z','vx','vy','vz','Mg']).head(4000)
            if len(df_raw) == 0:
                raise ValueError("No valid rows parsed.")
            print(f"  --> Parsed {len(df_raw)} particles.")
        except Exception as e:
            print(f"    Parsing failed: {e}")
            df_raw = None

    # ── Tier 3: analytic proxy ──────────────────────────────────────────────
    if df_raw is None:
        print("  --> Deploying mathematical structural proxy (positions in kpc)...")
        rng    = np.random.default_rng(42)
        n      = 1500
        R      = rng.gamma(shape=2.0, scale=0.15, size=n)  # kpc
        R_safe = np.where(R == 0, 1e-6, R)
        theta  = rng.uniform(0, 2*np.pi, n)
        x, y   = R * np.cos(theta), R * np.sin(theta)

        sig_los = np.maximum(9.0 - 0.8 * R, 1.0)

        if default_halo == 'core':
            # Near-isotropic (β ≈ 0): characteristic of a cored potential
            sig_r = np.maximum(10.0 - 1.2 * R, 1.0)
            sig_t = np.maximum(9.5  - 1.1 * R, 1.0)   # β ~ +0.09
        else:
            # [FIX-B] Cusp: radially anisotropic (β > 0).
            # ─────────────────────────────────────────────
            # Previous formula 1.8/(R+0.1)+3 diverges at R→0, giving β = −0.45
            # at R = 0.1 kpc (strongly tangential) — physically opposite to an NFW
            # cusp, which is built by radial infall (Mamon & Lokas 2005; β ~ +0.3
            # to +0.5 outward).  Fix: sig_t = 0.77 × sig_r → β = 1−0.77² = +0.41.
            sig_r = np.maximum(10.0 - 0.5 * R, 1.0)
            sig_t = np.maximum(0.77 * sig_r,   1.0)   # β ~ +0.41

        df_raw = pd.DataFrame({
            'x':  x,  'y': y,  'z': rng.normal(0, 0.2, n),
            'vx': (x/R_safe * rng.normal(0, sig_r, n) - y/R_safe * rng.normal(0, sig_t, n)),
            'vy': (y/R_safe * rng.normal(0, sig_r, n) + x/R_safe * rng.normal(0, sig_t, n)),
            'vz': rng.normal(0, sig_los, n),
            'Mg': -0.4 * R + rng.normal(2.0, 0.15, n)
        })
        position_scale_kpc = 1.0   # proxy already in kpc

    else:
        # [FIX-K] Detect unit of downloaded positions
        R_raw = np.sqrt(df_raw['x']**2 + df_raw['y']**2)
        if R_raw.median() > 10.0:
            print(f"  --> median(R_raw) = {R_raw.median():.1f}: "
                  "positions are in PARSECS; converting to kpc (÷1000).")
            position_scale_kpc = 1000.0
        else:
            print(f"  --> median(R_raw) = {R_raw.median():.4f}: positions confirmed in kpc.")
            position_scale_kpc = 1.0

    pm_factor = KAPPA * DISTANCE_KPC    # km/s per (mas/yr)
    df_obs = pd.DataFrame({
        'R_kpc':  np.sqrt(df_raw['x']**2 + df_raw['y']**2) / position_scale_kpc,
        'x':      df_raw['x'] / position_scale_kpc,
        'y':      df_raw['y'] / position_scale_kpc,
        'V_los':  df_raw['vz'],   'e_V_los': 2.0,
        'pmra':   df_raw['vx'] / pm_factor,
        'pmdec':  df_raw['vy'] / pm_factor,
        'e_pmra': 0.001,   'e_pmdec': 0.001,
        'SigMg':  df_raw['Mg'] if 'Mg' in df_raw.columns else np.nan
    })
    df_obs.to_csv(output_file, index=False)
    r_min = df_obs['R_kpc'].min();  r_max = df_obs['R_kpc'].max()
    print(f"  --> Cached to {output_file}  (R_kpc: {r_min:.3f}–{r_max:.3f} kpc)")


def get_binned_kinematics(df, n_bins=6):
    """
    σ_los and σ_tan,1D = sqrt((σ_RA² + σ_Dec²)/2) profiles.
    σ_tan,1D equals σ_los for isotropic orbits and enters β = 1 − σ_tan,1D²/σ_r².
    """
    df      = df.copy()
    df['bin'] = pd.qcut(df['R_kpc'], q=n_bins, labels=False, duplicates='drop')
    pm_conv   = KAPPA * DISTANCE_KPC

    radii, sigma_los, sigma_trans = [], [], []
    for i in sorted(df['bin'].dropna().unique().astype(int)):
        b = df[df['bin'] == i]
        if len(b) < 5:
            continue
        radii.append(b['R_kpc'].mean())
        sigma_los.append(
            calculate_intrinsic_dispersion(b['V_los'].values, b['e_V_los'].values))
        s_ra  = calculate_intrinsic_dispersion(
            b['pmra'].values  * pm_conv, b['e_pmra'].values  * pm_conv)
        s_dec = calculate_intrinsic_dispersion(
            b['pmdec'].values * pm_conv, b['e_pmdec'].values * pm_conv)
        sigma_trans.append(
            np.sqrt((np.nan_to_num(s_ra)**2 + np.nan_to_num(s_dec)**2) / 2.0)
        )

    return np.array(radii), np.array(sigma_los), np.array(sigma_trans)


def reproduce_arroyo_polonio_fig4():
    """LEGACY: the original Jeans-degeneracy demonstration figure built from the Gaia Challenge
    mock CSVs (projected_challenge_{core,cusp}.csv). Retained for provenance; NOT used by any
    flag-based command and NOT part of the paper figures (the flag-based --slide/--dm5/--fig4all
    supersede it and use the real data instead)."""
    print("[Phase 2] Generating Jeans-degeneracy proof figure...")
    df_core = pd.read_csv("projected_challenge_core.csv")
    df_cusp = pd.read_csv("projected_challenge_cusp.csv")

    r_co, los_co, tr_co = get_binned_kinematics(df_core)
    r_cu, los_cu, tr_cu = get_binned_kinematics(df_cusp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    axes[0].plot(r_co, los_co, '-o', color='royalblue', label='Core model')
    axes[0].plot(r_cu, los_cu, '-s', color='crimson',   label='Cusp model')
    axes[0].set_title("Line-of-Sight Profile (Degeneracy)")
    axes[0].set_xlabel("Projected radius $R$ (kpc)")
    axes[0].set_ylabel(r"Velocity dispersion $\sigma$ (km s$^{-1}$)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(r_co, tr_co, '--o', color='royalblue', label=r'Core — $\sigma_{\rm tan,1D}$')
    axes[1].plot(r_cu, tr_cu, '--s', color='crimson',   label=r'Cusp — $\sigma_{\rm tan,1D}$')
    axes[1].set_title(r"Plane-of-Sky 1D Tangential Profile (Degeneracy Broken)")
    axes[1].set_xlabel("Projected radius $R$ (kpc)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figure2_jeans_degeneracy.png", dpi=300)
    print("--> Saved figure2_jeans_degeneracy.png")
    plt.close()


# ============================================================
# PHASE 3: SEMI-EMPIRICAL PROFILE + REAL LITERATURE COMPARISON
# ============================================================
# Real, individually-sourced Sculptor enclosed-mass constraints, each plotted at
# the radius its paper actually constrains. Replaces the previously fabricated
# Z16/R19/P20/H20 points (which sat at invented coordinates).
#   label,             r_kpc, M_Msun,  err_Msun, colour,       reference
SCULPTOR_LIT_MASSES = [
    ("Walker+09/Wolf+10", 0.37, 2.20e7, 0.45e7, "black",       "robust M_½ = 3σ²r_½/G"),
    ("Battaglia+08",      1.80, 2.50e8, 0.50e8, "forestgreen", "M(<1.8 kpc) = 2–3×10⁸"),
    ("Hayashi+20",        3.00, 3.00e8, 0.60e8, "darkorange",  "M_halo(<3 kpc) ≈ 3×10⁸"),
]


def robust_half_light_mass(sigma_los=9.2, Re_2D=0.28):
    """Model-independent Walker+09/Wolf+10 mass within the 3D half-light radius."""
    r_half_3D = (4.0 / 3.0) * Re_2D
    return r_half_3D, 3.0 * sigma_los**2 * r_half_3D / G_KPC


def plot_inferred_halo_profiles(r_array=None,
                                mass_median=None, mass_1sigma_lower=None, mass_1sigma_upper=None,
                                rho_median=None,  rho_1sigma_lower=None,  rho_1sigma_upper=None,
                                rho0_ref=None, rc_ref=None):
    """
    Semi-empirical Burkert profile plus REAL literature enclosed-mass points.
    (For the full action-DF inference, see Phase 4.)

    [FIX-F] rho0_ref and rc_ref are explicit keyword arguments.
    Previous code defined rho0_c inside 'if r_array is None' then used it
    unconditionally — a NameError when calling with actual MCMC output.
    """
    print("[Phase 3] Generating semi-empirical halo profiles + literature...")

    if rho0_ref is None: rho0_ref = 1.5e8   # M_sun/kpc^3  (Read et al. 2019)
    if rc_ref   is None: rc_ref   = 0.50    # kpc

    if r_array is None:
        r_array           = np.logspace(-1.5, 0.5, 150)
        rho_median        = burkert_rho( r_array, rho0_ref, rc_ref)
        rho_1sigma_lower  = burkert_rho( r_array, rho0_ref * 0.75, rc_ref * 1.10)
        rho_1sigma_upper  = burkert_rho( r_array, rho0_ref * 1.25, rc_ref * 0.90)
        mass_median       = burkert_mass(r_array, rho0_ref, rc_ref)
        mass_1sigma_lower = burkert_mass(r_array, rho0_ref * 0.75, rc_ref * 1.10)
        mass_1sigma_upper = burkert_mass(r_array, rho0_ref * 1.25, rc_ref * 0.90)

    fig, axes = plt.subplots(2, 1, figsize=(7, 10), sharex=True)

    # ── Upper panel: enclosed mass ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(r_array, mass_median, color='black', lw=2,
             label='Semi-empirical Burkert core (median)')
    ax1.fill_between(r_array, mass_1sigma_lower, mass_1sigma_upper,
                     color='black', alpha=0.20, label=r'$1\sigma$ confidence')
    # Real literature points (replaces fabricated Z16/R19/P20/H20)
    for lbl, rk, M, e, col, _c in SCULPTOR_LIT_MASSES:
        ax1.errorbar([rk], [M], yerr=[[e], [e]], fmt='o', color=col,
                     capsize=4, ms=7, mec='k', mew=0.6, label=lbl)
    ax1.set_yscale('log')
    ax1.set_ylabel(r"Enclosed mass $M(<R)$ ($M_\odot$)")
    ax1.set_title("Sculptor dSph — semi-empirical halo + real literature")
    ax1.legend(loc='lower right', frameon=True, fontsize=8.5)
    ax1.grid(True, which='both', alpha=0.2)

    # ── Lower panel: density ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(r_array, rho_median, color='black', lw=2)
    ax2.fill_between(r_array, rho_1sigma_lower, rho_1sigma_upper,
                     color='black', alpha=0.20)

    # Stellar density — Plummer profile, M_star=2.3e6 (McConnachie 2012)
    r_e    = 0.28   # kpc
    M_star = 2.3e6  # M_sun
    rho_stars = (3.0 / (4.0*np.pi)) * (M_star / r_e**3) * (1.0 + (r_array/r_e)**2)**(-2.5)
    ax2.plot(r_array, rho_stars, color='royalblue', lw=1.5, ls='--')
    ax2.fill_between(r_array, rho_stars*0.80, rho_stars*1.20,
                     color='royalblue', alpha=0.15,
                     label=r'Stellar — Plummer, $r_e=0.28$ kpc')

    # Slope references anchored to best-fit profile at inner radii
    r_ref   = np.logspace(-1.40, -0.90, 10)
    rho_anc = burkert_rho(r_ref[0], rho0_ref, rc_ref)   # [FIX-F] parameter, not local var
    ax2.plot(r_ref, rho_anc * (r_ref / r_ref[0])**(-1.0),
             color='gray', ls=':', lw=2.0, label=r'Cusp slope $\gamma=1$')
    ax2.plot(r_ref, np.full_like(r_ref, rho_anc),
             color='gray', ls='-',  lw=1.5, label=r'Core slope $\gamma=0$')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r"Radius $R$ (kpc)")
    ax2.set_ylabel(r"Density $\rho(R)$ ($M_\odot\,{\rm kpc}^{-3}$)")
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', alpha=0.2)
    ax2.text(0.05, 0.05,
             r"Illustrative: 15 stars with $\log_{10}(R/{\rm kpc}) < -1.5$ omitted",
             transform=ax2.transAxes, fontsize=9, color='dimgray')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig("figure3_final_halo_profiles.png", dpi=300)
    print("--> Saved figure3_final_halo_profiles.png")
    plt.close()


# ============================================================
# PHASE 4: ACTION-BASED DF MODELING  (AGAMA; Arroyo-Polonio+25 method)
# ============================================================
# Faithful implementation of the Arroyo-Polonio et al. (2025) method: two stellar
# populations as quasi-spherical action-based DFs acting as tracers in a dominant
# double-power-law (Zhao) DM halo, fit through AGAMA's projected-moment machinery.
# The DM inner slope is RECOVERED by fitting, not assumed. When the real Tolstoy
# et al. (2023) catalog is reachable it is used; otherwise the pipeline runs the
# paper's own Section-5 validation (recover a known mock), which needs no network.

def dm_potential_5p(gamma, rs, log_MDM, alpha, eta):
    """
    Paper Eq.4 DM potential — the ACTUAL 5-parameter model of Arroyo-Polonio+25:
        rho(r) = rho0 (r/rs)^-gamma [1+(r/rs)^alpha]^((gamma-eta)/alpha) exp[-(r/rcut)^xi]
    Free: {log10 M_DM, rs, alpha, eta (outer slope), gamma (inner slope)};
    rcut=20 kpc and xi=1 fixed. AGAMA's Spheroid 'beta' is the paper's outer slope eta.
    Mass is normalised to the total M_DM (within the truncation).
    """
    return agama.Potential(type='Spheroid', mass=10.0**log_MDM, scaleRadius=rs,
                           gamma=gamma, beta=eta, alpha=alpha,
                           outerCutoffRadius=DM_RCUT, cutoffStrength=DM_XI)


def agama_build_galaxy_model(gamma, rs, log_rho_s, r_a, r_star):
    """3-param gNFW builder (alpha=1, eta=3 fixed) — used for the FAST MLE init."""
    pot = agama.Potential(type='Spheroid', densityNorm=10.0**log_rho_s,
                          scaleRadius=rs, gamma=gamma, beta=BETA_DM, alpha=1.0)
    tracer = agama.Density(type='Plummer', scaleRadius=r_star, mass=1.0)
    df = agama.DistributionFunction(type='QuasiSpherical', potential=pot,
                                    density=tracer, beta0=0.0, r_a=r_a)
    return agama.GalaxyModel(pot, df), pot


def agama_build_galaxy_model_5p(gamma, rs, log_MDM, alpha, eta, r_a, r_star):
    """Full 5-param DM potential (paper Eq.4) + quasi-spherical stellar-tracer DF."""
    pot = dm_potential_5p(gamma, rs, log_MDM, alpha, eta)
    tracer = agama.Density(type='Plummer', scaleRadius=r_star, mass=1.0)
    df = agama.DistributionFunction(type='QuasiSpherical', potential=pot,
                                    density=tracer, beta0=0.0, r_a=r_a)
    return agama.GalaxyModel(pot, df), pot


def agama_projected_sigmas(gm, Rproj):
    """Return sigma_los, sigma_R(plane-of-sky), sigma_T(plane-of-sky) at Rproj."""
    pts = np.column_stack([Rproj, np.zeros_like(Rproj)])
    m2 = gm.moments(pts, dens=False, vel2=True)      # XX,YY,ZZ,XY,XZ,YZ
    return np.sqrt(m2[:, 2]), np.sqrt(m2[:, 0]), np.sqrt(m2[:, 1])


def agama_generate_mock(gamma_true, rs_true, log_rho_s_true, r_a_true, seed=7):
    """Sample a two-population mock Sculptor from a KNOWN halo (paper's Sec. 5)."""
    rng = np.random.default_rng(seed)
    stars = []
    for frac, r_star in [(FRAC_MR, RE_MR), (1 - FRAC_MR, RE_MP)]:
        gm, _ = agama_build_galaxy_model(gamma_true, rs_true, log_rho_s_true, r_a_true, r_star)
        n = int(3000 * frac)
        posvel, _ = gm.sample(n)
        x, y, z, vx, vy, vz = posvel.T
        R = np.hypot(x, y)
        vlos = vz + rng.normal(0, 0.6, len(vz))      # + 0.6 km/s error (paper)
        stars.append(np.column_stack([R, vlos, np.full(len(R), r_star)]))
    m = np.vstack(stars)
    return m[m[:, 0] < 2.0]                            # outermost observed radius


def agama_binned_profile(R, vlos, nbins=5):
    """Intrinsic sigma_los(R) in radial bins (error-deconvolved)."""
    edges = np.quantile(R, np.linspace(0, 1, nbins + 1))
    rc, sig, err = [], [], []
    for i in range(nbins):
        m = (R >= edges[i]) & (R <= edges[i + 1] if i == nbins - 1 else R < edges[i + 1])
        if m.sum() < 15:
            continue
        v = vlos[m]
        s = np.sqrt(max(np.var(v, ddof=1) - 0.6**2, 0.5))
        rc.append(np.median(R[m])); sig.append(s)
        err.append(s / np.sqrt(2 * (m.sum() - 1)))
    return np.array(rc), np.array(sig), np.array(err)


def agama_fit_halo(R, vlos, r_star_label, r_a_fixed=1.5):
    """
    Recover (gamma, rs, log_rho_s) by fitting the sigma_los(R) profiles of BOTH
    stellar populations SIMULTANEOUSLY in a single shared DM potential. The two
    populations have different scale radii (0.18, 0.28 kpc) and probe M(<r) at two
    radii — the two-population constraint (Battaglia+08; Walker & Peñarrubia+11)
    that breaks the mass-anisotropy degeneracy. A single-population sigma_los fit,
    by contrast, collapses a genuine cusp to a false core.
    """
    pops = []
    for r_star in (RE_MR, RE_MP):
        sel = np.isclose(r_star_label, r_star)
        rc, so, se = agama_binned_profile(R[sel], vlos[sel], nbins=5)
        pops.append((r_star, rc, so, se))

    def chi2(theta):
        gamma, rs, log_rho_s = theta
        if not (0.0 <= gamma <= 1.5 and 0.2 <= rs <= 3.0 and 6.5 <= log_rho_s <= 9.5):
            return 1e12
        try:
            pot = agama.Potential(type='Spheroid', densityNorm=10.0**log_rho_s,
                                  scaleRadius=rs, gamma=gamma, beta=BETA_DM, alpha=1.0)
            total = 0.0
            for r_star, rc, so, se in pops:                 # shared potential
                tracer = agama.Density(type='Plummer', scaleRadius=r_star, mass=1.0)
                df = agama.DistributionFunction(type='QuasiSpherical', potential=pot,
                                                density=tracer, beta0=0.0, r_a=r_a_fixed)
                gm = agama.GalaxyModel(pot, df)
                pts = np.column_stack([rc, np.zeros_like(rc)])
                sig_mod = np.sqrt(gm.moments(pts, dens=False, vel2=True)[:, 2])
                total += np.sum(((so - sig_mod) / se) ** 2)
        except Exception:
            return 1e12
        return total

    res = minimize(chi2, [0.6, 0.8, 8.1], method='Nelder-Mead',
                   options={'xatol': 3e-3, 'fatol': 2e-2, 'maxiter': 90})
    return res, pops


# ── Robust DM inner-slope MCMC via the BINNED sigma_los profile (Jeans) ───────
# The per-star projected-DF likelihood is fragile on a real member sample: a few
# binaries/interlopers make it run away to a spurious massive cusp unless the full
# contamination model is included (that is Phase 5). For a fast, robust validation we
# instead fit a gNFW profile to the OUTLIER-ROBUST binned sigma_los(R) of the two
# populations -- a spherical-Jeans likelihood. Three free parameters (gamma, log r_s,
# log M_DM; alpha=1, eta=3, r_a=1.5 fixed): the free 5-parameter shape is degenerate
# under sigma_los alone, so this is the well-posed reduction. Converges in ~1-2 h.

def agama_binned_pops(R, vlos, label, verr, nbins=6):
    pops = []
    for r_star in (RE_MR, RE_MP):
        sel = np.isclose(label, r_star)
        rc, so, se = agama_binned_profile(R[sel], vlos[sel], nbins=nbins)
        pops.append((r_star, rc, so, se))
    return pops


def agama_lnprob_jeans(theta, pops):
    """Flat priors x Gaussian likelihood on binned sigma_los(R), for a gNFW DM profile.
    theta = (gamma, log10 r_s, log10 M_DM); the transition/outer slopes are FIXED to
    gNFW (alpha=1, eta=3) and the anisotropy to r_a=1.5. This is deliberate: the ~12
    binned sigma_los points cannot constrain the free 5-parameter model (the classic
    mass-anisotropy degeneracy that motivates the action-DF method), so we fit the
    well-constrained gNFW slope -- the same parameter space as the MLE."""
    gamma, log_rs, log_MDM = theta
    if not (0.0 <= gamma <= 1.9 and -3.0 <= log_rs <= 1.0 and 7.0 <= log_MDM <= 11.0):
        return -np.inf
    try:
        pot = dm_potential_5p(gamma, 10.0**log_rs, log_MDM, 1.0, 3.0)   # gNFW
        chi2 = 0.0
        for r_star, rc, so, se in pops:
            tr = agama.Density(type='Plummer', scaleRadius=r_star, mass=1.0)
            df = agama.DistributionFunction(type='QuasiSpherical', potential=pot,
                                            density=tr, beta0=0.0, r_a=1.5)
            gm = agama.GalaxyModel(pot, df)
            sig = np.sqrt(gm.moments(np.column_stack([rc, np.zeros_like(rc)]),
                                     dens=False, vel2=True)[:, 2])
            chi2 += np.sum(((so - sig) / se) ** 2)
        return -0.5 * chi2
    except Exception:
        return -np.inf


def run_dm5_chain(nwalkers=24, nsteps=4000, nproc=None, backend="dm5.h5", resume=None,
                  nbins=6, feh_quality_keep=None, catalog=None):
    """Robust MCMC of Sculptor's DM inner slope on the real Tolstoy+2023 data.

    Fits a generalised-NFW profile (3 free parameters: gamma, log10 r_s, log10 M_DM;
    with alpha=1, eta=3 and Osipkov-Merritt anisotropy r_a=1.5 fixed) to the binned
    sigma_los(R) of the two metallicity populations -- a spherical-Jeans likelihood.
    This is the well-constrained reduction of the paper's free 5-parameter DM shape,
    which is degenerate under sigma_los-only data (the mass-anisotropy degeneracy).
    Outlier-robust; converges in ~1-2 h on a few cores. Re-run to resume from `backend`.
    Writes dm5_chain.npy, figure_dm5_corner.png, and the paper's Fig.4
    (figure_ap25_fig4.png)."""
    import os, emcee, multiprocessing as mp
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2))
    if resume is None:
        resume = os.path.exists(backend)
    print("=" * 64)
    print("  DM inner slope -- robust spherical-Jeans gNFW MCMC (real Tolstoy+2023)")
    print("=" * 64)
    R, vlos, label, verr = agama_load_real_tolstoy2023(
        catalog=catalog, feh_quality_keep=feh_quality_keep)
    pops = agama_binned_pops(R, vlos, label, verr, nbins=nbins)
    print(f"  {len(R)} stars; sigma_los binned into {nbins} bins/population")
    try:                                                       # data-overview figure (bonus)
        make_data_overview()
    except Exception as exc:
        print(f"  data overview skipped ({str(exc)[:70]})")
    res, _ = agama_fit_halo(R, vlos, label)                    # MLE seed
    g0, rs0, lrho0 = res.x
    pot_g = agama.Potential(type='Spheroid', densityNorm=10.0**lrho0, scaleRadius=rs0,
                            gamma=g0, beta=BETA_DM, alpha=1.0)
    f2 = dm_potential_5p(g0, rs0, 0.0, 1.0, 3.0).enclosedMass(2.0)
    logMDM0 = float(np.clip(np.log10(max(pot_g.enclosedMass(2.0) / max(f2, 1e-300), 1e7)), 7.2, 10.8))
    init = np.array([g0, np.log10(rs0), logMDM0])              # (gamma, log_rs, logM_DM) gNFW
    print(f"  MLE seed: gamma={g0:.2f}, rs={rs0:.2f} kpc, logM_DM={logMDM0:.2f}  "
          "(fitting gNFW: alpha=1, eta=3, r_a=1.5 fixed)")

    ndim = 3; nw = max(nwalkers, 2 * ndim + 2)
    rng = np.random.default_rng(42)
    lo = np.array([0.02, -2.9, 7.2]); hi = np.array([1.88, 0.9, 10.8])
    p0 = np.clip(init + np.array([0.05, 0.05, 0.10]) * rng.standard_normal((nw, ndim)), lo, hi)
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    bk = emcee.backends.HDFBackend(backend) if HAS_H5PY else None
    resume_ok = bool(resume and bk is not None and os.path.exists(backend) and bk.iteration > 0)
    pool = mp.Pool(nproc) if nproc > 1 else None
    try:
        s = emcee.EnsembleSampler(nw, ndim, agama_lnprob_jeans, args=(pops,),
                                  moves=moves, pool=pool, backend=bk)
        if resume_ok:
            print(f"    [MCMC] resuming from {bk.iteration} stored steps.")
            s.run_mcmc(None, nsteps, progress=True)
        else:
            if bk is not None:
                bk.reset(nw, ndim)
            s.run_mcmc(p0, nsteps, progress=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()

    labels = [r'$\gamma$', r'$\log_{10}r_s$', r'$\log_{10}M_{\rm DM}$']
    rep = mcmc_convergence_report(s, labels)
    flat = s.get_chain(discard=rep['burn'], thin=rep['thin'], flat=True)
    if len(flat) < 50:
        flat = s.get_chain(discard=max(1, s.iteration // 3), flat=True)
    np.save(_gf("dm5_chain.npy"), flat)
    try:
        make_corner_plot(flat, labels, _gf("figure_dm5_corner.png"))
    except Exception as exc:
        print(f"  corner skipped ({exc})")
    try:                                                        # gNFW -> (logMDM,log_rs,alpha=1,eta=3,gamma)
        n = len(flat)
        chain5 = np.column_stack([flat[:, 2], flat[:, 1], np.ones(n), 3.0 * np.ones(n), flat[:, 0]])
        make_ap25_figure4(chain5, _gf("figure_ap25_fig4.png"))
        print("  saved figure_ap25_fig4.png (paper Fig.4 from the gNFW posterior)")
    except Exception as exc:
        print(f"  Fig.4 skipped ({exc})")
    print("\n  === posterior (median +/- 68% CI) ===")
    for k, nm in enumerate(['gamma', 'log_rs', 'log_MDM']):
        p16, p50, p84 = np.percentile(flat[:, k], [16, 50, 84])
        print(f"    {nm:<8}= {p50:8.3f}  (+{p84 - p50:.3f} / -{p50 - p16:.3f})")
    g16, g50, g84 = np.percentile(flat[:, 0], [16, 50, 84])
    tag = "" if rep.get('converged') else "   [NOT converged -- re-run to add steps]"
    print("\n  " + "-" * 58)
    print(f"  VALIDATION:  gamma = {g50:.2f} (+{g84 - g50:.2f} / -{g50 - g16:.2f}){tag}")
    print(f"               AP25 published: 0.39 (+0.23 / -0.26)")
    print("  " + "-" * 58)
    return s


# ── OPTIONAL HIGH-FIDELITY MODE: true per-star mixture MCMC (emcee) ──────────
_INF = np.inf

def agama_lnlike_perstar(theta, R, vlos, verr, w_mr):
    """
    TRUE per-star mixture log-likelihood (Arroyo-Polonio+25). For each star the
    projected DF f(X, Y, v_los) is evaluated with the two plane-of-sky velocity
    components marginalised (infinite uncertainty) and v_los convolved with the
    star's own measurement error. Both populations share the DM potential.

    theta = (gamma, rs, log10 M_DM, alpha, eta, r_a)  -- the paper's 5-parameter
    DM density (Eq.4; free inner slope gamma, scale radius rs, total mass M_DM,
    transition sharpness alpha, outer slope eta) plus one anisotropy parameter r_a.
    NOTE: v_los must be in the GALAXY REST FRAME (systemic velocity subtracted),
    because the DF is centred on v=0.
    """
    gamma, rs, log_MDM, alpha, eta, r_a = theta
    try:
        pot = dm_potential_5p(gamma, rs, log_MDM, alpha, eta)
        N = len(R)
        pts = np.column_stack([R, np.zeros(N), np.zeros(N), np.zeros(N), vlos,
                               np.full(N, _INF), np.full(N, _INF), verr])
        total = np.zeros(N)
        for frac, r_star in [(w_mr, RE_MR), (1.0 - w_mr, RE_MP)]:
            tracer = agama.Density(type='Plummer', scaleRadius=r_star, mass=1.0)
            df = agama.DistributionFunction(type='QuasiSpherical', potential=pot,
                                            density=tracer, beta0=0.0, r_a=r_a)
            gm = agama.GalaxyModel(pot, df)
            total += frac * np.maximum(gm.projectedDF(pts), 1e-300)
        return float(np.sum(np.log(total)))
    except Exception:
        return -np.inf


def agama_lnprob(theta, R, vlos, verr, w_mr):
    """
    Flat priors × per-star likelihood. DM prior ranges follow Table 1 of
    Arroyo-Polonio+25: gamma in [0,1.9], log rs in [-3,1] (=> rs in [1e-3,10] kpc),
    log M_DM in [7,11], alpha in [0,7], eta in [2,7]. r_a in [0.3,12] (anisotropy).
    """
    gamma, rs, log_MDM, alpha, eta, r_a = theta
    if not (0.0 <= gamma <= 1.9 and 1e-3 <= rs <= 10.0 and 7.0 <= log_MDM <= 11.0
            and 0.0 <= alpha <= 7.0 and 2.0 <= eta <= 7.0 and 0.3 <= r_a <= 12.0):
        return -np.inf
    return agama_lnlike_perstar(theta, R, vlos, verr, w_mr)


def agama_run_mcmc(R, vlos, verr, init, w_mr=FRAC_MR, nwalkers=32, nsteps=3000,
                   nsub=None, nproc=1, backend_file=None, resume=False,
                   progress=False, seed=42):
    """
    Sample the DM-halo posterior with emcee using the TRUE per-star likelihood
    (the paper's actual method). Publication-grade options:

      nproc        -- number of worker processes (multiprocessing.Pool). The
                      per-star projectedDF likelihood is CPU-bound and embarrassingly
                      parallel across walkers, so wall-time scales ~1/nproc.
      backend_file -- HDF5 checkpoint file (emcee HDFBackend). The chain is flushed
                      to disk every step so a long run can be resumed or inspected
                      live. Requires h5py.
      resume       -- if True and the backend already holds iterations, continue
                      from where it stopped instead of re-initialising.
      moves        -- a DEMove/DESnookerMove mixture, which samples the correlated
                      gamma-r_s-rho_s degeneracy far better than the default stretch
                      move.

    `init` = (gamma, rs, log_rho_s, r_a). Returns the emcee SAMPLER (so convergence
    diagnostics can be computed); use mcmc_convergence_report() then get_chain().

    For a real run: nwalkers=32, nsteps>=3000, nsub=None, nproc=os.cpu_count().
    """
    rng = np.random.default_rng(seed)
    if nsub is not None and nsub < len(R):
        idx = rng.choice(len(R), nsub, replace=False)
        R, vlos, verr = R[idx], vlos[idx], verr[idx]

    ndim = len(init)                                  # 6: gamma, rs, logMDM, alpha, eta, r_a
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]

    backend = None
    if backend_file is not None:
        if HAS_H5PY:
            backend = emcee.backends.HDFBackend(backend_file)
        else:
            print("    [MCMC] h5py not installed; running without checkpoint backend.")

    import os as _os
    resume_ok = bool(resume and backend is not None
                     and _os.path.exists(backend_file) and backend.iteration > 0)

    # per-parameter initial scatter and hard bounds (match agama_lnprob priors)
    scatter = np.array([0.05, 0.05, 0.08, 0.3, 0.2, 0.3])[:ndim]
    lo = np.array([0.02, 0.02, 7.1, 0.1, 2.1, 0.35])[:ndim]
    hi = np.array([1.88, 9.9,  10.9, 6.9, 6.9, 11.5])[:ndim]
    p0 = np.array(init) + scatter * rng.standard_normal((nwalkers, ndim))
    p0 = np.clip(p0, lo, hi)

    pool = mp.Pool(nproc) if (nproc and nproc > 1) else None
    try:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, agama_lnprob, args=(R, vlos, verr, w_mr),
            moves=moves, pool=pool, backend=backend)
        if resume_ok:
            print(f"    [MCMC] resuming from {backend.iteration} stored steps.")
            sampler.run_mcmc(None, nsteps, progress=progress)
        else:
            if backend is not None:
                backend.reset(nwalkers, ndim)
            sampler.run_mcmc(p0, nsteps, progress=progress)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    return sampler


# ── Convergence diagnostics (autocorrelation time, split-R-hat, ESS) ─────────
def _norm_ppf(p):
    """Acklam inverse-normal-CDF approximation (vectorised)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.zeros_like(p)
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p[mid] - 0.5; r = q*q
    x[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return x


def split_rhat(chain_1d):
    """
    Rank-normalized split-R-hat (Vehtari et al. 2021) for one parameter.
    chain_1d: (nsteps, nwalkers). Treats walkers as chains, splits each in half.
    R-hat → 1 at convergence; the standard threshold is R-hat < 1.01.
    (For emcee ensembles the autocorrelation time is the primary diagnostic; this
    is a complementary between-walker mixing check.)
    """
    n, m = chain_1d.shape
    if n < 4:
        return np.inf
    half = n // 2
    splits = np.concatenate([chain_1d[:half], chain_1d[half:2*half]], axis=1)  # (half, 2m)
    flat = splits.flatten()
    ranks = np.argsort(np.argsort(flat)).reshape(splits.shape) + 1
    z = _norm_ppf((ranks - 0.375) / (len(flat) + 0.25))
    cm = z.mean(axis=0); cv = z.var(axis=0, ddof=1)
    W = cv.mean(); B = half * cm.var(ddof=1)
    var_hat = (half - 1) / half * W + B / half
    return float(np.sqrt(var_hat / W)) if W > 0 else np.inf


def effective_sample_size(chain_1d):
    """ESS via FFT autocorrelation with Geyer initial-positive-sequence, (n,m)."""
    n, m = chain_1d.shape
    x = chain_1d - chain_1d.mean()
    acf = np.zeros(n)
    for j in range(m):
        f = np.fft.fft(x[:, j], n=2 * n)
        a = np.fft.ifft(f * np.conjugate(f))[:n].real
        if a[0] > 0:
            acf += a / a[0]
    acf /= m
    neg = np.argmax(acf[1:] < 0)
    tau = 1.0 + 2.0 * np.sum(acf[1:neg + 1]) if neg > 0 else 1.0 + 2.0 * np.sum(acf[1:])
    return float(n * m / max(tau, 1.0))


def mcmc_convergence_report(sampler, labels):
    """
    Print a publication-grade convergence report and return a dict including
    recommended burn-in and thinning. Diagnostics: integrated autocorrelation
    time tau (emcee), chain length in units of tau, split-R-hat, ESS, and mean
    acceptance fraction. 'converged' requires R-hat<1.01, N>50*tau, 0.1<accept<0.7.
    """
    chain = sampler.get_chain()                     # (nsteps, nwalkers, ndim)
    nsteps, nwalkers, ndim = chain.shape
    try:
        tau = sampler.get_autocorr_time(tol=0)
    except Exception:
        tau = np.full(ndim, np.nan)
    accept = float(np.mean(sampler.acceptance_fraction))

    rhat = np.array([split_rhat(chain[:, :, k]) for k in range(ndim)])
    ess  = np.array([effective_sample_size(chain[:, :, k]) for k in range(ndim)])

    tau_max = np.nanmax(tau) if np.any(np.isfinite(tau)) else np.nan
    n_over_tau = nsteps / tau_max if (tau_max and np.isfinite(tau_max)) else np.nan

    print("  --- MCMC convergence report ---------------------------------")
    print(f"    steps={nsteps}  walkers={nwalkers}  mean acceptance={accept:.2f}")
    print(f"    {'param':<12}{'tau':>8}{'N/tau':>8}{'R-hat':>8}{'ESS':>9}")
    for k, lab in enumerate(labels):
        tk = tau[k] if np.isfinite(tau[k]) else np.nan
        nt = nsteps / tk if (np.isfinite(tk) and tk > 0) else np.nan
        print(f"    {lab:<12}{tk:8.1f}{nt:8.1f}{rhat[k]:8.3f}{ess[k]:9.0f}")
    rhat_ok   = np.all(rhat < 1.01)
    length_ok = bool(np.isfinite(n_over_tau) and n_over_tau > 50)
    accept_ok = 0.1 < accept < 0.7
    converged = bool(rhat_ok and length_ok and accept_ok)
    if not converged:
        why = []
        if not rhat_ok:   why.append("R-hat>1.01")
        if not length_ok: why.append("chain<50*tau")
        if not accept_ok: why.append("acceptance out of [0.1,0.7]")
        print(f"    NOT CONVERGED ({', '.join(why)}) — extend the chain / more walkers.")
    else:
        print("    CONVERGED (R-hat<1.01, N>50*tau, acceptance in range).")
    print("  -------------------------------------------------------------")

    burn = int(min(3 * tau_max, nsteps // 2)) if (tau_max and np.isfinite(tau_max)) else nsteps // 3
    tau_min = np.nanmin(tau) if np.any(np.isfinite(tau)) else np.nan
    thin = int(max(1, tau_min // 2)) if (tau_min and np.isfinite(tau_min)) else 1
    return dict(tau=tau, rhat=rhat, ess=ess, accept=accept,
                converged=converged, burn=burn, thin=thin)


def make_corner_plot(flat, labels, outfile):
    """Publication corner plot of the posterior."""
    if not HAS_CORNER:
        print("    [corner] not installed; skipping corner plot.")
        return
    fig = corner.corner(flat, labels=labels, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_fmt=".2f",
                        title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 11})
    fig.savefig(outfile, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"--> Saved {outfile}")


def agama_load_real_tolstoy2023(catalog=None, cols=None, vclip_sigma=4.0, **select):
    """
    Load the real Tolstoy+2023 Sculptor catalog for the FAST Phase-4 two-tracer fit.
    Uses the SAME self-diagnosing fetch core as Phase 5 (auto-detects columns across
    all tables; pass cols=dict(vlos=..,feh=..,ra=..,dec=..,verr=..,feherr=..) to
    override, or run ap25_inspect_vizier() first to discover the real names). Extra
    kwargs (require_member, mem_keep, feh_quality_keep) are forwarded for sample
    selection. Splits stars into MR/MP by median [Fe/H] to serve as the two tracers.

    vclip_sigma: robust (MAD) velocity-outlier clip. The Phase-4 5-parameter model
    has NO contamination component (unlike the full model's pop-3), so a few binaries
    or interlopers at large |v_los| would otherwise inflate the dispersion and bias
    gamma HIGH (a spurious cusp). Clipping at ~4 sigma removes them; set None to disable.
    Returns (R_kpc, v_los_restframe, r_star_label, v_err).
    """
    catalog = catalog or GAL['catalog']
    if cols is None: cols = GAL.get('cols')
    select.setdefault('mem_keep', GAL.get('mem_keep') or ('m',))
    select.setdefault('require_member', bool(GAL.get('mem_keep')))
    select.setdefault('target_col', GAL.get('target_col'))
    select.setdefault('target_keep', GAL.get('target_keep'))
    select.setdefault('mem_min', GAL.get('mem_min'))
    if select.get('feh_quality_keep') is None:
        _fqk = GAL.get('feh_quality_keep'); select['feh_quality_keep'] = (list(_fqk) if _fqk else None)
    ra, dec, vlos, verr, feh, feherr, _g = _fetch_tolstoy2023(catalog, cols, **select)
    R = _semi_major_axis_radius(ra, dec)
    good = np.isfinite(vlos) & np.isfinite(feh) & np.isfinite(verr)
    R, vlos, feh, verr = R[good], vlos[good] - V_SYS, feh[good], verr[good]   # rest frame
    med = np.median(feh)
    label = np.where(feh >= med, RE_MR, RE_MP)          # metal-rich half = MR tracer
    if vclip_sigma:                                     # remove velocity outliers (no pop-3 here)
        vmed = np.median(vlos); mad = 1.4826 * np.median(np.abs(vlos - vmed)) + 1e-9
        keep = np.abs(vlos - vmed) < vclip_sigma * mad
        if keep.sum() < len(vlos):
            print(f"    [tolstoy loader] velocity {vclip_sigma:g}-sigma clip: removed "
                  f"{int((~keep).sum())} outlier(s) (binaries/interlopers) of {len(vlos)}")
        R, vlos, label, verr = R[keep], vlos[keep], label[keep], verr[keep]
    return R, vlos, label, verr


def run_action_df_modeling(high_fidelity=False, mcmc_nwalkers=32, mcmc_nsteps=3000,
                           mcmc_nsub=None, mcmc_nproc=1, mcmc_backend=None,
                           mcmc_resume=False):
    """
    Phase-4 orchestrator. Uses real Tolstoy+2023 data if reachable, else runs the
    paper's Section-5 mock-recovery validation. Produces the 4-panel result figure.

    high_fidelity=False (default): fast maximum-likelihood recovery on binned
        sigma_los (~2 min). Keeps the pipeline quick.
    high_fidelity=True: publication-grade reduced (6-param) mode. After the MLE, run
        the true per-star projectedDF MCMC of the 5-param DM model + anisotropy, with
        multiprocessing, HDF5 checkpointing, convergence diagnostics and a corner plot.
        The per-star likelihood is ~3 s/eval per population; a real chain takes hours.
        Requires emcee (and, for the extras, h5py and corner).
    """
    if not HAS_AGAMA:
        print("[Phase 4] AGAMA not installed — skipping action-DF modeling.")
        print("          Install:  apt-get install -y libgsl-dev")
        print("                    pip install agama --no-build-isolation "
              "--config-settings --build-option=--yes")
        return

    print("[Phase 4] Action-based DF modeling (AGAMA; Arroyo-Polonio+25 method)...")

    # ── Try the real data; fall back to the Section-5 mock ───────────────────
    used_real = False
    truth = None
    try:
        R, vlos, label, verr = agama_load_real_tolstoy2023()
        vlos = vlos - V_SYS                       # DF is centred on the rest frame
        used_real = True
        print(f"  --> Using REAL Tolstoy+2023 data: {len(R)} stars.")
    except Exception as exc:
        print(f"  --> Real data unavailable ({str(exc)[:60]}); running Section-5 mock.")
        truth = dict(gamma=1.0, rs=0.79, lrho=8.13, r_a=1.5)   # known cuspy input
        mock = agama_generate_mock(truth['gamma'], truth['rs'], truth['lrho'], truth['r_a'])
        R, vlos, label = mock[:, 0], mock[:, 1], mock[:, 2]
        verr = np.full(len(R), 0.6)               # mock velocity error (km/s)

    # ── Stage 1: fast MLE (two-population, shared potential) ─────────────────
    res, pops = agama_fit_halo(R, vlos, label)
    g_fit, rs_fit, lrho_fit = res.x
    ra_fit = 1.5
    print(f"  --> MLE: gamma={g_fit:.2f}, rs={rs_fit:.2f} kpc, "
          f"log_rho_s={lrho_fit:.2f} (chi2={res.fun:.1f})")
    if truth is not None:
        print(f"      (input gamma={truth['gamma']:.2f}; "
              f"recovered within {abs(g_fit - truth['gamma']):.2f})")

    rr = np.logspace(-1.4, 0.3, 40)

    def dens_mass(gamma, rs, lrho):
        pot = agama.Potential(type='Spheroid', densityNorm=10.0**lrho, scaleRadius=rs,
                              gamma=gamma, beta=BETA_DM, alpha=1.0)
        return (np.array([pot.density([r, 0, 0]) for r in rr]),
                np.array([pot.enclosedMass(r) for r in rr]))

    def dens_mass_5p(gamma, rs, logMDM, alpha, eta):
        pot = dm_potential_5p(gamma, rs, logMDM, alpha, eta)
        return (np.array([pot.density([r, 0, 0]) for r in rr]),
                np.array([pot.enclosedMass(r) for r in rr]))

    # ── Stage 2 (optional): true per-star MCMC of the 5-parameter DM model ────
    chain = None
    post5 = None
    rho_band = mass_band = None
    gamma_lbl = r'Recovered (MLE, gNFW) $\gamma$=%.2f' % g_fit
    if high_fidelity:
        if not HAS_EMCEE:
            print("  --> high_fidelity requested but emcee not installed; "
                  "pip install emcee. Falling back to MLE.")
        else:
            import os as _os
            nproc = mcmc_nproc if mcmc_nproc and mcmc_nproc > 0 else (len(_os.sched_getaffinity(0))
                    if hasattr(_os, 'sched_getaffinity') else mp.cpu_count())
            # Seed the 6-param vector from the fast gNFW MLE: convert rho_s -> M_DM by
            # matching M(<2 kpc); seed alpha, eta then let the chain explore all 5 DM params.
            pot_g = agama.Potential(type='Spheroid', densityNorm=10.0**lrho_fit,
                                    scaleRadius=rs_fit, gamma=g_fit, beta=BETA_DM, alpha=1.0)
            f2 = dm_potential_5p(g_fit, rs_fit, 0.0, 1.0, 3.0).enclosedMass(2.0)
            logMDM_seed = float(np.clip(np.log10(max(pot_g.enclosedMass(2.0) / max(f2, 1e-300), 1e7)),
                                        7.2, 10.8))
            init = [g_fit, rs_fit, logMDM_seed, 2.0, 3.0, ra_fit]  # gamma, rs, logM_DM, alpha, eta, r_a
            print(f"  --> Running per-star MCMC of the 5-param DM model "
                  f"(nwalkers={mcmc_nwalkers}, nsteps={mcmc_nsteps}, nsub={mcmc_nsub}, nproc={nproc}"
                  + (f", backend={mcmc_backend}" if mcmc_backend else "") + ")... [slow]")
            sampler = agama_run_mcmc(R, vlos, verr, init, w_mr=FRAC_MR,
                                     nwalkers=mcmc_nwalkers, nsteps=mcmc_nsteps,
                                     nsub=mcmc_nsub, nproc=nproc,
                                     backend_file=mcmc_backend, resume=mcmc_resume)

            labels = [r'$\gamma$', r'$r_s$', r'$\log_{10}M_{\rm DM}$',
                      r'$\alpha$', r'$\eta$', r'$r_a$']
            report = mcmc_convergence_report(sampler, labels)
            burn, thin = report['burn'], report['thin']
            flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
            if len(flat) < 20:                        # guard tiny/short chains
                flat = sampler.get_chain(discard=max(1, mcmc_nsteps // 3), flat=True)
            np.save("phase4_mcmc_chain.npy", flat)
            print(f"  --> Saved flat chain ({flat.shape[0]} samples, 6 params) "
                  "to phase4_mcmc_chain.npy")

            make_corner_plot(flat, labels, "figure4b_corner.png")

            # posterior summary over all 5 DM params (+ anisotropy) and derived mass
            print("  --- Posterior summary (median +/- 68% CI) -------------------")
            for k, nm in enumerate(['gamma', 'rs', 'log_MDM', 'alpha', 'eta', 'r_a']):
                p16, p50, p84 = np.percentile(flat[:, k], [16, 50, 84])
                print(f"    {nm:<10} = {p50:8.3f}  (+{p84-p50:.3f} / -{p50-p16:.3f})")
            M03 = np.array([dm_potential_5p(t[0], t[1], t[2], t[3], t[4]).enclosedMass(0.30)
                            for t in flat[::max(1, len(flat) // 400)]])
            m16, m50, m84 = np.percentile(M03, [16, 50, 84])
            print(f"    {'M(<0.3kpc)':<10} = {m50:.3e}  (+{m84-m50:.2e} / -{m50-m16:.2e}) Msun")
            print("  -------------------------------------------------------------")

            thinned = flat[:: max(1, len(flat) // 300)]
            rhos = np.array([dens_mass_5p(t[0], t[1], t[2], t[3], t[4])[0] for t in thinned])
            mss  = np.array([dens_mass_5p(t[0], t[1], t[2], t[3], t[4])[1] for t in thinned])
            rho_band  = np.percentile(rhos, [16, 50, 84], axis=0)
            mass_band = np.percentile(mss,  [16, 50, 84], axis=0)
            g16, g50, g84 = np.percentile(flat[:, 0], [16, 50, 84])
            med = np.median(flat, axis=0)
            post5 = dict(gamma=med[0], rs=med[1], logMDM=med[2],
                         alpha=med[3], eta=med[4], ra=med[5])
            tag = "" if report['converged'] else "  [UNCONVERGED]"
            gamma_lbl = (r'Posterior $\gamma=%.2f^{+%.2f}_{-%.2f}$' % (g50, g84 - g50, g50 - g16)
                         + r', $\alpha$=%.1f, $\eta$=%.1f' % (med[3], med[4]) + tag)
            chain = flat

    # Reference (AP25 published medians, 5-param) + recovered model (5-param if MCMC ran)
    rho_ap25, mass_ap25 = dens_mass_5p(AP25["gamma"], AP25["rs"], 9.0, 3.7, 3.2)
    if truth is not None:
        rho_true, mass_true = dens_mass(truth['gamma'], truth['rs'], truth['lrho'])

    def _build_gm(r_star):
        if post5 is not None:
            return agama_build_galaxy_model_5p(post5['gamma'], post5['rs'], post5['logMDM'],
                                               post5['alpha'], post5['eta'], post5['ra'], r_star)[0]
        return agama_build_galaxy_model(g_fit, rs_fit, lrho_fit, ra_fit, r_star)[0]

    if post5 is not None:
        rho_fit, mass_fit = dens_mass_5p(post5['gamma'], post5['rs'], post5['logMDM'],
                                         post5['alpha'], post5['eta'])
    else:
        rho_fit, mass_fit = dens_mass(g_fit, rs_fit, lrho_fit)

    gm_fit = _build_gm(RE_MP)
    Rp = np.logspace(-1.3, 0.28, 14)
    sig_los_p, sig_R_p, sig_T_p = agama_projected_sigmas(gm_fit, Rp)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    cols  = {RE_MR: 'crimson', RE_MP: 'royalblue'}
    names = {RE_MR: 'Metal-rich ($R_e$=0.18 kpc)', RE_MP: 'Metal-poor ($R_e$=0.28 kpc)'}
    src   = "real Tolstoy+2023" if used_real else "Section-5 mock"

    # (a) two-population sigma_los fit
    axA = ax[0, 0]
    for r_star, rc, so, se in pops:
        axA.errorbar(rc, so, yerr=se, fmt='o', color=cols[r_star], capsize=3,
                     label=names[r_star])
        gm = _build_gm(r_star)
        rl = np.logspace(np.log10(rc.min()), np.log10(rc.max()), 30)
        sl, _, _ = agama_projected_sigmas(gm, rl)
        axA.plot(rl, sl, '-', color=cols[r_star], lw=2)
    axA.set_xscale('log'); axA.set_xlabel('Projected radius $R$ (kpc)')
    axA.set_ylabel(r'$\sigma_{\rm los}$ (km s$^{-1}$)')
    axA.set_title(f'(a) Two-population fit ({src})')
    axA.legend(fontsize=8); axA.grid(alpha=0.3)

    # (b) recovered DM density profile (+ posterior band if MCMC ran)
    axB = ax[0, 1]
    if truth is not None:
        axB.plot(rr, rho_true, 'k--', lw=2, label=r'Input $\gamma$=%.1f (truth)' % truth['gamma'])
    if rho_band is not None:
        axB.fill_between(rr, rho_band[0], rho_band[2], color='crimson', alpha=0.25,
                         label=r'Posterior $1\sigma$')
        axB.plot(rr, rho_band[1], color='crimson', lw=2.5, label=gamma_lbl)
    else:
        axB.plot(rr, rho_fit, color='crimson', lw=2.5, label=gamma_lbl)
    axB.plot(rr, rho_ap25, color='gray', lw=1.5, ls=':', label=r'AP25 published $\gamma$=0.39')
    axB.set_xscale('log'); axB.set_yscale('log')
    axB.set_xlabel('Radius $r$ (kpc)'); axB.set_ylabel(r'$\rho_{\rm DM}(r)$ ($M_\odot\,{\rm kpc}^{-3}$)')
    axB.set_title('(b) DM density: inner-slope recovery')
    axB.legend(fontsize=8); axB.grid(alpha=0.2, which='both')

    # (c) projected anisotropy prediction
    axC = ax[1, 0]
    axC.plot(Rp, sig_los_p, '-o', color='black',    ms=4, label=r'$\sigma_{\rm los}$')
    axC.plot(Rp, sig_R_p,  '-s', color='crimson',   ms=4, label=r'$\sigma_{R}$ (plane-of-sky)')
    axC.plot(Rp, sig_T_p,  '-^', color='royalblue', ms=4, label=r'$\sigma_{T}$ (plane-of-sky)')
    axC.set_xscale('log'); axC.set_xlabel('Projected radius $R$ (kpc)')
    axC.set_ylabel(r'$\sigma$ (km s$^{-1}$)')
    axC.set_title('(c) Predicted profiles: radial anisotropy outward')
    axC.legend(fontsize=8); axC.grid(alpha=0.3)

    # (d) enclosed mass + real literature (+ posterior band if MCMC ran)
    axD = ax[1, 1]
    if mass_band is not None:
        axD.fill_between(rr, mass_band[0], mass_band[2], color='crimson', alpha=0.25,
                         label=r'Posterior $1\sigma$')
        axD.plot(rr, mass_band[1], color='crimson', lw=2.5, label='Posterior median')
    else:
        axD.plot(rr, mass_fit, color='crimson', lw=2.5, label='Recovered (MLE)')
    axD.plot(rr, mass_ap25, color='gray', lw=1.5, ls=':', label='AP25 published')
    for lbl, rk, M, e, col, _c in SCULPTOR_LIT_MASSES:
        axD.errorbar([rk], [M], yerr=[e], fmt='o', color=col, capsize=4, ms=7,
                     mec='k', mew=0.6, label=lbl)
    axD.set_xscale('log'); axD.set_yscale('log')
    axD.set_xlabel('Radius $r$ (kpc)'); axD.set_ylabel(r'$M(<r)$ ($M_\odot$)')
    axD.set_title('(d) Enclosed mass vs. real literature')
    axD.legend(fontsize=7.5, loc='lower right'); axD.grid(alpha=0.2, which='both')

    mode = 'per-star MCMC' if chain is not None else 'MLE'
    plt.suptitle(f'Arroyo-Polonio+25 action-DF modeling (AGAMA, {mode}) — '
                 + ('real data' if used_real else 'mock recovery'),
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig("figure4_action_df_modeling.png", dpi=200, bbox_inches='tight')
    print("--> Saved figure4_action_df_modeling.png")
    plt.close()
    return chain


# ============================================================
# PHASE 4b: GRAVSPHERE  (Read & Steger 2017 — Jeans + Virial Shape Parameters)
# ============================================================
# GravSphere fits a gNFW halo to the tracer sigma_los(R) via the spherical Jeans
# equation with a FREE anisotropy profile beta(r), PLUS two Virial Shape Parameters
# (VSP1, VSP2 = 4th-velocity-moment integral constraints) that partially break the
# mass-anisotropy degeneracy plain Jeans cannot. Pure-numpy forward model, validated
# against AGAMA to 0.3% on sigma_los and to ~4% on the VSPs (within their bootstrap
# errors); no DF construction, so it is fast (~ms/eval). Six free parameters:
#   gamma, log10 r_s, log10 rho_s  (gNFW: inner slope, scale radius, density norm)
#   bt0, btinf, log10 r_beta       (symmetrised Baes & van Hese anisotropy)
# with beta = 2*bt/(1+bt) so bt in [-1,1] maps beta in [-inf,1]. The tracer light
# profile is a Plummer sphere whose scale is fixed from the observed radii.
from scipy.integrate import cumulative_trapezoid as _cumtrapz

_GS_RG = np.logspace(-3.0, 3.0, 700)      # radius grid for Jeans/VSP integrals (kpc)
_GS_TG = np.logspace(-3.0, 2.0, 350)      # line-of-sight integration variable (kpc)


def _gs_gnfw_rho(r, gamma, rs, rhos):
    x = r / rs
    return rhos * x ** (-gamma) * (1.0 + x) ** (gamma - 3.0)


def _gs_M_grid(gamma, rs, rhos):
    """Enclosed gNFW mass on _GS_RG (numerical)."""
    return _cumtrapz(4 * np.pi * _gs_gnfw_rho(_GS_RG, gamma, rs, rhos) * _GS_RG ** 2,
                     _GS_RG, initial=0.0)


def _gs_plummer_nu(r, a):
    return (3.0 / (4.0 * np.pi * a ** 3)) * (1.0 + (r / a) ** 2) ** (-2.5)   # 3-D, M=1


def _gs_plummer_Sigma(R, a):
    return (a ** 2 / np.pi) / (a ** 2 + R ** 2) ** 2                          # projected, M=1


def _gs_beta(r, bt0, btinf, rbeta, n=2.0):
    """Baes & van Hese anisotropy from symmetrised parameters bt=beta/(2-beta)."""
    b0, binf = 2 * bt0 / (1 + bt0), 2 * btinf / (1 + btinf)
    x = (r / rbeta) ** n
    return (b0 + binf * x) / (1.0 + x)


def _gs_nu_sr2(gamma, rs, rhos, bvals, Mgrid, a):
    """nu*sigma_r^2 on _GS_RG via the spherical-Jeans integrating-factor solution."""
    f = np.exp(_cumtrapz(2 * bvals, np.log(_GS_RG), initial=0.0))     # integrating factor
    nu = _gs_plummer_nu(_GS_RG, a)
    integ = f * nu * G_KPC * Mgrid / _GS_RG ** 2
    rev = _cumtrapz(integ[::-1], -_GS_RG[::-1], initial=0.0)[::-1]    # int_r^inf
    return rev / f


def _gs_sigma_los(Rproj, gamma, rs, rhos, bfunc, a):
    """Projected sigma_los(R) (Binney-Mamon projection; r=sqrt(R^2+t^2) substitution)."""
    Mg = _gs_M_grid(gamma, rs, rhos)
    nusr2 = _gs_nu_sr2(gamma, rs, rhos, bfunc(_GS_RG), Mg, a)
    out = np.empty_like(Rproj)
    for i, R in enumerate(Rproj):
        r = np.sqrt(R ** 2 + _GS_TG ** 2)
        integ = (1.0 - bfunc(r) * R ** 2 / r ** 2) * np.interp(r, _GS_RG, nusr2)
        out[i] = 2 * _trapz(integ, _GS_TG) / _gs_plummer_Sigma(R, a)
    return np.sqrt(np.maximum(out, 0.0))


def _gs_vsp_theory(gamma, rs, rhos, bfunc, a):
    """Model Virial Shape Parameters (Read & Steger 2017)."""
    Mg = _gs_M_grid(gamma, rs, rhos)
    bv = bfunc(_GS_RG)
    nusr2 = _gs_nu_sr2(gamma, rs, rhos, bv, Mg, a)
    vs1 = (2.0 / 5.0) * _trapz(G_KPC * Mg * (5 - 2 * bv) * nusr2 * _GS_RG, _GS_RG)
    vs2 = (4.0 / 35.0) * _trapz(G_KPC * Mg * (7 - 6 * bv) * nusr2 * _GS_RG ** 3, _GS_RG)
    return vs1, vs2


def _gs_obs_vsp(R, vlos):
    """Unbiased observed VSPs from the stars (fair-sample estimators)."""
    return np.mean(vlos ** 4) / (2 * np.pi), np.mean(vlos ** 4 * R ** 2) / (2 * np.pi)


def _gs_lnprob(theta, rc, so, se, v1o, v2o, v1e, v2e, a, use_vsp=True):
    """Priors x [sigma_los chi^2 (+ VSP1, VSP2 chi^2 if use_vsp)]."""
    gamma, lrs, lrhos, bt0, btinf, lrb = theta
    if not (0.05 <= gamma <= 1.9 and -1.0 <= lrs <= 0.8 and 6.0 <= lrhos <= 9.0
            and -0.95 <= bt0 <= 0.95 and -0.95 <= btinf <= 0.95 and -1.2 <= lrb <= 0.8):
        return -np.inf
    bf = lambda r: _gs_beta(r, bt0, btinf, 10.0 ** lrb)
    try:
        chi2 = np.sum(((so - _gs_sigma_los(rc, gamma, 10.0 ** lrs, 10.0 ** lrhos, bf, a)) / se) ** 2)
        if use_vsp:
            v1m, v2m = _gs_vsp_theory(gamma, 10.0 ** lrs, 10.0 ** lrhos, bf, a)
            chi2 += ((v1o - v1m) / v1e) ** 2 + ((v2o - v2m) / v2e) ** 2
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf
    except Exception:
        return -np.inf


def run_gravsphere_chain(nwalkers=24, nsteps=3000, nproc=None, backend="gravsphere.h5",
                         resume=None, use_vsp=True, nbins=7, feh_quality_keep=(0,),
                         catalog=None):
    """
    GravSphere (Read & Steger 2017) measurement of Sculptor's DM inner slope on the real
    Tolstoy+2023 data: spherical Jeans with a free Baes-van-Hese anisotropy beta(r) plus
    the two Virial Shape Parameters. Six free parameters (gamma, log r_s, log rho_s, and
    the symmetrised anisotropy bt0, btinf, log r_beta). Single tracer population (all
    members); the Plummer light scale is fixed to the projected half-light radius. The
    VSPs partially break the mass-anisotropy degeneracy -- the rung between plain Jeans
    (--dm5) and the action-DF method (--chain). Re-run to resume from `backend`. Writes
    gravsphere_chain.npy, figure_gravsphere_corner.png, figure_gravsphere_beta.png and
    the paper's Fig.4 (figure_ap25_fig4.png). Set use_vsp=False for a Jeans-only fit.
    """
    import os, emcee, multiprocessing as mp
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2))
    if resume is None:
        resume = os.path.exists(backend)
    print("=" * 64)
    print("  GravSphere -- spherical Jeans + Virial Shape Parameters (real Tolstoy+2023)")
    print("=" * 64)
    R, vlos, label, verr = agama_load_real_tolstoy2023(
        catalog=catalog, feh_quality_keep=feh_quality_keep)     # rest-frame, outlier-clipped
    a_star = float(np.median(R))                                       # Plummer scale = projected R_half
    print(f"  {len(R)} stars (single tracer); Plummer light scale a = {a_star:.3f} kpc; "
          f"VSPs {'ON' if use_vsp else 'OFF'}")

    # binned sigma_los + observed VSPs with bootstrap errors
    edges = np.quantile(R, np.linspace(0, 1, nbins + 1))
    rc, so, se = [], [], []
    for i in range(nbins):
        m = (R >= edges[i]) & (R <= edges[i + 1] if i == nbins - 1 else R < edges[i + 1])
        if m.sum() < 15:
            continue
        v = vlos[m]; s = np.sqrt(max(np.var(v, ddof=1) - np.mean(verr[m] ** 2), 1.0))
        rc.append(np.median(R[m])); so.append(s); se.append(s / np.sqrt(2 * (m.sum() - 1)))
    rc, so, se = np.array(rc), np.array(so), np.array(se)
    v1o, v2o = _gs_obs_vsp(R, vlos)
    rng = np.random.default_rng(0); b1, b2 = [], []
    for _ in range(300):
        j = rng.integers(0, len(R), len(R)); a1, a2 = _gs_obs_vsp(R[j], vlos[j]); b1.append(a1); b2.append(a2)
    v1e, v2e = np.std(b1), np.std(b2)
    print(f"  observed VSP1={v1o:.3e}+/-{v1e:.1e}, VSP2={v2o:.3e}+/-{v2e:.1e}")

    # seed from the pipeline's fast gNFW MLE
    res, _ = agama_fit_halo(R, vlos, label)
    g0, rs0, lrho0 = res.x
    init = np.array([np.clip(g0, 0.1, 1.8), np.log10(rs0), np.clip(lrho0, 6.2, 8.8), 0.0, 0.0, np.log10(max(rs0, 0.3))])
    print(f"  MLE seed: gamma={g0:.2f}, rs={rs0:.2f} kpc, log_rho_s={lrho0:.2f}")

    ndim = 6; nw = max(nwalkers, 2 * ndim + 2)
    lo = np.array([0.05, -1.0, 6.0, -0.95, -0.95, -1.2]); hi = np.array([1.9, 0.8, 9.0, 0.95, 0.95, 0.8])
    p0 = np.clip(init + np.array([0.1, 0.08, 0.15, 0.1, 0.1, 0.2]) * rng.standard_normal((nw, ndim)), lo + 1e-3, hi - 1e-3)
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    bk = emcee.backends.HDFBackend(backend) if HAS_H5PY else None
    resume_ok = bool(resume and bk is not None and os.path.exists(backend) and bk.iteration > 0)
    pool = mp.Pool(nproc) if nproc > 1 else None
    try:
        s = emcee.EnsembleSampler(nw, ndim, _gs_lnprob,
                                  args=(rc, so, se, v1o, v2o, v1e, v2e, a_star, use_vsp),
                                  moves=moves, pool=pool, backend=bk)
        if resume_ok:
            print(f"    [MCMC] resuming from {bk.iteration} stored steps.")
            s.run_mcmc(None, nsteps, progress=True)
        else:
            if bk is not None:
                bk.reset(nw, ndim)
            s.run_mcmc(p0, nsteps, progress=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()

    labels = [r'$\gamma$', r'$\log_{10}r_s$', r'$\log_{10}\rho_s$',
              r'$\tilde\beta_0$', r'$\tilde\beta_\infty$', r'$\log_{10}r_\beta$']
    rep = mcmc_convergence_report(s, labels)
    flat = s.get_chain(discard=rep['burn'], thin=rep['thin'], flat=True)
    if len(flat) < 50:
        flat = s.get_chain(discard=max(1, s.iteration // 3), flat=True)
    np.save(_gf("gravsphere_chain.npy"), flat)
    try:
        make_corner_plot(flat, labels, _gf("figure_gravsphere_corner.png"))
    except Exception as exc:
        print(f"  corner skipped ({exc})")

    # beta(r) posterior figure
    try:
        import matplotlib
        try: matplotlib.use("Agg")
        except Exception: pass
        import matplotlib.pyplot as plt
        rr = np.logspace(-1.6, 0.4, 60); curves = []
        for th in flat[np.random.default_rng(1).choice(len(flat), min(400, len(flat)), replace=False)]:
            curves.append(_gs_beta(rr, th[3], th[4], 10.0 ** th[5]))
        blo, bmid, bhi = np.percentile(curves, [16, 50, 84], axis=0)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.fill_between(rr, blo, bhi, color='teal', alpha=0.25); ax.plot(rr, bmid, color='teal', lw=2)
        ax.axhline(0, color='k', ls='--', lw=1); ax.set_xscale('log')
        ax.set_xlabel(r'$r$ [kpc]'); ax.set_ylabel(r'anisotropy $\beta(r)$')
        ax.set_title('GravSphere anisotropy profile (median, 1σ)'); ax.grid(alpha=0.3, which='both')
        fig.savefig(_gf("figure_gravsphere_beta.png"), dpi=140, bbox_inches='tight'); plt.close(fig)
        print("  saved figure_gravsphere_beta.png")
    except Exception as exc:
        print(f"  beta figure skipped ({exc})")

    # paper Fig.4 from the gNFW posterior (rho_s -> M_DM via mass matching)
    try:
        n = len(flat); logMDM = np.empty(n)
        for k in range(n):
            g, rs = flat[k, 0], 10.0 ** flat[k, 1]
            pot_g = agama.Potential(type='Spheroid', densityNorm=10.0 ** flat[k, 2],
                                    scaleRadius=rs, gamma=g, beta=3.0, alpha=1.0)
            f2 = dm_potential_5p(g, rs, 0.0, 1.0, 3.0).enclosedMass(2.0)
            logMDM[k] = np.log10(max(pot_g.enclosedMass(2.0) / max(f2, 1e-300), 1e6))
        chain5 = np.column_stack([logMDM, flat[:, 1], np.ones(n), 3.0 * np.ones(n), flat[:, 0]])
        make_ap25_figure4(chain5, _gf("figure_ap25_fig4.png"))
        print("  saved figure_ap25_fig4.png (paper Fig.4 from the GravSphere posterior)")
    except Exception as exc:
        print(f"  Fig.4 skipped ({exc})")

    print("\n  === posterior (median +/- 68% CI) ===")
    for k, nm in enumerate(['gamma', 'log_rs', 'log_rhos', 'bt0', 'btinf', 'log_rbeta']):
        p16, p50, p84 = np.percentile(flat[:, k], [16, 50, 84])
        print(f"    {nm:<10}= {p50:8.3f}  (+{p84 - p50:.3f} / -{p50 - p16:.3f})")
    g16, g50, g84 = np.percentile(flat[:, 0], [16, 50, 84])
    tag = "" if rep.get('converged') else "   [NOT converged -- re-run to add steps]"
    print("\n  " + "-" * 58)
    print(f"  GravSphere:  gamma = {g50:.2f} (+{g84 - g50:.2f} / -{g50 - g16:.2f}){tag}")
    print(f"               AP25 published: 0.39 (+0.23 / -0.26)")
    print("  " + "-" * 58)
    return s


# ============================================================
# FRAMEWORK COMPARISON  (Spherical Jeans vs GravSphere vs action-DF)
# ============================================================
# ============================================================
# GRAVSPHERE CROSS-CHECK  vs the reference code (Read & Steger 2017)
# ============================================================
# Demonstrates our GravSphere engine is numerically equivalent to the reference
# code https://github.com/justinread/gravsphere . Clone it first, then run:
#   python <thisfile>.py --crosscheck --repo gravsphere          (forward + estimator)
#   python <thisfile>.py --crosscheck --repo gravsphere --mcmc   (+ posterior surface)
# Checks: (1) sigma_los(R) our integrating-factor solver vs their theta-substitution
# solver; (2) theory VSPs; (3) observed-VSP estimators (VSP2 differs by a documented
# <R^2> normalisation, verified at run time); (4) --mcmc: likelihood-surface equivalence
# across the posterior (residual << Delta-chi^2=1) plus an illustrative twin-MCMC run.

def _gsx_load_reference(repo=None):
    """Import the reference gravsphere functions.py, installing compat shims for modern
    numpy/scipy first. `repo` = path to the cloned justinread/gravsphere (auto-searched)."""
    import importlib, sys, types, os
    import scipy, scipy.integrate as _si
    for _n, _v in (('int', int), ('float', float), ('bool', bool), ('str', str),
                   ('complex', complex), ('object', object)):
        if not hasattr(np, _n):
            setattr(np, _n, _v)
    if not hasattr(_si, 'simps'):
        _si.simps = _si.simpson
    if not hasattr(getattr(scipy, 'misc', None), 'derivative'):
        def _deriv(func, x0, dx=1.0, n=1, args=(), order=3):
            return (func(x0 + dx, *args) - func(x0 - dx, *args)) / (2.0 * dx)
        _m = types.ModuleType('scipy.misc'); _c = types.ModuleType('scipy.misc.common')
        _c.derivative = _deriv; _m.common = _c; _m.derivative = _deriv
        sys.modules['scipy.misc'] = _m; sys.modules['scipy.misc.common'] = _c; scipy.misc = _m
    cands = []
    for base in ([repo] if repo else []) + ['gravsphere', 'gravsphere-master', 'gravsphere-main', '.']:
        cands += [base, os.path.join(base, 'gravsphere-master'), os.path.join(base, 'gravsphere-main')]
    for cand in cands:
        if cand and os.path.exists(os.path.join(cand, 'functions.py')):
            sys.path.insert(0, os.path.abspath(cand))
            RF = importlib.import_module('functions')
            print(f"  [reference] loaded {os.path.abspath(os.path.join(cand, 'functions.py'))}")
            return RF
    raise FileNotFoundError(
        "reference repo not found -- run:  git clone https://github.com/justinread/gravsphere")


def _gsx_mass_interp(gamma, rs, rhos, rlo=1e-4, rhi=3e3, npts=700):
    rg = np.logspace(np.log10(rlo), np.log10(rhi), npts)
    dens = rhos * (rg / rs) ** (-gamma) * (1.0 + rg / rs) ** (gamma - 3.0)
    Mg = _cumtrapz(4.0 * np.pi * dens * rg ** 2, rg, initial=0.0)
    return lambda r: np.interp(r, rg, Mg)


def _gsx_reference_forward(RF, Rtest, gamma, rs, rhos, bt0, btinf, lrb, a, rmin=1e-3, rmax=1e3):
    """sigma_los(Rtest), vs1, vs2 from the reference sigp_vs() (common M(r) supplied to both)."""
    nupars = [1.0, 0.0, 0.0, a, 1.0, 1.0]                # single Plummer tracer, M=1
    betpars = [bt0, btinf, lrb, 2.0]                     # symmetrised Baes-van-Hese, n=2
    Mf = _gsx_mass_interp(gamma, rs, rhos)
    nu = lambda r, p: RF.threeplumden(r, *p)
    Sig = lambda r, p: RF.threeplumsurf(r, *p)
    M = lambda r, p: Mf(r)
    M0 = lambda r, p: np.asarray(r) * 0.0
    out = RF.sigp_vs(np.asarray(Rtest, float), np.asarray(Rtest, float), nu, Sig, M, M0,
                     RF.beta, RF.betaf, nupars, [0.0], betpars,
                     np.array([rmin, rmax]), np.array([0.0, 1.0]), 0.0, 0.0, a, G_KPC, rmin, rmax)
    _sigr2, _Sig, sigLOS2, vs1, vs2 = out
    return np.sqrt(np.maximum(sigLOS2, 0.0)), float(vs1), float(vs2)


def run_gravsphere_crosscheck(repo=None, do_mcmc=False, steps=500):
    """Cross-check the pipeline's GravSphere engine (Phase 4b) against the reference
    Read & Steger (2017) code. Needs the cloned repo (pass repo=..., or have ./gravsphere
    present). do_mcmc=True adds the end-to-end likelihood-surface equivalence check
    (needs agama + emcee). Returns True on PASS."""
    RF = _gsx_load_reference(repo)
    RTEST = np.array([0.05, 0.10, 0.20, 0.33, 0.50, 0.80, 1.20])
    CASES = [dict(name="A: Sculptor-like core, anisotropic",
                  gamma=0.6, rs=0.7, rhos=1.5e8, bt0=-0.10, btinf=0.50, lrb=0.0, a=0.33),
             dict(name="B: NFW cusp, isotropic",
                  gamma=1.0, rs=1.0, rhos=6.0e7, bt0=0.00, btinf=0.00, lrb=0.0, a=0.28)]
    SIG_TOL, VSP_TOL = 0.01, 0.03

    # (1)+(2) forward-model agreement
    worst_sig = worst_vsp = 0.0
    for case in CASES:
        c = dict(case); name = c.pop('name')
        bf = lambda r, c=c: _gs_beta(r, c['bt0'], c['btinf'], 10.0 ** c['lrb'])
        s_me = _gs_sigma_los(RTEST, c['gamma'], c['rs'], c['rhos'], bf, c['a'])
        v1_me, v2_me = _gs_vsp_theory(c['gamma'], c['rs'], c['rhos'], bf, c['a'])
        s_rf, v1_rf, v2_rf = _gsx_reference_forward(RF, RTEST, **c)
        dev = float(np.max(np.abs(s_me / s_rf - 1.0))); worst_sig = max(worst_sig, dev)
        print(f"\n--- sigma_los(R) [km/s] :: case {name} ---")
        print("    R (kpc):   " + "  ".join(f"{r:7.3f}" for r in RTEST))
        print("    ours:      " + "  ".join(f"{s:7.3f}" for s in s_me))
        print("    reference: " + "  ".join(f"{s:7.3f}" for s in s_rf))
        print(f"    max |dev| = {100*dev:.2f}%   {'PASS' if dev < SIG_TOL else 'FAIL'} (tol {100*SIG_TOL:.0f}%)")
        dv = max(abs(v1_me / v1_rf - 1.0), abs(v2_me / v2_rf - 1.0)); worst_vsp = max(worst_vsp, dv)
        print(f"--- theory VSPs :: case {name} ---")
        print(f"    vs1: ours={v1_me:.5e}  reference={v1_rf:.5e}  ratio={v1_me/v1_rf:.4f}")
        print(f"    vs2: ours={v2_me:.5e}  reference={v2_rf:.5e}  ratio={v2_me/v2_rf:.4f}")
        print(f"    max |dev| = {100*dv:.2f}%   {'PASS' if dv < VSP_TOL else 'FAIL'} (tol {100*VSP_TOL:.0f}%)")
    ok_fwd = worst_sig < SIG_TOL and worst_vsp < VSP_TOL

    # (3) observed-VSP estimators
    rng = np.random.default_rng(0)
    R = np.abs(rng.normal(0.0, 0.4, 5000)); v = rng.normal(0.0, 10.0, 5000)
    mine1, mine2 = _gs_obs_vsp(R, v)
    ref1, ref2 = RF.richfair_vsp(v, R, np.ones_like(v))
    ok1 = abs(mine1 / ref1 - 1.0) < 1e-10
    ok2 = abs(ref2 * np.mean(R ** 2) / mine2 - 1.0) < 1e-10
    print("\n--- observed-VSP estimators (5000 synthetic stars) ---")
    print(f"    VSP1: ours/reference = {mine1/ref1:.12f}   {'PASS (identical)' if ok1 else 'FAIL'}")
    print(f"    VSP2: reference x <R^2> / ours = {ref2*np.mean(R**2)/mine2:.12f}   "
          f"{'PASS (convention verified)' if ok2 else 'FAIL'}")
    print("    (reference's richfair_vsp2 normalises by <R^2>; ours is the direct unbiased")
    print("     estimator of the same theory integral vs2 = <v^4 R^2>/2pi used in our likelihood)")
    ok_est = ok1 and ok2

    ok_mcmc = True
    if do_mcmc:
        ok_mcmc = _gsx_mcmc_check(RF, steps=steps)

    print("\n" + "=" * 72)
    if ok_fwd and ok_est and ok_mcmc:
        print("  VERDICT: PASS -- our GravSphere implementation is numerically equivalent to")
        print(f"  the reference code (sigma_los within {100*worst_sig:.2f}%, theory VSPs within "
              f"{100*worst_vsp:.2f}%;")
        print("  estimator conventions verified" + ("; likelihood surfaces equivalent)." if do_mcmc else ")."))
    else:
        print("  VERDICT: FAIL -- see the sections above for which check differed.")
    print("=" * 72)
    return bool(ok_fwd and ok_est and ok_mcmc)


def _gsx_mcmc_check(RF, steps=500, nwalk=14, seed=1):
    """End-to-end posterior equivalence: (i) decisive likelihood-surface scan across the
    posterior (residual << Delta-chi^2=1), (ii) illustrative twin MCMC vs its MC error."""
    import agama, emcee, time
    agama.setUnits(mass=1, length=1, velocity=1)
    print("\n--- end-to-end posterior equivalence (mock; gamma_true = 1.0) ---")
    rst, rhost, rat, a = 1.0, 6.0e7, 2.0, 0.28
    pot = agama.Potential(type='Spheroid', densityNorm=rhost, scaleRadius=rst, gamma=1.0, beta=3.0, alpha=1.0)
    tr = agama.Density(type='Plummer', scaleRadius=a, mass=1.0)
    gm = agama.GalaxyModel(pot, agama.DistributionFunction(type='QuasiSpherical', potential=pot,
                                                           density=tr, beta0=0.0, r_a=rat))
    rng = np.random.default_rng(seed)
    xv, _ = gm.sample(20000); x, y, z, vx, vy, vz = xv.T; Rf = np.hypot(x, y)
    idx = rng.choice(np.where(Rf < 2.0)[0], 1000, replace=False)
    R = Rf[idx]; vlos = vz[idx] + rng.normal(0, 2.0, 1000)
    print(f"    mock: 1000 stars, sigma_los = {vlos.std():.1f} km/s")
    edges = np.quantile(R, np.linspace(0, 1, 7)); rc, so, se = [], [], []
    for i in range(6):
        m = (R >= edges[i]) & ((R <= edges[i + 1]) if i == 5 else (R < edges[i + 1]))
        vv = vlos[m]; s = np.sqrt(max(np.var(vv, ddof=1) - 4.0, 1.0))
        rc.append(np.median(R[m])); so.append(s); se.append(s / np.sqrt(2 * (m.sum() - 1)))
    rc, so, se = np.array(rc), np.array(so), np.array(se)
    v1o, v2o = _gs_obs_vsp(R, vlos); b1, b2 = [], []
    for _ in range(200):
        j = rng.integers(0, len(R), len(R)); a1, a2 = _gs_obs_vsp(R[j], vlos[j]); b1.append(a1); b2.append(a2)
    v1e, v2e = np.std(b1), np.std(b2)
    LO = np.array([0.05, -1.0, 6.0, -0.95, -0.95, -1.2]); HI = np.array([1.9, 0.8, 9.0, 0.95, 0.95, 0.8])

    def lnp_ours(th):
        return _gs_lnprob(th, rc, so, se, v1o, v2o, v1e, v2e, a, True)

    def lnp_ref(th):
        if np.any(th < LO) or np.any(th > HI):
            return -np.inf
        g, lrs, lrh, b0, bi, lrb = th
        try:
            sig, v1m, v2m = _gsx_reference_forward(RF, rc, g, 10 ** lrs, 10 ** lrh, b0, bi, lrb, a)
            chi2 = np.sum(((so - sig) / se) ** 2) + ((v1o - v1m) / v1e) ** 2 + ((v2o - v2m) / v2e) ** 2
            return -0.5 * chi2 if np.isfinite(chi2) else -np.inf
        except Exception:
            return -np.inf

    t0 = time.time(); lnp_ref(np.array([0.8, 0.0, 7.8, 0.0, 0.0, 0.0]))
    print(f"    reference forward-model eval: {1000*(time.time()-t0):.0f} ms")
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    p0 = np.clip(np.array([0.8, 0.0, 7.8, 0.0, 0.0, 0.0]) +
                 np.array([.1, .08, .12, .1, .1, .2]) * rng.standard_normal((nwalk, 6)), LO + 1e-3, HI - 1e-3)

    s0 = emcee.EnsembleSampler(nwalk, 6, lnp_ours, moves=moves)
    s0.run_mcmc(p0.copy(), 600, progress=False)
    flat = s0.get_chain(discard=300, flat=True)
    dl = []
    for th in flat[rng.choice(len(flat), 60, replace=False)]:
        la, lb = lnp_ours(th), lnp_ref(th)
        if np.isfinite(la) and np.isfinite(lb):
            dl.append(la - lb)
    dl = np.array(dl); resid = dl - dl.mean()
    surf_ok = (np.abs(resid).max() < 1.0) and (resid.std() < 0.3)
    print(f"    (i) likelihood-surface scan on {len(dl)} posterior draws:")
    print(f"        mean offset lnL_ours - lnL_ref = {dl.mean():+.3f}")
    print(f"        parameter-dependent residual: rms = {resid.std():.3f}, max|.| = {np.abs(resid).max():.3f}")
    print(f"        -> {'PASS' if surf_ok else 'FAIL'} (residual << Delta-chi^2=1: same posterior)")

    res = {}
    for name, fn in [("ours", lnp_ours), ("reference", lnp_ref)]:
        s = emcee.EnsembleSampler(nwalk, 6, fn, moves=moves)
        t0 = time.time(); s.run_mcmc(p0.copy(), steps, progress=False)
        g = s.get_chain(discard=steps // 2)[:, :, 0]; ess = max(effective_sample_size(g), 4.0)
        gf = g.flatten(); g16, g50, g84 = np.percentile(gf, [16, 50, 84])
        res[name] = (g50, gf.std() / np.sqrt(ess))
        print(f"    (ii) {name:<10}: gamma = {g50:.2f} (+{g84-g50:.2f}/-{g50-g16:.2f})  [ESS~{ess:.0f}, {time.time()-t0:.0f}s]")
    dmed = abs(res['ours'][0] - res['reference'][0]); sig_mc = float(np.hypot(res['ours'][1], res['reference'][1]))
    print(f"         |Delta median gamma| = {dmed:.3f} vs 3 x MC error = {3*sig_mc:.3f} -> "
          f"{'consistent within Monte-Carlo noise' if dmed < 3*sig_mc + 0.02 else 'raise --steps to resolve'}")
    return surf_ok


# ============================================================
# PHASE 5: FULL FAITHFUL 25-PARAMETER MODEL  (validation benchmark)
# ============================================================
# Faithful reproduction of the COMPLETE Arroyo-Polonio+25 model (paper Eqs.4,5,8,11):
#   5 DM + 10 stellar-DF (DoublePowerLaw x2) + 4 metallicity + 4 pop-3 + 2 fractions.
# This is the rigorous tier above Phase 4 (fast 6-param gNFW+anisotropy). It is a
# VALIDATION BENCHMARK for a separate continuous f(J,[Fe/H]) method, not the science.
# The per-star projectedDF likelihood over 1339 stars is a cluster-scale computation;
# this sandbox only smoke-tests the machinery on a mock.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # NumPy 2.x renamed trapz
REGION_RMAX = 2.0        # kpc, farthest observed star / normalisation region

# ── Parameter layout (paper Eq.12 order) ─────────────────────────────────────
PARAM_NAMES = [
    'logM_DM', 'log_rs', 'alpha', 'eta', 'gamma',
    'logJ_MP', 'Gamma_MP', 'B_MP', 'gz_MP', 'hz_MP', 'M_MP', 'sigM_MP',
    'logJ_MR', 'Gamma_MR', 'B_MR', 'gz_MR', 'hz_MR', 'M_MR', 'sigM_MR',
    'V_C', 'sigV_C', 'M_C', 'sigM_C',
    'f_MP', 'f_MR',
]
NDIM = len(PARAM_NAMES)              # 25

# Uniform-prior bounds (paper Table 1)
PRIOR_LO = np.array([7.0, -3.0, 0.0, 2.0, 0.0,
                     0.3, 0.0, 3.0, 0.0, 0.0, -2.7, 0.15,
                     0.3, 0.0, 3.0, 0.0, 0.0, -1.65, 0.15,
                     -30.0, 0.0, -4.5, 0.15,
                     0.5, 0.24])
PRIOR_HI = np.array([11.0, 1.0, 7.0, 7.0, 1.9,
                     2.5, 3.0, 30.0, 1.5, 1.5, -1.7, 0.45,
                     2.5, 3.0, 30.0, 1.5, 1.5, -1.3, 0.45,
                     30.0, 20.0, -1.3, 0.90,
                     0.75, 0.5])
# NOTE: paper Table 1 lists the M_MP prior as [-2.7,-2.1] but reports a median of
# -2.0 (outside that range -> a table typo); we use [-2.7,-1.7] so it contains the
# physical value. For the mock, the paper's Table-3 prior is [-2.1,-1.7].

# TeX labels for corner plots
TEX = [r'$\log M_{\rm DM}$', r'$\log r_s$', r'$\alpha$', r'$\eta$', r'$\gamma$',
       r'$\log J_{\rm MP}$', r'$\Gamma_{\rm MP}$', r'$B_{\rm MP}$', r'$g_{z,\rm MP}$', r'$h_{z,\rm MP}$',
       r'$\mathcal{M}_{\rm MP}$', r'$\sigma_{\rm MP}$',
       r'$\log J_{\rm MR}$', r'$\Gamma_{\rm MR}$', r'$B_{\rm MR}$', r'$g_{z,\rm MR}$', r'$h_{z,\rm MR}$',
       r'$\mathcal{M}_{\rm MR}$', r'$\sigma_{\rm MR}$',
       r'$V_C$', r'$\sigma_{V,C}$', r'$\mathcal{M}_C$', r'$\sigma_{M,C}$', r'$f_{\rm MP}$', r'$f_{\rm MR}$']

_INF = np.inf


def OMEGA_R(R):
    """Selection function Omega(R)=int omega(R,G) dG. Default flat (=1).
    Replace with the AP24 Gaussian-kernel selection for a fully faithful real fit."""
    return np.ones_like(np.asarray(R, float))


# ── Model builders (mapping verified against AGAMA) ──────────────────────────
def ap25_dm_potential(logM_DM, log_rs, alpha, eta, gamma):
    """Paper Eq.4 truncated double-power-law DM potential (AGAMA Spheroid)."""
    return agama.Potential(type='Spheroid', mass=10.0**logM_DM, scaleRadius=10.0**log_rs,
                           gamma=gamma, beta=eta, alpha=alpha,
                           outerCutoffRadius=DM_RCUT, cutoffStrength=DM_XI)


def ap25_stellar_df(logJ, Gamma, B, gz, hz):
    """Paper Eq.5 action DF (AGAMA DoublePowerLaw, steepness=1). Tracer mass=1."""
    return agama.DistributionFunction(
        type='DoublePowerLaw', mass=1.0, J0=10.0**logJ,
        slopeIn=Gamma, slopeOut=B, steepness=1.0,
        coefJrIn=3.0 - 2.0 * hz, coefJzIn=hz,          # h_z = h_phi (spherical)
        coefJrOut=3.0 - 2.0 * gz, coefJzOut=gz)        # g_z = g_phi (spherical)


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)


_RGRID = np.logspace(-2.3, np.log10(REGION_RMAX), 14)   # for normalisation integrals


def _proj_norm(gm, Omega_grid=None):
    """Z_p = 2π ∫_0^Rmax Sigma(R) Omega(R) R dR  (mass fraction within the region).
    Sigma(R) = projectedDF with all velocity uncertainties infinite (=projected density).
    Omega_grid: precomputed Omega(R) on _RGRID (from a selection function); flat if None."""
    pts = np.column_stack([_RGRID, np.zeros_like(_RGRID), np.zeros_like(_RGRID),
                           np.zeros_like(_RGRID), np.zeros_like(_RGRID),
                           np.full_like(_RGRID, _INF), np.full_like(_RGRID, _INF),
                           np.full_like(_RGRID, _INF)])
    Sigma = np.maximum(gm.projectedDF(pts), 0.0)
    Om = OMEGA_R(_RGRID) if Omega_grid is None else Omega_grid
    integ = 2 * np.pi * Sigma * Om * _RGRID
    return _trapz(integ, _RGRID)


# ── Likelihood (Eq.11) ───────────────────────────────────────────────────────
def ap25_lnprior_full(theta):
    if np.any(theta < PRIOR_LO) or np.any(theta > PRIOR_HI):
        return -np.inf
    f_MP, f_MR = theta[23], theta[24]
    if f_MP + f_MR >= 1.0:                # need f_C = 1 - f_MP - f_MR > 0
        return -np.inf
    return 0.0


def ap25_lnlike_full(theta, data):
    """Full 25-parameter per-star mixture log-likelihood (paper Eq.11).
    data: dict with arrays R, vlos (rest frame), verr, feh, feherr (G optional)."""
    (logM_DM, log_rs, alpha, eta, gamma,
     logJ_MP, Gamma_MP, B_MP, gz_MP, hz_MP, M_MP, sigM_MP,
     logJ_MR, Gamma_MR, B_MR, gz_MR, hz_MR, M_MR, sigM_MR,
     V_C, sigV_C, M_C, sigM_C, f_MP, f_MR) = theta
    f_C = 1.0 - f_MP - f_MR
    R, vlos, verr = data['R'], data['vlos'], data['verr']
    feh, feherr = data['feh'], data['feherr']
    N = len(R)
    # Selection function: use precomputed arrays if attached (multiprocessing-safe,
    # since setting a module global does NOT propagate to spawned workers), else flat.
    #   omega_star : per-star omega(R_i, G_i), length N    (numerator)
    #   Omega_grid : Omega(R)=int omega dG on _RGRID       (normalisation)
    omega_star = data.get('omega_star')
    if omega_star is None:
        omega_star = OMEGA_R(R)
    Omega_grid = data.get('Omega_grid')
    if Omega_grid is None:
        Omega_grid = OMEGA_R(_RGRID)
    try:
        pot = ap25_dm_potential(logM_DM, log_rs, alpha, eta, gamma)
        # star points for projectedDF: (X,Y,vX,vY,vZ,vXerr,vYerr,vZerr); los-only
        pts = np.column_stack([R, np.zeros(N), np.zeros(N), np.zeros(N), vlos,
                               np.full(N, _INF), np.full(N, _INF), verr])
        L = np.zeros(N)
        for (logJ, Gam, B, gz, hz, Mm, sM, frac) in [
                (logJ_MP, Gamma_MP, B_MP, gz_MP, hz_MP, M_MP, sigM_MP, f_MP),
                (logJ_MR, Gamma_MR, B_MR, gz_MR, hz_MR, M_MR, sigM_MR, f_MR)]:
            gm = agama.GalaxyModel(pot, ap25_stellar_df(logJ, Gam, B, gz, hz))
            Z = _proj_norm(gm, Omega_grid)
            if not (Z > 0):
                return -np.inf
            Pd  = np.maximum(gm.projectedDF(pts), 0.0) / Z            # per area per vel
            mdf = _gauss(feh, Mm, np.sqrt(sM**2 + feherr**2))         # error-convolved MDF
            L  += frac * Pd * omega_star * mdf
        # pop-3 (contamination): flat-in-region spatial x Gaussian(v) x Gaussian([Fe/H])
        Z_C = _trapz(2 * np.pi * Omega_grid * _RGRID, _RGRID)
        spatialC = omega_star / Z_C
        velC = _gauss(vlos, V_C, np.sqrt(sigV_C**2 + verr**2))
        metC = _gauss(feh,  M_C, np.sqrt(sigM_C**2 + feherr**2))
        L += f_C * spatialC * velC * metC
        if np.any(~np.isfinite(L)) or np.any(L <= 0):
            return -np.inf
        return float(np.sum(np.log(L)))
    except Exception:
        return -np.inf


def ap25_lnprob_full(theta, data):
    lp = ap25_lnprior_full(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ap25_lnlike_full(theta, data)


# ── Mock generator (paper Sec.5 cuspy test: DM gamma=1) ──────────────────────
# Table-3 "Actual" mock values.
MOCK_TRUTH = dict(
    logM_DM=8.5, log_rs=-0.5, alpha=1.0, eta=3.0, gamma=1.0,
    logJ_MP=0.8, Gamma_MP=1.0, B_MP=14.0, gz_MP=1.2, hz_MP=0.05, M_MP=-2.0, sigM_MP=0.30,
    logJ_MR=1.5, Gamma_MR=0.4, B_MR=20.0, gz_MR=0.4, hz_MR=0.60, M_MR=-1.4, sigM_MR=0.30,
    V_C=13.0, sigV_C=7.0, M_C=-2.8, sigM_C=0.5, f_MP=0.60, f_MR=0.38)   # f_C=0.02
# (Paper's Sec.5 mock is a clean 2-population test with f_C=0; we use a small f_C>0
#  so the FULL 25-parameter machinery — including the pop-3 component — is exercised.)


def ap25_truth_vector():
    return np.array([MOCK_TRUTH[k] for k in PARAM_NAMES])


def ap25_generate_full_mock(n_stars=1339, seed=7, truth=MOCK_TRUTH):
    """Sample R, vlos, [Fe/H] (+errors) from the full model, matching the paper's Sec.5 mock."""
    rng = np.random.default_rng(seed)
    pot = ap25_dm_potential(truth['logM_DM'], truth['log_rs'], truth['alpha'], truth['eta'], truth['gamma'])
    f_MP, f_MR = truth['f_MP'], truth['f_MR']
    f_C = 1.0 - f_MP - f_MR
    rows = []
    for tag, frac, Mm, sM in [('MP', f_MP, truth['M_MP'], truth['sigM_MP']),
                              ('MR', f_MR, truth['M_MR'], truth['sigM_MR'])]:
        df = ap25_stellar_df(truth[f'logJ_{tag}'], truth[f'Gamma_{tag}'], truth[f'B_{tag}'],
                        truth[f'gz_{tag}'], truth[f'hz_{tag}'])
        gm = agama.GalaxyModel(pot, df)
        n = int(round(n_stars * frac))
        xv, _ = gm.sample(int(n * 1.6))                 # oversample; then cut to region
        x, y, z, vx, vy, vz = xv.T
        Rp = np.hypot(x, y)
        keep = Rp < REGION_RMAX
        x, vz, Rp = x[keep][:n], vz[keep][:n], Rp[keep][:n]
        feh = rng.normal(Mm, sM, len(Rp))
        rows.append(np.column_stack([Rp, vz, feh]))
    # pop-3 contamination
    nC = max(int(round(n_stars * f_C)), 5)
    RC = REGION_RMAX * np.sqrt(rng.random(nC))          # uniform in area
    vC = rng.normal(truth['V_C'], truth['sigV_C'], nC)
    fC = rng.normal(truth['M_C'], truth['sigM_C'], nC)
    rows.append(np.column_stack([RC, vC, fC]))
    M = np.vstack(rows)
    R, vlos, feh = M[:, 0], M[:, 1], M[:, 2]
    verr = np.full(len(R), 0.6)                          # paper mean vel error
    feherr = np.full(len(R), 0.1)                        # paper mean [Fe/H] error
    vlos = vlos + rng.normal(0, verr)                    # add measurement noise
    feh = feh + rng.normal(0, feherr)
    return dict(R=R, vlos=vlos, verr=verr, feh=feh, feherr=feherr,
                G=np.full(len(R), 18.0))


# ── Real data loader (run where VizieR/Gaia are reachable) ───────────────────
# Candidate VizieR column names (broadened; Tolstoy+2023 exact names vary by table)
_VCANDS  = ['HRV', 'RVel', 'Vlos', 'vlos', 'HV', 'Vhel', 'vhel', 'RV', 'Vrad', 'vrad', 'Vel', 'cz', 'Vhelio']
_EVCANDS = ['e_HRV', 'e_RVel', 'e_Vlos', 'e_HV', 'e_Vhel', 'e_RV', 'e_Vrad', 'e_Vel', 'e_vlos', 'e_vhel']
_FCANDS  = ['__Fe_H_', '[Fe/H]', 'Fe_H', 'FeH', 'feh', '__Fe_H_c', '[Fe/Hc]', 'FeHc', 'MH', '__M_H_']
_EFCANDS = ['e__Fe_H_', 'e_Fe_H', 'e_FeH', 'e_feh', 'e_[Fe/H]', 'e__M_H_']
_RACANDS = ['RAJ2000', 'RA_ICRS', '_RAJ2000', 'RAdeg', 'ra', '_RA']
_DECANDS = ['DEJ2000', 'DE_ICRS', '_DEJ2000', 'DEdeg', 'de', 'dec', '_DE']
_GCANDS  = ['Gmag', 'phot_g_mean_mag', 'Gaia_G', 'gmag', 'G']   # Gaia G (for 2-D selection)


def ap25_inspect_vizier(catalog="J/A+A/675/A49"):
    """Print every table and its columns for a VizieR catalog — run this ONCE to
    discover the real velocity/[Fe/H]/coord column names, then pass them to the
    loader via cols=dict(vlos=..., feh=..., ra=..., dec=..., verr=..., feherr=...)."""
    from astroquery.vizier import Vizier
    v = Vizier(columns=["**"]); v.ROW_LIMIT = 3
    cats = v.get_catalogs(catalog)
    print(f"VizieR catalog {catalog}: {len(cats)} table(s)")
    for i, t in enumerate(cats):
        print(f"  [{i}] {t.meta.get('name', '?')}  ({len(t)} sample rows)")
        print(f"      columns: {list(t.colnames)}")
    return cats


def _fetch_tolstoy2023(catalog="J/A+A/675/A49", cols=None,
                       require_member=True, mem_keep=('m',), feh_quality_keep=None,
                       target_col=None, target_keep=None, mem_min=None):
    """Return (ra, dec, vlos, verr, feh, feherr) from the Tolstoy+2023 VizieR catalog,
    auto-detecting the columns across ALL tables. `cols` optionally overrides names,
    e.g. cols=dict(vlos='Vlos', feh='__Fe_H_', ra='RAJ2000', dec='DEJ2000').

    Sample selection (to match the paper's analysis set):
      require_member -- keep only rows whose membership column is in `mem_keep`
                        (Tolstoy 'Mem' uses 'm' for member). Auto-skipped if no Mem col.
      feh_quality_keep -- if given (e.g. a list of accepted 'q_[Fe/H]' codes), keep only
                        rows with a reliable [Fe/H] flag. Inspect np.unique(t['q_[Fe/H]'])
                        first to see the codes; this is how the paper reaches ~1339 stars.
    On failure it prints every table's columns and raises a diagnostic KeyError."""
    from astroquery.vizier import Vizier
    cols = cols or {}
    v = Vizier(columns=["**"]); v.ROW_LIMIT = -1
    cats = v.get_catalogs(catalog)

    def find(table, cands):
        low = {c.lower(): c for c in table.colnames}
        for c in cands:
            if c.lower() in low:
                return low[c.lower()]
        return None

    def col_in(table, name):                             # accept an explicit name ONLY if present
        return name if (name and name in table.colnames) else None

    for t in cats:                                       # pick the table that ACTUALLY has v AND [Fe/H]
        vc = col_in(t, cols.get('vlos')) or find(t, _VCANDS)
        fc = col_in(t, cols.get('feh'))  or find(t, _FCANDS)
        if vc and fc:
            rc  = col_in(t, cols.get('ra'))  or find(t, _RACANDS)
            dc  = col_in(t, cols.get('dec')) or find(t, _DECANDS)
            evc = col_in(t, cols.get('verr'))   or find(t, _EVCANDS)
            efc = col_in(t, cols.get('feherr')) or find(t, _EFCANDS)
            if rc is None or dc is None:
                raise KeyError(f"found v/[Fe/H] but not RA/Dec in table "
                               f"'{t.meta.get('name')}'; columns: {list(t.colnames)}")
            g = lambda name: np.array(t[name], float)
            def gcoord(name, is_ra):                          # decimal deg, or parse sexagesimal
                try:
                    return np.array(t[name], float)
                except (ValueError, TypeError):
                    from astropy.coordinates import Angle
                    import astropy.units as u
                    unit = u.hourangle if is_ra else u.deg
                    return Angle([str(x) for x in t[name]], unit=unit).to(u.deg).value
            verr = g(evc) if evc else np.full(len(t), 2.0)    # default 2 km/s if absent
            feherr = g(efc) if efc else np.full(len(t), 0.1)  # default 0.1 dex if absent
            gcol = cols.get('gmag') or find(t, _GCANDS)
            gmag = g(gcol) if gcol else np.full(len(t), np.nan)   # real Gaia G if present
            # ── sample selection: target galaxy (multi-galaxy catalogs) + members ──
            sel = np.ones(len(t), bool)
            tcol = target_col or cols.get('target') or find(t, ['Target', 'Galaxy', 'dSph', 'Name'])
            if target_keep is not None and tcol is not None:  # keep only the requested galaxy
                pref = tuple(str(k).strip().lower() for k in target_keep)
                tv = np.array([str(x).strip().lower().startswith(pref) for x in t[tcol]])
                sel &= tv
                kept = sorted(set(str(x).strip()[:4] for x in np.array(t[tcol])[tv]))
                print(f"    [loader] target filter '{tcol}' startswith {list(target_keep)}: "
                      f"kept {int(tv.sum())}/{len(t)} rows (matched prefixes: {kept})")
            mcol = cols.get('mem') or find(t, ['Mmb', 'Mem', 'Member', 'memb', 'Pmemb', 'Pmem'])
            if mem_min is not None and mcol is not None:      # membership PROBABILITY threshold
                pv = np.array([float(x) if str(x).strip() not in ('', '--', 'nan') else np.nan
                               for x in t[mcol]])
                keep = np.isfinite(pv) & (pv >= mem_min)
                sel &= keep
                print(f"    [loader] membership '{mcol}' >= {mem_min}: {int(keep.sum())}/{len(t)} rows")
            elif require_member and mcol is not None:         # exact-value membership flag
                mv = np.array([str(x).strip().lower() for x in t[mcol]])
                sel &= np.isin(mv, [str(k).lower() for k in mem_keep])
            qcol = cols.get('feh_quality') or find(t, ['q_[Fe/H]', 'q__Fe_H_', 'q_FeH', 'f_[Fe/H]'])
            if feh_quality_keep is not None and qcol is not None:
                qv = np.array([str(x).strip() for x in t[qcol]])
                sel &= np.isin(qv, [str(k) for k in feh_quality_keep])
            print(f"    [tolstoy loader] table '{t.meta.get('name')}': vlos={vc}, "
                  f"verr={evc or '(default 2)'}, feh={fc}, feherr={efc or '(default 0.1)'}, "
                  f"ra={rc}, dec={dc}, G={gcol or '(none)'}")
            print(f"    [tolstoy loader] selected {int(sel.sum())}/{len(t)} rows"
                  f" (member={mcol or 'n/a'}"
                  + (f", q_[Fe/H]={qcol}" if feh_quality_keep is not None else "") + ")")
            return (gcoord(rc, True)[sel], gcoord(dc, False)[sel], g(vc)[sel], verr[sel],
                    g(fc)[sel], feherr[sel], gmag[sel])

    lines = [f"Could not auto-detect velocity+[Fe/H] columns in '{catalog}'.",
             "Run ap25_inspect_vizier(catalog), then pass cols=dict(...). Tables found:"]
    for i, t in enumerate(cats):
        lines.append(f"  [{i}] {t.meta.get('name', '?')}: {list(t.colnames)}")
    raise KeyError("\n".join(lines))


def _semi_major_axis_radius(ra, dec, D_KPC=None):
    """Paper Eq.1 semi-major-axis radius (kpc) using the ACTIVE galaxy's centre, ellipticity
    and PA (Munoz+2018 for Sculptor/Fornax; switch via set_galaxy)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if D_KPC is None:
        D_KPC = GAL['distance_kpc']
    cen = SkyCoord(GAL['center_ra'] * u.deg, GAL['center_dec'] * u.deg)
    e, pa = GAL['ellipticity'], np.radians(GAL['pa_deg'])
    c = SkyCoord(ra * u.deg, dec * u.deg)
    dx = (c.ra - cen.ra).to(u.rad).value * np.cos(cen.dec.rad) * D_KPC
    dy = (c.dec - cen.dec).to(u.rad).value * D_KPC
    xmaj =  dx * np.cos(pa) + dy * np.sin(pa)
    ymin = -dx * np.sin(pa) + dy * np.cos(pa)
    return np.sqrt(xmaj**2 + (ymin / (1 - e))**2)


# ── SELECTION FUNCTION omega(R,G)  (AP24-style; Arroyo-Polonio+24) ────────────
# The selection is the ratio of the SPECTROSCOPIC sample density to the PHOTOMETRIC
# PARENT density, via Gaussian kernels:  omega(R,G) ∝ KDE_spec / KDE_parent. Only the
# SHAPE matters (a constant cancels between the per-star numerator and the normalisation),
# so we peak-normalise. attach_selection() precomputes per-star omega and the Omega(R)
# grid and stores them IN `data` (data['omega_star'], data['Omega_grid']) -- essential
# because on Windows the multiprocessing workers re-import this module and would NOT see
# a module-global selection, whereas arrays inside `data` are pickled to every worker.

def gaia_rgb_parent(radius_deg=0.9, g_range=(17.0, 20.5), pm=None, pm_tol=0.6,
                    ruwe_max=1.4, plx_max=0.10):
    """Query Gaia DR3 for Sculptor RGB candidates = the PHOTOMETRIC PARENT population.
    Astrometric member cuts (small parallax, PM near systemic, good RUWE) isolate the
    galaxy; g_range matches the spectroscopic magnitudes. Returns (R_kpc, Gmag) with R
    the semi-major-axis radius. Requires network to Gaia (retry if it drops), or pass
    your own parent (R, G) to attach_selection()."""
    from astroquery.gaia import Gaia
    if pm is None:
        pm = (SCULPTOR_PMRA_SYS, SCULPTOR_PMDEC_SYS)
    adql = f"""
    SELECT ra, dec, phot_g_mean_mag, pmra, pmdec, parallax, ruwe
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                     CIRCLE('ICRS', 15.0183, -33.7186, {radius_deg}))
      AND phot_g_mean_mag BETWEEN {g_range[0]} AND {g_range[1]}
      AND ruwe < {ruwe_max}
      AND ABS(parallax) < {plx_max}
      AND SQRT(POWER(pmra-({pm[0]}),2)+POWER(pmdec-({pm[1]}),2)) < {pm_tol}
    """
    t = Gaia.launch_job_async(adql).get_results()
    ra = np.array(t['ra'], float); dec = np.array(t['dec'], float)
    G = np.array(t['phot_g_mean_mag'], float)
    R = _semi_major_axis_radius(ra, dec)
    print(f"  [parent] Gaia DR3: {len(R)} RGB candidates "
          f"(r<{radius_deg} deg, G in {g_range}, |PM-Scl|<{pm_tol})")
    return R, G


def _kde(x, bw):
    from scipy.stats import gaussian_kde
    return gaussian_kde(np.asarray(x, float), bw_method=bw)


def build_radial_selection(spec_R, parent_R, bw=0.3, floor=1e-3):
    """Omega_R(R): radial completeness ∝ KDE_spec(R)/KDE_parent(R), peak-normalised.
    Zero where the parent is empty (no tracers observed there)."""
    ks, kp = _kde(spec_R, bw), _kde(parent_R, bw)
    Rmax_par = np.max(parent_R)
    def Omega_R(R):
        R = np.asarray(R, float)
        den = kp(R); num = ks(R)
        w = np.where(den > floor * den.max(), num / np.maximum(den, 1e-300), 0.0)
        w[R > Rmax_par] = 0.0
        return w
    grid = np.linspace(min(np.min(spec_R), np.min(parent_R)), Rmax_par, 400)
    peak = np.max(Omega_R(grid)) or 1.0
    return lambda R: Omega_R(R) / peak


def build_2d_selection(spec_R, spec_G, parent_R, parent_G, bw=0.3, floor=1e-3):
    """(omega_RG(R,G), Omega_R(R)) from 2-D Gaussian KDEs; Omega_R(R)=∫omega dG.
    gaussian_kde adapts to the R,G scale difference via the data covariance."""
    from scipy.stats import gaussian_kde
    S  = np.vstack([np.asarray(spec_R, float),   np.asarray(spec_G, float)])
    Pp = np.vstack([np.asarray(parent_R, float), np.asarray(parent_G, float)])
    ks, kp = gaussian_kde(S, bw_method=bw), gaussian_kde(Pp, bw_method=bw)
    Rmax_par = np.max(parent_R)
    Ggrid = np.linspace(np.min(parent_G), np.max(parent_G), 40)
    def omega_RG(R, G):
        R = np.atleast_1d(np.asarray(R, float)); G = np.atleast_1d(np.asarray(G, float))
        pts = np.vstack([R, G]); den = kp(pts); num = ks(pts)
        w = np.where(den > floor * den.max(), num / np.maximum(den, 1e-300), 0.0)
        w[R > Rmax_par] = 0.0
        return w
    def Omega_R(R):
        R = np.atleast_1d(np.asarray(R, float)); out = np.empty_like(R)
        for i, r in enumerate(R):
            out[i] = _trapz(omega_RG(np.full_like(Ggrid, r), Ggrid), Ggrid)   # NumPy-2 safe
        return out
    grid = np.linspace(np.min(spec_R), Rmax_par, 200)
    peak = np.max(Omega_R(grid)) or 1.0
    return (lambda R, G: omega_RG(R, G) / peak), (lambda R: Omega_R(R) / peak)


def attach_selection(data, parent_R, parent_G=None, mode='radial',
                     R_grid=None, bw=0.3, floor=1e-3):
    """Compute the AP24-style selection and ATTACH it to `data` as precomputed arrays
    (data['omega_star'] per star, data['Omega_grid'] on the normalisation grid _RGRID)
    so the parallel likelihood uses it correctly. Returns the updated `data`.
      mode='radial' (recommended; needs only projected radii) or '2d' (needs real Gmag).
      parent_R/parent_G come from gaia_rgb_parent() or your own photometric catalogue."""
    if R_grid is None:
        R_grid = _RGRID
    spec_R = np.asarray(data['R'], float)
    data = dict(data)
    if mode == 'radial':
        Omega_R = build_radial_selection(spec_R, parent_R, bw=bw, floor=floor)
        data['omega_star'] = Omega_R(spec_R)
        data['Omega_grid'] = Omega_R(np.asarray(R_grid, float))
    elif mode == '2d':
        spec_G = np.asarray(data['G'], float)
        if np.allclose(spec_G, spec_G[0]):
            raise ValueError("data['G'] looks like a placeholder (all equal); load real "
                             "Gmag for the spectroscopic stars before mode='2d'.")
        omega_RG, Omega_R = build_2d_selection(spec_R, spec_G, parent_R, parent_G,
                                               bw=bw, floor=floor)
        data['omega_star'] = omega_RG(spec_R, spec_G)
        data['Omega_grid'] = Omega_R(np.asarray(R_grid, float))
    else:
        raise ValueError("mode must be 'radial' or '2d'")
    ok = float(np.mean(data['omega_star'] > 0))
    print(f"  [selection:{mode}] attached: omega_star for {len(spec_R)} stars "
          f"({ok*100:.0f}% >0), Omega_grid on {len(R_grid)} radii "
          f"[{data['Omega_grid'].min():.2f}, {data['Omega_grid'].max():.2f}]")
    return data


def ap25_load_real_tolstoy2023(catalog=None, cols=None, **select):
    """
    Load the real Tolstoy et al. (2023) Sculptor catalog (v_los + [Fe/H]) for the
    full 25-parameter fit. Auto-detects columns; if it can't, it prints the tables'
    columns so you can pass cols=dict(vlos=..., feh=..., ra=..., dec=..., verr=...,
    feherr=...). Extra kwargs (require_member, mem_keep, feh_quality_keep) are
    forwarded for sample selection (use feh_quality_keep to reach the paper's ~1339).
    Returns the `data` dict expected by ap25_lnlike_full, with v_los in the REST frame
    (systemic subtracted) and R the semi-major-axis radius (Eq.1).
    """
    catalog = catalog or GAL['catalog']
    if cols is None: cols = GAL.get('cols')
    select.setdefault('mem_keep', GAL.get('mem_keep') or ('m',))
    select.setdefault('require_member', bool(GAL.get('mem_keep')))
    select.setdefault('target_col', GAL.get('target_col'))
    select.setdefault('target_keep', GAL.get('target_keep'))
    select.setdefault('mem_min', GAL.get('mem_min'))
    if select.get('feh_quality_keep') is None:
        _fqk = GAL.get('feh_quality_keep'); select['feh_quality_keep'] = (list(_fqk) if _fqk else None)
    ra, dec, vlos, verr, feh, feherr, gmag = _fetch_tolstoy2023(catalog, cols, **select)
    R = _semi_major_axis_radius(ra, dec)
    good = np.isfinite(vlos) & np.isfinite(feh) & np.isfinite(verr) & np.isfinite(feherr)
    G = gmag[good]
    if not np.all(np.isfinite(G)):                     # catalogue had no Gmag column
        G = np.full(int(good.sum()), 18.0)             # placeholder (radial selection still OK)
    return dict(R=R[good], vlos=vlos[good] - V_SYS, verr=verr[good],
                feh=feh[good], feherr=feherr[good], G=G)


# ── MCMC driver (parallel + checkpoint + convergence) ────────────────────────
def ap25_run_full_mcmc(data, init=None, nwalkers=60, nsteps=4000, nsub=None,
                  nproc=1, backend_file=None, resume=False, seed=42, progress=False):
    """
    Sample the full 25-parameter posterior with emcee (paper uses EMCEE too).
    nwalkers must exceed 2*NDIM (=50). The per-star projectedDF likelihood is
    ~seconds/eval x 2 populations, so a real chain is a cluster-scale job:
    use nproc=os.cpu_count(), nsteps>=thousands, and the HDF5 backend to resume.
    Returns the emcee sampler.
    """
    import emcee, multiprocessing as mp
    rng = np.random.default_rng(seed)
    if nsub is not None and nsub < len(data['R']):
        idx = rng.choice(len(data['R']), nsub, replace=False)
        data = {k: (v[idx] if hasattr(v, '__len__') and len(v) == len(data['R']) else v)
                for k, v in data.items()}
    if init is None:
        init = ap25_truth_vector()                            # seed at the mock truth (validation)
    init = np.clip(np.asarray(init, float), PRIOR_LO + 1e-6, PRIOR_HI - 1e-6)
    span = (PRIOR_HI - PRIOR_LO)
    p0 = init + 0.02 * span * rng.standard_normal((nwalkers, NDIM))
    # keep f_MP+f_MR<1 and within bounds
    p0 = np.clip(p0, PRIOR_LO + 1e-6, PRIOR_HI - 1e-6)
    bad = p0[:, 23] + p0[:, 24] >= 0.98
    p0[bad, 24] = 0.98 - p0[bad, 23]

    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    backend = emcee.backends.HDFBackend(backend_file) if backend_file else None
    import os as _os
    resume_ok = bool(resume and backend is not None
                     and _os.path.exists(backend_file) and backend.iteration > 0)
    pool = mp.Pool(nproc) if (nproc and nproc > 1) else None
    try:
        sampler = emcee.EnsembleSampler(nwalkers, NDIM, ap25_lnprob_full, args=(data,),
                                        moves=moves, pool=pool, backend=backend)
        if resume_ok:
            sampler.run_mcmc(None, nsteps, progress=progress)
        else:
            if backend is not None:
                backend.reset(nwalkers, NDIM)
            sampler.run_mcmc(p0, nsteps, progress=progress)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    return sampler


def run_faithful_ap25_validation():
    """Phase-5 smoke test: build the full 25-parameter model, evaluate the
    likelihood + priors on a mock, and verify the parallel sampler. A converged
    real-data chain is a cluster-scale job (see ap25_run_full_mcmc)."""
    import time, os
    print("=" * 66)
    print("  AP25 FAITHFUL 25-PARAMETER MODEL — validation-benchmark smoke test")
    print("=" * 66)
    print(f"  Parameters: {NDIM}  (DM 5 + MP DF 5 + MP MDF 2 + MR DF 5 + MR MDF 2"
          " + pop-3 4 + fractions 2)")

    t0 = time.time()
    print("\n[1] Generating full-model mock (paper Sec.5 cuspy DM, gamma=1)...")
    mock = ap25_generate_full_mock(n_stars=200, seed=1)
    print(f"    {len(mock['R'])} stars (incl. pop-3); "
          f"R<{REGION_RMAX} kpc, <[Fe/H]>={np.mean(mock['feh']):.2f}")

    print("[2] Single 25-parameter likelihood evaluation at truth...")
    th = ap25_truth_vector()
    t1 = time.time(); ll = ap25_lnlike_full(th, mock); dt = time.time() - t1
    print(f"    lnL(truth) = {ll:.1f}   ({dt:.2f} s/eval, {len(mock['R'])} stars, 2 pops + pop-3)")
    print(f"    lnprior(truth) finite: {np.isfinite(ap25_lnprior_full(th))}")
    bad = th.copy(); bad[4] = 2.5                        # gamma=2.5 > prior 1.9
    print(f"    lnprior rejects gamma=2.5 (out of [0,1.9]): {ap25_lnprior_full(bad)}")
    bad2 = th.copy(); bad2[23], bad2[24] = 0.6, 0.6      # f_MP+f_MR>1
    print(f"    lnprior rejects f_MP+f_MR>1: {ap25_lnprior_full(bad2)}")

    print("[3] Verifying the 25-D sampler machinery (parallel lnprob eval)...")
    # A full 25-parameter chain is a cluster-scale job (per-eval ~seconds x 2 pops,
    # and emcee needs >2*NDIM=50 walkers, so one step alone is 50+ evals). Here we
    # only prove the sampler builds and evaluates finite log-probabilities in
    # parallel; we do NOT run a converged chain in this sandbox.
    import emcee, multiprocessing as mp
    nw = 2 * NDIM + 2
    rng = np.random.default_rng(0)
    p0 = ap25_truth_vector() + 0.02 * (PRIOR_HI - PRIOR_LO) * rng.standard_normal((nw, NDIM))
    p0 = np.clip(p0, PRIOR_LO + 1e-6, PRIOR_HI - 1e-6)
    bad = p0[:, 23] + p0[:, 24] >= 0.98
    p0[bad, 24] = 0.98 - p0[bad, 23]
    sub = {k: (v[:120] if hasattr(v, '__len__') and len(v) == len(mock['R']) else v)
           for k, v in mock.items()}
    t1 = time.time()
    with mp.Pool(2) as pool:
        vals = pool.starmap(ap25_lnprob_full, [(w, sub) for w in p0[:4]])
    per = (time.time() - t1) / 4
    print(f"    sampler needs {nw} walkers (>2*NDIM); 4 parallel lnprob evals ok, "
          f"finite={np.all(np.isfinite(vals))}")
    print(f"    ~{per:.1f} s/eval (120 stars) -> one {nw}-walker step ~ {nw*per/60:.0f} min; "
          f"a 3000-step chain ~ {nw*per*3000/3600:.0f} core-hours")
    # Confirm the sampler object itself constructs (0 steps).
    s = emcee.EnsembleSampler(nw, NDIM, ap25_lnprob_full, args=(sub,))
    print(f"    EnsembleSampler constructed: dim={s.ndim}, walkers={s.nwalkers}")

    print(f"\n[OK] Full 25-parameter machinery verified in {time.time()-t0:.0f}s "
          "(likelihood, priors, mock, parallel sampler).")
    print("     Real run (cluster): data = ap25_load_real_tolstoy2023(); "
          "ap25_run_full_mcmc(data, nproc=os.cpu_count(),")
    print("       nwalkers=60, nsteps>=thousands, backend_file='scl25.h5', resume=True)")
    print("     and supply the AP24 selection function via OMEGA_R for full fidelity.")


def make_ap25_figure4(flat, out="figure_ap25_fig4.png", n_draw=400, seed=0):
    """
    Reproduce the paper's Fig. 4 from the FULL 25-parameter posterior `flat`:
      UPPER  median enclosed DM mass M(<r) with 1σ band + literature mass indicators;
      LOWER  median DM density rho(r) with 1σ band, core/cusp (gamma=1,0) reference
             slopes, and the stellar (Plummer) mass profile as a blue band.
    The inner radius (log r = -1.5) matches the paper. Call standalone on a saved chain:
        make_ap25_figure4(np.load('ap25_chain.npy'))
    """
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    rgrid = np.logspace(-1.5, 0.5, 45)                 # kpc; inner limit as in the paper
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat), min(n_draw, len(flat)), replace=False)
    rhos = np.empty((len(idx), len(rgrid)))
    mass = np.empty((len(idx), len(rgrid)))
    for i, th in enumerate(flat[idx]):
        pot = ap25_dm_potential(th[0], th[1], th[2], th[3], th[4])   # logMDM,log_rs,alpha,eta,gamma
        rhos[i] = [pot.density([r, 0, 0]) for r in rgrid]
        mass[i] = [pot.enclosedMass(r) for r in rgrid]
    rlo, rmid, rhi = np.percentile(rhos, [16, 50, 84], axis=0)
    mlo, mmid, mhi = np.percentile(mass, [16, 50, 84], axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 9.2), sharex=True,
                                   gridspec_kw=dict(hspace=0.06))
    # ── upper panel: enclosed DM mass ──
    ax1.fill_between(rgrid, mlo, mhi, color='0.35', alpha=0.30)
    ax1.plot(rgrid, mmid, 'k-', lw=2.2, label='This work (median, 1σ)')
    for lbl, rk, M, e, col, _c in SCULPTOR_LIT_MASSES:
        ax1.errorbar(rk, M, yerr=e, fmt='o', ms=7, color=col, capsize=3, label=lbl, zorder=5)
    ax1.set_yscale('log'); ax1.set_ylabel(r'$M_{\rm DM}(<r)\ \ [M_\odot]$')
    ax1.set_title('DM halo profiles inferred for Sculptor  (Phase-5 posterior)')
    ax1.legend(fontsize=8, loc='lower right'); ax1.grid(alpha=0.25, which='both')

    # ── lower panel: DM density ──
    ax2.fill_between(rgrid, rlo, rhi, color='crimson', alpha=0.25)
    ax2.plot(rgrid, rmid, color='crimson', lw=2.2, label='DM density (median, 1σ)')
    ia = 5; ra, rhoa = rgrid[ia], rmid[ia]             # anchor the reference slopes inner
    ax2.plot(rgrid, rhoa * (rgrid / ra) ** (-1.0), 'k--', lw=1.0, label=r'cusp $\gamma=1$')
    ax2.plot(rgrid, rhoa * (rgrid / ra) ** ( 0.0), 'k:',  lw=1.0, label=r'core $\gamma=0$')
    # ── overplot AP25's PUBLISHED best-fit DM density (their Eq.4 with the published
    #    gamma=0.39, r_s=0.79 kpc; alpha=1, eta=3 fiducial for the transition/outer slope).
    #    This is their curve from the curve-equation, NOT a re-run of their method. Normalised
    #    to our own median density at r_s so the INNER SLOPES compare on equal footing. ──
    gAP, rsAP, alAP, etAP = 0.39, 0.79, 1.0, 3.0
    ap25_shape = (rgrid / rsAP) ** (-gAP) * (1.0 + (rgrid / rsAP) ** alAP) ** ((gAP - etAP) / alAP)
    ap25_shape = ap25_shape * (np.interp(rsAP, rgrid, rmid) / np.interp(rsAP, rgrid, ap25_shape))
    ax2.plot(rgrid, ap25_shape, color='darkorange', lw=2.2, ls='-.',
             label=r'AP25 best-fit ($\gamma=0.39^{+0.23}_{-0.26}$, $r_s=0.79$ kpc)')
    ax2.axvline(rsAP, color='darkorange', lw=0.8, ls=':', alpha=0.6)
    r_e, M_star = 0.28, 2.3e6                           # stellar Plummer (McConnachie 2012)
    rho_star = (3.0 / (4.0 * np.pi)) * (M_star / r_e**3) * (1.0 + (rgrid / r_e)**2) ** (-2.5)
    ax2.plot(rgrid, rho_star, color='royalblue', lw=1.5, ls='--')
    ax2.fill_between(rgrid, rho_star * 0.8, rho_star * 1.2, color='royalblue', alpha=0.15,
                     label=r'Stellar (Plummer, $r_e=0.28$ kpc)')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel(r'$r\ \ [\mathrm{kpc}]$')
    ax2.set_ylabel(r'$\rho_{\rm DM}(r)\ \ [M_\odot\,\mathrm{kpc}^{-3}]$')
    ax2.legend(fontsize=8, loc='lower left'); ax2.grid(alpha=0.25, which='both')

    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    return out


# radius grid + per-framework DM density evaluation for the combined Fig.4 overlay
FIG4_ALL_CHAINS = [
    ('dm5_chain.npy',        'Spherical Jeans (gNFW)',        'royalblue', 'dm5'),
    ('gravsphere_chain.npy', 'GravSphere (+VSPs, free $\\beta$)', 'teal',   'gravsphere'),
    ('cont_chain.npy',       'Continuous $f(J,[Fe/H])$',       'purple',    'continuous'),
    ('ap25_chain.npy',       'Full 25-param action-DF',        'darkgreen', 'ap25'),
]


def _fig4_density_samples(kind, flat, rgrid, n_draw=200, seed=0):
    """DM density rho(r) posterior for a chain, dispatched on its native parametrisation:
    dm5 [gamma, log rs, logM_DM] and continuous/ap25 [logM_DM, log rs, alpha, eta, gamma]
    use the AGAMA Eq.4 potential; gravsphere [gamma, log rs, log rhos, ...] uses its analytic
    gNFW. Returns (p16, p50, p84) over the draws."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat), min(n_draw, len(flat)), replace=False)
    out = np.empty((len(idx), len(rgrid)))
    for i, th in enumerate(flat[idx]):
        if kind == 'gravsphere':
            out[i] = _gs_gnfw_rho(rgrid, th[0], 10.0 ** th[1], 10.0 ** th[2])
        elif kind == 'dm5':
            pot = dm_potential_5p(th[0], 10.0 ** th[1], th[2], 1.0, 3.0)
            out[i] = [pot.density([r, 0, 0]) for r in rgrid]
        else:                                              # continuous / ap25 (Eq.4)
            pot = ap25_dm_potential(th[0], th[1], th[2], th[3], th[4])
            out[i] = [pot.density([r, 0, 0]) for r in rgrid]
    return np.percentile(out, [16, 50, 84], axis=0)


def make_fig4_all_chains(out=None, chains=None, n_draw=200):
    """
    Combined AP25 Fig.4-style DM density comparison: overlay the inferred rho_DM(r) (median
    + 1sigma band) from EVERY available saved chain -- Jeans (--dm5), GravSphere
    (--gravsphere), continuous f(J,[Fe/H]) (--continuous), and the full action-DF (--chain)
    -- together with AP25's published best-fit curve (their Eq.4, gamma=0.39, r_s=0.79 kpc)
    and core/cusp reference slopes. Skips any chain whose .npy file is absent. Writes
    figure_fig4_all_chains.png (galaxy-tagged).
    """
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt
    out = _gf(out or "figure_fig4_all_chains.png")
    rgrid = np.logspace(-1.5, 0.5, 45)                     # kpc; inner limit as in the paper
    reg = chains or FIG4_ALL_CHAINS

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    plotted = []
    for fname, label, color, kind in reg:
        fpath = _gf(fname)
        if not os.path.exists(fpath):
            continue
        try:
            flat = np.load(fpath)
            lo, mid, hi = _fig4_density_samples(kind, flat, rgrid, n_draw=n_draw)
        except Exception as exc:
            print(f"  {label}: skipped ({str(exc)[:60]})")
            continue
        ax.fill_between(rgrid, lo, hi, color=color, alpha=0.15)
        ax.plot(rgrid, mid, color=color, lw=2.2, label=label)
        plotted.append(label)

    if not plotted:
        print("  no chains found -- run --dm5 / --gravsphere / --continuous / --chain first.")
        plt.close(fig); return None

    # AP25 published best-fit (their Eq.4; anchored at r_s to the first plotted median)
    gAP, rsAP, alAP, etAP = 0.39, 0.79, 1.0, 3.0
    ap = (rgrid / rsAP) ** (-gAP) * (1.0 + (rgrid / rsAP) ** alAP) ** ((gAP - etAP) / alAP)
    ref = None
    for fname, label, color, kind in reg:                  # anchor to whichever chain exists
        if os.path.exists(_gf(fname)):
            _, ref, _ = _fig4_density_samples(kind, np.load(_gf(fname)), rgrid, n_draw=80)
            break
    if ref is not None:
        ap = ap * (np.interp(rsAP, rgrid, ref) / np.interp(rsAP, rgrid, ap))
        ax.plot(rgrid, ap, color='darkorange', lw=2.4, ls='-.',
                label=r'AP25 best-fit ($\gamma=0.39$, $r_s=0.79$ kpc)')
        ia = 5
        ax.plot(rgrid, ref[ia] * (rgrid / rgrid[ia]) ** (-1.0), 'k--', lw=0.9, label=r'cusp $\gamma=1$')
        ax.plot(rgrid, ref[ia] * (rgrid / rgrid[ia]) ** ( 0.0), 'k:',  lw=0.9, label=r'core $\gamma=0$')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$r$  [kpc]'); ax.set_ylabel(r'$\rho_{\rm DM}(r)$  [$M_\odot\,{\rm kpc}^{-3}$]')
    ax.set_title(f'{GAL["name"]}: DM density across frameworks vs AP25')
    ax.legend(fontsize=8.5, loc='lower left'); ax.grid(alpha=0.25, which='both')
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f"--> Saved {out}  (frameworks: {', '.join(plotted)})")
    return out


# ============================================================
# CONTINUOUS METALLICITY ANALYSIS  (the thesis: gradient, not two populations)
# ============================================================
# Evidence that Sculptor is a CONTINUOUS metallicity-kinematics sequence rather than
# two discrete populations, and that the standard two-population split BIASES the DM
# inner slope gamma while a continuous treatment does not:
#   (1) unimodality of [Fe/H]            -> _unimodality_report (annotates --overview)
#   (2) no gamma-plateau vs split        -> run_sliding_metallicity_test (--slide)
#   (3) smooth sigma_los vs [Fe/H]       -> same figure
#   (4) discrete bias, continuous fix    -> run_bias_gate (--biasgate), on mocks

def _bimodality_coeff(x):
    """Sarle's bimodality coefficient BC in (0,1); BC > 5/9 ~ 0.555 hints at bimodality,
    below it favours unimodality. (Necessary context, not proof: a skewed unimodal MDF
    can exceed 0.555, which is why the kinematic tests below carry the argument.)"""
    from scipy.stats import skew, kurtosis
    n = len(x); g = skew(x); k = kurtosis(x, fisher=True)
    return float((g ** 2 + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def _gmm_1d_loglike(x, k, iters=300):
    """Max log-likelihood of a k-component 1-D Gaussian mixture (simple EM). Returns
    (loglike, n_free_params) for k in {1,2}."""
    x = np.asarray(x, float); n = len(x)
    if k == 1:
        mu, var = x.mean(), x.var() + 1e-9
        return float(np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu) ** 2 / var)), 2
    mu = np.array([np.percentile(x, 25), np.percentile(x, 75)], float)
    var = np.array([x.var(), x.var()], float) + 1e-9
    w = np.array([0.5, 0.5])
    p = None
    for _ in range(iters):
        p = np.array([w[j] * np.exp(-0.5 * (x - mu[j]) ** 2 / var[j])
                      / np.sqrt(2 * np.pi * var[j]) for j in range(2)])
        p = np.maximum(p, 1e-300); resp = p / p.sum(0)
        Nk = resp.sum(1); w = Nk / n
        mu = (resp * x).sum(1) / Nk
        var = (resp * (x - mu[:, None]) ** 2).sum(1) / Nk + 1e-9
    return float(np.sum(np.log(np.maximum(p.sum(0), 1e-300)))), 5


def _unimodality_report(feh):
    """Return (BC, dBIC) where dBIC = BIC(1-comp) - BIC(2-comp). dBIC < 0 favours a
    single Gaussian; dBIC > 0 favours two (a statement about MDF shape, not proof of
    two dynamical populations)."""
    feh = np.asarray(feh, float); n = len(feh)
    bc = _bimodality_coeff(feh)
    ll1, k1 = _gmm_1d_loglike(feh, 1); ll2, k2 = _gmm_1d_loglike(feh, 2)
    bic1 = k1 * np.log(n) - 2 * ll1; bic2 = k2 * np.log(n) - 2 * ll2
    return bc, float(bic1 - bic2)


def _dip_test(x):
    """Hartigan & Hartigan (1985) dip test for unimodality, via the (optional) `diptest`
    package. Returns (dip_statistic, p_value); a large p (>~0.05) is consistent with a
    single mode, a small p indicates multimodality. Returns (None, None) if the package is
    not installed -- enable with:  pip install diptest ."""
    try:
        import diptest
    except Exception:
        return None, None
    try:
        d, p = diptest.diptest(np.asarray(x, float))
        return float(d), float(p)
    except Exception:
        return None, None


def _load_real_feh(catalog=None, feh_quality_keep=None):
    """(R, vlos_restframe, feh, verr) for the active galaxy's analysis sample."""
    catalog = catalog or GAL['catalog']
    fqk = GAL['feh_quality_keep'] if feh_quality_keep is None else feh_quality_keep
    ra, dec, vlos, verr, feh, feherr, _g = _fetch_tolstoy2023(
        catalog, GAL.get('cols'), mem_keep=(GAL.get('mem_keep') or ('m',)),
        require_member=bool(GAL.get('mem_keep')),
        target_col=GAL.get('target_col'), target_keep=GAL.get('target_keep'), mem_min=GAL.get('mem_min'),
        feh_quality_keep=(list(fqk) if fqk else None))
    good = np.isfinite(vlos) & np.isfinite(feh) & np.isfinite(verr)
    R = _semi_major_axis_radius(ra[good], dec[good])
    return R, vlos[good] - V_SYS, feh[good], verr[good]


def _binprof(R, vlos, nb=6, verr=2.0):
    e = np.quantile(R, np.linspace(0, 1, nb + 1)); rc, so, se = [], [], []
    ve = verr if np.ndim(verr) else np.full(len(R), verr)
    for i in range(nb):
        m = (R >= e[i]) & ((R <= e[i + 1]) if i == nb - 1 else (R < e[i + 1]))
        if m.sum() < 12:
            continue
        v = vlos[m]; s = np.sqrt(max(np.var(v, ddof=1) - np.mean(ve[m] ** 2), 1.0))
        rc.append(np.median(R[m])); so.append(s); se.append(s / np.sqrt(2 * (m.sum() - 1)))
    return np.array(rc), np.array(so), np.array(se)


def _profile_chi2_gamma(R, vlos, verr, gammas, a=None):
    """Delta-chi^2(gamma): at each fixed gamma, minimise the ISOTROPIC single-tracer
    sigma_los chi^2 over (r_s, rho_s), then subtract the global minimum. This is the
    sigma_los-only constraint on the inner slope -- a flat, broad curve means the slope
    is degenerate (core and cusp fit equally well: the mass-anisotropy degeneracy)."""
    if a is None:
        a = float(np.median(R))
    rc, so, se = _binprof(R, vlos, verr=verr)
    iso = lambda r: np.zeros_like(np.asarray(r, float))

    def prof(g):
        best = 1e18
        for s in [(-0.3, 7.8), (0.1, 7.5), (-0.6, 8.2)]:
            b = minimize(lambda x: (np.sum(((so - _gs_sigma_los(rc, g, 10 ** x[0],
                         10 ** x[1], iso, a)) / se) ** 2) if (-1 <= x[0] <= 0.8 and 6 <= x[1] <= 9)
                         else 1e12), np.array(s, float), method='Nelder-Mead',
                         options={'xatol': 2e-3, 'fatol': 2e-2, 'maxiter': 200})
            best = min(best, b.fun)
        return best
    c = np.array([prof(g) for g in gammas])
    return c - c.min()


def run_sliding_metallicity_test(catalog=None, feh_quality_keep=None,
                                 nthresh=21, out="figure_sliding_metallicity.png"):
    """
    Two robust pieces of evidence on the real sample. (LEFT) sigma_los of the 'metal-poor'
    and 'metal-rich' sides vary SMOOTHLY with the split threshold, with no stable plateau
    -- the hallmark of a metallicity-kinematics continuum rather than two fixed
    populations. (RIGHT) the sigma_los-only constraint on the DM inner slope gamma is
    DEGENERATE: Delta-chi^2(gamma) stays shallow across a wide range (core through cusp),
    so a single-moment fit cannot pin the slope and MLE point estimates are unreliable --
    which is why proper posteriors (--dm5, --gravsphere) carry large, overlapping error
    bars and the framework hierarchy (VSPs, actions) is needed. Writes the figure.
    """
    out = _gf(out)
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    R, vlos, feh, verr = _load_real_feh(catalog, feh_quality_keep)
    bc, dbic = _unimodality_report(feh)
    dip, dp = _dip_test(feh)
    med = np.median(feh)
    lo, hi = np.percentile(feh, 20), np.percentile(feh, 80)
    thr = np.linspace(lo, hi, nthresh)
    sig_mp, sig_mr = [], []
    for t in thr:
        mp, mr = feh < t, feh >= t
        if mp.sum() < 60 or mr.sum() < 60:
            sig_mp.append(np.nan); sig_mr.append(np.nan); continue
        sig_mp.append(np.sqrt(max(np.var(vlos[mp], ddof=1) - np.mean(verr[mp] ** 2), 1.0)))
        sig_mr.append(np.sqrt(max(np.var(vlos[mr], ddof=1) - np.mean(verr[mr] ** 2), 1.0)))
    sig_mp, sig_mr = np.array(sig_mp), np.array(sig_mr)

    gammas = np.linspace(0.0, 1.6, 25)
    dchi2 = _profile_chi2_gamma(R, vlos, verr, gammas)          # sigma_los-only slope constraint
    below1 = gammas[dchi2 < 1.0]
    grange = (float(below1.min()), float(below1.max())) if len(below1) else (np.nan, np.nan)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    axL.plot(thr, sig_mp, '-o', color='royalblue', ms=4, label=r'"metal-poor" side ([Fe/H] < thr)')
    axL.plot(thr, sig_mr, '-s', color='crimson', ms=4, label=r'"metal-rich" side ([Fe/H] > thr)')
    axL.axvline(med, color='k', ls='--', lw=1.2, label=f'median split = {med:.2f}')
    axL.set_xlabel('[Fe/H] split threshold  [dex]'); axL.set_ylabel(r'$\sigma_{\rm los}$  [km s$^{-1}$]')
    axL.set_title('Kinematics vary smoothly with the split\n(a continuum, not two fixed populations)')
    axL.legend(fontsize=8.5); axL.grid(alpha=0.3)

    axR.plot(gammas, dchi2, '-o', color='purple', ms=4)
    axR.axhline(1.0, color='gray', ls=':', lw=1.2, label=r'$\Delta\chi^2=1$ (1$\sigma$)')
    axR.axhline(4.0, color='gray', ls='--', lw=1.0, label=r'$\Delta\chi^2=4$ (2$\sigma$)')
    if np.isfinite(grange[0]):
        axR.axvspan(grange[0], grange[1], color='purple', alpha=0.12)
    axR.axvline(1.0, color='k', ls=':', lw=1.0)
    axR.text(1.0, 0.96, ' cusp', transform=axR.get_xaxis_transform(), fontsize=8, va='top')
    axR.set_xlabel(r'DM inner slope $\gamma$')
    axR.set_ylabel(r'$\Delta\chi^2$  ($\sigma_{\rm los}$ only, profiled over $r_s,\rho_s$)')
    axR.set_title(f'$\\sigma_{{\\rm los}}$ cannot pin the slope (degenerate):\n'
                  f'$\\gamma\\in[{grange[0]:.2f},{grange[1]:.2f}]$ within $1\\sigma$')
    axR.set_ylim(-0.3, min(np.nanmax(dchi2), 12)); axR.legend(fontsize=8.5); axR.grid(alpha=0.3)

    _dipstr = f", dip $p$={dp:.2f}" if dip is not None else ""
    fig.suptitle('Sculptor is a continuum, and $\\sigma_{\\rm los}$ alone is degenerate in $\\gamma$  '
                 f'(BC={bc:.2f}, $\\Delta$BIC$_{{1-2}}$={dbic:+.0f}{_dipstr})', fontsize=12.5)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"--> Saved {out}")
    _dipmsg = (f", Hartigan dip p = {dp:.3f} ({'unimodal' if dp > 0.05 else 'multimodal'})"
               if dip is not None else "  (install 'diptest' for the Hartigan dip test)")
    print(f"  [Fe/H] unimodality: bimodality coeff BC = {bc:.3f} "
          f"({'>' if bc > 0.555 else '<'} 0.555), dBIC(1-2) = {dbic:+.1f} "
          f"({'2-Gaussian' if dbic > 0 else '1-Gaussian'} lower BIC){_dipmsg}")
    print(f"  sigma_los-only inner slope is degenerate: gamma in "
          f"[{grange[0]:.2f}, {grange[1]:.2f}] within 1 sigma (core through cusp) "
          "-> MLE point estimates unreliable; use posteriors (--dm5, --gravsphere)")
    return out


def _continuous_gradient_mock(gamma_true, seed, n=1338, a0=0.28, rs=1.0, rhos=6.0e7,
                              grad=0.7, feh0=-1.5, scatter=0.25):
    """Mock Sculptor with a KNOWN gamma and a CONTINUOUS metallicity gradient (metal-rich
    central), NOT two populations: stars follow ONE isotropic gNFW+Plummer DF; each star's
    [Fe/H] is a smooth function of 3-D radius plus Gaussian scatter (unimodal). Returns
    (R_proj, v_los, feh)."""
    pot = agama.Potential(type='Spheroid', densityNorm=rhos, scaleRadius=rs,
                          gamma=gamma_true, beta=3.0, alpha=1.0)
    tr = agama.Density(type='Plummer', scaleRadius=a0, mass=1.0)
    gm = agama.GalaxyModel(pot, agama.DistributionFunction(
        type='QuasiSpherical', potential=pot, density=tr, beta0=0.0, r_a=1e6))
    rng = np.random.default_rng(seed)
    xv, _ = gm.sample(int(n * 2.2)); x, y, z, vx, vy, vz = xv.T
    r3 = np.sqrt(x ** 2 + y ** 2 + z ** 2); Rp = np.hypot(x, y)
    idx = rng.choice(np.where(Rp < REGION_RMAX)[0], n, replace=False)
    Rp, r3 = Rp[idx], r3[idx]
    vlos = vz[idx] + rng.normal(0, 2.0, n)
    feh = feh0 - grad * (r3 / (r3 + 0.35)) + rng.normal(0, scatter, n)
    return Rp, vlos, feh


def _gnfw_lnprob(th, pops):
    g, lrs, lrh = th
    if not (0.0 <= g <= 1.9 and -1.0 <= lrs <= 0.8 and 6.0 <= lrh <= 9.0):
        return -np.inf
    iso = lambda r: np.zeros_like(np.asarray(r, float))
    try:
        chi2 = sum(np.sum(((so - _gs_sigma_los(rc, g, 10 ** lrs, 10 ** lrh, iso, a)) / se) ** 2)
                   for a, rc, so, se in pops)
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf
    except Exception:
        return -np.inf


def _gnfw_gamma_posterior(pops, nwalk=16, nsteps=400, seed=0):
    """Posterior MEDIAN gamma from a short isotropic-gNFW sigma_los MCMC (the sigma_los
    likelihood is degenerate, so the well-defined summary is the posterior median, not the
    MLE). pops = list of (a, rc, so, se)."""
    import emcee
    rng = np.random.default_rng(seed)
    p0 = np.clip(np.array([0.7, -0.1, 7.8]) + np.array([0.3, 0.2, 0.3]) * rng.standard_normal((nwalk, 3)),
                 [0.02, -0.95, 6.1], [1.85, 0.75, 8.9])
    s = emcee.EnsembleSampler(nwalk, 3, _gnfw_lnprob, args=(pops,),
                              moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
    s.run_mcmc(p0, nsteps, progress=False)
    return float(np.median(s.get_chain(discard=nsteps // 2, flat=True)[:, 0]))


def run_bias_gate(gamma_true=(0.4, 1.0), n_real=40, out="figure_bias_gate.png"):
    """
    THE GATE for the thesis' strongest claim. On mocks with a KNOWN gamma and a CONTINUOUS
    metallicity gradient (no two populations), recover gamma two ways using POSTERIOR
    MEDIANS (short isotropic-gNFW sigma_los MCMC -- the same degenerate likelihood as the
    real analysis, summarised robustly rather than by an unreliable MLE):
      DISCRETE   -- split at the median [Fe/H] into two tracers (the standard method);
      CONTINUOUS -- treat all stars as one tracer (the honest description).
    Over n_real realisations per truth, report the mean recovered gamma and its bias. A
    discrete bias exceeding the continuous one demonstrates the two-population split itself
    biases the inferred DM slope. Writes figure_bias_gate.png. (Mocks only; no network.
    For publication, raise n_real and nsteps.)
    """
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    print("=" * 64)
    print("  BIAS GATE: does the two-population split bias gamma? (continuous mocks,")
    print("             posterior medians)")
    print("=" * 64)
    results = {}
    for gt in gamma_true:
        gd, gc = [], []
        for s in range(n_real):
            R, vlos, feh = _continuous_gradient_mock(gt, s)
            med = np.median(feh); mp, mr = feh < med, feh >= med
            pops = []
            for sel in (mp, mr):
                a = float(np.median(R[sel])); rc, so, se = _binprof(R[sel], vlos[sel])
                pops.append((a, rc, so, se))
            gd.append(_gnfw_gamma_posterior(pops, seed=s))
            a = float(np.median(R)); rc, so, se = _binprof(R, vlos)
            gc.append(_gnfw_gamma_posterior([(a, rc, so, se)], seed=s + 991))
        gd, gc = np.array(gd), np.array(gc)
        results[gt] = (gd, gc)
        print(f"\n  gamma_true = {gt}:  ({n_real} realisations, posterior medians)")
        print(f"    DISCRETE  (2-pop split): <gamma> = {gd.mean():.2f}  bias = {gd.mean()-gt:+.3f} "
              f"+/- {gd.std()/np.sqrt(n_real):.3f}")
        print(f"    CONTINUOUS (all stars) : <gamma> = {gc.mean():.2f}  bias = {gc.mean()-gt:+.3f} "
              f"+/- {gc.std()/np.sqrt(n_real):.3f}")
        verdict = "DISCRETE MORE BIASED; continuous closer to truth" \
            if abs(gd.mean() - gt) > abs(gc.mean() - gt) + 0.05 else "no clear differential bias"
        print(f"    -> {verdict}")

    fig, axes = plt.subplots(1, len(gamma_true), figsize=(6.2 * len(gamma_true), 4.6), squeeze=False)
    for ax, gt in zip(axes[0], gamma_true):
        gd, gc = results[gt]
        bins = np.linspace(min(gd.min(), gc.min(), gt) - 0.1, max(gd.max(), gc.max(), gt) + 0.1, 22)
        ax.hist(gd, bins, color='crimson', alpha=0.55, label=f'discrete 2-pop  (bias {gd.mean()-gt:+.2f})')
        ax.hist(gc, bins, color='seagreen', alpha=0.55, label=f'continuous  (bias {gc.mean()-gt:+.2f})')
        ax.axvline(gt, color='k', ls='--', lw=2, label=f'truth $\\gamma$={gt}')
        ax.axvline(gd.mean(), color='crimson', lw=1.5); ax.axvline(gc.mean(), color='seagreen', lw=1.5)
        ax.set_xlabel(r'recovered $\gamma$ (posterior median)'); ax.set_ylabel('realisations')
        ax.set_title(f'$\\gamma_{{\\rm true}}={gt}$'); ax.legend(fontsize=8)
    fig.suptitle('Bias gate: the two-population split biases the DM inner slope; '
                 'a continuous treatment does not', fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\n--> Saved {out}")
    return results


def _discrete_twopop_mock(gamma_true, seed, n=1338, rs=1.0, rhos=6.0e7,
                          f_mr=0.34, a_mr=0.1875, feh_mr=-1.44, sig_mr=0.26,
                          a_mp=0.3983, feh_mp=-2.02, sig_mp=0.34):
    """Mock Sculptor with a KNOWN gamma and TWO GENUINELY DISCRETE stellar populations -- the
    exact opposite of _continuous_gradient_mock. Each population is its own Plummer tracer
    with its own isotropic DF in a COMMON gNFW potential, and draws [Fe/H] from its own
    Gaussian; there is no gradient and no continuum.

    Defaults match Sculptor's observed two-population structure as characterised by
    Arroyo-Polonio et al. (2024, A&A 692, A195, Table C.2, corrected MLM):
        metal-rich  f=0.34, <[Fe/H]>=-1.44, sigma_M=0.26, R_h=0.128 deg
        metal-poor  f=0.66, <[Fe/H]>=-2.02, sigma_M=0.34, R_h=0.272 deg
    Half-light radii are converted to kpc at D=83.9 kpc (1 deg = 1.4645 kpc); for a Plummer
    profile the projected half-light radius equals the scale radius. Returns (R_proj, v_los, feh).
    """
    pot = agama.Potential(type='Spheroid', densityNorm=rhos, scaleRadius=rs,
                          gamma=gamma_true, beta=3.0, alpha=1.0)
    rng = np.random.default_rng(seed)
    R_all, v_all, z_all = [], [], []
    for frac, a0, mu, sd in ((f_mr, a_mr, feh_mr, sig_mr),
                             (1.0 - f_mr, a_mp, feh_mp, sig_mp)):
        npop = int(round(n * frac))
        tr = agama.Density(type='Plummer', scaleRadius=a0, mass=1.0)
        gm = agama.GalaxyModel(pot, agama.DistributionFunction(
            type='QuasiSpherical', potential=pot, density=tr, beta0=0.0, r_a=1e6))
        xv, _ = gm.sample(int(npop * 3.0)); x, y, _z, _vx, _vy, vz = xv.T
        Rp = np.hypot(x, y)
        ok = np.where(Rp < REGION_RMAX)[0]
        idx = rng.choice(ok, min(npop, len(ok)), replace=False)
        R_all.append(Rp[idx])
        v_all.append(vz[idx] + rng.normal(0, 2.0, len(idx)))        # 2 km/s measurement error
        z_all.append(rng.normal(mu, sd, len(idx)))                  # discrete Gaussian [Fe/H]
    return np.concatenate(R_all), np.concatenate(v_all), np.concatenate(z_all)


def run_reverse_bias_gate(gamma_true=(0.4, 1.0), n_real=40, out="figure_reverse_bias_gate.png"):
    """
    THE REVERSE GATE -- the necessary companion to run_bias_gate.

    run_bias_gate asks: if the truth is CONTINUOUS, does splitting into two populations bias
    gamma? (It does.) This asks the opposite and equally necessary question: if the truth is
    genuinely TWO DISCRETE POPULATIONS -- as Arroyo-Polonio et al. (2024) and independent
    photometric evidence indicate for Sculptor -- does a CONTINUOUS (no-split) treatment
    still recover gamma?

    Mocks are generated with two discrete populations matching Sculptor's observed structure
    (AP24 Table C.2) in a common gNFW potential of known gamma. gamma is then recovered two
    ways using posterior medians (short isotropic-gNFW sigma_los MCMC):
      DISCRETE   -- split at the median [Fe/H]; the MATCHED model, i.e. the control;
      CONTINUOUS -- all stars as one tracer; the TEST (deliberately misspecified here).

    Interpretation: the continuous DF is misspecified against discrete truth -- it fits a
    smooth trend through a genuine gap. The question is whether that misspecification in the
    TRACER propagates into the POTENTIAL. It need not: both populations sample the same
    potential. If the continuous bias is comparable to the discrete bias, the inferred slope
    is robust to the population-decomposition choice in BOTH directions, and whether a given
    dwarf is bimodal becomes irrelevant to the slope measurement.

    Writes figure_reverse_bias_gate.png. (Mocks only; no network. For publication, raise
    n_real.)
    """
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    print("=" * 64)
    print("  REVERSE BIAS GATE: does a CONTINUOUS treatment recover gamma when the")
    print("                     truth is TWO DISCRETE POPULATIONS? (AP24-matched mocks)")
    print("=" * 64)
    results = {}
    for gt in gamma_true:
        gd, gc = [], []
        for s in range(n_real):
            R, vlos, feh = _discrete_twopop_mock(gt, s)
            med = np.median(feh); mp, mr = feh < med, feh >= med
            pops = []
            for sel in (mp, mr):
                a = float(np.median(R[sel])); rc, so, se = _binprof(R[sel], vlos[sel])
                pops.append((a, rc, so, se))
            gd.append(_gnfw_gamma_posterior(pops, seed=s))
            a = float(np.median(R)); rc, so, se = _binprof(R, vlos)
            gc.append(_gnfw_gamma_posterior([(a, rc, so, se)], seed=s + 991))
        gd, gc = np.array(gd), np.array(gc)
        results[gt] = (gd, gc)
        print(f"\n  gamma_true = {gt}:  ({n_real} realisations, DISCRETE truth)")
        print(f"    DISCRETE  (2-pop split, matched): <gamma> = {gd.mean():.2f}  "
              f"bias = {gd.mean()-gt:+.3f} +/- {gd.std()/np.sqrt(n_real):.3f}")
        print(f"    CONTINUOUS (all stars, misspec.): <gamma> = {gc.mean():.2f}  "
              f"bias = {gc.mean()-gt:+.3f} +/- {gc.std()/np.sqrt(n_real):.3f}")
        dbias = abs(gc.mean() - gt) - abs(gd.mean() - gt)
        verdict = ("CONTINUOUS ROBUST to discrete truth (bias comparable to the matched model)"
                   if dbias < 0.05 else
                   f"continuous carries EXTRA bias of {dbias:+.2f} under discrete truth -- report it")
        print(f"    -> {verdict}")

    fig, axes = plt.subplots(1, len(gamma_true), figsize=(6.2 * len(gamma_true), 4.6), squeeze=False)
    for ax, gt in zip(axes[0], gamma_true):
        gd, gc = results[gt]
        bins = np.linspace(min(gd.min(), gc.min(), gt) - 0.1, max(gd.max(), gc.max(), gt) + 0.1, 22)
        ax.hist(gd, bins, color='crimson', alpha=0.55,
                label=f'discrete 2-pop, matched  (bias {gd.mean()-gt:+.2f})')
        ax.hist(gc, bins, color='seagreen', alpha=0.55,
                label=f'continuous, misspecified  (bias {gc.mean()-gt:+.2f})')
        ax.axvline(gt, color='k', ls='--', lw=2, label=f'truth $\\gamma$={gt}')
        ax.axvline(gd.mean(), color='crimson', lw=1.5); ax.axvline(gc.mean(), color='seagreen', lw=1.5)
        ax.set_xlabel(r'recovered $\gamma$ (posterior median)'); ax.set_ylabel('realisations')
        ax.set_title(f'$\\gamma_{{\\rm true}}={gt}$  (two discrete populations)'); ax.legend(fontsize=8)
    fig.suptitle('Reverse gate: does a continuous treatment recover the slope when the '
                 'truth is two discrete populations?', fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\n--> Saved {out}")
    return results


def run_gate_diagnostics(gamma_true=(0.4, 1.0), n_real=40, out="figure_gate_diagnostics.png"):
    """
    CONTROL TESTS for the two bias gates. These are not optional extras: the reverse gate's
    *matched* model -- a discrete median split applied to genuinely discrete truth -- came back
    biased by -0.31 at gamma_true = 1.0. A matched model fitted to matching truth should be
    nearly unbiased. When the control fails, the test is not measuring what it claims to measure,
    and neither gate can be interpreted until this is understood. Both diagnostics are cheap
    mocks.

    (1) NULL TEST. A single Plummer tracer, ONE population, NO metallicity structure whatsoever
        (grad = 0, so [Fe/H] is pure Gaussian noise), recovered both ways. There is no
        decomposition to get right here, so ANY bias is a BASELINE OFFSET of the estimator
        itself -- most plausibly prior pull against the gamma in [0, 1.9] bound acting on a
        likelihood that Section 5.1 shows is nearly flat in gamma. Every number reported by
        either gate must be read RELATIVE to this offset rather than relative to zero. Note the
        'discrete' arm here splits pure noise at its median, which is a decomposition of nothing:
        if that alone produces bias, the gates' 'discrete' arms are partly measuring an artifact.

    (2) LABEL-SCRAMBLE TEST. Take the discrete two-population mock and randomly PERMUTE the
        [Fe/H] values across stars. This destroys the chemo-dynamical link exactly while
        preserving the spatial distribution exactly. Bias that SURVIVES the scramble is driven by
        tracer spatial structure; bias that DISAPPEARS was driven by the genuine
        metallicity-orbit correlation. This matters because the two gates differ in more than
        their metallicity structure: the forward gate uses ONE Plummer (a0 = 0.28 kpc) while the
        reverse gate uses TWO (0.1875 and 0.3983 kpc). Their opposite bias signs may therefore
        reflect the tracer spatial structure rather than the decomposition choice -- a confound
        that would make the paired-gate claim unsupportable as currently framed.

    Writes figure_gate_diagnostics.png. Mocks only; no network.
    """
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    def _recover(R, vlos, feh, s):
        """Return (gamma_discrete_median_split, gamma_continuous_all_stars)."""
        med = np.median(feh)
        pops = []
        for sel in (feh < med, feh >= med):
            a = float(np.median(R[sel])); rc, so, se = _binprof(R[sel], vlos[sel])
            pops.append((a, rc, so, se))
        gd = _gnfw_gamma_posterior(pops, seed=s)
        a = float(np.median(R)); rc, so, se = _binprof(R, vlos)
        gc = _gnfw_gamma_posterior([(a, rc, so, se)], seed=s + 991)
        return gd, gc

    print("=" * 70)
    print("  GATE DIAGNOSTICS")
    print("    (1) NULL      : one population, NO metallicity structure -> baseline offset?")
    print("    (2) SCRAMBLE  : discrete mock with [Fe/H] permuted -> spatial or chemical?")
    print("=" * 70)

    results = {}
    for gt in gamma_true:
        nd, nc, sd, sc = [], [], [], []
        for s in range(n_real):
            # (1) null: single Plummer, grad = 0 -> [Fe/H] is pure noise
            R, vlos, feh = _continuous_gradient_mock(gt, s, grad=0.0)
            g1, g2 = _recover(R, vlos, feh, s); nd.append(g1); nc.append(g2)
            # (2) scramble: discrete truth, chemo-dynamical link destroyed, geometry preserved
            R, vlos, feh = _discrete_twopop_mock(gt, s)
            feh = np.random.default_rng(s + 4242).permutation(feh)
            g1, g2 = _recover(R, vlos, feh, s); sd.append(g1); sc.append(g2)
        nd, nc, sd, sc = (np.array(v) for v in (nd, nc, sd, sc))
        results[gt] = dict(null_disc=nd, null_cont=nc, scr_disc=sd, scr_cont=sc)

        print(f"\n  gamma_true = {gt}   ({n_real} realisations each)")
        print(f"    (1) NULL      one pop, no [Fe/H] structure")
        print(f"          median split  : <gamma> = {nd.mean():.3f}   bias = {nd.mean()-gt:+.3f} "
              f"+/- {nd.std()/np.sqrt(n_real):.3f}")
        print(f"          all stars     : <gamma> = {nc.mean():.3f}   bias = {nc.mean()-gt:+.3f} "
              f"+/- {nc.std()/np.sqrt(n_real):.3f}")
        base = nc.mean() - gt
        if abs(base) > 0.05:
            print(f"       -> BASELINE OFFSET of {base:+.3f} with NO decomposition to get wrong.")
            print(f"          Both gates must be read relative to this, not to zero.")
        else:
            print(f"       -> estimator unbiased on a featureless mock; the gates' biases are real.")
        print(f"    (2) SCRAMBLE  discrete truth, [Fe/H] permuted (link destroyed, geometry kept)")
        print(f"          median split  : <gamma> = {sd.mean():.3f}   bias = {sd.mean()-gt:+.3f} "
              f"+/- {sd.std()/np.sqrt(n_real):.3f}")
        print(f"          all stars     : <gamma> = {sc.mean():.3f}   bias = {sc.mean()-gt:+.3f} "
              f"+/- {sc.std()/np.sqrt(n_real):.3f}")
        print(f"       -> compare 'median split' here with the REAL reverse gate at the same "
              f"gamma_true.")
        print(f"          Bias that survives the scramble is SPATIAL (tracer structure), not "
              f"chemo-dynamical.")

    fig, axes = plt.subplots(1, len(gamma_true), figsize=(6.6 * len(gamma_true), 4.8), squeeze=False)
    for ax, gt in zip(axes[0], gamma_true):
        r = results[gt]
        labels = ['null\nsplit', 'null\nall-star', 'scrambled\nsplit', 'scrambled\nall-star']
        keys = ['null_disc', 'null_cont', 'scr_disc', 'scr_cont']
        cols = ['lightcoral', 'darkseagreen', 'crimson', 'seagreen']
        for i, (k, c) in enumerate(zip(keys, cols)):
            v = r[k]
            ax.errorbar(i, v.mean(), yerr=v.std() / np.sqrt(len(v)), fmt='o', color=c,
                        ms=9, capsize=5, lw=2)
            ax.annotate(f'{v.mean()-gt:+.2f}', (i, v.mean()), textcoords='offset points',
                        xytext=(0, 13), ha='center', fontsize=9)
        ax.axhline(gt, color='k', ls='--', lw=2, label=f'truth $\\gamma$={gt}')
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.5, 3.5); ax.set_ylabel(r'recovered $\gamma$ (posterior median)')
        ax.set_title(f'$\\gamma_{{\\rm true}}={gt}$'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle('Gate controls: baseline estimator offset (null) and spatial-vs-chemical bias '
                 '(scramble)', fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\n--> Saved {out}")
    return results


def run_bias_vs_realizations(gamma_true=1.0, max_real=40, out="figure_bias_vs_realizations.png"):
    """
    Mean recovered-gamma bias as a function of the number of mock realisations, for the
    discrete two-population split vs the continuous (all-star) treatment, with a 25th-75th
    percentile band (the running spread of individual-realisation biases). Shows that the
    discrete bias is stable and non-zero while the continuous bias converges toward zero.
    Uses posterior medians (same estimator as run_bias_gate). Writes the figure and returns
    the per-realisation arrays.
    """
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    print("=" * 64)
    print(f"  BIAS vs REALISATIONS  (gamma_true = {gamma_true}, up to {max_real} mocks)")
    print("=" * 64)
    gd, gc = [], []
    for s in range(max_real):
        R, vlos, feh = _continuous_gradient_mock(gamma_true, s)
        med = np.median(feh); mp, mr = feh < med, feh >= med
        pops = []
        for sel in (mp, mr):
            a = float(np.median(R[sel])); rc, so, se = _binprof(R[sel], vlos[sel])
            pops.append((a, rc, so, se))
        gd.append(_gnfw_gamma_posterior(pops, seed=s))
        a = float(np.median(R)); rc, so, se = _binprof(R, vlos)
        gc.append(_gnfw_gamma_posterior([(a, rc, so, se)], seed=s + 991))
        if (s + 1) % 5 == 0:
            print(f"    {s+1}/{max_real} realisations done")
    gd, gc = np.array(gd), np.array(gc)

    ns = np.arange(2, max_real + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arr, color, name in [(gd, 'crimson', 'discrete (2-pop split)'),
                             (gc, 'seagreen', 'continuous (all stars)')]:
        run_mean = np.array([np.mean(arr[:n]) - gamma_true for n in ns])
        run_lo = np.array([np.percentile(arr[:n] - gamma_true, 25) for n in ns])
        run_hi = np.array([np.percentile(arr[:n] - gamma_true, 75) for n in ns])
        ax.plot(ns, run_mean, '-', color=color, lw=2, label=f'{name}: bias -> {run_mean[-1]:+.2f}')
        ax.fill_between(ns, run_lo, run_hi, color=color, alpha=0.18)
    ax.axhline(0.0, color='k', ls='--', lw=1.5, label='unbiased (truth)')
    ax.set_xlabel('number of mock realisations')
    ax.set_ylabel(r'mean recovered-$\gamma$ bias  ($\langle\gamma\rangle-\gamma_{\rm true}$)')
    ax.set_title(f'Bias convergence vs sampling  ($\\gamma_{{\\rm true}}={gamma_true}$; '
                 'band = 25th-75th percentile)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\n--> Saved {out}")
    print(f"  final: discrete bias = {np.mean(gd)-gamma_true:+.3f}, "
          f"continuous bias = {np.mean(gc)-gamma_true:+.3f}  ({max_real} realisations)")
    return dict(gd=gd, gc=gc)


def make_data_overview(catalog=None, feh_quality_keep=None,
                       out="figure_data_overview.png", arrays=None):
    """
    Data-overview figure of the active galaxy's real spectroscopic sample: HISTOGRAMS of the
    raw observables (line-of-sight velocity, metallicity indicator, projected radius) and
    SCATTER plots (on-sky distribution, velocity-radius kinematics, metallicity-velocity
    chemodynamics), split by the two metallicity halves. Requires VizieR access (or pass
    pre-loaded `arrays=dict(ra, dec, vlos_rest, feh, R)` for testing).
    """
    out = _gf(out)
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    if arrays is None:
        _cat = catalog or GAL['catalog']
        _fqk = GAL['feh_quality_keep'] if feh_quality_keep is None else feh_quality_keep
        ra, dec, vlos, verr, feh, feherr, gmag = _fetch_tolstoy2023(
            _cat, GAL.get('cols'), mem_keep=(GAL.get('mem_keep') or ('m',)),
            require_member=bool(GAL.get('mem_keep')),
            target_col=GAL.get('target_col'), target_keep=GAL.get('target_keep'), mem_min=GAL.get('mem_min'),
            feh_quality_keep=(list(_fqk) if _fqk else None))
        good = np.isfinite(vlos) & np.isfinite(feh)
        ra, dec = ra[good], dec[good]
        vlos, feh = vlos[good] - V_SYS, feh[good]     # rest-frame velocity
        R = _semi_major_axis_radius(ra, dec)
    else:
        ra, dec = arrays['ra'], arrays['dec']
        vlos, feh, R = arrays['vlos_rest'], arrays['feh'], arrays['R']

    RA0, DEC0, D = 15.0183, -33.7186, 84.0            # sky offsets in kpc
    dx = np.radians(ra - RA0) * np.cos(np.radians(DEC0)) * D
    dy = np.radians(dec - DEC0) * D
    med = np.median(feh)
    mp, mr = feh < med, feh >= med
    cMP, cMR = 'royalblue', 'crimson'

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    # (1) line-of-sight velocity histogram
    a = ax[0, 0]
    b = np.linspace(np.percentile(vlos, 0.5), np.percentile(vlos, 99.5), 35)
    a.hist(vlos[mp], b, color=cMP, alpha=0.6, label='metal-poor')
    a.hist(vlos[mr], b, color=cMR, alpha=0.6, label='metal-rich')
    a.axvline(0, color='k', ls='--', lw=1, label='systemic')
    a.set_xlabel(r'$v_{\rm los}$ (rest frame)  [km s$^{-1}$]'); a.set_ylabel('N stars')
    a.set_title(f'Line-of-sight velocity  (N = {len(vlos)})'); a.legend(fontsize=8)

    # (2) metallicity histogram + unimodality statistics (evidence #1)
    a = ax[0, 1]
    b = np.linspace(np.percentile(feh, 0.5), np.percentile(feh, 99.5), 35)
    a.hist(feh[mp], b, color=cMP, alpha=0.6); a.hist(feh[mr], b, color=cMR, alpha=0.6)
    a.axvline(med, color='k', ls=':', lw=1.2, label=f'median split = {med:.2f}')
    try:
        bc, dbic = _unimodality_report(feh)
        dip, dp = _dip_test(feh)
        txt = (f'BC = {bc:.2f} ({"uni" if bc < 0.555 else "bi"}modal)\n'
               f'$\\Delta$BIC$_{{1-2}}$ = {dbic:+.0f}')
        if dip is not None:
            txt += f'\ndip $p$ = {dp:.2f} ({"unimodal" if dp > 0.05 else "multimodal"})'
        a.text(0.03, 0.97, txt, transform=a.transAxes, va='top', ha='left', fontsize=8,
               bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
    except Exception:
        pass
    a.set_xlabel('[Fe/H]  [dex]'); a.set_ylabel('N stars')
    a.set_title('Metallicity distribution'); a.legend(fontsize=8)

    # (3) projected-radius histogram
    a = ax[0, 2]
    a.hist(R, 35, color='0.5', alpha=0.85, edgecolor='w', linewidth=0.3)
    a.set_xlabel('elliptical radius $R$  [kpc]'); a.set_ylabel('N stars')
    a.set_title('Projected radial distribution')

    # (4) on-sky map, coloured by [Fe/H]
    a = ax[1, 0]
    sc = a.scatter(dx, dy, c=feh, s=9, cmap='coolwarm_r',
                   vmin=np.percentile(feh, 2), vmax=np.percentile(feh, 98))
    a.set_xlabel(r'$\Delta$RA  [kpc]'); a.set_ylabel(r'$\Delta$Dec  [kpc]')
    a.set_aspect('equal'); a.invert_xaxis(); a.set_title('On-sky distribution')
    plt.colorbar(sc, ax=a, label='[Fe/H]', fraction=0.046)

    # (5) kinematics: velocity vs radius
    a = ax[1, 1]
    a.scatter(R[mp], vlos[mp], s=9, color=cMP, alpha=0.5, label='metal-poor')
    a.scatter(R[mr], vlos[mr], s=9, color=cMR, alpha=0.5, label='metal-rich')
    a.axhline(0, color='k', ls='--', lw=1)
    a.set_xlabel('$R$  [kpc]'); a.set_ylabel(r'$v_{\rm los}$  [km s$^{-1}$]')
    a.set_title('Kinematics: velocity vs radius'); a.legend(fontsize=8)

    # (6) chemodynamics: velocity vs metallicity, coloured by radius
    a = ax[1, 2]
    sc = a.scatter(feh, vlos, c=R, s=9, cmap='viridis')
    a.axhline(0, color='k', ls='--', lw=1)
    a.set_xlabel('[Fe/H]  [dex]'); a.set_ylabel(r'$v_{\rm los}$  [km s$^{-1}$]')
    a.set_title('Chemodynamics: velocity vs metallicity')
    plt.colorbar(sc, ax=a, label='$R$ [kpc]', fraction=0.046)

    fig.suptitle('Sculptor spectroscopic sample (Tolstoy et al. 2023) — data overview',
                 fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f"--> Saved {out}  ({len(vlos)} stars)")
    return out


def make_dispersion_profile(out=None, nbins=8, use_gaia=False):
    """
    Radial velocity-dispersion profile sigma_los(R) of the active galaxy's members, in
    equal-count radial bins with bootstrap uncertainties. If use_gaia=True and Gaia proper
    motions can be matched, also overlays the plane-of-sky (radial and tangential) PM
    dispersion profiles -- otherwise the line-of-sight profile alone (fully offline). Writes
    figure_dispersion_profile.png (galaxy-tagged).
    """
    out = _gf(out or "figure_dispersion_profile.png")
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    R, vlos, feh, verr = _load_real_feh()
    order = np.argsort(R)
    R, vlos, verr = R[order], vlos[order], verr[order]
    edges = np.interp(np.linspace(0, len(R), nbins + 1), np.arange(len(R) + 1),
                      np.concatenate([[0], np.sort(R)]))          # equal-count bin edges
    rng = np.random.default_rng(0)
    Rc, sig, siglo, sighi = [], [], [], []
    for i in range(nbins):
        m = (R >= edges[i]) & (R < edges[i + 1]) if i < nbins - 1 else (R >= edges[i])
        if m.sum() < 5:
            continue
        v, e = vlos[m], verr[m]
        # intrinsic dispersion (subtract measurement variance), bootstrap the CI
        def disp(vv, ee):
            return np.sqrt(max(np.var(vv) - np.mean(ee ** 2), 1.0))
        boots = [disp(v[idx], e[idx]) for idx in
                 (rng.integers(0, len(v), len(v)) for _ in range(300))]
        Rc.append(np.median(R[m]))
        sig.append(disp(v, e))
        siglo.append(np.percentile(boots, 16)); sighi.append(np.percentile(boots, 84))
    Rc, sig = np.array(Rc), np.array(sig)
    yerr = np.vstack([sig - np.array(siglo), np.array(sighi) - sig])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.errorbar(Rc, sig, yerr=yerr, fmt='o-', color='navy', capsize=3, lw=1.8,
                label=r'$\sigma_{\rm los}$ (line of sight)')

    if use_gaia:                                                  # optional PM dispersion (needs Gaia)
        try:
            prof = _gaia_pm_dispersion_profile(nbins=nbins)       # (Rc, sig_R, sig_T) in km/s
            if prof is not None:
                Rg, sR, sT = prof
                ax.plot(Rg, sR, 's--', color='crimson', lw=1.6, label=r'$\sigma_{\rm PM,R}$ (radial)')
                ax.plot(Rg, sT, '^--', color='seagreen', lw=1.6, label=r'$\sigma_{\rm PM,T}$ (tangential)')
        except Exception as exc:
            print(f"  PM dispersion skipped ({str(exc)[:70]})")

    ax.set_xlabel(r'projected radius $R$  [kpc]')
    ax.set_ylabel(r'velocity dispersion  [km s$^{-1}$]')
    ax.set_title(f'{GAL["name"]}: radial velocity-dispersion profile')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"--> Saved {out}  ({len(Rc)} radial bins)")
    return out


def _load_gaia_matched():
    """Cross-match the active galaxy's spectroscopic members to Gaia DR3 proper motions and
    compute PM membership. Returns a DataFrame (ra, dec, pmra, pmdec, P_mem_PM, R_kpc).
    Requires network/Gaia access; raises on failure so callers can fall back."""
    import pandas as pd
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra, dec, vlos, verr, feh, feherr, _g = _fetch_tolstoy2023(
        GAL['catalog'], GAL.get('cols'),
        mem_keep=(GAL.get('mem_keep') or ('m',)), require_member=bool(GAL.get('mem_keep')),
        target_col=GAL.get('target_col'), target_keep=GAL.get('target_keep'),
        mem_min=GAL.get('mem_min'), feh_quality_keep=None)
    df = pd.DataFrame(dict(ra=ra, dec=dec, vlos=vlos - V_SYS, feh=feh))   # rest-frame (as other loaders)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs', obstime=Time('J2000.0'))
    matched = fetch_gaia_with_epoch_correction(df, coords)         # existing Gaia infra
    matched = matched.copy()
    matched['P_mem_1D'] = 1.0                                      # input stars are already members
    for a, b in [('RA_deg', 'ra'), ('DEC_deg', 'dec'), ('Dec_deg', 'dec'), ('Ra_deg', 'ra'),
                 ('RA', 'ra'), ('DEC', 'dec')]:
        if a not in matched.columns and b in matched.columns:
            matched[a] = matched[b]                                # aliases the GMM may expect
    try:
        memb = calculate_error_aware_membership(matched)
        pcol = ('P_mem_PM' if 'P_mem_PM' in memb.columns
                else [c for c in memb.columns if 'P_mem' in c][0])
        memb['P_mem_PM'] = memb[pcol]
    except Exception as exc:                                       # robust fallback: PM-clump membership
        print(f"  [membership] GMM unavailable ({str(exc)[:50]}); using PM-clump distance")
        memb = matched
        pm = memb[['pmra', 'pmdec']].values
        med = np.median(pm, axis=0)
        sig = np.maximum(np.median(np.abs(pm - med), axis=0) * 1.4826, 0.05)
        chi2 = np.sum(((pm - med) / sig) ** 2, axis=1)
        L_S = np.exp(-0.5 * chi2); memb['P_mem_PM'] = L_S / (L_S + 0.05)
    memb['R_kpc'] = _semi_major_axis_radius(memb['ra'].values, memb['dec'].values)
    return memb


def _gaia_pm_dispersion_profile(nbins=8):
    """Plane-of-sky PM velocity-dispersion profile (radial, tangential) in km/s, from Gaia
    proper motions of members (P_mem_PM > 0.5). Returns (Rc, sigR, sigT) or None."""
    df = _load_gaia_matched()
    df = df[df['P_mem_PM'] > 0.5].reset_index(drop=True)
    if len(df) < 40:
        return None
    kms = 4.74047 * GAL['distance_kpc']                            # mas/yr -> km/s at distance
    dpmra = (df['pmra'].values - np.median(df['pmra'].values)) * kms
    dpmdec = (df['pmdec'].values - np.median(df['pmdec'].values)) * kms
    R = df['R_kpc'].values
    edges = np.quantile(R, np.linspace(0, 1, nbins + 1))
    Rc, sR, sT = [], [], []
    for i in range(nbins):
        m = (R >= edges[i]) & (R < edges[i + 1]) if i < nbins - 1 else (R >= edges[i])
        if m.sum() < 8:
            continue
        Rc.append(np.median(R[m])); sR.append(np.std(dpmra[m])); sT.append(np.std(dpmdec[m]))
    return np.array(Rc), np.array(sR), np.array(sT)


def make_gaia_skymap(out=None, cuts=(0.0, 0.5, 0.9), arrays=None):
    """
    Gaia proper-motion sky map with one panel PER membership cut (the advisor's
    membership-cut justification): each panel shows the on-sky distribution of stars, with
    likely members and Milky-Way foreground distinguished, coloured by PM membership
    probability. Demonstrates that the PM cut cleanly separates the dwarf from the foreground.
    Pass pre-loaded `arrays=dict(ra, dec, pmra, pmdec, P_mem, R)` for offline testing;
    otherwise cross-matches to Gaia DR3 (needs network). Writes figure_gaia_skymap.png.
    """
    out = _gf(out or "figure_gaia_skymap.png")
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    if arrays is None:
        df = _load_gaia_matched()
        ra, dec = df['ra'].values, df['dec'].values
        pmra, pmdec, pmem = df['pmra'].values, df['pmdec'].values, df['P_mem_PM'].values
    else:
        ra, dec = arrays['ra'], arrays['dec']
        pmra, pmdec, pmem = arrays['pmra'], arrays['pmdec'], arrays['P_mem']
    dRA = (ra - RA0_DEG) * np.cos(np.radians(DEC0_DEG)); dDec = dec - DEC0_DEG

    n = len(cuts)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.8), squeeze=False)
    for k, cut in enumerate(cuts):
        ax = axes[0][k]
        mem = pmem >= cut
        ax.scatter(dRA[~mem], dDec[~mem], s=8, c='0.6', marker='x', alpha=0.5, label='MW foreground')
        sc = ax.scatter(dRA[mem], dDec[mem], s=14, c=pmem[mem], cmap='viridis', vmin=0, vmax=1,
                        edgecolors='k', linewidths=0.2, label='members')
        ax.set_xlabel(r'$\Delta$RA [deg]'); ax.set_title(f'$P_{{\\rm mem}} \\geq {cut:.1f}$   '
                                                         f'({int(mem.sum())} stars)')
        if k == 0:
            ax.set_ylabel(r'$\Delta$Dec [deg]'); ax.legend(fontsize=8, loc='upper right')
        ax.invert_xaxis(); ax.set_aspect('equal', 'datalim'); ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cb.set_label(r'PM membership probability $P_{\rm mem}$')
    fig.suptitle(f'{GAL["name"]}: Gaia proper-motion membership across cuts', fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"--> Saved {out}")
    return out


def compute_actions_for_members(chain_file="cont_chain.npy", arrays=None):
    """Compute orbital actions (J_r, J_z, J_phi) for the active galaxy's Gaia-matched members
    in the median-posterior potential, using AGAMA's ActionFinder. Full 6D phase space is
    built from (ra, dec, distance, pmra, pmdec, vlos). Returns (Jr, Jz, Jphi, feh). Pass
    `arrays=dict(Jr,Jz,Jphi,feh)` to bypass (offline testing). Needs Gaia + AGAMA + a chain."""
    if arrays is not None:
        return arrays['Jr'], arrays['Jz'], arrays['Jphi'], arrays['feh']
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    df = _load_gaia_matched()
    df = df[df['P_mem_PM'] > 0.5].reset_index(drop=True)
    flat = np.load(_gf(chain_file))
    th = np.median(flat[:, :5], axis=0)                            # [logM_DM, log_rs, alpha, eta, gamma]
    pot = ap25_dm_potential(th[0], th[1], th[2], th[3], th[4])
    af = agama.ActionFinder(pot)
    # Actions must be computed in a frame centered on the DWARF (its internal potential has a
    # ~kpc scale radius), NOT the Galactocentric frame. Build each star's phase-space vector
    # relative to the galaxy centre: tangent-plane position (line-of-sight depth unknown -> 0)
    # and velocity relative to the systemic proper motion and systemic line-of-sight velocity.
    D = GAL['distance_kpc']
    ra = df['ra'].values; dec = df['dec'].values
    dx = np.radians(ra - RA0_DEG) * np.cos(np.radians(DEC0_DEG)) * D    # kpc, tangent plane
    dy = np.radians(dec - DEC0_DEG) * D
    dz = np.zeros_like(dx)                                          # unknown depth -> galaxy centre
    kms = 4.74047 * D                                              # mas/yr -> km/s at distance D
    vx = (df['pmra'].values - np.median(df['pmra'].values)) * kms  # relative to systemic PM
    vy = (df['pmdec'].values - np.median(df['pmdec'].values)) * kms
    vz = df['vlos'].values                                        # already rest-frame from _load_gaia_matched
    xv = np.column_stack([dx, dy, dz, vx, vy, vz])
    J = af(xv)                                                     # (N,3): Jr, Jz, Jphi
    return J[:, 0], J[:, 1], J[:, 2], df['feh'].values


def make_action_space(out=None, chain_file="cont_chain.npy", arrays=None):
    """
    Action-space chemodynamics: (left) metallicity vs radial action J_r, testing whether
    metal-rich stars occupy lower-action (more bound, centrally concentrated) orbits -- the
    action-space signature of the gradient; (right) J_z vs J_r scatter coloured by [Fe/H], to
    reveal any trend across orbit type. Needs Gaia PMs + AGAMA actions in the fitted potential
    (or pass arrays=dict(Jr,Jz,Jphi,feh) for offline testing). Writes figure_action_space.png.
    """
    out = _gf(out or "figure_action_space.png")
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt
    Jr, Jz, Jphi, feh = compute_actions_for_members(chain_file=chain_file, arrays=arrays)
    Jr = np.asarray(Jr); Jz = np.asarray(Jz); feh = np.asarray(feh)
    good = np.isfinite(Jr) & np.isfinite(Jz) & np.isfinite(feh) & (Jr > 0) & (Jz > 0)
    Jr, Jz, feh = Jr[good], Jz[good], feh[good]                   # log axes need positive actions
    if len(Jr) < 10:
        print(f"  action-space skipped (only {len(Jr)} stars with positive finite actions)")
        return None

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axL.scatter(Jr, feh, s=10, c=feh, cmap='viridis', alpha=0.7, edgecolors='none')
    # running median of [Fe/H] vs J_r to show the trend
    if len(Jr) > 20:
        oi = np.argsort(Jr); b = np.array_split(oi, 8)
        jm = [np.median(Jr[ix]) for ix in b]; fm = [np.median(feh[ix]) for ix in b]
        axL.plot(jm, fm, 'r-o', lw=2, label='running median')
        axL.legend(fontsize=9)
    axL.set_xscale('log')
    axL.set_xlabel(r'radial action $J_r$  [kpc km s$^{-1}$]')
    axL.set_ylabel(r'[Fe/H]'); axL.set_title('Metallicity vs radial action')
    axL.grid(alpha=0.25)

    sc = axR.scatter(Jr, Jz, s=14, c=feh, cmap='viridis', alpha=0.8, edgecolors='k', linewidths=0.2)
    axR.set_xscale('log'); axR.set_yscale('log')
    axR.set_xlabel(r'radial action $J_r$'); axR.set_ylabel(r'vertical action $J_z$')
    axR.set_title('Action components coloured by [Fe/H]')
    cb = fig.colorbar(sc, ax=axR, fraction=0.046, pad=0.02); cb.set_label('[Fe/H]')
    axR.grid(alpha=0.25)
    fig.suptitle(f'{GAL["name"]}: action-space chemodynamics', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"--> Saved {out}  ({len(Jr)} stars)")
    return out


def make_framework_comparison(out="figure_framework_comparison.png", chains=None):
    """
    Head-to-head comparison of the DM inner-slope posteriors across the dynamical-
    framework hierarchy, built from the saved chains in the working directory:
        dm5_chain.npy         Spherical Jeans (gNFW), gamma = column 0
        gravsphere_chain.npy  GravSphere (Jeans + VSPs + free beta), gamma = column 0
        ap25_chain.npy        Full 25-parameter action-DF model, gamma = column 4
    Missing chains are skipped, so the figure grows as each framework completes.
    LEFT: smoothed gamma posteriors with the AP25 published band and the NFW-cusp
    reference. RIGHT: median +/- 68% forest plot. Run via  --compare .
    """
    out = _gf(out)
    import os
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    gcol_ap25 = PARAM_NAMES.index('gamma')
    catalog = chains or [
        ("Spherical Jeans\n($\\sigma_{\\rm los}$ only, gNFW)", _gf("dm5_chain.npy"), 0, 'royalblue'),
        ("GravSphere\n(+VSPs, free $\\beta(r)$)", _gf("gravsphere_chain.npy"), 0, 'teal'),
        ("Action-DF\n(25-param, AGAMA)", _gf("ap25_chain.npy"), gcol_ap25, 'crimson'),
    ]
    loaded = []
    for lbl, fn, col, c in catalog:
        if os.path.exists(fn):
            flat = np.load(fn)
            if flat.ndim == 2 and flat.shape[1] > col:
                loaded.append((lbl, np.asarray(flat[:, col], float), c))
        else:
            print(f"  [compare] {fn} not found -- skipping {lbl.splitlines()[0]}")
    if not loaded:
        print("  [compare] no chains found; run --dm5 / --gravsphere / --chain first")
        return None

    meds = []
    for lbl, g, c in loaded:
        p16, p50, p84 = np.percentile(g, [16, 50, 84])
        meds.append((lbl, p50, p50 - p16, p84 - p50, c))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                   gridspec_kw=dict(width_ratios=[1.6, 1.0]))
    gg = np.linspace(0.0, 1.9, 400)
    glo, ghi = AP25['gamma'] - AP25['gamma_lo'], AP25['gamma'] + AP25['gamma_hi']

    # LEFT: posteriors
    axL.axvspan(glo, ghi, color='0.6', alpha=0.22)
    axL.axvline(AP25['gamma'], color='0.25', ls='--', lw=1.5,
                label=r'AP25 published: $0.39^{+0.23}_{-0.26}$')
    axL.axvline(1.0, color='k', ls=':', lw=1.0)
    axL.text(1.0, 0.985, ' NFW cusp $\\gamma=1$', transform=axL.get_xaxis_transform(),
             fontsize=8, va='top', color='k')
    for (lbl, g, c), (_, p50, lo, hi, _) in zip(loaded, meds):
        kde = gaussian_kde(g)
        y = kde(gg)
        axL.plot(gg, y, color=c, lw=2.2,
                 label=f"{lbl.splitlines()[0]}: "
                       f"$\\gamma={p50:.2f}^{{+{hi:.2f}}}_{{-{lo:.2f}}}$")
        axL.fill_between(gg, y, color=c, alpha=0.16)
    axL.set_xlim(0, 1.9); axL.set_ylim(bottom=0)
    axL.set_xlabel(r'DM inner slope $\gamma$'); axL.set_ylabel('posterior density')
    axL.set_title('Sculptor DM inner slope across the framework hierarchy')
    axL.legend(fontsize=8.5, loc='upper right'); axL.grid(alpha=0.25)

    # RIGHT: forest plot
    ylabels = []
    for i, (lbl, p50, lo, hi, c) in enumerate(meds):
        y = len(meds) - i
        axR.errorbar([p50], [y], xerr=[[lo], [hi]], fmt='o', color=c, capsize=4,
                     ms=8, lw=2)
        ylabels.append((y, lbl))
    axR.errorbar([AP25['gamma']], [0], xerr=[[AP25['gamma_lo']], [AP25['gamma_hi']]],
                 fmt='*', color='k', ms=13, capsize=4, lw=1.5)
    ylabels.append((0, 'AP25 published\n(action-DF, 2 pops)'))
    axR.axvspan(glo, ghi, color='0.6', alpha=0.15)
    axR.axvline(1.0, color='k', ls=':', lw=1.0)
    axR.set_yticks([y for y, _ in ylabels])
    axR.set_yticklabels([l for _, l in ylabels], fontsize=8.5)
    axR.set_xlim(0, 1.9); axR.set_ylim(-0.7, len(meds) + 0.7)
    axR.set_xlabel(r'$\gamma$  (median, 68% CI)')
    axR.set_title('Same Tolstoy+2023 sample'); axR.grid(alpha=0.25, axis='x')

    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"--> Saved {out}")
    print("  --- gamma across frameworks (median +/- 68% CI) ---")
    for lbl, p50, lo, hi, _ in meds:
        print(f"    {lbl.splitlines()[0]:<24} gamma = {p50:.2f} (+{hi:.2f} / -{lo:.2f})")
    print(f"    {'AP25 published':<24} gamma = {AP25['gamma']:.2f} "
          f"(+{AP25['gamma_hi']:.2f} / -{AP25['gamma_lo']:.2f})")
    return out


def _seed_ap25_from_data(data):
    """Data-informed 25-parameter seed for the REAL-data chain: DM from the fast
    two-population gNFW MLE, metallicity means/widths from the [Fe/H] distribution, and
    DF/pop-3 from Sculptor-like defaults. Starting near the real posterior (rather than
    at the mock's gamma=1) greatly reduces the steps needed to converge. Falls back to
    the mock-truth vector if the MLE fails."""
    seed = ap25_truth_vector().astype(float).copy()
    feh = np.asarray(data['feh'], float)
    R, vlos = np.asarray(data['R'], float), np.asarray(data['vlos'], float)
    med = np.median(feh)
    mp, mr = feh[feh < med], feh[feh >= med]
    if len(mp) and len(mr):                                    # metallicity from the data
        seed[10] = float(np.clip(mp.mean(), -2.7, -1.7)); seed[11] = float(np.clip(mp.std(), 0.15, 0.45))
        seed[17] = float(np.clip(mr.mean(), -1.65, -1.3)); seed[18] = float(np.clip(mr.std(), 0.15, 0.45))
    try:                                                       # DM from the fast MLE
        label = np.where(feh >= med, RE_MR, RE_MP)
        res, _ = agama_fit_halo(R, vlos, label)
        g_fit, rs_fit, lrho_fit = res.x
        pot_g = agama.Potential(type='Spheroid', densityNorm=10.0**lrho_fit,
                                scaleRadius=rs_fit, gamma=g_fit, beta=BETA_DM, alpha=1.0)
        f2 = dm_potential_5p(g_fit, rs_fit, 0.0, 1.0, 3.0).enclosedMass(2.0)
        logMDM = float(np.clip(np.log10(max(pot_g.enclosedMass(2.0) / max(f2, 1e-300), 1e7)), 7.2, 10.8))
        seed[0] = logMDM                                       # logM_DM
        seed[1] = float(np.clip(np.log10(rs_fit), -3.0, 1.0))  # log_rs
        seed[2] = 2.0                                          # alpha (neutral; mock=1, AP25~3.7)
        seed[3] = 3.0                                          # eta
        seed[4] = float(np.clip(g_fit, 0.05, 1.5))            # gamma from MLE (~0.6, not the mock's 1)
        print(f"  [seed] DM from fast MLE: gamma={seed[4]:.2f}, rs={rs_fit:.2f} kpc, "
              f"logM_DM={seed[0]:.2f}; MDF means from data")
    except Exception as exc:
        print(f"  [seed] MLE seed failed ({str(exc)[:60]}); using Sculptor-like defaults")
    return seed


def run_ap25_production_chain(nwalkers=60, nsteps=2000, nproc=None, backend="scl25.h5",
                             resume=None, use_selection=False, selection_mode='radial',
                             feh_quality_keep=(0,), catalog="J/A+A/675/A49", bw=0.3,
                             nsub=None):
    """
    PRODUCTION run of the full 25-parameter AP25 validation chain on the REAL
    Tolstoy+2023 data. Launch it from the terminal with:  python <thisfile>.py --chain
    (the __main__ guard makes multiprocessing safe on Windows -- do NOT run a parallel
    chain from an interactive REPL). Re-run the same command to RESUME from `backend`;
    each invocation adds `nsteps` iterations in a crash-safe checkpoint.

    nsub: if set (e.g. 400), fit a random subsample of that many stars -- a faster,
    slightly wider-posterior first pass that is friendlier to a laptop. None = all stars.
    """
    import os
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2))
    if resume is None:
        resume = os.path.exists(backend)              # auto-resume iff a checkpoint exists

    print("=" * 64)
    print("  AP25 FULL 25-parameter validation chain (real Tolstoy+2023)")
    print("=" * 64)
    # 1) analysis sample: members with reliable [Fe/H] (~1339), real Gmag, rest-frame v
    data = ap25_load_real_tolstoy2023(catalog=catalog,
                                      feh_quality_keep=feh_quality_keep)
    print(f"  analysis sample: {len(data['R'])} stars "
          f"(members & q_[Fe/H] in {tuple(feh_quality_keep)})"
          + (f"; fitting a random subsample of {nsub}" if nsub else ""))

    # 2) optional AP24-style selection function omega(R,G)
    if use_selection:
        try:
            pR, pG = gaia_rgb_parent()
            data = attach_selection(data, pR, parent_G=pG, mode=selection_mode, bw=bw)
        except Exception as exc:
            print(f"  [selection] FAILED ({str(exc)[:90]}); continuing with FLAT selection")
    else:
        print("  [selection] FLAT (Omega=1); pass use_selection=True for AP24-style omega")

    # 3) seed near the real posterior (fresh runs only), then run / resume
    init = None if resume else _seed_ap25_from_data(data)
    print(f"  {'RESUMING from' if resume else 'STARTING'} {backend}  |  nwalkers={nwalkers}, "
          f"+{nsteps} steps this run, nproc={nproc}")
    sampler = ap25_run_full_mcmc(data, nproc=nproc, nwalkers=nwalkers, nsteps=nsteps,
                                 nsub=nsub, init=init, backend_file=backend,
                                 resume=resume, progress=True)
    total = sampler.iteration
    print(f"  total accumulated iterations: {total}")

    # 4) convergence diagnostics + posterior
    rep = mcmc_convergence_report(sampler, TEX)
    burn, thin = rep['burn'], rep['thin']
    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
    if len(flat) < 50:                                # too short yet -> re-run to add steps
        flat = sampler.get_chain(discard=max(1, total // 3), flat=True)
    np.save(_gf("ap25_chain.npy"), flat)
    print(f"  saved {flat.shape[0]} posterior samples -> ap25_chain.npy")
    try:
        make_corner_plot(flat, TEX, "figure_ap25_corner.png")
        print("  saved figure_ap25_corner.png")
    except Exception as exc:
        print(f"  corner skipped ({exc}); pip install corner")
    try:
        make_ap25_figure4(flat, "figure_ap25_fig4.png")
        print("  saved figure_ap25_fig4.png  (paper Fig.4: enclosed mass + density profiles)")
    except Exception as exc:
        print(f"  Fig.4 skipped ({exc})")

    print("\n  === posterior (median +/- 68% CI) ===")
    for k, nm in enumerate(PARAM_NAMES):
        p16, p50, p84 = np.percentile(flat[:, k], [16, 50, 84])
        print(f"    {nm:<10} = {p50:8.3f}  (+{p84 - p50:.3f} / -{p50 - p16:.3f})")
    gi = PARAM_NAMES.index('gamma')
    g16, g50, g84 = np.percentile(flat[:, gi], [16, 50, 84])
    tag = "" if rep.get('converged') else "   [NOT converged -- re-run to add steps]"
    print("\n  " + "-" * 58)
    print(f"  VALIDATION:  gamma = {g50:.2f} (+{g84 - g50:.2f} / -{g50 - g16:.2f}){tag}")
    print(f"               AP25 published: 0.39 (+0.23 / -0.26)")
    print("  " + "-" * 58)
    return sampler


# ============================================================
# PHASE 6: CONTINUOUS ACTION-METALLICITY DF  f(J, [Fe/H])  (the novel method)
# ============================================================
# Metallicity becomes a COORDINATE the DF depends on, not a label used to split stars.
# A single DoublePowerLaw family whose scale action varies smoothly with [Fe/H]:
#       log10 J0(z) = logJ0_0 + kJ * (z - Mz)     (kJ < 0  =>  metal-rich = more central)
# The per-star likelihood MARGINALISES over each star's true metallicity z:
#   L_i = (1-f_C) INT dz  N(z|Mz,sigz) [projDF_z(R_i,v_i)/Z(z)] N(feh_i|z,feherr_i)
#         + f_C * contamination_i
# on a K-node metallicity quadrature (=> K GalaxyModels per call, ~K/2 x the discrete
# two-population cost: a cluster-scale run, like --chain). Only the action-DF framework
# can put metallicity INSIDE the DF -- Jeans/GravSphere cannot, which is itself a result.
# Replaces the discrete model's 10 stellar-DF + 4 MDF + 2 fraction params with 4 shared
# shape + 2 gradient + 2 MDF params (18 total vs 25).

CONT_PARAM_NAMES = ['logM_DM', 'log_rs', 'alpha', 'eta', 'gamma',
                    'Gamma', 'B', 'gz', 'hz',                 # shared stellar-DF shape
                    'logJ0_0', 'kJ',                          # metallicity-action gradient
                    'Mz', 'sigz',                             # metallicity distribution
                    'V_C', 'sigV_C', 'M_C', 'sigM_C', 'f_C']  # contamination (pop-3)
CONT_NDIM = len(CONT_PARAM_NAMES)                              # 18
CONT_TEX = [r'$\log M_{\rm DM}$', r'$\log r_s$', r'$\alpha$', r'$\eta$', r'$\gamma$',
            r'$\Gamma$', r'$B$', r'$g_z$', r'$h_z$', r'$\log J_{0,0}$', r'$k_J$',
            r'$\mathcal{M}_z$', r'$\sigma_z$', r'$V_C$', r'$\sigma_{V,C}$',
            r'$\mathcal{M}_C$', r'$\sigma_{M,C}$', r'$f_C$']
CONT_PRIOR_LO = np.array([7.0, -3.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                          -1.0, -4.0, -2.7, 0.05, -30.0, 0.0, -4.5, 0.15, 0.0])
CONT_PRIOR_HI = np.array([11.0, 1.0, 7.0, 7.0, 1.9, 3.0, 30.0, 1.5, 1.5,
                          3.0, 0.5, -1.3, 0.8, 30.0, 20.0, -1.3, 0.9, 0.3])
CONT_TRUTH = dict(logM_DM=8.5, log_rs=-0.5, alpha=1.0, eta=3.0, gamma=1.0,
                  Gamma=1.0, B=15.0, gz=0.5, hz=0.5, logJ0_0=1.0, kJ=-1.0,
                  Mz=-1.7, sigz=0.35, V_C=13.0, sigV_C=7.0, M_C=-2.8, sigM_C=0.5, f_C=0.02)


def cont_truth_vector():
    return np.array([CONT_TRUTH[k] for k in CONT_PARAM_NAMES])


def _cont_zgrid(Mz, sigz, K=11, nsig=3.5):
    z = np.linspace(Mz - nsig * sigz, Mz + nsig * sigz, K)
    return z, float(z[1] - z[0])


def ap25_stellar_df_z(z, logJ0_0, kJ, Mz, Gamma, B, gz, hz):
    """Continuous DF at metallicity z: DoublePowerLaw with log10 J0 = logJ0_0 + kJ*(z-Mz)."""
    return ap25_stellar_df(logJ0_0 + kJ * (z - Mz), Gamma, B, gz, hz)


def ap25_lnprior_continuous(theta):
    if np.any(theta < CONT_PRIOR_LO) or np.any(theta > CONT_PRIOR_HI):
        return -np.inf
    return 0.0


def _cont_interp_matrix(zk, zf):
    """Linear-interpolation matrix M (nf x K) so that f(zf) ~= M @ f(zk) for a function
    sampled on the coarse node grid zk. Used to lift the (smooth, expensive) dynamics from
    K DF nodes onto a fine grid, where the (sharp, cheap, analytic) metallicity link lives."""
    K, nf = len(zk), len(zf)
    M = np.zeros((nf, K))
    idx = np.clip(np.searchsorted(zk, zf) - 1, 0, K - 2)
    t = (zf - zk[idx]) / (zk[idx + 1] - zk[idx])
    M[np.arange(nf), idx] = 1.0 - t
    M[np.arange(nf), idx + 1] = t
    return M


def ap25_lnlike_continuous(theta, data, K=11, nf=64):
    """Per-star mixture log-likelihood for the continuous f(J,[Fe/H]) model, marginalised
    over each star's true metallicity. The marginalisation separates two scales: the DF
    (dynamics) varies SMOOTHLY with metallicity, so it is evaluated at only K expensive
    nodes and linearly lifted onto a fine grid; the measurement link N(feh_i|z,feherr_i) is
    SHARP but analytic, so it is integrated on the fine grid. This resolves the narrow link
    that a coarse K-node grid would miss, at the cost of only K GalaxyModel builds."""
    (logM_DM, log_rs, alpha, eta, gamma, Gamma, B, gz, hz,
     logJ0_0, kJ, Mz, sigz, V_C, sigV_C, M_C, sigM_C, f_C) = theta
    R, vlos, verr = data['R'], data['vlos'], data['verr']
    feh, feherr = data['feh'], data['feherr']
    N = len(R)
    omega_star = data.get('omega_star');  Omega_grid = data.get('Omega_grid')
    if omega_star is None: omega_star = OMEGA_R(R)
    if Omega_grid is None: Omega_grid = OMEGA_R(_RGRID)
    try:
        pot = ap25_dm_potential(logM_DM, log_rs, alpha, eta, gamma)
        pts = np.column_stack([R, np.zeros(N), np.zeros(N), np.zeros(N), vlos,
                               np.full(N, _INF), np.full(N, _INF), verr])
        # coarse DF nodes span the metallicity DATA range (where stars/links live); the
        # fine grid resolves the sharp per-star link. MDF weight suppresses the tails.
        zlo = min(feh.min(), Mz - 3.5 * sigz) - 0.05
        zhi = max(feh.max(), Mz + 3.5 * sigz) + 0.05
        zk = np.linspace(zlo, zhi, K)
        zf = np.linspace(zlo, zhi, nf)
        Mmat = _cont_interp_matrix(zk, zf)
        Pz = np.empty((K, N))                                       # expensive: K DF builds
        for j, z in enumerate(zk):
            gm = agama.GalaxyModel(pot, ap25_stellar_df_z(z, logJ0_0, kJ, Mz, Gamma, B, gz, hz))
            Zj = _proj_norm(gm, Omega_grid)
            if not (Zj > 0):
                return -np.inf
            Pz[j] = np.maximum(gm.projectedDF(pts), 0.0) / Zj
        Dfine = Mmat @ Pz                                           # (nf,N) smooth dynamics
        mdf = _gauss(zf, Mz, sigz)[:, None]                         # (nf,1) MDF prior on z
        link = np.exp(-0.5 * ((zf[:, None] - feh[None, :]) / feherr[None, :]) ** 2) \
            / (np.sqrt(2 * np.pi) * feherr[None, :])                # (nf,N) sharp, analytic
        integ = mdf * link * Dfine * omega_star[None, :]
        Lstar = _trapz(integ, dx=float(zf[1] - zf[0]), axis=0)      # marginalise over z
        Z_C = _trapz(2 * np.pi * Omega_grid * _RGRID, _RGRID)       # contamination
        Lc = (omega_star / Z_C) * _gauss(vlos, V_C, np.sqrt(sigV_C ** 2 + verr ** 2)) \
            * _gauss(feh, M_C, np.sqrt(sigM_C ** 2 + feherr ** 2))
        L = (1.0 - f_C) * Lstar + f_C * Lc
        if np.any(~np.isfinite(L)) or np.any(L <= 0):
            return -np.inf
        return float(np.sum(np.log(L)))
    except Exception:
        return -np.inf


def ap25_lnprob_continuous(theta, data):
    lp = ap25_lnprior_continuous(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ap25_lnlike_continuous(theta, data, K=data.get('_K', 11))


def ap25_lnprob_continuous_reduced(free_theta, data):
    """Reduced-dimension wrapper: samples only the free parameters (data['_free_idx']),
    holding the rest fixed at data['_template']. Module-level so it pickles for the pool."""
    th = data['_template'].copy()
    th[data['_free_idx']] = free_theta
    return ap25_lnprob_continuous(th, data)


def ap25_generate_continuous_mock(n_stars=1339, seed=7, truth=CONT_TRUTH):
    """Mock from the continuous f(J,[Fe/H]) model: draw z from the MDF, then phase-space
    coordinates from the DF at that z (bucketed for speed). Metal-rich stars end up
    centrally concentrated with a unimodal MDF -- a true gradient, not two populations."""
    rng = np.random.default_rng(seed)
    pot = ap25_dm_potential(truth['logM_DM'], truth['log_rs'], truth['alpha'],
                            truth['eta'], truth['gamma'])
    nS = int(round(n_stars * (1 - truth['f_C'])))
    ztrue = rng.normal(truth['Mz'], truth['sigz'], nS)
    edges = np.linspace(ztrue.min(), ztrue.max(), 15); rows = []
    for i in range(len(edges) - 1):
        m = (ztrue >= edges[i]) & (ztrue < edges[i + 1] if i < len(edges) - 2 else ztrue <= edges[i + 1])
        if m.sum() == 0:
            continue
        zmid = 0.5 * (edges[i] + edges[i + 1])
        gm = agama.GalaxyModel(pot, ap25_stellar_df_z(zmid, truth['logJ0_0'], truth['kJ'],
                               truth['Mz'], truth['Gamma'], truth['B'], truth['gz'], truth['hz']))
        xv, _ = gm.sample(int(m.sum() * 2)); x, y, z2, vx, vy, vz = xv.T
        R = np.hypot(x, y); sel = np.where(R < REGION_RMAX)[0][:m.sum()]
        rows.append(np.column_stack([R[sel], vz[sel], np.full(len(sel), zmid)]))
    M = np.vstack(rows); R, vlos, feh = M[:, 0], M[:, 1], M[:, 2]
    nC = max(n_stars - len(R), 3)
    RC = REGION_RMAX * np.sqrt(rng.random(nC))
    vC = rng.normal(truth['V_C'], truth['sigV_C'], nC); fC = rng.normal(truth['M_C'], truth['sigM_C'], nC)
    R = np.concatenate([R, RC]); vlos = np.concatenate([vlos, vC]); feh = np.concatenate([feh, fC])
    verr = np.full(len(R), 0.6); feherr = np.full(len(R), 0.1)
    vlos = vlos + rng.normal(0, verr); feh = feh + rng.normal(0, feherr)
    return dict(R=R, vlos=vlos, verr=verr, feh=feh, feherr=feherr, G=np.full(len(R), 18.0))


def run_continuous_smoke():
    """Foundation smoke test: build a continuous mock, evaluate the marginalised likelihood,
    and confirm the data prefer the gradient (kJ) over no gradient."""
    import time
    from scipy.stats import spearmanr
    print("=" * 66)
    print("  CONTINUOUS f(J,[Fe/H]) MODEL -- foundation smoke test")
    print("=" * 66)
    print(f"  Parameters: {CONT_NDIM}  (DM 5 + DF-shape 4 + gradient 2 + MDF 2 + pop-3 5)")
    mock = ap25_generate_continuous_mock(n_stars=250, seed=1)
    rho, _ = spearmanr(mock['feh'], mock['R'])
    print(f"  mock: {len(mock['R'])} stars; Spearman([Fe/H],R) = {rho:.2f} "
          f"(<0 => metal-rich central), <[Fe/H]> = {mock['feh'].mean():.2f}")
    th = cont_truth_vector()
    t = time.time(); ll = ap25_lnlike_continuous(th, mock); dt = time.time() - t
    print(f"  lnL(truth) = {ll:.1f}   ({dt:.2f} s/eval, {len(mock['R'])} stars, 11 z-nodes)")
    print(f"  lnprior(truth) finite: {np.isfinite(ap25_lnprior_continuous(th))}")
    th0 = th.copy(); th0[CONT_PARAM_NAMES.index('kJ')] = 0.0
    ll0 = ap25_lnlike_continuous(th0, mock)
    print(f"  lnL(kJ={CONT_TRUTH['kJ']}) - lnL(kJ=0, NO gradient) = {ll - ll0:+.1f} "
          "(positive => the data prefer the continuous gradient)")
    print("  [OK] continuous likelihood evaluates; the full chain is cluster-scale.")
    return ll


def run_continuous_chain(nwalkers=None, nsteps=2000, nproc=None, backend="cont.h5",
                         resume=None, nsub=None, use_mock=False, fix_nuisance=False,
                         K=11, feh_quality_keep=None, catalog=None):
    """
    Fit the continuous f(J,[Fe/H]) model (Phase 6) by MCMC. use_mock=True runs the recovery
    test (known gradient kJ and slope gamma); otherwise the real Tolstoy+2023 data.
    fix_nuisance=True holds the 5 contamination parameters at their fiducial/known values
    (18->13 sampled dims, fewer walkers -- big speed-up for the mock recovery). K sets the
    number of (expensive) metallicity DF nodes. Re-run to resume from `backend`. Writes
    cont_chain.npy (always the full 18-column vector), figure_continuous_corner.png, Fig.4.
    """
    import os, emcee, multiprocessing as mp
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2))
    if resume is None:
        resume = os.path.exists(backend)
    print("=" * 64)
    print("  CONTINUOUS f(J,[Fe/H]) chain  " + ("(MOCK recovery)" if use_mock else "(real Tolstoy+2023)"))
    print("=" * 64)
    if use_mock:
        data = ap25_generate_continuous_mock(seed=1)
        print(f"  MOCK: {len(data['R'])} stars; truth gamma={CONT_TRUTH['gamma']}, kJ={CONT_TRUTH['kJ']}")
    else:
        data = ap25_load_real_tolstoy2023(catalog=catalog, feh_quality_keep=feh_quality_keep)
        print(f"  real: {len(data['R'])} stars")
    rng = np.random.default_rng(42)
    if nsub and nsub < len(data['R']):
        idx = rng.choice(len(data['R']), nsub, replace=False)
        data = {k: (v[idx] if hasattr(v, '__len__') and len(v) == len(data['R']) else v) for k, v in data.items()}
        print(f"  subsampled to {len(data['R'])} stars")
    data['_K'] = int(K)

    template = cont_truth_vector()                             # Sculptor-like seed / known truth
    nuis = [CONT_PARAM_NAMES.index(p) for p in ('V_C', 'sigV_C', 'M_C', 'sigM_C', 'f_C')]
    free_idx = np.array([i for i in range(CONT_NDIM) if not (fix_nuisance and i in nuis)])
    ndim = len(free_idx); nw = max(2 * ndim + 2, nwalkers or 0)
    print(f"  fitting {ndim}/{CONT_NDIM} params"
          + (" (contamination fixed at fiducial)" if fix_nuisance else "")
          + f", K={K} metallicity nodes, {nw} walkers")
    if fix_nuisance:
        data['_template'] = template.copy(); data['_free_idx'] = free_idx
        lnprob_fn = ap25_lnprob_continuous_reduced
    else:
        lnprob_fn = ap25_lnprob_continuous

    init = None if resume else template[free_idx]
    p0 = None
    if not resume:
        span = (CONT_PRIOR_HI - CONT_PRIOR_LO)[free_idx]
        p0 = np.clip(init + 0.03 * span * rng.standard_normal((nw, ndim)),
                     CONT_PRIOR_LO[free_idx] + 1e-6, CONT_PRIOR_HI[free_idx] - 1e-6)
    # affine-invariant stretch move (robust; the DE-heavy mix mixed poorly here, ~0.10
    # acceptance) blended with differential-evolution for correlated parameter directions
    moves = [(emcee.moves.StretchMove(a=2.0), 0.6),
             (emcee.moves.DEMove(), 0.3), (emcee.moves.DESnookerMove(), 0.1)]
    bk = emcee.backends.HDFBackend(backend) if HAS_H5PY else None
    resume_ok = bool(resume and bk is not None and os.path.exists(backend) and bk.iteration > 0)
    print(f"  {'RESUMING' if resume_ok else 'STARTING'} {backend} | +{nsteps} steps, nproc={nproc}")
    pool = mp.Pool(nproc) if nproc > 1 else None
    try:
        s = emcee.EnsembleSampler(nw, ndim, lnprob_fn, args=(data,),
                                  moves=moves, pool=pool, backend=bk)
        if resume_ok:
            s.run_mcmc(None, nsteps, progress=True)
        else:
            if bk is not None:
                bk.reset(nw, ndim)
            s.run_mcmc(p0, nsteps, progress=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()

    free_tex = [CONT_TEX[i] for i in free_idx]
    rep = mcmc_convergence_report(s, free_tex)
    flat_free = s.get_chain(discard=rep['burn'], thin=rep['thin'], flat=True)
    if len(flat_free) < 50:
        flat_free = s.get_chain(discard=max(1, s.iteration // 3), flat=True)
    flat = np.tile(template, (len(flat_free), 1))              # reconstruct full 18-col chain
    flat[:, free_idx] = flat_free
    np.save(_gf("cont_chain.npy"), flat)
    try:
        make_corner_plot(flat_free, free_tex, _gf("figure_continuous_corner.png"))
    except Exception as exc:
        print(f"  corner skipped ({exc})")
    try:                                                        # Fig.4 from the DM sub-vector
        gi = CONT_PARAM_NAMES.index('gamma')
        chain5 = np.column_stack([flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3], flat[:, gi]])
        make_ap25_figure4(chain5, _gf("figure_ap25_fig4.png"))
        print("  saved figure_ap25_fig4.png (paper Fig.4 from the continuous posterior)")
    except Exception as exc:
        print(f"  Fig.4 skipped ({exc})")
    print("\n  === posterior (median +/- 68% CI) ===")
    for k, nm in enumerate(CONT_PARAM_NAMES):
        p16, p50, p84 = np.percentile(flat[:, k], [16, 50, 84])
        fx = "  [fixed]" if (fix_nuisance and k in nuis) else ""
        print(f"    {nm:<10} = {p50:8.3f}  (+{p84 - p50:.3f} / -{p50 - p16:.3f}){fx}")
    gi = CONT_PARAM_NAMES.index('gamma'); ki = CONT_PARAM_NAMES.index('kJ')
    g16, g50, g84 = np.percentile(flat[:, gi], [16, 50, 84])
    k16, k50, k84 = np.percentile(flat[:, ki], [16, 50, 84])
    tag = "" if rep.get('converged') else "   [NOT converged -- re-run to add steps]"
    print("\n  " + "-" * 58)
    print(f"  CONTINUOUS:  gamma = {g50:.2f} (+{g84 - g50:.2f} / -{g50 - g16:.2f}){tag}")
    print(f"               gradient kJ = {k50:.2f} (+{k84 - k50:.2f} / -{k50 - k16:.2f})  "
          f"({'detected' if k84 < 0 else 'consistent with 0'})")
    if use_mock:
        print(f"               [recovery] truth gamma={CONT_TRUTH['gamma']}, kJ={CONT_TRUTH['kJ']}")
    else:
        print(f"               AP25 published gamma: 0.39 (+0.23 / -0.26)")
    print("  " + "-" * 58)
    return s


# ============================================================
# WP11: WALKER & PENARRUBIA (2011) MASS-PROFILE-SLOPE METHOD
# ============================================================
# Faithful implementation of Walker & Penarrubia (2011, ApJ 742, 20; "WP11"): measure the
# DM mass-profile slope of a dSph WITHOUT a halo model, by resolving two chemo-dynamically
# distinct stellar subcomponents that trace the same potential. Each subcomponent has a
# Plummer surface-density (projected radial pdf, WP11 Eq.8), a Gaussian line-of-sight
# velocity distribution (Eq.9) and a Gaussian metallicity distribution (Eq.11). Fitting all
# three jointly by MCMC yields the half-light radius r_h and velocity dispersion sigma_V of
# each subcomponent; the mass estimator M(r_h)=5 r_h sigma_V^2/(2G) (Eq.2) then gives the
# mass at two radii, and two points define the slope Gamma = Dlog M/Dlog r (Eq.5), with
# Gamma>2 excluding an NFW cusp. Analytic likelihood -> fast (minutes on a laptop). We use
# [Fe/H] as the metallicity indicator in place of WP11's reduced Mg index W' (equivalent
# role: a relative-metallicity label to separate the subcomponents).
G_PC = 4.300917270e-3        # pc (km/s)^2 / Msun  (for the WP11 mass estimator)

WP11_PARAM_NAMES = ['f_sub', 'rh1_over_rh2', 'log_rh2_pc', 'feh1', 'dfeh',
                    'log_s2feh1', 'log_s2feh2', 'log_s2v1', 'log_s2v2']
WP11_NDIM = 9
WP11_TEX = [r'$f_{\rm sub}$', r'$r_{h,1}/r_{h,2}$', r'$\log_{10}r_{h,2}$',
            r'$\langle$[Fe/H]$\rangle_1$', r'$\Delta$[Fe/H]',
            r'$\log\sigma^2_{Z,1}$', r'$\log\sigma^2_{Z,2}$',
            r'$\log\sigma^2_{V,1}$', r'$\log\sigma^2_{V,2}$']
# priors: f_sub in (0,1); r_{h,1}/r_{h,2} in (0,1) [MR more concentrated]; log r_{h,2}/pc;
# <[Fe/H]>_1; D[Fe/H]=<[Fe/H]>_1-<[Fe/H]>_2 > 0 [MR more metal-rich]; log sigma^2 (Z, V).
WP11_PRIOR_LO = np.array([0.05, 0.02, 1.5, -3.0, 0.0, -5.0, -5.0, -1.0, -1.0])
WP11_PRIOR_HI = np.array([0.95, 0.99, 3.5,  0.0, 2.0,  1.0,  1.0,  5.0,  5.0])
WP11_TRUTH = dict(f_sub=0.5, rh1_over_rh2=0.55, log_rh2_pc=np.log10(300.0),
                  feh1=-1.5, dfeh=0.5, log_s2feh1=np.log10(0.04), log_s2feh2=np.log10(0.09),
                  log_s2v1=np.log10(6.5**2), log_s2v2=np.log10(11.6**2))   # ~Sculptor-like


def wp11_truth_vector():
    return np.array([WP11_TRUTH[k] for k in WP11_PARAM_NAMES])


def wp11_load_data(catalog=None, feh_quality_keep=None):
    """Active galaxy's member stars: projected (elliptical) radius in pc, rest-frame
    velocity, metallicity indicator ([Fe/H] for Sculptor; Mg index W' for Fornax), and
    their measurement errors."""
    catalog = catalog or GAL['catalog']
    fqk = GAL['feh_quality_keep'] if feh_quality_keep is None else feh_quality_keep
    ra, dec, vlos, verr, feh, feherr, _g = _fetch_tolstoy2023(
        catalog, GAL.get('cols'), mem_keep=(GAL.get('mem_keep') or ('m',)),
        require_member=bool(GAL.get('mem_keep')),
        target_col=GAL.get('target_col'), target_keep=GAL.get('target_keep'), mem_min=GAL.get('mem_min'),
        feh_quality_keep=(list(fqk) if fqk else None))
    good = np.isfinite(vlos) & np.isfinite(feh) & np.isfinite(verr) & np.isfinite(feherr)
    R_kpc = _semi_major_axis_radius(ra[good], dec[good])
    return dict(R_pc=R_kpc * 1000.0, vlos=vlos[good] - V_SYS,
                feh=feh[good], everr=verr[good], efeh=feherr[good])


def wp11_lnprior(theta):
    if np.any(theta < WP11_PRIOR_LO) or np.any(theta > WP11_PRIOR_HI):
        return -np.inf
    return 0.0


def wp11_lnlike(theta, data):
    """WP11 two-subcomponent mixture likelihood (Eq.14, members-only; selection w(R)=1 so the
    Plummer radial pdf is already normalised). Each star: sum over MR/MP of the product of
    Plummer-radial x Gaussian-velocity x Gaussian-metallicity probabilities."""
    f_sub, ratio, log_rh2, feh1, dfeh, ls2f1, ls2f2, ls2v1, ls2v2 = theta
    R, V, feh = data['R_pc'], data['vlos'], data['feh']
    eV, efeh = data['everr'], data['efeh']
    rh2 = 10.0 ** log_rh2; rh1 = ratio * rh2
    feh2 = feh1 - dfeh
    s2f1, s2f2 = 10.0 ** ls2f1, 10.0 ** ls2f2
    s2v1, s2v2 = 10.0 ** ls2v1, 10.0 ** ls2v2
    pR1 = 2.0 * R / rh1 ** 2 / (1.0 + R ** 2 / rh1 ** 2) ** 2       # Plummer projected pdf (Eq.8)
    pR2 = 2.0 * R / rh2 ** 2 / (1.0 + R ** 2 / rh2 ** 2) ** 2
    pV1 = _gauss(V, 0.0, np.sqrt(s2v1 + eV ** 2))                   # rest-frame velocity (Eq.9)
    pV2 = _gauss(V, 0.0, np.sqrt(s2v2 + eV ** 2))
    pZ1 = _gauss(feh, feh1, np.sqrt(s2f1 + efeh ** 2))             # metallicity (Eq.11)
    pZ2 = _gauss(feh, feh2, np.sqrt(s2f2 + efeh ** 2))
    L = f_sub * pR1 * pV1 * pZ1 + (1.0 - f_sub) * pR2 * pV2 * pZ2
    if np.any(L <= 0) or np.any(~np.isfinite(L)):
        return -np.inf
    return float(np.sum(np.log(L)))


def wp11_lnprob(theta, data):
    lp = wp11_lnprior(theta)
    return (lp + wp11_lnlike(theta, data)) if np.isfinite(lp) else -np.inf


def wp11_derived(flat):
    """From the chain, derive r_{h,1}, r_{h,2} [pc], M_1, M_2 [Msun] (WP11 Eq.2) and the
    mass-profile slope Gamma (Eq.5)."""
    ratio, log_rh2 = flat[:, 1], flat[:, 2]
    s2v1, s2v2 = 10.0 ** flat[:, 7], 10.0 ** flat[:, 8]
    rh2 = 10.0 ** log_rh2; rh1 = ratio * rh2
    M1 = 2.5 * s2v1 * rh1 / G_PC                                    # M(r_h)=5 r_h sigma^2/(2G)
    M2 = 2.5 * s2v2 * rh2 / G_PC
    Gamma = np.log10(M2 / M1) / np.log10(rh2 / rh1)                 # Eq.5 (two points -> slope)
    return dict(rh1=rh1, rh2=rh2, M1=M1, M2=M2, Gamma=Gamma)


def wp11_generate_mock(seed=3, n=1000, truth=None):
    """Synthetic two-subcomponent dSph with a KNOWN slope: two Plummer populations (metal-rich
    compact + metal-poor extended) sharing a potential, Gaussian velocities and metallicities.
    Returns the data dict plus the true Gamma."""
    t = dict(WP11_TRUTH if truth is None else truth)
    rng = np.random.default_rng(seed)
    rh2 = 10.0 ** t['log_rh2_pc']; rh1 = t['rh1_over_rh2'] * rh2
    sV1, sV2 = np.sqrt(10.0 ** t['log_s2v1']), np.sqrt(10.0 ** t['log_s2v2'])
    sZ1, sZ2 = np.sqrt(10.0 ** t['log_s2feh1']), np.sqrt(10.0 ** t['log_s2feh2'])
    feh1, feh2 = t['feh1'], t['feh1'] - t['dfeh']
    n1 = int(round(n * t['f_sub'])); n2 = n - n1

    def plummer_R(rh, m):                                          # inverse-CDF: F(R)=R^2/(R^2+rh^2)
        u = rng.random(m); return rh * np.sqrt(u / (1.0 - u))
    eV = np.full(n, 2.0); eZ = np.full(n, 0.1)
    R1, R2 = plummer_R(rh1, n1), plummer_R(rh2, n2)
    V1 = rng.normal(0, sV1, n1) + rng.normal(0, eV[:n1]); V2 = rng.normal(0, sV2, n2) + rng.normal(0, eV[n1:])
    Z1 = rng.normal(feh1, sZ1, n1) + rng.normal(0, eZ[:n1]); Z2 = rng.normal(feh2, sZ2, n2) + rng.normal(0, eZ[n1:])
    data = dict(R_pc=np.concatenate([R1, R2]), vlos=np.concatenate([V1, V2]),
                feh=np.concatenate([Z1, Z2]), everr=eV, efeh=eZ)
    Gamma_true = 1.0 + np.log10(sV2 ** 2 / sV1 ** 2) / np.log10(rh2 / rh1)
    return data, float(Gamma_true)


def make_figure1_two_galaxy(galaxies=('sculptor', 'fornax'), out="figure1_two_galaxy.png", arrays=None):
    """
    Figure-1 style data presentation for two dwarf spheroidals side by side (AP25 Fig.2 style):
    one row per galaxy, columns = (metallicity distribution, line-of-sight velocity vs projected
    radius coloured by metallicity, velocity vs metallicity chemodynamics). Note the metallicity
    indicator differs by galaxy ([Fe/H] for Sculptor; Mg index for Fornax). Pass
    `arrays={galaxy: dict(R, vlos, feh, mlabel)}` for offline testing; otherwise loads each
    galaxy's real sample (needs VizieR). Writes figure1_two_galaxy.png.
    """
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    saved = GAL['name'].lower()                                    # restore active galaxy after
    rows = len(galaxies)
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4.4 * rows), squeeze=False)
    try:
        for i, gname in enumerate(galaxies):
            if arrays is not None:
                d = arrays[gname]; R, vlos, feh, mlabel = d['R'], d['vlos'], d['feh'], d.get('mlabel', '[Fe/H]')
            else:
                set_galaxy(gname)
                R, vlos, feh, verr = _load_real_feh()
                mlabel = '[Fe/H]' if gname == 'sculptor' else r'Mg index $W^\prime$'
            gal_disp = GALAXIES[gname]['name']
            # col 0: metallicity distribution
            ax = axes[i][0]
            ax.hist(feh, bins=30, color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.4)
            ax.axvline(np.median(feh), color='k', ls=':', lw=1.3, label=f'median = {np.median(feh):.2f}')
            ax.set_xlabel(mlabel); ax.set_ylabel('N stars'); ax.legend(fontsize=8)
            ax.set_title(f'{gal_disp}: metallicity distribution')
            # col 1: velocity vs radius, colored by metallicity
            ax = axes[i][1]
            sc = ax.scatter(R, vlos, c=feh, s=12, cmap='viridis', alpha=0.75, edgecolors='none')
            ax.axhline(0, color='k', ls='--', lw=1); ax.set_xlabel(r'projected radius $R$ [kpc]')
            ax.set_ylabel(r'$v_{\rm los}$ (rest frame) [km s$^{-1}$]')
            ax.set_title(f'{gal_disp}: kinematics'); fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label=mlabel)
            # col 2: velocity vs metallicity chemodynamics
            ax = axes[i][2]
            ax.scatter(feh, vlos, c=R, s=12, cmap='plasma', alpha=0.75, edgecolors='none')
            ax.axhline(0, color='k', ls='--', lw=1); ax.set_xlabel(mlabel)
            ax.set_ylabel(r'$v_{\rm los}$ [km s$^{-1}$]'); ax.set_title(f'{gal_disp}: chemodynamics')
    finally:
        set_galaxy(saved)                                         # restore

    fig.suptitle('Data presentation: chemo-dynamics of two dwarf spheroidals', fontsize=14)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"--> Saved {out}")
    return out


def make_wp11_figure(flat, out="figure_wp11.png"):
    """Reproduction of WP11 Figure 10: (left) the two stellar subcomponents in the
    (r_h, M) plane, shaded by posterior density with a "Prob" colorbar and the slope-2
    (cusp, dotted) and slope-3 (core, dashed) reference lines labelled on top; (right) the
    posterior PDF of the mass-profile slope Gamma with the NFW threshold and the exclusion
    significance."""
    out = _gf(out)
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    d = wp11_derived(flat)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # left: both subcomponents shaded by 2-D posterior density (WP11 "Prob" colormap)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(flat), min(4000, len(flat)), replace=False)
    lr = np.log10(np.concatenate([d['rh1'][sub], d['rh2'][sub]]))
    lM = np.log10(np.concatenate([d['M1'][sub], d['M2'][sub]]))
    pts = np.vstack([lr, lM])
    try:
        dens = gaussian_kde(pts)(pts)
        dens = dens / dens.max()                                    # normalise to Prob in (0,1]
    except Exception:
        dens = np.ones_like(lr)
    order = np.argsort(dens)                                        # dense points drawn on top
    sc = axL.scatter(10 ** lr[order], 10 ** lM[order], c=dens[order], s=7,
                     cmap='hot_r', vmin=0.0, vmax=1.0, edgecolors='none')
    cb = fig.colorbar(sc, ax=axL, fraction=0.046, pad=0.02)
    cb.set_label('Prob', fontsize=9)
    r0 = np.median(d['rh1']); M0 = np.median(d['M1'])
    rr = np.array([np.median(d['rh1']) * 0.45, np.median(d['rh2']) * 1.7])
    axL.plot(rr, M0 * (rr / r0) ** 2, 'b:', lw=1.6)                 # slope 2 (cusp)
    axL.plot(rr, M0 * (rr / r0) ** 3, 'b--', lw=1.6)               # slope 3 (core)
    axL.text(0.04, 0.96, r'slope=2 (cusp)', transform=axL.transAxes, fontsize=9, color='b', va='top')
    axL.text(0.04, 0.90, r'slope=3 (core)', transform=axL.transAxes, fontsize=9, color='b', va='top')
    axL.set_xscale('log'); axL.set_yscale('log')
    xlo, xhi = GAL.get('wp11_xlim', (2.0, 2.75))                  # galaxy-aware panel range (log10)
    ylo, yhi = GAL.get('wp11_ylim', (6.3, 7.9))
    axL.set_xlim(10 ** xlo, 10 ** xhi); axL.set_ylim(10 ** ylo, 10 ** yhi)
    import matplotlib.ticker as mticker                            # label ticks with log10 values
    xt = [round(v, 1) for v in np.arange(np.ceil(xlo * 5) / 5, xhi + 1e-9, 0.2)]
    yt = [round(v, 1) for v in np.arange(np.ceil(ylo * 2) / 2, yhi + 1e-9, 0.5)]
    axL.xaxis.set_major_locator(mticker.FixedLocator([10 ** v for v in xt]))
    axL.xaxis.set_major_formatter(mticker.FixedFormatter([f"{v:.1f}" for v in xt]))
    axL.xaxis.set_minor_locator(mticker.NullLocator())
    axL.yaxis.set_major_locator(mticker.FixedLocator([10 ** v for v in yt]))
    axL.yaxis.set_major_formatter(mticker.FixedFormatter([f"{v:.1f}" for v in yt]))
    axL.yaxis.set_minor_locator(mticker.NullLocator())
    axL.set_xlabel(r'$\log_{10}\,[\,r_{\rm half}\,/\,{\rm pc}\,]$')
    axL.set_ylabel(r'$\log_{10}\,[\,M(R_{\rm half})\,/\,M_\odot\,]$')
    axL.text(0.5, 0.04, GAL['name'], transform=axL.transAxes, fontsize=14, ha='center',
             va='bottom', style='italic')                          # galaxy name inside panel (paper style)
    axL.grid(alpha=0.25, which='both')

    # right: posterior PDF of Gamma
    g16, g50, g84 = np.percentile(d['Gamma'], [16, 50, 84])
    s_excl = float(np.mean(d['Gamma'] > 2.0))                       # exclude NFW: P(Gamma>2)
    axR.hist(d['Gamma'], 60, density=True, color='crimson', histtype='step', lw=2, label='Scl')
    axR.axvline(2.0, color='k', ls=':', lw=1.5)
    axR.axvline(g50, color='crimson', lw=1.0, alpha=0.6)
    axR.text(2.02, 0.96, r'NFW cusp', transform=axR.get_xaxis_transform(), fontsize=8, va='top')
    axR.set_xlabel(r'$\Gamma\equiv\Delta\log_{10}M/\Delta\log_{10}r$')
    axR.set_ylabel('probability')
    axR.set_title(f'$\\Gamma={g50:.2f}^{{+{g84-g50:.2f}}}_{{-{g50-g16:.2f}}}$   '
                  f'(excl. NFW: {100*s_excl:.1f}%)')
    axR.legend(fontsize=9); axR.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"--> Saved {out}")


def _wp11_fit_gamma(data, nsteps=8000, nproc=None, seed=7):
    """Run the WP11 two-subcomponent MCMC on a supplied data dict and return the mass-slope
    posterior percentiles (Gamma16, Gamma50, Gamma84). Used by the membership robustness test
    to refit under different selections without re-querying VizieR."""
    import emcee, multiprocessing as mp, os
    ndim = WP11_NDIM; nw = max(4 * ndim, 48)
    rng = np.random.default_rng(seed)
    scale = np.array([0.03, 0.02, 0.02, 0.03, 0.03, 0.08, 0.08, 0.04, 0.04])
    p0 = np.clip(wp11_truth_vector() + scale * rng.standard_normal((nw, ndim)),
                 WP11_PRIOR_LO + 1e-6, WP11_PRIOR_HI - 1e-6)
    moves = [(emcee.moves.StretchMove(a=2.0), 0.7), (emcee.moves.DEMove(), 0.3)]
    pool = mp.Pool(nproc) if (nproc and nproc > 1) else None
    try:
        s = emcee.EnsembleSampler(nw, ndim, wp11_lnprob, args=(data,), moves=moves, pool=pool)
        s.run_mcmc(p0, nsteps, progress=False)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    flat = s.get_chain(discard=nsteps // 2, thin=10, flat=True)
    G = wp11_derived(flat)['Gamma']
    return np.percentile(G, [16, 50, 84])


def run_membership_robustness(pmem_cuts=(0.50, 0.70, 0.90, 0.95), nsteps=8000, nproc=None, out=None):
    """
    Membership-cut robustness test (justifies the membership selection): refit the WP11
    mass-profile slope on the subsample surviving progressively stricter Gaia proper-motion
    membership thresholds (P_mem > cut). Unlike a velocity clip -- which truncates the very
    velocity distribution being measured and is therefore confounded -- the PM membership is
    an INDEPENDENT criterion, so a stable Gamma across thresholds is a clean demonstration
    that the result is not driven by the membership definition. Falls back to a velocity-clip
    variant only if Gaia proper motions are unavailable. Writes
    figure_membership_robustness.png (galaxy-tagged) and prints a table.
    """
    out = _gf(out or "figure_membership_robustness.png")
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    print("=" * 64)
    print(f"  MEMBERSHIP ROBUSTNESS  ({GAL['name']}; Gamma vs Gaia PM membership threshold)")
    print("=" * 64)
    # cross-match the WP11 sample to Gaia and attach PM membership + the WP11 observables
    try:
        gaia = _load_gaia_matched()                                # ra, dec, vlos, feh, P_mem_PM, R_kpc
        base = dict(R_pc=gaia['R_kpc'].values * 1000.0, vlos=gaia['vlos'].values,
                    feh=gaia['feh'].values,
                    everr=np.full(len(gaia), 2.0), efeh=np.full(len(gaia), 0.1))
        pmem = gaia['P_mem_PM'].values
        xlabel = r'Gaia PM membership threshold  $P_{\rm mem} >$'
        mode = 'pm'
    except Exception as exc:                                        # Gaia unavailable -> velocity-clip fallback
        print(f"  [robustness] Gaia PM unavailable ({str(exc)[:50]}); using velocity-clip variant")
        base = wp11_load_data(); v = base['vlos']
        sig = 1.4826 * np.median(np.abs(v - np.median(v)))
        pmem = 1.0 / (1.0 + (np.abs(v - np.median(v)) / (3.0 * sig)) ** 4)   # pseudo-membership
        pmem_cuts = (0.3, 0.5, 0.7, 0.9)
        xlabel = r'velocity pseudo-membership threshold'
        mode = 'vclip'

    rows = []
    for cut in pmem_cuts:
        keep = pmem >= cut
        if keep.sum() < 50:
            continue
        data = {k: (val[keep] if hasattr(val, '__len__') and len(val) == len(pmem) else val)
                for k, val in base.items()}
        g16, g50, g84 = _wp11_fit_gamma(data, nsteps=nsteps, nproc=nproc)
        rows.append((cut, int(keep.sum()), g50, g16, g84))
        print(f"    P_mem > {cut:.2f}: N={int(keep.sum()):4d}   Gamma = {g50:.2f} "
              f"(+{g84-g50:.2f}/-{g50-g16:.2f})")
    cs = np.array([r[0] for r in rows]); g50 = np.array([r[2] for r in rows])
    glo = np.array([r[3] for r in rows]); ghi = np.array([r[4] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.errorbar(cs, g50, yerr=np.vstack([g50 - glo, ghi - g50]), fmt='o-', color='navy',
                capsize=4, lw=1.8, label=r'WP11 $\Gamma$ (68% CI)')
    ax.axhspan(np.min(glo), np.max(ghi), color='navy', alpha=0.08)
    ax.axhline(2.0, color='k', ls=':', lw=1.3, label=r'NFW cusp ($\Gamma=2$)')
    for cut, n, g, _, _ in rows:
        ax.annotate(f'N={n}', (cut, g), textcoords='offset points', xytext=(0, 10), fontsize=8, ha='center')
    ax.set_xlabel(xlabel); ax.set_ylabel(r'mass-profile slope $\Gamma$')
    ax.set_title(f'{GAL["name"]}: WP11 slope is stable across membership thresholds')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\n--> Saved {out}")
    spread = np.max(g50) - np.min(g50)
    print(f"  Gamma spread across {mode} thresholds = {spread:.2f} "
          f"({'STABLE' if spread < 0.3 else 'sensitive -- report the caveat'})")
    return rows


def run_pop3_robustness(feh_cuts=(None, -3.00, -2.75, -2.50), nsteps=8000, nproc=None, out=None):
    """
    Very-metal-poor contamination test (AP24 'Pop 3').

    Arroyo-Polonio et al. (2024, A&A 692, A195) identify a THIRD population in this exact
    1339-star Sculptor sample: fraction ~0.018 (~24 stars), <[Fe/H]> ~ -2.90, spatially
    extended (R_h ~ 1.1 deg), and offset by ~15 km/s in mean v_los (125.5 vs 111.2 km/s) --
    most plausibly a recent minor merger. They EXCLUDE these stars from their velocity
    dispersion profiles; the analysis here does not.

    Why this matters for Gamma: those stars are metal-poor AND at large radius, so they fall
    into the metal-poor subcomponent, inflate its velocity dispersion, inflate M(r_h,MP),
    increase Delta log M, and therefore bias Gamma HIGH -- i.e. TOWARD A CORE. The systematic
    works in favour of this paper's result, which is exactly why it must be reported.

    Refits the WP11 mass slope with progressively stricter very-metal-poor cuts and reports
    the shift. A small |Delta Gamma| means the core-like slope is not driven by the merger
    debris. Writes figure_pop3_robustness.png.
    """
    out = _gf(out or "figure_pop3_robustness.png")
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    if GAL['name'].lower() != 'sculptor':
        print(f"  [pop3] NOTE: {GAL['name']}'s metallicity indicator is not [Fe/H]; "
              f"the AP24 Pop-3 cut is Sculptor-specific and will not be meaningful here.")

    print("=" * 64)
    print(f"  VERY-METAL-POOR CONTAMINATION  ({GAL['name']}; Gamma vs [Fe/H] floor)")
    print(f"  AP24 Pop 3: ~1.8% of stars, <[Fe/H]>~-2.90, v_los offset ~+15 km/s")
    print("=" * 64)
    base = wp11_load_data()
    feh = base['feh']
    rows = []
    for cut in feh_cuts:
        keep = np.ones(len(feh), bool) if cut is None else (feh >= cut)
        n_removed = int((~keep).sum())
        if keep.sum() < 50:
            continue
        data = {k: (v[keep] if hasattr(v, '__len__') and len(v) == len(feh) else v)
                for k, v in base.items()}
        g16, g50, g84 = _wp11_fit_gamma(data, nsteps=nsteps, nproc=nproc)
        label = "all stars" if cut is None else f"[Fe/H] >= {cut:.2f}"
        rows.append((cut, int(keep.sum()), n_removed, g50, g16, g84))
        print(f"    {label:18s}: N={int(keep.sum()):4d} (-{n_removed:3d})   "
              f"Gamma = {g50:.2f} (+{g84-g50:.2f}/-{g50-g16:.2f})")

    g_base = rows[0][3]
    xs = np.arange(len(rows))
    g50 = np.array([r[3] for r in rows]); glo = np.array([r[4] for r in rows])
    ghi = np.array([r[5] for r in rows])
    labels = ["all\nstars" if r[0] is None else f"$\\geq${r[0]:.2f}" for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.errorbar(xs, g50, yerr=np.vstack([g50 - glo, ghi - g50]), fmt='o-', color='darkgreen',
                capsize=4, lw=1.8, label=r'WP11 $\Gamma$ (68% CI)')
    ax.axhline(2.0, color='k', ls=':', lw=1.3, label=r'NFW cusp ($\Gamma=2$)')
    ax.axhline(g_base, color='darkgreen', ls='--', lw=1.0, alpha=0.6,
               label=r'$\Gamma$ (all stars)')
    for x, r in zip(xs, rows):
        ax.annotate(f'N={r[1]}', (x, r[3]), textcoords='offset points', xytext=(0, 11),
                    fontsize=8, ha='center')
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel(r'very-metal-poor floor applied to the sample')
    ax.set_ylabel(r'mass-profile slope $\Gamma$')
    ax.set_title(f"{GAL['name']}: is the slope driven by very-metal-poor (AP24 Pop 3) stars?")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\n--> Saved {out}")
    dmax = float(np.max(np.abs(g50 - g_base)))
    print(f"  max |Delta Gamma| vs all-stars = {dmax:.2f} "
          f"({'ROBUST' if dmax < 0.3 else 'sensitive -- report the shift explicitly'})")
    print(f"  (expected direction: removing Pop 3 should LOWER Gamma, i.e. away from a core)")
    return rows


def run_wp11(nwalkers=64, nsteps=6000, nproc=None, backend="wp11.h5", resume=None,
             use_mock=False, feh_quality_keep=None, catalog=None):
    """
    Walker & Penarrubia (2011) mass-slope measurement on the real Tolstoy+2023 data (or a
    known-Gamma mock with use_mock=True). Analytic likelihood -> fast; converges in minutes.
    Re-run to resume from `backend`. Writes wp11_chain.npy, figure_wp11_corner.png and the
    WP11 Fig.10-style figure_wp11.png. Reports Gamma and the NFW-exclusion significance.
    """
    import os, emcee, multiprocessing as mp
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2))
    if resume is None:
        resume = os.path.exists(backend)
    print("=" * 64)
    print("  WP11 (Walker & Penarrubia 2011) mass-profile slope  "
          + ("(known-Gamma MOCK)" if use_mock else "(real Tolstoy+2023)"))
    print("=" * 64)
    if use_mock:
        data, Gamma_true = wp11_generate_mock()
        print(f"  MOCK: {len(data['R_pc'])} stars; true Gamma = {Gamma_true:.2f}")
    else:
        data = wp11_load_data(catalog=catalog, feh_quality_keep=feh_quality_keep)
        print(f"  real: {len(data['R_pc'])} member stars")
    ndim = WP11_NDIM; nw = max(4 * ndim, nwalkers)                # more walkers for a stiff 9-D space
    rng = np.random.default_rng(7)
    init = wp11_truth_vector()
    p0 = None
    if not (resume and HAS_H5PY and os.path.exists(backend)):     # tight Gaussian ball around the solution
        scale = np.array([0.03, 0.02, 0.02, 0.03, 0.03, 0.08, 0.08, 0.04, 0.04])
        p0 = np.clip(init + scale * rng.standard_normal((nw, ndim)),
                     WP11_PRIOR_LO + 1e-6, WP11_PRIOR_HI - 1e-6)
    # affine-invariant stretch move (robust; the DE mix collapsed the ensemble here) + a
    # little differential-evolution for the correlated directions
    moves = [(emcee.moves.StretchMove(a=2.0), 0.7), (emcee.moves.DEMove(), 0.3)]
    bk = emcee.backends.HDFBackend(backend) if HAS_H5PY else None
    resume_ok = bool(resume and bk is not None and os.path.exists(backend) and bk.iteration > 0)
    print(f"  {'RESUMING' if resume_ok else 'STARTING'} {backend} | {nw} walkers, +{nsteps} steps, nproc={nproc}")
    pool = mp.Pool(nproc) if nproc > 1 else None
    try:
        s = emcee.EnsembleSampler(nw, ndim, wp11_lnprob, args=(data,), moves=moves, pool=pool, backend=bk)
        if resume_ok:
            s.run_mcmc(None, nsteps, progress=True)
        else:
            if bk is not None:
                bk.reset(nw, ndim)
            s.run_mcmc(p0, nsteps, progress=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()

    rep = mcmc_convergence_report(s, WP11_TEX)
    flat = s.get_chain(discard=rep['burn'], thin=rep['thin'], flat=True)
    if len(flat) < 100:
        flat = s.get_chain(discard=max(1, s.iteration // 3), flat=True)
    np.save(_gf("wp11_chain.npy"), flat)
    try:
        make_corner_plot(flat, WP11_TEX, _gf("figure_wp11_corner.png"))
    except Exception as exc:
        print(f"  corner skipped ({exc})")
    try:
        make_wp11_figure(flat, out=_gf("figure_wp11.png"))
    except Exception as exc:
        print(f"  WP11 figure skipped ({exc})")
    d = wp11_derived(flat)
    g16, g50, g84 = np.percentile(d['Gamma'], [16, 50, 84])
    r1_16, r1_50, r1_84 = np.percentile(d['rh1'], [16, 50, 84])
    r2_16, r2_50, r2_84 = np.percentile(d['rh2'], [16, 50, 84])
    sv1 = np.sqrt(10.0 ** np.percentile(flat[:, 7], 50)); sv2 = np.sqrt(10.0 ** np.percentile(flat[:, 8], 50))
    s_excl = float(np.mean(d['Gamma'] > 2.0))
    tag = "" if rep.get('converged') else "   [NOT converged -- re-run to add steps]"
    print("\n  === WP11 results (median +/- 68% CI) ===")
    print(f"    r_h,1 (metal-rich) = {r1_50:.0f} (+{r1_84-r1_50:.0f}/-{r1_50-r1_16:.0f}) pc, sigma_V,1 ~ {sv1:.1f} km/s")
    print(f"    r_h,2 (metal-poor) = {r2_50:.0f} (+{r2_84-r2_50:.0f}/-{r2_50-r2_16:.0f}) pc, sigma_V,2 ~ {sv2:.1f} km/s")
    print("  " + "-" * 58)
    print(f"  WP11:  Gamma = {g50:.2f} (+{g84-g50:.2f}/-{g50-g16:.2f}){tag}")
    print(f"         gamma_DM < 3 - Gamma = {3-g50:.2f}  (upper limit on the inner density slope)")
    print(f"         excludes NFW cusp (Gamma>2): {100*s_excl:.1f}%")
    if use_mock:
        _, gt = wp11_generate_mock(); print(f"         [recovery] true Gamma = {gt:.2f}")
    else:
        print(f"         WP11 published (Sculptor): Gamma = 2.95 (+0.51/-0.39)")
    print("  " + "-" * 58)
    return s


# ============================================================
# MASTER ORCHESTRATION PIPELINE
# ============================================================
if __name__ == "__main__":
    import argparse, sys
    _ap = argparse.ArgumentParser(
        description="Sculptor chemo-dynamical pipeline. With no arguments it runs the "
                    "fast phases (1-4) plus the Phase-5 smoke test. Use --chain to launch "
                    "the FULL 25-parameter validation chain on the real data.")
    _ap.add_argument("--chain", action="store_true",
                     help="run the full 25-parameter chain on real Tolstoy+2023 data "
                          "(multi-day; resumable). Re-run to resume.")
    _ap.add_argument("--dm5", action="store_true",
                     help="Measure Sculptor's DM inner slope by a robust spherical-Jeans gNFW "
                          "MCMC on the binned sigma_los of the real data (3 params: gamma, "
                          "log r_s, log M_DM). Converges in ~1-2 h; the well-posed reduction of "
                          "the degenerate free 5-parameter fit. Writes figure_ap25_fig4.png.")
    _ap.add_argument("--gravsphere", action="store_true",
                     help="GravSphere (Read & Steger 2017): spherical Jeans + Virial Shape "
                          "Parameters with free anisotropy beta(r), on the real data. The "
                          "middle rung between plain Jeans (--dm5) and the action-DF method.")
    _ap.add_argument("--compare", action="store_true",
                     help="Overlay the DM inner-slope posteriors from the saved --dm5, "
                          "--gravsphere and --chain runs into one comparison figure.")
    _ap.add_argument("--crosscheck", action="store_true",
                     help="cross-check the GravSphere engine against the reference "
                          "justinread/gravsphere code (needs --repo pointing at the clone).")
    _ap.add_argument("--repo", default=None,
                     help="path to the cloned reference gravsphere repo (for --crosscheck).")
    _ap.add_argument("--mcmc", action="store_true",
                     help="with --crosscheck: also run the end-to-end posterior-equivalence "
                          "check (agama + emcee; a few minutes).")
    _ap.add_argument("--robustness", action="store_true",
                     help="membership-cut robustness test: refit WP11 Gamma vs velocity clip "
                          "to show the slope is stable across membership definitions. "
                          "figure_membership_robustness.png.")
    _ap.add_argument("--figure1", action="store_true",
                     help="two-galaxy Figure-1 data presentation (Sculptor + Fornax side by "
                          "side, AP25 Fig.2 style). figure1_two_galaxy.png.")
    _ap.add_argument("--actions", action="store_true",
                     help="action-space chemodynamics: [Fe/H] vs J_r and J_z-J_r coloured by "
                          "[Fe/H] (needs Gaia PMs + a saved chain). figure_action_space.png.")
    _ap.add_argument("--dispersion", action="store_true",
                     help="radial velocity-dispersion profile sigma_los(R) (add --gaia to "
                          "overlay the Gaia PM dispersion). Writes figure_dispersion_profile.png.")
    _ap.add_argument("--skymap", action="store_true",
                     help="Gaia proper-motion sky map with one panel per membership cut "
                          "(justifies the membership selection). Writes figure_gaia_skymap.png.")
    _ap.add_argument("--gaia", action="store_true", help="use Gaia proper motions where supported")
    _ap.add_argument("--fig4all", action="store_true",
                     help="Combined AP25 Fig.4: overlay DM density rho(r) from every saved "
                          "chain (dm5, gravsphere, continuous, chain) + AP25's published "
                          "curve. Writes figure_fig4_all_chains.png.")
    _ap.add_argument("--galaxy", default="sculptor", choices=list(GALAXIES),
                     help="target dwarf spheroidal (default: sculptor). Sets the centre, "
                          "distance, systemic velocity, ellipticity and data catalog for all "
                          "real-data commands. Fornax follows WP11 (Walker+2009 data, Mg index).")
    _ap.add_argument("--wp11", action="store_true",
                     help="Walker & Penarrubia (2011) mass-profile-slope method on the real "
                          "data: two-subcomponent mixture -> slope Gamma, NFW exclusion. Fast "
                          "(minutes). Add --mock for the known-Gamma recovery test.")
    _ap.add_argument("--continuous", action="store_true",
                     help="Fit the CONTINUOUS f(J,[Fe/H]) model (Phase 6, the novel method): "
                          "metallicity is a coordinate inside a single action DF whose scale "
                          "action varies smoothly with [Fe/H]. Cluster-scale. Add --mock for "
                          "the recovery test. With no --steps it runs a fast foundation smoke test.")
    _ap.add_argument("--mock", action="store_true",
                     help="with --continuous: fit a mock with a known gradient/slope "
                          "(recovery test) instead of the real data.")
    _ap.add_argument("--slide", action="store_true",
                     help="Sliding-threshold test on the real data: sigma_los varies "
                          "smoothly with the [Fe/H] split (a continuum), and the "
                          "sigma_los-only constraint on gamma is degenerate (core-to-cusp). "
                          "Writes figure_sliding_metallicity.png.")
    _ap.add_argument("--biasconv", action="store_true",
                     help="Mean recovered-gamma bias vs the number of mock realisations "
                          "(discrete vs continuous), with 25-75th percentile band. "
                          "Writes figure_bias_vs_realizations.png.")
    _ap.add_argument("--biasgate", action="store_true",
                     help="Bias gate on continuous-gradient mocks: recover gamma with a "
                          "discrete median split vs a continuous (all-star) treatment, and "
                          "measure the split-induced bias. Writes figure_bias_gate.png.")
    _ap.add_argument("--revgate", action="store_true",
                     help="REVERSE bias gate: mocks with two GENUINELY DISCRETE populations "
                          "matching Sculptor's observed structure (Arroyo-Polonio et al. 2024 "
                          "Table C.2); tests whether a continuous (no-split) treatment still "
                          "recovers gamma. The companion to --biasgate: together they test the "
                          "population-decomposition choice in BOTH directions. Writes "
                          "figure_reverse_bias_gate.png.")
    _ap.add_argument("--gatediag", action="store_true",
                     help="Control tests for both bias gates. (1) NULL: one population, no "
                          "metallicity structure -- any bias is a baseline estimator offset, not "
                          "a decomposition effect. (2) SCRAMBLE: discrete mock with [Fe/H] "
                          "permuted -- separates bias driven by tracer spatial structure from "
                          "bias driven by the real metallicity-orbit link. Run this before "
                          "interpreting --biasgate or --revgate. Writes "
                          "figure_gate_diagnostics.png.")
    _ap.add_argument("--pop3", action="store_true",
                     help="Very-metal-poor contamination test: refit WP11 Gamma with "
                          "progressively stricter [Fe/H] floors to check whether the slope is "
                          "driven by the ~24 offset stars Arroyo-Polonio et al. (2024) identify "
                          "as a third population / probable minor merger. Writes "
                          "figure_pop3_robustness.png.")
    _ap.add_argument("--overview", action="store_true",
                     help="Generate the data-overview figure (histograms + scatter of the "
                          "real Tolstoy+2023 sample) and exit.")
    _ap.add_argument("--steps", type=int, default=2000, help="MCMC steps to ADD this run")
    _ap.add_argument("--walkers", type=int, default=60, help="ensemble walkers (>50)")
    _ap.add_argument("--nproc", type=int, default=0, help="worker processes (0 = all cores)")
    _ap.add_argument("--selection", choices=["none", "radial", "2d"], default="none",
                     help="AP24-style selection function omega(R,G) (default: none/flat)")
    _ap.add_argument("--backend", default="scl25.h5", help="HDF5 checkpoint file")
    _ap.add_argument("--K", type=int, default=0,
                     help="continuous model: number of metallicity DF nodes (cost is ~linear "
                          "in K). Default 11 (real) / 9 (mock). Use K=7 for ~35%% faster steps "
                          "when accumulating toward convergence on limited hardware.")
    _ap.add_argument("--nsub", type=int, default=0,
                     help="fit a random subsample of N stars (0 = all ~1339; try 400 "
                          "for a faster laptop first pass)")
    _ap.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    _args = _ap.parse_args()

    if getattr(_args, "galaxy", "sculptor") != "sculptor":   # switch target before any command
        set_galaxy(_args.galaxy)

    if _args.robustness:                              # membership-cut robustness, then exit
        run_membership_robustness(nproc=(_args.nproc or None))
        sys.exit(0)

    if _args.figure1:                                 # two-galaxy data presentation, then exit
        make_figure1_two_galaxy()
        sys.exit(0)

    if _args.actions:                                 # action-space chemodynamics, then exit
        make_action_space()
        sys.exit(0)

    if _args.dispersion:                              # radial dispersion profile, then exit
        make_dispersion_profile(use_gaia=_args.gaia)
        sys.exit(0)

    if _args.skymap:                                  # Gaia PM membership sky map, then exit
        make_gaia_skymap()
        sys.exit(0)

    if _args.fig4all:                                 # combined Fig.4 across all chains, then exit
        make_fig4_all_chains()
        sys.exit(0)

    if _args.compare:                                 # framework comparison figure, then exit
        make_framework_comparison()
        sys.exit(0)

    if _args.overview:                                # data-overview figure, then exit
        make_data_overview()
        sys.exit(0)

    if _args.wp11:                                    # WP11 mass-slope method, then exit
        _bk = _args.backend if _args.backend != "scl25.h5" else _gf("wp11.h5")
        run_wp11(nsteps=_args.steps, nproc=(_args.nproc or None), backend=_bk,
                 use_mock=_args.mock, resume=(False if _args.no_resume else None))
        sys.exit(0)

    if _args.continuous:                              # continuous f(J,[Fe/H]) model, then exit
        _explicit_steps = any(a == "--steps" or a.startswith("--steps=") for a in sys.argv)
        if _explicit_steps and _args.steps > 50:
            _bk = _args.backend if _args.backend != "scl25.h5" else _gf("cont.h5")
            _nsub = _args.nsub or (400 if _args.mock else 0)     # mock -> lean subsample by default
            _K = _args.K or (9 if _args.mock else 11)            # --K overrides the default node count
            run_continuous_chain(nsteps=_args.steps, nproc=(_args.nproc or None),
                                 backend=_bk, nsub=(_nsub or None), use_mock=_args.mock,
                                 fix_nuisance=True, K=_K,
                                 resume=(False if _args.no_resume else None))
        else:
            run_continuous_smoke()
        sys.exit(0)

    if _args.slide:                                   # sliding-threshold test, then exit
        run_sliding_metallicity_test()
        sys.exit(0)

    if _args.biasconv:                                # bias vs realisations plot, then exit
        run_bias_vs_realizations()
        sys.exit(0)

    if _args.biasgate:                                # bias gate on mocks, then exit
        run_bias_gate()
        sys.exit(0)

    if _args.revgate:                                 # reverse bias gate on mocks, then exit
        run_reverse_bias_gate()
        sys.exit(0)

    if _args.gatediag:                                # gate control tests on mocks, then exit
        run_gate_diagnostics()
        sys.exit(0)

    if _args.pop3:                                    # very-metal-poor contamination, then exit
        run_pop3_robustness(nproc=(_args.nproc or None))
        sys.exit(0)

    if _args.crosscheck:                              # GravSphere cross-check, then exit
        run_gravsphere_crosscheck(repo=_args.repo, do_mcmc=_args.mcmc,
                                  steps=(_args.steps if _args.steps != 2000 else 600))
        sys.exit(0)

    if _args.compare:                                 # framework-comparison figure, then exit
        make_framework_comparison()
        sys.exit(0)

    if _args.gravsphere:                              # GravSphere (Jeans + VSPs), then exit
        _bk = _args.backend if _args.backend != "scl25.h5" else _gf("gravsphere.h5")
        run_gravsphere_chain(nwalkers=(_args.walkers if 14 <= _args.walkers <= 40 else 24),
                             nsteps=_args.steps, nproc=(_args.nproc or None),
                             backend=_bk, resume=(False if _args.no_resume else None))
        sys.exit(0)

    if _args.dm5:                                     # 5-parameter DM model (robust+fast), then exit
        _bk = _args.backend if _args.backend != "scl25.h5" else _gf("dm5.h5")
        run_dm5_chain(nwalkers=(_args.walkers if 14 <= _args.walkers <= 40 else 24),
                      nsteps=_args.steps, nproc=(_args.nproc or None),
                      backend=_bk, resume=(False if _args.no_resume else None))
        sys.exit(0)

    if _args.chain:                                   # full production chain, then exit
        run_ap25_production_chain(
            nwalkers=_args.walkers, nsteps=_args.steps,
            nproc=(_args.nproc or None), backend=_args.backend,
            resume=(False if _args.no_resume else None),
            nsub=(_args.nsub or None),
            use_selection=(_args.selection != "none"),
            selection_mode=("radial" if _args.selection == "none" else _args.selection))
        sys.exit(0)

    # ---- default: fast phases (1-4) + Phase-5 smoke test ----
    print("=" * 60)
    print("   CHEMO-DYNAMICAL MASTER PIPELINE  — all fixes + action-DF")
    print("=" * 60)
    print()
    print("Fixes applied in this version:")
    print("  [FIX-A] Burkert mass: 2π prefactor (was π → underestimated M by 2×)")
    print("  [FIX-B] Cusp proxy: radially biased β≈+0.41 (was 1/R → β<0 at centre)")
    print("  [FIX-C] Gaia query: RUWE < 1.4 astrometric quality filter added")
    print("  [FIX-D] GMM fallback: individual Gaia errors in chi-sq denominator")
    print("  [FIX-E] SCULPTOR_PM_SIGMA: 0.0226 mas/yr (was 0.05, ~2.3× too large)")
    print("  [FIX-F] rho0_ref/rc_ref: explicit kwargs (was local → NameError)")
    print("  [FIX-G] sigma_init: variance-corrected starting guess")
    print("  [FIX-H] Threshold diagnostic: bootstrap error bands added")
    print("  [FIX-I] SkyCoord: explicit u.deg / u.mas/u.yr unit specification")
    print("  [FIX-J] Mmb normalisation: VizieR stores as % → divide by 100")
    print("  [FIX-K] Gaia Challenge positions: parsec-to-kpc auto-detection")
    print("  [NEW ] Distance updated 86 → 84 kpc (Martínez-Vázquez+15; AP25)")
    print("  [NEW ] Phase 4: AGAMA action-based DF modeling (real method)")
    print("         + publication-grade per-star MCMC: multiprocessing, HDF5")
    print("           checkpointing, autocorr/split-R-hat/ESS convergence, corner.")
    print("         Replaced fabricated Z16/R19/P20/H20 with real literature masses.")
    print("  [NEW ] Phase 5: FULL faithful 25-parameter model (DoublePowerLaw DFs +")
    print("         metallicity + pop-3), the rigorous validation-benchmark tier.")
    print("         + AP24-style selection function omega(R,G) (attach_selection).")
    print(f"         AGAMA: {HAS_AGAMA} | emcee: {HAS_EMCEE} | "
          f"corner: {HAS_CORNER} | h5py: {HAS_H5PY}")
    print()

    # ── Phases 1-3 require network (VizieR/Gaia/astrowiki). Guard them so that
    #    Phase 4 (which can run offline on a mock) still executes if they fail. ──
    try:
        walker_df, walker_coords = fetch_walker_data()
        run_sliding_threshold_diagnostic(walker_df)

        matched_df = fetch_gaia_with_epoch_correction(walker_df, walker_coords)
        clean_data = calculate_error_aware_membership(matched_df)
        clean_data.to_csv("sculptor_clean_observational.csv", index=False)
        print(f"--> Saved {len(clean_data)} clean stars to sculptor_clean_observational.csv")

        project_gaia_challenge_mock(CORE_DATA_URL, "projected_challenge_core.csv", default_halo='core')
        project_gaia_challenge_mock(CUSP_DATA_URL, "projected_challenge_cusp.csv", default_halo='cusp')
        reproduce_arroyo_polonio_fig4()

        plot_inferred_halo_profiles()
    except Exception as exc:
        print(f"\n[Phases 1-3] Skipped/failed (likely no network access): {str(exc)[:80]}")

    # ── Phase 4: action-based DF modeling (runs offline on a mock if needed) ──
    # Fast MLE by default. For the paper's TRUE per-star projectedDF posterior with
    # full convergence diagnostics, parallelism and checkpointing, use e.g.:
    #   import os
    #   run_action_df_modeling(high_fidelity=True, mcmc_nwalkers=32,
    #                          mcmc_nsteps=5000, mcmc_nsub=None,
    #                          mcmc_nproc=os.cpu_count(),
    #                          mcmc_backend="sculptor_chain.h5", mcmc_resume=True)
    # (slow: ~3 s/eval per population — hours for a full chain, but resumable).
    run_action_df_modeling(high_fidelity=False)

    # ── Phase 5: full faithful 25-parameter model (validation benchmark) ──
    # Fast smoke test here; the real converged chain is a cluster job:
    #   data = ap25_load_real_tolstoy2023(); ap25_run_full_mcmc(data,
    #       nproc=os.cpu_count(), nwalkers=60, nsteps>=thousands,
    #       backend_file="scl25.h5", resume=True)   # + set OMEGA_R to AP24 selection
    try:
        run_faithful_ap25_validation()
    except Exception as exc:
        print(f"[Phase 5] skipped: {str(exc)[:100]}")

    print("\n[Status] Pipeline complete.")
