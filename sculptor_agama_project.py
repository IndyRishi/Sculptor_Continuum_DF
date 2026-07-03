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
  5  The full faithful 25-parameter AP25 model (--chain) with an AP24-style selection
     function -- a cluster-scale run.

Command line:
  python sculptor_agama_project.py            # fast phases 1-4 + Phase-5 smoke test
  python sculptor_agama_project.py --dm5      # robust DM inner-slope measurement (real data)
  python sculptor_agama_project.py --chain    # full 25-parameter chain (cluster-scale)
  python sculptor_agama_project.py --help     # all options
"""

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
                  nbins=6, feh_quality_keep=(0,), catalog="J/A+A/675/A49"):
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
        catalog=catalog, feh_quality_keep=list(feh_quality_keep))
    pops = agama_binned_pops(R, vlos, label, verr, nbins=nbins)
    print(f"  {len(R)} stars; sigma_los binned into {nbins} bins/population")
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
    np.save("dm5_chain.npy", flat)
    try:
        make_corner_plot(flat, labels, "figure_dm5_corner.png")
    except Exception as exc:
        print(f"  corner skipped ({exc})")
    try:                                                        # gNFW -> (logMDM,log_rs,alpha=1,eta=3,gamma)
        n = len(flat)
        chain5 = np.column_stack([flat[:, 2], flat[:, 1], np.ones(n), 3.0 * np.ones(n), flat[:, 0]])
        make_ap25_figure4(chain5, "figure_ap25_fig4.png")
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


def agama_load_real_tolstoy2023(catalog="J/A+A/675/A49", cols=None, vclip_sigma=4.0, **select):
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
                       require_member=True, mem_keep=('m',), feh_quality_keep=None):
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

    for t in cats:                                       # pick the table with v AND [Fe/H]
        vc = cols.get('vlos') or find(t, _VCANDS)
        fc = cols.get('feh')  or find(t, _FCANDS)
        if vc and fc:
            rc  = cols.get('ra')  or find(t, _RACANDS)
            dc  = cols.get('dec') or find(t, _DECANDS)
            evc = cols.get('verr')   or find(t, _EVCANDS)
            efc = cols.get('feherr') or find(t, _EFCANDS)
            if rc is None or dc is None:
                raise KeyError(f"found v/[Fe/H] but not RA/Dec in table "
                               f"'{t.meta.get('name')}'; columns: {list(t.colnames)}")
            g = lambda name: np.array(t[name], float)
            verr = g(evc) if evc else np.full(len(t), 2.0)    # default 2 km/s if absent
            feherr = g(efc) if efc else np.full(len(t), 0.1)  # default 0.1 dex if absent
            gcol = cols.get('gmag') or find(t, _GCANDS)
            gmag = g(gcol) if gcol else np.full(len(t), np.nan)   # real Gaia G if present
            # ── sample selection: members (+ optional [Fe/H] quality) ──
            sel = np.ones(len(t), bool)
            mcol = cols.get('mem') or find(t, ['Mem', 'Member', 'memb', 'Pmemb', 'Pmem'])
            if require_member and mcol is not None:
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
            return (g(rc)[sel], g(dc)[sel], g(vc)[sel], verr[sel],
                    g(fc)[sel], feherr[sel], gmag[sel])

    lines = [f"Could not auto-detect velocity+[Fe/H] columns in '{catalog}'.",
             "Run ap25_inspect_vizier(catalog), then pass cols=dict(...). Tables found:"]
    for i, t in enumerate(cats):
        lines.append(f"  [{i}] {t.meta.get('name', '?')}: {list(t.colnames)}")
    raise KeyError("\n".join(lines))


def _semi_major_axis_radius(ra, dec, D_KPC=84.0):
    """Paper Eq.1 semi-major-axis radius (kpc) using Munoz+2018 centre/ellipticity/PA."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cen = SkyCoord(15.0183 * u.deg, -33.7186 * u.deg)
    e, pa = 0.32, np.radians(92.0)
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


def ap25_load_real_tolstoy2023(catalog="J/A+A/675/A49", cols=None, **select):
    """
    Load the real Tolstoy et al. (2023) Sculptor catalog (v_los + [Fe/H]) for the
    full 25-parameter fit. Auto-detects columns; if it can't, it prints the tables'
    columns so you can pass cols=dict(vlos=..., feh=..., ra=..., dec=..., verr=...,
    feherr=...). Extra kwargs (require_member, mem_keep, feh_quality_keep) are
    forwarded for sample selection (use feh_quality_keep to reach the paper's ~1339).
    Returns the `data` dict expected by ap25_lnlike_full, with v_los in the REST frame
    (systemic subtracted) and R the semi-major-axis radius (Eq.1).
    """
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
                                      feh_quality_keep=list(feh_quality_keep))
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
    np.save("ap25_chain.npy", flat)
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
    _ap.add_argument("--steps", type=int, default=2000, help="MCMC steps to ADD this run")
    _ap.add_argument("--walkers", type=int, default=60, help="ensemble walkers (>50)")
    _ap.add_argument("--nproc", type=int, default=0, help="worker processes (0 = all cores)")
    _ap.add_argument("--selection", choices=["none", "radial", "2d"], default="none",
                     help="AP24-style selection function omega(R,G) (default: none/flat)")
    _ap.add_argument("--backend", default="scl25.h5", help="HDF5 checkpoint file")
    _ap.add_argument("--nsub", type=int, default=0,
                     help="fit a random subsample of N stars (0 = all ~1339; try 400 "
                          "for a faster laptop first pass)")
    _ap.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    _args = _ap.parse_args()

    if _args.dm5:                                     # 5-parameter DM model (robust+fast), then exit
        _bk = _args.backend if _args.backend != "scl25.h5" else "dm5.h5"
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
