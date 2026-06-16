import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import multiprocessing

from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution, minimize

# ── scipy.integrate.trapezoid cross-version compatibility ───────────────────
try:
    from scipy.integrate import trapezoid as _trapz
except ImportError:
    from scipy.integrate import trapz as _trapz

# ── emcee (required for MCMC inference) ─────────────────────────────────────
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    warnings.warn(
        "emcee not found. Install with:  pip install emcee\n"
        "fit_posterior_grid() will raise ImportError at runtime.",
        stacklevel=2,
    )

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
RA0_DEG = 15.039
DEC0_DEG = -33.709
DISTANCE_KPC = 86.0
SEARCH_RADIUS_DEG = 0.5
MEMBERSHIP_MIN = 0.90
MIN_BIN_N = 10
N_RADIAL_BINS = 6
G_KPC_KMS2_MSUN = 4.30091e-6
H0_KM_S_MPC = 70.0
RHO_CRIT_MSUN_KPC3 = 3.0 * (H0_KM_S_MPC / 1000.0) ** 2 / (8.0 * np.pi * G_KPC_KMS2_MSUN)

DERIVED_HALO_RADII = [
    (0.15, "150"),
    (0.30, "300"),
    (0.50, "500"),
]
DERIVED_HALO_PARAMS = ["log10_M150", "log10_M300", "log10_M500", "log10_rho150"]
NATIVE_HALO_PARAMS  = ["log10_M200", "log10_rs"]
ORBIT_TENSION_PARAMS = ["beta"]
STRUCTURAL_PARAMS = ["log10_a", "V_sys"]

PARAM_NAMES = ["log10_M200", "log10_rs", "beta", "log10_a", "V_sys"]
PARAM_BOUNDS = np.array([
    (8.0, 11.0),    # log10_M200
    (-1.2, 1.0),    # log10_rs
    (-1.5, 0.7),    # beta
    (-1.0, 0.5),    # log10_a (Plummer scale radius in kpc)
    (100.0, 120.0), # V_sys (Systemic velocity of Sculptor in km/s)
], dtype=float)


# ============================================================
# BASIC STATS
# ============================================================
def weighted_mean(values, weights):
    values  = np.asarray(values,  dtype=float)
    weights = np.asarray(weights, dtype=float)
    weight_sum = np.sum(weights)
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        return np.nan
    return np.sum(weights * values) / weight_sum


def effective_n(weights):
    weights = np.asarray(weights, dtype=float)
    weight_sum = np.sum(weights)
    weight_sq_sum = np.sum(weights ** 2)
    if weight_sum <= 0 or weight_sq_sum <= 0:
        return np.nan
    return weight_sum ** 2 / weight_sq_sum


def intrinsic_dispersion_mle(values, errors, weights=None):
    values  = np.asarray(values,  dtype=float)
    errors  = np.asarray(errors,  dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)
    good = (
        np.isfinite(values) & np.isfinite(errors) &
        np.isfinite(weights) & (errors > 0) & (weights > 0)
    )
    values  = values[good]
    errors  = errors[good]
    weights = weights[good]
    if len(values) < 3:
        return np.nan, np.nan, np.nan

    mu0  = weighted_mean(values, weights)
    var0 = max(
        np.average((values - mu0) ** 2, weights=weights) -
        np.average(errors ** 2,         weights=weights),
        0.01,
    )
    sig0 = np.sqrt(var0)

    def nll(params):
        mu_v  = params[0]
        sig_v = np.exp(params[1])
        var_v = sig_v ** 2 + errors ** 2
        return 0.5 * np.sum(
            weights * (np.log(2.0 * np.pi * var_v) + (values - mu_v) ** 2 / var_v)
        )

    opt    = minimize(nll, x0=np.array([mu0, np.log(sig0)]), method="Nelder-Mead")
    mu_hat  = opt.x[0]
    sig_hat = np.exp(opt.x[1])
    n_eff   = effective_n(weights)
    sig_err = sig_hat / np.sqrt(max(2.0 * (n_eff - 1.0), 1.0))
    return mu_hat, sig_hat, sig_err


def make_equal_count_bins(dataframe, col_name, n_bins, min_n):
    labels = pd.Series(index=dataframe.index, dtype="Int64")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if min_n <= 0:
        raise ValueError("min_n must be positive")

    work = dataframe[np.isfinite(dataframe[col_name])].sort_values(col_name).copy()
    if len(work) < min_n:
        return labels

    actual_bins = min(int(n_bins), max(1, len(work) // int(min_n)))
    bins = np.array_split(work.index.to_numpy(), actual_bins)
    for label_val, idx_vals in enumerate(bins):
        if len(idx_vals) >= min_n:
            labels.loc[idx_vals] = label_val
    return labels


def radial_profile(dataframe, n_bins=N_RADIAL_BINS, min_n=MIN_BIN_N):
    """Generates a binned radial profile (only used for plotting now)."""
    work = dataframe.copy()
    work["bin"] = make_equal_count_bins(work, "R_kpc", n_bins, min_n)
    rows = []
    for bin_val in sorted(work["bin"].dropna().unique()):
        b = work[work["bin"] == bin_val]
        if len(b) < min_n:
            continue
        mu_v, sig_v, sig_err = intrinsic_dispersion_mle(
            b["V_los"].values, b["e_V_los"].values, b["P_mem"].values
        )
        if not np.isfinite(sig_v) or not np.isfinite(sig_err) or sig_err <= 0:
            continue
        rows.append({
            "R_kpc":          np.average(b["R_kpc"].values, weights=b["P_mem"].values),
            "sigma_kms":      sig_v,
            "sigma_err_kms":  sig_err,
            "v_mean_kms":     mu_v,
            "N":              len(b),
            "N_eff":          effective_n(b["P_mem"].values),
        })
    if not rows:
        return pd.DataFrame(
            columns=["R_kpc", "sigma_kms", "sigma_err_kms", "v_mean_kms", "N", "N_eff"]
        )
    return pd.DataFrame(rows).sort_values("R_kpc").reset_index(drop=True)


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_sculptor_data(cache_csv="sculptor_clean_member_catalog.csv", force_fetch=False):
    """
    Fetches the Walker et al. (2009) catalog.
    No longer cross-matches with Gaia to prevent massive data leakage from proper motion shifts.
    """
    if os.path.exists(cache_csv) and not force_fetch:
        return pd.read_csv(cache_csv)

    print("Fetching Walker et al. catalog...")
    Vizier.ROW_LIMIT = -1
    walker = Vizier.get_catalogs("J/AJ/137/3100")[0]
    
    walker_coords = SkyCoord(
        np.array(walker["RAJ2000"]).astype(str),
        np.array(walker["DEJ2000"]).astype(str),
        unit=(u.hourangle, u.deg),
        frame="icrs",
    )
    center = SkyCoord(RA0_DEG * u.deg, DEC0_DEG * u.deg, frame="icrs")
    
    # Use Walker coordinates directly to calculate R_kpc
    R_kpc = walker_coords.separation(center).radian * DISTANCE_KPC
    
    df = pd.DataFrame({
        "Target":    np.array(walker["Target"]).astype(str),
        "V_los":     np.array(walker["<HV>"], dtype=float),
        "e_V_los":   np.array(walker["e_<HV>"], dtype=float),
        "P_mem":     np.array(walker["Mmb"], dtype=float),
        "SigMg":     np.array(walker["<SigMg>"], dtype=float),
        "R_kpc":     R_kpc
    })
    
    df = df.dropna(subset=["V_los", "e_V_los", "P_mem", "SigMg"])
    df = df[(df["e_V_los"] > 0) & (df["P_mem"] >= MEMBERSHIP_MIN)].copy()
    
    df.to_csv(cache_csv, index=False)
    print(f"Saved {cache_csv}. Recovered {len(df)} stars.")
    return df


# ============================================================
# JEANS MODEL
# ============================================================
def nfw_mass_enclosed(r_kpc, log10_m200, log10_rs):
    m200 = 10.0 ** log10_m200
    rs   = 10.0 ** log10_rs
    r200 = (3.0 * m200 / (4.0 * np.pi * 200.0 * RHO_CRIT_MSUN_KPC3)) ** (1.0 / 3.0)
    c200 = r200 / rs
    x    = np.asarray(r_kpc, dtype=float) / rs
    fx   = np.log1p(x) - x / (1.0 + x)
    fc   = np.log1p(c200) - c200 / (1.0 + c200)
    return m200 * fx / fc


def nfw_density(r_kpc, log10_m200, log10_rs):
    m200 = 10.0 ** log10_m200
    rs   = 10.0 ** log10_rs
    r200 = (3.0 * m200 / (4.0 * np.pi * 200.0 * RHO_CRIT_MSUN_KPC3)) ** (1.0 / 3.0)
    c200 = r200 / rs
    fc   = np.log1p(c200) - c200 / (1.0 + c200)
    rho_s = m200 / (4.0 * np.pi * rs ** 3 * fc)
    x     = np.asarray(r_kpc, dtype=float) / rs
    return rho_s / (x * (1.0 + x) ** 2)


def add_derived_halo_quantities(samples_df):
    samples_df = samples_df.copy()
    for radius_kpc, radius_label in DERIVED_HALO_RADII:
        mass_vals = nfw_mass_enclosed(
            radius_kpc,
            samples_df["log10_M200"].values,
            samples_df["log10_rs"].values,
        )
        samples_df["log10_M" + radius_label] = np.log10(np.clip(mass_vals, 1e-30, None))
    rho150 = nfw_density(
        0.15,
        samples_df["log10_M200"].values,
        samples_df["log10_rs"].values,
    )
    samples_df["log10_rho150"] = np.log10(np.clip(rho150, 1e-30, None))
    return samples_df


def plummer_nu(r_kpc, a_kpc):
    r_kpc = np.asarray(r_kpc, dtype=float)
    return (1.0 + (r_kpc / a_kpc) ** 2) ** (-2.5)


def projected_jeans_sigma_los(R_eval_kpc, log10_m200, log10_rs, beta, a_kpc):
    """
    Projected line-of-sight velocity dispersion from the spherical Jeans
    equation with a Plummer tracer profile and NFW dark-matter halo.
    """
    R_eval_kpc = np.asarray(R_eval_kpc, dtype=float)
    r_min = max(1e-4, np.nanmin(R_eval_kpc) * 0.05)
    r_max = max(20.0, np.nanmax(R_eval_kpc) * 80.0)
    r_grid = np.geomspace(r_min, r_max, 520)

    nu_grid   = plummer_nu(r_grid, a_kpc)
    mass_grid = nfw_mass_enclosed(r_grid, log10_m200, log10_rs)

    # 3-D radial velocity dispersion from the Jeans equation
    integrand = nu_grid * G_KPC_KMS2_MSUN * mass_grid * r_grid ** (2.0 * beta - 2.0)
    rev_int  = cumulative_trapezoid(integrand[::-1], r_grid[::-1], initial=0.0)
    radial_int = -rev_int[::-1]

    with np.errstate(all="ignore"):
        sigma_r2 = radial_int / (nu_grid * r_grid ** (2.0 * beta))
    sigma_r2 = np.where(np.isfinite(sigma_r2), sigma_r2, 0.0)
    sigma_r2 = np.clip(sigma_r2, 0.0, None)

    out = []
    for R_val in R_eval_kpc:
        lower   = max(R_val * (1.0 + 1e-5), r_min)
        r_local = np.geomspace(lower, r_max, 420)
        nu_local  = plummer_nu(r_local, a_kpc)
        sr2_local = np.interp(r_local, r_grid, sigma_r2)

        geom  = r_local / np.sqrt(r_local ** 2 - R_val ** 2)
        denom = 2.0 * _trapz(nu_local * geom, r_local)
        if denom <= 0:
            out.append(np.nan)
            continue

        kernel = 1.0 - beta * R_val ** 2 / r_local ** 2
        numer  = 2.0 * _trapz(kernel * nu_local * sr2_local * geom, r_local)
        out.append(np.sqrt(max(numer / denom, 0.0)))

    return np.array(out)


# ============================================================
# LOG-POSTERIOR FOR MCMC (UNBINNED)
# ============================================================
def log_prior(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(PARAM_NAMES),) or not np.all(np.isfinite(theta)):
        return -np.inf
    if np.all((theta >= PARAM_BOUNDS[:, 0]) & (theta <= PARAM_BOUNDS[:, 1])):
        return 0.0
    return -np.inf


def log_likelihood_unbinned(theta, df):
    """
    Unbinned discrete log-likelihood for individual stellar velocities.
    Evaluates the Jeans integral on a sparse grid and interpolates to individual stars.
    """
    log10_m200, log10_rs, beta, log10_a, v_sys = theta
    a_kpc = 10.0 ** log10_a
    
    R_i = df["R_kpc"].values
    V_i = df["V_los"].values
    e_V_i = df["e_V_los"].values
    
    # 1. Fast Grid Evaluation
    R_grid = np.geomspace(max(1e-4, R_i.min() * 0.9), R_i.max() * 1.1, 40)
    try:
        sigma_grid = projected_jeans_sigma_los(R_grid, log10_m200, log10_rs, beta, a_kpc)
    except Exception:
        return -np.inf
        
    if not np.all(np.isfinite(sigma_grid)) or np.any(sigma_grid <= 0.0):
        return -np.inf

    # 2. Vectorized Interpolation to exact stellar radii
    sigma_model_i = np.interp(R_i, R_grid, sigma_grid)
    
    # 3. Unbinned Gaussian Likelihood for each star
    variance_tot = sigma_model_i**2 + e_V_i**2
    
    ll = -0.5 * np.sum(
        ((V_i - v_sys)**2 / variance_tot) + np.log(2.0 * np.pi * variance_tot)
    )
    return float(ll)


def log_posterior(theta, df):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_unbinned(theta, df)


def neg_log_like(theta, df):
    """Objective function for differential evolution MAP finding."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return 1e50
    ll = log_likelihood_unbinned(theta, df)
    if not np.isfinite(ll):
        return 1e50
    return -ll


# ============================================================
# MCMC POSTERIOR INFERENCE
# ============================================================
def _draw_initial_walkers(best, bounds, scale, n_walkers, rng):
    ndim = len(best)
    p0 = np.empty((n_walkers, ndim))
    for i in range(ndim):
        low, high = bounds[i]
        std = min(scale[i], (high - low) * 0.1)
        vals = rng.normal(best[i], std, size=n_walkers)
        vals = np.clip(vals, low + 1e-4, high - 1e-4)
        p0[:, i] = vals
    return p0


def fit_posterior_grid(
    df,
    n_draws=2000,
    rng=None,
    de_maxiter=160,
    de_popsize=10,
    polish=True,
    n_walkers=32,
    n_burnin=300,
    n_prod_steps=600,
    progress=False,
):
    if not HAS_EMCEE:
        raise ImportError("Install emcee:  pip install emcee")
    if rng is None:
        rng = np.random.default_rng(7)
    if len(df) < 10:
        return None

    ndim = len(PARAM_NAMES)
    
    # ── 1. MAP via differential evolution ────────────────────────────────────
    bounds = [tuple(row) for row in PARAM_BOUNDS]
    opt = differential_evolution(
        lambda x: neg_log_like(x, df),
        bounds=bounds,
        seed=int(rng.integers(0, 2**31 - 1)),
        tol=0.04,
        polish=polish,
        workers=1,
        maxiter=de_maxiter,
        popsize=de_popsize,
    )
    best      = opt.x
    best_nll  = opt.fun
    if not np.all(np.isfinite(best)) or not np.isfinite(best_nll):
        return None

    # ── 2. Initialise walkers in a tight ball around MAP ──────────────────────
    init_scale = np.array([0.12, 0.08, 0.08, 0.05, 1.0])
    p0 = _draw_initial_walkers(best, bounds, init_scale, n_walkers, rng)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior, args=[df]
    )

    # ── 3. Burn-in ────────────────────────────────────────────────────────────
    state = sampler.run_mcmc(p0, n_burnin, progress=progress)
    sampler.reset()

    # ── 4. Production ─────────────────────────────────────────────────────────
    actual_steps = max(n_prod_steps, int(np.ceil(n_draws / n_walkers)))
    sampler.run_mcmc(state, actual_steps, progress=progress)

    # ── 5. Thin by integrated autocorrelation time ────────────────────────────
    tau  = np.full(ndim, np.nan)
    thin = 1
    try:
        tau  = sampler.get_autocorr_time(quiet=True)
        thin = max(1, int(np.ceil(0.5 * float(np.nanmax(tau)))))
    except Exception:
        pass

    flat_samples     = sampler.get_chain(flat=True, thin=thin)
    acceptance_frac  = float(np.mean(sampler.acceptance_fraction))

    if len(flat_samples) == 0:
        return None
    if len(flat_samples) > n_draws:
        keep_idx = np.sort(rng.choice(len(flat_samples), size=n_draws, replace=False))
        flat_samples = flat_samples[keep_idx]

    # ── 6. Build output DataFrame ─────────────────────────────────────────────
    samples_df = pd.DataFrame(flat_samples, columns=PARAM_NAMES)
    samples_df["rs_kpc"]             = 10.0 ** samples_df["log10_rs"]
    samples_df["nll_best"]           = best_nll
    samples_df["best_log10_M200"]    = best[0]
    samples_df["best_log10_rs"]      = best[1]
    samples_df["best_beta"]          = best[2]
    samples_df["best_log10_a"]       = best[3]
    samples_df["best_V_sys"]         = best[4]
    samples_df["acceptance_fraction"]= acceptance_frac
    samples_df["tau_log10_M200"]     = tau[0]
    samples_df["tau_log10_rs"]       = tau[1]
    samples_df["tau_beta"]           = tau[2]
    samples_df["tau_log10_a"]        = tau[3]
    samples_df["tau_V_sys"]          = tau[4]
    samples_df["n_prod_steps"]       = actual_steps
    samples_df["thin"]               = thin
    samples_df["n_samples_returned"] = len(samples_df)
    samples_df = add_derived_halo_quantities(samples_df)
    return samples_df


# ============================================================
# TENSION METRICS
# ============================================================
def _finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _shared_histogram_range(a, b, lo_pct=0.5, hi_pct=99.5):
    combined = _finite_values(np.concatenate([a, b]))
    if len(combined) == 0:
        return None
    lo = np.nanpercentile(combined, lo_pct)
    hi = np.nanpercentile(combined, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        pad = max(1e-6, abs(float(lo)) * 1e-6)
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def scalar_tension_from_arrays(a, b):
    a = _finite_values(a)
    b = _finite_values(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    med_a = np.nanmedian(a)
    med_b = np.nanmedian(b)
    sig_a = 0.5 * (np.nanpercentile(a, 84) - np.nanpercentile(a, 16))
    sig_b = 0.5 * (np.nanpercentile(b, 84) - np.nanpercentile(b, 16))
    denom = np.sqrt(sig_a ** 2 + sig_b ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return np.abs(med_a - med_b) / denom


def scalar_tension(samples_a, samples_b, param):
    return scalar_tension_from_arrays(
        samples_a[param].values, samples_b[param].values
    )


def bootstrap_scalar_tension(samples_a, samples_b, param, n_resamples=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    a = _finite_values(samples_a[param].values)
    b = _finite_values(samples_b[param].values)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    vals = [
        scalar_tension_from_arrays(
            rng.choice(a, size=len(a), replace=True),
            rng.choice(b, size=len(b), replace=True),
        )
        for _ in range(n_resamples)
    ]
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return np.nanpercentile(vals, 16), np.nanpercentile(vals, 84)


def posterior_overlap_1d(samples_a, samples_b, param, bins=40):
    a = _finite_values(samples_a[param].values)
    b = _finite_values(samples_b[param].values)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    hist_range = _shared_histogram_range(a, b)
    if hist_range is None:
        return np.nan
    h_a, edges = np.histogram(a, bins=bins, range=hist_range, density=True)
    h_b, _     = np.histogram(b, bins=edges,                    density=True)
    if not np.all(np.isfinite(h_a)) or not np.all(np.isfinite(h_b)):
        return np.nan
    return float(np.clip(np.sum(np.minimum(h_a, h_b) * np.diff(edges)), 0.0, 1.0))


def jsd_1d(samples_a, samples_b, param, bins=40):
    a = _finite_values(samples_a[param].values)
    b = _finite_values(samples_b[param].values)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    hist_range = _shared_histogram_range(a, b)
    if hist_range is None:
        return np.nan
    h_a, edges = np.histogram(a, bins=bins, range=hist_range)
    h_b, _     = np.histogram(b, bins=edges)
    p = h_a.astype(float) + 1e-12
    q = h_b.astype(float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m  = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def parameter_group(param):
    if param in DERIVED_HALO_PARAMS:   return "derived_halo"
    if param in NATIVE_HALO_PARAMS:    return "native_halo"
    if param in ORBIT_TENSION_PARAMS:  return "orbital_nuisance"
    if param in STRUCTURAL_PARAMS:     return "structural"
    return "other"


def summarize_tension(samples_low, samples_high, rng=None, n_mc_resamples=200):
    if rng is None:
        rng = np.random.default_rng(321)
    rows   = []
    params = DERIVED_HALO_PARAMS + NATIVE_HALO_PARAMS + ORBIT_TENSION_PARAMS + STRUCTURAL_PARAMS
    for param in params:
        t_val       = scalar_tension(samples_low, samples_high, param)
        t_p16, t_p84 = bootstrap_scalar_tension(
            samples_low, samples_high, param,
            n_resamples=n_mc_resamples, rng=rng,
        )
        rows.append({
            "param_group":         parameter_group(param),
            "param":               param,
            "sigma_tension":       t_val,
            "sigma_tension_p16":   t_p16,
            "sigma_tension_p84":   t_p84,
            "overlap":             posterior_overlap_1d(samples_low, samples_high, param),
            "jsd_nats":            jsd_1d(samples_low, samples_high, param),
            "low_median":          np.nanmedian(samples_low[param].values),
            "high_median":         np.nanmedian(samples_high[param].values),
            "low_p16":             np.nanpercentile(samples_low[param].values,  16),
            "low_p84":             np.nanpercentile(samples_low[param].values,  84),
            "high_p16":            np.nanpercentile(samples_high[param].values, 16),
            "high_p84":            np.nanpercentile(samples_high[param].values, 84),
        })
    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================
def save_profile_plot(profile_low, profile_high, samples_low, samples_high,
                      threshold_q, output_path):
    plt.figure(figsize=(8, 5))
    colors = {"Low-Mg-index": "tab:blue", "High-Mg-index": "tab:red"}
    for label_name, prof, samp in [
        ("Low-Mg-index",  profile_low,  samples_low),
        ("High-Mg-index", profile_high, samples_high),
    ]:
        if prof is None or len(prof) == 0 or samp is None:
            continue
        plt.errorbar(
            prof["R_kpc"], prof["sigma_kms"], yerr=prof["sigma_err_kms"],
            fmt="o", color=colors[label_name], label=label_name + " data",
        )
        theta = [
            np.nanmedian(samp["log10_M200"]),
            np.nanmedian(samp["log10_rs"]),
            np.nanmedian(samp["beta"]),
            np.nanmedian(samp["log10_a"])
        ]
        r_plot = np.linspace(
            max(0.01, prof["R_kpc"].min() * 0.8),
            prof["R_kpc"].max() * 1.15, 100,
        )
        plt.plot(
            r_plot, projected_jeans_sigma_los(r_plot, theta[0], theta[1], theta[2], 10.0**theta[3]),
            color=colors[label_name], lw=2, label=label_name + " Jeans",
        )
    plt.xlabel("Projected radius [kpc]")
    plt.ylabel("Intrinsic sigma_los [km/s]")
    plt.title("Threshold q=" + str(round(threshold_q, 2)) +
              ": chemically selected tracer subsets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_tension_group(results_df, params, output_path, title,
                       xlabel="Mg-index cut value"):
    plt.figure(figsize=(9, 5))
    for param in params:
        sub = results_df[results_df["param"] == param].sort_values("SigMg_limit")
        if len(sub) == 0:
            continue
        x = sub["SigMg_limit"].values
        plt.plot(x, sub["sigma_tension"].values, marker="o", label=param)
        plt.fill_between(x, sub["sigma_tension_p16"].values,
                         sub["sigma_tension_p84"].values, alpha=0.15)
    plt.axhline(1.0, color="gray", ls="--", lw=1)
    plt.axhline(2.0, color="gray", ls=":",  lw=1)
    plt.xlabel(xlabel)
    plt.ylabel("Approx. posterior tension [sigma]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_threshold_summary_plot(results_df, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plot_specs = [
        (DERIVED_HALO_PARAMS,  "Derived inner-halo tension"),
        (NATIVE_HALO_PARAMS,   "Native NFW parameter tension"),
        (ORBIT_TENSION_PARAMS, "Orbital nuisance tension"),
    ]
    for ax, (params, title) in zip(axes, plot_specs):
        for param in params:
            sub = results_df[results_df["param"] == param].sort_values("SigMg_limit")
            if len(sub) == 0:
                continue
            x = sub["SigMg_limit"].values
            ax.plot(x, sub["sigma_tension"].values, marker="o", label=param)
            ax.fill_between(x, sub["sigma_tension_p16"].values,
                            sub["sigma_tension_p84"].values, alpha=0.15)
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.axhline(2.0, color="gray", ls=":",  lw=1)
        ax.set_ylabel("tension [sigma]")
        ax.set_title(title)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Actual Mg-index cut value")
    fig.suptitle("Sliding-threshold robustness test", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_baseline_diagnostic_plot(df, full_profile, output_path):
    mg_cut = df["SigMg"].median()
    work   = df.copy()
    work["pop"] = np.where(work["SigMg"] > mg_cut, "High-Mg-index", "Low-Mg-index")

    plt.figure(figsize=(13, 10))
    plt.subplot(2, 2, 1)
    plt.hist(work["V_los"], bins=35, weights=work["P_mem"],
             color="steelblue", alpha=0.85)
    plt.xlabel("V_los [km/s]")
    plt.ylabel("membership-weighted count")
    plt.title("Clean member velocity distribution")

    plt.subplot(2, 2, 2)
    plt.errorbar(full_profile["R_kpc"], full_profile["sigma_kms"],
                 yerr=full_profile["sigma_err_kms"], fmt="o-", color="black")
    plt.xlabel("Projected radius [kpc]")
    plt.ylabel("Intrinsic dispersion [km/s]")
    plt.title("Error-corrected dispersion profile")

    plt.subplot(2, 2, 3)
    for pop_name in ["High-Mg-index", "Low-Mg-index"]:
        sub = work[work["pop"] == pop_name]
        pop_profile = radial_profile(sub, n_bins=4, min_n=max(8, MIN_BIN_N // 2))
        if len(pop_profile) > 0:
            plt.errorbar(
                pop_profile["R_kpc"], pop_profile["sigma_kms"],
                yerr=pop_profile["sigma_err_kms"], fmt="o-", label=pop_name,
            )
    plt.xlabel("Projected radius [kpc]")
    plt.ylabel("Intrinsic dispersion [km/s]")
    plt.title("Dispersion by median Mg-index split")
    plt.legend()

    plt.subplot(2, 2, 4)
    for pop_name in ["High-Mg-index", "Low-Mg-index"]:
        sub = work[work["pop"] == pop_name]
        plt.hist(sub["V_los"], bins=25, alpha=0.55, density=True, label=pop_name)
    plt.xlabel("V_los [km/s]")
    plt.ylabel("density")
    plt.title("Velocity distributions by median Mg-index split")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_full_sample_jeans_plot(full_profile, full_samples, output_path):
    if full_samples is None or len(full_profile) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        full_profile["R_kpc"], full_profile["sigma_kms"],
        yerr=full_profile["sigma_err_kms"],
        fmt="o", color="black", label="Observed intrinsic dispersion",
    )
    theta = [
        np.nanmedian(full_samples["log10_M200"]),
        np.nanmedian(full_samples["log10_rs"]),
        np.nanmedian(full_samples["beta"]),
        np.nanmedian(full_samples["log10_a"])
    ]
    r_plot = np.linspace(
        max(0.01, full_profile["R_kpc"].min() * 0.8),
        full_profile["R_kpc"].max() * 1.15, 120,
    )
    plt.plot(r_plot, projected_jeans_sigma_los(r_plot, theta[0], theta[1], theta[2], 10.0**theta[3]),
             color="crimson", lw=2, label="Projected Jeans model")
    
    acc  = full_samples["acceptance_fraction"].iloc[0]
    tau_max = np.nanmax([
        full_samples["tau_log10_M200"].iloc[0],
        full_samples["tau_log10_rs"].iloc[0],
        full_samples["tau_beta"].iloc[0],
        full_samples["tau_log10_a"].iloc[0]
    ])
    plt.xlabel("Projected radius [kpc]")
    plt.ylabel("sigma_los [km/s]")
    plt.title(
        f"Jeans fit: NFW + Plummer tracer  |  "
        f"accept={acc:.2f}  tau_max={tau_max:.1f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def threshold_file_tag(value):
    return str(np.round(float(value), 4)).replace(".", "p").replace("-", "m")


def build_unique_threshold_table(df, threshold_grid):
    rows = [
        {"threshold_q": q, "SigMg_limit": float(df["SigMg"].quantile(q))}
        for q in threshold_grid
    ]
    raw     = pd.DataFrame(rows)
    grouped = []
    for limit in sorted(raw["SigMg_limit"].unique()):
        sub    = raw[raw["SigMg_limit"] == limit]
        q_vals = sub["threshold_q"].values
        grouped.append({
            "threshold_q":        float(np.mean(q_vals)),
            "threshold_q_min":    float(np.min(q_vals)),
            "threshold_q_max":    float(np.max(q_vals)),
            "threshold_q_values": ",".join([str(np.round(x, 3)) for x in q_vals]),
            "SigMg_limit":        float(limit),
        })
    return pd.DataFrame(grouped)


def evaluate_fixed_cut_tensions(df, threshold_table, n_draws, rng,
                                 fit_kwargs=None, n_mc_resamples=80):
    if fit_kwargs is None:
        fit_kwargs = {}
    all_rows = []
    for row_idx in range(len(threshold_table)):
        threshold_q        = threshold_table.loc[row_idx, "threshold_q"]
        threshold_q_min    = threshold_table.loc[row_idx, "threshold_q_min"]
        threshold_q_max    = threshold_table.loc[row_idx, "threshold_q_max"]
        threshold_q_values = threshold_table.loc[row_idx, "threshold_q_values"]
        limit              = threshold_table.loc[row_idx, "SigMg_limit"]
        low_df             = df[df["SigMg"] <= limit].copy()
        high_df            = df[df["SigMg"] >  limit].copy()
        
        low_samples        = fit_posterior_grid(low_df,  n_draws=n_draws, rng=rng, **fit_kwargs)
        high_samples       = fit_posterior_grid(high_df, n_draws=n_draws, rng=rng, **fit_kwargs)
        if low_samples is None or high_samples is None:
            continue
            
        tension = summarize_tension(low_samples, high_samples,
                                    rng=rng, n_mc_resamples=n_mc_resamples)
        tension["threshold_q"]        = threshold_q
        tension["threshold_q_min"]    = threshold_q_min
        tension["threshold_q_max"]    = threshold_q_max
        tension["threshold_q_values"] = threshold_q_values
        tension["SigMg_limit"]        = limit
        tension["N_low"]              = len(low_df)
        tension["N_high"]             = len(high_df)
        all_rows.append(tension)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def max_info_from_subset(subset, prefix):
    if subset is None or len(subset) == 0:
        return {prefix + k: v for k, v in [
            ("_max_tension", np.nan), ("_max_param", ""), ("_max_SigMg_limit", np.nan)
        ]}
    finite_subset = subset[np.isfinite(subset["sigma_tension"].values)]
    if len(finite_subset) == 0:
        return {prefix + k: v for k, v in [
            ("_max_tension", np.nan), ("_max_param", ""), ("_max_SigMg_limit", np.nan)
        ]}
    idx_val = finite_subset["sigma_tension"].idxmax()
    return {
        prefix + "_max_tension":    finite_subset.loc[idx_val, "sigma_tension"],
        prefix + "_max_param":      finite_subset.loc[idx_val, "param"],
        prefix + "_max_SigMg_limit": finite_subset.loc[idx_val, "SigMg_limit"],
    }


def summarize_max_tensions(results_df):
    out = {}
    if results_df is None or len(results_df) == 0:
        for prefix in ("derived_halo", "native_halo", "beta"):
            out.update(max_info_from_subset(pd.DataFrame(), prefix))
        return out
    out.update(max_info_from_subset(
        results_df[results_df["param_group"] == "derived_halo"], "derived_halo"))
    out.update(max_info_from_subset(
        results_df[results_df["param_group"] == "native_halo"],  "native_halo"))
    out.update(max_info_from_subset(
        results_df[results_df["param"] == "beta"],               "beta"))
    return out


def empirical_p_value(null_vals, observed_val):
    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if len(null_vals) == 0 or not np.isfinite(observed_val):
        return np.nan
    return (np.sum(null_vals >= observed_val) + 1.0) / (len(null_vals) + 1.0)


def save_data_bootstrap_plot(bootstrap_max_df, observed_max, output_path):
    if bootstrap_max_df is None or len(bootstrap_max_df) == 0:
        return
    metrics = [
        ("derived_halo_max_tension", "derived inner-halo"),
        ("beta_max_tension",         "beta"),
    ]
    data_vals    = []
    labels       = []
    observed_vals = []
    for metric, label_val in metrics:
        vals = bootstrap_max_df[metric].dropna().values
        if len(vals) == 0:
            continue
        data_vals.append(vals)
        labels.append(label_val)
        observed_vals.append(observed_max.get(metric, np.nan))
    if not data_vals:
        return
    plt.figure(figsize=(8, 5))
    plt.boxplot(data_vals, showfliers=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    for i, obs in enumerate(observed_vals):
        if np.isfinite(obs):
            plt.scatter(i + 1, obs, color="crimson", marker="*", s=140,
                        zorder=5, label="observed" if i == 0 else None)
    plt.axhline(1.0, color="gray", ls="--", lw=1)
    plt.axhline(2.0, color="gray", ls=":",  lw=1)
    plt.ylabel("Maximum tension across cuts [sigma]")
    plt.title("Data-level star bootstrap stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_permutation_null_plot(null_max_df, observed_max, output_path):
    if null_max_df is None or len(null_max_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    specs = [
        ("derived_halo_max_tension", "Derived inner-halo max tension"),
        ("beta_max_tension",         "Beta max tension"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        vals = null_max_df[metric].dropna().values
        if len(vals) == 0:
            continue
        obs = observed_max.get(metric, np.nan)
        ax.hist(vals, bins=min(15, max(5, len(vals))),
                color="steelblue", alpha=0.75)
        if np.isfinite(obs):
            p_val = empirical_p_value(vals, obs)
            ax.axvline(obs, color="crimson", lw=2,
                       label=f"observed, p={p_val:.3f}")
        ax.set_xlabel("Maximum tension across cuts [sigma]")
        ax.set_ylabel("permutations")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ============================================================
# MULTIPROCESSING WORKERS
# ============================================================
def _bootstrap_worker(args):
    rep_idx, df, threshold_table, n_draws, seed, fit_kwargs = args
    rng        = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(df), size=len(df))
    boot_df    = df.iloc[sample_idx].reset_index(drop=True)
    rep_results = evaluate_fixed_cut_tensions(
        boot_df, threshold_table,
        n_draws=n_draws, rng=rng, fit_kwargs=fit_kwargs, n_mc_resamples=40,
    )
    if len(rep_results) == 0:
        return None
    rep_results["replicate"]  = rep_idx
    rep_results["test_type"]  = "star_bootstrap"
    return rep_results


def _permutation_worker(args):
    rep_idx, df, threshold_table, n_draws, seed, fit_kwargs = args
    rng     = np.random.default_rng(seed)
    perm_df = df.copy().reset_index(drop=True)
    perm_df["SigMg"] = rng.permutation(perm_df["SigMg"].values)
    rep_results = evaluate_fixed_cut_tensions(
        perm_df, threshold_table,
        n_draws=n_draws, rng=rng, fit_kwargs=fit_kwargs, n_mc_resamples=40,
    )
    if len(rep_results) == 0:
        return None
    rep_results["replicate"] = rep_idx
    rep_results["test_type"] = "mg_permutation_null"
    return rep_results


def _run_pool(worker_fn, worker_args, n_processes, desc, outdir,
              all_path, max_path, existing_all, existing_max,
              observed_results):
    all_results = [] if existing_all is None else [existing_all]
    max_rows    = [] if existing_max is None else [existing_max]

    n_processes = max(1, int(n_processes))
    if n_processes == 1:
        iterator = (worker_fn(args) for args in worker_args)
        progress_iter = tqdm(iterator, total=len(worker_args), desc=desc)
        for result in progress_iter:
            if result is None:
                continue
            all_results.append(result)
            max_row    = summarize_max_tensions(result)
            max_row_df = pd.DataFrame([max_row])
            max_row_df["replicate"] = result["replicate"].iloc[0]
            max_rows.append(max_row_df)

            pd.concat(all_results, ignore_index=True).to_csv(all_path, index=False)
            pd.concat(max_rows,    ignore_index=True).to_csv(max_path, index=False)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(worker_fn, worker_args),
                total=len(worker_args), desc=desc,
            ):
                if result is None:
                    continue
                all_results.append(result)
                max_row    = summarize_max_tensions(result)
                max_row_df = pd.DataFrame([max_row])
                max_row_df["replicate"] = result["replicate"].iloc[0]
                max_rows.append(max_row_df)

                pd.concat(all_results, ignore_index=True).to_csv(all_path, index=False)
                pd.concat(max_rows,    ignore_index=True).to_csv(max_path, index=False)

    if not all_results:
        return None, None
    all_df = pd.concat(all_results, ignore_index=True)
    max_df = pd.concat(max_rows,    ignore_index=True)
    all_df.to_csv(all_path, index=False)
    max_df.to_csv(max_path, index=False)
    return all_df, max_df


def run_star_bootstrap_test(df, threshold_table, observed_results,
                             n_reps, n_draws, rng, outdir, fit_kwargs=None,
                             n_processes=2):
    if n_reps <= 0:
        return None, None
    if fit_kwargs is None:
        fit_kwargs = {}
    all_path = os.path.join(outdir, "data_bootstrap_all_tensions.csv")
    max_path = os.path.join(outdir, "data_bootstrap_max_tensions.csv")

    existing_all, existing_max, completed_reps = None, None, set()
    if os.path.exists(all_path) and os.path.exists(max_path):
        existing_all = pd.read_csv(all_path)
        existing_max = pd.read_csv(max_path)
        if "replicate" in existing_max.columns:
            completed_reps = set(
                existing_max["replicate"].dropna().astype(int).unique()
            )
        if len(completed_reps) >= n_reps:
            print(f"Using existing star-bootstrap ({len(completed_reps)} reps)")
            obs_max = summarize_max_tensions(observed_results)
            save_data_bootstrap_plot(
                existing_max, obs_max,
                os.path.join(outdir, "data_bootstrap_max_tension.png"),
            )
            return existing_all, existing_max
        print(f"Resuming star-bootstrap from {len(completed_reps)} completed reps")

    remaining = [i for i in range(n_reps) if i not in completed_reps]
    worker_args = [
        (i, df, threshold_table, n_draws, rng.integers(0, 2**31 - 1), fit_kwargs)
        for i in remaining
    ]
    all_df, max_df = _run_pool(
        _bootstrap_worker, worker_args, n_processes,
        "Star bootstrap", outdir, all_path, max_path,
        existing_all, existing_max, observed_results,
    )
    if all_df is None:
        return None, None
    obs_max = summarize_max_tensions(observed_results)
    save_data_bootstrap_plot(
        max_df, obs_max, os.path.join(outdir, "data_bootstrap_max_tension.png")
    )
    return all_df, max_df


def run_mg_permutation_null_test(df, threshold_table, observed_results,
                                  n_reps, n_draws, rng, outdir, fit_kwargs=None,
                                  n_processes=2):
    if n_reps <= 0:
        return None, None, None
    if fit_kwargs is None:
        fit_kwargs = {}
    all_path     = os.path.join(outdir, "mg_permutation_all_tensions.csv")
    max_path     = os.path.join(outdir, "mg_permutation_max_tensions.csv")
    summary_path = os.path.join(outdir, "mg_permutation_null_summary.csv")

    existing_all, existing_max, completed_reps = None, None, set()
    if all(os.path.exists(p) for p in (all_path, max_path, summary_path)):
        existing_all     = pd.read_csv(all_path)
        existing_max     = pd.read_csv(max_path)
        existing_summary = pd.read_csv(summary_path)
        if "replicate" in existing_max.columns:
            completed_reps = set(
                existing_max["replicate"].dropna().astype(int).unique()
            )
        if len(completed_reps) >= n_reps:
            print(f"Using existing Mg permutation ({len(completed_reps)} reps)")
            obs_max = summarize_max_tensions(observed_results)
            save_permutation_null_plot(
                existing_max, obs_max,
                os.path.join(outdir, "mg_permutation_null.png"),
            )
            return existing_all, existing_max, existing_summary
        print(f"Resuming Mg permutation from {len(completed_reps)} completed reps")

    remaining = [i for i in range(n_reps) if i not in completed_reps]
    worker_args = [
        (i, df, threshold_table, n_draws, rng.integers(0, 2**31 - 1), fit_kwargs)
        for i in remaining
    ]
    all_df, max_df = _run_pool(
        _permutation_worker, worker_args, n_processes,
        "Mg-index permutation null", outdir, all_path, max_path,
        existing_all, existing_max, observed_results,
    )
    if all_df is None:
        return None, None, None

    obs_max = summarize_max_tensions(observed_results)
    summary_rows = []
    for metric in ["derived_halo_max_tension", "native_halo_max_tension",
                   "beta_max_tension"]:
        vals = max_df[metric].values
        summary_rows.append({
            "metric":                  metric,
            "observed":                obs_max.get(metric, np.nan),
            "null_median":             np.nanmedian(vals),
            "null_p16":                np.nanpercentile(vals, 16),
            "null_p84":                np.nanpercentile(vals, 84),
            "empirical_p_ge_observed": empirical_p_value(
                vals, obs_max.get(metric, np.nan)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    save_permutation_null_plot(
        max_df, obs_max, os.path.join(outdir, "mg_permutation_null.png")
    )
    return all_df, max_df, summary_df


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(
    force_fetch=False,
    n_draws=1800,
    outdir="sculptor_threshold_outputs",
    n_star_bootstraps=0,
    n_permutations=0,
    n_processes=2,
    n_walkers=32,
    n_burnin=300,
    n_prod_steps=600,
    robustness_n_draws=450,
    robustness_de_maxiter=45,
    robustness_de_popsize=6,
):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(42)
    df  = fetch_sculptor_data(force_fetch=force_fetch)
    print("Clean member catalog rows:", len(df))

    full_profile = radial_profile(df)
    full_profile.to_csv(os.path.join(outdir, "full_sample_radial_profile.csv"), index=False)
    print("Full-sample radial profile:\n", full_profile)
    save_baseline_diagnostic_plot(df, full_profile,
                                  os.path.join(outdir, "baseline_diagnostic_plots.png"))
    fit_kwargs = {
        "n_walkers": n_walkers,
        "n_burnin": n_burnin,
        "n_prod_steps": n_prod_steps,
    }
    
    # Notice we now pass df directly instead of full_profile
    full_samples = fit_posterior_grid(
        df, n_draws=n_draws, rng=rng, **fit_kwargs
    )
    if full_samples is not None:
        full_samples.to_csv(os.path.join(outdir, "posterior_full_sample.csv"), index=False)
        save_full_sample_jeans_plot(full_profile, full_samples,
                                    os.path.join(outdir, "full_sample_jeans_fit.png"))

    threshold_grid  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_table = build_unique_threshold_table(df, threshold_grid)
    threshold_table.to_csv(os.path.join(outdir, "unique_threshold_table.csv"), index=False)
    print("Unique Mg-index cuts:\n", threshold_table)

    all_rows = []
    for row_idx in tqdm(range(len(threshold_table))):
        threshold_q        = threshold_table.loc[row_idx, "threshold_q"]
        threshold_q_min    = threshold_table.loc[row_idx, "threshold_q_min"]
        threshold_q_max    = threshold_table.loc[row_idx, "threshold_q_max"]
        threshold_q_values = threshold_table.loc[row_idx, "threshold_q_values"]
        limit              = threshold_table.loc[row_idx, "SigMg_limit"]
        file_tag           = threshold_file_tag(limit)

        low_df   = df[df["SigMg"] <= limit].copy()
        high_df  = df[df["SigMg"] >  limit].copy()
        
        # We still calculate profiles just for output/plotting
        low_profile  = radial_profile(low_df)
        high_profile = radial_profile(high_df)
        low_profile.to_csv(os.path.join(outdir,  "profile_low_cut" + file_tag + ".csv"), index=False)
        high_profile.to_csv(os.path.join(outdir, "profile_high_cut" + file_tag + ".csv"), index=False)

        # But we pass unbinned dataframes to the fitter
        low_samples  = fit_posterior_grid(
            low_df, n_draws=n_draws, rng=rng, **fit_kwargs
        )
        high_samples = fit_posterior_grid(
            high_df, n_draws=n_draws, rng=rng, **fit_kwargs
        )
        if low_samples is None or high_samples is None:
            continue

        low_samples.to_csv(os.path.join(outdir,  "posterior_low_cut"  + file_tag + ".csv"), index=False)
        high_samples.to_csv(os.path.join(outdir, "posterior_high_cut" + file_tag + ".csv"), index=False)

        tension = summarize_tension(low_samples, high_samples, rng=rng)
        tension["threshold_q"]        = threshold_q
        tension["threshold_q_min"]    = threshold_q_min
        tension["threshold_q_max"]    = threshold_q_max
        tension["threshold_q_values"] = threshold_q_values
        tension["SigMg_limit"]        = limit
        tension["N_low"]              = len(low_df)
        tension["N_high"]             = len(high_df)
        all_rows.append(tension)

        save_profile_plot(
            low_profile, high_profile, low_samples, high_samples,
            threshold_q, os.path.join(outdir, "jeans_split_cut" + file_tag + ".png"),
        )

    if not all_rows:
        raise RuntimeError(
            "No threshold split produced valid posteriors. "
            "Inspect sample size."
        )

    results_df = pd.concat(all_rows, ignore_index=True)
    results_df = results_df[[
        "threshold_q", "threshold_q_min", "threshold_q_max", "threshold_q_values",
        "SigMg_limit", "N_low", "N_high",
        "param_group", "param",
        "sigma_tension", "sigma_tension_p16", "sigma_tension_p84",
        "overlap", "jsd_nats",
        "low_median", "high_median", "low_p16", "low_p84", "high_p16", "high_p84",
    ]]
    results_csv = os.path.join(outdir, "threshold_tension_summary.csv")
    results_df.to_csv(results_csv, index=False)
    save_threshold_summary_plot(results_df, os.path.join(outdir, "threshold_tension_summary.png"))
    plot_tension_group(results_df, DERIVED_HALO_PARAMS,
                       os.path.join(outdir, "derived_inner_halo_tension.png"),
                       "Derived inner-halo posterior tension")
    plot_tension_group(results_df, NATIVE_HALO_PARAMS,
                       os.path.join(outdir, "native_nfw_parameter_tension.png"),
                       "Native NFW parameter posterior tension")
    plot_tension_group(results_df, ORBIT_TENSION_PARAMS,
                       os.path.join(outdir, "orbital_nuisance_tension.png"),
                       "Orbital anisotropy posterior tension")

    observed_max    = summarize_max_tensions(results_df)
    observed_max_df = pd.DataFrame([observed_max])
    observed_max_df.to_csv(os.path.join(outdir, "observed_max_tensions.csv"), index=False)
    print("Observed maximum tensions:\n", observed_max_df)

    # ── Robustness tests ───────────────────────────────────────────────────────
    robustness_fit_kwargs = {
        "de_maxiter":  robustness_de_maxiter,
        "de_popsize":  robustness_de_popsize,
        "polish":      False,
        "n_walkers":    n_walkers,
        "n_burnin":    150,     
        "n_prod_steps": 300,    
    }

    if n_star_bootstraps > 0:
        run_star_bootstrap_test(
            df, threshold_table, results_df,
            n_reps=n_star_bootstraps, n_draws=robustness_n_draws,
            rng=rng, outdir=outdir, fit_kwargs=robustness_fit_kwargs,
            n_processes=n_processes,
        )
    else:
        with open(os.path.join(outdir, "data_bootstrap_NOT_RUN.txt"), "w") as fh:
            fh.write("Data-level star bootstrap was not run.\n"
                     "Run with --n-star-bootstraps 10 to generate data_bootstrap_* files.\n")

    if n_permutations > 0:
        _, _, perm_summary = run_mg_permutation_null_test(
            df, threshold_table, results_df,
            n_reps=n_permutations, n_draws=robustness_n_draws,
            rng=rng, outdir=outdir, fit_kwargs=robustness_fit_kwargs,
            n_processes=n_processes,
        )
        if perm_summary is not None:
            print("Mg-index permutation null summary:\n", perm_summary)
    else:
        with open(os.path.join(outdir, "mg_permutation_NOT_RUN.txt"), "w") as fh:
            fh.write("Mg-index permutation null test was not run.\n"
                     "Run with --n-permutations 20 to generate mg_permutation_* files.\n")

    print("Saved:", results_csv)
    print("Output directory:", outdir)
    return results_df


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--force-fetch",           action="store_true")
    parser.add_argument("--n-draws",               type=int, default=1800)
    parser.add_argument("--outdir",                type=str, default="sculptor_threshold_outputs")
    parser.add_argument("--n-star-bootstraps",     type=int, default=0)
    parser.add_argument("--n-permutations",        type=int, default=0)
    parser.add_argument("--n-processes",           type=int, default=2)
    parser.add_argument("--n-walkers",             type=int, default=32)
    parser.add_argument("--n-burnin",              type=int, default=300)
    parser.add_argument("--n-prod-steps",          type=int, default=600)
    parser.add_argument("--robustness-n-draws",    type=int, default=450)
    parser.add_argument("--robustness-de-maxiter", type=int, default=45)
    parser.add_argument("--robustness-de-popsize", type=int, default=6)
    args = parser.parse_args()

    run_pipeline(
        force_fetch=args.force_fetch,
        n_draws=args.n_draws,
        outdir=args.outdir,
        n_star_bootstraps=args.n_star_bootstraps,
        n_permutations=args.n_permutations,
        n_processes=args.n_processes,
        n_walkers=args.n_walkers,
        n_burnin=args.n_burnin,
        n_prod_steps=args.n_prod_steps,
        robustness_n_draws=args.robustness_n_draws,
        robustness_de_maxiter=args.robustness_de_maxiter,
        robustness_de_popsize=args.robustness_de_popsize,
    )