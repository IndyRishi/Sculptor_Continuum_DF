import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import multiprocessing

from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution, minimize
from scipy.special import logsumexp

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
RA0_DEG = 15.039
DEC0_DEG = -33.709
DISTANCE_KPC = 86.0
SEARCH_RADIUS_DEG = 0.5
XMATCH_RADIUS_ARCSEC = 1.0
MEMBERSHIP_MIN = 0.90
MIN_BIN_N = 10
N_RADIAL_BINS = 6
PLUMMER_RHALF_KPC = 0.283
G_KPC_KMS2_MSUN = 4.30091e-6
H0_KM_S_MPC = 70.0
RHO_CRIT_MSUN_KPC3 = 3.0 * (H0_KM_S_MPC / 1000.0) ** 2 / (8.0 * np.pi * G_KPC_KMS2_MSUN)
DERIVED_HALO_RADII = [
    (0.15, "150"),
    (0.30, "300"),
    (0.50, "500")
]
DERIVED_HALO_PARAMS = [
    "log10_M150",
    "log10_M300",
    "log10_M500",
    "log10_rho150"
]
NATIVE_HALO_PARAMS = [
    "log10_M200",
    "log10_rs"
]
ORBIT_TENSION_PARAMS = [
    "beta"
]


# ============================================================
# BASIC STATS
# ============================================================
def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return np.sum(weights * values) / np.sum(weights)


def effective_n(weights):
    weights = np.asarray(weights, dtype=float)
    return np.sum(weights) ** 2 / np.sum(weights ** 2)


def intrinsic_dispersion_mle(values, errors, weights=None):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(errors) & np.isfinite(weights) & (errors > 0) & (weights > 0)
    values = values[good]
    errors = errors[good]
    weights = weights[good]
    if len(values) < 3:
        return np.nan, np.nan, np.nan

    mu0 = weighted_mean(values, weights)
    sig0 = np.sqrt(max(np.average((values - mu0) ** 2, weights=weights) - np.average(errors ** 2, weights=weights), 1.0))

    def nll(params):
        mu_val = params[0]
        log_sig = params[1]
        sig_val = np.exp(log_sig)
        var_val = sig_val ** 2 + errors ** 2
        return 0.5 * np.sum(weights * (np.log(2.0 * np.pi * var_val) + (values - mu_val) ** 2 / var_val))

    opt = minimize(nll, x0=np.array([mu0, np.log(sig0)]), method="Nelder-Mead")
    mu_hat = opt.x[0]
    sig_hat = np.exp(opt.x[1])
    n_eff = effective_n(weights)
    sig_err = sig_hat / np.sqrt(max(2.0 * (n_eff - 1.0), 1.0))
    return mu_hat, sig_hat, sig_err


def make_equal_count_bins(dataframe, col_name, n_bins, min_n):
    work = dataframe.sort_values(col_name).copy()
    bins = np.array_split(work.index.to_numpy(), n_bins)
    labels = pd.Series(index=dataframe.index, dtype="float")
    label_val = 0
    for idx_vals in bins:
        if len(idx_vals) >= min_n:
            labels.loc[idx_vals] = label_val
            label_val += 1
    return labels


def radial_profile(dataframe, n_bins=N_RADIAL_BINS, min_n=MIN_BIN_N):
    work = dataframe.copy()
    work["bin"] = make_equal_count_bins(work, "R_kpc", n_bins, min_n)
    rows = []
    for bin_val in sorted(work["bin"].dropna().unique()):
        b = work[work["bin"] == bin_val]
        if len(b) < min_n:
            continue
        mu_val, sig_val, sig_err = intrinsic_dispersion_mle(b["V_los"].values, b["e_V_los"].values, b["P_mem"].values)
        if not np.isfinite(sig_val) or not np.isfinite(sig_err) or sig_err <= 0:
            continue
        rows.append({
            "R_kpc": np.average(b["R_kpc"].values, weights=b["P_mem"].values),
            "sigma_kms": sig_val,
            "sigma_err_kms": sig_err,
            "v_mean_kms": mu_val,
            "N": len(b),
            "N_eff": effective_n(b["P_mem"].values)
        })
    if len(rows) == 0:
        return pd.DataFrame(columns=["R_kpc", "sigma_kms", "sigma_err_kms", "v_mean_kms", "N", "N_eff"])
    return pd.DataFrame(rows).sort_values("R_kpc").reset_index(drop=True)


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_sculptor_data(cache_csv="sculptor_clean_member_catalog.csv", force_fetch=False):
    if os.path.exists(cache_csv) and not force_fetch:
        df = pd.read_csv(cache_csv)
        return df

    print("Fetching Walker et al. catalog")
    Vizier.ROW_LIMIT = -1
    walker = Vizier.get_catalogs("J/AJ/137/3100")[0]
    walker_coords = SkyCoord(np.array(walker["RAJ2000"]).astype(str), np.array(walker["DEJ2000"]).astype(str), unit=(u.hourangle, u.deg), frame="icrs")
    center = SkyCoord(RA0_DEG * u.deg, DEC0_DEG * u.deg, frame="icrs")
    near_walker = walker_coords.separation(center) < SEARCH_RADIUS_DEG * u.deg
    walker_sub = walker[near_walker]
    walker_sub_coords = walker_coords[near_walker]
    print("Walker rows inside search radius: " + str(len(walker_sub)))

    print("Fetching Gaia DR3 cone")
    adql = """
    SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {rad})
    )
    """.format(ra=RA0_DEG, dec=DEC0_DEG, rad=SEARCH_RADIUS_DEG)
    gaia = Gaia.launch_job_async(adql).get_results()
    gaia_coords = SkyCoord(np.array(gaia["ra"], dtype=float), np.array(gaia["dec"], dtype=float), unit=u.deg, frame="icrs")

    idx, sep2d, _ = walker_sub_coords.match_to_catalog_sky(gaia_coords)
    mask = sep2d < XMATCH_RADIUS_ARCSEC * u.arcsec
    matches = pd.DataFrame({
        "walker_local_idx": np.where(mask)[0],
        "gaia_idx": idx[mask],
        "sep_arcsec": sep2d[mask].arcsec
    })
    matches = matches.sort_values("sep_arcsec")
    matches = matches.drop_duplicates("gaia_idx", keep="first")
    matches = matches.drop_duplicates("walker_local_idx", keep="first")
    matches = matches.sort_values("walker_local_idx").reset_index(drop=True)
    print("Unique closest matches: " + str(len(matches)))

    widx = matches["walker_local_idx"].values
    gidx = matches["gaia_idx"].values
    df = pd.DataFrame({
        "Target": np.array(walker_sub["Target"]).astype(str)[widx],
        "V_los": np.array(walker_sub["<HV>"], dtype=float)[widx],
        "e_V_los": np.array(walker_sub["e_<HV>"], dtype=float)[widx],
        "P_mem": np.array(walker_sub["Mmb"], dtype=float)[widx],
        "SigMg": np.array(walker_sub["<SigMg>"], dtype=float)[widx],
        "e_SigMg": np.array(walker_sub["e_<SigMg>"], dtype=float)[widx],
        "RA": np.array(gaia["ra"], dtype=float)[gidx],
        "Dec": np.array(gaia["dec"], dtype=float)[gidx],
        "source_id": np.array(gaia["source_id"])[gidx],
        "sep_arcsec": matches["sep_arcsec"].values
    })
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["V_los", "e_V_los", "P_mem", "SigMg", "RA", "Dec"])
    df = df[(df["e_V_los"] > 0) & (df["P_mem"] >= MEMBERSHIP_MIN)].copy()
    coords = SkyCoord(df["RA"].values * u.deg, df["Dec"].values * u.deg, frame="icrs")
    df["R_kpc"] = coords.separation(center).radian * DISTANCE_KPC
    df.to_csv(cache_csv, index=False)
    print("Saved " + cache_csv)
    return df


# ============================================================
# JEANS MODEL
# ============================================================
def nfw_mass_enclosed(r_kpc, log10_m200, log10_rs):
    m200 = 10.0 ** log10_m200
    rs = 10.0 ** log10_rs
    r200 = (3.0 * m200 / (4.0 * np.pi * 200.0 * RHO_CRIT_MSUN_KPC3)) ** (1.0 / 3.0)
    c200 = r200 / rs
    x = np.asarray(r_kpc, dtype=float) / rs
    fx = np.log1p(x) - x / (1.0 + x)
    fc = np.log1p(c200) - c200 / (1.0 + c200)
    return m200 * fx / fc


def nfw_density(r_kpc, log10_m200, log10_rs):
    m200 = 10.0 ** log10_m200
    rs = 10.0 ** log10_rs
    r200 = (3.0 * m200 / (4.0 * np.pi * 200.0 * RHO_CRIT_MSUN_KPC3)) ** (1.0 / 3.0)
    c200 = r200 / rs
    fc = np.log1p(c200) - c200 / (1.0 + c200)
    rho_s = m200 / (4.0 * np.pi * rs ** 3 * fc)
    x = np.asarray(r_kpc, dtype=float) / rs
    return rho_s / (x * (1.0 + x) ** 2)


def add_derived_halo_quantities(samples_df):
    samples_df = samples_df.copy()
    for radius_kpc, radius_label in DERIVED_HALO_RADII:
        mass_vals = nfw_mass_enclosed(
            radius_kpc,
            samples_df["log10_M200"].values,
            samples_df["log10_rs"].values
        )
        samples_df["log10_M" + radius_label] = np.log10(np.clip(mass_vals, 1e-30, None))
    rho150 = nfw_density(
        0.15,
        samples_df["log10_M200"].values,
        samples_df["log10_rs"].values
    )
    samples_df["log10_rho150"] = np.log10(np.clip(rho150, 1e-30, None))
    return samples_df


def plummer_nu(r_kpc, a_kpc=PLUMMER_RHALF_KPC):
    r_kpc = np.asarray(r_kpc, dtype=float)
    return (1.0 + (r_kpc / a_kpc) ** 2) ** (-2.5)


def projected_jeans_sigma_los(R_eval_kpc, log10_m200, log10_rs, beta):
    R_eval_kpc = np.asarray(R_eval_kpc, dtype=float)
    r_min = max(1e-4, np.nanmin(R_eval_kpc) * 0.05)
    r_max = max(20.0, np.nanmax(R_eval_kpc) * 80.0)
    r_grid = np.geomspace(r_min, r_max, 520)
    nu_grid = plummer_nu(r_grid)
    mass_grid = nfw_mass_enclosed(r_grid, log10_m200, log10_rs)
    integrand = nu_grid * G_KPC_KMS2_MSUN * mass_grid * r_grid ** (2.0 * beta - 2.0)
    rev_int = cumulative_trapezoid(integrand[::-1], r_grid[::-1], initial=0.0)
    radial_int = -rev_int[::-1]
    sigma_r2 = radial_int / (nu_grid * r_grid ** (2.0 * beta))
    sigma_r2 = np.clip(sigma_r2, 1e-8, None)

    out = []
    for R_val in R_eval_kpc:
        lower = max(R_val * (1.0 + 1e-5), r_min)
        r_local = np.geomspace(lower, r_max, 420)
        nu_local = plummer_nu(r_local)
        sr2_local = np.interp(r_local, r_grid, sigma_r2)
        geom = r_local / np.sqrt(r_local ** 2 - R_val ** 2)
        denom = 2.0 * np.trapezoid(nu_local * geom, r_local)
        kernel = 1.0 - beta * R_val ** 2 / r_local ** 2
        numer = 2.0 * np.trapezoid(kernel * nu_local * sr2_local * geom, r_local)
        out.append(np.sqrt(max(numer / denom, 1e-8)))
    return np.array(out)


def chi2_jeans(theta, profile):
    log10_m200 = theta[0]
    log10_rs = theta[1]
    beta = theta[2]
    if beta >= 0.9:
        return 1e50
    R_b = profile["R_kpc"].values
    S_b = profile["sigma_kms"].values
    E_b = profile["sigma_err_kms"].values
    model = projected_jeans_sigma_los(R_b, log10_m200, log10_rs, beta)
    if not np.all(np.isfinite(model)):
        return 1e50
    return np.sum(((S_b - model) / E_b) ** 2)


def prior_ok(samples):
    samples = np.asarray(samples)
    ok = (
        (samples[:, 0] >= 8.0) & (samples[:, 0] <= 11.0) &
        (samples[:, 1] >= -1.2) & (samples[:, 1] <= 1.0) &
        (samples[:, 2] >= -1.5) & (samples[:, 2] <= 0.7)
    )
    return ok


def fit_posterior_grid(profile, n_draws=2500, rng=None, de_maxiter=160, de_popsize=10, polish=True):
    if rng is None:
        rng = np.random.default_rng(7)
    if len(profile) < 3:
        return None

    bounds = [(8.0, 11.0), (-1.2, 1.0), (-1.5, 0.7)]
    opt = differential_evolution(
        lambda x: chi2_jeans(x, profile),
        bounds=bounds,
        seed=7,
        tol=0.04,
        polish=polish,
        workers=1,
        maxiter=de_maxiter,
        popsize=de_popsize
    )
    best = opt.x
    best_chi2 = opt.fun

    proposal_sd = np.array([0.45, 0.35, 0.35])
    raw = rng.normal(loc=best, scale=proposal_sd, size=(n_draws * 5, 3))
    ok = prior_ok(raw)
    raw = raw[ok]
    if len(raw) < 50:
        return None

    chi2_vals = np.array([chi2_jeans(row, profile) for row in raw])
    logw = -0.5 * (chi2_vals - np.nanmin(chi2_vals))
    logw = logw - logsumexp(logw)
    weights = np.exp(logw)
    weights = weights / np.sum(weights)
    take_n = min(n_draws, len(raw))
    idx_vals = rng.choice(np.arange(len(raw)), size=take_n, replace=True, p=weights)
    samples = raw[idx_vals].copy()
    samples_df = pd.DataFrame(samples, columns=["log10_M200", "log10_rs", "beta"])
    samples_df["rs_kpc"] = 10.0 ** samples_df["log10_rs"]
    samples_df["chi2_best"] = best_chi2
    samples_df["best_log10_M200"] = best[0]
    samples_df["best_log10_rs"] = best[1]
    samples_df["best_beta"] = best[2]
    samples_df = add_derived_halo_quantities(samples_df)
    return samples_df


# ============================================================
# TENSION METRICS
# ============================================================
def scalar_tension_from_arrays(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    med_a = np.nanmedian(a)
    med_b = np.nanmedian(b)
    sig_a = 0.5 * (np.nanpercentile(a, 84) - np.nanpercentile(a, 16))
    sig_b = 0.5 * (np.nanpercentile(b, 84) - np.nanpercentile(b, 16))
    denom = np.sqrt(sig_a ** 2 + sig_b ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return np.abs(med_a - med_b) / denom


def scalar_tension(samples_a, samples_b, param):
    return scalar_tension_from_arrays(samples_a[param].values, samples_b[param].values)


def bootstrap_scalar_tension(samples_a, samples_b, param, n_resamples=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    a = samples_a[param].values
    b = samples_b[param].values
    vals = []
    for _ in range(n_resamples):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        vals.append(scalar_tension_from_arrays(aa, bb))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return np.nanpercentile(vals, 16), np.nanpercentile(vals, 84)


def posterior_overlap_1d(samples_a, samples_b, param, bins=40):
    a = samples_a[param].values
    b = samples_b[param].values
    lo = np.nanpercentile(np.concatenate([a, b]), 0.5)
    hi = np.nanpercentile(np.concatenate([a, b]), 99.5)
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hist_b, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    overlap = np.sum(np.minimum(hist_a, hist_b) * widths)
    return overlap


def jsd_1d(samples_a, samples_b, param, bins=40):
    a = samples_a[param].values
    b = samples_b[param].values
    lo = np.nanpercentile(np.concatenate([a, b]), 0.5)
    hi = np.nanpercentile(np.concatenate([a, b]), 99.5)
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hist_b, _ = np.histogram(b, bins=edges, density=False)
    p = hist_a.astype(float) + 1e-12
    q = hist_b.astype(float) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def parameter_group(param):
    if param in DERIVED_HALO_PARAMS:
        return "derived_halo"
    if param in NATIVE_HALO_PARAMS:
        return "native_halo"
    if param in ORBIT_TENSION_PARAMS:
        return "orbital_nuisance"
    return "other"


def summarize_tension(samples_low, samples_high, rng=None, n_mc_resamples=200):
    if rng is None:
        rng = np.random.default_rng(321)
    rows = []
    params = DERIVED_HALO_PARAMS + NATIVE_HALO_PARAMS + ORBIT_TENSION_PARAMS
    for param in params:
        tension_val = scalar_tension(samples_low, samples_high, param)
        tension_p16, tension_p84 = bootstrap_scalar_tension(
            samples_low,
            samples_high,
            param,
            n_resamples=n_mc_resamples,
            rng=rng
        )
        rows.append({
            "param_group": parameter_group(param),
            "param": param,
            "sigma_tension": tension_val,
            "sigma_tension_p16": tension_p16,
            "sigma_tension_p84": tension_p84,
            "overlap": posterior_overlap_1d(samples_low, samples_high, param),
            "jsd_nats": jsd_1d(samples_low, samples_high, param),
            "low_median": np.nanmedian(samples_low[param].values),
            "high_median": np.nanmedian(samples_high[param].values),
            "low_p16": np.nanpercentile(samples_low[param].values, 16),
            "low_p84": np.nanpercentile(samples_low[param].values, 84),
            "high_p16": np.nanpercentile(samples_high[param].values, 16),
            "high_p84": np.nanpercentile(samples_high[param].values, 84)
        })
    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================
def save_profile_plot(profile_low, profile_high, samples_low, samples_high, threshold_q, output_path):
    plt.figure(figsize=(8, 5))
    colors = {"Low-Mg-index": "tab:blue", "High-Mg-index": "tab:red"}
    for label_name, prof, samples in [
        ("Low-Mg-index", profile_low, samples_low),
        ("High-Mg-index", profile_high, samples_high)
    ]:
        if prof is None or len(prof) == 0 or samples is None:
            continue
        plt.errorbar(prof["R_kpc"], prof["sigma_kms"], yerr=prof["sigma_err_kms"], fmt="o", color=colors[label_name], label=label_name + " data")
        theta = [
            np.nanmedian(samples["log10_M200"]),
            np.nanmedian(samples["log10_rs"]),
            np.nanmedian(samples["beta"])
        ]
        r_plot = np.linspace(max(0.01, prof["R_kpc"].min() * 0.8), prof["R_kpc"].max() * 1.15, 100)
        plt.plot(r_plot, projected_jeans_sigma_los(r_plot, theta[0], theta[1], theta[2]), color=colors[label_name], lw=2, label=label_name + " Jeans")
    plt.xlabel("Projected radius [kpc]")
    plt.ylabel("Intrinsic sigma_los [km/s]")
    plt.title("Threshold q=" + str(round(threshold_q, 2)) + ": chemically selected tracer subsets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_tension_group(results_df, params, output_path, title, xlabel="Mg-index cut value"):
    plt.figure(figsize=(9, 5))
    for param in params:
        subset = results_df[results_df["param"] == param].sort_values("SigMg_limit")
        if len(subset) == 0:
            continue
        x_vals = subset["SigMg_limit"].values
        y_vals = subset["sigma_tension"].values
        y_low = subset["sigma_tension_p16"].values
        y_high = subset["sigma_tension_p84"].values
        plt.plot(x_vals, y_vals, marker="o", label=param)
        plt.fill_between(x_vals, y_low, y_high, alpha=0.15)
    plt.axhline(1.0, color="gray", ls="--", lw=1)
    plt.axhline(2.0, color="gray", ls=":", lw=1)
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
        (DERIVED_HALO_PARAMS, "Derived inner-halo tension"),
        (NATIVE_HALO_PARAMS, "Native NFW parameter tension"),
        (ORBIT_TENSION_PARAMS, "Orbital nuisance tension")
    ]
    for ax, spec in zip(axes, plot_specs):
        params = spec[0]
        title = spec[1]
        for param in params:
            subset = results_df[results_df["param"] == param].sort_values("SigMg_limit")
            if len(subset) == 0:
                continue
            x_vals = subset["SigMg_limit"].values
            y_vals = subset["sigma_tension"].values
            y_low = subset["sigma_tension_p16"].values
            y_high = subset["sigma_tension_p84"].values
            ax.plot(x_vals, y_vals, marker="o", label=param)
            ax.fill_between(x_vals, y_low, y_high, alpha=0.15)
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.axhline(2.0, color="gray", ls=":", lw=1)
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
    work = df.copy()
    work["pop"] = np.where(work["SigMg"] > mg_cut, "High-Mg-index", "Low-Mg-index")

    plt.figure(figsize=(13, 10))

    plt.subplot(2, 2, 1)
    plt.hist(work["V_los"], bins=35, weights=work["P_mem"], color="steelblue", alpha=0.85)
    plt.xlabel("V_los [km/s]")
    plt.ylabel("membership-weighted count")
    plt.title("Clean member velocity distribution")

    plt.subplot(2, 2, 2)
    plt.errorbar(full_profile["R_kpc"], full_profile["sigma_kms"], yerr=full_profile["sigma_err_kms"], fmt="o-", color="black")
    plt.xlabel("Projected radius [kpc]")
    plt.ylabel("Intrinsic dispersion [km/s]")
    plt.title("Error-corrected dispersion profile")

    plt.subplot(2, 2, 3)
    for pop_name in ["High-Mg-index", "Low-Mg-index"]:
        subset = work[work["pop"] == pop_name]
        pop_profile = radial_profile(subset, n_bins=4, min_n=max(8, MIN_BIN_N // 2))
        if len(pop_profile) > 0:
            plt.errorbar(pop_profile["R_kpc"], pop_profile["sigma_kms"], yerr=pop_profile["sigma_err_kms"], fmt="o-", label=pop_name)
    plt.xlabel("Projected radius [kpc]")
    plt.ylabel("Intrinsic dispersion [km/s]")
    plt.title("Dispersion by median Mg-index split")
    plt.legend()

    plt.subplot(2, 2, 4)
    for pop_name in ["High-Mg-index", "Low-Mg-index"]:
        subset = work[work["pop"] == pop_name]
        plt.hist(subset["V_los"], bins=25, alpha=0.55, density=True, label=pop_name)
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
        full_profile["R_kpc"],
        full_profile["sigma_kms"],
        yerr=full_profile["sigma_err_kms"],
        fmt="o",
        color="black",
        label="Observed intrinsic dispersion"
    )
    theta = [
        np.nanmedian(full_samples["log10_M200"]),
        np.nanmedian(full_samples["log10_rs"]),
        np.nanmedian(full_samples["beta"])
    ]
    r_plot = np.linspace(max(0.01, full_profile["R_kpc"].min() * 0.8), full_profile["R_kpc"].max() * 1.15, 120)
    plt.plot(
        r_plot,
        projected_jeans_sigma_los(r_plot, theta[0], theta[1], theta[2]),
        color="crimson",
        lw=2,
        label="Projected Jeans model"
    )
    plt.xlabel("Projected radius [kpc]")
    plt.ylabel("sigma_los [km/s]")
    plt.title("Projected Jeans fit with Plummer tracer + NFW halo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def threshold_file_tag(value):
    return str(np.round(float(value), 4)).replace(".", "p").replace("-", "m")


def build_unique_threshold_table(df, threshold_grid):
    rows = []
    for threshold_q in threshold_grid:
        limit = float(df["SigMg"].quantile(threshold_q))
        rows.append({
            "threshold_q": threshold_q,
            "SigMg_limit": limit
        })
    raw = pd.DataFrame(rows)
    grouped = []
    for limit in sorted(raw["SigMg_limit"].unique()):
        sub = raw[raw["SigMg_limit"] == limit]
        q_vals = sub["threshold_q"].values
        grouped.append({
            "threshold_q": float(np.mean(q_vals)),
            "threshold_q_min": float(np.min(q_vals)),
            "threshold_q_max": float(np.max(q_vals)),
            "threshold_q_values": ",".join([str(np.round(x, 3)) for x in q_vals]),
            "SigMg_limit": float(limit)
        })
    return pd.DataFrame(grouped)


def evaluate_fixed_cut_tensions(df, threshold_table, n_draws, rng, fit_kwargs=None, n_mc_resamples=80):
    if fit_kwargs is None:
        fit_kwargs = {}
    all_rows = []
    for row_idx in range(len(threshold_table)):
        threshold_q = threshold_table.loc[row_idx, "threshold_q"]
        threshold_q_min = threshold_table.loc[row_idx, "threshold_q_min"]
        threshold_q_max = threshold_table.loc[row_idx, "threshold_q_max"]
        threshold_q_values = threshold_table.loc[row_idx, "threshold_q_values"]
        limit = threshold_table.loc[row_idx, "SigMg_limit"]
        low_df = df[df["SigMg"] <= limit].copy()
        high_df = df[df["SigMg"] > limit].copy()
        low_profile = radial_profile(low_df)
        high_profile = radial_profile(high_df)
        low_samples = fit_posterior_grid(low_profile, n_draws=n_draws, rng=rng, **fit_kwargs)
        high_samples = fit_posterior_grid(high_profile, n_draws=n_draws, rng=rng, **fit_kwargs)
        if low_samples is None or high_samples is None:
            continue
        tension = summarize_tension(low_samples, high_samples, rng=rng, n_mc_resamples=n_mc_resamples)
        tension["threshold_q"] = threshold_q
        tension["threshold_q_min"] = threshold_q_min
        tension["threshold_q_max"] = threshold_q_max
        tension["threshold_q_values"] = threshold_q_values
        tension["SigMg_limit"] = limit
        tension["N_low"] = len(low_df)
        tension["N_high"] = len(high_df)
        all_rows.append(tension)
    if len(all_rows) == 0:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def max_info_from_subset(subset, prefix):
    if subset is None or len(subset) == 0:
        return {
            prefix + "_max_tension": np.nan,
            prefix + "_max_param": "",
            prefix + "_max_SigMg_limit": np.nan
        }
    idx_val = subset["sigma_tension"].idxmax()
    return {
        prefix + "_max_tension": subset.loc[idx_val, "sigma_tension"],
        prefix + "_max_param": subset.loc[idx_val, "param"],
        prefix + "_max_SigMg_limit": subset.loc[idx_val, "SigMg_limit"]
    }


def summarize_max_tensions(results_df):
    out = {}
    if results_df is None or len(results_df) == 0:
        out.update(max_info_from_subset(pd.DataFrame(), "derived_halo"))
        out.update(max_info_from_subset(pd.DataFrame(), "native_halo"))
        out.update(max_info_from_subset(pd.DataFrame(), "beta"))
        return out
    out.update(max_info_from_subset(results_df[results_df["param_group"] == "derived_halo"], "derived_halo"))
    out.update(max_info_from_subset(results_df[results_df["param_group"] == "native_halo"], "native_halo"))
    out.update(max_info_from_subset(results_df[results_df["param"] == "beta"], "beta"))
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
        ("beta_max_tension", "beta")
    ]
    data_vals = []
    labels = []
    observed_vals = []
    for metric, label_val in metrics:
        vals = bootstrap_max_df[metric].dropna().values
        if len(vals) == 0:
            continue
        data_vals.append(vals)
        labels.append(label_val)
        observed_vals.append(observed_max.get(metric, np.nan))
    if len(data_vals) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.boxplot(data_vals, showfliers=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    for idx_val, observed_val in enumerate(observed_vals):
        if np.isfinite(observed_val):
            plt.scatter(idx_val + 1, observed_val, color="crimson", marker="*", s=140, zorder=5, label="observed" if idx_val == 0 else None)
    plt.axhline(1.0, color="gray", ls="--", lw=1)
    plt.axhline(2.0, color="gray", ls=":", lw=1)
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
        ("beta_max_tension", "Beta max tension")
    ]
    for ax, spec in zip(axes, specs):
        metric = spec[0]
        title = spec[1]
        vals = null_max_df[metric].dropna().values
        if len(vals) == 0:
            continue
        observed_val = observed_max.get(metric, np.nan)
        ax.hist(vals, bins=min(15, max(5, len(vals))), color="steelblue", alpha=0.75)
        if np.isfinite(observed_val):
            p_val = empirical_p_value(vals, observed_val)
            ax.axvline(observed_val, color="crimson", lw=2, label="observed, p=" + str(round(p_val, 3)))
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
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(df), size=len(df))
    boot_df = df.iloc[sample_idx].reset_index(drop=True)
    rep_results = evaluate_fixed_cut_tensions(
        boot_df,
        threshold_table,
        n_draws=n_draws,
        rng=rng,
        fit_kwargs=fit_kwargs,
        n_mc_resamples=40
    )
    if len(rep_results) == 0:
        return None
    rep_results["replicate"] = rep_idx
    rep_results["test_type"] = "star_bootstrap"
    return rep_results

def _permutation_worker(args):
    rep_idx, df, threshold_table, n_draws, seed, fit_kwargs = args
    rng = np.random.default_rng(seed)
    perm_df = df.copy().reset_index(drop=True)
    perm_df["SigMg"] = rng.permutation(perm_df["SigMg"].values)
    rep_results = evaluate_fixed_cut_tensions(
        perm_df,
        threshold_table,
        n_draws=n_draws,
        rng=rng,
        fit_kwargs=fit_kwargs,
        n_mc_resamples=40
    )
    if len(rep_results) == 0:
        return None
    rep_results["replicate"] = rep_idx
    rep_results["test_type"] = "mg_permutation_null"
    return rep_results


def run_star_bootstrap_test(df, threshold_table, observed_results, n_reps, n_draws, rng, outdir, fit_kwargs=None):
    if n_reps <= 0:
        return None, None
    if fit_kwargs is None:
        fit_kwargs = {}
    all_path = os.path.join(outdir, "data_bootstrap_all_tensions.csv")
    max_path = os.path.join(outdir, "data_bootstrap_max_tensions.csv")
    existing_all = None
    existing_max = None
    completed_reps = set()
    if os.path.exists(all_path) and os.path.exists(max_path):
        existing_all = pd.read_csv(all_path)
        existing_max = pd.read_csv(max_path)
        if "replicate" in existing_max.columns:
            completed_reps = set(existing_max["replicate"].dropna().astype(int).unique().tolist())
        existing_reps = len(completed_reps)
        if existing_reps >= n_reps:
            print("Using existing star-bootstrap outputs with " + str(existing_reps) + " replicates")
            observed_max = summarize_max_tensions(observed_results)
            save_data_bootstrap_plot(existing_max, observed_max, os.path.join(outdir, "data_bootstrap_max_tension.png"))
            return existing_all, existing_max
        print("Resuming star-bootstrap from " + str(existing_reps) + " completed replicates")
    all_results = []
    max_rows = []
    if existing_all is not None:
        all_results.append(existing_all)
    if existing_max is not None:
        max_rows.append(existing_max)
    remaining_reps = [rep_idx for rep_idx in range(n_reps) if rep_idx not in completed_reps]
    
    worker_args = [(rep_idx, df, threshold_table, n_draws, rng.integers(0, 2**31 - 1), fit_kwargs) for rep_idx in remaining_reps]
    
    num_processes = 2 
    print(f"Running bootstrap safely with {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(_bootstrap_worker, worker_args), total=len(worker_args), desc="Star bootstrap"):
            if result is not None:
                all_results.append(result)
                max_row = summarize_max_tensions(result)
                max_row_df = pd.DataFrame([max_row])
                max_row_df["replicate"] = result["replicate"].iloc[0]
                max_rows.append(max_row_df)

                all_df = pd.concat(all_results, ignore_index=True)
                max_df = pd.concat(max_rows, ignore_index=True)
                all_df.to_csv(all_path, index=False)
                max_df.to_csv(max_path, index=False)

    if len(all_results) == 0:
        return None, None
    all_df = pd.concat(all_results, ignore_index=True)
    max_df = pd.concat(max_rows, ignore_index=True)
    all_df.to_csv(all_path, index=False)
    max_df.to_csv(max_path, index=False)
    observed_max = summarize_max_tensions(observed_results)
    save_data_bootstrap_plot(max_df, observed_max, os.path.join(outdir, "data_bootstrap_max_tension.png"))
    return all_df, max_df


def run_mg_permutation_null_test(df, threshold_table, observed_results, n_reps, n_draws, rng, outdir, fit_kwargs=None):
    if n_reps <= 0:
        return None, None, None
    if fit_kwargs is None:
        fit_kwargs = {}
    all_path = os.path.join(outdir, "mg_permutation_all_tensions.csv")
    max_path = os.path.join(outdir, "mg_permutation_max_tensions.csv")
    summary_path = os.path.join(outdir, "mg_permutation_null_summary.csv")
    existing_all = None
    existing_max = None
    completed_reps = set()
    if os.path.exists(all_path) and os.path.exists(max_path) and os.path.exists(summary_path):
        existing_all = pd.read_csv(all_path)
        existing_max = pd.read_csv(max_path)
        existing_summary = pd.read_csv(summary_path)
        if "replicate" in existing_max.columns:
            completed_reps = set(existing_max["replicate"].dropna().astype(int).unique().tolist())
        existing_reps = len(completed_reps)
        if existing_reps >= n_reps:
            print("Using existing Mg-index permutation outputs with " + str(existing_reps) + " replicates")
            observed_max = summarize_max_tensions(observed_results)
            save_permutation_null_plot(existing_max, observed_max, os.path.join(outdir, "mg_permutation_null.png"))
            return existing_all, existing_max, existing_summary
        print("Resuming Mg-index permutation outputs from " + str(existing_reps) + " completed replicates")
    all_results = []
    max_rows = []
    if existing_all is not None:
        all_results.append(existing_all)
    if existing_max is not None:
        max_rows.append(existing_max)
    remaining_reps = [rep_idx for rep_idx in range(n_reps) if rep_idx not in completed_reps]

    worker_args = [(rep_idx, df, threshold_table, n_draws, rng.integers(0, 2**31 - 1), fit_kwargs) for rep_idx in remaining_reps]

    num_processes = 2
    print(f"Running permutation safely with {num_processes} processes...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(_permutation_worker, worker_args), total=len(worker_args), desc="Mg-index permutation null"):
            if result is not None:
                all_results.append(result)
                max_row = summarize_max_tensions(result)
                max_row_df = pd.DataFrame([max_row])
                max_row_df["replicate"] = result["replicate"].iloc[0]
                max_rows.append(max_row_df)

                all_df = pd.concat(all_results, ignore_index=True)
                max_df = pd.concat(max_rows, ignore_index=True)
                all_df.to_csv(all_path, index=False)
                max_df.to_csv(max_path, index=False)

    if len(all_results) == 0:
        return None, None, None
    all_df = pd.concat(all_results, ignore_index=True)
    max_df = pd.concat(max_rows, ignore_index=True)
    all_df.to_csv(all_path, index=False)
    max_df.to_csv(max_path, index=False)
    observed_max = summarize_max_tensions(observed_results)
    summary_rows = []
    for metric in ["derived_halo_max_tension", "native_halo_max_tension", "beta_max_tension"]:
        summary_rows.append({
            "metric": metric,
            "observed": observed_max.get(metric, np.nan),
            "null_median": np.nanmedian(max_df[metric].values),
            "null_p16": np.nanpercentile(max_df[metric].values, 16),
            "null_p84": np.nanpercentile(max_df[metric].values, 84),
            "empirical_p_ge_observed": empirical_p_value(max_df[metric].values, observed_max.get(metric, np.nan))
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    save_permutation_null_plot(max_df, observed_max, os.path.join(outdir, "mg_permutation_null.png"))
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
    robustness_n_draws=450,
    robustness_de_maxiter=45,
    robustness_de_popsize=6
):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(42)
    df = fetch_sculptor_data(force_fetch=force_fetch)
    print("Clean member catalog rows: " + str(len(df)))
    print(df.head())

    full_profile = radial_profile(df)
    full_profile.to_csv(os.path.join(outdir, "full_sample_radial_profile.csv"), index=False)
    print("Full-sample radial profile")
    print(full_profile)
    save_baseline_diagnostic_plot(df, full_profile, os.path.join(outdir, "baseline_diagnostic_plots.png"))
    full_samples = fit_posterior_grid(full_profile, n_draws=n_draws, rng=rng)
    if full_samples is not None:
        full_samples.to_csv(os.path.join(outdir, "posterior_full_sample.csv"), index=False)
        save_full_sample_jeans_plot(full_profile, full_samples, os.path.join(outdir, "full_sample_jeans_fit.png"))

    threshold_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_table = build_unique_threshold_table(df, threshold_grid)
    threshold_table.to_csv(os.path.join(outdir, "unique_threshold_table.csv"), index=False)
    print("Unique Mg-index cuts")
    print(threshold_table)
    all_rows = []

    for row_idx in tqdm(range(len(threshold_table))):
        threshold_q = threshold_table.loc[row_idx, "threshold_q"]
        threshold_q_min = threshold_table.loc[row_idx, "threshold_q_min"]
        threshold_q_max = threshold_table.loc[row_idx, "threshold_q_max"]
        threshold_q_values = threshold_table.loc[row_idx, "threshold_q_values"]
        limit = threshold_table.loc[row_idx, "SigMg_limit"]
        file_tag = threshold_file_tag(limit)
        low_df = df[df["SigMg"] <= limit].copy()
        high_df = df[df["SigMg"] > limit].copy()
        low_profile = radial_profile(low_df)
        high_profile = radial_profile(high_df)
        low_profile.to_csv(os.path.join(outdir, "profile_low_cut" + file_tag + ".csv"), index=False)
        high_profile.to_csv(os.path.join(outdir, "profile_high_cut" + file_tag + ".csv"), index=False)

        low_samples = fit_posterior_grid(low_profile, n_draws=n_draws, rng=rng)
        high_samples = fit_posterior_grid(high_profile, n_draws=n_draws, rng=rng)
        if low_samples is None or high_samples is None:
            continue

        low_samples.to_csv(os.path.join(outdir, "posterior_low_cut" + file_tag + ".csv"), index=False)
        high_samples.to_csv(os.path.join(outdir, "posterior_high_cut" + file_tag + ".csv"), index=False)
        tension = summarize_tension(low_samples, high_samples, rng=rng)
        tension["threshold_q"] = threshold_q
        tension["threshold_q_min"] = threshold_q_min
        tension["threshold_q_max"] = threshold_q_max
        tension["threshold_q_values"] = threshold_q_values
        tension["SigMg_limit"] = limit
        tension["N_low"] = len(low_df)
        tension["N_high"] = len(high_df)
        all_rows.append(tension)
        save_profile_plot(
            low_profile,
            high_profile,
            low_samples,
            high_samples,
            threshold_q,
            os.path.join(outdir, "jeans_split_cut" + file_tag + ".png")
        )

    if len(all_rows) == 0:
        raise RuntimeError("No threshold split produced valid posteriors. Lower MIN_BIN_N or inspect sample size.")

    results_df = pd.concat(all_rows, ignore_index=True)
    results_df = results_df[[
        "threshold_q",
        "threshold_q_min",
        "threshold_q_max",
        "threshold_q_values",
        "SigMg_limit",
        "N_low",
        "N_high",
        "param_group",
        "param",
        "sigma_tension",
        "sigma_tension_p16",
        "sigma_tension_p84",
        "overlap",
        "jsd_nats",
        "low_median",
        "high_median",
        "low_p16",
        "low_p84",
        "high_p16",
        "high_p84"
    ]]
    results_csv = os.path.join(outdir, "threshold_tension_summary.csv")
    results_df.to_csv(results_csv, index=False)
    save_threshold_summary_plot(results_df, os.path.join(outdir, "threshold_tension_summary.png"))
    plot_tension_group(
        results_df,
        DERIVED_HALO_PARAMS,
        os.path.join(outdir, "derived_inner_halo_tension.png"),
        "Derived inner-halo posterior tension"
    )
    plot_tension_group(
        results_df,
        NATIVE_HALO_PARAMS,
        os.path.join(outdir, "native_nfw_parameter_tension.png"),
        "Native NFW parameter posterior tension"
    )
    plot_tension_group(
        results_df,
        ORBIT_TENSION_PARAMS,
        os.path.join(outdir, "orbital_nuisance_tension.png"),
        "Orbital anisotropy posterior tension"
    )
    observed_max = summarize_max_tensions(results_df)
    observed_max_df = pd.DataFrame([observed_max])
    observed_max_df.to_csv(os.path.join(outdir, "observed_max_tensions.csv"), index=False)
    print("Observed maximum tensions")
    print(observed_max_df)

    robustness_fit_kwargs = {
        "de_maxiter": robustness_de_maxiter,
        "de_popsize": robustness_de_popsize,
        "polish": False
    }
    if n_star_bootstraps > 0:
        run_star_bootstrap_test(
            df,
            threshold_table,
            results_df,
            n_reps=n_star_bootstraps,
            n_draws=robustness_n_draws,
            rng=rng,
            outdir=outdir,
            fit_kwargs=robustness_fit_kwargs
        )
    else:
        with open(os.path.join(outdir, "data_bootstrap_NOT_RUN.txt"), "w") as file_obj:
            file_obj.write("Data-level star bootstrap was not run.\n")
            file_obj.write("Run with --n-star-bootstraps 10 to generate data_bootstrap_* files.\n")
    if n_permutations > 0:
        _, _, permutation_summary = run_mg_permutation_null_test(
            df,
            threshold_table,
            results_df,
            n_reps=n_permutations,
            n_draws=robustness_n_draws,
            rng=rng,
            outdir=outdir,
            fit_kwargs=robustness_fit_kwargs
        )
        if permutation_summary is not None:
            print("Mg-index permutation null summary")
            print(permutation_summary)
    else:
        with open(os.path.join(outdir, "mg_permutation_NOT_RUN.txt"), "w") as file_obj:
            file_obj.write("Mg-index permutation null test was not run.\n")
            file_obj.write("Run with --n-permutations 20 to generate mg_permutation_* files.\n")

    print("Saved " + results_csv)
    print("Output directory: " + outdir)
    print("Robustness files are only generated when --n-star-bootstraps or --n-permutations are greater than 0.")
    print(results_df.head(12))
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--n-draws", type=int, default=1800)
    parser.add_argument("--outdir", type=str, default="sculptor_threshold_outputs")
    parser.add_argument("--n-star-bootstraps", type=int, default=0)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--robustness-n-draws", type=int, default=450)
    parser.add_argument("--robustness-de-maxiter", type=int, default=45)
    parser.add_argument("--robustness-de-popsize", type=int, default=6)
    args = parser.parse_args()
    run_pipeline(
        force_fetch=args.force_fetch,
        n_draws=args.n_draws,
        outdir=args.outdir,
        n_star_bootstraps=args.n_star_bootstraps,
        n_permutations=args.n_permutations,
        robustness_n_draws=args.robustness_n_draws,
        robustness_de_maxiter=args.robustness_de_maxiter,
        robustness_de_popsize=args.robustness_de_popsize
    )