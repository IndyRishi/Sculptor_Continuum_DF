import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

# ============================================================
# CONFIGURATION
# ============================================================
RA0_DEG = 15.039
DEC0_DEG = -33.709
DISTANCE_KPC = 86.0

# Walker's coordinates have small astrometric errors. Since Sculptor's 
# proper motion only moves stars ~0.002 arcsec over 16 years, a slightly 
# wider search radius (2.5") captures the stars we were losing before.
XMATCH_RADIUS_ARCSEC = 2.5  

# We relax the initial 1D velocity cut to 80% to let the 3D Gaia data 
# have the final say on membership.
WALKER_MEMBERSHIP_MIN = 0.80  
FINAL_JOINT_MEMBERSHIP_MIN = 0.95

# ============================================================
# 1. FETCH WALKER CATALOG (1D Velocity + Chemistry)
# ============================================================
def fetch_walker_data():
    print("Fetching Walker et al. (2009) catalog...")
    Vizier.ROW_LIMIT = -1
    walker = Vizier.get_catalogs("J/AJ/137/3100")[0]
    
    df = pd.DataFrame({
        "Target":    np.array(walker["Target"]).astype(str),
        "RA_J2000":  np.array(walker["RAJ2000"]).astype(str),
        "Dec_J2000": np.array(walker["DEJ2000"]).astype(str),
        "V_los":     np.array(walker["<HV>"], dtype=float),
        "e_V_los":   np.array(walker["e_<HV>"], dtype=float),
        "P_mem_1D":  np.array(walker["Mmb"], dtype=float),
        "SigMg":     np.array(walker["<SigMg>"], dtype=float),
    })
    
    # Filter for stars with valid kinematics and relaxed initial membership
    df = df.dropna(subset=["V_los", "e_V_los", "P_mem_1D", "SigMg"])
    df = df[(df["e_V_los"] > 0) & (df["P_mem_1D"] >= WALKER_MEMBERSHIP_MIN)].copy()
    
    # Convert string coords to degrees for cross-matching
    coords = SkyCoord(df["RA_J2000"], df["Dec_J2000"], unit=(u.hourangle, u.deg), frame="icrs")
    df["RA_deg"] = coords.ra.deg
    df["Dec_deg"] = coords.dec.deg
    
    print(f"Recovered {len(df)} potential Walker members.")
    return df, coords

# ============================================================
# 2. FETCH GAIA DR3 & CROSS-MATCH (Proper Motions)
# ============================================================
def fetch_gaia_and_crossmatch(walker_df, walker_coords):
    print("Querying Gaia DR3 for Proper Motions (This may take a moment)...")
    
    # Forcefully bypass the server's default row limits by explicitly requesting 
    # up to 500,000 rows natively in the ADQL query.
    query = f"""
    SELECT TOP 500000 source_id, ra, dec, pmra, pmdec, pmra_error, pmdec_error
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
      POINT('ICRS', ra, dec),
      CIRCLE('ICRS', {RA0_DEG}, {DEC0_DEG}, 1.0)
    ) AND pmra IS NOT NULL AND pmdec IS NOT NULL
    """
    job = Gaia.launch_job_async(query)
    gaia_data = job.get_results()
    
    print(f"Downloaded {len(gaia_data)} stars from the Gaia server.")
    
    gaia_coords = SkyCoord(ra=gaia_data['ra'], dec=gaia_data['dec'], unit=(u.deg, u.deg), frame='icrs')
    
    # Cross-match Walker to Gaia (expanded to 4.0" to account for 2009 ground-based fiber precision)
    XMATCH_RADIUS_ARCSEC = 4.0
    print(f"Cross-matching with a {XMATCH_RADIUS_ARCSEC} arcsec radius...")
    idx, d2d, d3d = walker_coords.match_to_catalog_sky(gaia_coords)
    
    # Create mask for valid matches
    match_mask = d2d.arcsec < XMATCH_RADIUS_ARCSEC
    
    # Merge datasets
    matched_walker = walker_df[match_mask].copy()
    matched_gaia = gaia_data[idx[match_mask]]
    
    matched_walker["pmra"] = np.array(matched_gaia["pmra"])
    matched_walker["pmdec"] = np.array(matched_gaia["pmdec"])
    matched_walker["e_pmra"] = np.array(matched_gaia["pmra_error"])
    matched_walker["e_pmdec"] = np.array(matched_gaia["pmdec_error"])
    
    print(f"Successfully cross-matched {len(matched_walker)} stars with Gaia DR3.")
    return matched_walker

# ============================================================
# 3. GAUSSIAN MIXTURE MODEL (Separating Milky Way Foreground)
# ============================================================
def calculate_3d_membership(df):
    print("Running 2-Component Gaussian Mixture Model on Proper Motions...")
    
    # Extract PMs into a 2D array for the GMM
    X_pm = df[["pmra", "pmdec"]].dropna().values
    
    # We expect 2 components: The tight Sculptor clump, and the diffuse Milky Way background
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X_pm)
    
    # The component with the smaller covariance (tighter clump) is Sculptor
    covariances = [np.trace(cov) for cov in gmm.covariances_]
    sculptor_component_idx = np.argmin(covariances)
    
    # Get probability of belonging to the Sculptor PM clump
    pm_probabilities = gmm.predict_proba(X_pm)[:, sculptor_component_idx]
    df["P_mem_PM"] = pm_probabilities
    
    # Combine Walker's 1D velocity probability with Gaia's 2D PM probability
    df["P_mem_Joint"] = df["P_mem_1D"] * df["P_mem_PM"]
    
    # Final Strict Cut
    final_df = df[df["P_mem_Joint"] >= FINAL_JOINT_MEMBERSHIP_MIN].copy()
    
    # Calculate Projected Radius (R_kpc) for the final sample
    center = SkyCoord(RA0_DEG * u.deg, DEC0_DEG * u.deg, frame="icrs")
    final_coords = SkyCoord(final_df["RA_deg"], final_df["Dec_deg"], unit=(u.deg, u.deg), frame="icrs")
    final_df["R_kpc"] = final_coords.separation(center).radian * DISTANCE_KPC
    
    print(f"GMM Filtering Complete. Final Highly-Pure 3D Sample: {len(final_df)} stars.")
    return final_df, gmm

# ============================================================
# 4. DIAGNOSTIC PLOTTING
# ============================================================
def plot_gmm_results(raw_df, final_df, output_file="proper_motion_gmm.png"):
    # Expanded to 15x5 to accommodate 3 plots side-by-side
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Proper Motion Space
    plt.subplot(1, 3, 1)
    plt.scatter(raw_df["pmra"], raw_df["pmdec"], c='lightgray', s=10, label="All Walker-Gaia Matches", alpha=0.5)
    plt.scatter(final_df["pmra"], final_df["pmdec"], c='crimson', s=15, label="Final Sculptor Members (>95%)")
    plt.xlabel(r"$\mu_\alpha \cos\delta$ (mas/yr)")
    plt.ylabel(r"$\mu_\delta$ (mas/yr)")
    plt.title("Proper Motion Cleaning via GMM")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    
    # Plot 2: Continuous Metallicity Gradient
    plt.subplot(1, 3, 2)
    plt.scatter(final_df["R_kpc"], final_df["SigMg"], c='steelblue', s=15, alpha=0.7)
    plt.xlabel("Projected Radius $R_{kpc}$")
    plt.ylabel("Metallicity Indicator (Mg-index)")
    plt.title("Continuous Chemistry of Final Sample")

    # Plot 3: Metallicity Abundance Histogram
    plt.subplot(1, 3, 3)
    # Increased bins from 25 to 75 for a much higher resolution distribution
    plt.hist(final_df["SigMg"].dropna(), bins=75, color='mediumpurple', edgecolor='black', alpha=0.75)
    plt.xlabel("Metallicity Indicator (Mg-index)")
    plt.ylabel("Star Count")
    plt.title("Metallicity Abundance Distribution")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    print(f"Saved diagnostic plot to {output_file}")

# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    # 1. Get Walker Data
    walker_df, walker_coords = fetch_walker_data()
    
    # 2. Get Gaia 3D Kinematics
    matched_df = fetch_gaia_and_crossmatch(walker_df, walker_coords)
    
    # 3. Clean with GMM
    final_clean_df, gmm_model = calculate_3d_membership(matched_df)
    
    # 4. Save the continuous, 3D kinematic dataset for AGAMA
    output_csv = "sculptor_3d_kinematics_continuous.csv"
    final_clean_df.to_csv(output_csv, index=False)
    print(f"Saved final pipeline dataset to {output_csv}")
    
    # 5. Plot diagnostics to verify the GMM worked
    plot_gmm_results(matched_df, final_clean_df)