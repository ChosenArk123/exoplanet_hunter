import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import lightkurve as lk
from astroquery.mast import Catalogs
from astropy import units as u

# --- CONFIGURATION & CONSTANTS ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress minor warnings from dependencies to keep CLI clean
warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")
warnings.filterwarnings("ignore", category=RuntimeWarning)

CONSTANTS = {
    "AU_TO_SOLAR_RADII": 215.032,
    "SOLAR_RADIUS_TO_JUPITER": 9.73,
    "SOLAR_RADIUS_TO_EARTH": 109.076,
    "DEFAULT_PERIOD_MIN": 0.5,
    "DEFAULT_PERIOD_MAX": 15.0,
    "DEFAULT_N_PERIODS": 10000,
}


# --- DATACLASSES ---
@dataclass
class StarParams:
    name: str
    radius: float  # Solar Radii
    mass: float  # Solar Masses
    teff: float  # Kelvin
    provenance: str  # Where did this data come from?


@dataclass
class BlsResult:
    period: float
    t0: float
    duration: float
    depth: float
    power: float
    period_grid: np.ndarray
    power_grid: np.ndarray
    model_fold_time: np.ndarray
    model_fold_flux: np.ndarray


@dataclass
class PlanetEstimate:
    radius_jupiter: float
    radius_earth: float
    semi_major_axis_au: float
    equilibrium_temp: float
    transit_depth_ppm: float
    snr: float  # Signal-to-Noise Ratio estimate


# --- CORE FUNCTIONS ---

def fetch_star_params(target: str) -> StarParams:
    """
    Queries the TESS Input Catalog (TIC) via MAST for stellar parameters.
    Returns defaults if query fails or fields are missing.
    """
    logger.info(f"Fetching stellar parameters for {target}...")
    try:
        # Search TIC (covers both Kepler and TESS targets)
        catalog_data = Catalogs.query_object(target, radius=0.001, catalog="TIC")

        if len(catalog_data) == 0:
            logger.warning("Star not found in TIC. Using Solar defaults.")
            return StarParams(target, 1.0, 1.0, 5777.0, "Solar Default")

        # Extract with fallbacks for NaNs
        row = catalog_data[0]
        radius = row['rad'] if not np.isnan(row['rad']) else 1.0
        mass = row['mass'] if not np.isnan(row['mass']) else 1.0
        teff = row['Teff'] if not np.isnan(row['Teff']) else 5777.0

        logger.info(f"Star Params: R={radius:.2f} solRad, M={mass:.2f} solMass, T={teff:.0f} K")
        return StarParams(target, float(radius), float(mass), float(teff), "TIC")

    except Exception as e:
        logger.error(f"Catalog query failed: {e}. Using defaults.")
        return StarParams(target, 1.0, 1.0, 5777.0, "Error Default")


def download_lightcurve(target: str, mission: str = 'auto', **kwargs) -> lk.LightCurve:
    """
    Downloads and stitches light curves.
    Args:
        target: Star name
        mission: 'kepler', 'tess', or 'auto'
        kwargs: 'quarter' (Kepler) or 'sector' (TESS)
    """
    logger.info(f"Searching for light curve data ({mission})...")

    search_criteria = {'target': target}

    # Handle Mission Specifics
    if mission == 'kepler':
        search_criteria['author'] = 'Kepler'
        if kwargs.get('quarter') is not None:
            search_criteria['quarter'] = kwargs['quarter']
    elif mission == 'tess':
        search_criteria['author'] = 'SPOC'
        if kwargs.get('sector') is not None:
            search_criteria['sector'] = kwargs['sector']

    try:
        search = lk.search_lightcurve(**search_criteria)

        if len(search) == 0:
            # Fallback for auto mode
            if mission == 'auto':
                logger.info("Auto-search: Retrying with broad search...")
                search = lk.search_lightcurve(target)

            if len(search) == 0:
                raise ValueError(f"No light curve data found for {target}")

        logger.info(f"Found {len(search)} observation(s). Downloading...")
        # Download all found and stitch them together (handles multi-quarter/sector)
        lc_collection = search.download_all()
        if lc_collection is None or len(lc_collection) == 0:
            raise ValueError("Download failed or returned empty.")

        # Stitching is crucial for long-period detection, but simple download is safer for demos
        # We will use the first result if stitching fails, or stitch if possible.
        try:
            lc = lc_collection.stitch()
        except:
            logger.warning("Stitching failed, using first available light curve.")
            lc = lc_collection[0]

        return lc

    except Exception as e:
        raise RuntimeError(f"Light curve acquisition failed: {e}")


def preprocess_lightcurve(lc: lk.LightCurve, window_length: int = 401) -> lk.LightCurve:
    """Cleans, normalizes, and flattens the light curve."""
    logger.info("Preprocessing: Removing NaNs and Flattening...")
    lc = lc.remove_nans().normalize()

    # Flatten removes stellar variability (rotation, spots)
    # The window_length must be longer than the transit duration!
    flat_lc = lc.flatten(window_length=window_length)
    return flat_lc


def run_bls(flat_lc: lk.LightCurve, p_min: float, p_max: float, n_periods: int) -> BlsResult:
    """Runs the Box Least Squares (BLS) periodogram."""
    logger.info(f"Running BLS Search (Period: {p_min}-{p_max}d, Steps: {n_periods})...")

    period_grid = np.linspace(p_min, p_max, n_periods)

    # Calculate BLS
    periodogram = flat_lc.to_periodogram(method='bls', period=period_grid, frequency_factor=500)

    best_period = periodogram.period_at_max_power.value
    best_t0 = periodogram.transit_time_at_max_power.value
    best_dur = periodogram.duration_at_max_power.value
    best_depth = periodogram.depth_at_max_power.value
    max_power = periodogram.max_power.value

    logger.info(f"BLS Complete. Best Period: {best_period:.4f} days")

    # Generate the model for plotting
    planet_model = periodogram.get_transit_model(
        period=best_period,
        transit_time=best_t0,
        duration=best_dur
    )
    folded_model = planet_model.fold(best_period, best_t0)

    return BlsResult(
        period=float(best_period),
        t0=float(best_t0),
        duration=float(best_dur),
        depth=float(best_depth),
        power=float(max_power),
        period_grid=periodogram.period.value,
        power_grid=periodogram.power.value,
        model_fold_time=folded_model.time.value,
        model_fold_flux=folded_model.flux.value
    )


def estimate_planet_params(star: StarParams, bls: BlsResult, flat_lc: lk.LightCurve) -> PlanetEstimate:
    """Calculates physical properties and SNR."""

    # 1. Radius Calculation
    # Rp = R* * sqrt(Depth)
    r_planet_solar = star.radius * np.sqrt(bls.depth)
    r_jup = r_planet_solar * CONSTANTS["SOLAR_RADIUS_TO_JUPITER"]
    r_earth = r_planet_solar * CONSTANTS["SOLAR_RADIUS_TO_EARTH"]

    # 2. Semi-Major Axis (Kepler's 3rd Law)
    # a(AU) approx cuberoot(M* * P_years^2)
    p_years = bls.period / 365.25
    a_au = (star.mass * (p_years ** 2)) ** (1 / 3)

    # 3. Equilibrium Temperature
    # Teq = T* * sqrt(R* / 2a)
    # Convert 'a' to Solar Radii for units to cancel
    a_solar_radii = a_au * CONSTANTS["AU_TO_SOLAR_RADII"]
    t_eq = star.teff * np.sqrt(star.radius / (2 * a_solar_radii))

    # 4. SNR Estimation
    # Simple proxy: Depth / Noise * sqrt(N_transit_points)
    # We estimate noise as the standard deviation of the flux
    noise_est = np.std(flat_lc.flux.value)
    # Approx points in transit = Total Points * (Duration / Period)
    n_points = len(flat_lc)
    n_transit_points = n_points * (bls.duration / bls.period)
    snr = (bls.depth / noise_est) * np.sqrt(n_transit_points)

    return PlanetEstimate(
        radius_jupiter=r_jup,
        radius_earth=r_earth,
        semi_major_axis_au=a_au,
        equilibrium_temp=t_eq,
        transit_depth_ppm=bls.depth * 1e6,
        snr=snr
    )


def plot_outputs(lc_flat: lk.LightCurve, bls_res: BlsResult, params: PlanetEstimate,
                 star: StarParams, outdir: Path, show: bool):
    """Generates and saves diagnostic plots."""
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Periodogram ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(bls_res.period_grid, bls_res.power_grid, color='black', lw=1)
    ax1.axvline(bls_res.period, color='red', ls='--', alpha=0.6,
                label=f'Peak: {bls_res.period:.3f} d')
    ax1.set_xlabel("Period [days]")
    ax1.set_ylabel("BLS Power")
    ax1.set_title(f"Periodogram: {star.name}")
    ax1.legend()
    fig1.savefig(outdir / "periodogram.png", dpi=150)

    # --- Plot 2: Folded Transit ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Fold data
    folded_lc = lc_flat.fold(period=bls_res.period, epoch_time=bls_res.t0)

    # Scatter plot of all points
    ax2.scatter(folded_lc.time.value, folded_lc.flux.value,
                s=1, alpha=0.3, color='gray', label='Folded Data')

    # Binned average (to see the trend clearly)
    binned_lc = folded_lc.bin(time_bin_size=0.01)
    ax2.scatter(binned_lc.time.value, binned_lc.flux.value,
                s=20, color='blue', alpha=0.8, label='Binned Avg')

    # BLS Model
    # Sort model times for clean line plotting
    sort_idx = np.argsort(bls_res.model_fold_time)
    ax2.plot(bls_res.model_fold_time[sort_idx], bls_res.model_fold_flux[sort_idx],
             color='red', lw=2, label='BLS Model')

    ax2.set_xlim(-0.3 * bls_res.period, 0.3 * bls_res.period)  # Zoom to phase
    ax2.set_xlabel("Phase [days]")
    ax2.set_ylabel("Normalized Flux")
    ax2.set_title(f"Folded Transit: {star.name} (Epoch: {bls_res.t0:.2f})")

    # Add stats box
    stats_text = (
        f"Period: {bls_res.period:.4f} d\n"
        f"Radius: {params.radius_jupiter:.2f} R_Jup\n"
        f"Temp: {params.equilibrium_temp:.0f} K\n"
        f"SNR: {params.snr:.1f}"
    )
    ax2.text(0.02, 0.05, stats_text, transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8), fontsize=9)

    ax2.legend(loc='lower right')
    fig2.savefig(outdir / "folded_transit.png", dpi=150)

    logger.info(f"Plots saved to {outdir}")
    if show:
        plt.show()
    plt.close('all')


# --- MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(
        description="Exoplanet Hunter: CLI for analyzing Kepler/TESS light curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("target", type=str, help="Target Star Name (e.g., 'Kepler-10', 'TIC 25155310')")

    # Mission settings
    parser.add_argument("--mission", type=str, default="auto", choices=["kepler", "tess", "auto"],
                        help="Mission data source.")
    parser.add_argument("--quarter", type=int, help="Kepler Quarter (optional).")
    parser.add_argument("--sector", type=int, help="TESS Sector (optional).")

    # Analysis settings
    parser.add_argument("--period-min", type=float, default=CONSTANTS["DEFAULT_PERIOD_MIN"],
                        help="Min period to search (days).")
    parser.add_argument("--period-max", type=float, default=CONSTANTS["DEFAULT_PERIOD_MAX"],
                        help="Max period to search (days).")
    parser.add_argument("--n-periods", type=int, default=CONSTANTS["DEFAULT_N_PERIODS"],
                        help="Resolution of period search.")
    parser.add_argument("--window-length", type=int, default=401, help="Flattening window length (odd int).")

    # Output settings
    parser.add_argument("--outdir", type=str, default="./outputs", help="Directory to save results.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    parser.add_argument("--json", action="store_true", help="Save metrics to results.json.")

    args = parser.parse_args()

    try:
        # 1. Fetch Star Params
        star_params = fetch_star_params(args.target)

        # 2. Download Light Curve
        kwargs = {}
        if args.quarter: kwargs['quarter'] = args.quarter
        if args.sector: kwargs['sector'] = args.sector

        lc_raw = download_lightcurve(args.target, args.mission, **kwargs)

        # 3. Preprocess
        lc_flat = preprocess_lightcurve(lc_raw, window_length=args.window_length)

        # 4. Run BLS
        bls_results = run_bls(
            lc_flat,
            p_min=args.period_min,
            p_max=args.period_max,
            n_periods=args.n_periods
        )

        # 5. Estimate Physics
        planet_estimates = estimate_planet_params(star_params, bls_results, lc_flat)

        # 6. Report to Console
        print("\n" + "=" * 40)
        print(f"REPORT: {star_params.name}")
        print("=" * 40)
        print(f"Orbital Period:     {bls_results.period:.5f} days")
        print(f"Transit Depth:      {planet_estimates.transit_depth_ppm:.0f} ppm")
        print(f"Est. Radius (Jup):  {planet_estimates.radius_jupiter:.3f} R_J")
        print(f"Est. Radius (Earth):{planet_estimates.radius_earth:.2f} R_E")
        print(f"Est. Temperature:   {planet_estimates.equilibrium_temp:.0f} K")
        print(f"Semi-Major Axis:    {planet_estimates.semi_major_axis_au:.4f} AU")
        print(f"Detection SNR:      {planet_estimates.snr:.1f}")
        print("=" * 40 + "\n")

        # 7. Outputs
        out_path = Path(args.outdir) / args.target.replace(" ", "_")

        if not args.no_plot:
            plot_outputs(lc_flat, bls_results, planet_estimates, star_params, out_path, args.show)

        if args.json:
            json_path = out_path / "results.json"
            data = {
                "star": asdict(star_params),
                "bls": {k: v for k, v in asdict(bls_results).items() if "grid" not in k and "model" not in k},
                "planet": asdict(planet_estimates)
            }
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"JSON results saved to {json_path}")

    except Exception as e:
        logger.critical(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()