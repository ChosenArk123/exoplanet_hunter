import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Catalogs


def get_star_radius(target_name):
    """
    Queries the TESS Input Catalog (TIC) via MAST to find the star's true radius.
    """
    print(f"--- FETCHING STELLAR PARAMETERS FOR {target_name} ---")
    try:
        # Search the TIC catalog (which includes Kepler & TESS stars)
        # We search within a tiny radius (0.001 deg) to get the exact star
        catalog_data = Catalogs.query_object(target_name, radius=0.001, catalog="TIC")

        # The 'rad' column contains the radius in Solar Radii
        r_star = catalog_data[0]['rad']
        print(f"   >> STAR FOUND: {target_name}")
        print(f"   >> STAR RADIUS: {r_star:.3f} Solar Radii")
        return r_star
    except Exception as e:
        print(f"   !! WARNING: Could not find star radius. Defaulting to 1.0 (Sun-like). Error: {e}")
        return 1.0


def master_analysis(target_name):
    # 1. GET STAR PHYSICS
    r_star = get_star_radius(target_name)

    # 2. DOWNLOAD DATA
    print(f"\n--- DOWNLOADING LIGHT CURVE ---")
    # We download all available data for better accuracy, but limit to 1 quarter for speed in this demo
    # Change quarter=None to download EVERYTHING (warning: takes longer/more RAM)
    search = lk.search_lightcurve(target_name, author='Kepler', quarter=10)

    if len(search) == 0:
        print("No data found.")
        return

    lc = search.download().remove_nans().normalize()

    # 3. CLEANING (FLATTENING)
    flat_lc = lc.flatten(window_length=401)

    # 4. BLIND SEARCH (BLS)
    print(f"\n--- HUNTING FOR PLANETS ---")
    # Scan periods from 0.5 to 20 days
    periodogram = flat_lc.to_periodogram(method='bls', period=np.linspace(0.5, 20, 50000))

    best_period = periodogram.period_at_max_power.value
    best_t0 = periodogram.transit_time_at_max_power.value
    best_depth = periodogram.depth_at_max_power.value

    print(f"   >> PERIOD DETECTED: {best_period:.5f} days")
    print(f"   >> TRANSIT DEPTH: {best_depth:.5f}")

    # 5. CALCULATE PLANET SIZE (The Physics Part)
    # Formula: R_planet = R_star * sqrt(Depth)
    # We convert R_star (Solar Radii) to Jupiter Radii for easier visualization
    # 1 Sun Radius = 9.73 Jupiter Radii

    r_star_jup = r_star * 9.73
    r_planet = r_star_jup * np.sqrt(best_depth)

    print(f"   >> CALCULATED PLANET RADIUS: {r_planet:.2f} Jupiter Radii")

    # 6. PLOTTING
    print("\n--- GENERATING REPORT ---")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot Periodogram
    periodogram.plot(ax=ax1, lw=0, marker='o', markersize=2, color='purple')
    ax1.set_title(f"Periodogram for {target_name} (R_star = {r_star:.2f} R_sun)")
    ax1.axvline(best_period, color='red', ls='--', label=f'Peak: {best_period:.2f} d')
    ax1.legend()

    # Plot Folded Transit
    folded_lc = flat_lc.fold(period=best_period, epoch_time=best_t0)
    folded_lc.scatter(ax=ax2, color='blue', alpha=0.3, s=10, label='Observation')

    # Add the model
    planet_model = periodogram.get_transit_model(period=best_period,
                                                 transit_time=best_t0,
                                                 duration=periodogram.duration_at_max_power.value)
    planet_model.fold(best_period, best_t0).plot(ax=ax2, color='red', lw=2, label='Model')

    ax2.set_title(f"Transit: {target_name} b (Radius: {r_planet:.2f} R_Jup)")
    ax2.set_xlim(-0.5, 0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Try different stars to see the radius calculation adapt!
    # Kepler-8: A large star (approx 1.48 R_sun)
    # Kepler-10: A smaller star (approx 1.06 R_sun)
    master_analysis("Kepler-8")