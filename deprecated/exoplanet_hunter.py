import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np


def find_and_characterize(target_name):
    print(f"--- ANALYZING {target_name} ---")

    # 1. DOWNLOAD & CLEAN
    print("1. Downloading data...")
    # We download a specific quarter to keep it fast, but real science uses all quarters
    search = lk.search_lightcurve(target_name, author='Kepler', quarter=10)
    if len(search) == 0:
        print("No data found.")
        return

    lc = search.download().remove_nans().normalize()

    # Flatten: Remove stellar variability (spots/rotation) so only transits remain
    flat_lc = lc.flatten(window_length=401)

    # 2. BLIND SEARCH (The BLS Periodogram)
    print("2. Running Box Least Squares (BLS) search...")

    # Create a periodogram: check periods between 1 and 10 days
    periodogram = flat_lc.to_periodogram(method='bls', period=np.linspace(1, 10, 10000))

    # Extract the period with the highest "power" (best fit)
    best_period = periodogram.period_at_max_power.value
    best_t0 = periodogram.transit_time_at_max_power.value
    best_dur = periodogram.duration_at_max_power.value

    print(f"   >> DETECTED PERIOD: {best_period:.4f} days")
    print(f"   >> TRANSIT EPOCH: {best_t0:.4f}")

    # 3. CALCULATE PLANET RADIUS
    # We need the transit depth. We can get a rough estimate from the BLS model.
    # Depth ≈ (Radius_Planet / Radius_Star)^2
    # Therefore: Radius_Planet ≈ sqrt(Depth) * Radius_Star

    transit_depth = periodogram.depth_at_max_power.value
    # Approximate radius in Jupiter radii (assuming star is Sun-sized for simplicity)
    # 1 Solar Radius approx = 9.73 Jupiter Radii
    r_star_in_jupiters = 9.73
    r_planet_approx = np.sqrt(transit_depth) * r_star_in_jupiters

    print(f"   >> TRANSIT DEPTH: {transit_depth:.5f} (flux fraction)")
    print(f"   >> EST. RADIUS: {r_planet_approx:.2f} Jupiter Radii")

    # 4. PLOTTING
    print("3. Generating dashboard...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: The Periodogram (The "Search Result")
    periodogram.plot(ax=ax1, lw=0, marker='o', markersize=2, color='purple')
    ax1.set_title(f"Periodogram: Peaks indicate likely planets")
    ax1.axvline(best_period, color='red', linestyle='--', label=f'Peak: {best_period:.2f} d')
    ax1.legend()

    # Plot 2: The Folded Light Curve (The Visual Proof)
    folded_lc = flat_lc.fold(period=best_period, epoch_time=best_t0)
    folded_lc.scatter(ax=ax2, color='blue', alpha=0.3, s=10, label='Data')

    # Plot the BLS model over it (the red line)
    # We use the model derived from the periodogram
    planet_model = periodogram.get_transit_model(period=best_period,
                                                 transit_time=best_t0,
                                                 duration=best_dur)
    planet_model.fold(best_period, best_t0).plot(ax=ax2, color='red', lw=2, label='BLS Model')

    ax2.set_title(f"Folded Transit (Period: {best_period:.4f} days)")
    ax2.set_xlim(-0.3, 0.3)  # Zoom on the transit

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Try with "Kepler-8" again to see if it finds the same 3.52 day period!
    # Or try "Kepler-10" (Period ~0.8 days) or "Kepler-1" (TrES-2, Period ~2.47 days)
    find_and_characterize("Kepler-8")