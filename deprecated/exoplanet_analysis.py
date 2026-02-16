import lightkurve as lk
import matplotlib.pyplot as plt


def analyze_transit(target_name, orbital_period, t0):
    """
    Downloads, cleans, and folds a light curve to show an exoplanet transit.

    Parameters:
    - target_name: The ID of the star (e.g., "Kepler-8")
    - orbital_period: The time it takes the planet to orbit (in days)
    - t0: The time of the first recorded transit (to align the fold)
    """
    print(f"1. Searching for data on {target_name}...")
    # Search for data. We limit to 'Kepler' author to get the official science data.
    search_result = lk.search_lightcurve(target_name, author='Kepler', quarter=10)

    if len(search_result) == 0:
        print("No data found!")
        return

    # Download the data
    print("2. Downloading light curve...")
    lc = search_result.download()

    # Remove NaN values and Normalize (make the average flux 1.0)
    lc = lc.remove_nans().normalize()

    # 'Flatten' the light curve.
    # This removes the variability of the star itself (spots, rotation),
    # leaving mostly just the sharp dips from the planet.
    flat_lc = lc.flatten(window_length=401)

    print("3. Folding data...")
    # 'Folding' stacks the time series on top of itself based on the period.
    # This combines many transits into one clear signal.
    folded_lc = flat_lc.fold(period=orbital_period, epoch_time=t0)

    # Plotting
    print("4. Generating plot...")

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: The full timeline (raw-ish data)
    lc.plot(ax=ax1, color='gray', alpha=0.5, label='Original')
    flat_lc.plot(ax=ax1, color='black', label='Flattened')
    ax1.set_title(f"Light Curve: {target_name}")

    # Plot 2: The folded transit (the planet signal)
    # We use 'scatter' to show individual data points
    folded_lc.scatter(ax=ax2, color='blue', alpha=0.5, label='Folded Data')
    # Bin the data to see the average trend line (the red line)
    folded_lc.bin(time_bin_size=0.01).plot(ax=ax2, color='red', lw=2, label='Binned Model')
    ax2.set_title("Phase Folded Transit (The Planet Dip)")
    ax2.set_xlim(-0.2, 0.2)  # Zoom in on the center of the transit

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Kepler-8b known parameters
    # Period: ~3.5225 days
    # Transit Epoch (t0): 2455003.497
    analyze_transit("Kepler-8", orbital_period=3.5225, t0=2455003.497)