import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Catalogs

# --- PHYSICS CONSTANTS ---
AU_TO_SOLAR_RADII = 215.032  # 1 Astronomical Unit = ~215 Solar Radii
SOLAR_MASS_IN_KG = 1.989e30
GRAVITATIONAL_CONSTANT = 6.674e-11


def get_star_params(target_name):
    """
    Fetches Star Radius, Mass, and Temperature from the TESS Input Catalog (TIC).
    """
    print(f"\n--- 1. FETCHING STAR DATA FOR {target_name.upper()} ---")
    try:
        # Query MAST for the star's properties
        catalog_data = Catalogs.query_object(target_name, radius=0.001, catalog="TIC")

        # Extract the parameters (with fallback values if missing)
        r_star = catalog_data[0]['rad'] if not np.isnan(catalog_data[0]['rad']) else 1.0
        m_star = catalog_data[0]['mass'] if not np.isnan(catalog_data[0]['mass']) else 1.0
        t_star = catalog_data[0]['Teff'] if not np.isnan(catalog_data[0]['Teff']) else 5777.0

        print(f"   >> STAR RADIUS:      {r_star:.3f} Solar Radii")
        print(f"   >> STAR MASS:        {m_star:.3f} Solar Mass")
        print(f"   >> STAR TEMP (Teff): {t_star:.0f} K")

        return r_star, m_star, t_star
    except Exception as e:
        print(f"   !! WARNING: Could not resolve star. Using Sun-like defaults. ({e})")
        return 1.0, 1.0, 5777.0


def calculate_planet_temp(period_days, r_star, m_star, t_star):
    """
    Calculates the semi-major axis (distance) and Equilibrium Temperature.
    """
    # 1. Kepler's 3rd Law: Calculate Distance (Semi-Major Axis) in AU
    # Formula: a (AU) = cubic_root( (Period_years^2) * Mass_solar )
    period_years = period_days / 365.25
    a_au = ((period_years ** 2) * m_star) ** (1 / 3)

    # 2. Convert Distance to Solar Radii (to match R_star units)
    a_radii = a_au * AU_TO_SOLAR_RADII

    # 3. Calculate Temperature (Equilibrium)
    # T_eq = T_star * sqrt( R_star / 2a )
    # This assumes the planet is a blackbody (Albedo=0) and redistributes heat evenly.
    t_planet = t_star * np.sqrt(r_star / (2 * a_radii))

    return a_au, t_planet


def analyze_system(target_name):
    # STEP 1: Get Star Physics
    r_star, m_star, t_star = get_star_params(target_name)

    # STEP 2: Download Light Curve
    print(f"\n--- 2. DOWNLOADING SATELLITE DATA ---")
    # Using 'Kepler' data. You can change this to 'TESS' for newer targets.
    search = lk.search_lightcurve(target_name, author='Kepler', quarter=10)

    if len(search) == 0:
        print(f"   !! No Kepler data found for {target_name}. Trying TESS...")
        search = lk.search_lightcurve(target_name, author='SPOC')  # SPOC is the official TESS pipeline
        if len(search) == 0:
            print("   !! No data found at all. Check the name!")
            return

    lc = search[0].download().remove_nans().normalize()
    flat_lc = lc.flatten(window_length=401)

    # STEP 3: Hunt for Period (BLS)
    print(f"\n--- 3. SCANNING FOR TRANSITS ---")
    # Scans periods from 0.5 to 20 days
    periodogram = flat_lc.to_periodogram(method='bls', period=np.linspace(0.5, 20, 50000))

    best_period = periodogram.period_at_max_power.value
    best_t0 = periodogram.transit_time_at_max_power.value
    best_depth = periodogram.depth_at_max_power.value

    print(f"   >> ORBITAL PERIOD:   {best_period:.5f} days")

    # STEP 4: Calculate Planet Properties
    # Radius
    r_star_jup = r_star * 9.73  # Convert Sun radii to Jupiter radii
    r_planet = r_star_jup * np.sqrt(best_depth)

    # Temperature & Distance
    dist_au, t_planet = calculate_planet_temp(best_period, r_star, m_star, t_star)

    print(f"\n--- 4. PLANET REPORT: {target_name.upper()} b ---")
    print(f"   >> RADIUS:           {r_planet:.2f} Jupiter Radii")
    print(f"   >> DISTANCE:         {dist_au:.4f} AU")
    print(f"   >> TEMPERATURE:      {t_planet:.0f} Kelvin ({t_planet - 273.15:.0f} Celsius)")

    # Habitability Check
    if 260 < t_planet < 390:
        print("   >> STATUS:           POTENTIALLY HABITABLE ZONE (Liquid Water Possible)")
    elif t_planet > 1000:
        print("   >> STATUS:           INFERNO (Molten Surface)")
    else:
        print("   >> STATUS:           TOO COLD / TOO HOT")

    # STEP 5: Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Periodogram
    periodogram.plot(ax=ax1, lw=0, marker='o', markersize=2, color='purple')
    ax1.set_title(f"Period Scan for {target_name}")

    # Plot 2: Folded Transit
    folded_lc = flat_lc.fold(period=best_period, epoch_time=best_t0)
    folded_lc.scatter(ax=ax2, color='blue', alpha=0.3, s=10, label='Data')

    # Model
    model = periodogram.get_transit_model(period=best_period, transit_time=best_t0,
                                          duration=periodogram.duration_at_max_power.value)
    model.fold(best_period, best_t0).plot(ax=ax2, color='red', lw=2, label='Model')

    ax2.set_title(f"Transit View (Temp: {t_planet:.0f} K)")
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ask user for input
    target = input("Enter a star name (e.g., Kepler-8, Kepler-12, TESS): ")
    analyze_system(target)