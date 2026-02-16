import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Catalogs

# --- PAGE CONFIGURATION (NASA DARK MODE) ---
st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "Mission Control" look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1, h2, h3 {
        color: #58a6ff; 
        font-family: 'Helvetica Neue', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---
def get_star_data(target_name):
    """Fetches star parameters from TIC."""
    try:
        catalog_data = Catalogs.query_object(target_name, radius=0.001, catalog="TIC")
        r_star = catalog_data[0]['rad'] if not np.isnan(catalog_data[0]['rad']) else 1.0
        t_star = catalog_data[0]['Teff'] if not np.isnan(catalog_data[0]['Teff']) else 5777.0
        return r_star, t_star
    except:
        return 1.0, 5777.0


def analyze_star(target_name):
    """Downloads data and performs BLS search."""
    # 1. Get Star Data
    r_star, t_star = get_star_data(target_name)

    # 2. Download Light Curve
    try:
        search = lk.search_lightcurve(target_name, author='Kepler', quarter=10)
        if len(search) == 0:
            search = lk.search_lightcurve(target_name, author='SPOC', sector=1)  # TESS fallback

        if len(search) == 0:
            st.error(f"No data found for {target_name}. Try 'Kepler-10' or 'TIC 25155310'.")
            return None

        # Download and Clean
        lc = search[0].download().remove_nans().normalize()
        flat_lc = lc.flatten(window_length=401)

        # 3. BLS Search
        periodogram = flat_lc.to_periodogram(method='bls', period=np.linspace(0.5, 15, 10000))
        best_period = periodogram.period_at_max_power.value
        best_t0 = periodogram.transit_time_at_max_power.value
        best_depth = periodogram.depth_at_max_power.value

        # 4. Calculate Physics
        r_planet = (r_star * 9.73) * np.sqrt(best_depth)  # In Jupiter Radii

        # Temp calc
        a_au = ((best_period / 365.25) ** 2 * 1.0) ** (1 / 3)  # Assuming Sun-mass
        t_planet = t_star * np.sqrt(r_star / (2 * (a_au * 215.032)))

        return {
            'period': best_period,
            't0': best_t0,
            'r_planet': r_planet,
            't_planet': t_planet,
            'lc_raw': lc,  # Added raw light curve
            'lc_flat': flat_lc,  # Added flattened light curve
            'periodogram': periodogram
        }
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# --- APP LAYOUT ---
st.title("🪐 Exo-Hunter: Mission Dashboard")
st.markdown("Analyze Kepler & TESS data to discover exoplanets in real-time.")

# Sidebar
with st.sidebar:
    st.header("🔭 Target Selection")
    target = st.text_input("Enter Star ID:", value="Kepler-8")
    if st.button("Initialize Scan"):
        with st.spinner(f"Acquiring signal from {target}..."):
            st.session_state['results'] = analyze_star(target)
            st.session_state['target'] = target

# Main Dashboard
if 'results' in st.session_state and st.session_state['results']:
    results = st.session_state['results']

    # 1. TOP METRICS ROW
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Orbital Period", f"{results['period']:.4f} days")
    col2.metric("Planet Radius", f"{results['r_planet']:.2f} x Jupiter")
    col3.metric("Surface Temp", f"{results['t_planet']:.0f} K")

    # Habitability Status
    status = "🏜️ Too Hot" if results['t_planet'] > 400 else "❄️ Too Cold"
    if 260 < results['t_planet'] < 390: status = "🌍 Habitable Zone!"
    col4.metric("Status", status)

    st.divider()

    # 2. RAW LIGHT CURVE (New Section)
    st.subheader("1. Raw Data Telemetry (Time Series)")
    with st.expander("Show Graph Details", expanded=True):
        col_raw_L, col_raw_R = st.columns([3, 1])

        with col_raw_L:
            fig_raw, ax_raw = plt.subplots(figsize=(10, 4))
            fig_raw.patch.set_facecolor('#0e1117')
            ax_raw.set_facecolor('#0e1117')
            ax_raw.tick_params(colors='white')
            ax_raw.xaxis.label.set_color('white')
            ax_raw.yaxis.label.set_color('white')
            for spine in ax_raw.spines.values(): spine.set_color('white')

            # Plot raw flux
            ax_raw.plot(results['lc_raw'].time.value, results['lc_raw'].flux.value,
                        color='white', alpha=0.3, lw=0.5, label='Raw Flux')
            # Plot flattened trend
            ax_raw.plot(results['lc_flat'].time.value, results['lc_flat'].flux.value,
                        color='#58a6ff', alpha=0.8, lw=1, label='Processed')

            ax_raw.set_xlabel("Time (Barycentric Julian Date)")
            ax_raw.set_ylabel("Normalized Flux")
            ax_raw.legend(facecolor='#161b22', labelcolor='white')
            st.pyplot(fig_raw)

        with col_raw_R:
            st.markdown("""
            **Observation Timeline**

            This graph shows the brightness of the star over several weeks.

            * **White Noise:** The raw data, including starspots and rotation.
            * **Blue Line:** The cleaned data.
            * **The Spikes:** If you look closely at the Blue line, those sharp downward spikes are the transits!
            """)

    # 3. FOLDED TRANSIT
    st.subheader("2. Transit Detection (Folded)")
    col_fold_L, col_fold_R = st.columns([3, 1])

    with col_fold_L:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values(): spine.set_color('white')

        # Fold and Plot
        folded = results['lc_flat'].fold(period=results['period'], epoch_time=results['t0'])
        folded.scatter(ax=ax, c='#58a6ff', s=10, alpha=0.5, label='Sensor Data')

        # Model
        model = results['periodogram'].get_transit_model(
            period=results['period'], transit_time=results['t0'],
            duration=results['periodogram'].duration_at_max_power.value)
        model.fold(results['period'], results['t0']).plot(ax=ax, c='#ff4b4b', lw=3, label='Computer Model')

        ax.set_xlim(-0.5, 0.5)
        ax.legend(facecolor='#161b22', labelcolor='white')
        st.pyplot(fig)

    with col_fold_R:
        st.markdown("""
        **Confirmed Signal**

        We stack the data on top of itself based on the orbital period.

        * **The "U" Shape:** This is the shadow of the planet.
        * **Depth:** Tells us the size of the planet (deeper = bigger).
        * **Width:** Tells us the speed of the transit.
        """)

    st.divider()

    # 4. PERIODOGRAM
    st.subheader("3. Period Search (BLS)")
    with st.expander("Click to view Frequency Analysis"):
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        ax2.tick_params(colors='white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        for spine in ax2.spines.values(): spine.set_color('white')

        results['periodogram'].plot(ax=ax2, lw=0, marker='o', markersize=2, color='yellow')
        ax2.axvline(results['period'], color='#ff4b4b', ls='--', label=f'Peak: {results["period"]:.2f} d')
        ax2.legend(facecolor='#161b22', labelcolor='white')
        st.pyplot(fig2)