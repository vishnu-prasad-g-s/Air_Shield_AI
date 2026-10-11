"""Optional Streamlit dashboard for Air Shield AI forecast arrays."""
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from air_shield_ai.forecast import load_prediction_array, summarize_forecast
from air_shield_ai.aqi import pm25_to_aqi

st.set_page_config(page_title="Air Shield AI", layout="wide")
st.title("Air Shield AI")
st.caption("PM2.5 forecast intelligence and risk visualization")

uploaded = st.file_uploader("Upload a NumPy forecast file (.npy)", type=["npy"])
path = st.text_input("Or enter a local .npy path", value="")

if uploaded is None and not path:
    st.info("Provide a forecast array to begin.")
    st.stop()

if uploaded is not None:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        f.write(uploaded.getbuffer())
        source = f.name
else:
    source = path

try:
    pred = load_prediction_array(source)
    summary = summarize_forecast(pred)
except Exception as exc:
    st.error(f"Could not load forecast: {exc}")
    st.stop()

o = summary["overall"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast samples", summary["samples"])
c2.metric("Forecast horizon", f"{summary['horizon']} steps")
c3.metric("Mean PM2.5", f"{o['mean_pm25']:.2f} µg/m³")
c4.metric("Highest risk", o["highest_category"])

st.subheader("Mean PM2.5 by forecast step")
fig, ax = plt.subplots()
ax.plot(np.arange(1, summary["horizon"] + 1), summary["mean_by_step"], marker="o")
ax.set_xlabel("Forecast step")
ax.set_ylabel("Mean PM2.5 (µg/m³)")
ax.grid(True, alpha=0.25)
st.pyplot(fig)
plt.close(fig)

# Show first sample/last selected horizon as a simple spatial map when layout is N,H,W,T.
if pred.ndim == 4 and pred.shape[-1] <= 48:
    step = st.slider("Spatial forecast step", 1, pred.shape[-1], 1) - 1
    sample = st.number_input("Sample index", 0, pred.shape[0] - 1, 0)
    field = pred[int(sample), :, :, step]
    st.subheader(f"PM2.5 spatial forecast — sample {sample}, step {step + 1}")
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(field, origin="lower")
    fig2.colorbar(im, ax=ax2, label="PM2.5 (µg/m³)")
    ax2.set_xlabel("Grid X")
    ax2.set_ylabel("Grid Y")
    st.pyplot(fig2)
    plt.close(fig2)
    st.write(f"Approx. AQI at spatial mean: {pm25_to_aqi(float(np.nanmean(field))):.0f}")

st.warning("This dashboard provides model-based estimates and is not a substitute for official air-quality monitoring or health guidance.")
