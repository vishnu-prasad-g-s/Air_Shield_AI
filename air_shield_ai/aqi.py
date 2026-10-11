"""PM2.5-to-AQI utilities using Indian AQI PM2.5 breakpoints.

This module is an Air Shield AI addition. It does not change the original
forecasting model or its pretrained weights.
"""
import numpy as np

# PM2.5 concentration (ug/m3) and corresponding India AQI bands.
PM25_BREAKPOINTS = np.array([0, 30, 60, 90, 120, 250, np.inf], dtype=float)
AQI_BREAKPOINTS = np.array([0, 50, 100, 200, 300, 400, 500], dtype=float)
LABELS = ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe")


def pm25_to_aqi(pm25):
    """Convert PM2.5 concentration(s) in ug/m3 to an AQI value.

    Values above the final breakpoint are clipped to 500, matching the
    bounded AQI scale used by this project.
    """
    x = np.asarray(pm25, dtype=float)
    x = np.maximum(x, 0.0)
    out = np.empty_like(x)
    for lo, hi, a_lo, a_hi in zip(PM25_BREAKPOINTS[:-1], PM25_BREAKPOINTS[1:],
                                  AQI_BREAKPOINTS[:-1], AQI_BREAKPOINTS[1:]):
        mask = (x >= lo) & (x <= hi)
        if np.any(mask):
            if np.isinf(hi):
                out[mask] = a_hi
            else:
                out[mask] = ((a_hi - a_lo) / (hi - lo)) * (x[mask] - lo) + a_lo
    out = np.clip(out, 0, 500)
    return float(out) if out.ndim == 0 else out


def classify_aqi(aqi):
    """Return the AQI category for scalar or array AQI values."""
    x = np.asarray(aqi, dtype=float)
    idx = np.select([x <= 50, x <= 100, x <= 200, x <= 300, x <= 400],
                    [0, 1, 2, 3, 4], default=5)
    labels = np.asarray(LABELS, dtype=object)[idx]
    return labels.item() if labels.ndim == 0 else labels
