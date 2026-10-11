"""Risk/alert logic added for Air Shield AI."""
import numpy as np
from .aqi import pm25_to_aqi, classify_aqi


def build_alerts(pm25):
    """Create a compact risk report for a PM2.5 forecast array.

    Returns a dictionary containing AQI values, categories and a single
    highest-severity status. This is an informational alert layer, not medical
    advice.
    """
    values = np.asarray(pm25, dtype=float)
    aqi = pm25_to_aqi(values)
    categories = classify_aqi(aqi)
    severity = {name: i for i, name in enumerate(
        ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"))}
    flat = np.asarray(categories, dtype=object).reshape(-1)
    highest = max(flat, key=lambda x: severity[str(x)]) if flat.size else "Good"
    return {
        "max_pm25": float(np.nanmax(values)) if values.size else float("nan"),
        "mean_pm25": float(np.nanmean(values)) if values.size else float("nan"),
        "max_aqi": float(np.nanmax(aqi)) if np.size(aqi) else float("nan"),
        "highest_category": str(highest),
        "categories": categories,
        "aqi": aqi,
    }
