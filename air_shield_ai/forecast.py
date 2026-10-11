"""Utilities for inspecting saved PM2.5 forecast arrays."""
from pathlib import Path
import numpy as np
from .alerts import build_alerts


def load_prediction_array(path):
    """Load an .npy forecast file and validate it is numeric."""
    arr = np.load(Path(path), allow_pickle=False)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("Prediction array must contain numeric values.")
    return arr.astype(np.float32, copy=False)


def summarize_forecast(predictions):
    """Summarize a forecast array and derive an Air Shield AI risk report.

    Expected shapes include (N, H, W, horizon) or (N, horizon, H, W).
    The function does not assume one layout for the summary statistics.
    """
    x = np.asarray(predictions, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError("Expected a 4-D forecast array.")
    # The source notebook writes (N, H, W, 16); use the last axis as horizon
    # when it looks like a short forecast dimension.
    horizon_axis = 3 if x.shape[3] <= 48 else 1
    horizon = x.shape[horizon_axis]
    spatial_axes = tuple(i for i in range(4) if i not in (0, horizon_axis))
    per_step = np.nanmean(x, axis=(0,) + spatial_axes)
    overall = build_alerts(x)
    return {
        "samples": int(x.shape[0]),
        "horizon": int(horizon),
        "shape": tuple(int(v) for v in x.shape),
        "mean_by_step": per_step,
        "overall": overall,
    }
