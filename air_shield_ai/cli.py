"""Command-line summary for Air Shield AI."""
import argparse
from .forecast import load_prediction_array, summarize_forecast

def main():
    p = argparse.ArgumentParser(description="Summarize an Air Shield AI PM2.5 forecast .npy file")
    p.add_argument("prediction_file")
    args = p.parse_args()
    s = summarize_forecast(load_prediction_array(args.prediction_file))
    o = s["overall"]
    print(f"Shape: {s['shape']}")
    print(f"Samples: {s['samples']}")
    print(f"Horizon: {s['horizon']} steps")
    print(f"Mean PM2.5: {o['mean_pm25']:.3f} ug/m3")
    print(f"Maximum PM2.5: {o['max_pm25']:.3f} ug/m3")
    print(f"Maximum AQI: {o['max_aqi']:.1f}")
    print(f"Highest risk category: {o['highest_category']}")

if __name__ == "__main__":
    main()
