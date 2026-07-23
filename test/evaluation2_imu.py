#!/usr/bin/env python3

import argparse
import bisect
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = "/home/jasonxue/splg_lo_origin/splg_fusion"
DEFAULT_GNSS_CSV = os.path.join(BASE_DIR, "result", "extract_gnss_position.csv")
DEFAULT_VISION_CSV = os.path.join(BASE_DIR, "result", "vision_position.csv")
DEFAULT_IMU_CSV = os.path.join(BASE_DIR, "result", "imu_position.csv")
DEFAULT_FUSION_CSV = os.path.join(BASE_DIR, "result", "fusion_position.csv")
DEFAULT_OUTPUT_PNG = os.path.join(BASE_DIR, "result", "evaluation2_imu_error_plot.png")
DEFAULT_DISTANCE_OUTPUT_PNG = os.path.join(BASE_DIR, "result", "evaluation2_imu_distance_plot.png")
EARTH_RADIUS_M = 6378137.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare vision, IMU-only, and fusion positions against GNSS."
    )
    parser.add_argument("--gnss_csv", default=DEFAULT_GNSS_CSV)
    parser.add_argument("--vision_csv", default=DEFAULT_VISION_CSV)
    parser.add_argument("--imu_csv", default=DEFAULT_IMU_CSV)
    parser.add_argument("--fusion_csv", default=DEFAULT_FUSION_CSV)
    parser.add_argument("--output_png", default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--distance_output_png", default=DEFAULT_DISTANCE_OUTPUT_PNG)
    parser.add_argument("--thresholds", default="10,30,50,80,100")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_float(text):
    if text is None:
        return math.nan
    value = str(text).strip()
    if not value or value.lower() == "nan":
        return math.nan
    return float(value)


def parse_thresholds(text):
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            value = float(token)
            if value <= 0.0:
                raise ValueError("Threshold must be positive: {}".format(token))
            values.append(value)
    if not values:
        raise ValueError("At least one threshold is required.")
    return sorted(set(values))


def read_csv_points(path, stamp_key, lon_key, lat_key, available_key=None):
    if not os.path.isfile(path):
        raise FileNotFoundError("CSV not found: {}".format(path))
    points = []
    with open(path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        required = [stamp_key, lon_key, lat_key]
        if available_key:
            required.append(available_key)
        missing = [key for key in required if key not in fieldnames]
        if missing:
            raise KeyError("Missing columns {} in {}".format(", ".join(missing), path))
        for row in reader:
            if available_key and parse_float(row.get(available_key)) != 1.0:
                continue
            stamp = parse_float(row.get(stamp_key))
            lon = parse_float(row.get(lon_key))
            lat = parse_float(row.get(lat_key))
            if math.isfinite(stamp) and math.isfinite(lon) and math.isfinite(lat):
                points.append((stamp, lon, lat))
    points.sort(key=lambda point: point[0])
    if not points:
        raise ValueError("No valid rows found in {}".format(path))
    return points


def interpolate_gnss(points, stamp):
    stamps = [point[0] for point in points]
    index = bisect.bisect_left(stamps, stamp)
    if index < len(points) and abs(points[index][0] - stamp) < 1e-9:
        return points[index][1], points[index][2]
    if index == 0 or index >= len(points):
        return None
    t0, lon0, lat0 = points[index - 1]
    t1, lon1, lat1 = points[index]
    if t1 <= t0:
        return None
    ratio = (stamp - t0) / (t1 - t0)
    return lon0 + ratio * (lon1 - lon0), lat0 + ratio * (lat1 - lat0)


def error_to_meters(sample_lon, sample_lat, ref_lon, ref_lat):
    lat_ref_rad = math.radians((sample_lat + ref_lat) * 0.5)
    east_m = math.radians(sample_lon - ref_lon) * EARTH_RADIUS_M * math.cos(lat_ref_rad)
    north_m = math.radians(sample_lat - ref_lat) * EARTH_RADIUS_M
    return east_m, north_m, math.hypot(east_m, north_m)


def build_series(name, data_points, gnss_points, start_stamp):
    series = {"name": name, "t": [], "east_m": [], "north_m": [], "distance_m": []}
    for stamp, lon, lat in data_points:
        ref = interpolate_gnss(gnss_points, stamp)
        if ref is None:
            continue
        east_m, north_m, distance_m = error_to_meters(lon, lat, ref[0], ref[1])
        series["t"].append(stamp - start_stamp)
        series["east_m"].append(east_m)
        series["north_m"].append(north_m)
        series["distance_m"].append(distance_m)
    if not series["t"]:
        raise ValueError("No valid overlap samples for {}".format(name))
    return series


def compute_stats(values):
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        "max_abs": max(abs(value) for value in values),
    }


def threshold_stats(values, thresholds):
    return [(threshold, sum(value < threshold for value in values)) for threshold in thresholds]


def plot_error_series(series_list, output_png, start_stamp, end_stamp):
    ensure_parent_dir(output_png)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels = [("east_m", "east error (m)"), ("north_m", "north error (m)"), ("distance_m", "horizontal error (m)")]
    for axis, (key, ylabel) in zip(axes, labels):
        for series in series_list:
            axis.plot(series["t"], series[key], label=series["name"], linewidth=1.3)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("time in shared interval (s)")
    fig.suptitle("Localization error vs GNSS [{:.3f}, {:.3f}]".format(start_stamp, end_stamp), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_distance_series(series_list, output_png, start_stamp, end_stamp):
    ensure_parent_dir(output_png)
    fig, axis = plt.subplots(1, 1, figsize=(14, 5.5))
    for series in series_list:
        axis.plot(series["t"], series["distance_m"], label=series["name"], linewidth=1.3)
    axis.set_xlabel("time in shared interval (s)")
    axis.set_ylabel("distance error (m)")
    axis.set_title("Absolute distance error vs GNSS [{:.3f}, {:.3f}]".format(start_stamp, end_stamp))
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    try:
        thresholds = parse_thresholds(args.thresholds)
        gnss = read_csv_points(args.gnss_csv, "fix_stamp_sec", "longitude", "latitude")
        vision = read_csv_points(args.vision_csv, "image_stamp_sec", "camera_lon", "camera_lat")
        imu = read_csv_points(args.imu_csv, "fix_stamp_sec", "imu_lon", "imu_lat", "imu_available")
        fusion = read_csv_points(args.fusion_csv, "fix_stamp_sec", "pred_lon", "pred_lat", "pred_available")

        start_stamp = max(points[0][0] for points in (gnss, vision, imu, fusion))
        end_stamp = min(points[-1][0] for points in (gnss, vision, imu, fusion))
        if start_stamp >= end_stamp:
            raise ValueError("The four files do not have a shared timestamp interval.")
        gnss_shared = [point for point in gnss if start_stamp <= point[0] <= end_stamp]
        series_list = [
            build_series("vision", [p for p in vision if start_stamp <= p[0] <= end_stamp], gnss_shared, start_stamp),
            build_series("imu", [p for p in imu if start_stamp <= p[0] <= end_stamp], gnss_shared, start_stamp),
            build_series("fusion", [p for p in fusion if start_stamp <= p[0] <= end_stamp], gnss_shared, start_stamp),
        ]

        plot_error_series(series_list, args.output_png, start_stamp, end_stamp)
        plot_distance_series(series_list, args.distance_output_png, start_stamp, end_stamp)
        print("Shared stamp interval: [{:.6f}, {:.6f}]".format(start_stamp, end_stamp))
        print("Saved error plot to {}".format(args.output_png))
        print("Saved distance plot to {}".format(args.distance_output_png))
        for series in series_list:
            stats = compute_stats(series["distance_m"])
            print("{} distance error: count={count}, mean={mean:.3f} m, rmse={rmse:.3f} m, max_abs={max_abs:.3f} m".format(series["name"], **stats))
            print("{} threshold statistics:".format(series["name"]))
            for threshold, count in threshold_stats(series["distance_m"], thresholds):
                print("  error < {:.3f} m : {:.2f}% ({}/{})".format(threshold, 100.0 * count / stats["count"], count, stats["count"]))
    except Exception as exc:
        print("Failed to evaluate localization results: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
