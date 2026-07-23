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
DEFAULT_FUSION_CSV = os.path.join(BASE_DIR, "result", "fusion_position.csv")
DEFAULT_OUTPUT_PNG = os.path.join(BASE_DIR, "result", "evaluation2_error_plot.png")
DEFAULT_DISTANCE_OUTPUT_PNG = os.path.join(BASE_DIR, "result", "evaluation2_distance_plot.png")

EARTH_RADIUS_M = 6378137.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare vision_position.csv and fusion_position.csv against "
            "extract_gnss_position.csv inside the shared timestamp interval."
        )
    )
    parser.add_argument("--gnss_csv", default=DEFAULT_GNSS_CSV, help="Path to extract_gnss_position.csv")
    parser.add_argument("--vision_csv", default=DEFAULT_VISION_CSV, help="Path to vision_position.csv")
    parser.add_argument("--fusion_csv", default=DEFAULT_FUSION_CSV, help="Path to fusion_position.csv")
    parser.add_argument("--output_png", default=DEFAULT_OUTPUT_PNG, help="Path to the signed error plot")
    parser.add_argument(
        "--distance_output_png",
        default=DEFAULT_DISTANCE_OUTPUT_PNG,
        help="Path to the absolute distance error plot",
    )
    parser.add_argument(
        "--thresholds",
        default="10,30,50,80,100",
        help="Comma separated distance thresholds in meters, e.g. 10,30,50,80,100",
    )
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
    thresholds = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0:
            raise ValueError("Threshold must be positive: {}".format(token))
        thresholds.append(value)
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return sorted(set(thresholds))


def read_csv_points(csv_path, stamp_key, lon_key, lat_key):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("CSV not found: {}".format(csv_path))

    points = []
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [key for key in (stamp_key, lon_key, lat_key) if key not in reader.fieldnames]
        if missing:
            raise KeyError("Missing columns {} in {}".format(", ".join(missing), csv_path))

        for row in reader:
            stamp = parse_float(row.get(stamp_key))
            lon = parse_float(row.get(lon_key))
            lat = parse_float(row.get(lat_key))
            if not (math.isfinite(stamp) and math.isfinite(lon) and math.isfinite(lat)):
                continue
            points.append((stamp, lon, lat))

    points.sort(key=lambda item: item[0])
    if not points:
        raise ValueError("No valid rows found in {}".format(csv_path))
    return points


def filter_points_in_interval(points, start_stamp, end_stamp):
    return [point for point in points if start_stamp <= point[0] <= end_stamp]


def interpolate_gnss(gnss_points, stamp):
    stamps = [point[0] for point in gnss_points]
    index = bisect.bisect_left(stamps, stamp)

    if index < len(gnss_points) and abs(gnss_points[index][0] - stamp) < 1e-9:
        _, lon, lat = gnss_points[index]
        return lon, lat

    if index == 0 or index >= len(gnss_points):
        return None

    t0, lon0, lat0 = gnss_points[index - 1]
    t1, lon1, lat1 = gnss_points[index]
    if t1 <= t0:
        return None

    ratio = (stamp - t0) / (t1 - t0)
    lon = lon0 + ratio * (lon1 - lon0)
    lat = lat0 + ratio * (lat1 - lat0)
    return lon, lat


def lon_lat_error_to_meter(sample_lon, sample_lat, ref_lon, ref_lat):
    dlon_rad = math.radians(sample_lon - ref_lon)
    dlat_rad = math.radians(sample_lat - ref_lat)
    lat_ref_rad = math.radians((sample_lat + ref_lat) * 0.5)

    east_m = dlon_rad * EARTH_RADIUS_M * math.cos(lat_ref_rad)
    north_m = dlat_rad * EARTH_RADIUS_M
    horizontal_m = math.hypot(east_m, north_m)
    return east_m, north_m, horizontal_m


def build_error_series(name, data_points, gnss_points, start_stamp):
    stamps_rel = []
    lon_err_m = []
    lat_err_m = []
    horizontal_err_m = []

    for stamp, lon, lat in data_points:
        ref = interpolate_gnss(gnss_points, stamp)
        if ref is None:
            continue

        ref_lon, ref_lat = ref
        east_m, north_m, horizontal_m = lon_lat_error_to_meter(lon, lat, ref_lon, ref_lat)
        stamps_rel.append(stamp - start_stamp)
        lon_err_m.append(east_m)
        lat_err_m.append(north_m)
        horizontal_err_m.append(horizontal_m)

    if not stamps_rel:
        raise ValueError("No valid overlap samples after interpolation for {}".format(name))

    return {
        "name": name,
        "t": stamps_rel,
        "lon_err_m": lon_err_m,
        "lat_err_m": lat_err_m,
        "horizontal_err_m": horizontal_err_m,
    }


def compute_stats(values):
    count = len(values)
    mean_val = sum(values) / count
    rmse = math.sqrt(sum(value * value for value in values) / count)
    max_abs = max(abs(value) for value in values)
    return {"count": count, "mean": mean_val, "rmse": rmse, "max_abs": max_abs}


def compute_threshold_stats(values, thresholds):
    count = len(values)
    result = []
    for threshold in thresholds:
        within_count = sum(1 for value in values if value < threshold)
        ratio = 100.0 * within_count / count
        result.append({"threshold": threshold, "count": within_count, "ratio": ratio})
    return result


def plot_series(vision_series, fusion_series, output_png, start_stamp, end_stamp):
    ensure_parent_dir(output_png)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(vision_series["t"], vision_series["lon_err_m"], label="vision", linewidth=1.6)
    axes[0].plot(
        fusion_series["t"], fusion_series["lon_err_m"], label="fusion_position", linewidth=1.2
    )
    axes[0].set_ylabel("lon err (m)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right")

    axes[1].plot(vision_series["t"], vision_series["lat_err_m"], label="vision", linewidth=1.6)
    axes[1].plot(
        fusion_series["t"], fusion_series["lat_err_m"], label="fusion_position", linewidth=1.2
    )
    axes[1].set_ylabel("lat err (m)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(
        vision_series["t"], vision_series["horizontal_err_m"], label="vision", linewidth=1.6
    )
    axes[2].plot(
        fusion_series["t"],
        fusion_series["horizontal_err_m"],
        label="fusion_position",
        linewidth=1.2,
    )
    axes[2].set_ylabel("horizontal err (m)")
    axes[2].set_xlabel("time in shared interval (s)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Error vs GNSS in shared stamp interval [{:.3f}, {:.3f}]".format(start_stamp, end_stamp),
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_distance_series(vision_series, fusion_series, output_png, start_stamp, end_stamp):
    ensure_parent_dir(output_png)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
    ax.plot(
        vision_series["t"],
        vision_series["horizontal_err_m"],
        label="vision horizontal distance error",
        linewidth=1.5,
    )
    ax.plot(
        fusion_series["t"],
        fusion_series["horizontal_err_m"],
        label="fusion_position horizontal distance error",
        linewidth=1.2,
    )
    ax.set_xlabel("time in shared interval (s)")
    ax.set_ylabel("distance error (m)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    ax.set_title(
        "Absolute distance error vs GNSS in shared stamp interval [{:.3f}, {:.3f}]".format(
            start_stamp, end_stamp
        )
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()

    try:
        thresholds = parse_thresholds(args.thresholds)
        gnss_points = read_csv_points(args.gnss_csv, "fix_stamp_sec", "longitude", "latitude")
        vision_points = read_csv_points(args.vision_csv, "image_stamp_sec", "camera_lon", "camera_lat")
        fusion_points = read_csv_points(args.fusion_csv, "fix_stamp_sec", "pred_lon", "pred_lat")

        start_stamp = max(gnss_points[0][0], vision_points[0][0], fusion_points[0][0])
        end_stamp = min(gnss_points[-1][0], vision_points[-1][0], fusion_points[-1][0])
        if start_stamp >= end_stamp:
            raise ValueError("The three files do not have a shared timestamp interval.")

        gnss_shared = filter_points_in_interval(gnss_points, start_stamp, end_stamp)
        vision_shared = filter_points_in_interval(vision_points, start_stamp, end_stamp)
        fusion_shared = filter_points_in_interval(fusion_points, start_stamp, end_stamp)

        if len(gnss_shared) < 2:
            raise ValueError("Not enough GNSS points inside the shared interval for interpolation.")
        if not vision_shared:
            raise ValueError("No valid vision points inside the shared interval.")
        if not fusion_shared:
            raise ValueError("No valid fusion points inside the shared interval.")

        vision_series = build_error_series("vision", vision_shared, gnss_shared, start_stamp)
        fusion_series = build_error_series("fusion_position", fusion_shared, gnss_shared, start_stamp)

        plot_series(vision_series, fusion_series, args.output_png, start_stamp, end_stamp)
        plot_distance_series(vision_series, fusion_series, args.distance_output_png, start_stamp, end_stamp)

        vision_stats = compute_stats(vision_series["horizontal_err_m"])
        fusion_stats = compute_stats(fusion_series["horizontal_err_m"])
        vision_threshold_stats = compute_threshold_stats(vision_series["horizontal_err_m"], thresholds)
        fusion_threshold_stats = compute_threshold_stats(fusion_series["horizontal_err_m"], thresholds)

        print("Shared stamp interval: [{:.6f}, {:.6f}]".format(start_stamp, end_stamp))
        print("Saved signed error plot to {}".format(args.output_png))
        print("Saved absolute distance error plot to {}".format(args.distance_output_png))
        print(
            "Vision distance error: count={count}, mean={mean:.3f} m, rmse={rmse:.3f} m, max_abs={max_abs:.3f} m".format(
                **vision_stats
            )
        )
        print(
            "Fusion distance error: count={count}, mean={mean:.3f} m, rmse={rmse:.3f} m, max_abs={max_abs:.3f} m".format(
                **fusion_stats
            )
        )
        print("Vision threshold statistics:")
        for item in vision_threshold_stats:
            print(
                "  error < {:.3f} m : {:.2f}% ({}/{})".format(
                    item["threshold"], item["ratio"], item["count"], vision_stats["count"]
                )
            )
        print("Fusion threshold statistics:")
        for item in fusion_threshold_stats:
            print(
                "  error < {:.3f} m : {:.2f}% ({}/{})".format(
                    item["threshold"], item["ratio"], item["count"], fusion_stats["count"]
                )
            )
    except Exception as exc:
        print("Failed to evaluate localization results: {}".format(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
