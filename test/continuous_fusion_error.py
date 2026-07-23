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
RESULT_DIR = os.path.join(BASE_DIR, "result")

DEFAULT_GNSS_CSV = os.path.join(RESULT_DIR, "extract_gnss_position.csv")
DEFAULT_CONTINUOUS_CSV = os.path.join(RESULT_DIR, "continuous_fusion.csv")
DEFAULT_ERROR_PNG = os.path.join(RESULT_DIR, "continuous_fusion_error_plot.png")
DEFAULT_DISTANCE_PNG = os.path.join(RESULT_DIR, "continuous_fusion_distance_plot.png")
DEFAULT_TRACK_PNG = os.path.join(RESULT_DIR, "continuous_fusion_track_plot.png")

EARTH_RADIUS_M = 6378137.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare continuous_fusion.csv against extract_gnss_position.csv."
    )
    parser.add_argument("--gnss_csv", default=DEFAULT_GNSS_CSV, help="Path to extract_gnss_position.csv")
    parser.add_argument(
        "--continuous_csv", default=DEFAULT_CONTINUOUS_CSV, help="Path to continuous_fusion.csv"
    )
    parser.add_argument(
        "--error_png", default=DEFAULT_ERROR_PNG, help="Path to the signed east/north/up error plot"
    )
    parser.add_argument(
        "--distance_png", default=DEFAULT_DISTANCE_PNG, help="Path to the absolute distance error plot"
    )
    parser.add_argument(
        "--track_png", default=DEFAULT_TRACK_PNG, help="Path to the continuous fusion and GNSS track plot"
    )
    parser.add_argument(
        "--thresholds",
        default="1,2,3,5,10",
        help="Comma separated distance thresholds in meters, e.g. 1,2,3,5,10",
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


def read_csv_points(csv_path, stamp_key, lon_key, lat_key, alt_key):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("CSV not found: {}".format(csv_path))

    points = []
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [key for key in (stamp_key, lon_key, lat_key, alt_key) if key not in (reader.fieldnames or [])]
        if missing:
            raise KeyError("Missing columns {} in {}".format(", ".join(missing), csv_path))

        for row in reader:
            stamp = parse_float(row.get(stamp_key))
            lon = parse_float(row.get(lon_key))
            lat = parse_float(row.get(lat_key))
            alt = parse_float(row.get(alt_key))
            if not (math.isfinite(stamp) and math.isfinite(lon) and math.isfinite(lat) and math.isfinite(alt)):
                continue
            points.append((stamp, lon, lat, alt))

    points.sort(key=lambda item: item[0])
    if not points:
        raise ValueError("No valid rows found in {}".format(csv_path))
    return points


def interpolate_reference(points, stamp):
    stamps = [point[0] for point in points]
    index = bisect.bisect_left(stamps, stamp)

    if index < len(points) and abs(points[index][0] - stamp) < 1e-9:
        _, lon, lat, alt = points[index]
        return lon, lat, alt

    if index == 0 or index >= len(points):
        return None

    t0, lon0, lat0, alt0 = points[index - 1]
    t1, lon1, lat1, alt1 = points[index]
    if t1 <= t0:
        return None

    ratio = (stamp - t0) / (t1 - t0)
    lon = lon0 + ratio * (lon1 - lon0)
    lat = lat0 + ratio * (lat1 - lat0)
    alt = alt0 + ratio * (alt1 - alt0)
    return lon, lat, alt


def lon_lat_error_to_meter(sample_lon, sample_lat, ref_lon, ref_lat):
    dlon_rad = math.radians(sample_lon - ref_lon)
    dlat_rad = math.radians(sample_lat - ref_lat)
    lat_ref_rad = math.radians((sample_lat + ref_lat) * 0.5)
    east_m = dlon_rad * EARTH_RADIUS_M * math.cos(lat_ref_rad)
    north_m = dlat_rad * EARTH_RADIUS_M
    return east_m, north_m


def build_error_series(continuous_points, gnss_points):
    start_stamp = max(continuous_points[0][0], gnss_points[0][0])
    end_stamp = min(continuous_points[-1][0], gnss_points[-1][0])
    if start_stamp >= end_stamp:
        raise ValueError("continuous_fusion.csv and extract_gnss_position.csv do not overlap in time.")

    stamps_rel = []
    east_err_m = []
    north_err_m = []
    up_err_m = []
    horizontal_err_m = []
    total_err_m = []
    continuous_track_lon = []
    continuous_track_lat = []
    gnss_track_lon = []
    gnss_track_lat = []

    for stamp, lon, lat, alt in continuous_points:
        if stamp < start_stamp or stamp > end_stamp:
            continue

        ref = interpolate_reference(gnss_points, stamp)
        if ref is None:
            continue

        ref_lon, ref_lat, ref_alt = ref
        east_m, north_m = lon_lat_error_to_meter(lon, lat, ref_lon, ref_lat)
        up_m = alt - ref_alt
        horizontal_m = math.hypot(east_m, north_m)
        total_m = math.sqrt(east_m * east_m + north_m * north_m + up_m * up_m)

        stamps_rel.append(stamp - start_stamp)
        east_err_m.append(east_m)
        north_err_m.append(north_m)
        up_err_m.append(up_m)
        horizontal_err_m.append(horizontal_m)
        total_err_m.append(total_m)
        continuous_track_lon.append(lon)
        continuous_track_lat.append(lat)
        gnss_track_lon.append(ref_lon)
        gnss_track_lat.append(ref_lat)

    if not stamps_rel:
        raise ValueError("No valid overlap samples after GNSS interpolation.")

    return {
        "start_stamp": start_stamp,
        "end_stamp": end_stamp,
        "t": stamps_rel,
        "east_err_m": east_err_m,
        "north_err_m": north_err_m,
        "up_err_m": up_err_m,
        "horizontal_err_m": horizontal_err_m,
        "total_err_m": total_err_m,
        "continuous_track_lon": continuous_track_lon,
        "continuous_track_lat": continuous_track_lat,
        "gnss_track_lon": gnss_track_lon,
        "gnss_track_lat": gnss_track_lat,
    }


def compute_stats(values):
    count = len(values)
    mean_val = sum(values) / count
    rmse = math.sqrt(sum(value * value for value in values) / count)
    max_abs = max(abs(value) for value in values)
    return {"count": count, "mean": mean_val, "rmse": rmse, "max_abs": max_abs}


def compute_threshold_stats(values, thresholds):
    count = len(values)
    stats = []
    for threshold in thresholds:
        within_count = sum(1 for value in values if value < threshold)
        ratio = 100.0 * within_count / count
        stats.append({"threshold": threshold, "count": within_count, "ratio": ratio})
    return stats


def plot_signed_errors(series, output_png):
    ensure_parent_dir(output_png)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(series["t"], series["east_err_m"], linewidth=1.2, label="east")
    axes[0].set_ylabel("east err (m)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right")

    axes[1].plot(series["t"], series["north_err_m"], linewidth=1.2, label="north")
    axes[1].set_ylabel("north err (m)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right")

    axes[2].plot(series["t"], series["up_err_m"], linewidth=1.2, label="up")
    axes[2].set_ylabel("up err (m)")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="upper right")

    axes[3].plot(series["t"], series["horizontal_err_m"], linewidth=1.3, label="horizontal")
    axes[3].set_ylabel("horizontal err (m)")
    axes[3].set_xlabel("time in shared interval (s)")
    axes[3].grid(True, linestyle="--", alpha=0.4)
    axes[3].legend(loc="upper right")

    fig.suptitle(
        "Continuous Fusion Error vs GNSS [{:.3f}, {:.3f}]".format(
            series["start_stamp"], series["end_stamp"]
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_distance_errors(series, output_png):
    ensure_parent_dir(output_png)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
    ax.plot(series["t"], series["horizontal_err_m"], linewidth=1.2, label="horizontal")
    ax.plot(series["t"], series["total_err_m"], linewidth=1.2, label="3d total")
    ax.set_xlabel("time in shared interval (s)")
    ax.set_ylabel("distance error (m)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    ax.set_title(
        "Continuous Fusion Distance Error vs GNSS [{:.3f}, {:.3f}]".format(
            series["start_stamp"], series["end_stamp"]
        )
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_track(series, output_png):
    ensure_parent_dir(output_png)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(
        series["gnss_track_lon"],
        series["gnss_track_lat"],
        linewidth=1.4,
        label="GNSS",
        color="tab:blue",
    )
    ax.plot(
        series["continuous_track_lon"],
        series["continuous_track_lat"],
        linewidth=1.2,
        label="continuous_fusion",
        color="tab:red",
    )
    ax.scatter(series["gnss_track_lon"][0], series["gnss_track_lat"][0], color="tab:blue", s=28, marker="o")
    ax.scatter(series["gnss_track_lon"][-1], series["gnss_track_lat"][-1], color="tab:blue", s=34, marker="x")
    ax.scatter(
        series["continuous_track_lon"][0],
        series["continuous_track_lat"][0],
        color="tab:red",
        s=28,
        marker="o",
    )
    ax.scatter(
        series["continuous_track_lon"][-1],
        series["continuous_track_lat"][-1],
        color="tab:red",
        s=34,
        marker="x",
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        "Continuous Fusion Track vs GNSS [{:.3f}, {:.3f}]".format(
            series["start_stamp"], series["end_stamp"]
        )
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def print_stats(name, values):
    stats = compute_stats(values)
    print(
        "{}: count={count}, mean={mean:.3f} m, rmse={rmse:.3f} m, max_abs={max_abs:.3f} m".format(
            name, **stats
        )
    )
    return stats


def main():
    args = parse_args()

    try:
        thresholds = parse_thresholds(args.thresholds)
        gnss_points = read_csv_points(
            args.gnss_csv,
            "fix_stamp_sec",
            "longitude",
            "latitude",
            "altitude",
        )
        continuous_points = read_csv_points(
            args.continuous_csv,
            "stamp_sec",
            "lon",
            "lat",
            "alt",
        )

        series = build_error_series(continuous_points, gnss_points)
        plot_signed_errors(series, args.error_png)
        plot_distance_errors(series, args.distance_png)
        plot_track(series, args.track_png)

        print("Shared stamp interval: [{:.6f}, {:.6f}]".format(series["start_stamp"], series["end_stamp"]))
        print("Saved signed error plot to {}".format(args.error_png))
        print("Saved distance error plot to {}".format(args.distance_png))
        print("Saved track plot to {}".format(args.track_png))
        print_stats("East error", series["east_err_m"])
        print_stats("North error", series["north_err_m"])
        print_stats("Up error", series["up_err_m"])
        horizontal_stats = print_stats("Horizontal distance error", series["horizontal_err_m"])
        print_stats("3D total distance error", series["total_err_m"])
        print("Horizontal distance threshold statistics:")
        for item in compute_threshold_stats(series["horizontal_err_m"], thresholds):
            print(
                "  error < {:.3f} m : {:.2f}% ({}/{})".format(
                    item["threshold"], item["ratio"], item["count"], horizontal_stats["count"]
                )
            )
    except Exception as exc:
        print("Failed to evaluate continuous fusion error: {}".format(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
