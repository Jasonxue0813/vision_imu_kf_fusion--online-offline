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
DEFAULT_OUTPUT_PNG = os.path.join(BASE_DIR, "result", "evaluation3_error_plot.png")
DEFAULT_DISTANCE_OUTPUT_PNG = os.path.join(BASE_DIR, "result", "evaluation3_distance_plot.png")

EARTH_RADIUS_M = 6378137.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align vision and fusion positions to GNSS using the first 10 seconds, "
            "then compare errors after the alignment interval."
        )
    )
    parser.add_argument("--gnss_csv", default=DEFAULT_GNSS_CSV)
    parser.add_argument("--vision_csv", default=DEFAULT_VISION_CSV)
    parser.add_argument("--fusion_csv", default=DEFAULT_FUSION_CSV)
    parser.add_argument("--output_png", default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--distance_output_png", default=DEFAULT_DISTANCE_OUTPUT_PNG)
    parser.add_argument("--alignment_sec", type=float, default=10.0)
    parser.add_argument(
        "--thresholds",
        default="10,30,50,80,100",
        help="Comma separated distance thresholds in meters",
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
        fieldnames = reader.fieldnames or []
        missing = [key for key in (stamp_key, lon_key, lat_key) if key not in fieldnames]
        if missing:
            raise KeyError("Missing columns {} in {}".format(", ".join(missing), csv_path))

        for row in reader:
            stamp = parse_float(row.get(stamp_key))
            lon = parse_float(row.get(lon_key))
            lat = parse_float(row.get(lat_key))
            if math.isfinite(stamp) and math.isfinite(lon) and math.isfinite(lat):
                points.append((stamp, lon, lat))

    points.sort(key=lambda item: item[0])
    if not points:
        raise ValueError("No valid rows found in {}".format(csv_path))
    return points


def interpolate_gnss(gnss_points, stamp):
    stamps = [point[0] for point in gnss_points]
    index = bisect.bisect_left(stamps, stamp)
    if index < len(gnss_points) and abs(gnss_points[index][0] - stamp) < 1e-9:
        return gnss_points[index][1], gnss_points[index][2]
    if index == 0 or index >= len(gnss_points):
        return None

    t0, lon0, lat0 = gnss_points[index - 1]
    t1, lon1, lat1 = gnss_points[index]
    if t1 <= t0:
        return None
    ratio = (stamp - t0) / (t1 - t0)
    return lon0 + ratio * (lon1 - lon0), lat0 + ratio * (lat1 - lat0)


def lon_lat_to_enu(sample_lon, sample_lat, ref_lon, ref_lat):
    lat_ref_rad = math.radians((sample_lat + ref_lat) * 0.5)
    east_m = math.radians(sample_lon - ref_lon) * EARTH_RADIUS_M * math.cos(lat_ref_rad)
    north_m = math.radians(sample_lat - ref_lat) * EARTH_RADIUS_M
    return east_m, north_m


def mean_alignment_offset(vision_points, gnss_points, alignment_sec, start_stamp):
    offsets = []
    for stamp, lon, lat in vision_points:
        relative_stamp = stamp - start_stamp
        if 0.0 <= relative_stamp < alignment_sec:
            ref = interpolate_gnss(gnss_points, stamp)
            if ref is not None:
                east_m, north_m = lon_lat_to_enu(lon, lat, ref[0], ref[1])
                offsets.append((east_m, north_m))
    if not offsets:
        raise ValueError("No valid vision/GNSS samples in the alignment interval.")
    return (
        sum(offset[0] for offset in offsets) / len(offsets),
        sum(offset[1] for offset in offsets) / len(offsets),
        len(offsets),
    )


def build_error_series(name, data_points, gnss_points, start_stamp, alignment_sec, offset):
    stamps_rel = []
    east_err_m = []
    north_err_m = []
    horizontal_err_m = []
    offset_east_m, offset_north_m = offset

    for stamp, lon, lat in data_points:
        relative_stamp = stamp - start_stamp
        if relative_stamp < alignment_sec:
            continue
        ref = interpolate_gnss(gnss_points, stamp)
        if ref is None:
            continue
        east_m, north_m = lon_lat_to_enu(lon, lat, ref[0], ref[1])
        east_m -= offset_east_m
        north_m -= offset_north_m
        stamps_rel.append(relative_stamp)
        east_err_m.append(east_m)
        north_err_m.append(north_m)
        horizontal_err_m.append(math.hypot(east_m, north_m))

    if not stamps_rel:
        raise ValueError("No valid samples after the alignment interval for {}".format(name))
    return {"name": name, "t": stamps_rel, "east_err_m": east_err_m, "north_err_m": north_err_m, "horizontal_err_m": horizontal_err_m}


def compute_stats(values):
    count = len(values)
    return {
        "count": count,
        "mean": sum(values) / count,
        "rmse": math.sqrt(sum(value * value for value in values) / count),
        "max_abs": max(abs(value) for value in values),
    }


def compute_threshold_stats(values, thresholds):
    return [
        {
            "threshold": threshold,
            "count": sum(1 for value in values if value < threshold),
            "ratio": 100.0 * sum(1 for value in values if value < threshold) / len(values),
        }
        for threshold in thresholds
    ]


def plot_series(vision_series, fusion_series, output_png, alignment_sec, start_stamp, end_stamp):
    ensure_parent_dir(output_png)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for axis, key, label in (
        (axes[0], "east_err_m", "east error (m)"),
        (axes[1], "north_err_m", "north error (m)"),
        (axes[2], "horizontal_err_m", "horizontal error (m)"),
    ):
        axis.plot(vision_series["t"], vision_series[key], label="vision", linewidth=1.6)
        axis.plot(fusion_series["t"], fusion_series[key], label="fusion_position", linewidth=1.2)
        axis.set_ylabel(label)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="upper right")
    axes[2].set_xlabel("time from shared start (s)")
    fig.suptitle(
        "Aligned error after {:.1f}s calibration [{:.3f}, {:.3f}]".format(
            alignment_sec, start_stamp, end_stamp
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_distance_series(vision_series, fusion_series, output_png, alignment_sec):
    ensure_parent_dir(output_png)
    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
    ax.plot(vision_series["t"], vision_series["horizontal_err_m"], label="vision", linewidth=1.5)
    ax.plot(fusion_series["t"], fusion_series["horizontal_err_m"], label="fusion_position", linewidth=1.2)
    ax.set_xlabel("time from shared start (s)")
    ax.set_ylabel("aligned distance error (m)")
    ax.set_title("Aligned distance error after {:.1f}s calibration".format(alignment_sec))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    try:
        if args.alignment_sec <= 0.0:
            raise ValueError("alignment_sec must be positive.")
        thresholds = parse_thresholds(args.thresholds)
        gnss_points = read_csv_points(args.gnss_csv, "fix_stamp_sec", "longitude", "latitude")
        vision_points = read_csv_points(args.vision_csv, "image_stamp_sec", "camera_lon", "camera_lat")
        fusion_points = read_csv_points(args.fusion_csv, "fix_stamp_sec", "pred_lon", "pred_lat")

        start_stamp = max(gnss_points[0][0], vision_points[0][0], fusion_points[0][0])
        end_stamp = min(gnss_points[-1][0], vision_points[-1][0], fusion_points[-1][0])
        if start_stamp >= end_stamp or end_stamp - start_stamp <= args.alignment_sec:
            raise ValueError("The shared timestamp interval is too short for alignment.")

        gnss_shared = [point for point in gnss_points if start_stamp <= point[0] <= end_stamp]
        vision_shared = [point for point in vision_points if start_stamp <= point[0] <= end_stamp]
        fusion_shared = [point for point in fusion_points if start_stamp <= point[0] <= end_stamp]
        if len(gnss_shared) < 2 or not vision_shared or not fusion_shared:
            raise ValueError("Not enough valid samples inside the shared timestamp interval.")

        offset_east_m, offset_north_m, alignment_count = mean_alignment_offset(
            vision_shared, gnss_shared, args.alignment_sec, start_stamp
        )
        vision_series = build_error_series(
            "vision", vision_shared, gnss_shared, start_stamp, args.alignment_sec, (offset_east_m, offset_north_m)
        )
        fusion_series = build_error_series(
            "fusion_position", fusion_shared, gnss_shared, start_stamp, args.alignment_sec, (offset_east_m, offset_north_m)
        )

        plot_series(vision_series, fusion_series, args.output_png, args.alignment_sec, start_stamp, end_stamp)
        plot_distance_series(vision_series, fusion_series, args.distance_output_png, args.alignment_sec)

        vision_stats = compute_stats(vision_series["horizontal_err_m"])
        fusion_stats = compute_stats(fusion_series["horizontal_err_m"])
        print("Shared stamp interval: [{:.6f}, {:.6f}]".format(start_stamp, end_stamp))
        print("Alignment samples: {}".format(alignment_count))
        print("Fixed bias: east={:.3f} m, north={:.3f} m, distance={:.3f} m".format(
            offset_east_m, offset_north_m, math.hypot(offset_east_m, offset_north_m)
        ))
        print("Saved aligned signed error plot to {}".format(args.output_png))
        print("Saved aligned distance error plot to {}".format(args.distance_output_png))
        print("Vision aligned distance error: count={count}, mean={mean:.3f} m, rmse={rmse:.3f} m, max_abs={max_abs:.3f} m".format(**vision_stats))
        print("Fusion aligned distance error: count={count}, mean={mean:.3f} m, rmse={rmse:.3f} m, max_abs={max_abs:.3f} m".format(**fusion_stats))
        for name, values in (("Vision", vision_series["horizontal_err_m"]), ("Fusion", fusion_series["horizontal_err_m"])):
            print("{} threshold statistics:".format(name))
            for item in compute_threshold_stats(values, thresholds):
                print("  error < {:.3f} m : {:.2f}% ({}/{})".format(item["threshold"], item["ratio"], item["count"], len(values)))
    except Exception as exc:
        print("Failed to evaluate aligned localization results: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
