#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = "/home/jasonxue/splg_lo_origin/splg_fusion"
DEFAULT_GNSS_CSV = os.path.join(BASE_DIR, "result", "extract_gnss_position.csv")
DEFAULT_FUSION_CSV = os.path.join(BASE_DIR, "result", "fusion_position.csv")
DEFAULT_VISION_CSV = os.path.join(BASE_DIR, "result", "vision_position.csv")
DEFAULT_OUTPUT_PNG = os.path.join(BASE_DIR, "result", "flight_path3.png")
EARTH_RADIUS_M = 6378137.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw GNSS, vision, and fusion paths after removing the first 10s vision-GNSS bias."
    )
    parser.add_argument("--gnss_csv", default=DEFAULT_GNSS_CSV)
    parser.add_argument("--fusion_csv", default=DEFAULT_FUSION_CSV)
    parser.add_argument("--vision_csv", default=DEFAULT_VISION_CSV)
    parser.add_argument("--output_png", default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--alignment_sec", type=float, default=10.0)
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


def interpolate_points(points, stamp):
    if stamp <= points[0][0]:
        return points[0][1], points[0][2]
    if stamp >= points[-1][0]:
        return points[-1][1], points[-1][2]
    for index in range(1, len(points)):
        t0, lon0, lat0 = points[index - 1]
        t1, lon1, lat1 = points[index]
        if stamp <= t1:
            ratio = (stamp - t0) / (t1 - t0)
            return lon0 + ratio * (lon1 - lon0), lat0 + ratio * (lat1 - lat0)
    return None


def lon_lat_to_enu(sample_lon, sample_lat, ref_lon, ref_lat):
    lat_ref_rad = math.radians((sample_lat + ref_lat) * 0.5)
    east_m = math.radians(sample_lon - ref_lon) * EARTH_RADIUS_M * math.cos(lat_ref_rad)
    north_m = math.radians(sample_lat - ref_lat) * EARTH_RADIUS_M
    return east_m, north_m


def enu_to_lon_lat(east_m, north_m, ref_lon, ref_lat):
    lat_ref_rad = math.radians(ref_lat)
    lon = ref_lon + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(lat_ref_rad)))
    lat = ref_lat + math.degrees(north_m / EARTH_RADIUS_M)
    return lon, lat


def mean_alignment_offset(vision_points, gnss_points, alignment_sec, start_stamp):
    offsets = []
    for stamp, lon, lat in vision_points:
        if not (0.0 <= stamp - start_stamp < alignment_sec):
            continue
        ref = interpolate_points(gnss_points, stamp)
        if ref is not None:
            offsets.append(lon_lat_to_enu(lon, lat, ref[0], ref[1]))
    if not offsets:
        raise ValueError("No valid vision/GNSS samples in the alignment interval.")
    return (
        sum(item[0] for item in offsets) / len(offsets),
        sum(item[1] for item in offsets) / len(offsets),
        len(offsets),
    )


def corrected_path(data_points, gnss_points, start_stamp, alignment_sec, offset):
    corrected = []
    for stamp, lon, lat in data_points:
        if stamp - start_stamp < alignment_sec:
            continue
        ref = interpolate_points(gnss_points, stamp)
        if ref is None:
            continue
        east_m, north_m = lon_lat_to_enu(lon, lat, ref[0], ref[1])
        east_m -= offset[0]
        north_m -= offset[1]
        corrected_lon, corrected_lat = enu_to_lon_lat(east_m, north_m, ref[0], ref[1])
        corrected.append((stamp, corrected_lon, corrected_lat))
    return corrected


def shared_gnss_path(gnss_points, start_stamp, end_stamp):
    return [point for point in gnss_points if start_stamp <= point[0] <= end_stamp]


def plot_flight_path(gnss_points, fusion_points, vision_points, output_png, alignment_sec, offset, alignment_count):
    ensure_parent_dir(output_png)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot([point[1] for point in gnss_points], [point[2] for point in gnss_points], color="tab:blue", linewidth=1.6, label="GNSS")
    if fusion_points:
        ax.plot([point[1] for point in fusion_points], [point[2] for point in fusion_points], color="tab:red", linewidth=1.3, label="Fusion aligned")
    if vision_points:
        ax.plot([point[1] for point in vision_points], [point[2] for point in vision_points], color="tab:green", linewidth=1.3, label="Vision aligned")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        "Flight Path with {:.1f}s Vision-GNSS Bias Removed\n"
        "bias east={:.2f}m, north={:.2f}m, samples={}".format(
            alignment_sec, offset[0], offset[1], alignment_count
        )
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    try:
        if args.alignment_sec <= 0.0:
            raise ValueError("alignment_sec must be positive.")
        gnss_points = read_csv_points(args.gnss_csv, "fix_stamp_sec", "longitude", "latitude")
        fusion_points = read_csv_points(args.fusion_csv, "fix_stamp_sec", "pred_lon", "pred_lat")
        vision_points = read_csv_points(args.vision_csv, "image_stamp_sec", "camera_lon", "camera_lat")

        start_stamp = max(gnss_points[0][0], vision_points[0][0], fusion_points[0][0])
        end_stamp = min(gnss_points[-1][0], vision_points[-1][0], fusion_points[-1][0])
        if start_stamp >= end_stamp or end_stamp - start_stamp <= args.alignment_sec:
            raise ValueError("The shared timestamp interval is too short for alignment.")

        gnss_shared = shared_gnss_path(gnss_points, start_stamp, end_stamp)
        vision_shared = [point for point in vision_points if start_stamp <= point[0] <= end_stamp]
        fusion_shared = [point for point in fusion_points if start_stamp <= point[0] <= end_stamp]
        offset_east_m, offset_north_m, alignment_count = mean_alignment_offset(
            vision_shared, gnss_shared, args.alignment_sec, start_stamp
        )

        vision_aligned = corrected_path(
            vision_shared, gnss_shared, start_stamp, args.alignment_sec, (offset_east_m, offset_north_m)
        )
        fusion_aligned = corrected_path(
            fusion_shared, gnss_shared, start_stamp, args.alignment_sec, (offset_east_m, offset_north_m)
        )
        gnss_plot = [point for point in gnss_shared if point[0] - start_stamp >= args.alignment_sec]
        if not vision_aligned or not fusion_aligned or not gnss_plot:
            raise ValueError("No valid path samples after the alignment interval.")

        plot_flight_path(
            gnss_plot,
            fusion_aligned,
            vision_aligned,
            args.output_png,
            args.alignment_sec,
            (offset_east_m, offset_north_m),
            alignment_count,
        )
        print("Shared stamp interval: [{:.6f}, {:.6f}]".format(start_stamp, end_stamp))
        print("Alignment samples: {}".format(alignment_count))
        print("Removed fixed bias: east={:.3f} m, north={:.3f} m, distance={:.3f} m".format(
            offset_east_m, offset_north_m, math.hypot(offset_east_m, offset_north_m)
        ))
        print("GNSS points  : {}".format(len(gnss_plot)))
        print("Fusion points: {}".format(len(fusion_aligned)))
        print("Vision points: {}".format(len(vision_aligned)))
        print("Saved aligned flight path to {}".format(args.output_png))
    except Exception as exc:
        print("Failed to draw aligned flight path: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
