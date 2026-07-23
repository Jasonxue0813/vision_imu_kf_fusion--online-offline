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
DEFAULT_OUTPUT_PNG = os.path.join(BASE_DIR, "result", "flight_path.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read GNSS, fusion, and vision positions and draw them on one flight path plot."
    )
    parser.add_argument("--gnss_csv", default=DEFAULT_GNSS_CSV, help="Path to extract_gnss_position.csv")
    parser.add_argument("--fusion_csv", default=DEFAULT_FUSION_CSV, help="Path to fusion_position.csv")
    parser.add_argument("--vision_csv", default=DEFAULT_VISION_CSV, help="Path to vision_position.csv")
    parser.add_argument("--output_png", default=DEFAULT_OUTPUT_PNG, help="Path to the output flight path plot")
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


def plot_flight_path(gnss_points, fusion_points, vision_points, output_png):
    ensure_parent_dir(output_png)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.plot(
        [point[1] for point in gnss_points],
        [point[2] for point in gnss_points],
        color="tab:blue",
        linewidth=1.6,
        label="GNSS",
    )
    ax.plot(
        [point[1] for point in fusion_points],
        [point[2] for point in fusion_points],
        color="tab:red",
        linewidth=1.3,
        label="Fusion",
    )
    ax.plot(
        [point[1] for point in vision_points],
        [point[2] for point in vision_points],
        color="tab:green",
        linewidth=1.3,
        label="Vision",
    )

    ax.scatter(gnss_points[0][1], gnss_points[0][2], color="tab:blue", s=28, marker="o")
    ax.scatter(gnss_points[-1][1], gnss_points[-1][2], color="tab:blue", s=34, marker="x")
    ax.scatter(fusion_points[0][1], fusion_points[0][2], color="tab:red", s=28, marker="o")
    ax.scatter(fusion_points[-1][1], fusion_points[-1][2], color="tab:red", s=34, marker="x")
    ax.scatter(vision_points[0][1], vision_points[0][2], color="tab:green", s=28, marker="o")
    ax.scatter(vision_points[-1][1], vision_points[-1][2], color="tab:green", s=34, marker="x")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Flight Path Comparison")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    try:
        gnss_points = read_csv_points(args.gnss_csv, "fix_stamp_sec", "longitude", "latitude")
        fusion_points = read_csv_points(args.fusion_csv, "fix_stamp_sec", "pred_lon", "pred_lat")
        vision_points = read_csv_points(args.vision_csv, "image_stamp_sec", "camera_lon", "camera_lat")

        plot_flight_path(gnss_points, fusion_points, vision_points, args.output_png)
    except Exception as exc:
        print("Failed to draw flight path: {}".format(exc), file=sys.stderr)
        return 1

    print("GNSS points  : {}".format(len(gnss_points)))
    print("Fusion points: {}".format(len(fusion_points)))
    print("Vision points: {}".format(len(vision_points)))
    print("Saved flight path plot to {}".format(args.output_png))
    return 0


if __name__ == "__main__":
    sys.exit(main())
