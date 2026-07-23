#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys


BASE_DIR = "/home/jasonxue/splg_lo_origin/splg_fusion"
DEFAULT_INPUT = os.path.join(BASE_DIR, "result", "extract_gnss_position.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "result", "extract_gnss_position_processed.csv")
EARTH_RADIUS_M = 6378137.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect GNSS position jumps and replace anomalous positions "
            "with time-interpolated values."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input GNSS CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output processed CSV path")
    parser.add_argument(
        "--max_step_m",
        type=float,
        default=10.0,
        help="Maximum allowed distance between consecutive accepted points",
    )
    return parser.parse_args()


def parse_float(value):
    if value is None:
        return math.nan
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return math.nan
    return float(value)


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_points(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("Input CSV not found: {}".format(path))

    with open(path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        required = ["fix_stamp_sec", "longitude", "latitude", "altitude"]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise KeyError("Missing columns: {}".format(", ".join(missing)))

        points = []
        for row_number, row in enumerate(reader, start=2):
            try:
                point = {
                    "fix_index": row.get("fix_index", str(row_number - 1)),
                    "fix_stamp_sec": parse_float(row.get("fix_stamp_sec")),
                    "longitude": parse_float(row.get("longitude")),
                    "latitude": parse_float(row.get("latitude")),
                    "altitude": parse_float(row.get("altitude")),
                }
            except ValueError as exc:
                raise ValueError("Invalid numeric value at CSV line {}: {}".format(row_number, exc)) from exc

            if not all(
                math.isfinite(point[key])
                for key in ("fix_stamp_sec", "longitude", "latitude", "altitude")
            ):
                raise ValueError("Non-finite GNSS value at CSV line {}".format(row_number))
            points.append(point)

    if not points:
        raise ValueError("No valid GNSS rows found in {}".format(path))
    points.sort(key=lambda point: point["fix_stamp_sec"])
    return points


def distance_m(point_a, point_b):
    """计算两个经纬度点的近似水平距离。"""
    lat_ref_rad = math.radians((point_a["latitude"] + point_b["latitude"]) * 0.5)
    east_m = math.radians(point_a["longitude"] - point_b["longitude"])
    east_m *= EARTH_RADIUS_M * math.cos(lat_ref_rad)
    north_m = math.radians(point_a["latitude"] - point_b["latitude"]) * EARTH_RADIUS_M
    return math.hypot(east_m, north_m)


def interpolate_point(point_a, point_b, stamp_sec):
    """按时间在线性插值经纬度和高度。"""
    t0 = point_a["fix_stamp_sec"]
    t1 = point_b["fix_stamp_sec"]
    if t1 <= t0:
        return dict(point_a)

    ratio = (stamp_sec - t0) / (t1 - t0)
    result = dict(point_a)
    result["fix_stamp_sec"] = stamp_sec
    for key in ("longitude", "latitude", "altitude"):
        result[key] = point_a[key] + ratio * (point_b[key] - point_a[key])
    return result


def mark_anomalies(points, max_step_m):
    """按最近正常点检测连续跳点，并返回正常点索引和异常点索引。"""
    valid_indices = [0]
    anomaly_indices = set()
    anchor_index = 0
    index = 1

    while index < len(points):
        if distance_m(points[anchor_index], points[index]) <= max_step_m:
            valid_indices.append(index)
            anchor_index = index
            index += 1
            continue

        next_valid_index = None
        for candidate in range(index + 1, len(points)):
            if distance_m(points[anchor_index], points[candidate]) <= max_step_m:
                next_valid_index = candidate
                break

        if next_valid_index is None:
            anomaly_indices.update(range(index, len(points)))
            break

        anomaly_indices.update(range(index, next_valid_index))
        valid_indices.append(next_valid_index)
        anchor_index = next_valid_index
        index = next_valid_index + 1

    return valid_indices, anomaly_indices


def replace_anomalies(points, valid_indices, anomaly_indices):
    """用异常点前后的正常点按时间插值替换异常点。"""
    if not anomaly_indices:
        return [dict(point) for point in points]

    valid_set = set(valid_indices)
    result = [dict(point) for point in points]
    for index in sorted(anomaly_indices):
        previous_valid = next(
            (candidate for candidate in range(index - 1, -1, -1) if candidate in valid_set),
            None,
        )
        next_valid = next(
            (candidate for candidate in range(index + 1, len(points)) if candidate in valid_set),
            None,
        )

        if previous_valid is not None and next_valid is not None:
            result[index] = interpolate_point(
                points[previous_valid], points[next_valid], points[index]["fix_stamp_sec"]
            )
        elif previous_valid is not None:
            result[index] = dict(points[previous_valid])
            result[index]["fix_stamp_sec"] = points[index]["fix_stamp_sec"]
        elif next_valid is not None:
            result[index] = dict(points[next_valid])
            result[index]["fix_stamp_sec"] = points[index]["fix_stamp_sec"]

    return result


def write_points(path, points):
    ensure_parent_dir(path)
    fieldnames = ["fix_index", "fix_stamp_sec", "longitude", "latitude", "altitude"]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "fix_index": point["fix_index"],
                    "fix_stamp_sec": "{:.9f}".format(point["fix_stamp_sec"]),
                    "longitude": "{:.10f}".format(point["longitude"]),
                    "latitude": "{:.10f}".format(point["latitude"]),
                    "altitude": "{:.6f}".format(point["altitude"]),
                }
            )


def main():
    args = parse_args()
    try:
        if args.max_step_m <= 0.0:
            raise ValueError("max_step_m must be positive.")

        points = read_points(args.input)
        valid_indices, anomaly_indices = mark_anomalies(points, args.max_step_m)
        processed_points = replace_anomalies(points, valid_indices, anomaly_indices)
        write_points(args.output, processed_points)

        print("Input points     : {}".format(len(points)))
        print("Anomaly points   : {}".format(len(anomaly_indices)))
        print("Max allowed step : {:.3f} m".format(args.max_step_m))
        print("Saved processed CSV to {}".format(args.output))
    except Exception as exc:
        print("Failed to process GNSS CSV: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
