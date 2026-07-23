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

DEFAULT_CONTINUOUS_CSV = os.path.join(RESULT_DIR, "continuous_fusion.csv")
DEFAULT_VISION_CSV = os.path.join(RESULT_DIR, "vision_position.csv")
DEFAULT_GNSS_CSV = os.path.join(RESULT_DIR, "extract_gnss_position.csv")
DEFAULT_ERROR_PNG = os.path.join(RESULT_DIR, "evalution_error.png")
DEFAULT_TRACK_PNG = os.path.join(RESULT_DIR, "evalution_track.png")

EARTH_RADIUS_M = 6378137.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare continuous_fusion.csv and vision_position against gnss_position, "
            "then draw error plots and flight tracks."
        )
    )
    parser.add_argument("--continuous_csv", default=DEFAULT_CONTINUOUS_CSV, help="Path to continuous fusion csv")
    parser.add_argument("--vision_csv", default=DEFAULT_VISION_CSV, help="Path to vision_position.csv")
    parser.add_argument("--gnss_csv", default=DEFAULT_GNSS_CSV, help="Path to gnss_position.csv")
    parser.add_argument("--error_png", default=DEFAULT_ERROR_PNG, help="Path to the error figure")
    parser.add_argument("--track_png", default=DEFAULT_TRACK_PNG, help="Path to the track figure")
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


def choose_key(fieldnames, candidates, csv_path, label):
    for key in candidates:
        if key in fieldnames:
            return key
    raise KeyError(
        "Missing {} columns {} in {}".format(label, ", ".join(candidates), csv_path)
    )


def read_csv_points(csv_path, stamp_key, lon_key, lat_key, alt_key):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("CSV not found: {}".format(csv_path))

    points = []
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        stamp_candidates = stamp_key if isinstance(stamp_key, (list, tuple)) else [stamp_key]
        lon_candidates = lon_key if isinstance(lon_key, (list, tuple)) else [lon_key]
        lat_candidates = lat_key if isinstance(lat_key, (list, tuple)) else [lat_key]
        alt_candidates = alt_key if isinstance(alt_key, (list, tuple)) else [alt_key]

        stamp_field = choose_key(fieldnames, stamp_candidates, csv_path, "stamp")
        lon_field = choose_key(fieldnames, lon_candidates, csv_path, "longitude")
        lat_field = choose_key(fieldnames, lat_candidates, csv_path, "latitude")
        alt_field = choose_key(fieldnames, alt_candidates, csv_path, "altitude")

        for row in reader:
            stamp = parse_float(row.get(stamp_field))
            lon = parse_float(row.get(lon_field))
            lat = parse_float(row.get(lat_field))
            alt = parse_float(row.get(alt_field))
            if not (math.isfinite(stamp) and math.isfinite(lon) and math.isfinite(lat) and math.isfinite(alt)):
                continue
            points.append((stamp, lon, lat, alt))

    points.sort(key=lambda item: item[0])
    if not points:
        raise ValueError("No valid rows found in {}".format(csv_path))
    return points


def interpolate_reference(ref_points, stamp):
    ref_stamps = [point[0] for point in ref_points]
    index = bisect.bisect_left(ref_stamps, stamp)

    if index < len(ref_points) and abs(ref_points[index][0] - stamp) < 1e-9:
        _, lon, lat, alt = ref_points[index]
        return lon, lat, alt

    if index == 0 or index >= len(ref_points):
        return None

    t0, lon0, lat0, alt0 = ref_points[index - 1]
    t1, lon1, lat1, alt1 = ref_points[index]
    if t1 <= t0:
        return None

    ratio = (stamp - t0) / (t1 - t0)
    lon = lon0 + ratio * (lon1 - lon0)
    lat = lat0 + ratio * (lat1 - lat0)
    alt = alt0 + ratio * (alt1 - alt0)
    return lon, lat, alt


def lon_lat_to_meter(sample_lon, sample_lat, ref_lon, ref_lat):
    dlon_rad = math.radians(sample_lon - ref_lon)
    dlat_rad = math.radians(sample_lat - ref_lat)
    lat_ref_rad = math.radians((sample_lat + ref_lat) * 0.5)
    lon_err_m = dlon_rad * EARTH_RADIUS_M * math.cos(lat_ref_rad)
    lat_err_m = dlat_rad * EARTH_RADIUS_M
    return lon_err_m, lat_err_m


def build_error_series(name, sample_points, gnss_points):
    start_stamp = max(sample_points[0][0], gnss_points[0][0])
    end_stamp = min(sample_points[-1][0], gnss_points[-1][0])
    if start_stamp >= end_stamp:
        raise ValueError("{} and gnss_position do not have a shared timestamp interval.".format(name))

    t = []
    lon_err_m = []
    lat_err_m = []
    height_err_m = []
    total_err_m = []
    track_lon = []
    track_lat = []
    gnss_track_lon = []
    gnss_track_lat = []

    for stamp, lon, lat, alt in sample_points:
        if stamp < start_stamp or stamp > end_stamp:
            continue
        ref = interpolate_reference(gnss_points, stamp)
        if ref is None:
            continue

        ref_lon, ref_lat, ref_alt = ref
        east_m, north_m = lon_lat_to_meter(lon, lat, ref_lon, ref_lat)
        up_m = alt - ref_alt
        total_m = math.sqrt(east_m * east_m + north_m * north_m + up_m * up_m)

        t.append(stamp - start_stamp)
        lon_err_m.append(east_m)
        lat_err_m.append(north_m)
        height_err_m.append(up_m)
        total_err_m.append(total_m)
        track_lon.append(lon)
        track_lat.append(lat)
        gnss_track_lon.append(ref_lon)
        gnss_track_lat.append(ref_lat)

    if not t:
        raise ValueError("No valid overlap samples after interpolation for {}".format(name))

    return {
        "name": name,
        "start_stamp": start_stamp,
        "end_stamp": end_stamp,
        "t": t,
        "lon_err_m": lon_err_m,
        "lat_err_m": lat_err_m,
        "height_err_m": height_err_m,
        "total_err_m": total_err_m,
        "track_lon": track_lon,
        "track_lat": track_lat,
        "gnss_track_lon": gnss_track_lon,
        "gnss_track_lat": gnss_track_lat,
    }


def mean_abs(values):
    return sum(abs(value) for value in values) / len(values)


def mean_value(values):
    return sum(values) / len(values)


def format_mean_text(series_list, key, title):
    lines = [title]
    for series in series_list:
        lines.append("{}: {:.3f} m".format(series["name"], mean_abs(series[key])))
    return "\n".join(lines)


def plot_error_figure(continuous_series, vision_series, output_png):
    ensure_parent_dir(output_png)
    series_list = [continuous_series, vision_series]

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    metric_specs = [
        ("lon_err_m", "Longitude Error (m)", "lon"),
        ("lat_err_m", "Latitude Error (m)", "lat"),
        ("height_err_m", "Height Error (m)", "height"),
        ("total_err_m", "Total Error (m)", "total"),
    ]
    styles = {
        continuous_series["name"]: {"color": "tab:blue", "linewidth": 1.2},
        vision_series["name"]: {"color": "tab:orange", "linewidth": 1.2},
    }

    for ax, (key, ylabel, title_key) in zip(axes, metric_specs):
        for series in series_list:
            style = styles[series["name"]]
            ax.plot(series["t"], series[key], label=series["name"], **style)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        mean_text = format_mean_text(series_list, key, "Mean |{} error|".format(title_key))
        ax.text(
            0.015,
            0.97,
            mean_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
        )

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time in shared interval (s)")
    fig.suptitle(
        "Continuous Fusion / Vision Position Error vs GNSS\n"
        "[{:.3f}, {:.3f}]".format(
            max(continuous_series["start_stamp"], vision_series["start_stamp"]),
            min(continuous_series["end_stamp"], vision_series["end_stamp"]),
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_track_figure(continuous_series, vision_series, output_png):
    ensure_parent_dir(output_png)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(
        continuous_series["gnss_track_lon"],
        continuous_series["gnss_track_lat"],
        label="gnss_position",
        linewidth=1.5,
        color="tab:green",
    )
    ax.plot(
        continuous_series["track_lon"],
        continuous_series["track_lat"],
        label="continuous_fusion",
        linewidth=1.2,
        color="tab:blue",
    )
    ax.plot(
        vision_series["track_lon"],
        vision_series["track_lat"],
        label="vision_position",
        linewidth=1.2,
        color="tab:orange",
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Flight Track")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def print_summary(series):
    print(
        "{}: samples={}, mean_abs_lon={:.3f} m, mean_abs_lat={:.3f} m, "
        "mean_abs_height={:.3f} m, mean_total={:.3f} m".format(
            series["name"],
            len(series["t"]),
            mean_abs(series["lon_err_m"]),
            mean_abs(series["lat_err_m"]),
            mean_abs(series["height_err_m"]),
            mean_value(series["total_err_m"]),
        )
    )


def main():
    args = parse_args()
    try:
        gnss_points = read_csv_points(
            args.gnss_csv,
            "fix_stamp_sec",
            ("gt_lon", "longitude"),
            ("gt_lat", "latitude"),
            ("gt_alt", "altitude"),
        )
        vision_points = read_csv_points(
            args.vision_csv, "image_stamp_sec", "camera_lon", "camera_lat", "camera_alt"
        )
        continuous_points = read_csv_points(args.continuous_csv, "stamp_sec", "lon", "lat", "alt")

        continuous_series = build_error_series("continuous_fusion", continuous_points, gnss_points)
        vision_series = build_error_series("vision_position", vision_points, gnss_points)

        plot_error_figure(continuous_series, vision_series, args.error_png)
        plot_track_figure(continuous_series, vision_series, args.track_png)

        print_summary(continuous_series)
        print_summary(vision_series)
        print("Saved error figure to {}".format(args.error_png))
        print("Saved track figure to {}".format(args.track_png))
    except Exception as exc:
        print("Failed to generate evaluation plots: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
