#!/usr/bin/env python3

import argparse
import bisect
import collections
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
DEFAULT_FUSION_CSV = os.path.join(RESULT_DIR, "fusion_position.csv")

DEFAULT_CONTINUOUS_DIAG_PNG = os.path.join(RESULT_DIR, "continuous_fusion_diagnostics.png")
DEFAULT_FUSION_ERROR_PNG = os.path.join(RESULT_DIR, "fusion_position_error_plot.png")
DEFAULT_FUSION_TRACK_PNG = os.path.join(RESULT_DIR, "fusion_position_track_plot.png")

EARTH_RADIUS_M = 6378137.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze continuous_fusion.csv for fusion health and evaluate "
            "fusion_position.csv against GNSS."
        )
    )
    parser.add_argument("--gnss_csv", default=DEFAULT_GNSS_CSV, help="Path to extract_gnss_position.csv")
    parser.add_argument(
        "--continuous_csv", default=DEFAULT_CONTINUOUS_CSV, help="Path to continuous_fusion.csv"
    )
    parser.add_argument("--fusion_csv", default=DEFAULT_FUSION_CSV, help="Path to fusion_position.csv")
    parser.add_argument(
        "--continuous_diag_png",
        default=DEFAULT_CONTINUOUS_DIAG_PNG,
        help="Path to the continuous fusion diagnostics plot",
    )
    parser.add_argument(
        "--fusion_error_png",
        default=DEFAULT_FUSION_ERROR_PNG,
        help="Path to the fusion error plot against GNSS",
    )
    parser.add_argument(
        "--fusion_track_png",
        default=DEFAULT_FUSION_TRACK_PNG,
        help="Path to the fusion and GNSS track comparison plot",
    )
    parser.add_argument(
        "--thresholds",
        default="1,2,3,5,10",
        help="Comma separated fusion distance thresholds in meters, e.g. 1,2,3,5,10",
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


def parse_int(text, default=-1):
    value = parse_float(text)
    if not math.isfinite(value):
        return default
    return int(round(value))


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


def read_csv_dicts(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("CSV not found: {}".format(csv_path))
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError("CSV has no header: {}".format(csv_path))
    return rows


def read_csv_points(csv_path, stamp_key, lon_key, lat_key):
    rows = read_csv_dicts(csv_path)
    points = []
    for row in rows:
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
    return east_m, north_m, math.hypot(east_m, north_m)


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


def build_continuous_diagnostics(rows):
    diag_rows = []
    for row in rows:
        stamp = parse_float(row.get("stamp_sec"))
        lon = parse_float(row.get("lon"))
        lat = parse_float(row.get("lat"))
        if not (math.isfinite(stamp) and math.isfinite(lon) and math.isfinite(lat)):
            continue

        cov_xx = parse_float(row.get("cov_xx_m2"))
        cov_yy = parse_float(row.get("cov_yy_m2"))
        cov_zz = parse_float(row.get("cov_zz_m2"))
        sigma_x = math.sqrt(max(0.0, cov_xx)) if math.isfinite(cov_xx) else math.nan
        sigma_y = math.sqrt(max(0.0, cov_yy)) if math.isfinite(cov_yy) else math.nan
        sigma_z = math.sqrt(max(0.0, cov_zz)) if math.isfinite(cov_zz) else math.nan

        diag_rows.append(
            {
                "stamp": stamp,
                "source": (row.get("source") or "").strip() or "unknown",
                "speed_mps": parse_float(row.get("speed_mps")),
                "propagate_dt_sec": parse_float(row.get("propagate_dt_sec")),
                "visual_update_applied": parse_int(row.get("visual_update_applied"), default=0),
                "visual_nis": parse_float(row.get("visual_nis")),
                "visual_cov_scale": parse_float(row.get("visual_cov_scale")),
                "visual_gate_passed": parse_int(row.get("visual_gate_passed"), default=-1),
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "sigma_z": sigma_z,
            }
        )

    if not diag_rows:
        raise ValueError("No valid rows found in continuous fusion csv.")

    diag_rows.sort(key=lambda item: item["stamp"])
    source_counter = collections.Counter(row["source"] for row in diag_rows)
    visual_rows = [row for row in diag_rows if row["visual_gate_passed"] in (0, 1)]
    nis_values = [row["visual_nis"] for row in visual_rows if math.isfinite(row["visual_nis"])]
    update_count = sum(1 for row in diag_rows if row["visual_update_applied"] == 1)
    gate_fail_count = sum(1 for row in visual_rows if row["visual_gate_passed"] == 0)

    return {
        "rows": diag_rows,
        "source_counter": source_counter,
        "total_count": len(diag_rows),
        "update_count": update_count,
        "visual_row_count": len(visual_rows),
        "gate_fail_count": gate_fail_count,
        "nis_values": nis_values,
    }


def plot_continuous_diagnostics(diag, output_png):
    ensure_parent_dir(output_png)
    rows = diag["rows"]
    start_stamp = rows[0]["stamp"]
    t = [row["stamp"] - start_stamp for row in rows]
    source_names = list(diag["source_counter"].keys())
    source_to_index = {name: idx for idx, name in enumerate(source_names)}
    source_y = [source_to_index[row["source"]] for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    axes[0].scatter(t, source_y, s=8, alpha=0.8)
    axes[0].set_yticks(list(source_to_index.values()))
    axes[0].set_yticklabels(source_names)
    axes[0].set_ylabel("source")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(t, [row["speed_mps"] for row in rows], label="speed_mps", linewidth=1.2)
    axes[1].plot(
        t,
        [row["propagate_dt_sec"] for row in rows],
        label="propagate_dt_sec",
        linewidth=1.0,
    )
    axes[1].set_ylabel("speed / dt")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(t, [row["sigma_x"] for row in rows], label="sigma_x", linewidth=1.2)
    axes[2].plot(t, [row["sigma_y"] for row in rows], label="sigma_y", linewidth=1.2)
    axes[2].plot(t, [row["sigma_z"] for row in rows], label="sigma_z", linewidth=1.2)
    axes[2].set_ylabel("sigma (m)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    axes[3].plot(t, [row["visual_nis"] for row in rows], label="visual_nis", linewidth=1.0)
    axes[3].plot(
        t,
        [row["visual_cov_scale"] for row in rows],
        label="visual_cov_scale",
        linewidth=1.0,
    )
    gate_fail_t = [row["stamp"] - start_stamp for row in rows if row["visual_gate_passed"] == 0]
    gate_fail_nis = [row["visual_nis"] for row in rows if row["visual_gate_passed"] == 0]
    if gate_fail_t:
        axes[3].scatter(gate_fail_t, gate_fail_nis, label="gate_failed", s=12, color="red")
    axes[3].set_ylabel("nis / cov_scale")
    axes[3].set_xlabel("time from first continuous sample (s)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Continuous Fusion Diagnostics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def build_fusion_error_series(fusion_points, gnss_points):
    start_stamp = max(gnss_points[0][0], fusion_points[0][0])
    end_stamp = min(gnss_points[-1][0], fusion_points[-1][0])
    if start_stamp >= end_stamp:
        raise ValueError("Fusion and GNSS do not have a shared timestamp interval.")

    stamps_rel = []
    east_err_m = []
    north_err_m = []
    horizontal_err_m = []
    fusion_track_lon = []
    fusion_track_lat = []
    gnss_track_lon = []
    gnss_track_lat = []

    for stamp, lon, lat in fusion_points:
        if stamp < start_stamp or stamp > end_stamp:
            continue
        ref = interpolate_gnss(gnss_points, stamp)
        if ref is None:
            continue
        ref_lon, ref_lat = ref
        east_m, north_m, horizontal_m = lon_lat_error_to_meter(lon, lat, ref_lon, ref_lat)
        stamps_rel.append(stamp - start_stamp)
        east_err_m.append(east_m)
        north_err_m.append(north_m)
        horizontal_err_m.append(horizontal_m)
        fusion_track_lon.append(lon)
        fusion_track_lat.append(lat)
        gnss_track_lon.append(ref_lon)
        gnss_track_lat.append(ref_lat)

    if not stamps_rel:
        raise ValueError("No valid fusion samples inside the shared GNSS interval.")

    return {
        "start_stamp": start_stamp,
        "end_stamp": end_stamp,
        "t": stamps_rel,
        "east_err_m": east_err_m,
        "north_err_m": north_err_m,
        "horizontal_err_m": horizontal_err_m,
        "fusion_track_lon": fusion_track_lon,
        "fusion_track_lat": fusion_track_lat,
        "gnss_track_lon": gnss_track_lon,
        "gnss_track_lat": gnss_track_lat,
    }


def plot_fusion_error(series, output_png):
    ensure_parent_dir(output_png)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(series["t"], series["east_err_m"], linewidth=1.2, label="east error")
    axes[0].set_ylabel("east err (m)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right")

    axes[1].plot(series["t"], series["north_err_m"], linewidth=1.2, label="north error")
    axes[1].set_ylabel("north err (m)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right")

    axes[2].plot(series["t"], series["horizontal_err_m"], linewidth=1.3, label="distance error")
    axes[2].set_ylabel("distance err (m)")
    axes[2].set_xlabel("time in shared interval (s)")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="upper right")

    fig.suptitle(
        "Fusion Position vs GNSS Error [{:.3f}, {:.3f}]".format(
            series["start_stamp"], series["end_stamp"]
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_fusion_track(series, output_png):
    ensure_parent_dir(output_png)
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.plot(series["gnss_track_lon"], series["gnss_track_lat"], label="GNSS", linewidth=1.3)
    ax.plot(
        series["fusion_track_lon"],
        series["fusion_track_lat"],
        label="fusion_position",
        linewidth=1.1,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Fusion Position Track vs GNSS")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def print_continuous_summary(diag):
    print("=== Continuous Fusion Diagnostics ===")
    print("Total continuous rows: {}".format(diag["total_count"]))
    print("Source counts:")
    for source_name, count in diag["source_counter"].most_common():
        print("  {} : {}".format(source_name, count))
    print("Visual updates applied: {}".format(diag["update_count"]))
    print("Visual rows with gate info: {}".format(diag["visual_row_count"]))
    print("Visual gate failed rows: {}".format(diag["gate_fail_count"]))
    if diag["visual_row_count"] > 0:
        gate_fail_ratio = 100.0 * diag["gate_fail_count"] / diag["visual_row_count"]
        print("Visual gate failed ratio: {:.2f}%".format(gate_fail_ratio))
    if diag["nis_values"]:
        nis_stats = compute_stats(diag["nis_values"])
        print(
            "Visual NIS: count={count}, mean={mean:.3f}, rmse={rmse:.3f}, max_abs={max_abs:.3f}".format(
                **nis_stats
            )
        )
    else:
        print("Visual NIS: no valid values")

    print("Quick check:")
    if diag["update_count"] == 0:
        print("  Warning: no visual update was applied in continuous fusion.")
    elif diag["visual_row_count"] > 0 and diag["gate_fail_count"] == diag["visual_row_count"]:
        print("  Warning: all visual rows failed the gate.")
    else:
        print("  Continuous fusion contains IMU prediction rows and usable visual updates.")


def print_fusion_summary(series, thresholds):
    stats = compute_stats(series["horizontal_err_m"])
    threshold_stats = compute_threshold_stats(series["horizontal_err_m"], thresholds)
    print("=== Fusion Position Evaluation ===")
    print(
        "Shared stamp interval: [{:.6f}, {:.6f}]".format(
            series["start_stamp"], series["end_stamp"]
        )
    )
    print(
        "Fusion distance error: count={count}, mean={mean:.3f} m, rmse={rmse:.3f} m, max_abs={max_abs:.3f} m".format(
            **stats
        )
    )
    print("Fusion threshold statistics:")
    for item in threshold_stats:
        print(
            "  error < {:.3f} m : {:.2f}% ({}/{})".format(
                item["threshold"], item["ratio"], item["count"], stats["count"]
            )
        )


def main():
    args = parse_args()
    try:
        thresholds = parse_thresholds(args.thresholds)

        continuous_rows = read_csv_dicts(args.continuous_csv)
        continuous_diag = build_continuous_diagnostics(continuous_rows)
        plot_continuous_diagnostics(continuous_diag, args.continuous_diag_png)

        gnss_points = read_csv_points(args.gnss_csv, "fix_stamp_sec", "longitude", "latitude")
        fusion_points = read_csv_points(args.fusion_csv, "fix_stamp_sec", "pred_lon", "pred_lat")
        fusion_series = build_fusion_error_series(fusion_points, gnss_points)
        plot_fusion_error(fusion_series, args.fusion_error_png)
        plot_fusion_track(fusion_series, args.fusion_track_png)

        print_continuous_summary(continuous_diag)
        print("Saved continuous diagnostics plot to {}".format(args.continuous_diag_png))
        print_fusion_summary(fusion_series, thresholds)
        print("Saved fusion error plot to {}".format(args.fusion_error_png))
        print("Saved fusion track plot to {}".format(args.fusion_track_png))
    except Exception as exc:
        print("Failed to analyze fusion outputs: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
