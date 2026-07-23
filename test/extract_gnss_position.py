#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import rosbag


DEFAULT_BAG = "/home/jasonxue/splg_lo_origin/18402.bag"
DEFAULT_TOPIC = "/fix"
DEFAULT_OUTPUT = "/home/jasonxue/splg_lo_origin/splg_fusion/result/extract_gnss_position.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract GNSS lon/lat/alt from a rosbag /fix topic into a CSV file."
    )
    parser.add_argument(
        "--bag",
        default=DEFAULT_BAG,
        help="Path to the input rosbag.",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="GNSS topic name to read, default is /fix.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to the output CSV file.",
    )
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def extract_gnss_to_csv(bag_path, topic_name, output_csv):
    if not os.path.isfile(bag_path):
        raise FileNotFoundError("Rosbag not found: {}".format(bag_path))

    ensure_parent_dir(output_csv)

    count = 0
    with rosbag.Bag(bag_path, "r") as bag, open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "fix_index",
                "fix_stamp_sec",
                "longitude",
                "latitude",
                "altitude",
            ]
        )

        for bag_message in bag.read_messages(topics=[topic_name]):
            msg = bag_message.message
            stamp_sec = msg.header.stamp.to_sec()
            writer.writerow(
                [
                    count + 1,
                    "{:.9f}".format(stamp_sec),
                    "{:.12f}".format(msg.longitude),
                    "{:.12f}".format(msg.latitude),
                    "{:.9f}".format(msg.altitude),
                ]
            )
            count += 1

    return count


def main():
    args = parse_args()

    try:
        count = extract_gnss_to_csv(args.bag, args.topic, args.output)
    except Exception as exc:
        print("Failed to extract GNSS positions: {}".format(exc), file=sys.stderr)
        return 1

    print("Wrote {} GNSS rows to {}".format(count, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
