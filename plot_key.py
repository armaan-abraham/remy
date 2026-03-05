#!/usr/bin/env python3
"""Extract all instances of key=value from a file and plot as a line plot."""

import argparse
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_values(filepath, key):
    pattern = re.compile(rf"{re.escape(key)}\s*=\s*([^\s,\])\}}&]+)")
    values = []
    with open(filepath) as f:
        for line in f:
            for match in pattern.finditer(line):
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    pass
    return values


def main():
    parser = argparse.ArgumentParser(description="Extract key=value from log files and plot.")
    parser.add_argument("ids", nargs="+", type=int, help="Job IDs to plot")
    parser.add_argument("key", help="Key to search for (e.g. 'score')")
    parser.add_argument("-p", "--path-template", default="/iris/u/armaana/jobs/logs/remy_{id}.out",
                        help="Path template with {id} placeholder (default: /iris/u/armaana/jobs/logs/remy_{id}).out")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory for plots")
    parser.add_argument("--ymin", type=float, default=None, help="Y-axis minimum")
    parser.add_argument("--ymax", type=float, default=None, help="Y-axis maximum")
    args = parser.parse_args()

    for job_id in args.ids:
        filepath = args.path_template.format(id=job_id)
        if not os.path.isfile(filepath):
            print(f"Skipping {job_id}: {filepath} not found")
            continue

        values = extract_values(filepath, args.key)
        if not values:
            print(f"No instances of '{args.key}=' found in {filepath}")
            continue

        print(f"[{job_id}] Found {len(values)} values for '{args.key}'")

        filename = f"{args.key}-{job_id}.png"
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out = os.path.join(args.output_dir, filename)
        else:
            out = filename
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(values)), values, marker="o", markersize=3, linewidth=1)
        plt.xlabel("Step")
        plt.ylabel(args.key)
        plt.title(f"{args.key} over time — job {job_id} ({len(values)} points)")
        if args.ymin is not None:
            plt.ylim(bottom=args.ymin)
        if args.ymax is not None:
            plt.ylim(top=args.ymax)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
