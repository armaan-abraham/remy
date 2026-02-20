#!/usr/bin/env python3
"""Extract all instances of key=value from a file and plot as a line plot."""

import argparse
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
    parser = argparse.ArgumentParser(description="Extract key=value from a file and plot.")
    parser.add_argument("filepath", help="Path to the log file")
    parser.add_argument("key", help="Key to search for (e.g. 'score')")
    parser.add_argument("-o", "--output", default=None, help="Output image path (default: <key>.png)")
    args = parser.parse_args()

    values = extract_values(args.filepath, args.key)
    print(values)
    if not values:
        print(f"No instances of '{args.key}=' found in {args.filepath}")
        return

    print(f"Found {len(values)} values for '{args.key}'")

    out = args.output or f"{args.key}.png"
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(values)), values, marker="o", markersize=3, linewidth=1)
    plt.xlabel("Step")
    plt.ylabel(args.key)
    plt.title(f"{args.key} over time ({len(values)} points)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
