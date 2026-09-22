"""Create a static, exportable comparison figure from benchmark output."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/benchmark"))
    args = parser.parse_args()
    with (args.results / "runs.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    methods = ["FIFO", "SPT", "MWKR", "AUCTION_ORIGINAL"]
    names = ["FIFO", "SPT", "MWKR", "Auction\noriginal"]
    colors = ["#718096", "#3182ce", "#2f855a", "#805ad5"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, instance in zip(axes.flat, ["ft10", "la31", "ta51", "ta71"]):
        group = {r["method"]: r for r in rows
                 if r["instance"] == instance and r["scenario"] == "static"}
        vals = [float(group[m]["makespan"]) for m in methods]
        bars = ax.bar(names, vals, color=colors)
        ax.bar_label(bars, fmt="%.0f", padding=3)
        optimum = group["FIFO"]["static_optimum"]
        if optimum:
            ax.axhline(float(optimum), color="#c53030", linestyle="--", label="Known optimum")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{instance} — {group['FIFO']['jobs']} jobs x {group['FIFO']['machines']} machines")
        ax.set_ylabel("Makespan (lower is better)")
        ax.set_ylim(0, max(vals) * 1.23)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Static JSPLIB: original auction equals MWKR without due dates", fontsize=13)
    fig.savefig(args.results / "static_comparison.png", dpi=160)
    fig.savefig(args.results / "static_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
