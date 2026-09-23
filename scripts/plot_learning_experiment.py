"""Export validation curves and test run means without selecting on the test."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/adaptive_atc_v1"))
    args = parser.parse_args()
    def read(name):
        return json.loads((args.results / name).read_text(encoding="utf-8"))
    with (args.results / "validation_grid.csv").open(encoding="utf-8") as f:
        grid = list(csv.DictReader(f))
    with (args.results / "training_history.csv").open(encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    selection = read("selection.json")
    summary = read("test_summary.json")
    result = summary["comparisons"]["normalized_tardiness"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    axes[0].plot([float(r["k"]) for r in grid],
                 [float(r["validation_normalized_tardiness"]) for r in grid], "o-")
    axes[0].axvline(selection["fixed_k"], color="#b83280", linestyle="--",
                    label=f"Selected k={selection['fixed_k']}")
    axes[0].set(title="Fixed ATC: validation grid", xlabel="k",
                ylabel="Mean normalized tardiness")
    axes[0].legend()
    for model in selection["models"]:
        points = [r for r in history if float(r["alpha"]) == selection["selected_alpha"]
                  and int(r["training_seed"]) == model["training_seed"]]
        axes[1].plot([int(p["episode"]) for p in points],
                     [float(p["validation_loss"]) for p in points],
                     marker=".", label=f"seed {model['training_seed']}")
    axes[1].axhline(selection["baseline_validation_loss"], color="black", linestyle="--",
                    label="Fixed ATC")
    axes[1].set(title="Q-learning: validation checkpoints", xlabel="Training episodes",
                ylabel="Mean normalized tardiness")
    axes[1].legend(fontsize=8)
    run_seeds = [m["training_seed"] for m in selection["models"]]
    axes[2].scatter(run_seeds, result["per_run_means"], color="#2b6cb0",
                     label="Selected Q checkpoints")
    axes[2].axhline(result["baseline_mean"], color="black", linestyle="--", label="Fixed ATC")
    axes[2].set(title="Closed test: mean of each training run", xlabel="Training seed",
                ylabel="Mean normalized tardiness")
    axes[2].set_xticks(run_seeds)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Adaptive ATC — lower is better; selection uses validation only", fontsize=14)
    fig.savefig(args.results / "learning_summary.png", dpi=160)
    fig.savefig(args.results / "learning_summary.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
