#!/usr/bin/env python3
"""
Plot per-generation fitness for a single GA run.

Given a run folder like:

  scripts/GA and MAP-Elites/ga_run_logs/runs/local_two_ai_test_YYYYMMDD_HHMMSS/

this script reads generations.csv and produces:
  - fitness_over_generations.png   (best vs avg fitness)
  - components_over_generations.png (balance, duration, diversity)

Usage (from project root):

  python scripts/GA\ and\ MAP-Elites/plot_ga_run_fitness.py \
      scripts/GA\ and\ MAP-Elites/ga_run_logs/runs/local_two_ai_test_YYYYMMDD_HHMMSS
"""

import csv
import os
import sys
from typing import List, Dict


def read_generations_csv(gen_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(gen_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "generation": int(row["generation"]),
                    "best_fitness": float(row["best_fitness"]),
                    "avg_fitness": float(row["avg_fitness"]),
                    "best_balance": float(row["best_balance"]),
                    "best_duration": float(row["best_duration"]),
                    "best_diversity": float(row["best_diversity"]),
                }
            )
    return rows


def plot_fitness(run_dir: str) -> None:
    gen_path = os.path.join(run_dir, "generations.csv")
    if not os.path.exists(gen_path):
        print(f"ERROR: generations.csv not found in {run_dir}")
        sys.exit(1)

    data = read_generations_csv(gen_path)
    gens = [d["generation"] for d in data]

    best_f = [d["best_fitness"] for d in data]
    avg_f = [d["avg_fitness"] for d in data]

    best_b = [d["best_balance"] for d in data]
    best_d = [d["best_duration"] for d in data]
    best_div = [d["best_diversity"] for d in data]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"ERROR: Could not import matplotlib: {e}")
        sys.exit(1)

    # 1) Best vs avg fitness
    plt.figure(figsize=(8, 5))
    plt.plot(gens, best_f, "b-o", label="Best fitness", markersize=4)
    plt.plot(gens, avg_f, "g-s", label="Avg fitness", markersize=4)
    for g, f in zip(gens, best_f):
        plt.annotate(str(g), (g, f), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best & Avg Fitness per Generation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(run_dir, "fitness_over_generations.png")
    plt.savefig(out1, dpi=120)
    plt.close()
    print(f"Wrote {out1}")

    # 2) Components per generation
    plt.figure(figsize=(8, 5))
    plt.plot(gens, best_b, "r-o", label="Balance", markersize=4)
    plt.plot(gens, best_d, "c-s", label="Duration", markersize=4)
    plt.plot(gens, best_div, "m-^", label="Diversity", markersize=4)
    for g, b in zip(gens, best_b):
        plt.annotate(str(g), (g, b), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)
    plt.xlabel("Generation")
    plt.ylabel("Component score")
    plt.title("Best Balance / Duration / Diversity per Generation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(run_dir, "components_over_generations.png")
    plt.savefig(out2, dpi=120)
    plt.close()
    print(f"Wrote {out2}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_ga_run_fitness.py <run_dir>")
        print("Example:")
        print("  python scripts/GA\\ and\\ MAP-Elites/plot_ga_run_fitness.py "
              "scripts/GA\\ and\\ MAP-Elites/ga_run_logs/runs/local_two_ai_test_YYYYMMDD_HHMMSS")
        sys.exit(1)

    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    plot_fitness(run_dir)


if __name__ == "__main__":
    main()

