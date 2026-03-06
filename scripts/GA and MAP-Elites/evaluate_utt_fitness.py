#!/usr/bin/env python3
"""
Evaluate a single UTT JSON file with the GA fitness function (no evolution).

This script runs one pass of the WorkingGAEvaluator on a given UTT and prints
its balance / duration / diversity / overall_fitness across the configured AI set.

Usage (from project root):

  python scripts/GA\ and\ MAP-Elites/evaluate_utt_fitness.py \
      scripts/GA\ and\ MAP-Elites/ga_run_logs/runs/local_two_ai_test_YYYYMMDD_HHMMSS/utt_log/gen0_ind0.json

You can also pass a UTT in the Java folder, e.g.:

  python scripts/GA\ and\ MAP-Elites/evaluate_utt_fitness.py \
      gym_microrts/microrts/utts/gen0_ind0.json
"""

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from core.ga_working_evaluator import WorkingGAEvaluator  # type: ignore  # noqa: E402
from core.ga_chromosome import MicroRTSChromosome         # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a single UTT JSON with GA fitness.")
    p.add_argument("utt_json", help="Path to UTT JSON file (relative or absolute).")
    p.add_argument(
        "--map",
        default="maps/8x8/basesWorkers8x8A.xml",
        help="Map path to use for evaluation (default: %(default)s)",
    )
    p.add_argument(
        "--games-per-eval",
        type=int,
        default=10,
        help="Games per evaluation per pair per ordering (default: %(default)s)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="Max steps per game (default: %(default)s)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    utt_path = Path(args.utt_json)
    if not utt_path.is_file():
        print(f"ERROR: UTT file not found: {utt_path}")
        return 1

    with utt_path.open("r", encoding="utf-8") as f:
        json_str = f.read()

    chrom = MicroRTSChromosome.from_json(json_str)

    # Use the same AI set and weights as run_ga_local_test.py by default
    ai_agents = [
        # Fully observable rush/baselines
        "lightRushAI",
        "workerRushAI",
        "heavyRushAI",
        "rangedRushAI",
        # Partially observable rush variants
        "POHeavyRush",
        "POLightRush",
        "PORangedRush",
        "POWorkerRush",
    ]

    print("Evaluating UTT fitness (single run, no evolution)")
    print("================================================")
    print(f"UTT file: {utt_path}")
    print(f"Map:      {args.map}")
    print(f"Games:    {args.games_per_eval} per pair per ordering")
    print(f"Max steps per game: {args.max_steps}")
    print(f"AI agents: {ai_agents}")
    print("================================================\n")

    evaluator = WorkingGAEvaluator(
        alpha=0.5,        # balance weight
        beta=0.3,         # duration weight
        gamma=0.2,        # diversity weight
        max_steps=args.max_steps,
        map_path=args.map,
        map_paths=[args.map],
        games_per_eval=args.games_per_eval,
        ai_agents=ai_agents,
        use_nondeterministic=True,
        use_both_orderings=True,
        target_duration=100,
        duration_tolerance=80,
    )

    fitness = evaluator.evaluate_chromosome(chrom)

    print("Fitness components:")
    print(f"  Balance:   {fitness.balance:.4f}")
    print(f"  Duration:  {fitness.duration:.4f}")
    print(f"  Diversity: {fitness.strategy_diversity:.4f}")
    print(f"  Overall:   {fitness.overall_fitness:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

