#!/usr/bin/env python3
"""
Run the GA with multiple AI matchups (round-robin over several AIs).

Unlike run_ga_local_test.py (single pair: lightRush vs workerRush), this script
evaluates each UTT across all pairs from AI_AGENTS. With 4 AIs you get 6 pairs;
with 5 AIs you get 10 pairs. Fitness (balance, duration, diversity) is computed
over all matchups so the evolved UTT is balanced across multiple strategies.

Run from project root:

  python scripts/GA\ and\ MAP-Elites/run_ga_multi_matchup.py

Or from this directory:

  python run_ga_multi_matchup.py
"""

import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from core.ga_algorithm import GAConfig
from core.ga_config_manager import ExperimentManager
from run_ga import run_genetic_algorithm
from run_ga_local_test import (
    GA_RUN_LOGS_DIR,
    RUN_HISTORY_CSV,
    Tee,
    _write_local_run_logs,
)

# -----------------------------------------------------------------------------
# Multi-matchup config: more AIs => more pairs (round-robin)
# n AIs → n*(n-1)/2 pairs. 6 AIs=15 pairs, 8 AIs=28 pairs.
# -----------------------------------------------------------------------------
AI_AGENTS = [
    # Rush
    "workerRushAI",
    "lightRushAI",
    "heavyRushAI",
    "rangedRushAI",
    # Balanced / strategic
    "coacAI",
    "naiveMCTSAI",
    # Defensive / economic
    "passiveAI",
    # Random baseline
    "randomBiasedAI",
]

# Tune these: more matchups = more games per individual (lower games_per_eval if runs are slow)
# Allow overriding via environment variables for cluster runs.
GENERATIONS = int(os.getenv("GA_GENERATIONS", "30"))
POPULATION = int(os.getenv("GA_POPULATION", "20"))
GAMES_PER_EVAL = int(os.getenv("GA_GAMES_PER_EVAL", "4"))  # per ordering per map
MAX_STEPS = int(os.getenv("GA_MAX_STEPS", "20000"))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ga_multi_matchup_output")
EXPERIMENT_NAME = "local_multi_ai_test"


def main():
    os.makedirs(GA_RUN_LOGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(GA_RUN_LOGS_DIR, f"{EXPERIMENT_NAME}_{ts}.log")
    tee = Tee(sys.stdout, log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        return _main(ts, log_path)
    finally:
        sys.stdout = old_stdout
        tee.close()


def _main(ts: str, log_path: str):
    MAP_PATHS = ["maps/8x8/basesWorkers8x8A.xml"]
    n_pairs = len(AI_AGENTS) * (len(AI_AGENTS) - 1) // 2

    print("=" * 60)
    print("GA multi-matchup: round-robin over multiple AIs")
    print("=" * 60)
    print(f"  AI agents:      {AI_AGENTS}")
    print(f"  Matchup pairs:  {n_pairs}")
    print(f"  Log file:       {log_path}")
    print(f"  Generations:    {GENERATIONS}")
    print(f"  Population:    {POPULATION}")
    print(f"  Games per eval: {GAMES_PER_EVAL} × {len(MAP_PATHS)} map(s) (per pair, both orderings)")
    print(f"  Max steps:     {MAX_STEPS}")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print("=" * 60)

    config = GAConfig(
        population_size=POPULATION,
        generations=GENERATIONS,
        crossover_rate=0.6,
        mutation_rate=0.3,
        use_real_microrts=True,
        use_working_evaluator=True,
        max_steps=MAX_STEPS,
        map_path=MAP_PATHS[0],
        map_paths=MAP_PATHS,
        games_per_evaluation=GAMES_PER_EVAL,
        ai_agents=AI_AGENTS,
        fitness_alpha=0.7,
        fitness_beta=0.15,
        fitness_gamma=0.15,
        target_duration=1000,
        duration_tolerance=600,
        max_generations_without_improvement=GENERATIONS,
        random_immigrant_interval=5,
        use_nondeterministic=True,
        use_both_orderings=True,
        verbose=True,
    )

    experiment_manager = ExperimentManager(OUTPUT_DIR)
    results, experiment_dir = run_genetic_algorithm(
        config,
        experiment_manager,
        EXPERIMENT_NAME,
        checkpoint_dir=None,
        resume_from=None,
    )

    print()
    print("=" * 60)
    print("Multi-matchup run finished")
    print("=" * 60)
    print(f"  Best fitness:   {results.best_fitness.overall_fitness:.4f}")
    print(f"  Balance:        {results.best_fitness.balance:.4f}")
    print(f"  Duration:       {results.best_fitness.duration:.4f}")
    print(f"  Diversity:      {results.best_fitness.strategy_diversity:.4f}")
    print(f"  Results saved: {experiment_dir}")
    _write_local_run_logs(results, config, experiment_dir, EXPERIMENT_NAME, ts=ts)
    print(f"  Full run log:  {log_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
