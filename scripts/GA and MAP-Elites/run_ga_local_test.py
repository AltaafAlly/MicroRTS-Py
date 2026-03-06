#!/usr/bin/env python3
"""
Run the GA locally with a single matchup (RangedRush vs WorkerRush) for quick testing.
Uses a deterministic rush that actually plays; Ranged is often weaker than Light vs WorkerRush so balance can vary with UTT.

Uses a small population and few generations so it finishes in a few minutes.
Logs are written to ga_run_logs/: full terminal output as a .log file, plus CSV run history
and per-run generations CSV (not used on cluster).

Run from project root:

  python scripts/GA\ and\ MAP-Elites/run_ga_local_test.py

Or from this directory (scripts/GA and MAP-Elites):

  python run_ga_local_test.py
"""

import csv
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Project root (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)  # so "core" is GA's core package
sys.path.insert(0, PROJECT_ROOT)

from core.ga_algorithm import MicroRTSGeneticAlgorithm, GAConfig, GAResults
from core.ga_chromosome import MicroRTSChromosome
from core.ga_config_manager import ExperimentManager
from run_ga import run_genetic_algorithm

# Unit type -> which AI agents are most affected (for UTT change reporting)
UNIT_TYPE_AFFECTS_AI = {
    "Worker": ["workerRushAI"],
    "Light": ["lightRushAI"],
    "Heavy": ["heavyRushAI"],
    "Ranged": ["rangedRushAI"],
    "Barracks": ["workerRushAI", "lightRushAI"],
    "Base": ["workerRushAI", "lightRushAI"],
    "Resource": [],
}

# Local-only: folder for CSV run logs and full run output (compare local test runs)
GA_RUN_LOGS_DIR = os.path.join(SCRIPT_DIR, "ga_run_logs")
RUN_HISTORY_CSV = "run_history.csv"


class Tee:
    """Write to both stdout and a file so the full run is logged."""
    def __init__(self, stream, filepath):
        self.stream = stream
        self.filepath = filepath
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

# Local test config (override with GA_GENERATIONS, GA_POPULATION, GA_GAMES_PER_EVAL if needed)
GENERATIONS = int(os.getenv("GA_GENERATIONS", "25"))
POPULATION = int(os.getenv("GA_POPULATION", "10"))
GAMES_PER_EVAL = int(os.getenv("GA_GAMES_PER_EVAL", "8"))  # per map, per ordering
MAX_STEPS = 20000    # Cap per game; decisive games end in hundreds, draws stop here
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ga_local_test_output")
EXPERIMENT_NAME = "local_two_ai_test"
# Save per-game snapshots and match_outputs/*.txt — change to False for faster runs, less disk
SAVE_GAME_DETAILS = False
if "GA_SAVE_GAME_DETAILS" in os.environ:
    SAVE_GAME_DETAILS = os.environ["GA_SAVE_GAME_DETAILS"].strip().lower() in ("1", "true", "yes")


# Param order for UTT export
_UTT_PARAMS = ["cost", "produceTime", "hp", "minDamage", "maxDamage", "attackRange",
               "attackTime", "moveTime", "sightRadius", "harvestTime", "returnTime", "harvestAmount"]


def _default_utt_value(unit_type: str, param_name: str):
    """Default value for (unit_type, param) from PARAMETER_BOUNDS midpoint."""
    bounds = MicroRTSChromosome.PARAMETER_BOUNDS.get(unit_type, {}).get(param_name)
    if bounds is None:
        return 1
    lo, hi = bounds
    return (lo + hi) // 2


def _utt_diff(
    old_chrom: MicroRTSChromosome,
    new_chrom: MicroRTSChromosome,
    to_generation: int,
) -> List[Dict[str, Any]]:
    """Compare two chromosomes and return list of (unit_type, param, old_val, new_val, affects_ai)."""
    rows = []
    for unit_type in MicroRTSChromosome.UNIT_TYPES:
        u_old = old_chrom.unit_params.get(unit_type)
        u_new = new_chrom.unit_params.get(unit_type)
        if u_old is None or u_new is None:
            continue
        affects = UNIT_TYPE_AFFECTS_AI.get(unit_type, [])
        affects_ai = ";".join(affects) if affects else "all"
        for attr in _UTT_PARAMS:
            if not hasattr(u_old, attr) or not hasattr(u_new, attr):
                continue
            v_old = getattr(u_old, attr)
            v_new = getattr(u_new, attr)
            if v_old != v_new:
                rows.append({
                    "generation": to_generation,
                    "unit_type": unit_type,
                    "param_name": attr,
                    "old_value": v_old,
                    "new_value": v_new,
                    "affects_ai": affects_ai,
                })
    return rows


def _utt_gen0_rows(chrom: MicroRTSChromosome, generation: int = 0, only_diffs: bool = True) -> List[Dict[str, Any]]:
    """Gen 0 best: old_value = default (midpoint), new_value = evolved. If only_diffs, list only params that changed."""
    rows = []
    for unit_type in MicroRTSChromosome.UNIT_TYPES:
        u = chrom.unit_params.get(unit_type)
        if u is None:
            continue
        affects = UNIT_TYPE_AFFECTS_AI.get(unit_type, [])
        affects_ai = ";".join(affects) if affects else "all"
        for attr in _UTT_PARAMS:
            if not hasattr(u, attr):
                continue
            default_val = _default_utt_value(unit_type, attr)
            evolved_val = getattr(u, attr)
            if only_diffs and default_val == evolved_val:
                continue
            rows.append({
                "generation": generation,
                "unit_type": unit_type,
                "param_name": attr,
                "old_value": default_val,
                "new_value": evolved_val,
                "affects_ai": affects_ai,
            })
    return rows


def _write_best_utt_summary(run_dir: str, results: GAResults, config: GAConfig) -> None:
    """Write best_utt_summary.txt and best_utt_config.json: why it was best, full config, diff from default."""
    best = getattr(results, "best_individual", None)
    if best is None:
        return
    best_path = os.path.join(run_dir, "best_utt_summary.txt")
    json_path = os.path.join(run_dir, "best_utt_config.json")
    lines = [
        "Best UTT (final generation)",
        "=" * 60,
        "",
        "Why this config was best:",
        f"  Overall fitness:  {results.best_fitness.overall_fitness:.4f}",
        f"  Balance:          {results.best_fitness.balance:.4f}",
        f"  Duration:         {results.best_fitness.duration:.4f}",
        f"  Diversity:        {results.best_fitness.strategy_diversity:.4f}",
        "",
        "How it differs from default (default = midpoint of parameter bounds):",
        "-" * 60,
    ]
    diffs = []
    for unit_type in MicroRTSChromosome.UNIT_TYPES:
        u = best.unit_params.get(unit_type)
        if u is None:
            continue
        for attr in _UTT_PARAMS:
            if not hasattr(u, attr):
                continue
            default_val = _default_utt_value(unit_type, attr)
            evolved_val = getattr(u, attr)
            if default_val != evolved_val:
                aff = UNIT_TYPE_AFFECTS_AI.get(unit_type, [])
                aff_str = ",".join(aff) if aff else "all"
                diffs.append((unit_type, attr, default_val, evolved_val, aff_str))
    if not diffs:
        lines.append("  (No parameters differ from default.)")
    else:
        for unit_type, attr, default_val, evolved_val, aff_str in diffs:
            lines.append(f"  {unit_type}.{attr}:  {default_val} -> {evolved_val}  (affects: {aff_str})")
    lines.extend([
        "",
        "Full best UTT (all parameters):",
        "-" * 60,
    ])
    for unit_type in MicroRTSChromosome.UNIT_TYPES:
        u = best.unit_params.get(unit_type)
        if u is None:
            continue
        lines.append(f"\n  {unit_type}:")
        for attr in _UTT_PARAMS:
            if hasattr(u, attr):
                lines.append(f"    {attr}: {getattr(u, attr)}")
    with open(best_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(best.to_json())


def _write_local_run_logs(
    results: GAResults,
    config: GAConfig,
    experiment_dir: str,
    experiment_name: str,
    ts: Optional[str] = None,
) -> str:
    """Write all run outputs into a single folder ga_run_logs/runs/<experiment>_<ts>/ and return its path."""
    os.makedirs(GA_RUN_LOGS_DIR, exist_ok=True)
    if ts is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ai_agents = ",".join(config.ai_agents) if config.ai_agents else ""

    # Run folder: one folder per run so everything is sorted and stored in one place
    run_id = f"{experiment_name}_{ts}"
    run_dir = os.path.join(GA_RUN_LOGS_DIR, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    save_game_details = getattr(config, "save_game_details", True)
    match_outputs_dir = os.path.join(run_dir, "match_outputs")
    if save_game_details:
        os.makedirs(match_outputs_dir, exist_ok=True)

    # 1) Append one row to run_history.csv (top-level) for easy comparison across runs
    history_path = os.path.join(GA_RUN_LOGS_DIR, RUN_HISTORY_CSV)
    write_header = not os.path.exists(history_path)
    with open(history_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "experiment_name", "generations", "population_size", "games_per_eval", "max_steps",
                "ai_agents", "best_fitness", "best_balance", "best_duration", "best_diversity",
                "total_time_sec", "experiment_dir"
            ])
        w.writerow([
            ts, experiment_name, config.generations, config.population_size, config.games_per_evaluation,
            config.max_steps, ai_agents,
            round(results.best_fitness.overall_fitness, 4), round(results.best_fitness.balance, 4),
            round(results.best_fitness.duration, 4), round(results.best_fitness.strategy_diversity, 4),
            round(results.total_time, 2), experiment_dir
        ])

    # 2) Per-run generations CSV (fitness over generations) — inside run folder
    gen_path = os.path.join(run_dir, "generations.csv")
    with open(gen_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_fitness", "avg_fitness", "best_balance", "best_duration", "best_diversity", "time_elapsed"])
        for s in results.generation_stats:
            w.writerow([
                s.generation, round(s.best_fitness, 4), round(s.avg_fitness, 4),
                round(s.best_balance, 4), round(s.best_duration, 4), round(s.best_diversity, 4),
                round(s.time_elapsed, 2)
            ])

    # 3) UTT changes — only real value changes: gen 0 vs default (midpoint), then gen-to-gen diffs
    utt_changes_path = os.path.join(run_dir, "utt_changes.csv")
    utt_changes_rows = []
    best_per_gen = getattr(results, "best_individual_per_generation", None)
    if best_per_gen and len(best_per_gen) > 0:
        gen0, chrom0 = best_per_gen[0]
        utt_changes_rows.extend(_utt_gen0_rows(chrom0, generation=gen0, only_diffs=True))
        for i in range(1, len(best_per_gen)):
            gen_prev, chrom_prev = best_per_gen[i - 1]
            gen_curr, chrom_curr = best_per_gen[i]
            utt_changes_rows.extend(_utt_diff(chrom_prev, chrom_curr, to_generation=gen_curr))
    with open(utt_changes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "unit_type", "param_name", "old_value", "new_value", "affects_ai"])
        for r in utt_changes_rows:
            w.writerow([r["generation"], r["unit_type"], r["param_name"], r["old_value"], r["new_value"], r["affects_ai"]])

    # 3b) Best UTT summary: why it was best, full config, how it differs from default
    _write_best_utt_summary(run_dir, results, config)

    # 4) Match log + match_outputs/*.txt (unit composition per matchup, like runtime_utt_change/outputs)
    matches_path = os.path.join(run_dir, "matches.csv")
    match_log = getattr(results, "run_match_log", None) or []
    with open(matches_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "generation", "individual_index", "ai_left", "ai_right",
            "left_wins", "right_wins", "draws", "winner",
            "left_unit_composition", "right_unit_composition",
        ])
        for idx, m in enumerate(match_log):
            w.writerow([
                m.get("generation", ""),
                m.get("individual_index", ""),
                m.get("ai_left", ""),
                m.get("ai_right", ""),
                m.get("left_wins", ""),
                m.get("right_wins", ""),
                m.get("draws", ""),
                m.get("winner", ""),
                m.get("left_unit_composition", "N/A"),
                m.get("right_unit_composition", "N/A"),
            ])
            # One .txt per matchup (only when save_game_details is on)
            if save_game_details:
                gen = m.get("generation", 0)
                ind = m.get("individual_index", 0)
                ai_left = m.get("ai_left", "").replace(" ", "_")
                ai_right = m.get("ai_right", "").replace(" ", "_")
                subfolder = os.path.join(match_outputs_dir, f"gen{gen}_ind{ind}")
                os.makedirs(subfolder, exist_ok=True)
                matchup_name = f"{ai_left}_vs_{ai_right}".replace("/", "_")
                txt_path = os.path.join(subfolder, f"{matchup_name}.txt")
                with open(txt_path, "w", encoding="utf-8") as tf:
                    left_comp = m.get("left_unit_composition", "N/A")
                    right_comp = m.get("right_unit_composition", "N/A")
                    winner = m.get("winner", "")
                    tf.write(f"Match: {m.get('ai_left', '')} (left) vs {m.get('ai_right', '')} (right)\n")
                    tf.write(f"Result: {m.get('left_wins', 0)}-{m.get('right_wins', 0)} (draws: {m.get('draws', 0)})\n")
                    tf.write(f"Winner: {winner}\n")
                    tf.write(f"Left unit composition (end of last game): {left_comp or '(none)'}\n")
                    tf.write(f"Right unit composition (end of last game): {right_comp or '(none)'}\n")
                    # Win condition: how MicroRTS decides a winner
                    tf.write("\n--- Win condition (MicroRTS) ---\n")
                    tf.write("A game ends when one player has no units left (elimination). That player loses; the other wins.\n")
                    tf.write("Draw if both still have units when max steps is reached, or both have zero units.\n")
                    # Why this side won: explicit reason from end state
                    tf.write("\n--- Why this result ---\n")
                    if left_comp not in ("N/A", "", None) or right_comp not in ("N/A", "", None):
                        if winner == "left":
                            tf.write(f"Left ({m.get('ai_left', '')}) won: Right had no units left (eliminated). ")
                            tf.write(f"End state: Left had {left_comp or 'units'}; Right had {right_comp or 'none'}.\n")
                        elif winner == "right":
                            tf.write(f"Right ({m.get('ai_right', '')}) won: Left had no units left (eliminated). ")
                            tf.write(f"End state: Left had {left_comp or 'none'}; Right had {right_comp or 'units'}.\n")
                        elif winner == "draw":
                            tf.write(f"Draw: both sides still had units (or tied at zero). ")
                            tf.write(f"End state: Left had {left_comp or 'none'}; Right had {right_comp or 'none'}.\n")
                    else:
                        tf.write("(Unit composition not captured; see snapshots below for end state.)\n")
                    tf.write("\n(Unit composition is captured from the last game of the matchup when capture_composition is enabled.)\n")
                    all_game_snapshots = m.get("_game_snapshots") or []
                    per_game_compositions = m.get("_per_game_compositions") or []  # list of {winner, left, right, game_index}
                    games_per_ordering = m.get("_games_per_ordering")  # when both orderings: 5 so Games 1-5 = one ordering, 6-10 = other

                    def _comp_str(comp_dict):
                        if not comp_dict or not isinstance(comp_dict, dict):
                            return "none"
                        return ",".join(f"{k}:{v}" for k, v in sorted(comp_dict.items()))

                    if all_game_snapshots:
                        tf.write("\n" + "=" * 60 + "\n")
                        tf.write("Game state snapshots – every game, step 0 then every N steps then final\n")
                        tf.write("=" * 60 + "\n")
                        # New format: list of per-game [(step, text), ...]; old format: flat [(step, text), ...] for one game
                        if all_game_snapshots and isinstance(all_game_snapshots[0], list):
                            for game_idx, snapshots in enumerate(all_game_snapshots, 1):
                                if games_per_ordering and game_idx == 1:
                                    tf.write(f"\n(Ordering 1: Left={m.get('ai_left', '')}, Right={m.get('ai_right', '')} — Games 1–{games_per_ordering})\n")
                                elif games_per_ordering and game_idx == games_per_ordering + 1:
                                    tf.write(f"\n(Ordering 2: Left={m.get('ai_right', '')}, Right={m.get('ai_left', '')} — Games {games_per_ordering + 1}–{len(all_game_snapshots)})\n")
                                tf.write(f"\n--- Game {game_idx} ---\n")
                                # Per-game winner and why (from _per_game_compositions when available)
                                if game_idx <= len(per_game_compositions):
                                    pg = per_game_compositions[game_idx - 1]
                                    pg_winner = pg.get("winner", "")
                                    left_c = _comp_str(pg.get("left"))
                                    right_c = _comp_str(pg.get("right"))
                                    if pg_winner == "left":
                                        tf.write(f"Winner: Left ({m.get('ai_left', '')}) — Right had no units (elimination). Left: {left_c}; Right: {right_c}.\n")
                                    elif pg_winner == "right":
                                        tf.write(f"Winner: Right ({m.get('ai_right', '')}) — Left had no units (elimination). Left: {left_c}; Right: {right_c}.\n")
                                    else:
                                        tf.write(f"Winner: Draw — Both sides still had units. Left: {left_c}; Right: {right_c}.\n")
                                for step, text in snapshots:
                                    tf.write(f"\n  Step {step}\n")
                                    tf.write(text)
                                    if not text.endswith("\n"):
                                        tf.write("\n")
                        else:
                            tf.write("\n--- Game 1 (legacy single-game capture) ---\n")
                            if per_game_compositions:
                                pg = per_game_compositions[0]
                                pg_winner = pg.get("winner", "")
                                left_c = _comp_str(pg.get("left"))
                                right_c = _comp_str(pg.get("right"))
                                if pg_winner == "left":
                                    tf.write(f"Winner: Left — Right had no units (elimination). Left: {left_c}; Right: {right_c}.\n")
                                elif pg_winner == "right":
                                    tf.write(f"Winner: Right — Left had no units (elimination). Left: {left_c}; Right: {right_c}.\n")
                                else:
                                    tf.write(f"Winner: Draw — Both sides still had units. Left: {left_c}; Right: {right_c}.\n")
                            for step, text in all_game_snapshots:
                                tf.write(f"\n  Step {step}\n")
                                tf.write(text)
                                if not text.endswith("\n"):
                                    tf.write("\n")
                    else:
                        tf.write("\n(No snapshots: enable capture_snapshots and ensure game_state_utils is available.)\n")

    # 5) Fitness over generations plot — inside run folder
    plot_path = os.path.join(run_dir, "fitness_plot.png")
    if results.generation_stats:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            gens = [s.generation for s in results.generation_stats]
            best_f = [s.best_fitness for s in results.generation_stats]
            avg_f = [s.avg_fitness for s in results.generation_stats]
            plt.figure(figsize=(8, 5))
            plt.plot(gens, best_f, "b-o", label="Best fitness", markersize=4)
            plt.plot(gens, avg_f, "g-s", label="Avg fitness", markersize=4)
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.title("Fitness over generations")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=120)
            plt.close()
        except Exception as e:
            print(f"  (Could not save fitness plot: {e})")

    print(f"  Run logs: {GA_RUN_LOGS_DIR}")
    print(f"    History: {RUN_HISTORY_CSV}")
    print(f"    This run (all in one folder): {run_dir}")
    print(f"      generations.csv, utt_changes.csv, best_utt_summary.txt, best_utt_config.json,")
    print(f"      matches.csv, fitness_plot.png," + (" match_outputs/<genN_indM>/*.txt," if save_game_details else " (match_outputs off),") + " utt_log/ (every gen/ind UTT)")
    return run_dir


def main():
    os.makedirs(GA_RUN_LOGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{EXPERIMENT_NAME}_{ts}"
    run_dir = os.path.join(GA_RUN_LOGS_DIR, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"{EXPERIMENT_NAME}_{ts}.log")
    tee = Tee(sys.stdout, log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        return _main(ts, log_path)
    finally:
        sys.stdout = old_stdout
        tee.close()


def _main(ts: str, log_path: str):
    # Single map for fast debug; add more maps for better balance estimates
    MAP_PATHS = [
        "maps/8x8/basesWorkers8x8A.xml",
    ]
    # Run folder and UTT log: create at start so we can save every evaluated UTT for comparison
    run_id = f"{EXPERIMENT_NAME}_{ts}"
    run_dir = os.path.join(GA_RUN_LOGS_DIR, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    utt_log_dir = os.path.join(run_dir, "utt_log")
    os.makedirs(utt_log_dir, exist_ok=True)

    print("=" * 60)
    print("GA local test: LightRush vs WorkerRush")
    print("=" * 60)
    print(f"  Log file:       {log_path}")
    print(f"  UTT log:        {utt_log_dir} (every gen/ind saved)")
    print(f"  Generations:    {GENERATIONS}")
    print(f"  Population:     {POPULATION}")
    print(f"  Games per eval: {GAMES_PER_EVAL} × {len(MAP_PATHS)} map(s)")
    print(f"  Max steps:      {MAX_STEPS}")
    print(f"  Save game details: {SAVE_GAME_DETAILS} (edit SAVE_GAME_DETAILS in this script or set GA_SAVE_GAME_DETAILS=0)")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print("=" * 60)

    config = GAConfig(
        population_size=POPULATION,
        generations=GENERATIONS,
        crossover_rate=0.6,   # slightly less crossover so mutations stick
        mutation_rate=0.3,   # higher mutation to escape 2-0-10 plateau and explore 6-6
        use_real_microrts=True,
        use_working_evaluator=True,
        max_steps=MAX_STEPS,
        map_path=MAP_PATHS[0],
        map_paths=MAP_PATHS,
        games_per_evaluation=GAMES_PER_EVAL,
        ai_agents=[
            # Fully observable rush/baselines
            "lightRushAI",
            "workerRushAI",
            "heavyRushAI",
            "rangedRushAI",
            #"randomBiasedAI",
            # Partially observable rush variants
            "POHeavyRush",
            "POLightRush",
            "PORangedRush",
            "POWorkerRush",
        ],
        # Weights: slightly less dominance for balance so duration can break ties between 5-5-0 UTTs
        fitness_alpha=0.5,   # balance weight
        fitness_beta=0.3,    # duration weight
        fitness_gamma=0.2,   # diversity weight
        # Duration target: typical avg steps/game ~50 with wide tolerance band [10, 90]
        target_duration=50,
        duration_tolerance=40,
        max_generations_without_improvement=GENERATIONS,  # run full 20 gens, don't early-stop at 5
        random_immigrant_interval=5,  # inject random UTT every 5 gens to escape 2-0-10 plateau
        use_nondeterministic=True,  # random move conflicts + wider damage ranges
        use_both_orderings=True,    # run (ai1,ai2) and (ai2,ai1); symmetric UTT → 50-50 gives balance signal
        verbose=True,
        utt_log_dir=utt_log_dir,    # save every evaluated UTT (gen{N}_ind{M}.json) for comparison
        save_game_details=SAVE_GAME_DETAILS,  # per-game snapshots + match_outputs/*.txt (set GA_SAVE_GAME_DETAILS=0 to disable)
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
    print("Local test finished")
    print("=" * 60)
    print(f"  Best fitness:   {results.best_fitness.overall_fitness:.4f}")
    print(f"  Balance:        {results.best_fitness.balance:.4f}")
    print(f"  Duration:       {results.best_fitness.duration:.4f}")
    print(f"  Diversity:      {results.best_fitness.strategy_diversity:.4f}")
    print(f"  Results saved:  {experiment_dir}")
    _write_local_run_logs(results, config, experiment_dir, EXPERIMENT_NAME, ts=ts)
    print(f"  Full run log:   {log_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
