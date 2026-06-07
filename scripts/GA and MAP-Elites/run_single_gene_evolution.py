"""
Single-gene evolution driver (light vs worker, one UTT parameter).

Uses WorkingGAEvaluator for real matches, then applies optional single-gene fitness
shaping from ``core.single_gene_fitness`` so the GA can move an intentionally extreme
parameter (e.g. Light maxDamage) back toward balanced matchups.

Outputs (per run, under ``single_gene_experiment/runs/<timestamp>/``):
- ``gene_trajectory.csv`` — per individual: generation, gene values, base vs shaped fitness
- ``generations_summary.csv`` — one row per generation: mean/std/best gene, best fitness
- ``gene_trajectory.png`` — mean ± std and best gene (denormalized) over time
- ``fitness_trajectory.png`` — best / mean shaped and base overall fitness per generation
- ``run.log`` — human-readable log appended after each generation

Run from repo root:
  cd /path/to/MicroRTS-Py-Research
  python "scripts/GA and MAP-Elites/run_single_gene_evolution.py"
"""

from __future__ import annotations

import csv
import datetime
import math
import os
import random
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

# Allow running as a script from this directory
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from core.ga_chromosome import MicroRTSChromosome  # noqa: E402
from core.ga_working_evaluator import (  # noqa: E402
    WorkingGAEvaluator,
)
from core.single_gene_fitness import apply_single_gene_fitness, stagnation_penalty  # noqa: E402

# Reuse helpers from the original single-gene experiment module
from run_single_gene_two_ai_experiment import (  # noqa: E402
    compute_gene_index,
    denormalize_gene_value,
    mutate_gene,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Tunables ---
AI_LEFT = "lightRushAI"
AI_RIGHT = "workerRushAI"
MAP_PATHS = [
    "maps/10x10/basesWorkers10x10.xml",
]
CANDIDATE_MAPS = [
    "maps/10x10/basesWorkers10x10.xml",
    "maps/12x12/basesWorkers12x12F.xml",
    "maps/16x16/basesWorkers16x16G.xml",
    "maps/8x8/basesWorkers8x8.xml",
    "maps/8x8/basesWorkers8x8A.xml",
    "maps/8x8/basesWorkers8x8B.xml",
    "maps/12x12/basesWorkers12x12.xml",
    "maps/12x12/basesWorkers12x12A.xml",
    "maps/12x12/basesWorkers12x12B.xml",
    "maps/12x12/basesWorkers12x12C.xml",
    "maps/12x12/basesWorkers12x12D.xml",
    "maps/12x12/basesWorkers12x12E.xml",
    "maps/12x12/basesWorkers12x12G.xml",
    "maps/12x12/basesWorkers12x12H.xml",
    "maps/16x16/basesWorkers16x16.xml",
    "maps/16x16/basesWorkers16x16A.xml",
    "maps/16x16/basesWorkers16x16B.xml",
    "maps/16x16/basesWorkers16x16C.xml",
    "maps/16x16/basesWorkers16x16D.xml",
    "maps/16x16/basesWorkers16x16E.xml",
    "maps/16x16/basesWorkers16x16F.xml",
]
TARGET_UNIT = "Worker"
TARGET_PARAM = "harvestAmount"
TARGET_GENE_CANDIDATES = [
    ("Light", "maxDamage"),
    ("Light", "hp"),
    ("Light", "attackRange"),
    ("Worker", "maxDamage"),
    ("Worker", "hp"),
    ("Worker", "attackRange"),
    ("Worker", "cost"),
]
GENERATIONS = 12
POPULATION_SIZE = 6
MUTATION_SIGMA = 0.08
CROSSOVER_RATE = 0.5
# Start gene near maximum (unbalanced combat)
INITIAL_GENE_HIGH_NORMALIZED = 1.0
# Evaluator weights (balance-first)
EVAL_ALPHA = 0.72
EVAL_BETA = 0.18
EVAL_GAMMA = 0.10
GAMES_PER_EVAL = 5
MAX_STEPS = 20000
# Probe search budget (keeps long map lists practical)
MAX_PROBE_MAPS = int(os.environ.get("SINGLE_GENE_MAX_PROBE_MAPS", "12"))
# Duration score is computed from AVERAGE steps per game (not total over all games).
# With 5 games x 2 orderings, logs around ~550 total steps imply ~55 avg/game.
TARGET_DURATION = int(os.environ.get("SINGLE_GENE_TARGET_DURATION", "60"))
DURATION_TOLERANCE = int(os.environ.get("SINGLE_GENE_DURATION_TOLERANCE", "60"))

# Single-gene experiment override: allow Light.maxDamage to exceed default upper bound (16).
# This keeps the experiment "one gene" while enabling truly extreme starts.
LIGHT_MAX_DAMAGE_OVERRIDE_MAX = int(os.environ.get("SINGLE_GENE_LIGHT_MAXDAMAGE_MAX", "40"))
# Single-gene shaping: "repair_imbalance" rewards lower normalized gene when balance is poor
SINGLE_GENE_MODE = os.environ.get("SINGLE_GENE_FITNESS_MODE", "directional_balance_gene")
REPAIR_BALANCE_THRESHOLD = 0.55
REPAIR_WEIGHT = float(os.environ.get("SINGLE_GENE_REPAIR_WEIGHT", "0.22"))
STAGNATION_TOLERANCE = float(os.environ.get("SINGLE_GENE_STAGNATION_TOL", "0.01"))
STAGNATION_WEIGHT = float(os.environ.get("SINGLE_GENE_STAGNATION_WEIGHT", "0.08"))
IMBALANCE_GATE = float(os.environ.get("SINGLE_GENE_IMBALANCE_GATE", "0.08"))
SIDE_BIAS_WEIGHT = float(os.environ.get("SINGLE_GENE_SIDE_BIAS_WEIGHT", "0.40"))
MIN_PROBE_DELTA = float(os.environ.get("SINGLE_GENE_MIN_PROBE_DELTA", "0.15"))
AUTO_SELECT_MAP = os.environ.get("SINGLE_GENE_AUTO_SELECT_MAP", "1") != "0"
# Keep experiment strictly single-gene by default.
# Set SINGLE_GENE_AUTO_SELECT_GENE=1 only if you explicitly want cross-gene auto-discovery.
AUTO_SELECT_GENE = os.environ.get("SINGLE_GENE_AUTO_SELECT_GENE", "0") != "0"
# If false, continue GA even when probe signal is weak (warn instead of abort).
REQUIRE_STRONG_PROBE = os.environ.get("SINGLE_GENE_REQUIRE_STRONG_PROBE", "0") != "0"


def _ensure_run_dir() -> Path:
    base = PROJECT_ROOT / "scripts" / "GA and MAP-Elites" / "single_gene_experiment" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"single_gene_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _midpoint_genome() -> List[float]:
    """
    Deterministic baseline genome: each evolvable parameter is at the middle
    of its legal range. Fixed-range parameters are encoded as 0.
    """
    g: List[float] = []
    for ut in MicroRTSChromosome.UNIT_TYPES:
        bounds = MicroRTSChromosome.PARAMETER_BOUNDS[ut]
        for _pname, (lo, hi) in bounds.items():
            if hi <= lo:
                g.append(0.0)
            else:
                g.append(0.5)
    # Include global parameters (currently moveConflictResolutionStrategy)
    # so genome length matches MicroRTSChromosome.from_genome().
    g.append(0.5)
    return g


def _evaluate_probe(
    evaluator: WorkingGAEvaluator,
    base_genome: List[float],
    gene_idx: int,
    gene_norm: float,
) -> tuple[float, int, int, int]:
    """
    Evaluate one genome probe and return:
    (ai1_win_rate_decisive, ai1_wins, ai2_wins, draws)
    """
    run_log: List[dict] = []
    setattr(evaluator, "run_match_log", run_log)
    setattr(evaluator, "run_match_log_generation", -1)
    setattr(evaluator, "run_match_log_individual_index", 0)
    g = list(base_genome)
    g[gene_idx] = max(0.0, min(1.0, gene_norm))
    ch = MicroRTSChromosome.from_genome(g)
    _ = evaluator.evaluate_chromosome(ch)
    row = run_log[-1] if run_log else {}
    lw = int(row.get("left_wins", 0))
    rw = int(row.get("right_wins", 0))
    dr = int(row.get("draws", 0))
    dec = lw + rw
    rate = (lw / dec) if dec > 0 else 0.5
    return rate, lw, rw, dr


def _build_evaluator_for_map(map_path: str) -> WorkingGAEvaluator:
    return WorkingGAEvaluator(
        alpha=EVAL_ALPHA,
        beta=EVAL_BETA,
        gamma=EVAL_GAMMA,
        max_steps=MAX_STEPS,
        map_path=map_path,
        map_paths=[map_path],
        games_per_eval=GAMES_PER_EVAL,
        ai_agents=[AI_LEFT, AI_RIGHT],
        use_nondeterministic=True,
        use_both_orderings=True,
        target_duration=TARGET_DURATION,
        duration_tolerance=DURATION_TOLERANCE,
    )


def main() -> None:
    os.chdir(PROJECT_ROOT)
    run_dir = _ensure_run_dir()
    csv_path = run_dir / "gene_trajectory.csv"
    summary_path = run_dir / "generations_summary.csv"
    plot_gene = run_dir / "gene_trajectory.png"
    plot_fit = run_dir / "fitness_trajectory.png"
    log_path = run_dir / "run.log"

    # Optional range override to allow "very high" Light.maxDamage starts (>16) for this experiment.
    if TARGET_UNIT == "Light" and TARGET_PARAM == "maxDamage":
        lo, hi = MicroRTSChromosome.PARAMETER_BOUNDS["Light"]["maxDamage"]
        if LIGHT_MAX_DAMAGE_OVERRIDE_MAX > hi:
            MicroRTSChromosome.PARAMETER_BOUNDS["Light"]["maxDamage"] = (lo, LIGHT_MAX_DAMAGE_OVERRIDE_MAX)

    gene = compute_gene_index(TARGET_UNIT, TARGET_PARAM)
    evaluator = _build_evaluator_for_map(MAP_PATHS[0])

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log("Single-gene evolution run")
    log(f"Started: {datetime.datetime.now().isoformat()}")
    log(f"Run directory: {run_dir}")
    log(f"AIs: {AI_LEFT} vs {AI_RIGHT}")
    log(f"Maps: {MAP_PATHS}")
    log(f"Gene: {gene.unit_type}.{gene.param_name} in [{gene.min_val}, {gene.max_val}]")
    log(f"Auto-select map={AUTO_SELECT_MAP}, auto-select gene={AUTO_SELECT_GENE}")
    log(f"Require strong probe signal={REQUIRE_STRONG_PROBE} (min_delta={MIN_PROBE_DELTA:.3f})")
    log(f"Duration target/tolerance={TARGET_DURATION}/{DURATION_TOLERANCE}")
    log(f"Generations={GENERATIONS} population={POPULATION_SIZE}")
    log(f"Single-gene fitness mode: {SINGLE_GENE_MODE} (repair_weight={REPAIR_WEIGHT}, threshold={REPAIR_BALANCE_THRESHOLD})")
    log("=" * 60)

    random.seed(42)
    base_genome = _midpoint_genome()
    # Probe low/high values to verify the selected setup has a usable directional signal.
    selected_map = MAP_PATHS[0]
    selected_gene = gene
    log(
        f"Probing setup candidates (auto-map={AUTO_SELECT_MAP}, auto-gene={AUTO_SELECT_GENE}) "
        f"with minimum delta {MIN_PROBE_DELTA:.3f}..."
    )
    genes_to_try = (
        [compute_gene_index(u, p) for (u, p) in TARGET_GENE_CANDIDATES]
        if AUTO_SELECT_GENE
        else [selected_gene]
    )
    maps_to_try = list(dict.fromkeys(([selected_map] + CANDIDATE_MAPS) if AUTO_SELECT_MAP else [selected_map]))
    if AUTO_SELECT_MAP and MAX_PROBE_MAPS > 0 and len(maps_to_try) > MAX_PROBE_MAPS:
        # Keep the current selected map first, then cap the exploration budget.
        maps_to_try = maps_to_try[:MAX_PROBE_MAPS]
    best = None
    for gi in genes_to_try:
        for mp in maps_to_try:
            e2 = _build_evaluator_for_map(mp)
            if hasattr(e2, "close_cached_env"):
                e2.close_cached_env()
            lr, llw, lrw, ldr = _evaluate_probe(e2, base_genome, gi.index, 0.0)
            hr, hlw, hrw, hdr = _evaluate_probe(e2, base_genome, gi.index, 1.0)
            delta = abs(hr - lr)
            direction = hr - lr
            log(
                f"  Candidate {gi.unit_type}.{gi.param_name} @ {mp}: "
                f"low={llw}-{lrw}-{ldr} (rate={lr:.3f}), "
                f"high={hlw}-{hrw}-{hdr} (rate={hr:.3f}), "
                f"delta={delta:.3f}, dir={direction:+.3f}"
            )
            # Prefer cases where high gene makes AI1 stronger (positive direction).
            score = delta + (0.05 if direction > 0 else 0.0)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "delta": delta,
                    "direction": direction,
                    "gene": gi,
                    "map": mp,
                    "low_tuple": (llw, lrw, ldr),
                    "high_tuple": (hlw, hrw, hdr),
                    "low_rate": lr,
                    "high_rate": hr,
                }

    assert best is not None
    selected_gene = best["gene"]
    selected_map = best["map"]
    best_delta = float(best["delta"])
    if selected_gene != gene:
        gene = selected_gene
        log(f"Selected stronger-signal gene: {gene.unit_type}.{gene.param_name}")
    if selected_map != MAP_PATHS[0]:
        MAP_PATHS[:] = [selected_map]
        log(f"Selected stronger-signal map: {selected_map}")
    evaluator = _build_evaluator_for_map(selected_map)
    log(
        f"Chosen probe setup => {gene.unit_type}.{gene.param_name} @ {selected_map}: "
        f"low={best['low_tuple'][0]}-{best['low_tuple'][1]}-{best['low_tuple'][2]} (rate={best['low_rate']:.3f}), "
        f"high={best['high_tuple'][0]}-{best['high_tuple'][1]}-{best['high_tuple'][2]} (rate={best['high_rate']:.3f}), "
        f"delta={best_delta:.3f}"
    )
    if best_delta < MIN_PROBE_DELTA:
        msg = (
            f"Weak dominance signal across tested gene/map candidates "
            f"(best={gene.unit_type}.{gene.param_name} @ {selected_map}, delta={best_delta:.3f} < {MIN_PROBE_DELTA:.3f}). "
            "GA can still run, but outcome-shift signal may be muted."
        )
        if REQUIRE_STRONG_PROBE:
            raise RuntimeError(
                msg + " Add better candidates, or lower SINGLE_GENE_MIN_PROBE_DELTA."
            )
        log("WARNING: " + msg + " Continuing because SINGLE_GENE_REQUIRE_STRONG_PROBE=0.")

    population: List[List[float]] = []
    for _ in range(POPULATION_SIZE):
        # Start extremely high (near max) with tiny jitter so the run begins in a clearly "high-gene" regime.
        hi = INITIAL_GENE_HIGH_NORMALIZED + random.uniform(-0.002, 0.0)
        gi = list(base_genome)
        gi[gene.index] = min(1.0, max(0.0, hi))
        population.append(gi)

    gen_means: List[float] = []
    gen_stds: List[float] = []
    gen_best_gene: List[float] = []
    gen_best_shaped: List[float] = []
    gen_mean_shaped: List[float] = []
    gen_best_base: List[float] = []
    prev_gen_mean_ai1_win_rate = None
    prev_gen_mean_draw_rate = None

    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(
            [
                "generation",
                "individual_index",
                "normalized_gene",
                "denormalized_gene",
                "balance",
                "duration",
                "strategy_diversity",
                "overall_fitness_base",
                "overall_fitness_shaped",
                "ai1_win_rate_decisive",
                "ai1_wins",
                "ai2_wins",
                "draws",
                "draw_rate",
                "stagnation_penalty",
                "side_bias",
                "side_bias_penalty",
            ]
        )

        for gen in range(GENERATIONS):
            chromosomes = [MicroRTSChromosome.from_genome(g) for g in population]
            # Evaluate one-by-one so we can capture directional outcome (light vs worker win-rate)
            base_fitness = []
            run_log: List[dict] = []
            setattr(evaluator, "run_match_log", run_log)
            setattr(evaluator, "run_match_log_generation", gen)
            if hasattr(evaluator, "close_cached_env"):
                evaluator.close_cached_env()
            for i, ch in enumerate(chromosomes):
                setattr(evaluator, "run_match_log_individual_index", i)
                base_fitness.append(evaluator.evaluate_chromosome(ch))

            shaped: List[float] = []
            denorm_vals: List[float] = []
            gen_ai1_rates: List[float] = []
            gen_draw_rates: List[float] = []
            for i, (g, fc) in enumerate(zip(population, base_fitness)):
                norm = g[gene.index]
                # Pull directional matchup stats from evaluator run log.
                # With use_both_orderings=True, left_wins/right_wins are by AI identity:
                # ai1 (light) vs ai2 (worker).
                row = next(
                    (
                        r
                        for r in reversed(run_log)
                        if r.get("generation") == gen and r.get("individual_index") == i
                    ),
                    None,
                )
                lw = int((row or {}).get("left_wins", 0))
                rw = int((row or {}).get("right_wins", 0))
                dr = int((row or {}).get("draws", 0))
                o1_ai1 = int((row or {}).get("_ordering1_ai1_wins", 0))
                o1_ai2 = int((row or {}).get("_ordering1_ai2_wins", 0))
                o2_ai1 = int((row or {}).get("_ordering2_ai1_wins", 0))
                o2_ai2 = int((row or {}).get("_ordering2_ai2_wins", 0))
                decisive = lw + rw
                ai1_win_rate = (lw / decisive) if decisive > 0 else 0.5
                total_games = lw + rw + dr
                draw_rate = (dr / total_games) if total_games > 0 else 0.0
                o1_dec = o1_ai1 + o1_ai2
                o2_dec = o2_ai1 + o2_ai2
                o1_rate = (o1_ai1 / o1_dec) if o1_dec > 0 else 0.5
                o2_rate = (o2_ai1 / o2_dec) if o2_dec > 0 else 0.5
                side_bias = abs(o1_rate - o2_rate)
                fc2 = apply_single_gene_fitness(
                    fc,
                    norm,
                    mode=SINGLE_GENE_MODE,
                    balance_threshold=REPAIR_BALANCE_THRESHOLD,
                    w_repair=REPAIR_WEIGHT,
                    ai1_win_rate=ai1_win_rate,
                    target_ai1_win_rate=0.5,
                    w_directional=0.65,
                    w_gene_down=0.0,
                )
                stg_pen = stagnation_penalty(
                    ai1_win_rate=ai1_win_rate,
                    draw_rate=draw_rate,
                    prev_ai1_win_rate=prev_gen_mean_ai1_win_rate,
                    prev_draw_rate=prev_gen_mean_draw_rate,
                    tolerance=STAGNATION_TOLERANCE,
                    weight=STAGNATION_WEIGHT,
                    imbalance_gate=IMBALANCE_GATE,
                )
                side_pen = SIDE_BIAS_WEIGHT * side_bias
                final_shaped = max(0.0, fc2.overall_fitness - stg_pen - side_pen)
                dv = denormalize_gene_value(gene, norm)
                denorm_vals.append(dv)
                shaped.append(final_shaped)
                gen_ai1_rates.append(ai1_win_rate)
                gen_draw_rates.append(draw_rate)
                w.writerow(
                    [
                        gen,
                        i,
                        norm,
                        dv,
                        fc.balance,
                        fc.duration,
                        fc.strategy_diversity,
                        fc.overall_fitness,
                        final_shaped,
                        ai1_win_rate,
                        lw,
                        rw,
                        dr,
                        draw_rate,
                        stg_pen,
                        side_bias,
                        side_pen,
                    ]
                )

            mean_val = sum(denorm_vals) / len(denorm_vals)
            var = sum((x - mean_val) ** 2 for x in denorm_vals) / len(denorm_vals)
            std_val = math.sqrt(var)
            best_i = max(range(len(shaped)), key=lambda i: shaped[i])
            best_gene_val = denorm_vals[best_i]
            best_shaped = shaped[best_i]
            mean_shaped = sum(shaped) / len(shaped)
            best_base = base_fitness[best_i].overall_fitness

            gen_means.append(mean_val)
            gen_stds.append(std_val)
            gen_best_gene.append(best_gene_val)
            gen_best_shaped.append(best_shaped)
            gen_mean_shaped.append(mean_shaped)
            gen_best_base.append(max(f.overall_fitness for f in base_fitness))

            log(
                f"[Gen {gen}] gene {gene.unit_type}.{gene.param_name}: "
                f"mean={mean_val:.3f} std={std_val:.3f} best_gene={best_gene_val:.3f} | "
                f"best_shaped={best_shaped:.4f} mean_shaped={mean_shaped:.4f} best_base={best_base:.4f} "
                f"| mean_ai1_win={sum(gen_ai1_rates)/len(gen_ai1_rates):.3f} mean_draw={sum(gen_draw_rates)/len(gen_draw_rates):.3f}"
            )

            prev_gen_mean_ai1_win_rate = sum(gen_ai1_rates) / len(gen_ai1_rates)
            prev_gen_mean_draw_rate = sum(gen_draw_rates) / len(gen_draw_rates)

            # Selection + offspring (tournament on shaped fitness)
            def tournament_pick() -> List[float]:
                k = 3
                cand = random.sample(range(POPULATION_SIZE), k)
                bi = max(cand, key=lambda i: shaped[i])
                return population[bi]

            new_pop: List[List[float]] = []
            while len(new_pop) < POPULATION_SIZE:
                p1, p2 = tournament_pick(), tournament_pick()
                c1, c2 = list(p1), list(p2)
                if random.random() < CROSSOVER_RATE:
                    if random.random() < 0.5:
                        c1[gene.index], c2[gene.index] = c2[gene.index], c1[gene.index]
                c1 = mutate_gene(c1, gene, sigma=MUTATION_SIGMA)
                c2 = mutate_gene(c2, gene, sigma=MUTATION_SIGMA)
                new_pop.append(c1)
                if len(new_pop) < POPULATION_SIZE:
                    new_pop.append(c2)
            population = new_pop

    with summary_path.open("w", newline="", encoding="utf-8") as fs:
        sw = csv.writer(fs)
        sw.writerow(
            [
                "generation",
                "mean_gene_denorm",
                "std_gene_denorm",
                "best_gene_denorm",
                "mean_shaped_fitness",
                "best_shaped_fitness",
                "best_base_fitness",
            ]
        )
        for g in range(GENERATIONS):
            sw.writerow(
                [
                    g,
                    gen_means[g],
                    gen_stds[g],
                    gen_best_gene[g],
                    gen_mean_shaped[g],
                    gen_best_shaped[g],
                    gen_best_base[g],
                ]
            )

    gens = list(range(GENERATIONS))
    plt.figure(figsize=(9, 4.5))
    plt.plot(gens, gen_means, label="Mean gene (denorm)", color="C0")
    plt.fill_between(
        gens,
        [m - s for m, s in zip(gen_means, gen_stds)],
        [m + s for m, s in zip(gen_means, gen_stds)],
        color="C0",
        alpha=0.2,
        label="±1 std",
    )
    plt.plot(gens, gen_best_gene, "--", color="C1", label="Best individual gene")
    plt.xlabel("Generation")
    plt.ylabel(f"{gene.unit_type}.{gene.param_name}")
    plt.title(f"Gene trajectory: {AI_LEFT} vs {AI_RIGHT}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_gene, dpi=120)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(gens, gen_best_shaped, "b-o", markersize=4, label="Best shaped fitness")
    plt.plot(gens, gen_mean_shaped, "g-s", markersize=4, label="Mean shaped fitness")
    plt.plot(gens, gen_best_base, color="gray", linestyle=":", label="Best base overall (evaluator)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over generations (selection uses shaped overall)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_fit, dpi=120)
    plt.close()

    log("")
    log("Finished: " + datetime.datetime.now().isoformat())
    log(f"CSV (per individual): {csv_path}")
    log(f"CSV (per generation): {summary_path}")
    log(f"Plot (gene): {plot_gene}")
    log(f"Plot (fitness): {plot_fit}")
    log("=" * 60)


if __name__ == "__main__":
    main()
