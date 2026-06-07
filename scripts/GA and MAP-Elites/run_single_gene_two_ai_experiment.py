"""
Single-gene GA experiment for MicroRTS.

For the newer pipeline (repair fitness shaping, per-run logs under
``single_gene_experiment/runs/``, extra plots), see ``run_single_gene_evolution.py``.

Goal:
- Use only TWO AI agents (e.g. lightRushAI vs workerRushAI).
- Isolate ONE gene (a single UTT parameter) and let a small GA evolve it.
- Start that gene near its maximum (very unbalanced) and see if evolution
  pushes it toward values that make the matchup more balanced.
- Log the gene value per individual and per generation, plus mean and variance.
- Plot the trajectory of the gene (mean ± std) over generations.

This script is intentionally standalone and does NOT use the full GAAlgorithm /
GAConfig pipeline. It works directly with MicroRTSChromosome genomes and the
WorkingGAEvaluator.

Optional FE parity (matches ``diagnose_side_bias_light_vs_worker.py`` with ``DIAG_USE_FE_UTT=1``): set
``SINGLE_GENE_FE_UTT=1`` to pass ``fe_utt_builtin=(3, 1)`` into the evaluator. Games then ignore chromosome JSON
(evolved gene does not affect fitness—use only to validate the experiment harness against the GUI).
"""

import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

from core.ga_chromosome import MicroRTSChromosome  # type: ignore
from core.ga_working_evaluator import WorkingGAEvaluator, evaluate_population_fitness_working  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class GeneSpec:
    unit_type: str
    param_name: str
    index: int
    min_val: int
    max_val: int


def compute_gene_index(unit_type: str, param_name: str) -> GeneSpec:
    """
    Compute the index of a specific (unit_type, param_name) in the normalized genome.
    """
    idx = 0
    for ut in MicroRTSChromosome.UNIT_TYPES:
        bounds = MicroRTSChromosome.PARAMETER_BOUNDS[ut]
        for pname, (lo, hi) in bounds.items():
            if ut == unit_type and pname == param_name:
                return GeneSpec(unit_type=ut, param_name=pname, index=idx, min_val=lo, max_val=hi)
            idx += 1
    raise ValueError(f"Could not find gene ({unit_type}, {param_name}) in chromosome bounds.")


def make_initial_genome(gene: GeneSpec, high_normalized: float = 0.95) -> List[float]:
    """
    Create a baseline genome: random chromosome -> genome, then overwrite the
    target gene with a high normalized value.
    """
    base_chrom = MicroRTSChromosome()
    genome = base_chrom.to_genome()
    genome[gene.index] = min(1.0, max(0.0, high_normalized))
    return genome


def mutate_gene(genome: List[float], gene: GeneSpec, sigma: float = 0.08) -> List[float]:
    """
    Mutate only the target gene with Gaussian noise, keeping it in [0, 1].
    """
    new = list(genome)
    val = new[gene.index]
    val += random.gauss(0.0, sigma)
    val = max(0.0, min(1.0, val))
    new[gene.index] = val
    return new


def denormalize_gene_value(gene: GeneSpec, normalized: float) -> float:
    return gene.min_val + normalized * (gene.max_val - gene.min_val)


def main() -> None:
    os.chdir(PROJECT_ROOT)

    # --------- Experiment configuration ---------
    ai_left = "lightRushAI"
    ai_right = "workerRushAI"
    # Use the near-neutral basesWorkers maps from the side-bias diagnostics
    map_paths = [
        "maps/10x10/basesWorkers10x10.xml",
        "maps/12x12/basesWorkers12x12F.xml",
        "maps/16x16/basesWorkers16x16G.xml",
    ]

    # Isolated gene: Light unit attack damage (maxDamage) – very strong effect on combat.
    # You can change this to e.g. ("Worker", "cost") or others from PARAMETER_BOUNDS.
    target_unit = "Light"
    target_param = "maxDamage"
    gene = compute_gene_index(target_unit, target_param)

    generations = 10
    population_size = 6
    mutation_sigma = 0.08
    crossover_rate = 0.5  # simple 1-point crossover

    fe_utt = (3, 1) if os.environ.get("SINGLE_GENE_FE_UTT", "").strip().lower() in ("1", "true", "yes", "on") else None

    # --------- Evaluator setup (two AI agents only) ---------
    evaluator = WorkingGAEvaluator(
        alpha=0.7,        # emphasize balance
        beta=0.2,
        gamma=0.1,
        max_steps=20000,
        map_path=map_paths[0],
        map_paths=map_paths,
        games_per_eval=3,               # 3 games per map
        ai_agents=[ai_left, ai_right],  # ONLY two AIs
        use_nondeterministic=False if fe_utt else True,
        fe_utt_builtin=fe_utt,
        # Single ordering: ai_left (P0) vs ai_right (P1) only.
        use_both_orderings=True,
        # Duration is avg steps *per game* (total_steps/total_games). Logs show ~3k–7k per game.
        target_duration=6000,
        duration_tolerance=4000,
    )

    out_dir = PROJECT_ROOT / "scripts" / "GA and MAP-Elites" / "single_gene_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gene_trajectory.csv"
    plot_path = out_dir / "gene_trajectory.png"
    log_path = out_dir / "experiment_log.txt"

    # Simple logger: write to both stdout and log file
    log_file = log_path.open("w", buffering=1)

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")

    log("Single-gene two-AI experiment starting...")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"AI left:  {ai_left}")
    log(f"AI right: {ai_right}")
    log(f"Maps: {', '.join(map_paths)}")
    log(f"Gene: {gene.unit_type}.{gene.param_name} in [{gene.min_val}, {gene.max_val}]")

    # --------- GA initialization ---------
    random.seed(42)

    population: List[List[float]] = []
    for _ in range(population_size):
        # Start with very high attack damage (normalized close to 1.0)
        g = make_initial_genome(gene, high_normalized=0.98 + random.uniform(-0.01, 0.01))
        population.append(g)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation",
                "individual_index",
                "normalized_value",
                "denormalized_value",
                "balance",
                "overall_fitness",
            ]
        )

        gen_means: List[float] = []
        gen_stds: List[float] = []
        gen_best_vals: List[float] = []

        for gen in range(generations):
            # Decode genomes into chromosomes
            chromosomes = [MicroRTSChromosome.from_genome(g) for g in population]

            # Evaluate fitness
            fitness_components = evaluate_population_fitness_working(chromosomes, evaluator)

            # Log per-individual gene + balance
            denorm_vals: List[float] = []
            for i, (g, fc) in enumerate(zip(population, fitness_components)):
                norm = g[gene.index]
                val = denormalize_gene_value(gene, norm)
                denorm_vals.append(val)
                writer.writerow(
                    [
                        gen,
                        i,
                        norm,
                        val,
                        fc.balance,
                        fc.overall_fitness,
                    ]
                )

            # Per-generation stats
            mean_val = sum(denorm_vals) / len(denorm_vals)
            var = sum((x - mean_val) ** 2 for x in denorm_vals) / len(denorm_vals)
            std_val = math.sqrt(var)
            best_val = denorm_vals[max(range(len(denorm_vals)), key=lambda i: fitness_components[i].overall_fitness)]

            gen_means.append(mean_val)
            gen_stds.append(std_val)
            gen_best_vals.append(best_val)

            log(
                f"Gen {gen}: {gene.unit_type}.{gene.param_name} "
                f"mean={mean_val:.2f}, std={std_val:.2f}, best={best_val:.2f}"
            )

            # --------- Selection + variation ---------
            # Simple tournament selection on overall_fitness
            def tournament_pick() -> List[float]:
                k = 3
                cand_idxs = random.sample(range(population_size), k)
                best_i = max(cand_idxs, key=lambda i: fitness_components[i].overall_fitness)
                return population[best_i]

            new_population: List[List[float]] = []
            while len(new_population) < population_size:
                parent1 = tournament_pick()
                parent2 = tournament_pick()
                child1, child2 = list(parent1), list(parent2)
                if random.random() < crossover_rate:
                    # single-point crossover on the single gene (effectively just swap)
                    if random.random() < 0.5:
                        child1[gene.index], child2[gene.index] = child2[gene.index], child1[gene.index]
                child1 = mutate_gene(child1, gene, sigma=mutation_sigma)
                child2 = mutate_gene(child2, gene, sigma=mutation_sigma)
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)
            population = new_population

    # --------- Plot gene trajectory (mean ± std, plus best) ---------
    gens = list(range(generations))
    plt.figure(figsize=(8, 4))
    means = gen_means
    stds = gen_stds
    bests = gen_best_vals

    plt.plot(gens, means, label="Mean gene value", color="C0")
    plt.fill_between(
        gens,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        color="C0",
        alpha=0.2,
        label="±1 std dev",
    )
    plt.plot(gens, bests, label="Best individual", color="C1", linestyle="--")

    plt.xlabel("Generation")
    plt.ylabel(f"{gene.unit_type}.{gene.param_name} (denormalized)")
    plt.title(f"Evolution of {gene.unit_type}.{gene.param_name} for {ai_left} vs {ai_right}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    log(f"\nSingle-gene experiment complete.")
    log(f"- Gene: {gene.unit_type}.{gene.param_name} in [{gene.min_val}, {gene.max_val}]")
    log(f"- CSV:  {csv_path}")
    log(f"- Plot: {plot_path}")
    log(f"- Log:  {log_path}")
    log_file.close()


if __name__ == "__main__":
    main()

