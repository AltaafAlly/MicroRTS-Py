#!/usr/bin/env python3
"""
UTT Evolution with Genetic Algorithm and MAP-Elites

This script demonstrates how to evolve Unit Type Table (UTT) configurations
using a genetic algorithm with AI-based fitness evaluation.

Usage:
    python run_utt_evolution.py --generations 10 --population 20 --ai randomBiasedAI
    python run_utt_evolution.py --generations 5 --population 10 --ai workerRushAI --map 8x8
"""

import argparse
import sys
import os
import time
from typing import List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.improved_fitness_evaluator import AIGameSimulationFitness, MultiObjectiveFitness
from core.utt_genetic_algorithm import UTTGeneEncoder, GeneticAlgorithm
from core.utt_map_elites import MAPElitesAlgorithm, BehaviorDescriptorExtractor
from core.utt_utils import UTTVisualizer


def run_ga_evolution(args):
    """Run genetic algorithm evolution."""
    print("🧬 Running Genetic Algorithm Evolution")
    print("=" * 50)
    
    # Setup
    base_utt_path = "/home/altaaf/projects/MicroRTS-Py-Research/gym_microrts/microrts/utts/AsymmetricP1UTT.json"
    encoder = UTTGeneEncoder(base_utt_path)
    
    # Create fitness evaluator
    if args.fitness_type == "ai":
        fitness_evaluator = AIGameSimulationFitness(
            map_paths=[f"maps/{args.map}/basesWorkers{args.map}A.xml"],
            num_games=args.games_per_eval,
            max_steps=args.max_steps,
            ai_agents=[args.ai]
        )
        print(f"   Using AI-based fitness evaluation with {args.ai}")
    else:
        fitness_evaluator = MultiObjectiveFitness(
            map_paths=[f"maps/{args.map}/basesWorkers{args.map}A.xml"]
        )
        print(f"   Using multi-objective fitness evaluation")
    
    # Create genetic algorithm
    ga = GeneticAlgorithm(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        encoder=encoder,
        fitness_evaluator=fitness_evaluator
    )
    
    print(f"   Population: {args.population}, Generations: {args.generations}")
    print(f"   Mutation rate: {args.mutation_rate}, Crossover rate: {args.crossover_rate}")
    print()
    
    # Run evolution
    start_time = time.time()
    best_individuals = ga.run()
    total_time = time.time() - start_time
    
    # Display results
    print(f"\n🏆 EVOLUTION COMPLETE!")
    print(f"   Total time: {total_time:.1f}s")
    if best_individuals:
        print(f"   Best fitness: {best_individuals.fitness:.2f}")
    else:
        print(f"   No results returned")
    
    return best_individuals


def run_map_elites_evolution(args):
    """Run MAP-Elites evolution."""
    print("🗺️ Running MAP-Elites Evolution")
    print("=" * 50)
    
    # Setup
    base_utt_path = "/home/altaaf/projects/MicroRTS-Py-Research/gym_microrts/microrts/utts/AsymmetricP1UTT.json"
    encoder = UTTGeneEncoder(base_utt_path)
    
    # Create fitness evaluator
    if args.fitness_type == "ai":
        fitness_evaluator = AIGameSimulationFitness(
            map_paths=[f"maps/{args.map}/basesWorkers{args.map}A.xml"],
            num_games=args.games_per_eval,
            max_steps=args.max_steps,
            ai_agents=[args.ai]
        )
    else:
        fitness_evaluator = MultiObjectiveFitness(
            map_paths=[f"maps/{args.map}/basesWorkers{args.map}A.xml"]
        )
    
    # Create behavior descriptor extractor
    behavior_extractor = BehaviorDescriptorExtractor()
    
    # Create MAP-Elites algorithm
    map_elites = MAPElitesAlgorithm(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        encoder=encoder,
        fitness_evaluator=fitness_evaluator,
        behavior_extractor=behavior_extractor
    )
    
    print(f"   Population: {args.population}, Generations: {args.generations}")
    print(f"   Behavior dimensions: {behavior_extractor.num_dimensions}")
    print()
    
    # Run evolution
    start_time = time.time()
    archive = map_elites.run()
    total_time = time.time() - start_time
    
    # Display results
    print(f"\n🏆 MAP-ELITES EVOLUTION COMPLETE!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Archive size: {len(archive)}")
    if archive:
        valid_individuals = [ind for ind in archive.values() if ind is not None]
        if valid_individuals:
            best_fitness = max(ind.fitness for ind in valid_individuals)
            print(f"   Best fitness: {best_fitness:.2f}")
        else:
            print(f"   No valid individuals in archive")
    else:
        print(f"   Empty archive")
    
    return archive


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evolve UTT configurations using GA or MAP-Elites")
    
    # Algorithm choice
    parser.add_argument("--algorithm", choices=["ga", "map-elites"], default="ga",
                       help="Evolution algorithm to use")
    
    # Evolution parameters
    parser.add_argument("--generations", type=int, default=5,
                       help="Number of generations")
    parser.add_argument("--population", type=int, default=10,
                       help="Population size")
    parser.add_argument("--mutation-rate", type=float, default=0.3,
                       help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.7,
                       help="Crossover rate")
    
    # Fitness evaluation
    parser.add_argument("--fitness-type", choices=["ai", "multi-objective"], default="ai",
                       help="Type of fitness evaluation")
    parser.add_argument("--ai", default="randomBiasedAI",
                       help="AI agent for evaluation")
    parser.add_argument("--games-per-eval", type=int, default=2,
                       help="Number of games per fitness evaluation")
    parser.add_argument("--max-steps", type=int, default=300,
                       help="Maximum steps per game")
    
    # Map settings
    parser.add_argument("--map", default="8x8",
                       help="Map size (e.g., 8x8, 10x10)")
    
    # Output
    parser.add_argument("--save-results", action="store_true",
                       help="Save results to file")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualizations")
    
    args = parser.parse_args()
    
    print("🚀 UTT EVOLUTION EXPERIMENT")
    print("=" * 50)
    print(f"Algorithm: {args.algorithm}")
    print(f"Fitness: {args.fitness_type}")
    print(f"AI: {args.ai}")
    print(f"Map: {args.map}")
    print()
    
    # Run evolution
    if args.algorithm == "ga":
        results = run_ga_evolution(args)
    else:
        results = run_map_elites_evolution(args)
    
    # Save results if requested
    if args.save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/evolution_{args.algorithm}_{timestamp}.json"
        print(f"\n💾 Saving results to {filename}")
        # TODO: Implement result saving
    
    # Create visualizations if requested
    if args.visualize:
        print(f"\n📊 Creating visualizations...")
        # TODO: Implement visualization
    
    print(f"\n✅ Experiment complete!")


if __name__ == "__main__":
    main()
