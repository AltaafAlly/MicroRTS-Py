#!/usr/bin/env python3
"""
Main Runner Script for MicroRTS Genetic Algorithm

This script provides a command-line interface for running the genetic algorithm
to evolve balanced MicroRTS game configurations.

Usage:
    python run_new_ga.py --generations 10 --population 20
    python run_new_ga.py --config fast --save-results
    python run_new_ga.py --list-experiments
"""

import argparse
import sys
import os
import json
from typing import Optional

# Add the core module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.ga_algorithm import MicroRTSGeneticAlgorithm, GAConfig, create_default_ga, create_fast_ga, create_comprehensive_ga
from core.ga_config_manager import ExperimentManager, MicroRTSConfigConverter, ConfigValidator
from core.ga_working_evaluator import WorkingGAEvaluator, evaluate_population_fitness_working


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="MicroRTS Genetic Algorithm for evolving balanced game configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (uses real MicroRTS by default)
  python run_ga.py --config fast
  
  # Use simulation instead of real MicroRTS (faster for testing)
  python run_ga.py --config fast --use-simulation
  
  # Custom run with real MicroRTS matches
  python run_ga.py --generations 15 --population 30 --mutation-rate 0.15 --max-steps 500
  
  # Comprehensive run with results saving
  python run_ga.py --config comprehensive --save-results --experiment-name "balance_test"
  
  # List previous experiments
  python run_ga.py --list-experiments
  
  # Load and analyze previous results
  python run_ga.py --load-experiment experiments/balance_test_1234567890
        """
    )
    
    # Algorithm configuration
    config_group = parser.add_argument_group('Algorithm Configuration')
    config_group.add_argument('--config', choices=['default', 'fast', 'comprehensive'], 
                             default='default', help='Predefined configuration preset')
    config_group.add_argument('--generations', type=int, help='Number of generations to evolve')
    config_group.add_argument('--population', type=int, help='Population size')
    config_group.add_argument('--crossover-rate', type=float, help='Crossover probability (0-1)')
    config_group.add_argument('--mutation-rate', type=float, help='Mutation probability (0-1)')
    config_group.add_argument('--mutation-strength', type=float, help='Mutation strength (0-1)')
    
    # Selection parameters
    selection_group = parser.add_argument_group('Selection Parameters')
    selection_group.add_argument('--tournament-size', type=int, help='Tournament selection size')
    selection_group.add_argument('--elite-size', type=int, help='Number of elite individuals to preserve')
    
    # Fitness evaluation
    fitness_group = parser.add_argument_group('Fitness Evaluation')
    fitness_group.add_argument('--fitness-alpha', type=float, help='Balance component weight (0-1)')
    fitness_group.add_argument('--fitness-beta', type=float, help='Duration component weight (0-1)')
    fitness_group.add_argument('--fitness-gamma', type=float, help='Strategy diversity component weight (0-1)')
    fitness_group.add_argument('--target-duration', type=int, help='Target match duration in steps')
    fitness_group.add_argument('--duration-tolerance', type=int, help='Acceptable duration deviation')
    
    # Real MicroRTS settings
    microrts_group = parser.add_argument_group('Real MicroRTS Settings')
    microrts_group.add_argument('--use-real-microrts', action='store_true', default=True,
                               help='Use real MicroRTS matches instead of simulation')
    microrts_group.add_argument('--use-simulation', action='store_true',
                               help='Use simulation instead of real MicroRTS matches')
    microrts_group.add_argument('--use-working-evaluator', action='store_true',
                               help='Use the working UTT evaluator (bypasses UTT loading bug)')
    microrts_group.add_argument('--max-steps', type=int, help='Maximum steps per game')
    microrts_group.add_argument('--map-path', type=str, help='Path to map file')
    microrts_group.add_argument('--games-per-eval', type=int, help='Games per chromosome evaluation')
    
    # Termination criteria
    termination_group = parser.add_argument_group('Termination Criteria')
    termination_group.add_argument('--max-generations-without-improvement', type=int, 
                                  help='Max generations without improvement before stopping')
    termination_group.add_argument('--target-fitness', type=float, 
                                  help='Target fitness value to reach (0-1)')
    
    # Output and storage
    output_group = parser.add_argument_group('Output and Storage')
    output_group.add_argument('--save-results', action='store_true', 
                             help='Save results to experiment directory')
    output_group.add_argument('--experiment-name', type=str, default='ga_experiment',
                             help='Name for the experiment (used in directory naming)')
    output_group.add_argument('--output-dir', type=str, default='experiments',
                             help='Base directory for experiment storage')
    output_group.add_argument('--verbose', action='store_true', default=True,
                             help='Enable verbose output')
    output_group.add_argument('--quiet', action='store_true', 
                             help='Disable verbose output')
    
    # Experiment management
    experiment_group = parser.add_argument_group('Experiment Management')
    experiment_group.add_argument('--list-experiments', action='store_true',
                                 help='List all available experiments')
    experiment_group.add_argument('--load-experiment', type=str,
                                 help='Load and display results from a previous experiment')
    experiment_group.add_argument('--compare-experiments', nargs='+',
                                 help='Compare multiple experiments')
    
    # Analysis and visualization
    analysis_group = parser.add_argument_group('Analysis and Visualization')
    analysis_group.add_argument('--analyze-best', action='store_true',
                               help='Analyze the best individual configuration')
    analysis_group.add_argument('--export-config', type=str,
                               help='Export best configuration to MicroRTS format')
    analysis_group.add_argument('--validate-config', action='store_true',
                               help='Validate the best configuration')
    
    return parser


def create_ga_config(args) -> GAConfig:
    """Create GA configuration from command line arguments."""
    
    # Start with default configuration
    if args.config == 'fast':
        ga = create_fast_ga()
        config = ga.config
    elif args.config == 'comprehensive':
        ga = create_comprehensive_ga()
        config = ga.config
    else:
        config = GAConfig()
    
    # Override with command line arguments
    if args.generations is not None:
        config.generations = args.generations
    if args.population is not None:
        config.population_size = args.population
    if args.crossover_rate is not None:
        config.crossover_rate = args.crossover_rate
    if args.mutation_rate is not None:
        config.mutation_rate = args.mutation_rate
    if args.mutation_strength is not None:
        config.mutation_strength = args.mutation_strength
    
    # Selection parameters
    if args.tournament_size is not None:
        config.tournament_size = args.tournament_size
    if args.elite_size is not None:
        config.elite_size = args.elite_size
    
    # Fitness parameters
    if args.fitness_alpha is not None:
        config.fitness_alpha = args.fitness_alpha
    if args.fitness_beta is not None:
        config.fitness_beta = args.fitness_beta
    if args.fitness_gamma is not None:
        config.fitness_gamma = args.fitness_gamma
    if args.target_duration is not None:
        config.target_duration = args.target_duration
    if args.duration_tolerance is not None:
        config.duration_tolerance = args.duration_tolerance
    
    # Real MicroRTS settings
    if args.use_simulation:
        config.use_real_microrts = False
    if args.use_working_evaluator:
        config.use_working_evaluator = True
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.map_path is not None:
        config.map_path = args.map_path
    if args.games_per_eval is not None:
        config.games_per_evaluation = args.games_per_eval
    
    # Termination criteria
    if args.max_generations_without_improvement is not None:
        config.max_generations_without_improvement = args.max_generations_without_improvement
    if args.target_fitness is not None:
        config.target_fitness = args.target_fitness
    
    # Output settings
    config.verbose = args.verbose and not args.quiet
    
    return config


def run_genetic_algorithm(config: GAConfig, experiment_manager: ExperimentManager, 
                         experiment_name: str) -> tuple:
    """Run the genetic algorithm and return results."""
    
    # Create experiment directory
    experiment_dir = experiment_manager.create_experiment_dir(experiment_name)
    
    # Save configuration
    experiment_manager.save_experiment_config(config, experiment_dir)
    
    # Create and run GA
    ga = MicroRTSGeneticAlgorithm(config)
    results = ga.run()
    
    # Save results
    experiment_manager.save_experiment_results(results, experiment_dir)
    experiment_manager.save_best_config(results.best_individual, experiment_dir)
    
    return results, experiment_dir


def list_experiments(experiment_manager: ExperimentManager):
    """List all available experiments."""
    experiments = experiment_manager.list_experiments()
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"Found {len(experiments)} experiments:")
    print("-" * 80)
    
    for experiment in experiments:
        experiment_dir = os.path.join(experiment_manager.base_dir, experiment)
        summary = experiment_manager.get_experiment_summary(experiment_dir)
        
        if 'error' in summary:
            print(f"❌ {experiment} (Error: {summary['error']})")
        else:
            print(f"✅ {experiment}")
            print(f"   Best Fitness: {summary['best_fitness']:.4f}")
            print(f"   Generations: {summary['total_generations']}")
            print(f"   Time: {summary['total_time']:.1f}s")
            if summary['convergence_generation']:
                print(f"   Converged at generation: {summary['convergence_generation']}")
            print()


def load_experiment(experiment_manager: ExperimentManager, experiment_path: str):
    """Load and display experiment results."""
    try:
        results = experiment_manager.load_experiment_results(experiment_path)
        config = experiment_manager.load_experiment_config(experiment_path)
        
        print(f"Experiment: {os.path.basename(experiment_path)}")
        print("=" * 60)
        
        # Display configuration
        print("Configuration:")
        print(f"  Population Size: {config.population_size}")
        print(f"  Generations: {config.total_generations}")
        print(f"  Crossover Rate: {config.crossover_rate}")
        print(f"  Mutation Rate: {config.mutation_rate}")
        print(f"  Fitness Weights: α={config.fitness_alpha}, β={config.fitness_beta}, γ={config.fitness_gamma}")
        print()
        
        # Display results
        print("Results:")
        print(f"  Best Fitness: {results.best_fitness.overall_fitness:.4f}")
        print(f"    Balance: {results.best_fitness.balance:.4f}")
        print(f"    Duration: {results.best_fitness.duration:.4f}")
        print(f"    Diversity: {results.best_fitness.strategy_diversity:.4f}")
        print(f"  Total Generations: {results.total_generations}")
        print(f"  Total Time: {results.total_time:.1f} seconds")
        if results.convergence_generation:
            print(f"  Converged at Generation: {results.convergence_generation}")
        print()
        
        # Display generation statistics
        if results.generation_stats:
            print("Generation Statistics:")
            print("Gen | Best Fit | Avg Fit | Diversity | Time")
            print("-" * 50)
            for stats in results.generation_stats[-10:]:  # Show last 10 generations
                print(f"{stats.generation:3d} | {stats.best_fitness:8.4f} | {stats.avg_fitness:7.4f} | {stats.population_diversity:9.4f} | {stats.time_elapsed:4.1f}s")
        
    except Exception as e:
        print(f"Error loading experiment: {e}")


def analyze_best_individual(results, export_path: Optional[str] = None, validate: bool = False):
    """Analyze the best individual from the results."""
    best = results.best_individual
    
    print("Best Individual Analysis:")
    print("=" * 40)
    
    # Display unit parameters
    print("Unit Parameters:")
    for unit_type, params in best.unit_params.items():
        print(f"  {unit_type}:")
        for param_name, value in params.to_dict().items():
            print(f"    {param_name}: {value}")
        print()
    
    # Display global parameters
    print("Global Parameters:")
    for param_name, value in best.global_params.to_dict().items():
        print(f"  {param_name}: {value}")
    print()
    
    # Validate configuration if requested
    if validate:
        warnings = ConfigValidator.validate_chromosome(best)
        if warnings:
            print("Validation Warnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        else:
            print("✅ Configuration is valid")
        print()
    
    # Export to MicroRTS format if requested
    if export_path:
        microrts_config = MicroRTSConfigConverter.chromosome_to_microrts_config(best)
        MicroRTSConfigConverter.save_microrts_config(microrts_config, export_path)
        print(f"✅ Configuration exported to: {export_path}")


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Initialize experiment manager
    experiment_manager = ExperimentManager(args.output_dir)
    
    # Handle special commands
    if args.list_experiments:
        list_experiments(experiment_manager)
        return
    
    if args.load_experiment:
        load_experiment(experiment_manager, args.load_experiment)
        return
    
    # Create configuration
    try:
        config = create_ga_config(args)
    except Exception as e:
        print(f"Error creating configuration: {e}")
        return 1
    
    # Validate fitness weights
    total_weight = config.fitness_alpha + config.fitness_beta + config.fitness_gamma
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: Fitness weights sum to {total_weight:.3f}, not 1.0")
        print("Normalizing weights...")
        config.fitness_alpha /= total_weight
        config.fitness_beta /= total_weight
        config.fitness_gamma /= total_weight
    
    # Run genetic algorithm
    try:
        print("Starting MicroRTS Genetic Algorithm...")
        results, experiment_dir = run_genetic_algorithm(config, experiment_manager, args.experiment_name)
        
        print(f"\n✅ Evolution completed successfully!")
        print(f"📁 Results saved to: {experiment_dir}")
        
        # Analyze best individual if requested
        if args.analyze_best or args.export_config or args.validate_config:
            print("\n" + "=" * 60)
            analyze_best_individual(results, args.export_config, args.validate_config)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Evolution interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during evolution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
