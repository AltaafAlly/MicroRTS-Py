#!/usr/bin/env python3
"""
Hybrid GA System: Use simulation for evolution, test best UTTs in real matches
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the core module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.ga_algorithm import MicroRTSGeneticAlgorithm, GAConfig
from core.ga_chromosome import MicroRTSChromosome
from utt_manager import UTTManager

def run_hybrid_ga(
    generations: int = 10,
    population: int = 20,
    experiment_name: str = "hybrid_research",
    test_best_count: int = 5
):
    """
    Run GA with simulation, then test best UTTs in real matches.
    
    Args:
        generations: Number of generations to evolve
        population: Population size
        experiment_name: Name for the experiment
        test_best_count: Number of best UTTs to test in real matches
    """
    
    print("🧬 Hybrid GA System: Simulation + Real Testing")
    print("=" * 60)
    print(f"Generations: {generations}")
    print(f"Population: {population}")
    print(f"Experiment: {experiment_name}")
    print(f"Will test top {test_best_count} UTTs in real matches")
    print()
    
    # Step 1: Run GA with simulation
    print("📊 Step 1: Running GA with simulation...")
    config = GAConfig(
        generations=generations,
        population_size=population,
        use_real_microrts=False,  # Use simulation
        use_working_evaluator=False
    )
    
    ga = MicroRTSGeneticAlgorithm(config)
    results = ga.run()
    
    print(f"✅ Simulation evolution completed!")
    print(f"   Best fitness: {results.best_fitness.overall_fitness:.4f}")
    print(f"   Best individual: {results.best_individual}")
    print()
    
    # Step 2: Extract best UTTs
    print("🏆 Step 2: Extracting best UTTs...")
    best_utts = []
    
    # Get the best individuals from the final generation
    final_population = ga.population
    final_fitness = ga.fitness_scores
    
    # Sort by fitness
    sorted_indices = sorted(range(len(final_fitness)), 
                           key=lambda i: final_fitness[i].overall_fitness, 
                           reverse=True)
    
    for i in range(min(test_best_count, len(sorted_indices))):
        idx = sorted_indices[i]
        chromosome = final_population[idx]
        fitness = final_fitness[idx]
        
        best_utts.append({
            'chromosome': chromosome,
            'fitness': fitness,
            'rank': i + 1
        })
        
        print(f"   Rank {i+1}: Fitness {fitness.overall_fitness:.4f}")
    
    print()
    
    # Step 3: Test best UTTs in real matches
    print("🎮 Step 3: Testing best UTTs in real matches...")
    utt_manager = UTTManager()
    
    real_test_results = []
    
    for i, utt_data in enumerate(best_utts):
        print(f"   Testing UTT {i+1}/{len(best_utts)} (Fitness: {utt_data['fitness'].overall_fitness:.4f})...")
        
        try:
            # Generate UTT config
            utt_config = utt_data['chromosome'].to_microrts_config()
            
            # Save with metadata
            utt_path = utt_manager.save_utt_with_metadata(
                utt_config=utt_config,
                experiment_id=experiment_name,
                generation=generations,
                individual_id=i+1,
                fitness={
                    'overall_fitness': utt_data['fitness'].overall_fitness,
                    'balance': utt_data['fitness'].balance,
                    'duration': utt_data['fitness'].duration,
                    'strategy_diversity': utt_data['fitness'].strategy_diversity
                },
                description=f"Best UTT #{i+1} from hybrid GA evolution"
            )
            
            # Test in real match
            from test_utt import test_utt
            match_result = test_utt(utt_path, "POHeavyRush", "POLightRush", 3)
            
            real_test_results.append({
                'rank': i + 1,
                'simulation_fitness': utt_data['fitness'].overall_fitness,
                'real_match_result': match_result,
                'utt_path': utt_path
            })
            
            print(f"     Real match: {match_result}")
            
        except Exception as e:
            print(f"     ❌ Error testing UTT: {e}")
            real_test_results.append({
                'rank': i + 1,
                'simulation_fitness': utt_data['fitness'].overall_fitness,
                'real_match_result': None,
                'error': str(e)
            })
    
    print()
    
    # Step 4: Generate report
    print("📋 Step 4: Hybrid GA Results Report")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Simulation Evolution: {generations} generations, {population} population")
    print(f"Best Simulation Fitness: {results.best_fitness.overall_fitness:.4f}")
    print()
    
    print("🏆 Top UTTs (Simulation → Real Testing):")
    for result in real_test_results:
        print(f"  Rank {result['rank']}:")
        print(f"    Simulation Fitness: {result['simulation_fitness']:.4f}")
        if result['real_match_result']:
            print(f"    Real Match: {result['real_match_result']}")
        else:
            print(f"    Real Match: Failed ({result.get('error', 'Unknown error')})")
        print()
    
    print("✅ Hybrid GA completed successfully!")
    print(f"📁 UTTs saved in: evolved_utts/experiments/{experiment_name}/")
    
    return {
        'simulation_results': results,
        'real_test_results': real_test_results,
        'best_utts': best_utts
    }

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid GA: Simulation + Real Testing")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--population", type=int, default=20, help="Population size")
    parser.add_argument("--experiment", type=str, default="hybrid_research", help="Experiment name")
    parser.add_argument("--test-count", type=int, default=5, help="Number of best UTTs to test")
    
    args = parser.parse_args()
    
    run_hybrid_ga(
        generations=args.generations,
        population=args.population,
        experiment_name=args.experiment,
        test_best_count=args.test_count
    )

if __name__ == "__main__":
    main()
