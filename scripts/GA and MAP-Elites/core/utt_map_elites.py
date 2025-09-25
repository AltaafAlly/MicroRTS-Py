"""
MAP-Elites implementation for evolving diverse MicroRTS UTT configurations.

This extends the genetic algorithm with MAP-Elites to maintain a diverse archive
of high-performing solutions across different behavioral dimensions.
"""

import json
import random
import numpy as np
import copy
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import os
import sys
import time
from collections import defaultdict
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utt_genetic_algorithm import (
    UTTGeneEncoder, Individual, GeneticAlgorithm, 
    GameSimulationFitness, GeneBounds, UnitTypeConstraints
)


@dataclass
class MAPElitesArchive:
    """Archive for MAP-Elites algorithm."""
    cells: Dict[Tuple, Individual] = field(default_factory=dict)
    dimensions: List[str] = field(default_factory=list)
    bounds: List[Tuple[float, float]] = field(default_factory=list)
    resolution: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        self.total_cells = np.prod(self.resolution) if self.resolution else 0
    
    def get_cell_index(self, behavior_descriptor: np.ndarray) -> Tuple:
        """Convert behavior descriptor to cell index."""
        cell_index = []
        for i, (bd_val, (min_val, max_val), res) in enumerate(zip(behavior_descriptor, self.bounds, self.resolution)):
            # Clamp to bounds
            clamped_val = np.clip(bd_val, min_val, max_val)
            # Convert to cell index
            normalized = (clamped_val - min_val) / (max_val - min_val)
            cell_idx = int(normalized * (res - 1))
            cell_index.append(cell_idx)
        return tuple(cell_index)
    
    def add_individual(self, individual: Individual, behavior_descriptor: np.ndarray):
        """Add individual to archive if it's better than existing occupant."""
        cell_index = self.get_cell_index(behavior_descriptor)
        
        if cell_index not in self.cells or individual.fitness > self.cells[cell_index].fitness:
            self.cells[cell_index] = copy.deepcopy(individual)
            return True
        return False
    
    def get_random_individual(self) -> Optional[Individual]:
        """Get a random individual from the archive."""
        if not self.cells:
            return None
        return random.choice(list(self.cells.values()))
    
    def get_filled_cells(self) -> List[Tuple]:
        """Get list of filled cell indices."""
        return list(self.cells.keys())
    
    def get_coverage(self) -> float:
        """Get coverage (fraction of filled cells)."""
        return len(self.cells) / max(self.total_cells, 1)
    
    def get_qd_score(self) -> float:
        """Get Quality-Diversity score (sum of all fitnesses)."""
        return sum(ind.fitness for ind in self.cells.values())
    
    def get_max_fitness(self) -> float:
        """Get maximum fitness in archive."""
        return max((ind.fitness for ind in self.cells.values()), default=0.0)
    
    def get_avg_fitness(self) -> float:
        """Get average fitness in archive."""
        if not self.cells:
            return 0.0
        return np.mean([ind.fitness for ind in self.cells.values()])


class BehaviorDescriptorExtractor:
    """Extracts behavior descriptors from UTT configurations."""
    
    def __init__(self):
        """Initialize behavior descriptor extractor."""
        pass
    
    def extract_descriptor(self, utt_data: Dict) -> np.ndarray:
        """Extract behavior descriptor from UTT data."""
        descriptors = []
        
        # Get unit types (excluding Resource)
        unit_types = [unit for unit in utt_data["unitTypes"] if unit["name"] != "Resource"]
        
        # 1. Average unit cost (economic dimension)
        avg_cost = np.mean([unit["cost"] for unit in unit_types])
        descriptors.append(avg_cost)
        
        # 2. Average unit HP (durability dimension)
        avg_hp = np.mean([unit["hp"] for unit in unit_types])
        descriptors.append(avg_hp)
        
        # 3. Average damage per second (DPS dimension)
        dps_values = []
        for unit in unit_types:
            if unit["attackTime"] > 0:
                avg_damage = (unit["minDamage"] + unit["maxDamage"]) / 2
                dps = avg_damage / unit["attackTime"]
                dps_values.append(dps)
        avg_dps = np.mean(dps_values) if dps_values else 0
        descriptors.append(avg_dps)
        
        # 4. Average movement speed (mobility dimension)
        speed_values = []
        for unit in unit_types:
            if unit["moveTime"] > 0:
                speed = 1.0 / unit["moveTime"]  # Higher speed = lower moveTime
                speed_values.append(speed)
        avg_speed = np.mean(speed_values) if speed_values else 0
        descriptors.append(avg_speed)
        
        # 5. Average attack range (range dimension)
        avg_range = np.mean([unit["attackRange"] for unit in unit_types])
        descriptors.append(avg_range)
        
        # 6. Production time ratio (production speed dimension)
        production_times = [unit["produceTime"] for unit in unit_types if unit["produceTime"] > 0]
        avg_production_time = np.mean(production_times) if production_times else 0
        descriptors.append(avg_production_time)
        
        return np.array(descriptors)


class MAPElitesAlgorithm:
    """MAP-Elites algorithm for evolving diverse UTT configurations."""
    
    def __init__(self,
                 archive: MAPElitesArchive,
                 encoder: UTTGeneEncoder,
                 fitness_evaluator: GameSimulationFitness,
                 behavior_extractor: BehaviorDescriptorExtractor,
                 iterations: int = 1000,
                 batch_size: int = 10,
                 mutation_rate: float = 0.1):
        """Initialize MAP-Elites algorithm."""
        self.archive = archive
        self.encoder = encoder
        self.fitness_evaluator = fitness_evaluator
        self.behavior_extractor = behavior_extractor
        self.iterations = iterations
        self.batch_size = batch_size
        self.mutation_rate = mutation_rate
        
        self.iteration = 0
        self.stats = {
            "coverage": [],
            "qd_score": [],
            "max_fitness": [],
            "avg_fitness": [],
            "improvements": []
        }
    
    def initialize_archive(self, num_initial: int = 100):
        """Initialize archive with random individuals."""
        print(f"Initializing archive with {num_initial} random individuals...")
        
        for _ in range(num_initial):
            # Generate random genome
            genome = np.random.random(self.encoder.genome_size)
            utt_data = self.encoder.decode_genome_to_utt(genome)
            
            # Create individual
            individual = Individual(genome=genome, utt_data=utt_data)
            
            # Evaluate fitness
            individual.fitness = self.fitness_evaluator.evaluate(individual)
            
            # Extract behavior descriptor
            behavior_descriptor = self.behavior_extractor.extract_descriptor(utt_data)
            
            # Add to archive
            self.archive.add_individual(individual, behavior_descriptor)
        
        print(f"Archive initialized with {len(self.archive.cells)} filled cells")
    
    def evolve_batch(self):
        """Evolve one batch of individuals."""
        new_individuals = []
        
        for _ in range(self.batch_size):
            # Select parent from archive
            parent = self.archive.get_random_individual()
            if parent is None:
                # If archive is empty, create random individual
                genome = np.random.random(self.encoder.genome_size)
                utt_data = self.encoder.decode_genome_to_utt(genome)
                parent = Individual(genome=genome, utt_data=utt_data)
            
            # Create offspring through mutation
            offspring = self._mutate_individual(parent)
            
            # Evaluate offspring
            offspring.fitness = self.fitness_evaluator.evaluate(offspring)
            
            # Extract behavior descriptor
            behavior_descriptor = self.behavior_extractor.extract_descriptor(offspring.utt_data)
            
            # Add to archive
            improved = self.archive.add_individual(offspring, behavior_descriptor)
            if improved:
                new_individuals.append(offspring)
        
        return new_individuals
    
    def _mutate_individual(self, individual: Individual) -> Individual:
        """Create mutated offspring from parent."""
        mutated_genome = individual.genome.copy()
        
        for i in range(len(mutated_genome)):
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation_strength = 0.1
                mutation = np.random.normal(0, mutation_strength)
                mutated_genome[i] = np.clip(mutated_genome[i] + mutation, 0.0, 1.0)
        
        mutated_utt = self.encoder.decode_genome_to_utt(mutated_genome)
        return Individual(genome=mutated_genome, utt_data=mutated_utt)
    
    def run(self):
        """Run MAP-Elites algorithm."""
        print("Starting MAP-Elites algorithm...")
        print(f"Iterations: {self.iterations}")
        print(f"Batch size: {self.batch_size}")
        print(f"Archive dimensions: {self.archive.dimensions}")
        print(f"Archive resolution: {self.archive.resolution}")
        print(f"Total cells: {self.archive.total_cells}")
        
        # Initialize archive
        self.initialize_archive()
        
        # Main evolution loop
        for iteration in range(self.iterations):
            self.iteration = iteration
            
            # Evolve batch
            new_individuals = self.evolve_batch()
            
            # Record statistics
            self._record_stats(len(new_individuals))
            
            # Print progress
            if iteration % 50 == 0:
                self._print_progress()
        
        print("MAP-Elites algorithm completed!")
        self._print_final_stats()
        
        return self.archive
    
    def _record_stats(self, improvements: int):
        """Record algorithm statistics."""
        self.stats["coverage"].append(self.archive.get_coverage())
        self.stats["qd_score"].append(self.archive.get_qd_score())
        self.stats["max_fitness"].append(self.archive.get_max_fitness())
        self.stats["avg_fitness"].append(self.archive.get_avg_fitness())
        self.stats["improvements"].append(improvements)
    
    def _print_progress(self):
        """Print current progress."""
        coverage = self.archive.get_coverage()
        qd_score = self.archive.get_qd_score()
        max_fitness = self.archive.get_max_fitness()
        avg_fitness = self.archive.get_avg_fitness()
        
        print(f"Iteration {self.iteration}: "
              f"Coverage={coverage:.3f}, "
              f"QD-Score={qd_score:.3f}, "
              f"Max Fitness={max_fitness:.3f}, "
              f"Avg Fitness={avg_fitness:.3f}")
    
    def _print_final_stats(self):
        """Print final statistics."""
        print("\n=== Final Statistics ===")
        print(f"Total filled cells: {len(self.archive.cells)}")
        print(f"Coverage: {self.archive.get_coverage():.3f}")
        print(f"QD-Score: {self.archive.get_qd_score():.3f}")
        print(f"Max Fitness: {self.archive.get_max_fitness():.3f}")
        print(f"Avg Fitness: {self.archive.get_avg_fitness():.3f}")
    
    def save_results(self, output_dir: str):
        """Save results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save archive
        archive_data = {
            "cells": {},
            "dimensions": self.archive.dimensions,
            "bounds": self.archive.bounds,
            "resolution": self.archive.resolution,
            "stats": self.stats
        }
        
        for cell_index, individual in self.archive.cells.items():
            archive_data["cells"][str(cell_index)] = {
                "genome": individual.genome.tolist(),
                "fitness": individual.fitness,
                "utt_data": individual.utt_data,
                "metadata": individual.metadata
            }
        
        archive_path = os.path.join(output_dir, "map_elites_archive.json")
        with open(archive_path, 'w') as f:
            json.dump(archive_data, f, indent=2)
        
        # Save best individuals
        best_individuals = sorted(self.archive.cells.values(), 
                                key=lambda x: x.fitness, reverse=True)[:10]
        
        for i, individual in enumerate(best_individuals):
            best_path = os.path.join(output_dir, f"best_individual_{i+1}.json")
            with open(best_path, 'w') as f:
                json.dump(individual.utt_data, f, indent=2)
        
        # Save statistics
        stats_path = os.path.join(output_dir, "map_elites_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Results saved to: {output_dir}")
    
    def plot_results(self, output_dir: str):
        """Create visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot fitness evolution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.stats["max_fitness"])
        plt.title("Max Fitness Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("Max Fitness")
        
        plt.subplot(2, 2, 2)
        plt.plot(self.stats["avg_fitness"])
        plt.title("Average Fitness Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("Average Fitness")
        
        plt.subplot(2, 2, 3)
        plt.plot(self.stats["coverage"])
        plt.title("Archive Coverage Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("Coverage")
        
        plt.subplot(2, 2, 4)
        plt.plot(self.stats["qd_score"])
        plt.title("QD-Score Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("QD-Score")
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "map_elites_evolution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create behavior space visualization (2D projection)
        if len(self.archive.dimensions) >= 2:
            self._plot_behavior_space(output_dir)
        
        print(f"Plots saved to: {output_dir}")
    
    def _plot_behavior_space(self, output_dir: str):
        """Create 2D behavior space visualization."""
        if len(self.archive.cells) == 0:
            return
        
        # Extract behavior descriptors for all individuals
        descriptors = []
        fitnesses = []
        
        for individual in self.archive.cells.values():
            bd = self.behavior_extractor.extract_descriptor(individual.utt_data)
            descriptors.append(bd[:2])  # Use first 2 dimensions
            fitnesses.append(individual.fitness)
        
        descriptors = np.array(descriptors)
        fitnesses = np.array(fitnesses)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(descriptors[:, 0], descriptors[:, 1], 
                            c=fitnesses, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Fitness')
        plt.xlabel(f'Behavior Dimension 1: {self.archive.dimensions[0]}')
        plt.ylabel(f'Behavior Dimension 2: {self.archive.dimensions[1]}')
        plt.title('MAP-Elites Behavior Space')
        
        plot_path = os.path.join(output_dir, "behavior_space.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_utt_archive(dimensions: List[str], 
                      bounds: List[Tuple[float, float]], 
                      resolution: List[int]) -> MAPElitesArchive:
    """Create MAP-Elites archive for UTT evolution."""
    return MAPElitesArchive(
        dimensions=dimensions,
        bounds=bounds,
        resolution=resolution
    )


def main():
    """Main function to run MAP-Elites algorithm."""
    # Configuration
    base_utt_path = "/home/altaaf/projects/MicroRTS-Py-Research/gym_microrts/microrts/utts/AsymmetricP1UTT.json"
    map_paths = ["maps/10x10/basesTwoWorkers10x10.xml"]
    output_dir = "/home/altaaf/projects/MicroRTS-Py-Research/experiments/map_elites_results"
    
    # Create archive with behavior dimensions
    dimensions = [
        "avg_cost",      # Economic dimension
        "avg_hp",        # Durability dimension  
        "avg_dps",       # Damage dimension
        "avg_speed",     # Mobility dimension
        "avg_range",     # Range dimension
        "avg_production" # Production dimension
    ]
    
    bounds = [
        (1.0, 15.0),    # avg_cost
        (3.0, 50.0),    # avg_hp
        (0.0, 2.0),     # avg_dps
        (0.05, 0.2),    # avg_speed
        (1.0, 4.0),     # avg_range
        (50.0, 200.0)   # avg_production
    ]
    
    resolution = [10, 10, 10, 10, 10, 10]  # 10x10x10x10x10x10 grid
    
    archive = create_utt_archive(dimensions, bounds, resolution)
    
    # Initialize components
    encoder = UTTGeneEncoder(base_utt_path)
    fitness_evaluator = GameSimulationFitness(
        map_paths=map_paths,
        num_games=2,  # Reduced for faster testing
        max_steps=500
    )
    behavior_extractor = BehaviorDescriptorExtractor()
    
    # Create and run MAP-Elites
    map_elites = MAPElitesAlgorithm(
        archive=archive,
        encoder=encoder,
        fitness_evaluator=fitness_evaluator,
        behavior_extractor=behavior_extractor,
        iterations=200,  # Reduced for testing
        batch_size=5,
        mutation_rate=0.1
    )
    
    # Run algorithm
    final_archive = map_elites.run()
    
    # Save results
    map_elites.save_results(output_dir)
    map_elites.plot_results(output_dir)
    
    print(f"\nMAP-Elites completed! Results saved to: {output_dir}")
    print(f"Final archive size: {len(final_archive.cells)} cells")
    print(f"Final coverage: {final_archive.get_coverage():.3f}")
    print(f"Final QD-score: {final_archive.get_qd_score():.3f}")


if __name__ == "__main__":
    main()
