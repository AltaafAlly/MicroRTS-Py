"""
Genetic Algorithm for evolving MicroRTS Unit Type Table (UTT) parameters.

This implementation evolves unit stats while preserving the game's tech tree structure
and maintaining valid constraints between unit parameters.
"""

import json
import random
import numpy as np
import copy
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


@dataclass
class GeneBounds:
    """Defines the bounds for a gene parameter."""
    min_val: float
    max_val: float
    is_int: bool = True
    step: float = 1.0


@dataclass
class UnitTypeConstraints:
    """Constraints for a specific unit type."""
    name: str
    # Evolvable parameters with bounds
    cost: GeneBounds = field(default_factory=lambda: GeneBounds(1, 20, True))
    hp: GeneBounds = field(default_factory=lambda: GeneBounds(1, 100, True))
    min_damage: GeneBounds = field(default_factory=lambda: GeneBounds(1, 10, True))
    max_damage: GeneBounds = field(default_factory=lambda: GeneBounds(1, 10, True))
    attack_range: GeneBounds = field(default_factory=lambda: GeneBounds(1, 6, True))
    produce_time: GeneBounds = field(default_factory=lambda: GeneBounds(10, 300, True))
    move_time: GeneBounds = field(default_factory=lambda: GeneBounds(5, 20, True))
    attack_time: GeneBounds = field(default_factory=lambda: GeneBounds(3, 15, True))
    harvest_time: GeneBounds = field(default_factory=lambda: GeneBounds(10, 30, True))
    return_time: GeneBounds = field(default_factory=lambda: GeneBounds(5, 15, True))
    harvest_amount: GeneBounds = field(default_factory=lambda: GeneBounds(1, 5, True))
    sight_radius: GeneBounds = field(default_factory=lambda: GeneBounds(1, 10, True))
    
    # Special constraints
    def validate_ranged_constraints(self, unit_data: Dict) -> bool:
        """Validate special constraints for Ranged units."""
        if self.name == "Ranged":
            return (unit_data["attackRange"] >= 2 and 
                   unit_data["minDamage"] <= unit_data["maxDamage"])
        return True
    
    def validate_damage_constraints(self, unit_data: Dict) -> bool:
        """Validate that minDamage <= maxDamage."""
        return unit_data["minDamage"] <= unit_data["maxDamage"]


class UTTGeneEncoder:
    """Encodes/decodes UTT parameters to/from genetic representation."""
    
    def __init__(self, base_utt_path: str):
        """Initialize with base UTT configuration."""
        with open(base_utt_path, 'r') as f:
            self.base_utt = json.load(f)
        
        # Define unit type constraints
        self.unit_constraints = {
            "Base": UnitTypeConstraints("Base", 
                cost=GeneBounds(10, 25, True),
                hp=GeneBounds(60, 120, True),
                sight_radius=GeneBounds(6, 12, True)),
            
            "Barracks": UnitTypeConstraints("Barracks",
                cost=GeneBounds(5, 15, True),
                hp=GeneBounds(30, 60, True),
                sight_radius=GeneBounds(3, 6, True)),
            
            "Worker": UnitTypeConstraints("Worker",
                cost=GeneBounds(1, 4, True),
                hp=GeneBounds(2, 6, True),
                min_damage=GeneBounds(1, 3, True),
                max_damage=GeneBounds(1, 3, True),
                move_time=GeneBounds(8, 16, True),
                attack_time=GeneBounds(4, 8, True),
                harvest_time=GeneBounds(15, 25, True),
                return_time=GeneBounds(8, 15, True),
                harvest_amount=GeneBounds(1, 4, True),
                sight_radius=GeneBounds(3, 6, True)),
            
            "Light": UnitTypeConstraints("Light",
                cost=GeneBounds(2, 5, True),
                hp=GeneBounds(5, 12, True),
                min_damage=GeneBounds(2, 5, True),
                max_damage=GeneBounds(2, 5, True),
                move_time=GeneBounds(6, 12, True),
                attack_time=GeneBounds(4, 8, True),
                sight_radius=GeneBounds(3, 6, True)),
            
            "Heavy": UnitTypeConstraints("Heavy",
                cost=GeneBounds(4, 10, True),
                hp=GeneBounds(10, 25, True),
                min_damage=GeneBounds(4, 8, True),
                max_damage=GeneBounds(4, 8, True),
                move_time=GeneBounds(8, 16, True),
                attack_time=GeneBounds(4, 8, True),
                sight_radius=GeneBounds(2, 5, True)),
            
            "Ranged": UnitTypeConstraints("Ranged",
                cost=GeneBounds(3, 7, True),
                hp=GeneBounds(2, 6, True),
                min_damage=GeneBounds(1, 3, True),  # Keep low damage
                max_damage=GeneBounds(2, 4, True),  # Keep low damage
                attack_range=GeneBounds(2, 6, True),  # Must be >= 2
                move_time=GeneBounds(8, 16, True),
                attack_time=GeneBounds(4, 8, True),
                sight_radius=GeneBounds(3, 6, True))
        }
        
        # Global parameters
        self.global_bounds = {
            "moveConflictResolutionStrategy": GeneBounds(0, 2, True)
        }
        
        # Build gene mapping
        self._build_gene_mapping()
    
    def _build_gene_mapping(self):
        """Build mapping from gene indices to UTT parameters."""
        self.gene_mapping = []
        self.gene_bounds = []
        
        # Add global parameters
        for param, bounds in self.global_bounds.items():
            self.gene_mapping.append(("global", param))
            self.gene_bounds.append(bounds)
        
        # Add unit-specific parameters
        for unit_type in self.base_utt["unitTypes"]:
            unit_name = unit_type["name"]
            if unit_name == "Resource":  # Skip resource units
                continue
                
            constraints = self.unit_constraints.get(unit_name)
            if not constraints:
                continue
                
            # Add all evolvable parameters for this unit
            for param_name in ["cost", "hp", "minDamage", "maxDamage", "attackRange", 
                             "produceTime", "moveTime", "attackTime", "harvestTime", 
                             "returnTime", "harvestAmount", "sightRadius"]:
                if hasattr(constraints, param_name.lower()):
                    bounds = getattr(constraints, param_name.lower())
                    self.gene_mapping.append((unit_name, param_name))
                    self.gene_bounds.append(bounds)
        
        self.genome_size = len(self.gene_mapping)
        print(f"Genome size: {self.genome_size} genes")
    
    def encode_utt_to_genome(self, utt_data: Dict) -> np.ndarray:
        """Convert UTT data to genome representation."""
        genome = np.zeros(self.genome_size)
        
        for i, (scope, param) in enumerate(self.gene_mapping):
            bounds = self.gene_bounds[i]
            
            if scope == "global":
                value = utt_data.get(param, 0)
            else:
                # Find the unit type
                unit_data = None
                for unit in utt_data["unitTypes"]:
                    if unit["name"] == scope:
                        unit_data = unit
                        break
                
                if unit_data:
                    value = unit_data.get(param, 0)
                else:
                    value = bounds.min_val
            
            # Normalize to [0, 1] range
            if bounds.max_val > bounds.min_val:
                normalized = (value - bounds.min_val) / (bounds.max_val - bounds.min_val)
                genome[i] = np.clip(normalized, 0.0, 1.0)
            else:
                genome[i] = 0.0
        
        return genome
    
    def decode_genome_to_utt(self, genome: np.ndarray) -> Dict:
        """Convert genome to UTT data."""
        utt_data = copy.deepcopy(self.base_utt)
        
        for i, (scope, param) in enumerate(self.gene_mapping):
            bounds = self.gene_bounds[i]
            
            # Denormalize from [0, 1] to actual range
            normalized = np.clip(genome[i], 0.0, 1.0)
            value = bounds.min_val + normalized * (bounds.max_val - bounds.min_val)
            
            # Round to integer if needed
            if bounds.is_int:
                value = int(round(value))
                value = max(bounds.min_val, min(bounds.max_val, value))
            
            if scope == "global":
                utt_data[param] = value
            else:
                # Find and update the unit type
                for unit in utt_data["unitTypes"]:
                    if unit["name"] == scope:
                        unit[param] = value
                        break
        
        # Apply constraints
        self._apply_constraints(utt_data)
        
        return utt_data
    
    def _apply_constraints(self, utt_data: Dict):
        """Apply validation constraints to the UTT."""
        for unit in utt_data["unitTypes"]:
            unit_name = unit["name"]
            
            # Ensure minDamage <= maxDamage
            if unit["minDamage"] > unit["maxDamage"]:
                unit["maxDamage"] = unit["minDamage"]
            
            # Special constraints for Ranged units
            if unit_name == "Ranged":
                unit["attackRange"] = max(2, unit["attackRange"])
                # Keep damage low for ranged units
                unit["minDamage"] = min(3, unit["minDamage"])
                unit["maxDamage"] = min(4, unit["maxDamage"])


@dataclass
class Individual:
    """Represents a single individual in the population."""
    genome: np.ndarray
    fitness: float = 0.0
    utt_data: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.utt_data is None:
            self.utt_data = {}


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation."""
    
    @abstractmethod
    def evaluate(self, individual: Individual) -> float:
        """Evaluate fitness of an individual."""
        pass


class GameSimulationFitness(FitnessEvaluator):
    """Fitness evaluator based on game simulations."""
    
    def __init__(self, 
                 map_paths: List[str],
                 num_games: int = 5,
                 max_steps: int = 2000,
                 opponent_ai: str = "RandomBiasedAI"):
        """Initialize fitness evaluator."""
        self.map_paths = map_paths
        self.num_games = num_games
        self.max_steps = max_steps
        self.opponent_ai = opponent_ai
        
    def evaluate(self, individual: Individual) -> float:
        """Evaluate fitness through game simulations."""
        try:
            # Create temporary UTT file
            temp_utt_path = f"/tmp/temp_utt_{random.randint(10000, 99999)}.json"
            with open(temp_utt_path, 'w') as f:
                json.dump(individual.utt_data, f)
            
            total_wins = 0
            total_games = 0
            
            for _ in range(self.num_games):
                try:
                    # Create environment with evolved UTT
                    env = MicroRTSGridModeVecEnv(
                        num_selfplay_envs=0,
                        num_bot_envs=1,
                        max_steps=self.max_steps,
                        map_paths=self.map_paths,
                        utt_json_p0=temp_utt_path,
                        utt_json_p1=None,  # Use default UTT for opponent
                        autobuild=True
                    )
                    
                    # Run game simulation
                    obs = env.reset()
                    done = False
                    steps = 0
                    
                    while not done and steps < self.max_steps:
                        # Random actions for both players
                        actions = [env.action_space.sample() for _ in range(env.num_envs)]
                        obs, rewards, dones, infos = env.step(actions)
                        done = any(dones)
                        steps += 1
                    
                    # Check winner (simplified - in practice you'd need more sophisticated evaluation)
                    if steps < self.max_steps:
                        # Game ended early, check final state
                        total_wins += 0.5  # Neutral result
                    else:
                        # Game timed out, evaluate based on resources/units
                        total_wins += 0.5  # Neutral result
                    
                    total_games += 1
                    env.close()
                    
                except Exception as e:
                    print(f"Game simulation error: {e}")
                    total_games += 1
                    total_wins += 0.0  # Penalty for failed games
            
            # Clean up
            if os.path.exists(temp_utt_path):
                os.remove(temp_utt_path)
            
            # Return win rate as fitness
            fitness = total_wins / max(total_games, 1)
            return fitness
            
        except Exception as e:
            print(f"Fitness evaluation error: {e}")
            return 0.0


class GeneticAlgorithm:
    """Main genetic algorithm implementation."""
    
    def __init__(self,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5,
                 tournament_size: int = 3,
                 encoder: UTTGeneEncoder = None,
                 fitness_evaluator: FitnessEvaluator = None):
        """Initialize genetic algorithm."""
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.encoder = encoder
        self.fitness_evaluator = fitness_evaluator
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        
    def initialize_population(self):
        """Initialize random population."""
        print("Initializing population...")
        self.population = []
        
        for _ in range(self.population_size):
            # Generate random genome
            genome = np.random.random(self.encoder.genome_size)
            
            # Decode to UTT
            utt_data = self.encoder.decode_genome_to_utt(genome)
            
            individual = Individual(genome=genome, utt_data=utt_data)
            self.population.append(individual)
        
        print(f"Initialized population of {len(self.population)} individuals")
    
    def evaluate_population(self):
        """Evaluate fitness for entire population."""
        print(f"Evaluating generation {self.generation}...")
        
        for i, individual in enumerate(self.population):
            if individual.fitness == 0.0:  # Only evaluate if not already evaluated
                individual.fitness = self.fitness_evaluator.evaluate(individual)
                print(f"Individual {i+1}/{len(self.population)}: fitness = {individual.fitness:.3f}")
        
        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(current_best)
        
        # Record statistics
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        self.fitness_history.append(avg_fitness)
        
        print(f"Generation {self.generation} - Avg fitness: {avg_fitness:.3f}, Best: {self.best_individual.fitness:.3f}")
    
    def tournament_selection(self) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # Uniform crossover
        mask = np.random.random(self.encoder.genome_size) < 0.5
        
        child1_genome = np.where(mask, parent1.genome, parent2.genome)
        child2_genome = np.where(mask, parent2.genome, parent1.genome)
        
        child1_utt = self.encoder.decode_genome_to_utt(child1_genome)
        child2_utt = self.encoder.decode_genome_to_utt(child2_genome)
        
        child1 = Individual(genome=child1_genome, utt_data=child1_utt)
        child2 = Individual(genome=child2_genome, utt_data=child2_utt)
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """Apply mutation to individual."""
        mutated_genome = individual.genome.copy()
        
        for i in range(len(mutated_genome)):
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation_strength = 0.1
                mutation = np.random.normal(0, mutation_strength)
                mutated_genome[i] = np.clip(mutated_genome[i] + mutation, 0.0, 1.0)
        
        mutated_utt = self.encoder.decode_genome_to_utt(mutated_genome)
        return Individual(genome=mutated_genome, utt_data=mutated_utt)
    
    def evolve_generation(self):
        """Evolve one generation."""
        new_population = []
        
        # Elitism - keep best individuals
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elite_size]
        new_population.extend([copy.deepcopy(ind) for ind in elite])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def run(self):
        """Run the complete genetic algorithm."""
        print("Starting Genetic Algorithm...")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Genome size: {self.encoder.genome_size}")
        
        self.initialize_population()
        
        for gen in range(self.generations):
            self.evaluate_population()
            
            if gen < self.generations - 1:  # Don't evolve after last generation
                self.evolve_generation()
        
        print("Genetic Algorithm completed!")
        print(f"Best fitness achieved: {self.best_individual.fitness:.3f}")
        
        return self.best_individual


def main():
    """Main function to run the genetic algorithm."""
    # Configuration
    base_utt_path = "/home/altaaf/projects/MicroRTS-Py-Research/gym_microrts/microrts/utts/AsymmetricP1UTT.json"
    map_paths = ["maps/10x10/basesTwoWorkers10x10.xml"]
    
    # Initialize components
    encoder = UTTGeneEncoder(base_utt_path)
    fitness_evaluator = GameSimulationFitness(
        map_paths=map_paths,
        num_games=3,  # Reduced for faster testing
        max_steps=1000
    )
    
    # Create and run genetic algorithm
    ga = GeneticAlgorithm(
        population_size=20,  # Reduced for faster testing
        generations=10,      # Reduced for faster testing
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=3,
        encoder=encoder,
        fitness_evaluator=fitness_evaluator
    )
    
    best_individual = ga.run()
    
    # Save best result
    output_path = "/home/altaaf/projects/MicroRTS-Py-Research/experiments/best_evolved_utt.json"
    with open(output_path, 'w') as f:
        json.dump(best_individual.utt_data, f, indent=2)
    
    print(f"Best evolved UTT saved to: {output_path}")
    
    # Save fitness history
    history_path = "/home/altaaf/projects/MicroRTS-Py-Research/experiments/fitness_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            "fitness_history": ga.fitness_history,
            "best_fitness": best_individual.fitness,
            "generations": ga.generations,
            "population_size": ga.population_size
        }, f, indent=2)
    
    print(f"Fitness history saved to: {history_path}")


if __name__ == "__main__":
    main()
