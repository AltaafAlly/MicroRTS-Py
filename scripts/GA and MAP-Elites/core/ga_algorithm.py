"""
Main Genetic Algorithm Implementation for MicroRTS Parameter Evolution

This module implements the complete genetic algorithm that evolves MicroRTS
game configurations to achieve balanced, engaging, and strategically diverse gameplay.
"""

import random
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .ga_chromosome import MicroRTSChromosome, create_population
from .ga_fitness_evaluator import FitnessEvaluator, FitnessComponents, evaluate_population_fitness
# Removed ga_real_microrts_evaluator - has UTT loading bug
from .ga_working_evaluator import WorkingGAEvaluator, evaluate_population_fitness_working
from .ga_genetic_operators import GeneticOperators, SelectionConfig


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm."""
    
    # Population settings
    population_size: int = 20
    generations: int = 10
    
    # Genetic operator rates
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    # Selection settings
    tournament_size: int = 3
    elite_size: int = 2
    
    # Fitness evaluation settings
    fitness_alpha: float = 0.4  # Balance weight
    fitness_beta: float = 0.3   # Duration weight
    fitness_gamma: float = 0.3  # Strategy diversity weight
    target_duration: int = 200
    duration_tolerance: int = 50
    
    # Real MicroRTS settings
    use_real_microrts: bool = True  # Use real MicroRTS instead of simulation
    use_working_evaluator: bool = False  # Use working UTT evaluator
    max_steps: int = 300  # Max steps per game
    map_path: str = "maps/8x8/basesWorkers8x8L.xml"  # Map to use
    games_per_evaluation: int = 3  # Games per chromosome evaluation
    
    # Termination criteria
    max_generations_without_improvement: int = 5
    target_fitness: float = 0.9
    
    # Output settings
    verbose: bool = True
    save_best_individuals: bool = True
    save_generation_stats: bool = True


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    best_balance: float
    best_duration: float
    best_diversity: float
    population_diversity: float
    convergence_rate: float
    time_elapsed: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'worst_fitness': self.worst_fitness,
            'best_balance': self.best_balance,
            'best_duration': self.best_duration,
            'best_diversity': self.best_diversity,
            'population_diversity': self.population_diversity,
            'convergence_rate': self.convergence_rate,
            'time_elapsed': self.time_elapsed
        }


@dataclass
class GAResults:
    """Results from running the genetic algorithm."""
    
    best_individual: MicroRTSChromosome
    best_fitness: FitnessComponents
    generation_stats: List[GenerationStats]
    total_generations: int
    total_time: float
    convergence_generation: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'best_individual': self.best_individual.to_microrts_config(),
            'best_fitness': {
                'balance': self.best_fitness.balance,
                'duration': self.best_fitness.duration,
                'strategy_diversity': self.best_fitness.strategy_diversity,
                'overall_fitness': self.best_fitness.overall_fitness
            },
            'generation_stats': [stats.to_dict() for stats in self.generation_stats],
            'total_generations': self.total_generations,
            'total_time': self.total_time,
            'convergence_generation': self.convergence_generation
        }


class MicroRTSGeneticAlgorithm:
    """
    Main genetic algorithm class for evolving MicroRTS configurations.
    
    This class implements the complete evolutionary cycle:
    1. Initialize random population
    2. Evaluate fitness via AI vs AI matches
    3. Select parents
    4. Apply crossover and mutation
    5. Replace population
    6. Repeat until convergence or max generations
    """
    
    def __init__(self, config: GAConfig = None):
        """
        Initialize the genetic algorithm.
        
        Args:
            config: GA configuration parameters
        """
        self.config = config or GAConfig()
        
        # Initialize components
        if self.config.use_working_evaluator:
            # Use the working UTT evaluator that bypasses the UTT loading bug
            self.fitness_evaluator = WorkingGAEvaluator(
                alpha=self.config.fitness_alpha,
                beta=self.config.fitness_beta,
                gamma=self.config.fitness_gamma,
                max_steps=self.config.max_steps,
                map_path=self.config.map_path,
                games_per_eval=self.config.games_per_evaluation
            )
        elif self.config.use_real_microrts:
            # Real MicroRTS evaluator removed due to UTT loading bug
            # Use working evaluator instead
            print("⚠️  Real MicroRTS evaluator has UTT loading bug. Using working evaluator instead.")
            self.fitness_evaluator = WorkingGAEvaluator(
                alpha=self.config.fitness_alpha,
                beta=self.config.fitness_beta,
                gamma=self.config.fitness_gamma,
                max_steps=self.config.max_steps,
                map_path=self.config.map_path,
                games_per_eval=self.config.games_per_evaluation
            )
        else:
            self.fitness_evaluator = FitnessEvaluator(
                alpha=self.config.fitness_alpha,
                beta=self.config.fitness_beta,
                gamma=self.config.fitness_gamma,
                target_duration=self.config.target_duration,
                duration_tolerance=self.config.duration_tolerance
            )
        
        selection_config = SelectionConfig(
            tournament_size=self.config.tournament_size,
            elite_size=self.config.elite_size
        )
        
        self.genetic_operators = GeneticOperators(selection_config)
        
        # Evolution state
        self.population: List[MicroRTSChromosome] = []
        self.fitness_scores: List[FitnessComponents] = []
        self.generation_stats: List[GenerationStats] = []
        self.best_individual: Optional[MicroRTSChromosome] = None
        self.best_fitness: Optional[FitnessComponents] = None
        
        # Convergence tracking
        self.generations_without_improvement = 0
        self.last_best_fitness = 0.0
    
    def initialize_population(self) -> None:
        """Initialize the population with random chromosomes."""
        if self.config.verbose:
            print(f"Initializing population of {self.config.population_size} individuals...")
        
        self.population = create_population(self.config.population_size)
        
        if self.config.verbose:
            print("Population initialized successfully.")
    
    def evaluate_population(self) -> None:
        """Evaluate fitness for the entire population."""
        if self.config.verbose:
            evaluator_type = "real MicroRTS matches" if self.config.use_real_microrts else "simulation"
            print(f"Evaluating population fitness using {evaluator_type}...")
        
        start_time = time.time()
        
        try:
            if self.config.use_working_evaluator or self.config.use_real_microrts:
                # Both use the working evaluator now
                self.fitness_scores = evaluate_population_fitness_working(self.population, self.fitness_evaluator)
            else:
                self.fitness_scores = evaluate_population_fitness(self.population, self.fitness_evaluator)
        except Exception as e:
            print(f"❌ Error during fitness evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        evaluation_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"Population evaluation completed in {evaluation_time:.2f} seconds.")
    
    def update_best_individual(self) -> None:
        """Update the best individual found so far."""
        if not self.fitness_scores:
            return
        
        # Find best individual in current population
        best_idx = max(range(len(self.fitness_scores)), 
                      key=lambda i: self.fitness_scores[i].overall_fitness)
        
        current_best_fitness = self.fitness_scores[best_idx].overall_fitness
        
        # Update global best if improved
        if self.best_fitness is None or current_best_fitness > self.best_fitness.overall_fitness:
            self.best_individual = self.population[best_idx].copy()
            self.best_fitness = self.fitness_scores[best_idx]
            self.generations_without_improvement = 0
            self.last_best_fitness = current_best_fitness
            
            if self.config.verbose:
                print(f"New best individual found! Fitness: {current_best_fitness:.4f}")
        else:
            self.generations_without_improvement += 1
    
    def calculate_generation_stats(self, generation: int, start_time: float) -> GenerationStats:
        """Calculate statistics for the current generation."""
        if not self.fitness_scores:
            return GenerationStats(
                generation=generation,
                best_fitness=0.0,
                avg_fitness=0.0,
                worst_fitness=0.0,
                best_balance=0.0,
                best_duration=0.0,
                best_diversity=0.0,
                population_diversity=0.0,
                convergence_rate=0.0,
                time_elapsed=time.time() - start_time
            )
        
        # Calculate fitness statistics
        fitness_values = [score.overall_fitness for score in self.fitness_scores]
        balance_values = [score.balance for score in self.fitness_scores]
        duration_values = [score.duration for score in self.fitness_scores]
        diversity_values = [score.strategy_diversity for score in self.fitness_scores]
        
        best_fitness = max(fitness_values)
        avg_fitness = np.mean(fitness_values)
        worst_fitness = min(fitness_values)
        
        best_balance = max(balance_values)
        best_duration = max(duration_values)
        best_diversity = max(diversity_values)
        
        # Calculate population diversity
        population_diversity = self.genetic_operators.calculate_population_diversity(self.population)
        
        # Calculate convergence rate
        if self.last_best_fitness > 0:
            convergence_rate = (best_fitness - self.last_best_fitness) / self.last_best_fitness
        else:
            convergence_rate = 0.0
        
        return GenerationStats(
            generation=generation,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            worst_fitness=worst_fitness,
            best_balance=best_balance,
            best_duration=best_duration,
            best_diversity=best_diversity,
            population_diversity=population_diversity,
            convergence_rate=convergence_rate,
            time_elapsed=time.time() - start_time
        )
    
    def should_terminate(self, generation: int) -> bool:
        """Check if the algorithm should terminate."""
        # Check maximum generations
        if generation >= self.config.generations:
            if self.config.verbose:
                print(f"Terminating: Reached maximum generations ({self.config.generations})")
            return True
        
        # Check target fitness
        if self.best_fitness and self.best_fitness.overall_fitness >= self.config.target_fitness:
            if self.config.verbose:
                print(f"Terminating: Reached target fitness ({self.config.target_fitness})")
            return True
        
        # Check convergence (no improvement for several generations)
        if self.generations_without_improvement >= self.config.max_generations_without_improvement:
            if self.config.verbose:
                print(f"Terminating: No improvement for {self.generations_without_improvement} generations")
            return True
        
        return False
    
    def evolve_generation(self, generation: int) -> None:
        """Evolve one generation of the population."""
        if self.config.verbose:
            print(f"\n--- Generation {generation} ---")
        
        start_time = time.time()
        
        # Evaluate current population
        self.evaluate_population()
        
        # Update best individual
        self.update_best_individual()
        
        # Calculate generation statistics
        stats = self.calculate_generation_stats(generation, start_time)
        self.generation_stats.append(stats)
        
        # Print generation summary
        if self.config.verbose:
            print(f"Best fitness: {stats.best_fitness:.4f}")
            print(f"Average fitness: {stats.avg_fitness:.4f}")
            print(f"Population diversity: {stats.population_diversity:.4f}")
            print(f"Convergence rate: {stats.convergence_rate:.4f}")
            print(f"Time elapsed: {stats.time_elapsed:.2f}s")
        
        # Evolve population (except for last generation)
        if not self.should_terminate(generation + 1):
            if self.config.verbose:
                print("Evolving population...")
            
            try:
                self.population = self.genetic_operators.evolve_generation(
                    self.population,
                    self.fitness_scores,
                    self.config.crossover_rate,
                    self.config.mutation_rate
                )
            except Exception as e:
                print(f"❌ Error during evolution: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    def run(self) -> GAResults:
        """
        Run the complete genetic algorithm.
        
        Returns:
            GAResults containing the best individual and evolution statistics
        """
        if self.config.verbose:
            print("=" * 60)
            print("MICRO RTS GENETIC ALGORITHM")
            print("=" * 60)
            print(f"Population size: {self.config.population_size}")
            print(f"Generations: {self.config.generations}")
            print(f"Crossover rate: {self.config.crossover_rate}")
            print(f"Mutation rate: {self.config.mutation_rate}")
            print(f"Fitness weights: α={self.config.fitness_alpha}, β={self.config.fitness_beta}, γ={self.config.fitness_gamma}")
            print("=" * 60)
        
        start_time = time.time()
        
        # Initialize population
        self.initialize_population()
        
        # Run evolution
        generation = 0
        convergence_generation = None
        
        while not self.should_terminate(generation):
            self.evolve_generation(generation)
            
            # Check for convergence
            if convergence_generation is None and self.generations_without_improvement >= 3:
                convergence_generation = generation
            
            generation += 1
        
        total_time = time.time() - start_time
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("EVOLUTION COMPLETED")
            print("=" * 60)
            print(f"Total generations: {generation}")
            print(f"Total time: {total_time:.2f} seconds")
            if self.best_fitness:
                print(f"Best fitness: {self.best_fitness.overall_fitness:.4f}")
                print(f"Best balance: {self.best_fitness.balance:.4f}")
                print(f"Best duration: {self.best_fitness.duration:.4f}")
                print(f"Best diversity: {self.best_fitness.strategy_diversity:.4f}")
            print("=" * 60)
        
        # Create results
        results = GAResults(
            best_individual=self.best_individual or self.population[0],
            best_fitness=self.best_fitness or self.fitness_scores[0],
            generation_stats=self.generation_stats,
            total_generations=generation,
            total_time=total_time,
            convergence_generation=convergence_generation
        )
        
        return results
    
    def save_results(self, results: GAResults, filename: str = None) -> str:
        """
        Save results to a JSON file.
        
        Args:
            results: GAResults to save
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"ga_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        if self.config.verbose:
            print(f"Results saved to: {filename}")
        
        return filename
    
    def load_results(self, filename: str) -> GAResults:
        """
        Load results from a JSON file.
        
        Args:
            filename: Path to results file
            
        Returns:
            Loaded GAResults
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Reconstruct results
        best_individual = MicroRTSChromosome.from_json(json.dumps(data['best_individual']))
        best_fitness = FitnessComponents(**data['best_fitness'])
        
        generation_stats = []
        for stats_data in data['generation_stats']:
            generation_stats.append(GenerationStats(**stats_data))
        
        results = GAResults(
            best_individual=best_individual,
            best_fitness=best_fitness,
            generation_stats=generation_stats,
            total_generations=data['total_generations'],
            total_time=data['total_time'],
            convergence_generation=data.get('convergence_generation')
        )
        
        return results


def create_default_ga() -> MicroRTSGeneticAlgorithm:
    """Create a genetic algorithm with default configuration."""
    config = GAConfig()
    return MicroRTSGeneticAlgorithm(config)


def create_fast_ga() -> MicroRTSGeneticAlgorithm:
    """Create a genetic algorithm optimized for speed."""
    config = GAConfig(
        population_size=10,
        generations=5,
        verbose=True
    )
    return MicroRTSGeneticAlgorithm(config)


def create_comprehensive_ga() -> MicroRTSGeneticAlgorithm:
    """Create a genetic algorithm optimized for thoroughness."""
    config = GAConfig(
        population_size=50,
        generations=20,
        max_generations_without_improvement=10,
        verbose=True
    )
    return MicroRTSGeneticAlgorithm(config)
