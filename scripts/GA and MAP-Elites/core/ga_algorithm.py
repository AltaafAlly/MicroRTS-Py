"""
Main Genetic Algorithm Implementation for MicroRTS Parameter Evolution

This module implements the complete genetic algorithm that evolves MicroRTS
game configurations to achieve balanced, engaging, and strategically diverse gameplay.
"""

import random
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Sequence
from dataclasses import dataclass, field, fields
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
    # Slightly prioritize balance, but still leave room for duration/diversity
    fitness_alpha: float = 0.4  # Balance weight
    fitness_beta: float = 0.3   # Duration weight
    fitness_gamma: float = 0.3  # Strategy diversity weight
    target_duration: int = 200
    duration_tolerance: int = 50
    
    # Real MicroRTS settings
    use_real_microrts: bool = True  # Use real MicroRTS instead of simulation
    use_working_evaluator: bool = False  # Use working UTT evaluator
    max_steps: int = 300  # Max steps per game
    map_path: str = "maps/8x8/basesWorkers8x8A.xml"  # Map to use
    map_paths: Optional[Sequence[str]] = None  # If set, eval on each map and aggregate for balance signal (e.g. 8-7)
    games_per_evaluation: int = 3  # Games per chromosome evaluation
    # Optional: restrict to specific AIs (e.g. ["lightRushAI", "workerRushAI"] for single-matchup debugging)
    ai_agents: Optional[Sequence[str]] = None
    # If True, evaluator forces nondeterministic UTT (random move conflicts + damage ranges) so 1 map gives balance signal
    use_nondeterministic: bool = False
    # If True, run (ai1,ai2) and (ai2,ai1) and aggregate; balanced UTT gives ~60-60 so balance > 0
    use_both_orderings: bool = False
    
    # Termination criteria
    max_generations_without_improvement: int = 5
    target_fitness: float = 0.9
    # Diversity: every N generations replace one non-elite individual with a random chromosome (0 = off)
    random_immigrant_interval: int = 0
    
    # Output settings
    verbose: bool = True
    save_best_individuals: bool = True
    save_generation_stats: bool = True
    # Optional: directory to save every evaluated UTT (gen{N}_ind{M}.json) for comparison
    utt_log_dir: Optional[str] = None


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
    # Optional: per-generation best UTT for diffing and match log for CSV export
    best_individual_per_generation: Optional[List[Tuple[int, MicroRTSChromosome]]] = None
    run_match_log: Optional[List[Dict[str, Any]]] = None
    
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
        ai_list = list(self.config.ai_agents) if self.config.ai_agents else None
        use_nondet = getattr(self.config, "use_nondeterministic", False)
        use_both = getattr(self.config, "use_both_orderings", False)
        if self.config.use_working_evaluator:
            # Use the working UTT evaluator that bypasses the UTT loading bug
            self.fitness_evaluator = WorkingGAEvaluator(
                alpha=self.config.fitness_alpha,
                beta=self.config.fitness_beta,
                gamma=self.config.fitness_gamma,
                max_steps=self.config.max_steps,
                map_path=self.config.map_path,
                map_paths=getattr(self.config, "map_paths", None),
                games_per_eval=self.config.games_per_evaluation,
                ai_agents=ai_list,
                use_nondeterministic=use_nondet,
                use_both_orderings=use_both,
                target_duration=getattr(self.config, "target_duration", 500),
                duration_tolerance=getattr(self.config, "duration_tolerance", 400),
            )
        setattr(self.fitness_evaluator, "utt_log_dir", getattr(self.config, "utt_log_dir", None))
        if self.config.use_real_microrts:
            # Real MicroRTS evaluator removed due to UTT loading bug
            # Use working evaluator instead
            print("⚠️  Real MicroRTS evaluator has UTT loading bug. Using working evaluator instead.")
            self.fitness_evaluator = WorkingGAEvaluator(
                alpha=self.config.fitness_alpha,
                beta=self.config.fitness_beta,
                gamma=self.config.fitness_gamma,
                max_steps=self.config.max_steps,
                map_path=self.config.map_path,
                map_paths=getattr(self.config, "map_paths", None),
                games_per_eval=self.config.games_per_evaluation,
                ai_agents=ai_list,
                use_nondeterministic=use_nondet,
                use_both_orderings=use_both,
                target_duration=getattr(self.config, "target_duration", 500),
                duration_tolerance=getattr(self.config, "duration_tolerance", 400),
            )
            setattr(self.fitness_evaluator, "utt_log_dir", getattr(self.config, "utt_log_dir", None))
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
        
        # Checkpointing for resume capability
        self.checkpoint_dir = None
        self.current_generation = 0
        self.previous_total_time = 0.0  # Track time from previous runs when resuming
        
        # Optional: for run_ga_local_test CSV/plot (best UTT per gen, match log)
        self.best_individual_history: List[Tuple[int, MicroRTSChromosome]] = []
        self.run_match_log: List[Dict[str, Any]] = []
    
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
        
        # Optional: let evaluator log each match (generation, individual_index, ai1, ai2, wins, unit composition, etc.)
        # Set on evaluator every time (evaluator doesn't define these by default, so hasattr would be False)
        setattr(self.fitness_evaluator, "run_match_log", self.run_match_log)
        setattr(self.fitness_evaluator, "run_match_log_generation", generation)
        setattr(self.fitness_evaluator, "run_match_capture_composition", True)
        setattr(self.fitness_evaluator, "run_match_capture_snapshots", True)
        # Evaluate current population
        self.evaluate_population()
        
        # Update best individual
        self.update_best_individual()
        # Record best UTT this generation for later UTT-diff and CSV
        if self.best_individual is not None:
            self.best_individual_history.append((generation, self.best_individual.copy()))
        
        # Calculate generation statistics
        stats = self.calculate_generation_stats(generation, start_time)
        self.generation_stats.append(stats)
        
        # Print generation summary
        if self.config.verbose:
            # Count how many individuals are balanced (60-60) so evolution over time is visible
            n_balanced = sum(1 for s in self.fitness_scores if s.balance >= 0.99)
            pop_size = len(self.fitness_scores)
            # One-line progress so evolution over time is visible when scanning the log
            print(f"  Gen {generation}: best={stats.best_fitness:.3f} avg={stats.avg_fitness:.3f} balanced={n_balanced}/{pop_size}")
            print(f"Best fitness: {stats.best_fitness:.4f}")
            print(f"Average fitness: {stats.avg_fitness:.4f}")
            print(f"Balanced (60-60): {n_balanced}/{pop_size}  ← population evolving toward balance")
            print(f"Population diversity: {stats.population_diversity:.4f}")
            print(f"Convergence rate: {stats.convergence_rate:.4f}")
            print(f"Time elapsed: {stats.time_elapsed:.2f}s")
            # Single-matchup mode: show fitness for the one pair so we can see if it changes over generations
            if self.config.ai_agents and len(self.config.ai_agents) == 2 and self.best_fitness:
                a1, a2 = self.config.ai_agents[0], self.config.ai_agents[1]
                print(f"  Single matchup ({a1} vs {a2}): balance={self.best_fitness.balance:.4f}, duration={self.best_fitness.duration:.4f}, diversity={self.best_fitness.strategy_diversity:.4f}, overall={self.best_fitness.overall_fitness:.4f}")
        
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
                # Optional: inject a random immigrant every N generations to escape plateaus
                interval = getattr(self.config, 'random_immigrant_interval', 0)
                if interval > 0 and (generation + 1) % interval == 0 and len(self.population) > getattr(self.config, 'elite_size', 2):
                    elite_sz = self.config.elite_size
                    idx = random.randint(elite_sz, len(self.population) - 1)
                    self.population[idx] = create_population(1)[0]
                    if self.config.verbose:
                        print(f"  Injected random immigrant at index {idx}")
            except Exception as e:
                print(f"❌ Error during evolution: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    def save_checkpoint(self, checkpoint_dir: str, generation: int, current_run_time: float = 0.0) -> None:
        """
        Save current GA state to checkpoint for resuming later.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            generation: Current generation number
            current_run_time: Time elapsed in current run (for cumulative tracking)
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_data = {
            'generation': generation,
            'population': [json.loads(chrom.to_json()) for chrom in self.population],
            'fitness_scores': [
                {
                    'balance': f.balance,
                    'duration': f.duration,
                    'strategy_diversity': f.strategy_diversity,
                    'overall_fitness': f.overall_fitness
                } for f in self.fitness_scores
            ],
            'best_individual': json.loads(self.best_individual.to_json()) if self.best_individual else None,
            'best_fitness': {
                'balance': self.best_fitness.balance,
                'duration': self.best_fitness.duration,
                'strategy_diversity': self.best_fitness.strategy_diversity,
                'overall_fitness': self.best_fitness.overall_fitness
            } if self.best_fitness else None,
            'generation_stats': [stats.to_dict() for stats in self.generation_stats],
            'generations_without_improvement': self.generations_without_improvement,
            'last_best_fitness': self.last_best_fitness,
            'previous_total_time': self.previous_total_time + current_run_time,
            'config': {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'crossover_rate': self.config.crossover_rate,
                'mutation_rate': self.config.mutation_rate,
                'mutation_strength': self.config.mutation_strength,
                'tournament_size': self.config.tournament_size,
                'elite_size': self.config.elite_size,
                'fitness_alpha': self.config.fitness_alpha,
                'fitness_beta': self.config.fitness_beta,
                'fitness_gamma': self.config.fitness_gamma,
                # Persist evaluator / environment settings so resume behaves identically
                'use_real_microrts': self.config.use_real_microrts,
                'use_working_evaluator': getattr(self.config, 'use_working_evaluator', False),
                'max_steps': self.config.max_steps,
                'map_path': self.config.map_path,
                'map_paths': list(self.config.map_paths) if getattr(self.config, 'map_paths', None) else None,
                'games_per_evaluation': self.config.games_per_evaluation,
                'ai_agents': list(self.config.ai_agents) if self.config.ai_agents else None,
                'max_generations_without_improvement': self.config.max_generations_without_improvement,
                'random_immigrant_interval': getattr(self.config, 'random_immigrant_interval', 0),
            }
        }
        
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_gen_{generation}.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Also save latest checkpoint
        latest_checkpoint = os.path.join(checkpoint_dir, "checkpoint_latest.json")
        with open(latest_checkpoint, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        if self.config.verbose:
            print(f"Checkpoint saved: {checkpoint_file}")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_file: str) -> 'MicroRTSGeneticAlgorithm':
        """
        Load GA state from checkpoint.
        
        Args:
            checkpoint_file: Path to checkpoint JSON file
            
        Returns:
            MicroRTSGeneticAlgorithm instance restored from checkpoint
        """
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Recreate config (only pass known GAConfig fields for backward compatibility)
        config_dict = checkpoint_data['config']
        valid_keys = {f.name for f in fields(GAConfig)}
        config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = GAConfig(**config_dict)
        
        # Create GA instance
        ga = cls(config)
        
        # Restore population
        from .ga_chromosome import MicroRTSChromosome
        ga.population = [
            MicroRTSChromosome.from_json(json.dumps(chrom_data))
            for chrom_data in checkpoint_data['population']
        ]
        
        # Restore fitness scores
        from .ga_fitness_evaluator import FitnessComponents
        ga.fitness_scores = [
            FitnessComponents(**f_data)
            for f_data in checkpoint_data['fitness_scores']
        ]
        
        # Restore best individual
        if checkpoint_data['best_individual']:
            ga.best_individual = MicroRTSChromosome.from_json(
                json.dumps(checkpoint_data['best_individual'])
            )
        
        # Restore best fitness
        if checkpoint_data['best_fitness']:
            ga.best_fitness = FitnessComponents(**checkpoint_data['best_fitness'])
        
        # Restore generation stats
        ga.generation_stats = [
            GenerationStats(**stats_data)
            for stats_data in checkpoint_data['generation_stats']
        ]
        
        # Restore convergence tracking
        ga.generations_without_improvement = checkpoint_data['generations_without_improvement']
        ga.last_best_fitness = checkpoint_data['last_best_fitness']
        ga.current_generation = checkpoint_data['generation']
        
        # Restore previous total time if available
        if 'previous_total_time' in checkpoint_data:
            ga.previous_total_time = checkpoint_data['previous_total_time']
        elif ga.generation_stats:
            # Estimate from generation stats
            ga.previous_total_time = sum(stats.time_elapsed for stats in ga.generation_stats)
        
        return ga
    
    def run(self, checkpoint_dir: str = None, resume_from: str = None) -> GAResults:
        """
        Run the complete genetic algorithm.
        
        Args:
            checkpoint_dir: Directory to save checkpoints (optional)
            resume_from: Path to checkpoint file to resume from (optional)
        
        Returns:
            GAResults containing the best individual and evolution statistics
        """
        start_time = time.time()
        
        # Resume from checkpoint if provided
        if resume_from:
            if self.config.verbose:
                print("=" * 60)
                print("RESUMING FROM CHECKPOINT")
                print("=" * 60)
                print(f"Loading checkpoint: {resume_from}")
            
            # Load checkpoint
            resumed_ga = self.load_checkpoint(resume_from)
            # Copy state to self
            self.population = resumed_ga.population
            self.fitness_scores = resumed_ga.fitness_scores
            self.best_individual = resumed_ga.best_individual
            self.best_fitness = resumed_ga.best_fitness
            self.generation_stats = resumed_ga.generation_stats
            self.generations_without_improvement = resumed_ga.generations_without_improvement
            self.last_best_fitness = resumed_ga.last_best_fitness
            self.previous_total_time = resumed_ga.previous_total_time
            generation = resumed_ga.current_generation + 1  # Start from next generation
            
            if self.config.verbose:
                print(f"Resuming from generation {generation}")
                print(f"Best fitness so far: {self.best_fitness.overall_fitness:.4f}")
                print("=" * 60)
        else:
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
            
            # Initialize population
            self.initialize_population()
            generation = 0
            # Clear optional logs for this run (UTT history + match log for CSV/plot)
            self.run_match_log = []
            self.best_individual_history = []
        
        # Set checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        convergence_generation = None
        
        while not self.should_terminate(generation):
            gen_start_time = time.time()
            self.evolve_generation(generation)
            gen_elapsed = time.time() - gen_start_time
            
            # Save checkpoint after each generation
            if self.checkpoint_dir:
                self.save_checkpoint(self.checkpoint_dir, generation, current_run_time=time.time() - start_time)
            
            # Check for convergence
            if convergence_generation is None and self.generations_without_improvement >= 3:
                convergence_generation = generation
            
            generation += 1
        
        total_time = self.previous_total_time + (time.time() - start_time)
        
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
        
        # Release cached env in working evaluator so JVM/client can be closed
        if hasattr(self.fitness_evaluator, "close_cached_env"):
            self.fitness_evaluator.close_cached_env()
        
        # Create results
        results = GAResults(
            best_individual=self.best_individual or self.population[0],
            best_fitness=self.best_fitness or self.fitness_scores[0],
            generation_stats=self.generation_stats,
            total_generations=generation,
            total_time=total_time,
            convergence_generation=convergence_generation,
            best_individual_per_generation=self.best_individual_history if self.best_individual_history else None,
            run_match_log=self.run_match_log if self.run_match_log else None,
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
