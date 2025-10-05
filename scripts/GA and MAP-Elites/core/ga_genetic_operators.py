"""
Genetic Operators for MicroRTS Genetic Algorithm

This module implements selection, crossover, and mutation operators
for the genetic algorithm evolution process.
"""

import random
import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass

from .ga_chromosome import MicroRTSChromosome
from .ga_fitness_evaluator import FitnessComponents


@dataclass
class SelectionConfig:
    """Configuration for selection operators."""
    
    tournament_size: int = 3
    elite_size: int = 2
    selection_pressure: float = 1.5  # For rank-based selection


class SelectionOperator:
    """Base class for selection operators."""
    
    def select(self, population: List[MicroRTSChromosome], 
               fitness_scores: List[FitnessComponents], 
               num_parents: int) -> List[MicroRTSChromosome]:
        """
        Select parents from population based on fitness.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent chromosomes
        """
        raise NotImplementedError


class TournamentSelection(SelectionOperator):
    """
    Tournament selection operator.
    
    Randomly selects k individuals and chooses the best one.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament
        """
        self.tournament_size = tournament_size
    
    def select(self, population: List[MicroRTSChromosome], 
               fitness_scores: List[FitnessComponents], 
               num_parents: int) -> List[MicroRTSChromosome]:
        """Select parents using tournament selection."""
        parents = []
        
        for _ in range(num_parents):
            # Randomly select tournament participants
            tournament_indices = random.sample(range(len(population)), 
                                             min(self.tournament_size, len(population)))
            
            # Find the best individual in the tournament
            best_index = tournament_indices[0]
            best_fitness = fitness_scores[best_index].overall_fitness
            
            for idx in tournament_indices[1:]:
                if fitness_scores[idx].overall_fitness > best_fitness:
                    best_index = idx
                    best_fitness = fitness_scores[idx].overall_fitness
            
            parents.append(population[best_index])
        
        return parents


class RankBasedSelection(SelectionOperator):
    """
    Rank-based selection operator.
    
    Assigns selection probabilities based on rank rather than raw fitness.
    """
    
    def __init__(self, selection_pressure: float = 1.5):
        """
        Initialize rank-based selection.
        
        Args:
            selection_pressure: Selection pressure (1.0 = uniform, >1.0 = more selective)
        """
        self.selection_pressure = selection_pressure
    
    def select(self, population: List[MicroRTSChromosome], 
               fitness_scores: List[FitnessComponents], 
               num_parents: int) -> List[MicroRTSChromosome]:
        """Select parents using rank-based selection."""
        # Handle edge case of population size 1
        if len(population) == 1:
            return [population[0]] * num_parents
        
        # Sort population by fitness (descending)
        sorted_indices = sorted(range(len(population)), 
                              key=lambda i: fitness_scores[i].overall_fitness, 
                              reverse=True)
        
        # Calculate selection probabilities based on rank
        n = len(population)
        probabilities = []
        
        for rank in range(n):
            # Linear ranking: P(rank) = (2 - s) / n + 2 * s * (n - rank - 1) / (n * (n - 1))
            # Handle division by zero when n = 1 (already handled above)
            prob = (2 - self.selection_pressure) / n + \
                   2 * self.selection_pressure * (n - rank - 1) / (n * (n - 1))
            probabilities.append(prob)
        
        # Select parents based on probabilities
        parents = []
        for _ in range(num_parents):
            selected_index = np.random.choice(sorted_indices, p=probabilities)
            parents.append(population[selected_index])
        
        return parents


class ElitismSelection(SelectionOperator):
    """
    Elitism selection operator.
    
    Always keeps the best individuals in the population.
    """
    
    def __init__(self, elite_size: int = 2):
        """
        Initialize elitism selection.
        
        Args:
            elite_size: Number of elite individuals to preserve
        """
        self.elite_size = elite_size
    
    def select(self, population: List[MicroRTSChromosome], 
               fitness_scores: List[FitnessComponents], 
               num_parents: int) -> List[MicroRTSChromosome]:
        """Select elite individuals."""
        # Sort by fitness (descending)
        sorted_indices = sorted(range(len(population)), 
                              key=lambda i: fitness_scores[i].overall_fitness, 
                              reverse=True)
        
        # Select top elite individuals
        elite_count = min(self.elite_size, num_parents, len(population))
        elite_parents = [population[i] for i in sorted_indices[:elite_count]]
        
        return elite_parents


class CrossoverOperator:
    """Base class for crossover operators."""
    
    def crossover(self, parent1: MicroRTSChromosome, parent2: MicroRTSChromosome, 
                  crossover_rate: float = 0.7) -> Tuple[MicroRTSChromosome, MicroRTSChromosome]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            crossover_rate: Probability of crossover occurring
            
        Returns:
            Tuple of two offspring chromosomes
        """
        raise NotImplementedError


class SinglePointCrossover(CrossoverOperator):
    """Single-point crossover operator."""
    
    def crossover(self, parent1: MicroRTSChromosome, parent2: MicroRTSChromosome, 
                  crossover_rate: float = 0.7) -> Tuple[MicroRTSChromosome, MicroRTSChromosome]:
        """Perform single-point crossover."""
        return parent1.crossover(parent2, crossover_rate)


class UniformCrossover(CrossoverOperator):
    """Uniform crossover operator."""
    
    def crossover(self, parent1: MicroRTSChromosome, parent2: MicroRTSChromosome, 
                  crossover_rate: float = 0.7) -> Tuple[MicroRTSChromosome, MicroRTSChromosome]:
        """Perform uniform crossover."""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Convert to genomes
        genome1 = parent1.to_genome()
        genome2 = parent2.to_genome()
        
        # Create offspring genomes
        offspring1_genome = []
        offspring2_genome = []
        
        for i in range(len(genome1)):
            if random.random() < 0.5:
                offspring1_genome.append(genome1[i])
                offspring2_genome.append(genome2[i])
            else:
                offspring1_genome.append(genome2[i])
                offspring2_genome.append(genome1[i])
        
        # Convert back to chromosomes
        offspring1 = MicroRTSChromosome.from_genome(offspring1_genome)
        offspring2 = MicroRTSChromosome.from_genome(offspring2_genome)
        
        return offspring1, offspring2


class ArithmeticCrossover(CrossoverOperator):
    """Arithmetic crossover operator for continuous parameters."""
    
    def crossover(self, parent1: MicroRTSChromosome, parent2: MicroRTSChromosome, 
                  crossover_rate: float = 0.7) -> Tuple[MicroRTSChromosome, MicroRTSChromosome]:
        """Perform arithmetic crossover."""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Convert to genomes
        genome1 = parent1.to_genome()
        genome2 = parent2.to_genome()
        
        # Create offspring genomes using arithmetic combination
        alpha = random.uniform(0, 1)  # Mixing parameter
        
        offspring1_genome = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(genome1, genome2)]
        offspring2_genome = [(1 - alpha) * g1 + alpha * g2 for g1, g2 in zip(genome1, genome2)]
        
        # Convert back to chromosomes
        offspring1 = MicroRTSChromosome.from_genome(offspring1_genome)
        offspring2 = MicroRTSChromosome.from_genome(offspring2_genome)
        
        return offspring1, offspring2


class MutationOperator:
    """Base class for mutation operators."""
    
    def mutate(self, chromosome: MicroRTSChromosome, mutation_rate: float = 0.1) -> MicroRTSChromosome:
        """
        Apply mutation to a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated chromosome
        """
        raise NotImplementedError


class GaussianMutation(MutationOperator):
    """Gaussian mutation operator."""
    
    def __init__(self, mutation_strength: float = 0.1):
        """
        Initialize Gaussian mutation.
        
        Args:
            mutation_strength: Standard deviation of Gaussian mutation
        """
        self.mutation_strength = mutation_strength
    
    def mutate(self, chromosome: MicroRTSChromosome, mutation_rate: float = 0.1) -> MicroRTSChromosome:
        """Apply Gaussian mutation."""
        mutated_chromosome = chromosome.copy()
        mutated_chromosome.mutate(mutation_rate, self.mutation_strength)
        return mutated_chromosome


class AdaptiveMutation(MutationOperator):
    """Adaptive mutation operator that adjusts mutation strength based on diversity."""
    
    def __init__(self, initial_mutation_strength: float = 0.1, 
                 min_mutation_strength: float = 0.01,
                 max_mutation_strength: float = 0.5):
        """
        Initialize adaptive mutation.
        
        Args:
            initial_mutation_strength: Starting mutation strength
            min_mutation_strength: Minimum allowed mutation strength
            max_mutation_strength: Maximum allowed mutation strength
        """
        self.initial_mutation_strength = initial_mutation_strength
        self.min_mutation_strength = min_mutation_strength
        self.max_mutation_strength = max_mutation_strength
        self.current_mutation_strength = initial_mutation_strength
    
    def mutate(self, chromosome: MicroRTSChromosome, mutation_rate: float = 0.1) -> MicroRTSChromosome:
        """Apply adaptive mutation."""
        mutated_chromosome = chromosome.copy()
        mutated_chromosome.mutate(mutation_rate, self.current_mutation_strength)
        return mutated_chromosome
    
    def update_mutation_strength(self, population_diversity: float):
        """
        Update mutation strength based on population diversity.
        
        Args:
            population_diversity: Measure of population diversity (0-1)
        """
        # Increase mutation strength when diversity is low
        # Decrease mutation strength when diversity is high
        diversity_factor = 1.0 - population_diversity
        
        self.current_mutation_strength = self.initial_mutation_strength * (1.0 + diversity_factor)
        self.current_mutation_strength = max(self.min_mutation_strength, 
                                           min(self.max_mutation_strength, 
                                               self.current_mutation_strength))


class GeneticOperators:
    """
    Main class that coordinates all genetic operators.
    """
    
    def __init__(self, selection_config: SelectionConfig = None):
        """
        Initialize genetic operators.
        
        Args:
            selection_config: Configuration for selection operators
        """
        self.selection_config = selection_config or SelectionConfig()
        
        # Default operators
        self.selection_operator = TournamentSelection(self.selection_config.tournament_size)
        self.crossover_operator = SinglePointCrossover()
        self.mutation_operator = GaussianMutation()
        
        # Elitism operator
        self.elitism_operator = ElitismSelection(self.selection_config.elite_size)
    
    def set_selection_operator(self, operator: SelectionOperator):
        """Set the selection operator."""
        self.selection_operator = operator
    
    def set_crossover_operator(self, operator: CrossoverOperator):
        """Set the crossover operator."""
        self.crossover_operator = operator
    
    def set_mutation_operator(self, operator: MutationOperator):
        """Set the mutation operator."""
        self.mutation_operator = operator
    
    def evolve_generation(self, population: List[MicroRTSChromosome], 
                         fitness_scores: List[FitnessComponents],
                         crossover_rate: float = 0.7,
                         mutation_rate: float = 0.1) -> List[MicroRTSChromosome]:
        """
        Evolve one generation of the population.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            
        Returns:
            New population for next generation
        """
        new_population = []
        
        # Preserve elite individuals
        elite_individuals = self.elitism_operator.select(population, fitness_scores, 
                                                        self.selection_config.elite_size)
        new_population.extend(elite_individuals)
        
        # Generate offspring to fill remaining population slots
        offspring_needed = len(population) - len(elite_individuals)
        
        while len(new_population) < len(population):
            # Select parents
            parents = self.selection_operator.select(population, fitness_scores, 2)
            
            # Perform crossover
            offspring1, offspring2 = self.crossover_operator.crossover(parents[0], parents[1], 
                                                                      crossover_rate)
            
            # Apply mutation
            offspring1 = self.mutation_operator.mutate(offspring1, mutation_rate)
            offspring2 = self.mutation_operator.mutate(offspring2, mutation_rate)
            
            # Add offspring to new population
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        return new_population[:len(population)]
    
    def calculate_population_diversity(self, population: List[MicroRTSChromosome]) -> float:
        """
        Calculate population diversity based on genetic distance.
        
        Args:
            population: Current population
            
        Returns:
            Diversity measure (0-1, higher is more diverse)
        """
        if len(population) < 2:
            return 0.0
        
        # Convert all chromosomes to genomes
        genomes = [chromosome.to_genome() for chromosome in population]
        
        # Safety check for empty genomes
        if not genomes or len(genomes[0]) == 0:
            return 0.0
        
        # Calculate pairwise distances
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                # Euclidean distance between genomes
                distance = np.sqrt(sum((g1 - g2) ** 2 for g1, g2 in zip(genomes[i], genomes[j])))
                total_distance += distance
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        avg_distance = total_distance / pair_count
        
        # Normalize diversity (assuming max possible distance is sqrt(num_genes))
        max_possible_distance = np.sqrt(len(genomes[0]))
        if max_possible_distance == 0:
            return 0.0
        
        normalized_diversity = min(1.0, avg_distance / max_possible_distance)
        
        return normalized_diversity
