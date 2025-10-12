"""
Core modules for the new MicroRTS Genetic Algorithm implementation.
"""

# New GA implementation
from .ga_chromosome import MicroRTSChromosome, UnitParameters, GlobalParameters, create_random_chromosome, create_population
from .ga_fitness_evaluator import FitnessEvaluator, FitnessComponents, MicroRTSMatchSimulator, MatchResult, evaluate_population_fitness
from .ga_working_evaluator import WorkingGAEvaluator, evaluate_population_fitness_working
from .ga_genetic_operators import (
    SelectionOperator, TournamentSelection, RankBasedSelection, ElitismSelection,
    CrossoverOperator, SinglePointCrossover, UniformCrossover, ArithmeticCrossover,
    MutationOperator, GaussianMutation, AdaptiveMutation, GeneticOperators, SelectionConfig
)
from .ga_algorithm import (
    MicroRTSGeneticAlgorithm, GAConfig, GAResults, GenerationStats,
    create_default_ga, create_fast_ga, create_comprehensive_ga
)
from .ga_config_manager import (
    MicroRTSGameConfig, MicroRTSConfigConverter, ExperimentManager, ConfigValidator
)

__all__ = [
    # Chromosome representation
    "MicroRTSChromosome",
    "UnitParameters", 
    "GlobalParameters",
    "create_random_chromosome",
    "create_population",
    
    # Fitness evaluation
    "FitnessEvaluator",
    "FitnessComponents",
    "MicroRTSMatchSimulator",
    "MatchResult",
    "evaluate_population_fitness",
    
    # Working UTT evaluation
    "WorkingGAEvaluator",
    "evaluate_population_fitness_working",
    
    # Genetic operators
    "SelectionOperator",
    "TournamentSelection",
    "RankBasedSelection", 
    "ElitismSelection",
    "CrossoverOperator",
    "SinglePointCrossover",
    "UniformCrossover",
    "ArithmeticCrossover",
    "MutationOperator",
    "GaussianMutation",
    "AdaptiveMutation",
    "GeneticOperators",
    "SelectionConfig",
    
    # Main algorithm
    "MicroRTSGeneticAlgorithm",
    "GAConfig",
    "GAResults",
    "GenerationStats",
    "create_default_ga",
    "create_fast_ga",
    "create_comprehensive_ga",
    
    # Configuration management
    "MicroRTSGameConfig",
    "MicroRTSConfigConverter",
    "ExperimentManager",
    "ConfigValidator"
]
