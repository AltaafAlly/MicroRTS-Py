"""
GA and MAP-Elites for MicroRTS UTT Evolution

This package provides genetic algorithm implementations for evolving 
Unit Type Table (UTT) configurations in MicroRTS.
"""

# Core modules
from .core.ga_chromosome import MicroRTSChromosome, UnitParameters, GlobalParameters, create_random_chromosome, create_population
from .core.ga_fitness_evaluator import FitnessEvaluator, FitnessComponents, MicroRTSMatchSimulator, MatchResult, evaluate_population_fitness
from .core.ga_real_microrts_evaluator import RealMicroRTSFitnessEvaluator, RealMicroRTSMatchRunner, evaluate_population_fitness_real
from .core.ga_genetic_operators import (
    GeneticOperators, 
    SelectionConfig, 
    TournamentSelection, 
    RankBasedSelection,
    UniformCrossover, 
    SinglePointCrossover,
    GaussianMutation, 
    UniformMutation
)
from .core.ga_algorithm import (
    MicroRTSGeneticAlgorithm,
    GAConfig, 
    GAResults, 
    GenerationStats,
    create_default_config,
    create_fast_config,
    create_comprehensive_config
)
from .core.ga_config_manager import GAConfigManager

__version__ = "2.0.0"
__author__ = "MicroRTS Research Team"

__all__ = [
    # Core GA components
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
    "RealMicroRTSFitnessEvaluator",
    "RealMicroRTSMatchRunner", 
    "evaluate_population_fitness_real",
    
    # Genetic operators
    "GeneticOperators",
    "SelectionConfig",
    "TournamentSelection",
    "RankBasedSelection", 
    "UniformCrossover",
    "SinglePointCrossover",
    "GaussianMutation",
    "UniformMutation",
    
    # Main algorithm
    "MicroRTSGeneticAlgorithm",
    "GAConfig",
    "GAResults", 
    "GenerationStats",
    "create_default_config",
    "create_fast_config", 
    "create_comprehensive_config",
    
    # Configuration management
    "GAConfigManager"
]