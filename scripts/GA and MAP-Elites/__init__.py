"""
GA and MAP-Elites for MicroRTS UTT Evolution

This package provides genetic algorithm and MAP-Elites implementations
for evolving Unit Type Table (UTT) configurations in MicroRTS.
"""

# Core modules
from .core.utt_genetic_algorithm import (
    UTTGeneEncoder, 
    Individual, 
    GeneticAlgorithm,
    FitnessEvaluator
)

from .core.utt_map_elites import (
    MAPElites,
    BehaviorDescriptor,
    Archive
)

from .core.utt_utils import (
    UTTAnalyzer,
    UTTVisualizer,
    UTTValidator
)

from .core.improved_fitness_evaluator import (
    AIGameSimulationFitness,
    MultiObjectiveFitness
)

__version__ = "1.0.0"
__author__ = "MicroRTS Research Team"

__all__ = [
    # Core GA components
    "UTTGeneEncoder",
    "Individual", 
    "GeneticAlgorithm",
    "FitnessEvaluator",
    
    # MAP-Elites components
    "MAPElites",
    "BehaviorDescriptor", 
    "Archive",
    
    # Utilities
    "UTTAnalyzer",
    "UTTVisualizer",
    "UTTValidator",
    
    # Fitness evaluators
    "AIGameSimulationFitness",
    "MultiObjectiveFitness"
]
