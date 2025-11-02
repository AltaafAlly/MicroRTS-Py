"""
Genetic Algorithm Chromosome Implementation for MicroRTS Parameter Evolution

This module defines the chromosome structure that represents a complete MicroRTS
game configuration, including unit parameters and global game parameters.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import json


@dataclass
class UnitParameters:
    """Represents the evolvable parameters for a single unit type."""
    
    # Resource and production
    cost: int = 1
    produceTime: int = 1
    
    # Combat stats
    hp: int = 1
    minDamage: int = 1
    maxDamage: int = 1
    attackRange: int = 1
    attackTime: int = 1
    
    # Movement and vision
    moveTime: int = 1
    sightRadius: int = 1
    
    # Resource gathering (for workers)
    harvestTime: int = 1
    returnTime: int = 1
    harvestAmount: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'cost': self.cost,
            'produceTime': self.produceTime,
            'hp': self.hp,
            'minDamage': self.minDamage,
            'maxDamage': self.maxDamage,
            'attackRange': self.attackRange,
            'attackTime': self.attackTime,
            'moveTime': self.moveTime,
            'sightRadius': self.sightRadius,
            'harvestTime': self.harvestTime,
            'returnTime': self.returnTime,
            'harvestAmount': self.harvestAmount
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnitParameters':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GlobalParameters:
    """Represents global game parameters that can be evolved."""
    
    moveConflictResolutionStrategy: int = 1  # 1-3 for different strategies
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'moveConflictResolutionStrategy': self.moveConflictResolutionStrategy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlobalParameters':
        """Create from dictionary."""
        return cls(**data)


class MicroRTSChromosome:
    """
    Represents a complete MicroRTS game configuration as a chromosome.
    
    This chromosome contains:
    1. Unit parameters for each unit type (Worker, Light, Heavy, Ranged, Base, Barracks)
    2. Global game parameters
    """
    
    # Define the unit types that can be evolved
    UNIT_TYPES = ['Resource', 'Worker', 'Light', 'Heavy', 'Ranged', 'Base', 'Barracks']
    
    # Parameter bounds for each unit type
    PARAMETER_BOUNDS = {
        'Resource': {
            'cost': (1, 1),
            'produceTime': (10, 10),
            'hp': (1, 1),
            'minDamage': (1, 1),
            'maxDamage': (1, 1),
            'attackRange': (1, 1),
            'attackTime': (10, 10),
            'moveTime': (10, 10),
            'harvestTime': (10, 10),
            'returnTime': (10, 10),
            'harvestAmount': (1, 1),
            'sightRadius': (0, 0)
        },
        'Worker': {
            'cost': (2, 6),      # More reasonable cost range
            'produceTime': (2, 4), # Faster production
            'hp': (5, 15),       # More survivable
            'minDamage': (2, 4), # Effective damage
            'maxDamage': (2, 4), # Effective damage
            'attackRange': (1, 2), # Close combat
            'attackTime': (2, 4), # Reasonable attack speed
            'moveTime': (1, 2),  # Fast movement
            'sightRadius': (2, 4), # Good vision
            'harvestTime': (2, 4), # Efficient harvesting
            'returnTime': (1, 2), # Quick return
            'harvestAmount': (3, 8) # Good harvest amount
        },
        'Light': {
            'cost': (8, 15),     # More reasonable cost
            'produceTime': (3, 6), # Faster production
            'hp': (15, 30),     # More survivable
            'minDamage': (6, 12), # Effective damage
            'maxDamage': (6, 12), # Effective damage
            'attackRange': (1, 3), # Good range
            'attackTime': (3, 6), # Reasonable attack speed
            'moveTime': (1, 3),  # Fast movement
            'sightRadius': (3, 5), # Good vision
            'harvestTime': (1, 1), # Fixed: Java requires positive bounds
            'returnTime': (1, 1), # Fixed: Java requires positive bounds
            'harvestAmount': (1, 1) # Fixed: Java requires positive bounds
        },
        'Heavy': {
            'cost': (15, 40),
            'produceTime': (5, 15),
            'hp': (30, 100),
            'minDamage': (10, 30),
            'maxDamage': (10, 30),
            'attackRange': (1, 3),
            'attackTime': (3, 10),
            'moveTime': (2, 6),
            'sightRadius': (2, 5),
            'harvestTime': (1, 1),  # Fixed: Java requires positive bounds
            'returnTime': (1, 1),   # Fixed: Java requires positive bounds
            'harvestAmount': (1, 1) # Fixed: Java requires positive bounds
        },
        'Ranged': {
            'cost': (10, 30),
            'produceTime': (3, 10),
            'hp': (15, 40),
            'minDamage': (8, 20),
            'maxDamage': (8, 20),
            'attackRange': (2, 6),
            'attackTime': (2, 6),
            'moveTime': (1, 3),
            'sightRadius': (3, 8),
            'harvestTime': (1, 1),  # Fixed: Java requires positive bounds
            'returnTime': (1, 1),   # Fixed: Java requires positive bounds
            'harvestAmount': (1, 1) # Fixed: Java requires positive bounds
        },
        'Base': {
            'cost': (50, 100),
            'produceTime': (10, 30),
            'hp': (100, 300),
            'minDamage': (1, 1),    # Fixed: Java requires positive bounds
            'maxDamage': (1, 1),    # Fixed: Java requires positive bounds
            'attackRange': (1, 1),  # Fixed: Java requires positive bounds
            'attackTime': (1, 1),   # Fixed: Java requires positive bounds
            'moveTime': (1, 1),     # Fixed: Java requires positive bounds
            'sightRadius': (3, 8),
            'harvestTime': (1, 1),  # Fixed: Java requires positive bounds
            'returnTime': (1, 1),   # Fixed: Java requires positive bounds
            'harvestAmount': (1, 1) # Fixed: Java requires positive bounds
        },
        'Barracks': {
            'cost': (20, 50),
            'produceTime': (5, 15),
            'hp': (50, 150),
            'minDamage': (1, 1),    # Fixed: Java requires positive bounds
            'maxDamage': (1, 1),    # Fixed: Java requires positive bounds
            'attackRange': (1, 1),  # Fixed: Java requires positive bounds
            'attackTime': (1, 1),   # Fixed: Java requires positive bounds
            'moveTime': (1, 1),     # Fixed: Java requires positive bounds
            'sightRadius': (2, 6),
            'harvestTime': (1, 1),  # Fixed: Java requires positive bounds
            'returnTime': (1, 1),   # Fixed: Java requires positive bounds
            'harvestAmount': (1, 1) # Fixed: Java requires positive bounds
        }
    }
    
    def __init__(self, unit_params: Dict[str, UnitParameters] = None, 
                 global_params: GlobalParameters = None):
        """
        Initialize chromosome with unit and global parameters.
        
        Args:
            unit_params: Dictionary mapping unit type names to UnitParameters
            global_params: Global game parameters
        """
        self.unit_params = unit_params or {}
        self.global_params = global_params or GlobalParameters()
        
        # Initialize unit parameters if not provided
        if not self.unit_params:
            self._initialize_random_unit_params()
    
    def _initialize_random_unit_params(self):
        """Initialize unit parameters with random values within bounds."""
        for unit_type in self.UNIT_TYPES:
            bounds = self.PARAMETER_BOUNDS[unit_type]
            params = {}
            
            for param_name, (min_val, max_val) in bounds.items():
                if min_val == max_val == 0:  # Not applicable parameter
                    params[param_name] = 0
                elif min_val == max_val:  # Fixed value
                    params[param_name] = min_val
                else:
                    params[param_name] = random.randint(min_val, max_val)
            
            self.unit_params[unit_type] = UnitParameters(**params)
    
    def to_genome(self) -> List[float]:
        """
        Convert chromosome to a genome (list of normalized values).
        
        Returns:
            List of normalized parameter values for genetic operations
        """
        genome = []
        
        # Add unit parameters
        for unit_type in self.UNIT_TYPES:
            if unit_type in self.unit_params:
                unit = self.unit_params[unit_type]
                bounds = self.PARAMETER_BOUNDS[unit_type]
                
                for param_name in bounds.keys():
                    value = getattr(unit, param_name)
                    min_val, max_val = bounds[param_name]
                    
                    if min_val == max_val == 0:  # Not applicable parameter
                        genome.append(0.0)  # Use 0.0 for not applicable
                    elif min_val == max_val:  # Fixed value
                        genome.append(0.5)  # Use 0.5 for fixed values
                    else:
                        # Normalize to [0, 1]
                        normalized = (value - min_val) / (max_val - min_val)
                        genome.append(normalized)
        
        # Add global parameters (normalize 1-3 to 0-1)
        genome.append((self.global_params.moveConflictResolutionStrategy - 1) / 2.0)
        
        return genome
    
    @classmethod
    def from_genome(cls, genome: List[float]) -> 'MicroRTSChromosome':
        """
        Create chromosome from genome.
        
        Args:
            genome: List of normalized parameter values
            
        Returns:
            MicroRTSChromosome instance
        """
        unit_params = {}
        genome_idx = 0
        
        # Reconstruct unit parameters
        for unit_type in cls.UNIT_TYPES:
            bounds = cls.PARAMETER_BOUNDS[unit_type]
            params = {}
            
            for param_name, (min_val, max_val) in bounds.items():
                if min_val == max_val == 0:  # Not applicable parameter
                    params[param_name] = 0
                elif min_val == max_val:  # Fixed value
                    params[param_name] = min_val
                else:
                    # Denormalize from [0, 1]
                    normalized = genome[genome_idx]
                    value = int(min_val + normalized * (max_val - min_val))
                    params[param_name] = max(min_val, min(value, max_val))
                genome_idx += 1
            
            unit_params[unit_type] = UnitParameters(**params)
        
        # Reconstruct global parameters (convert 0-1 back to 1-3)
        move_strategy = max(1, min(3, int(genome[genome_idx] * 2) + 1))  # Ensure 1-3 range
        global_params = GlobalParameters(
            moveConflictResolutionStrategy=move_strategy
        )
        
        return cls(unit_params, global_params)
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """
        Apply mutation to the chromosome.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Strength of mutation (0-1)
        """
        genome = self.to_genome()
        
        for i in range(len(genome)):
            if random.random() < mutation_rate:
                # Apply Gaussian mutation
                mutation = np.random.normal(0, mutation_strength)
                genome[i] = max(0.0, min(1.0, genome[i] + mutation))
        
        # Reconstruct chromosome from mutated genome
        mutated_chromosome = self.from_genome(genome)
        self.unit_params = mutated_chromosome.unit_params
        self.global_params = mutated_chromosome.global_params
    
    def crossover(self, other: 'MicroRTSChromosome', crossover_rate: float = 0.7) -> Tuple['MicroRTSChromosome', 'MicroRTSChromosome']:
        """
        Perform crossover with another chromosome.
        
        Args:
            other: Another MicroRTSChromosome
            crossover_rate: Probability of crossover
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if random.random() > crossover_rate:
            return self.copy(), other.copy()
        
        genome1 = self.to_genome()
        genome2 = other.to_genome()
        
        # Single-point crossover
        crossover_point = random.randint(1, len(genome1) - 1)
        
        offspring1_genome = genome1[:crossover_point] + genome2[crossover_point:]
        offspring2_genome = genome2[:crossover_point] + genome1[crossover_point:]
        
        offspring1 = self.from_genome(offspring1_genome)
        offspring2 = self.from_genome(offspring2_genome)
        
        return offspring1, offspring2
    
    def copy(self) -> 'MicroRTSChromosome':
        """Create a deep copy of the chromosome."""
        unit_params_copy = {}
        for unit_type, params in self.unit_params.items():
            unit_params_copy[unit_type] = UnitParameters(**params.to_dict())
        
        global_params_copy = GlobalParameters(**self.global_params.to_dict())
        
        return MicroRTSChromosome(unit_params_copy, global_params_copy)
    
    def to_microrts_config(self) -> Dict[str, Any]:
        """
        Convert chromosome to MicroRTS configuration format.
        
        Returns:
            Dictionary in MicroRTS configuration format
        """
        config = {
            'moveConflictResolutionStrategy': self.global_params.moveConflictResolutionStrategy,
            'unitTypes': []
        }
        
        # Unit type mapping with IDs and capabilities
        unit_mapping = {
            'Resource': {'ID': 0, 'isResource': True, 'isStockpile': False, 'canHarvest': False, 'canMove': False, 'canAttack': False, 'produces': [], 'producedBy': []},
            'Base': {'ID': 1, 'isResource': False, 'isStockpile': True, 'canHarvest': False, 'canMove': False, 'canAttack': False, 'produces': ['Worker'], 'producedBy': ['Worker']},
            'Barracks': {'ID': 2, 'isResource': False, 'isStockpile': False, 'canHarvest': False, 'canMove': False, 'canAttack': False, 'produces': ['Light', 'Heavy', 'Ranged'], 'producedBy': ['Worker']},
            'Worker': {'ID': 3, 'isResource': False, 'isStockpile': False, 'canHarvest': True, 'canMove': True, 'canAttack': True, 'produces': [], 'producedBy': ['Base']},
            'Light': {'ID': 4, 'isResource': False, 'isStockpile': False, 'canHarvest': False, 'canMove': True, 'canAttack': True, 'produces': [], 'producedBy': ['Barracks']},
            'Heavy': {'ID': 5, 'isResource': False, 'isStockpile': False, 'canHarvest': False, 'canMove': True, 'canAttack': True, 'produces': [], 'producedBy': ['Barracks']},
            'Ranged': {'ID': 6, 'isResource': False, 'isStockpile': False, 'canHarvest': False, 'canMove': True, 'canAttack': True, 'produces': [], 'producedBy': ['Barracks']}
        }
        
        for unit_type, params in self.unit_params.items():
            if unit_type in unit_mapping:
                unit_config = {
                    'ID': unit_mapping[unit_type]['ID'],
                    'name': unit_type,
                    **params.to_dict(),
                    **unit_mapping[unit_type]
                }
                config['unitTypes'].append(unit_config)
        
        return config
    
    def to_json(self) -> str:
        """Convert chromosome to JSON string."""
        return json.dumps(self.to_microrts_config(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MicroRTSChromosome':
        """Create chromosome from JSON string."""
        config = json.loads(json_str)
        
        unit_params = {}
        for unit_type, params_dict in config['unitTypes'].items():
            unit_params[unit_type] = UnitParameters.from_dict(params_dict)
        
        global_params = GlobalParameters.from_dict(config['globalParameters'])
        
        return cls(unit_params, global_params)
    
    def __str__(self) -> str:
        """String representation of the chromosome."""
        return f"MicroRTSChromosome(units={len(self.unit_params)}, global={self.global_params.moveConflictResolutionStrategy})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


def create_random_chromosome() -> MicroRTSChromosome:
    """Create a random chromosome for initialization."""
    return MicroRTSChromosome()


def create_population(population_size: int) -> List[MicroRTSChromosome]:
    """Create a random population of chromosomes."""
    return [create_random_chromosome() for _ in range(population_size)]
