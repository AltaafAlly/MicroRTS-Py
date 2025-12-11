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
                    # Ensure we don't generate invalid values
                    value = random.randint(min_val, max_val)
                    # Extra safety: ensure positive parameters are at least 1
                    positive_params = ['cost', 'hp', 'produceTime', 'moveTime', 'attackTime', 
                                     'harvestTime', 'returnTime', 'harvestAmount', 'attackRange',
                                     'minDamage', 'maxDamage']
                    if param_name in positive_params and value <= 0:
                        value = max(1, min_val)
                    params[param_name] = value
            
            # Ensure damage values are valid
            if 'minDamage' in params and 'maxDamage' in params:
                if params['maxDamage'] < params['minDamage']:
                    params['maxDamage'] = params['minDamage']
            
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
            
            # Validate and fix parameters before creating UnitParameters
            # Ensure all positive parameters are at least 1
            if 'cost' in params and params['cost'] <= 0:
                params['cost'] = 1
            if 'hp' in params and params['hp'] <= 0:
                params['hp'] = 1
            if 'produceTime' in params and params['produceTime'] <= 0:
                params['produceTime'] = 1
            if 'moveTime' in params and params['moveTime'] <= 0:
                params['moveTime'] = 1
            if 'attackTime' in params and params['attackTime'] <= 0:
                params['attackTime'] = 1
            if 'harvestTime' in params and params['harvestTime'] <= 0:
                params['harvestTime'] = 1
            if 'returnTime' in params and params['returnTime'] <= 0:
                params['returnTime'] = 1
            if 'harvestAmount' in params and params['harvestAmount'] <= 0:
                params['harvestAmount'] = 1
            if 'attackRange' in params and params['attackRange'] <= 0:
                params['attackRange'] = 1
            
            # Ensure damage values are valid
            if 'minDamage' in params and params['minDamage'] < 1:
                params['minDamage'] = 1
            if 'maxDamage' in params and params['maxDamage'] < 1:
                params['maxDamage'] = 1
            if 'minDamage' in params and 'maxDamage' in params:
                if params['maxDamage'] < params['minDamage']:
                    params['maxDamage'] = params['minDamage']
            
            # sightRadius can be 0, but ensure it's non-negative
            if 'sightRadius' in params and params['sightRadius'] < 0:
                params['sightRadius'] = 0
            
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
                # Get parameters and validate/fix them for Java compatibility
                param_dict = params.to_dict()
                
                # Ensure all time parameters are positive (Java requirement)
                time_params = ['produceTime', 'moveTime', 'attackTime', 'harvestTime', 'returnTime']
                for time_param in time_params:
                    if time_param in param_dict and param_dict[time_param] <= 0:
                        param_dict[time_param] = 1
                
                # Ensure cost, hp, and other positive parameters are > 0
                if param_dict.get('cost', 1) <= 0:
                    param_dict['cost'] = 1
                if param_dict.get('hp', 1) <= 0:
                    param_dict['hp'] = 1
                if param_dict.get('harvestAmount', 1) <= 0:
                    param_dict['harvestAmount'] = 1
                
                # Ensure damage values are valid
                # minDamage and maxDamage must be >= 1, and maxDamage >= minDamage
                if param_dict.get('minDamage', 1) < 1:
                    param_dict['minDamage'] = 1
                if param_dict.get('maxDamage', 1) < 1:
                    param_dict['maxDamage'] = 1
                if param_dict.get('maxDamage', 1) < param_dict.get('minDamage', 1):
                    param_dict['maxDamage'] = param_dict['minDamage']
                
                # Ensure attackRange is positive
                if param_dict.get('attackRange', 1) <= 0:
                    param_dict['attackRange'] = 1
                
                # sightRadius can be 0 for some units (like Resource), but ensure it's non-negative
                if param_dict.get('sightRadius', 0) < 0:
                    param_dict['sightRadius'] = 0
                
                unit_config = {
                    'ID': unit_mapping[unit_type]['ID'],
                    'name': unit_type,
                    **param_dict,
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
