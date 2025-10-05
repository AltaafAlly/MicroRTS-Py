"""
Configuration Management System for MicroRTS Genetic Algorithm

This module handles the conversion between evolved chromosome parameters
and actual MicroRTS game configurations, as well as managing experiment
configurations and results.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

from .ga_chromosome import MicroRTSChromosome
from .ga_algorithm import GAConfig, GAResults


@dataclass
class MicroRTSGameConfig:
    """Represents a complete MicroRTS game configuration."""
    
    # Global game parameters
    moveConflictResolutionStrategy: int = 0
    
    # Unit type definitions
    unitTypes: Dict[str, Dict[str, Any]] = None
    
    # Map settings
    mapSize: str = "8x8"
    mapType: str = "basesWorkers8x8"
    
    # Game settings
    maxGameLength: int = 300
    maxCycles: int = 300
    
    def __post_init__(self):
        if self.unitTypes is None:
            self.unitTypes = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MicroRTSGameConfig':
        """Create from dictionary."""
        return cls(**data)


class MicroRTSConfigConverter:
    """
    Converts between GA chromosomes and MicroRTS game configurations.
    """
    
    # Default MicroRTS unit type structure
    DEFAULT_UNIT_STRUCTURE = {
        'Worker': {
            'canAttack': True,
            'canMove': True,
            'canHarvest': True,
            'canReturn': True,
            'canProduce': False,
            'produces': [],
            'cost': 1,
            'hp': 1,
            'minDamage': 1,
            'maxDamage': 1,
            'attackRange': 1,
            'produceTime': 1,
            'moveTime': 1,
            'attackTime': 1,
            'harvestTime': 1,
            'returnTime': 1,
            'harvestAmount': 1,
            'sightRadius': 1
        },
        'Light': {
            'canAttack': True,
            'canMove': True,
            'canHarvest': False,
            'canReturn': False,
            'canProduce': False,
            'produces': [],
            'cost': 2,
            'hp': 4,
            'minDamage': 2,
            'maxDamage': 2,
            'attackRange': 1,
            'produceTime': 1,
            'moveTime': 1,
            'attackTime': 1,
            'harvestTime': 0,
            'returnTime': 0,
            'harvestAmount': 0,
            'sightRadius': 2
        },
        'Heavy': {
            'canAttack': True,
            'canMove': True,
            'canHarvest': False,
            'canReturn': False,
            'canProduce': False,
            'produces': [],
            'cost': 2,
            'hp': 4,
            'minDamage': 4,
            'maxDamage': 4,
            'attackRange': 1,
            'produceTime': 2,
            'moveTime': 2,
            'attackTime': 2,
            'harvestTime': 0,
            'returnTime': 0,
            'harvestAmount': 0,
            'sightRadius': 1
        },
        'Ranged': {
            'canAttack': True,
            'canMove': True,
            'canHarvest': False,
            'canReturn': False,
            'canProduce': False,
            'produces': [],
            'cost': 2,
            'hp': 1,
            'minDamage': 1,
            'maxDamage': 1,
            'attackRange': 2,
            'produceTime': 1,
            'moveTime': 1,
            'attackTime': 1,
            'harvestTime': 0,
            'returnTime': 0,
            'harvestAmount': 0,
            'sightRadius': 2
        },
        'Base': {
            'canAttack': False,
            'canMove': False,
            'canHarvest': False,
            'canReturn': False,
            'canProduce': True,
            'produces': ['Worker'],
            'cost': 10,
            'hp': 10,
            'minDamage': 0,
            'maxDamage': 0,
            'attackRange': 0,
            'produceTime': 0,
            'moveTime': 0,
            'attackTime': 0,
            'harvestTime': 0,
            'returnTime': 0,
            'harvestAmount': 0,
            'sightRadius': 5
        },
        'Barracks': {
            'canAttack': False,
            'canMove': False,
            'canHarvest': False,
            'canReturn': False,
            'canProduce': True,
            'produces': ['Light', 'Heavy', 'Ranged'],
            'cost': 5,
            'hp': 4,
            'minDamage': 0,
            'maxDamage': 0,
            'attackRange': 0,
            'produceTime': 0,
            'moveTime': 0,
            'attackTime': 0,
            'harvestTime': 0,
            'returnTime': 0,
            'harvestAmount': 0,
            'sightRadius': 3
        }
    }
    
    @classmethod
    def chromosome_to_microrts_config(cls, chromosome: MicroRTSChromosome) -> MicroRTSGameConfig:
        """
        Convert a GA chromosome to a MicroRTS game configuration.
        
        Args:
            chromosome: GA chromosome with evolved parameters
            
        Returns:
            MicroRTSGameConfig ready for use in MicroRTS
        """
        config = MicroRTSGameConfig()
        
        # Set global parameters
        config.moveConflictResolutionStrategy = chromosome.global_params.moveConflictResolutionStrategy
        
        # Set unit types with evolved parameters
        for unit_type, evolved_params in chromosome.unit_params.items():
            if unit_type in cls.DEFAULT_UNIT_STRUCTURE:
                # Start with default structure
                unit_config = cls.DEFAULT_UNIT_STRUCTURE[unit_type].copy()
                
                # Override with evolved parameters
                unit_config.update(evolved_params.to_dict())
                
                config.unitTypes[unit_type] = unit_config
        
        return config
    
    @classmethod
    def microrts_config_to_chromosome(cls, config: MicroRTSGameConfig) -> MicroRTSChromosome:
        """
        Convert a MicroRTS game configuration to a GA chromosome.
        
        Args:
            config: MicroRTSGameConfig
            
        Returns:
            MicroRTSChromosome for use in GA
        """
        from .ga_chromosome import UnitParameters, GlobalParameters
        
        # Extract global parameters
        global_params = GlobalParameters(
            moveConflictResolutionStrategy=config.moveConflictResolutionStrategy
        )
        
        # Extract unit parameters
        unit_params = {}
        for unit_type, unit_config in config.unitTypes.items():
            if unit_type in cls.DEFAULT_UNIT_STRUCTURE:
                # Extract only the evolvable parameters
                evolvable_params = {}
                for param_name in UnitParameters.__dataclass_fields__.keys():
                    if param_name in unit_config:
                        evolvable_params[param_name] = unit_config[param_name]
                
                unit_params[unit_type] = UnitParameters(**evolvable_params)
        
        return MicroRTSChromosome(unit_params, global_params)
    
    @classmethod
    def save_microrts_config(cls, config: MicroRTSGameConfig, filename: str) -> None:
        """
        Save MicroRTS configuration to file.
        
        Args:
            config: MicroRTSGameConfig to save
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    @classmethod
    def load_microrts_config(cls, filename: str) -> MicroRTSGameConfig:
        """
        Load MicroRTS configuration from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded MicroRTSGameConfig
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return MicroRTSGameConfig.from_dict(data)


class ExperimentManager:
    """
    Manages GA experiments, including configuration, execution, and results storage.
    """
    
    def __init__(self, base_dir: str = "experiments"):
        """
        Initialize experiment manager.
        
        Args:
            base_dir: Base directory for storing experiments
        """
        self.base_dir = base_dir
        self.ensure_experiment_dir()
    
    def ensure_experiment_dir(self) -> None:
        """Ensure the experiment directory exists."""
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_experiment_dir(self, experiment_name: str) -> str:
        """
        Create a new experiment directory.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Path to the experiment directory
        """
        timestamp = int(time.time())
        experiment_dir = os.path.join(self.base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    def save_experiment_config(self, config: GAConfig, experiment_dir: str) -> str:
        """
        Save experiment configuration.
        
        Args:
            config: GA configuration
            experiment_dir: Experiment directory path
            
        Returns:
            Path to saved config file
        """
        config_file = os.path.join(experiment_dir, "ga_config.json")
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        return config_file
    
    def save_experiment_results(self, results: GAResults, experiment_dir: str) -> str:
        """
        Save experiment results.
        
        Args:
            results: GA results
            experiment_dir: Experiment directory path
            
        Returns:
            Path to saved results file
        """
        results_file = os.path.join(experiment_dir, "ga_results.json")
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        return results_file
    
    def save_best_config(self, chromosome: MicroRTSChromosome, experiment_dir: str) -> str:
        """
        Save the best evolved configuration as a MicroRTS config.
        
        Args:
            chromosome: Best chromosome from evolution
            experiment_dir: Experiment directory path
            
        Returns:
            Path to saved MicroRTS config file
        """
        microrts_config = MicroRTSConfigConverter.chromosome_to_microrts_config(chromosome)
        config_file = os.path.join(experiment_dir, "best_microrts_config.json")
        MicroRTSConfigConverter.save_microrts_config(microrts_config, config_file)
        return config_file
    
    def load_experiment_config(self, experiment_dir: str) -> GAConfig:
        """
        Load experiment configuration.
        
        Args:
            experiment_dir: Experiment directory path
            
        Returns:
            Loaded GA configuration
        """
        config_file = os.path.join(experiment_dir, "ga_config.json")
        with open(config_file, 'r') as f:
            data = json.load(f)
        return GAConfig(**data)
    
    def load_experiment_results(self, experiment_dir: str) -> GAResults:
        """
        Load experiment results.
        
        Args:
            experiment_dir: Experiment directory path
            
        Returns:
            Loaded GA results
        """
        results_file = os.path.join(experiment_dir, "ga_results.json")
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Reconstruct results
        from .ga_chromosome import MicroRTSChromosome
        from .ga_fitness_evaluator import FitnessComponents
        
        best_individual = MicroRTSChromosome.from_json(json.dumps(data['best_individual']))
        best_fitness = FitnessComponents(**data['best_fitness'])
        
        from .ga_algorithm import GenerationStats
        generation_stats = [GenerationStats(**stats) for stats in data['generation_stats']]
        
        return GAResults(
            best_individual=best_individual,
            best_fitness=best_fitness,
            generation_stats=generation_stats,
            total_generations=data['total_generations'],
            total_time=data['total_time'],
            convergence_generation=data.get('convergence_generation')
        )
    
    def list_experiments(self) -> List[str]:
        """
        List all available experiments.
        
        Returns:
            List of experiment directory names
        """
        if not os.path.exists(self.base_dir):
            return []
        
        experiments = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                experiments.append(item)
        
        return sorted(experiments)
    
    def get_experiment_summary(self, experiment_dir: str) -> Dict[str, Any]:
        """
        Get a summary of an experiment.
        
        Args:
            experiment_dir: Experiment directory path
            
        Returns:
            Dictionary with experiment summary
        """
        try:
            config = self.load_experiment_config(experiment_dir)
            results = self.load_experiment_results(experiment_dir)
            
            return {
                'name': os.path.basename(experiment_dir),
                'config': asdict(config),
                'best_fitness': results.best_fitness.overall_fitness,
                'total_generations': results.total_generations,
                'total_time': results.total_time,
                'convergence_generation': results.convergence_generation
            }
        except Exception as e:
            return {
                'name': os.path.basename(experiment_dir),
                'error': str(e)
            }


class ConfigValidator:
    """
    Validates MicroRTS configurations for correctness and balance.
    """
    
    @staticmethod
    def validate_chromosome(chromosome: MicroRTSChromosome) -> List[str]:
        """
        Validate a chromosome for potential issues.
        
        Args:
            chromosome: Chromosome to validate
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check unit parameter bounds
        for unit_type, params in chromosome.unit_params.items():
            bounds = MicroRTSChromosome.PARAMETER_BOUNDS.get(unit_type, {})
            
            for param_name, value in params.to_dict().items():
                if param_name in bounds:
                    min_val, max_val = bounds[param_name]
                    if min_val == max_val == 0:  # Not applicable
                        continue
                    
                    if value < min_val or value > max_val:
                        warnings.append(f"{unit_type}.{param_name} = {value} is outside bounds [{min_val}, {max_val}]")
        
        # Check for extreme values that might break gameplay
        for unit_type, params in chromosome.unit_params.items():
            if params.cost <= 0:
                warnings.append(f"{unit_type}.cost must be positive")
            
            if params.hp <= 0:
                warnings.append(f"{unit_type}.hp must be positive")
            
            if params.minDamage > params.maxDamage:
                warnings.append(f"{unit_type}.minDamage ({params.minDamage}) > maxDamage ({params.maxDamage})")
        
        # Check global parameters
        if not (0 <= chromosome.global_params.moveConflictResolutionStrategy <= 3):
            warnings.append("moveConflictResolutionStrategy must be between 0 and 3")
        
        return warnings
    
    @staticmethod
    def validate_microrts_config(config: MicroRTSGameConfig) -> List[str]:
        """
        Validate a MicroRTS configuration.
        
        Args:
            config: MicroRTSGameConfig to validate
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check required unit types
        required_units = ['Worker', 'Light', 'Heavy', 'Ranged', 'Base', 'Barracks']
        for unit_type in required_units:
            if unit_type not in config.unitTypes:
                warnings.append(f"Missing required unit type: {unit_type}")
        
        # Check unit configurations
        for unit_type, unit_config in config.unitTypes.items():
            # Check required fields
            required_fields = ['cost', 'hp', 'minDamage', 'maxDamage', 'attackRange']
            for field in required_fields:
                if field not in unit_config:
                    warnings.append(f"{unit_type} missing required field: {field}")
            
            # Check logical constraints
            if 'minDamage' in unit_config and 'maxDamage' in unit_config:
                if unit_config['minDamage'] > unit_config['maxDamage']:
                    warnings.append(f"{unit_type}: minDamage > maxDamage")
            
            if 'cost' in unit_config and unit_config['cost'] <= 0:
                warnings.append(f"{unit_type}: cost must be positive")
            
            if 'hp' in unit_config and unit_config['hp'] <= 0:
                warnings.append(f"{unit_type}: hp must be positive")
        
        return warnings
