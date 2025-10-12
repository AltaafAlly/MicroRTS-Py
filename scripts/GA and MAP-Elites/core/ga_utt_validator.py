#!/usr/bin/env python3
"""
UTT Validator: Ensures generated UTTs are Java-compatible
"""

from typing import Dict, Any, List, Tuple
import random

class UTTValidator:
    """Validates and fixes UTT configurations for Java compatibility."""
    
    # Safe parameter bounds that work with Java environment
    SAFE_BOUNDS = {
        'Resource': {
            'cost': (1, 1),
            'produceTime': (10, 10),
            'hp': (1, 1),
            'minDamage': (1, 1),
            'maxDamage': (1, 1),
            'attackRange': (1, 1),
            'attackTime': (10, 10),
            'moveTime': (10, 10),
            'sightRadius': (0, 0),
            'harvestTime': (10, 10),
            'returnTime': (10, 10),
            'harvestAmount': (1, 1)
        },
        'Worker': {
            'cost': (1, 10),
            'produceTime': (1, 5),
            'hp': (1, 20),
            'minDamage': (1, 5),
            'maxDamage': (1, 5),
            'attackRange': (1, 3),
            'attackTime': (1, 5),
            'moveTime': (1, 3),
            'sightRadius': (1, 5),
            'harvestTime': (1, 5),
            'returnTime': (1, 3),
            'harvestAmount': (1, 10)
        },
        'Light': {
            'cost': (5, 20),
            'produceTime': (2, 8),
            'hp': (10, 50),
            'minDamage': (5, 15),
            'maxDamage': (5, 15),
            'attackRange': (1, 4),
            'attackTime': (2, 8),
            'moveTime': (1, 4),
            'sightRadius': (2, 6),
            'harvestTime': (0, 0),
            'returnTime': (0, 0),
            'harvestAmount': (0, 0)
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
            'harvestTime': (0, 0),
            'returnTime': (0, 0),
            'harvestAmount': (0, 0)
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
            'harvestTime': (0, 0),
            'returnTime': (0, 0),
            'harvestAmount': (0, 0)
        },
        'Base': {
            'cost': (50, 100),
            'produceTime': (10, 30),
            'hp': (100, 300),
            'minDamage': (0, 0),
            'maxDamage': (0, 0),
            'attackRange': (0, 0),
            'attackTime': (0, 0),
            'moveTime': (0, 0),
            'sightRadius': (3, 8),
            'harvestTime': (0, 0),
            'returnTime': (0, 0),
            'harvestAmount': (0, 0)
        },
        'Barracks': {
            'cost': (20, 50),
            'produceTime': (5, 15),
            'hp': (50, 150),
            'minDamage': (0, 0),
            'maxDamage': (0, 0),
            'attackRange': (0, 0),
            'attackTime': (0, 0),
            'moveTime': (0, 0),
            'sightRadius': (2, 6),
            'harvestTime': (0, 0),
            'returnTime': (0, 0),
            'harvestAmount': (0, 0)
        }
    }
    
    @classmethod
    def validate_and_fix_utt(cls, utt_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix a UTT configuration to ensure Java compatibility.
        
        Args:
            utt_config: UTT configuration dictionary
            
        Returns:
            Fixed UTT configuration
        """
        fixed_config = utt_config.copy()
        
        # Fix move conflict resolution strategy
        if 'moveConflictResolutionStrategy' in fixed_config:
            strategy = fixed_config['moveConflictResolutionStrategy']
            if strategy < 1 or strategy > 3:
                fixed_config['moveConflictResolutionStrategy'] = random.randint(1, 3)
        
        # Fix unit types
        if 'unitTypes' in fixed_config:
            fixed_units = []
            for unit in fixed_config['unitTypes']:
                fixed_unit = cls._fix_unit(unit)
                fixed_units.append(fixed_unit)
            fixed_config['unitTypes'] = fixed_units
        
        return fixed_config
    
    @classmethod
    def _fix_unit(cls, unit: Dict[str, Any]) -> Dict[str, Any]:
        """Fix a single unit configuration."""
        unit_name = unit.get('name', '')
        fixed_unit = unit.copy()
        
        if unit_name in cls.SAFE_BOUNDS:
            bounds = cls.SAFE_BOUNDS[unit_name]
            
            for param_name, (min_val, max_val) in bounds.items():
                if param_name in fixed_unit:
                    current_value = fixed_unit[param_name]
                    
                    if min_val == max_val == 0:
                        # Not applicable parameter
                        fixed_unit[param_name] = 0
                    elif min_val == max_val:
                        # Fixed value
                        fixed_unit[param_name] = min_val
                    else:
                        # Ensure value is within bounds
                        if current_value < min_val or current_value > max_val:
                            fixed_unit[param_name] = random.randint(min_val, max_val)
        
        return fixed_unit
    
    @classmethod
    def validate_utt_safety(cls, utt_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if a UTT configuration is safe for Java environment.
        
        Args:
            utt_config: UTT configuration dictionary
            
        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []
        
        # Check move conflict resolution strategy
        if 'moveConflictResolutionStrategy' in utt_config:
            strategy = utt_config['moveConflictResolutionStrategy']
            if strategy < 1 or strategy > 3:
                issues.append(f"Invalid moveConflictResolutionStrategy: {strategy} (must be 1-3)")
        
        # Check unit types
        if 'unitTypes' in utt_config:
            for unit in utt_config['unitTypes']:
                unit_issues = cls._check_unit_safety(unit)
                issues.extend(unit_issues)
        
        return len(issues) == 0, issues
    
    @classmethod
    def _check_unit_safety(cls, unit: Dict[str, Any]) -> List[str]:
        """Check if a single unit is safe."""
        issues = []
        unit_name = unit.get('name', '')
        
        if unit_name in cls.SAFE_BOUNDS:
            bounds = cls.SAFE_BOUNDS[unit_name]
            
            for param_name, (min_val, max_val) in bounds.items():
                if param_name in unit:
                    value = unit[param_name]
                    
                    if min_val == max_val == 0:
                        # Not applicable parameter - should be 0
                        if value != 0:
                            issues.append(f"{unit_name}.{param_name} should be 0 (not applicable)")
                    elif min_val == max_val:
                        # Fixed value
                        if value != min_val:
                            issues.append(f"{unit_name}.{param_name} should be {min_val} (fixed value)")
                    else:
                        # Variable parameter - check bounds
                        if value < min_val or value > max_val:
                            issues.append(f"{unit_name}.{param_name} = {value} is out of bounds [{min_val}, {max_val}]")
        
        return issues

def create_safe_utt_config() -> Dict[str, Any]:
    """Create a safe UTT configuration for testing."""
    from core.ga_chromosome import create_random_chromosome
    
    chromosome = create_random_chromosome()
    utt_config = chromosome.to_microrts_config()
    
    return UTTValidator.validate_and_fix_utt(utt_config)

if __name__ == "__main__":
    # Test the validator
    print("Testing UTT Validator...")
    
    # Create a random UTT
    from core.ga_chromosome import create_random_chromosome
    chromosome = create_random_chromosome()
    utt_config = chromosome.to_microrts_config()
    
    print("Original UTT:")
    print(f"  moveConflictResolutionStrategy: {utt_config.get('moveConflictResolutionStrategy')}")
    
    # Validate and fix
    fixed_config = UTTValidator.validate_and_fix_utt(utt_config)
    
    print("Fixed UTT:")
    print(f"  moveConflictResolutionStrategy: {fixed_config.get('moveConflictResolutionStrategy')}")
    
    # Check safety
    is_safe, issues = UTTValidator.validate_utt_safety(fixed_config)
    print(f"  Is safe: {is_safe}")
    if issues:
        print(f"  Issues: {issues}")
    else:
        print("  ✅ No issues found!")
