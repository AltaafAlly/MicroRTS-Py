"""
Fitness Evaluation System for MicroRTS Genetic Algorithm

This module implements the three-component fitness function:
1. Balance: Measures fairness between AI agents
2. Duration: Evaluates match length appropriateness  
3. Strategy Diversity: Measures tactical variation across matches
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import statistics

from .ga_chromosome import MicroRTSChromosome


@dataclass
class MatchResult:
    """Represents the result of a single AI vs AI match."""
    
    ai1_name: str
    ai2_name: str
    winner: int  # 0 for AI1, 1 for AI2, -1 for draw
    duration: int  # Number of game steps
    ai1_actions: List[str]  # Action sequence for AI1
    ai2_actions: List[str]  # Action sequence for AI2
    ai1_units_created: Dict[str, int]  # Unit type counts for AI1
    ai2_units_created: Dict[str, int]  # Unit type counts for AI2
    ai1_resources_gathered: int
    ai2_resources_gathered: int
    
    def is_balanced(self) -> bool:
        """Check if the match was balanced (close win/loss)."""
        return self.winner == -1  # Draw is considered balanced
    
    def get_balance_score(self) -> float:
        """Get balance score (0-1, higher is more balanced)."""
        if self.winner == -1:  # Draw
            return 1.0
        else:
            # Penalize decisive wins
            return 0.0


@dataclass
class FitnessComponents:
    """Components of the fitness function."""
    
    balance: float  # 0-1, higher is better
    duration: float  # 0-1, higher is better  
    strategy_diversity: float  # 0-1, higher is better
    overall_fitness: float  # Weighted combination
    
    def __str__(self) -> str:
        return f"Fitness(balance={self.balance:.3f}, duration={self.duration:.3f}, diversity={self.strategy_diversity:.3f}, overall={self.overall_fitness:.3f})"


class MicroRTSMatchSimulator:
    """
    Simulates AI vs AI matches in MicroRTS with evolved parameters.
    
    This is a placeholder implementation that simulates realistic match outcomes
    based on the chromosome parameters. In a real implementation, this would
    interface with the actual MicroRTS engine.
    """
    
    # Available AI agents for evaluation
    AI_AGENTS = [
        'randomBiasedAI',
        'workerRushAI', 
        'lightRushAI',
        'heavyRushAI',
        'passiveAI'
    ]
    
    def __init__(self, max_steps: int = 300, games_per_evaluation: int = 3):
        """
        Initialize the match simulator.
        
        Args:
            max_steps: Maximum number of steps per game
            games_per_evaluation: Number of games to run per evaluation
        """
        self.max_steps = max_steps
        self.games_per_evaluation = games_per_evaluation
    
    def simulate_match(self, chromosome: MicroRTSChromosome, 
                      ai1: str, ai2: str) -> MatchResult:
        """
        Simulate a single match between two AIs with the given chromosome.
        
        Args:
            chromosome: The evolved MicroRTS configuration
            ai1: First AI agent name
            ai2: Second AI agent name
            
        Returns:
            MatchResult containing match outcome and statistics
        """
        # Simulate realistic match based on chromosome parameters
        # This is a simplified simulation - in reality, you'd run actual MicroRTS
        
        # Extract key parameters that affect gameplay
        worker_cost = chromosome.unit_params['Worker'].cost
        light_cost = chromosome.unit_params['Light'].cost
        heavy_cost = chromosome.unit_params['Heavy'].cost
        
        worker_hp = chromosome.unit_params['Worker'].hp
        light_hp = chromosome.unit_params['Light'].hp
        heavy_hp = chromosome.unit_params['Heavy'].hp
        
        # Simulate AI behavior based on their strategies
        ai1_actions, ai1_units, ai1_resources = self._simulate_ai_behavior(
            ai1, worker_cost, light_cost, heavy_cost, worker_hp, light_hp, heavy_hp
        )
        ai2_actions, ai2_units, ai2_resources = self._simulate_ai_behavior(
            ai2, worker_cost, light_cost, heavy_cost, worker_hp, light_hp, heavy_hp
        )
        
        # Determine winner based on resources and units
        ai1_power = ai1_resources + sum(ai1_units.values()) * 10
        ai2_power = ai2_resources + sum(ai2_units.values()) * 10
        
        if abs(ai1_power - ai2_power) < 50:  # Close match
            winner = -1  # Draw
        elif ai1_power > ai2_power:
            winner = 0  # AI1 wins
        else:
            winner = 1  # AI2 wins
        
        # Simulate duration based on unit costs and HP
        avg_cost = (worker_cost + light_cost + heavy_cost) / 3
        avg_hp = (worker_hp + light_hp + heavy_hp) / 3
        duration = int(100 + (avg_cost * 2) + (avg_hp * 0.5) + random.randint(-50, 50))
        duration = max(50, min(duration, self.max_steps))
        
        return MatchResult(
            ai1_name=ai1,
            ai2_name=ai2,
            winner=winner,
            duration=duration,
            ai1_actions=ai1_actions,
            ai2_actions=ai2_actions,
            ai1_units_created=ai1_units,
            ai2_units_created=ai2_units,
            ai1_resources_gathered=ai1_resources,
            ai2_resources_gathered=ai2_resources
        )
    
    def _simulate_ai_behavior(self, ai_name: str, worker_cost: int, light_cost: int, 
                             heavy_cost: int, worker_hp: int, light_hp: int, 
                             heavy_hp: int) -> Tuple[List[str], Dict[str, int], int]:
        """Simulate AI behavior based on strategy type."""
        
        actions = []
        units = {'Worker': 0, 'Light': 0, 'Heavy': 0, 'Ranged': 0}
        resources = 0
        
        # Simulate different AI strategies
        if ai_name == 'randomBiasedAI':
            # Random but biased toward useful actions
            for _ in range(random.randint(20, 50)):
                action = random.choice(['move', 'attack', 'harvest', 'produce'])
                actions.append(action)
                if action == 'harvest':
                    resources += random.randint(5, 15)
                elif action == 'produce':
                    unit_type = random.choice(['Worker', 'Light', 'Heavy'])
                    units[unit_type] += 1
        
        elif ai_name == 'workerRushAI':
            # Early aggressive worker rush
            for _ in range(random.randint(15, 30)):
                if random.random() < 0.7:  # Focus on workers
                    actions.append('produce_worker')
                    units['Worker'] += 1
                else:
                    actions.append('harvest')
                    resources += random.randint(3, 8)
        
        elif ai_name == 'lightRushAI':
            # Fast light unit attacks
            for _ in range(random.randint(25, 40)):
                if random.random() < 0.6:  # Focus on light units
                    actions.append('produce_light')
                    units['Light'] += 1
                else:
                    actions.append('harvest')
                    resources += random.randint(4, 10)
        
        elif ai_name == 'heavyRushAI':
            # Slower but strong heavy units
            for _ in range(random.randint(30, 60)):
                if random.random() < 0.5:  # Focus on heavy units
                    actions.append('produce_heavy')
                    units['Heavy'] += 1
                else:
                    actions.append('harvest')
                    resources += random.randint(5, 12)
        
        elif ai_name == 'passiveAI':
            # Defensive strategy
            for _ in range(random.randint(40, 80)):
                if random.random() < 0.3:  # Less aggressive
                    actions.append('produce_worker')
                    units['Worker'] += 1
                else:
                    actions.append('harvest')
                    resources += random.randint(6, 15)
        
        return actions, units, resources


class FitnessEvaluator:
    """
    Evaluates fitness of MicroRTS chromosomes using the three-component function.
    
    Fitness = α * Balance + β * Duration + γ * StrategyDiversity
    """
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
                 target_duration: int = 200, duration_tolerance: int = 50):
        """
        Initialize the fitness evaluator.
        
        Args:
            alpha: Weight for balance component
            beta: Weight for duration component  
            gamma: Weight for strategy diversity component
            target_duration: Target match duration in steps
            duration_tolerance: Acceptable deviation from target duration
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.target_duration = target_duration
        self.duration_tolerance = duration_tolerance
        
        self.simulator = MicroRTSMatchSimulator()
        
        # Validate weights sum to 1
        total_weight = alpha + beta + gamma
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Fitness weights must sum to 1.0, got {total_weight}")
    
    def evaluate_fitness(self, chromosome: MicroRTSChromosome, 
                        ai_pairs: List[Tuple[str, str]] = None) -> FitnessComponents:
        """
        Evaluate fitness of a chromosome.
        
        Args:
            chromosome: The chromosome to evaluate
            ai_pairs: List of AI pairs to test against (default: random pairs)
            
        Returns:
            FitnessComponents with all fitness metrics
        """
        if ai_pairs is None:
            ai_pairs = self._generate_ai_pairs()
        
        # Run matches for all AI pairs
        match_results = []
        for ai1, ai2 in ai_pairs:
            result = self.simulator.simulate_match(chromosome, ai1, ai2)
            match_results.append(result)
        
        # Calculate fitness components
        balance_score = self._calculate_balance_score(match_results)
        duration_score = self._calculate_duration_score(match_results)
        diversity_score = self._calculate_strategy_diversity_score(match_results)
        
        # Calculate overall fitness
        overall_fitness = (self.alpha * balance_score + 
                          self.beta * duration_score + 
                          self.gamma * diversity_score)
        
        return FitnessComponents(
            balance=balance_score,
            duration=duration_score,
            strategy_diversity=diversity_score,
            overall_fitness=overall_fitness
        )
    
    def _generate_ai_pairs(self) -> List[Tuple[str, str]]:
        """Generate random AI pairs for evaluation."""
        ai_agents = self.simulator.AI_AGENTS.copy()
        pairs = []
        
        # Generate pairs ensuring each AI is tested
        for _ in range(self.simulator.games_per_evaluation):
            ai1, ai2 = random.sample(ai_agents, 2)
            pairs.append((ai1, ai2))
        
        return pairs
    
    def _calculate_balance_score(self, match_results: List[MatchResult]) -> float:
        """
        Calculate balance score based on match outcomes.
        
        Higher score = more balanced matches (closer to 50/50 win rate)
        """
        if not match_results:
            return 0.0
        
        # Count wins for each side
        ai1_wins = sum(1 for result in match_results if result.winner == 0)
        ai2_wins = sum(1 for result in match_results if result.winner == 1)
        draws = sum(1 for result in match_results if result.winner == -1)
        
        total_matches = len(match_results)
        
        # Calculate balance (closer to 0.5 is better)
        if total_matches == 0:
            return 0.0
        
        # Reward draws and close win rates
        draw_ratio = draws / total_matches
        win_balance = 1.0 - abs(ai1_wins - ai2_wins) / total_matches
        
        # Combine draw ratio and win balance
        balance_score = 0.7 * draw_ratio + 0.3 * win_balance
        
        return min(1.0, max(0.0, balance_score))
    
    def _calculate_duration_score(self, match_results: List[MatchResult]) -> float:
        """
        Calculate duration score based on match lengths.
        
        Higher score = matches closer to target duration
        """
        if not match_results:
            return 0.0
        
        durations = [result.duration for result in match_results]
        avg_duration = statistics.mean(durations)
        
        # Gaussian penalty for being too far from target
        duration_diff = abs(avg_duration - self.target_duration)
        
        if duration_diff <= self.duration_tolerance:
            # Within acceptable range
            duration_score = 1.0 - (duration_diff / self.duration_tolerance) * 0.2
        else:
            # Outside acceptable range - exponential penalty
            excess = duration_diff - self.duration_tolerance
            duration_score = 0.8 * np.exp(-excess / 100.0)
        
        return min(1.0, max(0.0, duration_score))
    
    def _calculate_strategy_diversity_score(self, match_results: List[MatchResult]) -> float:
        """
        Calculate strategy diversity score based on action variety.
        
        Higher score = more diverse strategies across matches
        """
        if not match_results:
            return 0.0
        
        # Collect all actions from all matches
        all_actions = []
        for result in match_results:
            all_actions.extend(result.ai1_actions)
            all_actions.extend(result.ai2_actions)
        
        if not all_actions:
            return 0.0
        
        # Calculate action diversity using entropy
        action_counts = defaultdict(int)
        for action in all_actions:
            action_counts[action] += 1
        
        total_actions = len(all_actions)
        entropy = 0.0
        
        for count in action_counts.values():
            probability = count / total_actions
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Normalize entropy (max entropy is log2(num_unique_actions))
        max_entropy = np.log2(len(action_counts)) if len(action_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Also consider unit diversity
        unit_diversity = self._calculate_unit_diversity(match_results)
        
        # Combine action and unit diversity
        diversity_score = 0.6 * normalized_entropy + 0.4 * unit_diversity
        
        return min(1.0, max(0.0, diversity_score))
    
    def _calculate_unit_diversity(self, match_results: List[MatchResult]) -> float:
        """Calculate diversity based on unit type usage."""
        if not match_results:
            return 0.0
        
        # Collect unit counts from all matches
        all_units = defaultdict(int)
        for result in match_results:
            for unit_type, count in result.ai1_units_created.items():
                all_units[unit_type] += count
            for unit_type, count in result.ai2_units_created.items():
                all_units[unit_type] += count
        
        if not all_units:
            return 0.0
        
        # Calculate unit diversity using entropy
        total_units = sum(all_units.values())
        entropy = 0.0
        
        for count in all_units.values():
            probability = count / total_units
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Normalize entropy
        max_entropy = np.log2(len(all_units)) if len(all_units) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy


def evaluate_population_fitness(population: List[MicroRTSChromosome], 
                              evaluator: FitnessEvaluator) -> List[FitnessComponents]:
    """
    Evaluate fitness for an entire population.
    
    Args:
        population: List of chromosomes to evaluate
        evaluator: FitnessEvaluator instance
        
    Returns:
        List of FitnessComponents for each chromosome
    """
    fitness_scores = []
    
    for i, chromosome in enumerate(population):
        print(f"Evaluating chromosome {i+1}/{len(population)}...")
        fitness = evaluator.evaluate_fitness(chromosome)
        fitness_scores.append(fitness)
        print(f"  {fitness}")
    
    return fitness_scores
