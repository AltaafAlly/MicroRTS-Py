"""
Improved fitness evaluator that uses actual AI agents to evaluate UTT configurations.

This module provides a more sophisticated fitness evaluation that uses real AI agents
to play games and determine the quality of evolved UTT configurations.
"""

import json
import random
import numpy as np
import os
import sys
import time
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
from .utt_genetic_algorithm import FitnessEvaluator, Individual


class AIGameSimulationFitness(FitnessEvaluator):
    """Fitness evaluator using actual AI agents for game simulations."""
    
    def __init__(self, 
                 map_paths: List[str],
                 num_games: int = 5,
                 max_steps: int = 2000,
                 ai_agents: List[str] = None,
                 evaluation_metrics: List[str] = None):
        """Initialize fitness evaluator with AI agents.
        
        Args:
            map_paths: List of map files to use
            num_games: Number of games to play per evaluation
            max_steps: Maximum steps per game
            ai_agents: List of AI agent names to use for evaluation
            evaluation_metrics: List of metrics to use for fitness calculation
        """
        self.map_paths = map_paths
        self.num_games = num_games
        self.max_steps = max_steps
        
        # Default AI agents if none provided - using working AI names from run_match_configured.py
        if ai_agents is None:
            self.ai_agents = [
                "randomBiasedAI",
                "workerRushAI", 
                "lightRushAI",
                "passiveAI"
            ]
        else:
            self.ai_agents = ai_agents
        
        # Default evaluation metrics
        if evaluation_metrics is None:
            self.evaluation_metrics = [
                "win_rate",      # Win/loss ratio
                "resource_efficiency",  # Resource gathering efficiency
                "unit_production",      # Unit production rate
                "combat_effectiveness"  # Combat performance
            ]
        else:
            self.evaluation_metrics = evaluation_metrics
    
    def evaluate(self, individual: Individual) -> float:
        """Evaluate fitness using AI agents."""
        try:
            # Create temporary UTT file
            temp_utt_path = f"/tmp/temp_utt_{random.randint(10000, 99999)}.json"
            with open(temp_utt_path, 'w') as f:
                json.dump(individual.utt_data, f)
            
            total_fitness = 0.0
            total_games = 0
            
            # Test against different AI agents
            for map_path in self.map_paths:
                for ai_agent in self.ai_agents:
                    for _ in range(self.num_games // len(self.ai_agents)):
                        try:
                            fitness = self._play_game_with_ai(
                                individual.utt_data, 
                                temp_utt_path, 
                                ai_agent,
                                map_path
                            )
                            total_fitness += fitness
                            total_games += 1
                            
                        except Exception as e:
                            print(f"Game simulation error with {ai_agent}: {e}")
                            total_games += 1
                            total_fitness += 0.0  # Penalty for failed games
            
            # Clean up
            if os.path.exists(temp_utt_path):
                os.remove(temp_utt_path)
            
            # Return average fitness
            avg_fitness = total_fitness / max(total_games, 1)
            return avg_fitness
            
        except Exception as e:
            print(f"Fitness evaluation error: {e}")
            return 0.0
    
    def _play_game_with_ai(self, utt_data: Dict, temp_utt_path: str, ai_agent: str, map_path: str) -> float:
        """Play a single game with the specified AI agent using the correct pattern from run_match_configured.py."""
        try:
            # Get the AI function using the same pattern as run_match_configured.py
            ai_function = getattr(microrts_ai, ai_agent)
            if ai_function is None:
                raise ValueError(f"Unknown AI agent: {ai_agent}")
            
            # Create environment with evolved UTT for player 0, default UTT for player 1
            # Player 0: Evolved UTT with AI agent (to test how well AI performs with evolved UTT)
            # Player 1: Default UTT with AI agent (the opponent)
            env = MicroRTSBotVecEnv(
                ai1s=[ai_function],  # AI function for player 0 (with evolved UTT)
                ai2s=[ai_function],  # AI function for player 1 (with default UTT)
                max_steps=self.max_steps,
                map_paths=[map_path],
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                autobuild=True,
                utt_json_p0=temp_utt_path,  # Evolved UTT for player 0
                utt_json_p1=None,           # Default UTT for player 1 (AI)
            )
            
            # Run game simulation using the same pattern as run_match_configured.py
            _ = env.reset()
            h, w = env.height, env.width
            L = 7 * h * w
            dummy_actions = [[[0] * L, [0] * L]]
            
            results = {"left_wins": 0, "right_wins": 0, "draws": 0}
            
            # Play the game
            steps = 0
            while True:
                obs, rewards, done, info = env.step(dummy_actions)
                steps += 1
                if isinstance(done, (list, tuple, np.ndarray)):
                    done_flag = bool(done[0]) if len(done) else False
                else:
                    done_flag = bool(done)
                if not done_flag:
                    continue
                
                # Determine winner using the same logic as run_match_configured.py
                inf = info[0] if isinstance(info, list) and info else info
                winner = "draw"
                if isinstance(inf, dict) and "raw_rewards" in inf:
                    rr = inf["raw_rewards"]
                    rr = rr.tolist() if hasattr(rr, "tolist") else rr
                    if rr and rr[0] > 0:
                        winner = "left"  # Player 0 (evolved UTT) wins
                    elif rr and rr[0] < 0:
                        winner = "right"  # Player 1 (AI) wins
                
                if winner == "left":
                    results["left_wins"] = 1
                elif winner == "right":
                    results["right_wins"] = 1
                else:
                    results["draws"] = 1
                break
            
            # Close environment properly (don't call env.close() to keep JVM alive)
            try:
                env.vec_client.close()
            except Exception:
                pass
            
            # Calculate fitness based on results
            fitness = self._calculate_fitness_from_results(results, steps)
            return fitness
            
        except Exception as e:
            print(f"Error during game simulation with {ai_agent} on {map_path}: {e}")
            return 0.0

    def _calculate_fitness_from_results(self, results: Dict, steps: int) -> float:
        """Calculate fitness based on game results."""
        fitness_score = 0.0
        
        # Win/loss component (most important)
        if results["left_wins"] > 0:
            fitness_score += 100.0  # Big reward for winning
        elif results["right_wins"] > 0:
            fitness_score += 0.0    # No reward for losing
        else:
            fitness_score += 25.0   # Small reward for drawing
        
        # Game length component (longer games might indicate better balance)
        if steps > self.max_steps * 0.8:
            fitness_score += 10.0   # Bonus for long games
        
        return fitness_score


class MultiObjectiveFitness(FitnessEvaluator):
    """Multi-objective fitness evaluator for different aspects of UTT quality."""
    
    def __init__(self, 
                 map_paths: List[str],
                 objectives: List[str] = None,
                 weights: List[float] = None):
        """Initialize multi-objective fitness evaluator.
        
        Args:
            map_paths: List of map files
            objectives: List of objectives to optimize
            weights: Weights for each objective
        """
        self.map_paths = map_paths
        
        if objectives is None:
            self.objectives = [
                "balance",      # Game balance
                "diversity",    # Unit diversity
                "efficiency",   # Resource efficiency
                "fun_factor"    # Fun factor (subjective)
            ]
        else:
            self.objectives = objectives
        
        if weights is None:
            self.weights = [0.4, 0.2, 0.2, 0.2]  # Equal weights
        else:
            self.weights = weights
        
        # Create individual evaluators for each objective
        self.evaluators = {
            "balance": self._evaluate_balance,
            "diversity": self._evaluate_diversity,
            "efficiency": self._evaluate_efficiency,
            "fun_factor": self._evaluate_fun_factor
        }
    
    def evaluate(self, individual: Individual) -> float:
        """Evaluate multi-objective fitness."""
        try:
            scores = []
            
            for objective in self.objectives:
                if objective in self.evaluators:
                    score = self.evaluators[objective](individual)
                    scores.append(score)
                else:
                    scores.append(0.0)
            
            # Weighted sum
            weighted_fitness = sum(score * weight for score, weight in zip(scores, self.weights))
            return weighted_fitness
            
        except Exception as e:
            print(f"Multi-objective evaluation error: {e}")
            return 0.0
    
    def _evaluate_balance(self, individual: Individual) -> float:
        """Evaluate game balance."""
        # Check if unit costs are reasonable relative to their stats
        utt_data = individual.utt_data
        balance_score = 0.0
        
        for unit in utt_data["unitTypes"]:
            if unit["name"] == "Resource":
                continue
            
            # Cost-effectiveness ratio
            if unit["cost"] > 0:
                effectiveness = (unit["hp"] + unit["minDamage"] + unit["maxDamage"]) / 2
                cost_effectiveness = effectiveness / unit["cost"]
                balance_score += min(cost_effectiveness, 2.0)
        
        return balance_score / max(len(utt_data["unitTypes"]) - 1, 1)
    
    def _evaluate_diversity(self, individual: Individual) -> float:
        """Evaluate unit diversity."""
        utt_data = individual.utt_data
        units = [unit for unit in utt_data["unitTypes"] if unit["name"] != "Resource"]
        
        if len(units) < 2:
            return 0.0
        
        # Calculate variance in key stats
        costs = [unit["cost"] for unit in units]
        hps = [unit["hp"] for unit in units]
        damages = [(unit["minDamage"] + unit["maxDamage"]) / 2 for unit in units]
        
        diversity_score = (
            np.var(costs) + np.var(hps) + np.var(damages)
        ) / 3.0
        
        return min(diversity_score / 100.0, 1.0)  # Normalize
    
    def _evaluate_efficiency(self, individual: Individual) -> float:
        """Evaluate resource efficiency."""
        utt_data = individual.utt_data
        efficiency_score = 0.0
        
        for unit in utt_data["unitTypes"]:
            if unit["name"] == "Resource":
                continue
            
            # Production time efficiency
            if unit["produceTime"] > 0:
                time_efficiency = 100.0 / unit["produceTime"]
                efficiency_score += time_efficiency
            
            # Harvest efficiency for workers
            if unit["canHarvest"] and unit["harvestTime"] > 0:
                harvest_efficiency = unit["harvestAmount"] / unit["harvestTime"]
                efficiency_score += harvest_efficiency * 10
        
        return min(efficiency_score / 10.0, 1.0)  # Normalize
    
    def _evaluate_fun_factor(self, individual: Individual) -> float:
        """Evaluate fun factor (heuristic)."""
        utt_data = individual.utt_data
        fun_score = 0.0
        
        # Prefer moderate costs (not too cheap, not too expensive)
        for unit in utt_data["unitTypes"]:
            if unit["name"] == "Resource":
                continue
            
            cost = unit["cost"]
            if 2 <= cost <= 8:  # Sweet spot for costs
                fun_score += 1.0
            elif cost < 2 or cost > 15:
                fun_score += 0.2  # Penalty for extreme costs
        
        # Prefer reasonable HP values
        for unit in utt_data["unitTypes"]:
            if unit["name"] == "Resource":
                continue
            
            hp = unit["hp"]
            if 3 <= hp <= 20:  # Sweet spot for HP
                fun_score += 1.0
            elif hp < 1 or hp > 50:
                fun_score += 0.2  # Penalty for extreme HP
        
        return fun_score / max(len(utt_data["unitTypes"]) - 1, 1)


def create_fitness_evaluator(evaluator_type: str = "ai_game", **kwargs) -> FitnessEvaluator:
    """Factory function to create fitness evaluators."""
    
    if evaluator_type == "ai_game":
        return AIGameSimulationFitness(**kwargs)
    elif evaluator_type == "multi_objective":
        return MultiObjectiveFitness(**kwargs)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


# Example usage
if __name__ == "__main__":
    # Test the improved fitness evaluator
    map_paths = ["maps/8x8/basesWorkers8x8A.xml"]
    
    # Create AI-based evaluator
    ai_evaluator = AIGameSimulationFitness(
        map_paths=map_paths,
        num_games=2,
        max_steps=500,
        ai_agents=["randomBiasedAI", "workerRushAI"]
    )
    
    # Create multi-objective evaluator
    mo_evaluator = MultiObjectiveFitness(
        map_paths=map_paths,
        objectives=["balance", "diversity", "efficiency"],
        weights=[0.5, 0.3, 0.2]
    )
    
    print("Improved fitness evaluators created successfully!")
    print(f"AI evaluator: {type(ai_evaluator).__name__}")
    print(f"Multi-objective evaluator: {type(mo_evaluator).__name__}")