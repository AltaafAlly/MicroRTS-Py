#!/usr/bin/env python3
"""
Working GA Evaluator that uses the successful UTT testing approach.
This bypasses the UTT loading bug by using the working test_utt.py approach.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
import tempfile

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.ga_chromosome import MicroRTSChromosome
from core.ga_fitness_evaluator import FitnessEvaluator, FitnessComponents, MatchResult

class WorkingGAEvaluator(FitnessEvaluator):
    """
    GA Evaluator that uses the working UTT testing approach.
    This creates UTT files and tests them using the successful test_utt.py method.
    """
    
    def __init__(self, 
                 alpha: float = 0.5,  # Increased from 0.4 to prioritize balance
                 beta: float = 0.25,  # Reduced from 0.3
                 gamma: float = 0.25,  # Reduced from 0.3
                 max_steps: int = 1000,
                 map_path: str = "maps/8x8/basesWorkers8x8A.xml",
                 games_per_eval: int = 3,
                 ai_agents: List[str] = None,
                 min_balance_threshold: float = 0.2,  # Minimum acceptable balance per matchup
                 use_strict_balance: bool = True):  # Use stricter balance penalties
        """
        Initialize the working GA evaluator.
        
        Args:
            alpha: Balance weight (increased default to prioritize balance)
            beta: Duration weight
            gamma: Strategy diversity weight
            max_steps: Maximum steps per game
            map_path: Path to map file
            games_per_eval: Number of games per evaluation
            ai_agents: List of AI agents to use for testing
            min_balance_threshold: Minimum acceptable balance score per matchup (0.0-1.0)
            use_strict_balance: If True, apply exponential penalty for very imbalanced matchups
        """
        super().__init__(alpha, beta, gamma)
        
        self.max_steps = max_steps
        self.map_path = map_path
        self.games_per_eval = games_per_eval
        self.min_balance_threshold = min_balance_threshold
        self.use_strict_balance = use_strict_balance
        
        # Baseline AI agents for comprehensive evaluation
        # Covers diverse strategies: rush, balanced, defensive, competition AIs
        # Expanded to 8 AIs for better balance evaluation (28 pairs vs. 15 pairs = 87% more matchups)
        # This provides better balance assessment while keeping evaluation time reasonable
        self.baseline_ais = [
            # Rush strategies (early aggression)
            "workerRushAI",      # Classic worker rush - early aggression
            "lightRushAI",       # Light unit rush - fast military units
            
            # Balanced/Strategic AIs
            "coacAI",            # Strong balanced AI - good overall strategy
            "naiveMCTSAI",       # Monte Carlo Tree Search - planning-based
            "droplet",           # Strong competition AI - strategic gameplay (added for diversity)
            "mixedBot",          # Mixed strategy bot - adaptive gameplay (added for diversity)
            
            # Defensive/Economic
            "passiveAI",         # Defensive baseline - economic focus
            
            # Baseline/Random
            "randomBiasedAI",    # Biased random - slightly strategic randomness
            # Note: Excluded "randomAI" to keep at 8 AIs (28 pairs = 140 games/chromosome with 5 games/pair)
            # Note: Excluded "guidedRojoA3N" (causes issues), "rojo", "izanagi", "tiamat", "mayari" 
            #       to keep evaluation time manageable. Can be added if needed via ai_agents parameter.
        ]
        
        # Use provided AI agents or default to baseline
        self.ai_agents = ai_agents or self.baseline_ais
        
        # Round-robin: All AI agents play against each other
        # This gives maximum variety and comprehensive balance assessment
        # With 8 AIs, this creates 8 choose 2 = 28 unique pairs (vs. 15 with 6 AIs)
        self.comprehensive_test_pairs = self._generate_round_robin_pairs(self.baseline_ais)
        
        # Create temporary directory for UTT files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ga_utts_"))
        print(f"Created temporary UTT directory: {self.temp_dir}")
    
    def _generate_round_robin_pairs(self, ai_list: List[str]) -> List[Tuple[str, str]]:
        """
        Generate all unique pairs for round-robin tournament.
        
        Args:
            ai_list: List of AI agent names
            
        Returns:
            List of (ai1, ai2) tuples for all unique matchups
        """
        pairs = []
        for i in range(len(ai_list)):
            for j in range(i + 1, len(ai_list)):
                pairs.append((ai_list[i], ai_list[j]))
        return pairs
    
    def evaluate_chromosome(self, chromosome: MicroRTSChromosome) -> FitnessComponents:
        """
        Evaluate a chromosome by creating a UTT file and testing it.
        
        Args:
            chromosome: The chromosome to evaluate
            
        Returns:
            FitnessComponents with the evaluation results
        """
        print(f"  Evaluating chromosome with working UTT approach...")
        
        try:
            # Create UTT file from chromosome
            utt_path = self._create_utt_file(chromosome)
            
            # Test the UTT using the working approach
            match_results = self._test_utt_file(utt_path)
            
            # Calculate fitness from match results
            fitness = self._calculate_fitness(match_results)
            
            # Clean up temporary file
            if utt_path.exists():
                utt_path.unlink()
            
            print(f"    Fitness: {fitness.overall_fitness:.3f} (balance={fitness.balance:.3f}, duration={fitness.duration:.3f}, diversity={fitness.strategy_diversity:.3f})")
            
            return fitness
            
        except Exception as e:
            print(f"    Error evaluating chromosome: {e}")
            # Return default fitness on error
            return FitnessComponents(
                balance=0.5,
                duration=0.5,
                strategy_diversity=0.0,
                overall_fitness=0.3
            )
    
    def _create_utt_file(self, chromosome: MicroRTSChromosome) -> Path:
        """Create a UTT file from a chromosome."""
        
        # Generate UTT config
        utt_config = chromosome.to_microrts_config()
        
        # Create unique filename
        timestamp = int(time.time() * 1000)
        utt_filename = f"ga_utt_{timestamp}.json"
        utt_path = self.temp_dir / utt_filename
        
        # Save UTT file
        with open(utt_path, 'w') as f:
            json.dump(utt_config, f, indent=2)
        
        print(f"    Created UTT file: {utt_path}")
        return utt_path
    
    def _test_utt_file(self, utt_path: Path) -> List[Dict]:
        """
        Test a UTT file using round-robin tournament approach.
        
        All baseline AI agents play against each other in a round-robin format.
        With 8 AIs, this creates 28 unique matchups (vs. 15 with 6 AIs), providing:
        - Maximum variety in strategy matchups
        - Comprehensive balance assessment across all AI types
        - Better diversity measurement
        - More statistical coverage for balance evaluation
        """
        
        match_results = []
        
        # Use comprehensive baseline test pairs
        test_pairs = self.comprehensive_test_pairs
        
        print(f"    Testing {len(test_pairs)} AI pairs: {test_pairs}")
        
        # Calculate games per pair (at least 3 for meaningful balance stats)
        games_per_pair = max(3, self.games_per_eval)
        print(f"    Running {games_per_pair} games per pair ({len(test_pairs)} pairs = {games_per_pair * len(test_pairs)} total games)")
        
        for pair_idx, (ai1, ai2) in enumerate(test_pairs, 1):
            try:
                print(f"      [{pair_idx}/{len(test_pairs)}] Testing {ai1} vs {ai2} ({games_per_pair} games)...")
                
                # Copy UTT to microrts directory
                microrts_utt_path = self._copy_utt_to_microrts(utt_path)
                
                # Run match using the working approach
                result = self._run_match_with_utt(ai1, ai2, microrts_utt_path, games_per_pair)
                
                if result:
                    match_results.append({
                        'ai1': ai1,
                        'ai2': ai2,
                        'result': result
                    })
                    # Print quick summary
                    total = result.get('left_wins', 0) + result.get('right_wins', 0) + result.get('draws', 0)
                    if total > 0:
                        print(f"        Result: {result.get('left_wins', 0)}-{result.get('right_wins', 0)}-{result.get('draws', 0)} (L-W-D)")
                else:
                    print(f"        Warning: No result returned for {ai1} vs {ai2}")
                    # Continue - don't fail entire evaluation if one matchup fails
                
                # Clean up
                if microrts_utt_path.exists():
                    microrts_utt_path.unlink()
                    
            except Exception as e:
                print(f"        Error testing {ai1} vs {ai2}: {e}")
                print(f"        Skipping this matchup and continuing evaluation...")
                import traceback
                traceback.print_exc()
                # Continue evaluation even if one AI pair fails (some AIs may not be available)
                continue
        
        print(f"    Completed {len(match_results)}/{len(test_pairs)} match pairs successfully")
        
        return match_results
    
    def _copy_utt_to_microrts(self, utt_path: Path) -> Path:
        """Copy UTT file to microrts directory."""
        
        microrts_dir = project_root / "gym_microrts" / "microrts" / "utts"
        microrts_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = microrts_dir / "test_utt.json"
        import shutil
        shutil.copy2(utt_path, dest_path)
        
        return dest_path
    
    def _run_match_with_utt(self, ai1: str, ai2: str, utt_path: Path, games: int = None) -> Dict:
        """
        Run a match using the working approach.
        
        Args:
            ai1: First AI agent name
            ai2: Second AI agent name
            utt_path: Path to UTT file
            games: Number of games to run (defaults to self.games_per_eval or 3)
        """
        
        try:
            # Import the working match runner
            sys.path.insert(0, str(project_root / "scripts" / "Running Simulations"))
            from run_match_configured import run_pair
            
            # Run multiple games per pair for meaningful balance statistics
            # With only 1 game, balance is always 0.0 (one-sided) or 1.0 (draw)
            # With 3+ games, we can get meaningful win/loss ratios
            if games is None:
                games = max(3, self.games_per_eval)  # At least 3 games per pair
            
            result = run_pair(
                ai_left=ai1,
                ai_right=ai2,
                map_path=self.map_path,
                max_steps=self.max_steps,
                games=games,  # Multiple games for meaningful balance stats
                autobuild=False,
                utt_json=None,
                utt_json_p0="utts/test_utt.json",
                utt_json_p1="utts/test_utt.json"
            )
            
            return result
            
        except Exception as e:
            print(f"    Error running match: {e}")
            return None
    
    def _calculate_fitness(self, match_results: List[Dict]) -> FitnessComponents:
        """
        Calculate fitness from comprehensive match results.
        
        Uses multiple AI pairs to evaluate:
        - Balance: How fair matches are across different strategy matchups
        - Duration: How appropriate match lengths are
        - Strategy Diversity: How varied the outcomes are across different AI pairs
        """
        
        if not match_results:
            return FitnessComponents(
                balance=0.5,
                duration=0.5,
                strategy_diversity=0.0,
                overall_fitness=0.3
            )
        
        # Calculate balance (how fair the matches are)
        balance_scores = []
        duration_scores = []
        all_ai_names = set()  # Track which AIs were tested for diversity
        
        for match in match_results:
            result = match['result']
            if result:
                # Track AI diversity
                all_ai_names.add(match['ai1'])
                all_ai_names.add(match['ai2'])
                
                # Balance: closer to 50-50 is better
                total_games = result.get('left_wins', 0) + result.get('right_wins', 0) + result.get('draws', 0)
                if total_games > 0:
                    if result.get('left_wins', 0) == 0 and result.get('right_wins', 0) == 0:
                        # All draws - this is actually balanced!
                        balance = 1.0
                    else:
                        win_ratio = result.get('left_wins', 0) / total_games
                        # Perfect balance is 0.5 (50-50 win rate)
                        # Score: 1.0 for perfect balance, 0.0 for completely one-sided
                        imbalance = abs(win_ratio - 0.5)  # 0.0 = perfect, 0.5 = completely one-sided
                        base_balance = 1.0 - imbalance * 2
                        
                        # Apply stricter penalty for very imbalanced matchups
                        if self.use_strict_balance:
                            # Exponential penalty: very imbalanced matchups get penalized more
                            # This encourages the GA to avoid configurations with any highly imbalanced matchups
                            if imbalance > 0.35:  # More than 70-30 split
                                # Apply exponential penalty: (imbalance/0.5)^2
                                # Example: 0.4 imbalance (80-20) -> penalty factor of 0.64
                                penalty_factor = (imbalance / 0.5) ** 2
                                balance = base_balance * penalty_factor
                            elif imbalance > 0.25:  # Between 50-30 and 70-30
                                # Apply quadratic penalty: (imbalance/0.5)^1.5
                                penalty_factor = (imbalance / 0.5) ** 1.5
                                balance = base_balance * penalty_factor
                            else:
                                # Relatively balanced (within 60-40), use base score
                                balance = base_balance
                        else:
                            # Original linear penalty
                            balance = base_balance
                        
                        # Ensure balance is non-negative
                        balance = max(0.0, balance)
                    balance_scores.append(balance)
                    
                    # Duration: Since we don't have step count in results, we estimate based on outcomes
                    # Games that complete (have wins/losses) are better than all draws (too long)
                    # We reward decisive outcomes as indicators of reasonable game length
                    draws = result.get('draws', 0)
                    wins = result.get('left_wins', 0) + result.get('right_wins', 0)
                    
                    if total_games > 0:
                        draw_ratio = draws / total_games
                        # Lower draw ratio = games are completing = reasonable duration
                        # Some draws are OK (0-0.3), too many draws (0.5+) suggests games too long
                        if draw_ratio <= 0.3:
                            duration_score = 1.0  # Good: games completing
                        elif draw_ratio <= 0.5:
                            duration_score = 0.8  # Acceptable: some draws
                        elif draw_ratio <= 0.7:
                            duration_score = 0.6  # Concerning: many draws
                        else:
                            duration_score = 0.4  # Poor: mostly draws (games too long)
                        
                        duration_scores.append(duration_score)
                    else:
                        # No games completed - very poor duration
                        duration_scores.append(0.2)
        
        # Calculate balance using geometric mean to penalize very imbalanced matchups more
        # Geometric mean gives lower weight to outliers, so a single very imbalanced matchup
        # will significantly lower the overall balance score
        if balance_scores:
            if self.use_strict_balance:
                # Use geometric mean: more sensitive to very low scores
                # This means one very imbalanced matchup (e.g., 5-0-0) will significantly hurt the score
                import math
                # Avoid zero values (add small epsilon) and use geometric mean
                balanced_scores = [max(0.001, score) for score in balance_scores]  # Avoid log(0)
                log_sum = sum(math.log(score) for score in balanced_scores)
                geometric_mean = math.exp(log_sum / len(balanced_scores))
                
                # Also check minimum threshold: penalize if any matchup is below threshold
                min_balance = min(balance_scores)
                if min_balance < self.min_balance_threshold:
                    # Apply penalty: reduce overall balance if any matchup is very imbalanced
                    penalty = (min_balance / self.min_balance_threshold) ** 0.5  # Square root penalty
                    balance = geometric_mean * penalty
                else:
                    balance = geometric_mean
            else:
                # Simple arithmetic mean (original approach)
                balance = sum(balance_scores) / len(balance_scores)
        else:
            balance = 0.5
        
        # Calculate average duration score
        duration = sum(duration_scores) / len(duration_scores) if duration_scores else 0.5
        
        # Strategy Diversity: Based on:
        # 1. Number of different AIs tested (more = more diverse)
        # 2. Variance in balance scores across matchups (more variance = more diverse outcomes)
        # 3. Variety in match outcomes (wins, losses, draws)
        
        ai_diversity = min(1.0, len(all_ai_names) / len(self.baseline_ais))  # Normalize by number of baseline AIs
        
        if len(balance_scores) > 1:
            # Calculate variance in balance scores across different matchups
            balance_variance = sum((score - balance) ** 2 for score in balance_scores) / len(balance_scores)
            # Higher variance = more diverse outcomes (some matchups favor one side, others favor the other)
            # But we want some variance (not all 50-50, not all one-sided)
            variance_score = min(1.0, balance_variance * 2)  # Scale variance to 0-1
            
            # Count unique outcome patterns
            outcome_patterns = set()
            for match in match_results:
                result = match['result']
                if result:
                    pattern = (result.get('left_wins', 0), result.get('right_wins', 0), result.get('draws', 0))
                    outcome_patterns.add(pattern)
            pattern_diversity = min(1.0, len(outcome_patterns) / len(match_results))
            
            # Combine diversity metrics
            strategy_diversity = 0.4 * ai_diversity + 0.3 * variance_score + 0.3 * pattern_diversity
        else:
            # Single matchup - lower diversity
            strategy_diversity = 0.3 * ai_diversity
        
        # Calculate overall fitness
        overall = (self.alpha * balance + 
                  self.beta * duration + 
                  self.gamma * strategy_diversity)
        
        return FitnessComponents(
            balance=balance,
            duration=duration,
            strategy_diversity=strategy_diversity,
            overall_fitness=overall
        )
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

def evaluate_population_fitness_working(population: List[MicroRTSChromosome], 
                                      evaluator: WorkingGAEvaluator) -> List[FitnessComponents]:
    """
    Evaluate population fitness using the working approach.
    
    Args:
        population: List of chromosomes to evaluate
        evaluator: Working GA evaluator
        
    Returns:
        List of fitness components
    """
    print(f"Evaluating population of {len(population)} individuals using working UTT approach...")
    
    fitness_scores = []
    
    for i, chromosome in enumerate(population):
        print(f"  Evaluating individual {i+1}/{len(population)}...")
        
        fitness = evaluator.evaluate_chromosome(chromosome)
        chromosome.fitness = fitness
        fitness_scores.append(fitness)
    
    print(f"Population evaluation completed!")
    return fitness_scores
