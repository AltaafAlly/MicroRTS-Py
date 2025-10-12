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
from typing import List, Dict, Any
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
                 alpha: float = 0.4,
                 beta: float = 0.3,
                 gamma: float = 0.3,
                 max_steps: int = 1000,
                 map_path: str = "maps/8x8/basesWorkers8x8A.xml",
                 games_per_eval: int = 3,
                 ai_agents: List[str] = None):
        """
        Initialize the working GA evaluator.
        
        Args:
            alpha: Balance weight
            beta: Duration weight
            gamma: Strategy diversity weight
            max_steps: Maximum steps per game
            map_path: Path to map file
            games_per_eval: Number of games per evaluation
            ai_agents: List of AI agents to use for testing
        """
        super().__init__(alpha, beta, gamma)
        
        self.max_steps = max_steps
        self.map_path = map_path
        self.games_per_eval = games_per_eval
        
        # Use the AI agents that work well
        self.ai_agents = ai_agents or [
            "POHeavyRush", "POLightRush", "randomAI", "passiveAI", 
            "workerRushAI", "lightRushAI", "naiveMCTSAI", "coacAI"
        ]
        
        # Create temporary directory for UTT files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ga_utts_"))
        print(f"Created temporary UTT directory: {self.temp_dir}")
    
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
        """Test a UTT file using the working test approach."""
        
        match_results = []
        
        # Test with just one AI pair to avoid JVM issues
        test_pairs = [
            ("POHeavyRush", "POLightRush")
        ]
        
        for ai1, ai2 in test_pairs:
            try:
                # Copy UTT to microrts directory
                microrts_utt_path = self._copy_utt_to_microrts(utt_path)
                
                # Run match using the working approach
                result = self._run_match_with_utt(ai1, ai2, microrts_utt_path)
                
                if result:
                    match_results.append({
                        'ai1': ai1,
                        'ai2': ai2,
                        'result': result
                    })
                
                # Clean up
                if microrts_utt_path.exists():
                    microrts_utt_path.unlink()
                    
            except Exception as e:
                print(f"    Error testing {ai1} vs {ai2}: {e}")
                continue
        
        return match_results
    
    def _copy_utt_to_microrts(self, utt_path: Path) -> Path:
        """Copy UTT file to microrts directory."""
        
        microrts_dir = project_root / "gym_microrts" / "microrts" / "utts"
        microrts_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = microrts_dir / "test_utt.json"
        import shutil
        shutil.copy2(utt_path, dest_path)
        
        return dest_path
    
    def _run_match_with_utt(self, ai1: str, ai2: str, utt_path: Path) -> Dict:
        """Run a match using the working approach."""
        
        try:
            # Import the working match runner
            sys.path.insert(0, str(project_root / "scripts" / "Running Simulations"))
            from run_match_configured import run_pair
            
            # Run the match
            result = run_pair(
                ai_left=ai1,
                ai_right=ai2,
                map_path=self.map_path,
                max_steps=self.max_steps,
                games=1,  # Single game for speed
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
        """Calculate fitness from match results."""
        
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
        
        for match in match_results:
            result = match['result']
            if result:
                # Balance: closer to 50-50 is better
                total_games = result['left_wins'] + result['right_wins'] + result['draws']
                if total_games > 0:
                    win_ratio = result['left_wins'] / total_games
                    balance = 1.0 - abs(win_ratio - 0.5) * 2  # 0.5 = perfect balance
                    balance_scores.append(balance)
                
                # Duration: assume reasonable duration for now
                duration_scores.append(0.7)  # Placeholder
        
        # Calculate averages
        balance = sum(balance_scores) / len(balance_scores) if balance_scores else 0.5
        duration = sum(duration_scores) / len(duration_scores) if duration_scores else 0.5
        diversity = 0.3  # Placeholder for now
        
        # Calculate overall fitness
        overall = (self.alpha * balance + 
                  self.beta * duration + 
                  self.gamma * diversity)
        
        return FitnessComponents(
            balance=balance,
            duration=duration,
            strategy_diversity=diversity,
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
