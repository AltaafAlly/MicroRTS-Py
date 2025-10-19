#!/usr/bin/env python3
"""
Comprehensive UTT Impact Tournament
===================================

This script runs a large-scale tournament to analyze the impact of custom UTTs
vs default UTTs on different AI agent performance. Designed for cluster execution.

Key Features:
- Tests multiple UTT configurations (default vs custom)
- Uses strategically selected baseline AI agents
- Comprehensive result tracking and analysis
- Cluster-ready with progress logging
- Detailed statistical analysis
"""

import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import argparse

import numpy as np

# Let gym_microrts handle JVM startup automatically

from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai


class UTTImpactTournament:
    """Comprehensive tournament system for analyzing UTT impact on AI performance."""
    
    def __init__(self, output_dir: str = "utt_tournament_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategic baseline AI agents for comprehensive analysis
        # Optimized selection: 6 diverse agents covering different strategies
        self.baseline_ais = [
            # Rush strategies (aggressive)
            "workerRushAI",      # Classic worker rush
            "lightRushAI",       # Light unit rush
            
            # Balanced strategies
            "coacAI",            # Strong balanced AI
            "naiveMCTSAI",       # Monte Carlo Tree Search
            
            # Defensive/economic strategies
            "passiveAI",         # Defensive baseline
            "randomBiasedAI",    # Biased random (slightly strategic)
        ]
        
        # UTT configurations to test - Focused comparison
        self.utt_configs = {
            "default_original": {"utt_json_p0": None, "utt_json_p1": None},  # Default original UTT (baseline)
            "custom_demo": {"utt_json_p0": "utts/CustomDemoUTT.json", "utt_json_p1": "utts/CustomDemoUTT.json"},  # Your custom UTT
        }
        
        # Tournament settings
        self.tournament_config = {
            "map_path": "maps/8x8/basesWorkers8x8A.xml",
            "max_steps": 4000,
            "max_steps_long": 8000,  # For retry on high draws
            "games_per_pair": 10,    # More games for statistical significance
            "draw_retry_threshold": 0.5,  # Retry if >50% draws
            "autobuild": False,
        }
        
        self.results = {}
        self.start_time = datetime.now()
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp and level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def run_utt_comparison(self, utt_name: str, utt_config: Dict) -> Dict:
        """Run tournament for a specific UTT configuration."""
        self.log(f"Starting tournament for UTT: {utt_name}")
        
        # Filter available AIs
        available_ais = [ai for ai in self.baseline_ais if hasattr(microrts_ai, ai)]
        self.log(f"Available AIs: {', '.join(available_ais)}")
        
        # Initialize results structure
        utt_results = {
            "utt_name": utt_name,
            "utt_config": utt_config,
            "timestamp": datetime.now().isoformat(),
            "standings": {ai: {"wins": 0, "losses": 0, "draws": 0, "points": 0.0} for ai in available_ais},
            "pair_results": [],
            "statistics": {}
        }
        
        # Run round-robin tournament
        total_pairs = len(available_ais) * (len(available_ais) - 1) // 2
        pair_count = 0
        
        for i in range(len(available_ais)):
            for j in range(i + 1, len(available_ais)):
                pair_count += 1
                ai_left, ai_right = available_ais[i], available_ais[j]
                
                self.log(f"Pair {pair_count}/{total_pairs}: {ai_left} vs {ai_right}")
                
                # Run the match
                pair_result = self.run_pair(
                    ai_left, ai_right, utt_config, pair_count, total_pairs
                )
                
                # Update standings
                lw, rw, d = pair_result["left_wins"], pair_result["right_wins"], pair_result["draws"]
                utt_results["standings"][ai_left]["wins"] += lw
                utt_results["standings"][ai_left]["losses"] += rw
                utt_results["standings"][ai_left]["draws"] += d
                utt_results["standings"][ai_right]["wins"] += rw
                utt_results["standings"][ai_right]["losses"] += lw
                utt_results["standings"][ai_right]["draws"] += d
                
                # Calculate points (wins + 0.5*draws)
                utt_results["standings"][ai_left]["points"] = (
                    utt_results["standings"][ai_left]["wins"] + 
                    0.5 * utt_results["standings"][ai_left]["draws"]
                )
                utt_results["standings"][ai_right]["points"] = (
                    utt_results["standings"][ai_right]["wins"] + 
                    0.5 * utt_results["standings"][ai_right]["draws"]
                )
                
                utt_results["pair_results"].append(pair_result)
                
                self.log(f"Result: {ai_left} {lw}-{rw} {ai_right} (draws: {d})")
        
        # Calculate statistics
        utt_results["statistics"] = self.calculate_utt_statistics(utt_results)
        
        # Save individual UTT results
        self.save_utt_results(utt_name, utt_results)
        
        return utt_results
    
    def run_pair(self, ai_left: str, ai_right: str, utt_config: Dict, 
                 pair_num: int, total_pairs: int) -> Dict:
        """Run a single pair match with retry logic for high draws."""
        a1 = getattr(microrts_ai, ai_left)
        a2 = getattr(microrts_ai, ai_right)
        
        # Create environment with UTT configuration
        env = MicroRTSBotVecEnv(
            ai1s=[a1], ai2s=[a2],
            max_steps=self.tournament_config["max_steps"],
            map_paths=[self.tournament_config["map_path"]],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            autobuild=self.tournament_config["autobuild"],
            utt_json_p0=utt_config["utt_json_p0"],
            utt_json_p1=utt_config["utt_json_p1"],
        )
        
        _ = env.reset()
        h, w = env.height, env.width
        L = 7 * h * w
        dummy_actions = [[[0] * L, [0] * L]]
        
        results = {"left_wins": 0, "right_wins": 0, "draws": 0}
        games = self.tournament_config["games_per_pair"]
        
        for game_num in range(games):
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
                    
                inf = info[0] if isinstance(info, list) and info else info
                winner = "draw"
                if isinstance(inf, dict) and "raw_rewards" in inf:
                    rr = inf["raw_rewards"]
                    rr = rr.tolist() if hasattr(rr, "tolist") else rr
                    if rr and rr[0] > 0:
                        winner = "left"
                    elif rr and rr[0] < 0:
                        winner = "right"
                        
                if winner == "left":
                    results["left_wins"] += 1
                elif winner == "right":
                    results["right_wins"] += 1
                else:
                    results["draws"] += 1
                    
                _ = env.reset()
                break
        
        # Check for high draw rate and retry with longer horizon
        draw_rate = results["draws"] / games
        if (draw_rate >= self.tournament_config["draw_retry_threshold"] and 
            self.tournament_config["max_steps_long"]):
            
            self.log(f"High draw rate ({draw_rate:.1%}) for {ai_left} vs {ai_right}. Retrying with longer horizon...")
            
            # Retry with longer horizon
            env_long = MicroRTSBotVecEnv(
                ai1s=[a1], ai2s=[a2],
                max_steps=self.tournament_config["max_steps_long"],
                map_paths=[self.tournament_config["map_path"]],
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                autobuild=self.tournament_config["autobuild"],
                utt_json_p0=utt_config["utt_json_p0"],
                utt_json_p1=utt_config["utt_json_p1"],
            )
            
            _ = env_long.reset()
            dummy_actions_long = [[[0] * L, [0] * L]]
            
            results_long = {"left_wins": 0, "right_wins": 0, "draws": 0}
            
            for game_num in range(games):
                steps = 0
                while True:
                    obs, rewards, done, info = env_long.step(dummy_actions_long)
                    steps += 1
                    if isinstance(done, (list, tuple, np.ndarray)):
                        done_flag = bool(done[0]) if len(done) else False
                    else:
                        done_flag = bool(done)
                    if not done_flag:
                        continue
                        
                    inf = info[0] if isinstance(info, list) and info else info
                    winner = "draw"
                    if isinstance(inf, dict) and "raw_rewards" in inf:
                        rr = inf["raw_rewards"]
                        rr = rr.tolist() if hasattr(rr, "tolist") else rr
                        if rr and rr[0] > 0:
                            winner = "left"
                        elif rr and rr[0] < 0:
                            winner = "right"
                            
                    if winner == "left":
                        results_long["left_wins"] += 1
                    elif winner == "right":
                        results_long["right_wins"] += 1
                    else:
                        results_long["draws"] += 1
                        
                    _ = env_long.reset()
                    break
            
            # Use the longer horizon results
            results = results_long
            self.log(f"Retry result: {ai_left} {results['left_wins']}-{results['right_wins']} {ai_right} (draws: {results['draws']})")
            
            try:
                env_long.vec_client.close()
            except Exception:
                pass
        
        # Close environment
        try:
            env.vec_client.close()
        except Exception:
            pass
        
        # Add metadata to results
        results.update({
            "ai_left": ai_left,
            "ai_right": ai_right,
            "utt_config": utt_config,
            "games": games,
            "map_path": self.tournament_config["map_path"],
            "pair_number": pair_num,
            "total_pairs": total_pairs,
        })
        
        return results
    
    def calculate_utt_statistics(self, utt_results: Dict) -> Dict:
        """Calculate comprehensive statistics for a UTT configuration."""
        standings = utt_results["standings"]
        pair_results = utt_results["pair_results"]
        
        # Basic statistics
        total_games = sum(len([p for p in pair_results if p["ai_left"] == ai or p["ai_right"] == ai]) * 
                         self.tournament_config["games_per_pair"] for ai in standings.keys())
        
        total_wins = sum(rec["wins"] for rec in standings.values())
        total_losses = sum(rec["losses"] for rec in standings.values())
        total_draws = sum(rec["draws"] for rec in standings.values())
        
        # Win rate analysis
        win_rates = {ai: rec["wins"] / max(1, rec["wins"] + rec["losses"] + rec["draws"]) 
                    for ai, rec in standings.items()}
        
        # Ranking analysis
        sorted_ais = sorted(standings.items(), key=lambda x: x[1]["points"], reverse=True)
        
        # Draw rate analysis
        draw_rates = [p["draws"] / p["games"] for p in pair_results]
        avg_draw_rate = np.mean(draw_rates) if draw_rates else 0
        
        return {
            "total_games": total_games,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_draws": total_draws,
            "overall_draw_rate": total_draws / max(1, total_games),
            "average_draw_rate_per_pair": avg_draw_rate,
            "win_rates": win_rates,
            "ranking": [{"ai": ai, "points": rec["points"], "wins": rec["wins"], 
                        "losses": rec["losses"], "draws": rec["draws"]} 
                       for ai, rec in sorted_ais],
            "top_performer": sorted_ais[0][0] if sorted_ais else None,
            "bottom_performer": sorted_ais[-1][0] if sorted_ais else None,
        }
    
    def save_utt_results(self, utt_name: str, utt_results: Dict):
        """Save results for a specific UTT configuration."""
        utt_dir = self.output_dir / utt_name
        utt_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        with open(utt_dir / "detailed_results.json", "w") as f:
            json.dump(utt_results, f, indent=2)
        
        # Save standings as CSV
        with open(utt_dir / "standings.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ai", "points", "wins", "losses", "draws", "win_rate"])
            for ai, rec in utt_results["standings"].items():
                win_rate = rec["wins"] / max(1, rec["wins"] + rec["losses"] + rec["draws"])
                writer.writerow([ai, rec["points"], rec["wins"], rec["losses"], rec["draws"], win_rate])
        
        # Save pair results as CSV
        with open(utt_dir / "pair_results.csv", "w", newline="") as f:
            if utt_results["pair_results"]:
                writer = csv.DictWriter(f, fieldnames=utt_results["pair_results"][0].keys())
                writer.writeheader()
                writer.writerows(utt_results["pair_results"])
    
    def run_full_tournament(self):
        """Run the complete tournament across all UTT configurations."""
        self.log("Starting UTT Impact Tournament")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Testing {len(self.utt_configs)} UTT configurations")
        self.log(f"Using {len(self.baseline_ais)} baseline AI agents")
        
        all_results = {}
        
        for utt_name, utt_config in self.utt_configs.items():
            self.log(f"\n{'='*60}")
            self.log(f"UTT Configuration: {utt_name}")
            self.log(f"{'='*60}")
            
            utt_results = self.run_utt_comparison(utt_name, utt_config)
            all_results[utt_name] = utt_results
            
            # Print summary
            self.log(f"\nSummary for {utt_name}:")
            sorted_ais = sorted(utt_results["standings"].items(), 
                              key=lambda x: x[1]["points"], reverse=True)
            for i, (ai, rec) in enumerate(sorted_ais[:5]):  # Top 5
                self.log(f"  {i+1}. {ai}: {rec['points']} pts (W{rec['wins']} L{rec['losses']} D{rec['draws']})")
        
        # Generate comparative analysis
        self.generate_comparative_analysis(all_results)
        
        # Save overall tournament results
        self.save_tournament_summary(all_results)
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.log(f"\nTournament completed in {duration}")
        self.log(f"Results saved to: {self.output_dir}")
    
    def generate_comparative_analysis(self, all_results: Dict):
        """Generate comparative analysis across UTT configurations."""
        self.log("\nGenerating comparative analysis...")
        
        # Create comparison matrix
        comparison_data = []
        
        for utt_name, utt_results in all_results.items():
            for ai, rec in utt_results["standings"].items():
                win_rate = rec["wins"] / max(1, rec["wins"] + rec["losses"] + rec["draws"])
                comparison_data.append({
                    "utt": utt_name,
                    "ai": ai,
                    "points": rec["points"],
                    "wins": rec["wins"],
                    "losses": rec["losses"],
                    "draws": rec["draws"],
                    "win_rate": win_rate,
                })
        
        # Save comparison data
        with open(self.output_dir / "utt_comparison.csv", "w", newline="") as f:
            if comparison_data:
                writer = csv.DictWriter(f, fieldnames=comparison_data[0].keys())
                writer.writeheader()
                writer.writerows(comparison_data)
        
        # Generate UTT impact analysis
        self.analyze_utt_impact(all_results)
    
    def analyze_utt_impact(self, all_results: Dict):
        """Analyze the impact of different UTTs on AI performance."""
        self.log("Analyzing UTT impact on AI performance...")
        
        # Get baseline (default original) results
        baseline_utt = "default_original"
        if baseline_utt not in all_results:
            self.log(f"Warning: Baseline UTT {baseline_utt} not found")
            return
        
        baseline_results = all_results[baseline_utt]["standings"]
        impact_analysis = {}
        
        for utt_name, utt_results in all_results.items():
            if utt_name == baseline_utt:
                continue
                
            utt_standings = utt_results["standings"]
            impact_analysis[utt_name] = {}
            
            for ai in baseline_results.keys():
                if ai in utt_standings:
                    baseline_points = baseline_results[ai]["points"]
                    utt_points = utt_standings[ai]["points"]
                    point_change = utt_points - baseline_points
                    percent_change = (point_change / max(1, baseline_points)) * 100
                    
                    impact_analysis[utt_name][ai] = {
                        "baseline_points": baseline_points,
                        "utt_points": utt_points,
                        "point_change": point_change,
                        "percent_change": percent_change,
                    }
        
        # Save impact analysis
        with open(self.output_dir / "utt_impact_analysis.json", "w") as f:
            json.dump(impact_analysis, f, indent=2)
        
        # Print summary of biggest impacts
        self.log("\nBiggest UTT Impact Changes:")
        for utt_name, impacts in impact_analysis.items():
            sorted_impacts = sorted(impacts.items(), 
                                  key=lambda x: abs(x[1]["percent_change"]), 
                                  reverse=True)
            self.log(f"\n{utt_name} vs {baseline_utt}:")
            for ai, impact in sorted_impacts[:3]:  # Top 3 changes
                direction = "↑" if impact["percent_change"] > 0 else "↓"
                self.log(f"  {ai}: {direction}{abs(impact['percent_change']):.1f}% "
                        f"({impact['baseline_points']:.1f} → {impact['utt_points']:.1f})")
    
    def save_tournament_summary(self, all_results: Dict):
        """Save overall tournament summary."""
        summary = {
            "tournament_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
                "utt_configurations": list(self.utt_configs.keys()),
                "ai_agents": self.baseline_ais,
                "tournament_config": self.tournament_config,
            },
            "results_summary": {}
        }
        
        for utt_name, utt_results in all_results.items():
            stats = utt_results["statistics"]
            summary["results_summary"][utt_name] = {
                "total_games": stats["total_games"],
                "overall_draw_rate": stats["overall_draw_rate"],
                "top_performer": stats["top_performer"],
                "bottom_performer": stats["bottom_performer"],
                "ranking": stats["ranking"][:5],  # Top 5
            }
        
        with open(self.output_dir / "tournament_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="UTT Impact Tournament")
    parser.add_argument("--output-dir", default="utt_tournament_results", 
                       help="Output directory for results")
    parser.add_argument("--utt", choices=["default_original", "default_finetuned", "custom_demo", "asymmetric_p1"],
                       help="Run tournament for specific UTT only")
    parser.add_argument("--games", type=int, default=10,
                       help="Number of games per pair")
    parser.add_argument("--max-steps", type=int, default=4000,
                       help="Maximum steps per game")
    
    args = parser.parse_args()
    
    # Create tournament instance
    tournament = UTTImpactTournament(output_dir=args.output_dir)
    
    # Override config if specified
    if args.games:
        tournament.tournament_config["games_per_pair"] = args.games
    if args.max_steps:
        tournament.tournament_config["max_steps"] = args.max_steps
    
    # Run tournament
    if args.utt:
        # Run single UTT
        utt_config = tournament.utt_configs[args.utt]
        tournament.run_utt_comparison(args.utt, utt_config)
    else:
        # Run full tournament
        tournament.run_full_tournament()


if __name__ == "__main__":
    main()
