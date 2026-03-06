#!/usr/bin/env python3
"""
Working GA Evaluator that uses the successful UTT testing approach.
This bypasses the UTT loading bug by using the working test_utt.py approach.
"""

import os
import shutil
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import tempfile

# Add the project root and runtime_utt_change to the path (same as scripts/Running Simulations/runtime_utt_change)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
_runtime_utt_dir = project_root / "scripts" / "Running Simulations" / "runtime_utt_change"
if _runtime_utt_dir.is_dir() and str(_runtime_utt_dir) not in sys.path:
    sys.path.insert(0, str(_runtime_utt_dir))

from core.ga_chromosome import MicroRTSChromosome
from core.ga_fitness_evaluator import FitnessEvaluator, FitnessComponents, MatchResult
from core.ga_utt_validator import UTTValidator

class WorkingGAEvaluator(FitnessEvaluator):
    """
    GA Evaluator that uses the working UTT testing approach.
    This creates UTT files and tests them using the successful test_utt.py method.
    """
    
    def __init__(self, 
                 alpha: float = 0.4,  # Slightly prioritize balance, but not too dominant
                 beta: float = 0.3,
                 gamma: float = 0.3,
                 max_steps: int = 1000,
                 map_path: str = "maps/8x8/basesWorkers8x8A.xml",
                 map_paths: Optional[List[str]] = None,
                 games_per_eval: int = 3,
                 ai_agents: List[str] = None,
                 min_balance_threshold: float = 0.1,  # Minimum acceptable balance per matchup (softened)
                 use_strict_balance: bool = True,  # Use stricter balance penalties
                 use_nondeterministic: bool = False,  # If True, force random move conflicts and damage ranges
                 use_both_orderings: bool = False,  # If True, run (ai1,ai2) and (ai2,ai1) and aggregate → 50-50 when UTT is balanced
                 target_duration: int = 500,  # Target avg steps per game for duration score (sweet spot)
                 duration_tolerance: int = 400):  # Acceptable deviation; score decays outside [target±tolerance]
        """
        Initialize the working GA evaluator.
        
        Args:
            alpha: Balance weight (increased default to prioritize balance)
            beta: Duration weight
            gamma: Strategy diversity weight
            max_steps: Maximum steps per game
            map_path: Path to map file (used when map_paths is not set)
            map_paths: If set, run each matchup on every map and aggregate; gives mixed outcomes (e.g. 8-7) so balance can be non-zero
            games_per_eval: Number of games per evaluation (per map when map_paths is used)
            ai_agents: List of AI agents to use for testing
            min_balance_threshold: Minimum acceptable balance score per matchup (0.0-1.0)
            use_strict_balance: If True, apply exponential penalty for very imbalanced matchups
            use_nondeterministic: If True, write UTT with moveConflictResolutionStrategy=2 (random) and ensure combat units have minDamage < maxDamage so outcomes vary per game on one map
            use_both_orderings: If True, for each pair run (ai1,ai2) and (ai2,ai1) and aggregate; balanced UTT gives ~50-50 so balance > 0
            target_duration: Target average steps per game for duration score (sweet spot)
            duration_tolerance: Duration score decays when avg steps/game is outside [target ± tolerance]
        """
        super().__init__(alpha, beta, gamma)
        
        self.max_steps = max_steps
        self.map_path = map_path
        self.map_paths = map_paths if map_paths is not None else [map_path]
        self.games_per_eval = games_per_eval
        self.min_balance_threshold = min_balance_threshold
        self.use_strict_balance = use_strict_balance
        self.use_nondeterministic = use_nondeterministic
        self.use_both_orderings = use_both_orderings
        self.target_duration = target_duration
        self.duration_tolerance = duration_tolerance
        
        # Baseline AI agents for comprehensive evaluation
        # Covers diverse strategies: rush, balanced, defensive
        # Kept at 6 AIs for now to keep evaluation time reasonable while we tune balance
        self.baseline_ais = [
            # Rush strategies (early aggression)
            "workerRushAI",      # Classic worker rush - early aggression
            "lightRushAI",       # Light unit rush - fast military units
            
            # Balanced/Strategic AIs
            "coacAI",            # Strong balanced AI - good overall strategy
            "naiveMCTSAI",       # Monte Carlo Tree Search - planning-based
            
            # Defensive/Economic
            "passiveAI",         # Defensive baseline - economic focus
            
            # Baseline/Random
            "randomBiasedAI",    # Biased random - slightly strategic randomness
            # Note: Competition AIs (droplet, mixedBot, rojo, izanagi, tiamat, mayari)
            # are excluded here for now to reduce evaluation time and make balance tuning easier.
            # They can be reintroduced later via ai_agents if needed.
        ]
        
        # Use provided AI agents or default to baseline
        self.ai_agents = ai_agents or self.baseline_ais

        # Cached env for reuse across chromosome evaluations (avoids create/close each time)
        self._cached_env = None
        self._cached_env_key = None
        
        # Round-robin: All AI agents play against each other (use ai_agents so single-matchup mode works)
        # With 2 AIs this is 1 pair (e.g. lightRushAI vs workerRushAI); with 6 AIs, 15 pairs
        self.comprehensive_test_pairs = self._generate_round_robin_pairs(self.ai_agents)
        
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
            # Optional: log every UTT to a folder (gen{N}_ind{M}.json) for later comparison
            utt_log_dir = getattr(self, "utt_log_dir", None)
            if utt_log_dir:
                gen = getattr(self, "run_match_log_generation", 0)
                ind = getattr(self, "run_match_log_individual_index", 0)
                os.makedirs(utt_log_dir, exist_ok=True)
                dest = os.path.join(utt_log_dir, f"gen{gen}_ind{ind}.json")
                shutil.copy2(utt_path, dest)
            # Test the UTT using the working approach
            match_results = self._test_utt_file(utt_path)
            
            # Optional: append to run_match_log for CSV export (GA sets run_match_log, run_match_log_generation, run_match_log_individual_index)
            run_log = getattr(self, "run_match_log", None)
            if run_log is not None and getattr(self, "run_match_log_generation", None) is not None:
                gen = getattr(self, "run_match_log_generation", 0)
                ind_idx = getattr(self, "run_match_log_individual_index", 0)
                for match in match_results:
                    r = match.get("result") or {}
                    run_log.append({
                        "generation": gen,
                        "individual_index": ind_idx,
                        "ai_left": match.get("ai1", ""),
                        "ai_right": match.get("ai2", ""),
                        "left_wins": r.get("left_wins", 0),
                        "right_wins": r.get("right_wins", 0),
                        "draws": r.get("draws", 0),
                        "winner": "left" if r.get("left_wins", 0) > r.get("right_wins", 0) else ("right" if r.get("right_wins", 0) > r.get("left_wins", 0) else "draw"),
                        "left_unit_composition": r.get("_left_unit_composition", "N/A"),
                        "right_unit_composition": r.get("_right_unit_composition", "N/A"),
                        "_per_game_compositions": r.get("_per_game_compositions"),  # for match_outputs/*.txt
                        "_game_snapshots": r.get("_game_snapshots", []),  # (step, text) for match_outputs full map
                        "_games_per_ordering": r.get("_games_per_ordering"),  # when both orderings: 5 so we can label Games 1-5 vs 6-10
                    })
            
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
        
        # Generate UTT config and clamp to safe bounds (e.g. Worker cost <= 5 so Base can produce with 5 starting resources)
        utt_config = UTTValidator.validate_and_fix_utt(chromosome.to_microrts_config())
        
        # Nondeterministic mode: random move conflicts + wider damage ranges so outcomes can flip (balance signal)
        if self.use_nondeterministic:
            utt_config["moveConflictResolutionStrategy"] = 2  # CANCEL_RANDOM (Java UnitTypeTable.MOVE_CONFLICT_RESOLUTION_CANCEL_RANDOM)
            # Stronger nondeterminism: ensure combat units have at least 3-point damage spread (e.g. 2-5) so
            # damage variance is enough to flip close fights across games and get mixed results per map
            DAMAGE_SPREAD = 3  # minDamage+3 for more variance (was 2)
            combat_names = {"Worker", "Light", "Heavy", "Ranged"}
            for u in utt_config.get("unitTypes", []):
                if u.get("name") in combat_names and u.get("canAttack"):
                    mn = u.get("minDamage", 1)
                    mx = u.get("maxDamage", 1)
                    if mx <= mn:
                        u["maxDamage"] = max(mx, mn + DAMAGE_SPREAD)
                    elif mx < mn + DAMAGE_SPREAD:
                        # Widen existing range to at least DAMAGE_SPREAD for stronger variance
                        u["maxDamage"] = mn + DAMAGE_SPREAD
        
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
        With 6 AIs, this creates 15 unique matchups, providing:
        - Variety in strategy matchups
        - Comprehensive balance assessment across all AI types
        - Reasonable evaluation time while we tune balance
        """
        
        match_results = []
        
        # Use comprehensive baseline test pairs
        test_pairs = self.comprehensive_test_pairs
        
        print(f"    Testing {len(test_pairs)} AI pairs: {test_pairs}")
        
        # Calculate games per pair (at least 3 for meaningful balance stats)
        games_per_pair = max(1, self.games_per_eval)  # allow 1 game per map to avoid reinforcing same winner
        total_games_per_pair = games_per_pair * len(self.map_paths)
        print(f"    Running {games_per_pair} games per map × {len(self.map_paths)} map(s) = {total_games_per_pair} games per pair ({len(test_pairs)} pairs)")
        
        for pair_idx, (ai1, ai2) in enumerate(test_pairs, 1):
            try:
                orderings_note = " (both orderings)" if self.use_both_orderings else ""
                print(f"      [{pair_idx}/{len(test_pairs)}] Testing {ai1} vs {ai2}{orderings_note} ({games_per_pair} games × {len(self.map_paths)} map(s))...")
                
                # Copy UTT to a unique filename so the Java client cannot return cached results
                unique_id = str(time.time_ns())
                microrts_utt_path = self._copy_utt_to_microrts(utt_path, unique_suffix=unique_id)

                def _run_one_ordering(left_ai: str, right_ai: str):
                    """Run left_ai vs right_ai on all maps; return aggregated {left_wins, right_wins, draws, _total_steps}."""
                    if len(self.map_paths) == 1:
                        r = self._run_match_with_utt(left_ai, right_ai, microrts_utt_path, games_per_pair, map_path_override=self.map_paths[0])
                        return r if r else {"left_wins": 0, "right_wins": 0, "draws": 0, "_total_steps": 0}
                    agg = {"left_wins": 0, "right_wins": 0, "draws": 0, "_total_steps": 0}
                    for map_p in self.map_paths:
                        r = self._run_match_with_utt(left_ai, right_ai, microrts_utt_path, games_per_pair, map_path_override=map_p)
                        if r:
                            agg["left_wins"] += r.get("left_wins", 0)
                            agg["right_wins"] += r.get("right_wins", 0)
                            agg["draws"] += r.get("draws", 0)
                            agg["_total_steps"] += r.pop("_total_steps", 0)
                    return agg
                
                if self.use_both_orderings:
                    # Run (ai1, ai2) and (ai2, ai1); aggregate by AI identity so balance = "ai1 vs ai2" not "left vs right".
                    # R1: ai1=left, ai2=right  → ai1_wins=R1.left, ai2_wins=R1.right
                    # R2: ai2=left, ai1=right  → ai1_wins=R2.right, ai2_wins=R2.left
                    R1 = _run_one_ordering(ai1, ai2)
                    R2 = _run_one_ordering(ai2, ai1)
                    ai1_wins = R1.get("left_wins", 0) + R2.get("right_wins", 0)
                    ai2_wins = R1.get("right_wins", 0) + R2.get("left_wins", 0)
                    combined = {
                        "ai1_wins": ai1_wins,
                        "ai2_wins": ai2_wins,
                        "draws": R1.get("draws", 0) + R2.get("draws", 0),
                        "_total_steps": R1.get("_total_steps", 0) + R2.get("_total_steps", 0),
                        # For run_log/CSV: left/right = ai1/ai2 when both orderings
                        "left_wins": ai1_wins,
                        "right_wins": ai2_wins,
                    }
                    # Pass through capture data for match_outputs: merge both orderings so all 10 games are shown
                    s1 = R1.get("_game_snapshots") or []
                    s2 = R2.get("_game_snapshots") or []
                    combined["_game_snapshots"] = s1 + s2
                    # Unit composition summary: use last game of R1 (or R2) for left/right in ai1/ai2 terms
                    combined["_left_unit_composition"] = R1.get("_left_unit_composition", "N/A")
                    combined["_right_unit_composition"] = R1.get("_right_unit_composition", "N/A")
                    # Per-game compositions: R1 as-is; for R2 swap left/right and winner so Left=ai1, Right=ai2
                    p1 = R1.get("_per_game_compositions") or []
                    p2_raw = R2.get("_per_game_compositions") or []
                    p2_swapped = []
                    for pg in p2_raw:
                        p2_swapped.append({
                            "left": pg.get("right"),
                            "right": pg.get("left"),
                            "winner": "right" if pg.get("winner") == "left" else ("left" if pg.get("winner") == "right" else "draw"),
                            "game_index": len(p1) + len(p2_swapped),
                        })
                    combined["_per_game_compositions"] = p1 + p2_swapped
                    combined["_games_per_ordering"] = len(s1)  # so writer can label "Games 1-5" vs "Games 6-10"
                    total_steps = combined.get("_total_steps", 0)
                    if total_steps:
                        print(f"        (total steps this run: {total_steps})")
                    d = combined["draws"]
                    print(f"        Result: {ai1_wins}-{ai2_wins}-{d} ({ai1} vs {ai2} wins) (both orderings)")
                    match_results.append({"ai1": ai1, "ai2": ai2, "result": combined})
                elif len(self.map_paths) == 1:
                    result = self._run_match_with_utt(ai1, ai2, microrts_utt_path, games_per_pair,
                                                      map_path_override=self.map_paths[0])
                    results_to_append = [result] if result else []
                else:
                    results_to_append = []
                    total_steps_agg = 0
                    for map_p in self.map_paths:
                        r = self._run_match_with_utt(ai1, ai2, microrts_utt_path, games_per_pair, map_path_override=map_p)
                        if r:
                            total_steps_agg += r.get("_total_steps", 0)
                            results_to_append.append(r)
                    # For logging only: aggregated result (so we still print e.g. "30-0-0")
                    if results_to_append:
                        agg = {"left_wins": sum(x.get("left_wins", 0) for x in results_to_append),
                               "right_wins": sum(x.get("right_wins", 0) for x in results_to_append),
                               "draws": sum(x.get("draws", 0) for x in results_to_append),
                               "_total_steps": total_steps_agg}
                    else:
                        agg = None
                    result = agg  # used below for logging / sanity check
                
                if not self.use_both_orderings and results_to_append:
                    # For logging: use aggregated result (single map) or agg (multi-map); keep _total_steps in result for fitness
                    result_for_log = results_to_append[0] if len(self.map_paths) == 1 else agg
                    total_steps = result_for_log.get("_total_steps") if len(self.map_paths) == 1 else (agg.get("_total_steps") if agg else None)
                    if total_steps is not None:
                        print(f"        (total steps this run: {total_steps})")
                    total_games = sum(r.get('left_wins', 0) + r.get('right_wins', 0) + r.get('draws', 0) for r in results_to_append)
                    draws_total = sum(r.get('draws', 0) for r in results_to_append)
                    
                    # If total_steps is suspiciously low, re-run (single map only for simplicity)
                    MIN_STEPS_PER_GAME = 10
                    if len(self.map_paths) == 1 and total_steps is not None and total_games >= 3 and total_steps < MIN_STEPS_PER_GAME * total_games:
                        print(f"        WARNING: total steps ({total_steps}) suspiciously low for {total_games} games; re-running...")
                        self.close_cached_env()
                        result = self._run_match_with_utt(ai1, ai2, microrts_utt_path, games_per_pair, map_path_override=self.map_paths[0])
                        if result:
                            total_steps = result.pop("_total_steps", None)
                            results_to_append = [result]
                            total_games = result.get('left_wins', 0) + result.get('right_wins', 0) + result.get('draws', 0)
                            draws_total = result.get('draws', 0)
                    
                    # If all games were draws, rerun once (single map only)
                    if len(self.map_paths) == 1 and total_games >= 3 and draws_total == total_games:
                        print(f"        First run: 0-0-{draws_total} (all draws). Rerunning...")
                        rerun_result = self._run_match_with_utt(ai1, ai2, microrts_utt_path, games_per_pair, map_path_override=self.map_paths[0])
                        if rerun_result:
                            results_to_append = [rerun_result]
                            total_games = rerun_result.get('left_wins', 0) + rerun_result.get('right_wins', 0) + rerun_result.get('draws', 0)
                            print(f"        Rerun result: {rerun_result.get('left_wins', 0)}-{rerun_result.get('right_wins', 0)}-{rerun_result.get('draws', 0)} (L-W-D).")
                    
                    # Append one match_results entry per map (so balance is computed per map and can be > 0)
                    for r in results_to_append:
                        match_results.append({'ai1': ai1, 'ai2': ai2, 'result': r})
                    # Print aggregated summary
                    l = sum(x.get('left_wins', 0) for x in results_to_append)
                    w = sum(x.get('right_wins', 0) for x in results_to_append)
                    d = sum(x.get('draws', 0) for x in results_to_append)
                    print(f"        Result: {l}-{w}-{d} (L-W-D)" + (f" (per map: {len(results_to_append)} maps)" if len(results_to_append) > 1 else ""))
                elif not self.use_both_orderings:
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
    
    def _copy_utt_to_microrts(self, utt_path: Path, unique_suffix: str = None) -> Path:
        """Copy UTT file to microrts directory. Use unique_suffix to avoid path-based caching."""
        import shutil
        microrts_dir = project_root / "gym_microrts" / "microrts" / "utts"
        microrts_dir.mkdir(parents=True, exist_ok=True)
        if unique_suffix:
            dest_path = microrts_dir / f"test_utt_{unique_suffix}.json"
        else:
            dest_path = microrts_dir / "test_utt.json"
        shutil.copy2(utt_path, dest_path)
        return dest_path
    
    def _run_match_with_utt(self, ai1: str, ai2: str, utt_path: Path, games: int = None,
                            map_path_override: Optional[str] = None) -> Dict:
        """
        Run a match using the working approach.
        Always creates a new env (run_pair) per call so every evaluation runs real games.
        
        Args:
            ai1: First AI agent name
            ai2: Second AI agent name
            utt_path: Path to UTT file (caller copies to a unique file under utts/)
            games: Number of games to run (defaults to self.games_per_eval or 3)
            map_path_override: If set, use this map instead of self.map_path
        """
        # Ensure runtime_utt_change is on path so run_pair can import game_state_utils (capture/snapshots)
        _run_sim = project_root / "scripts" / "Running Simulations"
        if str(_run_sim) not in sys.path:
            sys.path.insert(0, str(_run_sim))
        if _runtime_utt_dir.is_dir() and str(_runtime_utt_dir) not in sys.path:
            sys.path.insert(0, str(_runtime_utt_dir))
        from run_match_configured import run_pair

        if games is None:
            games = max(1, self.games_per_eval)
        map_to_use = map_path_override if map_path_override is not None else self.map_path

        # Symmetric UTT: both players use the same evolved UTT (fair, interesting games)
        utt_rel = "utts/" + utt_path.name
        capture_composition = getattr(self, "run_match_capture_composition", False)
        capture_snapshots = getattr(self, "run_match_capture_snapshots", False)
        # Snapshot every N steps so short games (~40-80 steps) still show multiple states
        snapshot_interval = getattr(self, "run_match_snapshot_interval", 15)
        try:
            return run_pair(
                ai_left=ai1,
                ai_right=ai2,
                map_path=map_to_use,
                max_steps=self.max_steps,
                games=games,
                autobuild=False,
                utt_json=None,
                utt_json_p0=utt_rel,
                utt_json_p1=utt_rel,
                capture_composition=capture_composition,
                capture_snapshots=capture_snapshots,
                snapshot_interval=snapshot_interval,
            )
        except Exception as e:
            print(f"    Error running match: {e}")
            return None

    def close_cached_env(self) -> None:
        """Close the cached MicroRTS env, if any. Call at end of GA run to free resources."""
        if getattr(self, "_cached_env", None) is not None:
            try:
                self._cached_env.vec_client.close()
            except Exception:
                pass
            self._cached_env = None
            self._cached_env_key = None

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
                
                # Balance: by AI identity when both orderings (ai1_wins vs ai2_wins), else by side (left vs right)
                if "ai1_wins" in result and "ai2_wins" in result:
                    wins_ai1 = result.get("ai1_wins", 0)
                    wins_ai2 = result.get("ai2_wins", 0)
                    total_games = wins_ai1 + wins_ai2 + result.get("draws", 0)
                    decisive = wins_ai1 + wins_ai2
                else:
                    wins_ai1 = result.get("left_wins", 0)
                    wins_ai2 = result.get("right_wins", 0)
                    total_games = wins_ai1 + wins_ai2 + result.get("draws", 0)
                    decisive = wins_ai1 + wins_ai2
                if total_games > 0:
                    if decisive == 0:
                        # All draws = no competitive signal; treat as poor balance so GA doesn't optimize for timeouts
                        balance = 0.0
                    else:
                        # Win ratio for the two AIs (of decisive games, is it 50-50?)
                        win_ratio = wins_ai1 / decisive
                        # Perfect balance is 0.5 (50-50 win rate between the two AIs)
                        # Score: 1.0 for perfect balance, 0.0 for completely one-sided
                        imbalance = abs(win_ratio - 0.5)  # 0.0 = perfect, 0.5 = completely one-sided
                        base_balance = 1.0 - imbalance * 2
                        
                        # Apply stricter penalty for very imbalanced matchups
                        if self.use_strict_balance:
                            # Exponential penalty: very imbalanced matchups get penalized more
                            # This encourages the GA to avoid configurations with any highly imbalanced matchups
                            if imbalance > 0.4:  # More than 80-20 split
                                # Apply exponential penalty: (imbalance/0.5)^2
                                # Example: 0.4 imbalance (80-20) -> penalty factor of 0.64
                                penalty_factor = (imbalance / 0.5) ** 2
                                balance = base_balance * penalty_factor
                            elif imbalance > 0.3:  # Between ~60-40 and 80-20
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
                    
                    # Duration: prefer step-based (sensitive under symmetric UTT); fallback to draw-based
                    total_steps = result.get('_total_steps')
                    if total_games > 0 and total_steps is not None and total_steps >= 0:
                        avg_steps_per_game = total_steps / total_games
                        # Reward "good" length: peak at target_duration, decay outside [target ± tolerance]
                        dev = abs(avg_steps_per_game - self.target_duration)
                        if dev <= self.duration_tolerance:
                            duration_score = 1.0 - (dev / self.duration_tolerance)  # 1.0 at target, 0 at ±tolerance
                        else:
                            duration_score = 0.0
                        duration_scores.append(max(0.0, duration_score))
                    else:
                        # Fallback: use draw ratio (games completing = reasonable duration)
                        draws = result.get('draws', 0)
                        draw_ratio = draws / total_games if total_games > 0 else 1.0
                        if draw_ratio <= 0.3:
                            duration_score = 1.0
                        elif draw_ratio <= 0.5:
                            duration_score = 0.8
                        elif draw_ratio <= 0.7:
                            duration_score = 0.5
                        else:
                            duration_score = 0.15
                        duration_scores.append(duration_score)
        
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
        
        ai_diversity = min(1.0, len(all_ai_names) / len(self.ai_agents))  # Normalize by number of AIs in this run
        
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
    
    # Start each generation with no cached env so the first evaluation creates a fresh one.
    # Reusing across generations led to ~0.08s evals (cached/wrong results); reusing only
    # within a generation (at most 6 evals per env) keeps runs correct.
    if hasattr(evaluator, "close_cached_env"):
        evaluator.close_cached_env()
    
    fitness_scores = []
    
    for i, chromosome in enumerate(population):
        print(f"  Evaluating individual {i+1}/{len(population)}...")
        setattr(evaluator, "run_match_log_individual_index", i)
        fitness = evaluator.evaluate_chromosome(chromosome)
        chromosome.fitness = fitness
        fitness_scores.append(fitness)
    
    print(f"Population evaluation completed!")
    return fitness_scores
