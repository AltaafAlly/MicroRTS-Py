#!/usr/bin/env python3
"""
Local UTT Impact Tournament
===========================

Simplified version for local testing without cluster dependencies.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add the scripts directory to the path so we can import the tournament class
sys.path.append(str(Path(__file__).parent.parent / "scripts" / "Running Simulations"))

from utt_impact_tournament import UTTImpactTournament

def main():
    print("=" * 60)
    print("LOCAL UTT IMPACT TOURNAMENT")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Create output directory
    output_dir = Path("local_testing/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}")
    
    # Create tournament instance
    tournament = UTTImpactTournament(output_dir=str(output_dir))
    
    # Override config for local testing (shorter/faster)
    tournament.tournament_config["games_per_pair"] = 5  
    tournament.tournament_config["max_steps"] = 5000    
    tournament.tournament_config["max_steps_long"] = 10000  
    
    print("\nStarting tournament...")
    print(f"Testing {len(tournament.utt_configs)} UTT configurations")
    print(f"Using {len(tournament.baseline_ais)} AI agents")
    
    try:
        # Run the tournament
        tournament.run_full_tournament()
        
        print("\n" + "=" * 60)
        print("TOURNAMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {output_dir.absolute()}")
        
        # Show summary
        results_file = output_dir / "tournament_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print("\nQUICK SUMMARY:")
            for utt_name, utt_data in results.items():
                print(f"\n{utt_name}:")
                for ai_name, ai_data in utt_data.items():
                    wins = ai_data.get('wins', 0)
                    losses = ai_data.get('losses', 0)
                    draws = ai_data.get('draws', 0)
                    print(f"  {ai_name}: {wins}W-{losses}L-{draws}D")
        
    except Exception as e:
        print(f"\nERROR: Tournament failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
