#!/usr/bin/env python3
"""
Simple script to test evolved UTTs with run_match_configured.py
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the working example
sys.path.insert(0, str(project_root / "scripts" / "Running Simulations"))
from run_match_configured import run_pair
from utt_manager import UTTManager

def test_utt(utt_path: str, ai1: str = "POHeavyRush", ai2: str = "POLightRush", games: int = 3):
    """Test a specific UTT file."""
    
    print(f"🧪 Testing UTT: {Path(utt_path).name}")
    print(f"   AI1: {ai1}")
    print(f"   AI2: {ai2}")
    print(f"   Games: {games}")
    print()
    
    # Copy UTT to microrts directory
    manager = UTTManager()
    microrts_utt_path = manager.copy_utt_to_microrts(utt_path, "test_utt")
    
    try:
        # Run match
        result = run_pair(
            ai_left=ai1,
            ai_right=ai2,
            map_path="maps/8x8/basesWorkers8x8A.xml",
            max_steps=1000,
            games=games,
            autobuild=False,
            utt_json=None,
            utt_json_p0="utts/test_utt.json",
            utt_json_p1="utts/test_utt.json"
        )
        
        print(f"✅ Results: {result['left_wins']}-{result['right_wins']} (draws: {result['draws']})")
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    finally:
        # Clean up
        test_utt_path = Path("gym_microrts/microrts/utts/test_utt.json")
        if test_utt_path.exists():
            test_utt_path.unlink()

def test_best_utt(experiment_id: str = None, ai1: str = "POHeavyRush", ai2: str = "POLightRush", games: int = 3):
    """Test the best UTT from an experiment."""
    
    manager = UTTManager()
    best_utt = manager.get_best_utt(experiment_id)
    
    if not best_utt:
        print("❌ No UTTs found.")
        return None
    
    # Get the full path to the UTT file
    utt_filename = best_utt['utt_filename']
    if experiment_id:
        utt_path = manager.base_dir / "experiments" / experiment_id / utt_filename
    else:
        # Find the UTT file
        utt_path = None
        for meta_file in manager.base_dir.rglob("*.meta.json"):
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                if metadata['utt_filename'] == utt_filename:
                    utt_path = meta_file.parent / utt_filename
                    break
    
    if not utt_path or not utt_path.exists():
        print(f"❌ UTT file not found: {utt_filename}")
        return None
    
    print(f"🏆 Testing best UTT from experiment: {experiment_id or 'all'}")
    return test_utt(str(utt_path), ai1, ai2, games)

def compare_utts(utt1_path: str, utt2_path: str, ai1: str = "POHeavyRush", ai2: str = "POLightRush", games: int = 3):
    """Compare two UTT files."""
    
    print("🆚 Comparing UTTs")
    print("=" * 40)
    
    # Test first UTT
    print("1. Testing UTT 1...")
    result1 = test_utt(utt1_path, ai1, ai2, games)
    
    print()
    
    # Test second UTT
    print("2. Testing UTT 2...")
    result2 = test_utt(utt2_path, ai1, ai2, games)
    
    print()
    print("📊 Comparison Results:")
    if result1 and result2:
        print(f"   UTT 1: {result1['left_wins']}-{result1['right_wins']} (draws: {result1['draws']})")
        print(f"   UTT 2: {result2['left_wins']}-{result2['right_wins']} (draws: {result2['draws']})")
        
        # Determine which is better
        if result1['left_wins'] > result2['left_wins']:
            print("   🏆 UTT 1 performed better!")
        elif result2['left_wins'] > result1['left_wins']:
            print("   🏆 UTT 2 performed better!")
        else:
            print("   🤝 Both UTTs performed equally!")

def main():
    parser = argparse.ArgumentParser(description="Test evolved UTT files")
    parser.add_argument("--utt", type=str, help="Path to UTT file to test")
    parser.add_argument("--best", type=str, help="Test best UTT from experiment ID")
    parser.add_argument("--compare", nargs=2, help="Compare two UTT files")
    parser.add_argument("--ai1", type=str, default="POHeavyRush", help="First AI agent")
    parser.add_argument("--ai2", type=str, default="POLightRush", help="Second AI agent")
    parser.add_argument("--games", type=int, default=3, help="Number of games to play")
    
    args = parser.parse_args()
    
    if args.utt:
        test_utt(args.utt, args.ai1, args.ai2, args.games)
    elif args.best:
        test_best_utt(args.best, args.ai1, args.ai2, args.games)
    elif args.compare:
        compare_utts(args.compare[0], args.compare[1], args.ai1, args.ai2, args.games)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
