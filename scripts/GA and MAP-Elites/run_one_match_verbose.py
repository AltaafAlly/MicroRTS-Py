#!/usr/bin/env python3
"""
Run a single match and log game state at multiple steps to verify the game is played correctly.
Output: one file with snapshots at step 0, then every N steps, and at game end.

Usage (from project root):
  python "scripts/GA and MAP-Elites/run_one_match_verbose.py"
  python "scripts/GA and MAP-Elites/run_one_match_verbose.py" --steps 30  # snapshot every 30 steps
  python "scripts/GA and MAP-Elites/run_one_match_verbose.py" --out my_match.txt
"""

import os
import sys
import argparse
from datetime import datetime

# Project root and paths (same as run_ga_local_test)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
runtime_utt = os.path.join(PROJECT_ROOT, "scripts", "Running Simulations", "runtime_utt_change")
if runtime_utt not in sys.path and os.path.isdir(runtime_utt):
    sys.path.insert(0, runtime_utt)

import numpy as np
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai

# Import snapshot helper (same as runtime_utt_change)
try:
    from game_state_utils import get_game_snapshot_text
except ImportError:
    get_game_snapshot_text = None


def run_one_game_verbose(
    ai_left: str = "lightRushAI",
    ai_right: str = "workerRushAI",
    map_path: str = "maps/8x8/basesWorkers8x8A.xml",
    max_steps: int = 2000,
    snapshot_every: int = 20,
    utt_path_p0: str = None,
    utt_path_p1: str = None,
) -> tuple:
    """
    Run one game and collect game-state snapshots at step 0 and every snapshot_every steps.
    Returns (list of (step, snapshot_text), total_steps, winner_str).
    """
    a1 = getattr(microrts_ai, ai_left)
    a2 = getattr(microrts_ai, ai_right)
    utt_p0 = utt_path_p0 or "utts/TestUnitTypeTable.json"
    utt_p1 = utt_path_p1 or utt_path_p0 or "utts/TestUnitTypeTable.json"

    env = MicroRTSBotVecEnv(
        ai1s=[a1], ai2s=[a2],
        max_steps=max_steps,
        map_paths=[map_path],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False,
        utt_json=None,
        utt_json_p0=utt_p0,
        utt_json_p1=utt_p1,
    )
    snapshots = []
    _ = env.reset()
    h, w = env.height, env.width
    L = 7 * h * w
    dummy_actions = [[[0] * L, [0] * L]]

    # Step 0 (initial state)
    if get_game_snapshot_text:
        snapshots.append((0, get_game_snapshot_text(env, ai_left, ai_right)))
    else:
        snapshots.append((0, "(game_state_utils not available)"))

    step = 0
    winner = "draw"
    while step < max_steps:
        obs, rewards, done, info = env.step(dummy_actions)
        step += 1
        done_flag = bool(done[0]) if isinstance(done, (list, tuple, np.ndarray)) and len(done) else bool(done)

        # Snapshot at this step (every snapshot_every steps, or last step)
        if get_game_snapshot_text and (step % snapshot_every == 0 or done_flag):
            snapshots.append((step, get_game_snapshot_text(env, ai_left, ai_right)))

        if done_flag:
            inf = info[0] if isinstance(info, list) and info else info
            if isinstance(inf, dict) and "raw_rewards" in inf:
                rr = inf["raw_rewards"]
                rr = rr.tolist() if hasattr(rr, "tolist") else rr
                if rr and rr[0] > 0:
                    winner = "left"
                elif rr and rr[0] < 0:
                    winner = "right"
            break

    try:
        env.vec_client.close()
    except Exception:
        pass
    return snapshots, step, winner


def main():
    parser = argparse.ArgumentParser(description="Run one match with step-by-step state snapshots")
    parser.add_argument("--left", default="lightRushAI", help="Left AI name")
    parser.add_argument("--right", default="workerRushAI", help="Right AI name")
    parser.add_argument("--map", default="maps/8x8/basesWorkers8x8A.xml", help="Map path")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per game")
    parser.add_argument("--steps", type=int, default=20, dest="snapshot_every",
                        help="Log snapshot every N steps")
    parser.add_argument("--out", default=None, help="Output file (default: ga_run_logs/verbose_match_<timestamp>.txt)")
    args = parser.parse_args()

    out_dir = os.path.join(SCRIPT_DIR, "ga_run_logs")
    os.makedirs(out_dir, exist_ok=True)
    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(out_dir, args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"verbose_match_{args.left}_vs_{args.right}_{ts}.txt")

    print(f"Running one game: {args.left} vs {args.right} (snapshot every {args.snapshot_every} steps)...")
    snapshots, total_steps, winner = run_one_game_verbose(
        ai_left=args.left,
        ai_right=args.right,
        map_path=args.map,
        max_steps=args.max_steps,
        snapshot_every=args.snapshot_every,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Match: {args.left} (left) vs {args.right} (right)\n")
        f.write(f"Map: {args.map}  Max steps: {args.max_steps}\n")
        f.write(f"Result: Game ended at step {total_steps}  Winner: {winner}\n")
        f.write("=" * 70 + "\n\n")
        for step, text in snapshots:
            f.write(f"--- Step {step} ---\n")
            f.write(text)
            f.write("\n\n")
    print(f"Done. Total steps: {total_steps}, Winner: {winner}")
    print(f"Snapshots written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
