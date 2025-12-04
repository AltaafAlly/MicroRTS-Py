"""
Test any AI agent with runtime UTT change (Quiet version - minimal output).

Simply modify the configuration variables at the top to test different AI agents
and their corresponding favorable UTT files.
"""

import os
import sys

# Calculate project root: go up from test_single_matchup_quiet.py -> runtime_utt_change -> Running Simulations -> scripts -> project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai


def main():
    # Change these values to test different scenarios
    # AI Agents
    ai1_name = "lightRushAI"  
    opponent_ai_name = "coacAI"
    
    # Initial UTTs (set at game start)
    initial_utt_p0 = "utts/TestUnitTypeTable.json"  # Default UTT for Player 0
    initial_utt_p1 = "utts/TestUnitTypeTable.json"  # Default UTT for Player 1
    # "utts/TestUnitTypeTable.json", "utts/CustomDemoUTT.json", "utts/FavorWorkerRushUTT.json", "utts/BetterCombatViewUTT.json"
    
    # Runtime UTT Change
    change_utt = True  # Set to True to change UTT during game, False to keep initial UTTs
    utt_change_step = 20  # Step at which to change UTT (e.g., 1, 5, 10, 20, etc.)
    new_utt_p0 = "utts/CustomDemoUTT.json"  # New UTT for Player 0 at change step (None to keep unchanged)
    new_utt_p1 = None  # New UTT for Player 1 at change step (None to keep unchanged)
    
    # Game settings
    max_steps = 1000  # Maximum number of steps for the game
    
    # Get AI functions
    ai1_func = getattr(microrts_ai, ai1_name, microrts_ai.lightRushAI)
    opponent_ai_func = getattr(microrts_ai, opponent_ai_name, microrts_ai.passiveAI)
    
    print(f"Match: {ai1_name} (P0) vs {opponent_ai_name} (P1)")
    if change_utt:
        print(f"UTT change: Step {utt_change_step}")
    
    env = MicroRTSBotVecEnv(
        ai1s=[ai1_func],
        ai2s=[opponent_ai_func],
        map_paths=["maps/8x8/basesWorkers8x8A.xml"],
        max_steps=max_steps,
        utt_json_p0=initial_utt_p0,
        utt_json_p1=initial_utt_p1,
        autobuild=False,
    )
    
    obs = env.reset()
    
    utt_changed = False
    step = 0
    winner = None
    final_step = None
    
    while step < max_steps:
        actions = [0] * env.num_envs
        obs, rewards, dones, infos = env.step(actions)
        
        # Change UTT at specified step if requested
        if change_utt and step == utt_change_step and not utt_changed:
            success = env.change_utt(
                utt_json_p0=new_utt_p0,
                utt_json_p1=new_utt_p1
            )
            if success:
                print(f"✓ UTT changed at step {step}")
                utt_changed = True
            else:
                print(f"✗ Failed to change UTT at step {step}")
        
        if dones[0]:
            final_step = step
            # Determine winner
            if isinstance(infos, list) and len(infos) > 0:
                info = infos[0] if isinstance(infos[0], dict) else {}
                if "raw_rewards" in info:
                    raw_rewards = info["raw_rewards"]
                    if hasattr(raw_rewards, "tolist"):
                        raw_rewards = raw_rewards.tolist()
                    if isinstance(raw_rewards, (list, tuple)) and len(raw_rewards) > 0:
                        if raw_rewards[0] > 0:
                            winner = f"{ai1_name} (Player 0)"
                        elif raw_rewards[0] < 0:
                            winner = f"{opponent_ai_name} (Player 1)"
                        else:
                            winner = "Draw"
            break
        
        step += 1
    
    if winner is None:
        winner = "Draw (max steps reached)"
        final_step = max_steps
    
    env.close()
    
    # Minimal results output
    print(f"\nWinner: {winner}")
    print(f"Final Step: {final_step}")
    if change_utt:
        print(f"UTT Changed: {utt_changed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        import jpype
        if jpype.isJVMStarted():
            jpype.shutdownJVM()

