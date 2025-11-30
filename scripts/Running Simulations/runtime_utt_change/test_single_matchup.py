"""
Test any AI agent with runtime UTT change.

Simply modify the configuration variables at the top to test different AI agents
and their corresponding favorable UTT files.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai


def main():
    # ===== CONFIGURATION - Change these values to test different scenarios =====
    # AI for Player 0 (will get favorable UTT)
    ai1_name = "workerRushAI"  # Options: "lightRushAI", "workerRushAI", etc.
    ai1_utt_file = "utts/FavorWorkerRushUTT.json"  # Favorable UTT for ai1_name
    # For workerRushAI, use: "utts/FavorWorkerRushUTT.json"
    
    # AI for Player 1 (opponent, keeps neutral UTT)
    opponent_ai_name = "coacAI"  # Options: passiveAI, randomAI, workerRushAI, lightRushAI, coacAI, etc.
    
    change_utt = True  # Set to False for control (no UTT change)
    utt_change_step = 50  # Step at which to change UTT (e.g., 1, 5, 10, 20, etc.)
    # ============================================================================
    
    # Get AI functions
    ai1_func = getattr(microrts_ai, ai1_name, microrts_ai.lightRushAI)
    opponent_ai_func = getattr(microrts_ai, opponent_ai_name, microrts_ai.passiveAI)
    
    # UTT descriptions for display
    utt_descriptions = {
        "utts/CustomDemoUTT.json": {
            "name": "CustomDemoUTT",
            "light": "Light: HP=60, Damage=20, Cost=1, Produce Time=2",
            "worker": "Worker: HP=10, Damage=5, Cost=1, Produce Time=10, Harvest=5"
        },
        "utts/FavorWorkerRushUTT.json": {
            "name": "FavorWorkerRushUTT",
            "light": "Light: HP=8, Damage=3, Cost=3, Produce Time=80",
            "worker": "Worker: HP=25, Damage=10, Cost=1, Produce Time=8, Harvest=8"
        }
    }
    
    utt_desc = utt_descriptions.get(ai1_utt_file, {
        "name": ai1_utt_file,
        "light": "Custom UTT",
        "worker": "Custom UTT"
    })
    
    print("=" * 70)
    print("Runtime UTT Change: Single Matchup Test")
    print("=" * 70)
    print(f"\nMatch: {ai1_name} (P0) vs {opponent_ai_name} (P1)")
    print(f"UTT Change: {'YES (at step ' + str(utt_change_step) + ')' if change_utt else 'NO (control)'}")
    if change_utt:
        print(f"\n{ai1_name} gets {utt_desc['name']} at step {utt_change_step}:")
        if "light" in ai1_name.lower():
            print(f"  {utt_desc['light']}")
        elif "worker" in ai1_name.lower():
            print(f"  {utt_desc['worker']}")
        print(f"{opponent_ai_name} keeps neutral UTT:")
        print("  Light: HP=4, Damage=2, Cost=2, Produce Time=65")
        print("  Worker: HP=1, Damage=1, Cost=1, Produce Time=50")
    print("=" * 70)
    
    env = MicroRTSBotVecEnv(
        ai1s=[ai1_func],
        ai2s=[opponent_ai_func],
        map_paths=["maps/8x8/basesWorkers8x8A.xml"],
        max_steps=2000,
        utt_json_p0="utts/TestUnitTypeTable.json",  # Start neutral
        utt_json_p1="utts/TestUnitTypeTable.json",  # Start neutral
        autobuild=False,
    )
    
    print("\n✓ Environment created")
    obs = env.reset()
    print("✓ Environment reset\n")
    
    utt_changed = False
    step = 0
    winner = None
    final_step = None
    
    while step < 2000:
        actions = [0] * env.num_envs
        obs, rewards, dones, infos = env.step(actions)
        
        # Change UTT at specified step if requested
        if change_utt and step == utt_change_step and not utt_changed:
            print(f">>> Step {step}: Changing UTT configuration...")
            success = env.change_utt(
                utt_json_p0=ai1_utt_file,  # Favorable UTT for ai1_name
                utt_json_p1=None  # Keep P1 unchanged
            )
            if success:
                print("✓ UTT changed successfully!")
                print(f"  Player 0 ({ai1_name}) now has {utt_desc['name']}")
                if "light" in ai1_name.lower():
                    print(f"    {utt_desc['light']}")
                elif "worker" in ai1_name.lower():
                    print(f"    {utt_desc['worker']}")
                print(f"  Player 1 ({opponent_ai_name}) keeps neutral UTT")
                print()
                utt_changed = True
            else:
                print("✗ Failed to change UTT")
        
        # Print progress every 50 steps and verify UTT is still changed
        if step > 0 and step % 50 == 0:
            # Verify UTT is still changed
            try:
                utt_check = env.real_utt_p0
                unit_types = utt_check.getUnitTypes()
                if "light" in ai1_name.lower():
                    # Check Light unit for lightRushAI
                    light_hp = None
                    for i in range(unit_types.size()):
                        if unit_types.get(i).name == "Light":
                            light_hp = unit_types.get(i).hp
                            break
                    if light_hp and light_hp >= 50:
                        print(f"  Step {step}: Game running... (reward: {rewards[0]:.2f}) [UTT verified: Light HP={light_hp}]")
                    else:
                        print(f"  Step {step}: Game running... (reward: {rewards[0]:.2f}) ⚠ UTT may have reverted!")
                elif "worker" in ai1_name.lower():
                    # Check Worker unit for workerRushAI
                    worker_hp = None
                    for i in range(unit_types.size()):
                        if unit_types.get(i).name == "Worker":
                            worker_hp = unit_types.get(i).hp
                            break
                    if worker_hp and worker_hp >= 20:
                        print(f"  Step {step}: Game running... (reward: {rewards[0]:.2f}) [UTT verified: Worker HP={worker_hp}]")
                    else:
                        print(f"  Step {step}: Game running... (reward: {rewards[0]:.2f}) ⚠ UTT may have reverted!")
                else:
                    print(f"  Step {step}: Game running... (reward: {rewards[0]:.2f})")
            except:
                print(f"  Step {step}: Game running... (reward: {rewards[0]:.2f})")
        
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
        final_step = 2000
    
    env.close()
    
    # Results
    print("\n" + "=" * 70)
    print("MATCH RESULT")
    print("=" * 70)
    print(f"Winner: {winner}")
    print(f"Final Step: {final_step}")
    print(f"UTT Changed: {utt_changed if change_utt else 'N/A (control)'}")
    
    if change_utt:
        if winner == f"{ai1_name} (Player 0)":
            print(f"\n✓ SUCCESS: {ai1_name} won with the favorable UTT!")
            print("  This confirms the runtime UTT change is working correctly.")
        elif winner == f"{opponent_ai_name} (Player 1)":
            print(f"\n⚠ {ai1_name} lost despite favorable UTT.")
            print("  Possible reasons:")
            print("    - Opponent is too strong")
            print("    - Map doesn't favor this strategy")
            print(f"    - UTT change happened too late (step {utt_change_step})")
            print("    - Early game momentum")
        else:
            print("\n→ Game ended in a draw.")
            print("  UTT change may have balanced the matchup, or game needs more steps.")
    
    print("=" * 70)
    print(f"\nTo test different scenarios, modify the configuration at the top of this script:")
    print(f"  - ai1_name: AI that gets favorable UTT (lightRushAI, workerRushAI, etc.)")
    print(f"  - ai1_utt_file: UTT file for ai1 (utts/CustomDemoUTT.json, utts/FavorWorkerRushUTT.json)")
    print(f"  - opponent_ai_name: Opponent AI (passiveAI, randomAI, workerRushAI, lightRushAI, coacAI, etc.)")
    print(f"  - utt_change_step: Step at which to change UTT (1, 5, 10, 20, etc.)")


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

