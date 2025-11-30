import os
import sys
import json

# Add project root to path (go up 3 levels: runtime_utt_change -> Running Simulations -> scripts -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai


def inspect_utt_properties(env, utt_name="UTT", player=0):
    """Inspect UTT properties to verify configuration"""
    try:
        # Get the UTT object
        utt = env.real_utt_p0 if player == 0 else env.real_utt_p1
        
        # Find Light unit type - iterate through unit types
        light_type = None
        unit_types = utt.getUnitTypes()
        for i in range(unit_types.size()):
            unit_type = unit_types.get(i)
            if unit_type.name == "Light":
                light_type = unit_type
                break
        
        if light_type:
            print(f"\n  {utt_name} (Player {player}) - Light Unit Stats:")
            print(f"    HP: {light_type.hp}")
            print(f"    Damage: {light_type.minDamage}-{light_type.maxDamage}")
            print(f"    Cost: {light_type.cost}")
            print(f"    Produce Time: {light_type.produceTime}")
            print(f"    Move Time: {light_type.moveTime}")
            print(f"    Attack Range: {light_type.attackRange}")
            
            # Also check Barracks
            barracks_type = None
            for i in range(unit_types.size()):
                unit_type = unit_types.get(i)
                if unit_type.name == "Barracks":
                    barracks_type = unit_type
                    break
            if barracks_type:
                print(f"    Barracks Cost: {barracks_type.cost}, Produce Time: {barracks_type.produceTime}")
        
        return light_type
    except Exception as e:
        print(f"  ⚠ Could not inspect {utt_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_utt_stats(utt1, utt2, unit_name="Light"):
    """Compare stats of a unit type between two UTTs"""
    try:
        unit_types1 = utt1.getUnitTypes()
        unit_types2 = utt2.getUnitTypes()
        
        type1 = None
        type2 = None
        
        for i in range(unit_types1.size()):
            if unit_types1.get(i).name == unit_name:
                type1 = unit_types1.get(i)
                break
        for i in range(unit_types2.size()):
            if unit_types2.get(i).name == unit_name:
                type2 = unit_types2.get(i)
                break
        
        if type1 and type2:
            return {
                'hp': (type1.hp, type2.hp),
                'damage': (f"{type1.minDamage}-{type1.maxDamage}", f"{type2.minDamage}-{type2.maxDamage}"),
                'cost': (type1.cost, type2.cost),
                'produce_time': (type1.produceTime, type2.produceTime),
            }
    except:
        pass
    return None


def main():
    print("=" * 70)
    print("Runtime UTT Change Example")
    print("=" * 70)
    print("\nCreating environment with initial UTT configuration...")
    
    # Create environment with initial UTT
    ai1_name = "lightRushAI"
    ai2_name = "workerRushAI"
    
    env = MicroRTSBotVecEnv(
        ai1s=[microrts_ai.lightRushAI],  # Use AI objects, not strings
        ai2s=[microrts_ai.workerRushAI],  # Use AI objects, not strings
        map_paths=["maps/8x8/basesWorkers8x8A.xml"],  # Must be a list (smaller map favors light rush)
        max_steps=2000,
        utt_json_p0="utts/TestUnitTypeTable.json",  # Start both with neutral UTT
        utt_json_p1="utts/TestUnitTypeTable.json",  # Start both with neutral UTT
        autobuild=False,  # Don't rebuild, we already built it
    )
    
    print("✓ Environment created")
    print(f"  Player 0 (AI1): {ai1_name}")
    print(f"  Player 1 (AI2): {ai2_name}")
    
    # Inspect initial UTT configurations
    print("\n📊 Initial UTT Configurations:")
    inspect_utt_properties(env, "Initial UTT (P0)", player=0)
    inspect_utt_properties(env, "Initial UTT (P1)", player=1)
    
    print("\nResetting environment...")
    obs = env.reset()
    print("✓ Environment reset")
    
    print("\nRunning game...")
    print("  Both players start with TestUnitTypeTable (neutral)")
    print("  UTT will change at step 1 to favor lightRushAI (Player 0)")
    print("  (This demonstrates the runtime UTT change effect)")
    print("(Game will run until completion or max 2000 steps)")
    print("-" * 70)
    
    utt_changed = False
    step = 0
    max_steps = 2000  # Use the environment's max_steps limit
    reward_before_change = None
    reward_after_change = []
    
    while step < max_steps:
        # For MicroRTSBotVecEnv, actions are simple integers (both AIs play automatically)
        actions = [0] * env.num_envs  # Simple actions for bot vs bot
        obs, rewards, dones, infos = env.step(actions)
        
        # Track reward before UTT change
        if step == 0:
            reward_before_change = rewards[0]
        
        # Change UTT immediately after first step to give lightRushAI (Player 0) the advantage
        if step == 1 and not utt_changed:
            print(f"\n>>> Step {step}: Changing UTT configuration at runtime...")
            print("  Changing Player 0's UTT to CustomDemoUTT (favors light rush)...")
            print(f"  Reward before change: {reward_before_change}")
            
            # Store UTT before change for comparison
            utt_before_p0 = env.real_utt_p0
            
            # Change UTT for Player 0 to favor light rush
            success = env.change_utt(
                utt_json_p0="utts/CustomDemoUTT.json",  # New UTT for Player 0 (favors light rush)
                utt_json_p1=None  # Keep Player 1's UTT unchanged (neutral)
            )
            
            if success:
                print("\n✓ UTT changed successfully!")
                
                # Verify UTT change by comparing UTT objects
                print("\n📊 Verification: UTT Change Comparison")
                utt_after_p0 = env.real_utt_p0
                comparison = compare_utt_stats(utt_before_p0, utt_after_p0, "Light")
                if comparison:
                    print("\n  Light Unit Stats Changed:")
                    print(f"    HP: {comparison['hp'][0]} → {comparison['hp'][1]} ", end="")
                    if comparison['hp'][0] != comparison['hp'][1]:
                        print("✓ CHANGED")
                    else:
                        print("(unchanged)")
                    print(f"    Damage: {comparison['damage'][0]} → {comparison['damage'][1]} ", end="")
                    if comparison['damage'][0] != comparison['damage'][1]:
                        print("✓ CHANGED")
                    else:
                        print("(unchanged)")
                    print(f"    Cost: {comparison['cost'][0]} → {comparison['cost'][1]} ", end="")
                    if comparison['cost'][0] != comparison['cost'][1]:
                        print("✓ CHANGED")
                    else:
                        print("(unchanged)")
                    print(f"    Produce Time: {comparison['produce_time'][0]} → {comparison['produce_time'][1]} ", end="")
                    if comparison['produce_time'][0] != comparison['produce_time'][1]:
                        print("✓ CHANGED")
                    else:
                        print("(unchanged)")
                
                # Inspect UTT properties
                print("\n📊 UTT Configurations AFTER change:")
                inspect_utt_properties(env, "New UTT (P0)", player=0)
                inspect_utt_properties(env, "Unchanged UTT (P1)", player=1)
                
                print(f"\n  Player 0 ({ai1_name}) now has CustomDemoUTT (favors light rush)")
                print(f"  Player 1 ({ai2_name}) keeps TestUnitTypeTable (neutral)")
                print("  Watch how rewards change after this point...")
                utt_changed = True
            else:
                print("✗ Failed to change UTT (e.g., unit type mismatch)")
        
        # Track rewards after change
        if utt_changed and step > 1:
            reward_after_change.append(rewards[0])
        
        if step % 10 == 0 and step > 0:
            reward_trend = ""
            if utt_changed and len(reward_after_change) > 5:
                recent_rewards = reward_after_change[-5:]
                if all(r > reward_before_change for r in recent_rewards):
                    reward_trend = " (↑ improving for P0)"
                elif all(r < reward_before_change for r in recent_rewards):
                    reward_trend = " (↓ declining for P0)"
            
            # Periodically verify UTT is still changed
            if utt_changed and step % 50 == 0:
                # Verify by checking UTT HP value
                try:
                    utt_p0 = env.real_utt_p0
                    unit_types = utt_p0.getUnitTypes()
                    light_type = None
                    for i in range(unit_types.size()):
                        if unit_types.get(i).name == "Light":
                            light_type = unit_types.get(i)
                            break
                    if light_type and light_type.hp >= 50:  # CustomDemoUTT has HP=60
                        verification = "✓ (UTT verified: Light HP={})".format(light_type.hp)
                    else:
                        verification = "⚠ (UTT may not have applied)"
                    print(f"  Step {step}: Game running... (reward: {rewards[0]}){reward_trend} {verification}")
                except:
                    print(f"  Step {step}: Game running... (reward: {rewards[0]}){reward_trend}")
            else:
                print(f"  Step {step}: Game running... (reward: {rewards[0]}){reward_trend}")
        
        if dones[0]:
            if not utt_changed:
                print(f"\n⚠ Game ended at step {step} before UTT change could occur")
            else:
                print(f"\nGame ended at step {step}")
                print("  (UTT was changed at step 1, observe the outcome)")
                if reward_after_change:
                    avg_reward_after = sum(reward_after_change) / len(reward_after_change)
                    print(f"  Reward before UTT change: {reward_before_change}")
                    print(f"  Average reward after UTT change: {avg_reward_after:.2f}")
                    if avg_reward_after > reward_before_change:
                        print("  ✓ Reward improved after UTT change (favoring Player 0)")
                    elif avg_reward_after < reward_before_change:
                        print("  ⚠ Reward decreased after UTT change")
                    else:
                        print("  → Reward remained similar")
            
            # Get winner immediately when game ends
            winner = "Draw"
            if isinstance(infos, list) and len(infos) > 0:
                info = infos[0] if isinstance(infos[0], dict) else {}
                if "raw_rewards" in info:
                    raw_rewards = info["raw_rewards"]
                    if hasattr(raw_rewards, "tolist"):
                        raw_rewards = raw_rewards.tolist()
                    if isinstance(raw_rewards, (list, tuple)) and len(raw_rewards) > 0:
                        if raw_rewards[0] > 0:
                            winner = f"Player 0 ({ai1_name})"
                        elif raw_rewards[0] < 0:
                            winner = f"Player 1 ({ai2_name})"
                        else:
                            winner = "Draw"
            print(f"🏆 Winner: {winner}")
            break
        
        step += 1
    
    # Determine winner from final rewards
    winner = "Draw"
    if isinstance(infos, list) and len(infos) > 0:
        info = infos[0] if isinstance(infos[0], dict) else {}
        if "raw_rewards" in info:
            raw_rewards = info["raw_rewards"]
            if hasattr(raw_rewards, "tolist"):
                raw_rewards = raw_rewards.tolist()
            if isinstance(raw_rewards, (list, tuple)) and len(raw_rewards) > 0:
                # raw_rewards[0] is the win/loss reward: +1 = P0 wins, -1 = P1 wins, 0 = draw
                if raw_rewards[0] > 0:
                    winner = f"Player 0 ({ai1_name})"
                elif raw_rewards[0] < 0:
                    winner = f"Player 1 ({ai2_name})"
                else:
                    winner = "Draw"
    
    # Only show final summary if game didn't end naturally (already shown above)
    if not dones[0]:
        print("\n" + "-" * 70)
        print("Game completed!")
        print(f"⚠ Game reached maximum steps ({max_steps}) without natural conclusion")
        print("  (This may indicate a very long/balanced game)")
        print(f"🏆 Winner: {winner}")
        print("=" * 70)
    
    # Clean up
    env.close()
    print("\nEnvironment closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Make sure to shutdown JVM on error
        import jpype
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
