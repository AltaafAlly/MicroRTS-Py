"""
Test any AI agent with runtime UTT change.

Simply modify the configuration variables at the top to test different AI agents
and their corresponding favorable UTT files.
"""

import os
import sys
import datetime
from io import StringIO

# Calculate project root: go up from test_single_matchup.py -> runtime_utt_change -> Running Simulations -> scripts -> project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai


def main():
    # Change these values to test different scenarios
    # AI Agents
    ai1_name = "lightRushAI"  
    opponent_ai_name = "coacAI"  # Same AI for mirror match
    
    # Initial UTTs (set at game start)
    initial_utt_p0 = "utts/TestUnitTypeTable.json"  # Default UTT for Player 0
    initial_utt_p1 = "utts/TestUnitTypeTable.json"  # Default UTT for Player 1
    # "utts/TestUnitTypeTable.json", "utts/CustomDemoUTT.json", "utts/FavorWorkerRushUTT.json", "utts/BetterCombatViewUTT.json"
    
    # Runtime UTT Change
    change_utt = True  # No runtime UTT change
    utt_change_step = 20  # Step at which to change UTT (only used if change_utt = True)
    new_utt_p0 = "utts/CustomDemoUTT.json"  # No runtime change
    new_utt_p1 = None  # No runtime change
    
    # Game settings
    max_steps = 1000  # Maximum number of steps for the game
    
    # Output saving
    save_output = True  # Set to True to save output to file
    output_dir = "scripts/Running Simulations/runtime_utt_change/outputs"  # Directory to save outputs
    
    # BetterCombatViewUTT features:
    # - Worker: harvestAmount=2 (harvests faster), HP=2 (not overpowered), Damage=1 (normal)
    # - Barracks: cost=3 (cheaper), produceTime=40 (faster to build)
    # - Light: cost=1, HP=15, Damage=6, produceTime=30 (quick to produce, balanced)
    # This allows combat units to be produced and visible before game ends
    
    
    ai1_func = getattr(microrts_ai, ai1_name, microrts_ai.lightRushAI)
    opponent_ai_func = getattr(microrts_ai, opponent_ai_name, microrts_ai.passiveAI)
    
    # Setup output file if saving enabled
    output_file = None
    if save_output:
        # Create output directory if it doesn't exist
        # project_root already points to project root, so just join the relative path
        if os.path.isabs(output_dir):
            output_path = output_dir
        else:
            output_path = os.path.join(project_root, output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Create filename with timestamp and matchup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"match_{ai1_name}_vs_{opponent_ai_name}_{timestamp}.txt"
        output_file = os.path.join(output_path, filename)
        print(f"\n💾 Output will be saved to: {output_file}\n")
    
    # Setup output buffering if saving enabled
    if save_output:
        output_buffer = StringIO()
        import builtins
        original_print = builtins.print
        
        def buffered_print(*args, **kwargs):
            """Print to both console and buffer"""
            # Use original print to avoid recursion
            original_print(*args, **kwargs)
            # Write to buffer separately, handling file parameter
            buffer_kwargs = {k: v for k, v in kwargs.items() if k != 'file'}
            buffer_kwargs['file'] = output_buffer
            original_print(*args, **buffer_kwargs)
        
        # Store original print and replace with buffered version
        builtins._original_print = original_print
        builtins.print = buffered_print
        # Store in globals for finally block
        globals()['output_file'] = output_file
        globals()['output_buffer'] = output_buffer
    
    # Make output_file available for finally block
    globals()['output_file'] = output_file
    if save_output:
        globals()['output_buffer'] = output_buffer
    
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
        },
        "utts/BetterCombatViewUTT.json": {
            "name": "BetterCombatViewUTT",
            "light": "Light: HP=15, Damage=6, Cost=1, Produce Time=30",
            "worker": "Worker: HP=2, Damage=1, Cost=1, Produce Time=40, Harvest=2",
            "barracks": "Barracks: Cost=3, Produce Time=40"
        }
    }
    
    # Add TestUnitTypeTable description
    utt_descriptions["utts/TestUnitTypeTable.json"] = {
        "name": "TestUnitTypeTable",
        "light": "Light: HP=4, Damage=2, Cost=2, Produce Time=65",
        "worker": "Worker: HP=1, Damage=1, Cost=1, Produce Time=50"
    }
    
    print("=" * 70)
    print("Runtime UTT Change: Single Matchup Test")
    print("=" * 70)
    print(f"\nMatch: {ai1_name} (P0) vs {opponent_ai_name} (P1)")
    
    # Show initial UTT configuration
    initial_desc_p0 = utt_descriptions.get(initial_utt_p0, {"name": "Custom"})
    initial_desc_p1 = utt_descriptions.get(initial_utt_p1, {"name": "Custom"})
    print(f"\nInitial UTTs:")
    print(f"  Player 0 ({ai1_name}): {initial_desc_p0.get('name', initial_utt_p0)}")
    print(f"  Player 1 ({opponent_ai_name}): {initial_desc_p1.get('name', initial_utt_p1)}")
    
    # Show runtime change if enabled
    if change_utt:
        print(f"\nRuntime UTT Change: YES (at step {utt_change_step})")
        if new_utt_p0:
            new_desc = utt_descriptions.get(new_utt_p0, {"name": new_utt_p0})
            print(f"  Player 0 ({ai1_name}) will change to: {new_desc['name']}")
            if "light" in ai1_name.lower() and "light" in new_desc:
                print(f"    {new_desc['light']}")
            elif "worker" in ai1_name.lower() and "worker" in new_desc:
                print(f"    {new_desc['worker']}")
        if new_utt_p1:
            new_desc = utt_descriptions.get(new_utt_p1, {"name": new_utt_p1})
            print(f"  Player 1 ({opponent_ai_name}) will change to: {new_desc['name']}")
        if not new_utt_p0 and not new_utt_p1:
            print("  (No runtime change configured)")
    else:
        print(f"\nRuntime UTT Change: NO")
        print("  Both players keep their initial UTTs throughout the game")
    
    print("=" * 70)
    
    env = MicroRTSBotVecEnv(
        ai1s=[ai1_func],
        ai2s=[opponent_ai_func],
        map_paths=["maps/8x8/basesWorkers8x8A.xml"],
        max_steps=max_steps,  # Use configured max_steps
        utt_json_p0=initial_utt_p0,  # Initial UTT for Player 0
        utt_json_p1=initial_utt_p1,  # Initial UTT for Player 1
        autobuild=False,
    )
    
    print("\n✓ Environment created")
    obs = env.reset()
    print("✓ Environment reset\n")
    
    utt_changed = False
    step = 0
    winner = None
    final_step = None
    
    while step < max_steps:
        actions = [0] * env.num_envs
        obs, rewards, dones, infos = env.step(actions)
        
        # Change UTT at specified step if requested
        if change_utt and step == utt_change_step and not utt_changed:
            print(f">>> Step {step}: Changing UTT configuration...")
            success = env.change_utt(
                utt_json_p0=new_utt_p0,  # New UTT for Player 0 (None to keep unchanged)
                utt_json_p1=new_utt_p1   # New UTT for Player 1 (None to keep unchanged)
            )
            if success:
                print("✓ UTT changed successfully!")
                if new_utt_p0:
                    new_desc = utt_descriptions.get(new_utt_p0, {"name": new_utt_p0})
                    print(f"  Player 0 ({ai1_name}) changed to: {new_desc['name']}")
                    if "light" in ai1_name.lower() and "light" in new_desc:
                        print(f"    {new_desc['light']}")
                    elif "worker" in ai1_name.lower() and "worker" in new_desc:
                        print(f"    {new_desc['worker']}")
                if new_utt_p1:
                    new_desc = utt_descriptions.get(new_utt_p1, {"name": new_utt_p1})
                    print(f"  Player 1 ({opponent_ai_name}) changed to: {new_desc['name']}")
                if not new_utt_p0 and not new_utt_p1:
                    print("  (No changes made)")
                print()
                utt_changed = True
                
                # Capture a snapshot right after UTT change to see current state
                mid_game_snapshot = get_game_snapshot(env, ai1_name, opponent_ai_name)
                print("  Snapshot at step", step, "(right after UTT change):")
                print(mid_game_snapshot)
                print()
            else:
                print("✗ Failed to change UTT")
        
        # Print progress and capture snapshots more frequently to catch combat units
        if step > 0 and step % 10 == 0:  # Check every 10 steps instead of 50
            # Verify UTT if runtime change was made
            try:
                if utt_changed and new_utt_p0:
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
                    
                    # Capture snapshots at key steps to catch combat units
                    if step in [30, 40, 50, 60, 70, 80, 90, 100]:
                        mid_game_snapshot = get_game_snapshot(env, ai1_name, opponent_ai_name)
                        print(f"\n  Snapshot at step {step}:")
                        print(mid_game_snapshot)
                        print()
                    else:
                        print(f"  Step {step}: Game running... (reward: {rewards[0]:.2f})")
            except:
                print(f"  Step {step}: Game running... (reward: {rewards[0]:.2f})")
        
        # Always capture snapshots at key steps to see combat units, even if UTT wasn't changed
        if not change_utt or not utt_changed:
            if step > 0 and step % 50 == 0:
                mid_game_snapshot = get_game_snapshot(env, ai1_name, opponent_ai_name)
                print(f"\n  Snapshot at step {step}:")
                print(mid_game_snapshot)
                print()
        
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
    
    # Capture game state snapshot before closing
    snapshot = get_game_snapshot(env, ai1_name, opponent_ai_name)
    
    env.close()
    
    # Results
    print("\n" + "=" * 70)
    print("MATCH RESULT")
    print("=" * 70)
    print(f"Winner: {winner}")
    print(f"Final Step: {final_step}")
    print(f"UTT Changed: {utt_changed if change_utt else 'N/A (control)'}")
    
    #if change_utt:
    #    if winner == f"{ai1_name} (Player 0)":
    #        print(f"\n✓ SUCCESS: {ai1_name} won with the favorable UTT!")
    #        print("  This confirms the runtime UTT change is working correctly.")
    #    elif winner == f"{opponent_ai_name} (Player 1)":
    #        print(f"\n⚠ {ai1_name} lost despite favorable UTT.")
    #        print("  Possible reasons:")
    #        print("    - Opponent is too strong")
    #        print("    - Map doesn't favor this strategy")
    #        print(f"    - UTT change happened too late (step {utt_change_step})")
    #        print("    - Early game momentum")
    #    else:
    #        print("\n→ Game ended in a draw.")
    #        print("  UTT change may have balanced the matchup, or game needs more steps.")
    
    # Display game snapshot
    if snapshot:
        print("\n" + "=" * 70)
        print("GAME STATE SNAPSHOT")
        print("=" * 70)
        print(snapshot)
        print("=" * 70)


def get_game_snapshot(env, ai1_name, opponent_ai_name):
    """
    Capture and return a formatted snapshot of the current game state.
    Shows units, buildings, their positions, HP, and stats to verify UTT changes.
    """
    try:
        # Access the game state from the bot client
        bot_client = env.vec_client.botClients[0]
        gs = bot_client.getGameState()
        
        if not gs:
            return "⚠ Unable to access game state"
        
        # Get physical game state
        pgs = gs.getPhysicalGameState()
        if not pgs:
            return "⚠ Unable to access physical game state"
        
        # Get all units (including neutral resources)
        units = pgs.getUnits()
        
        # Check for combat units and track production status
        combat_unit_names = ["Light", "Heavy", "Ranged"]
        combat_units_found = []
        barracks_found = []
        
        for i in range(units.size()):
            unit = units.get(i)
            unit_name = unit.getType().name
            player_id = unit.getPlayer()
            
            if unit_name in combat_unit_names and player_id >= 0:
                combat_units_found.append({
                    'name': unit_name,
                    'player': player_id,
                    'pos': (unit.getX(), unit.getY()),
                    'hp': unit.getHitPoints()
                })
            elif unit_name == "Barracks" and player_id >= 0:
                barracks_found.append({
                    'player': player_id,
                    'pos': (unit.getX(), unit.getY()),
                    'hp': unit.getHitPoints()
                })
        
        # Add combat unit summary to snapshot
        snapshot_summary = []
        if combat_units_found:
            for cu in combat_units_found:
                snapshot_summary.append(f"  ⚔️  {cu['name']} (P{cu['player']}) at {cu['pos']}, HP: {cu['hp']}")
        else:
            snapshot_summary.append("  ⚠️  NO COMBAT UNITS on field")
        
        if barracks_found:
            for b in barracks_found:
                snapshot_summary.append(f"  🏭 Barracks (P{b['player']}) at {b['pos']}, HP: {b['hp']}")
        
        # Get UTTs for both players
        utt_p0 = gs.getUnitTypeTableP0()
        utt_p1 = gs.getUnitTypeTableP1()
        
        # Organize units by player
        player0_units = []
        player1_units = []
        neutral_units = []
        
        for i in range(units.size()):
            unit = units.get(i)
            player_id = unit.getPlayer()
            unit_type = unit.getType()
            unit_name = unit_type.name
            hp = unit.getHitPoints()
            max_hp = unit_type.hp
            x = unit.getX()
            y = unit.getY()
            resources = unit.getResources()
            
            # Skip neutral resources for main display (we'll show a summary)
            if player_id < 0 or unit_type.isResource:
                neutral_units.append({
                    'name': unit_name,
                    'x': x,
                    'y': y,
                    'hp': hp
                })
                continue
            
            # Get stats from the player's UTT
            if player_id == 0:
                player_utt = utt_p0
                player_name = ai1_name
            elif player_id == 1:
                player_utt = utt_p1
                player_name = opponent_ai_name
            else:
                continue
            
            # Get unit type from player's UTT
            player_unit_type = player_utt.getUnitType(unit_name)
            if player_unit_type:
                min_damage = player_unit_type.minDamage
                max_damage = player_unit_type.maxDamage
                cost = player_unit_type.cost
            else:
                min_damage = unit_type.minDamage
                max_damage = unit_type.maxDamage
                cost = unit_type.cost
            
            # Format damage display
            if min_damage == max_damage:
                damage_str = str(min_damage)
            else:
                damage_str = f"{min_damage}-{max_damage}"
            
            unit_info = {
                'name': unit_name,
                'hp': hp,
                'max_hp': max_hp,
                'damage': damage_str,
                'cost': cost,
                'x': x,
                'y': y,
                'resources': resources,
                'is_building': not unit_type.canMove and not unit_type.isResource
            }
            
            if player_id == 0:
                player0_units.append(unit_info)
            elif player_id == 1:
                player1_units.append(unit_info)
        
        # Get resources for both players
        player0_obj = pgs.getPlayer(0)
        player1_obj = pgs.getPlayer(1)
        resources_p0 = player0_obj.getResources() if player0_obj else 0
        resources_p1 = player1_obj.getResources() if player1_obj else 0
        
        # Format output
        snapshot = []
        snapshot.append(f"\nTotal units on field: {units.size()} (including {len(neutral_units)} neutral resources)")
        
        # Add combat unit summary at the top
        if snapshot_summary:
            snapshot.append("\n" + "\n".join(snapshot_summary))
        
        snapshot.append(f"\nPlayer 0 ({ai1_name}):")
        snapshot.append(f"  Resources: {resources_p0}")
        snapshot.append(f"  Units & Buildings: {len(player0_units)}")
        
        if player0_units:
            # Group by unit type
            by_type = {}
            for u in player0_units:
                key = u['name']
                if key not in by_type:
                    by_type[key] = []
                by_type[key].append(u)
            
            for unit_type, unit_list in sorted(by_type.items()):
                building_str = " (Building)" if unit_list[0]['is_building'] else ""
                snapshot.append(f"  {unit_type}{building_str}: {len(unit_list)} units")
                for u in unit_list[:5]:  # Show first 5 of each type
                    snapshot.append(f"    - ID: {u['name']}, Pos: ({u['x']}, {u['y']}), "
                                  f"HP: {u['hp']}/{u['max_hp']}, Damage: {u['damage']}, Cost: {u['cost']}")
                    if u['resources'] > 0:
                        snapshot[-1] += f", Resources: {u['resources']}"
                if len(unit_list) > 5:
                    snapshot.append(f"    ... and {len(unit_list) - 5} more")
        else:
            snapshot.append("  No units remaining")
        
        snapshot.append(f"\nPlayer 1 ({opponent_ai_name}):")
        snapshot.append(f"  Resources: {resources_p1}")
        snapshot.append(f"  Units & Buildings: {len(player1_units)}")
        
        if player1_units:
            # Group by unit type
            by_type = {}
            for u in player1_units:
                key = u['name']
                if key not in by_type:
                    by_type[key] = []
                by_type[key].append(u)
            
            for unit_type, unit_list in sorted(by_type.items()):
                building_str = " (Building)" if unit_list[0]['is_building'] else ""
                snapshot.append(f"  {unit_type}{building_str}: {len(unit_list)} units")
                for u in unit_list[:5]:  # Show first 5 of each type
                    snapshot.append(f"    - ID: {u['name']}, Pos: ({u['x']}, {u['y']}), "
                                  f"HP: {u['hp']}/{u['max_hp']}, Damage: {u['damage']}, Cost: {u['cost']}")
                    if u['resources'] > 0:
                        snapshot[-1] += f", Resources: {u['resources']}"
                if len(unit_list) > 5:
                    snapshot.append(f"    ... and {len(unit_list) - 5} more")
        else:
            snapshot.append("  No units remaining")
        
        # Show summary of neutral resources if any
        if neutral_units:
            resource_count = sum(1 for u in neutral_units if u['name'] == 'Resource')
            if resource_count > 0:
                snapshot.append(f"\nNeutral Resources: {resource_count} resource deposits on map")
        
        # Show UTT verification stats
        snapshot.append("\nUTT Verification (from game state):")
        
        # Check Player 0 UTT stats
        utt_p0_types = utt_p0.getUnitTypes()
        light_p0 = None
        worker_p0 = None
        for i in range(utt_p0_types.size()):
            ut = utt_p0_types.get(i)
            if ut.name == "Light":
                light_p0 = ut
            elif ut.name == "Worker":
                worker_p0 = ut
        
        if light_p0:
            dmg_p0 = f"{light_p0.minDamage}-{light_p0.maxDamage}" if light_p0.minDamage != light_p0.maxDamage else str(light_p0.minDamage)
            snapshot.append(f"  P0 Light: HP={light_p0.hp}, Damage={dmg_p0}, Cost={light_p0.cost}")
        if worker_p0:
            dmg_p0_w = f"{worker_p0.minDamage}-{worker_p0.maxDamage}" if worker_p0.minDamage != worker_p0.maxDamage else str(worker_p0.minDamage)
            snapshot.append(f"  P0 Worker: HP={worker_p0.hp}, Damage={dmg_p0_w}, Cost={worker_p0.cost}")
        
        # Check Player 1 UTT stats
        utt_p1_types = utt_p1.getUnitTypes()
        light_p1 = None
        worker_p1 = None
        for i in range(utt_p1_types.size()):
            ut = utt_p1_types.get(i)
            if ut.name == "Light":
                light_p1 = ut
            elif ut.name == "Worker":
                worker_p1 = ut
        
        if light_p1:
            dmg_p1 = f"{light_p1.minDamage}-{light_p1.maxDamage}" if light_p1.minDamage != light_p1.maxDamage else str(light_p1.minDamage)
            snapshot.append(f"  P1 Light: HP={light_p1.hp}, Damage={dmg_p1}, Cost={light_p1.cost}")
        if worker_p1:
            dmg_p1_w = f"{worker_p1.minDamage}-{worker_p1.maxDamage}" if worker_p1.minDamage != worker_p1.maxDamage else str(worker_p1.minDamage)
            snapshot.append(f"  P1 Worker: HP={worker_p1.hp}, Damage={dmg_p1_w}, Cost={worker_p1.cost}")
        
        return "\n".join(snapshot)
        
    except Exception as e:
        return f"⚠ Error capturing snapshot: {str(e)}"


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Use original print for error messages
        import builtins
        if hasattr(builtins, '_original_print'):
            builtins._original_print(f"\n❌ Error: {e}")
        else:
            print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        import jpype
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
    finally:
        # Save output buffer to file if it exists
        if 'output_buffer' in globals() and 'output_file' in globals():
            output_file_path = globals().get('output_file')
            output_buffer_obj = globals().get('output_buffer')
            
            if output_file_path and output_buffer_obj:
                try:
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(output_buffer_obj.getvalue())
                    # Use original print
                    import builtins
                    original_print = getattr(builtins, '_original_print', print)
                    original_print(f"\n💾 Output saved to: {output_file_path}")
                except Exception as e:
                    import builtins
                    original_print = getattr(builtins, '_original_print', print)
                    original_print(f"\n⚠️  Failed to save output: {e}")
        
        # Restore original print if it was modified
        import builtins
        if hasattr(builtins, '_original_print'):
            builtins.print = builtins._original_print

