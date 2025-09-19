import os
import csv
import argparse
import glob
from pathlib import Path
from typing import Dict, Tuple, List
import time

import numpy as np
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai


def discover_maps(maps_dir: str = "gym_microrts/microrts/maps") -> List[str]:
    """Discover all available maps in the maps directory."""
    maps = []
    if os.path.exists(maps_dir):
        # Look for .xml files in maps directory and subdirectories
        pattern = os.path.join(maps_dir, "**", "*.xml")
        full_paths = glob.glob(pattern, recursive=True)
        full_paths.sort()
        
        # Convert to relative paths that the environment expects
        for full_path in full_paths:
            # Convert gym_microrts/microrts/maps/8x8/file.xml to maps/8x8/file.xml
            if "gym_microrts/microrts/maps/" in full_path:
                relative_path = full_path.split("gym_microrts/microrts/maps/")[-1]
                maps.append(f"maps/{relative_path}")
            else:
                maps.append(full_path)
    return maps


def display_maps(maps: List[str]) -> None:
    """Display available maps in a nice format."""
    if not maps:
        print("❌ No maps found in gym_microrts/microrts/maps/ directory")
        return
    
    print("🗺️  Available Maps:")
    print("=" * 60)
    for i, map_path in enumerate(maps, 1):
        map_name = os.path.basename(map_path)
        # Extract the subdirectory after maps/
        map_dir = os.path.dirname(map_path)
        if "maps/" in map_dir:
            map_dir = map_dir.split("maps/")[-1]
        else:
            map_dir = "root"
        print(f"{i:2d}. {map_name:25s} ({map_dir})")
    print("=" * 60)


def select_map_interactive(maps: List[str]) -> str:
    """Allow user to interactively select a map."""
    if not maps:
        print("❌ No maps available!")
        return "maps/8x8/basesWorkers8x8A.xml"  # Default fallback
    
    display_maps(maps)
    
    while True:
        try:
            choice = input(f"\n🎮 Select map (1-{len(maps)}) or press Enter for default: ").strip()
            if not choice:
                return "maps/8x8/basesWorkers8x8A.xml"  # Default
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(maps):
                selected_map = maps[choice_num - 1]
                print(f"✅ Selected: {selected_map}")
                return selected_map
            else:
                print(f"❌ Please enter a number between 1 and {len(maps)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            return "maps/8x8/basesWorkers8x8A.xml"  # Default fallback


def run_pair_with_events(ai_left: str, ai_right: str, map_path: str, max_steps: int, games: int,
                        autobuild: bool, utt_json: str | None,
                        utt_json_p0: str | None = None, utt_json_p1: str | None = None,
                        verbose_events: bool = True) -> Tuple[Dict[str, int], List[Dict]]:
    """Run matches between two AIs with enhanced event tracking."""
    a1 = getattr(microrts_ai, ai_left)
    a2 = getattr(microrts_ai, ai_right)

    env = MicroRTSBotVecEnv(
        ai1s=[a1], ai2s=[a2],
        max_steps=max_steps,
        map_paths=[map_path],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=autobuild,
        utt_json=utt_json,
        utt_json_p0=utt_json_p0,
        utt_json_p1=utt_json_p1,
    )

    _ = env.reset()
    h, w = env.height, env.width
    L = 7 * h * w
    dummy_actions = [[[0] * L, [0] * L]]

    results = {"left_wins": 0, "right_wins": 0, "draws": 0}
    all_events = []
    
    for game_num in range(games):
        print(f"\n🎮 Game {game_num + 1}/{games}: {ai_left} vs {ai_right}")
        print("=" * 60)
        
        game_events = []
        steps = 0
        last_reward = 0
        
        while True:
            obs, rewards, done, info = env.step(dummy_actions)
            steps += 1
            
            # Track reward changes as a proxy for events
            current_reward = rewards[0] if len(rewards) > 0 else 0
            if current_reward != last_reward:
                event_type = "reward_change"
                if current_reward > last_reward:
                    event_desc = f"Positive reward change: {last_reward} → {current_reward}"
                else:
                    event_desc = f"Negative reward change: {last_reward} → {current_reward}"
                
                event = {
                    'game': game_num + 1,
                    'step': steps,
                    'event_type': event_type,
                    'description': event_desc,
                    'reward': current_reward,
                    'ai_left': ai_left,
                    'ai_right': ai_right,
                    'timestamp': time.time()
                }
                game_events.append(event)
                
                if verbose_events:
                    print(f"  [Step {steps}] {event_desc}")
                
                last_reward = current_reward
            
            # Print progress every 500 steps
            if steps % 500 == 0:
                print(f"  Step {steps}: Reward = {current_reward}")
            
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
            
            # Add game end event
            game_end_event = {
                'game': game_num + 1,
                'step': steps,
                'event_type': 'game_end',
                'description': f"Game ended - Winner: {winner}",
                'winner': winner,
                'final_reward': current_reward,
                'ai_left': ai_left,
                'ai_right': ai_right,
                'timestamp': time.time()
            }
            game_events.append(game_end_event)
            
            if winner == "left":
                results["left_wins"] += 1
            elif winner == "right":
                results["right_wins"] += 1
            else:
                results["draws"] += 1
            
            print(f"  🏆 Winner: {winner} (Steps: {steps}, Final Reward: {current_reward})")
            print(f"  📊 Events in this game: {len(game_events)}")
            
            all_events.extend(game_events)
            _ = env.reset()
            break

    # Important: do NOT call env.close() here because it shuts down the JVM,
    # and JPype cannot restart it in the same Python process. Instead, close
    # the underlying client only and keep the JVM alive for subsequent pairs.
    try:
        env.vec_client.close()
    except Exception:
        pass
    
    return results, all_events


def round_robin_with_events(ai_names: List[str], map_path: str, max_steps: int, games: int,
                           autobuild: bool, utt_json: str | None, out_dir: str = "results",
                           max_steps_long: int | None = None, draw_retry_threshold: float = 0.0,
                           utt_json_p0: str | None = None, utt_json_p1: str | None = None,
                           verbose_events: bool = True) -> None:
    """Run round-robin tournament with event tracking."""
    standings: Dict[str, Dict[str, int | float]] = {
        name: {"wins": 0, "losses": 0, "draws": 0, "points": 0.0} for name in ai_names
    }
    pair_rows: List[Dict[str, int | str]] = []
    all_events = []
    
    print(f"🏆 Starting Round-Robin Tournament with Event Tracking")
    print(f"🤖 AIs: {', '.join(ai_names)}")
    print(f"🎮 Games per pair: {games}")
    print(f"🗺️  Map: {map_path}")
    print("=" * 80)
    
    for i in range(len(ai_names)):
        for j in range(i + 1, len(ai_names)):
            left, right = ai_names[i], ai_names[j]
            print(f"\n⚔️  MATCH: {left} vs {right}")
            print("=" * 60)
            
            res, events = run_pair_with_events(left, right, map_path, max_steps, games, 
                                             autobuild, utt_json, utt_json_p0, utt_json_p1, verbose_events)
            lw, rw, d = res["left_wins"], res["right_wins"], res["draws"]
            print(f"\n📈 {left} vs {right}: {lw}-{rw} (draws {d}) over {games} games")
            print(f"📊 Total events tracked: {len(events)}")
            
            # Optional retry with longer horizon if draws dominate
            if max_steps_long and draw_retry_threshold > 0:
                if d / max(1, games) >= draw_retry_threshold:
                    print(f"🔄 High draw rate (>{draw_retry_threshold:.0%}). Retrying with longer time limit...")
                    res, retry_events = run_pair_with_events(left, right, map_path, max_steps_long, games, 
                                                           autobuild, utt_json, utt_json_p0, utt_json_p1, verbose_events)
                    lw, rw, d = res["left_wins"], res["right_wins"], res["draws"]
                    events.extend(retry_events)
                    print(f"🔄 Retry result {left} vs {right}: {lw}-{rw} (draws {d})")
            
            # Update standings
            standings[left]["wins"] += lw
            standings[left]["losses"] += rw
            standings[left]["draws"] += d
            standings[right]["wins"] += rw
            standings[right]["losses"] += lw
            standings[right]["draws"] += d
            
            # Update points (wins + 0.5*draws)
            standings[left]["points"] = standings[left]["wins"] + 0.5 * standings[left]["draws"]
            standings[right]["points"] = standings[right]["wins"] + 0.5 * standings[right]["draws"]
            
            pair_rows.append({
                "left": left,
                "right": right,
                "left_wins": lw,
                "right_wins": rw,
                "draws": d,
                "games": games,
                "map_path": map_path,
                "utt_json": utt_json or "default",
                "utt_json_p0": utt_json_p0 or "",
                "utt_json_p1": utt_json_p1 or "",
                "max_steps": max_steps_long if (max_steps_long and draw_retry_threshold > 0 and d / max(1, games) >= draw_retry_threshold) else max_steps,
            })
            
            # Add events to global list
            all_events.extend(events)

    # Print summary sorted by points
    ordered = sorted(standings.items(), key=lambda kv: kv[1]["points"], reverse=True)
    print(f"\n🏆 FINAL STANDINGS (points = wins + 0.5*draws)")
    print("=" * 80)
    for i, (name, rec) in enumerate(ordered, 1):
        print(f"{i:2d}. {name:15s}: {rec['points']:4.1f} pts | W{rec['wins']:2d} L{rec['losses']:2d} D{rec['draws']:2d}")

    # Write CSV outputs (overwrite on every run)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pairs_csv = out_path / "round_robin_pairs.csv"
    with pairs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "left", "right", "left_wins", "right_wins", "draws", "games", "map_path", "utt_json", "utt_json_p0", "utt_json_p1", "max_steps"
        ])
        writer.writeheader()
        writer.writerows(pair_rows)

    standings_csv = out_path / "round_robin_standings.csv"
    with standings_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ai", "points", "wins", "losses", "draws"])
        for name, rec in ordered:
            writer.writerow([name, rec["points"], rec["wins"], rec["losses"], rec["draws"]])
    
        # Write events CSV
        if all_events:
            events_csv = out_path / "match_events.csv"
            with events_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "game", "step", "event_type", "description", "reward", "final_reward", "winner", "ai_left", "ai_right", "timestamp"
                ])
                writer.writeheader()
                writer.writerows(all_events)
        print(f"\n📁 Results saved to:")
        print(f"   - Pairs: {pairs_csv}")
        print(f"   - Standings: {standings_csv}")
        print(f"   - Events: {events_csv}")
        print(f"📊 Total events tracked: {len(all_events)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MicroRTS Tournament with Event Tracking")
    
    # Map selection
    parser.add_argument("--map", "-m", type=str, help="Map file path (e.g., maps/8x8/basesWorkers8x8A.xml)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive map selection")
    parser.add_argument("--list-maps", "-l", action="store_true", help="List available maps and exit")
    
    # Tournament settings
    parser.add_argument("--games", "-g", type=int, default=2, help="Games per pair (default: 2)")
    parser.add_argument("--max-steps", type=int, default=4000, help="Maximum steps per game (default: 4000)")
    parser.add_argument("--ais", nargs="+", help="AI names to include (e.g., coacAI droplet randomAI)")
    
    # Event tracking
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose event output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (no event output)")
    
    # UTT settings
    parser.add_argument("--utt-p0", type=str, default="utts/CustomDemoUTT.json", help="Player 0 UTT file")
    parser.add_argument("--utt-p1", type=str, default="utts/AsymmetricP1UTT.json", help="Player 1 UTT file")
    
    return parser.parse_args()


def main():
    """Main function with event tracking enabled."""
    args = parse_arguments()
    
    # Discover available maps
    available_maps = discover_maps()
    
    # Handle map selection
    if args.list_maps:
        display_maps(available_maps)
        return
    
    if args.interactive:
        map_path = select_map_interactive(available_maps)
    elif args.map:
        map_path = args.map
        # Check if it's a relative path and convert to full path for existence check
        if map_path.startswith("maps/"):
            full_path = f"gym_microrts/microrts/{map_path}"
            if not os.path.exists(full_path):
                print(f"❌ Map file not found: {map_path}")
                print("Available maps:")
                display_maps(available_maps)
                return
        elif not os.path.exists(map_path):
            print(f"❌ Map file not found: {map_path}")
            print("Available maps:")
            display_maps(available_maps)
            return
    else:
        map_path = "maps/8x8/basesWorkers8x8A.xml"  # Default
    
    # Configure the tournament
    config = {
        "map_path": map_path,
        "max_steps": args.max_steps,
        "max_steps_long": args.max_steps * 3,  # 3x longer for retries
        "draw_retry_threshold": 0.6,
        "games_per_pair": args.games,
        "autobuild": False,
        "utt_json": None,
        "utt_json_p0": args.utt_p0,
        "utt_json_p1": args.utt_p1,
        "verbose_events": args.verbose and not args.quiet,
    }

    # AI selection
    if args.ais:
        requested_ai_names = args.ais
    else:
        requested_ai_names = [
            "coacAI",
            "droplet", 
            "randomAI",
        ]
    
    # Keep only those available in this install
    ai_names = [n for n in requested_ai_names if hasattr(microrts_ai, n)]
    
    if not ai_names:
        print("❌ No valid AI names found!")
        print("Available AIs:", [name for name in dir(microrts_ai) if not name.startswith('_')])
        return

    print("🎮 MicroRTS Tournament with Event Tracking")
    print("=" * 80)
    print(f"🤖 AIs: {', '.join(ai_names)}")
    print(f"📊 Event tracking: {'ENABLED' if config['verbose_events'] else 'DISABLED'}")
    print(f"🗺️  Map: {config['map_path']}")
    print(f"🎮 Games per pair: {config['games_per_pair']}")
    print(f"⏱️  Max steps: {config['max_steps']}")
    print(f"🎯 UTT P0: {config['utt_json_p0']}")
    print(f"🎯 UTT P1: {config['utt_json_p1']}")
    print("=" * 80)
    
    round_robin_with_events(
        ai_names=ai_names,
        map_path=config["map_path"],
        max_steps=config["max_steps"],
        games=config["games_per_pair"],
        autobuild=config["autobuild"],
        utt_json=config["utt_json"],
        out_dir="results",
        max_steps_long=config.get("max_steps_long"),
        draw_retry_threshold=config.get("draw_retry_threshold", 0.0),
        utt_json_p0=config["utt_json_p0"],
        utt_json_p1=config["utt_json_p1"],
        verbose_events=config["verbose_events"],
    )


if __name__ == "__main__":
    main()
