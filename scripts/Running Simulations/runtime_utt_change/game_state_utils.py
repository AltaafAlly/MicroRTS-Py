"""
Shared utilities to read unit composition and game state from a MicroRTS env.
Used by runtime_utt_change outputs and by the GA match runner to capture
end-of-game unit composition (why a side won or lost).
"""

from typing import Dict, List, Any, Optional


def get_unit_composition_dict(env) -> Optional[Dict[str, Any]]:
    """
    Read current game state from env and return compact unit composition for left/right.
    Call this when a game has just ended (done=True), before env.reset().

    Returns:
        Dict with:
          - "left": {"Worker": 2, "Base": 1, "Light": 1, ...}  (player 0)
          - "right": {...}  (player 1)
          - "left_resources": int
          - "right_resources": int
        or None if state cannot be read.
    """
    try:
        # Prefer render_client (same as botClients[0]) for MicroRTSBotVecEnv
        bot_client = getattr(env, "render_client", None)
        vec_client = getattr(env, "vec_client", None)
        if bot_client is None and vec_client is not None:
            bc = getattr(vec_client, "botClients", None)
            if bc is not None and len(bc) > 0:
                bot_client = bc[0]
        if bot_client is None:
            import sys
            print("  [game_state_utils] bot_client is None (env has no render_client/vec_client.botClients[0])", file=sys.stderr)
            return None
        # After a game ends, Java reset() runs inside step(); use last terminal state for composition.
        # If the JAR was built with getLastTerminalGameState (see gym_microrts/microrts), use that;
        # otherwise fall back to getGameState() (may be post-reset initial state, not end-of-game).
        gs = None
        if vec_client is not None and hasattr(vec_client, "getLastTerminalGameState"):
            try:
                gs = vec_client.getLastTerminalGameState(0)
            except Exception:
                pass
        if gs is None and bot_client is not None:
            gs = bot_client.getGameState()
        if gs is None:
            import sys
            print("  [game_state_utils] getGameState() / getLastTerminalGameState(0) returned None", file=sys.stderr)
            return None
        pgs = gs.getPhysicalGameState()
        if pgs is None:
            return None
        units = pgs.getUnits()
        try:
            n_units = int(units.size()) if hasattr(units, "size") else len(units)
        except Exception:
            n_units = 0

        left_counts = {}
        right_counts = {}
        for i in range(n_units):
            try:
                unit = units.get(i)
                player_id = unit.getPlayer()
                pid = int(player_id) if player_id is not None else -1
                unit_type = unit.getType()
                unit_name = str(unit_type.name) if hasattr(unit_type, "name") else ""
                if getattr(unit_type, "isResource", False):
                    continue
                if pid == 0:
                    left_counts[unit_name] = left_counts.get(unit_name, 0) + 1
                elif pid == 1:
                    right_counts[unit_name] = right_counts.get(unit_name, 0) + 1
            except Exception:
                continue

        try:
            player0_obj = pgs.getPlayer(0)
            player1_obj = pgs.getPlayer(1)
            left_resources = int(player0_obj.getResources()) if player0_obj else 0
            right_resources = int(player1_obj.getResources()) if player1_obj else 0
        except Exception:
            left_resources = right_resources = 0

        return {
            "left": left_counts,
            "right": right_counts,
            "left_resources": left_resources,
            "right_resources": right_resources,
        }
    except Exception as e:
        import sys
        print(f"  [game_state_utils] get_unit_composition_dict failed: {e}", file=sys.stderr)
        return None


def composition_to_string(comp: Dict[str, int]) -> str:
    """Format unit composition dict as a short string, e.g. 'Worker:2,Base:1,Light:1'."""
    if not comp:
        return ""
    return ",".join(f"{k}:{v}" for k, v in sorted(comp.items()))


def get_game_snapshot_text(
    env,
    ai_left_name: str = "P0",
    ai_right_name: str = "P1",
) -> str:
    """
    Produce a human-readable snapshot of the current game state (like runtime_utt_change outputs).
    Uses the same structure as test_single_matchup.get_game_snapshot for consistency.
    """
    try:
        bot_client = env.vec_client.botClients[0]
        gs = bot_client.getGameState()
        if not gs:
            return "Unable to access game state"
        pgs = gs.getPhysicalGameState()
        if not pgs:
            return "Unable to access physical game state"
        units = pgs.getUnits()

        player0_units = []
        player1_units = []
        for i in range(units.size()):
            unit = units.get(i)
            player_id = unit.getPlayer()
            unit_type = unit.getType()
            unit_name = unit_type.name
            if player_id < 0 or unit_type.isResource:
                continue
            unit_info = {
                "name": unit_name,
                "hp": unit.getHitPoints(),
                "max_hp": unit_type.hp,
                "x": unit.getX(),
                "y": unit.getY(),
                "is_building": not unit_type.canMove and not unit_type.isResource,
            }
            if player_id == 0:
                player0_units.append(unit_info)
            elif player_id == 1:
                player1_units.append(unit_info)

        player0_obj = pgs.getPlayer(0)
        player1_obj = pgs.getPlayer(1)
        resources_p0 = player0_obj.getResources() if player0_obj else 0
        resources_p1 = player1_obj.getResources() if player1_obj else 0

        def format_side(unit_list: List[Dict], name: str, resources: int) -> List[str]:
            lines = [f"\n{name}:", f"  Resources: {resources}", f"  Units & Buildings: {len(unit_list)}"]
            if not unit_list:
                lines.append("  No units remaining")
                return lines
            by_type = {}
            for u in unit_list:
                key = u["name"]
                if key not in by_type:
                    by_type[key] = []
                by_type[key].append(u)
            for unit_type, ulist in sorted(by_type.items()):
                building_str = " (Building)" if ulist[0]["is_building"] else ""
                lines.append(f"  {unit_type}{building_str}: {len(ulist)} units")
                for u in ulist[:5]:
                    lines.append(f"    - {u['name']} Pos:({u['x']},{u['y']}) HP:{u['hp']}/{u['max_hp']}")
                if len(ulist) > 5:
                    lines.append(f"    ... and {len(ulist) - 5} more")
            return lines

        out = [f"Total units on field: {units.size()}"]
        out.extend(format_side(player0_units, f"Player 0 ({ai_left_name})", resources_p0))
        out.extend(format_side(player1_units, f"Player 1 ({ai_right_name})", resources_p1))
        return "\n".join(out)
    except Exception as e:
        return f"Error capturing snapshot: {e}"
