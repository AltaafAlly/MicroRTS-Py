import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai


# Single-file launcher: set your match parameters here and run this file.


def resolve_winner_side_p0_p1(env: MicroRTSBotVecEnv, info0: Optional[dict]) -> str:
    """
    Map episode to ``left`` (player 0) / ``right`` (player 1) / ``draw``.

    Prefer ``GameState.winner()`` from the JNI bot client (``getLastTerminalGameState``), which is
    the same authority the MicroRTS GUI uses. When the episode ends on the **step cap** without
    ``gameover()``, fall back to WinLoss ``raw_rewards[0]`` sign (empirically: ``< 0`` → P0/left,
    ``> 0`` → P1/right in this bridge).
    """
    try:
        vc = getattr(env, "vec_client", None)
        if vc is not None and hasattr(vc, "getLastTerminalGameState"):
            gs = vc.getLastTerminalGameState(0)
            if gs is not None and bool(gs.gameover()):
                w = int(gs.winner())
                if w == 0:
                    return "left"
                if w == 1:
                    return "right"
                return "draw"
    except Exception:
        pass
    if isinstance(info0, dict) and "raw_rewards" in info0:
        rr = info0["raw_rewards"]
        rr = rr.tolist() if hasattr(rr, "tolist") else list(rr)
        if rr:
            v = float(rr[0])
            if v < 0:
                return "left"
            if v > 0:
                return "right"
    return "draw"


def run_games_with_env(env: MicroRTSBotVecEnv, games: int) -> Dict[str, int]:
    """
    Run N games with an existing MicroRTSBotVecEnv (same format as run_pair result).
    Does not create or close the env. Use this to reuse one env across many UTT evaluations.
    """
    h, w = env.height, env.width
    L = 7 * h * w
    dummy_actions = [[[0] * L, [0] * L]]
    results = {"left_wins": 0, "right_wins": 0, "draws": 0}
    total_steps_this_run = 0
    _ = env.reset()
    for _g in range(games):
        steps = 0
        while True:
            obs, rewards, done, info = env.step(dummy_actions)
            steps += 1
            if isinstance(done, (list, tuple, np.ndarray)):
                done_flag = bool(done[0]) if len(done) else False
            else:
                done_flag = bool(done)
            if not done_flag:
                continue
            total_steps_this_run += steps
            inf = info[0] if isinstance(info, list) and info else info
            winner = resolve_winner_side_p0_p1(env, inf if isinstance(inf, dict) else None)
            if winner == "left":
                results["left_wins"] += 1
            elif winner == "right":
                results["right_wins"] += 1
            else:
                results["draws"] += 1
            _ = env.reset()
            break
    results["_total_steps"] = total_steps_this_run
    return results


def run_pair(ai_left: str, ai_right: str, map_path: str, max_steps: int, games: int,
             autobuild: bool, utt_json: Optional[str],
             utt_json_p0: Optional[str] = None, utt_json_p1: Optional[str] = None,
             capture_composition: bool = False,
             capture_snapshots: bool = False,
             snapshot_interval: int = 50,
             verbose_games: bool = False,
             utt_builtin: Optional[Tuple[int, int]] = None,
             utt_json_overlay_builtin: Optional[Tuple[int, int]] = None) -> Dict:
    """
    Run games between ai_left (P0) and ai_right (P1).
    If capture_composition is True, end-of-game unit composition is captured and returned.
    If capture_snapshots is True, game-state snapshots (last game only) are captured every snapshot_interval steps.
    If verbose_games is True, prints each game's winner and step count, and sets results[\"per_game\"].
    If utt_builtin is (version, move_conflict), both players use that Java ``UnitTypeTable`` (e.g. ``(3, 1)``
    matches the MicroRTS FE preset **Nondeterministic-Both**); JSON paths are ignored when this is set.
    If utt_json_overlay_builtin is set (and utt_builtin is not), JSON unit stats are applied on top of that Java
    table (e.g. ``(3, 1)`` for GA / headless parity with the FE while still loading evolved ``utts/*.json``).
    """
    # Optional: import game_state_utils for composition and/or snapshots
    get_unit_composition_dict = None
    composition_to_string = None
    get_game_snapshot_text = None
    if capture_composition or capture_snapshots:
        _last_import_err = None
        # 1) Try using existing sys.path first (GA evaluator inserts runtime_utt_change)
        try:
            from game_state_utils import get_unit_composition_dict as _guc, composition_to_string as _cts
            get_unit_composition_dict = _guc
            composition_to_string = _cts
            try:
                from game_state_utils import get_game_snapshot_text as _gst
                get_game_snapshot_text = _gst
            except ImportError:
                pass
        except Exception as _e:
            _last_import_err = _e
        # 2) If that failed, try path candidates (run from any cwd)
        if get_unit_composition_dict is None:
            _run_match_dir = os.path.dirname(os.path.abspath(__file__))
            _scripts_dir = os.path.dirname(_run_match_dir)
            _project_root = os.path.dirname(_scripts_dir) if _scripts_dir else os.getcwd()
            for _candidate in [
                os.path.join(_run_match_dir, "runtime_utt_change"),
                os.path.join(_project_root, "scripts", "Running Simulations", "runtime_utt_change"),
                os.path.join(os.getcwd(), "scripts", "Running Simulations", "runtime_utt_change"),
                os.path.join(os.getcwd(), "runtime_utt_change"),
            ]:
                if not os.path.isdir(_candidate):
                    continue
                try:
                    if _candidate not in sys.path:
                        sys.path.insert(0, _candidate)
                    from game_state_utils import get_unit_composition_dict as _guc, composition_to_string as _cts
                    get_unit_composition_dict = _guc
                    composition_to_string = _cts
                    try:
                        from game_state_utils import get_game_snapshot_text as _gst
                        get_game_snapshot_text = _gst
                    except ImportError:
                        pass
                    break
                except Exception as _e:
                    _last_import_err = _e
                    continue
        if get_unit_composition_dict is None:
            capture_composition = False
        if get_game_snapshot_text is None:
            capture_snapshots = False
        if (capture_composition or capture_snapshots) and get_unit_composition_dict is None and get_game_snapshot_text is None:
            print(f"  [run_pair] Could not import game_state_utils: {_last_import_err}", file=sys.stderr)

    a1 = getattr(microrts_ai, ai_left)
    a2 = getattr(microrts_ai, ai_right)

    jp0, jp1 = utt_json_p0, utt_json_p1
    if utt_builtin is not None:
        jp0, jp1 = None, None

    overlay = utt_json_overlay_builtin if utt_builtin is None else None

    env = MicroRTSBotVecEnv(
        ai1s=[a1], ai2s=[a2],
        max_steps=max_steps,
        map_paths=[map_path],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=autobuild,
        utt_json=utt_json,
        utt_json_p0=jp0,
        utt_json_p1=jp1,
        utt_builtin=utt_builtin,
        utt_json_overlay_builtin=overlay,
    )

    _ = env.reset()
    h, w = env.height, env.width
    L = 7 * h * w
    dummy_actions = [[[0] * L, [0] * L]]

    results = {"left_wins": 0, "right_wins": 0, "draws": 0}
    total_steps_this_run = 0
    per_game_compositions = []
    per_game_rows: List[Dict[str, Union[int, str]]] = []
    game_snapshots = []  # list of per-game [(step, text), ...] when capture_snapshots (every game)
    for _g in range(games):
        steps = 0
        this_game_snapshots = []  # (step, text) for this game only

        def _snapshot(step_num: int, at_end: bool = False) -> None:
            if not capture_snapshots:
                return
            try:
                if get_game_snapshot_text:
                    text = get_game_snapshot_text(env, ai_left, ai_right)
                    this_game_snapshots.append((step_num, text))
                else:
                    this_game_snapshots.append((step_num, "(game_state_utils not loaded)"))
            except Exception as e:
                this_game_snapshots.append((step_num, f"[Snapshot unavailable: {e}]"))

        if capture_snapshots:
            _snapshot(0)
        while True:
            obs, rewards, done, info = env.step(dummy_actions)
            steps += 1
            if capture_snapshots and steps <= max_steps:
                if steps % snapshot_interval == 0 or steps == 1:
                    _snapshot(steps)
            if isinstance(done, (list, tuple, np.ndarray)):
                done_flag = bool(done[0]) if len(done) else False
            else:
                done_flag = bool(done)
            if not done_flag:
                continue
            total_steps_this_run += steps
            if capture_snapshots and (not this_game_snapshots or this_game_snapshots[-1][0] != steps):
                _snapshot(steps, at_end=True)
            if capture_snapshots:
                game_snapshots.append(this_game_snapshots)
            inf = info[0] if isinstance(info, list) and info else info
            winner = resolve_winner_side_p0_p1(env, inf if isinstance(inf, dict) else None)
            if winner == "left":
                results["left_wins"] += 1
            elif winner == "right":
                results["right_wins"] += 1
            else:
                results["draws"] += 1

            if verbose_games:
                if winner == "left":
                    w_ai = ai_left
                elif winner == "right":
                    w_ai = ai_right
                else:
                    w_ai = "draw"
                per_game_rows.append(
                    {
                        "game_index": _g,
                        "winner_side": winner,
                        "winner_ai": w_ai,
                        "steps": steps,
                    }
                )
                _map_tag = map_path.replace("\\", "/").split("/")[-1] if map_path else "?"
                print(
                    f"        [{_map_tag}] game {_g + 1}/{games}: winner={w_ai}  "
                    f"(P0/left={ai_left}, P1/right={ai_right})  steps={steps}",
                    flush=True,
                )

            # Capture end-of-game unit composition (before reset) when requested
            if capture_composition and get_unit_composition_dict and composition_to_string:
                comp = get_unit_composition_dict(env)
                if comp:
                    comp["winner"] = winner
                    comp["game_index"] = _g
                    per_game_compositions.append(comp)

            _ = env.reset()
            break
    results["_game_snapshots"] = game_snapshots

    # Sanity check: if total_steps is 0, games did not run (possible reuse/cache bug)
    if total_steps_this_run == 0:
        print("        WARNING: total steps this run = 0 (games may not have run)", file=sys.stderr)
    results["_total_steps"] = total_steps_this_run

    lw_, rw_, d_ = results.get("left_wins", 0), results.get("right_wins", 0), results.get("draws", 0)
    if games > 0 and d_ == games and lw_ == 0 and rw_ == 0:
        if total_steps_this_run >= games * max_steps - 1:
            print(
                "        NOTE: All games logged as draws and total steps ≈ games×max_steps — "
                f"each episode likely **hit the step cap** ({max_steps}/game) before natural elimination. "
                "Raise SINGLE_GENE_MAX_STEPS (run_single_gene uses 300000 default for BroodWar; try 400000+ if still all draws), "
                "or reduce SINGLE_GENE_DEC_GAMES while tuning.",
                flush=True,
            )

    if capture_composition:
        if per_game_compositions and composition_to_string:
            results["_per_game_compositions"] = per_game_compositions
            last = per_game_compositions[-1]
            results["_left_unit_composition"] = composition_to_string(last.get("left", {}))
            results["_right_unit_composition"] = composition_to_string(last.get("right", {}))
        else:
            results["_left_unit_composition"] = "N/A"
            results["_right_unit_composition"] = "N/A"
            print("  (Unit composition N/A: rebuild gym_microrts/microrts JAR for end-of-game capture)", file=sys.stderr)

    if verbose_games and per_game_rows:
        results["per_game"] = per_game_rows

    try:
        env.vec_client.close()
    except Exception:
        pass
    return results


def round_robin(ai_names: List[str], map_path: str, max_steps: int, games: int,
                autobuild: bool, utt_json: Optional[str], out_dir: str = "results",
                max_steps_long: Optional[int] = None, draw_retry_threshold: float = 0.0,
                utt_json_p0: Optional[str] = None, utt_json_p1: Optional[str] = None) -> None:
    standings: Dict[str, Dict[str, Union[int, float]]] = {
        name: {"wins": 0, "losses": 0, "draws": 0, "points": 0.0} for name in ai_names
    }
    pair_rows: List[Dict[str, int | str]] = []
    for i in range(len(ai_names)):
        for j in range(i + 1, len(ai_names)):
            left, right = ai_names[i], ai_names[j]
            res = run_pair(left, right, map_path, max_steps, games, autobuild, utt_json, utt_json_p0, utt_json_p1)
            lw, rw, d = res["left_wins"], res["right_wins"], res["draws"]
            print(f"{left} vs {right}: {lw}-{rw} (draws {d}) over {games} games")
            # Optional retry with longer horizon if draws dominate
            if max_steps_long and draw_retry_threshold > 0:
                if d / max(1, games) >= draw_retry_threshold:
                    print(f"High draw rate (>{draw_retry_threshold:.0%}). Retrying {left} vs {right} with max_steps={max_steps_long}...")
                    res = run_pair(left, right, map_path, max_steps_long, games, autobuild, utt_json, utt_json_p0, utt_json_p1)
                    lw, rw, d = res["left_wins"], res["right_wins"], res["draws"]
                    print(f"Retry result {left} vs {right}: {lw}-{rw} (draws {d})")
            
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

    # Print summary sorted by points
    ordered = sorted(standings.items(), key=lambda kv: kv[1]["points"], reverse=True)
    print("\nRound-robin standings (points = wins + 0.5*draws):")
    for name, rec in ordered:
        print(f"- {name}: {rec['points']} pts | W{rec['wins']} L{rec['losses']} D{rec['draws']}")

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


def main():
    # Configure the tournament and environment once here
    config = {
        "map_path": "maps/8x8/basesWorkers8x8A.xml",
        "max_steps": 4000,
        # If many games are draws at max_steps, retry that pairing once with this longer horizon:
        "max_steps_long": 12000,
        # Trigger retry when draw ratio >= threshold (0.0 disables retry)
        "draw_retry_threshold": 0.6,
        "games_per_pair": 1,            # games per pairing
        "autobuild": False,
        "utt_json": None,  # Not used when utt_json_p0/p1 are specified
        # Asymmetric UTT support - now using custom UTTs!
        "utt_json_p0": "utts/CustomDemoUTT.json",  # Player 0 UTT
        "utt_json_p1": "utts/AsymmetricP1UTT.json",  # Player 1 UTT
    }

    # Which AIs to include in the round-robin:
    # Use the curated list from microrts_ai.ALL_AIS
    # Full set requested
    requested_ai_names = [
        #"POHeavyRush",
        #"POLightRush",
        #"PORangedRush",
        #"POWorkerRush",
        "coacAI",
        "droplet",
        #"guidedRojoA3N", #this ai agent is causing problems 
        #"izanagi",
        #"lightRushAI",
        #"mayari",
        #"mixedBot",
        #"naiveMCTSAI",
        #"passiveAI",
        #"randomAI",
        #"randomBiasedAI",
        #"rojo",
        #"tiamat",
        #"workerRushAI",
    ]
    # Keep only those available in this install
    ai_names = [n for n in requested_ai_names if hasattr(microrts_ai, n)]

    print(f"AIs: {', '.join(ai_names)}")
    round_robin(
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
    )


if __name__ == "__main__":
    main()


