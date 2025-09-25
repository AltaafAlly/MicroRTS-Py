import os
import csv
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai


# Single-file launcher: set your match parameters here and run this file.


def run_pair(ai_left: str, ai_right: str, map_path: str, max_steps: int, games: int,
             autobuild: bool, utt_json: str | None,
             utt_json_p0: str | None = None, utt_json_p1: str | None = None) -> Dict[str, int]:
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
            inf = info[0] if isinstance(info, list) and info else info
            winner = "draw"
            if isinstance(inf, dict) and "raw_rewards" in inf:
                rr = inf["raw_rewards"]
                rr = rr.tolist() if hasattr(rr, "tolist") else rr
                if rr and rr[0] > 0:
                    winner = "left"
                elif rr and rr[0] < 0:
                    winner = "right"
            if winner == "left":
                results["left_wins"] += 1
            elif winner == "right":
                results["right_wins"] += 1
            else:
                results["draws"] += 1
            _ = env.reset()
            break

    # Important: do NOT call env.close() here because it shuts down the JVM,
    # and JPype cannot restart it in the same Python process. Instead, close
    # the underlying client only and keep the JVM alive for subsequent pairs.
    try:
        env.vec_client.close()
    except Exception:
        pass
    return results


def round_robin(ai_names: List[str], map_path: str, max_steps: int, games: int,
                autobuild: bool, utt_json: str | None, out_dir: str = "results",
                max_steps_long: int | None = None, draw_retry_threshold: float = 0.0,
                utt_json_p0: str | None = None, utt_json_p1: str | None = None) -> None:
    standings: Dict[str, Dict[str, int | float]] = {
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
        "games_per_pair": 5,            # games per pairing
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
        "POHeavyRush",
        "POLightRush",
        "PORangedRush",
        "POWorkerRush",
        "coacAI",
        "droplet",
        #"guidedRojoA3N", #this ai agent is causing problems 
        "izanagi",
        "lightRushAI",
        "mayari",
        "mixedBot",
        "naiveMCTSAI",
        "passiveAI",
        "randomAI",
        "randomBiasedAI",
        "rojo",
        "tiamat",
        "workerRushAI",
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


