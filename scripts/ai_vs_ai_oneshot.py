#!/usr/bin/env python3
import json, argparse, os, re, xml.etree.ElementTree as ET
import numpy as np
from typing import Any, Dict, Tuple
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai

#once the JVM is shut down inside a Python process, it cannot be started again in the same process.
#run each series in its own Python process

def parse_map_hw(map_path: str) -> Tuple[int,int]:
    try:
        root = ET.parse(map_path).getroot()
        def grab(tag):
            n = root.find(tag) or root.find(f".//{tag}")
            if n is not None and n.text and n.text.strip().isdigit(): return int(n.text.strip())
            for el in root.iter():
                if tag in el.attrib and str(el.attrib[tag]).isdigit(): return int(el.attrib[tag])
            return None
        h, w = grab("height"), grab("width")
        if h and w: return h, w
    except Exception:
        pass
    m = re.search(r"(\d+)\s*x\s*(\d+)", os.path.basename(map_path))
    if not m: raise RuntimeError(f"Could not infer size from '{map_path}'")
    a, b = int(m.group(1)), int(m.group(2))
    return b, a  # (h,w)

def make_dummy_actions(num_envs: int, h: int, w: int):
    L = 7 * h * w
    return [[ [0]*L, [0]*L ] for _ in range(num_envs)]

def winner_from_info(info: Dict[str, Any], rewards):
    if isinstance(info, dict) and "raw_rewards" in info:
        rr = info["raw_rewards"]; rr = rr.tolist() if hasattr(rr, "tolist") else rr
        if isinstance(rr, (list, tuple)) and len(rr)>=1:
            if rr[0] > 0: return "player_0"
            if rr[0] < 0: return "player_1"
            return "draw"
    if isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards)>=1:
        if rewards[0] > 0: return "player_0"
        if rewards[0] < 0: return "player_1"
    return "draw"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai1", required=True)
    ap.add_argument("--ai2", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--autobuild", action="store_true", default=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    a1 = getattr(microrts_ai, args.ai1)
    a2 = getattr(microrts_ai, args.ai2)

    env = MicroRTSBotVecEnv(
        ai1s=[a1], ai2s=[a2],
        max_steps=args.max_steps,
        map_paths=[args.map],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=args.autobuild,
    )
    _ = env.reset()
    h, w = parse_map_hw(args.map)
    dummy_actions = make_dummy_actions(1, h, w)

    left_wins = right_wins = draws = timeouts = 0
    for g in range(args.games):
        steps = 0
        while True:
            ret = env.step(dummy_actions)
            if len(ret) == 5:
                obs, rewards, terminated, truncated, info = ret
                done = bool(terminated or truncated)
            else:
                obs, rewards, done, info = ret
                if isinstance(done, (list, tuple, np.ndarray)):
                    done = bool(done[0]) if len(done) else False
            steps += 1
            if done:
                inf = info[0] if isinstance(info, list) and info else info
                w = winner_from_info(inf, rewards)
                if w == "player_0": left_wins += 1
                elif w == "player_1": right_wins += 1
                else: draws += 1
                if steps >= args.max_steps: timeouts += 1
                _ = env.reset()
                break

    env.close()
    out = {
        "ai_left": args.ai1, "ai_right": args.ai2, "map": args.map,
        "left_wins": left_wins, "right_wins": right_wins, "draws": draws,
        "timeouts": timeouts, "games": args.games, "h": h, "w": w
    }
    s = json.dumps(out)
    print(s)
    if args.json:
        with open(args.json, "w") as f: f.write(s)

if __name__ == "__main__":
    main()
