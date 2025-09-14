#!/usr/bin/env python3
import os, re, math, argparse, xml.etree.ElementTree as ET
from typing import Any, Dict, Tuple
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
import random

# ---------- utilities ----------
def parse_map_hw(map_path: str) -> Tuple[int, int]:
    try:
        tree = ET.parse(map_path)
        root = tree.getroot()
        def grab(tag):
            n = root.find(tag) or root.find(f".//{tag}")
            if n is not None and n.text and n.text.strip().isdigit():
                return int(n.text.strip())
            for el in root.iter():
                if tag in el.attrib and str(el.attrib[tag]).isdigit():
                    return int(el.attrib[tag])
            return None
        h, w = grab("height"), grab("width")
        if h and w: return h, w
    except Exception:
        pass
    m = re.search(r"(\d+)\s*x\s*(\d+)", os.path.basename(map_path))
    if not m:
        raise RuntimeError(f"Could not infer map size from '{map_path}'")
    a, b = int(m.group(1)), int(m.group(2))
    return b, a  # (h, w)

def make_dummy_actions(num_envs: int, h: int, w: int):
    L = 7 * h * w
    return [[ [0]*L, [0]*L ] for _ in range(num_envs)], L

def winner_from_info(info: Dict[str, Any], rewards) -> str:
    if isinstance(info, dict) and "raw_rewards" in info:
        rr = info["raw_rewards"]
        rr = rr.tolist() if hasattr(rr, "tolist") else rr
        if isinstance(rr, (list, tuple)) and len(rr) >= 1:
            if rr[0] > 0:  return "player_0"
            if rr[0] < 0:  return "player_1"
            return "draw"
    if isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards) >= 1:
        if rewards[0] > 0: return "player_0"
        if rewards[0] < 0: return "player_1"
    return "draw"

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z*z/(2*n)
    adj = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return p, max(0.0, lower), min(1.0, upper)

def binomial_p_value(k: int, n: int, p0: float = 0.5) -> float:
    # two-sided exact (Clopper-Pearson style: use tail doubling)
    from math import comb
    pmf = lambda i: comb(n, i) * (p0**i) * ((1-p0)**(n-i))
    # two-sided: sum probabilities <= observed tail prob
    obs_prob = pmf(k)
    return sum(pmf(i) for i in range(n+1) if pmf(i) <= obs_prob)

# ---------- core run ----------
def run_series(ai_left: str, ai_right: str, map_path: str, max_steps: int, games: int,
               autobuild: bool = True, seed: int = 123) -> Dict[str, int]:
    random.seed(seed); np.random.seed(seed)
    aL = getattr(microrts_ai, ai_left)
    aR = getattr(microrts_ai, ai_right)

    env = MicroRTSBotVecEnv(
        ai1s=[aL], ai2s=[aR],
        max_steps=max_steps,
        map_paths=[map_path],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=autobuild,
    )
    _ = env.reset()
    h, w = parse_map_hw(map_path)
    dummy_actions, L = make_dummy_actions(1, h, w)

    wins_L = wins_R = draws = 0
    timeouts = 0
    for g in range(1, games+1):
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
                wnr = winner_from_info(inf, rewards)
                if wnr == "player_0": wins_L += 1
                elif wnr == "player_1": wins_R += 1
                else: draws += 1
                if steps >= max_steps: timeouts += 1
                _ = env.reset()
                break
    env.close()
    return {"left_wins": wins_L, "right_wins": wins_R, "draws": draws, "timeouts": timeouts, "L": L, "h": h, "w": w}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai1", default="coacAI", help="Left player")
    ap.add_argument("--ai2", default="lightRushAI", help="Right player")
    ap.add_argument("--map", default="maps/8x8/basesWorkers8x8A.xml")
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--autobuild", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--determinism-check", action="store_true")
    args = ap.parse_args()

    # A vs B
    res1 = run_series(args.ai1, args.ai2, args.map, args.max_steps, args.games, args.autobuild, args.seed)
    # B vs A (swap sides)
    res2 = run_series(args.ai2, args.ai1, args.map, args.max_steps, args.games, args.autobuild, args.seed)

    def summarize(tag, res):
        total = res["left_wins"] + res["right_wins"] + res["draws"]
        dec = res["left_wins"] + res["right_wins"]
        p, lo, hi = wilson_interval(res["left_wins"], max(1, dec))
        print(f"\n[{tag}] h={res['h']} w={res['w']} L={res['L']}")
        print(f"  left_wins={res['left_wins']} right_wins={res['right_wins']} draws={res['draws']} timeouts={res['timeouts']}")
        print(f"  decisive={dec}/{total}  non-draw winrate_left={p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        if dec > 0:
            pval = binomial_p_value(res["left_wins"], dec, p0=0.5)
            print(f"  binomial two-sided p={pval:.4f} (H0: 50/50)")

    summarize(f"{args.ai1} (left) vs {args.ai2} (right)", res1)
    summarize(f"{args.ai2} (left) vs {args.ai1} (right)", res2)

    # Side-bias check: combine both orientations
    A_as_left = res1["left_wins"]
    A_as_right = res2["right_wins"]  # when swapped, right_wins belong to original A
    draws_total = res1["draws"] + res2["draws"]
    dec_total = (res1["left_wins"] + res1["right_wins"] +
                 res2["left_wins"] + res2["right_wins"])
    A_total = A_as_left + A_as_right
    pA, loA, hiA = wilson_interval(A_total, max(1, dec_total))
    side_diff = abs((res1["left_wins"] / max(1, (res1["left_wins"] + res1["right_wins"]))) -
                    (res2["right_wins"] / max(1, (res2["left_wins"] + res2["right_wins"]))))

    print("\n=== Combined (side-corrected) results ===")
    print(f"  {args.ai1} wins (non-draw): {A_total}/{dec_total}  winrate={pA:.3f}  95% CI [{loA:.3f}, {hiA:.3f}]")
    if dec_total > 0:
        pvalA = binomial_p_value(A_total, dec_total, 0.5)
        print(f"  two-sided binomial p={pvalA:.4f} (H0: equal strength)")

    print(f"  draws_total={draws_total}  draw_rate={draws_total/(2*args.games):.3f}")
    print(f"  side-bias (|winrate_left - winrate_right|) ≈ {side_diff:.3f}")

    # Determinism probe (optional)
    if args.determinism_check:
        res3 = run_series(args.ai1, args.ai2, args.map, args.max_steps, 5, args.autobuild, args.seed)
        res4 = run_series(args.ai1, args.ai2, args.map, args.max_steps, 5, args.autobuild, args.seed)
        assert res3 == res4, "Determinism check failed: identical seeds produced different outcomes."
        print("\nDeterminism check: PASS")

if __name__ == "__main__":
    main()


#How to run:
#python scripts/agent_validation.py --ai1 coacAI --ai2 lightRushAI --games 25 --map maps/8x8/basesWorkers8x8A.xml --max-steps 12000
