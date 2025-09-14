# scripts/ai_vs_ai_debug.py
import json
import numpy as np
import pprint
from typing import Any, Dict
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai

PP = pprint.PrettyPrinter(depth=4, width=100)

def run_matches(ai1="workerRushAI", ai2="coacAI",
                num_games=1,
                map_path="maps/8x8/basesWorkers8x8A.xml",
                max_steps=12000,
                autobuild=True):
    a1 = getattr(microrts_ai, ai1)
    a2 = getattr(microrts_ai, ai2)

    env = MicroRTSBotVecEnv(
        ai1s=[a1],
        ai2s=[a2],
        max_steps=max_steps,
        map_paths=[map_path],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=autobuild,
    )

    _ = env.reset()
    num_envs = getattr(env, "num_envs", 1)
    dummy_actions = [[[0]*8, [0]*8] for _ in range(num_envs)]  # exact Java int[][][]

    steps_in_game = 0
    while True:
        ret = env.step(dummy_actions)

        # Handle both 4-return (Gym) and 5-return (Gymnasium)
        if len(ret) == 5:
            obs, rewards, terminated, truncated, info = ret
            done = terminated or truncated
        elif len(ret) == 4:
            obs, rewards, done, info = ret
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(ret)}")

        # Reduce vectorized done if needed
        if isinstance(done, (list, tuple, np.ndarray)):
            done = bool(done[0]) if len(done) else False

        steps_in_game += 1
        if done:
            inf = info[0] if isinstance(info, list) and info else info
            # ---- DEBUG DUMP (print + write to file) ----
            print("\n--- DEBUG: final info (pretty) ---")
            try:
                print(PP.pformat(inf))
            except Exception as e:
                print("pretty print failed:", e, "\nraw repr:", repr(inf))

            try:
                with open("last_info.json", "w") as f:
                    json.dump(inf, f, indent=2, default=lambda o: str(o))
                print("\nWrote last_info.json next to this script.")
            except Exception as e:
                print("Failed to write last_info.json:", e)

            print("\n--- DEBUG: meta ---")
            print("rewards:", rewards)
            print("steps_in_game:", steps_in_game, "max_steps:", max_steps)
            print("time_limited:", steps_in_game >= max_steps)
            # --------------------------------------------

            break

    env.close()
    print("\nDone (debug).")

if __name__ == "__main__":
    # small map + aggressive bot to finish quickly
    run_matches(ai1="workerRushAI", ai2="coacAI", num_games=1)
