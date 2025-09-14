#!/usr/bin/env python3
import os
import numpy as np
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder

# ---- random masked policy helpers (same idea as hello_world.py) ----
def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def sample(logits):
    p = softmax(logits, axis=1)
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    return (u < c).argmax(axis=1).reshape(-1, 1)

def main(
    ai2_name="coacAI",
    map_path="maps/16x16/basesWorkers16x16.xml",
    max_steps=2000,
    out_dir="videos",
    seed=0,
):
    os.makedirs(out_dir, exist_ok=True)

    # Left side = our random masked policy; Right side = chosen built-in AI
    ai2 = getattr(microrts_ai, ai2_name)

    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=max_steps,
        render_theme=2,
        ai2s=[ai2],
        map_paths=[map_path],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    # SB3’s recorder expects this attribute:
    env.render_mode = "rgb_array"

    name_prefix = f"gridnet_one_{ai2_name}_{os.path.splitext(os.path.basename(map_path))[0]}"
    env = VecVideoRecorder(
        env,
        video_folder=out_dir,
        record_video_trigger=lambda ep_id: ep_id == 0,  # record only episode 0
        video_length=max_steps,                          # up to max_steps frames
        name_prefix=name_prefix,
    )

    env.action_space.seed(seed)
    obs = env.reset()
    nvec = env.action_space.nvec

    steps = 0
    while True:
        # Masked random action (valid per-cell action)
        action_mask = env.get_action_mask()
        action_mask = action_mask.reshape(-1, action_mask.shape[-1])
        action_mask[action_mask == 0] = -9e8  # disallow invalid choices

        action = np.concatenate(
            (
                sample(action_mask[:, 0:6]),                     # action type
                sample(action_mask[:, 6:10]),                    # move
                sample(action_mask[:, 10:14]),                   # harvest
                sample(action_mask[:, 14:18]),                   # return
                sample(action_mask[:, 18:22]),                   # produce dir
                sample(action_mask[:, 22:29]),                   # produce unit type
                sample(action_mask[:, 29: np.sum(nvec[1:])]),    # attack target
            ),
            axis=1,
        )

        ret = env.step(action)
        # Gym or Gymnasium API
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = ret

        steps += 1
        if done or steps >= max_steps:
            # Try to print who won (uses your build’s raw_rewards[0] convention)
            inf = info[0] if isinstance(info, list) and info else info
            winner = "draw"
            if isinstance(inf, dict) and "raw_rewards" in inf:
                rr = inf["raw_rewards"]; rr = rr.tolist() if hasattr(rr, "tolist") else rr
                if rr and rr[0] > 0: winner = "player_0"
                elif rr and rr[0] < 0: winner = "player_1"
            print(f"Winner: {winner} | steps={steps}")
            break

    env.close()
    print(f"Saved video to: {out_dir}/{name_prefix}-episode-0.mp4")

if __name__ == "__main__":
    main()
