import os, re, xml.etree.ElementTree as ET
import numpy as np
from typing import Any, Dict
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
from stable_baselines3.common.vec_env import VecVideoRecorder
def parse_map_hw(map_path: str) -> tuple[int, int]:
    """Robustly get (h,w) from MicroRTS map XML, with filename fallback like '8x8'."""
    # 1) Try XML
    try:
        tree = ET.parse(map_path)
        root = tree.getroot()

        def find_int(tag):
            node = root.find(tag) or root.find(f".//{tag}")
            if node is not None and node.text:
                t = node.text.strip()
                if t.isdigit(): return int(t)
            # also try attributes on any element
            for el in root.iter():
                if tag in el.attrib and str(el.attrib[tag]).isdigit():
                    return int(el.attrib[tag])
            return None

        h = find_int("height")
        w = find_int("width")
        if h and w: return h, w
    except Exception:
        pass

    # 2) Fallback: filename like ".../16x16/..." or ".../basesWorkers8x8A.xml"
    m = re.search(r"(\d+)\s*x\s*(\d+)", os.path.basename(map_path))
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        # In filenames it's usually WxH, but we just need both numbers:
        return b, a  # treat first as width, second as height → return (h,w)

    raise RuntimeError(f"Could not infer map size from '{map_path}'. "
                       f"Provide explicit h,w or use a standard map path.")

def make_dummy_actions(num_envs: int, h: int, w: int):
    """Build EXACT JNI payload: int[num_envs][2][7*h*w] as plain Python lists."""
    L = 7 * h * w
    return [[ [0]*L, [0]*L ] for _ in range(num_envs)], L

def winner_from_info(info: Dict[str, Any], rewards):
    # Trust raw_rewards[0]: +1 -> P0 win, -1 -> P1 win, 0 -> draw
    if isinstance(info, dict) and "raw_rewards" in info:
        rr = info["raw_rewards"]
        rr = rr.tolist() if hasattr(rr, "tolist") else rr
        if isinstance(rr, (list, tuple)) and len(rr) >= 1:
            if rr[0] > 0:  return "player_0"
            if rr[0] < 0:  return "player_1"
    # fallback: shaped reward sign
    if isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards) >= 1:
        if rewards[0] > 0: return "player_0"
        if rewards[0] < 0: return "player_1"
    return "draw"

def run_matches(ai1="randomAI", ai2="randomAI",
                num_games=10,
                map_path="maps/8x8/basesWorkers8x8A.xml",
                max_steps=4000,
                autobuild=True,
                num_envs=1,
                utt_json=None):
    a1 = getattr(microrts_ai, ai1)
    a2 = getattr(microrts_ai, ai2)

    env = MicroRTSBotVecEnv(
        ai1s=[a1]*num_envs, ai2s=[a2]*num_envs,
        max_steps=max_steps,
        map_paths=[map_path]*num_envs,
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=autobuild,
        utt_json=utt_json,
    )

    # We don't rely on obs shape at all
    _ = env.reset()

    h, w = parse_map_hw(map_path)
    dummy_actions, L = make_dummy_actions(num_envs, h, w)

    print(f"Map: {map_path} -> h={h}, w={w}, per-player action length L={L}")
    print(f"JNI action payload shape: [{len(dummy_actions)}][{len(dummy_actions[0])}][{len(dummy_actions[0][0])}]")

    a_w = b_w = d = 0
    for g in range(1, num_games+1):
        steps = 0
        while True:
            ret = env.step(dummy_actions)  # MUST be plain lists (int[][][])
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
                if wnr == "player_0": a_w += 1
                elif wnr == "player_1": b_w += 1
                else: d += 1
                # Print raw signal for sanity
                raw0 = (inf.get("raw_rewards")[0] if isinstance(inf, dict) and "raw_rewards" in inf else "n/a")
                shaped = (rewards[0] if isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards) else "n/a")
                print(f"[{g}/{num_games}] {ai1} vs {ai2} | steps={steps} | winner={wnr} | raw0={raw0} | reward0={shaped}")
                _ = env.reset()
                break

    env.close()
    print(f"\nSummary: {ai1} {a_w} – {b_w} {ai2} | draws {d}")


# NEW: one-episode video capture using GridNet env (Python-only, headless)
def record_one_game(ai2="coacAI", map_path="maps/8x8/basesWorkers8x8A.xml",
                    max_steps=4000, out_dir="videos", seed=0):
    import os
    os.makedirs(out_dir, exist_ok=True)

    # imports local to avoid JPype JVM restart conflicts with BotVecEnv runs
    

    def softmax(x, axis=None):
        x = x - x.max(axis=axis, keepdims=True); y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def sample(logits):
        p = softmax(logits, axis=1); c = p.cumsum(axis=1); u = np.random.rand(len(c), 1)
        return (u < c).argmax(axis=1).reshape(-1, 1)

    ai2_fn = getattr(microrts_ai, ai2)

    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=max_steps,
        render_theme=2,
        ai2s=[ai2_fn],
        map_paths=[map_path],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    # SB3’s recorder expects this attribute:
    env.render_mode = "rgb_array"

    name_prefix = f"gridnet_one_{ai2}_{os.path.splitext(os.path.basename(map_path))[0]}"
    env = VecVideoRecorder(
        env,
        video_folder=out_dir,
        record_video_trigger=lambda ep_id: ep_id == 0,  # only the first episode
        video_length=max_steps,
        name_prefix=name_prefix,
    )

    env.action_space.seed(seed)
    _ = env.reset()
    nvec = env.action_space.nvec

    steps = 0
    while True:
        # masked-random action for the left (agent) side
        action_mask = env.get_action_mask().reshape(-1, env.get_action_mask().shape[-1])
        action_mask[action_mask == 0] = -9e8
        action = np.concatenate(
            (
                sample(action_mask[:, 0:6]),
                sample(action_mask[:, 6:10]),
                sample(action_mask[:, 10:14]),
                sample(action_mask[:, 14:18]),
                sample(action_mask[:, 18:22]),
                sample(action_mask[:, 22:29]),
                sample(action_mask[:, 29: np.sum(nvec[1:])]),
            ),
            axis=1,
        )
        ret = env.step(action)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = ret
        steps += 1
        if done or steps >= max_steps:
            inf = info[0] if isinstance(info, list) and info else info
            winner = "draw"
            if isinstance(inf, dict) and "raw_rewards" in inf:
                rr = inf["raw_rewards"]; rr = rr.tolist() if hasattr(rr,"tolist") else rr
                if rr and rr[0] > 0: winner = "player_0"
                elif rr and rr[0] < 0: winner = "player_1"
            print(f"[record] Winner: {winner} | steps={steps}")
            break

    env.close()
    print(f"[record] Saved video to: {out_dir}/{name_prefix}-episode-0.mp4")



# ---------- AIs from gym_microrts/microrts_ai.py ----------
# 'POHeavyRush', 
#'POLightRush', 
#'PORangedRush', 
#'POWorkerRush', 
#'coacAI', 
#'droplet', 
#'guidedRojoA3N', 
#'izanagi', 
#'lightRushAI',
# 'mayari', 
# 'mixedBot
# ', 'naiveMCTSAI',
#  'passiveAI',
# 'randomAI', 
#'randomBiasedAI',
#  'rojo', 
#  'tiamat', 
#  'workerRushAI'