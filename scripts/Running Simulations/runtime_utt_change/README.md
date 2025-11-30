# Runtime UTT Change

This folder contains examples and tests for changing Unit Type Table (UTT) configuration at runtime during a game.

## Files

- **`runtime_utt_change_example.py`** - Main example script demonstrating runtime UTT changes with verification
- **`test_single_matchup.py`** - Simple test script to verify UTT changes work correctly

## Quick Start

```bash
conda activate microrts
python scripts/Running\ Simulations/runtime_utt_change/test_single_matchup.py
```

To test different opponents, modify `opponent_ai_name` in `test_single_matchup.py`.

## Usage

```python
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv

env = MicroRTSBotVecEnv(...)
env.reset()

# Change UTT during gameplay
env.change_utt(
    utt_json_p0="utts/CustomDemoUTT.json",  # New UTT for Player 0
    utt_json_p1=None  # Keep Player 1's UTT unchanged (or provide new path)
)
```
