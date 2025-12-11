# Runtime UTT Change Documentation

This folder contains examples and tests for changing Unit Type Table (UTT) configuration **at runtime** during a game. This feature allows you to modify unit stats, costs, and production times for AI agents while the game is actively running, enabling dynamic game balance testing and adaptive AI research.

## Table of Contents

- [Overview](#overview)
- [What is Runtime UTT Change?](#what-is-runtime-utt-change)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Code Examples](#code-examples)
- [Use Cases](#use-cases)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Files in This Directory](#files-in-this-directory)

---

## Overview

The runtime UTT change feature allows you to:
- **Modify unit statistics** (HP, damage, cost, production time) during gameplay
- **Test dynamic balance adjustments** without restarting the game
- **Study adaptive AI behavior** when game rules change mid-match
- **Experiment with progressive difficulty** by adjusting stats over time

---

## What is Runtime UTT Change?

A **Unit Type Table (UTT)** defines all unit properties in MicroRTS:
- Health Points (HP)
- Damage (min/max)
- Resource cost
- Production time
- Movement speed
- Attack range
- And more...

**Runtime UTT change** allows you to swap the UTT configuration for one or both players **while the game is running**, without resetting the environment. This means:
- Existing units keep their current stats
- **Newly produced units** will use the new UTT stats
- Buildings and production facilities immediately use new costs/times
- The change takes effect immediately for future actions

---

## Quick Start

### Prerequisites

```bash
conda activate microrts
```

### Run Example Scripts

**Simple test (minimal output):**
```bash
python scripts/Running\ Simulations/runtime_utt_change/test_single_matchup_quiet.py
```

**Detailed test (with snapshots and verification):**
```bash
python scripts/Running\ Simulations/runtime_utt_change/test_single_matchup.py
```

**Full example with verification:**
```bash
python scripts/Running\ Simulations/runtime_utt_change/runtime_utt_change_example.py
```

---

## Basic Usage

### Step 1: Create Environment with Initial UTTs

```python
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai

env = MicroRTSBotVecEnv(
    ai1s=[microrts_ai.lightRushAI],
    ai2s=[microrts_ai.workerRushAI],
    map_paths=["maps/8x8/basesWorkers8x8A.xml"],
    max_steps=1000,
    utt_json_p0="utts/TestUnitTypeTable.json",  # Initial UTT for Player 0
    utt_json_p1="utts/TestUnitTypeTable.json",  # Initial UTT for Player 1
    autobuild=False,
)
```

### Step 2: Reset Environment

```python
obs = env.reset()
```

### Step 3: Run Game Loop and Change UTT

```python
step = 0
while step < max_steps:
    actions = [0] * env.num_envs
    obs, rewards, dones, infos = env.step(actions)
    
    # Change UTT at a specific step
    if step == 20:  # Change at step 20
        success = env.change_utt(
            utt_json_p0="utts/CustomDemoUTT.json",  # New UTT for Player 0
            utt_json_p1=None  # Keep Player 1's UTT unchanged
        )
        if success:
            print(f"✓ UTT changed at step {step}")
        else:
            print(f"✗ Failed to change UTT at step {step}")
    
    if dones[0]:
        break
    
    step += 1
```

### Step 4: Clean Up

```python
env.close()
```

---

## Code Examples

### Example 1: Simple UTT Change at Fixed Step

```python
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai

# Configuration
ai1_name = "lightRushAI"
opponent_ai_name = "coacAI"
initial_utt = "utts/TestUnitTypeTable.json"
new_utt = "utts/CustomDemoUTT.json"
change_step = 20
max_steps = 1000

# Create environment
env = MicroRTSBotVecEnv(
    ai1s=[getattr(microrts_ai, ai1_name)],
    ai2s=[getattr(microrts_ai, opponent_ai_name)],
    map_paths=["maps/8x8/basesWorkers8x8A.xml"],
    max_steps=max_steps,
    utt_json_p0=initial_utt,
    utt_json_p1=initial_utt,
    autobuild=False,
)

obs = env.reset()
utt_changed = False

# Game loop
for step in range(max_steps):
    actions = [0] * env.num_envs
    obs, rewards, dones, infos = env.step(actions)
    
    # Change UTT at specified step
    if step == change_step and not utt_changed:
        success = env.change_utt(
            utt_json_p0=new_utt,
            utt_json_p1=None
        )
        if success:
            print(f"✓ UTT changed at step {step}")
            utt_changed = True
    
    if dones[0]:
        break

env.close()
```

### Example 2: Multiple UTT Changes Over Time

```python
# Change UTT multiple times during the game
utt_changes = [
    (10, "utts/CustomDemoUTT.json", None),   # Step 10: Change P0
    (50, None, "utts/FavorWorkerRushUTT.json"),  # Step 50: Change P1
    (100, "utts/BetterCombatViewUTT.json", None),  # Step 100: Change P0 again
]

for step in range(max_steps):
    actions = [0] * env.num_envs
    obs, rewards, dones, infos = env.step(actions)
    
    # Check if any UTT change is scheduled for this step
    for change_step, new_utt_p0, new_utt_p1 in utt_changes:
        if step == change_step:
            success = env.change_utt(
                utt_json_p0=new_utt_p0,
                utt_json_p1=new_utt_p1
            )
            if success:
                print(f"✓ UTT changed at step {step}")
    
    if dones[0]:
        break
```

### Example 3: Conditional UTT Change Based on Game State

```python
# Change UTT based on game conditions
player0_advantage = False

for step in range(max_steps):
    actions = [0] * env.num_envs
    obs, rewards, dones, infos = env.step(actions)
    
    # Change UTT if Player 0 is losing (negative reward)
    if step > 20 and rewards[0] < -0.5 and not player0_advantage:
        success = env.change_utt(
            utt_json_p0="utts/CustomDemoUTT.json",  # Give P0 a boost
            utt_json_p1=None
        )
        if success:
            print(f"✓ UTT changed at step {step} to help Player 0")
            player0_advantage = True
    
    if dones[0]:
        break
```

### Example 4: Verify UTT Change

```python
# Verify UTT change by inspecting unit stats
def verify_utt_change(env, player=0, expected_light_hp=60):
    """Verify UTT change by checking unit stats"""
    utt = env.real_utt_p0 if player == 0 else env.real_utt_p1
    unit_types = utt.getUnitTypes()
    
    for i in range(unit_types.size()):
        unit_type = unit_types.get(i)
        if unit_type.name == "Light":
            if unit_type.hp == expected_light_hp:
                print(f"✓ Player {player} UTT verified: Light HP={unit_type.hp}")
                return True
            else:
                print(f"⚠ Player {player} UTT mismatch: Light HP={unit_type.hp} (expected {expected_light_hp})")
                return False
    return False

# In game loop
if step == change_step:
    success = env.change_utt(utt_json_p0="utts/CustomDemoUTT.json")
    if success:
        verify_utt_change(env, player=0, expected_light_hp=60)
```

---

## Use Cases

### 1. **Dynamic Balance Testing**
Test how game balance changes when unit stats are adjusted mid-game:
```python
# Start with balanced UTT, then favor one player
env.change_utt(utt_json_p0="utts/CustomDemoUTT.json")  # Favor P0
```

### 2. **Adaptive Difficulty**
Gradually increase difficulty by adjusting unit stats:
```python
# Progressive difficulty: make opponent stronger over time
if step == 50:
    env.change_utt(utt_json_p1="utts/FavorWorkerRushUTT.json")
if step == 100:
    env.change_utt(utt_json_p1="utts/BetterCombatViewUTT.json")
```

### 3. **AI Behavior Research**
Study how AIs adapt when game rules change:
```python
# Change UTT mid-game and observe AI response
env.change_utt(utt_json_p0="utts/CustomDemoUTT.json")
# Monitor how AI strategy changes
```

### 4. **Tournament Testing**
Test fairness by switching UTTs between matches:
```python
# Run match 1: P0 gets favorable UTT
env.change_utt(utt_json_p0="utts/CustomDemoUTT.json")
# Run match 2: P1 gets favorable UTT
env.change_utt(utt_json_p0=None, utt_json_p1="utts/CustomDemoUTT.json")
```

### 5. **Progressive Game Modes**
Create game modes where stats evolve over time:
```python
# Early game: balanced
# Mid game: favor combat units
if step == 30:
    env.change_utt(utt_json_p0="utts/BetterCombatViewUTT.json")
# Late game: favor economy
if step == 100:
    env.change_utt(utt_json_p0="utts/FavorWorkerRushUTT.json")
```

---

## API Reference

### `MicroRTSBotVecEnv.change_utt()`

Changes the UTT configuration for one or both players during gameplay.

**Parameters:**
- `utt_json_p0` (str or None): Path to new UTT JSON file for Player 0. Use `None` to keep unchanged.
- `utt_json_p1` (str or None): Path to new UTT JSON file for Player 1. Use `None` to keep unchanged.

**Returns:**
- `bool`: `True` if UTT change was successful, `False` otherwise.

**Example:**
```python
success = env.change_utt(
    utt_json_p0="utts/CustomDemoUTT.json",  # Change P0's UTT
    utt_json_p1=None  # Keep P1's UTT unchanged
)
```

### Accessing Current UTTs

After creating the environment, you can access the current UTT objects:

```python
# Get UTT objects
utt_p0 = env.real_utt_p0  # Player 0's UTT
utt_p1 = env.real_utt_p1  # Player 1's UTT

# Inspect unit types
unit_types = utt_p0.getUnitTypes()
for i in range(unit_types.size()):
    unit_type = unit_types.get(i)
    if unit_type.name == "Light":
        print(f"Light HP: {unit_type.hp}")
        print(f"Light Damage: {unit_type.minDamage}-{unit_type.maxDamage}")
        print(f"Light Cost: {unit_type.cost}")
```

---

## Best Practices

### 1. **Change Timing**
- **Early changes** (steps 1-20): Affect early game strategy
- **Mid-game changes** (steps 20-100): Impact ongoing battles
- **Late changes** (steps 100+): May have limited effect if game is decided

### 2. **UTT File Paths**
- Use relative paths from project root: `"utts/TestUnitTypeTable.json"`
- Ensure UTT files exist and are valid JSON
- Test UTT files separately before using in runtime changes

### 3. **Error Handling**
```python
success = env.change_utt(utt_json_p0="utts/CustomDemoUTT.json")
if not success:
    print("⚠ UTT change failed - check file path and UTT validity")
    # Continue game with original UTT
```

### 4. **Verification**
Always verify UTT changes worked:
```python
# After change_utt(), verify the change
utt_p0 = env.real_utt_p0
unit_types = utt_p0.getUnitTypes()
# Check expected unit stats
```

### 5. **One-Time Changes**
Use a flag to prevent multiple changes:
```python
utt_changed = False
if step == change_step and not utt_changed:
    env.change_utt(...)
    utt_changed = True
```

### 6. **Game State Awareness**
Consider game state when changing UTTs:
```python
# Don't change UTT if game is about to end
if step < max_steps - 50:  # Leave buffer before game end
    env.change_utt(...)
```

---

## Troubleshooting

### Issue: UTT Change Returns `False`

**Possible causes:**
1. **Invalid file path**: Check that UTT file exists
   ```python
   import os
   utt_path = "utts/CustomDemoUTT.json"
   if not os.path.exists(utt_path):
       print(f"UTT file not found: {utt_path}")
   ```

2. **Invalid UTT format**: Ensure UTT JSON is valid
   ```python
   import json
   with open("utts/CustomDemoUTT.json") as f:
       utt_data = json.load(f)  # Will raise error if invalid JSON
   ```

3. **Unit type mismatch**: New UTT must have same unit types as original

### Issue: UTT Change Doesn't Seem to Take Effect

**Solutions:**
1. **Verify the change**: Check UTT stats after change
   ```python
   utt_p0 = env.real_utt_p0
   # Inspect unit types to confirm change
   ```

2. **Check timing**: Changes only affect **newly produced units**, not existing ones
   - Existing units keep their original stats
   - Only units produced after the change use new stats

3. **Game may have ended**: If game ends before new units are produced, change won't be visible

### Issue: Game Crashes After UTT Change

**Solutions:**
1. **Check JVM state**: Ensure JVM is properly initialized
   ```python
   import jpype
   if not jpype.isJVMStarted():
       # JVM not started - this shouldn't happen if env is created correctly
   ```

2. **Validate UTT files**: Test UTT files in a simple script first

3. **Error handling**: Wrap `change_utt()` in try-except
   ```python
   try:
       success = env.change_utt(...)
   except Exception as e:
       print(f"Error changing UTT: {e}")
   ```

---

## Files in This Directory

- **`test_single_matchup_quiet.py`** - Minimal test script with quiet output
- **`test_single_matchup.py`** - Detailed test script with game snapshots and verification
- **`runtime_utt_change_example.py`** - Comprehensive example with UTT verification and comparison
- **`README.md`** - This documentation file
- **`outputs/`** - Directory containing saved match outputs (if output saving is enabled)

---

## Additional Resources

- **UTT Files**: Located in `utts/` directory at project root
  - `TestUnitTypeTable.json` - Default/neutral UTT
  - `CustomDemoUTT.json` - Favors light rush (high HP/damage Light units)
  - `FavorWorkerRushUTT.json` - Favors worker rush (strong workers)
  - `BetterCombatViewUTT.json` - Balanced for visible combat

- **Related Documentation**:
  - `ASYMMETRIC_UTT_README.md` - Asymmetric UTT configuration
  - `HOW_TO_TEST_RUNTIME_UTT.md` - Testing guide
  - `RUNTIME_UTT_CHANGE_README.md` - Alternative documentation

---

## Summary

Runtime UTT change is a powerful feature for:
- ✅ Testing dynamic game balance
- ✅ Studying adaptive AI behavior
- ✅ Creating progressive difficulty modes
- ✅ Researching game mechanics

**Key Points:**
- Use `env.change_utt(utt_json_p0=..., utt_json_p1=...)` during game loop
- Changes affect **newly produced units**, not existing ones
- Always verify changes with `env.real_utt_p0` and `env.real_utt_p1`
- Handle errors gracefully - changes can fail if UTT files are invalid

For questions or issues, refer to the example scripts in this directory or check the troubleshooting section above.
