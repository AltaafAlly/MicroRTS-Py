# Runtime UTT Configuration Change

This document describes the implementation of runtime Unit Type Table (UTT) configuration changes, allowing you to modify unit stats, costs, and abilities during a running game without restarting the game or modifying the Java environment.

## Overview

The runtime UTT change feature enables:
- **Changing UTT during gameplay**: Modify unit configurations while a game is running
- **Per-player changes**: Change UTT for Player 0, Player 1, or both independently
- **No environment restart**: The game continues seamlessly with the new configuration
- **No Java environment modification**: Changes are isolated to the current game instance

## Implementation Details

### Java Changes

The following Java classes were modified to support runtime UTT changes:

1. **`JNIGridnetClient.java`**: Added `changeUTT(String newUttP0Json, String newUttP1Json)` method
2. **`JNIBotClient.java`**: Added `changeUTT(String newUttP0Json, String newUttP1Json)` method
3. **`JNIGridnetClientSelfPlay.java`**: Added `changeUTT(String newUttP0Json, String newUttP1Json)` method
4. **`JNIGridnetVecClient.java`**: Added `changeUTT(String newUttP0Json, String newUttP1Json)` method to change UTT for all clients
5. **`JNIAI.java`**: Added `setUTT(UnitTypeTable newUtt)` method to update AI's UTT reference

### Python Changes

The following Python classes were modified:

1. **`MicroRTSGridModeVecEnv`**: Added `change_utt(utt_json_p0=None, utt_json_p1=None)` method
2. **`MicroRTSBotVecEnv`**: Added `change_utt(utt_json_p0=None, utt_json_p1=None)` method

## Usage

### Basic Example

```python
from gym_microrts.envs import MicroRTSBotVecEnv

# Create environment with initial UTT
env = MicroRTSBotVecEnv(
    ai1s=["workerRushAI"],
    ai2s=["lightRushAI"],
    map_paths="maps/8x8/basesWorkers8x8A.xml",
    max_steps=2000,
    utt_json_p0="utts/CustomDemoUTT.json",
    utt_json_p1="utts/CustomDemoUTT.json",
)

obs = env.reset()

# Run game for some steps
for step in range(100):
    actions = [env.action_space.sample() for _ in range(env.num_envs)]
    obs, rewards, dones, infos = env.step(actions)
    
    # Change UTT at step 50
    if step == 50:
        success = env.change_utt(
            utt_json_p0="utts/CustomDemoUTT.json",  # New UTT for Player 0
            utt_json_p1=None  # Keep Player 1's UTT unchanged
        )
        
        if success:
            print("UTT changed successfully!")
        else:
            print("Failed to change UTT")
    
    if dones[0]:
        break

env.close()
```

### Change Both Players

```python
# Change UTT for both players
success = env.change_utt(
    utt_json_p0="utts/CustomDemoUTT.json",  # Player 0's new UTT
    utt_json_p1="utts/AsymmetricP1UTT.json"  # Player 1's new UTT
)
```

### Change Only One Player

```python
# Change only Player 0's UTT
success = env.change_utt(
    utt_json_p0="utts/CustomDemoUTT.json",
    utt_json_p1=None  # Keep Player 1's UTT unchanged
)

# Change only Player 1's UTT
success = env.change_utt(
    utt_json_p0=None,  # Keep Player 0's UTT unchanged
    utt_json_p1="utts/AsymmetricP1UTT.json"
)
```

## How It Works

1. **UTT Loading**: When `change_utt()` is called, the specified UTT JSON files are loaded
2. **GameState Cloning**: The current `GameState` is cloned using `cloneChangingUTTs()`, which:
   - Creates a new GameState with the new UTTs
   - Updates all units to use the new unit types
   - Adjusts unit HP if they were at full health
3. **State Updates**: The client's internal state is updated with the new GameState and UTTs
4. **AI Updates**: AIs are updated with the new UTT references if they support it
5. **Mask Updates**: Action masks are recalculated if the unit types changed

## Limitations

1. **Unit Type Compatibility**: The new UTT must contain all unit types that exist in the current game state. If a unit type is missing, the change will fail.

2. **Action Space**: The action space dimensions are fixed at environment initialization. If the new UTT has a different number of unit types, the action space won't automatically update. Consider resetting the environment if unit types changed significantly.

3. **Ongoing Actions**: Units with actions in progress will continue with those actions, but new actions will use the new UTT configuration.

4. **Unit HP Adjustments**: Only units at full HP are adjusted to match the new unit type's HP. Partially damaged units keep their current HP.

## Error Handling

The `change_utt()` method returns `True` if successful, `False` otherwise. Common failure reasons:

- Unit type mismatch: New UTT doesn't contain all required unit types
- Invalid JSON: UTT file is malformed
- File not found: UTT file path is incorrect

## Example Script

See `scripts/Running Simulations/runtime_utt_change_example.py` for a complete working example.

## Technical Notes

- The implementation uses `GameState.cloneChangingUTTs()` which is already present in the MicroRTS codebase
- Changes are isolated to the current game instance and don't affect other games or the Java environment
- The feature works with both symmetric and asymmetric UTT configurations
- All client types (JNIGridnetClient, JNIBotClient, JNIGridnetClientSelfPlay) support runtime UTT changes


