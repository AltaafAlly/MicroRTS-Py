# How to Test Runtime UTT Change

This guide shows you how to test if the runtime UTT change functionality works.

## Quick Test (Recommended)

Run the comprehensive test script:

```bash
cd /home/altaaf/projects/MicroRTS-Py-Research
python scripts/Running\ Simulations/runtime_utt_change/test_runtime_utt_working.py
```

Or run the main example to see it in action:

```bash
conda activate microrts
python scripts/Running\ Simulations/runtime_utt_change/runtime_utt_change_example.py
```

This script will:
1. ✅ Verify all Java methods are compiled
2. ✅ Create a game environment
3. ✅ Run a few game steps
4. ✅ Change UTT at runtime
5. ✅ Verify the change worked
6. ✅ Continue the game with new UTT

## Prerequisites

Before running the test, make sure:

1. **Java code is compiled:**
   ```bash
   bash build.sh
   ```

2. **Python environment is set up:**
   - If using conda: `conda activate microrts39`
   - Required packages: `gym`, `jpype1`, `numpy`

3. **JAR file exists:**
   ```bash
   ls -lh gym_microrts/microrts/microrts.jar
   ```

## Step-by-Step Manual Test

If you want to test manually:

### 1. Start Python and Import

```python
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai
```

### 2. Create Environment

```python
env = MicroRTSBotVecEnv(
    ai1s=["workerRushAI"],
    ai2s=["lightRushAI"],
    map_paths="maps/8x8/basesWorkers8x8A.xml",
    max_steps=2000,
    utt_json_p0="utts/CustomDemoUTT.json",  # Optional
    autobuild=False,
)
```

### 3. Reset and Run Some Steps

```python
obs = env.reset()

# Run a few steps
for i in range(10):
    actions = [env.action_space.sample() for _ in range(env.num_envs)]
    obs, rewards, dones, infos = env.step(actions)
    print(f"Step {i}: Game running...")
```

### 4. Change UTT at Runtime

```python
# Change UTT for Player 0
success = env.change_utt(
    utt_json_p0="utts/CustomDemoUTT.json",
    utt_json_p1=None  # Keep Player 1's UTT unchanged
)

if success:
    print("✓ UTT changed successfully!")
else:
    print("✗ UTT change failed")
```

### 5. Continue Game

```python
# Run more steps to verify game continues
for i in range(10, 20):
    actions = [env.action_space.sample() for _ in range(env.num_envs)]
    obs, rewards, dones, infos = env.step(actions)
    print(f"Step {i}: Game continues with new UTT...")
```

### 6. Clean Up

```python
env.close()
```

## What to Look For

### ✅ Success Indicators:

1. **Method exists**: `change_utt()` method is callable
2. **Returns True**: Method returns `True` when called
3. **Game continues**: Game runs normally after UTT change
4. **No errors**: No exceptions or crashes

### ❌ Failure Indicators:

1. **Method not found**: `AttributeError: 'MicroRTSBotVecEnv' object has no attribute 'change_utt'`
2. **Returns False**: Method returns `False` (usually unit type mismatch)
3. **Game crashes**: Exception after UTT change
4. **Import errors**: Missing dependencies

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'gym'"

**Solution:**
```bash
pip install gym jpype1 numpy
# or
conda install -c conda-forge gym jpype1 numpy
```

### Issue: "JAR file not found"

**Solution:**
```bash
bash build.sh
```

### Issue: "change_utt() returns False"

**Possible causes:**
- New UTT doesn't contain all unit types in current game
- Invalid UTT JSON file
- File path is incorrect

**Solution:**
- Ensure UTT file contains all required unit types
- Check UTT file path is correct
- Try using the same UTT file (should always work)

### Issue: "JVM already started" or JVM errors

**Solution:**
- Restart Python session
- Make sure you're not running multiple environments simultaneously

## Expected Output

When the test works correctly, you should see:

```
======================================================================
Testing Runtime UTT Change Functionality
======================================================================

[1/6] Importing modules...
   ✓ Imports successful

[2/6] Checking UTT files...
   ✓ Found UTT file: .../utts/CustomDemoUTT.json

[3/6] Creating environment...
   ✓ Environment created

[4/6] Getting initial UTT configuration...
   ✓ Initial UTT has X unit types

[5/6] Running game steps...
   ✓ Environment reset
   Step 0: Game running...
   Step 5: Game running...
   ✓ Ran 10 steps successfully

[6/6] Changing UTT at runtime...
   Attempting to change UTT configuration...
   ✓ UTT changed successfully!
   ✓ New UTT has X unit types

   Running 5 more steps to verify game continues...
   ✓ Game continued successfully after UTT change!

[Cleanup] Closing environment...
   ✓ Environment closed

======================================================================
✅ TEST PASSED: Runtime UTT change works correctly!
======================================================================
```

## Alternative: Quick Verification

If you just want to verify the methods are compiled (without running a full game):

```bash
javap -cp gym_microrts/microrts/microrts.jar tests.JNIGridnetClient | grep changeUTT
```

Should output:
```
public boolean changeUTT(java.lang.String, java.lang.String) throws java.lang.Exception;
```

Or use the simple test script:

```bash
python scripts/Running\ Simulations/runtime_utt_change/test_runtime_utt_simple.py
```

## Next Steps

Once the test passes, you can:

1. Use `change_utt()` in your own code
2. Experiment with different UTT configurations
3. Test changing UTT at different game stages
4. Try asymmetric UTT changes (different UTTs for each player)

See `RUNTIME_UTT_CHANGE_README.md` for detailed usage examples.

## File Organization

All runtime UTT change files are organized in:
- **Folder:** `scripts/Running Simulations/runtime_utt_change/`
- **Main example:** `runtime_utt_change_example.py`
- **Tests:** `test_runtime_utt_working.py`, `test_runtime_utt_simple.py`
- **Documentation:** See `README.md` in that folder

