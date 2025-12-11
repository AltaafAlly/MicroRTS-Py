# Local Tournament Runner - Sequential Game Execution Pattern

This directory contains a pattern for running multiple MicroRTS games **sequentially** (one after another) in an organized tournament format. This pattern is useful for testing AI agents, comparing UTT configurations, and running batch simulations.

## Table of Contents

- [Overview](#overview)
- [What This Pattern Does](#what-this-pattern-does)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Code Structure](#code-structure)
- [Using This Pattern in Other Cases](#using-this-pattern-in-other-cases)
- [Customization Guide](#customization-guide)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The local tournament runner demonstrates a **sequential execution pattern** for running multiple MicroRTS games:

1. **Organized Structure**: Games are organized into pairs, rounds, and tournaments
2. **Sequential Execution**: Games run one after another (not concurrently)
3. **Result Tracking**: Comprehensive result collection and analysis
4. **Error Handling**: Robust error handling for long-running tournaments
5. **Progress Logging**: Real-time progress updates during execution

**Key Feature**: This pattern runs games **sequentially** (one after another), which is ideal for:
- Local testing (no cluster needed)
- Resource-constrained environments
- Debugging and development
- Small to medium-scale tournaments

---

## What This Pattern Does

### Execution Flow

```
Tournament Start
    ↓
For each UTT Configuration:
    ↓
    For each AI Pair (round-robin):
        ↓
        Run N games sequentially:
            Game 1 → Game 2 → Game 3 → ... → Game N
        ↓
        Collect results
    ↓
    Calculate statistics
    ↓
Save results
```

### Example: 2 UTTs × 6 AIs × 15 games per pair

- **Total pairs**: 6 × 5 / 2 = 15 pairs per UTT
- **Total games per UTT**: 15 pairs × 15 games = 225 games
- **Total games**: 225 × 2 UTTs = 450 games
- **Execution**: All games run sequentially, one after another

---

## Quick Start

### Option 1: Using the Shell Script (Recommended)

```bash
cd scripts/Running\ Simulations/local_testing
./run_tournament.sh
```

### Option 2: Running Python Directly

```bash
cd scripts/Running\ Simulations/local_testing
conda activate microrts
python run_local_tournament.py
```

### What Happens

1. Creates tournament instance with configuration
2. Runs games sequentially for each UTT configuration
3. Collects results for each pair
4. Calculates statistics and rankings
5. Saves results to `results/` directory

---

## Architecture

### File Structure

```
local_testing/
├── run_local_tournament.py    # Main entry point (simplified config)
├── run_tournament.sh          # Shell script wrapper
├── README.md                  # This file
└── results/                   # Output directory (created automatically)
    ├── default_original/
    │   ├── detailed_results.json
    │   ├── standings.csv
    │   └── pair_results.csv
    ├── custom_demo/
    │   └── ...
    ├── tournament_summary.json
    └── utt_comparison.csv
```

### Component Overview

1. **`run_local_tournament.py`**: Simplified wrapper that:
   - Sets up local testing configuration
   - Creates tournament instance
   - Runs tournament
   - Displays summary

2. **`UTTImpactTournament` class** (in `utt_impact_tournament.py`):
   - Core tournament logic
   - Game execution
   - Result collection
   - Statistical analysis

3. **`run_tournament.sh`**: Shell script that:
   - Activates conda environment
   - Sets up Java environment
   - Runs Python script

---

## How It Works

### Step-by-Step Execution

#### 1. Tournament Initialization

```python
tournament = UTTImpactTournament(output_dir="results")
tournament.tournament_config["games_per_pair"] = 1  # Local: 1 game
tournament.tournament_config["max_steps"] = 5000     # Local: 5000 steps
```

#### 2. UTT Configuration Loop

```python
for utt_name, utt_config in tournament.utt_configs.items():
    # Run tournament for this UTT
    utt_results = tournament.run_utt_comparison(utt_name, utt_config)
```

#### 3. Round-Robin Pair Generation

```python
# Generate all unique pairs: (AI1, AI2), (AI1, AI3), ..., (AI5, AI6)
for i in range(len(available_ais)):
    for j in range(i + 1, len(available_ais)):
        ai_left, ai_right = available_ais[i], available_ais[j]
        # Run match between these two AIs
```

#### 4. Sequential Game Execution

```python
# Run N games sequentially for this pair
for game_num in range(games_per_pair):
    # Create/reset environment
    obs = env.reset()
    
    # Run game until completion
    while not done:
        obs, rewards, done, info = env.step(actions)
    
    # Determine winner and record result
    # Reset for next game
    obs = env.reset()
```

#### 5. Result Collection

```python
# After all games for a pair:
results = {
    "left_wins": 5,
    "right_wins": 3,
    "draws": 2,
    "games": 10
}

# Update standings
standings[ai_left]["wins"] += results["left_wins"]
standings[ai_right]["wins"] += results["right_wins"]
```

#### 6. Statistical Analysis

```python
# Calculate statistics after all pairs complete
statistics = {
    "total_games": 225,
    "win_rates": {...},
    "ranking": [...],
    "top_performer": "coacAI"
}
```

---

## Code Structure

### Main Entry Point: `run_local_tournament.py`

```python
def main():
    # 1. Create output directory
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Create tournament instance
    tournament = UTTImpactTournament(output_dir=str(output_dir))
    
    # 3. Override config for local testing
    tournament.tournament_config["games_per_pair"] = 1
    tournament.tournament_config["max_steps"] = 5000
    
    # 4. Run tournament
    tournament.run_full_tournament()
    
    # 5. Display summary
    # ...
```

### Core Tournament Class: `UTTImpactTournament`

Key methods:

- **`run_full_tournament()`**: Main orchestrator
  - Loops through UTT configurations
  - Calls `run_utt_comparison()` for each UTT

- **`run_utt_comparison(utt_name, utt_config)`**: Runs tournament for one UTT
  - Generates round-robin pairs
  - Calls `run_pair()` for each pair
  - Calculates statistics

- **`run_pair(ai_left, ai_right, utt_config, ...)`**: Runs match between two AIs
  - Creates environment with UTT config
  - Runs N games sequentially
  - Collects results
  - Returns win/loss/draw counts

- **`calculate_utt_statistics(utt_results)`**: Statistical analysis
  - Win rates, rankings, draw rates
  - Top/bottom performers

---

## Using This Pattern in Other Cases

This sequential execution pattern can be adapted for various use cases:

### Use Case 1: Simple Batch Testing

Run multiple games with different configurations:

```python
from gym_microrts.envs.vec_env import MicroRTSBotVecEnv
from gym_microrts import microrts_ai

def run_batch_games(ai_pairs, num_games_per_pair=10):
    """Run multiple games sequentially."""
    results = []
    
    for ai1_name, ai2_name in ai_pairs:
        print(f"Running {ai1_name} vs {ai2_name}...")
        
        # Create environment
        env = MicroRTSBotVecEnv(
            ai1s=[getattr(microrts_ai, ai1_name)],
            ai2s=[getattr(microrts_ai, ai2_name)],
            map_paths=["maps/8x8/basesWorkers8x8A.xml"],
            max_steps=5000,
            autobuild=False,
        )
        
        obs = env.reset()
        wins_ai1 = 0
        wins_ai2 = 0
        draws = 0
        
        # Run games sequentially
        for game_num in range(num_games_per_pair):
            done = False
            while not done:
                actions = [[[0] * (7 * env.height * env.width)] * 2]
                obs, rewards, done, info = env.step(actions)
                
                if done[0]:
                    # Determine winner
                    info_dict = info[0] if isinstance(info, list) else info
                    if "raw_rewards" in info_dict:
                        raw_reward = info_dict["raw_rewards"][0]
                        if raw_reward > 0:
                            wins_ai1 += 1
                        elif raw_reward < 0:
                            wins_ai2 += 1
                        else:
                            draws += 1
                    obs = env.reset()
                    break
        
        results.append({
            "ai1": ai1_name,
            "ai2": ai2_name,
            "wins_ai1": wins_ai1,
            "wins_ai2": wins_ai2,
            "draws": draws,
        })
        
        env.close()
    
    return results

# Usage
ai_pairs = [
    ("lightRushAI", "workerRushAI"),
    ("coacAI", "naiveMCTSAI"),
]
results = run_batch_games(ai_pairs, num_games_per_pair=10)
```

### Use Case 2: UTT Comparison Testing

Compare different UTT configurations:

```python
def compare_utts(utt_configs, ai_pair, num_games=10):
    """Compare multiple UTT configurations."""
    results = {}
    
    for utt_name, utt_config in utt_configs.items():
        print(f"Testing UTT: {utt_name}")
        
        env = MicroRTSBotVecEnv(
            ai1s=[getattr(microrts_ai, ai_pair[0])],
            ai2s=[getattr(microrts_ai, ai_pair[1])],
            map_paths=["maps/8x8/basesWorkers8x8A.xml"],
            max_steps=5000,
            utt_json_p0=utt_config.get("utt_json_p0"),
            utt_json_p1=utt_config.get("utt_json_p1"),
            autobuild=False,
        )
        
        # Run games sequentially
        wins_p0 = 0
        for _ in range(num_games):
            obs = env.reset()
            done = False
            while not done:
                actions = [[[0] * (7 * env.height * env.width)] * 2]
                obs, rewards, done, info = env.step(actions)
                if done[0]:
                    info_dict = info[0] if isinstance(info, list) else info
                    if "raw_rewards" in info_dict and info_dict["raw_rewards"][0] > 0:
                        wins_p0 += 1
                    obs = env.reset()
                    break
        
        results[utt_name] = {"wins_p0": wins_p0, "total": num_games}
        env.close()
    
    return results

# Usage
utt_configs = {
    "default": {"utt_json_p0": None, "utt_json_p1": None},
    "custom": {"utt_json_p0": "utts/CustomDemoUTT.json", "utt_json_p1": "utts/CustomDemoUTT.json"},
}
results = compare_utts(utt_configs, ("lightRushAI", "workerRushAI"), num_games=20)
```

### Use Case 3: Map Testing

Test different maps sequentially:

```python
def test_maps(maps, ai_pair, num_games=10):
    """Test AI performance across different maps."""
    results = {}
    
    for map_name, map_path in maps.items():
        print(f"Testing map: {map_name}")
        
        env = MicroRTSBotVecEnv(
            ai1s=[getattr(microrts_ai, ai_pair[0])],
            ai2s=[getattr(microrts_ai, ai_pair[1])],
            map_paths=[map_path],
            max_steps=5000,
            autobuild=False,
        )
        
        # Run games sequentially
        wins_p0 = 0
        for _ in range(num_games):
            obs = env.reset()
            done = False
            while not done:
                actions = [[[0] * (7 * env.height * env.width)] * 2]
                obs, rewards, done, info = env.step(actions)
                if done[0]:
                    info_dict = info[0] if isinstance(info, list) else info
                    if "raw_rewards" in info_dict and info_dict["raw_rewards"][0] > 0:
                        wins_p0 += 1
                    obs = env.reset()
                    break
        
        results[map_name] = {"wins_p0": wins_p0, "total": num_games}
        env.close()
    
    return results
```

### Use Case 4: Parameter Sweep

Test different game parameters:

```python
def parameter_sweep(param_name, param_values, base_config, num_games=10):
    """Test different parameter values."""
    results = {}
    
    for param_value in param_values:
        print(f"Testing {param_name}={param_value}")
        
        # Update config with parameter value
        config = base_config.copy()
        config[param_name] = param_value
        
        env = MicroRTSBotVecEnv(
            ai1s=[getattr(microrts_ai, config["ai1"])],
            ai2s=[getattr(microrts_ai, config["ai2"])],
            map_paths=[config["map_path"]],
            max_steps=config["max_steps"],
            autobuild=False,
        )
        
        # Run games sequentially
        wins_p0 = 0
        for _ in range(num_games):
            obs = env.reset()
            done = False
            while not done:
                actions = [[[0] * (7 * env.height * env.width)] * 2]
                obs, rewards, done, info = env.step(actions)
                if done[0]:
                    info_dict = info[0] if isinstance(info, list) else info
                    if "raw_rewards" in info_dict and info_dict["raw_rewards"][0] > 0:
                        wins_p0 += 1
                    obs = env.reset()
                    break
        
        results[param_value] = {"wins_p0": wins_p0, "total": num_games}
        env.close()
    
    return results
```

---

## Customization Guide

### Customizing Tournament Configuration

Edit `run_local_tournament.py`:

```python
# Override tournament config
tournament.tournament_config["games_per_pair"] = 5      # More games per pair
tournament.tournament_config["max_steps"] = 10000       # Longer games
tournament.tournament_config["max_steps_long"] = 20000  # Longer retry games
```

### Adding Custom AI Agents

Edit `utt_impact_tournament.py`:

```python
self.baseline_ais = [
    "lightRushAI",
    "workerRushAI",
    "coacAI",
    "yourCustomAI",  # Add your AI here
]
```

### Adding Custom UTT Configurations

Edit `utt_impact_tournament.py`:

```python
self.utt_configs = {
    "default_original": {"utt_json_p0": None, "utt_json_p1": None},
    "custom_demo": {"utt_json_p0": "utts/CustomDemoUTT.json", "utt_json_p1": "utts/CustomDemoUTT.json"},
    "my_custom_utt": {"utt_json_p0": "utts/MyUTT.json", "utt_json_p1": "utts/MyUTT.json"},  # Add here
}
```

### Customizing Maps

Edit `utt_impact_tournament.py`:

```python
self.tournament_config = {
    "map_path": "maps/10x10/basesWorkers10x10A.xml",  # Change map
    # ...
}
```

### Custom Result Processing

Add custom analysis in `run_local_tournament.py`:

```python
# After tournament completes
results_file = output_dir / "tournament_results.json"
if results_file.exists():
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Your custom analysis here
    for utt_name, utt_data in results.items():
        # Process results
        pass
```

---

## Best Practices

### 1. **Resource Management**

Always close environments properly:

```python
try:
    env.vec_client.close()
except Exception:
    pass
```

### 2. **Error Handling**

Wrap game execution in try-except:

```python
try:
    # Run game
    obs, rewards, done, info = env.step(actions)
except Exception as e:
    print(f"Error in game: {e}")
    # Handle error (skip game, retry, etc.)
```

### 3. **Progress Logging**

Log progress for long-running tournaments:

```python
total_games = len(pairs) * games_per_pair
completed = 0

for pair in pairs:
    for game_num in range(games_per_pair):
        # Run game
        completed += 1
        if completed % 10 == 0:
            print(f"Progress: {completed}/{total_games} ({100*completed/total_games:.1f}%)")
```

### 4. **Result Saving**

Save intermediate results for long tournaments:

```python
# Save after each UTT configuration
if utt_name not in saved_results:
    save_results(utt_name, utt_results)
```

### 5. **Memory Management**

For very long tournaments, consider:

```python
# Close and recreate environment periodically
if game_num % 100 == 0:
    env.close()
    env = create_new_environment()
```

### 6. **Configuration Validation**

Validate configuration before starting:

```python
# Check AI availability
available_ais = [ai for ai in self.baseline_ais if hasattr(microrts_ai, ai)]
if len(available_ais) < 2:
    raise ValueError("Need at least 2 available AIs")
```

---

## Troubleshooting

### Issue: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'utt_impact_tournament'`

**Solution**:
```bash
# Make sure you're in the correct directory
cd scripts/Running\ Simulations/local_testing

# Or run from project root
cd /path/to/MicroRTS-Py-Research
python scripts/Running\ Simulations/local_testing/run_local_tournament.py
```

### Issue: Java/JVM Errors

**Problem**: `JVM not found` or `Java version mismatch`

**Solution**:
```bash
# Check Java version
java -version

# Set JAVA_HOME if needed
export JAVA_HOME=/path/to/java
export PATH="$JAVA_HOME/bin:$PATH"
```

### Issue: Games Taking Too Long

**Problem**: Games exceed `max_steps` frequently

**Solution**:
```python
# Increase max_steps
tournament.tournament_config["max_steps"] = 10000

# Or reduce for faster testing
tournament.tournament_config["max_steps"] = 2000
```

### Issue: High Draw Rate

**Problem**: Many games end in draws

**Solution**:
```python
# Increase max_steps for retry
tournament.tournament_config["max_steps_long"] = 20000

# Or adjust draw retry threshold
tournament.tournament_config["draw_retry_threshold"] = 0.3  # Retry if >30% draws
```

### Issue: Out of Memory

**Problem**: System runs out of memory during long tournaments

**Solution**:
```python
# Close environments more frequently
if pair_num % 5 == 0:
    env.close()
    # Force garbage collection
    import gc
    gc.collect()
```

### Issue: Results Not Saving

**Problem**: Results directory not created or files not written

**Solution**:
```python
# Check directory permissions
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

# Verify write permissions
test_file = output_dir / "test.txt"
test_file.write_text("test")
test_file.unlink()  # Clean up
```

---

## Summary

This sequential execution pattern provides:

✅ **Organized Structure**: Clear hierarchy (tournament → UTT → pairs → games)  
✅ **Sequential Execution**: Games run one after another (simple, reliable)  
✅ **Result Tracking**: Comprehensive result collection and analysis  
✅ **Easy Customization**: Simple configuration overrides  
✅ **Error Resilience**: Robust error handling  
✅ **Progress Visibility**: Real-time logging  

**Key Pattern**: 
```
For each configuration:
    For each pair:
        For each game:
            Run game → Collect result → Reset
        Aggregate pair results
    Calculate statistics
Save all results
```

This pattern is ideal for:
- Local testing and development
- Small to medium-scale tournaments
- Debugging and validation
- Resource-constrained environments

For questions or issues, refer to the example code in `run_local_tournament.py` and `utt_impact_tournament.py`.
