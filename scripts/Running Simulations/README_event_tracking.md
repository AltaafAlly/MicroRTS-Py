# MicroRTS Event Tracking Script

## 🎮 Enhanced Tournament Script with Map Selection

The `run_match_configured_with_events.py` script now supports easy map switching and flexible configuration!

## 🚀 Quick Start

### **1. List Available Maps**
```bash
python scripts/run_match_configured_with_events.py --list-maps
```

### **2. Interactive Map Selection**
```bash
python scripts/run_match_configured_with_events.py --interactive
```

### **3. Specify Map Directly**
```bash
python scripts/run_match_configured_with_events.py --map maps/10x10/basesTwoWorkers10x10.xml
```

## 📋 Command Line Options

### **Map Selection**
- `--map MAP_PATH` or `-m MAP_PATH` - Specify map file directly
- `--interactive` or `-i` - Interactive map selection menu
- `--list-maps` or `-l` - List all available maps and exit

### **Tournament Settings**
- `--games N` or `-g N` - Games per pair (default: 2)
- `--max-steps N` - Maximum steps per game (default: 4000)
- `--ais AI1 AI2 AI3` - Specify which AIs to include

### **Event Tracking**
- `--verbose` or `-v` - Enable verbose event output
- `--quiet` or `-q` - Quiet mode (no event output)

### **UTT Settings**
- `--utt-p0 FILE` - Player 0 UTT file (default: utts/CustomDemoUTT.json)
- `--utt-p1 FILE` - Player 1 UTT file (default: utts/AsymmetricP1UTT.json)

## 🎯 Usage Examples

### **Basic Tournament (Default Settings)**
```bash
conda activate microrts
python scripts/run_match_configured_with_events.py
```

### **Interactive Map Selection**
```bash
conda activate microrts
python scripts/run_match_configured_with_events.py --interactive
```

### **Custom Map with More Games**
```bash
conda activate microrts
python scripts/run_match_configured_with_events.py --map gym_microrts/microrts/maps/16x16/basesWorkers16x16A.xml --games 5
```

### **Specific AIs with Verbose Output**
```bash
conda activate microrts
python scripts/run_match_configured_with_events.py --ais coacAI droplet randomAI --verbose
```

### **Quiet Mode (No Event Output)**
```bash
conda activate microrts
python scripts/run_match_configured_with_events.py --quiet --games 10
```

### **Custom UTT Files**
```bash
conda activate microrts
python scripts/run_match_configured_with_events.py --utt-p0 utts/MyCustomUTT.json --utt-p1 utts/AnotherUTT.json
```

## 📊 Output Files

The script generates three CSV files in `results/`:
- `round_robin_pairs.csv` - Match results
- `round_robin_standings.csv` - Tournament standings
- `match_events.csv` - All tracked events

## 🗺️ Map Discovery

The script automatically discovers all `.xml` map files in the `gym_microrts/microrts/maps/` directory and subdirectories. Maps are organized by size:
- `gym_microrts/microrts/maps/8x8/` - Small maps (8x8)
- `gym_microrts/microrts/maps/10x10/` - Medium maps (10x10)
- `gym_microrts/microrts/maps/16x16/` - Large maps (16x16)
- `gym_microrts/microrts/maps/24x24/` - Extra large maps (24x24)
- `gym_microrts/microrts/maps/32x32/` - Huge maps (32x32)

## 🎮 Interactive Mode

When using `--interactive`, you'll see:
```
🗺️  Available Maps:
============================================================
 1. basesWorkers8x8A.xml      (8x8)
 2. basesWorkers8x8B.xml      (8x8)
 3. basesTwoWorkers10x10.xml  (10x10)
 4. basesWorkers16x16A.xml    (16x16)
============================================================

🎮 Select map (1-4) or press Enter for default: 
```

## 🔧 Tips

1. **Use `--list-maps`** to see all available maps before running
2. **Use `--interactive`** for easy map selection during development
3. **Use `--verbose`** to see real-time event tracking
4. **Use `--quiet`** for faster execution when you don't need event details
5. **Combine options** for maximum flexibility

## 🎯 Example Workflow

```bash
# 1. See what maps are available
python scripts/run_match_configured_with_events.py --list-maps

# 2. Run interactive tournament
python scripts/run_match_configured_with_events.py --interactive --games 3 --verbose

# 3. Run specific map with custom AIs
python scripts/run_match_configured_with_events.py --map maps/10x10/basesTwoWorkers10x10.xml --ais coacAI droplet --games 5
```

Enjoy your enhanced MicroRTS tournaments! 🎉
