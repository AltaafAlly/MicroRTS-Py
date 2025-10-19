# Local UTT Impact Tournament Testing

This directory contains a simplified setup for running the UTT Impact Tournament locally for testing purposes.

## Quick Start

### Option 1: Using the shell script (recommended)
```bash
cd local_testing
./run_tournament.sh
```

### Option 2: Running Python directly
```bash
cd local_testing
python run_local_tournament.py
```

## What it does

- Runs a **simplified tournament** with only 3 games per match (vs 15 on cluster)
- Uses **shorter game length** (1000 steps vs 5000 on cluster)
- Tests the **same UTT configurations** as the cluster version
- Uses the **same AI agents** as the cluster version
- Saves results to `results/`

## Configuration

The local version uses these settings (vs cluster):
- **Games per match**: 3 (vs 15)
- **Max steps**: 1000 (vs 5000)
- **Output directory**: `results/`
- **Verbose logging**: Enabled

## Expected Runtime

- **Local version**: ~2-5 minutes
- **Cluster version**: ~30-60 minutes

## Troubleshooting

If you get import errors:
```bash
# Make sure you're in the project root
cd /path/to/MicroRTS-Py-Research

# Then run the local tournament
cd local_testing
python run_local_tournament.py
```

If you get Java/JVM errors:
```bash
# Check Java version
java -version

# Check if microrts.jar exists
ls -la ../gym_microrts/microrts/microrts.jar
```

## Results

Results are saved to `results/` with the same format as the cluster version:
- `tournament_results.json` - Detailed results
- `tournament_summary.txt` - Human-readable summary
- Individual match logs

## Files

- `run_local_tournament.py` - Main Python script
- `run_tournament.sh` - Shell script wrapper
- `README.md` - This file
- `results/` - Output directory (created automatically)
