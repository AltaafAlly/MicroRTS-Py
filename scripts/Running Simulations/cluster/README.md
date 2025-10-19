# Cluster Testing and Execution

This folder contains scripts and configurations for running MicroRTS tournaments and tests on a compute cluster.

## Files

### Test Scripts
- **`simple_cluster_test.py`** - Basic cluster compatibility test that checks:
  - Basic imports (gym_microrts, JPype)
  - JVM startup and environment creation
  - Custom UTT loading
  - Single match execution
- **`run_manual_test.sh`** - Shell script to run the simple cluster test manually

### SLURM Batch Scripts
- **`test_cluster.sbatch`** - SLURM batch script for running the simple cluster test
- **`submit_tournament.sbatch`** - SLURM batch script for running the full UTT impact tournament

## Usage

### Local Testing
```bash
# Run the simple cluster test locally
./run_manual_test.sh

# Or directly with Python
python simple_cluster_test.py
```

### Cluster Submission
```bash
# Submit simple test job
sbatch test_cluster.sbatch

# Submit full tournament job
sbatch submit_tournament.sbatch
```

## Cluster Configuration

The scripts are configured for the mscluster environment with:
- Partition: `bigbatch`
- Output directory: `/home-mscluster/%u/`
- Conda environment: `microrts39`
- Java: OpenJDK 17 (via conda)

## Troubleshooting

If tests fail, check:
1. Java version compatibility
2. JAR file compilation and classpath
3. Python dependencies (PIL/Pillow)
4. File paths and permissions
