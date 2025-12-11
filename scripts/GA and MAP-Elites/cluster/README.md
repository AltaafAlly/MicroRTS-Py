# GA Cluster Execution

This folder contains scripts for running the Genetic Algorithm (GA) on a compute cluster.

## Files

### SLURM Batch Scripts
- **`submit_ga.sbatch`** - SLURM batch script for running long GA evolution runs on the cluster

### Manual Test Scripts
- **`run_ga_manual.sh`** - Shell script to run the GA manually (for testing or local execution)

## Usage

### Local/Manual Testing
```bash
# Run a quick GA test locally
cd /path/to/MicroRTS-Py-Research
./scripts/GA\ and\ MAP-Elites/cluster/run_ga_manual.sh

# Or with custom parameters
export GA_GENERATIONS=10
export GA_POPULATION=20
export GA_EXPERIMENT_NAME="my_test"
./scripts/GA\ and\ MAP-Elites/cluster/run_ga_manual.sh
```

### Cluster Submission

#### Basic Submission
```bash
# Submit with default parameters (30 generations, 30 population)
sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
```

#### Custom Parameters via Environment Variables
```bash
# Set custom parameters before submission
export GA_GENERATIONS=50
export GA_POPULATION=40
export GA_EXPERIMENT_NAME="long_evolution"
export GA_GAMES_PER_EVAL=5
export GA_MAX_STEPS=2000

# Submit the job
sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
```

#### Inline Parameter Submission
```bash
# Submit with inline environment variables
GA_GENERATIONS=50 GA_POPULATION=40 GA_EXPERIMENT_NAME="custom_run" \
    sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
```

## Configuration

### Default Parameters
- **Generations**: 30
- **Population**: 30
- **Games per evaluation**: 3 (per AI pair, so 18 total games per chromosome)
- **Max steps per game**: 1000
- **Experiment name**: "cluster_long_run"
- **Time limit**: 24 hours
- **Partition**: bigbatch

### Customizing Parameters

You can customize the GA run by setting environment variables:

- `GA_GENERATIONS` - Number of generations to evolve (default: 30)
- `GA_POPULATION` - Population size (default: 30)
- `GA_EXPERIMENT_NAME` - Name for the experiment (default: "cluster_long_run")
- `GA_GAMES_PER_EVAL` - Games per AI pair (default: 3, total = 3 × 6 pairs = 18 games per chromosome)
- `GA_MAX_STEPS` - Maximum steps per game (default: 1000)

### Example: Long Evolution Run
```bash
# Run for 100 generations with larger population
export GA_GENERATIONS=100
export GA_POPULATION=50
export GA_EXPERIMENT_NAME="long_evolution_100gen"
export GA_GAMES_PER_EVAL=5  # More games = more accurate but slower

sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
```

## Output

Results are saved to:
```
/home-mscluster/${USER}/ga_runs/ga_experiment_YYYYMMDD_HHMMSS_job${SLURM_JOB_ID}/
├── experiments/
│   └── ${EXPERIMENT_NAME}_${TIMESTAMP}/
│       ├── ga_config.json
│       ├── ga_results.json
│       └── best_microrts_config.json
├── ga_stdout.log
└── microrts.jar (rebuilt for cluster)
```

## Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### View Output
```bash
# View latest output
tail -f /home-mscluster/${USER}/job_logs/ga_evolution_*.out

# View errors
tail -f /home-mscluster/${USER}/job_logs/ga_evolution_*.err
```

### Check Results
```bash
# Find the latest experiment
LATEST_RUN=$(ls -td /home-mscluster/${USER}/ga_runs/ga_experiment_* | head -1)
echo "Latest run: $LATEST_RUN"

# View experiment summary
python -c "
import json
import sys
with open('$LATEST_RUN/experiments/*/ga_results.json', 'r') as f:
    results = json.load(f)
print(f\"Best Fitness: {results['best_fitness']['overall_fitness']:.4f}\")
print(f\"Generations: {results['total_generations']}\")
"
```

## Cluster Configuration

The scripts are configured for the mscluster environment with:
- Partition: `bigbatch`
- Output directory: `/home-mscluster/%u/`
- Conda environment: `microrts39`
- Java: OpenJDK (via conda or system)
- Time limit: 24 hours (adjustable in sbatch script)

## Troubleshooting

### Job Fails Immediately
- Check Java version compatibility
- Verify conda environment is activated
- Check that microrts.jar exists and is valid

### GA Runs But Gets Low Fitness
- This is normal for early generations
- Check that matches are completing (not all draws)
- Verify AI agents are working correctly

### Out of Memory
- Reduce population size
- Reduce games_per_eval
- Reduce max_steps

### Time Limit Exceeded
- Reduce generations or population
- Reduce games_per_eval
- Increase time limit in sbatch script: `#SBATCH --time=48:00:00`

## Performance Tips

1. **For faster runs**: Reduce `GA_GAMES_PER_EVAL` to 2 (12 total games per chromosome)
2. **For better results**: Increase `GA_GAMES_PER_EVAL` to 5 (30 total games per chromosome)
3. **For exploration**: Increase `GA_POPULATION` to 50+
4. **For convergence**: Increase `GA_GENERATIONS` to 50+

