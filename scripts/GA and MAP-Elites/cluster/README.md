# GA Cluster Execution

This folder contains scripts for running the Genetic Algorithm (GA) on a compute cluster.

## Files

### SLURM Batch Scripts
- **`submit_ga.sbatch`** - SLURM batch script for running long GA evolution runs on the cluster

### Manual Test Scripts
- **`run_ga_manual.sh`** - Shell script to run the GA manually (for testing or local execution)
- **`sync_to_cluster.sh`** - Script to sync code to the cluster using rsync
- **`auto_resubmit_ga.sh`** - (Optional) Auto-resubmit wrapper for automatic resubmission (requires screen/tmux)

## Usage

### Syncing Code to Cluster

Before running the GA on the cluster, sync your code:

```bash
# Using the sync script (recommended)
cd /path/to/MicroRTS-Py-Research
./scripts/GA\ and\ MAP-Elites/cluster/sync_to_cluster.sh

# Or manually with rsync
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='experiments/' \
    --exclude='results/' \
    --exclude='gym_microrts/microrts/microrts.jar' \
    ./ username@cluster:~/Research/MicroRTS-Py-Research/
```

The sync script automatically excludes:
- Git files, Python cache, compiled files
- Experiment results and logs
- JAR files (will be rebuilt on cluster)
- IDE files and temporary files

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
# Submit with default parameters (20 generations, 25 population)
sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
```

#### Resuming After Time Limit
When the job hits the 3-day time limit, simply resubmit with the same command:
```bash
# Job finished at generation 12? Just resubmit:
sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch

# The script automatically detects and resumes from the latest checkpoint
# No need to find or specify the checkpoint file - it's automatic!
```

**Note**: The checkpoint detection and resume happens automatically - you just need to resubmit the job manually when it finishes.

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

### Balance Improvements (v2)

**New in this version**: Improved balance calculation with stricter penalties for imbalanced matchups.

#### Key Improvements:
1. **Increased Balance Weight**: Default `alpha` (balance weight) increased from 0.4 to 0.5
   - Balance now has 50% weight (up from 40%)
   - Duration and diversity weights reduced to 25% each (from 30%)

2. **Strict Balance Penalties**: Exponential penalties for very imbalanced matchups
   - Matchups with >70-30 split get exponential penalty: `(imbalance/0.5)^2`
   - Matchups with 60-40 to 70-30 split get quadratic penalty: `(imbalance/0.5)^1.5`
   - This means a single 5-0-0 matchup will significantly hurt the overall balance score

3. **Geometric Mean Aggregation**: Balance scores now use geometric mean instead of arithmetic mean
   - Geometric mean is more sensitive to very low scores
   - One very imbalanced matchup (e.g., 5-0-0) will significantly lower the overall score
   - This encourages the GA to find configurations where ALL matchups are reasonably balanced

4. **Minimum Balance Threshold**: Configurable threshold (default 0.2)
   - If any matchup is below the threshold, an additional penalty is applied
   - This prevents configurations with extremely imbalanced matchups from surviving

#### Example Impact:
**Before** (arithmetic mean, linear penalty):
- Matchups: [0.0, 0.0, 0.3, 0.5, 0.7, ...] (some very imbalanced)
- Average balance: ~0.28 (low but not terrible)

**After** (geometric mean, exponential penalty):
- Same matchups: [0.0, 0.0, 0.15, 0.3, 0.5, ...] (penalized more)
- Geometric mean balance: ~0.15 (much lower, encourages improvement)
- GA will evolve away from configurations with any very imbalanced matchups

#### Expected Results:
- **Higher balance scores**: Target 0.4-0.5+ (up from 0.28)
- **Fewer one-sided matchups**: Expect fewer 5-0-0 or 0-5-0 results
- **More balanced overall**: Even if some matchups favor one side, they should be closer to 60-40 or better
- **Slower convergence**: May take more generations to find good solutions, but solutions will be more balanced

### Default Parameters
- **Generations**: 20
- **Population**: 25
- **AI agents**: 8 AIs (workerRushAI, lightRushAI, coacAI, naiveMCTSAI, droplet, mixedBot, passiveAI, randomBiasedAI)
- **Games per evaluation**: 5 (per AI pair, so 140 total games per chromosome with 28 pairs)
- **Note**: Expanded from 6 to 8 AIs (87% more matchups) for better balance evaluation
- **Max steps per game**: 5000
- **Experiment name**: "cluster_long_run"
- **Time limit**: 3 days (with automatic checkpoint/resume)
- **Partition**: bigbatch

### Customizing Parameters

You can customize the GA run by setting environment variables:

- `GA_GENERATIONS` - Number of generations to evolve (default: 20)
- `GA_POPULATION` - Population size (default: 25)
- `GA_EXPERIMENT_NAME` - Name for the experiment (default: "cluster_long_run")
- `GA_GAMES_PER_EVAL` - Games per AI pair (default: 5, total = 5 × 28 pairs = 140 games per chromosome with 8 AIs)
- `GA_MAX_STEPS` - Maximum steps per game (default: 5000)

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
│       ├── best_microrts_config.json
│       └── checkpoints/
│           ├── checkpoint_gen_0.json
│           ├── checkpoint_gen_1.json
│           ├── ...
│           └── checkpoint_latest.json  # Always points to most recent
├── ga_stdout.log
└── microrts.jar (rebuilt for cluster)
```

## Checkpointing and Resuming

The GA automatically saves checkpoints after each generation, allowing you to resume long-running jobs that hit time limits.

### How It Works

1. **Automatic Checkpointing**: After each generation, the GA saves:
   - Current population
   - Fitness scores
   - Best individual found so far
   - Generation statistics
   - Convergence tracking state

2. **Semi-Automatic Resume**: 
   - **You need to**: Manually resubmit the job after it hits the time limit
   - **The script automatically**: 
     - Searches for the latest checkpoint (`checkpoint_latest.json`)
     - Resumes from that checkpoint if found
     - Continues evolution from the next generation
   - **No manual work needed**: You don't need to find or specify the checkpoint file

3. **Manual Resume** (if you want more control): You can also resume manually:
   ```bash
   python run_ga.py --resume-from /path/to/checkpoint_latest.json [other args...]
   ```

### Benefits

- **No Lost Progress**: If a job hits the 3-day time limit, simply resubmit and it continues
- **Incremental Evolution**: Run for 3 days, analyze results, then continue if needed
- **Fault Tolerance**: If the job crashes, you can resume from the last checkpoint

### Example: Resuming After Time Limit

**Simple workflow - just resubmit when the job finishes:**

```bash
# Step 1: Submit initial job
sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
# Job ID: 12345

# Step 2: Wait for job to finish (check status)
squeue -u $USER
# or check the output log: tail -f /home-mscluster/${USER}/job_logs/ga_evolution_*.out

# Step 3: When job finishes (hits time limit), simply resubmit:
sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
# Job ID: 12346
# The script automatically:
# 1. Detects checkpoint_latest.json from the previous run
# 2. Loads the population and state
# 3. Continues from the next generation
# 4. No manual work needed - checkpoint detection is automatic!

# Step 4: If it hits time limit again, repeat step 3
sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
# Continues seamlessly...
```

**What happens:**
- ✅ **Automatic**: Checkpoint detection and resume (you don't need to find the checkpoint file)
- ⚠️ **Manual**: You need to resubmit the job using `sbatch` after it stops (simple one-line command)
- ✅ **Automatic**: The script finds the latest checkpoint and resumes seamlessly

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
- **Automatic Resume**: The batch script automatically detects and resumes from checkpoints. Simply resubmit the same job:
  ```bash
  sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
  ```
  The script will automatically find the latest checkpoint and resume from where it left off.
- **Manual Resume**: If you need to resume manually:
  ```bash
  # Find the latest checkpoint
  LATEST_CHECKPOINT=$(find ~/Research/MicroRTS-Py-Research/experiments -name "checkpoint_latest.json" | sort -r | head -1)
  
  # Resume from checkpoint
  cd ~/Research/MicroRTS-Py-Research/scripts/GA\ and\ MAP-Elites
  python run_ga.py \
      --resume-from "$LATEST_CHECKPOINT" \
      --generations 20 \
      --population 25 \
      --experiment-name "cluster_long_run" \
      --use-working-evaluator \
      --verbose
  ```
- **Alternative**: Reduce generations or population, reduce games_per_eval, or increase time limit in sbatch script: `#SBATCH --time=3-00:00:00` (3 days)

## Performance Tips

1. **For faster runs**: Reduce `GA_GAMES_PER_EVAL` to 3 (84 total games per chromosome with 8 AIs, 28 pairs)
2. **For better results**: Increase `GA_GAMES_PER_EVAL` to 7-10 (196-280 total games per chromosome with 8 AIs)
3. **For exploration**: Increase `GA_POPULATION` to 40+
4. **For convergence**: Increase `GA_GENERATIONS` to 30+
5. **With checkpointing**: You can now afford higher `GA_GAMES_PER_EVAL` values (5-10) since jobs can resume after time limits

