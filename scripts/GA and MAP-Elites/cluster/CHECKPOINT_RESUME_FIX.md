# Checkpoint Resume Fix

## Problem Identified

When a GA job hits the 3-day time limit and is resubmitted, it was **not automatically resuming from the checkpoint** even though checkpoints were being saved.

### Root Cause

Each SLURM job creates a **new run directory** with a unique timestamp:
```
/home-mscluster/aally/ga_runs/ga_experiment_20260110_140337_job172717/
/home-mscluster/aally/ga_runs/ga_experiment_20260114_135210_job173459/
```

The checkpoint detection logic was only searching in the **current run directory** (`${EXPERIMENT_OUTPUT_DIR}`), which is empty for a new job. The checkpoints from the previous job are in a **different directory**, so they weren't found.

### Evidence

From the output files:
- **ga_evolution_172717.out**: Reached time limit, saved checkpoints up to generation 11
- **ga_evolution_173459.out**: Started new run, showed "No checkpoint found. Starting new run." (line 53)

Both jobs should have been able to resume, but the second job couldn't find the checkpoints from the first job.

## Solution

Modified `submit_ga.sbatch` to search for checkpoints in **ALL previous run directories**, not just the current one:

1. **Search in parent directory**: Look in `/home-mscluster/${USER}/ga_runs/` recursively
2. **Prefer same experiment**: First try to find checkpoints from the same experiment name
3. **Fallback to any**: If no matching experiment found, use the most recent checkpoint
4. **Better logging**: Show which checkpoint was found and when it was modified

### Changes Made

**File**: `scripts/GA and MAP-Elites/cluster/submit_ga.sbatch`

**Before**:
```bash
LATEST_CHECKPOINT=$(find ${EXPERIMENT_OUTPUT_DIR} -name "checkpoint_latest.json" -type f 2>/dev/null | sort -r | head -1)
```

**After**:
```bash
GA_RUNS_BASE="/home-mscluster/${USER}/ga_runs"
# First try to find checkpoint from same experiment name
EXPERIMENT_CHECKPOINT=$(find ${GA_RUNS_BASE} -path "*/experiments/${EXPERIMENT_NAME}_*/checkpoints/checkpoint_latest.json" -type f 2>/dev/null | sort -r | head -1)

# If not found, use any latest checkpoint
if [ -z "${EXPERIMENT_CHECKPOINT}" ]; then
    LATEST_CHECKPOINT=$(find ${GA_RUNS_BASE} -name "checkpoint_latest.json" -type f 2>/dev/null | sort -r | head -1)
else
    LATEST_CHECKPOINT="${EXPERIMENT_CHECKPOINT}"
fi
```

## How It Works Now

1. **First job** (e.g., job 172717):
   - Creates directory: `ga_experiment_20260110_140337_job172717/`
   - Saves checkpoints: `experiments/cluster_long_run_XXXXX/checkpoints/checkpoint_latest.json`
   - Hits time limit after generation 11

2. **Resubmit job** (e.g., job 173459):
   - Creates NEW directory: `ga_experiment_20260114_135210_job173459/`
   - Searches in `/home-mscluster/aally/ga_runs/` for `checkpoint_latest.json`
   - Finds checkpoint from previous job
   - **Automatically resumes** from generation 12

## Testing

To verify the fix works:

1. **Submit a job**:
   ```bash
   sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
   ```

2. **Wait for it to save at least one checkpoint** (after generation 0 completes)

3. **Cancel the job** (simulate time limit):
   ```bash
   scancel <JOB_ID>
   ```

4. **Resubmit the job**:
   ```bash
   sbatch scripts/GA\ and\ MAP-ELites/cluster/submit_ga.sbatch
   ```

5. **Check the output** - should see:
   ```
   ==========================================
   AUTOMATIC RESUME DETECTED
   ==========================================
   Found checkpoint: /home-mscluster/aally/ga_runs/.../checkpoint_latest.json
   Resuming from checkpoint automatically...
   ==========================================
   ```

6. **Verify it resumes** - should start from generation 1 (or whatever the last checkpoint was)

## Notes

- **Experiment name matching**: The script prefers checkpoints from the same experiment name (`cluster_long_run` by default), but will use any checkpoint if none match
- **Multiple experiments**: If you run multiple experiments with different names, each will only resume from its own checkpoints
- **Manual resume**: You can still manually specify a checkpoint with `--resume-from` if needed
- **Checkpoint location**: Checkpoints are saved in `experiments/<EXPERIMENT_NAME>_<TIMESTAMP>/checkpoints/`

## Related Files

- `scripts/GA and MAP-Elites/cluster/submit_ga.sbatch` - SLURM batch script with checkpoint detection
- `scripts/GA and MAP-Elites/core/ga_algorithm.py` - GA algorithm with checkpoint save/load
- `scripts/GA and MAP-Elites/run_ga.py` - Main runner script that accepts `--resume-from`
