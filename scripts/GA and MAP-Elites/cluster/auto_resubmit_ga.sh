#!/bin/bash
# Auto-resubmit wrapper for GA evolution jobs
# This script monitors and automatically resubmits the GA job when it hits time limits
# Usage: ./auto_resubmit_ga.sh [max_resubmits]

set -euo pipefail

# Maximum number of times to resubmit (default: unlimited, set to 0 for unlimited)
MAX_RESUBMITS=${1:-0}
RESUBMIT_COUNT=0

# Track the original job ID
ORIGINAL_JOB_ID=""

echo "=========================================="
echo "GA Auto-Resubmit Wrapper"
echo "=========================================="
echo "This script will automatically resubmit the GA job"
echo "when it hits the time limit until completion."
echo ""

# Function to submit the job
submit_job() {
    local job_id
    job_id=$(sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch | grep -oP '\d+')
    
    if [ -z "$job_id" ]; then
        echo "ERROR: Failed to submit job"
        exit 1
    fi
    
    echo "Submitted job ID: $job_id"
    
    if [ -z "$ORIGINAL_JOB_ID" ]; then
        ORIGINAL_JOB_ID=$job_id
        echo "Original job ID: $ORIGINAL_JOB_ID"
    fi
    
    echo $job_id
}

# Function to check if job completed successfully
check_job_completion() {
    local job_id=$1
    local job_state
    
    # Check if job is still running
    job_state=$(squeue -j $job_id -h -o %T 2>/dev/null || echo "COMPLETED")
    
    if [ "$job_state" = "COMPLETED" ] || [ "$job_state" = "CANCELLED" ] || [ "$job_state" = "TIMEOUT" ]; then
        # Job finished, check if it completed all generations
        local output_file="/home-mscluster/${USER}/job_logs/ga_evolution_${job_id}.out"
        
        if [ -f "$output_file" ]; then
            # Check if we see "EVOLUTION COMPLETED" in the output
            if grep -q "EVOLUTION COMPLETED" "$output_file" 2>/dev/null; then
                echo "✅ Job $job_id completed successfully!"
                return 0  # Successfully completed
            else
                echo "⚠️  Job $job_id finished but may not have completed all generations"
                echo "   Checking for checkpoint..."
                return 1  # Need to resubmit
            fi
        else
            echo "⚠️  Job $job_id finished but output file not found"
            return 1  # Need to resubmit
        fi
    else
        # Job still running
        return 2  # Still running
    fi
}

# Function to wait for job and monitor
wait_and_monitor() {
    local job_id=$1
    
    echo ""
    echo "Monitoring job $job_id..."
    echo "You can check status with: squeue -j $job_id"
    echo "You can view output with: tail -f /home-mscluster/${USER}/job_logs/ga_evolution_${job_id}.out"
    echo ""
    
    # Wait for job to complete
    while true; do
        sleep 60  # Check every minute
        
        check_job_completion $job_id
        local status=$?
        
        if [ $status -eq 0 ]; then
            # Successfully completed
            echo ""
            echo "=========================================="
            echo "✅ EVOLUTION COMPLETED SUCCESSFULLY!"
            echo "=========================================="
            echo "Final job ID: $job_id"
            echo "Total resubmits: $RESUBMIT_COUNT"
            return 0
        elif [ $status -eq 1 ]; then
            # Finished but needs resubmit
            echo ""
            echo "Job $job_id finished. Checking if resubmit is needed..."
            
            # Check max resubmits
            if [ $MAX_RESUBMITS -gt 0 ] && [ $RESUBMIT_COUNT -ge $MAX_RESUBMITS ]; then
                echo "❌ Maximum resubmits ($MAX_RESUBMITS) reached. Stopping."
                return 1
            fi
            
            RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1))
            echo "Resubmitting... (attempt $RESUBMIT_COUNT)"
            
            # Wait a bit before resubmitting
            sleep 5
            
            local new_job_id=$(submit_job)
            job_id=$new_job_id
            echo "Continuing with job $job_id..."
        fi
        # If status is 2, job is still running, continue loop
    done
}

# Main execution
echo "Submitting initial job..."
INITIAL_JOB_ID=$(submit_job)

echo ""
echo "Starting auto-resubmit monitoring..."
echo "Press Ctrl+C to stop monitoring (job will continue running)"
echo ""

# Trap Ctrl+C to allow graceful exit
trap 'echo ""; echo "Monitoring stopped. Job $job_id will continue running."; exit 0' INT

wait_and_monitor $INITIAL_JOB_ID

