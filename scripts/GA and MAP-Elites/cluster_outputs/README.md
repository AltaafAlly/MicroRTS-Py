# Cluster run outputs

Outputs from GA runs submitted via **cluster/** (e.g. `submit_ga.sbatch`).

- **ga_runs/** — One folder per job: `ga_experiment_<date>_job<id>/` (checkpoints, configs, best UTT).
- **Raw outputs/** — SLURM stdout/stderr logs (`ga_evolution_<jobid>.out`, `.err`).
- **cluster_long_run_*** — Legacy single-run results (config, checkpoints, best_microrts_config.json).

These are **not** used by local runs; local runs use **ga_run_logs/**.
