# GA run logs (local testing only)

This folder is used **only** when you run the local test script:

## How to tell if the GA is working

- **Matches are running** — In `matches.csv` you see real results (e.g. `left_wins: 5`, `right_wins: 0`, `winner: left`). If you see these, the GA is playing games and fitness is being computed.
- **Generations** — `generations.csv` has one row per generation with `best_fitness`, `avg_fitness`, etc. If these numbers are present (e.g. 0.195), evolution is running.
- **UTT changes** — `utt_changes.csv` has at least the initial best UTT (generation 0, `old_value: initial`). When the best individual changes between generations, you get extra rows.
- **Unit composition** — If you see `N/A` in `left_unit_composition` / `right_unit_composition`, games are still running correctly; only the end-of-game unit count capture failed (often fixed by rebuilding `gym_microrts/microrts` JAR with `build_no_ant.sh` or `ant`). Run again and check stderr for `[game_state_utils]` messages to see why.

```bash
python run_ga_local_test.py
```

## Layout

- **`run_history.csv`** — One row per run: timestamp, config, best fitness, balance, duration, diversity, total time. Use this to compare local test runs.

- **`runs/<experiment>_<timestamp>/`** — **One folder per run** (created on each new run). Everything for that run is stored here (like `runtime_utt_change/outputs`):
  - **`generations.csv`** — Per-generation stats (best_fitness, avg_fitness, best_balance, etc.).
  - **`utt_changes.csv`** — What changed in the best UTT from the previous generation (`unit_type`, `param_name`, `old_value`, `new_value`, `affects_ai`).
  - **`matches.csv`** — Every matchup: `ai_left`, `ai_right`, `left_wins`, `right_wins`, `draws`, `winner`, **`left_unit_composition`**, **`right_unit_composition`** (actual end-of-game unit counts when capture is enabled).
  - **`fitness_plot.png`** — Fitness over generations (best and average).
  - **`match_outputs/`** — One `.txt` file per matchup with winner and unit composition (why one side won or lost), similar to `scripts/Running Simulations/runtime_utt_change/outputs`.

Unit composition uses the same code as `runtime_utt_change` (`game_state_utils.get_unit_composition_dict`). For **true end-of-game** counts, rebuild the MicroRTS JAR (`gym_microrts/microrts/build_no_ant.sh` or `ant`); otherwise you may see post-reset (initial) composition.

- **`archive/`** — Old flat files from before the `runs/` layout (e.g. `*_generations.csv`, `*.log`) are moved here to keep the root folder clean.

Cluster runs (`run_ga.py` / `submit_ga.sbatch`) do **not** write here.
