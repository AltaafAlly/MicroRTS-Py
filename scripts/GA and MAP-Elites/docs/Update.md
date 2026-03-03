# Update 1 — 17/02/2026

## Why the results were bad

### All-draw was scored as best balance
When every game was a draw (0-0-3), the code set **balance = 1.0**. So the GA was rewarded for UTTs that never produce a winner.

### Decisive matchups were scored worse
Results like 3-0-0 or 0-3-0 got balance = 0 and lower fitness, so the GA was pushed away from UTTs that produce clear wins/losses.

### Population converged to the wrong behavior
By generation 2, everyone was 0-0-3 and “best” fitness was the all-draw case. The GA was effectively optimizing for “games don’t end” instead of “50–50 win rate.”

---

## Change made

In **`core/ga_working_evaluator.py`**, the all-draw case is no longer treated as perfect balance:

| Case | Before | After |
|------|--------|-------|
| `left_wins == 0` and `right_wins == 0` (all draws) | balance = 1.0 | balance = 0.0 |

So:
- **0-0-3** (all draws) now gets **low** balance and low overall fitness.
- **2-1** or **1-2** (decisive but close) gets **high** balance and higher fitness.

The GA should now favor UTTs that give close, decisive matchups (e.g. 2-1, 1-2) instead of UTTs that make every game a draw.

---

# Update 2 — 17/02/2026

## Why results were still not good (all 3-0 or 0-3, balance stuck at 0)

With **only 3 games per evaluation**, outcomes are almost always **3-0** or **0-3**, so balance stays 0 and fitness is stuck at 0.39. We almost never see **2-1** or **1-2**, which are the only outcomes that give a non-zero balance score in 3 games.

## Change made

In **`run_ga_local_test.py`** (local test only):

| Setting           | Before | After   |
|-------------------|--------|--------|
| `GAMES_PER_EVAL`  | 3      | 5      |
| `GENERATIONS`     | 3      | 5      |
| `POPULATION`      | 4      | 6      |
| `MAX_STEPS`       | 1e6    | 20_000 |

- **More games per eval (5)** so we can get 3-2, 2-3, 4-1, etc. and a real balance signal.
- **Slightly more generations and population** so the GA has more search to find UTTs that balance the matchup.
- **Lower max_steps** so local runs stay fast; increase again for cluster if needed.

---

# Update 3 — 28/01/2026 (reverted)

## High-resource map made things worse

We added `basesWorkers8x8_high_resources.xml` (50k starting resources, 999999 per node) to avoid resource-starvation draws. In practice **every** game then ran to the step limit and ended in a draw: with effectively infinite resources, both AIs keep building and never eliminate each other within 100k steps. So the GA saw only 0-0-5 and fitness dropped (all 0.21).

## Revert

- **Map** set back to **`maps/8x8/basesWorkers8x8A.xml`** (original 5 + 20 per node). We again get a mix of decisive games (5-0-0, etc.) and some draws.
- **Local test** `MAX_STEPS` set to **20_000** so when a game does draw we cap at 20k steps instead of 100k, keeping runs faster.
- The file `basesWorkers8x8_high_resources.xml` remains in `gym_microrts/microrts/maps/8x8/` for optional use (e.g. `--map_path maps/8x8/basesWorkers8x8_high_resources.xml`) if you want to experiment.

---

# Update 4 — 28/01/2026

## Gen 3–4 completing in ~0.08s (evaluations not actually running)

When reusing a single env across all generations, later generations (e.g. 3 and 4) reported “Population evaluation completed in 0.08 seconds” for 6×5 games — impossible for real simulations. The reused Java client was likely returning cached or stale state.

## Changes made

1. **Close cached env at the start of each generation**  
   In `evaluate_population_fitness_working()`, call `evaluator.close_cached_env()` before the loop over chromosomes. Each generation then creates a new env on the first evaluation and reuses it only within that generation (at most 6 evals per env). No reuse across generations.

2. **Sanity check and re-run on suspicious total_steps**  
   In `_test_utt_file()`, if `total_steps` is present and &lt; 10×number of games (e.g. &lt; 50 for 5 games), the run is treated as suspicious (possibly cached). We close the cached env and re-run the matchup once with a fresh env and use that result.

3. **Unique UTT path per evaluation (Update 4b)**  
   The Java client was still returning instantly for evals 2–6 within a generation when reusing the env and loading `utts/test_utt.json` again (path-based caching). We now copy each chromosome’s UTT to a **unique file** (`test_utt_<time_ns>.json`) and pass that path to `change_utt()`. Each evaluation then loads a different path, so the client cannot return a cached result. `_copy_utt_to_microrts()` accepts an optional `unique_suffix`; `_run_match_with_utt()` uses `"utts/" + utt_path.name` for both the initial env and `change_utt()`.

4. **No env reuse: fresh env per evaluation (Update 4c)**  
   Even with unique UTT paths, reusing the same env (change_utt + run_games_with_env) still led to Gen 1/3/4 completing in ~0.1s. So we **disabled reuse**: `_run_match_with_utt()` now always calls `run_pair()` (create env, run games, close client). Every evaluation gets a brand-new env. Runs are slower but each generation’s timings should reflect real simulation time. Cached-env logic and `run_games_with_env` are no longer used in the evaluator; `close_cached_env` remains for cleanup but is effectively a no-op.

---

# Update 5 — Multi-map evaluation for non-zero balance

## Why it’s always 5-0 or 0-5 (balance stays 0)

With **deterministic AIs** (LightRush, WorkerRush), the same UTT gives the same winner every game. So 5 games on one map → always 5-0 or 0-5, and the balance term never gets a gradient toward 50–50.

## Change: evaluate on multiple maps and aggregate

- **`WorkingGAEvaluator`** now accepts optional **`map_paths`** (list of map paths). If set, each matchup is run on **every** map (same games per map), and results are **aggregated** (sum of left_wins, right_wins, draws). So e.g. 5 games × 3 maps = 15 games per pair; map A might give 5-0, map B 0-5, map C 3-2 → **8-7-0** and a **non-zero balance**.
- **`GAConfig`** has **`map_paths`** (optional). When present, the evaluator uses it; otherwise it uses a single `map_path` as before.
- **Local test** uses **3 maps** (A, Obstacle, L) so each evaluation can produce mixed totals (e.g. 2-1, 1-2) and balance &gt; 0 when obstacles/different layout flip the winner for the same UTT.

---

# Update 6 — Weaker left agent for more balanced matchup

## LightRush was too strong vs WorkerRush

With **LightRush vs WorkerRush**, LightRush tended to dominate, so outcomes were often 5-0 or 0-5 and the GA had little gradient toward 50–50.

## Change: use RangedRush vs WorkerRush (deterministic, plays actively)

- **`run_ga_local_test.py`** and **`run_ga.py`** (when `--single_matchup`) now use **`rangedRushAI`** vs **`workerRushAI`** instead of LightRush vs WorkerRush.
- **RandomBiased** is random → noisy balance signal. **Passive** barely acts → poor signal. **RangedRush** is a full scripted rush (builds barracks, ranged units, attacks), deterministic, and often weaker than LightRush vs WorkerRush so UTT/maps can produce mixed results (2-1, 1-2). **heavyRushAI** and **rangedRushAI** were added to `gym_microrts/microrts_ai.py` for use as alternatives.
