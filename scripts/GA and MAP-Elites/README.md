# MicroRTS Genetic Algorithm

Evolves **unit parameters and global game settings** for balanced MicroRTS via AI vs AI evaluation. Run locally with `run_ga.py` or `run_ga_local_test.py`; cluster submission is in **cluster/**.

**Where things live:** scripts → **run_ga.py** / **run_ga_local_test.py** · code → **core/** · docs → **docs/** · local logs → **ga_run_logs/** · cluster outputs → **cluster_outputs/**

## 🆕 What's New

This is a **complete rewrite** of the previous GA implementation with the following improvements:

- **Cleaner Architecture**: Modular design with separate components for chromosomes, fitness evaluation, genetic operators, and configuration management
- **Better Fitness Function**: Three-component fitness based on balance, duration, and strategy diversity
- **Realistic Simulation**: AI vs AI match simulation for more accurate fitness evaluation
- **Comprehensive Configuration**: Evolves unit stats, costs, timing, and global game parameters
- **Robust Validation**: Built-in configuration validation and error checking
- **Easy Experimentation**: Command-line interface with predefined configurations

## 🚀 Quick Start

### Basic GA Evolution
```bash
python run_ga.py --config fast
```

### Custom Parameters
```bash
python run_ga.py --generations 15 --population 30 --mutation-rate 0.15
```

### Comprehensive Run with Results Saving
```bash
python run_ga.py --config comprehensive --save-results --experiment-name "balance_test"
```

### Local test (short run, logs + CSVs)
```bash
python run_ga_local_test.py
```
Optional: **turn off** per-game snapshots and `match_outputs/*.txt` by setting `SAVE_GAME_DETAILS = False` at the top of `run_ga_local_test.py`, or set env `GA_SAVE_GAME_DETAILS=0` (faster runs, less disk). When on (default), each matchup gets detailed game state and winner/reason per game.

## 📁 Project structure

```
GA and MAP-Elites/
├── README.md                 # This file — start here
├── run_ga.py                 # Main CLI: run GA (local or configs)
├── run_ga_local_test.py      # Local test: short run, logs to ga_run_logs/
├── test_utt.py               # Test a single UTT file with AI vs AI
├── hybrid_ga.py              # Hybrid GA utilities
├── utt_manager.py            # UTT file helpers
│
├── core/                     # GA framework (chromosomes, fitness, evolution)
│   ├── ga_chromosome.py      # Chromosome & unit parameter encoding
│   ├── ga_fitness_evaluator.py
│   ├── ga_working_evaluator.py   # Real MicroRTS evaluation (UTT runs)
│   ├── ga_genetic_operators.py
│   ├── ga_algorithm.py      # Main GA loop
│   ├── ga_config_manager.py
│   └── ga_utt_validator.py
│
├── docs/                     # Extra documentation (design, balance, UTT)
│   └── README.md             # Index of docs
│
├── ga_run_logs/              # Local test outputs (run_ga_local_test.py)
│   ├── run_history.csv       # One row per run
│   ├── runs/<run_id>/        # One folder per run (CSVs, plot, match_outputs/)
│   └── archive/              # Old flat logs moved here
│
├── cluster/                  # Submit GA jobs to cluster (SLURM)
│   ├── submit_ga.sbatch
│   └── README.md
│
├── cluster_outputs/          # Outputs from cluster jobs (ga_runs/, Raw outputs/)
├── experiments/              # Experiment results (saved configs, checkpoints)
└── .gitignore
```

**Entry points:** `run_ga.py` (full runs), `run_ga_local_test.py` (quick local test with logs).

## 🧬 Core Components (New Implementation)

### 1. Chromosome Representation (`ga_chromosome.py`)
- **MicroRTSChromosome**: Complete game configuration as a chromosome
- **UnitParameters**: Evolvable parameters for each unit type
- **GlobalParameters**: Global game settings that can be evolved
- **Genome Encoding**: Conversion between chromosome and genetic representation

### 2. Fitness Evaluation (`ga_fitness_evaluator.py`)
- **FitnessEvaluator**: Three-component fitness function
- **MicroRTSMatchSimulator**: AI vs AI match simulation
- **MatchResult**: Detailed match outcome analysis
- **FitnessComponents**: Balance, duration, and strategy diversity metrics

### 3. Genetic Operators (`ga_genetic_operators.py`)
- **SelectionOperator**: Tournament, rank-based, and elitism selection
- **CrossoverOperator**: Single-point, uniform, and arithmetic crossover
- **MutationOperator**: Gaussian and adaptive mutation
- **GeneticOperators**: Coordinated genetic operations

### 4. Main Algorithm (`ga_algorithm.py`)
- **MicroRTSGeneticAlgorithm**: Complete GA implementation
- **GAConfig**: Comprehensive configuration management
- **GAResults**: Evolution results and statistics
- **GenerationStats**: Per-generation performance metrics

### 5. Configuration Management (`ga_config_manager.py`)
- **MicroRTSConfigConverter**: Convert between GA and MicroRTS formats
- **ExperimentManager**: Experiment storage and retrieval
- **ConfigValidator**: Configuration validation and error checking

## 🎯 What Gets Evolved

### Unit Parameters (Per Unit Type):
- `cost` - Resource cost to produce
- `hp` - Hit points/health
- `minDamage` / `maxDamage` - Attack damage range
- `attackRange` - Attack range
- `produceTime` - Time to produce unit
- `moveTime` - Movement speed
- `attackTime` - Attack speed
- `harvestTime` / `returnTime` / `harvestAmount` - Resource gathering
- `sightRadius` - Vision range

### Global Parameters:
- `moveConflictResolutionStrategy` - How to handle movement conflicts

### What's Frozen (Not Evolved):
- Unit IDs and names
- Boolean capability flags (canAttack, canMove, etc.)
- Production relationships (Base→Worker, Barracks→Combat units)
- Resource unit stats

### Worker harvest bounds (avoid instant-drain economy)

Worker harvest parameters are **bounded** so evolved UTTs stay playable in the GUI and don’t produce “instant drain” economies:

| Parameter        | Bounds (chromosome) | Purpose |
|-----------------|----------------------|--------|
| **harvestTime** | 6–20                 | Harvest takes at least 6 cycles (no near-instant harvest). |
| **returnTime**  | 4–12                 | Return to base takes at least 4 cycles (no 1-cycle return). |
| **harvestAmount** | 1–4                | At most 4 per trip so map resources don’t empty in a few steps. |

Without these bounds, workers can drain the map almost instantly, then idle with no harvest targets and no resources left for the Barracks to train units. Bounds are defined in `core/ga_chromosome.py` (`DEFAULT_PARAMETER_BOUNDS['Worker']`) and enforced in `core/ga_utt_validator.py` (`SAFE_BOUNDS['Worker']`).

## 🤖 AI Agents for Evaluation

The system supports various AI agents for fitness evaluation:
- `randomBiasedAI` - Random actions with bias toward useful actions
- `workerRushAI` - Early worker rush strategy
- `lightRushAI` - Light unit rush strategy
- `passiveAI` - Defensive strategy
- `lightRushAI` - Light unit rush
- `heavyRushAI` - Heavy unit rush

## 🧠 Fitness Function

The new GA uses a **three-component fitness function**:

```
Fitness = α × Balance + β × Duration + γ × StrategyDiversity
```

### Components:

1. **Balance (α = 0.4)**: Measures fairness between AI agents
   - Rewards close matches and draws
   - Penalizes one-sided victories
   - Range: 0-1 (higher is more balanced)

2. **Duration (β = 0.3)**: Evaluates match length appropriateness
   - Target duration: 200 steps
   - Gaussian penalty for being too fast/slow
   - Range: 0-1 (higher is better duration)

3. **Strategy Diversity (γ = 0.3)**: Measures tactical variation
   - Action entropy across matches
   - Unit type diversity
   - Range: 0-1 (higher is more diverse)

## 📊 Command Line Options (New Implementation)

```bash
python run_ga.py [OPTIONS]

Algorithm Configuration:
  --config {default,fast,comprehensive}  Predefined configuration preset
  --generations INT              Number of generations
  --population INT               Population size
  --crossover-rate FLOAT         Crossover probability (0-1)
  --mutation-rate FLOAT          Mutation probability (0-1)
  --mutation-strength FLOAT      Mutation strength (0-1)

Selection Parameters:
  --tournament-size INT          Tournament selection size
  --elite-size INT               Number of elite individuals

Fitness Evaluation:
  --fitness-alpha FLOAT          Balance component weight (0-1)
  --fitness-beta FLOAT           Duration component weight (0-1)
  --fitness-gamma FLOAT          Strategy diversity weight (0-1)
  --target-duration INT          Target match duration in steps
  --duration-tolerance INT       Acceptable duration deviation

Termination Criteria:
  --max-generations-without-improvement INT  Max generations without improvement
  --target-fitness FLOAT         Target fitness value to reach (0-1)

Output and Storage:
  --save-results                 Save results to experiment directory
  --experiment-name STRING       Name for the experiment
  --output-dir STRING            Base directory for experiment storage
  --verbose                      Enable verbose output
  --quiet                        Disable verbose output

Experiment Management:
  --list-experiments             List all available experiments
  --load-experiment PATH         Load and display results from previous experiment
  --compare-experiments PATH...  Compare multiple experiments

Analysis and Visualization:
  --analyze-best                 Analyze the best individual configuration
  --export-config PATH           Export best configuration to MicroRTS format
  --validate-config              Validate the best configuration
```

## 🔬 Example Experiments (New Implementation)

### 1. Quick Test (2-3 minutes)
```bash
python run_ga.py --config fast
```

### 2. Balanced Configuration (10-15 minutes)
```bash
python run_ga.py --generations 10 --population 20 --fitness-alpha 0.5 --fitness-beta 0.3 --fitness-gamma 0.2
```

### 3. Comprehensive Run (30-45 minutes)
```bash
python run_ga.py --config comprehensive --save-results --experiment-name "comprehensive_test"
```

### 4. Custom Focus on Duration (15-20 minutes)
```bash
python run_ga.py --generations 12 --population 25 --fitness-alpha 0.3 --fitness-beta 0.5 --fitness-gamma 0.2 --target-duration 250
```

### 5. High Diversity Focus (20-25 minutes)
```bash
python run_ga.py --generations 15 --population 30 --fitness-alpha 0.3 --fitness-beta 0.2 --fitness-gamma 0.5 --mutation-rate 0.15
```

### 6. Experiment Management
```bash
# List all experiments
python run_ga.py --list-experiments

# Load and analyze previous results
python run_ga.py --load-experiment experiments/comprehensive_test_1234567890

# Export best configuration
python run_ga.py --load-experiment experiments/comprehensive_test_1234567890 --export-config best_config.json --validate-config
```

## 🎮 How It Works (New Implementation)

1. **Initialization**: Create random population of MicroRTS game configurations
2. **Evaluation**: Test each configuration via AI vs AI match simulation
3. **Fitness Calculation**: Compute balance, duration, and strategy diversity scores
4. **Selection**: Choose best-performing configurations as parents (tournament selection)
5. **Crossover**: Combine parent configurations to create offspring
6. **Mutation**: Randomly modify configuration parameters
7. **Elitism**: Preserve best individuals across generations
8. **Replacement**: Replace old population with new generation
9. **Convergence Check**: Stop if target fitness reached or no improvement
10. **Repeat**: Continue until termination criteria met

## 📈 Expected Results (New Implementation)

- **Fitness Improvement**: Better configurations should achieve higher overall fitness scores
- **Balanced Gameplay**: Evolved configurations should produce more fair matches between AIs
- **Appropriate Duration**: Matches should converge toward target duration (200 steps)
- **Strategic Diversity**: Evolved configurations should encourage varied tactical approaches
- **Parameter Optimization**: Unit costs, stats, and timing should be optimized for balance
- **Convergence**: Algorithm should converge to stable, high-quality solutions

## 🛠️ Requirements

- Python 3.7+
- MicroRTS environment
- Required packages: numpy, matplotlib, seaborn (optional)

## 🚨 Notes

- AI-based evaluation is slower but more realistic
- Multi-objective evaluation is faster but less realistic
- Results may vary due to randomness in AI behavior
- Longer runs (more generations) generally produce better results