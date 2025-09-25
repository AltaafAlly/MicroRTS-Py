# UTT Evolution with Genetic Algorithm and MAP-Elites

This project implements a genetic algorithm and MAP-Elites algorithm for evolving Unit Type Table (UTT) configurations in MicroRTS. The system uses AI-based fitness evaluation to determine the quality of evolved UTT configurations.

## 🚀 Quick Start

### Basic GA Evolution
```bash
python run_utt_evolution.py --generations 10 --population 20 --ai randomBiasedAI
```

### MAP-Elites Evolution
```bash
python run_utt_evolution.py --algorithm map-elites --generations 10 --population 20 --ai workerRushAI
```

### Multi-Objective Fitness (Faster)
```bash
python run_utt_evolution.py --fitness-type multi-objective --generations 20 --population 30
```

## 📁 Project Structure

```
GA and MAP-Elites/
├── core/                           # Core framework
│   ├── utt_genetic_algorithm.py   # GA implementation
│   ├── utt_map_elites.py          # MAP-Elites implementation
│   ├── improved_fitness_evaluator.py # AI-based fitness evaluation
│   ├── utt_utils.py               # Utility functions
│   └── utt_evolution_config.json  # Configuration file
├── run_utt_evolution.py           # Main experiment script
├── main.py                        # Alternative entry point
└── README.md                      # This file
```

## 🧬 Core Components

### 1. Genetic Algorithm (`utt_genetic_algorithm.py`)
- **UTTGeneEncoder**: Encodes/decodes UTT parameters to/from genome
- **Individual**: Represents a single UTT configuration
- **GeneticAlgorithm**: Main GA implementation with selection, crossover, mutation

### 2. MAP-Elites (`utt_map_elites.py`)
- **MAPElitesAlgorithm**: Quality-diversity evolution
- **BehaviorDescriptorExtractor**: Extracts behavior descriptors for archive
- **MAPElitesArchive**: Manages the archive of diverse solutions

### 3. Fitness Evaluation (`improved_fitness_evaluator.py`)
- **AIGameSimulationFitness**: Real AI vs AI game evaluation
- **MultiObjectiveFitness**: Fast design-principle-based evaluation

### 4. Utilities (`utt_utils.py`)
- **UTTGenerator**: Creates and modifies UTT configurations
- **UTTAnalyzer**: Analyzes UTT properties and statistics
- **UTTVisualizer**: Creates visualizations of UTT configurations

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

## 🤖 AI Agents for Evaluation

The system supports various AI agents for fitness evaluation:
- `randomBiasedAI` - Random actions with bias toward useful actions
- `workerRushAI` - Early worker rush strategy
- `lightRushAI` - Light unit rush strategy
- `passiveAI` - Defensive strategy
- `lightRushAI` - Light unit rush
- `heavyRushAI` - Heavy unit rush

## 📊 Command Line Options

```bash
python run_utt_evolution.py [OPTIONS]

Algorithm Options:
  --algorithm {ga,map-elites}    Evolution algorithm (default: ga)

Evolution Parameters:
  --generations INT              Number of generations (default: 5)
  --population INT               Population size (default: 10)
  --mutation-rate FLOAT          Mutation rate (default: 0.3)
  --crossover-rate FLOAT         Crossover rate (default: 0.7)

Fitness Evaluation:
  --fitness-type {ai,multi-objective}  Fitness type (default: ai)
  --ai STRING                    AI agent for evaluation (default: randomBiasedAI)
  --games-per-eval INT           Games per evaluation (default: 2)
  --max-steps INT                Max steps per game (default: 300)

Map Settings:
  --map STRING                   Map size (default: 8x8)

Output:
  --save-results                 Save results to file
  --visualize                    Create visualizations
```

## 🔬 Example Experiments

### 1. Quick Test (5 minutes)
```bash
python run_utt_evolution.py --generations 3 --population 8 --ai randomBiasedAI
```

### 2. Comprehensive GA Run (30 minutes)
```bash
python run_utt_evolution.py --generations 10 --population 20 --ai workerRushAI --games-per-eval 3
```

### 3. MAP-Elites Diversity Exploration (1 hour)
```bash
python run_utt_evolution.py --algorithm map-elites --generations 15 --population 30 --ai lightRushAI
```

### 4. Multi-AI Evaluation
```bash
# Test against multiple AI strategies
python run_utt_evolution.py --ai randomBiasedAI --generations 5
python run_utt_evolution.py --ai workerRushAI --generations 5
python run_utt_evolution.py --ai lightRushAI --generations 5
```

## 🎮 How It Works

1. **Initialization**: Create random population of UTT configurations
2. **Evaluation**: Test each UTT by playing AI vs AI games
3. **Selection**: Choose best-performing UTTs as parents
4. **Crossover**: Combine parent UTTs to create offspring
5. **Mutation**: Randomly modify UTT parameters
6. **Replacement**: Replace old population with new generation
7. **Repeat**: Continue for specified number of generations

## 📈 Expected Results

- **Fitness Improvement**: Better UTTs should achieve higher fitness scores
- **Strategy Specialization**: UTTs may evolve to counter specific AI strategies
- **Balanced Configurations**: MAP-Elites should find diverse, high-quality solutions
- **Performance Differences**: Different UTTs should show varying performance against different AIs

## 🛠️ Requirements

- Python 3.7+
- MicroRTS environment
- Required packages: numpy, matplotlib, seaborn (optional)

## 🚨 Notes

- AI-based evaluation is slower but more realistic
- Multi-objective evaluation is faster but less realistic
- Results may vary due to randomness in AI behavior
- Longer runs (more generations) generally produce better results