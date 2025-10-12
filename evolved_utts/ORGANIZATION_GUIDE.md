# 🧬 UTT Organization Guide

## 🎯 **Problem Solved**

The previous setup was messy with UTT files scattered everywhere. Now we have a **proper organization system** for managing evolved UTT files.

## 📁 **New Directory Structure**

```
evolved_utts/
├── experiments/          # UTTs organized by experiment
│   └── exp_001/         # Experiment 1 UTTs
│       ├── exp_001_gen_001_ind_001_fitness_0.385.json
│       ├── exp_001_gen_001_ind_001_fitness_0.385.meta.json
│       └── ...
├── generations/         # UTTs organized by generation
├── archive/             # Archived/backup UTTs
├── templates/           # Template UTT files
└── README.md           # Documentation
```

## 🛠️ **New Tools**

### 1. **UTT Manager** (`utt_manager.py`)
- **Organize**: Save UTTs with proper naming and metadata
- **List**: View all UTTs with fitness scores
- **Find Best**: Get the best UTT from experiments
- **Report**: Generate performance reports
- **Cleanup**: Remove temporary files

### 2. **UTT Tester** (`test_utt.py`)
- **Test Single UTT**: Test any UTT file
- **Test Best UTT**: Test the best UTT from an experiment
- **Compare UTTs**: Compare two UTT files side-by-side

### 3. **GA Evolution** (`run_ga.py`)
- **Real Evolution**: Evolves UTTs using genetic algorithm
- **Organized Output**: Saves evolved UTTs with proper structure
- **Metadata**: Includes fitness scores and descriptions
- **Experiment Tracking**: Organizes by experiment ID

## 🚀 **Usage Examples**

### Evolve UTTs for a new experiment:
```bash
python scripts/GA\ and\ MAP-Elites/run_ga.py \
    --use-working-evaluator \
    --generations 10 \
    --population 20 \
    --experiment-name exp_002
```

### List all UTTs:
```bash
python scripts/GA\ and\ MAP-Elites/utt_manager.py --list
```

### Show the best UTT:
```bash
python scripts/GA\ and\ MAP-Elites/utt_manager.py --best
```

### Generate performance report:
```bash
python scripts/GA\ and\ MAP-Elites/utt_manager.py --report
```

### Test the best UTT:
```bash
python scripts/GA\ and\ MAP-Elites/test_utt.py --best exp_001
```

### Compare two UTTs:
```bash
python scripts/GA\ and\ MAP-Elites/test_utt.py \
    --compare utt1.json utt2.json
```

## 📊 **Metadata System**

Each UTT file now has a companion `.meta.json` file with:

```json
{
    "experiment_id": "exp_001",
    "generation": 1,
    "individual_id": 4,
    "fitness": {
        "overall_fitness": 0.892,
        "balance": 0.737,
        "duration": 0.720,
        "strategy_diversity": 0.230
    },
    "created_at": "2024-01-15T10:30:00Z",
    "description": "Random individual 4 from generation 1",
    "parent_individuals": [],
    "mutation_count": 0
}
```

## 🧹 **Cleanup**

- **Automatic**: UTT Manager automatically cleans up temporary files
- **Manual**: Use `--cleanup` flag to remove temp files
- **Archive**: Move old UTTs to archive when no longer needed

## ✅ **Benefits**

1. **Organized**: No more scattered UTT files
2. **Trackable**: Full metadata for each UTT
3. **Searchable**: Easy to find best UTTs
4. **Comparable**: Side-by-side UTT testing
5. **Scalable**: Handles many experiments and generations
6. **Clean**: Automatic cleanup of temporary files

## 🎯 **Next Steps**

1. **Run Experiments**: Generate UTTs for different experiments
2. **Test Performance**: Use the testing tools to evaluate UTTs
3. **Compare Results**: Find the best UTT configurations
4. **Scale Up**: Run larger experiments with more generations
5. **Analyze**: Use reports to understand evolution progress

The system is now **clean, organized, and ready for serious research**! 🎉
