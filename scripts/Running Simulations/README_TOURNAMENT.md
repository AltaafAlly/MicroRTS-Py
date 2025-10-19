# UTT Impact Tournament System

This directory contains a comprehensive tournament system for analyzing the impact of custom Unit Type Tables (UTTs) on AI agent performance in MicroRTS.

## 🎯 Purpose

The tournament system is designed to:
- Compare AI performance across different UTT configurations
- Analyze how custom UTTs affect different AI strategies
- Generate comprehensive statistical analysis and visualizations
- Provide insights for UTT design and AI development

## 📁 Files Overview

### Core Tournament Scripts
- **`utt_impact_tournament.py`** - Main tournament runner with comprehensive analysis
- **`run_match_configured.py`** - Original match runner (your existing script)
- **`test_tournament.py`** - Quick test script to verify everything works

### Cluster Execution
- **`submit_tournament.sh`** - SLURM job submission script for cluster execution
- **`analyze_tournament_results.py`** - Post-tournament analysis and visualization

## 🚀 Quick Start

### 1. Test Locally (Recommended First)
```bash
cd /home/altaaf/projects/MicroRTS-Py-Research
conda activate microrts
python scripts/Running\ Simulations/test_tournament.py
```

### 2. Run Full Tournament Locally
```bash
python scripts/Running\ Simulations/utt_impact_tournament.py \
    --output-dir "my_tournament_results" \
    --games 10 \
    --max-steps 4000
```

### 3. Submit to Cluster
```bash
sbatch scripts/Running\ Simulations/submit_tournament.sh
```

## 🎮 Tournament Configuration

### AI Agents Tested
The tournament uses strategically selected baseline AI agents:

**Rush Strategies (Aggressive):**
- `workerRushAI` - Classic worker rush
- `lightRushAI` - Light unit rush  
- `POHeavyRush` - Heavy unit rush (partial obs)
- `PORangedRush` - Ranged unit rush (partial obs)

**Balanced Strategies:**
- `coacAI` - Strong balanced AI
- `naiveMCTSAI` - Monte Carlo Tree Search
- `mixedBot` - Mixed strategy bot

**Defensive/Economic:**
- `passiveAI` - Defensive baseline
- `randomBiasedAI` - Biased random (slightly strategic)

**Advanced Strategies:**
- `izanagi` - Advanced AI
- `droplet` - Sophisticated bot
- `tiamat` - Complex strategy AI

### UTT Configurations
- **`default_original`** - Default original UTT (baseline)
- **`default_finetuned`** - Default finetuned UTT
- **`custom_demo`** - Your custom overpowered UTT
- **`asymmetric_p1`** - Asymmetric UTT for player 1

## 📊 Results Structure

The tournament generates comprehensive results in the output directory:

```
tournament_results/
├── tournament_summary.json          # Overall tournament summary
├── utt_comparison.csv               # Cross-UTT comparison data
├── utt_impact_analysis.json         # Detailed impact analysis
├── default_original/                # Results for each UTT
│   ├── detailed_results.json
│   ├── standings.csv
│   └── pair_results.csv
├── custom_demo/
│   ├── detailed_results.json
│   ├── standings.csv
│   └── pair_results.csv
└── analysis/                        # Post-tournament analysis
    ├── performance_rankings.json
    ├── utt_impact_report.json
    ├── executive_summary.json
    ├── tournament_report.md
    ├── win_rate_heatmap.png
    ├── points_distribution.png
    ├── utt_impact_scatter.png
    └── ranking_changes.png
```

## 🔍 Key Analysis Features

### 1. Performance Rankings
- AI rankings for each UTT configuration
- Cross-UTT ranking comparisons
- Statistical significance testing

### 2. UTT Impact Analysis
- Performance change percentages
- Most/least affected AI agents
- Statistical significance of changes

### 3. Visualizations
- Win rate heatmaps across UTTs
- Points distribution box plots
- UTT impact scatter plots
- Ranking change visualizations

### 4. Executive Summary
- Key findings and insights
- Recommendations for UTT design
- Markdown report for easy reading

## ⚙️ Customization

### Modify AI Agents
Edit the `baseline_ais` list in `utt_impact_tournament.py`:
```python
self.baseline_ais = [
    "your_ai_1",
    "your_ai_2",
    # ... add more
]
```

### Add UTT Configurations
Edit the `utt_configs` dictionary:
```python
self.utt_configs = {
    "your_utt_name": "path/to/your/utt.json",
    # ... add more
}
```

### Adjust Tournament Settings
Modify `tournament_config`:
```python
self.tournament_config = {
    "map_path": "maps/8x8/basesWorkers8x8A.xml",
    "max_steps": 4000,
    "games_per_pair": 10,
    # ... adjust as needed
}
```

## 📈 Expected Results

Based on your custom UTT modifications, you should see:

### Custom Demo UTT Impact
- **Dramatic performance changes** due to overpowered units
- **Faster game resolution** due to increased damage/speed
- **Different AI strategy effectiveness** due to changed unit balance
- **Higher win rates** for aggressive AIs due to faster production

### Key Insights to Look For
1. **Which AIs benefit most** from the custom UTT
2. **How strategy effectiveness changes** with different unit balance
3. **Whether certain AI types** (rush vs defensive) are more affected
4. **Statistical significance** of UTT impact on different AIs

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Make sure you're in the correct conda environment
2. **File not found**: Check that UTT JSON files exist in the correct paths
3. **Memory issues**: Reduce `games_per_pair` or `max_steps` for testing
4. **Long execution**: Use the test script first to verify setup

### Debug Mode
Run with verbose logging:
```bash
python scripts/Running\ Simulations/utt_impact_tournament.py --output-dir "debug_results" --games 1
```

## 📞 Support

If you encounter issues:
1. Check the log output for error messages
2. Verify all file paths are correct
3. Ensure the conda environment is properly activated
4. Try the test script first to isolate issues

## 🎉 Success Metrics

A successful tournament run should show:
- Clear performance differences between UTTs
- Statistical significance in UTT impact
- Meaningful insights about AI strategy effectiveness
- Comprehensive visualizations and reports

The tournament is designed to provide actionable insights for UTT design and AI development in MicroRTS!
