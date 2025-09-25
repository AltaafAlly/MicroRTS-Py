"""
Utility functions for UTT (Unit Type Table) manipulation and analysis.

This module provides tools for analyzing, comparing, and manipulating UTT configurations
for the genetic algorithm experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from dataclasses import dataclass
import os


@dataclass
class UTTStats:
    """Statistics for a UTT configuration."""
    avg_cost: float
    avg_hp: float
    avg_damage: float
    avg_dps: float
    avg_speed: float
    avg_range: float
    avg_production_time: float
    total_units: int
    combat_units: int
    economic_units: int


class UTTAnalyzer:
    """Analyzer for UTT configurations."""
    
    def __init__(self):
        """Initialize UTT analyzer."""
        pass
    
    def load_utt(self, file_path: str) -> Dict:
        """Load UTT from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def save_utt(self, utt_data: Dict, file_path: str):
        """Save UTT to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(utt_data, f, indent=2)
    
    def get_unit_stats(self, utt_data: Dict, unit_name: str) -> Optional[Dict]:
        """Get statistics for a specific unit type."""
        for unit in utt_data["unitTypes"]:
            if unit["name"] == unit_name:
                return unit
        return None
    
    def calculate_utt_stats(self, utt_data: Dict) -> UTTStats:
        """Calculate comprehensive statistics for a UTT."""
        # Filter out Resource units
        units = [unit for unit in utt_data["unitTypes"] if unit["name"] != "Resource"]
        
        if not units:
            return UTTStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate averages
        costs = [unit["cost"] for unit in units]
        hps = [unit["hp"] for unit in units]
        damages = [(unit["minDamage"] + unit["maxDamage"]) / 2 for unit in units]
        
        # Calculate DPS (damage per second)
        dps_values = []
        for unit in units:
            if unit["attackTime"] > 0:
                avg_damage = (unit["minDamage"] + unit["maxDamage"]) / 2
                dps = avg_damage / unit["attackTime"]
                dps_values.append(dps)
        
        # Calculate speed (1/moveTime)
        speeds = [1.0 / unit["moveTime"] if unit["moveTime"] > 0 else 0 for unit in units]
        
        ranges = [unit["attackRange"] for unit in units]
        production_times = [unit["produceTime"] for unit in units]
        
        # Count unit types
        combat_units = sum(1 for unit in units if unit["canAttack"])
        economic_units = sum(1 for unit in units if unit["canHarvest"] or unit["isStockpile"])
        
        return UTTStats(
            avg_cost=np.mean(costs),
            avg_hp=np.mean(hps),
            avg_damage=np.mean(damages),
            avg_dps=np.mean(dps_values) if dps_values else 0,
            avg_speed=np.mean(speeds),
            avg_range=np.mean(ranges),
            avg_production_time=np.mean(production_times),
            total_units=len(units),
            combat_units=combat_units,
            economic_units=economic_units
        )
    
    def compare_utts(self, utt1: Dict, utt2: Dict) -> Dict[str, float]:
        """Compare two UTT configurations."""
        stats1 = self.calculate_utt_stats(utt1)
        stats2 = self.calculate_utt_stats(utt2)
        
        comparison = {}
        for field in UTTStats.__dataclass_fields__:
            val1 = getattr(stats1, field)
            val2 = getattr(stats2, field)
            if val1 != 0:
                comparison[field] = (val2 - val1) / val1 * 100  # Percentage change
            else:
                comparison[field] = val2
        
        return comparison
    
    def validate_utt(self, utt_data: Dict) -> List[str]:
        """Validate UTT configuration and return list of issues."""
        issues = []
        
        for unit in utt_data["unitTypes"]:
            unit_name = unit["name"]
            
            # Check damage constraints
            if unit["minDamage"] > unit["maxDamage"]:
                issues.append(f"{unit_name}: minDamage ({unit['minDamage']}) > maxDamage ({unit['maxDamage']})")
            
            # Check Ranged unit constraints
            if unit_name == "Ranged":
                if unit["attackRange"] < 2:
                    issues.append(f"{unit_name}: attackRange ({unit['attackRange']}) < 2")
                if unit["minDamage"] > 3:
                    issues.append(f"{unit_name}: minDamage ({unit['minDamage']}) > 3 (should be low)")
                if unit["maxDamage"] > 4:
                    issues.append(f"{unit_name}: maxDamage ({unit['maxDamage']}) > 4 (should be low)")
            
            # Check for reasonable values
            if unit["cost"] <= 0:
                issues.append(f"{unit_name}: cost ({unit['cost']}) <= 0")
            if unit["hp"] <= 0:
                issues.append(f"{unit_name}: hp ({unit['hp']}) <= 0")
            if unit["produceTime"] <= 0:
                issues.append(f"{unit_name}: produceTime ({unit['produceTime']}) <= 0")
        
        return issues
    
    def create_utt_summary(self, utt_data: Dict) -> str:
        """Create a human-readable summary of UTT configuration."""
        stats = self.calculate_utt_stats(utt_data)
        issues = self.validate_utt(utt_data)
        
        summary = f"""
UTT Configuration Summary:
========================

Unit Statistics:
- Total Units: {stats.total_units}
- Combat Units: {stats.combat_units}
- Economic Units: {stats.economic_units}

Average Values:
- Cost: {stats.avg_cost:.2f}
- HP: {stats.avg_hp:.2f}
- Damage: {stats.avg_damage:.2f}
- DPS: {stats.avg_dps:.3f}
- Speed: {stats.avg_speed:.3f}
- Range: {stats.avg_range:.2f}
- Production Time: {stats.avg_production_time:.2f}

Unit Details:
"""
        
        for unit in utt_data["unitTypes"]:
            if unit["name"] == "Resource":
                continue
                
            summary += f"""
{unit['name']}:
  Cost: {unit['cost']}, HP: {unit['hp']}
  Damage: {unit['minDamage']}-{unit['maxDamage']}, Range: {unit['attackRange']}
  Times: Move={unit['moveTime']}, Attack={unit['attackTime']}, Produce={unit['produceTime']}
  Capabilities: Move={unit['canMove']}, Attack={unit['canAttack']}, Harvest={unit['canHarvest']}
"""
        
        if issues:
            summary += f"\nValidation Issues:\n"
            for issue in issues:
                summary += f"- {issue}\n"
        else:
            summary += "\nNo validation issues found.\n"
        
        return summary


class UTTVisualizer:
    """Visualization tools for UTT analysis."""
    
    def __init__(self):
        """Initialize UTT visualizer."""
        if sns:
            plt.style.use('seaborn-v0_8')
        else:
            plt.style.use('default')
    
    def plot_utt_comparison(self, utts: List[Dict], labels: List[str], output_path: str):
        """Create comparison plot for multiple UTTs."""
        stats_list = []
        analyzer = UTTAnalyzer()
        
        for utt in utts:
            stats = analyzer.calculate_utt_stats(utt)
            stats_list.append(stats)
        
        # Create comparison DataFrame
        data = []
        for i, (stats, label) in enumerate(zip(stats_list, labels)):
            data.append({
                'UTT': label,
                'Avg Cost': stats.avg_cost,
                'Avg HP': stats.avg_hp,
                'Avg Damage': stats.avg_damage,
                'Avg DPS': stats.avg_dps,
                'Avg Speed': stats.avg_speed,
                'Avg Range': stats.avg_range,
                'Avg Production Time': stats.avg_production_time
            })
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        metrics = ['Avg Cost', 'Avg HP', 'Avg Damage', 'Avg DPS', 
                  'Avg Speed', 'Avg Range', 'Avg Production Time']
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                df.plot(x='UTT', y=metric, kind='bar', ax=axes[i], 
                       color='skyblue', edgecolor='black')
                axes[i].set_title(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_unit_radar(self, utt_data: Dict, output_path: str):
        """Create radar chart for unit capabilities."""
        analyzer = UTTAnalyzer()
        
        # Get combat units
        combat_units = [unit for unit in utt_data["unitTypes"] 
                       if unit["canAttack"] and unit["name"] != "Resource"]
        
        if not combat_units:
            print("No combat units found for radar chart")
            return
        
        # Create radar chart for each unit
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, unit in enumerate(combat_units[:6]):  # Limit to 6 units
            if i >= len(axes):
                break
                
            # Normalize values for radar chart
            categories = ['Cost', 'HP', 'Damage', 'Range', 'Speed', 'DPS']
            
            # Calculate normalized values (0-1 scale)
            cost_norm = min(unit["cost"] / 10.0, 1.0)
            hp_norm = min(unit["hp"] / 20.0, 1.0)
            damage_norm = min((unit["minDamage"] + unit["maxDamage"]) / 2 / 8.0, 1.0)
            range_norm = min(unit["attackRange"] / 6.0, 1.0)
            speed_norm = min(1.0 / unit["moveTime"] / 0.2, 1.0)
            dps_norm = min(((unit["minDamage"] + unit["maxDamage"]) / 2 / unit["attackTime"]) / 2.0, 1.0)
            
            values = [cost_norm, hp_norm, damage_norm, range_norm, speed_norm, dps_norm]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            ax = axes[i]
            ax.plot(angles, values, 'o-', linewidth=2, label=unit["name"])
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title(unit["name"])
            ax.grid(True)
        
        # Remove empty subplots
        for i in range(len(combat_units), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_evolution_progress(self, fitness_history: List[float], output_path: str):
        """Plot evolution progress."""
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, linewidth=2, color='blue')
        plt.title('Evolution Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(fitness_history) > 1:
            z = np.polyfit(range(len(fitness_history)), fitness_history, 1)
            p = np.poly1d(z)
            plt.plot(p(range(len(fitness_history))), "--", color='red', alpha=0.7, label='Trend')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class UTTGenerator:
    """Generator for creating UTT variations."""
    
    def __init__(self, base_utt_path: str):
        """Initialize with base UTT."""
        with open(base_utt_path, 'r') as f:
            self.base_utt = json.load(f)
    
    def create_random_variation(self, variation_strength: float = 0.2) -> Dict:
        """Create random variation of base UTT."""
        utt = copy.deepcopy(self.base_utt)
        
        for unit in utt["unitTypes"]:
            if unit["name"] == "Resource":
                continue
            
            # Vary parameters
            unit["cost"] = max(1, int(unit["cost"] * (1 + random.uniform(-variation_strength, variation_strength))))
            unit["hp"] = max(1, int(unit["hp"] * (1 + random.uniform(-variation_strength, variation_strength))))
            unit["minDamage"] = max(1, int(unit["minDamage"] * (1 + random.uniform(-variation_strength, variation_strength))))
            unit["maxDamage"] = max(unit["minDamage"], int(unit["maxDamage"] * (1 + random.uniform(-variation_strength, variation_strength))))
            unit["attackRange"] = max(1, int(unit["attackRange"] * (1 + random.uniform(-variation_strength, variation_strength))))
            unit["produceTime"] = max(10, int(unit["produceTime"] * (1 + random.uniform(-variation_strength, variation_strength))))
            unit["moveTime"] = max(5, int(unit["moveTime"] * (1 + random.uniform(-variation_strength, variation_strength))))
            unit["attackTime"] = max(3, int(unit["attackTime"] * (1 + random.uniform(-variation_strength, variation_strength))))
            unit["sightRadius"] = max(1, int(unit["sightRadius"] * (1 + random.uniform(-variation_strength, variation_strength))))
            
            # Special constraints for Ranged
            if unit["name"] == "Ranged":
                unit["attackRange"] = max(2, unit["attackRange"])
                unit["minDamage"] = min(3, unit["minDamage"])
                unit["maxDamage"] = min(4, unit["maxDamage"])
        
        return utt
    
    def create_balanced_variation(self) -> Dict:
        """Create balanced variation with specific design goals."""
        utt = copy.deepcopy(self.base_utt)
        
        # Design goals: balanced economy, diverse combat roles
        for unit in utt["unitTypes"]:
            if unit["name"] == "Resource":
                continue
            
            if unit["name"] == "Worker":
                # Make workers efficient but vulnerable
                unit["cost"] = 2
                unit["hp"] = 3
                unit["harvestAmount"] = 2
                unit["harvestTime"] = 20
                
            elif unit["name"] == "Light":
                # Fast, cheap, low damage
                unit["cost"] = 3
                unit["hp"] = 7
                unit["minDamage"] = 3
                unit["maxDamage"] = 3
                unit["moveTime"] = 9
                
            elif unit["name"] == "Heavy":
                # Slow, expensive, high damage
                unit["cost"] = 6
                unit["hp"] = 15
                unit["minDamage"] = 6
                unit["maxDamage"] = 6
                unit["moveTime"] = 12
                
            elif unit["name"] == "Ranged":
                # Medium cost, low damage, high range
                unit["cost"] = 4
                unit["hp"] = 3
                unit["minDamage"] = 2
                unit["maxDamage"] = 3
                unit["attackRange"] = 4
                unit["moveTime"] = 12
        
        return utt


def analyze_utt_file(file_path: str, output_dir: str = None):
    """Analyze a UTT file and generate reports."""
    analyzer = UTTAnalyzer()
    visualizer = UTTVisualizer()
    
    # Load UTT
    utt_data = analyzer.load_utt(file_path)
    
    # Generate summary
    summary = analyzer.create_utt_summary(utt_data)
    print(summary)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        summary_path = os.path.join(output_dir, "utt_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        # Create visualizations
        radar_path = os.path.join(output_dir, "unit_radar.png")
        visualizer.plot_unit_radar(utt_data, radar_path)
        
        print(f"Analysis saved to: {output_dir}")


def compare_utt_files(file_paths: List[str], labels: List[str], output_path: str):
    """Compare multiple UTT files."""
    analyzer = UTTAnalyzer()
    visualizer = UTTVisualizer()
    
    # Load UTTs
    utts = [analyzer.load_utt(path) for path in file_paths]
    
    # Create comparison plot
    visualizer.plot_utt_comparison(utts, labels, output_path)
    
    # Print detailed comparison
    print("UTT Comparison:")
    print("===============")
    
    for i, (utt, label) in enumerate(zip(utts, labels)):
        stats = analyzer.calculate_utt_stats(utt)
        print(f"\n{label}:")
        print(f"  Avg Cost: {stats.avg_cost:.2f}")
        print(f"  Avg HP: {stats.avg_hp:.2f}")
        print(f"  Avg DPS: {stats.avg_dps:.3f}")
        print(f"  Avg Speed: {stats.avg_speed:.3f}")
    
    print(f"\nComparison plot saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    base_utt_path = "/home/altaaf/projects/MicroRTS-Py-Research/gym_microrts/microrts/utts/AsymmetricP1UTT.json"
    
    # Analyze base UTT
    print("Analyzing base UTT...")
    analyze_utt_file(base_utt_path, "/home/altaaf/projects/MicroRTS-Py-Research/experiments/utt_analysis")
    
    # Create and compare variations
    generator = UTTGenerator(base_utt_path)
    
    variations = [
        generator.create_random_variation(0.3),
        generator.create_balanced_variation()
    ]
    
    labels = ["Random Variation", "Balanced Variation"]
    
    # Save variations
    for i, (variation, label) in enumerate(zip(variations, labels)):
        output_path = f"/home/altaaf/projects/MicroRTS-Py-Research/experiments/{label.lower().replace(' ', '_')}.json"
        with open(output_path, 'w') as f:
            json.dump(variation, f, indent=2)
        print(f"Saved {label} to: {output_path}")
    
    # Compare variations
    compare_utt_files(
        [base_utt_path] + [f"/home/altaaf/projects/MicroRTS-Py-Research/experiments/{label.lower().replace(' ', '_')}.json" for label in labels],
        ["Base UTT"] + labels,
        "/home/altaaf/projects/MicroRTS-Py-Research/experiments/utt_comparison.png"
    )
