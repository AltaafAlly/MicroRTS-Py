#!/usr/bin/env python3
"""
Tournament Results Analysis
==========================

This script analyzes the results from the UTT Impact Tournament and generates
comprehensive reports, visualizations, and insights about UTT impact on AI performance.
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


class TournamentAnalyzer:
    """Analyzes tournament results and generates comprehensive reports."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tournament summary
        self.summary = self.load_tournament_summary()
        self.comparison_data = self.load_comparison_data()
        self.impact_analysis = self.load_impact_analysis()
        
    def load_tournament_summary(self) -> Dict:
        """Load tournament summary."""
        summary_path = self.input_dir / "tournament_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                return json.load(f)
        return {}
    
    def load_comparison_data(self) -> pd.DataFrame:
        """Load UTT comparison data."""
        comparison_path = self.input_dir / "utt_comparison.csv"
        if comparison_path.exists():
            return pd.read_csv(comparison_path)
        return pd.DataFrame()
    
    def load_impact_analysis(self) -> Dict:
        """Load UTT impact analysis."""
        impact_path = self.input_dir / "utt_impact_analysis.json"
        if impact_path.exists():
            with open(impact_path, "r") as f:
                return json.load(f)
        return {}
    
    def generate_performance_rankings(self):
        """Generate performance rankings across UTTs."""
        if self.comparison_data.empty:
            print("No comparison data available")
            return
        
        # Create ranking analysis
        rankings = {}
        for utt in self.comparison_data['utt'].unique():
            utt_data = self.comparison_data[self.comparison_data['utt'] == utt]
            utt_data = utt_data.sort_values('points', ascending=False)
            rankings[utt] = utt_data[['ai', 'points', 'win_rate']].to_dict('records')
        
        # Save rankings
        with open(self.output_dir / "performance_rankings.json", "w") as f:
            json.dump(rankings, f, indent=2)
        
        # Create ranking comparison table
        ranking_df = pd.DataFrame()
        for utt, ranking in rankings.items():
            for i, ai_data in enumerate(ranking):
                ranking_df = pd.concat([ranking_df, pd.DataFrame({
                    'utt': [utt],
                    'ai': [ai_data['ai']],
                    'rank': [i + 1],
                    'points': [ai_data['points']],
                    'win_rate': [ai_data['win_rate']]
                })], ignore_index=True)
        
        ranking_df.to_csv(self.output_dir / "ranking_comparison.csv", index=False)
        
        print("Performance rankings generated")
    
    def generate_utt_impact_report(self):
        """Generate detailed UTT impact report."""
        if not self.impact_analysis:
            print("No impact analysis data available")
            return
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": {},
            "detailed_impacts": self.impact_analysis
        }
        
        # Calculate summary statistics
        for utt_name, impacts in self.impact_analysis.items():
            changes = [impact["percent_change"] for impact in impacts.values()]
            report["summary"][utt_name] = {
                "average_change": np.mean(changes),
                "median_change": np.median(changes),
                "std_change": np.std(changes),
                "max_improvement": max(changes),
                "max_decline": min(changes),
                "ais_improved": sum(1 for c in changes if c > 0),
                "ais_declined": sum(1 for c in changes if c < 0),
                "total_ais": len(changes)
            }
        
        # Save report
        with open(self.output_dir / "utt_impact_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate summary table
        summary_df = pd.DataFrame(report["summary"]).T
        summary_df.to_csv(self.output_dir / "utt_impact_summary.csv")
        
        print("UTT impact report generated")
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        if self.comparison_data.empty:
            print("No data available for visualizations")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Win Rate Comparison Heatmap
        self.create_win_rate_heatmap()
        
        # 2. Points Distribution Box Plot
        self.create_points_distribution_plot()
        
        # 3. UTT Impact Scatter Plot
        self.create_utt_impact_scatter()
        
        # 4. AI Performance Ranking Changes
        self.create_ranking_changes_plot()
        
        print("Visualizations created")
    
    def create_win_rate_heatmap(self):
        """Create win rate comparison heatmap."""
        pivot_data = self.comparison_data.pivot(index='ai', columns='utt', values='win_rate')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Win Rate'})
        plt.title('AI Win Rates Across Different UTTs', fontsize=16, fontweight='bold')
        plt.xlabel('UTT Configuration', fontsize=12)
        plt.ylabel('AI Agent', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "win_rate_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_points_distribution_plot(self):
        """Create points distribution box plot."""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.comparison_data, x='utt', y='points')
        plt.title('Points Distribution Across UTTs', fontsize=16, fontweight='bold')
        plt.xlabel('UTT Configuration', fontsize=12)
        plt.ylabel('Points', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "points_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_utt_impact_scatter(self):
        """Create UTT impact scatter plot."""
        if not self.impact_analysis:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (utt_name, impacts) in enumerate(self.impact_analysis.items()):
            if i >= 4:  # Limit to 4 subplots
                break
                
            ais = list(impacts.keys())
            changes = [impacts[ai]["percent_change"] for ai in ais]
            
            axes[i].scatter(range(len(ais)), changes, alpha=0.7, s=60)
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[i].set_title(f'{utt_name} vs Default Original', fontweight='bold')
            axes[i].set_xlabel('AI Agent Index')
            axes[i].set_ylabel('Performance Change (%)')
            axes[i].grid(True, alpha=0.3)
            
            # Add AI labels for significant changes
            for j, (ai, change) in enumerate(zip(ais, changes)):
                if abs(change) > 20:  # Significant change threshold
                    axes[i].annotate(ai, (j, change), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=8)
        
        plt.suptitle('UTT Impact on AI Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "utt_impact_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_ranking_changes_plot(self):
        """Create ranking changes visualization."""
        if self.comparison_data.empty:
            return
        
        # Calculate ranking changes
        baseline_utt = 'default_original'
        if baseline_utt not in self.comparison_data['utt'].values:
            print(f"Baseline UTT {baseline_utt} not found")
            return
        
        baseline_ranks = self.comparison_data[
            self.comparison_data['utt'] == baseline_utt
        ].sort_values('points', ascending=False).reset_index(drop=True)
        baseline_ranks['baseline_rank'] = baseline_ranks.index + 1
        
        ranking_changes = {}
        for utt in self.comparison_data['utt'].unique():
            if utt == baseline_utt:
                continue
                
            utt_ranks = self.comparison_data[
                self.comparison_data['utt'] == utt
            ].sort_values('points', ascending=False).reset_index(drop=True)
            utt_ranks['utt_rank'] = utt_ranks.index + 1
            
            # Merge with baseline
            merged = pd.merge(baseline_ranks[['ai', 'baseline_rank']], 
                            utt_ranks[['ai', 'utt_rank']], on='ai')
            merged['rank_change'] = merged['baseline_rank'] - merged['utt_rank']
            ranking_changes[utt] = merged
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (utt_name, changes) in enumerate(ranking_changes.items()):
            if i >= 4:
                break
                
            axes[i].bar(range(len(changes)), changes['rank_change'], 
                       color=['green' if x > 0 else 'red' if x < 0 else 'gray' 
                             for x in changes['rank_change']])
            axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[i].set_title(f'Ranking Changes: {utt_name}', fontweight='bold')
            axes[i].set_xlabel('AI Agent Index')
            axes[i].set_ylabel('Rank Change (Positive = Better)')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('AI Ranking Changes Across UTTs', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "ranking_changes.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_executive_summary(self):
        """Generate executive summary report."""
        summary = {
            "tournament_overview": self.summary.get("tournament_info", {}),
            "key_findings": [],
            "recommendations": [],
            "detailed_analysis": {}
        }
        
        # Analyze key findings
        if self.impact_analysis:
            for utt_name, impacts in self.impact_analysis.items():
                changes = [impact["percent_change"] for impact in impacts.values()]
                avg_change = np.mean(changes)
                
                if abs(avg_change) > 10:  # Significant impact
                    direction = "improves" if avg_change > 0 else "degrades"
                    summary["key_findings"].append(
                        f"UTT '{utt_name}' {direction} AI performance by {abs(avg_change):.1f}% on average"
                    )
                
                # Find most/least affected AIs
                max_impact = max(impacts.items(), key=lambda x: abs(x[1]["percent_change"]))
                summary["key_findings"].append(
                    f"AI '{max_impact[0]}' shows the largest impact with {max_impact[1]['percent_change']:.1f}% change under '{utt_name}'"
                )
        
        # Generate recommendations
        if self.comparison_data.empty == False:
            # Find best performing UTT
            utt_avg_points = self.comparison_data.groupby('utt')['points'].mean()
            best_utt = utt_avg_points.idxmax()
            summary["recommendations"].append(
                f"UTT '{best_utt}' shows the highest average performance across all AIs"
            )
            
            # Find most balanced UTT (lowest std deviation)
            utt_std_points = self.comparison_data.groupby('utt')['points'].std()
            most_balanced = utt_std_points.idxmin()
            summary["recommendations"].append(
                f"UTT '{most_balanced}' shows the most balanced performance across AIs"
            )
        
        # Save executive summary
        with open(self.output_dir / "executive_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        self.create_markdown_report(summary)
        
        print("Executive summary generated")
    
    def create_markdown_report(self, summary: Dict):
        """Create markdown report for easy reading."""
        md_content = f"""# UTT Impact Tournament Results

## Tournament Overview
- **Start Time**: {summary['tournament_overview'].get('start_time', 'N/A')}
- **Duration**: {summary['tournament_overview'].get('duration_minutes', 'N/A'):.1f} minutes
- **UTT Configurations**: {', '.join(summary['tournament_overview'].get('utt_configurations', []))}
- **AI Agents**: {len(summary['tournament_overview'].get('ai_agents', []))} agents tested

## Key Findings
"""
        
        for finding in summary["key_findings"]:
            md_content += f"- {finding}\n"
        
        md_content += "\n## Recommendations\n"
        for rec in summary["recommendations"]:
            md_content += f"- {rec}\n"
        
        md_content += f"""
## Files Generated
- `performance_rankings.json`: Detailed rankings for each UTT
- `utt_impact_report.json`: Comprehensive impact analysis
- `win_rate_heatmap.png`: Visual comparison of win rates
- `points_distribution.png`: Distribution of points across UTTs
- `utt_impact_scatter.png`: Scatter plot of UTT impacts
- `ranking_changes.png`: Changes in AI rankings across UTTs

## Analysis Completed
**Timestamp**: {datetime.now().isoformat()}
"""
        
        with open(self.output_dir / "tournament_report.md", "w") as f:
            f.write(md_content)
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("Starting tournament results analysis...")
        
        self.generate_performance_rankings()
        self.generate_utt_impact_report()
        self.create_visualizations()
        self.generate_executive_summary()
        
        print(f"\nAnalysis completed! Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in self.output_dir.iterdir():
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description="Analyze UTT Tournament Results")
    parser.add_argument("--input-dir", required=True, help="Input directory with tournament results")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis")
    
    args = parser.parse_args()
    
    analyzer = TournamentAnalyzer(args.input_dir, args.output_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
