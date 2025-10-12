#!/usr/bin/env python3
"""
UTT Manager - Organize and manage evolved UTT files
"""

import os
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

class UTTManager:
    """Manages evolved UTT files with proper organization and metadata."""
    
    def __init__(self, base_dir: str = "evolved_utts"):
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        self.base_dir = project_root / base_dir
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_dir / "generations").mkdir(exist_ok=True)
        (self.base_dir / "experiments").mkdir(exist_ok=True)
        (self.base_dir / "archive").mkdir(exist_ok=True)
        (self.base_dir / "templates").mkdir(exist_ok=True)
    
    def save_utt_with_metadata(self, 
                              utt_config: dict,
                              experiment_id: str,
                              generation: int,
                              individual_id: int,
                              fitness: dict,
                              description: str = "",
                              parent_individuals: List[int] = None,
                              mutation_count: int = 0) -> str:
        """
        Save UTT with proper naming and metadata.
        
        Args:
            utt_config: The UTT configuration dictionary
            experiment_id: Experiment identifier (e.g., "exp_001")
            generation: Generation number
            individual_id: Individual number within generation
            fitness: Fitness scores dictionary
            description: Human-readable description
            parent_individuals: List of parent individual IDs
            mutation_count: Number of mutations applied
            
        Returns:
            Path to saved UTT file
        """
        
        # Create filename
        fitness_str = f"{fitness.get('overall_fitness', 0.0):.3f}"
        utt_filename = f"{experiment_id}_gen_{generation:03d}_ind_{individual_id:03d}_fitness_{fitness_str}.json"
        meta_filename = f"{experiment_id}_gen_{generation:03d}_ind_{individual_id:03d}_fitness_{fitness_str}.meta.json"
        
        # Determine directory
        if experiment_id.startswith("exp_"):
            utt_dir = self.base_dir / "experiments" / experiment_id
        else:
            utt_dir = self.base_dir / "generations" / f"gen_{generation:03d}"
        
        utt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save UTT file
        utt_path = utt_dir / utt_filename
        with open(utt_path, 'w') as f:
            json.dump(utt_config, f, indent=2)
        
        # Create and save metadata
        metadata = {
            "experiment_id": experiment_id,
            "generation": generation,
            "individual_id": individual_id,
            "fitness": fitness,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "parent_individuals": parent_individuals or [],
            "mutation_count": mutation_count,
            "utt_filename": utt_filename
        }
        
        meta_path = utt_dir / meta_filename
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved UTT: {utt_path}")
        print(f"✅ Saved metadata: {meta_path}")
        
        return str(utt_path)
    
    def save_best_utt(self, utt_config: dict, experiment_id: str, fitness: dict) -> str:
        """Save the best UTT from an experiment."""
        return self.save_utt_with_metadata(
            utt_config=utt_config,
            experiment_id=experiment_id,
            generation=0,  # Special generation for best
            individual_id=0,  # Special individual for best
            fitness=fitness,
            description="Best UTT from experiment"
        )
    
    def list_utts(self, experiment_id: Optional[str] = None, generation: Optional[int] = None) -> List[Dict]:
        """List all UTTs with their metadata."""
        utts = []
        
        if experiment_id:
            # List UTTs from specific experiment
            exp_dir = self.base_dir / "experiments" / experiment_id
            if exp_dir.exists():
                for meta_file in exp_dir.glob("*.meta.json"):
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                        utts.append(metadata)
        elif generation is not None:
            # List UTTs from specific generation
            gen_dir = self.base_dir / "generations" / f"gen_{generation:03d}"
            if gen_dir.exists():
                for meta_file in gen_dir.glob("*.meta.json"):
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                        utts.append(metadata)
        else:
            # List all UTTs
            for meta_file in self.base_dir.rglob("*.meta.json"):
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    utts.append(metadata)
        
        # Sort by fitness
        utts.sort(key=lambda x: x.get('fitness', {}).get('overall_fitness', 0.0), reverse=True)
        return utts
    
    def get_best_utt(self, experiment_id: Optional[str] = None) -> Optional[Dict]:
        """Get the best UTT based on overall fitness."""
        utts = self.list_utts(experiment_id)
        if utts:
            return utts[0]
        return None
    
    def copy_utt_to_microrts(self, utt_path: str, new_name: Optional[str] = None) -> str:
        """Copy UTT file to gym_microrts/microrts/utts/ for testing."""
        utt_path = Path(utt_path)
        if not utt_path.exists():
            raise FileNotFoundError(f"UTT file not found: {utt_path}")
        
        # Determine new filename
        if new_name:
            new_filename = f"{new_name}.json"
        else:
            new_filename = utt_path.name
        
        # Copy to microrts directory
        microrts_dir = Path("gym_microrts/microrts/utts")
        microrts_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = microrts_dir / new_filename
        shutil.copy2(utt_path, dest_path)
        
        print(f"✅ Copied UTT to: {dest_path}")
        return str(dest_path)
    
    def cleanup_temp_utts(self):
        """Clean up temporary UTT files from gym_microrts directory."""
        microrts_dir = Path("gym_microrts/microrts/utts")
        if not microrts_dir.exists():
            return
        
        # Remove temporary GA UTT files
        temp_files = list(microrts_dir.glob("ga_utt_*.json"))
        for temp_file in temp_files:
            temp_file.unlink()
            print(f"🧹 Removed temp file: {temp_file}")
        
        print(f"✅ Cleaned up {len(temp_files)} temporary UTT files")
    
    def archive_old_utts(self, experiment_id: str, keep_best: int = 5):
        """Archive old UTTs, keeping only the best ones."""
        exp_dir = self.base_dir / "experiments" / experiment_id
        if not exp_dir.exists():
            return
        
        # Get all UTTs sorted by fitness
        utts = self.list_utts(experiment_id)
        
        # Keep only the best ones
        utts_to_keep = utts[:keep_best]
        utts_to_archive = utts[keep_best:]
        
        # Move to archive
        archive_dir = self.base_dir / "archive" / experiment_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        for utt_meta in utts_to_archive:
            utt_filename = utt_meta['utt_filename']
            meta_filename = utt_filename.replace('.json', '.meta.json')
            
            # Move files
            shutil.move(exp_dir / utt_filename, archive_dir / utt_filename)
            shutil.move(exp_dir / meta_filename, archive_dir / meta_filename)
        
        print(f"📦 Archived {len(utts_to_archive)} UTTs, kept {len(utts_to_keep)} best ones")
    
    def generate_report(self, experiment_id: Optional[str] = None) -> str:
        """Generate a report of UTT performance."""
        utts = self.list_utts(experiment_id)
        
        if not utts:
            return "No UTTs found."
        
        report = []
        report.append("🧬 UTT Performance Report")
        report.append("=" * 50)
        
        if experiment_id:
            report.append(f"Experiment: {experiment_id}")
        else:
            report.append("All Experiments")
        
        report.append(f"Total UTTs: {len(utts)}")
        report.append("")
        
        # Top 10 UTTs
        report.append("🏆 Top 10 UTTs:")
        for i, utt in enumerate(utts[:10], 1):
            fitness = utt.get('fitness', {})
            report.append(f"{i:2d}. {utt['utt_filename']}")
            report.append(f"    Overall: {fitness.get('overall_fitness', 0.0):.3f}")
            report.append(f"    Balance: {fitness.get('balance', 0.0):.3f}")
            report.append(f"    Duration: {fitness.get('duration', 0.0):.3f}")
            report.append(f"    Diversity: {fitness.get('strategy_diversity', 0.0):.3f}")
            report.append("")
        
        # Statistics
        overall_fitnesses = [utt.get('fitness', {}).get('overall_fitness', 0.0) for utt in utts]
        if overall_fitnesses:
            report.append("📊 Statistics:")
            report.append(f"Best fitness: {max(overall_fitnesses):.3f}")
            report.append(f"Worst fitness: {min(overall_fitnesses):.3f}")
            report.append(f"Average fitness: {sum(overall_fitnesses)/len(overall_fitnesses):.3f}")
        
        return "\n".join(report)

def main():
    """Command line interface for UTT Manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="UTT Manager - Organize evolved UTT files")
    parser.add_argument("--list", action="store_true", help="List all UTTs")
    parser.add_argument("--experiment", type=str, help="Filter by experiment ID")
    parser.add_argument("--generation", type=int, help="Filter by generation")
    parser.add_argument("--best", action="store_true", help="Show best UTT")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--cleanup", action="store_true", help="Clean up temporary UTT files")
    parser.add_argument("--archive", type=str, help="Archive old UTTs for experiment")
    
    args = parser.parse_args()
    
    manager = UTTManager()
    
    if args.list:
        utts = manager.list_utts(args.experiment, args.generation)
        print(f"Found {len(utts)} UTTs:")
        for utt in utts[:10]:  # Show top 10
            fitness = utt.get('fitness', {})
            print(f"  {utt['utt_filename']} - Fitness: {fitness.get('overall_fitness', 0.0):.3f}")
    
    elif args.best:
        best_utt = manager.get_best_utt(args.experiment)
        if best_utt:
            print("🏆 Best UTT:")
            print(f"  File: {best_utt['utt_filename']}")
            fitness = best_utt.get('fitness', {})
            print(f"  Overall fitness: {fitness.get('overall_fitness', 0.0):.3f}")
            print(f"  Balance: {fitness.get('balance', 0.0):.3f}")
            print(f"  Duration: {fitness.get('duration', 0.0):.3f}")
            print(f"  Diversity: {fitness.get('strategy_diversity', 0.0):.3f}")
        else:
            print("No UTTs found.")
    
    elif args.report:
        report = manager.generate_report(args.experiment)
        print(report)
    
    elif args.cleanup:
        manager.cleanup_temp_utts()
    
    elif args.archive:
        manager.archive_old_utts(args.archive)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
