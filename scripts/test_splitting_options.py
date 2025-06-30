#!/usr/bin/env python3
"""
Test Script for Dataset Splitting Options

This script tests all three splitting options (A, B, C) by:
1. Creating a backup of the current state
2. Running each option sequentially
3. Resetting the state between options
4. Comparing results

Author: ASV Dataset Preparation System
Version: 1.0.0
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.dataset_config import ASVDatasetConfig
from scripts.dataset_splitting_balancing import DatasetSplittingSystem
from utils.logging_utils import setup_logger

class SplittingOptionsTestSystem:
    """System to test all splitting options with reset functionality"""
    
    def __init__(self):
        self.config = ASVDatasetConfig()
        self.logger = setup_logger('splitting_test', self.config.paths.logs_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Test directories
        self.test_dir = self.config.paths.output_root / f"splitting_test_{self.timestamp}"
        self.backup_dir = self.test_dir / "backup"
        self.results_dir = self.test_dir / "results"
        
        # Create test directories
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Test system initialized at {self.test_dir}")
    
    def create_backup(self) -> bool:
        """Create backup of current output state"""
        try:
            self.logger.info("Creating backup of current state...")
            
            # Backup any existing output files
            output_files = list(self.config.paths.output_root.glob("*"))
            if output_files:
                for file_path in output_files:
                    if file_path.name.startswith("splitting_test_"):
                        continue  # Skip test directories
                    
                    backup_path = self.backup_dir / file_path.name
                    if file_path.is_file():
                        shutil.copy2(file_path, backup_path)
                    elif file_path.is_dir():
                        shutil.copytree(file_path, backup_path, dirs_exist_ok=True)
                
                self.logger.info(f"Backed up {len(output_files)} items")
            else:
                self.logger.info("No existing output files to backup")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def reset_to_backup(self) -> bool:
        """Reset output directory to backup state"""
        try:
            self.logger.info("Resetting to backup state...")
            
            # Remove all non-test files from output directory
            for file_path in self.config.paths.output_root.glob("*"):
                if file_path.name.startswith("splitting_test_"):
                    continue  # Keep test directories
                
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            
            # Restore from backup
            backup_files = list(self.backup_dir.glob("*"))
            for backup_path in backup_files:
                restore_path = self.config.paths.output_root / backup_path.name
                if backup_path.is_file():
                    shutil.copy2(backup_path, restore_path)
                elif backup_path.is_dir():
                    shutil.copytree(backup_path, restore_path, dirs_exist_ok=True)
            
            self.logger.info(f"Restored {len(backup_files)} items from backup")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset to backup: {e}")
            return False
    
    def test_option(self, option_name: str) -> Dict[str, Any]:
        """Test a specific splitting option"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TESTING OPTION {option_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Initialize splitting system (automatically loads file mappings)
            splitting_system = DatasetSplittingSystem()
            
            # Run the specific option
            if option_name.lower() == 'a':
                result = splitting_system.implement_split_option_a()
            elif option_name.lower() == 'b':
                result = splitting_system.implement_split_option_b()
            elif option_name.lower() == 'c':
                result = splitting_system.implement_split_option_c()
            else:
                raise ValueError(f"Unknown option: {option_name}")
            
            # Save results to test directory
            option_results_dir = self.results_dir / f"option_{option_name.lower()}"
            option_results_dir.mkdir(exist_ok=True)
            
            # Save comprehensive results
            results_file = option_results_dir / f"results_{option_name.lower()}_{self.timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Save file mappings
            mappings_file = option_results_dir / f"file_mappings_{option_name.lower()}_{self.timestamp}.csv"
            self.save_mappings_to_csv(result['file_mappings'], mappings_file)
            
            # Generate summary
            summary = self.generate_option_summary(option_name, result)
            summary_file = option_results_dir / f"summary_{option_name.lower()}_{self.timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            self.logger.info(f"✓ Option {option_name.upper()} completed successfully")
            self.logger.info(f"  Results saved to: {option_results_dir}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"✗ Option {option_name.upper()} failed: {e}")
            return {'error': str(e)}
    
    def save_mappings_to_csv(self, file_mappings: Dict[str, Any], output_file: Path):
        """Save file mappings to CSV format"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if not file_mappings:
                return
            
            # Get field names from first mapping
            first_mapping = next(iter(file_mappings.values()))
            fieldnames = ['original_filename'] + list(first_mapping.keys())
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for filename, mapping in file_mappings.items():
                row = {'original_filename': filename}
                row.update(mapping)
                writer.writerow(row)
    
    def generate_option_summary(self, option_name: str, result: Dict[str, Any]) -> str:
        """Generate a summary report for an option"""
        if 'error' in result:
            return f"Option {option_name.upper()} FAILED: {result['error']}"
        
        config = result['configuration']
        stats = result['global_stats']
        
        summary = []
        summary.append(f"OPTION {option_name.upper()} SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Configuration: {config['name']}")
        summary.append(f"Description: {config['description']}")
        summary.append("")
        
        # Split statistics
        summary.append("SPLIT STATISTICS:")
        summary.append("-" * 20)
        
        total_files = 0
        total_genuine = 0
        total_deepfake = 0
        
        for split_name, split_stats in stats.items():
            total_files += split_stats['total_files']
            total_genuine += split_stats['genuine_files']
            total_deepfake += split_stats['deepfake_files']
            
            genuine_pct = (split_stats['genuine_files'] / split_stats['total_files'] * 100) if split_stats['total_files'] > 0 else 0
            
            summary.append(f"{split_name.upper()}:")
            summary.append(f"  Total files: {split_stats['total_files']:,}")
            summary.append(f"  Genuine: {split_stats['genuine_files']:,} ({genuine_pct:.1f}%)")
            summary.append(f"  Deepfake: {split_stats['deepfake_files']:,} ({100-genuine_pct:.1f}%)")
            summary.append(f"  Users: {split_stats['users_count']}")
            summary.append(f"  Attack types: {len(split_stats['attack_distribution'])}")
            summary.append("")
        
        # Overall statistics
        summary.append("OVERALL STATISTICS:")
        summary.append("-" * 20)
        summary.append(f"Total files processed: {total_files:,}")
        summary.append(f"Total genuine files: {total_genuine:,} ({total_genuine/total_files*100:.1f}%)")
        summary.append(f"Total deepfake files: {total_deepfake:,} ({total_deepfake/total_files*100:.1f}%)")
        summary.append(f"Number of splits: {len(stats)}")
        
        return "\n".join(summary)
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run tests for all three options"""
        self.logger.info("Starting comprehensive splitting options test...")
        
        # Create initial backup
        if not self.create_backup():
            self.logger.error("Failed to create backup - aborting tests")
            return {}
        
        results = {}
        options = ['A', 'B', 'C']
        
        for i, option in enumerate(options):
            self.logger.info(f"\nTesting option {option} ({i+1}/{len(options)})...")
            
            # Reset to clean state before each test (except first)
            if i > 0:
                if not self.reset_to_backup():
                    self.logger.error(f"Failed to reset before option {option}")
                    continue
            
            # Test the option
            result = self.test_option(option)
            results[option] = result
            
            # Log completion
            if 'error' not in result:
                self.logger.info(f"✓ Option {option} test completed successfully")
            else:
                self.logger.error(f"✗ Option {option} test failed")
        
        # Generate comparison report
        self.generate_comparison_report(results)
        
        self.logger.info(f"\nAll tests completed! Results saved to: {self.test_dir}")
        return results
    
    def generate_comparison_report(self, results: Dict[str, Dict[str, Any]]):
        """Generate a comparison report of all options"""
        comparison_file = self.results_dir / f"options_comparison_{self.timestamp}.txt"
        
        comparison = []
        comparison.append("DATASET SPLITTING OPTIONS COMPARISON")
        comparison.append("=" * 60)
        comparison.append(f"Test timestamp: {self.timestamp}")
        comparison.append("")
        
        for option_name, result in results.items():
            if 'error' in result:
                comparison.append(f"OPTION {option_name}: FAILED - {result['error']}")
                comparison.append("")
                continue
            
            config = result['configuration']
            stats = result['global_stats']
            
            comparison.append(f"OPTION {option_name}: {config['name']}")
            comparison.append(f"Description: {config['description']}")
            comparison.append("")
            
            # Summary statistics
            total_files = sum(split_stats['total_files'] for split_stats in stats.values())
            total_users = sum(split_stats['users_count'] for split_stats in stats.values())
            
            comparison.append(f"  Total files: {total_files:,}")
            comparison.append(f"  Total users: {total_users}")
            comparison.append(f"  Number of splits: {len(stats)}")
            
            for split_name, split_stats in stats.items():
                genuine_pct = (split_stats['genuine_files'] / split_stats['total_files'] * 100) if split_stats['total_files'] > 0 else 0
                comparison.append(f"  {split_name.upper()}: {split_stats['total_files']:,} files "
                                f"({split_stats['genuine_files']:,} genuine [{genuine_pct:.1f}%], "
                                f"{split_stats['deepfake_files']:,} deepfake)")
            
            comparison.append("")
        
        # Write comparison report
        with open(comparison_file, 'w') as f:
            f.write("\n".join(comparison))
        
        # Also print to console
        print("\n" + "\n".join(comparison))

def main():
    """Main function to run all tests"""
    print("🧪 Starting Dataset Splitting Options Test Suite...")
    print("=" * 60)
    
    try:
        # Initialize test system
        test_system = SplittingOptionsTestSystem()
        
        # Run all tests
        results = test_system.run_all_tests()
        
        # Summary
        successful_tests = sum(1 for result in results.values() if 'error' not in result)
        total_tests = len(results)
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Successful: {successful_tests}/{total_tests}")
        print(f"   Results saved to: {test_system.test_dir}")
        
        if successful_tests == total_tests:
            print("✅ All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed - check logs for details")
            return 1
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 