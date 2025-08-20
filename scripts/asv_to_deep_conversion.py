#!/usr/bin/env python3
"""
ASV to Deep ASV Detection Dataset Conversion System
===================================================

This script converts ASVspoof-2019 dataset to Deep ASV Detection format
with seamless split option selection and copying to dataASV directory.

Features:
- Interactive split option selection
- Complete dataset conversion and copying
- Deep ASV Detection directory structure creation
- Progress tracking and validation
- Cleanup of temporary files

Author: ASV Dataset Preparation System
"""

import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.dataset_config import config
from utils.logging_utils import setup_logger
from scripts.dataset_splitting_balancing import DatasetSplittingSystem
from scripts.asv_dataset_integration import ASVDatasetIntegrator, IntegrationConfig, ConversionMode, ValidationLevel


class ASVToDeepConversionSystem:
    """Complete conversion system from ASVspoof-2019 to Deep ASV Detection format"""
    
    def __init__(self, config_instance=None):
        # Use provided config or fall back to global config
        self.config = config_instance if config_instance is not None else config
        self.logger = setup_logger('ASVToDeepConversion', self.config.paths.logs_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Deep ASV Detection paths
        self.deep_asv_root = Path("../../Deep ASV Detection")
        self.data_asv_dir = self.deep_asv_root / "dataASV"
        
        # Conversion results
        self.conversion_results = {}
        
    def _normalize_user_id(self, user_id_value) -> int:
        """Return numeric user id from either int or 'user_XX' string."""
        try:
            if isinstance(user_id_value, int):
                return user_id_value
            s = str(user_id_value)
            digits = ''.join(ch for ch in s if ch.isdigit())
            return int(digits) if digits else 0
        except Exception:
            return 0

    def display_split_options(self):
        """Display available split options with detailed information"""
        print("\n" + "="*70)
        print("🔀 DATASET SPLITTING OPTIONS")
        print("="*70)
        
        options = {
            'A': {
                'name': 'Original Splits',
                'description': 'Use train for training, dev for validation, eval for testing',
                'details': [
                    'TRAIN: 25,380 files (20 users) - Original train split',
                    'DEV: 24,844 files (19 users) - Original dev split', 
                    'TEST: 71,237 files (67 users) - Original eval split',
                    'Best for: Maintaining original dataset structure'
                ]
            },
            'B': {
                'name': 'Combined Redistribution',
                'description': 'Combine all and create new balanced splits (70/15/15)',
                'details': [
                    'TRAIN: ~84,945 files (106 users) - All users represented',
                    'DEV: ~18,125 files (106 users) - All users represented',
                    'TEST: ~18,391 files (106 users) - All users represented',
                    'Best for: Balanced training with all speakers in all splits'
                ]
            },
            'C': {
                'name': 'Train+Dev vs Eval',
                'description': 'Use train+dev for training, eval for testing',
                'details': [
                    'TRAIN: 50,224 files (39 users) - Combined train+dev',
                    'TEST: 71,237 files (67 users) - Original eval',
                    'DEV: 0 files - No separate validation set',
                    'Best for: Maximum training data, simple train/test split'
                ]
            }
        }
        
        for option, info in options.items():
            print(f"\n📊 OPTION {option}: {info['name']}")
            print(f"   {info['description']}")
            print("   Details:")
            for detail in info['details']:
                print(f"     • {detail}")
        
        print("\n" + "="*70)
    
    def get_user_split_choice(self) -> str:
        """Get user's choice for split option with validation"""
        self.display_split_options()
        
        while True:
            print("\n💡 Choose your splitting strategy:")
            choice = input("Enter A, B, or C (or 'help' for details): ").strip().upper()
            
            if choice in ['A', 'B', 'C']:
                return choice
            elif choice == 'HELP':
                self.display_split_options()
            else:
                print("❌ Invalid choice. Please enter A, B, or C.")
    
    def prepare_deep_asv_directory(self):
        """Create and prepare the dataASV directory structure"""
        self.logger.info("Preparing Deep ASV Detection directory structure...")
        
        # Create dataASV directory if it doesn't exist
        self.data_asv_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split subdirectories
        split_dirs = ['train', 'dev', 'test']
        for split_dir in split_dirs:
            (self.data_asv_dir / split_dir).mkdir(exist_ok=True)
        
        self.logger.info(f"✓ Created directory structure at: {self.data_asv_dir}")
        print(f"📁 Directory structure created at: {self.data_asv_dir}")
    
    def run_dataset_analysis(self) -> bool:
        """Run dataset analysis to generate required mappings"""
        print("\n🔍 STEP 1: RUNNING DATASET ANALYSIS")
        print("="*50)
        
        try:
            # Create integration config for analysis
            config = IntegrationConfig(
                conversion_mode=ConversionMode.ANALYSIS_ONLY,
                validation_level=ValidationLevel.BASIC,
                enable_backup=False
            )
            
            integrator = ASVDatasetIntegrator(config)
            success = integrator.run_integration()
            
            if success:
                print("✅ Dataset analysis completed successfully")
                self.logger.info("Dataset analysis completed")
                return True
            else:
                print("❌ Dataset analysis failed")
                self.logger.error("Dataset analysis failed")
                return False
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            self.logger.error(f"Analysis failed: {e}")
            return False
    
    def run_dataset_splitting(self, split_option: str) -> Dict[str, Any]:
        """Run dataset splitting with the chosen option"""
        print(f"\n🔀 STEP 2: RUNNING DATASET SPLITTING (OPTION {split_option})")
        print("="*60)
        
        try:
            splitting_system = DatasetSplittingSystem()
            
            # Run the appropriate splitting option
            if split_option == 'A':
                result = splitting_system.implement_split_option_a()
                print("✅ Option A: Original splits applied")
            elif split_option == 'B':
                result = splitting_system.implement_split_option_b()
                print("✅ Option B: Combined redistribution applied")
            elif split_option == 'C':
                result = splitting_system.implement_split_option_c()
                print("✅ Option C: Train+Dev vs Eval applied")
            else:
                raise ValueError(f"Invalid split option: {split_option}")
            
            # Save split results
            saved_files = splitting_system.save_split_results({f'option_{split_option.lower()}': result})
            
            print(f"📊 Split results saved to: {saved_files[0] if saved_files else 'N/A'}")
            self.logger.info(f"Dataset splitting completed with option {split_option}")
            
            return result
            
        except Exception as e:
            print(f"❌ Splitting failed: {e}")
            self.logger.error(f"Splitting failed: {e}")
            raise
    
    def convert_and_copy_files(self, split_result: Dict[str, Any], split_option: str):
        """Convert files to Deep ASV Detection format and copy to dataASV"""
        print(f"\n📁 STEP 3: CONVERTING AND COPYING FILES")
        print("="*50)
        
        file_mappings = split_result['file_mappings']
        global_stats = split_result['global_stats']
        
        # Data directories for original files - use correct paths based on dataset year and scenario
        if getattr(self.config, 'dataset_year', None) and self.config.dataset_year.value == '2021':
            # ASVspoof2021: eval only
            if getattr(self.config, 'scenario', None) and self.config.scenario.value == 'PA':
                data_dirs = {
                    'eval': self.config.paths.pa2021_eval_dir
                }
            else:
                data_dirs = {
                    'eval': self.config.paths.la2021_eval_dir
                }
        else:
            # ASVspoof2019: train, dev, eval
            if getattr(self.config, 'scenario', None) and self.config.scenario.value == 'PA':
                data_dirs = {
                    'train': self.config.paths.pa_train_dir,
                    'dev': self.config.paths.pa_dev_dir,
                    'eval': self.config.paths.pa_eval_dir
                }
            else:
                data_dirs = {
                    'train': self.config.paths.la_train_dir,
                    'dev': self.config.paths.la_dev_dir,
                    'eval': self.config.paths.la_eval_dir
                }
        
        # Progress tracking
        total_files = sum(stats['total_files'] for stats in global_stats.values())
        successful_copies = 0
        failed_copies = 0
        
        print(f"📊 Total files to process: {total_files:,}")
        
        # Process files by split
        with tqdm(total=total_files, desc="Converting files") as pbar:
            for filename, mapping in file_mappings.items():
                try:
                    # Determine source file path
                    original_split = mapping['original_split']
                    source_file = data_dirs[original_split] / filename
                    
                    if not source_file.exists():
                        self.logger.warning(f"Source file not found: {source_file}")
                        failed_copies += 1
                        pbar.update(1)
                        continue
                    
                    # Determine target file path and name
                    new_split = mapping['new_split']
                    user_id_num = self._normalize_user_id(mapping['user_id'])
                    label = mapping['label']
                    attack_category = mapping.get('attack_category', '')
                    
                    # Create user directory
                    user_dir = self.data_asv_dir / new_split / f"user_{user_id_num:02d}"
                    user_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate target filename
                    file_counter = len(list(user_dir.glob(f"user{user_id_num:02d}_{label}_*.flac"))) + 1
                    
                    if label == 'genuine':
                        target_filename = f"user{user_id_num:02d}_genuine_{file_counter:03d}.flac"
                    else:
                        # Map attack category to suffix
                        attack_suffix = self._get_attack_suffix(attack_category)
                        # Optionally include attack ID if available in mapping and config enabled
                        attack_id = mapping.get('attack_id')
                        if self.config.conversion.include_attack_id_in_filename and attack_id:
                            target_filename = f"user{user_id_num:02d}_deepfake_{attack_suffix}_{attack_id.lower()}_{file_counter:03d}.flac"
                        else:
                            target_filename = f"user{user_id_num:02d}_deepfake_{attack_suffix}_{file_counter:03d}.flac"
                    
                    target_file = user_dir / target_filename
                    
                    # Copy file
                    shutil.copy2(source_file, target_file)
                    successful_copies += 1
                    
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Failed to copy {filename}: {e}")
                    failed_copies += 1
                    pbar.update(1)
        
        # Report results
        print(f"\n📊 CONVERSION RESULTS:")
        print(f"  ✅ Successfully copied: {successful_copies:,} files")
        print(f"  ❌ Failed copies: {failed_copies:,} files")
        print(f"  📁 Target directory: {self.data_asv_dir}")
        
        # Save conversion summary
        self.conversion_results = {
            'split_option': split_option,
            'total_files': total_files,
            'successful_copies': successful_copies,
            'failed_copies': failed_copies,
            'target_directory': str(self.data_asv_dir),
            'split_statistics': global_stats,
            'timestamp': self.timestamp
        }
        
        self.logger.info(f"File conversion completed: {successful_copies}/{total_files} files")
    
    def _get_attack_suffix(self, attack_category: Optional[str]) -> str:
        """Get attack suffix for filename based on category"""
        if not attack_category:
            return "tts"
        
        category_map = {
            'TTS': 'tts',
            'VC': 'vc', 
            'TTS_VC': 'tts_vc',
            'pa_perfect': 'pa_perfect',
            'pa_high': 'pa_high',
            'pa_low': 'pa_low'
        }
        
        return category_map.get(attack_category, 'tts')
    
    def validate_conversion(self):
        """Validate the conversion results"""
        print(f"\n✅ STEP 4: VALIDATING CONVERSION")
        print("="*40)
        
        # Check directory structure
        required_splits = ['train', 'dev', 'test']
        
        for split in required_splits:
            split_dir = self.data_asv_dir / split
            if split_dir.exists():
                user_dirs = list(split_dir.glob("user_*"))
                total_files = sum(len(list(user_dir.glob("*.flac"))) for user_dir in user_dirs)
                print(f"  📁 {split.upper()}: {len(user_dirs)} users, {total_files:,} files")
            else:
                print(f"  ❌ {split.upper()}: Directory missing")
        
        # Check file naming conventions
        sample_files = list(self.data_asv_dir.rglob("*.flac"))[:5]
        print(f"\n📝 Sample filenames:")
        for file in sample_files:
            print(f"  • {file.name}")
        
        print(f"\n🎯 Conversion validation completed")
        print(f"📁 Dataset ready at: {self.data_asv_dir}")
    
    def save_conversion_report(self):
        """Save detailed conversion report"""
        report_file = self.config.paths.logs_dir / f"asv_to_deep_conversion_{self.timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.conversion_results, f, indent=2, default=str)
        
        print(f"📄 Conversion report saved: {report_file}")
        self.logger.info(f"Conversion report saved: {report_file}")
    
    def run_complete_conversion(self):
        """Run the complete conversion process"""
        print("🚀 ASV TO DEEP ASV DETECTION CONVERSION")
        print("="*50)
        print("This will convert ASVspoof-2019 to Deep ASV Detection format")
        print("and copy files to the dataASV directory.")
        
        try:
            # Step 1: Get user choice for split option
            split_option = self.get_user_split_choice()
            print(f"\n✅ Selected split option: {split_option}")
            
            # Step 2: Prepare directory structure
            self.prepare_deep_asv_directory()
            
            # Step 3: Run dataset analysis
            if not self.run_dataset_analysis():
                raise Exception("Dataset analysis failed")
            
            # Step 4: Run dataset splitting
            split_result = self.run_dataset_splitting(split_option)
            
            # Step 5: Convert and copy files
            self.convert_and_copy_files(split_result, split_option)
            
            # Step 6: Validate conversion
            self.validate_conversion()
            
            # Step 7: Save report
            self.save_conversion_report()
            
            print("\n🎉 CONVERSION COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"📁 Dataset location: {self.data_asv_dir}")
            print(f"📊 Split option used: {split_option}")
            print(f"✅ Files copied: {self.conversion_results.get('successful_copies', 0):,}")
            print("\n🎯 Your dataset is now ready for Deep ASV Detection main.py!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ CONVERSION FAILED: {e}")
            self.logger.error(f"Conversion failed: {e}")
            return False


def main():
    """Main execution function"""
    converter = ASVToDeepConversionSystem()
    success = converter.run_complete_conversion()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 