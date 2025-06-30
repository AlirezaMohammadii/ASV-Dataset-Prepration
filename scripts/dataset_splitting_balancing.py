#!/usr/bin/env python3
"""
Dataset Splitting and Balancing System
======================================

Task 4: Create balanced train/dev/test splits compatible with the Deep ASV Detection workflow.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.dataset_config import config
from utils.logging_utils import setup_logger


@dataclass
class SplitConfiguration:
    """Configuration for a dataset split"""
    name: str
    description: str
    train_sources: List[str]
    dev_sources: List[str] 
    test_sources: List[str]
    train_ratio: Optional[float] = None
    dev_ratio: Optional[float] = None
    test_ratio: Optional[float] = None


class DatasetSplittingSystem:
    """Comprehensive dataset splitting and balancing system"""
    
    def __init__(self):
        self.logger = setup_logger('DatasetSplittingSystem', config.paths.logs_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load data from previous tasks
        self.file_mappings = self._load_file_mappings()
        self.user_assignments = self._load_user_assignments()
        
        # Split configurations
        self.split_options = self._create_split_configurations()
        
        # Random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def _load_file_mappings(self) -> Dict[str, Any]:
        """Load file mappings from Task 2"""
        self.logger.info("Loading file mappings from Task 2...")
        
        analysis_dir = config.paths.analysis_output_dir
        mapping_files = list(analysis_dir.glob("file_label_mappings_*.csv"))
        
        if not mapping_files:
            raise FileNotFoundError("No file mapping results found from Task 2")
        
        latest_file = max(mapping_files, key=lambda p: p.stat().st_mtime)
        self.logger.info(f"Loading mappings from: {latest_file}")
        
        df = pd.read_csv(latest_file)
        mappings = {}
        
        for _, row in df.iterrows():
            mappings[row['Original_Filename']] = {
                'original_speaker': row['Original_Speaker'],
                'original_label': row['Original_Label'],
                'original_attack_type': row['Original_Attack_Type'] if pd.notna(row['Original_Attack_Type']) else None,
                'mapped_label': row['Mapped_Label'],
                'mapped_attack_category': row['Mapped_Attack_Category'],
                'split': row['Split'],
                'file_exists': row['File_Exists'],
                'file_size_mb': row['File_Size_MB'] if pd.notna(row['File_Size_MB']) else None
            }
        
        self.logger.info(f"✓ Loaded {len(mappings)} file mappings")
        return mappings
    
    def _load_user_assignments(self) -> List[Dict[str, Any]]:
        """Load user assignments (create from file mappings if needed)"""
        self.logger.info("Loading user assignments...")
        
        analysis_dir = config.paths.analysis_output_dir
        
        # Try to load fixed version first
        fixed_files = list(analysis_dir.glob("data_structure_analysis_fixed_*.json"))
        if fixed_files:
            latest_file = max(fixed_files, key=lambda p: p.stat().st_mtime)
            self.logger.info(f"Loading FIXED user assignments from: {latest_file}")
            with open(latest_file, 'r') as f:
                data = json.load(f)
            user_assignments = data['detailed_analyses']['user_mapping_strategy']['user_assignments']
            self.logger.info(f"✓ Loaded {len(user_assignments)} user assignments")
            return user_assignments
        
        # Create user assignments from file mappings if not available
        self.logger.info("Creating user assignments from file mappings...")
        
        # Extract unique speakers from file mappings
        speakers = set()
        for mapping in self.file_mappings.values():
            if mapping['file_exists']:
                speakers.add(mapping['original_speaker'])
        
        # Create user assignments (1 user per speaker for simplicity)
        user_assignments = []
        for i, speaker in enumerate(sorted(speakers), 1):
            user_assignments.append({
                'user_id': i,
                'speakers': [speaker],
                'total_files': sum(1 for m in self.file_mappings.values() 
                                 if m['file_exists'] and m['original_speaker'] == speaker),
                'splits': list(set(m['split'] for m in self.file_mappings.values() 
                                 if m['file_exists'] and m['original_speaker'] == speaker))
            })
        
        self.logger.info(f"✓ Created {len(user_assignments)} user assignments from speakers")
        return user_assignments
    
    def _create_split_configurations(self) -> Dict[str, SplitConfiguration]:
        """Create the three split configuration options"""
        
        configurations = {
            'option_a': SplitConfiguration(
                name="Option A: Original Splits",
                description="Use train for training, dev for validation, eval for testing",
                train_sources=['train'],
                dev_sources=['dev'],
                test_sources=['eval']
            ),
            
            'option_b': SplitConfiguration(
                name="Option B: Combined Redistribution", 
                description="Combine all and create new balanced splits",
                train_sources=['train', 'dev', 'eval'],
                dev_sources=['train', 'dev', 'eval'],
                test_sources=['train', 'dev', 'eval'],
                train_ratio=0.7,
                dev_ratio=0.15,
                test_ratio=0.15
            ),
            
            'option_c': SplitConfiguration(
                name="Option C: Train+Dev vs Eval",
                description="Use train+dev for training, eval for testing",
                train_sources=['train', 'dev'],
                dev_sources=[],  # No separate dev set
                test_sources=['eval']
            )
        }
        
        return configurations
    
    def create_speaker_to_user_mapping(self) -> Dict[str, str]:
        """Create mapping from speaker to user ID"""
        speaker_to_user = {}
        for user_assignment in self.user_assignments:
            user_id = user_assignment['user_id']
            for speaker in user_assignment['speakers']:
                speaker_to_user[speaker] = user_id
        
        self.logger.info(f"✓ Created speaker-to-user mapping for {len(speaker_to_user)} speakers")
        return speaker_to_user
    
    def analyze_current_distribution(self) -> Dict[str, Any]:
        """Analyze current file distribution across splits and users"""
        self.logger.info("Analyzing current file distribution...")
        
        speaker_to_user = self.create_speaker_to_user_mapping()
        
        # Distribution by original split
        split_stats = defaultdict(lambda: {
            'total_files': 0,
            'genuine_files': 0,
            'deepfake_files': 0,
            'attack_distribution': Counter(),
            'speakers': set(),
            'users': set()
        })
        
        for filename, mapping in self.file_mappings.items():
            if not mapping['file_exists']:
                continue
            
            split = mapping['split']
            speaker = mapping['original_speaker']
            label = mapping['mapped_label']
            attack_category = mapping['mapped_attack_category']
            
            # Skip if speaker not assigned
            if speaker not in speaker_to_user:
                continue
            
            user_id = speaker_to_user[speaker]
            
            # Update split stats
            split_stats[split]['total_files'] += 1
            split_stats[split]['speakers'].add(speaker)
            split_stats[split]['users'].add(user_id)
            
            if label == 'genuine':
                split_stats[split]['genuine_files'] += 1
            else:
                split_stats[split]['deepfake_files'] += 1
                split_stats[split]['attack_distribution'][attack_category] += 1
        
        # Convert sets to lists for JSON serialization
        for split_data in split_stats.values():
            split_data['speakers'] = list(split_data['speakers'])
            split_data['users'] = list(split_data['users'])
            split_data['attack_distribution'] = dict(split_data['attack_distribution'])
        
        analysis = {
            'by_original_split': dict(split_stats),
            'summary': {
                'total_files': sum(data['total_files'] for data in split_stats.values()),
                'total_speakers': len(speaker_to_user),
                'total_users': len(set(speaker_to_user.values())),
                'splits': list(split_stats.keys())
            }
        }
        
        self.logger.info("✓ Current distribution analysis completed")
        return analysis
    
    def implement_split_option_a(self) -> Dict[str, Any]:
        """Implement Option A: Use original splits as-is"""
        self.logger.info("Implementing Option A: Original Splits...")
        
        config = self.split_options['option_a']
        speaker_to_user = self.create_speaker_to_user_mapping()
        
        # Statistics for each split
        split_stats = {
            'train': {'total_files': 0, 'genuine_files': 0, 'deepfake_files': 0, 'attack_distribution': Counter(), 'users': set()},
            'dev': {'total_files': 0, 'genuine_files': 0, 'deepfake_files': 0, 'attack_distribution': Counter(), 'users': set()},
            'test': {'total_files': 0, 'genuine_files': 0, 'deepfake_files': 0, 'attack_distribution': Counter(), 'users': set()}
        }
        
        file_mappings = {}
        
        # Process each file
        for filename, mapping in self.file_mappings.items():
            if not mapping['file_exists']:
                continue
            
            speaker = mapping['original_speaker']
            if speaker not in speaker_to_user:
                continue
            
            user_id = speaker_to_user[speaker]
            original_split = mapping['split']
            label = mapping['mapped_label']
            attack_category = mapping['mapped_attack_category']
            
            # Map original splits to new split names
            if original_split == 'train':
                new_split = 'train'
            elif original_split == 'dev':
                new_split = 'dev'
            elif original_split == 'eval':
                new_split = 'test'
            else:
                continue
            
            # Update statistics
            split_stats[new_split]['total_files'] += 1
            split_stats[new_split]['users'].add(user_id)
            
            if label == 'genuine':
                split_stats[new_split]['genuine_files'] += 1
            else:
                split_stats[new_split]['deepfake_files'] += 1
                split_stats[new_split]['attack_distribution'][attack_category] += 1
            
            # Create file mapping
            file_mappings[filename] = {
                'original_filename': filename,
                'original_speaker': speaker,
                'original_split': original_split,
                'new_split': new_split,
                'user_id': user_id,
                'label': label,
                'attack_category': attack_category
            }
        
        # Convert sets to lists and counters to dicts
        for split_data in split_stats.values():
            split_data['users'] = list(split_data['users'])
            split_data['users_count'] = len(split_data['users'])
            split_data['attack_distribution'] = dict(split_data['attack_distribution'])
        
        result = {
            'configuration': {
                'name': config.name,
                'description': config.description,
                'train_sources': config.train_sources,
                'dev_sources': config.dev_sources,
                'test_sources': config.test_sources
            },
            'global_stats': split_stats,
            'file_mappings': file_mappings,
            'timestamp': self.timestamp
        }
        
        self.logger.info("✓ Option A implementation completed")
        return result
    
    def implement_split_option_b(self) -> Dict[str, Any]:
        """Implement Option B: Combine all and create new balanced splits"""
        self.logger.info("Implementing Option B: Combined Redistribution...")
        
        config = self.split_options['option_b']
        speaker_to_user = self.create_speaker_to_user_mapping()
        
        # Group files by user
        user_files = defaultdict(list)
        
        for filename, mapping in self.file_mappings.items():
            if not mapping['file_exists']:
                continue
            
            speaker = mapping['original_speaker']
            if speaker not in speaker_to_user:
                continue
            
            user_id = speaker_to_user[speaker]
            user_files[user_id].append((filename, mapping))
        
        # Statistics for each split
        split_stats = {
            'train': {'total_files': 0, 'genuine_files': 0, 'deepfake_files': 0, 'attack_distribution': Counter(), 'users': set()},
            'dev': {'total_files': 0, 'genuine_files': 0, 'deepfake_files': 0, 'attack_distribution': Counter(), 'users': set()},
            'test': {'total_files': 0, 'genuine_files': 0, 'deepfake_files': 0, 'attack_distribution': Counter(), 'users': set()}
        }
        
        file_mappings = {}
        
        for user_id, files in user_files.items():
            # Separate genuine and deepfake files
            genuine_files = [(f, m) for f, m in files if m['mapped_label'] == 'genuine']
            deepfake_files = [(f, m) for f, m in files if m['mapped_label'] == 'deepfake']
            
            # Shuffle for random distribution
            random.shuffle(genuine_files)
            random.shuffle(deepfake_files)
            
            # Split genuine files
            n_genuine = len(genuine_files)
            train_genuine_end = int(n_genuine * config.train_ratio)
            dev_genuine_end = train_genuine_end + int(n_genuine * config.dev_ratio)
            
            genuine_splits = {
                'train': genuine_files[:train_genuine_end],
                'dev': genuine_files[train_genuine_end:dev_genuine_end],
                'test': genuine_files[dev_genuine_end:]
            }
            
            # Split deepfake files
            n_deepfake = len(deepfake_files)
            train_deepfake_end = int(n_deepfake * config.train_ratio)
            dev_deepfake_end = train_deepfake_end + int(n_deepfake * config.dev_ratio)
            
            deepfake_splits = {
                'train': deepfake_files[:train_deepfake_end],
                'dev': deepfake_files[train_deepfake_end:dev_deepfake_end],
                'test': deepfake_files[dev_deepfake_end:]
            }
            
            # Process splits
            for split_name in ['train', 'dev', 'test']:
                if genuine_splits[split_name] or deepfake_splits[split_name]:
                    split_stats[split_name]['users'].add(user_id)
                
                # Process genuine files
                for filename, mapping in genuine_splits[split_name]:
                    split_stats[split_name]['total_files'] += 1
                    split_stats[split_name]['genuine_files'] += 1
                    
                    file_mappings[filename] = {
                        'original_filename': filename,
                        'original_speaker': mapping['original_speaker'],
                        'original_split': mapping['split'],
                        'new_split': split_name,
                        'user_id': user_id,
                        'label': 'genuine',
                        'attack_category': None
                    }
                
                # Process deepfake files
                for filename, mapping in deepfake_splits[split_name]:
                    attack_category = mapping['mapped_attack_category']
                    split_stats[split_name]['total_files'] += 1
                    split_stats[split_name]['deepfake_files'] += 1
                    split_stats[split_name]['attack_distribution'][attack_category] += 1
                    
                    file_mappings[filename] = {
                        'original_filename': filename,
                        'original_speaker': mapping['original_speaker'],
                        'original_split': mapping['split'],
                        'new_split': split_name,
                        'user_id': user_id,
                        'label': 'deepfake',
                        'attack_category': attack_category
                    }
        
        # Convert sets to lists and counters to dicts
        for split_data in split_stats.values():
            split_data['users'] = list(split_data['users'])
            split_data['users_count'] = len(split_data['users'])
            split_data['attack_distribution'] = dict(split_data['attack_distribution'])
        
        result = {
            'configuration': {
                'name': config.name,
                'description': config.description,
                'train_sources': config.train_sources,
                'dev_sources': config.dev_sources,
                'test_sources': config.test_sources,
                'train_ratio': config.train_ratio,
                'dev_ratio': config.dev_ratio,
                'test_ratio': config.test_ratio
            },
            'global_stats': split_stats,
            'file_mappings': file_mappings,
            'timestamp': self.timestamp
        }
        
        self.logger.info("✓ Option B implementation completed")
        return result
    
    def implement_split_option_c(self) -> Dict[str, Any]:
        """Implement Option C: Train+Dev for training, Eval for testing"""
        self.logger.info("Implementing Option C: Train+Dev vs Eval...")
        
        config = self.split_options['option_c']
        speaker_to_user = self.create_speaker_to_user_mapping()
        
        # Statistics for each split (only train and test)
        split_stats = {
            'train': {'total_files': 0, 'genuine_files': 0, 'deepfake_files': 0, 'attack_distribution': Counter(), 'users': set()},
            'test': {'total_files': 0, 'genuine_files': 0, 'deepfake_files': 0, 'attack_distribution': Counter(), 'users': set()}
        }
        
        file_mappings = {}
        
        # Process each file
        for filename, mapping in self.file_mappings.items():
            if not mapping['file_exists']:
                continue
            
            speaker = mapping['original_speaker']
            if speaker not in speaker_to_user:
                continue
            
            user_id = speaker_to_user[speaker]
            original_split = mapping['split']
            label = mapping['mapped_label']
            attack_category = mapping['mapped_attack_category']
            
            # Map original splits to new split names
            if original_split in ['train', 'dev']:
                new_split = 'train'
            elif original_split == 'eval':
                new_split = 'test'
            else:
                continue
            
            # Update statistics
            split_stats[new_split]['total_files'] += 1
            split_stats[new_split]['users'].add(user_id)
            
            if label == 'genuine':
                split_stats[new_split]['genuine_files'] += 1
            else:
                split_stats[new_split]['deepfake_files'] += 1
                split_stats[new_split]['attack_distribution'][attack_category] += 1
            
            # Create file mapping
            file_mappings[filename] = {
                'original_filename': filename,
                'original_speaker': speaker,
                'original_split': original_split,
                'new_split': new_split,
                'user_id': user_id,
                'label': label,
                'attack_category': attack_category
            }
        
        # Convert sets to lists and counters to dicts
        for split_data in split_stats.values():
            split_data['users'] = list(split_data['users'])
            split_data['users_count'] = len(split_data['users'])
            split_data['attack_distribution'] = dict(split_data['attack_distribution'])
        
        result = {
            'configuration': {
                'name': config.name,
                'description': config.description,
                'train_sources': config.train_sources,
                'dev_sources': config.dev_sources,
                'test_sources': config.test_sources
            },
            'global_stats': split_stats,
            'file_mappings': file_mappings,
            'timestamp': self.timestamp
        }
        
        self.logger.info("✓ Option C implementation completed")
        return result
    
    def save_split_results(self, results: Dict[str, Dict[str, Any]]) -> List[Path]:
        """Save all split results to files"""
        saved_files = []
        output_dir = config.paths.analysis_output_dir
        
        try:
            for option_name, result in results.items():
                # Save comprehensive analysis
                analysis_file = output_dir / f"dataset_splits_{option_name}_{self.timestamp}.json"
                
                with open(analysis_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                saved_files.append(analysis_file)
                self.logger.info(f"✓ Saved {option_name} analysis: {analysis_file}")
                
                # Save file mappings
                mappings_file = output_dir / f"file_mappings_{option_name}_{self.timestamp}.csv"
                
                mappings_data = []
                for filename, mapping in result['file_mappings'].items():
                    mappings_data.append({
                        'Original_Filename': mapping['original_filename'],
                        'Original_Speaker': mapping['original_speaker'],
                        'Original_Split': mapping['original_split'],
                        'New_Split': mapping['new_split'],
                        'User_ID': mapping['user_id'],
                        'Label': mapping['label'],
                        'Attack_Category': mapping['attack_category'] or ''
                    })
                
                if mappings_data:
                    df = pd.DataFrame(mappings_data)
                    df.to_csv(mappings_file, index=False)
                    saved_files.append(mappings_file)
                    self.logger.info(f"✓ Saved {option_name} mappings: {mappings_file}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Failed to save split results: {e}")
            return saved_files
    
    def print_option_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Print comparison of all three options"""
        
        print("\n" + "="*100)
        print("DATASET SPLITTING OPTIONS - COMPREHENSIVE COMPARISON")
        print("="*100)
        
        for option_name, result in results.items():
            config = result['configuration']
            stats = result['global_stats']
            
            print(f"\n🔹 {config['name'].upper()}")
            print(f"   {config['description']}")
            
            # Print split statistics
            for split_name, split_stats in stats.items():
                if split_stats['total_files'] == 0:
                    continue
                
                genuine_pct = (split_stats['genuine_files'] / split_stats['total_files']) * 100
                print(f"   📊 {split_name.upper()}: {split_stats['total_files']:,} files "
                      f"({split_stats['genuine_files']:,} genuine [{genuine_pct:.1f}%], "
                      f"{split_stats['deepfake_files']:,} deepfake)")
                print(f"      Users with data: {split_stats['users_count']}")
                
                # Top attack types
                if split_stats['attack_distribution']:
                    top_attacks = sorted(split_stats['attack_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)[:3]
                    attack_str = ", ".join([f"{att}: {count}" for att, count in top_attacks])
                    print(f"      Top attacks: {attack_str}")
            
            print()
        
        print("="*100)
        print("RECOMMENDATION GUIDE:")
        print("• Option A: Best for comparing with existing research using original splits")
        print("• Option B: Best for balanced training with custom validation")
        print("• Option C: Best for maximum training data with separate test set")
        print("="*100)


def main():
    """Main execution function for Task 4"""
    print("="*80)
    print("DATASET SPLITTING AND BALANCING SYSTEM - TASK 4")
    print("Create balanced train/dev/test splits for Deep ASV Detection")
    print("="*80)
    
    # Initialize system
    splitter = DatasetSplittingSystem()
    
    try:
        # Step 1: Analyze current distribution
        print("\n📊 Step 1: Analyzing current file distribution...")
        current_analysis = splitter.analyze_current_distribution()
        
        print(f"✓ Total files: {current_analysis['summary']['total_files']:,}")
        print(f"✓ Total speakers: {current_analysis['summary']['total_speakers']}")
        print(f"✓ Total users: {current_analysis['summary']['total_users']}")
        
        # Step 2: Implement all three options
        print("\n🔀 Step 2: Implementing all split options...")
        
        print("   Implementing Option A...")
        option_a_result = splitter.implement_split_option_a()
        
        print("   Implementing Option B...")
        option_b_result = splitter.implement_split_option_b()
        
        print("   Implementing Option C...")
        option_c_result = splitter.implement_split_option_c()
        
        results = {
            'option_a': option_a_result,
            'option_b': option_b_result,
            'option_c': option_c_result
        }
        
        # Step 3: Save results
        print("\n💾 Step 3: Saving split results...")
        saved_files = splitter.save_split_results(results)
        
        # Step 4: Display comparison
        splitter.print_option_comparison(results)
        
        print(f"\n✅ Task 4 completed successfully!")
        print(f"📁 Results saved to:")
        for file_path in saved_files:
            print(f"  • {file_path}")
        
        print(f"\n🔄 NEXT STEPS:")
        print("1. Review the three splitting options above")
        print("2. Choose the option that best fits your research needs")
        print("3. Use the corresponding file mappings for dataset organization")
        print("4. Proceed with Task 3 file organization using chosen split")
        
        return 0
        
    except Exception as e:
        splitter.logger.error(f"Task 4 failed: {e}")
        print(f"❌ Task 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
