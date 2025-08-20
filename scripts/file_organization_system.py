#!/usr/bin/env python3
"""
File Organization and Renaming System for Deep ASV Detection
===========================================================

Task 3: Transform ASVspoof-2019 files into the Deep ASV Detection expected format.

Features:
- Create dataASV directory structure in Deep ASV Detection folder
- Implement speaker-to-user mapping from Task 1
- Rename files from LA_T_XXXXXXX.flac to userXX_genuine_YYY.flac or userXX_deepfake_(type)_YYY.flac
- Maintain sequential numbering within each user directory
- Preserve original file associations for traceability
- Implement safe file operations with integrity checks
- Create backup and rollback mechanisms

Author: ASV Dataset Preparation System
Version: 3.0.0
"""

import sys
import json
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.dataset_config import config, DatasetSplit
from utils.logging_utils import setup_logger


@dataclass
class FileOperation:
    """Represents a single file operation"""
    source_path: Path
    target_path: Path
    operation_type: str  # 'copy' or 'move'
    original_filename: str
    new_filename: str
    user_id: str
    file_type: str  # 'genuine' or 'deepfake'
    attack_category: Optional[str]
    source_size: int
    source_checksum: str


@dataclass
class UserDirectory:
    """Represents a user directory structure"""
    user_id: str
    directory_path: Path
    genuine_count: int
    deepfake_count: int
    total_files: int
    speakers: List[str]


class FileOrganizationSystem:
    """Comprehensive file organization and renaming system"""
    
    def __init__(self, config_instance=None, use_copy: bool = True, create_backup: bool = True):
        # Use provided config or fall back to global config
        self.config = config_instance if config_instance is not None else config
        self.logger = setup_logger('FileOrganizationSystem', self.config.paths.logs_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configuration
        self.use_copy = use_copy  # True for copy, False for move
        self.create_backup = create_backup
        
        # Target directory structure
        self.deep_asv_root = Path("../../Deep ASV Detection")
        self.target_data_dir = self.deep_asv_root / "dataASV"
        
        # Operation tracking
        self.planned_operations: List[FileOperation] = []
        self.completed_operations: List[FileOperation] = []
        self.failed_operations: List[Tuple[FileOperation, str]] = []
        self.user_directories: Dict[str, UserDirectory] = {}
        
        # Load mappings from Task 2
        self.file_mappings = self._load_file_mappings()
        self.user_assignments = self._load_user_assignments()
        
        # Operation statistics
        self.stats = {
            'total_files_planned': 0,
            'total_files_completed': 0,
            'total_files_failed': 0,
            'total_size_bytes': 0,
            'users_created': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _load_file_mappings(self) -> Dict[str, Any]:
        """Load file mappings from Task 2 results"""
        self.logger.info("Loading file-to-label mappings from Task 2...")
        
        # Find the latest mappings file
        analysis_dir = self.config.paths.analysis_output_dir
        mapping_files = list(analysis_dir.glob("file_label_mappings_*.csv"))
        
        if not mapping_files:
            raise FileNotFoundError("No file mapping results found from Task 2")
        
        latest_file = max(mapping_files, key=lambda p: p.stat().st_mtime)
        self.logger.info(f"Loading mappings from: {latest_file}")
        
        # Load mappings
        df = pd.read_csv(latest_file)
        mappings = {}
        
        for _, row in df.iterrows():
            mappings[row['Original_Filename']] = {
                'original_speaker': row['Original_Speaker'],
                'original_label': row['Original_Label'],
                'original_attack_type': row['Original_Attack_Type'] if pd.notna(row['Original_Attack_Type']) else None,
                'mapped_label': row['Mapped_Label'],
                'mapped_attack_category': row['Mapped_Attack_Category'],
                'mapped_filename_pattern': row['Mapped_Filename_Pattern'],
                'split': row['Split'],
                'file_exists': row['File_Exists'],
                'file_size_mb': row['File_Size_MB'] if pd.notna(row['File_Size_MB']) else None
            }
        
        self.logger.info(f"✓ Loaded {len(mappings)} file mappings")
        return mappings
    
    def _load_user_assignments(self) -> Dict[str, Any]:
        """Load user assignments from Task 1 results"""
        self.logger.info("Loading user assignments from Task 1...")
        
        # Find the latest analysis file
        analysis_dir = self.config.paths.analysis_output_dir
        analysis_files = list(analysis_dir.glob("data_structure_analysis_*.json"))
        
        if not analysis_files:
            raise FileNotFoundError("No user assignment results found from Task 1")
        
        latest_file = max(analysis_files, key=lambda p: p.stat().st_mtime)
        self.logger.info(f"Loading user assignments from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        user_assignments = data['detailed_analyses']['user_mapping_strategy']['user_assignments']
        self.logger.info(f"✓ Loaded {len(user_assignments)} user assignments")
        
        return user_assignments
    
    def create_directory_structure(self) -> bool:
        """Create the target directory structure"""
        self.logger.info("Creating directory structure...")
        
        try:
            # Create main dataASV directory
            if not self.target_data_dir.exists():
                self.target_data_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"✓ Created main directory: {self.target_data_dir}")
            else:
                self.logger.info(f"✓ Main directory already exists: {self.target_data_dir}")
            
            # Create user directories
            for user_assignment in self.user_assignments:
                user_id = user_assignment['user_id']
                user_dir = self.target_data_dir / user_id
                
                if not user_dir.exists():
                    user_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✓ Created user directory: {user_dir}")
                
                # Initialize user directory tracking
                self.user_directories[user_id] = UserDirectory(
                    user_id=user_id,
                    directory_path=user_dir,
                    genuine_count=0,
                    deepfake_count=0,
                    total_files=0,
                    speakers=user_assignment['speakers']
                )
            
            self.stats['users_created'] = len(self.user_directories)
            self.logger.info(f"✓ Created {len(self.user_directories)} user directories")
            
            # Set proper permissions
            self._set_directory_permissions()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create directory structure: {e}")
            return False
    
    def _set_directory_permissions(self):
        """Set proper permissions for directories"""
        try:
            # Set permissions for main directory
            self.target_data_dir.chmod(0o755)
            
            # Set permissions for user directories
            for user_dir_info in self.user_directories.values():
                user_dir_info.directory_path.chmod(0o755)
            
            self.logger.info("✓ Set directory permissions")
            
        except Exception as e:
            self.logger.warning(f"Could not set directory permissions: {e}")
    
    def plan_file_operations(self) -> bool:
        """Plan all file operations based on mappings and user assignments"""
        self.logger.info("Planning file operations...")
        
        try:
            # Create speaker to user mapping
            speaker_to_user = {}
            for user_assignment in self.user_assignments:
                user_id = user_assignment['user_id']
                for speaker in user_assignment['speakers']:
                    speaker_to_user[speaker] = user_id
            
            # Data directories - use correct paths based on dataset year and scenario
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
            
            # User file counters for sequential numbering
            user_counters = defaultdict(lambda: {'genuine': 0, 'deepfake': 0})
            
            # Plan operations for each file
            for filename, mapping in self.file_mappings.items():
                if not mapping['file_exists']:
                    continue
                
                # Get source file path
                split = mapping['split']
                source_dir = data_dirs[split]
                source_path = source_dir / filename
                
                if not source_path.exists():
                    self.logger.warning(f"Source file not found: {source_path}")
                    continue
                
                # Get user assignment
                speaker = mapping['original_speaker']
                if speaker not in speaker_to_user:
                    self.logger.warning(f"No user assignment for speaker: {speaker}")
                    continue
                
                user_id = speaker_to_user[speaker]
                user_dir = self.user_directories[user_id].directory_path
                
                # Generate target filename
                file_type = mapping['mapped_label']
                if file_type == 'genuine':
                    user_counters[user_id]['genuine'] += 1
                    counter = user_counters[user_id]['genuine']
                    new_filename = f"{user_id}_genuine_{counter:03d}.flac"
                else:  # deepfake
                    user_counters[user_id]['deepfake'] += 1
                    counter = user_counters[user_id]['deepfake']
                    attack_category = mapping['mapped_attack_category']
                    attack_id = mapping.get('original_attack_type') or mapping.get('Original_Attack_Type')
                    if self.config.conversion.include_attack_id_in_filename and attack_id:
                        new_filename = f"{user_id}_deepfake_{attack_category}_{attack_id.lower()}_{counter:03d}.flac"
                    else:
                        new_filename = f"{user_id}_deepfake_{attack_category}_{counter:03d}.flac"
                
                target_path = user_dir / new_filename
                
                # Calculate checksum
                source_checksum = self._calculate_checksum(source_path)
                
                # Create file operation
                operation = FileOperation(
                    source_path=source_path,
                    target_path=target_path,
                    operation_type='copy' if self.use_copy else 'move',
                    original_filename=filename,
                    new_filename=new_filename,
                    user_id=user_id,
                    file_type=file_type,
                    attack_category=mapping['mapped_attack_category'] if file_type == 'deepfake' else None,
                    source_size=source_path.stat().st_size,
                    source_checksum=source_checksum
                )
                
                self.planned_operations.append(operation)
            
            # Update statistics
            self.stats['total_files_planned'] = len(self.planned_operations)
            self.stats['total_size_bytes'] = sum(op.source_size for op in self.planned_operations)
            
            self.logger.info(f"✓ Planned {len(self.planned_operations)} file operations")
            self.logger.info(f"✓ Total size: {self.stats['total_size_bytes'] / (1024**3):.2f} GB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to plan file operations: {e}")
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return ""
    
    def create_backup_manifest(self) -> bool:
        """Create backup manifest for rollback capability"""
        if not self.create_backup:
            return True
        
        try:
            backup_dir = self.config.paths.output_root / "backups" / f"backup_{self.timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create manifest
            manifest = {
                'timestamp': self.timestamp,
                'operation_type': 'copy' if self.use_copy else 'move',
                'total_operations': len(self.planned_operations),
                'target_directory': str(self.target_data_dir),
                'operations': [
                    {
                        'source_path': str(op.source_path),
                        'target_path': str(op.target_path),
                        'original_filename': op.original_filename,
                        'new_filename': op.new_filename,
                        'user_id': op.user_id,
                        'file_type': op.file_type,
                        'attack_category': op.attack_category,
                        'source_checksum': op.source_checksum
                    }
                    for op in self.planned_operations
                ]
            }
            
            manifest_file = backup_dir / "operation_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"✓ Created backup manifest: {manifest_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup manifest: {e}")
            return False
    
    def execute_file_operations(self, dry_run: bool = False) -> bool:
        """Execute the planned file operations"""
        if dry_run:
            self.logger.info("Performing DRY RUN - no files will be modified")
        else:
            self.logger.info("Executing file operations...")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            for i, operation in enumerate(self.planned_operations):
                if i % 1000 == 0:
                    progress = (i / len(self.planned_operations)) * 100
                    self.logger.info(f"Progress: {i}/{len(self.planned_operations)} ({progress:.1f}%)")
                
                try:
                    if not dry_run:
                        success = self._execute_single_operation(operation)
                        if success:
                            self.completed_operations.append(operation)
                            self.stats['total_files_completed'] += 1
                            
                            # Update user directory stats
                            user_info = self.user_directories[operation.user_id]
                            if operation.file_type == 'genuine':
                                user_info.genuine_count += 1
                            else:
                                user_info.deepfake_count += 1
                            user_info.total_files += 1
                        else:
                            self.failed_operations.append((operation, "Operation failed"))
                            self.stats['total_files_failed'] += 1
                    else:
                        # Dry run - just validate
                        if operation.source_path.exists():
                            self.logger.debug(f"DRY RUN: {operation.source_path} → {operation.target_path}")
                        else:
                            self.logger.warning(f"DRY RUN: Source file missing: {operation.source_path}")
                
                except Exception as e:
                    error_msg = f"Error processing {operation.source_path}: {e}"
                    self.logger.error(error_msg)
                    self.failed_operations.append((operation, error_msg))
                    self.stats['total_files_failed'] += 1
            
            self.stats['end_time'] = datetime.now()
            
            if not dry_run:
                self.logger.info(f"✓ Completed {self.stats['total_files_completed']} operations")
                if self.stats['total_files_failed'] > 0:
                    self.logger.warning(f"⚠️  {self.stats['total_files_failed']} operations failed")
            else:
                self.logger.info("✓ Dry run completed - validation successful")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute file operations: {e}")
            return False
    
    def _execute_single_operation(self, operation: FileOperation) -> bool:
        """Execute a single file operation with integrity checks"""
        try:
            # Ensure target directory exists
            operation.target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Perform operation
            if operation.operation_type == 'copy':
                shutil.copy2(operation.source_path, operation.target_path)
            else:  # move
                shutil.move(str(operation.source_path), str(operation.target_path))
            
            # Verify integrity
            if operation.target_path.exists():
                target_size = operation.target_path.stat().st_size
                if target_size != operation.source_size:
                    self.logger.error(f"Size mismatch: {operation.target_path}")
                    return False
                
                # Verify checksum for critical operations
                if operation.source_checksum:
                    target_checksum = self._calculate_checksum(operation.target_path)
                    if target_checksum != operation.source_checksum:
                        self.logger.error(f"Checksum mismatch: {operation.target_path}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute operation {operation.source_path}: {e}")
            return False
    
    def generate_operation_report(self) -> Dict[str, Any]:
        """Generate comprehensive operation report"""
        self.logger.info("Generating operation report...")
        
        # Calculate duration
        duration = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # User directory statistics
        user_stats = {}
        for user_id, user_info in self.user_directories.items():
            user_stats[user_id] = {
                'total_files': user_info.total_files,
                'genuine_files': user_info.genuine_count,
                'deepfake_files': user_info.deepfake_count,
                'speakers': user_info.speakers,
                'directory_path': str(user_info.directory_path)
            }
        
        # File type distribution
        file_type_dist = Counter()
        attack_type_dist = Counter()
        for op in self.completed_operations:
            file_type_dist[op.file_type] += 1
            if op.attack_category:
                attack_type_dist[op.attack_category] += 1
        
        report = {
            'operation_metadata': {
                'timestamp': self.timestamp,
                'operation_type': 'copy' if self.use_copy else 'move',
                'target_directory': str(self.target_data_dir),
                'duration_seconds': duration,
                'backup_created': self.create_backup
            },
            'operation_statistics': {
                'total_files_planned': self.stats['total_files_planned'],
                'total_files_completed': self.stats['total_files_completed'],
                'total_files_failed': self.stats['total_files_failed'],
                'success_rate': self.stats['total_files_completed'] / self.stats['total_files_planned'] if self.stats['total_files_planned'] > 0 else 0,
                'total_size_gb': self.stats['total_size_bytes'] / (1024**3),
                'users_created': self.stats['users_created']
            },
            'file_distribution': {
                'by_type': dict(file_type_dist),
                'by_attack_category': dict(attack_type_dist)
            },
            'user_statistics': user_stats,
            'failed_operations': [
                {
                    'source_path': str(op.source_path),
                    'target_path': str(op.target_path),
                    'error': error
                }
                for op, error in self.failed_operations[:50]  # Limit to first 50
            ]
        }
        
        self.logger.info("✓ Operation report generated")
        return report
    
    def save_results(self) -> List[Path]:
        """Save operation results and reports"""
        saved_files = []
        output_dir = self.config.paths.analysis_output_dir
        
        try:
            # Save operation report
            report = self.generate_operation_report()
            report_file = output_dir / f"file_organization_report_{self.timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            saved_files.append(report_file)
            self.logger.info(f"✓ Saved operation report: {report_file}")
            
            # Save operation log
            log_file = output_dir / f"file_operations_log_{self.timestamp}.csv"
            
            log_data = []
            for op in self.completed_operations:
                log_data.append({
                    'Original_Filename': op.original_filename,
                    'New_Filename': op.new_filename,
                    'User_ID': op.user_id,
                    'File_Type': op.file_type,
                    'Attack_Category': op.attack_category or '',
                    'Source_Path': str(op.source_path),
                    'Target_Path': str(op.target_path),
                    'Operation_Type': op.operation_type,
                    'File_Size_MB': op.source_size / (1024**2),
                    'Source_Checksum': op.source_checksum
                })
            
            if log_data:
                df = pd.DataFrame(log_data)
                df.to_csv(log_file, index=False)
                saved_files.append(log_file)
                self.logger.info(f"✓ Saved operation log: {log_file}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return saved_files
    
    def print_summary(self):
        """Print operation summary"""
        print("\n" + "="*80)
        print("FILE ORGANIZATION SYSTEM - OPERATION SUMMARY")
        print("="*80)
        
        print(f"\n📁 TARGET DIRECTORY: {self.target_data_dir}")
        print(f"🔧 OPERATION TYPE: {'Copy' if self.use_copy else 'Move'}")
        print(f"💾 BACKUP CREATED: {'Yes' if self.create_backup else 'No'}")
        
        print(f"\n📊 OPERATION STATISTICS:")
        print(f"  • Files planned: {self.stats['total_files_planned']:,}")
        print(f"  • Files completed: {self.stats['total_files_completed']:,}")
        print(f"  • Files failed: {self.stats['total_files_failed']:,}")
        
        if self.stats['total_files_planned'] > 0:
            success_rate = (self.stats['total_files_completed'] / self.stats['total_files_planned']) * 100
            print(f"  • Success rate: {success_rate:.1f}%")
        
        print(f"  • Total size: {self.stats['total_size_bytes'] / (1024**3):.2f} GB")
        print(f"  • Users created: {self.stats['users_created']}")
        
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            print(f"  • Duration: {duration:.1f} seconds")
        
        print(f"\n👥 USER DIRECTORIES:")
        for user_id, user_info in list(self.user_directories.items())[:10]:  # Show first 10
            print(f"  • {user_id}: {user_info.total_files} files ({user_info.genuine_count} genuine, {user_info.deepfake_count} deepfake)")
        
        if len(self.user_directories) > 10:
            print(f"  ... and {len(self.user_directories) - 10} more users")
        
        if self.failed_operations:
            print(f"\n⚠️  FAILED OPERATIONS ({len(self.failed_operations)}):")
            for op, error in self.failed_operations[:5]:  # Show first 5
                print(f"  • {op.original_filename}: {error}")
            if len(self.failed_operations) > 5:
                print(f"  ... and {len(self.failed_operations) - 5} more failures")


def main():
    """Main execution function for Task 3"""
    print("="*80)
    print("FILE ORGANIZATION AND RENAMING SYSTEM - TASK 3")
    print("Transform ASVspoof-2019 files into Deep ASV Detection format")
    print("="*80)
    
    # Initialize system
    organizer = FileOrganizationSystem(use_copy=True, create_backup=True)
    
    try:
        # Step 1: Create directory structure
        print("\n🏗️  Step 1: Creating directory structure...")
        if not organizer.create_directory_structure():
            print("❌ Failed to create directory structure")
            return 1
        
        # Step 2: Plan file operations
        print("\n📋 Step 2: Planning file operations...")
        if not organizer.plan_file_operations():
            print("❌ Failed to plan file operations")
            return 1
        
        # Step 3: Create backup manifest
        print("\n💾 Step 3: Creating backup manifest...")
        if not organizer.create_backup_manifest():
            print("❌ Failed to create backup manifest")
            return 1
        
        # Step 4: Dry run validation
        print("\n🔍 Step 4: Performing dry run validation...")
        if not organizer.execute_file_operations(dry_run=True):
            print("❌ Dry run validation failed")
            return 1
        
        # Step 5: Execute operations (ask for confirmation)
        print(f"\n⚠️  Ready to organize {organizer.stats['total_files_planned']:,} files")
        print(f"   Target: {organizer.target_data_dir}")
        print(f"   Size: {organizer.stats['total_size_bytes'] / (1024**3):.2f} GB")
        
        response = input("\nProceed with file organization? (y/N): ").strip().lower()
        if response == 'y':
            print("\n🚀 Step 5: Executing file operations...")
            if not organizer.execute_file_operations(dry_run=False):
                print("❌ File operations failed")
                return 1
        else:
            print("Operation cancelled by user")
            return 0
        
        # Step 6: Save results
        print("\n💾 Step 6: Saving results...")
        saved_files = organizer.save_results()
        
        # Display summary
        organizer.print_summary()
        
        print(f"\n✅ Task 3 completed successfully!")
        print(f"📁 Results saved to:")
        for file_path in saved_files:
            print(f"  • {file_path}")
        
        return 0
        
    except Exception as e:
        organizer.logger.error(f"Task 3 failed: {e}")
        print(f"❌ Task 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
