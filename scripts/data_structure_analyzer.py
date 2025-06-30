#!/usr/bin/env python3
"""
ASVspoof-2019 Data Structure Analyzer
====================================

This script performs comprehensive analysis of the ASVspoof-2019 dataset structure,
implementing Task 1 of the dataset preparation process.

Features:
- Complete dataset inventory and file counting
- Protocol file analysis and label distribution
- Speaker distribution analysis
- Attack type mapping and statistics
- Professional reporting with structured output

Author: ASV Dataset Preparation System
Version: 1.0.0
"""

import sys
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.dataset_config import config, DatasetSplit
from utils.protocol_parser import parser, FileInfo
from utils.logging_utils import setup_logger


class DataStructureAnalyzer:
    """Comprehensive analyzer for ASVspoof-2019 dataset structure"""
    
    def __init__(self):
        self.logger = setup_logger('DataStructureAnalyzer', config.paths.logs_dir)
        self.analysis_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def validate_dataset_paths(self) -> bool:
        """Validate that all required dataset paths exist"""
        self.logger.info("Validating dataset paths...")
        
        errors = config.validate_paths()
        if errors:
            self.logger.error("Dataset validation failed:")
            for error in errors:
                self.logger.error(f"  - {error}")
            return False
        
        self.logger.info("✓ All dataset paths validated successfully")
        return True
    
    def analyze_file_counts(self) -> Dict[str, Any]:
        """Analyze and count files in all dataset splits"""
        self.logger.info("Analyzing file counts across all splits...")
        
        file_counts = {}
        data_dirs = {
            'train': config.paths.la_train_dir,
            'dev': config.paths.la_dev_dir,
            'eval': config.paths.la_eval_dir
        }
        
        total_files = 0
        total_size = 0
        
        for split_name, data_dir in data_dirs.items():
            if not data_dir.exists():
                self.logger.warning(f"Directory not found: {data_dir}")
                file_counts[split_name] = {
                    'count': 0,
                    'size_mb': 0,
                    'files': []
                }
                continue
            
            # Count FLAC files
            flac_files = list(data_dir.glob("*.flac"))
            split_size = sum(f.stat().st_size for f in flac_files)
            
            file_counts[split_name] = {
                'count': len(flac_files),
                'size_bytes': split_size,
                'size_mb': split_size / (1024 * 1024),
                'size_gb': split_size / (1024 * 1024 * 1024),
                'files': [f.name for f in flac_files[:10]]  # Sample of first 10 files
            }
            
            total_files += len(flac_files)
            total_size += split_size
            
            self.logger.info(f"  {split_name}: {len(flac_files)} files ({split_size / (1024*1024):.1f} MB)")
        
        # Add totals
        file_counts['totals'] = {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024)
        }
        
        self.logger.info(f"✓ Total files analyzed: {total_files} ({total_size / (1024*1024*1024):.2f} GB)")
        return file_counts


def main():
    """Main execution function"""
    print("=" * 80)
    print("ASVspoof-2019 Data Structure Analyzer")
    print("Task 1: Data Structure Analysis and Preparation")
    print("=" * 80)
    
    analyzer = DataStructureAnalyzer()
    
    # Validate dataset paths
    if not analyzer.validate_dataset_paths():
        print("❌ Dataset validation failed. Please check paths and try again.")
        return 1
    
    try:
        # Simple test first
        print("\n🔍 Testing file count analysis...")
        file_counts = analyzer.analyze_file_counts()
        
        print("\n📊 FILE COUNT RESULTS")
        print("-" * 40)
        for split, data in file_counts.items():
            if split != 'totals':
                print(f"{split.upper()}: {data['count']} files ({data['size_mb']:.1f} MB)")
        
        totals = file_counts['totals']
        print(f"\nTOTAL: {totals['total_files']} files ({totals['total_size_gb']:.2f} GB)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
