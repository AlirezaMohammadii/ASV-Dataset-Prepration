"""
Integration Configuration Presets
=================================

This module provides predefined configuration presets for different
ASV dataset integration scenarios.

Author: ASV Dataset Preparation System
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, Any
from scripts.asv_dataset_integration import IntegrationConfig, ConversionMode, ValidationLevel


class IntegrationPresets:
    """Predefined integration configuration presets"""
    
    @staticmethod
    def quick_analysis() -> IntegrationConfig:
        """Quick analysis configuration for initial dataset exploration"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.ANALYSIS_ONLY,
            validation_level=ValidationLevel.BASIC,
            enable_backup=False,
            enable_rollback=False,
            generate_reports=True
        )
    
    @staticmethod
    def development_testing() -> IntegrationConfig:
        """Development testing configuration with subset processing"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.TEST_SUBSET,
            validation_level=ValidationLevel.STANDARD,
            max_users=10,
            test_subset_size=500,
            enable_backup=True,
            enable_rollback=True,
            dry_run=False,
            generate_reports=True
        )
    
    @staticmethod
    def production_conservative() -> IntegrationConfig:
        """Conservative production configuration with comprehensive validation"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.FULL_CONVERSION,
            validation_level=ValidationLevel.COMPREHENSIVE,
            max_users=50,
            min_files_per_user=15,
            max_files_per_user=150,
            target_genuine_ratio=0.6,
            balance_tolerance=0.05,
            splitting_option="A",
            enable_backup=True,
            enable_rollback=True,
            create_symlinks=False,
            preserve_original_structure=True,
            generate_reports=True
        )
    
    @staticmethod
    def production_balanced() -> IntegrationConfig:
        """Balanced production configuration with redistribution"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.FULL_CONVERSION,
            validation_level=ValidationLevel.STANDARD,
            max_users=75,
            min_files_per_user=10,
            max_files_per_user=200,
            target_genuine_ratio=0.55,
            balance_tolerance=0.1,
            splitting_option="B",
            train_ratio=0.7,
            dev_ratio=0.15,
            test_ratio=0.15,
            enable_backup=True,
            enable_rollback=True,
            create_symlinks=False,
            generate_reports=True
        )
    
    @staticmethod
    def production_fast() -> IntegrationConfig:
        """Fast production configuration with minimal validation"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.FULL_CONVERSION,
            validation_level=ValidationLevel.BASIC,
            max_users=100,
            min_files_per_user=5,
            max_files_per_user=300,
            target_genuine_ratio=0.5,
            balance_tolerance=0.15,
            splitting_option="C",
            enable_backup=False,
            enable_rollback=False,
            create_symlinks=True,
            generate_reports=False
        )
    
    @staticmethod
    def research_comprehensive() -> IntegrationConfig:
        """Comprehensive research configuration with detailed analysis"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.FULL_CONVERSION,
            validation_level=ValidationLevel.COMPREHENSIVE,
            max_users=200,
            min_files_per_user=3,
            max_files_per_user=500,
            target_genuine_ratio=0.6,
            balance_tolerance=0.2,
            splitting_option="B",
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            enable_backup=True,
            enable_rollback=True,
            create_symlinks=False,
            preserve_original_structure=True,
            generate_reports=True
        )
    
    @staticmethod
    def splitting_only_option_a() -> IntegrationConfig:
        """Configuration for splitting only with Option A"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.SPLIT_ONLY,
            validation_level=ValidationLevel.STANDARD,
            splitting_option="A",
            enable_backup=True,
            enable_rollback=True,
            generate_reports=True
        )
    
    @staticmethod
    def splitting_only_option_b() -> IntegrationConfig:
        """Configuration for splitting only with Option B"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.SPLIT_ONLY,
            validation_level=ValidationLevel.STANDARD,
            splitting_option="B",
            train_ratio=0.7,
            dev_ratio=0.15,
            test_ratio=0.15,
            enable_backup=True,
            enable_rollback=True,
            generate_reports=True
        )
    
    @staticmethod
    def splitting_only_option_c() -> IntegrationConfig:
        """Configuration for splitting only with Option C"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.SPLIT_ONLY,
            validation_level=ValidationLevel.STANDARD,
            splitting_option="C",
            enable_backup=True,
            enable_rollback=True,
            generate_reports=True
        )
    
    @staticmethod
    def custom_balanced_small() -> IntegrationConfig:
        """Custom configuration for small balanced dataset"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.FULL_CONVERSION,
            validation_level=ValidationLevel.STANDARD,
            max_users=25,
            min_files_per_user=20,
            max_files_per_user=100,
            target_genuine_ratio=0.65,
            balance_tolerance=0.05,
            splitting_option="B",
            train_ratio=0.6,
            dev_ratio=0.2,
            test_ratio=0.2,
            enable_backup=True,
            enable_rollback=True,
            generate_reports=True
        )
    
    @staticmethod
    def custom_imbalanced_large() -> IntegrationConfig:
        """Custom configuration for large imbalanced dataset"""
        return IntegrationConfig(
            conversion_mode=ConversionMode.FULL_CONVERSION,
            validation_level=ValidationLevel.BASIC,
            max_users=150,
            min_files_per_user=5,
            max_files_per_user=400,
            target_genuine_ratio=0.3,
            balance_tolerance=0.3,
            splitting_option="A",
            enable_backup=True,
            enable_rollback=True,
            create_symlinks=True,
            generate_reports=True
        )
    
    @classmethod
    def get_all_presets(cls) -> Dict[str, IntegrationConfig]:
        """Get all available presets"""
        return {
            'quick_analysis': cls.quick_analysis(),
            'development_testing': cls.development_testing(),
            'production_conservative': cls.production_conservative(),
            'production_balanced': cls.production_balanced(),
            'production_fast': cls.production_fast(),
            'research_comprehensive': cls.research_comprehensive(),
            'splitting_only_option_a': cls.splitting_only_option_a(),
            'splitting_only_option_b': cls.splitting_only_option_b(),
            'splitting_only_option_c': cls.splitting_only_option_c(),
            'custom_balanced_small': cls.custom_balanced_small(),
            'custom_imbalanced_large': cls.custom_imbalanced_large()
        }
    
    @classmethod
    def get_preset_descriptions(cls) -> Dict[str, str]:
        """Get descriptions of all presets"""
        return {
            'quick_analysis': 'Quick dataset analysis without conversion',
            'development_testing': 'Development testing with small subset',
            'production_conservative': 'Conservative production with comprehensive validation',
            'production_balanced': 'Balanced production with data redistribution',
            'production_fast': 'Fast production with minimal validation',
            'research_comprehensive': 'Comprehensive research configuration',
            'splitting_only_option_a': 'Splitting only using original splits',
            'splitting_only_option_b': 'Splitting only with combined redistribution',
            'splitting_only_option_c': 'Splitting only with train+dev vs eval',
            'custom_balanced_small': 'Custom small balanced dataset',
            'custom_imbalanced_large': 'Custom large imbalanced dataset'
        }


def save_preset_configs():
    """Save all preset configurations to JSON files"""
    import json
    from pathlib import Path
    
    config_dir = Path(__file__).parent / "presets"
    config_dir.mkdir(exist_ok=True)
    
    presets = IntegrationPresets.get_all_presets()
    descriptions = IntegrationPresets.get_preset_descriptions()
    
    for name, config in presets.items():
        config_file = config_dir / f"{name}.json"
        
        config_dict = {
            'name': name,
            'description': descriptions[name],
            'configuration': {
                'conversion_mode': config.conversion_mode.value,
                'validation_level': config.validation_level.value,
                'max_users': config.max_users,
                'min_files_per_user': config.min_files_per_user,
                'max_files_per_user': config.max_files_per_user,
                'target_genuine_ratio': config.target_genuine_ratio,
                'balance_tolerance': config.balance_tolerance,
                'splitting_option': config.splitting_option,
                'train_ratio': config.train_ratio,
                'dev_ratio': config.dev_ratio,
                'test_ratio': config.test_ratio,
                'test_subset_size': config.test_subset_size,
                'enable_backup': config.enable_backup,
                'enable_rollback': config.enable_rollback,
                'dry_run': config.dry_run,
                'create_symlinks': config.create_symlinks,
                'preserve_original_structure': config.preserve_original_structure,
                'generate_reports': config.generate_reports
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    print(f"Saved {len(presets)} preset configurations to {config_dir}")


if __name__ == "__main__":
    save_preset_configs() 