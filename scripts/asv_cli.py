#!/usr/bin/env python3
"""
ASV Dataset Preparation CLI
===========================

User-friendly command-line interface for ASV dataset preparation and conversion.

Author: ASV Dataset Preparation System
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.asv_dataset_integration import ASVDatasetIntegrator, IntegrationConfig, ConversionMode, ValidationLevel
from config.integration_configs import IntegrationPresets
from config.dataset_config import DatasetYear, Scenario


def list_presets():
    """List all available configuration presets"""
    presets = IntegrationPresets.get_all_presets()
    descriptions = IntegrationPresets.get_preset_descriptions()
    
    print("\n🔧 AVAILABLE CONFIGURATION PRESETS")
    print("=" * 50)
    
    categories = {
        "Analysis & Testing": ["quick_analysis", "development_testing"],
        "Production Configurations": ["production_conservative", "production_balanced", "production_fast"],
        "Research Configurations": ["research_comprehensive"],
        "Splitting Only": ["splitting_only_option_a", "splitting_only_option_b", "splitting_only_option_c"],
        "Custom Configurations": ["custom_balanced_small", "custom_imbalanced_large"]
    }
    
    for category, preset_names in categories.items():
        print(f"\n📂 {category}:")
        for name in preset_names:
            if name in descriptions:
                print(f"   {name:<25} - {descriptions[name]}")
    
    print(f"\nTotal: {len(presets)} presets available")
    print("\nUsage: python asv_cli.py run --preset <preset_name>")


def show_preset_details(preset_name: str):
    """Show detailed configuration for a preset"""
    presets = IntegrationPresets.get_all_presets()
    descriptions = IntegrationPresets.get_preset_descriptions()
    
    if preset_name not in presets:
        print(f"❌ Preset '{preset_name}' not found")
        print("Use 'python asv_cli.py list-presets' to see available presets")
        return
    
    config = presets[preset_name]
    description = descriptions[preset_name]
    
    print(f"\n🔧 PRESET DETAILS: {preset_name}")
    print("=" * 50)
    print(f"Description: {description}")
    print()
    
    print("Configuration:")
    print(f"  Conversion Mode:      {config.conversion_mode.value}")
    print(f"  Validation Level:     {config.validation_level.value}")
    print(f"  Splitting Option:     {config.splitting_option}")
    print()
    
    print("Dataset Settings:")
    print(f"  Max Users:            {config.max_users}")
    print(f"  Min Files/User:       {config.min_files_per_user}")
    print(f"  Max Files/User:       {config.max_files_per_user}")
    print(f"  Target Genuine Ratio: {config.target_genuine_ratio:.1%}")
    print(f"  Balance Tolerance:    {config.balance_tolerance:.1%}")
    print()
    
    if config.splitting_option == "B":
        print("Split Ratios:")
        print(f"  Train:                {config.train_ratio:.1%}")
        print(f"  Dev:                  {config.dev_ratio:.1%}")
        print(f"  Test:                 {config.test_ratio:.1%}")
        print()
    
    print("Processing Options:")
    print(f"  Enable Backup:        {config.enable_backup}")
    print(f"  Enable Rollback:      {config.enable_rollback}")
    print(f"  Create Symlinks:      {config.create_symlinks}")
    print(f"  Generate Reports:     {config.generate_reports}")
    print(f"  Dry Run:              {config.dry_run}")


def validate_environment():
    """Validate that the environment is properly set up"""
    print("🔍 VALIDATING ENVIRONMENT")
    print("=" * 30)
    
    # Check if we're in the right directory
    expected_files = [
        "config/dataset_config.py",
        "scripts/asv_dataset_integration.py",
        "utils/logging_utils.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Environment validation failed!")
        print("Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease run this script from the asv_dataset_preparation directory")
        return False
    
    print("✅ Environment validation passed")
    return True


def run_integration(args):
    """Run the integration process"""
    print("🚀 STARTING ASV DATASET INTEGRATION")
    print("=" * 40)
    
    # Validate environment first
    if not validate_environment():
        return 1
    
    try:
        # Get configuration
        if args.preset:
            presets = IntegrationPresets.get_all_presets()
            if args.preset not in presets:
                print(f"❌ Unknown preset: {args.preset}")
                print("Use 'python asv_cli.py list-presets' to see available presets")
                return 1
            
            config = presets[args.preset]
            print(f"📋 Using preset: {args.preset}")
            
            # Override with command line arguments
            if args.mode:
                config.conversion_mode = ConversionMode(args.mode)
            if args.validation:
                config.validation_level = ValidationLevel(args.validation)
            if args.split_option:
                config.splitting_option = args.split_option
            if args.dataset_year:
                config.dataset_year = DatasetYear(args.dataset_year)
            if args.scenario:
                config.scenario = Scenario(args.scenario)
            if args.dry_run:
                config.dry_run = True
            if args.no_backup:
                config.enable_backup = False
            if args.no_rollback:
                config.enable_rollback = False
                
        else:
            # Create config from command line arguments
            config = IntegrationConfig(
                conversion_mode=ConversionMode(args.mode or 'full_conversion'),
                validation_level=ValidationLevel(args.validation or 'standard'),
                splitting_option=args.split_option or 'A',
                dataset_year=DatasetYear(args.dataset_year or '2019'),
                scenario=Scenario(args.scenario or 'LA'),
                dry_run=args.dry_run,
                enable_backup=not args.no_backup,
                enable_rollback=not args.no_rollback
            )
        
        # Show configuration summary
        print("\n📋 CONFIGURATION SUMMARY")
        print("-" * 25)
        print(f"Mode:        {config.conversion_mode.value}")
        print(f"Validation:  {config.validation_level.value}")
        print(f"Split:       Option {config.splitting_option}")
        print(f"Dataset Year: {config.dataset_year.value}")
        print(f"Scenario:    {config.scenario.value}")
        print(f"Backup:      {'Enabled' if config.enable_backup else 'Disabled'}")
        print(f"Rollback:    {'Enabled' if config.enable_rollback else 'Disabled'}")
        print(f"Dry Run:     {'Yes' if config.dry_run else 'No'}")
        
        if config.dry_run:
            print("\n⚠️  DRY RUN MODE - No actual changes will be made")
        
        # Confirm before proceeding
        if not args.yes:
            response = input("\nProceed with integration? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Integration cancelled")
                return 0
        
        # Run integration
        integrator = ASVDatasetIntegrator(config)
        success = integrator.run_integration()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Integration interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return 1


def create_preset_configs():
    """Create preset configuration files"""
    print("📁 CREATING PRESET CONFIGURATION FILES")
    print("=" * 40)
    
    try:
        from config.integration_configs import save_preset_configs
        save_preset_configs()
        print("✅ Preset configuration files created successfully")
        return 0
    except Exception as e:
        print(f"❌ Failed to create preset configs: {e}")
        return 1


def reset_project(force: bool = False):
    """Reset the entire project to a clean state"""
    print("🔄 PROJECT RESET")
    print("=" * 30)
    
    try:
        # Import here to avoid circular imports
        from scripts.asv_dataset_integration import ASVDatasetIntegrator, IntegrationConfig, ConversionMode
        
        # Create a minimal config for reset operation
        config = IntegrationConfig(
            conversion_mode=ConversionMode.ANALYSIS_ONLY,
            enable_backup=False,
            enable_rollback=False
        )
        
        integrator = ASVDatasetIntegrator(config)
        
        # Get current project status
        status = integrator.get_project_status()
        
        print(f"📊 CURRENT PROJECT STATUS:")
        print(f"  Total files: {status['total_files']:,}")
        print(f"  Total size: {status['total_size_gb']:.2f} GB")
        if status['last_run_timestamp']:
            print(f"  Last run: {status['last_run_timestamp']}")
        print()
        
        # Show what exists
        for name, info in status['directories'].items():
            if info['exists']:
                print(f"  📁 {name}: {info['file_count']:,} files ({info['size_gb']:.2f} GB)")
        
        if status['total_files'] == 0:
            print("✅ Project is already clean - nothing to reset")
            return 0
        
        print(f"\n⚠️  WARNING: This will permanently delete all project outputs!")
        print(f"⚠️  This includes logs, reports, analysis results, and backups!")
        print(f"⚠️  Total data to be removed: {status['total_size_gb']:.2f} GB")
        
        if not force:
            response = input("\nAre you sure you want to reset the project? (type 'yes' to confirm): ")
            if response.lower() != 'yes':
                print("Reset cancelled")
                return 0
        
        # Perform reset
        print("\n🔄 Performing project reset...")
        success = integrator.reset_project(confirm=True)
        
        if success:
            print("✅ Project reset completed successfully!")
            print("🎯 Ready for a fresh start!")
            return 0
        else:
            print("❌ Project reset failed - check logs for details")
            return 1
            
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return 1


def run_split_recommender():
    """Run the split option recommender"""
    try:
        from scripts.optimal_split_recommender import OptimalSplitRecommender
        
        recommender = OptimalSplitRecommender()
        recommender.run_recommender()
        return 0
        
    except Exception as e:
        print(f"❌ Recommender failed: {e}")
        return 1


def run_dataset_conversion(args):
    """Run dataset conversion to Deep ASV Detection format"""
    try:
        from scripts.asv_to_deep_conversion import ASVToDeepConversionSystem
        from config.dataset_config import ASVDatasetConfig, DatasetYear, Scenario
        
        print("🚀 ASV TO DEEP ASV DETECTION CONVERSION")
        print("="*50)
        
        # Create config instance with dataset year and scenario
        dataset_config = ASVDatasetConfig(
            dataset_year=DatasetYear(args.dataset_year) if args.dataset_year else DatasetYear.ASV2019,
            scenario=Scenario(args.scenario) if args.scenario else Scenario.LA
        )
        
        converter = ASVToDeepConversionSystem(config_instance=dataset_config)
        
        # Handle test mode
        if args.test_mode:
            print("🧪 Running in test mode...")
            from scripts.test_conversion_system import ConversionSystemTester
            tester = ConversionSystemTester()
            success = tester.run_all_tests()
            return 0 if success else 1
        
        # Handle forced split option
        if args.split_option:
            # Mock user input for automated execution
            import builtins
            original_input = builtins.input
            
            def mock_input(prompt):
                print(f"{prompt}{args.split_option}")
                return args.split_option
            
            builtins.input = mock_input
            
            try:
                # Before conversion, set dataset-year via integration analyzer to generate mappings
                from scripts.asv_dataset_integration import ASVDatasetIntegrator, IntegrationConfig, ConversionMode
                ic = IntegrationConfig(conversion_mode=ConversionMode.ANALYSIS_ONLY, dataset_year=DatasetYear(args.dataset_year), scenario=Scenario(args.scenario), enable_backup=False, enable_rollback=False)
                integrator = ASVDatasetIntegrator(ic)
                integrator.run_integration()
                success = converter.run_complete_conversion()
                return 0 if success else 1
            finally:
                builtins.input = original_input
        else:
            # Interactive mode
            from scripts.asv_dataset_integration import ASVDatasetIntegrator, IntegrationConfig, ConversionMode
            ic = IntegrationConfig(conversion_mode=ConversionMode.ANALYSIS_ONLY, dataset_year=DatasetYear(args.dataset_year), scenario=Scenario(args.scenario), enable_backup=False, enable_rollback=False)
            integrator = ASVDatasetIntegrator(ic)
            integrator.run_integration()
            success = converter.run_complete_conversion()
            return 0 if success else 1
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1


def show_project_status():
    """Show current project status"""
    print("📊 PROJECT STATUS")
    print("=" * 25)
    
    try:
        from scripts.asv_dataset_integration import ASVDatasetIntegrator, IntegrationConfig, ConversionMode
        
        config = IntegrationConfig(
            conversion_mode=ConversionMode.ANALYSIS_ONLY,
            enable_backup=False
        )
        
        integrator = ASVDatasetIntegrator(config)
        status = integrator.get_project_status()
        
        print(f"📁 Total files: {status['total_files']:,}")
        print(f"💾 Total size: {status['total_size_gb']:.2f} GB")
        
        if status['last_run_timestamp']:
            print(f"🕒 Last run: {status['last_run_timestamp']}")
        else:
            print("🕒 Last run: Never")
        
        print(f"\n📂 Directory Status:")
        for name, info in status['directories'].items():
            if info['exists']:
                print(f"  ✅ {name:<15}: {info['file_count']:>6,} files ({info['size_gb']:>6.2f} GB)")
            else:
                print(f"  ❌ {name:<15}: Not found")
        
        if status['total_files'] == 0:
            print(f"\n🎯 Project is clean - ready for fresh start!")
        else:
            print(f"\n💡 Use 'python asv_cli.py reset' to clean the project")
        
        return 0
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return 1


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="ASV Dataset Preparation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available presets
  python asv_cli.py list-presets
  
  # Show preset details
  python asv_cli.py show-preset production_balanced
  
  # Run with preset
  python asv_cli.py run --preset quick_analysis
  
  # Run with custom settings
  python asv_cli.py run --mode full_conversion --split-option B --validation comprehensive
  
  # Dry run with preset
  python asv_cli.py run --preset production_conservative --dry-run
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List presets command
    list_parser = subparsers.add_parser('list-presets', help='List all available configuration presets')
    
    # Show preset command
    show_parser = subparsers.add_parser('show-preset', help='Show details of a specific preset')
    show_parser.add_argument('preset_name', help='Name of the preset to show')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the integration process')
    
    # Configuration options
    config_group = run_parser.add_argument_group('Configuration')
    config_group.add_argument('--preset', help='Use a predefined configuration preset')
    config_group.add_argument('--mode', choices=['analysis_only', 'split_only', 'full_conversion', 'test_subset'],
                             help='Conversion mode (overrides preset)')
    config_group.add_argument('--validation', choices=['basic', 'standard', 'comprehensive'],
                             help='Validation level (overrides preset)')
    config_group.add_argument('--split-option', choices=['A', 'B', 'C'],
                             help='Dataset splitting option (overrides preset)')
    config_group.add_argument('--dataset-year', choices=['2019', '2021'], help='Dataset year to process')
    config_group.add_argument('--scenario', choices=['LA', 'PA'], help='Scenario (LA or PA)')
    
    # Processing options
    process_group = run_parser.add_argument_group('Processing')
    process_group.add_argument('--dry-run', action='store_true', help='Perform dry run without actual changes')
    process_group.add_argument('--no-backup', action='store_true', help='Disable backup creation')
    process_group.add_argument('--no-rollback', action='store_true', help='Disable rollback capability')
    process_group.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompts')
    
    # Create configs command
    create_parser = subparsers.add_parser('create-configs', help='Create preset configuration files')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset project to clean state')
    reset_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show current project status')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate environment setup')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert dataset to Deep ASV Detection format')
    convert_parser.add_argument('--split-option', choices=['A', 'B', 'C'], help='Force specific split option')
    convert_parser.add_argument('--test-mode', action='store_true', help='Run in test mode with automated input')
    convert_parser.add_argument('--dataset-year', choices=['2019', '2021'], default='2019', help='Dataset year to process')
    convert_parser.add_argument('--scenario', choices=['LA', 'PA'], default='LA', help='Scenario (LA or PA)')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Get split option recommendation based on use case')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute commands
    if args.command == 'list-presets':
        list_presets()
        return 0
        
    elif args.command == 'show-preset':
        show_preset_details(args.preset_name)
        return 0
        
    elif args.command == 'run':
        return run_integration(args)
        
    elif args.command == 'create-configs':
        return create_preset_configs()
        
    elif args.command == 'reset':
        return reset_project(args.force)
        
    elif args.command == 'status':
        return show_project_status()
        
    elif args.command == 'validate':
        success = validate_environment()
        return 0 if success else 1
        
    elif args.command == 'convert':
        return run_dataset_conversion(args)
        
    elif args.command == 'recommend':
        return run_split_recommender()
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    exit(main()) 