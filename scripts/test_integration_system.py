#!/usr/bin/env python3
"""
Integration System Test Suite
=============================

Comprehensive test suite for the ASV dataset integration system.
Tests all components, error handling, and recovery mechanisms.

Author: ASV Dataset Preparation System
Version: 1.0.0
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.dataset_config import ASVDatasetConfig
from scripts.asv_dataset_integration import ASVDatasetIntegrator, IntegrationConfig, ConversionMode, ValidationLevel
from config.integration_configs import IntegrationPresets
from utils.logging_utils import setup_logger


class IntegrationTestSuite:
    """Comprehensive test suite for the integration system"""
    
    def __init__(self):
        self.config = ASVDatasetConfig()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger(f'integration_test_{self.timestamp}', self.config.paths.logs_dir)
        
        # Test directories
        self.test_dir = self.config.paths.output_root / f"integration_test_{self.timestamp}"
        self.backup_dir = self.test_dir / "backup"
        self.results_dir = self.test_dir / "results"
        
        # Create test directories
        for dir_path in [self.test_dir, self.backup_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Test results
        self.test_results = {
            'start_time': datetime.now(),
            'tests': [],
            'passed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        self.logger.info(f"Integration test suite initialized at {self.test_dir}")
    
    def create_test_backup(self) -> bool:
        """Create backup of current state for testing"""
        try:
            self.logger.info("Creating test backup...")
            
            # Backup output directory
            if self.config.paths.output_root.exists():
                backup_output = self.backup_dir / "output"
                shutil.copytree(
                    self.config.paths.output_root, 
                    backup_output, 
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(f"integration_test_{self.timestamp}")
                )
            
            self.logger.info("✓ Test backup created")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create test backup: {e}")
            return False
    
    def restore_from_backup(self) -> bool:
        """Restore from test backup"""
        try:
            self.logger.info("Restoring from test backup...")
            
            # Remove current output (except test directory)
            for item in self.config.paths.output_root.glob("*"):
                if item.name.startswith(f"integration_test_{self.timestamp}"):
                    continue
                
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            
            # Restore from backup
            backup_output = self.backup_dir / "output"
            if backup_output.exists():
                for item in backup_output.glob("*"):
                    target = self.config.paths.output_root / item.name
                    if item.is_file():
                        shutil.copy2(item, target)
                    elif item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
            
            self.logger.info("✓ Restored from test backup")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def run_test(self, test_name: str, test_func) -> Dict[str, Any]:
        """Run a single test and record results"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"RUNNING TEST: {test_name}")
        self.logger.info(f"{'='*60}")
        
        test_result = {
            'name': test_name,
            'start_time': datetime.now(),
            'status': 'running',
            'message': '',
            'details': {},
            'duration': 0
        }
        
        try:
            # Run the test
            result = test_func()
            
            if result is True:
                test_result['status'] = 'passed'
                test_result['message'] = 'Test passed successfully'
                self.test_results['passed'] += 1
                self.logger.info(f"✅ {test_name} PASSED")
                
            elif result is False:
                test_result['status'] = 'failed'
                test_result['message'] = 'Test failed'
                self.test_results['failed'] += 1
                self.logger.error(f"❌ {test_name} FAILED")
                
            elif isinstance(result, dict):
                test_result['status'] = result.get('status', 'unknown')
                test_result['message'] = result.get('message', '')
                test_result['details'] = result.get('details', {})
                
                if test_result['status'] == 'passed':
                    self.test_results['passed'] += 1
                    self.logger.info(f"✅ {test_name} PASSED")
                elif test_result['status'] == 'failed':
                    self.test_results['failed'] += 1
                    self.logger.error(f"❌ {test_name} FAILED")
                elif test_result['status'] == 'skipped':
                    self.test_results['skipped'] += 1
                    self.logger.info(f"⏭️  {test_name} SKIPPED")
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['message'] = f'Test threw exception: {str(e)}'
            self.test_results['failed'] += 1
            self.logger.error(f"❌ {test_name} FAILED with exception: {e}")
        
        finally:
            test_result['end_time'] = datetime.now()
            test_result['duration'] = (test_result['end_time'] - test_result['start_time']).total_seconds()
            self.test_results['tests'].append(test_result)
        
        return test_result
    
    def test_environment_validation(self) -> bool:
        """Test environment validation"""
        try:
            config = IntegrationConfig(conversion_mode=ConversionMode.ANALYSIS_ONLY)
            integrator = ASVDatasetIntegrator(config)
            
            # Test prerequisite validation
            validation_result = integrator.validate_prerequisites()
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Environment validation test failed: {e}")
            return False
    
    def test_preset_configurations(self) -> Dict[str, Any]:
        """Test all preset configurations"""
        try:
            presets = IntegrationPresets.get_all_presets()
            
            results = {
                'status': 'passed',
                'message': f'All {len(presets)} presets validated',
                'details': {
                    'total_presets': len(presets),
                    'valid_presets': 0,
                    'invalid_presets': 0,
                    'preset_details': {}
                }
            }
            
            for name, config in presets.items():
                try:
                    # Basic validation of preset configuration
                    if not isinstance(config, IntegrationConfig):
                        raise ValueError(f"Invalid config type for {name}")
                    
                    # Validate enum values
                    if not isinstance(config.conversion_mode, ConversionMode):
                        raise ValueError(f"Invalid conversion mode for {name}")
                    
                    if not isinstance(config.validation_level, ValidationLevel):
                        raise ValueError(f"Invalid validation level for {name}")
                    
                    # Validate ratios for Option B
                    if config.splitting_option == "B":
                        total_ratio = config.train_ratio + config.dev_ratio + config.test_ratio
                        if abs(total_ratio - 1.0) > 0.01:
                            raise ValueError(f"Invalid ratios for {name}: sum = {total_ratio}")
                    
                    results['details']['valid_presets'] += 1
                    results['details']['preset_details'][name] = 'valid'
                    
                except Exception as e:
                    results['details']['invalid_presets'] += 1
                    results['details']['preset_details'][name] = f'invalid: {e}'
                    self.logger.error(f"Invalid preset {name}: {e}")
            
            if results['details']['invalid_presets'] > 0:
                results['status'] = 'failed'
                results['message'] = f"{results['details']['invalid_presets']} invalid presets found"
            
            return results
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Preset configuration test failed: {e}',
                'details': {}
            }
    
    def test_analysis_only_mode(self) -> bool:
        """Test analysis-only mode"""
        try:
            config = IntegrationPresets.quick_analysis()
            integrator = ASVDatasetIntegrator(config)
            
            # Run analysis only
            success = integrator.run_integration()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Analysis-only mode test failed: {e}")
            return False
    
    def test_dry_run_mode(self) -> bool:
        """Test dry run mode"""
        try:
            config = IntegrationPresets.development_testing()
            config.dry_run = True
            
            integrator = ASVDatasetIntegrator(config)
            
            # Run in dry run mode
            success = integrator.run_integration()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Dry run mode test failed: {e}")
            return False
    
    def test_backup_restore_functionality(self) -> Dict[str, Any]:
        """Test backup and restore functionality"""
        try:
            config = IntegrationConfig(
                conversion_mode=ConversionMode.ANALYSIS_ONLY,
                enable_backup=True,
                enable_rollback=True
            )
            
            integrator = ASVDatasetIntegrator(config)
            
            # Test backup creation
            backup_success = integrator.create_system_backup()
            
            if not backup_success:
                return {
                    'status': 'failed',
                    'message': 'Backup creation failed',
                    'details': {}
                }
            
            # Test rollback
            rollback_success = integrator.rollback_changes()
            
            return {
                'status': 'passed' if rollback_success else 'failed',
                'message': 'Backup/restore functionality tested',
                'details': {
                    'backup_success': backup_success,
                    'rollback_success': rollback_success
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Backup/restore test failed: {e}',
                'details': {}
            }
    
    def test_validation_levels(self) -> Dict[str, Any]:
        """Test different validation levels"""
        try:
            results = {
                'status': 'passed',
                'message': 'All validation levels tested',
                'details': {}
            }
            
            for level in ValidationLevel:
                try:
                    config = IntegrationConfig(
                        conversion_mode=ConversionMode.ANALYSIS_ONLY,
                        validation_level=level
                    )
                    
                    integrator = ASVDatasetIntegrator(config)
                    
                    # Test validation
                    validation_success = integrator.run_validation_checks()
                    
                    results['details'][level.value] = {
                        'success': validation_success,
                        'message': 'Validation completed' if validation_success else 'Validation failed'
                    }
                    
                except Exception as e:
                    results['details'][level.value] = {
                        'success': False,
                        'message': f'Exception: {e}'
                    }
                    results['status'] = 'failed'
            
            return results
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Validation levels test failed: {e}',
                'details': {}
            }
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery"""
        try:
            # Test with invalid configuration
            config = IntegrationConfig(
                conversion_mode=ConversionMode.FULL_CONVERSION,
                max_users=0,  # Invalid value
                enable_rollback=True
            )
            
            integrator = ASVDatasetIntegrator(config)
            
            # This should fail gracefully
            success = integrator.run_integration()
            
            # Check if errors were recorded
            errors_recorded = len(integrator.state.errors) > 0
            
            return {
                'status': 'passed' if not success and errors_recorded else 'failed',
                'message': 'Error handling tested',
                'details': {
                    'integration_failed_as_expected': not success,
                    'errors_recorded': errors_recorded,
                    'error_count': len(integrator.state.errors)
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Error handling test failed: {e}',
                'details': {}
            }
    
    def test_splitting_options(self) -> Dict[str, Any]:
        """Test all splitting options"""
        try:
            results = {
                'status': 'passed',
                'message': 'All splitting options tested',
                'details': {}
            }
            
            for option in ['A', 'B', 'C']:
                try:
                    config = IntegrationConfig(
                        conversion_mode=ConversionMode.SPLIT_ONLY,
                        splitting_option=option,
                        enable_backup=True
                    )
                    
                    integrator = ASVDatasetIntegrator(config)
                    
                    # Test splitting
                    success = integrator.run_integration()
                    
                    results['details'][f'option_{option}'] = {
                        'success': success,
                        'message': 'Splitting completed' if success else 'Splitting failed'
                    }
                    
                    if not success:
                        results['status'] = 'failed'
                    
                    # Restore backup for next test
                    self.restore_from_backup()
                    
                except Exception as e:
                    results['details'][f'option_{option}'] = {
                        'success': False,
                        'message': f'Exception: {e}'
                    }
                    results['status'] = 'failed'
            
            return results
            
        except Exception as e:
            return {
                'status': 'failed',
                'message': f'Splitting options test failed: {e}',
                'details': {}
            }
    
    def test_report_generation(self) -> bool:
        """Test report generation"""
        try:
            config = IntegrationConfig(
                conversion_mode=ConversionMode.ANALYSIS_ONLY,
                generate_reports=True
            )
            
            integrator = ASVDatasetIntegrator(config)
            
            # Run integration
            success = integrator.run_integration()
            
            # Check if report was generated
            report_file = integrator.generate_integration_report()
            
            return success and report_file and report_file.exists()
            
        except Exception as e:
            self.logger.error(f"Report generation test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        self.logger.info("🧪 STARTING INTEGRATION SYSTEM TEST SUITE")
        self.logger.info("=" * 60)
        
        # Create test backup
        if not self.create_test_backup():
            self.logger.error("Failed to create test backup - aborting tests")
            return self.test_results
        
        # Define test suite
        tests = [
            ("Environment Validation", self.test_environment_validation),
            ("Preset Configurations", self.test_preset_configurations),
            ("Analysis Only Mode", self.test_analysis_only_mode),
            ("Dry Run Mode", self.test_dry_run_mode),
            ("Backup/Restore Functionality", self.test_backup_restore_functionality),
            ("Validation Levels", self.test_validation_levels),
            ("Error Handling", self.test_error_handling),
            ("Splitting Options", self.test_splitting_options),
            ("Report Generation", self.test_report_generation)
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            
            # Restore backup after each test (except first few)
            if test_name not in ["Environment Validation", "Preset Configurations"]:
                self.restore_from_backup()
        
        # Calculate final results
        self.test_results['end_time'] = datetime.now()
        self.test_results['total_duration'] = (
            self.test_results['end_time'] - self.test_results['start_time']
        ).total_seconds()
        
        # Save test results
        self.save_test_results()
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def save_test_results(self):
        """Save test results to file"""
        try:
            results_file = self.results_dir / f"integration_test_results_{self.timestamp}.json"
            
            # Convert datetime objects to strings for JSON serialization
            serializable_results = json.loads(json.dumps(self.test_results, default=str))
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Test results saved to: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save test results: {e}")
    
    def print_test_summary(self):
        """Print test summary"""
        total_tests = len(self.test_results['tests'])
        passed = self.test_results['passed']
        failed = self.test_results['failed']
        skipped = self.test_results['skipped']
        duration = self.test_results['total_duration']
        
        print(f"\n{'='*60}")
        print("🧪 INTEGRATION TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests:     {total_tests}")
        print(f"Passed:          {passed} ✅")
        print(f"Failed:          {failed} ❌")
        print(f"Skipped:         {skipped} ⏭️")
        print(f"Success Rate:    {(passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        print(f"Total Duration:  {duration:.1f} seconds")
        print(f"Results saved:   {self.results_dir}")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {failed} TESTS FAILED - Check logs for details")
        
        print(f"{'='*60}")


def main():
    """Main function to run the test suite"""
    print("🧪 ASV Dataset Integration System Test Suite")
    print("=" * 50)
    
    try:
        # Create and run test suite
        test_suite = IntegrationTestSuite()
        results = test_suite.run_all_tests()
        
        # Return appropriate exit code
        return 0 if results['failed'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 