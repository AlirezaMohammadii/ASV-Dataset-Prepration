"""
ASV Dataset Preparation Configuration
====================================

This module contains all configuration settings for ASV dataset preparation and conversion
to Deep ASV Detection format.

Author: ASV Dataset Preparation System
Version: 1.0.0
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from enum import Enum


class DatasetSplit(Enum):
    """Enumeration for dataset splits"""
    TRAIN = "train"
    DEV = "dev" 
    EVAL = "eval"


class AttackType(Enum):
    """Enumeration for ASVspoof attack types"""
    # TTS attacks
    A01 = "A01"  # TTS neural waveform model
    A02 = "A02"  # TTS vocoder
    A03 = "A03"  # TTS vocoder
    A04 = "A04"  # TTS waveform concatenation
    A07 = "A07"  # TTS vocoder+GAN
    A08 = "A08"  # TTS neural waveform
    A09 = "A09"  # TTS vocoder
    A10 = "A10"  # TTS neural waveform
    A11 = "A11"  # TTS griffin lim
    A12 = "A12"  # TTS neural waveform
    A16 = "A16"  # TTS waveform concatenation
    
    # VC attacks
    A05 = "A05"  # VC vocoder
    A06 = "A06"  # VC spectral filtering
    A17 = "A17"  # VC waveform filtering
    A18 = "A18"  # VC vocoder
    A19 = "A19"  # VC spectral filtering
    
    # TTS_VC attacks
    A13 = "A13"  # TTS_VC waveform concatenation+waveform filtering
    A14 = "A14"  # TTS_VC vocoder
    A15 = "A15"  # TTS_VC neural waveform


class DatasetYear(Enum):
    ASV2019 = "2019"
    ASV2021 = "2021"


class Scenario(Enum):
    LA = "LA"
    DF = "DF"
    PA = "PA"


@dataclass
class PathConfig:
    """Configuration for file paths"""
    # Base directories
    asv_datasets_root: Path = Path("../")
    asvspoof2019_root: Path = Path("../asvpoof-2019/versions/1")
    asvspoof2021_root: Path = Path("../ASVspoof_2021")
    deep_asv_detection_root: Path = Path("../../Deep ASV Detection")
    
    # ASVspoof-2019 specific paths
    la_root: Path = field(init=False)
    pa_root: Path = field(init=False)
    
    # 2019 Data directories
    la_train_dir: Path = field(init=False)
    la_dev_dir: Path = field(init=False)
    la_eval_dir: Path = field(init=False)
    
    # 2019 PA Data directories
    pa_train_dir: Path = field(init=False)
    pa_dev_dir: Path = field(init=False)
    pa_eval_dir: Path = field(init=False)
    
    # 2019 Protocol directories
    la_cm_protocols_dir: Path = field(init=False)
    la_asv_protocols_dir: Path = field(init=False)
    pa_cm_protocols_dir: Path = field(init=False)
    pa_asv_protocols_dir: Path = field(init=False)

    # ASVspoof-2021 specific (focus LA; DF/PA eval-only)
    la2021_root: Path = field(init=False)
    la2021_eval_dir: Path = field(init=False)
    la2021_cm_eval_protocol_file: Path = field(init=False)
    la2021_cm_keys_metadata_file: Path = field(init=False)
    
    # ASVspoof-2021 PA specific
    pa2021_root: Path = field(init=False)
    pa2021_eval_dir: Path = field(init=False)
    pa2021_cm_eval_protocol_file: Path = field(init=False)
    pa2021_cm_keys_metadata_file: Path = field(init=False)
    
    # ASVspoof-2021 DF specific (for future use)
    df2021_root: Path = field(init=False)
    df2021_eval_dir: Path = field(init=False)
    df2021_cm_eval_protocol_file: Path = field(init=False)
    df2021_cm_keys_metadata_file: Path = field(init=False)
    
    # Output directories
    output_root: Path = Path("./output")
    converted_data_dir: Path = field(init=False)
    analysis_output_dir: Path = field(init=False)
    logs_dir: Path = Path("./logs")
    
    def __post_init__(self):
        """Initialize derived paths"""
        # 2019 LA roots
        self.la_root = self.asvspoof2019_root / "LA" / "LA"
        self.pa_root = self.asvspoof2019_root / "PA" / "PA"
        
        # 2019 LA data directories
        self.la_train_dir = self.la_root / "ASVspoof2019_LA_train" / "flac"
        self.la_dev_dir = self.la_root / "ASVspoof2019_LA_dev" / "flac"
        self.la_eval_dir = self.la_root / "ASVspoof2019_LA_eval" / "flac"
        
        # 2019 PA data directories
        self.pa_train_dir = self.pa_root / "ASVspoof2019_PA_train" / "flac"
        self.pa_dev_dir = self.pa_root / "ASVspoof2019_PA_dev" / "flac"
        self.pa_eval_dir = self.pa_root / "ASVspoof2019_PA_eval" / "flac"
        
        # 2019 Protocol directories
        self.la_cm_protocols_dir = self.la_root / "ASVspoof2019_LA_cm_protocols"
        self.la_asv_protocols_dir = self.la_root / "ASVspoof2019_LA_asv_protocols"
        self.pa_cm_protocols_dir = self.pa_root / "ASVspoof2019_PA_cm_protocols"
        self.pa_asv_protocols_dir = self.pa_root / "ASVspoof2019_PA_asv_protocols"

        # 2021 LA
        self.la2021_root = self.asvspoof2021_root / "ASVspoof2021_LA_eval" / "ASVspoof2021_LA_eval"
        self.la2021_eval_dir = self.la2021_root / "flac"
        self.la2021_cm_eval_protocol_file = self.la2021_root / "ASVspoof2021.LA.cm.eval.trl.txt"
        self.la2021_cm_keys_metadata_file = self.asvspoof2021_root / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt"
        
        # 2021 PA
        self.pa2021_root = self.asvspoof2021_root / "ASVspoof2021_PA_eval_part00" / "ASVspoof2021_PA_eval"
        self.pa2021_eval_dir = self.pa2021_root / "flac"
        # PA2021 doesn't have separate protocol files, only metadata
        self.pa2021_cm_eval_protocol_file = None
        self.pa2021_cm_keys_metadata_file = self.asvspoof2021_root / "PA-keys-full" / "keys" / "PA" / "CM" / "trial_metadata.txt"
        
        # 2021 DF (for future use)
        self.df2021_root = self.asvspoof2021_root / "ASVspoof2021_DF_eval_part00" / "ASVspoof2021_DF_eval"
        self.df2021_eval_dir = self.df2021_root / "flac"
        self.df2021_cm_eval_protocol_file = self.df2021_root / "ASVspoof2021.DF.cm.eval.trl.txt"
        self.df2021_cm_keys_metadata_file = self.asvspoof2021_root / "DF-keys-full" / "keys" / "DF" / "CM" / "trial_metadata.txt"
        
        # Output directories
        self.converted_data_dir = self.output_root / "converted_data"
        self.analysis_output_dir = self.output_root / "analysis"
        
        # Ensure output directories exist
        self.output_root.mkdir(exist_ok=True)
        self.converted_data_dir.mkdir(exist_ok=True)
        self.analysis_output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


@dataclass
class ConversionConfig:
    """Configuration for dataset conversion"""
    # User mapping settings
    max_users: int = 50
    min_files_per_user: int = 10
    max_files_per_user: int = 200
    
    # Balance settings
    target_genuine_ratio: float = 0.6  # 60% genuine, 40% spoofed
    balance_tolerance: float = 0.1  # Allow 10% deviation
    
    # File naming settings
    user_id_format: str = "user_{:02d}"
    genuine_file_format: str = "user{:02d}_genuine_{:03d}.flac"
    deepfake_file_format: str = "user{:02d}_deepfake_{}__{:03d}.flac"
    include_attack_id_in_filename: bool = False
    
    # Attack type mapping
    preserve_attack_types: bool = True
    default_attack_type: str = "tts"
    
    # Processing settings
    copy_files: bool = True  # True to copy, False to move
    validate_audio: bool = True
    create_backup: bool = True


@dataclass
class AnalysisConfig:
    """Configuration for dataset analysis"""
    # Output formats
    generate_csv: bool = True
    generate_json: bool = True
    generate_plots: bool = True
    
    # Analysis depth
    detailed_speaker_analysis: bool = True
    attack_type_analysis: bool = True
    file_size_analysis: bool = True
    duration_analysis: bool = False  # Requires audio loading
    
    # Reporting
    create_summary_report: bool = True
    include_recommendations: bool = True


class ASVDatasetConfig:
    """Main configuration class for ASV dataset preparation"""
    
    def __init__(self, dataset_year: DatasetYear = DatasetYear.ASV2019, scenario: Scenario = Scenario.LA):
        self.paths = PathConfig()
        self.conversion = ConversionConfig()
        self.analysis = AnalysisConfig()
        self.dataset_year = dataset_year
        self.scenario = scenario
        
        # Attack type mappings
        self.attack_type_descriptions = {
            # LA attack types
            "A01": "TTS neural waveform model",
            "A02": "TTS vocoder",
            "A03": "TTS vocoder", 
            "A04": "TTS waveform concatenation",
            "A05": "VC vocoder",
            "A06": "VC spectral filtering",
            "A07": "TTS vocoder+GAN",
            "A08": "TTS neural waveform",
            "A09": "TTS vocoder",
            "A10": "TTS neural waveform",
            "A11": "TTS griffin lim",
            "A12": "TTS neural waveform",
            "A13": "TTS_VC waveform concatenation+waveform filtering",
            "A14": "TTS_VC vocoder",
            "A15": "TTS_VC neural waveform",
            "A16": "TTS waveform concatenation",
            "A17": "VC waveform filtering",
            "A18": "VC vocoder",
            "A19": "VC spectral filtering",
            # PA attack types (environment + replay device combinations)
            # Format: [replay_quality][room_size][distance]
            # Room sizes: a=small, b=medium, c=large
            # Distances: a=short, b=medium, c=long
            # Replay quality: a=perfect, b=high, c=low
            
            # Perfect quality replay devices (a**)
            "aaa": "PA_perfect_small_short",    # Perfect device, small room, short distance
            "aab": "PA_perfect_small_medium",   # Perfect device, small room, medium distance
            "aac": "PA_perfect_small_long",     # Perfect device, small room, long distance
            "aba": "PA_perfect_medium_short",   # Perfect device, medium room, short distance
            "abb": "PA_perfect_medium_medium",  # Perfect device, medium room, medium distance
            "abc": "PA_perfect_medium_long",    # Perfect device, medium room, long distance
            "aca": "PA_perfect_large_short",    # Perfect device, large room, short distance
            "acb": "PA_perfect_large_medium",   # Perfect device, large room, medium distance
            "acc": "PA_perfect_large_long",     # Perfect device, large room, long distance
            
            # High quality replay devices (b**)
            "baa": "PA_high_small_short",       # High quality device, small room, short distance
            "bab": "PA_high_small_medium",      # High quality device, small room, medium distance
            "bac": "PA_high_small_long",        # High quality device, small room, long distance
            "bba": "PA_high_medium_short",      # High quality device, medium room, short distance
            "bbb": "PA_high_medium_medium",     # High quality device, medium room, medium distance
            "bbc": "PA_high_medium_long",       # High quality device, medium room, long distance
            "bca": "PA_high_large_short",       # High quality device, large room, short distance
            "bcb": "PA_high_large_medium",      # High quality device, large room, medium distance
            "bcc": "PA_high_large_long",        # High quality device, large room, long distance
            
            # Low quality replay devices (c**)
            "caa": "PA_low_small_short",        # Low quality device, small room, short distance
            "cab": "PA_low_small_medium",       # Low quality device, small room, medium distance
            "cac": "PA_low_small_long",         # Low quality device, small room, long distance
            "cba": "PA_low_medium_short",       # Low quality device, medium room, short distance
            "cbb": "PA_low_medium_medium",      # Low quality device, medium room, medium distance
            "cbc": "PA_low_medium_long",        # Low quality device, medium room, long distance
            "cca": "PA_low_large_short",        # Low quality device, large room, short distance
            "ccb": "PA_low_large_medium",       # Low quality device, large room, medium distance
            "ccc": "PA_low_large_long",         # Low quality device, large room, long distance
            
            # PA Attack IDs (from protocol column 4)
            "AA": "PA_perfect_small_short",     # Perfect device, small room, short distance
            "AB": "PA_perfect_small_medium",    # Perfect device, small room, medium distance
            "AC": "PA_perfect_small_long",      # Perfect device, small room, long distance
            "BA": "PA_perfect_medium_short",    # Perfect device, medium room, short distance
            "BB": "PA_perfect_medium_medium",   # Perfect device, medium room, medium distance
            "BC": "PA_perfect_medium_long",     # Perfect device, medium room, long distance
            "CA": "PA_perfect_large_short",     # Perfect device, large room, short distance
            "CB": "PA_perfect_large_medium",    # Perfect device, large room, medium distance
            "CC": "PA_perfect_large_long"       # Perfect device, large room, long distance
        }
        
        # Attack type categories
        self.attack_categories = {
            "TTS": ["A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12", "A16"],
            "VC": ["A05", "A06", "A17", "A18", "A19"],
            "TTS_VC": ["A13", "A14", "A15"],
            "PA_PERFECT": ["aaa", "aab", "aac", "aba", "abb", "abc", "aca", "acb", "acc",
                          "AA", "AB", "AC", "BA", "BB", "BC", "CA", "CB", "CC"],
            "PA_HIGH": ["baa", "bab", "bac", "bba", "bbb", "bbc", "bca", "bcb", "bcc"],
            "PA_LOW": ["caa", "cab", "cac", "cba", "cbb", "cbc", "cca", "ccb", "ccc"]
        }
        
        # Deep ASV Detection compatibility
        self.deep_asv_patterns = {
            "subdir_pattern": r'^user_(\d{2})$',
            "genuine_pattern": r'^user(\d{2})_genuine_(\d{3})(\.(wav|flac))$',
            "deepfake_pattern": r'^user(\d{2})_deepfake_[a-z]+_(\d{3})(\.(wav|flac))$'
        }
        
        # Supported audio formats
        self.audio_extensions = {'.wav', '.flac', '.mp3', '.m4a'}
        
    def get_attack_category(self, attack_id: str) -> str:
        """Get the category (TTS/VC/TTS_VC/PA_PERFECT/PA_HIGH/PA_LOW) for an attack ID. Raise if unknown."""
        for category, attacks in self.attack_categories.items():
            if attack_id in attacks:
                return category.lower()
        
        # Handle PA attack IDs that are combinations of environment and replay device
        if self.scenario == Scenario.PA and attack_id != "-":
            # PA attack IDs are in format like "AA", "AB", etc. or with quality suffix
            if len(attack_id) == 2 and attack_id[0] in "ABC" and attack_id[1] in "ABC":
                return "pa_perfect"  # Default to perfect quality
            elif attack_id.endswith("_high"):
                return "pa_high"
            elif attack_id.endswith("_low"):
                return "pa_low"
        
        # Unknown attacks are not accepted
        raise ValueError(f"Unknown attack id '{attack_id}'. Extend attack mappings to proceed.")
    
    def get_deepfake_filename_format(self, attack_id: str) -> str:
        """Get the appropriate deepfake filename format for an attack ID"""
        category = self.get_attack_category(attack_id)
        return self.conversion.deepfake_file_format.replace("{}", category)
    
    def validate_paths(self) -> List[str]:
        """Validate that all required paths exist, based on dataset year and scenario"""
        errors: List[str] = []
        
        if self.dataset_year == DatasetYear.ASV2019:
            if not self.paths.asvspoof2019_root.exists():
                errors.append(f"ASVspoof-2019 root not found: {self.paths.asvspoof2019_root}")
            
            if self.scenario == Scenario.LA:
                if not self.paths.la_root.exists():
                    errors.append(f"LA root not found: {self.paths.la_root}")
                for split, path in [
                    ("train", self.paths.la_train_dir),
                    ("dev", self.paths.la_dev_dir), 
                    ("eval", self.paths.la_eval_dir)
                ]:
                    if not path.exists():
                        errors.append(f"LA {split} directory not found: {path}")
                if not self.paths.la_cm_protocols_dir.exists():
                    errors.append(f"LA CM protocols not found: {self.paths.la_cm_protocols_dir}")
            
            elif self.scenario == Scenario.PA:
                if not self.paths.pa_root.exists():
                    errors.append(f"PA root not found: {self.paths.pa_root}")
                for split, path in [
                    ("train", self.paths.pa_train_dir),
                    ("dev", self.paths.pa_dev_dir), 
                    ("eval", self.paths.pa_eval_dir)
                ]:
                    if not path.exists():
                        errors.append(f"PA {split} directory not found: {path}")
                if not self.paths.pa_cm_protocols_dir.exists():
                    errors.append(f"PA CM protocols not found: {self.paths.pa_cm_protocols_dir}")
            
            else:
                errors.append(f"Scenario {self.scenario.value} not supported for ASVspoof2019")
        
        else:  # 2021 (eval only)
            if not self.paths.asvspoof2021_root.exists():
                errors.append(f"ASVspoof-2021 root not found: {self.paths.asvspoof2021_root}")
            
            if self.scenario == Scenario.LA:
                if not self.paths.la2021_root.exists():
                    errors.append(f"ASVspoof2021 LA root not found: {self.paths.la2021_root}")
                if not self.paths.la2021_eval_dir.exists():
                    errors.append(f"ASVspoof2021 LA eval flac dir not found: {self.paths.la2021_eval_dir}")
                if not self.paths.la2021_cm_eval_protocol_file.exists():
                    errors.append(f"ASVspoof2021 LA eval protocol file not found: {self.paths.la2021_cm_eval_protocol_file}")
                if not self.paths.la2021_cm_keys_metadata_file.exists():
                    errors.append(f"ASVspoof2021 LA keys metadata not found: {self.paths.la2021_cm_keys_metadata_file}")
            
            elif self.scenario == Scenario.PA:
                if not self.paths.pa2021_root.exists():
                    errors.append(f"ASVspoof2021 PA root not found: {self.paths.pa2021_root}")
                if not self.paths.pa2021_eval_dir.exists():
                    errors.append(f"ASVspoof2021 PA eval flac dir not found: {self.paths.pa2021_eval_dir}")
                # PA2021 doesn't have separate protocol files, only metadata
                if not self.paths.pa2021_cm_keys_metadata_file.exists():
                    errors.append(f"ASVspoof2021 PA keys metadata not found: {self.paths.pa2021_cm_keys_metadata_file}")
            
            else:
                errors.append(f"Scenario {self.scenario.value} support not implemented yet for ASVspoof2021")
            
        return errors


# Global configuration instance (default to 2019 LA)
config = ASVDatasetConfig() 


def set_active_dataset(dataset_year: DatasetYear, scenario: Scenario) -> ASVDatasetConfig:
    """Update and return the global configuration to the requested dataset and scenario."""
    global config
    config = ASVDatasetConfig(dataset_year=dataset_year, scenario=scenario)
    return config 