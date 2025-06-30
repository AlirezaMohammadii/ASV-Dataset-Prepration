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


@dataclass
class PathConfig:
    """Configuration for file paths"""
    # Base directories
    asv_datasets_root: Path = Path("../")
    asvspoof2019_root: Path = Path("../asvpoof-2019/versions/1")
    deep_asv_detection_root: Path = Path("../../Deep ASV Detection")
    
    # ASVspoof-2019 specific paths
    la_root: Path = field(init=False)
    pa_root: Path = field(init=False)
    
    # Data directories
    la_train_dir: Path = field(init=False)
    la_dev_dir: Path = field(init=False)
    la_eval_dir: Path = field(init=False)
    
    # Protocol directories
    la_cm_protocols_dir: Path = field(init=False)
    la_asv_protocols_dir: Path = field(init=False)
    
    # Output directories
    output_root: Path = Path("./output")
    converted_data_dir: Path = field(init=False)
    analysis_output_dir: Path = field(init=False)
    logs_dir: Path = Path("./logs")
    
    def __post_init__(self):
        """Initialize derived paths"""
        self.la_root = self.asvspoof2019_root / "LA" / "LA"
        self.pa_root = self.asvspoof2019_root / "PA" / "PA"
        
        # LA data directories
        self.la_train_dir = self.la_root / "ASVspoof2019_LA_train" / "flac"
        self.la_dev_dir = self.la_root / "ASVspoof2019_LA_dev" / "flac"
        self.la_eval_dir = self.la_root / "ASVspoof2019_LA_eval" / "flac"
        
        # Protocol directories
        self.la_cm_protocols_dir = self.la_root / "ASVspoof2019_LA_cm_protocols"
        self.la_asv_protocols_dir = self.la_root / "ASVspoof2019_LA_asv_protocols"
        
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
    
    def __init__(self):
        self.paths = PathConfig()
        self.conversion = ConversionConfig()
        self.analysis = AnalysisConfig()
        
        # Attack type mappings
        self.attack_type_descriptions = {
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
            "A19": "VC spectral filtering"
        }
        
        # Attack type categories
        self.attack_categories = {
            "TTS": ["A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12", "A16"],
            "VC": ["A05", "A06", "A17", "A18", "A19"],
            "TTS_VC": ["A13", "A14", "A15"]
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
        """Get the category (TTS/VC/TTS_VC) for an attack ID"""
        for category, attacks in self.attack_categories.items():
            if attack_id in attacks:
                return category.lower()
        return "unknown"
    
    def get_deepfake_filename_format(self, attack_id: str) -> str:
        """Get the appropriate deepfake filename format for an attack ID"""
        category = self.get_attack_category(attack_id)
        if category == "unknown":
            category = self.conversion.default_attack_type
        return self.conversion.deepfake_file_format.replace("{}", category)
    
    def validate_paths(self) -> List[str]:
        """Validate that all required paths exist"""
        errors = []
        
        # Check ASVspoof-2019 paths
        if not self.paths.asvspoof2019_root.exists():
            errors.append(f"ASVspoof-2019 root not found: {self.paths.asvspoof2019_root}")
        
        if not self.paths.la_root.exists():
            errors.append(f"LA root not found: {self.paths.la_root}")
            
        # Check data directories
        for split, path in [
            ("train", self.paths.la_train_dir),
            ("dev", self.paths.la_dev_dir), 
            ("eval", self.paths.la_eval_dir)
        ]:
            if not path.exists():
                errors.append(f"LA {split} directory not found: {path}")
        
        # Check protocol directories
        if not self.paths.la_cm_protocols_dir.exists():
            errors.append(f"LA CM protocols not found: {self.paths.la_cm_protocols_dir}")
            
        return errors


# Global configuration instance
config = ASVDatasetConfig() 