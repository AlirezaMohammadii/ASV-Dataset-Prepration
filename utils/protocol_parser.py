"""
ASVspoof-2019 Protocol Parser
============================

This module provides utilities for parsing ASVspoof-2019 protocol files and extracting
labeling information for dataset conversion.

Author: ASV Dataset Preparation System
Version: 1.0.0
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.dataset_config import config, DatasetSplit
from config.dataset_config import DatasetYear
from config.dataset_config import Scenario


class ProtocolEntry(NamedTuple):
    """Represents a single entry in an ASVspoof protocol file"""
    speaker_id: str
    audio_file_name: str
    unused_field: str  # Always '-' in LA scenario
    system_id: str     # Attack ID (A01-A19) or '-' for bonafide
    key: str          # 'bonafide' or 'spoof'


@dataclass
class FileInfo:
    """Information about a single audio file"""
    filename: str
    speaker_id: str
    split: DatasetSplit
    is_genuine: bool
    attack_id: Optional[str] = None
    attack_category: Optional[str] = None
    file_path: Optional[Path] = None
    file_size: Optional[int] = None


class ProtocolParser:
    """Parser for ASVspoof-2019 protocol files"""
    
    def __init__(self, config_instance=None):
        # Use provided config or fall back to global config
        self.config = config_instance if config_instance is not None else config
        self.logger = logging.getLogger(__name__)
        self._protocol_cache = {}
        
        # Protocol file patterns (2019 LA)
        self.la_protocol_files = {
            DatasetSplit.TRAIN: "ASVspoof2019.LA.cm.train.trn.txt",
            DatasetSplit.DEV: "ASVspoof2019.LA.cm.dev.trl.txt", 
            DatasetSplit.EVAL: "ASVspoof2019.LA.cm.eval.trl.txt"
        }
        
        # Protocol file patterns (2019 PA)
        self.pa_protocol_files = {
            DatasetSplit.TRAIN: "ASVspoof2019.PA.cm.train.trn.txt",
            DatasetSplit.DEV: "ASVspoof2019.PA.cm.dev.trl.txt", 
            DatasetSplit.EVAL: "ASVspoof2019.PA.cm.eval.trl.txt"
        }
        
        # File name patterns (2019 LA)
        self.la_file_patterns = {
            DatasetSplit.TRAIN: re.compile(r'^LA_T_(\d+)\.flac$'),
            DatasetSplit.DEV: re.compile(r'^LA_D_(\d+)\.flac$'),
            DatasetSplit.EVAL: re.compile(r'^LA_E_(\d+)\.flac$')
        }
        
        # File name patterns (2019 PA)
        self.pa_file_patterns = {
            DatasetSplit.TRAIN: re.compile(r'^PA_T_(\d+)\.flac$'),
            DatasetSplit.DEV: re.compile(r'^PA_D_(\d+)\.flac$'),
            DatasetSplit.EVAL: re.compile(r'^PA_E_(\d+)\.flac$')
        }
        
        # Get protocol files based on scenario
        if self.config.scenario == Scenario.PA:
            self.protocol_files = self.pa_protocol_files
            self.file_patterns = self.pa_file_patterns
        else:
            self.protocol_files = self.la_protocol_files
            self.file_patterns = self.la_file_patterns
    
    def parse_protocol_file(self, protocol_file_path: Path) -> List[ProtocolEntry]:
        """
        Parse a single protocol file and return list of entries.
        Only used for 2019 LA style.
        """
        if not protocol_file_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_file_path}")
        
        entries: List[ProtocolEntry] = []
        
        try:
            with open(protocol_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        self.logger.warning(
                            f"Invalid line format at {protocol_file_path}:{line_num}: {line}"
                        )
                        continue
                    entry = ProtocolEntry(
                        speaker_id=parts[0],
                        audio_file_name=parts[1],
                        unused_field=parts[2],
                        system_id=parts[3],
                        key=parts[4]
                    )
                    if not self._validate_protocol_entry(entry):
                        self.logger.warning(
                            f"Invalid protocol entry at {protocol_file_path}:{line_num}: {line}"
                        )
                        continue
                    entries.append(entry)
        except Exception as e:
            raise ValueError(f"Error parsing protocol file {protocol_file_path}: {e}") from e
        
        self.logger.info(f"Parsed {len(entries)} entries from {protocol_file_path}")
        return entries
    
    def _validate_protocol_entry(self, entry: ProtocolEntry) -> bool:
        """Validate a protocol entry (2019 LA or PA)"""
        if self.config.scenario == Scenario.LA:
            # LA validation
            if not re.match(r'^LA_\d{4}$', entry.speaker_id):
                return False
            if not re.match(r'^LA_[TDE]_\d+$', entry.audio_file_name):
                return False
            if entry.unused_field != '-':
                return False
            if entry.key == 'bonafide':
                if entry.system_id != '-':
                    return False
            elif entry.key == 'spoof':
                if not re.match(r'^A\d{2}$', entry.system_id):
                    return False
            else:
                return False
        elif self.config.scenario == Scenario.PA:
            # PA validation
            if not re.match(r'^PA_\d{4}$', entry.speaker_id):
                return False
            if not re.match(r'^PA_[TDE]_\d+$', entry.audio_file_name):
                return False
            # PA has environment ID (3 letters) instead of unused field
            if not re.match(r'^[abc]{3}$', entry.unused_field):
                return False
            if entry.key == 'bonafide':
                if entry.system_id != '-':
                    return False
            elif entry.key == 'spoof':
                # PA attack IDs are combinations like "AA", "AB", etc.
                if not re.match(r'^[ABC]{2}$', entry.system_id):
                    return False
            else:
                return False
        else:
            return False
        return True
    
    def parse_all_protocols(self) -> Dict[DatasetSplit, List[ProtocolEntry]]:
        """
        Parse all protocol files and return organized entries.
        Switch behavior based on dataset year and scenario in config.
        """
        if getattr(self.config, 'dataset_year', None) and self.config.dataset_year.value == '2021':
            # 2021 LA eval only: use keys metadata for labels and attack IDs
            return self._parse_2021_la_keys_metadata()
        
        all_entries: Dict[DatasetSplit, List[ProtocolEntry]] = {}
        
        # Select protocol directory based on scenario
        if self.config.scenario == Scenario.PA:
            protocol_dir = self.config.paths.pa_cm_protocols_dir
        else:
            protocol_dir = self.config.paths.la_cm_protocols_dir
        
        for split, filename in self.protocol_files.items():
            protocol_path = protocol_dir / filename
            try:
                entries = self.parse_protocol_file(protocol_path)
                all_entries[split] = entries
                self.logger.info(f"Loaded {len(entries)} entries for {split.value} split")
            except Exception as e:
                self.logger.error(f"Failed to parse {split.value} protocol: {e}")
                all_entries[split] = []
        return all_entries
    
    def _parse_2021_la_keys_metadata(self) -> Dict[DatasetSplit, List[ProtocolEntry]]:
        """Parse ASVspoof2021 LA keys metadata (eval-only) into ProtocolEntry-like records."""
        path = self.config.paths.la2021_cm_keys_metadata_file
        if not path.exists():
            raise FileNotFoundError(f"Missing ASVspoof2021 LA keys metadata: {path}")
        entries_eval: List[ProtocolEntry] = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # Expected: speaker_id, file_id, codec, channel, system_id_or_dash, key, trimflag, split
                    if len(parts) < 8:
                        raise ValueError(f"Invalid metadata format at {path}:{line_num}: {line}")
                    speaker_id = parts[0]
                    audio_id = parts[1]  # e.g., LA_E_9332881
                    system_id = parts[4]
                    key = parts[5]  # 'spoof' or 'bonafide'
                    split_tag = parts[7]
                    if split_tag != 'eval':
                        continue
                    # ProtocolEntry expects audio_file_name without extension
                    entry = ProtocolEntry(
                        speaker_id=speaker_id,
                        audio_file_name=audio_id,
                        unused_field='-',
                        system_id=system_id if key == 'spoof' else '-',
                        key=key
                    )
                    entries_eval.append(entry)
        except Exception as e:
            raise ValueError(f"Error parsing 2021 keys metadata: {e}") from e
        return {DatasetSplit.EVAL: entries_eval}
    
    def create_file_mapping(self) -> Dict[str, FileInfo]:
        """
        Create a comprehensive mapping of all files with their metadata.
        
        Returns:
            Dictionary mapping filename to FileInfo
        """
        file_mapping = {}
        protocol_entries = self.parse_all_protocols()
        
        # Data directories for each split
        if getattr(self.config, 'dataset_year', None) and self.config.dataset_year == DatasetYear.ASV2021:
            data_dirs = {
                DatasetSplit.EVAL: self.config.paths.la2021_eval_dir
            }
        else:
            # 2019 - select data directories based on scenario
            if self.config.scenario == Scenario.PA:
                data_dirs = {
                    DatasetSplit.TRAIN: self.config.paths.pa_train_dir,
                    DatasetSplit.DEV: self.config.paths.pa_dev_dir,
                    DatasetSplit.EVAL: self.config.paths.pa_eval_dir
                }
            else:
                data_dirs = {
                    DatasetSplit.TRAIN: self.config.paths.la_train_dir,
                    DatasetSplit.DEV: self.config.paths.la_dev_dir,
                    DatasetSplit.EVAL: self.config.paths.la_eval_dir
                }
        
        for split, entries in protocol_entries.items():
            data_dir = data_dirs[split]
            
            for entry in entries:
                filename = entry.audio_file_name + ".flac"  # Add extension
                file_path = data_dir / filename
                
                # Get file size if file exists
                file_size = None
                if file_path.exists():
                    file_size = file_path.stat().st_size
                
                # Determine attack category
                attack_category = None
                if entry.key == 'spoof':
                    attack_category = self.config.get_attack_category(entry.system_id)
                
                file_info = FileInfo(
                    filename=filename,
                    speaker_id=entry.speaker_id,
                    split=split,
                    is_genuine=(entry.key == 'bonafide'),
                    attack_id=entry.system_id if entry.key == 'spoof' else None,
                    attack_category=attack_category,
                    file_path=file_path,
                    file_size=file_size
                )
                
                file_mapping[filename] = file_info
        
        self.logger.info(f"Created file mapping for {len(file_mapping)} files")
        return file_mapping
    
    def get_speaker_statistics(self) -> Dict[str, Dict]:
        """
        Get comprehensive statistics about speakers in the dataset.
        
        Returns:
            Dictionary with speaker statistics
        """
        file_mapping = self.create_file_mapping()
        speaker_stats = defaultdict(lambda: {
            'total_files': 0,
            'genuine_files': 0,
            'spoofed_files': 0,
            'attack_types': Counter(),
            'splits': Counter(),
            'total_size': 0
        })
        
        for file_info in file_mapping.values():
            speaker_id = file_info.speaker_id
            stats = speaker_stats[speaker_id]
            
            stats['total_files'] += 1
            stats['splits'][file_info.split.value] += 1
            
            if file_info.file_size:
                stats['total_size'] += file_info.file_size
            
            if file_info.is_genuine:
                stats['genuine_files'] += 1
            else:
                stats['spoofed_files'] += 1
                if file_info.attack_id:
                    stats['attack_types'][file_info.attack_id] += 1
        
        # Convert defaultdict to regular dict and add computed metrics
        result = {}
        for speaker_id, stats in speaker_stats.items():
            stats['genuine_ratio'] = stats['genuine_files'] / stats['total_files']
            stats['spoofed_ratio'] = stats['spoofed_files'] / stats['total_files']
            stats['avg_file_size'] = stats['total_size'] / stats['total_files'] if stats['total_files'] > 0 else 0
            result[speaker_id] = dict(stats)
        
        return result
    
    def get_attack_type_statistics(self) -> Dict[str, Dict]:
        """
        Get comprehensive statistics about attack types.
        
        Returns:
            Dictionary with attack type statistics
        """
        file_mapping = self.create_file_mapping()
        attack_stats = defaultdict(lambda: {
            'count': 0,
            'speakers': set(),
            'splits': Counter(),
            'total_size': 0,
            'category': None,
            'description': None
        })
        
        for file_info in file_mapping.values():
            if not file_info.is_genuine and file_info.attack_id:
                attack_id = file_info.attack_id
                stats = attack_stats[attack_id]
                
                stats['count'] += 1
                stats['speakers'].add(file_info.speaker_id)
                stats['splits'][file_info.split.value] += 1
                
                if file_info.file_size:
                    stats['total_size'] += file_info.file_size
                
                # Add metadata
                if not stats['category']:
                    stats['category'] = self.config.get_attack_category(attack_id)
                    stats['description'] = self.config.attack_type_descriptions.get(attack_id, "Unknown")
        
        # Convert sets to counts and add computed metrics
        result = {}
        for attack_id, stats in attack_stats.items():
            stats['unique_speakers'] = len(stats['speakers'])
            stats['speakers'] = list(stats['speakers'])  # Convert set to list for JSON serialization
            stats['avg_file_size'] = stats['total_size'] / stats['count'] if stats['count'] > 0 else 0
            result[attack_id] = dict(stats)
        
        return result
    
    def get_dataset_statistics(self) -> Dict:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            Dictionary with overall dataset statistics
        """
        file_mapping = self.create_file_mapping()
        
        # Overall statistics
        total_files = len(file_mapping)
        genuine_files = sum(1 for f in file_mapping.values() if f.is_genuine)
        spoofed_files = total_files - genuine_files
        
        # Split statistics
        split_stats = Counter()
        for file_info in file_mapping.values():
            split_stats[file_info.split.value] += 1
        
        # Speaker statistics
        unique_speakers = len(set(f.speaker_id for f in file_mapping.values()))
        
        # Attack type statistics
        attack_types = set()
        for file_info in file_mapping.values():
            if file_info.attack_id:
                attack_types.add(file_info.attack_id)
        
        # File size statistics
        file_sizes = [f.file_size for f in file_mapping.values() if f.file_size]
        total_size = sum(file_sizes) if file_sizes else 0
        avg_file_size = total_size / len(file_sizes) if file_sizes else 0
        
        return {
            'total_files': total_files,
            'genuine_files': genuine_files,
            'spoofed_files': spoofed_files,
            'genuine_ratio': genuine_files / total_files if total_files > 0 else 0,
            'spoofed_ratio': spoofed_files / total_files if total_files > 0 else 0,
            'unique_speakers': unique_speakers,
            'unique_attack_types': len(attack_types),
            'attack_types': list(attack_types),
            'split_distribution': dict(split_stats),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024),
            'avg_file_size_bytes': avg_file_size,
            'avg_file_size_kb': avg_file_size / 1024,
            'files_with_size_info': len(file_sizes),
            'files_missing_size_info': total_files - len(file_sizes)
        }


# Global parser instance
parser = ProtocolParser() 