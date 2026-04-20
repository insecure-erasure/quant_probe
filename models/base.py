"""
Formal contract (Protocol) that every architecture module must satisfy.

Each architecture module (wan.py, zimage.py, ...) must expose a module-level
CONFIG dict whose value is an instance of ArchitectureConfig.

registry.py collects these CONFIG objects and builds MODEL_CONFIGS.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple
import re


@dataclass
class ArchitectureConfig:
    """
    Full specification of a model architecture for quant_probe.

    Fields
    ------
    name : str
        Short identifier used on the CLI (e.g. "wan", "zimage").
    description : str
        Human-readable model description shown in output headers.
    convert_flag : str
        Corresponding flag(s) for convert_to_quant (shown in suggested params).
    layer_patterns : Dict[str, re.Pattern]
        Regex patterns for main transformer blocks.
        Each pattern must capture the block index in group 1.
    refiner_patterns : Optional[Dict[str, re.Pattern]]
        Regex patterns for refiner sub-graphs (e.g. Z-Image context/noise
        refiners). Each pattern must capture the subgraph name in group 1 and
        the block index in group 2.  None if the architecture has no refiners.
    detail_groups : Dict[str, List[str]]
        Grouping of layer_type names for the detail tables (main blocks).
    refiner_detail_groups : Optional[Dict[str, List[str]]]
        Same as detail_groups but for refiner patterns.  None if no refiners.
    default_args : Dict[str, float]
        Calibrated default values for the scoring / threshold parameters:
          fp8_percentile, keep_percentile, kurtosis_keep,
          fp8_min_score, min_group_spread.
    highprec_excluded : Optional[List[str]]
        Layer name fragments that receive high-precision treatment in
        convert_to_quant and are intentionally excluded from analysis.
        None if the architecture has no such layers.
    spread_filter_recommended : List[str]
        Layer types for which --spread-filter-exempt is strongly recommended
        (typically attention Q/K projections).
    main_subgraphs : Set[str]
        Subgraph identifiers that belong to the main transformer blocks
        (used to split metrics after analysis).
    refiner_subgraphs : Set[str]
        Subgraph identifiers that belong to refiner blocks.
        Empty set if the architecture has no refiners.
    infer_subgraph : Callable[[str], str]
        Given a full safetensors key, returns the subgraph identifier string
        (e.g. "layers", "blocks", "context_refiner", "noise_refiner").
        Must be consistent with main_subgraphs / refiner_subgraphs.
    layer_type_to_key : Callable[[str], str]
        Given a layer_type string (e.g. "attention.qkv"), returns the regex
        fragment that matches that layer's weight key suffix
        (e.g. "attention\\.qkv\\.weight" as a raw string).
        Used by core.py to build --custom-layers / --exclude-layers regexes.
    build_block_path : Callable[[str, str, str], str]
        Given (subgraph, block_alternation, key_pattern), returns the full
        regex string for one entry in the --custom-layers / --exclude-layers
        output (e.g. "layers.(0|1|2).attention.qkv.weight" as a raw regex).
    """

    name: str
    description: str
    convert_flag: str

    layer_patterns: Dict[str, "re.Pattern[str]"]
    refiner_patterns: Optional[Dict[str, "re.Pattern[str]"]]

    detail_groups: Dict[str, List[str]]
    refiner_detail_groups: Optional[Dict[str, List[str]]]

    default_args: Dict[str, float]
    highprec_excluded: Optional[List[str]]
    spread_filter_recommended: List[str]

    main_subgraphs: Set[str]
    refiner_subgraphs: Set[str]

    # Callable fields — architecture-specific logic injected by each module.
    infer_subgraph: Callable[[str], str]
    layer_type_to_key: Callable[[str], str]
    build_block_path: Callable[[str, str, str], str]
