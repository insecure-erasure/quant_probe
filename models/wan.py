"""
Architecture configuration for Wan 2.1 video diffusion model.

Transformer blocks use the blocks.* key namespace with separate cross-attention
and self-attention projections (k, v, q, o) plus feed-forward layers (ffn.0,
ffn.2).  No refiner sub-graphs.

Calibrated defaults are sourced from the Wan 2.1 14B score distribution.
"""

import re
from typing import Dict, Set

from quant_probe.models.base import ArchitectureConfig


# ---------------------------------------------------------------------------
# Layer patterns
# ---------------------------------------------------------------------------

_LAYER_PATTERNS = {
    "cross_attn.k":     re.compile(r"blocks\.(\d+)\.cross_attn\.k\.weight$"),
    "cross_attn.v":     re.compile(r"blocks\.(\d+)\.cross_attn\.v\.weight$"),
    "cross_attn.q":     re.compile(r"blocks\.(\d+)\.cross_attn\.q\.weight$"),
    "cross_attn.o":     re.compile(r"blocks\.(\d+)\.cross_attn\.o\.weight$"),
    "cross_attn.k_img": re.compile(r"blocks\.(\d+)\.cross_attn\.k_img\.weight$"),
    "cross_attn.v_img": re.compile(r"blocks\.(\d+)\.cross_attn\.v_img\.weight$"),
    "self_attn.k":      re.compile(r"blocks\.(\d+)\.self_attn\.k\.weight$"),
    "self_attn.v":      re.compile(r"blocks\.(\d+)\.self_attn\.v\.weight$"),
    "self_attn.q":      re.compile(r"blocks\.(\d+)\.self_attn\.q\.weight$"),
    "self_attn.o":      re.compile(r"blocks\.(\d+)\.self_attn\.o\.weight$"),
    "ffn.0":            re.compile(r"blocks\.(\d+)\.ffn\.0\.weight$"),
    "ffn.2":            re.compile(r"blocks\.(\d+)\.ffn\.2\.weight$"),
}

_DETAIL_GROUPS = {
    "cross_attn": [
        "cross_attn.k", "cross_attn.v", "cross_attn.q", "cross_attn.o",
        "cross_attn.k_img", "cross_attn.v_img",
    ],
    "self_attn": ["self_attn.k", "self_attn.v", "self_attn.q", "self_attn.o"],
    "ffn":       ["ffn.0", "ffn.2"],
}

# ---------------------------------------------------------------------------
# Calibrated defaults (Wan 2.1 14B score distribution)
# ---------------------------------------------------------------------------

_DEFAULT_ARGS = {
    "fp8_percentile":   75.0,
    "keep_percentile":  90.0,
    "kurtosis_keep":     8.0,
    "fp8_min_score":    0.50,
    "min_group_spread": 0.06,
}

# ---------------------------------------------------------------------------
# Subgraph sets
# ---------------------------------------------------------------------------

_MAIN_SUBGRAPHS: Set[str]    = {"blocks"}
_REFINER_SUBGRAPHS: Set[str] = set()

# ---------------------------------------------------------------------------
# Architecture-specific callables
# ---------------------------------------------------------------------------

def infer_subgraph(key: str) -> str:
    """All Wan 2.1 keys live under blocks.*."""
    return "blocks"


_LAYER_TYPE_KEY_MAP = {
    "cross_attn.k":     r"cross_attn\.k\.weight",
    "cross_attn.v":     r"cross_attn\.v\.weight",
    "cross_attn.q":     r"cross_attn\.q\.weight",
    "cross_attn.o":     r"cross_attn\.o\.weight",
    "cross_attn.k_img": r"cross_attn\.k_img\.weight",
    "cross_attn.v_img": r"cross_attn\.v_img\.weight",
    "self_attn.k":      r"self_attn\.k\.weight",
    "self_attn.v":      r"self_attn\.v\.weight",
    "self_attn.q":      r"self_attn\.q\.weight",
    "self_attn.o":      r"self_attn\.o\.weight",
    "ffn.0":            r"ffn\.0\.weight",
    "ffn.2":            r"ffn\.2\.weight",
}


def layer_type_to_key(layer_type: str) -> str:
    """Return the regex key fragment for a given layer type."""
    return _LAYER_TYPE_KEY_MAP.get(layer_type, layer_type)


def build_block_path(subgraph: str, block_alt: str, key_pattern: str) -> str:
    """Build the full regex for one (subgraph, blocks, layer) combination."""
    return rf"blocks\.({block_alt})\.{key_pattern}"


# ---------------------------------------------------------------------------
# CONFIG instance
# ---------------------------------------------------------------------------

CONFIG = ArchitectureConfig(
    name                      = "wan",
    description               = "Wan 2.1 video diffusion model",
    convert_flag              = "--wan",
    layer_patterns            = _LAYER_PATTERNS,
    refiner_patterns          = None,
    detail_groups             = _DETAIL_GROUPS,
    refiner_detail_groups     = None,
    default_args              = _DEFAULT_ARGS,
    highprec_excluded         = None,
    # Q/K projections: primary source of quantization error in DiT (QK^T
    # amplifies leptokurtic distributions).
    spread_filter_recommended = ["cross_attn.k", "cross_attn.q",
                                 "self_attn.k",  "self_attn.q"],
    main_subgraphs            = _MAIN_SUBGRAPHS,
    refiner_subgraphs         = _REFINER_SUBGRAPHS,
    infer_subgraph            = infer_subgraph,
    layer_type_to_key         = layer_type_to_key,
    build_block_path          = build_block_path,
)
