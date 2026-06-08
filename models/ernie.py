"""
Architecture configuration for ERNIE Image Turbo image diffusion model.

Transformer blocks use the layers.* key namespace with separate Q/K/V/O
self-attention projections and SwiGLU feed-forward layers (gate_proj, up_proj,
linear_fc2).  No cross-attention — text conditioning is injected via
adaLN_modulation + text_proj.  No refiner sub-graphs.

Calibrated defaults are cloned from Z-Image (same DiT lineage) and should be
recalibrated empirically from ERNIE's score distribution.
"""

import re
from typing import Set

from quant_probe.models.base import ArchitectureConfig


# ---------------------------------------------------------------------------
# Layer patterns — main blocks (layers.0-N)
# ---------------------------------------------------------------------------
# Attention is split into separate Q/K/V/O projections (NOT fused qkv).
# MLP uses SwiGLU with gate_proj, up_proj, linear_fc2 (not w1/w2/w3).

_LAYER_PATTERNS = {
    "self_attn.q": re.compile(r"layers\.(\d+)\.self_attention\.to_q\.weight$"),
    "self_attn.k": re.compile(r"layers\.(\d+)\.self_attention\.to_k\.weight$"),
    "self_attn.v": re.compile(r"layers\.(\d+)\.self_attention\.to_v\.weight$"),
    "self_attn.o": re.compile(r"layers\.(\d+)\.self_attention\.to_out\.0\.weight$"),
    "ffn.gate":    re.compile(r"layers\.(\d+)\.mlp\.gate_proj\.weight$"),
    "ffn.up":      re.compile(r"layers\.(\d+)\.mlp\.up_proj\.weight$"),
    "ffn.down":    re.compile(r"layers\.(\d+)\.mlp\.linear_fc2\.weight$"),
}

_DETAIL_GROUPS = {
    "self_attention": ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"],
    "ffn":            ["ffn.gate", "ffn.up", "ffn.down"],
}

# ---------------------------------------------------------------------------
# Layers excluded from analysis (highprec treatment in convert_to_quant)
# ---------------------------------------------------------------------------
# These correspond to the global sensitive layers that should remain in BF16.
# Block-level norms (adaLN_sa_ln, adaLN_mlp_ln, norm_k, norm_q) are 1D and
# naturally excluded since they don't match any 2D layer pattern.

_HIGHPREC_EXCLUDED = [
    "x_embedder.proj",
    "time_embedding.linear_1",
    "time_embedding.linear_2",
    "text_proj",
    "adaLN_modulation.1",
    "final_norm.linear",
    "final_linear",
]

# ---------------------------------------------------------------------------
# Calibrated defaults (cloned from Z-Image — not yet recalibrated)
# ---------------------------------------------------------------------------

_DEFAULT_ARGS = {
    "fp8_percentile":   50.0,
    "keep_percentile":  90.0,
    "kurtosis_keep":     8.0,
    "fp8_min_score":    0.0,
    "min_group_spread": 0.20,
}

# ---------------------------------------------------------------------------
# Subgraph sets
# ---------------------------------------------------------------------------

_MAIN_SUBGRAPHS: Set[str]    = {"layers"}
_REFINER_SUBGRAPHS: Set[str] = set()

# ---------------------------------------------------------------------------
# Architecture-specific callables
# ---------------------------------------------------------------------------

def infer_subgraph(key: str) -> str:
    """All ERNIE keys live under layers.*."""
    return "layers"


_LAYER_TYPE_KEY_MAP = {
    "self_attn.q": r"self_attention\.to_q\.weight",
    "self_attn.k": r"self_attention\.to_k\.weight",
    "self_attn.v": r"self_attention\.to_v\.weight",
    "self_attn.o": r"self_attention\.to_out\.0\.weight",
    "ffn.gate":    r"mlp\.gate_proj\.weight",
    "ffn.up":      r"mlp\.up_proj\.weight",
    "ffn.down":    r"mlp\.linear_fc2\.weight",
}


def layer_type_to_key(layer_type: str) -> str:
    """Return the regex key fragment for a given layer type."""
    return _LAYER_TYPE_KEY_MAP.get(layer_type, layer_type)


def build_block_path(subgraph: str, block_alt: str, key_pattern: str) -> str:
    """Build the full regex for one (subgraph, blocks, layer) combination."""
    return rf"layers\.({block_alt})\.{key_pattern}"


# ---------------------------------------------------------------------------
# CONFIG instance
# ---------------------------------------------------------------------------

# Hardcoded exclude-layers suffix combining global highprec layers
# and first-block attention/MLP projections that should always remain
# in BF16, as determined by empirical testing.
_EXCLUDE_LAYERS_SUFFIX = (
    r"(time_embedding|adaLN_modulation|final_linear|final_norm|"
    r"x_embedder|layers[.]0[.]self_attention|"
    r"layers[.]0[.]mlp.gate_proj|layers[.]0[.]mlp[.]up_proj|text_proj)"
)


CONFIG = ArchitectureConfig(
    name                      = "ernie",
    description               = "ERNIE Image Turbo image diffusion model",
    convert_flag              = "",
    layer_patterns            = _LAYER_PATTERNS,
    refiner_patterns          = None,
    detail_groups             = _DETAIL_GROUPS,
    refiner_detail_groups     = None,
    default_args              = _DEFAULT_ARGS,
    highprec_excluded         = _HIGHPREC_EXCLUDED,
    exclude_layers_suffix     = _EXCLUDE_LAYERS_SUFFIX,
    # Q/K projections: primary source of quantization error in DiT (QK^T
    # amplifies leptokurtic distributions).
    spread_filter_recommended = ["self_attn.q", "self_attn.k"],
    main_subgraphs            = _MAIN_SUBGRAPHS,
    refiner_subgraphs         = _REFINER_SUBGRAPHS,
    spread_filter_main_only   = False,  # no refiners — field is inert
    infer_subgraph            = infer_subgraph,
    layer_type_to_key         = layer_type_to_key,
    build_block_path          = build_block_path,
)
