"""
Architecture configuration for Z-Image / Z-Image Turbo image diffusion model.

Main transformer blocks use the layers.* key namespace (optionally prefixed
with model.diffusion_model.).  Attention Q/K/V are fused into a single
attention.qkv tensor.

Refiner sub-graphs (context_refiner, noise_refiner) share the same layer
structure as main blocks but use a separate key namespace and are handled
without positional classification.

Calibrated defaults are sourced from empirical analysis of Z-Image Turbo.
"""

import re
from typing import Set

from quant_probe.models.base import ArchitectureConfig


# ---------------------------------------------------------------------------
# Layer patterns — main blocks (layers.0-N)
# ---------------------------------------------------------------------------
# The (?:model\.diffusion_model\.)? prefix handles both Z-Image base and Turbo,
# which differ only in whether this prefix is present in their keys.

_LAYER_PATTERNS = {
    "attention.qkv":      re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.attention\.qkv\.weight$"),
    "attention.out":      re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.attention\.out\.weight$"),
    "feed_forward.w1":    re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.feed_forward\.w1\.weight$"),
    "feed_forward.w2":    re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.feed_forward\.w2\.weight$"),
    "feed_forward.w3":    re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.feed_forward\.w3\.weight$"),
    "adaLN_modulation.0": re.compile(r"(?:model\.diffusion_model\.)?layers\.(\d+)\.adaLN_modulation\.0\.weight$"),
}

_DETAIL_GROUPS = {
    "attention":    ["attention.qkv", "attention.out"],
    "feed_forward": ["feed_forward.w1", "feed_forward.w2", "feed_forward.w3"],
    "adaLN":        ["adaLN_modulation.0"],
}

# ---------------------------------------------------------------------------
# Layer patterns — refiners (context_refiner.0-1, noise_refiner.0-1)
# ---------------------------------------------------------------------------
# Each subgraph has its own set of patterns so that layer_type reflects the
# real key namespace (context_refiner.* / noise_refiner.*) rather than a
# synthetic "refiner." prefix.  Group 1 captures the block index directly,
# consistent with main block patterns.

_REFINER_PATTERNS = {
    "context_refiner.attention.qkv":   re.compile(r"(?:model\.diffusion_model\.)?context_refiner\.(\d+)\.attention\.qkv\.weight$"),
    "context_refiner.attention.out":   re.compile(r"(?:model\.diffusion_model\.)?context_refiner\.(\d+)\.attention\.out\.weight$"),
    "context_refiner.feed_forward.w1": re.compile(r"(?:model\.diffusion_model\.)?context_refiner\.(\d+)\.feed_forward\.w1\.weight$"),
    "context_refiner.feed_forward.w2": re.compile(r"(?:model\.diffusion_model\.)?context_refiner\.(\d+)\.feed_forward\.w2\.weight$"),
    "context_refiner.feed_forward.w3": re.compile(r"(?:model\.diffusion_model\.)?context_refiner\.(\d+)\.feed_forward\.w3\.weight$"),
    "noise_refiner.attention.qkv":     re.compile(r"(?:model\.diffusion_model\.)?noise_refiner\.(\d+)\.attention\.qkv\.weight$"),
    "noise_refiner.attention.out":     re.compile(r"(?:model\.diffusion_model\.)?noise_refiner\.(\d+)\.attention\.out\.weight$"),
    "noise_refiner.feed_forward.w1":   re.compile(r"(?:model\.diffusion_model\.)?noise_refiner\.(\d+)\.feed_forward\.w1\.weight$"),
    "noise_refiner.feed_forward.w2":   re.compile(r"(?:model\.diffusion_model\.)?noise_refiner\.(\d+)\.feed_forward\.w2\.weight$"),
    "noise_refiner.feed_forward.w3":   re.compile(r"(?:model\.diffusion_model\.)?noise_refiner\.(\d+)\.feed_forward\.w3\.weight$"),
}

_REFINER_DETAIL_GROUPS = {
    "context_refiner.attention":    ["context_refiner.attention.qkv", "context_refiner.attention.out"],
    "context_refiner.feed_forward": ["context_refiner.feed_forward.w1", "context_refiner.feed_forward.w2", "context_refiner.feed_forward.w3"],
    "noise_refiner.attention":      ["noise_refiner.attention.qkv", "noise_refiner.attention.out"],
    "noise_refiner.feed_forward":   ["noise_refiner.feed_forward.w1", "noise_refiner.feed_forward.w2", "noise_refiner.feed_forward.w3"],
}

# ---------------------------------------------------------------------------
# Layers excluded from analysis (highprec treatment in convert_to_quant)
# ---------------------------------------------------------------------------
# These correspond to ZIMAGE_LAYER_KEYNAMES in convert_to_quant and reflect an
# informed architectural decision by the model author.  The script notifies the
# user about their exclusion but does not analyse them.

_HIGHPREC_EXCLUDED = [
    "x_embedder",
    "clip_text_pooled_proj",
    "final_layer",
    "cap_embedder.1",
    "adaLN_modulation",   # global (not layers.*)
    "t_embedder",
    "time_text_embed",
]

# ---------------------------------------------------------------------------
# Calibrated defaults (empirical — Z-Image Turbo)
# ---------------------------------------------------------------------------

_DEFAULT_ARGS = {
    "fp8_percentile":   75.0,
    "keep_percentile":  90.0,
    "kurtosis_keep":     8.0,
    # 0.0: the natural separation at ~0.525 is already captured by fp8_percentile;
    # an artificial floor suppressed valid signal.
    "fp8_min_score":    0.0,
    # 0.20: inter-group positional spread is genuinely low in Z-Image;
    # the Wan default of 0.06 filtered real signal.
    "min_group_spread": 0.20,
}

# ---------------------------------------------------------------------------
# Subgraph sets
# ---------------------------------------------------------------------------

_MAIN_SUBGRAPHS: Set[str]    = {"layers"}
_REFINER_SUBGRAPHS: Set[str] = {"context_refiner", "noise_refiner"}

# ---------------------------------------------------------------------------
# Architecture-specific callables
# ---------------------------------------------------------------------------

def infer_subgraph(key: str) -> str:
    """
    Infer the subgraph identifier from a Z-Image safetensors key.

    Returns "context_refiner", "noise_refiner", or "layers".
    """
    if "context_refiner" in key:
        return "context_refiner"
    if "noise_refiner" in key:
        return "noise_refiner"
    return "layers"


_LAYER_TYPE_KEY_MAP = {
    "attention.qkv":      r"attention\.qkv\.weight",
    "attention.out":      r"attention\.out\.weight",
    "feed_forward.w1":    r"feed_forward\.w1\.weight",
    "feed_forward.w2":    r"feed_forward\.w2\.weight",
    "feed_forward.w3":    r"feed_forward\.w3\.weight",
    "adaLN_modulation.0": r"adaLN_modulation\.0\.weight",
    # Refiner layer types — key suffix is identical to main blocks;
    # the subgraph prefix is handled by build_block_path.
    "context_refiner.attention.qkv":   r"attention\.qkv\.weight",
    "context_refiner.attention.out":   r"attention\.out\.weight",
    "context_refiner.feed_forward.w1": r"feed_forward\.w1\.weight",
    "context_refiner.feed_forward.w2": r"feed_forward\.w2\.weight",
    "context_refiner.feed_forward.w3": r"feed_forward\.w3\.weight",
    "noise_refiner.attention.qkv":     r"attention\.qkv\.weight",
    "noise_refiner.attention.out":     r"attention\.out\.weight",
    "noise_refiner.feed_forward.w1":   r"feed_forward\.w1\.weight",
    "noise_refiner.feed_forward.w2":   r"feed_forward\.w2\.weight",
    "noise_refiner.feed_forward.w3":   r"feed_forward\.w3\.weight",
}


def layer_type_to_key(layer_type: str) -> str:
    """Return the regex key fragment for a given layer type."""
    return _LAYER_TYPE_KEY_MAP.get(layer_type, layer_type)


def build_block_path(subgraph: str, block_alt: str, key_pattern: str) -> str:
    """
    Build the full regex for one (subgraph, blocks, layer) combination.

    Refiner keys use the subgraph name directly as the namespace prefix;
    main block keys use 'layers'.
    """
    if subgraph in ("context_refiner", "noise_refiner"):
        return rf"{subgraph}\.({block_alt})\.{key_pattern}"
    return rf"layers\.({block_alt})\.{key_pattern}"


# ---------------------------------------------------------------------------
# CONFIG instance
# ---------------------------------------------------------------------------

CONFIG = ArchitectureConfig(
    name                      = "zimage",
    description               = "Z-Image / Z-Image Turbo image diffusion model",
    convert_flag              = "--zimage / --zimage_refiner",
    layer_patterns            = _LAYER_PATTERNS,
    refiner_patterns          = _REFINER_PATTERNS,
    detail_groups             = _DETAIL_GROUPS,
    refiner_detail_groups     = _REFINER_DETAIL_GROUPS,
    default_args              = _DEFAULT_ARGS,
    highprec_excluded         = _HIGHPREC_EXCLUDED,
    # Q/K/V are fused into attention.qkv — the full tensor must be treated as
    # sensitive.  attention.out projects back to the residual stream and is
    # equally sensitive to quantization error in DiT architectures.
    spread_filter_recommended = ["attention.qkv", "attention.out"],
    main_subgraphs            = _MAIN_SUBGRAPHS,
    refiner_subgraphs         = _REFINER_SUBGRAPHS,
    infer_subgraph            = infer_subgraph,
    layer_type_to_key         = layer_type_to_key,
    build_block_path          = build_block_path,
)
