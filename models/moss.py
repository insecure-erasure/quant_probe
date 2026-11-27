"""
Architecture configuration for the MOSS-Image diffusion model family.

Keys follow the ComfyUI single-file repack (Comfy-Org/MOSS-Image,
diffusion_models/moss_image_{base,sft}_bf16.safetensors).  The two releases are
separate upstream repos (OpenMOSS-Team/MOSS-Image-Base and
OpenMOSS-Team/MOSS-Image-SFT), but their headers are identical -- 379 tensors,
185 2D, same keys and same shapes -- so a single config covers both.  The SFT
release is the reference variant for calibration; it is a finetune of the base,
not a different architecture.

Layout: 42 blocks, hidden 3072, 24 heads x 128.  Each block carries a fused
QKV projection ([9216, 3072]), an output projection, a fused SwiGLU up+gate
([16384, 3072]) and a down projection ([3072, 8192]) -- four quantizable
matrices per block, 168 in total.  Q and K have their own RMSNorm (q_norm /
k_norm, 1D) and there is no separate K/V, so a --spread-filter-exempt entry can
only target attn.qkv / attn.o (as with Z-Image's fused attention.qkv).

Outside the blocks: a global adaLN head (shared_block_modulation, 18432x3072),
a two-block text_embedder stack, time_embedding.0/2, video_embedder,
video_head_modulation and final_linear_video.  All of them are 2D but are kept
in BF16 -- see _HIGHPREC_EXCLUDED.

No refiner sub-graphs of the Z-Image kind: the two text_embedder.blocks are
treated as high-precision globals, matching what Comfy-Org keeps in BF16 in
their own int8_convrot build of this checkpoint.

Calibrated defaults are cloned from Z-Image (same single-stream DiT lineage)
and have NOT been recalibrated for MOSS-Image.
"""

import re
from typing import Set

from quant_probe.models.base import ArchitectureConfig


# ---------------------------------------------------------------------------
# Layer patterns — transformer blocks (layers.0-41)
# ---------------------------------------------------------------------------
# Fully fused, single-stream block: one QKV projection and one up+gate
# projection.  Matching `layers.N.proj.*` also keeps the two
# `text_embedder.blocks.N` stacks out, since they use a different namespace
# (`text_embedder.blocks.N.qkv` / `.ffn_gate` / `.ffn_up`, no `proj.`).

_LAYER_PATTERNS = {
    "attn.qkv":    re.compile(r"layers\.(\d+)\.proj\.qkv\.weight$"),
    "attn.o":      re.compile(r"layers\.(\d+)\.proj\.o\.weight$"),
    "ffn.up_gate": re.compile(r"layers\.(\d+)\.proj\.ffn_up_gate\.weight$"),
    "ffn.down":    re.compile(r"layers\.(\d+)\.proj\.ffn_down\.weight$"),
}

_DETAIL_GROUPS = {
    "attention": ["attn.qkv", "attn.o"],
    # up+gate is fused (ComfyUI naming); there is no separate up projection.
    "ffn":       ["ffn.up_gate", "ffn.down"],
}

# ---------------------------------------------------------------------------
# Layers excluded from analysis (highprec treatment in convert_to_quant)
# ---------------------------------------------------------------------------
# Mirrors what Comfy-Org keeps in BF16 in their own int8_convrot build of this
# exact checkpoint: all 17 non-block 2D matrices -- shared_block_modulation,
# time_embedding.0/2, the two text_embedder blocks (5 matrices each),
# text_embedder.proj, video_embedder, video_head_modulation and
# final_linear_video -- plus the 1D tensors (norm1, norm2, q_norm, k_norm,
# final_norm_video, rope.bands), which the ndim check would skip anyway.

_HIGHPREC_EXCLUDED = [
    "shared_block_modulation",
    "time_embedding.0",
    "time_embedding.2",
    "text_embedder.proj",
    "text_embedder.blocks.0",
    "text_embedder.blocks.1",
    "video_embedder",
    "video_head_modulation",
    "final_linear_video",
    # 1D — not matched by any layer pattern, listed for completeness
    "final_norm_video",
    "layers.N.norm1 / norm2",
    "layers.N.q_norm / k_norm",
    "rope.bands",
]

# ---------------------------------------------------------------------------
# Calibrated defaults (cloned from Z-Image — not yet recalibrated)
# ---------------------------------------------------------------------------

_DEFAULT_ARGS = {
    "fp8_percentile":   50.0,
    "keep_percentile":  90.0,
    "kurtosis_keep":     8.0,
    "fp8_min_score":     0.0,
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
    """All MOSS-Image block keys live under layers.*."""
    return "layers"


_LAYER_TYPE_KEY_MAP = {
    "attn.qkv":    r"proj\.qkv\.weight",
    "attn.o":      r"proj\.o\.weight",
    "ffn.up_gate": r"proj\.ffn_up_gate\.weight",
    "ffn.down":    r"proj\.ffn_down\.weight",
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

# Hardcoded exclude-layers suffix: the 17 non-block 2D matrices plus the 1D
# block norms.  Self-contained on purpose -- convert_to_quant has no MOSS-Image
# preset (nothing matches "moss" on its `main` branch), so there is no model
# flag to lean on.  The leading `[.]` on the norm patterns keeps them anchored
# to a key segment boundary, so they cannot swallow a projection name.
_EXCLUDE_LAYERS_SUFFIX = (
    r"(text_embedder|time_embedding|video_embedder|video_head_modulation|"
    r"shared_block_modulation|final_linear|final_norm|rope[.]bands|"
    r"[.]norm1|[.]norm2|[.]q_norm|[.]k_norm)"
)


CONFIG = ArchitectureConfig(
    name                      = "moss",
    description               = "MOSS-Image diffusion model (base / SFT, single-stream DiT)",
    # No convert_to_quant preset targets MOSS-Image; the ctq command relies on
    # --exclude-layers alone, like ernie and qwen21.
    convert_flag              = "",
    layer_patterns            = _LAYER_PATTERNS,
    refiner_patterns          = None,
    detail_groups             = _DETAIL_GROUPS,
    refiner_detail_groups     = None,
    default_args              = _DEFAULT_ARGS,
    highprec_excluded         = _HIGHPREC_EXCLUDED,
    exclude_layers_suffix     = _EXCLUDE_LAYERS_SUFFIX,
    # Q/K/V are fused into proj.qkv, so the whole tensor must be decided
    # individually.  proj.o writes back into the residual stream and is equally
    # sensitive to a group-level demotion.  q_norm/k_norm are applied *after*
    # the projection, so the norm does not shield it from quantization error.
    spread_filter_recommended = ["attn.qkv", "attn.o"],
    main_subgraphs            = _MAIN_SUBGRAPHS,
    refiner_subgraphs         = _REFINER_SUBGRAPHS,
    spread_filter_main_only   = False,  # no refiners — field is inert
    infer_subgraph            = infer_subgraph,
    layer_type_to_key         = layer_type_to_key,
    build_block_path          = build_block_path,
)
