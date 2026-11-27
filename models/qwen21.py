"""
Architecture configuration for Qwen-Image 2.1 image diffusion model.

Keys follow the ComfyUI single-file repack (Comfy-Org/Qwen-Image-2.1,
diffusion_models/qwen_image_2.1_bf16.safetensors).  The diffusers original
(Uwen/Qwen-Image-2.1 transformer/) uses `img_mlp.gate_layer` +
`img_mlp.proj` where the ComfyUI repack fuses both into `img_mlp.gate_up`
([2*dim, dim] = [24576, 4096]).  Both namespaces are matched here so the
script also works on an unrepacked transformer.

Architecture: single-stream DiT, 32 blocks, hidden 4096, 32 heads x 128.
Text conditioning is injected once via txt_in and cached as a causal prefix
("mixed-granularity attention" + prefix KV reuse in the model card), so each
block only carries a joint attention sub-block plus an image MLP — there is
no per-block text MLP and no fused qkv: Q/K/V/out are separate 4096x4096
projections with RMSNorm on Q and K (norm_q/norm_k, 1D).

No refiner sub-graphs.

Calibrated defaults are cloned from Z-Image (same DiT lineage, same
single-stream layout) and have NOT been recalibrated for Qwen-Image 2.1.
"""

import re
from typing import Set

from quant_probe.models.base import ArchitectureConfig


# ---------------------------------------------------------------------------
# Layer patterns — transformer blocks (transformer_blocks.0-31)
# ---------------------------------------------------------------------------
# `gate_up` covers the ComfyUI repack (fused SwiGLU gate+up); `gate_layer`
# and `proj` cover the original diffusers naming.  Only one of the two
# shapes is ever present in a given file.

_LAYER_PATTERNS = {
    "attn.q":       re.compile(r"transformer_blocks\.(\d+)\.attn\.to_q\.weight$"),
    "attn.k":       re.compile(r"transformer_blocks\.(\d+)\.attn\.to_k\.weight$"),
    "attn.v":       re.compile(r"transformer_blocks\.(\d+)\.attn\.to_v\.weight$"),
    "attn.o":       re.compile(r"transformer_blocks\.(\d+)\.attn\.to_out\.0\.weight$"),
    "ffn.gate_up":  re.compile(r"transformer_blocks\.(\d+)\.img_mlp\.gate_up\.weight$"),
    "ffn.gate":     re.compile(r"transformer_blocks\.(\d+)\.img_mlp\.gate_layer\.weight$"),
    "ffn.up":       re.compile(r"transformer_blocks\.(\d+)\.img_mlp\.proj\.weight$"),
    "ffn.out":      re.compile(r"transformer_blocks\.(\d+)\.img_mlp\.out\.weight$"),
}

_DETAIL_GROUPS = {
    "attention": ["attn.q", "attn.k", "attn.v", "attn.o"],
    # gate_up (fused, ComfyUI) and gate+up (split, diffusers) describe the
    # same two matrices; a given checkpoint only yields one of them.
    "img_mlp":   ["ffn.gate_up", "ffn.gate", "ffn.up", "ffn.out"],
}

# ---------------------------------------------------------------------------
# Layers excluded from analysis (highprec treatment in convert_to_quant)
# ---------------------------------------------------------------------------
# Mirrors what Comfy-Org keeps in BF16 in their own int8_convrot build of this
# exact checkpoint: every non-block 2D matrix (img_in, txt_in.in_layer,
# txt_in.out_layer, time_text_embed.timestep_embedder.linear_{1,2},
# modulation.1, norm_out.linear, proj_out) plus the 1D block norms
# (attn.norm_q / attn.norm_k / txt_in.text_norm), which are 1D and would be
# skipped by the ndim check anyway.

_HIGHPREC_EXCLUDED = [
    "img_in",
    "txt_in.in_layer",
    "txt_in.out_layer",
    "txt_in.text_norm",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "modulation.1",
    "norm_out.linear",
    "proj_out",
    "attn.norm_q",
    "attn.norm_k",
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

_MAIN_SUBGRAPHS: Set[str]    = {"transformer_blocks"}
_REFINER_SUBGRAPHS: Set[str] = set()

# ---------------------------------------------------------------------------
# Architecture-specific callables
# ---------------------------------------------------------------------------

def infer_subgraph(key: str) -> str:
    """All Qwen-Image 2.1 keys live under transformer_blocks.*."""
    return "transformer_blocks"


_LAYER_TYPE_KEY_MAP = {
    "attn.q":      r"attn\.to_q\.weight",
    "attn.k":      r"attn\.to_k\.weight",
    "attn.v":      r"attn\.to_v\.weight",
    "attn.o":      r"attn\.to_out\.0\.weight",
    "ffn.gate_up": r"img_mlp\.gate_up\.weight",
    "ffn.gate":    r"img_mlp\.gate_layer\.weight",
    "ffn.up":      r"img_mlp\.proj\.weight",
    "ffn.out":     r"img_mlp\.out\.weight",
}


def layer_type_to_key(layer_type: str) -> str:
    """Return the regex key fragment for a given layer type."""
    return _LAYER_TYPE_KEY_MAP.get(layer_type, layer_type)


def build_block_path(subgraph: str, block_alt: str, key_pattern: str) -> str:
    """Build the full regex for one (subgraph, blocks, layer) combination."""
    return rf"transformer_blocks\.({block_alt})\.{key_pattern}"


# ---------------------------------------------------------------------------
# CONFIG instance
# ---------------------------------------------------------------------------

# Hardcoded exclude-layers suffix: every global 2D matrix plus the 1D block
# norms.  This list is self-contained on purpose — convert_to_quant has no
# Qwen-Image 2.1 preset, and its --qwen filter targets Qwen-Image v1 keys
# (`norm_added_k` / `norm_added_q` / `transformer_blocks.0.img_mod.1`) that do
# not exist in 2.1, while 2.1's global 16384x4096 `modulation.1` is not covered
# by it.  Passing --qwen would therefore look correct while silently
# quantizing `modulation.1`.
_EXCLUDE_LAYERS_SUFFIX = (
    r"(img_in|txt_in|time_text_embed|modulation|norm_out|proj_out|"
    r"attn[.]norm_q|attn[.]norm_k)"
)


CONFIG = ArchitectureConfig(
    name                      = "qwen21",
    description               = "Qwen-Image 2.1 image diffusion model (single-stream DiT)",
    # No convert_to_quant preset targets 2.1 — the --qwen filter is v1-only
    # (norm_added_* / img_mod) and misses 2.1's global modulation.1.  The
    # ctq command therefore relies on --exclude-layers alone.
    convert_flag              = "",
    layer_patterns            = _LAYER_PATTERNS,
    refiner_patterns          = None,
    detail_groups             = _DETAIL_GROUPS,
    refiner_detail_groups     = None,
    default_args              = _DEFAULT_ARGS,
    highprec_excluded         = _HIGHPREC_EXCLUDED,
    exclude_layers_suffix     = _EXCLUDE_LAYERS_SUFFIX,
    # Q/K drive QK^T, which amplifies leptokurtic distributions; V and out write
    # into the residual stream and are equally sensitive to a group-level
    # demotion.  In 2.1 Q and K each carry an RMSNorm (norm_q/norm_k) applied
    # *after* the projection, which does not shield the projection from
    # quantization error.
    spread_filter_recommended = ["attn.q", "attn.k", "attn.v", "attn.o"],
    main_subgraphs            = _MAIN_SUBGRAPHS,
    refiner_subgraphs         = _REFINER_SUBGRAPHS,
    spread_filter_main_only   = False,  # no refiners — field is inert
    infer_subgraph            = infer_subgraph,
    layer_type_to_key         = layer_type_to_key,
    build_block_path          = build_block_path,
)
