# quant_probe

A command-line tool to analyze weight tensors in diffusion model checkpoints and recommend a quantization format for each layer — `*KEEP*` (BF16), FP8, or NVFP4 — then outputs a complete `ctq` command with `--nvfp4`, `--save-quant-metadata`, `--comfy_quant`, and model-specific flags. Designed to feed directly into [ComfyUI](https://github.com/comfyanonymous/ComfyUI) quantization workflows.

Supported architectures: **Wan 2.1**, **Wan 2.2**, **Z-Image Turbo**, **ERNIE Image Turbo**, **Qwen-Image 2.1** and **MOSS-Image**.

## Features

- **Per-tensor sensitivity scoring** based on excess kurtosis, dynamic range, and aspect ratio
- **Automatic thresholds** derived from the model's own score distribution — no manual tuning required
- **Kurtosis hard floor** (`--kurtosis-keep`) to protect extremely leptokurtic tensors that score-based thresholds would miss
- **Spread filter** to demote group-level recommendations by one step (FP8 → NVFP4, `*KEEP*` → FP8) when score variation across block positions is too low to be meaningful
- **Spread filter exemptions** (`--spread-filter-exempt`) for layer types where individual tensor decisions should always take precedence over group-level ones
- **Refiner-aware analysis** (Z-Image): `context_refiner` and `noise_refiner` sub-graphs are always resolved tensor-by-tensor — they have too few blocks for positional spread analysis to be meaningful
- **Suggested `convert_to_quant` parameters** printed directly — `--custom-layers` and `--exclude-layers` regexes ready to copy
- **Estimated output file size** after applying recommendations, broken down by format
- **CSV export** with per-tensor metrics, raw recommendation, effective recommendation after all filters, and decision reason
- **`--lowram` mode** for systems where mmap-ing the full model file would exhaust virtual memory
- GPU-accelerated metric computation with automatic CPU fallback

## Requirements

```
torch
safetensors
```

```
pip install torch safetensors
```

## Usage

```bash
# Without installing
python -m quant_probe.cli model.safetensors --model wan21

# After pip install -e .
quant-probe model.safetensors --model zimage
```

### Export metrics to CSV

```bash
python -m quant_probe.cli model.safetensors --model wan21 --csv results.csv
```

### Protect attention K and Q layers from group-level overrides

Cross-attention K and Q projections (Wan) and fused QKV/out projections (Z-Image) are numerically sensitive — the $QK^T$ operation is the primary source of precision loss in quantized attention. Exempting them from the spread filter ensures their quantization format is always decided at individual tensor level:

```bash
# Wan 2.1
quant-probe model.safetensors --model wan21 \
    --spread-filter-exempt cross_attn.k cross_attn.q self_attn.k self_attn.q

# Z-Image
quant-probe model.safetensors --model zimage \
    --spread-filter-exempt attention.qkv attention.out
```

### Adjust quantization thresholds

```bash
# More conservative: protect more layers as FP8 instead of NVFP4
quant-probe model.safetensors --model wan21 --fp8-percentile 60

# Less conservative: protect fewer layers in BF16
quant-probe model.safetensors --model wan21 --keep-percentile 95
```

### Run on CPU or use low-RAM mode

```bash
quant-probe model.safetensors --model wan21 --device cpu

# Low-RAM: load tensors individually instead of mmap-ing the full file
quant-probe model.safetensors --model wan21 --lowram
```

## Compatibility with convert_to_quant

| Model | Flag |
|---|---|
| Wan 2.1 | `--wan` |
| Wan 2.2 | `--wan` |
| Z-Image Turbo | `--zimage` (quantize refiners) or `--zimage_refiner` (keep refiners in BF16) |
| ERNIE Image Turbo | *(none — `--exclude-layers` is self-contained)* |
| Qwen-Image 2.1 | *(none — `--exclude-layers` is self-contained; `--qwen` is v1-only)* |

The `--custom-layers` / `--exclude-layers` output from this script is compatible with both Z-Image flags.

## Scope

### Wan 2.1

| Group | Layers |
|---|---|
| `cross_attn` | `k`, `v`, `q`, `o`, `k_img`, `v_img` |
| `self_attn` | `k`, `v`, `q`, `o` |
| `ffn` | `0`, `2` |

### Wan 2.2

Same `blocks.*` namespace as Wan 2.1, minus the image-conditioned cross-attention projections (`k_img`, `v_img`). Both MoE experts (high-noise / low-noise) share the identical layout, so a single config covers both.

| Group | Layers |
|---|---|
| `cross_attn` | `k`, `v`, `q`, `o` |
| `self_attn` | `k`, `v`, `q`, `o` |
| `ffn` | `0`, `2` |

### Z-Image

Main blocks (`layers.*`):

| Group | Layers |
|---|---|
| `attention` | `qkv`, `out` |
| `feed_forward` | `w1`, `w2`, `w3` |
| `adaLN` | `adaLN_modulation.0` |

Refiners (`context_refiner.*`, `noise_refiner.*`): same layer structure as main blocks, analyzed flat without positional classification.

The following layers are intentionally excluded from analysis because they receive special high-precision treatment in `convert_to_quant` (`--zimage` / `--zimage_refiner`): `x_embedder`, `clip_text_pooled_proj`, `final_layer`, `cap_embedder.1`, `adaLN_modulation` (global), `t_embedder`, `time_text_embed`.

### ERNIE Image Turbo

Blocks (`layers.*`), 36 blocks. No cross-attention and no refiners — text conditioning is injected through `adaLN_modulation` + `text_proj`.

| Group | Layers |
|---|---|
| `self_attention` | `q`, `k`, `v`, `o` |
| `ffn` | `gate`, `up`, `down` |

Excluded from analysis (kept BF16): `x_embedder.proj`, `time_embedding.linear_1` / `linear_2`, `text_proj`, `adaLN_modulation.1`, `final_norm.linear`, `final_linear`, plus the 1D norms (`adaLN_sa_ln`, `adaLN_mlp_ln`, `self_attention.norm_q` / `norm_k`). See [`ernie-image-turbo.md`](ernie-image-turbo.md).

### Qwen-Image 2.1

Blocks (`transformer_blocks.0` … `.31`), single-stream DiT. No refiners.

| Group | Layers |
|---|---|
| `attn` | `q`, `k`, `v`, `o` |
| `img_mlp` | `gate_up` (ComfyUI fused) or `gate` + `up` (diffusers naming), `out` |

Excluded from analysis (kept BF16): the eight global 2D matrices — `img_in`, `txt_in.in_layer`, `txt_in.out_layer`, `time_text_embed.timestep_embedder.linear_1` / `linear_2`, `modulation.1`, `norm_out.linear`, `proj_out` — plus the 1D norms. See [`qwen-image-2.1.md`](qwen-image-2.1.md) for the verified 265-key inventory and why `--qwen` must not be used with it.

### MOSS-Image

Blocks (`layers.0` … `.41`), single-stream DiT with fully fused attention and FFN. The Base and SFT releases share an identical 379-key header, so one config covers both; **SFT is the reference variant** for calibration. No refiners.

| Group | Layers |
|---|---|
| `attention` | `qkv`, `o` |
| `ffn` | `up_gate`, `down` |

Excluded from analysis (kept BF16): `shared_block_modulation` (global adaLN), `text_embedder.proj`, the two `text_embedder.blocks.*` stacks, `time_embedding.0` / `.2`, `video_embedder`, `video_head_modulation`, `final_linear_video`, plus the 1D norms and `rope.bands`. See [`moss-image.md`](moss-image.md) for the verified 379-key inventory and the Comfy-Org int8 baseline.

## How recommendations are assigned

Each tensor receives a sensitivity score combining three metrics:

- **Excess kurtosis** (weight 0.6) — IQR-normalized globally; detects heavy-tailed distributions prone to quantization error
- **Dynamic range** (weight 0.3) — within-type min-max normalized; detects wide value spreads
- **Aspect ratio** (weight 0.1) — within-type normalized; inactive when all tensors of a type share the same shape

Scores are compared against two percentile-derived thresholds:

- Score ≥ `keep-percentile` → `*KEEP*` (BF16)
- Score ≥ `fp8-percentile` and ≥ `fp8-min-score` → FP8
- Otherwise → NVFP4

Two additional guards override the score:

- **Kurtosis hard floor** — tensors with excess kurtosis ≥ `--kurtosis-keep` are forced to `*KEEP*` regardless of score.
- **Spread filter** — when score variation across block position groups (first blocks, middle, last blocks) is below `--min-group-spread`, the group-level recommendation is demoted by one step: FP8 → NVFP4, `*KEEP*` → FP8. A tensor whose own score earned `*KEEP*` is never pushed below BF16 by this demotion. Layer types listed in `--spread-filter-exempt` bypass this filter entirely. For Z-Image, refiner sub-graphs always bypass it.

## CSV output

| Column | Description |
|---|---|
| `key` | Full safetensors key |
| `layer_type` | Layer group (e.g. `cross_attn.k`, `context_refiner.attention.out`) |
| `block_idx` | Transformer block index |
| `subgraph` | Sub-graph identifier (`blocks`, `layers`, `context_refiner`, `noise_refiner`) |
| `rows`, `cols` | Tensor shape |
| `excess_kurtosis` | Corrected excess kurtosis (0 = normal distribution) |
| `dynamic_range` | abs(max) − abs(min) |
| `std` | Standard deviation |
| `outlier_pct` | % of values beyond `outlier-sigma` × std |
| `aspect_ratio` | max(rows, cols) / min(rows, cols) |
| `score` | Combined sensitivity score |
| `recommendation` | Raw recommendation before group-level filters |
| `effective_recommendation` | Final recommendation after all filters |
| `reason` | Decision reason (see below) |

### `reason` values

| Value | Meaning |
|---|---|
| `score_percentile` | `*KEEP*` or FP8 driven by percentile threshold |
| `score_below_fp8_min` | NVFP4 because `fp8-min-score` guard blocked FP8 |
| `default` | NVFP4 because score is below all thresholds |
| `kurtosis_floor` | `*KEEP*` forced because excess kurtosis ≥ `kurtosis-keep` |
| `group_keep_resolved` | Group verdict was `*KEEP*`, but this tensor's own score earned only FP8 or NVFP4, so that is what it kept |
| `group_fp8_promotion` | Tensor was individually NVFP4 but carried up to FP8 by its block-position group |
| `group_keep_preserved` | Tensor's own score earned `*KEEP*`, so neither an FP8 group verdict nor a spread demotion could push it below BF16 |
| `group_spread_demotion` | Tensor was individually FP8 but its block-position group resolved to NVFP4 |
| `spread_demotion` | FP8→NVFP4 or `*KEEP*`→FP8 by the spread filter directly |

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `model_path` | *(required)* | Path to the BF16 safetensors model file |
| `--model` | *(required)* | Architecture: `wan21`, `wan22`, `zimage`, `ernie`, `qwen21` or `moss` |
| `--csv` | — | Export per-tensor metrics to a CSV file |
| `--device` | auto | Computation device: `cpu` or `cuda` |
| `--lowram` | off | Avoid full-file mmap; load tensors individually by byte offset |
| `--base-format` | `nvfp4` | Format of the least-sensitive tier: `nvfp4`, `mxfp8`, `int8` or `fp8` (no flag). Emitted as the main `ctq` flag |
| `--sensitive-format` | derived | Format of the sensitive tier, emitted as `--custom-type`. When unset it is FP8 if the base is `nvfp4`, otherwise the base format itself — so `--base-format mxfp8` alone gives a uniform MXFP8 build with no `--custom-layers` block |
| `--fp8-percentile` | model-specific | Score percentile threshold for FP8 recommendation |
| `--keep-percentile` | 90 | Score percentile threshold for `*KEEP*` recommendation |
| `--fp8-min-score` | model-specific | Absolute minimum score required for FP8; set to 0.0 to disable |
| `--kurtosis-keep` | 8.0 | Excess kurtosis hard floor for forced `*KEEP*`; set to `inf` to disable |
| `--min-group-spread` | model-specific | Minimum score spread across block position groups for per-group FP8 |
| `--spread-filter-exempt` | — | Layer types that bypass the spread filter |
| `--extreme-pct` | 10 | Percentage of blocks considered extreme at each end |
| `--outlier-sigma` | 3.0 | Standard deviation multiplier for outlier detection |
| `--kurtosis-weight` | 0.6 | Score weight for excess kurtosis |
| `--range-weight` | 0.3 | Score weight for dynamic range |
| `--ar-weight` | 0.1 | Score weight for aspect ratio |

### Model-specific defaults

| Parameter | Wan 2.1 | Wan 2.2 | Z-Image | ERNIE | Qwen-Image 2.1 | MOSS-Image |
|---|---|---|---|---|---|---|
| `fp8-percentile` | 75.0 | 50.0* | 50.0 | 50.0* | 50.0* | 50.0 |
| `keep-percentile` | 90.0 | 90.0* | 90.0 | 90.0* | 90.0* | 90.0 |
| `kurtosis-keep` | 8.0 | 8.0* | 8.0 | 8.0* | 8.0* | 8.0 |
| `fp8-min-score` | 0.50 | 0.50* | 0.0 | 0.0* | 0.0* | 0.0 |
| `min-group-spread` | 0.06 | 0.06* | 0.20 | 0.20* | 0.20* | 0.20 |

\* Wan 2.2 fp8_percentile set to 50 (empirical); other defaults cloned from Wan 2.1 — not yet recalibrated.
\* ERNIE and Qwen-Image 2.1 defaults are cloned from Z-Image — not yet recalibrated.

MOSS-Image defaults are numerically identical to Z-Image's, but they were not
inherited blindly: each one was checked against the SFT release's score
distribution and kept (no valley to cut on, a spread floor that never fires, a
kurtosis floor that catches exactly the three extreme tensors). The evidence is
in `doc/moss-image.md` §7.

See `doc/qwen-image-2.1.md` and `doc/moss-image.md` for the verified layer
inventories of the Comfy-Org BF16 repacks, and why `--qwen` must not be used
with Qwen-Image 2.1 (no MOSS-Image preset exists in convert_to_quant at all).

## Adding a new architecture

1. Create `quant_probe/models/<n>.py` exporting a `CONFIG` instance of `ArchitectureConfig` (see `models/base.py` for the full contract).
2. Add one import and one entry in `quant_probe/registry.py`.
