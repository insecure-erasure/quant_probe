# quant_probe

A command-line tool to analyze weight tensors in diffusion model checkpoints and recommend a quantization format for each layer — `*KEEP*` (BF16), FP8, or NVFP4 — before running `convert_to_quant`. Designed to feed directly into [ComfyUI](https://github.com/comfyanonymous/ComfyUI) quantization workflows.

Supported architectures: **Wan 2.1** and **Z-Image / Z-Image Turbo**.

## Features

- **Per-tensor sensitivity scoring** based on excess kurtosis, dynamic range, and aspect ratio
- **Automatic thresholds** derived from the model's own score distribution — no manual tuning required
- **Kurtosis hard floor** (`--kurtosis-keep`) to protect extremely leptokurtic tensors that score-based thresholds would miss
- **Spread filter** to suppress per-group FP8 recommendations when score variation across block positions is too low to be meaningful
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
python -m quant_probe.cli model.safetensors --model wan

# After pip install -e .
quant-probe model.safetensors --model zimage
```

### Export metrics to CSV

```bash
python -m quant_probe.cli model.safetensors --model wan --csv results.csv
```

### Protect attention K and Q layers from group-level overrides

Cross-attention K and Q projections (Wan) and fused QKV/out projections (Z-Image) are numerically sensitive — the $QK^T$ operation is the primary source of precision loss in quantized attention. Exempting them from the spread filter ensures their quantization format is always decided at individual tensor level:

```bash
# Wan 2.1
quant-probe model.safetensors --model wan \
    --spread-filter-exempt cross_attn.k cross_attn.q self_attn.k self_attn.q

# Z-Image
quant-probe model.safetensors --model zimage \
    --spread-filter-exempt attention.qkv attention.out
```

### Adjust quantization thresholds

```bash
# More conservative: protect more layers as FP8 instead of NVFP4
quant-probe model.safetensors --model wan --fp8-percentile 60

# Less conservative: protect fewer layers in BF16
quant-probe model.safetensors --model wan --keep-percentile 95
```

### Run on CPU or use low-RAM mode

```bash
quant-probe model.safetensors --model wan --device cpu

# Low-RAM: load tensors individually instead of mmap-ing the full file
quant-probe model.safetensors --model wan --lowram
```

## Compatibility with convert_to_quant

| Model | Flag |
|---|---|
| Wan 2.1 | `--wan` |
| Z-Image base / Turbo | `--zimage` (quantize refiners) or `--zimage_refiner` (keep refiners in BF16) |

The `--custom-layers` / `--exclude-layers` output from this script is compatible with both Z-Image flags.

## Scope

### Wan 2.1

| Group | Layers |
|---|---|
| `cross_attn` | `k`, `v`, `q`, `o`, `k_img`, `v_img` |
| `self_attn` | `k`, `v`, `q`, `o` |
| `ffn` | `0`, `2` |

### Z-Image / Z-Image Turbo

Main blocks (`layers.*`):

| Group | Layers |
|---|---|
| `attention` | `qkv`, `out` |
| `feed_forward` | `w1`, `w2`, `w3` |
| `adaLN` | `adaLN_modulation.0` |

Refiners (`context_refiner.*`, `noise_refiner.*`): same layer structure as main blocks, analyzed flat without positional classification.

The following layers are intentionally excluded from analysis because they receive special high-precision treatment in `convert_to_quant` (`--zimage` / `--zimage_refiner`): `x_embedder`, `clip_text_pooled_proj`, `final_layer`, `cap_embedder.1`, `adaLN_modulation` (global), `t_embedder`, `time_text_embed`.

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
- **Spread filter** — when score variation across block position groups (first blocks, middle, last blocks) is below `--min-group-spread`, per-group recommendations are suppressed and all tensors of that type fall back to NVFP4. Layer types listed in `--spread-filter-exempt` bypass this filter entirely. For Z-Image, refiner sub-graphs always bypass it.

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
| `group_keep_resolved` | Tensor was individually `*KEEP*` but demoted during per-tensor resolution within a `*KEEP*` group |
| `group_fp8_promotion` | Tensor was individually NVFP4 but carried up to FP8 by its block-position group |
| `group_spread_demotion` | Tensor was individually FP8 or `*KEEP*` but brought down by the group's spread filter result |
| `spread_demotion` | FP8→NVFP4 or `*KEEP*`→FP8 by spread filter directly |

## CLI reference

| Argument | Default | Description |
|---|---|---|
| `model_path` | *(required)* | Path to the BF16 safetensors model file |
| `--model` | *(required)* | Architecture: `wan` or `zimage` |
| `--csv` | — | Export per-tensor metrics to a CSV file |
| `--device` | auto | Computation device: `cpu` or `cuda` |
| `--lowram` | off | Avoid full-file mmap; load tensors individually by byte offset |
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

| Parameter | Wan 2.1 | Z-Image Turbo |
|---|---|---|
| `fp8-percentile` | 75.0 | 75.0 |
| `keep-percentile` | 90.0 | 90.0 |
| `kurtosis-keep` | 8.0 | 8.0 |
| `fp8-min-score` | 0.50 | 0.0 |
| `min-group-spread` | 0.06 | 0.20 |

## Adding a new architecture

1. Create `quant_probe/models/<n>.py` exporting a `CONFIG` instance of `ArchitectureConfig` (see `models/base.py` for the full contract).
2. Add one import and one entry in `quant_probe/registry.py`.
