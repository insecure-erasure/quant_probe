# quant_probe

Quantization sensitivity analysis for safetensors diffusion models. Analyzes weight tensors to recommend per-layer quantization format (BF16 / FP8 / NVFP4) and generates ready-to-use parameters for [convert_to_quant](https://github.com/silveroxides/convert_to_quant).

Supported architectures: **Wan 2.1** and **Z-Image Turbo**.

## How it works

For each target layer, the script computes:

- **Excess kurtosis** — heavy-tailed distributions are more sensitive to quantization
- **Dynamic range** — wider range means harder to represent at low precision
- **Aspect ratio** — shape-based proxy for quantization difficulty

These metrics are combined into a score and bucketed into three recommendations: `*KEEP*` (BF16), `FP8`, or `NVFP4`. Thresholds are derived automatically from the model's own score distribution using configurable percentiles.

A **spread filter** suppresses per-group positional variance when the score spread across block positions is too low to be meaningful. For Z-Image, refiner sub-graphs (`context_refiner`, `noise_refiner`) bypass this filter entirely — they only have 2 blocks, which is insufficient for positional spread analysis.

## Usage

```bash
# Without installing
python -m quant_probe.cli model.safetensors --model zimage \
  --spread-filter-exempt attention.qkv attention.out

# After pip install -e .
quant-probe model.safetensors --model wan \
  --spread-filter-exempt cross_attn.k cross_attn.q self_attn.k self_attn.q
```

The output includes `--custom-layers` and `--exclude-layers` parameters ready to pass to `convert_to_quant`.

## Key options

| Option | Default | Description |
|---|---|---|
| `--model` | required | Architecture: `wan` or `zimage` |
| `--fp8-percentile` | model-specific | Score percentile threshold for FP8 |
| `--keep-percentile` | 90.0 | Score percentile threshold for \*KEEP\* |
| `--spread-filter-exempt` | none | Layer types that bypass the spread filter |
| `--kurtosis-keep` | 8.0 | Kurtosis hard floor — forces \*KEEP\* |
| `--csv` | none | Export per-tensor metrics to CSV |
| `--lowram` | false | Avoid mmap-ing the full model file |
| `--device` | auto | `cpu` or `cuda` |

## Compatibility with convert_to_quant

| Model | Flag |
|---|---|
| Wan 2.1 | `--wan` |
| Z-Image base / Turbo | `--zimage` (quantize refiners) or `--zimage_refiner` (keep refiners in BF16) |

The `--exclude-layers` output from this script is compatible with both Z-Image flags.

## Adding a new architecture

1. Create `quant_probe/models/<name>.py` exporting a `CONFIG` instance of `ArchitectureConfig` (see `models/base.py` for the contract).
2. Add one import and one entry in `quant_probe/registry.py`.

## Requirements

- Python 3.10+
- `torch`
- `safetensors`
