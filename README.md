# quant_probe

Quantization sensitivity analysis for safetensors diffusion models. Analyzes weight tensors to recommend a per-layer quantization format and generates ready-to-use parameters for [convert_to_quant](https://github.com/silveroxides/convert_to_quant).

Supported architectures: **Wan 2.1**, **Wan 2.2**, **Z-Image Turbo**, **ERNIE Image Turbo**, **Qwen-Image 2.1** and **MOSS-Image**.

## How it works

For each target layer, the script computes:

- **Excess kurtosis** — heavy-tailed distributions are more sensitive to quantization
- **Dynamic range** — wider range means harder to represent at low precision
- **Aspect ratio** — shape-based proxy for quantization difficulty

These metrics are combined into a score and bucketed into three tiers: `*KEEP*` (BF16), a **sensitive** tier, and a **base** tier for everything else. Thresholds are derived automatically from the model's own score distribution using configurable percentiles.

The two quantized tiers are mapped onto concrete formats by `--base-format` and `--sensitive-format`. The default is NVFP4 base with FP8 for the sensitive tier; MXFP8, INT8 and plain FP8 are also available, and a base that is already 8-bit collapses the build to a single uniform format.

A **spread filter** suppresses per-group positional variance when the score spread across block positions is too low to be meaningful. For Z-Image, refiner sub-graphs (`context_refiner`, `noise_refiner`) bypass this filter entirely — they only have 2 blocks, which is insufficient for positional spread analysis.

## Usage

```bash
# Without installing
python -m quant_probe.cli model.safetensors --model zimage \
  --spread-filter-exempt attention.qkv attention.out

# After pip install -e .
quant-probe model.safetensors --model wan21 \
  --spread-filter-exempt cross_attn.k cross_attn.q self_attn.k self_attn.q
```

The output includes a ready-to-use `ctq` command with `--custom-layers` and `--exclude-layers` parameters, plus metadata and ComfyUI flags.

## Key options

| Option | Default | Description |
|---|---|---|
| `--model` | required | Architecture: `wan21`, `wan22`, `zimage`, `ernie`, `qwen21` or `moss` |
| `--base-format` | `nvfp4` | Format for the least-sensitive tier: `nvfp4`, `mxfp8`, `int8` or `fp8` (emits no flag — FP8 is `ctq`'s default) |
| `--sensitive-format` | derived | Format for the sensitive tier, emitted as `--custom-type`. Unset = FP8 when the base is `nvfp4`, otherwise the base itself (uniform build, no `--custom-layers` block) |
| `--fp8-percentile` | model-specific | Score percentile threshold for FP8 |
| `--keep-percentile` | 90.0 | Score percentile threshold for \*KEEP\* |
| `--spread-filter-exempt` | none | Layer types that bypass the spread filter |
| `--kurtosis-keep` | 8.0 | Kurtosis hard floor — forces \*KEEP\* |
| `--csv` | none | Export per-tensor metrics to CSV |
| `--lowram` | false | Avoid mmap-ing the full model file |
| `--device` | auto | `cpu` or `cuda` |

NVFP4 and MXFP8 require Blackwell GPUs (SM ≥ 10.0, CUDA 13.0+, PyTorch 2.10+); on Ada/Hopper use `--base-format int8` or `--base-format fp8`. The full CLI reference is in [`doc/README.md`](doc/README.md), with per-model layer inventories in [`doc/`](doc/).

## Compatibility with convert_to_quant

| Model | Flag |
|---|---|
| Wan 2.1 | `--wan` |
| Wan 2.2 | `--wan` |
| Z-Image base / Turbo | `--zimage` (quantize refiners) or `--zimage_refiner` (keep refiners in BF16) |
| ERNIE Image Turbo | *(none — `--exclude-layers` is self-contained)* |
| Qwen-Image 2.1 | *(none — `--exclude-layers` is self-contained; `--qwen` is v1-only)* |

The `--exclude-layers` output from this script is compatible with both Z-Image flags.

## Adding a new architecture

1. Create `quant_probe/models/<name>.py` exporting a `CONFIG` instance of `ArchitectureConfig` (see `models/base.py` for the contract).
2. Add one import and one entry in `quant_probe/registry.py`.

## Requirements

- Python 3.10+
- `torch`
- `safetensors`
