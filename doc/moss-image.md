# MOSS-Image — Layer Layout

**Files analysed (Comfy-Org repack, the ones tested with ComfyUI):**

| File | Size | Tensors |
|---|---|---|
| `diffusion_models/moss_image_base_bf16.safetensors` | 10.16 GB | 379 |
| `diffusion_models/moss_image_sft_bf16.safetensors` | 10.16 GB | 379 (identical key set + shapes) |

Header read remotely from Hugging Face (HTTP range request, no download).
Source repos: `OpenMOSS-Team/MOSS-Image-Base`, `OpenMOSS-Team/MOSS-Image-SFT`.

**Base vs SFT:** two separate upstream repos, but the headers are identical —
379 tensors, 185 2D + 194 1D, same keys, same shapes, same 10.16 GB payload.
The SFT release is a finetune of the base, not a different architecture, so a
single `--model moss` config covers both. **SFT is the reference variant**: it
is the one to calibrate against and the one ComfyUI ships a text-to-image
workflow for.

**Architecture:** single-stream DiT, **42 blocks**, hidden 3072, **24 heads ×
128**, FFN inner 8192, ~5.08B params in the diffusion transformer. Attention is
**fully fused** — one `proj.qkv` (`[9216, 3072]` = 3 × 3072) and one
`proj.ffn_up_gate` (`[16384, 3072]` = 2 × 8192) — so there is no separate K, V,
gate or up projection to map onto a layer type.

---

## 1. Global layers (outside transformer blocks) — 17 2D + 1D

| Layer | Shape | Params |
|---|---|---|
| `shared_block_modulation.weight` | `[18432, 3072]` | 56.62M |
| `text_embedder.blocks.0.qkv.weight` | `[9216, 3072]` | 28.31M |
| `text_embedder.blocks.0.proj.weight` | `[3072, 3072]` | 9.44M |
| `text_embedder.blocks.0.ffn_gate.weight` | `[8192, 3072]` | 25.17M |
| `text_embedder.blocks.0.ffn_up.weight` | `[8192, 3072]` | 25.17M |
| `text_embedder.blocks.0.ffn_down.weight` | `[3072, 8192]` | 25.17M |
| `text_embedder.blocks.1.*` | *(same as `.0`)* | 113.25M |
| `text_embedder.proj.weight` | `[3072, 4096]` | 12.58M |
| `time_embedding.0.weight` | `[3072, 256]` | 0.79M |
| `time_embedding.2.weight` | `[3072, 3072]` | 9.44M |
| `video_embedder.weight` | `[3072, 128]` | 0.39M |
| `video_head_modulation.weight` | `[6144, 3072]` | 18.87M |
| `final_linear_video.weight` | `[128, 3072]` | 0.39M |

> `shared_block_modulation` (18432 = 6 × hidden) is a **global** adaLN head used
> by every block, the exact analogue of `modulation.1` in Qwen-Image 2.1. It is
> the single most sensitive global matrix (56.6M) and must stay BF16.
>
> The `text_embedder.blocks.*` stack is only **2 blocks** — the same situation
> as Z-Image's refiners, where positional spread analysis is not meaningful.

1D tensors (194): `norm1` / `norm2` (`[3072]`) and `q_norm` / `k_norm`
(`[128]`) per block, `final_norm_video`, `rope.bands` (F32), plus the biases and
norms of the global layers above.

## 2. Transformer blocks (`layers.0` … `layers.41`) — 336 tensors

| Layer | Shape | Params |
|---|---|---|
| `proj.qkv.weight` | `[9216, 3072]` | 28.31M |
| `proj.o.weight` | `[3072, 3072]` | 9.44M |
| `proj.ffn_up_gate.weight` | `[16384, 3072]` | 50.33M |
| `proj.ffn_down.weight` | `[3072, 8192]` | 25.17M |
| `norm1.weight` | `[3072]` | 1D |
| `norm2.weight` | `[3072]` | 1D |
| `q_norm.weight` | `[128]` | 1D |
| `k_norm.weight` | `[128]` | 1D |

4 quantizable matrices × 42 blocks = **168 tensors**. There are no biases inside
`layers.*` (only in the `text_embedder` blocks).

The per-block count (113.25M) coincides with a `text_embedder` block only
because a fused `up+gate` equals a split `gate` + `up`; the namespaces differ and
`layers.N.proj.*` cannot match `text_embedder.blocks.N.*`.

## 3. Parameter summary

| Category | Params | % |
|---|---|---|
| Blocks (168 quantizable 2D) | 4.756B | 93.6% |
| Global (17 × 2D, kept BF16) | 0.326B | 6.4% |
| 1D tensors (194) | ~0% | ~0% |
| **Total** | **~5.08B** | **100%** |

---

## 4. quant_probe mapping

| Layer type | Regex |
|---|---|
| `attn.qkv` | `layers\.(\d+)\.proj\.qkv\.weight$` |
| `attn.o` | `layers\.(\d+)\.proj\.o\.weight$` |
| `ffn.up_gate` | `layers\.(\d+)\.proj\.ffn_up_gate\.weight$` |
| `ffn.down` | `layers\.(\d+)\.proj\.ffn_down\.weight$` |

`highprec_excluded` (not analysed, keep BF16): the 17 global 2D matrices above,
plus the 1D tensors (`norm1`, `norm2`, `q_norm`, `k_norm`, `final_norm_video`,
`rope.bands`).

Validated against both real 379-key headers (base and SFT): 168/168 block
matrices matched, zero 1D tensors matched, and the generated `--exclude-layers`
regex covers exactly the 211 non-target keys with no block key leaked.

---

## 5. Spread filter recommendations

```python
spread_filter_recommended = ["attn.qkv", "attn.o"]
```

Q/K/V are fused into a single `proj.qkv`, so — as with Z-Image's
`attention.qkv` — the whole tensor has to be decided individually; there is no
`attn.k` / `attn.v` to exempt separately. `proj.o` writes straight back into the
residual stream and is equally sensitive to a group-level demotion. The
`q_norm` / `k_norm` RMSNorms are applied *after* the projection and do not shield
it from quantization error.

```bash
quant-probe moss_image_sft_bf16.safetensors --model moss \
    --spread-filter-exempt attn.qkv attn.o
```

---

## 6. convert_to_quant flags

**There is no MOSS-Image preset in convert_to_quant** (nothing matches "moss"
on its `main` branch). `quant_probe` therefore emits **no model flag** for
`moss` and relies on its own `--exclude-layers`:

```
(text_embedder|time_embedding|video_embedder|video_head_modulation|shared_block_modulation|final_linear|final_norm|rope[.]bands|[.]norm1|[.]norm2|[.]q_norm|[.]k_norm)
```

```bash
# mixed: NVFP4 base, FP8 for the sensitive tier (default)
quant-probe moss_image_sft_bf16.safetensors --model moss

# mixed: NVFP4 base, MXFP8 for the sensitive tier
quant-probe moss_image_sft_bf16.safetensors --model moss \
    --base-format nvfp4 --sensitive-format mxfp8

# uniform MXFP8: only the BF16 protection applies
quant-probe moss_image_sft_bf16.safetensors --model moss --base-format mxfp8
```

The emitted command is ready to run; `--base-format` and `--sensitive-format`
map onto the main `ctq` flag and `--custom-type` respectively. When both are
equal the `--custom-layers` block is dropped.

### Reference: what Comfy-Org shipped

`moss_image_sft_int8_convrot.safetensors` (5.41 GB instead of 10.16 GB)
quantizes **exactly the 168 block matrices** and keeps all 17 global matrices in
BF16:

```
comfy_quant: {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

with a per-row `weight_scale` (`[out_features, 1]`, F32) plus a per-block
`comfy_attention.config`. That is the empirical baseline `moss` reproduces for
the exclude list — read from the file's own header, so it can be re-checked
without downloading anything.

---

## 7. Calibrated defaults

```python
default_args = {
    "fp8_percentile":   50.0,   # cloned from Z-Image — recalibrate
    "keep_percentile":  90.0,
    "kurtosis_keep":     8.0,
    "fp8_min_score":     0.0,
    "min_group_spread": 0.20,
}
```

Cloned from Z-Image (same single-stream DiT lineage) and **not yet
recalibrated**. First run should be:

```bash
quant-probe moss_image_sft_bf16.safetensors --model moss \
    --csv moss.csv --spread-filter-exempt attn.qkv attn.o
```

then inspect the score distribution in `moss.csv` before trusting the
FP8/*KEEP* split.
