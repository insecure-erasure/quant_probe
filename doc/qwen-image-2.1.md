# Qwen-Image 2.1 — Layer Layout

**Files analysed (Comfy-Org repack, the ones tested with ComfyUI):**

| File | Size | Tensors |
|---|---|---|
| `diffusion_models/qwen_image_2.1_bf16.safetensors` | 14.23 GB | 265 |
| `diffusion_models/qwen_image_2.1_turbo_bf16.safetensors` | 14.23 GB | 265 (identical key set + shapes) |

Header read remotely from Hugging Face (HTTP range request, no download).
Source repos: `Qwen/Qwen-Image-2.1`, `Qwen/Qwen-Image-2.1-Turbo`.

**Architecture:** single-stream DiT, **32 blocks**, hidden 4096, 32 heads × 128,
~7.1B params in the diffusion transformer. Model card: *"7B parameters in its
visual generation component (32 Single-Stream DiT layers)"*, *"mixed-granularity
attention and prefix KV cache reuse"* — text conditioning is injected once by
`txt_in` and reused as a causal prefix, which is why blocks carry **only**
`attn` + `img_mlp` (no per-block text MLP, no `img_mod`/`txt_mod`).

---

## 1. Global layers (outside transformer blocks) — 9 tensors

| Layer | Shape | Params | Comfy-Org build |
|---|---|---|---|
| `img_in.weight` | `[4096, 64]` | 0.26M | BF16 |
| `time_text_embed.timestep_embedder.linear_1.weight` | `[4096, 256]` | 1.05M | BF16 |
| `time_text_embed.timestep_embedder.linear_2.weight` | `[4096, 4096]` | 16.78M | BF16 |
| `txt_in.in_layer.weight` | `[4096, 4096]` | 16.78M | BF16 |
| `txt_in.out_layer.weight` | `[4096, 4096]` | 16.78M | BF16 |
| `modulation.1.weight` | `[16384, 4096]` | 67.11M | BF16 |
| `norm_out.linear.weight` | `[4096, 4096]` | 16.78M | BF16 |
| `proj_out.weight` | `[64, 4096]` | 0.26M | BF16 |
| `txt_in.text_norm.weight` | `[4096]` | 1D | BF16 |

> `modulation.1` is **global** in 2.1 (one 4×hidden adaLN head for all blocks).
> Qwen-Image v1 had a per-block `img_mod` / `txt_mod` instead — this is the
> single biggest key-layout difference between the two generations.

## 2. Transformer blocks (`transformer_blocks.0` … `.31`) — 256 tensors

| Layer | Shape | Params | Comfy-Org build |
|---|---|---|---|
| `attn.to_q.weight` | `[4096, 4096]` | 16.78M | INT8 |
| `attn.to_k.weight` | `[4096, 4096]` | 16.78M | INT8 |
| `attn.to_v.weight` | `[4096, 4096]` | 16.78M | INT8 |
| `attn.to_out.0.weight` | `[4096, 4096]` | 16.78M | INT8 |
| `img_mlp.gate_up.weight` | `[24576, 4096]` | 100.66M | INT8 |
| `img_mlp.out.weight` | `[4096, 12288]` | 50.33M | INT8 |
| `attn.norm_q.weight` | `[128]` | 1D | BF16 |
| `attn.norm_k.weight` | `[128]` | 1D | BF16 |

6 quantizable matrices × 32 blocks = **192 tensors**. `gate_up` is the ComfyUI
*fused* SwiGLU gate+up (`2 × 12288`); the diffusers original ships
`img_mlp.gate_layer` + `img_mlp.proj` separately — both namings are matched.

## 3. Parameter summary

| Category | Params | % |
|---|---|---|
| Blocks (192 quantizable 2D) | ~6.98B | 98.1% |
| Global (8 × 2D, kept BF16) | 0.135B | 1.9% |
| 1D norms (65 tensors) | ~0.04M | ~0% |
| **Total** | **~7.12B** | **100%** |

---

## 4. quant_probe mapping

| Layer type | Regex |
|---|---|
| `attn.q` | `transformer_blocks\.(\d+)\.attn\.to_q\.weight$` |
| `attn.k` | `transformer_blocks\.(\d+)\.attn\.to_k\.weight$` |
| `attn.v` | `transformer_blocks\.(\d+)\.attn\.to_v\.weight$` |
| `attn.o` | `transformer_blocks\.(\d+)\.attn\.to_out\.0\.weight$` |
| `ffn.gate_up` | `transformer_blocks\.(\d+)\.img_mlp\.gate_up\.weight$` |
| `ffn.gate` / `ffn.up` | `…img_mlp\.gate_layer\.weight$` / `…img_mlp\.proj\.weight$` (diffusers naming) |
| `ffn.out` | `transformer_blocks\.(\d+)\.img_mlp\.out\.weight$` |

`highprec_excluded` (not analysed, keep BF16): the 8 global 2D matrices, plus
`txt_in.text_norm`, `attn.norm_q`, `attn.norm_k`.

Validated against the real 265-key header: 192/192 block matrices matched, zero
1D tensors matched, and the generated `--exclude-layers` regex covers exactly the
73 non-target keys with no block key leaked.

---

## 5. Spread filter recommendations

```python
spread_filter_recommended = ["attn.q", "attn.k", "attn.v", "attn.o"]
```

Same rationale as ERNIE/other DiTs: the $QK^T$ product amplifies leptokurtic
weight distributions, so Q/K deserve per-tensor decisions instead of being
demoted by a group-level spread verdict. V and out are included as well — they
write directly into the residual stream, so a group-level demotion there is just
as costly. Note that in 2.1 Q and K each have an RMSNorm (`norm_q` / `norm_k`)
applied *after* the projection — the norm does not shield the projection itself
from quantization error.

```bash
quant-probe qwen_image_2.1_bf16.safetensors --model qwen21 \
    --spread-filter-exempt attn.q attn.k attn.v attn.o
```

**This list is load-bearing for `attn.q`.** Its measured positional spread is
**0.091**, less than half of the `min_group_spread` floor of 0.20, so without the
exemption the spread filter would demote every FP8 `attn.q` tensor in the block
range it resolved. The exemption is what keeps the projection decisions
tensor-by-tensor; dropping `attn.q` from it is not a neutral experiment.

---

## 6. convert_to_quant flags

**There is no Qwen-Image 2.1 preset in convert_to_quant** (checked on `main`,
`dev-clean-branch`, `dev/new` and the int4 branch, at commit dates up to
2026-11-16). The `--qwen` filter is **Qwen-Image v1 only**:

| Preset field | Value | Valid for 2.1? |
|---|---|---|
| `exclude` | `norm_added_k`, `norm_added_q`, `norm_k`, `norm_q`, `txt_norm` | `norm_added_*` don't exist in 2.1 (v1's added text-KV projections); the rest are 1D no-ops |
| `highprec` | `time_text_embed`, `img_in`, `norm_out`, `proj_out`, `transformer_blocks.0.img_mod.1`, `txt_in` | `img_mod.1` doesn't exist in 2.1 → **`modulation.1` (67M) is not protected** |

So `--qwen` would *look* correct while silently quantizing the global
`modulation.1`. `quant_probe` therefore emits **no model flag** for `qwen21` and
relies on its own `--exclude-layers`:

```
(img_in|txt_in|time_text_embed|modulation|norm_out|proj_out|attn[.]norm_q|attn[.]norm_k)
```

```bash
# mixed: NVFP4 base, MXFP8 for the sensitive tier
quant-probe model.safetensors --model qwen21 \
    --base-format nvfp4 --sensitive-format mxfp8

# uniform MXFP8: only the BF16 protection applies
quant-probe model.safetensors --model qwen21 --base-format mxfp8
```

The emitted command is ready to run; `--base-format` and `--sensitive-format`
map onto the main `ctq` flag and `--custom-type` respectively. When both are
equal the `--custom-layers` block is dropped, because everything that is not
excluded already receives the base format.

MXFP8 costs 8.25 bits per weight (E4M3 data plus an E8M0 scale every 32
weights), not 8.0, so the size table uses 1.03125 bytes per weight for it.

### Reference: what Comfy-Org shipped

`qwen_image_2.1_int8_convrot.safetensors` (7.26 GB instead of 14.23 GB) quantizes
**exactly the 192 block matrices** and keeps every global matrix in BF16:

```
comfy_quant: {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

with a per-row `weight_scale` (`[out_features, 1]`, F32). That is the empirical
baseline `qwen21` reproduces for the exclude list — reproduced from the file's own
header, so it can be re-checked without downloading anything.

---

## 7. Calibrated defaults

```python
default_args = {
    "fp8_percentile":   50.0,
    "keep_percentile":  90.0,
    "kurtosis_keep":     8.0,
    "fp8_min_score":     0.0,
    "min_group_spread": 0.20,
}
```

**Verified on a 224-tensor run.** The run used the *diffusers* naming
(`img_mlp.gate_layer` + `img_mlp.proj` instead of the ComfyUI repack's fused
`gate_up`), which also exercises the dual-namespace patterns in
`models/qwen21.py`: 7 matrices × 32 blocks. The Z-Image values survived
unchanged, and each one was checked against the measured distribution:

| Knob | Measurement | Outcome |
|---|---|---|
| `fp8_percentile 50` | Scores span 0.17–0.90: a smooth continuum below 0.55, largest gap in that region 0.02 | Cut lands at **0.386** (the median). Neutral, not tuned — there is no valley to cut on |
| `keep_percentile 90` | Cut lands at **0.653**, inside a dense band. The first real gap above it is **0.762 → 0.796** (0.034) | Kept. `keep-percentile 94` would land on that gap and isolate a 14-tensor cluster, but 0.034 in 224 samples is thin evidence for leaving 9 more tensors quantized |
| `fp8_min_score 0.0` | No low-score cluster | Guard stays off |
| `min_group_spread 0.20` | Never fires (`spread_demotion` absent): FFN positional spreads are 0.28 / 0.28 / 0.40 | Kept, but note §5 — `attn.q`'s own spread is **0.091**, so the exemption list is what protects attention, not this floor |
| `kurtosis_keep 8.0` | 5 tensors at kurtosis ≥ 8, the worst **27.47** (`attn.o` block 0) | Catches the pathological ones without over-reaching |

Split obtained: **89 FP8 / 112 NVFP4 / 23 \*KEEP\*** by raw score, resolved to
**83 / 118 / 23** after the group logic (8 promotions, 14 group demotions, 2
individual \*KEEP\* rescued back to BF16 within an FP8 group).

Caveat carried by the variant: the 224-tensor run has `ffn.gate` and `ffn.up` as
separate 12288×4096 matrices, while the Comfy-Org repack fuses them into a single
24576×4096 `gate_up`. Kurtosis — and therefore the FFN scores — are not
identical between the two packagings, so the thresholds above are as valid as
they can be for the fused file without a run of its own. Attention tensors are
byte-identical in both, and they dominate the \*KEEP\* set (17 of 23).
