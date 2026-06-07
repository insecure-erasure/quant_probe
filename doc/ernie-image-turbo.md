# ERNIE Image Turbo — Layer Layout

**File:** `ernie-image-turbo.safetensors`  
**Size:** 15 GB  
**Dtype:** `bfloat16`  
**Total params:** 8.03B  

---

## 1. Global Layers (outside transformer blocks)

### 1a. 2D weights (quantizable, but should be excluded as sensitive)

| Layer | Shape | Params | Notes |
|-------|-------|--------|-------|
| `x_embedder.proj.weight` | `[4096, 128, 1, 1]` | 0.52M | 4D (conv) |
| `time_embedding.linear_1.weight` | `[4096, 4096]` | 16.78M | bf16 |
| `time_embedding.linear_2.weight` | `[4096, 4096]` | 16.78M | bf16 |
| `text_proj.weight` | `[4096, 3072]` | 12.58M | bf16 |
| `adaLN_modulation.1.weight` | `[24576, 4096]` | 100.66M | bf16 |
| `final_norm.linear.weight` | `[8192, 4096]` | 33.55M | bf16 |
| `final_linear.weight` | `[128, 4096]` | 0.52M | bf16 |

### 1b. Biases

| Layer | Shape |
|-------|-------|
| `x_embedder.proj.bias` | `[4096]` |
| `time_embedding.linear_1.bias` | `[4096]` |
| `time_embedding.linear_2.bias` | `[4096]` |
| `adaLN_modulation.1.bias` | `[24576]` |
| `final_norm.linear.bias` | `[8192]` |
| `final_linear.bias` | `[128]` |

---

## 2. Transformer Blocks (36 blocks: `layers.0` through `layers.35`)

Each block has an identical structure.

### 2a. Self-Attention (separate projections, NOT fused)

| Layer | Shape | Params | Description |
|-------|-------|--------|-------------|
| `self_attention.to_q.weight` | `[4096, 4096]` | 16.78M | Q projection |
| `self_attention.to_k.weight` | `[4096, 4096]` | 16.78M | K projection |
| `self_attention.to_v.weight` | `[4096, 4096]` | 16.78M | V projection |
| `self_attention.to_out.0.weight` | `[4096, 4096]` | 16.78M | Output projection |
| `self_attention.norm_k.weight` | `[128]` | — | 1D (norm) |
| `self_attention.norm_q.weight` | `[128]` | — | 1D (norm) |

### 2b. MLP (Feed-Forward)

| Layer | Shape | Params | Description |
|-------|-------|--------|-------------|
| `mlp.gate_proj.weight` | `[12288, 4096]` | 50.33M | Gate (SwiGLU) |
| `mlp.up_proj.weight` | `[12288, 4096]` | 50.33M | Up |
| `mlp.linear_fc2.weight` | `[4096, 12288]` | 50.33M | Down |

### 2c. AdaLN norms (1D — always exclude)

| Layer | Shape | Description |
|-------|-------|-------------|
| `adaLN_sa_ln.weight` | `[4096]` | 1D norm |
| `adaLN_mlp_ln.weight` | `[4096]` | 1D norm |

> **Subtotal per block:** 218.10M params  
> **Total 36 blocks:** 7,851.60M params (7.85B)

---

## 3. Quantizable Parameter Summary

| Category | Params | % of total |
|----------|--------|------------|
| Blocks (quantizable) | 7,851.60M | 97.7% |
| Global (excluded) | 181.39M | 2.3% |
| 1D norms (excluded) | 0.30M | 0.0% |
| **TOTAL** | **8,033.0M** | **100.0%** |

---

## 4. quant_probe Mapping

### Layer types for analysis (blocks)

| Layer type | Regex pattern |
|------------|---------------|
| `self_attn.q` | `layers\.(\d+)\.self_attention\.to_q\.weight$` |
| `self_attn.k` | `layers\.(\d+)\.self_attention\.to_k\.weight$` |
| `self_attn.v` | `layers\.(\d+)\.self_attention\.to_v\.weight$` |
| `self_attn.o` | `layers\.(\d+)\.self_attention\.to_out\.0\.weight$` |
| `ffn.gate` | `layers\.(\d+)\.mlp\.gate_proj\.weight$` |
| `ffn.up` | `layers\.(\d+)\.mlp\.up_proj\.weight$` |
| `ffn.down` | `layers\.(\d+)\.mlp\.linear_fc2\.weight$` |

### `highprec_excluded` layers (do NOT analyze, keep BF16)

- `x_embedder.proj`
- `time_embedding.linear_1`
- `time_embedding.linear_2`
- `text_proj`
- `adaLN_modulation.1`
- `final_norm.linear`
- `final_linear`

### Layers excluded from analysis (1D norms)

- `adaLN_sa_ln`
- `adaLN_mlp_ln`
- `self_attention.norm_k`
- `self_attention.norm_q`

---

## 5. Spread Filter Recommendations

```python
spread_filter_recommended = [
    "self_attn.q",  # QK^T sensitive — crucial for precision
    "self_attn.k",  # QK^T sensitive
]
```

> **Note:** ERNIE has **no cross-attention**. Only self-attention.  
> Text conditioning is injected via `adaLN_modulation` + `text_proj`.

**Rationale:** In DiT architectures, Q and K projections determine attention quality. The QK<sup>T</sup> product amplifies leptokurtic distributions, making these layers particularly sensitive to quantization. The spread filter may demote FP8 to NVFP4 for entire groups when positional variation is low, but Q/K deserve individual per-tensor decisions.

---

## 6. Additional Details for `ArchitectureConfig`

| Parameter | Value |
|-----------|-------|
| `total_blocks` | 36 |
| `hidden_size` | 4096 |
| `ffn_size` | 12288 (3× hidden) |
| `num_heads` | 32 (128‑dim per head) |
| `attention` | split Q/K/V/O (NOT fused qkv) |
| `mlp_style` | SwiGLU (gate_proj, up_proj, linear_fc2 as down) |

### `detail_groups`

```python
detail_groups = {
    "self_attention": ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"],
    "ffn":            ["ffn.gate", "ffn.up", "ffn.down"],
}
```

### Suggested `default_args` (based on Z-Image as reference)

```python
default_args = {
    "fp8_percentile":   50.0,   # tune empirically
    "keep_percentile":  90.0,   # default
    "kurtosis_keep":     8.0,   # default
    "fp8_min_score":    0.0,    # no artificial floor
    "min_group_spread": 0.20,   # same scale as Z-Image
}
```

---

## 7. Empirical Quantization Tests (silveroxides)

[silveroxides](https://github.com/silveroxides), the author of [`convert_to_quant`](https://github.com/silveroxides/convert_to_quant), has run quantization tests on this model using the following `--exclude-layers` regex to keep sensitive layers in BF16:

```
(time_embedding|adaLN_modulation|final_linear|final_norm|x_embedder|layers[.]0[.]self_attention|layers[.]0[.]mlp.gate_proj|layers[.]0[.]mlp[.]up_proj|text_proj)
```

### Layers kept in BF16 by this pattern

| Layer(s) | Reason |
|----------|--------|
| `time_embedding.linear_1` / `linear_2` | Temporal embeddings — sensitive to precision loss |
| `adaLN_modulation.1` | Global modulation — controls all block conditioning |
| `final_linear` | Output projection — directly affects generation quality |
| `final_norm.linear` | Final normalization — compounds output error |
| `x_embedder.proj` | Patch embedding (4D conv) — first layer, sets up latent |
| `text_proj` | Text conditioning projection — drives semantic fidelity |
| `layers.0.self_attention.*` (Q, K, V, out) | First block attention — early processing, high sensitivity |
| `layers.0.mlp.gate_proj` | First block MLP gate |
| `layers.0.mlp.up_proj` | First block MLP up-projection |

### Usage

```bash
ctq -i ernie-image-turbo.safetensors \
    -o ernie-image-turbo-fp8.safetensors \
    --comfy_quant \
    --simple \
    --low-memory \
    --exclude-layers "(time_embedding|adaLN_modulation|final_linear|final_norm|x_embedder|layers[.]0[.]self_attention|layers[.]0[.]mlp.gate_proj|layers[.]0[.]mlp[.]up_proj|text_proj)"
```

This pattern excludes all global sensitive layers plus the entire first transformer block's attention and part of its MLP, keeping roughly **~134M params** (`2.3% global + ~218M from block 0 attention/gate/up`) in BF16 while quantizing the remaining **~35 blocks × ~184M** to FP8/INT8.
