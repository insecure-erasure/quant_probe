"""
Quantization sensitivity analysis — architecture-agnostic core.

All knowledge of specific model architectures is injected via the
ArchitectureConfig instance passed to each public function.  This module
contains no if/else branches on model names.
"""

import csv
import json
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors is required.  Install with: pip install safetensors")
    sys.exit(1)

from quant_probe.models.base import ArchitectureConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TensorMetrics:
    """Metrics for a single weight tensor."""
    key: str
    layer_type: str
    block_idx: int
    subgraph: str
    shape: Tuple[int, ...]
    excess_kurtosis: float
    dynamic_range: float
    std: float
    outlier_pct: float
    aspect_ratio: float
    score: float = 0.0
    recommendation: str = ""
    reason: str = ""


@dataclass
class AggregatedMetrics:
    """Mean metrics for a group of tensors."""
    layer_type: str
    block_range: str
    subgraph: str
    count: int
    excess_kurtosis: float
    kurtosis_max: float
    dynamic_range: float
    std: float
    outlier_pct: float
    aspect_ratio: float
    score: float = 0.0
    recommendation: str = ""
    reason: str = ""
    spread_filtered: bool = False


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(tensor: torch.Tensor, outlier_sigma: float) -> Dict[str, float]:
    """
    Compute quantization sensitivity metrics for a 2D weight tensor.

    Excess kurtosis is used (kurtosis - 3), so a normal distribution gives 0.
    """
    t = tensor.to(torch.float32).flatten()

    mean = t.mean()
    std  = t.std()

    if std > 0:
        excess_kurtosis = ((t - mean) ** 4).mean() / (std ** 4) - 3.0
    else:
        excess_kurtosis = torch.tensor(0.0)

    dynamic_range = t.abs().max() - t.abs().min()

    if std > 0:
        outlier_mask = (t - mean).abs() > outlier_sigma * std
        outlier_pct  = outlier_mask.float().mean() * 100.0
    else:
        outlier_pct = torch.tensor(0.0)

    rows, cols   = tensor.shape
    aspect_ratio = max(rows, cols) / min(rows, cols)

    return {
        "excess_kurtosis": excess_kurtosis.item(),
        "dynamic_range":   dynamic_range.item(),
        "std":             std.item(),
        "outlier_pct":     outlier_pct.item(),
        "aspect_ratio":    aspect_ratio,
    }


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_scores(
    all_metrics:     List[TensorMetrics],
    kurtosis_weight: float,
    range_weight:    float,
    ar_weight:       float,
) -> None:
    """Compute combined sensitivity score for each tensor (in-place)."""
    kurt_vals = torch.tensor([m.excess_kurtosis for m in all_metrics])
    q25 = torch.quantile(kurt_vals, 0.25).item()
    q75 = torch.quantile(kurt_vals, 0.75).item()
    iqr = q75 - q25

    def norm_iqr(val: float) -> float:
        lo      = q25 - 1.5 * iqr
        hi      = q75 + 1.5 * iqr
        clipped = max(lo, min(hi, val))
        return (clipped - lo) / (hi - lo) if hi > lo else 0.0

    by_type: Dict[str, List[TensorMetrics]] = defaultdict(list)
    for m in all_metrics:
        by_type[m.layer_type].append(m)

    type_range_bounds: Dict[str, Tuple[float, float]] = {}
    type_ar_bounds:    Dict[str, Tuple[float, float]] = {}

    for layer_type, group in by_type.items():
        dr_vals = [m.dynamic_range for m in group]
        ar_vals = [m.aspect_ratio  for m in group]
        type_range_bounds[layer_type] = (min(dr_vals), max(dr_vals))
        type_ar_bounds[layer_type]    = (min(ar_vals), max(ar_vals))

    for m in all_metrics:
        n_kurt  = norm_iqr(m.excess_kurtosis)

        dr_lo, dr_hi = type_range_bounds[m.layer_type]
        n_range = (m.dynamic_range - dr_lo) / (dr_hi - dr_lo) if dr_hi > dr_lo else 0.0

        ar_lo, ar_hi = type_ar_bounds[m.layer_type]
        n_ar    = (m.aspect_ratio - ar_lo) / (ar_hi - ar_lo) if ar_hi > ar_lo else 0.0

        m.score = (
            kurtosis_weight * n_kurt +
            range_weight    * n_range +
            ar_weight       * n_ar
        )


# ---------------------------------------------------------------------------
# Threshold and recommendation logic
# ---------------------------------------------------------------------------

def compute_auto_thresholds(
    all_metrics:     List[TensorMetrics],
    fp8_percentile:  float,
    keep_percentile: float,
) -> Tuple[float, float]:
    scores         = torch.tensor([m.score for m in all_metrics])
    fp8_threshold  = torch.quantile(scores, fp8_percentile  / 100.0).item()
    keep_threshold = torch.quantile(scores, keep_percentile / 100.0).item()
    return fp8_threshold, keep_threshold


def assign_recommendation(
    score:           float,
    fp8_threshold:   float,
    keep_threshold:  float,
    fp8_min_score:   float = 0.0,
    excess_kurtosis: float = 0.0,
    kurtosis_keep:   float = float("inf"),
) -> Tuple[str, str]:
    """Return (*KEEP* | FP8 | NVFP4, reason) based on score thresholds."""
    if excess_kurtosis >= kurtosis_keep:
        return "*KEEP*", "kurtosis_floor"
    if score >= keep_threshold:
        return "*KEEP*", "score_percentile"
    elif score >= fp8_threshold and score >= fp8_min_score:
        return "FP8", "score_percentile"
    elif score >= fp8_threshold and score < fp8_min_score:
        return "NVFP4", "score_below_fp8_min"
    else:
        return "NVFP4", "default"


# ---------------------------------------------------------------------------
# Low-RAM safetensors helpers
# ---------------------------------------------------------------------------

_ST_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16":  torch.float16,
    "F32":  torch.float32,
    "F64":  torch.float64,
    "I8":   torch.int8,
    "I16":  torch.int16,
    "I32":  torch.int32,
    "I64":  torch.int64,
    "U8":   torch.uint8,
}


def read_safetensors_header(model_path: str) -> Tuple[Dict, int]:
    with open(model_path, "rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise RuntimeError("File too short to be a valid safetensors file.")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size)
    header      = json.loads(header_json.decode("utf-8"))
    data_offset = 8 + header_size
    return header, data_offset


def load_tensor_lowram(
    model_path:  str,
    key:         str,
    header:      Dict,
    data_offset: int,
    device:      torch.device,
) -> torch.Tensor:
    meta                = header[key]
    dtype_str           = meta["dtype"]
    shape               = meta["shape"]
    byte_start, byte_end = meta["data_offsets"]

    torch_dtype = _ST_DTYPE_MAP.get(dtype_str)
    if torch_dtype is None:
        raise RuntimeError(f"Unsupported safetensors dtype: {dtype_str}")

    n_bytes = byte_end - byte_start
    with open(model_path, "rb") as f:
        f.seek(data_offset + byte_start)
        raw = f.read(n_bytes)

    tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype).reshape(shape)
    return tensor.to(device=device)


# ---------------------------------------------------------------------------
# Analysis pass
# ---------------------------------------------------------------------------

def _match_key(
    key:              str,
    cfg:              ArchitectureConfig,
) -> Optional[Tuple[str, int, str]]:
    """
    Try to match a safetensors key against all configured patterns.

    Returns (layer_type, block_idx, subgraph) or None if no match.
    Subgraph inference is delegated to cfg.infer_subgraph().
    """
    for layer_type, pattern in cfg.layer_patterns.items():
        m = pattern.search(key)
        if m:
            block_idx = int(m.group(1))
            subgraph  = cfg.infer_subgraph(key)
            return layer_type, block_idx, subgraph

    if cfg.refiner_patterns:
        for layer_type, pattern in cfg.refiner_patterns.items():
            m = pattern.search(key)
            if m:
                subgraph  = m.group(1)   # e.g. "context_refiner"
                block_idx = int(m.group(2))
                return layer_type, block_idx, subgraph

    return None


def analyze_model(
    model_path: str,
    device:     torch.device,
    cfg:        ArchitectureConfig,
    outlier_sigma: float,
    low_ram:    bool = False,
) -> List[TensorMetrics]:
    """
    Stream through the model file and compute metrics for all target tensors.
    """
    results: List[TensorMetrics] = []

    if low_ram:
        header, data_offset = read_safetensors_header(model_path)
        all_keys = [k for k in header.keys() if k != "__metadata__"]
    else:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())

    matching: List[Tuple[str, str, int, str]] = []
    for key in all_keys:
        result = _match_key(key, cfg)
        if result:
            layer_type, block_idx, subgraph = result
            matching.append((key, layer_type, block_idx, subgraph))

    total = len(matching)
    print(f"  Target tensors found: {total}")
    print()

    if low_ram:
        for i, (key, layer_type, block_idx, subgraph) in enumerate(matching, 1):
            print(f"\r  Analyzing {i}/{total}: {key[:70]:<70}", end="", flush=True)
            tensor = load_tensor_lowram(model_path, key, header, data_offset, device)
            if tensor.ndim != 2:
                del tensor
                continue
            metrics = compute_metrics(tensor, outlier_sigma)
            results.append(TensorMetrics(
                key             = key,
                layer_type      = layer_type,
                block_idx       = block_idx,
                subgraph        = subgraph,
                shape           = tuple(tensor.shape),
                excess_kurtosis = metrics["excess_kurtosis"],
                dynamic_range   = metrics["dynamic_range"],
                std             = metrics["std"],
                outlier_pct     = metrics["outlier_pct"],
                aspect_ratio    = metrics["aspect_ratio"],
            ))
            del tensor
    else:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for i, (key, layer_type, block_idx, subgraph) in enumerate(matching, 1):
                print(f"\r  Analyzing {i}/{total}: {key[:70]:<70}", end="", flush=True)
                tensor = f.get_tensor(key).to(device=device)
                if tensor.ndim != 2:
                    del tensor
                    continue
                metrics = compute_metrics(tensor, outlier_sigma)
                results.append(TensorMetrics(
                    key             = key,
                    layer_type      = layer_type,
                    block_idx       = block_idx,
                    subgraph        = subgraph,
                    shape           = tuple(tensor.shape),
                    excess_kurtosis = metrics["excess_kurtosis"],
                    dynamic_range   = metrics["dynamic_range"],
                    std             = metrics["std"],
                    outlier_pct     = metrics["outlier_pct"],
                    aspect_ratio    = metrics["aspect_ratio"],
                ))
                del tensor
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"\r  {'Done':<80}")
    return results


# ---------------------------------------------------------------------------
# Block position classification
# ---------------------------------------------------------------------------

def compute_extreme_ranges(
    total_blocks: int,
    extreme_pct:  float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    n_extreme = max(1, round(total_blocks * extreme_pct / 100.0))
    low  = (0, n_extreme - 1)
    high = (total_blocks - n_extreme, total_blocks - 1)
    return low, high


def classify_block(
    block_idx:    int,
    extreme_low:  Tuple[int, int],
    extreme_high: Tuple[int, int],
) -> str:
    if extreme_low[0] <= block_idx <= extreme_low[1]:
        return "extreme_low"
    if extreme_high[0] <= block_idx <= extreme_high[1]:
        return "extreme_high"
    return "middle"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def block_range_label(block_indices: List[int], total_blocks: int) -> str:
    lo, hi = min(block_indices), max(block_indices)
    if lo == 0 and hi == total_blocks - 1:
        return f"0–{hi}"
    return f"{lo}–{hi}"


def aggregate(
    metrics_list: List[TensorMetrics],
    layer_type:   str,
    block_range:  str,
    subgraph:     str,
) -> AggregatedMetrics:
    n = len(metrics_list)
    if n == 0:
        return AggregatedMetrics(layer_type, block_range, subgraph, 0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return AggregatedMetrics(
        layer_type      = layer_type,
        block_range     = block_range,
        subgraph        = subgraph,
        count           = n,
        excess_kurtosis = sum(m.excess_kurtosis for m in metrics_list) / n,
        kurtosis_max    = max(m.excess_kurtosis for m in metrics_list),
        dynamic_range   = sum(m.dynamic_range   for m in metrics_list) / n,
        std             = sum(m.std             for m in metrics_list) / n,
        outlier_pct     = sum(m.outlier_pct     for m in metrics_list) / n,
        aspect_ratio    = metrics_list[0].aspect_ratio,
        score           = sum(m.score           for m in metrics_list) / n,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

SEP = "─" * 115


def fmt_row_summary(agg: AggregatedMetrics) -> str:
    return (
        f"  {agg.layer_type:<28} {agg.block_range:<10} "
        f"{agg.excess_kurtosis:>14.3f}  {agg.dynamic_range:>10.3f}  "
        f"{agg.std:>6.4f}  {agg.outlier_pct:>9.1f}%  "
        f"{agg.aspect_ratio:>5.2f}  {agg.score:>6.3f}  {agg.recommendation}"
    )


def fmt_row_detail(agg: AggregatedMetrics) -> str:
    marker = ""
    if agg.spread_filtered:
        marker = "  [→NVFP4 ~spread]" if agg.recommendation == "FP8" else "  [→FP8 ~spread]"
    return (
        f"  {agg.layer_type:<28} {agg.block_range:<12} "
        f"{agg.excess_kurtosis:>14.3f}  {agg.kurtosis_max:>10.3f}  {agg.dynamic_range:>10.3f}  "
        f"{agg.std:>6.4f}  {agg.outlier_pct:>9.1f}%  "
        f"{agg.score:>6.3f}  {agg.recommendation}{marker}"
    )


def print_summary_table(aggregated: List[AggregatedMetrics], title_suffix: str = "") -> None:
    print()
    print(SEP)
    title = "  SUMMARY BY LAYER TYPE — weight tensors"
    if title_suffix:
        title += f"  [{title_suffix}]"
    print(title)
    print(SEP)
    print(
        f"  {'Layer':<28} {'Blocks':<10} {'Excess kurtosis':>14}  {'Dyn. range':>10}  "
        f"{'Std':>6}  {'Outliers%':>9}  {'AR':>5}  {'Score':>6}  Recommendation"
    )
    print(SEP)
    for agg in aggregated:
        print(fmt_row_summary(agg))
    print(SEP)


def print_detail_table(
    group_name:   str,
    rows:         List[AggregatedMetrics],
    title_suffix: str = "",
) -> None:
    print()
    print(SEP)
    title = f"  DETAIL: {group_name} — weight tensors by block position"
    if title_suffix:
        title += f"  [{title_suffix}]"
    print(title)
    print(SEP)
    print(
        f"  {'Layer':<28} {'Blocks':<12} {'Excess kurtosis':>14}  {'Kurt. max':>10}  {'Dyn. range':>10}  "
        f"{'Std':>6}  {'Outliers%':>9}  {'Score':>6}  Recommendation"
    )
    print(SEP)
    for row in rows:
        print(fmt_row_detail(row))
    print(SEP)


def print_highprec_notice(excluded: List[str], convert_flag: str) -> None:
    print()
    print(SEP)
    print("  LAYERS EXCLUDED FROM ANALYSIS")
    print(SEP)
    print(f"  The following layers are NOT included in this analysis because they")
    print(f"  receive special high-precision treatment in convert_to_quant")
    print(f"  ({convert_flag}) and are assumed to reflect an informed architectural")
    print(f"  decision by the model author.  Consult convert_to_quant documentation")
    print(f"  for details on their handling.")
    print()
    for name in excluded:
        print(f"    - {name}")
    print(SEP)


# ---------------------------------------------------------------------------
# convert_to_quant parameter generation
# ---------------------------------------------------------------------------

def _block_range_to_indices(block_range: str) -> List[int]:
    normalized = block_range.replace("–", "-").replace("—", "-")
    parts      = normalized.split("-")
    lo, hi     = int(parts[0]), int(parts[1])
    return list(range(lo, hi + 1))


def _blocks_to_alternation(indices: List[int]) -> str:
    return "|".join(str(i) for i in sorted(indices))


def build_convert_to_quant_params(
    all_detail_rows:      List[AggregatedMetrics],
    all_metrics:          List[TensorMetrics],
    min_group_spread:     float,
    spread_filter_exempt: set,
) -> Tuple[List[Tuple[str, List[int], str]], List[Tuple[str, List[int], str]]]:
    """
    Build FP8 and *KEEP* recommendations for convert_to_quant.

    Returns lists of (layer_type, block_indices, subgraph) tuples.
    """
    individual_rec: Dict[Tuple[str, int, str], str] = {
        (m.layer_type, m.block_idx, m.subgraph): m.recommendation for m in all_metrics
    }

    by_layer_type: Dict[str, List[AggregatedMetrics]] = defaultdict(list)
    for row in all_detail_rows:
        by_layer_type[row.layer_type].append(row)

    low_spread_types: set = set()
    for layer_type, rows in by_layer_type.items():
        if layer_type in spread_filter_exempt:
            continue
        scores = [r.score for r in rows]
        spread = max(scores) - min(scores)
        if spread < min_group_spread:
            low_spread_types.add(layer_type)

    fp8_entries:  List[Tuple[str, List[int], str]] = []
    keep_entries: List[Tuple[str, List[int], str]] = []

    for row in all_detail_rows:
        indices  = _block_range_to_indices(row.block_range)
        subgraph = row.subgraph

        if row.layer_type in spread_filter_exempt:
            keep_idxs, fp8_idxs = [], []
            for idx in indices:
                rec = individual_rec.get((row.layer_type, idx, subgraph), "NVFP4")
                if rec == "*KEEP*":
                    keep_idxs.append(idx)
                elif rec == "FP8":
                    fp8_idxs.append(idx)
            if keep_idxs:
                keep_entries.append((row.layer_type, keep_idxs, subgraph))
            if fp8_idxs:
                fp8_entries.append((row.layer_type, fp8_idxs, subgraph))

        elif row.layer_type in low_spread_types and row.recommendation == "FP8":
            row.spread_filtered = True

        elif row.layer_type in low_spread_types and row.recommendation == "*KEEP*":
            row.spread_filtered = True
            fp8_entries.append((row.layer_type, indices, subgraph))

        elif row.recommendation == "FP8":
            fp8_entries.append((row.layer_type, indices, subgraph))

        elif row.recommendation == "*KEEP*":
            keep_idxs, fp8_idxs = [], []
            for idx in indices:
                rec = individual_rec.get((row.layer_type, idx, subgraph), "NVFP4")
                if rec == "*KEEP*":
                    keep_idxs.append(idx)
                elif rec == "FP8":
                    fp8_idxs.append(idx)
            if keep_idxs:
                keep_entries.append((row.layer_type, keep_idxs, subgraph))
            if fp8_idxs:
                fp8_entries.append((row.layer_type, fp8_idxs, subgraph))

    return fp8_entries, keep_entries


def _build_regex_for_entries(
    entries: List[Tuple[str, List[int], str]],
    cfg:     ArchitectureConfig,
) -> Optional[str]:
    """Build a single alternation regex from (layer_type, block_indices, subgraph) tuples."""
    if not entries:
        return None

    patterns = []
    for layer_type, indices, subgraph in entries:
        key_pat   = cfg.layer_type_to_key(layer_type)
        block_alt = _blocks_to_alternation(indices)
        patterns.append(cfg.build_block_path(subgraph, block_alt, key_pat))

    return "|".join(patterns)


def print_suggested_params(
    fp8_entries:      List[Tuple[str, List[int], str]],
    keep_entries:     List[Tuple[str, List[int], str]],
    fp8_min_score:    float,
    min_group_spread: float,
    cfg:              ArchitectureConfig,
) -> None:
    print()
    print(SEP)
    print("  SUGGESTED convert_to_quant PARAMETERS")
    print(SEP)
    print(f"  Model: {cfg.name}  |  convert_to_quant flag: {cfg.convert_flag}")
    print(f"  Filters active — FP8 min score: {fp8_min_score}  |  min group spread: {min_group_spread}")
    print()

    fp8_regex = _build_regex_for_entries(fp8_entries, cfg)
    if fp8_regex:
        print(f'  --custom-layers "{fp8_regex}"')
        print( "  --custom-type fp8")
    else:
        print("  No layers recommended as FP8 (all NVFP4, or filtered by min score / spread)")

    print()

    keep_regex = _build_regex_for_entries(keep_entries, cfg)
    if keep_regex:
        print(f'  --exclude-layers "{keep_regex}"')
    else:
        print("  No additional --exclude-layers needed (model flag covers existing exclusions)")

    print(SEP)


# ---------------------------------------------------------------------------
# Effective recommendation builder
# ---------------------------------------------------------------------------

def build_effective_rec(
    all_detail_rows:      List[AggregatedMetrics],
    all_metrics:          List[TensorMetrics],
    spread_filter_exempt: set,
) -> Tuple[Dict, Dict]:
    """Build (effective_rec, effective_reason) dicts keyed by (layer_type, block_idx, subgraph)."""
    individual_rec: Dict[Tuple[str, int, str], str] = {
        (m.layer_type, m.block_idx, m.subgraph): m.recommendation for m in all_metrics
    }
    effective_rec:    Dict[Tuple[str, int, str], str] = {}
    effective_reason: Dict[Tuple[str, int, str], str] = {}

    for row in all_detail_rows:
        indices  = _block_range_to_indices(row.block_range)
        subgraph = row.subgraph

        if row.layer_type in spread_filter_exempt:
            for idx in indices:
                k = (row.layer_type, idx, subgraph)
                effective_rec[k] = individual_rec.get(k, "NVFP4")

        elif row.spread_filtered:
            eff = "NVFP4" if row.recommendation == "FP8" else "FP8"
            for idx in indices:
                k = (row.layer_type, idx, subgraph)
                effective_rec[k]    = eff
                effective_reason[k] = "spread_demotion"

        elif row.recommendation == "*KEEP*":
            for idx in indices:
                k       = (row.layer_type, idx, subgraph)
                ind_rec = individual_rec.get(k, row.recommendation)
                effective_rec[k] = ind_rec
                if ind_rec != "*KEEP*":
                    effective_reason[k] = "group_keep_resolved"

        else:
            for idx in indices:
                k       = (row.layer_type, idx, subgraph)
                effective_rec[k] = row.recommendation
                ind_rec = individual_rec.get(k, row.recommendation)
                if ind_rec != row.recommendation:
                    if row.recommendation == "FP8" and ind_rec == "NVFP4":
                        effective_reason[k] = "group_fp8_promotion"
                    else:
                        effective_reason[k] = "group_spread_demotion"

    return effective_rec, effective_reason


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    all_metrics:      List[TensorMetrics],
    output_path:      str,
    effective_rec:    Dict,
    effective_reason: Dict,
) -> None:
    fieldnames = [
        "key", "layer_type", "block_idx", "subgraph",
        "rows", "cols",
        "excess_kurtosis", "dynamic_range", "std", "outlier_pct",
        "aspect_ratio", "score", "recommendation", "effective_recommendation",
        "reason",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            key = (m.layer_type, m.block_idx, m.subgraph)
            eff = effective_rec.get(key, m.recommendation)
            rsn = effective_reason.get(key, m.reason)
            writer.writerow({
                "key":                      m.key,
                "layer_type":               m.layer_type,
                "block_idx":                m.block_idx,
                "subgraph":                 m.subgraph,
                "rows":                     m.shape[0],
                "cols":                     m.shape[1],
                "excess_kurtosis":          f"{m.excess_kurtosis:.4f}",
                "dynamic_range":            f"{m.dynamic_range:.4f}",
                "std":                      f"{m.std:.6f}",
                "outlier_pct":              f"{m.outlier_pct:.2f}",
                "aspect_ratio":             f"{m.aspect_ratio:.2f}",
                "score":                    f"{m.score:.4f}",
                "recommendation":           m.recommendation,
                "effective_recommendation": eff,
                "reason":                   rsn,
            })
    print(f"\n  CSV exported to: {output_path}")


# ---------------------------------------------------------------------------
# Output file size estimation
# ---------------------------------------------------------------------------

_BYTES_PER_WEIGHT: Dict[str, float] = {
    "*KEEP*": 2.0,
    "FP8":    1.0,
    "NVFP4":  0.5625,
}


def estimate_output_size(
    all_metrics:    List[TensorMetrics],
    effective_rec:  Dict,
    original_bytes: int,
) -> Dict:
    in_scope_original  = 0
    in_scope_estimated = 0.0
    per_format_bytes:  Dict[str, float] = {"*KEEP*": 0.0, "FP8": 0.0, "NVFP4": 0.0}
    per_format_counts: Dict[str, int]   = {"*KEEP*": 0,   "FP8": 0,   "NVFP4": 0}

    for m in all_metrics:
        n_weights          = m.shape[0] * m.shape[1]
        in_scope_original += n_weights * 2

        fmt          = effective_rec.get((m.layer_type, m.block_idx, m.subgraph), m.recommendation)
        bpw          = _BYTES_PER_WEIGHT.get(fmt, 2.0)
        tensor_bytes = n_weights * bpw
        in_scope_estimated          += tensor_bytes
        per_format_bytes[fmt]        = per_format_bytes.get(fmt, 0.0) + tensor_bytes
        per_format_counts[fmt]       = per_format_counts.get(fmt, 0)  + 1

    out_of_scope    = original_bytes - in_scope_original
    total_estimated = out_of_scope + in_scope_estimated

    return {
        "in_scope_original_bytes":  in_scope_original,
        "in_scope_estimated_bytes": in_scope_estimated,
        "per_format_bytes":         per_format_bytes,
        "per_format_counts":        per_format_counts,
        "out_of_scope_bytes":       out_of_scope,
        "total_estimated_bytes":    total_estimated,
        "original_bytes":           original_bytes,
    }


def print_size_estimate(est: Dict) -> None:
    def gb(b: float) -> str:
        return f"{b / 1024**3:.2f} GB"

    original  = est["original_bytes"]
    total     = est["total_estimated_bytes"]
    delta     = total - original
    delta_pct = delta / original * 100.0
    pfb       = est["per_format_bytes"]
    pfc       = est["per_format_counts"]
    n_scope   = sum(pfc.values())
    sign      = "+" if delta >= 0 else "−"
    abs_delta = abs(delta) / 1024**3

    print()
    print(SEP)
    print("  ESTIMATED OUTPUT FILE SIZE")
    print(SEP)
    print(f"  Tensors in scope ({n_scope})  —  original BF16: {gb(est['in_scope_original_bytes'])}")
    for fmt in ("*KEEP*", "FP8", "NVFP4"):
        label = f"    {fmt} ({'BF16' if fmt == '*KEEP*' else fmt})"
        print(f"  {label:<38}  {pfc[fmt]:>4} tensors  →  {gb(pfb[fmt]):>9}")
    print(f"  {'  Subtotal after quantization':<38}  {gb(est['in_scope_estimated_bytes']):>9}")
    print(f"  {'Tensors out of scope (BF16, unchanged)':<38}  {gb(est['out_of_scope_bytes']):>9}")
    print(f"  {'─' * 60}")
    print(f"  {'Total estimated':<38}  {gb(total):>9}")
    print(f"  {'Original file size':<38}  {gb(original):>9}")
    print(f"  {'Delta':<38}  {sign}{abs_delta:.2f} GB  ({sign}{abs(delta_pct):.1f}%)")
    print(SEP)
