"""
quant_probe CLI — entry point.

Parses arguments, resolves model-specific defaults, and orchestrates the
analysis pipeline defined in core.py.  Contains no analytical logic.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

from quant_probe.registry import MODEL_CONFIGS
from quant_probe import core


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantization sensitivity analysis for safetensors models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models:
  wan     - Wan 2.1 video diffusion model
  zimage  - Z-Image / Z-Image Turbo image diffusion model

Score composition:
  - Excess kurtosis (IQR-normalized globally)       default weight: 0.6
  - Dynamic range (within-type min-max normalized)  default weight: 0.3
  - Aspect ratio  (within-type min-max normalized)  default weight: 0.1
  Outlier% is shown in tables but excluded from score (correlated with kurtosis).

Thresholds and extreme block ranges are derived automatically from the model.

Default thresholds are calibrated per model:
  wan    - calibrated from Wan 2.1 14B score distribution
  zimage - calibrated empirically from Z-Image Turbo score distribution
""",
    )
    parser.add_argument("model_path", type=str,
                        help="Path to the BF16 safetensors model file.")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model architecture to analyze.")
    parser.add_argument("--csv", type=str, default=None, metavar="OUTPUT.csv",
                        help="Export per-tensor metrics to a CSV file.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for computation: 'cpu' or 'cuda' (default: auto-detect).")
    parser.add_argument("--outlier-sigma", type=float, default=3.0, metavar="N",
                        help="Standard deviation multiplier for outlier detection (default: 3.0).")
    parser.add_argument("--fp8-percentile", type=float, default=None, metavar="P",
                        help="Score percentile threshold for FP8 recommendation (default: model-specific).")
    parser.add_argument("--keep-percentile", type=float, default=None, metavar="P",
                        help="Score percentile threshold for *KEEP* recommendation (default: model-specific).")
    parser.add_argument("--extreme-pct", type=float, default=10.0, metavar="P",
                        help="Percentage of blocks to consider extreme at each end (default: 10).")
    parser.add_argument("--kurtosis-weight", type=float, default=0.6, metavar="W",
                        help="Weight for IQR-normalized excess kurtosis in combined score (default: 0.6).")
    parser.add_argument("--range-weight", type=float, default=0.3, metavar="W",
                        help="Weight for within-type dynamic range in combined score (default: 0.3).")
    parser.add_argument("--ar-weight", type=float, default=0.1, metavar="W",
                        help="Weight for within-type aspect ratio in combined score (default: 0.1).")
    parser.add_argument("--fp8-min-score", type=float, default=None, metavar="S",
                        help="Absolute minimum score required for FP8 (default: model-specific).")
    parser.add_argument("--min-group-spread", type=float, default=None, metavar="S",
                        help="Minimum score spread across block position groups for per-group FP8 (default: model-specific).")
    parser.add_argument("--spread-filter-exempt", nargs="*", default=[],
                        metavar="LAYER_TYPE",
                        help=(
                            "Layer types that bypass the spread filter, always using "
                            "their per-tensor individual recommendation.  "
                            "Strongly recommended for attention Q/K layers due to DiT "
                            "QK^T sensitivity.  "
                            "Recommended values — "
                            "wan: cross_attn.k cross_attn.q self_attn.k self_attn.q  |  "
                            "zimage: attention.qkv attention.out"
                        ))
    parser.add_argument("--kurtosis-keep", type=float, default=None, metavar="K",
                        help="Hard floor on excess kurtosis: forces *KEEP* (default: model-specific).")
    parser.add_argument("--lowram", action="store_true", default=False,
                        help="Avoid mmap-ing the full model file.")
    return parser


# ---------------------------------------------------------------------------
# Default resolver
# ---------------------------------------------------------------------------

def _resolve(cli_val, key: str, defaults: Dict) -> float:
    """Return CLI value if provided, otherwise the model-specific default."""
    if cli_val is not None:
        return cli_val
    return defaults[key]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Validate input file
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    # Validate score weights
    total_weight = args.kurtosis_weight + args.range_weight + args.ar_weight
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Error: --kurtosis-weight + --range-weight + --ar-weight must sum to 1.0 "
              f"(got {total_weight:.4f})")
        sys.exit(1)

    # Load architecture config
    cfg      = MODEL_CONFIGS[args.model]
    defaults = cfg.default_args

    # Resolve model-specific defaults
    fp8_percentile   = _resolve(args.fp8_percentile,   "fp8_percentile",   defaults)
    keep_percentile  = _resolve(args.keep_percentile,  "keep_percentile",  defaults)
    kurtosis_keep    = _resolve(args.kurtosis_keep,    "kurtosis_keep",    defaults)
    fp8_min_score    = _resolve(args.fp8_min_score,    "fp8_min_score",    defaults)
    min_group_spread = _resolve(args.min_group_spread, "min_group_spread", defaults)

    # Validate spread_filter_exempt layer types
    all_layer_types: Set[str] = set(cfg.layer_patterns.keys())
    if cfg.refiner_patterns:
        all_layer_types |= set(cfg.refiner_patterns.keys())
    for lt in args.spread_filter_exempt:
        if lt not in all_layer_types:
            print(f"Error: --spread-filter-exempt: unknown layer type '{lt}' for model '{args.model}'.")
            print(f"       Valid types: {', '.join(sorted(all_layer_types))}")
            sys.exit(1)

    # Warn if recommended attention layers are missing from spread_filter_exempt
    exempt_set = set(args.spread_filter_exempt)
    missing    = [lt for lt in cfg.spread_filter_recommended if lt not in exempt_set]
    if missing:
        print()
        print(f"  *** WARNING: the following attention layer types are NOT in")
        print(f"  *** --spread-filter-exempt and may produce suboptimal recommendations")
        print(f"  *** due to DiT QK^T sensitivity:")
        print(f"  ***   {' '.join(missing)}")
        print(f"  *** Consider re-running with:")
        print(f"  ***   --spread-filter-exempt {' '.join(cfg.spread_filter_recommended)}")

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        props      = torch.cuda.get_device_properties(device)
        vram_mb    = props.total_memory // (1024 * 1024)
        device_str = f"CUDA ({props.name}, {vram_mb} MB)"
    else:
        device_str = "CPU"

    exempt_str = ", ".join(sorted(exempt_set)) if exempt_set else "none"
    print()
    print(f"  Quantization Sensitivity Analysis — {model_path.name}")
    print(f"  Model: {args.model}  ({cfg.description})")
    print(f"  Device: {device_str}")
    print(f"  Outlier sigma: {args.outlier_sigma}  |  "
          f"FP8 percentile: {fp8_percentile}  |  "
          f"Keep percentile: {keep_percentile}  |  "
          f"FP8 min score: {fp8_min_score}  |  "
          f"Min group spread: {min_group_spread}  |  "
          f"Spread filter exempt: {exempt_str}  |  "
          f"Extreme blocks: {args.extreme_pct}%  |  "
          f"Kurtosis keep: {kurtosis_keep}  |  "
          f"Low-RAM: {'yes' if args.lowram else 'no'}")
    print()

    if cfg.highprec_excluded:
        core.print_highprec_notice(cfg.highprec_excluded, cfg.convert_flag)

    print()
    print("  Scanning model...")

    # -----------------------------------------------------------------------
    # Analysis pass
    # -----------------------------------------------------------------------
    all_metrics = core.analyze_model(
        str(model_path),
        device,
        cfg,
        args.outlier_sigma,
        low_ram=args.lowram,
    )

    if not all_metrics:
        print("Error: no target tensors found in model.")
        sys.exit(1)

    # Split metrics by subgraph
    main_metrics    = [m for m in all_metrics if m.subgraph in cfg.main_subgraphs]
    refiner_metrics = [m for m in all_metrics if m.subgraph in cfg.refiner_subgraphs]

    total_blocks = max(m.block_idx for m in main_metrics) + 1 if main_metrics else 0

    print(f"  Tensors analyzed: {len(all_metrics)} | Main blocks: {total_blocks}"
          + (f" | Refiner tensors: {len(refiner_metrics)}" if refiner_metrics else ""))

    # AR inactive check
    by_type_check: Dict = defaultdict(list)
    for m in all_metrics:
        by_type_check[m.layer_type].append(m)
    ar_inactive = all(
        max(m.aspect_ratio for m in g) == min(m.aspect_ratio for m in g)
        for g in by_type_check.values()
    )
    ar_note = " [inactive: no within-type AR variation — effective budget: 0.9]" if ar_inactive else ""
    print(f"  Score weights — kurtosis (IQR): {args.kurtosis_weight}  |  "
          f"dyn. range: {args.range_weight}  |  "
          f"aspect ratio: {args.ar_weight}{ar_note}")

    if total_blocks > 0:
        extreme_low, extreme_high = core.compute_extreme_ranges(total_blocks, args.extreme_pct)
        print(f"  Extreme blocks — low: {extreme_low[0]}–{extreme_low[1]}  |  "
              f"high: {extreme_high[0]}–{extreme_high[1]}")

    # -----------------------------------------------------------------------
    # Scoring and recommendations
    # -----------------------------------------------------------------------
    core.compute_scores(all_metrics, args.kurtosis_weight, args.range_weight, args.ar_weight)

    fp8_threshold, keep_threshold = core.compute_auto_thresholds(
        all_metrics, fp8_percentile, keep_percentile
    )
    if fp8_min_score > fp8_threshold:
        print(f"  Auto thresholds — FP8: score >= {fp8_min_score:.3f} "
              f"[fp8-min-score active; percentile gave {fp8_threshold:.3f}]  |  "
              f"*KEEP*: score >= {keep_threshold:.3f}")
    else:
        print(f"  Auto thresholds — FP8: score >= {fp8_threshold:.3f} [percentile]  |  "
              f"*KEEP*: score >= {keep_threshold:.3f}")

    for m in all_metrics:
        m.recommendation, m.reason = core.assign_recommendation(
            m.score, fp8_threshold, keep_threshold, fp8_min_score,
            excess_kurtosis=m.excess_kurtosis,
            kurtosis_keep=kurtosis_keep,
        )

    # -----------------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------------
    by_layer: Dict = defaultdict(list)
    for m in main_metrics:
        by_layer[m.layer_type].append(m)

    summary_aggs = []
    for layer_type in cfg.layer_patterns.keys():
        group = by_layer.get(layer_type, [])
        if not group:
            continue
        block_indices = [m.block_idx for m in group]
        agg = core.aggregate(group, layer_type,
                             core.block_range_label(block_indices, total_blocks),
                             group[0].subgraph)
        agg.recommendation, agg.reason = core.assign_recommendation(
            agg.score, fp8_threshold, keep_threshold, fp8_min_score,
            excess_kurtosis=agg.kurtosis_max,
            kurtosis_keep=kurtosis_keep,
        )
        summary_aggs.append(agg)

    core.print_summary_table(summary_aggs, title_suffix="main blocks")

    refiner_summary_aggs = []
    if refiner_metrics and cfg.refiner_patterns:
        by_refiner_layer: Dict = defaultdict(list)
        for m in refiner_metrics:
            by_refiner_layer[m.layer_type].append(m)

        for layer_type in cfg.refiner_patterns.keys():
            group = by_refiner_layer.get(layer_type, [])
            if not group:
                continue
            block_indices   = [m.block_idx for m in group]
            total_refiner   = max(block_indices) + 1
            agg = core.aggregate(group, layer_type,
                                 core.block_range_label(block_indices, total_refiner),
                                 group[0].subgraph)
            agg.recommendation, agg.reason = core.assign_recommendation(
                agg.score, fp8_threshold, keep_threshold, fp8_min_score,
                excess_kurtosis=agg.kurtosis_max,
                kurtosis_keep=kurtosis_keep,
            )
            refiner_summary_aggs.append(agg)

        core.print_summary_table(refiner_summary_aggs,
                                 title_suffix="context_refiner + noise_refiner")

    # -----------------------------------------------------------------------
    # Detail tables — main blocks
    # -----------------------------------------------------------------------
    all_detail_rows = []
    detail_rows_by_group: Dict = {}

    if total_blocks > 0:
        for group_name, layer_types in cfg.detail_groups.items():
            group_rows = []

            for layer_type in layer_types:
                group = by_layer.get(layer_type, [])
                if not group:
                    continue

                for position, subset in [
                    ("extreme_low",  [m for m in group if core.classify_block(m.block_idx, extreme_low, extreme_high) == "extreme_low"]),
                    ("middle",       [m for m in group if core.classify_block(m.block_idx, extreme_low, extreme_high) == "middle"]),
                    ("extreme_high", [m for m in group if core.classify_block(m.block_idx, extreme_low, extreme_high) == "extreme_high"]),
                ]:
                    if not subset:
                        continue
                    indices = [m.block_idx for m in subset]
                    label   = core.block_range_label(indices, total_blocks)
                    agg     = core.aggregate(subset, layer_type, label, subset[0].subgraph)
                    agg.recommendation, agg.reason = core.assign_recommendation(
                        agg.score, fp8_threshold, keep_threshold, fp8_min_score,
                        excess_kurtosis=agg.kurtosis_max,
                        kurtosis_keep=kurtosis_keep,
                    )
                    group_rows.append(agg)

            if group_rows:
                detail_rows_by_group[group_name] = group_rows
                all_detail_rows.extend(group_rows)

    # -----------------------------------------------------------------------
    # Detail tables — refiners
    # -----------------------------------------------------------------------
    refiner_detail_by_group: Dict = {}

    if refiner_metrics and cfg.refiner_detail_groups:
        by_refiner_layer: Dict = defaultdict(list)
        for m in refiner_metrics:
            by_refiner_layer[m.layer_type].append(m)

        for group_name, layer_types in cfg.refiner_detail_groups.items():
            group_rows = []
            for layer_type in layer_types:
                group = by_refiner_layer.get(layer_type, [])
                if not group:
                    continue
                block_indices = [m.block_idx for m in group]
                total_r       = max(block_indices) + 1
                label         = core.block_range_label(block_indices, total_r)
                agg           = core.aggregate(group, layer_type, label, group[0].subgraph)
                agg.recommendation, agg.reason = core.assign_recommendation(
                    agg.score, fp8_threshold, keep_threshold, fp8_min_score,
                    excess_kurtosis=agg.kurtosis_max,
                    kurtosis_keep=kurtosis_keep,
                )
                group_rows.append(agg)

            if group_rows:
                refiner_detail_by_group[group_name] = group_rows
                all_detail_rows.extend(group_rows)

    # -----------------------------------------------------------------------
    # Suggested parameters (must run before printing detail tables so
    # spread_filtered flags are set)
    # -----------------------------------------------------------------------
    fp8_entries, keep_entries = core.build_convert_to_quant_params(
        all_detail_rows, all_metrics,
        min_group_spread,
        spread_filter_exempt=exempt_set,
    )

    # Print detail tables
    for group_name, group_rows in detail_rows_by_group.items():
        core.print_detail_table(group_name, group_rows, title_suffix="main blocks")

    if refiner_metrics and cfg.refiner_detail_groups:
        for group_name, group_rows in refiner_detail_by_group.items():
            core.print_detail_table(group_name, group_rows,
                                    title_suffix="refiners — no positional classification")

    core.print_suggested_params(
        fp8_entries, keep_entries,
        fp8_min_score, min_group_spread,
        cfg,
    )

    # -----------------------------------------------------------------------
    # Effective recommendation, CSV export, size estimate
    # -----------------------------------------------------------------------
    effective_rec, effective_reason = core.build_effective_rec(
        all_detail_rows, all_metrics,
        spread_filter_exempt=exempt_set,
    )

    if args.csv:
        core.export_csv(all_metrics, args.csv, effective_rec, effective_reason)

    original_bytes = model_path.stat().st_size
    size_est       = core.estimate_output_size(all_metrics, effective_rec, original_bytes)
    core.print_size_estimate(size_est)


if __name__ == "__main__":
    main()
