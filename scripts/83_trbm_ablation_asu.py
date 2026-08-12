#!/usr/bin/env python3
"""
Optional offline ablations for TRBM-FLARE-VAX.

Run this ONLY after the full TRBM experiment has completed with:
    run_trbm_flare_vax_asu.py

This script does NOT call the ASU/OpenAI endpoint and does NOT repeat LLM inference.
It reuses calibration/test artifacts produced by the full run and computes selected
ablation baselines separately.

Available ablations
-------------------
1. hbm8_pattern_anchor
   Original empirical HBM8 pattern prior.

2. hbm_theory_prior
   Theory-constrained numerical prior before any residual-memory correction.

3. retrieval_residual_only
   Uses retrieved signed residuals numerically but removes the LLM mechanism gate.
   Its correction scale and threshold are calibrated on the calibration split.

The full method itself is reported by full_results.csv and is intentionally not
recomputed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_trbm_flare_vax_asu as core  # noqa: E402


ALL_ABLATIONS = [
    "hbm8_pattern_anchor",
    "hbm_theory_prior",
    "retrieval_residual_only",
]


def parse_csv_list(value: str) -> List[str]:
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if len(items) == 1 and items[0].lower() == "all":
        return list(ALL_ABLATIONS)
    unknown = sorted(set(items) - set(ALL_ABLATIONS))
    if unknown:
        raise ValueError(
            f"Unknown ablation(s): {unknown}. "
            f"Allowed: {', '.join(ALL_ABLATIONS)} or all"
        )
    if not items:
        raise ValueError("At least one ablation must be selected.")
    return items


def successful_rows(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).str.lower().eq("ok")].copy()
    if frame.empty:
        raise RuntimeError(f"No successful rows found in {label}")
    required = {"source_row_index", "actual", "theory_prior_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {sorted(missing)}")
    return frame.reset_index(drop=True)


def load_pattern_rates(variant_dir: Path) -> Dict[str, float]:
    path = variant_dir / "pattern_base_rates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pattern base-rate file: {path}")
    df = pd.read_csv(path)
    if not {"pattern", "smoothed_rate"}.issubset(df.columns):
        raise RuntimeError(
            f"{path} must contain pattern and smoothed_rate columns."
        )
    return {
        str(row["pattern"]): float(row["smoothed_rate"])
        for _, row in df.iterrows()
    }


def pattern_probabilities(frame: pd.DataFrame, rates: Mapping[str, float]) -> np.ndarray:
    if "hbm8_pattern" not in frame.columns:
        raise RuntimeError("Prediction file is missing hbm8_pattern.")
    overall = float(rates.get("__overall__", np.nan))
    if not np.isfinite(overall):
        usable = [v for k, v in rates.items() if k != "__overall__" and np.isfinite(v)]
        overall = float(np.mean(usable)) if usable else 0.5
    return np.array(
        [float(rates.get(str(p), overall)) for p in frame["hbm8_pattern"]],
        dtype=float,
    )


def load_run_defaults(exp_dir: Path) -> Dict[str, Any]:
    path = exp_dir / "run_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_experiments(root: Path) -> List[Tuple[str, str]]:
    full_results = root / "full_results.csv"
    if full_results.exists() and full_results.stat().st_size > 0:
        df = pd.read_csv(full_results)
        if {"variant", "model"}.issubset(df.columns):
            pairs = (
                df[["variant", "model"]]
                .drop_duplicates()
                .astype(str)
                .itertuples(index=False, name=None)
            )
            return sorted(set(pairs))

    out: List[Tuple[str, str]] = []
    for variant in ["v4", "v5"]:
        vd = root / variant
        if not vd.exists():
            continue
        for child in vd.iterdir():
            if not child.is_dir():
                continue
            if (child / "calibration_predictions.csv").exists() and (child / "test_predictions.csv").exists():
                out.append((variant, child.name))
    return sorted(set(out))


def survey_weights_for_test(root: Path, variant: str, test: pd.DataFrame) -> Optional[np.ndarray]:
    profile_path = root / variant / "selected_hbm_profiles.csv"
    if not profile_path.exists():
        return None
    scores = pd.read_csv(profile_path)
    if "source_row_index" in scores.columns:
        scores = scores.set_index("source_row_index")
    elif scores.columns[0].startswith("Unnamed"):
        scores = scores.set_index(scores.columns[0])
    else:
        # selected_hbm_profiles.csv was written with dataframe index.
        scores = scores.set_index(scores.columns[0])
    if "survey_weight" not in scores.columns:
        return None

    vals = []
    for source in test["source_row_index"].astype(int):
        try:
            vals.append(float(scores.loc[source, "survey_weight"]))
        except Exception:
            vals.append(np.nan)
    sw = np.asarray(vals, dtype=float)
    good = np.isfinite(sw) & (sw > 0)
    if not good.any():
        return None
    fill = float(np.nanmedian(sw[good]))
    return np.where(good, sw, fill)


def add_metric_row(
    rows: List[Dict[str, Any]],
    *,
    variant: str,
    model: str,
    method: str,
    metric_values: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    row = {
        "variant": variant,
        "model": model,
        "method": method,
        **dict(metric_values),
    }
    if extra:
        row.update(dict(extra))
    rows.append(row)


def run_experiment(
    root: Path,
    variant: str,
    model: str,
    ablations: Sequence[str],
    *,
    threshold_metric_override: Optional[str],
    max_scale_override: Optional[float],
    step_override: Optional[float],
    include_survey_weighted: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    exp_dir = root / variant / model
    cal_path = exp_dir / "calibration_predictions.csv"
    test_path = exp_dir / "test_predictions.csv"
    if not cal_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Full-run prediction artifacts not found for {variant}/{model}. "
            "Run run_trbm_flare_vax_asu.py first."
        )

    cal = successful_rows(pd.read_csv(cal_path), f"{variant}/{model}/calibration")
    test = successful_rows(pd.read_csv(test_path), f"{variant}/{model}/test")
    cfg = load_run_defaults(exp_dir)

    threshold_metric = (
        threshold_metric_override
        or str(cfg.get("threshold_metric", "balanced_accuracy"))
    )
    if threshold_metric not in {"balanced_accuracy", "f1", "accuracy"}:
        raise ValueError(f"Unsupported threshold metric: {threshold_metric}")

    max_scale = float(
        max_scale_override
        if max_scale_override is not None
        else cfg.get("max_correction_scale", 1.50)
    )
    step = float(
        step_override
        if step_override is not None
        else cfg.get("correction_scale_step", 0.05)
    )

    yc = cal["actual"].astype(int).to_numpy()
    yt = test["actual"].astype(int).to_numpy()
    pc = cal["theory_prior_probability"].astype(float).to_numpy()
    pt = test["theory_prior_probability"].astype(float).to_numpy()

    ab_dir = exp_dir / "ablation"
    ab_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {
        "variant": variant,
        "model": model,
        "threshold_metric": threshold_metric,
        "ablations": {},
    }

    if "hbm8_pattern_anchor" in ablations:
        rates = load_pattern_rates(root / variant)
        pcal = pattern_probabilities(cal, rates)
        ptest = pattern_probabilities(test, rates)
        threshold, table = core.calibrate_threshold(yc, pcal, threshold_metric)
        table.to_csv(ab_dir / "threshold_search_hbm8_pattern_anchor.csv", index=False)
        m = core.metrics(yt, ptest, threshold)
        add_metric_row(
            rows,
            variant=variant,
            model=model,
            method="hbm8_pattern_anchor",
            metric_values=m,
            extra={"selected_scale": np.nan},
        )
        details["ablations"]["hbm8_pattern_anchor"] = {
            "threshold": threshold,
            "metrics": m,
        }

    if "hbm_theory_prior" in ablations:
        threshold, table = core.calibrate_threshold(yc, pc, threshold_metric)
        table.to_csv(ab_dir / "threshold_search_hbm_theory_prior.csv", index=False)
        m = core.metrics(yt, pt, threshold)
        add_metric_row(
            rows,
            variant=variant,
            model=model,
            method="hbm_theory_prior",
            metric_values=m,
            extra={"selected_scale": np.nan},
        )
        details["ablations"]["hbm_theory_prior"] = {
            "threshold": threshold,
            "metrics": m,
        }

        if include_survey_weighted:
            sw = survey_weights_for_test(root, variant, test)
            if sw is not None:
                mw = core.metrics(yt, pt, threshold, sw)
                add_metric_row(
                    rows,
                    variant=variant,
                    model=model,
                    method="hbm_theory_prior_survey_weighted",
                    metric_values=mw,
                    extra={"selected_scale": np.nan},
                )
                details["ablations"]["hbm_theory_prior_survey_weighted"] = {
                    "threshold": threshold,
                    "metrics": mw,
                }

    if "retrieval_residual_only" in ablations:
        needed = "retrieval_only_raw_correction"
        if needed not in cal.columns or needed not in test.columns:
            raise RuntimeError(
                f"{variant}/{model} prediction files do not contain {needed}. "
                "Use the revised full-first TRBM driver, which stores this intermediate."
            )
        rc = cal[needed].astype(float).to_numpy()
        rt = test[needed].astype(float).to_numpy()

        scale, scale_table = core.select_correction_scale(
            yc, pc, rc, max_scale, step
        )
        pcal = core.apply_scale(pc, rc, scale)
        ptest = core.apply_scale(pt, rt, scale)
        threshold, threshold_table = core.calibrate_threshold(
            yc, pcal, threshold_metric
        )
        scale_table.to_csv(
            ab_dir / "correction_scale_search_retrieval_residual_only.csv",
            index=False,
        )
        threshold_table.to_csv(
            ab_dir / "threshold_search_retrieval_residual_only.csv",
            index=False,
        )
        m = core.metrics(yt, ptest, threshold)
        add_metric_row(
            rows,
            variant=variant,
            model=model,
            method="retrieval_residual_only",
            metric_values=m,
            extra={"selected_scale": scale},
        )
        details["ablations"]["retrieval_residual_only"] = {
            "selected_scale": scale,
            "threshold": threshold,
            "metrics": m,
        }

    return rows, details


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Optional OFFLINE TRBM ablations. Requires a completed full run; "
            "does not call the LLM API."
        )
    )
    p.add_argument(
        "--full-output-dir",
        required=True,
        help="Output directory previously created by run_trbm_flare_vax_asu.py",
    )
    p.add_argument(
        "--ablations",
        default="all",
        help=(
            "Comma-separated subset or 'all'. Choices: "
            + ",".join(ALL_ABLATIONS)
        ),
    )
    p.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated filter, e.g. v4 or v4,v5",
    )
    p.add_argument(
        "--models",
        default="",
        help="Optional comma-separated model-name filter",
    )
    p.add_argument(
        "--threshold-metric",
        choices=["balanced_accuracy", "f1", "accuracy"],
        default=None,
        help="Override the metric saved by the full run.",
    )
    p.add_argument("--max-correction-scale", type=float, default=None)
    p.add_argument("--correction-scale-step", type=float, default=None)
    p.add_argument(
        "--include-survey-weighted",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally add survey-weighted theory-prior metrics.",
    )
    p.add_argument(
        "--output-csv",
        default="",
        help="Default: <full-output-dir>/ablation_results.csv",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.full_output_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    ablations = parse_csv_list(args.ablations)
    experiments = discover_experiments(root)
    if not experiments:
        raise RuntimeError(
            "No completed full TRBM experiments were discovered. "
            "Run run_trbm_flare_vax_asu.py first."
        )

    variant_filter = {
        x.strip() for x in args.variants.split(",") if x.strip()
    }
    model_filter = {
        x.strip() for x in args.models.split(",") if x.strip()
    }
    if variant_filter:
        experiments = [x for x in experiments if x[0] in variant_filter]
    if model_filter:
        experiments = [x for x in experiments if x[1] in model_filter]
    if not experiments:
        raise RuntimeError("No experiments remain after applying filters.")

    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for variant, model in experiments:
        print(f"[ablation] {variant}/{model} -> {', '.join(ablations)}")
        rr, detail = run_experiment(
            root,
            variant,
            model,
            ablations,
            threshold_metric_override=args.threshold_metric,
            max_scale_override=args.max_correction_scale,
            step_override=args.correction_scale_step,
            include_survey_weighted=args.include_survey_weighted,
        )
        rows.extend(rr)
        summaries.append(detail)

    out_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else root / "ablation_results.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "variant", "model", "method", "selected_scale",
        "n", "threshold", "accuracy", "balanced_accuracy",
        "precision", "recall", "specificity", "f1", "roc_auc",
        "average_precision", "brier", "log_loss",
    ]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(out_csv, index=False)
    (out_csv.parent / "ablation_summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nSaved: {out_csv}")
    if rows:
        print(pd.DataFrame(rows).reindex(columns=cols).to_string(index=False))


if __name__ == "__main__":
    main()
