# GitHub update summary

This revised public package preserves the previous FLARE-VAX V4/V5 code, baselines, transferred methods, documentation, and compact result tables, while adding two completed V4/V5 extensions.

## Added methods

### RG-FLARE-VAX
- Added `scripts/81_rg_flare_vax_reward_memory_asu.py`.
- Reuses the V4/V5 HBM construction and feature boundaries.
- Adds a numerical reward-calibrated HBM integration layer.
- Adds five-fold OOF reward fitting for memory respondents.
- Adds optional LLM-guided sparse reward-weight revision with held-out acceptance.
- Adds empirical directional Q-values to reflective memories and Q-aware retrieval.
- Keeps the LLM frozen and bounds final residual corrections.

### TRBM-FLARE-VAX
- Added `scripts/82_trbm_flare_vax_asu.py`.
- Added `scripts/83_trbm_ablation_asu.py` for offline ablations.
- Added `docs/trbm_method_notes.md`.
- Replaces LLM base-probability estimation with a non-negative theory-constrained HBM logistic prior.
- Builds memory from out-of-fold signed theory residuals.
- Uses the LLM only for mechanism explanation and applicability/direction gating.
- Computes correction magnitude from historical residuals and calibrates one global scale.

## Added full-run results

- `results/reward_memory/benchmark_results_public.csv`
- `results/trbm/full_results_public.csv`
- `results/all_results_public.csv` regenerated to include both extensions.

Key full-run observations:
- RG V4 final reward-valued memory reaches ROC-AUC 0.8221 with Llama 4 Scout 17B and 0.8210 with Llama 3 70B.
- RG V5 final reward-valued memory reaches ROC-AUC 0.7204 / 0.7208.
- TRBM V4 reaches ROC-AUC 0.8205 and V5 0.7200 in the unweighted full-run rows.
- TRBM calibration selected `alpha = 0.00` in all supplied full runs, so its final reported predictions are effectively the theory-constrained prior; this is explicitly documented rather than hidden.

Row-level predictions, JSONL logs, notebooks with execution state, and raw NHIS data remain excluded from the public update.
