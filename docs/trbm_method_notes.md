# TRBM Method Notes

## Why this version changes the role of the LLM

The previous FLARE-VAX / RG-FLARE-VAX variants allowed the LLM to produce a residual probability adjustment. TRBM intentionally removes that responsibility. The design assumes that a frozen LLM is useful for semantic mechanism identification but may be unreliable as a calibrated tabular probability estimator.

The division of labor is therefore:

- **Behavioral theory:** defines which latent/construct-level quantities are allowed to create the base prior.
- **Numerical ML:** estimates the base probability from those theory-defined constructs.
- **LLM:** interprets why a respondent may depart from that prior and identifies a reusable mechanism.
- **Memory:** supplies historical evidence about how large such departures were in comparable cases.
- **Calibration split:** selects one global correction scale and classification thresholds.

## Why the theory prior is not the full-feature ML baseline

The TRBM prior deliberately does **not** receive all raw NHIS features. If a full-feature XGBoost/LightGBM model were used as the prior, the residual would primarily be an ML-model residual rather than a behavioral-theory residual.

The main TRBM prior uses only five positively oriented HBM-derived proxy dimensions and non-negative coefficients. A full-feature ML model should remain an external upper/comparison baseline.

## Exact V4/V5 distinction

### V4

The second theory feature is the existing `vaccine_acceptance_benefit_proxy`, which uses non-target vaccination behavior as an observed proxy of vaccine acceptance/preventive belief. V4 profiles and residual explanations can therefore use prior vaccine behavior.

### V5

The second theory feature is `preventive_engagement_proxy`. The supplied V5 construction excludes other-vaccine-history variables from scoring, observed profiles, retrieval, and prompts. The TRBM code preserves that boundary.

## Leakage controls

1. Memory/calibration/test splits remain separate.
2. Theory residuals for memory cases are based on out-of-fold predictions.
3. Only memory-split outcomes are used to create residual memories.
4. Calibration outcomes are used only for correction-scale/threshold selection.
5. Test outcomes are used only for final evaluation.
6. The test-time LLM never receives the target outcome.
7. The LLM never chooses the numerical correction magnitude.

## Residual definition

For memory respondent i:

`r_i = y_i - p_i^OOF`

- large positive residual: theory prior underestimated vaccination;
- large negative residual: theory prior overestimated vaccination.

The default reflection threshold is `|r_i| >= 0.40`, followed by balanced selection across residual direction and HBM8 pattern.

## Retrieval logic

TRBM separates theory similarity from contextual similarity. This is intentional.

A useful residual memory should answer two questions:

1. Did the earlier respondent occupy a similar HBM/theory state?
2. Does the target show similar observed contextual evidence that could explain the earlier theory failure?

The default score is:

`0.55 * theory_similarity + 0.35 * context_similarity + 0.10 * same_pattern`

Memory strength is used only as a small secondary ranking term.

## LLM outputs

### Reflection

The LLM must return a mechanism family plus a concrete rule and applicability/contradiction conditions. It is not allowed to output a probability.

### Gate

The LLM chooses `increase`, `decrease`, or `none`, and selects supporting memory IDs. It cannot invent new memory IDs and cannot choose a numerical delta.

## Numerical correction

For the LLM-selected memories, TRBM computes a similarity/confidence-weighted mean of historical signed residuals, shrinks the correction for small support, applies a weak/moderate/strong multiplier chosen by the LLM, and caps the unscaled correction.

The calibration split then selects a single global scale alpha by log loss. This keeps the LLM out of probability calibration.

## Full-first execution policy

The main `run_trbm_flare_vax_asu.py` driver now evaluates only the **complete TRBM** and writes `full_results.csv`.

The existing ablation comparisons have been removed from the automatic main-run report. They are implemented in `run_trbm_ablation_asu.py`, which is intentionally offline and optional.

The ablation script reuses artifacts already produced by the full pipeline:

- calibration/test theory-prior probabilities,
- HBM8 pattern assignments,
- retrieval-only raw residual corrections,
- pattern base rates.

Therefore the later ablation stage does not require extra ASU/OpenAI calls.

## Important ablations

The package automatically reports:

1. HBM8 pattern anchor
2. constrained HBM theory prior
3. residual retrieval without LLM gating
4. full LLM-gated TRBM

For a paper, useful additional ablations would be:

- unconstrained vs sign-constrained HBM prior;
- theory-only similarity vs context-only similarity;
- random residual memories vs theory-matched residual memories;
- residual memory with mechanism text removed;
- V4 vs V5 to isolate revealed prior-vaccine behavior.
