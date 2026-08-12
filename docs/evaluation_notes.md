# Evaluation notes

## Feature-policy comparison

- **V4** permits prior non-target vaccination history.
- **V5** excludes all non-target vaccination history from ML features, prompts, retrieval, and memory.

V4 and V5 should therefore be interpreted as distinct information settings rather than as interchangeable model variants.

## Thresholding

LLM methods output a probability or score. The classification threshold is selected on the calibration split and then frozen for the test split. Public tables report both ROC-AUC and threshold-dependent metrics.

## Pending rows

Blank metrics are deliberate. They identify configurations withheld for a unified rerun or not yet implemented; they do not represent zero performance.

## Coverage caveat

The V5 Llama 4 FLARE-VAX summary contains 11,057 evaluated test cases rather than the configured 12,852. This row is retained with an explicit status label and should not be treated as a full-coverage comparison.

## Survey-analysis caution

These experiments evaluate respondent-level prediction. They do not estimate national vaccination prevalence. Population inference would require the NHIS survey weights, strata, and primary sampling units to be handled according to the survey design.
