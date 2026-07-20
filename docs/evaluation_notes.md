# Evaluation Notes

## 1. Two feature-policy settings

The repository separates results into:

- **Without other vaccine history:** no non-target vaccine variable may enter the model.
- **With other vaccine history:** prior COVID-19, pneumonia, shingles, and hepatitis A vaccination variables may be used.

This distinction is essential because other vaccination behavior is a strong predictor of flu vaccination and can dominate the comparison.

## 2. Why the HBM2 development runs are not directly comparable

| Run | Sampling | Test size | Output type |
|---|---|---:|---|
| V1 smoke test | balanced | 6 | binary |
| V1-Async | balanced | 300 | binary |
| V2 reflection | natural/proportional | 300 | probability + calibrated class |
| V3 dual memory | proportional, three-way split | 250 | pattern anchor + probability |

Changes in sample composition, split size, and class balance can move accuracy, F1, and calibration independently of the method.

## 3. Current 5,000-person findings

### With other vaccine history

- Pattern-only AUC: 0.7658
- LLM with memory AUC: 0.7332
- XGBoost AUC: 0.8452

### Without other vaccine history

- Pattern-only AUC: 0.6812
- LLM without memory AUC: 0.6584
- LLM with memory AUC: 0.5808
- XGBoost AUC: 0.7570

## 4. What the memory results imply

The current evidence does not show that reflection memory improves generalization.

Potential explanations:

1. Only six memories survived distillation in the no-prior run.
2. Similarity in observed variables may not imply similarity in unobserved vaccination attitudes.
3. A correction rule derived from one error may overcorrect another respondent.
4. The LLM may overweight retrieved exceptions relative to the stable pattern prior.
5. Calibration was performed with memory, but the memory itself may shift the score distribution inconsistently.

## 5. Recommended controlled benchmark

For a final comparison:

1. Save one fixed 5,000-person sample and one fixed memory/calibration/test split.
2. Reuse those row indices for every HBM, LLM, and ML method.
3. Use the same feature-policy definition in every model.
4. Report accuracy, balanced accuracy, F1, AUC, average precision, Brier score, and log loss.
5. Add bootstrap confidence intervals and paired tests on identical test predictions.
6. Compare LLM probability against pattern probability at the respondent level.
7. Evaluate whether each memory item improves its nearest held-out neighbors before allowing it into final memory.

## 6. Survey-analysis caution

These experiments evaluate individual-level prediction. They do not estimate national vaccination prevalence. Population inference would require correct use of NHIS weights, strata, and primary sampling units.
