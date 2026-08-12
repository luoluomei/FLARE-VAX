# FLARE-VAX: Theory-Guided Influenza Vaccination Prediction from NHIS 2024

[中文说明](README_zh.md)

FLARE-VAX studies whether behavioral-theory structure can improve prediction of influenza vaccination from the **2024 National Health Interview Survey (NHIS) Sample Adult file**. The working raw file used in development contains approximately **32,629 respondents and 630 variables**. The prediction target is:

```text
SHTFLU12M_A: influenza vaccination during the past 12 months
```

After target and feature-policy filtering, V4 evaluates 32,132 respondents and V5 evaluates 32,130. The default split is **40% memory-build / 20% calibration / 40% test**. Raw NHIS data and respondent-level API logs are not redistributed.

> HBM quantities in this repository are observable, theory-guided proxies built from NHIS variables; they are not direct psychometric measurements of private beliefs.

## 1. Dataset and prediction setting

The Sample Adult file combines demographics, health status, chronic conditions, insurance and affordability, healthcare access, utilization, digital health engagement, and vaccination history. Conventional ML uses the selected raw/engineered features directly. LLM methods receive human-readable respondent profiles under the same V4 or V5 feature restrictions.

- **V4 — with other vaccination history:** 75 baseline features. Prior COVID-19, pneumonia, shingles/Shingrix, and hepatitis-A vaccination variables may be used.
- **V5 — without other vaccination history:** 67 baseline features. All non-target vaccination-history variables are excluded from scoring, prompts, retrieval, and memory construction.

## 2. FLARE-VAX V4 and V5

Both versions construct five deterministic HBM-inspired proxies: observed threat, acceptance/benefit or preventive engagement, structural barriers, healthcare cues, and navigation self-efficacy. They are collapsed into Motivation, Capability, and Activation; High/Low states define eight behavioral patterns. Pattern priors are estimated only from the memory-build split. Reflective rules are learned only from training-side errors, then frozen before calibration and test.

### V4 — With Other Vaccination History

**Code:** `scripts/40_flare_vax_v4_with_vaccine_history.py`

V4 uses non-target vaccination behavior as part of the acceptance/benefit proxy. The LLM receives the deterministic profile, pattern prior, observed context, and retrieved reflective rules, and makes one residual probability decision per respondent.

### V5 — Without Other Vaccination History

**Code:** `scripts/50_flare_vax_v5_without_vaccine_history.py`

V5 removes the strongest vaccination-history shortcut. Its acceptance signal is replaced by non-vaccine preventive engagement, including wellness behavior, health-information use, clinician communication, and result-review behavior. The remaining architecture is aligned with V4.

## 3. Two V4/V5 extensions added in this update

The two new methods below are **not new feature versions**. Both keep the original V4/V5 data boundary and deterministic HBM proxy construction. What changes is the layer that turns the HBM state into a probability and the way reflective memory is learned and used.

### 3.1 RG-FLARE-VAX — Reward-Guided HBM Integration + Reward-Valued Memory

**Code:** `scripts/81_rg_flare_vax_reward_memory_asu.py`

RG-FLARE-VAX keeps the original HBM8 pattern prior, but adds a small external numerical reward layer that learns how the five HBM constructs should adjust the pattern anchor. Memory-split respondents are scored with five-fold out-of-fold prediction so the same respondent is not used both to fit and evaluate its reward prior. The frozen LLM may propose sparse, pattern-specific reward-weight revisions, but a proposed revision is accepted only when it improves held-out out-of-fold log loss.

The second change is the reflective-memory mechanism. High-confidence errors are converted into reusable rules, and each rule receives an empirical directional **Q-value** measuring whether applying its correction direction historically reduced loss among similar memory respondents. Test-time retrieval therefore combines respondent similarity, HBM8 pattern match, prior memory quality, and the empirical Q-value. The final LLM is anchored to the reward-calibrated HBM prior and can make only a bounded residual correction (default ±15 percentage points).

In short, relative to the original V4/V5 pipeline, RG-FLARE-VAX changes **how HBM evidence is numerically integrated** and **how reflective memories are valued**, while leaving the LLM frozen and preserving the original V4/V5 feature restrictions. It is SILIC-inspired reward learning for cross-sectional choice prediction, not sequential IRL.

### 3.2 TRBM-FLARE-VAX — Theory-Residual Behavioral Memory

**Code:** `scripts/82_trbm_flare_vax_asu.py`  
**Optional offline ablations:** `scripts/83_trbm_ablation_asu.py`  
**Method notes:** `docs/trbm_method_notes.md`

TRBM makes a stronger change to the division of labor. Instead of asking the LLM to estimate or directly adjust the base vaccination probability, a small **theory-constrained logistic model** first estimates `P_HBM` using only the five positively oriented HBM-derived constructs. The model does not receive the full raw NHIS feature set, and its construct coefficients are constrained to be non-negative so the base probability remains interpretable as a theory-derived prior rather than a full-feature ML predictor.

Memory is then built around **theory residuals**. For each memory respondent, TRBM computes an out-of-fold residual `actual - P_HBM_OOF`. Large positive or negative residuals identify cases where the HBM prior systematically under- or over-predicts behavior. The LLM is used only to explain those failures as reusable mechanisms; it is explicitly prohibited from producing a probability or numerical correction.

At calibration/test time, retrieval combines similarity in HBM/theory state with similarity in observed context. The LLM only decides whether a retrieved mechanism applies and whether it points up, down, or nowhere. The numerical correction itself is computed from the historical signed residuals of the selected memories, and a single global correction scale `alpha` is selected on the calibration split and then frozen.

Relative to the original V4/V5 method, TRBM therefore moves probability estimation out of the LLM and uses the LLM as a **semantic mechanism gate over empirical theory failures**. This version is useful for testing whether LLMs add value through mechanism recognition even when calibration and correction magnitudes are handled numerically.


## 4. Zero-shot and few-shot LLM baselines

**Code:** `scripts/60_llm_icl_benchmark_asu.py`

These baselines provide no HBM scores, HBM pattern, pattern prior, reflective memory, or FLARE correction rule.

| Method | Demonstration policy | Main distinction |
|---|---|---|
| Zero-shot direct | No examples | Raw allowed profile directly to probability/label. |
| Random balanced 8-shot | One fixed set of 4 YES + 4 NO memory respondents | Tests ordinary balanced few-shot prompting. |
| Similarity-selected 8-shot | For each target, 4 nearest YES + 4 nearest NO memory respondents | Personalizes examples using preprocessed feature similarity. |
| Representative 8-shot | Classwise KMeans, then nearest observed respondent to each centroid | Uses a fixed set intended to cover central class structure. |
| Random 8-shot + generic CoT | Same fixed random 4+4 examples | Adds generic evidence-for/evidence-against reasoning without behavioral theory. |

Calibration selects the classification threshold; test evaluation uses the frozen prompt and selection policy.

## 5. Transferred methods from prior work

### 5.1 Methods that do **not** fine-tune the LLM

#### HBM-CoPB — theory-structured one-call reasoning

**Original paper:** [Chain-of-Planned-Behaviour Workflow Elicits Few-Shot Mobility Generation in LLMs](https://arxiv.org/abs/2402.09836)

The original CoPB workflow uses Theory of Planned Behaviour constructs—attitude, subjective norms, and perceived behavioral control—to reason about mobility intention before mapping intention to movement. The paper also studies a separate label-generation/fine-tuning extension, but the transferred baseline in this repository is the prompting-only version.

**FLARE-VAX transfer:** `scripts/70_hbm_copb_pbj_baselines_asu.py` replaces TPB mobility stages with five HBM-inspired vaccination stages. Eight balanced memory examples are first converted into `profile → structured HBM reasoning → label` demonstrations. Calibration and test respondents use the same scaffold. Deterministic FLARE scores, HBM8 patterns, priors, retrieval, and reflection are withheld.

#### HBM-PB&J — two-call theory-grounded persona construction

**Original paper:** [Improving Language Model Personas via Rationalization with Psychological Scaffolds](https://aclanthology.org/2025.findings-emnlp.1187/)

PB&J enriches a user persona by generating plausible rationales for observed prior judgments under psychological scaffolds such as Big Five traits, Schwartz values, or Primal World Beliefs, then uses the enriched persona to predict a new preference or opinion.

**FLARE-VAX transfer:** the V4 implementation uses a label-blind first call to rationalize demographics, health/access context, healthcare behavior, and non-target vaccination history into an HBM-scaffolded health persona. A second call combines the raw profile, persona, and eight memory demonstrations to predict influenza vaccination. PB&J is currently implemented only for V4 because non-target vaccine decisions are the closest analogue to the original seed judgments.

#### SILIC-inspired inverse contextual reward inference

**Original paper:** [Where You Go is Who You Are: Behavioral Theory-Guided LLMs for Inverse Reinforcement Learning](https://arxiv.org/abs/2505.17249)

SILIC combines behavioral theory, LLM guidance, inverse reinforcement learning, and cognitive-chain reasoning to infer sociodemographic attributes from mobility trajectories. The LLM helps initialize and update a latent reward representation whose behavioral fit is optimized against observed choices.

**Initial FLARE-VAX implementation:** `scripts/80_silic_v4_inverse_contextual_reward_asu.py` is a V4-only, cross-sectional adaptation. COVID-19, pneumonia, shingles, and hepatitis-A decisions are treated as contextual binary choices; a hierarchical model estimates a five-dimensional preventive reward vector; optional LLM initialization/update guides numerical refinement; and a final HBM-inspired call predicts flu vaccination. Because NHIS has no ordered transitions, this is described as inverse contextual choice—not sequential MDP/MaxEnt IRL. The LLM remains frozen; only external latent parameters are optimized. **Status: implementation available, full benchmark results pending.**

### 5.2 Method requiring supervised model fine-tuning

#### Persona-aware and Explainable Bikeability Assessment

**Original paper:** [Persona-aware and Explainable Bikeability Assessment: A Vision-Language Model Approach](https://arxiv.org/abs/2601.03534)

The paper conditions a vision-language model on theory-grounded cyclist personas, applies multi-granularity supervised fine-tuning using expert reasoning plus user ratings, and uses controlled AI-generated data augmentation to support explainable bikeability scoring. **Status in this repository: reference reviewed; no FLARE-VAX implementation yet.**

## 6. Results

All LLM tables use the calibration-selected threshold. Balanced accuracy is reported because several LLM configurations have asymmetric sensitivity and specificity.

### 6.1 Conventional ML baselines

| Version | Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|---|
| V4 | gradient_boosting | 0.7630 | 0.8444 | 0.7506 |
| V4 | knn | 0.7102 | 0.7689 | 0.6981 |
| V4 | logistic | 0.7604 | 0.8398 | 0.7484 |
| V4 | mlp | 0.7587 | 0.8402 | 0.7509 |
| V4 | random_forest | 0.7617 | 0.8377 | 0.7471 |
| V4 | svm | 0.7630 | 0.8356 | 0.7552 |
| V4 | xgboost | 0.7647 | 0.8452 | 0.7519 |
| V5 | gradient_boosting | 0.6867 | 0.7553 | 0.6691 |
| V5 | knn | 0.6289 | 0.6708 | 0.6119 |
| V5 | logistic | 0.6766 | 0.7433 | 0.6640 |
| V5 | mlp | 0.6824 | 0.7506 | 0.6580 |
| V5 | random_forest | 0.6829 | 0.7507 | 0.6660 |
| V5 | svm | 0.6786 | 0.7477 | 0.6636 |
| V5 | xgboost | 0.6871 | 0.7570 | 0.6697 |

V4 substantially outperforms V5 in conventional ML, showing the predictive strength of other-vaccination history.

### 6.2 Zero-shot and few-shot LLM baselines

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| V4 | Llama 3 70B | Random balanced 8-shot | 12853 | 5 | 0.4737 | 0.5000 | 0.5000 | 0.6428 | complete |
| V4 | Llama 3 70B | Random 8-shot + generic CoT | 12853 | 29 | 0.5186 | 0.5399 | 0.5399 | 0.6504 | complete |
| V4 | Llama 3 70B | Representative 8-shot | — | — | — | — | — | — | pending_rerun |
| V4 | Llama 3 70B | Similarity-selected 8-shot | — | — | — | — | — | — | pending_rerun |
| V4 | Llama 3 70B | Zero-shot direct | 12853 | 46 | 0.4787 | 0.5048 | 0.5048 | 0.6449 | complete |
| V4 | Llama 4 Scout 17B | Random balanced 8-shot | 12853 | 86 | 0.6432 | 0.6297 | 0.6851 | 0.4974 | complete |
| V4 | Llama 4 Scout 17B | Random 8-shot + generic CoT | 12853 | 73 | 0.5830 | 0.5622 | 0.6048 | 0.2761 | complete |
| V4 | Llama 4 Scout 17B | Representative 8-shot | 12853 | 75 | 0.6053 | 0.6211 | 0.6554 | 0.6887 | complete |
| V4 | Llama 4 Scout 17B | Similarity-selected 8-shot | 12853 | 79 | 0.6189 | 0.6225 | 0.6543 | 0.6317 | complete |
| V4 | Llama 4 Scout 17B | Zero-shot direct | 12853 | 81 | 0.6339 | 0.6453 | 0.7136 | 0.6904 | complete |
| V5 | Llama 3 70B | Random balanced 8-shot | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 3 70B | Random 8-shot + generic CoT | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 3 70B | Representative 8-shot | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 3 70B | Similarity-selected 8-shot | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 3 70B | Zero-shot direct | — | — | — | — | — | — | pending_rerun |
| V5 | Llama 4 Scout 17B | Random balanced 8-shot | 12852 | 82 | 0.5917 | 0.5932 | 0.6229 | 0.5903 | complete |
| V5 | Llama 4 Scout 17B | Random 8-shot + generic CoT | 12852 | 73 | 0.5461 | 0.5223 | 0.5389 | 0.1248 | complete |
| V5 | Llama 4 Scout 17B | Representative 8-shot | 12852 | 75 | 0.6218 | 0.6232 | 0.6251 | 0.6197 | complete |
| V5 | Llama 4 Scout 17B | Similarity-selected 8-shot | 12852 | 79 | 0.5773 | 0.5783 | 0.6051 | 0.5725 | complete |
| V5 | Llama 4 Scout 17B | Zero-shot direct | 12852 | 81 | 0.6307 | 0.6367 | 0.6622 | 0.6585 | complete |

The public progress table intentionally leaves **all V5 Llama 3 70B rows** and the **V4 70B similarity-selected and representative rows** blank pending rerun. Preliminary outputs for these configurations were near-constant, so they are not treated as finalized evidence.

### 6.3 Transferred baseline results

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| V4 | Llama 3 70B | HBM-CoPB | 12853 | 51 | 0.6419 | 0.6464 | 0.6698 | 0.6593 | complete |
| V4 | Llama 4 Scout 17B | HBM-CoPB | 12853 | 51 | 0.6726 | 0.6753 | 0.7131 | 0.6776 | complete |
| V5 | Llama 3 70B | HBM-CoPB | 12852 | 51 | 0.5664 | 0.5634 | 0.5719 | 0.5258 | complete |
| V5 | Llama 4 Scout 17B | HBM-CoPB | 12852 | 61 | 0.6376 | 0.6378 | 0.6652 | 0.6268 | complete |
| V4 | Llama 3 70B | HBM-PB&J | 12853 | 76 | 0.5831 | 0.5935 | 0.6243 | 0.6425 | complete |
| V4 | Llama 4 Scout 17B | HBM-PB&J | — | — | — | — | — | — | pending |

HBM-PB&J V4 with Llama 4 Scout 17B is retained as a visible pending row. SILIC and the fine-tuned persona-aware method do not yet have reportable FLARE-VAX results.

### 6.4 FLARE-VAX V4/V5 main results and ablations

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| V4 | Llama 4 Scout 17B | FLARE-VAX full run | 12853 | 51 | 0.7318 | 0.7312 | 0.7789 | 0.7174 | complete |
| V4 | Llama 3 70B | FLARE-VAX full run | 12853 | 46 | 0.7278 | 0.7285 | 0.7546 | 0.7207 | complete |
| V4 | No LLM | HBM8 pattern-only ablation | 12853 | 50 | 0.7277 | 0.7284 | 0.7693 | 0.7206 | complete |
| V5 | Llama 4 Scout 17B | FLARE-VAX full run | 11057 | 47 | 0.6101 | 0.6179 | 0.6334 | 0.6265 | complete_final_summary_11057_evaluated |
| V5 | Llama 3 70B | FLARE-VAX full run | 12852 | 47 | 0.6255 | 0.6325 | 0.6763 | 0.6597 | complete |
| V5 | No LLM | HBM8 pattern-only ablation | 12852 | 37 | 0.6255 | 0.6325 | 0.6797 | 0.6597 | complete |

The V5 Llama 4 summary evaluates 11,057 test cases although the configured test split contains 12,852; the repository preserves that coverage caveat rather than imputing missing predictions.


### 6.5 RG-FLARE-VAX reward-guided extension

The table below reports the reward prior and the final reward-valued-memory stage from the **full** runs. The public CSV also retains the HBM8 pattern anchor, the no-memory comparison, and survey-weighted metrics.

| Version | Model | Stage | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| V4 | Llama 3 70B | Reward-calibrated HBM prior | 12853 | 47 | 0.7371 | 0.7390 | 0.8218 | 0.7363 |
| V4 | Llama 3 70B | Final reward-valued memory | 12853 | 52 | 0.7384 | 0.7398 | 0.8210 | 0.7351 |
| V4 | Llama 4 Scout 17B | Reward-calibrated HBM prior | 12853 | 53 | 0.7417 | 0.7403 | 0.8216 | 0.7238 |
| V4 | Llama 4 Scout 17B | Final reward-valued memory | 12853 | 50 | 0.7390 | 0.7395 | 0.8221 | 0.7308 |
| V5 | Llama 3 70B | Reward-calibrated HBM prior | 12852 | 45 | 0.6590 | 0.6635 | 0.7223 | 0.6758 |
| V5 | Llama 3 70B | Final reward-valued memory | 12852 | 50 | 0.6604 | 0.6638 | 0.7208 | 0.6699 |
| V5 | Llama 4 Scout 17B | Reward-calibrated HBM prior | 12852 | 49 | 0.6580 | 0.6585 | 0.7222 | 0.6493 |
| V5 | Llama 4 Scout 17B | Final reward-valued memory | 12852 | 46 | 0.6604 | 0.6635 | 0.7204 | 0.6686 |

The reward layer produces a clear improvement over the HBM8 pattern anchor in discrimination. On V4, ROC-AUC rises from about 0.769 for the pattern anchor to about 0.822 for the reward-guided configurations. On V5, it rises from about 0.680 to about 0.720–0.722. The reflective memory changes the final operating point and F1/balanced accuracy in some settings, but it does **not** uniformly improve every probability metric over the reward prior or the no-memory comparison; the repository therefore reports these stages separately rather than attributing the whole gain to memory.

### 6.6 TRBM-FLARE-VAX theory-residual extension

| Version | Model | Test N | Threshold | Correction scale α | Accuracy | Balanced Acc. | ROC-AUC | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V4 | Llama 3 70B | 12853 | 0.48 | 0.00 | 0.7390 | 0.7371 | 0.8205 | 0.7181 |
| V4 | Llama 4 Scout 17B | 12853 | 0.48 | 0.00 | 0.7390 | 0.7371 | 0.8205 | 0.7181 |
| V5 | Llama 3 70B | 12853 | 0.44 | 0.00 | 0.6531 | 0.6574 | 0.7200 | 0.6690 |
| V5 | Llama 4 Scout 17B | 12853 | 0.44 | 0.00 | 0.6531 | 0.6574 | 0.7200 | 0.6690 |

For these full runs, calibration selected `alpha = 0.00` for every configuration. Therefore the reported `trbm_full` predictions are effectively the theory-constrained HBM prior, even though the reflection memories and LLM mechanism gates were built and evaluated. This is an informative result: under the current correction rule, calibration did not find positive log-loss value in applying the residual-memory correction on top of the theory prior. The V4 prior still reaches ROC-AUC 0.8205; V5 reaches 0.7200 in the unweighted evaluation. The supplied TRBM package reports 12,853 V5 test cases, and that count is preserved here as produced rather than silently harmonized with the 12,852-row V5 count in the earlier pipeline.

## 7. Curated repository structure

```text
scripts/
  01_ml_baselines.py
  40_flare_vax_v4_with_vaccine_history.py
  50_flare_vax_v5_without_vaccine_history.py
  60_llm_icl_benchmark_asu.py
  70_hbm_copb_pbj_baselines_asu.py
  80_silic_v4_inverse_contextual_reward_asu.py
  81_rg_flare_vax_reward_memory_asu.py
  82_trbm_flare_vax_asu.py
  83_trbm_ablation_asu.py
  90_collect_results.py
configs/
docs/
data/README.md
results/
  ml/
  icl/
  transfer_baselines/
  flare_vax/
  reward_memory/
  trbm/
```

Excluded from the public package: earlier HBM2 development scripts, notebooks with local execution state, `.ipynb_checkpoints`, raw respondent-level predictions, prompt logs, support maps, local paths, API failure traces, and the raw NHIS dataset.

## 8. Reproduction

```bash
pip install -r requirements.txt
```

Examples:

```bash
python scripts/01_ml_baselines.py --data_path /path/to/adult24.csv --output_csv results/ml/new_run.csv

python scripts/60_llm_icl_benchmark_asu.py \
  --input-csv /path/to/adult24.csv \
  --output-dir /path/to/output \
  --v4-reference-split /path/to/v4_split.csv \
  --v5-reference-split /path/to/v5_split.csv

python scripts/70_hbm_copb_pbj_baselines_asu.py --help
python scripts/80_silic_v4_inverse_contextual_reward_asu.py --help
python scripts/81_rg_flare_vax_reward_memory_asu.py --help
python scripts/82_trbm_flare_vax_asu.py --help
python scripts/83_trbm_ablation_asu.py --help
```

The LLM runners default to the ASU OpenAI-compatible endpoint and read credentials from environment variables or explicit CLI arguments. Run `--plan-only`/dry-run options before a full grid, and use a new output directory when changing experimental configuration.
