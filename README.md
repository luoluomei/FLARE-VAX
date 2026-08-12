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


## New extensions to V4/V5: RG-FLARE-VAX and TRBM-FLARE-VAX

The following two methods are follow-up extensions of the original V4/V5 framework rather than new feature versions. V4 still allows other vaccination history, V5 still excludes it, and both methods preserve the original five HBM-inspired proxies and the V4/V5 feature boundary.

The extensions address two questions: can the relatively coarse HBM8 pattern prior be converted into a more individualized numerical prior, and can reflective memory be evaluated by historical predictive usefulness rather than by LLM-generated plausibility alone?

### RG-FLARE-VAX — Reward-Guided HBM Integration + Reward-Valued Memory

**Code:** `scripts/81_rg_flare_vax_reward_memory_asu.py`

RG adds a **reward-guided probability-calibration layer** to the original FLARE-VAX and changes how reflective memory is valued. The original HBM8 pattern is preserved, but the pattern vaccination rate is no longer the only probability anchor. Instead, the five HBM constructs are first used to produce a more individualized `P_reward`, and the LLM/memory layer focuses on the residual information that remains unexplained.

The method can be summarized in four stages.

**1. Reuse the original V4/V5 HBM representation.**  
Each respondent receives the same five HBM proxies, Motivation/Capability/Activation state, and HBM8 pattern as in the original method. The memory-split vaccination rate for each pattern remains the initial behavioral anchor `P_pattern`.

**2. Refine `P_pattern` into a more individualized `P_reward`.**  
Respondents within the same HBM8 pattern can still differ substantially in threat, barriers, cues, and the other construct values. RG therefore fits a cross-fitted numerical reward model that learns how a respondent's deviation from the pattern-level construct profile should adjust `P_pattern`.

An optional pattern-level LLM-guidance call can propose sparse increases/decreases to construct weights using aggregate error and contribution statistics. The proposal is not accepted automatically: it must improve log loss on a separate OOF validation subset. The design is therefore `LLM proposes → data validates → accept/reject`.

**3. Build reward-valued reflective memory around `P_reward`.**  
For selected memory respondents, the LLM treats `P_reward` as the main probability anchor and makes only a small residual adjustment using more detailed observed context. High-confidence errors are reflected into reusable correction rules.

Unlike the original memory design, each rule is then assigned an empirical **Q-value**. The system evaluates whether applying the rule's proposed increase/decrease direction to similar historical respondents actually reduces prediction loss. A memory is therefore valued by both semantic plausibility and empirical predictive utility: it should not only sound reasonable, but should also have helped similar historical cases.

**4. Retrieve similar, high-value memories and make the final bounded correction.**  
At calibration/test time, retrieval combines respondent similarity with memory Q-value. The final LLM receives `P_reward`, the current profile, and the retrieved reward-valued memories, and decides whether a small increase, decrease, or no change is warranted. Calibration then selects the final classification threshold.

The main change from the original FLARE-VAX is therefore twofold: **RG learns a more individualized reward-calibrated prior from the five HBM constructs, and it upgrades reflective memory from an LLM-generated rule into a rule whose historical predictive value is explicitly evaluated.** The LLM itself remains frozen.

### TRBM-FLARE-VAX — Theory-Residual Behavioral Memory

**Code:** `scripts/82_trbm_flare_vax_asu.py`  
**Ablation:** `scripts/83_trbm_ablation_asu.py`

TRBM further reduces the LLM's role in numerical prediction. Its central idea is to first let the HBM representation produce an explicit theory-based probability, and then learn **when and why that theory prior fails**.

The method can also be summarized in four stages.

**1. Construct an explicit theory prior.**  
The same five V4/V5 HBM constructs are oriented so that larger values consistently represent stronger theoretical support for vaccination. A small non-negative constrained logistic model is then fitted using only these five theory-derived variables, producing `P_HBM`.

Because the full NHIS feature set is not used, `P_HBM` is intended to represent what the current HBM representation predicts, rather than a conventional full-feature ML prediction.

**2. Identify historical theory failures directly.**  
Cross-fitting on the memory split produces OOF `P_HBM` for each respondent. TRBM then computes the discrepancy between the theory prior and the observed outcome:

`theory residual = actual - P_HBM_OOF`

Large positive or negative residuals identify cases in which the five-dimensional HBM prior clearly under- or over-predicts observed vaccination behavior. Thus, TRBM memory begins from failures of the theory representation itself, not from errors made by an LLM prediction.

**3. Use the LLM to summarize and match theory-failure mechanisms.**  
For large residual cases, the LLM examines the HBM state, `P_HBM`, observed outcome, and detailed profile to identify what observable mechanism was not sufficiently represented by the HBM prior. These failures are summarized into reusable mechanism memories.

For a new calibration/test respondent, similar historical theory-failure memories are retrieved. The LLM acts only as a **mechanism gate**: it decides whether a retrieved mechanism genuinely applies, whether the supported direction is increase/decrease/no correction, and whether contradictions are present. It does not output a new probability or determine a numerical correction magnitude.

**4. Let historical residuals determine the correction size.**  
When the LLM judges that a historical mechanism transfers, the signed residuals from the selected historical memories are combined using similarity and confidence to produce an empirical numerical correction. Calibration then selects a global scale `alpha`:

`P_TRBM = P_HBM + alpha × empirical residual correction`

If calibration does not find reliable additional value from the memory correction, it can select `alpha = 0`, returning the final prediction to `P_HBM`.

TRBM therefore has a deliberately strict division of labor: **the HBM theory model provides the base probability, historical residuals provide the numerical correction magnitude, and the LLM only decides whether a historical theory-failure mechanism applies to the current respondent.**

| Method | Base probability | Main LLM role at prediction time | Numerical correction |
|---|---|---|---|
| Original FLARE-VAX | HBM8 pattern prior | Direct residual reasoning from respondent evidence | Determined by LLM |
| RG-FLARE-VAX | Reward-calibrated HBM prior | Small residual correction using Q-valued memory | Determined by LLM within a tighter bound |
| TRBM-FLARE-VAX | Theory-constrained `P_HBM` | Match/gate historical theory-failure mechanisms | Historical residuals + calibrated `alpha` |

## 3. Zero-shot and few-shot LLM baselines

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

## 4. Transferred methods from prior work

### 4.1 Methods that do **not** fine-tune the LLM

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

### 4.2 Method requiring supervised model fine-tuning

#### Persona-aware and Explainable Bikeability Assessment

**Original paper:** [Persona-aware and Explainable Bikeability Assessment: A Vision-Language Model Approach](https://arxiv.org/abs/2601.03534)

The paper conditions a vision-language model on theory-grounded cyclist personas, applies multi-granularity supervised fine-tuning using expert reasoning plus user ratings, and uses controlled AI-generated data augmentation to support explainable bikeability scoring. **Status in this repository: reference reviewed; no FLARE-VAX implementation yet.**

## 5. Results

All LLM tables use the calibration-selected threshold. Balanced accuracy is reported because several LLM configurations have asymmetric sensitivity and specificity.

### 5.1 Conventional ML baselines

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

### 5.2 Zero-shot and few-shot LLM baselines

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| V4 | Llama 3 70B | Random balanced 8-shot | 12853 | 5 | 0.4737 | 0.5000 | 0.5000 | 0.6428 | complete |
| V4 | Llama 3 70B | Random 8-shot + generic CoT | 12853 | 29 | 0.5186 | 0.5399 | 0.5399 | 0.6504 | complete |
| V4 | Llama 3 70B | Representative 8-shot | 12853 | 5 | 0.4737 | 0.5000 | 0.5000 | 0.6428 | complete |
| V4 | Llama 3 70B | Similarity-selected 8-shot | 12853 | 5 | 0.4737 | 0.5000 | 0.5000 | 0.6428 | complete |
| V4 | Llama 3 70B | Zero-shot direct | 12853 | 46 | 0.4787 | 0.5048 | 0.5048 | 0.6449 | complete |
| V4 | Llama 4 Scout 17B | Random balanced 8-shot | 12853 | 86 | 0.6432 | 0.6297 | 0.6851 | 0.4974 | complete |
| V4 | Llama 4 Scout 17B | Random 8-shot + generic CoT | 12853 | 73 | 0.5830 | 0.5622 | 0.6048 | 0.2761 | complete |
| V4 | Llama 4 Scout 17B | Representative 8-shot | 12853 | 75 | 0.6053 | 0.6211 | 0.6554 | 0.6887 | complete |
| V4 | Llama 4 Scout 17B | Similarity-selected 8-shot | 12853 | 79 | 0.6189 | 0.6225 | 0.6543 | 0.6317 | complete |
| V4 | Llama 4 Scout 17B | Zero-shot direct | 12853 | 81 | 0.6339 | 0.6453 | 0.7136 | 0.6904 | complete |
| V5 | Llama 3 70B | Random balanced 8-shot | 12852 | 5 | 0.4739 | 0.5000 | 0.5000 | 0.6430 | complete |
| V5 | Llama 3 70B | Random 8-shot + generic CoT | 12851 | 51 | 0.4877 | 0.5122 | 0.5122 | 0.6448 | complete* |
| V5 | Llama 3 70B | Representative 8-shot | 12852 | 5 | 0.4739 | 0.5000 | 0.5000 | 0.6430 | complete |
| V5 | Llama 3 70B | Similarity-selected 8-shot | 12852 | 5 | 0.4739 | 0.5000 | 0.5000 | 0.6430 | complete |
| V5 | Llama 3 70B | Zero-shot direct | 12852 | 46 | 0.4774 | 0.5034 | 0.5034 | 0.6445 | complete |
| V5 | Llama 4 Scout 17B | Random balanced 8-shot | 12852 | 82 | 0.5917 | 0.5932 | 0.6229 | 0.5903 | complete |
| V5 | Llama 4 Scout 17B | Random 8-shot + generic CoT | 12852 | 73 | 0.5461 | 0.5223 | 0.5389 | 0.1248 | complete |
| V5 | Llama 4 Scout 17B | Representative 8-shot | 12852 | 75 | 0.6218 | 0.6232 | 0.6251 | 0.6197 | complete |
| V5 | Llama 4 Scout 17B | Similarity-selected 8-shot | 12852 | 79 | 0.5773 | 0.5783 | 0.6051 | 0.5725 | complete |
| V5 | Llama 4 Scout 17B | Zero-shot direct | 12852 | 81 | 0.6307 | 0.6367 | 0.6622 | 0.6585 | complete |

The previously pending Llama 3 70B benchmark rows have now been completed. Several direct 8-shot configurations still collapse to nearly constant positive predictions (balanced accuracy ≈ 0.50), which is reported as an observed benchmark outcome rather than treated as a missing run. `*` The V5 70B generic-CoT run returned 12,851 test predictions (test success rate 0.999922) rather than the configured 12,852.

### 5.3 Transferred baseline results

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| V4 | Llama 3 70B | HBM-CoPB | 12853 | 51 | 0.6419 | 0.6464 | 0.6698 | 0.6593 | complete |
| V4 | Llama 4 Scout 17B | HBM-CoPB | 12853 | 51 | 0.6726 | 0.6753 | 0.7131 | 0.6776 | complete |
| V5 | Llama 3 70B | HBM-CoPB | 12852 | 51 | 0.5664 | 0.5634 | 0.5719 | 0.5258 | complete |
| V5 | Llama 4 Scout 17B | HBM-CoPB | 12852 | 61 | 0.6376 | 0.6378 | 0.6652 | 0.6268 | complete |
| V4 | Llama 3 70B | HBM-PB&J | 12853 | 76 | 0.5831 | 0.5935 | 0.6243 | 0.6425 | complete |
| V4 | Llama 4 Scout 17B | HBM-PB&J | 12853 | 61 | 0.7007 | 0.6970 | 0.7610 | 0.6651 | complete |

The previously pending V4 Llama 4 Scout 17B HBM-PB&J run is now complete and reaches ROC-AUC 0.7610 with balanced accuracy 0.6970. V5 HBM-PB&J, SILIC, and the fine-tuned persona-aware method do not yet have reportable FLARE-VAX results.

### 5.4 FLARE-VAX V4/V5 main results and ablations

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| V4 | Llama 4 Scout 17B | FLARE-VAX full run | 12853 | 51 | 0.7318 | 0.7312 | 0.7789 | 0.7174 | complete |
| V4 | Llama 3 70B | FLARE-VAX full run | 12853 | 46 | 0.7278 | 0.7285 | 0.7546 | 0.7207 | complete |
| V4 | No LLM | HBM8 pattern-only ablation | 12853 | 50 | 0.7277 | 0.7284 | 0.7693 | 0.7206 | complete |
| V5 | Llama 4 Scout 17B | FLARE-VAX full run | 11057 | 47 | 0.6101 | 0.6179 | 0.6334 | 0.6265 | complete_final_summary_11057_evaluated |
| V5 | Llama 3 70B | FLARE-VAX full run | 12852 | 47 | 0.6255 | 0.6325 | 0.6763 | 0.6597 | complete |
| V5 | No LLM | HBM8 pattern-only ablation | 12852 | 37 | 0.6255 | 0.6325 | 0.6797 | 0.6597 | complete |

The V5 Llama 4 summary evaluates 11,057 test cases although the configured test split contains 12,852; the repository preserves that coverage caveat rather than imputing missing predictions.


### 5.5 RG-FLARE-VAX extension

The table reports the reward-calibrated prior and the final reward-valued-memory stage from the full runs.

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

The clearest gain comes from the reward-calibrated prior itself: V4 ROC-AUC increases from roughly 0.769 for the HBM8 pattern anchor to about 0.822, while V5 increases from roughly 0.680 to about 0.720–0.722. Reward-valued memory changes the final operating point and helps some metrics/settings, but does not uniformly outperform the reward prior on every probability metric.

### 5.6 TRBM-FLARE-VAX extension

The TRBM experiments are still being completed. For this week's update, only the **V4 + Llama 4 Scout 17B** runs are treated as finalized; the remaining planned configurations are listed with `--` placeholders.

| Version | Model | Method | Test N | Memories | Threshold | Correction scale α | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V4 | Llama 4 Scout 17B | TRBM full | 12853 | 459 | 0.48 | 0.00 | 0.7390 | 0.7371 | 0.8205 | 0.7181 | complete |
| V4 | Llama 4 Scout 17B | TRBM full — survey weighted | 12853 | 459 | 0.48 | 0.00 | 0.7315 | 0.7236 | 0.8113 | 0.6812 | complete |
| V4 | Llama 3 70B | TRBM full | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V4 | Llama 3 70B | TRBM full — survey weighted | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V5 | Llama 4 Scout 17B | TRBM full | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V5 | Llama 4 Scout 17B | TRBM full — survey weighted | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V5 | Llama 3 70B | TRBM full | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |
| V5 | Llama 3 70B | TRBM full — survey weighted | -- | -- | -- | -- | -- | -- | -- | -- | not completed yet |

For the two completed V4 Scout 17B runs, calibration selected `alpha = 0.00`. Thus the final TRBM probability currently reduces to the theory-constrained `P_HBM`: the reflective-memory and mechanism-gating components were executed, but their residual correction was not retained by calibration. The unweighted theory prior reaches ROC-AUC 0.8205; the survey-weighted version reaches 0.8113. The other TRBM configurations are intentionally left blank until their runs are finalized.

## 6. Curated repository structure

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

## 7. Reproduction

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
