# FLARE-VAX: Theory-Guided Influenza Vaccination Prediction from NHIS 2024

[中文详细说明](README_zh.md)

This repository documents a sequence of experiments that transfer and extend the **FLARE** framework from wildfire evacuation prediction to influenza-vaccination behavior prediction using the **2024 National Health Interview Survey (NHIS) Sample Adult file**.

The prediction target is:

```text
SHTFLU12M_A: whether the respondent received an influenza vaccination during the past 12 months
```

The repository is organized as a **method-evolution record**, rather than as a claim that every later version performs better. The progression starts with a close three-call transfer of the original FLARE logic, then introduces structured reflection, probability calibration, pattern anchors, dual memory, five observable HBM-inspired proxies, eight behavioral patterns, and finally a strict version that removes all non-target vaccine-history variables.

> **Important interpretation.** The constructs in this repository are theory-guided proxies derived from observable NHIS variables. They are not direct psychometric measurements of private beliefs such as perceived susceptibility, perceived benefit, perceived barriers, cues to action, or self-efficacy.

---

## 1. Research objective

The central question is not only whether an LLM can predict flu-vaccination behavior, but also **how behavioral theory should be incorporated into an LLM decision system**.

The experiments test several increasingly structured roles for the LLM:

1. **Construct assessor:** separately infer threat and barriers from survey variables.
2. **Direct decision maker:** combine construct assessments and context into a binary decision.
3. **Probability estimator:** return a calibrated probability rather than only YES/NO.
4. **Error reflector:** convert selected training errors into reusable correction rules.
5. **Residual adjuster:** begin from a training-derived pattern probability and make only a bounded individual-level adjustment.
6. **Structured reasoner:** explicitly move through construct interpretation, pattern interpretation, residual context, and decision mapping.

This produces two main empirical settings:

- **With other-vaccine history:** prior COVID-19, pneumonia, shingles, Shingrix, and hepatitis-A vaccination variables may contribute to the behavioral profile.
- **Without other-vaccine history:** every vaccine-related predictor other than the flu-vaccination target is excluded from scoring, prompting, similarity retrieval, and memory construction.

---

## 2. Repository structure

```text
scripts/
  00_prepare_hbm2_data.py
  01_ml_baselines.py
  10_hbm2_three_call.py
  11_hbm2_three_call_async.py
  20_hbm2_reflection_calibration.py
  30_hbm2_pattern_anchor_dual_memory.py
  40_hbm5_prior_vaccine_reflective_memory.py
  50_hbm5_no_prior_vaccine_reflective_memory.py
  90_collect_results.py

configs/
  experiment_catalog.json
  baseline_feature_manifest.example.json

docs/
  method_evolution.md
  variable_dictionary.md
  evaluation_notes.md
  replication_commands.md

results/
  run_index.csv
  reported_llm_hbm_metrics.csv
  baseline_all_models.csv
  benchmark_without_other_vaccine_history.csv
  benchmark_with_other_vaccine_history.csv
  raw_logs/
  summaries/
```

The full prompt text and JSON schemas are preserved directly in each experiment script. This README explains their structure and role without duplicating every line of code.

---

## 3. Evolution at a glance

| Version | Behavioral representation | Main LLM role | Regular LLM calls per respondent | Conditional reflection call | Prediction anchor | Memory type | Other-vaccine history |
|---|---|---|---:|---|---|---|---|
| V1 | Threat + Barrier; four HBM2 patterns | assess two constructs, then directly classify | 3 | No separate reflection call | qualitative HBM2 pattern | raw incorrect training examples | excluded |
| V1-Async | same as V1 | same as V1 | 3, executed concurrently | No | same as V1 | same examples, updated by batch | excluded |
| V2 | authoritative Threat + Barrier; four HBM2 patterns | refine constructs, estimate probability, explain decision | 3 | one additional call for each selected/wrong training case | qualitative pattern prior or training pattern rate | LLM-generated correction rules | excluded |
| V3 | authoritative HBM2 + training pattern base rate | output only a bounded residual adjustment | 3 | selected high-confidence errors | numeric pattern base rate | correct prototypes + error reflections | excluded |
| V4 | five deterministic proxies; three meta-dimensions; eight patterns | one integrated residual probability decision | 1 | selected high-confidence errors | numeric HBM8 base rate | distilled reflective exception rules | allowed |
| V5 | five deterministic proxies; three meta-dimensions; eight patterns | one four-stage structured residual decision | 1 | selected high-confidence errors | numeric HBM8 base rate | stage-specific reflective rules | fully excluded |

The key architectural change is that **LLM call count decreases as the deterministic theory layer becomes richer**. V1–V3 normally make three prediction-related calls per respondent. V4–V5 construct the behavioral profile in Python and make one integrated decision call per respondent.

---

# 4. Detailed method implementations

## 4.1 V1 — HBM2 three-call direct transfer

**Script:** `scripts/10_hbm2_three_call.py`  
**Version string:** `hbm2_openai_colab_v1`

### 4.1.1 Purpose

V1 is the closest minimal transfer of the original FLARE architecture. It represents the decision through two broad HBM-inspired constructs:

- **Threat:** observable susceptibility and severity-related risk.
- **Barrier:** observable financial, insurance, access, transportation, language, and digital friction.

The two constructs form four rule-based patterns:

| Pattern | Threat | Barrier | Theoretical tendency |
|---|---|---|---|
| P0 | High | Low | most favorable profile for vaccination |
| P1 | High | High | motivation is present but barriers may block action |
| P2 | Low | Low | access is favorable but motivation may be weaker |
| P3 | Low | High | least favorable profile |

The rule-based pattern is computed before the LLM calls. The LLM-generated 1–5 scores do **not** define the pattern.

### 4.1.2 Step 1 — deterministic HBM2 profile construction

`00_prepare_hbm2_data.py` converts the raw NHIS file into a cleaned row-level dataset.

The threat side uses variables such as:

- age and age-group indicators;
- self-rated health;
- diabetes, COPD, cancer, heart disease, angina, heart attack, stroke, hypertension, asthma, kidney and liver conditions;
- disability and functional difficulty;
- BMI and smoking-related context.

The barrier side uses variables such as:

- current and recent insurance status;
- cost-related loss of coverage;
- delayed or forgone care because of cost;
- difficulty paying medical bills;
- usual source of care;
- transportation barriers;
- language or communication difficulty;
- internet and digital-access indicators.

Python creates:

```text
hbm_threat_score
hbm_threat_level
hbm_barrier_score
hbm_barrier_level
hbm2_pattern
```

The LLM sees a human-readable profile rather than a raw vector of unexplained survey codes.

### 4.1.3 CALL 1 — threat assessment

#### LLM role

The model acts only as an HBM threat assessor. It is explicitly prohibited from predicting vaccination or discussing barriers, benefits, or cues.

#### Input

A formatted profile containing age, health status, chronic/risk conditions, functional limitations, disability, BMI, and related observable evidence.

#### System-prompt constraints

The system prompt tells the LLM to:

- combine susceptibility and severity into an observable perceived-threat proxy;
- use only supplied evidence;
- avoid inferring private attitudes;
- avoid predicting the vaccination target;
- follow a 1–5 rubric;
- return one concise reason.

#### User-prompt structure

```text
TASK
  Assess the HBM-aligned influenza threat proxy.

DEFINITION
  Susceptibility + severity; observable proxies, not private beliefs.

SCORING RUBRIC
  1 = very low ... 5 = very high.

INPUT FORMAT
  age, health status, conditions, functional limitations, BMI, etc.

OUTPUT FORMAT
  structured JSON only.
```

#### Output schema

```json
{
  "score": 4,
  "level": "high",
  "reason": "The respondent has concentrated age and chronic-risk evidence."
}
```

Allowed levels are `very_low`, `low`, `moderate`, `high`, and `very_high`.

### 4.1.4 CALL 2 — barrier assessment

#### LLM role

The model evaluates only the barrier construct. It is prohibited from using threat information or directly predicting vaccination.

#### Input

A formatted profile containing insurance, affordability, usual care, medical-bill stress, transportation, language, and digital-access evidence.

#### System-prompt constraints

The system prompt requires the model to:

- focus only on observable barriers;
- avoid inferring vaccine attitudes;
- ignore threat evidence;
- use a 1–5 rubric;
- return one concise sentence.

#### User-prompt structure

```text
TASK
  Assess the HBM-aligned barrier proxy.

DEFINITION
  Financial, insurance, healthcare-access, transportation,
  language, and digital obstacles.

SCORING RUBRIC
  1 = very low ... 5 = very high.

INPUT FORMAT
  insurance and access profile.

OUTPUT FORMAT
  structured JSON only.
```

#### Output schema

```json
{
  "score": 3,
  "level": "moderate",
  "reason": "The respondent has cost concerns and no stable usual-care location."
}
```

### 4.1.5 Memory retrieval before CALL 3

During the training phase, each incorrect prediction is converted into a memory record containing:

- the rule-based pattern;
- the CALL 1 and CALL 2 outputs;
- selected context variables;
- the predicted decision;
- the true training label.

The system also creates a standardized numeric feature vector. For a new respondent, cosine similarity retrieves the top `k` incorrect training cases. No additional LLM call is used to summarize these errors.

Therefore, V1 memory is best described as:

```text
raw error-example memory
```

rather than reflective rule memory.

### 4.1.6 CALL 3 — final binary vaccination decision

#### Input

The decision prompt contains six blocks:

```text
1. CALL 1 — THREAT ASSESSMENT
2. CALL 2 — BARRIER ASSESSMENT
3. RULE-BASED HBM2 PATTERN
4. OPTIONAL PATTERN PRIOR
5. ADDITIONAL CONTEXT VARIABLES
6. SIMILAR TRAINING ERRORS
```

The pattern prior can be configured as:

- `theory`: qualitative ordering only;
- `train_rate`: training-only vaccination rate for the current pattern;
- `none`: no explicit pattern prior.

#### Decision-prompt constraints

The LLM is instructed to:

1. integrate the two construct assessments first;
2. treat the HBM2 pattern as directional rather than deterministic;
3. use additional context only when it adds clear evidence;
4. use memory only when genuinely similar;
5. output no explanation beyond the required JSON field.

#### Output schema

```json
{
  "decision": "YES"
}
```

or

```json
{
  "decision": "NO"
}
```

### 4.1.7 Training and test order

```text
training respondent
  → CALL 1
  → CALL 2
  → retrieve previous training errors
  → CALL 3
  → compare with training label
  → if wrong, append raw error example to memory

end of training
  → freeze memory

test respondent
  → CALL 1
  → CALL 2
  → retrieve frozen training errors
  → CALL 3
  → no memory update
```

### 4.1.8 What the LLM contributes

V1 gives the LLM broad responsibility:

- infer two latent-style construct scores;
- integrate the constructs;
- decide how much to follow the pattern;
- interpret background context;
- decide whether similar errors should override the prior;
- directly return the final label.

This makes V1 close to the original FLARE idea, but also makes it difficult to identify which component caused an error.

---

## 4.2 V1-Async — concurrent three-call implementation

**Script:** `scripts/11_hbm2_three_call_async.py`  
**Version string:** `hbm2_openai_colab_v2_concurrent`

V1-Async uses the **same prompts, schemas, features, patterns, and decision logic** as V1. It is an engineering implementation rather than a new behavioral model.

### 4.2.1 Execution changes

For each batch:

1. CALL 1 and CALL 2 are launched concurrently for each respondent.
2. Multiple respondents can be processed simultaneously.
3. Once the two assessments are available, CALL 3 is launched.
4. Completed cases are written in data-index order.
5. Training errors are added to memory after the batch completes.

### 4.2.2 Consequence for memory timing

Sequential V1 allows training case `i` to influence case `i+1` immediately. V1-Async uses one pre-batch memory snapshot for every respondent in the same batch.

```text
Sequential V1:
case 1 error → memory → case 2 can retrieve it

V1-Async:
batch starts with memory M
case 1, case 2, ..., case b all use M
batch finishes → new errors are added
```

This normally provides a large speed improvement, but it is not perfectly identical to online sample-by-sample learning.

---

## 4.3 V2 — authoritative HBM2, reflective memory, and threshold calibration

**Script:** `scripts/20_hbm2_reflection_calibration.py`  
**Version string:** `hbm2_openai_colab_v3_reflection_calibrated`

### 4.3.1 Main changes from V1

V2 introduces four changes:

1. the rule-based High/Low threat and barrier classes become authoritative;
2. CALL 3 outputs a probability and reason rather than only YES/NO;
3. incorrect training predictions receive a separate reflection call;
4. the final classification threshold is selected from completed training predictions.

> **Implementation note:** the preserved V2 script calibrates the threshold on the training decisions, not on an independent calibration split. Independent memory/calibration/test splitting begins in V3.

### 4.3.2 CALL 1 and CALL 2 — constrained construct refinement

Python first supplies the authoritative High/Low class. The LLM is allowed to explain and refine the class but cannot change it.

For a Low class, the structured-output schema permits scores `1`, `2`, or `3`. For a High class, it permits `3`, `4`, or `5`. Score `3` is the shared boundary value.

The prompt contains:

```text
AUTHORITATIVE RULE-BASED CLASSIFICATION
  rule score
  rule level
  explicit instruction that the class cannot change

1–5 NUANCE RUBRIC
  score meaning within the supplied class

INDIVIDUAL VARIABLES
  observable respondent evidence
```

Outputs remain:

```json
{
  "score": 4,
  "level": "high",
  "reason": "..."
}
```

The role of CALL 1/2 has therefore changed from **construct classification** to **construct explanation and within-class nuance**.

### 4.3.3 CALL 3 — probability decision

The prompt contains:

```text
1. authoritative HBM2 pattern and rule scores
2. optional pattern prior
3. CALL 1 threat explanation
4. CALL 2 barrier explanation
5. additional observed context
6. retrieved reflection memories
7. instructions for probability and decision consistency
```

The LLM must return:

```json
{
  "probability_yes": 68,
  "decision": "YES",
  "reason": "High threat and stable access outweigh the remaining cost concern."
}
```

The internal validation requires:

```text
decision = YES when probability_yes >= 50
decision = NO  when probability_yes < 50
```

The final reported label may later use a calibrated threshold rather than 50.

### 4.3.4 Reflection call for an incorrect training prediction

If a training prediction is wrong at the internal 50% cutoff, the true training outcome becomes available to the reflection prompt.

The reflection input contains:

```text
TRAINING ERROR
  actual outcome
  predicted probability
  predicted label
  original reason

AUTHORITATIVE HBM2 PATTERN
  pattern ID and rule scores

CALL 1 RESULT
CALL 2 RESULT
ADDITIONAL CONTEXT
MEMORY USED BEFORE THE ERROR
```

The reflection system prompt asks the LLM to identify a transferable error rather than merely state that the label differed.

Output schema:

```json
{
  "error_cause": "The model over-weighted financial barriers.",
  "missed_or_overweighted_signal": "Recent repeated healthcare contact was under-weighted.",
  "correction_rule": "For similar P1 respondents, increase probability when recent healthcare contact is strong despite moderate cost concern.",
  "applicable_pattern": "P1"
}
```

This creates a structured correction rule that can be retrieved for later respondents.

### 4.3.5 Online reflection-memory update

Training CALL 3 is sequential. After an incorrect case is reflected on, the resulting rule can influence the next training case.

```text
training case
  → CALL 1/2
  → retrieve existing reflections
  → CALL 3 probability
  → if wrong at 50%, REFLECTION CALL
  → store rule immediately
```

After training, memory is frozen and cannot be updated by test labels.

### 4.3.6 Threshold calibration

Python searches candidate probability cutoffs and selects the one maximizing the configured metric:

- balanced accuracy;
- F1;
- accuracy.

The calibration is deterministic and does not call an LLM.

### 4.3.7 What the LLM contributes

Relative to V1, the LLM is more constrained in construct formation but more expressive in the final decision:

- CALL 1/2 explain fixed classes;
- CALL 3 produces a probability and rationale;
- the reflection call transforms errors into reusable rules.

The main research question becomes whether **LLM-generated correction rules** improve future predictions more than raw error examples.

---

## 4.4 V3 — pattern anchor, bounded residual adjustment, and dual memory

**Script:** `scripts/30_hbm2_pattern_anchor_dual_memory.py`  
**Version string:** `hbm2_openai_colab_v4_pattern_anchor_dual_memory`

### 4.4.1 Main changes from V2

V3 separates the prediction into two components:

```text
final probability
  = training-only HBM2 pattern base rate
  + bounded LLM residual adjustment
```

The pattern is no longer only qualitative. It supplies an empirical numeric anchor.

V3 also creates a proper three-way split:

```text
memory-build split
calibration split
test split
```

### 4.4.2 Step 1 — memory/calibration/test split

The default ratios are:

```text
memory-build = 0.50
calibration  = 0.25
test         = 0.25
```

The split can preserve vaccination-class and HBM2-pattern composition.

Only the memory-build split is used to:

- estimate pattern base rates;
- construct prototypes;
- construct reflection memories.

### 4.4.3 Step 2 — fit pattern probability anchors

For each HBM2 pattern, Python estimates a smoothed vaccination rate from the memory-build split.

Example:

```text
P0 High Threat / Low Barrier  → 62.6%
P1 High Threat / High Barrier → 48.3%
P2 Low Threat / Low Barrier   → 38.8%
P3 Low Threat / High Barrier  → 31.2%
```

For a memory-build respondent, a leave-one-out version is used so that the respondent's own label does not directly contribute to its anchor. Calibration and test respondents use the full memory-build pattern rate.

### 4.4.4 CALL 1 and CALL 2

The calls use the authoritative-class prompt design introduced in V2:

- the High/Low class cannot change;
- the LLM supplies a 1–5 within-class nuance score and one concise explanation;
- the calls do not predict vaccination.

All selected respondents' CALL 1/2 outputs are precomputed concurrently.

### 4.4.5 Dual-memory retrieval before CALL 3

V3 has two memory channels.

#### Prototype memory

A prototype is a high-confidence **correct** memory-build case. It represents typical evidence supporting the normal pattern tendency.

Prototype creation does not require a separate LLM call. It stores the observed case and its successful decision behavior.

#### Reflection memory

A reflection is generated from a high-confidence **incorrect** memory-build case. It represents a narrow exception rule.

It requires an additional reflection LLM call.

#### Retrieval policy

The decision prompt receives a configurable mixture, such as:

```text
prototype_k  = 2
reflection_k = 1
```

The prompt explicitly states that prototypes represent typical cases and reflections represent exceptional cases. A reflection should not overturn the pattern anchor unless the current respondent closely matches its applicability conditions.

### 4.4.6 CALL 3 — bounded residual adjustment

The user prompt contains:

```text
AUTHORITATIVE HBM2 CLASSIFICATION
  pattern ID
  rule threat score and level
  rule barrier score and level
  theoretical tendency

TRAINING-ONLY PROBABILITY ANCHOR
  pattern base rate
  number of memory-build observations

CALL 1 — THREAT EXPLANATION/NUANCE
CALL 2 — BARRIER EXPLANATION/NUANCE
ADDITIONAL CONTEXT
BALANCED MEMORY EVIDENCE
OUTPUT RULES
```

The system prompt tells the LLM that the base rate is authoritative as the starting probability and that it must not invent a new absolute probability.

Output schema:

```json
{
  "adjustment": 10,
  "reason": "Recent healthcare engagement supports a modest upward adjustment within P0."
}
```

Allowed adjustment:

```text
-20 to +20 percentage points
```

Python calculates:

```text
probability_yes = clip(pattern_base_rate + adjustment, 0, 100)
```

The LLM therefore acts as a **residual model**, not the entire classifier.

### 4.4.7 Reflection call

A high-confidence error can trigger:

```json
{
  "error_cause": "The model treated the pattern tendency as too deterministic.",
  "missed_or_overweighted_signal": "The respondent's repeated recent healthcare use was under-weighted.",
  "correction_rule": "Apply an upward exception only to P3 respondents with similarly strong recent healthcare engagement.",
  "applicable_pattern": "P3"
}
```

The prompt includes the anchor, CALL 3 adjustment, final probability, pattern, CALL 1/2 results, context, and any memory used before the error.

### 4.4.8 Memory-build, calibration, and test sequence

```text
MEMORY-BUILD
  precompute CALL 1/2
  sequential CALL 3
  add high-confidence correct prototypes
  reflect on selected high-confidence errors
  freeze final dual memory

CALIBRATION
  CALL 3 with frozen dual memory
  select decision threshold
  no memory updates

TEST
  CALL 3 with the same frozen memory
  apply frozen threshold
  no memory updates
```

### 4.4.9 Controlled ablation

V3 can evaluate the pattern anchor without LLM adjustment on the same test set. This is important because it asks:

```text
Does CALL 3 add predictive information beyond the HBM2 pattern rate?
```

---

## 4.5 V4 — HBM5 with prior-vaccine acceptance proxy

**Script:** `scripts/40_hbm5_prior_vaccine_reflective_memory.py`  
**Version string:** `hbm5_openai_v1_offline_reflective_memory`

### 4.5.1 Main change from HBM2

V4 no longer uses separate LLM calls to score Threat and Barrier. Instead, Python constructs five reproducible observable proxies directly from raw NHIS variables.

| Proxy | Conceptual role | Main observed evidence |
|---|---|---|
| `observed_threat_proxy` | vulnerability and severity-related risk | age, health status, chronic burden, functional and immune vulnerability |
| `vaccine_acceptance_benefit_proxy` | revealed acceptance of preventive vaccination | prior COVID-19, pneumonia, shingles, Shingrix, and hepatitis-A vaccination behavior |
| `structural_barrier_proxy` | difficulty converting intention into access | insurance instability, cost-related unmet care, medical-bill stress, transportation, communication difficulty |
| `healthcare_cue_proxy` | opportunities that may activate preventive action | recent doctor/wellness contact, retail/virtual care, urgent care, emergency or hospital contact |
| `navigation_self_efficacy_proxy` | ability to navigate healthcare and information systems | usual care, care setting, internet access, health-information and communication tools |

These remain observed proxies, not direct HBM psychometric scales.

### 4.5.2 Step 1 — deterministic score construction

Each proxy is calculated in Python from recoded NHIS evidence. The calculation is reproducible and independent of the LLM.

The five proxy scores are then collapsed into three meta-dimensions:

```text
Motivation = mean(Observed Threat, Vaccine Acceptance/Benefit Proxy)

Capability = mean(5 - Structural Barriers,
                  Navigation Self-Efficacy)

Activation = Healthcare Cues
```

### 4.5.3 Step 2 — eight-pattern profile

Each meta-dimension is divided into High/Low using thresholds fitted on the memory-build split or fixed thresholds.

This produces eight patterns:

```text
High Motivation / High Capability / Strong Cue
High Motivation / High Capability / Weak Cue
High Motivation / Low Capability  / Strong Cue
High Motivation / Low Capability  / Weak Cue
Low Motivation  / High Capability / Strong Cue
Low Motivation  / High Capability / Weak Cue
Low Motivation  / Low Capability  / Strong Cue
Low Motivation  / Low Capability  / Weak Cue
```

Python estimates a smoothed training-only base probability for each pattern.

### 4.5.4 Data split

The default split is:

```text
memory-build = 40%
calibration  = 20%
test         = 40%
```

The memory-build split is used to fit:

- High/Low thresholds;
- HBM8 pattern base rates;
- similarity space;
- reflective memory.

### 4.5.5 Regular LLM decision call

Unlike V1–V3, there are no separate Threat and Barrier calls. Every respondent normally receives **one integrated decision call**.

#### Input payload

The prompt contains:

```text
pattern prior
  HBM8 pattern
  training-only base probability
  number of memory-build observations

respondent observed profile
  five proxy scores
  three meta-dimensions
  underlying observed evidence
  selected background variables

retrieved reflective memories
  similarity
  source pattern
  correction direction
  residual factor
  supporting variables
  correction rule
  applicability conditions
  non-generalization warning
```

#### System-prompt structure

The system prompt tells the LLM:

1. Threat and prior-vaccine acceptance jointly inform motivation.
2. Low barriers and high navigation self-efficacy inform capability.
3. Healthcare contact opportunities inform activation.
4. The HBM8 base rate is a prior, not a deterministic label.
5. The LLM should identify residual observed factors within the pattern.
6. Memory items are exception rules and should be applied only when their conditions match.
7. The true target label is not supplied.

#### Output schema

```json
{
  "residual_adjustment": 8,
  "probability_yes": 74.2,
  "deviation_direction": "higher_than_pattern",
  "dominant_hbm_factors": [
    "high vaccine acceptance",
    "strong healthcare cues"
  ],
  "residual_observed_factors": [
    "recent wellness contact"
  ],
  "reason": "The observed profile supports a modest increase above the pattern base rate."
}
```

Allowed adjustment:

```text
-30 to +30 percentage points
```

Python validates consistency between the base probability, adjustment, and final probability.

### 4.5.6 Offline reflective-memory construction

V4 changes memory learning from online to offline.

#### Stage A — no-memory decision pass

Every memory-build respondent first receives one decision call **without memory**. These predictions do not influence one another.

#### Stage B — select reflection candidates

Python identifies high-confidence errors, including:

- strong underpredictions;
- strong overpredictions;
- pattern-specific error buckets.

The number of candidates and calls per bucket is capped.

#### Stage C — reflection call

The reflection prompt receives:

```text
required correction direction
required error type
actual training outcome
pattern base probability
initial no-memory decision
respondent observed profile
```

Output schema:

```json
{
  "correction_direction": "increase",
  "error_type": "underprediction",
  "residual_factor": "recent repeated preventive healthcare contact",
  "supporting_variables": ["WELLNESS_A", "LASTDR_A"],
  "correction_rule": "Increase probability for genuinely similar respondents with strong recent preventive contact.",
  "applicability_conditions": ["same pattern", "recent preventive contact"],
  "non_generalization_warning": "Do not apply when healthcare contact is absent or only emergency-driven.",
  "reflection_confidence": 0.82,
  "estimated_memory_value": 0.76
}
```

#### Stage D — distill and freeze memory

Python filters reflections by:

- reflection confidence;
- estimated memory value;
- novelty;
- pattern/error bucket limits;
- global memory-size limit.

Only the final filtered rules are available during calibration and test.

### 4.5.7 Calibration and test

```text
calibration respondents
  → retrieve frozen reflections
  → one decision call
  → select LLM threshold
  → separately select pattern-only threshold

test respondents
  → retrieve the same frozen reflections
  → one decision call
  → apply frozen LLM threshold
```

An optional test-without-memory pass can be enabled, but the reported V4 run used the frozen-memory test pass.

### 4.5.8 What the LLM contributes

The deterministic layer already supplies:

- construct scores;
- meta-dimensions;
- pattern;
- pattern base probability.

The LLM's only predictive task is to identify **within-pattern residual evidence**. This makes the LLM contribution directly comparable with the pattern-only baseline.

---

## 4.6 V5 — HBM5-NV without other-vaccine history

**Script:** `scripts/50_hbm5_no_prior_vaccine_reflective_memory.py`  
**Version string:** `flare_vax_no_prior_vax_cot_memory_v1`

### 4.6.1 Main purpose

V5 removes the strongest but potentially shortcut-like predictor family: other-vaccine history.

The following variables are explicitly prohibited from:

- proxy scoring;
- prompt construction;
- similarity vectors;
- memory retrieval;
- reflection content.

```text
SHTCVD191_A
SHTCVD19NM2_A
SHTPNUEV_A
SHTPNEUNB_A
SHTSHINGL1_A
SHINGRIX3_A
SHTHEPA_A
SHTFLUM_A
SHTFLUY_A
```

### 4.6.2 Replacement of the vaccine-acceptance proxy

The prior-vaccine acceptance/benefit proxy is replaced by:

```text
preventive_engagement_proxy
```

This proxy is based on non-vaccine behaviors such as:

- wellness-visit behavior;
- health-information seeking;
- online communication with healthcare providers;
- online review of test results;
- broader health-management engagement.

It should not be interpreted as a direct perceived-benefit scale.

### 4.6.3 Deterministic representation

Python constructs:

```text
observed_threat_proxy
preventive_engagement_proxy
structural_barrier_proxy
healthcare_cue_proxy
navigation_self_efficacy_proxy
```

Then:

```text
Motivation = mean(Threat, Preventive Engagement)
Capability = mean(5 - Barriers, Navigation Self-Efficacy)
Activation = Healthcare Cues
```

High/Low thresholds produce the same eight-pattern structure used in V4.

### 4.6.4 Decision-call prompt structure

The prompt is a compact JSON payload containing:

```text
pattern_prior
  pattern
  general tendency
  base probability
  memory-build pattern count

respondent_profile
  five proxies
  three meta-dimensions
  HBM8 pattern
  raw observed evidence
  allowed background variables

retrieved_reflective_memories
  similarity
  correction direction
  failed reasoning stage
  correction rule
  corrected reasoning path
  applicability conditions
  contradiction conditions
  retrieval keys
  memory value
```

The system prompt explicitly instructs the model never to infer or mention any other-vaccine history.

### 4.6.5 Four-stage structured reasoning output

V5 asks the LLM to expose a concise observable reasoning trace, not unrestricted hidden chain-of-thought.

Output schema:

```json
{
  "construct_interpretation": {
    "threat": "high",
    "preventive_engagement": "moderate",
    "barriers": "low",
    "self_efficacy": "high",
    "cues": "moderate"
  },
  "reasoning_trace": {
    "stage_1_evidence_synthesis": "Summarize the observed construct evidence.",
    "stage_2_pattern_interpretation": "Explain the HBM8 pattern tendency.",
    "stage_3_residual_context": "Identify evidence not captured by the pattern average.",
    "stage_4_decision_mapping": "Map the anchor and residual evidence to probability."
  },
  "residual_adjustment": -10,
  "probability_yes": 48.0,
  "deviation_direction": "lower_than_pattern",
  "dominant_observed_factors": [
    "weak preventive engagement",
    "limited recent healthcare cues"
  ],
  "memory_application": [],
  "confidence": 0.73
}
```

Allowed residual adjustment:

```text
-25 to +25 percentage points
```

The returned probability must equal the pattern base probability plus the residual adjustment, clipped to `[0, 100]`.

### 4.6.6 Stage-specific reflection call

For a selected high-confidence error, the reflection call diagnoses where the observable reasoning process failed.

Allowed failure stages:

```text
construct_inference
pattern_interpretation
context_integration
decision_mapping
memory_misapplication
```

Output schema:

```json
{
  "correction_direction": "increase",
  "error_type": "underprediction",
  "failure_stage": "context_integration",
  "incorrect_assumption": "The model treated weak activation as decisive despite strong preventive engagement.",
  "corrected_reasoning_path": {
    "evidence_reinterpretation": "Preventive engagement provides independent positive evidence.",
    "pattern_exception": "This profile can exceed the pattern average despite weaker cues.",
    "decision_correction": "Apply a moderate upward residual adjustment."
  },
  "supporting_variables": ["WELLNESS_A", "HITCOMM_A"],
  "correction_rule": "Increase predictions for similar profiles when sustained preventive engagement offsets weak cues.",
  "applicability_conditions": ["high preventive engagement", "low structural barriers"],
  "contradiction_conditions": ["no recent health-management behavior"],
  "retrieval_keys": ["preventive engagement", "weak cue exception"],
  "reflection_confidence": 0.81,
  "estimated_memory_value": 0.77
}
```

The memory is intended to preserve not only a correction direction, but also:

- the failed stage;
- the corrected path;
- conditions under which the rule applies;
- conditions under which it should not apply.

### 4.6.7 Complete execution sequence

For the reported 5,000-person run:

```text
MEMORY-BUILD SPLIT: 2,000
  2,000 decision calls without memory
  select high-confidence errors
  reflection calls on selected candidates
  distill 6 final memory items

CALIBRATION SPLIT: 1,000
  1,000 decision calls with frozen memory
  select LLM threshold = 52
  select pattern-only threshold = 42

TEST SPLIT: 2,000
  2,000 decision calls with frozen memory
  2,000 decision calls without memory
  pattern-only predictions require no LLM call
```

This produces a same-test comparison of:

1. HBM8 pattern only;
2. LLM residual decision without memory;
3. LLM residual decision with memory.

---

# 5. Detailed comparison across methods

## 5.1 Behavioral representation and LLM responsibility

| Dimension | V1 | V1-Async | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|---|
| Main constructs | Threat, Barrier | same | Threat, Barrier | Threat, Barrier | five proxies | five proxies |
| Construct source | LLM 1–5 assessments plus rule scores | same | authoritative rule class + LLM nuance | authoritative rule class + LLM nuance | deterministic Python scores | deterministic Python scores |
| Number of patterns | 4 | 4 | 4 | 4 | 8 | 8 |
| Pattern role | qualitative direction | qualitative direction | authoritative class; directional decision prior | numeric probability anchor | numeric probability anchor | numeric probability anchor |
| LLM decision role | full binary classifier | full binary classifier | free probability estimator | bounded residual adjuster | integrated residual adjuster | structured four-stage residual adjuster |
| Final LLM output | YES/NO | YES/NO | probability + YES/NO + reason | adjustment + reason | adjustment + probability + factors | structured trace + adjustment + probability |
| Direct interpretability | low–moderate | low–moderate | moderate | high | high | highest |

## 5.2 Prompt architecture

| Version | Prompt blocks supplied to final decision call | Key system constraint |
|---|---|---|
| V1 | Threat result; Barrier result; HBM2 pattern; optional prior; context; similar raw errors | pattern is directional; output only YES/NO |
| V1-Async | identical to V1 | identical to V1 |
| V2 | authoritative pattern; CALL 1/2 nuance; context; reflected rules | return probability consistent with internal 50% decision |
| V3 | authoritative pattern; numeric base rate; CALL 1/2 nuance; context; prototypes; reflections | output only a bounded `[-20,+20]` adjustment |
| V4 | five proxies; three dimensions; HBM8 pattern/base rate; raw evidence; reflections | identify only residual factors; adjustment `[-30,+30]` |
| V5 | same high-level profile but no vaccine history; stage-aware memories | follow four observable stages; adjustment `[-25,+25]`; never infer other vaccines |

## 5.3 Memory design

| Version | What enters memory | How memory is created | When it updates | What is retrieved |
|---|---|---|---|---|
| V1 | incorrect training case | deterministic record; no reflection call | immediately after each sequential training error | nearest raw errors |
| V1-Async | incorrect training case | same as V1 | after each batch | nearest raw errors from pre-batch memory |
| V2 | structured correction rule | one reflection call after a wrong training prediction | immediately during sequential training | similar reflected rules |
| V3 prototype | high-confidence correct case | no extra LLM call | during memory-build | typical correct examples |
| V3 reflection | high-confidence incorrect case | additional reflection call | during memory-build | narrow exception rules |
| V4 | high-value residual rule | offline no-memory pass, candidate selection, reflection, filtering | built once, then frozen | similar exception rules |
| V5 | stage-specific correction rule | same offline design, with applicability and contradiction logic | built once, then frozen | similar stage-aware exceptions |

## 5.4 Calibration and leakage control

| Version | Split structure | Threshold source | Test-memory update | Strongest leakage control |
|---|---|---|---|---|
| V1 | train/test | fixed binary decision | no | test labels never enter memory |
| V1-Async | train/test | fixed binary decision | no | test labels never enter memory |
| V2 | train/test | selected from training predictions | no | frozen training reflection memory |
| V3 | memory/calibration/test | independent calibration split | no | pattern rates, memory, and threshold frozen before test |
| V4 | memory/calibration/test | independent calibration split | no | offline memory plus frozen pattern and LLM thresholds |
| V5 | memory/calibration/test | independent calibration split | no | explicit feature exclusion plus same-test memory ablation |

## 5.5 Approximate LLM-call complexity

Let:

```text
N  = total selected respondents
R  = number of reflection candidates actually sent to the LLM
Nt = test-set size
```

| Version | Approximate number of calls |
|---|---:|
| V1 | `3N` |
| V1-Async | `3N`, but parallelized |
| V2 | `3N + R` |
| V3 | `3N + R` |
| V4 | `N + R`; add `Nt` if optional no-memory test ablation is run |
| V5 reported design | `N + Nt + R`, because test is run both with and without memory |

For V4 and V5, five proxy scores do **not** mean five LLM calls. They are computed deterministically before the single integrated decision call.

## 5.6 What becomes more constrained over time

| Component | V1 | V2 | V3 | V4/V5 |
|---|---|---|---|---|
| Construct class | LLM can independently score it | Python class is authoritative | Python class is authoritative | entirely computed by Python |
| Pattern | qualitative input | authoritative classification | empirical probability anchor | empirical probability anchor |
| Absolute probability | not produced | freely produced by LLM | computed from anchor + bounded adjustment | computed from anchor + bounded adjustment |
| Error learning | raw examples | LLM correction rules | prototype + exception rule | filtered offline exception rules |
| LLM freedom | highest | lower | substantially lower | narrow residual role |

This progression is the main methodological contribution of the repository: it tests whether prediction improves when the LLM is shifted from an unconstrained end-to-end classifier toward a theory-bounded residual reasoner.

---

# 6. Script-to-result mapping

| Version | Script | Sample / test | Result artifact | Main reported result |
|---|---|---:|---|---|
| V1 | `scripts/10_hbm2_three_call.py` | 20 / 6 | `results/summaries/hbm2_v1_smoke_test.json` | Acc 0.6667; F1 0.5000; smoke test only |
| V1-Async | `scripts/11_hbm2_three_call_async.py` | 1000 / 300 | `results/raw_logs/hbm2_async_1000.txt` | Acc 0.6100; F1 0.5551 |
| V2 | `scripts/20_hbm2_reflection_calibration.py` | 1000 / 300 | `results/raw_logs/hbm2_reflection_1000.txt` | Acc 0.5133; AUC 0.5409; F1 0.5756 |
| V3 | `scripts/30_hbm2_pattern_anchor_dual_memory.py` | 1000 / 250 | `results/raw_logs/hbm2_dual_memory_1000.txt` | LLM Acc 0.6240/AUC 0.6633; pattern-only Acc 0.6160/AUC 0.6512 |
| V4 | `scripts/40_hbm5_prior_vaccine_reflective_memory.py` | 5000 / 2000 | `results/raw_logs/hbm5_prior_vaccine_5000.txt` | pattern-only Acc 0.7260/AUC 0.7658; LLM-memory Acc 0.7195/AUC 0.7332 |
| V5 | `scripts/50_hbm5_no_prior_vaccine_reflective_memory.py` | 5000 / 2000 | `results/summaries/hbm5_no_prior_vaccine_5000.json` | pattern-only Acc 0.6255/AUC 0.6812; LLM-no-memory Acc 0.6135/AUC 0.6584; LLM-memory Acc 0.5705/AUC 0.5808 |

Machine-readable mappings are available in:

```text
results/run_index.csv
results/reported_llm_hbm_metrics.csv
```

---

# 7. Final reported comparison

## 7.1 Without other-vaccine history

| Method | Accuracy | Balanced Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| HBM5-NV pattern only | 0.6255 | 0.6326 | 0.6812 | 0.6597 |
| HBM5-NV LLM without memory | 0.6135 | 0.6162 | 0.6584 | 0.6201 |
| HBM5-NV LLM with reflective memory | 0.5705 | 0.5653 | 0.5808 | 0.5077 |
| Logistic regression, 67 variables | 0.6766 | — | 0.7433 | 0.6640 |
| Random forest, 67 variables | 0.6829 | — | 0.7507 | 0.6660 |
| Gradient boosting, 67 variables | 0.6867 | — | 0.7553 | 0.6691 |
| XGBoost, 67 variables | **0.6871** | — | **0.7570** | **0.6697** |

Interpretation:

- the deterministic eight-pattern model outperformed both LLM variants;
- the no-memory LLM was better than the reflective-memory LLM;
- the six retained memory rules did not generalize well enough to improve the test set;
- conventional ML remained the strongest predictive benchmark.

## 7.2 With other-vaccine history

| Method | Accuracy | Balanced Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| HBM5 pattern only | 0.7260 | 0.7266 | 0.7658 | 0.7169 |
| HBM5 LLM with reflective memory | 0.7195 | 0.7177 | 0.7332 | 0.6976 |
| Logistic regression, 75 variables | 0.7604 | — | 0.8398 | 0.7484 |
| Random forest, 75 variables | 0.7617 | — | 0.8377 | 0.7471 |
| Gradient boosting, 75 variables | 0.7630 | — | 0.8444 | 0.7506 |
| XGBoost, 75 variables | **0.7647** | — | **0.8452** | 0.7519 |
| SVM, 75 variables | 0.7630 | — | 0.8356 | **0.7552** |

Interpretation:

- other-vaccine behavior provides a large amount of predictive information;
- the HBM5 pattern representation captures much of that signal without requiring an LLM call;
- the residual LLM layer did not improve over the pattern-only anchor;
- ML models still achieved the highest discrimination.

---

# 8. What can and cannot be concluded

## Supported by the current experiments

1. A deterministic theory layer can create strongly ordered behavioral groups.
2. Other-vaccine history is a highly predictive feature family for flu vaccination.
3. Pattern-only models can outperform an LLM residual layer.
4. Reflection memory can reduce performance when exception rules are sparse, noisy, or overgeneralized.
5. Constraining the LLM to a residual role makes its incremental value directly testable.

## Not yet supported

1. Later versions are not automatically superior because they are more complex.
2. The HBM2 development results are not a controlled version-by-version benchmark; they use different sampling and test configurations.
3. The current results do not show that LLM memory reliably learns transportable behavioral rules.
4. The observed proxies should not be interpreted as validated psychometric HBM constructs.
5. The current baseline results were supplied as aggregate console output; exact shared split assignments were not available for every model.

A paper-quality final comparison should save one common split, run every method on the same respondents, and report bootstrap confidence intervals and paired significance tests.

---

# 9. Reproduction

## 9.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set the API key through the environment:

```bash
export OPENAI_API_KEY="..."
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
```

Do not commit API keys to GitHub.

## 9.2 Data placement

Download the NHIS 2024 Sample Adult CSV and save it as:

```text
data/adult24.csv
```

The raw NHIS file is not redistributed in this repository.

## 9.3 Prepare the HBM2 cleaned dataset

```bash
python scripts/00_prepare_hbm2_data.py \
  --input_csv data/adult24.csv \
  --output_csv data/nhis2024_hbm2_clean.csv
```

## 9.4 Inspect prompts without API calls

V1:

```bash
python scripts/10_hbm2_three_call.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/hbm2_v1 \
  --sample_size 20 \
  --dry_run
```

V3:

```bash
python scripts/30_hbm2_pattern_anchor_dual_memory.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/hbm2_v3 \
  --sample_size 100 \
  --dry_run
```

V5:

```bash
python scripts/50_hbm5_no_prior_vaccine_reflective_memory.py \
  --input-csv data/adult24.csv \
  --output-dir results/runtime/hbm5_no_prior \
  --sample-size 100 \
  --dry-run
```

`--dry_run` / `--dry-run` prints representative prompts and validates the data flow without sending OpenAI requests.

## 9.5 Run ML baselines

```bash
python scripts/01_ml_baselines.py \
  --input_csv data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/ml_baselines
```

See `configs/baseline_feature_manifest.example.json` and `docs/replication_commands.md` for additional settings.

---

# 10. Research design summary

The methodological progression can be summarized as:

```text
V1
observable variables
→ LLM construct inference
→ LLM binary decision
→ raw error examples

V2
rule-based construct class
→ LLM within-class nuance
→ LLM probability
→ LLM reflection rule

V3
rule-based construct class
→ empirical HBM2 pattern probability
→ bounded LLM residual adjustment
→ prototype + reflection dual memory

V4
five deterministic proxies
→ three meta-dimensions
→ eight-pattern probability anchor
→ one LLM residual decision
→ offline reflective memory

V5
same structured design
→ remove all other-vaccine predictors
→ replace vaccine acceptance with preventive engagement
→ four-stage structured decision trace
→ same-test pattern / no-memory / memory comparison
```

The repository therefore tests a substantive design question:

> Should an LLM infer the full human decision from survey data, or should deterministic behavioral theory define the main structure while the LLM is restricted to explaining and adjusting individual deviations?

---

## Reference

Chen, R., Wang, C., Sun, Y., Zhao, X., and Xu, S. (2025). *From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs*. Proceedings of ACL 2025.
