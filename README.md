# FLARE-VAX: Theory-Guided Influenza Vaccination Prediction from NHIS 2024

This repository summarizes a sequence of experiments that transfer and extend the FLARE framework from wildfire evacuation prediction to influenza vaccination behavior prediction using the **2024 National Health Interview Survey (NHIS) Sample Adult file**.

- **Outcome:** whether a respondent received a flu vaccine during the past 12 months (`SHTFLU12M_A`).
- **Starting point:** FLARE uses behavioral theory, latent-perception reasoning paths, LLM inference, and memory from past errors for human decision prediction.
- **Research progression:** a minimal two-construct transfer is gradually expanded into five observable Health Belief Model (HBM)-inspired proxies, eight behavioral patterns, explicit pattern anchors, reflective memory, calibration, and a strict no-other-vaccine-history setting.

Original FLARE paper: Chen et al. (2025), *From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs* — https://aclanthology.org/2025.acl-long.1438/

> **Interpretation warning:** the HBM variables in this repository are **theory-guided observed proxies**, not direct psychometric measurements of private beliefs such as perceived susceptibility, perceived benefits, or self-efficacy.

## Repository map

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

docs/
  method_evolution.md
  variable_dictionary.md
  evaluation_notes.md
  replication_commands.md

results/
  baseline_all_models.csv
  reported_llm_hbm_metrics.csv
  benchmark_without_other_vaccine_history.csv
  benchmark_with_other_vaccine_history.csv
  summaries/
  raw_logs/
```

## Method evolution

### V1 — HBM2 three-call direct transfer

**Purpose:** construct the closest minimal transfer from the original two-perception FLARE pipeline.

**Observed variables**

- Threat: age, self-rated health, chronic conditions, functional difficulty, disability, BMI.
- Barriers: insurance, cost-related unmet care, medical-bill stress, usual source of care, transportation, language, and internet access.
- Decision context: recent healthcare contact, digital health use, education, income-to-poverty category, wellness recency, and optionally race/Hispanic public-use group.

**Constructs and patterns**

1. A deterministic HBM2 threat score is computed from cleaned NHIS variables.
2. A deterministic HBM2 barrier score is computed from cleaned NHIS variables.
3. These form four patterns:
   - P0: High Threat / Low Barrier
   - P1: High Threat / High Barrier
   - P2: Low Threat / Low Barrier
   - P3: Low Threat / High Barrier

**LLM calls per respondent: 3**

1. CALL 1 converts the threat profile into a 1–5 threat assessment.
2. CALL 2 converts the barrier profile into a 1–5 barrier assessment.
3. CALL 3 receives both assessments, the precomputed pattern, selected context, and retrieved training errors, then outputs `YES` or `NO`.

**Memory:** incorrect training cases are stored directly and retrieved by cosine similarity. Test labels never update memory.

**Script:** `scripts/10_hbm2_three_call.py`

### V1-Async — concurrent implementation

This version preserves the V1 research design but runs CALL 1 and CALL 2 concurrently and processes multiple respondents at once. It is an **engineering acceleration**, not a distinct theoretical method.

**Script:** `scripts/11_hbm2_three_call_async.py`

### V2 — HBM2 reflective memory and calibration

V2 adds a more faithful reflection loop:

1. CALL 1: threat assessment.
2. CALL 2: barrier assessment.
3. CALL 3: probability of vaccination plus a concise reason.
4. Selected training errors receive an additional reflection call.
5. Structured error explanations are stored as memory.
6. A decision threshold is selected on a held-out calibration set.
7. Memory and the threshold are frozen before test evaluation.

The main difference from V1 is that memory contains **correction rules derived from error reflection**, rather than only raw error examples.

**Script:** `scripts/20_hbm2_reflection_calibration.py`

### V3 — HBM2 pattern anchor and balanced dual memory

V3 separates the pattern prior from the LLM residual decision.

1. Estimate a smoothed vaccination base rate for each HBM2 pattern using the memory-build split only.
2. Use the pattern base rate as the respondent's starting probability.
3. Ask the LLM for a bounded residual adjustment rather than an unconstrained prediction.
4. Store two memory types:
   - **Prototypes:** high-confidence correct cases.
   - **Reflections:** high-confidence incorrect cases with correction rules.
5. Balance retrieval across the two memory types.
6. Select the final threshold on an independent calibration split.

This version tests whether the LLM adds information **beyond** the HBM2 pattern anchor.

**Script:** `scripts/30_hbm2_pattern_anchor_dual_memory.py`

### V4 — HBM5 with prior-vaccine acceptance proxy

V4 expands the behavioral representation from two constructs to five deterministic proxies.

| Proxy | Main observed inputs |
|---|---|
| Observed threat | age, health status, chronic burden, functional vulnerability, immune vulnerability |
| Vaccine acceptance / benefit proxy | COVID-19, pneumonia, shingles, Shingrix, and hepatitis A vaccination history |
| Structural barriers | insurance instability, cost-related unmet care, bill stress, transportation, communication difficulty |
| Healthcare cues | doctor/wellness recency, retail or virtual care, urgent/emergency/hospital contact |
| Navigation self-efficacy | usual source of care, care setting, internet access, digital health use, communication capacity |

The five proxies are collapsed into three meta-dimensions:

```text
Motivation = mean(Threat, Vaccine-Acceptance/Benefit Proxy)
Capability = mean(5 - Barriers, Navigation Self-Efficacy)
Activation = Healthcare Cues
```

High/low splits on the three dimensions produce **eight HBM-style patterns**.

**LLM calls per respondent: 1 decision call.** The five proxy scores are deterministic; the LLM receives the complete structured profile, a training-only pattern base rate, and retrieved reflective memories. It returns a residual probability adjustment. Additional reflection calls are made only for selected high-confidence training errors.

**Script:** `scripts/40_hbm5_prior_vaccine_reflective_memory.py`

### V5 — HBM5 without other-vaccine history

V5 is the strictest and most policy-relevant version. Every vaccine variable other than the flu-vaccination target is excluded from scoring, prompting, retrieval, and similarity representations.

The prior-vaccine acceptance proxy is replaced by **preventive engagement**:

- wellness-visit behavior,
- health-information seeking,
- online doctor communication,
- online test-result review.

The three meta-dimensions remain:

```text
Motivation = mean(Threat, Preventive Engagement)
Capability = mean(5 - Barriers, Navigation Self-Efficacy)
Activation = Healthcare Cues
```

This version evaluates the same test cases three ways:

1. HBM8 pattern only.
2. LLM without memory.
3. LLM with frozen reflective memory.

**Script:** `scripts/50_hbm5_no_prior_vaccine_reflective_memory.py`

## Script-to-result index

The preserved scripts and supplied run artifacts are mapped below. The HBM2 development runs use different sampling policies and test sizes, so their metric differences should not be interpreted as a controlled causal effect of added complexity.

| Version | Script | Sample / test | Result artifact | Main reported result |
|---|---|---:|---|---|
| V1 | `scripts/10_hbm2_three_call.py` | 20 / 6 | `results/summaries/hbm2_v1_smoke_test.json` | Acc 0.6667; F1 0.5000; smoke test only |
| V1-Async | `scripts/11_hbm2_three_call_async.py` | 1000 / 300 | `results/raw_logs/hbm2_async_1000.txt` | Acc 0.6100; F1 0.5551 |
| V2 | `scripts/20_hbm2_reflection_calibration.py` | 1000 / 300 | `results/raw_logs/hbm2_reflection_1000.txt` | Acc 0.5133; AUC 0.5409; F1 0.5756 |
| V3 | `scripts/30_hbm2_pattern_anchor_dual_memory.py` | 1000 / 250 | `results/raw_logs/hbm2_dual_memory_1000.txt` | LLM Acc 0.6240/AUC 0.6633; pattern-only Acc 0.6160/AUC 0.6512 |
| V4 | `scripts/40_hbm5_prior_vaccine_reflective_memory.py` | 5000 / 2000 | `results/raw_logs/hbm5_prior_vaccine_5000.txt` | Pattern-only Acc 0.7260/AUC 0.7658; LLM-memory Acc 0.7195/AUC 0.7332 |
| V5 | `scripts/50_hbm5_no_prior_vaccine_reflective_memory.py` | 5000 / 2000 | `results/summaries/hbm5_no_prior_vaccine_5000.json` | Pattern-only Acc 0.6255/AUC 0.6812; LLM-no-memory Acc 0.6135/AUC 0.6584; LLM-memory Acc 0.5705/AUC 0.5808 |

A machine-readable mapping is available at `results/run_index.csv`; full metrics are in `results/reported_llm_hbm_metrics.csv`.

## Final reported results

### A. Other vaccine history is not allowed

| Model | Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|
| HBM5-NV pattern only | 0.6255 | 0.6812 | 0.6597 |
| HBM5-NV LLM without memory | 0.6135 | 0.6584 | 0.6201 |
| HBM5-NV LLM with reflective memory | 0.5705 | 0.5808 | 0.5077 |
| Logistic regression, 67 cleaned variables | 0.6766 | 0.7433 | 0.6640 |
| XGBoost, 67 cleaned variables | **0.6871** | **0.7570** | **0.6697** |

**Finding:** the eight-pattern rule system outperformed both LLM variants. Reflective memory substantially reduced performance in this run, suggesting that the distilled exception rules were too sparse, too noisy, or insufficiently transferable.

### B. Other vaccine history is allowed

| Model | Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|
| HBM5 pattern only | 0.7260 | 0.7658 | 0.7169 |
| HBM5 LLM with reflective memory | 0.7195 | 0.7332 | 0.6976 |
| Logistic regression, 75 cleaned variables | 0.7604 | 0.8398 | 0.7484 |
| XGBoost, 75 cleaned variables | **0.7647** | **0.8452** | **0.7519** |

**Finding:** adding other vaccine history produces a large performance increase for both HBM and ML approaches. The HBM5 pattern-only model again outperformed the LLM residual-decision layer, while the conventional ML baselines remained strongest.

Detailed metrics for every supplied model are in `results/`.

## Important comparison caveat

The archived experiments were not all run on one identical split:

- HBM2 development versions use different sample sizes, class sampling policies, and test sizes.
- HBM5 experiments use 5,000 selected respondents with 2,000-person test sets.
- The supplied ML console output did not include exact split assignments.

Therefore, the tables above summarize **reported performance**, not a definitive controlled head-to-head benchmark. A final paper-quality comparison should reuse one saved split assignment across all methods and report bootstrap confidence intervals.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the NHIS 2024 Sample Adult CSV and save it as `data/adult24.csv`.

### Prepare the HBM2 data

```bash
python scripts/00_prepare_hbm2_data.py \
  --input_csv data/adult24.csv \
  --output_csv data/nhis2024_hbm2_clean.csv
```

### Inspect prompts without API calls

```bash
python scripts/10_hbm2_three_call.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/hbm2_v1 \
  --sample_size 20 \
  --dry_run
```

```bash
python scripts/50_hbm5_no_prior_vaccine_reflective_memory.py \
  --input-csv data/adult24.csv \
  --output-dir results/runtime/hbm5_no_prior \
  --sample-size 100 \
  --dry-run
```

### Run with OpenAI

Set the key through the environment rather than committing it:

```bash
export OPENAI_API_KEY="..."
```

Example:

```bash
python scripts/50_hbm5_no_prior_vaccine_reflective_memory.py \
  --input-csv data/adult24.csv \
  --output-dir results/runtime/hbm5_no_prior_5000 \
  --sample-size 5000 \
  --model gpt-4o-mini-2024-07-18 \
  --run-test-without-memory
```

## Main conclusions from the current experiments

1. **Behavioral structure is useful.** The deterministic HBM5 pattern model is competitive with simpler HBM2 versions and remains interpretable.
2. **Other vaccine history is highly predictive.** Removing it causes a substantial drop across both theory-guided and ML approaches.
3. **More LLM complexity did not guarantee better prediction.** In both 5,000-person experiments, pattern-only prediction beat the LLM with memory.
4. **Memory needs stronger validation.** Future work should ablate memory retrieval, require minimum support for each correction rule, and estimate whether a memory item improves held-out neighbors before retaining it.
5. **ML remains the predictive upper benchmark.** XGBoost achieved the strongest reported accuracy and AUC in both feature-policy settings.

## Reproducibility status

- All six supplied experiment scripts are preserved as self-contained snapshots with clearer filenames.
- Supplied console logs and extracted JSON summaries are archived under `results/`.
- The HBM2 cleaner and ML baseline runner were added to make the repository easier to rerun.
- Exact reproduction of the archived 67/75-variable ML results requires the original cleaned feature manifest and split assignments, which were not present in the supplied files.
