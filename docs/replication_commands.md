# Replication Commands

All commands assume execution from the repository root.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."
```

## 2. Prepare HBM2 data

```bash
python scripts/00_prepare_hbm2_data.py \
  --input_csv data/adult24.csv \
  --output_csv data/nhis2024_hbm2_clean.csv
```

## 3. V1 HBM2 smoke test

```bash
python scripts/10_hbm2_three_call.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/v1_hbm2 \
  --sample_size 20 \
  --balanced_sample \
  --memory_k 3
```

## 4. V1-Async HBM2

```bash
python scripts/11_hbm2_three_call_async.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/v1_async_1000 \
  --sample_size 1000 \
  --balanced_sample \
  --concurrent_samples 24 \
  --max_concurrent_requests 48
```

## 5. V2 reflection and calibration

```bash
python scripts/20_hbm2_reflection_calibration.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/v2_reflection_1000 \
  --sample_size 1000 \
  --no-balanced_sample
```


## 6. V3 pattern anchor and dual memory

```bash
python scripts/30_hbm2_pattern_anchor_dual_memory.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --output_dir results/runtime/v3_dual_memory_1000 \
  --sample_size 1000 \
  --class_sampling proportional
```

## 7. V4 HBM5 with other vaccine history

```bash
python scripts/40_hbm5_prior_vaccine_reflective_memory.py \
  --input-csv data/adult24.csv \
  --output-dir results/runtime/v4_hbm5_prior_5000 \
  --sample-size 5000 \
  --class-sampling proportional \
  --memory-ratio 0.40 \
  --calibration-ratio 0.20 \
  --test-ratio 0.40
```

## 8. V5 HBM5 without other vaccine history

```bash
python scripts/50_hbm5_no_prior_vaccine_reflective_memory.py \
  --input-csv data/adult24.csv \
  --output-dir results/runtime/v5_hbm5_no_prior_5000 \
  --sample-size 5000 \
  --memory-ratio 0.40 \
  --calibration-ratio 0.20 \
  --test-ratio 0.40 \
  --run-test-without-memory
```

## 9. ML baselines

Generic baseline from a cleaned table:

```bash
python scripts/01_ml_baselines.py \
  --data_path data/nhis2024_hbm2_clean.csv \
  --target vaccinated \
  --sample_size 2000 \
  --models logistic,xgboost \
  --output_csv results/runtime/ml_baselines_smoke.csv
```

The prepared HBM2 table does not contain other-vaccine variables, so the command above is only a runner smoke test. For exact reproduction of the archived 67/75-feature comparison, pass the original broader cleaned table and feature manifest:

```bash
python scripts/01_ml_baselines.py \
  --data_path path/to/original_cleaned_table.csv \
  --target vaccinated \
  --feature_manifest path/to/selected_llm_variable_manifest.json \
  --output_csv results/runtime/ml_baselines_exact.csv
```

## 10. Rebuild result tables

```bash
python scripts/90_collect_results.py
```
