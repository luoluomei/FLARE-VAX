# Replication commands

Use `python <script> --help` for the full argument list. The public repository keeps only summary outputs; full row-level outputs should be written outside the Git checkout.

## ML

```bash
python scripts/01_ml_baselines.py --data_path /path/to/adult24.csv --output_csv /path/to/output/ml.csv
```

## Unified ICL grid

```bash
python scripts/60_llm_icl_benchmark_asu.py --input-csv /path/to/adult24.csv --output-dir /path/to/output/icl --v4-reference-split /path/to/v4_split.csv --v5-reference-split /path/to/v5_split.csv
```

## HBM-CoPB / HBM-PB&J

```bash
python scripts/70_hbm_copb_pbj_baselines_asu.py --help
```

## SILIC-inspired V4

```bash
python scripts/80_silic_v4_inverse_contextual_reward_asu.py --help
```

Do not commit API keys, raw NHIS data, respondent-level predictions, or local output paths.

## RG-FLARE-VAX reward-guided extension

```bash
python scripts/81_rg_flare_vax_reward_memory_asu.py --help
```

## TRBM-FLARE-VAX theory-residual extension

```bash
python scripts/82_trbm_flare_vax_asu.py --help
```

Optional ablations reuse artifacts from a completed TRBM run and do not require new API calls:

```bash
python scripts/83_trbm_ablation_asu.py --help
```
