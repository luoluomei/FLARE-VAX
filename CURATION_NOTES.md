# Curation notes

## Retained

- V4 and V5 FLARE-VAX scripts.
- Conventional ML baselines.
- Unified zero-shot/few-shot ICL benchmark.
- HBM-CoPB and HBM-PB&J transfer implementation.
- Initial SILIC-inspired V4 implementation.
- RG-FLARE-VAX reward-guided V4/V5 extension and compact full-run results.
- TRBM-FLARE-VAX theory-residual V4/V5 extension, offline ablation script, method notes, and compact full-run results.
- Compact metrics, configuration examples, documentation, and references.

## Excluded

- Raw NHIS data and respondent-level profiles.
- Row-level predictions, JSONL API transcripts, support-set artifacts, threshold-search traces, checkpoints, and notebooks with execution state.
- Earlier HBM2 development files superseded by V4/V5.
- Local absolute paths, credentials, and failure-only run directories.

## Intentionally pending

- HBM-PB&J V4 with Llama 4 Scout 17B.
- All V5 Llama 3 70B unified ICL rows.
- V4 Llama 3 70B similarity-selected and representative 8-shot rows.
- Full benchmark of the original `80_silic_v4_inverse_contextual_reward_asu.py` implementation (distinct from the completed RG/TRBM extensions).
- FLARE-VAX transfer of the supervised fine-tuning method from arXiv:2601.03534.

Blank public metrics are deliberate and must not be interpreted as zero performance.
