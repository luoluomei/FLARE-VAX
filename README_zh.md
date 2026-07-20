# FLARE-VAX：基于 NHIS 2024 的流感疫苗接种行为预测

这个仓库整理了从原始 FLARE 框架迁移到流感疫苗接种预测问题的一系列实验。预测目标是 NHIS 2024 Sample Adult 数据中的 `SHTFLU12M_A`，即受访者过去 12 个月是否接种流感疫苗。

原始 FLARE 论文通过行为理论、隐含心理状态、推理模式、LLM 决策以及错误反思记忆来预测山火疏散行为。本项目从一个最接近原版结构的 HBM2 版本出发，逐步加入反思、概率校准、pattern anchor、双记忆、五个 HBM proxy、八类 pattern，以及完全排除其他疫苗接种历史的严格版本。

> 所有 HBM 分数都是由 NHIS 可观测变量构造的 theory-guided proxy，不是直接测量的心理量表。

## 版本演进

| 版本 | 核心构念 | Pattern | 每个样本的主要 LLM 调用 | Memory | 是否使用其他疫苗历史 |
|---|---|---:|---:|---|---|
| V1 HBM2 | Threat + Barrier | 4 | 3 | 直接保存训练错误 | 否 |
| V1-Async | 与 V1 相同 | 4 | 3，并发 | 与 V1 相同 | 否 |
| V2 HBM2 Reflection | Threat + Barrier | 4 | 3 + 选择性 reflection | 结构化错误修正规则 | 否 |
| V3 Pattern Anchor | Threat + Barrier | 4 | 3 + 选择性 reflection | 正确 prototype + 错误 reflection | 否 |
| V4 HBM5 | Threat、Vaccine Acceptance、Barrier、Cue、Self-Efficacy | 8 | 1 + 选择性 reflection | 反思型 exception rule | 是 |
| V5 HBM5-NV | Threat、Preventive Engagement、Barrier、Cue、Self-Efficacy | 8 | 1 + 选择性 reflection | 同时测试有/无 memory | 否 |

### V1：HBM2 三次调用

1. 从原始变量构造 rule-based threat score 和 barrier score。
2. 将两者组合为四个 pattern。
3. CALL 1 读取年龄、健康状况、慢性病和功能障碍，输出 1–5 threat score。
4. CALL 2 读取保险、费用、就医可及性、交通、语言和网络条件，输出 1–5 barrier score。
5. CALL 3 读取两个 LLM 评分、rule-based pattern、背景变量和相似训练错误，输出是否接种。
6. 训练阶段预测错误被写入 memory；测试阶段 memory 冻结。

### V2：加入反思和校准

错误样本不再只是原样存储，而是额外调用一次 LLM，生成错误原因、遗漏信号、修正规则和适用 pattern。之后在独立 calibration set 上选择分类阈值。

### V3：Pattern Anchor + 双记忆

先根据 memory split 计算每个 pattern 的训练期接种率，把它作为 base probability。LLM 不再从零预测，而只负责输出一个有限范围内的 residual adjustment。Memory 同时包含高置信正确样本 prototype 和高置信错误样本 reflection。

### V4：扩展到五个 Proxy 和八类 Pattern

五个 proxy 分别是：

- Observed Threat
- Vaccine Acceptance / Benefit Proxy
- Structural Barrier
- Healthcare Cue
- Navigation Self-Efficacy

再组合成：

```text
Motivation = mean(Threat, Vaccine Acceptance)
Capability = mean(5 - Barrier, Self-Efficacy)
Activation = Cue
```

三个维度各自划分 High/Low，形成八类 pattern。五个 proxy 均由 Python 确定性计算，因此每个样本只需要一次 LLM decision call。

### V5：完全排除其他疫苗历史

所有非目标疫苗变量都不会进入 score、prompt、memory retrieval 或 similarity representation。原来的 vaccine acceptance proxy 被 preventive engagement 替代，使用 wellness visit、健康信息搜索、在线联系医生和查看检查结果等行为变量。

## 代码与运行结果对应

下表将每个保留脚本与对应的运行记录直接连接起来。HBM2 开发版本的抽样方式和测试集大小不同，因此只能用于记录方法演进，不能把数值变化直接解释为版本升级带来的因果提升。

| 版本 | 脚本 | 样本/测试集 | 对应结果 | 主要指标 |
|---|---|---:|---|---|
| V1 | `scripts/10_hbm2_three_call.py` | 20 / 6 | `results/summaries/hbm2_v1_smoke_test.json` | Acc 0.6667；F1 0.5000，仅为 smoke test |
| V1-Async | `scripts/11_hbm2_three_call_async.py` | 1000 / 300 | `results/raw_logs/hbm2_async_1000.txt` | Acc 0.6100；F1 0.5551 |
| V2 | `scripts/20_hbm2_reflection_calibration.py` | 1000 / 300 | `results/raw_logs/hbm2_reflection_1000.txt` | Acc 0.5133；AUC 0.5409；F1 0.5756 |
| V3 | `scripts/30_hbm2_pattern_anchor_dual_memory.py` | 1000 / 250 | `results/raw_logs/hbm2_dual_memory_1000.txt` | LLM Acc 0.6240 / AUC 0.6633；Pattern-only Acc 0.6160 / AUC 0.6512 |
| V4 | `scripts/40_hbm5_prior_vaccine_reflective_memory.py` | 5000 / 2000 | `results/raw_logs/hbm5_prior_vaccine_5000.txt` | Pattern-only Acc 0.7260 / AUC 0.7658；LLM-memory Acc 0.7195 / AUC 0.7332 |
| V5 | `scripts/50_hbm5_no_prior_vaccine_reflective_memory.py` | 5000 / 2000 | `results/summaries/hbm5_no_prior_vaccine_5000.json` | Pattern-only Acc 0.6255 / AUC 0.6812；LLM-no-memory Acc 0.6135 / AUC 0.6584；LLM-memory Acc 0.5705 / AUC 0.5808 |

机器可读的完整索引位于 `results/run_index.csv`，全部指标位于 `results/reported_llm_hbm_metrics.csv`。

## 最终结果

### 不允许使用其他疫苗接种历史

| 方法 | Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|
| HBM5-NV Pattern Only | 0.6255 | 0.6812 | 0.6597 |
| HBM5-NV LLM Without Memory | 0.6135 | 0.6584 | 0.6201 |
| HBM5-NV LLM With Memory | 0.5705 | 0.5808 | 0.5077 |
| Logistic Regression | 0.6766 | 0.7433 | 0.6640 |
| XGBoost | **0.6871** | **0.7570** | **0.6697** |

### 允许使用其他疫苗接种历史

| 方法 | Accuracy | ROC-AUC | F1 |
|---|---:|---:|---:|
| HBM5 Pattern Only | 0.7260 | 0.7658 | 0.7169 |
| HBM5 LLM With Memory | 0.7195 | 0.7332 | 0.6976 |
| Logistic Regression | 0.7604 | 0.8398 | 0.7484 |
| XGBoost | **0.7647** | **0.8452** | **0.7519** |

当前结果不能支持“越复杂越好”：两个 HBM5 实验中，pattern-only 都优于 LLM with memory；在不使用其他疫苗历史时，memory 还明显降低了表现。因此后续研究重点应该是验证 LLM residual 是否真正带来 pattern 之外的信息，以及 memory rule 是否能够在 held-out neighbor 上产生稳定增益。

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

准备 HBM2 数据：

```bash
python scripts/00_prepare_hbm2_data.py \
  --input_csv data/adult24.csv \
  --output_csv data/nhis2024_hbm2_clean.csv
```

只检查 prompt、不调用 API：

```bash
python scripts/50_hbm5_no_prior_vaccine_reflective_memory.py \
  --input-csv data/adult24.csv \
  --output-dir results/runtime/hbm5_no_prior \
  --sample-size 100 \
  --dry-run
```

更完整的方法说明、变量映射和复现实验命令位于 `docs/`。
