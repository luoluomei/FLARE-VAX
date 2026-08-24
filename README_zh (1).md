# FLARE-VAX：基于 NHIS 2024 的行为理论引导流感疫苗接种预测

[English](README.md)

FLARE-VAX 研究行为理论结构能否提升基于 **2024 National Health Interview Survey (NHIS) Sample Adult** 的流感疫苗接种预测。开发时使用的原始文件约包含 **32,629 名 respondent、630 个变量**，预测目标为：

```text
SHTFLU12M_A：过去 12 个月是否接种流感疫苗
```

经过 target 与 feature-policy 筛选后，V4 评估 32,132 人，V5 评估 32,130 人。默认划分为 **40% memory-build / 20% calibration / 40% test**。原始 NHIS 数据以及逐 respondent 的 API log 不在 GitHub 中公开。

> 本项目中的 HBM quantity 是根据 NHIS 可观察变量构造的 theory-guided proxy，不是对个体私人信念的直接 psychometric measurement。

## 1. 数据与预测设置

Sample Adult 文件包含人口统计、健康状态、慢性病、保险与医疗负担、healthcare access、utilization、digital health engagement 以及 vaccination history。传统 ML 直接使用选定的 raw/engineered feature；LLM 方法则在相同 V4/V5 feature restriction 下接收 human-readable respondent profile。

- **V4 — 保留其他疫苗史：** baseline 75 个 feature，可使用 COVID-19、pneumonia、shingles/Shingrix、hepatitis-A 等非目标疫苗历史。
- **V5 — 去除其他疫苗史：** baseline 67 个 feature，所有 non-target vaccination-history variable 都从 scoring、prompt、retrieval 和 memory construction 中排除。

## 2. V4 与 V5

- **V4（包含其他疫苗史）**：允许 COVID、肺炎、带状疱疹和甲肝等非目标疫苗史进入 vaccine acceptance/benefit proxy，共 75 个 ML 特征。
- **V5（排除其他疫苗史）**：从分数、prompt、retrieval 和 memory 中排除全部非目标疫苗史，用 wellness、健康信息查询、医生沟通和结果查看等非疫苗预防行为替代，共 67 个 ML 特征。

两者都构造 threat、acceptance/engagement、barriers、healthcare cues、navigation self-efficacy 五个 proxy，再合并成 Motivation、Capability、Activation，并形成 8 个行为 pattern。Pattern prior 只从 memory split 估计，reflection 只基于训练侧错误建立并在 calibration/test 前冻结。

## 3. 从早期 Error Analysis 到当前重构

这一版不再继续沿用后来逐层增加 numerical prior、failure router 和 residual model 的路线，而是回到原始 V4/V5 的核心问题：

> **HBM 应该给 LLM 提供什么结构，历史错误应该怎样变成可迁移的 evidence，而不是由另一个 ML 模型直接替代 LLM 完成个体概率预测。**

### 3.1 早期 error analysis 暴露出的稳定问题

此前的 exploratory error analysis 给出了几个非常清楚的趋势。它们不是最终因果结论，而是帮助我们重新设计方法的 empirical diagnostics：

- **Low-Motivation pattern 更容易出现 false negative，而 High-Motivation pattern 更容易出现 false positive。** 最难的 case 往往正是实际行为与 HBM-derived tendency 方向相反的人。
- **年龄呈现 behavioral reversal。** 较年轻的实际 vaccinator 更容易被低估，而较老的实际 non-vaccinator 更容易被高估。这说明 objective risk / age 对总体预测很有用，但不能被当成个体行为的确定规则。
- **V4 的其他疫苗史非常强，但也形成了明显的 cross-vaccine transfer failure。** “其他疫苗行为 → 流感疫苗行为”的 transfer 在总体上有效，但不是对每个人都成立。
- **Healthcare cue 存在 granularity 问题。** 急诊、住院、一般 healthcare contact 和 preventive wellness contact 都可能提高 aggregate cue，但它们的行为含义并不相同；NHIS 中的 healthcare contact 更接近“有接触/有机会”，不能直接解释成“医生推荐”。
- **Construct conflict 反复出现。** 一些 respondent 的五个 HBM-derived support 内部高度不一致，最终被压缩成同一个 HBM8 pattern 后会损失信息。
- **旧 residual-memory correction 并不总是安全。** 在一次 V4 diagnostic 中，大量 false negative 反而收到了 decrease correction，说明 memory/LLM 如果只重复先验的主导方向，可能把真正的 theory exception 推得更错。

这些发现仍然保留，但它们现在只用来定义 **error-analysis descriptor、theory-failure memory 和 retrieval dimension**，不再直接训练一个新的 outcome predictor。

### 3.2 为什么不继续使用后来的 numerical reward / ML refinement 作为最终方法

后续几个实验版本确实显著提高了预测性能，但也逐渐偏离了原始研究问题。

第一类改版先把原来的 `P_pattern` 进一步拟合成更个体化的 numerical prior。这个 prior 使用 HBM-derived state 进行监督式数值学习，效果比原始 pattern anchor 强很多。之后 error analysis 又进一步训练了一个 supervised ML router，学习“这个 numerical theory prior 在什么地方会失败”，并把 router 的结果重新作用到 probability correction 上。

更后面的 residual 版本又把同 split 的 XGBoost 当作诊断上限，专门分析 `ML correct / theory-guided model wrong` 的 case，然后训练 residual model 去学习 `actual - current_probability`。这些版本最后可以非常接近 full-feature XGBoost 的 performance，但整个 information flow 已经逐渐变成：

```text
HBM representation
   ↓
strong supervised numerical prior
   ↓
ML failure / residual model
   ↓
probability correction
   ↓
LLM auxiliary reasoning
```

这在工程上并不是错误，甚至 accuracy 很高；问题是它回答的越来越像 **“如何用 HBM 特征辅助一个 ML ensemble”**，而不是我们真正想研究的：

> **behavioral theory 如何组织 LLM 的个体化判断，以及 LLM 能否利用 historical theory failures 去判断一个新 respondent 是否应该偏离 theory tendency。**

因此当前版本保留这些探索带来的 error-analysis insight，但放弃让 ML 直接生成新的 prior、failure probability 或 residual probability。

---

## 4. 当前主方法：TFM-FLARE-VAX

**TFM-FLARE-VAX = Theory-Failure Memory + Reward-Valued Contrastive Retrieval**

代码：`scripts/81_theory_failure_memory_flare_vax_asu.py`  
Notebook：`notebooks/81_theory_failure_memory_flare_vax_asu.ipynb`

默认测试模型：

- `llama4-scout-17b`
- `llama4-maverick-17b`

完整流程：

```text
                         NHIS respondent
                               │
                               ▼
                     HBM-derived profile
                               │
                               ▼
                         HBM8 pattern
                               │
                               ▼
                    Pattern empirical anchor
                         P_pattern
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ordinary behavioral       theory-failure
                memory                 memory
                    │                     │
                    └──────────┬──────────┘
                               │
                     Reward/Q retrieval
                               │
                               ▼
                       contrastive evidence
                               │
                               ▼
                              LLM
                  theory-consistent evidence
                  theory-conflicting evidence
                  historical exception check
                  individualized integration
                               │
                               ▼
                            P_final
```

### 4.1 回到原始 `P_pattern`：不再用 ML 计算 individualized prior

当前版本恢复原 V4/V5 的先验设计：五个 deterministic HBM-derived proxy 形成 Motivation / Capability / Activation；memory split 的 median 将三个 meta-dimension 划成 High / Low；得到 8 个 HBM pattern；`P_pattern` 只是该 pattern 在 memory split 中的 smoothed empirical vaccination rate。

同一个 HBM8 pattern 内的 respondent 在进入 LLM 前拥有相同的 population-level behavioral anchor。Memory respondent 自己计算 anchor 时使用 leave-one-out，避免自己的 label 泄漏到自己的 `P_pattern`。

这里**没有 logistic regression、GBDT、XGBoost 或其他 supervised ML probability model**。

### 4.2 Error analysis 放在哪里：只在 offline memory-building 阶段分析 theory failure

Error analysis 不作为 test-time predictor。它只使用 memory split，并在 memory 内进一步划成 `error discovery` 与 `error validation`，用于发现并验证可重复的 same-pattern failure signature。

最基础的 strong theory failure 默认定义为：

```text
strong positive theory exception:
actual = vaccinated AND P_pattern <= 35%

strong negative theory exception:
actual = not vaccinated AND P_pattern >= 65%
```

35% / 65% 可通过 CLI 调整。普通 case 则是 `P_pattern` 在 50% classification boundary 上与实际 outcome 一致的 historical respondent。

Error analysis 最重要的比较不是“所有错 vs 所有对”，而是：

```text
同一个 HBM8 pattern 内
    theory exception
        vs.
    ordinary respondent
```

这样才能回答：**在 theory state 已经相同的情况下，还有哪些 observed configuration 与偏离 theory tendency 有关？**

### 4.3 新增的 error-analysis feature：描述 theory compression / mismatch，而不是预测 label

当前脚本从原始 V4/V5 observed profile 中额外构造一组 deterministic diagnostic feature。这些 feature 不进入任何 supervised probability model，只用于 error analysis、memory description 和 retrieval。

共同 feature 包括：

- `theory_support_mean / std / range`
- `theory_support_high_count / low_count`
- `age`、`chronic_count`
- `threat_component_std / range`
- `capability_component_gap`
- `preventive_contact_support`
- `acute_contact_support`
- `acute_minus_preventive_contact`
- `cue_component_spread`
- `cost_unmet_care_count / hard_access_barrier_count`

V4 进一步增加：`cross_vaccine_mean`、`cross_vaccine_std / range`、`cross_vaccine_positive_count`、`cross_vaccine_mixedness`、`cross_vaccine_theory_disagreement`。这些变量专门描述前面 error analysis 暴露出的 **cross-vaccine transfer mismatch**。

V5 进一步增加：`engagement_component_std / range`、`care_connected_engagement`、`information_minus_care_connected`、`information_only_engagement`，用于区分“信息搜索很多”和“真正与 preventive/care action 连接的 engagement”。

### 4.4 如何判断这些 feature 真的是稳定的 failure signature

脚本在 memory split 内把 error-analysis 数据再次分成 discovery / validation 两半。对于每个 HBM8 pattern，比较 strong theory exception 与 same-pattern ordinary case 的 standardized mean difference（SMD）。

一个 feature 只有满足 discovery 中达到最小 SMD、validation 中方向一致且仍达到最小 SMD，才标记为 `stable signature`。这些 stable signature 的绝对效应大小会转成 **retrieval feature weight**。

因此 error analysis 的结果不是变成一个新的 classifier，而是告诉 retrieval：

> 在寻找 historical analogue / exception 时，哪些 observed dimensions 应该被更认真地比较。

### 4.5 Deterministic failure-profile tag

脚本还根据 discovery split 的 quantile 构造不依赖 outcome label 的 profile descriptor，例如：

- `high_construct_conflict`
- `capability_component_mismatch`
- `acute_contact_not_preventive`
- `cue_granularity_mismatch`
- `threat_component_heterogeneity`
- V4：`mixed_cross_vaccine_behavior`
- V4：`cross_vaccine_transfer_mismatch_candidate`
- V5：`information_only_engagement`
- V5：`engagement_granularity_mismatch`
- V5：`information_vs_care_connected_mismatch`
- `younger_low_theory_support_profile`
- `older_high_theory_support_profile`

这些 tag **不是“这个人一定会预测错”的结论，更不是因果机制**。它们只是把 error analysis 发现的 observed configuration 变成检索时可以直接比较的 descriptor。脚本另外输出 discovery/validation 中这些 tag 在 theory exception 与 ordinary case 之间的 prevalence difference。

### 4.6 两类 historical memory

**Ordinary behavioral memory**：`P_pattern` 的方向与真实 outcome 一致的 historical respondent。脚本在每个 HBM8 pattern 内选择靠近该 pattern observed-feature centroid 的代表性 ordinary case。

**Theory-failure memory**：`P_pattern` 与真实 behavior 强烈冲突的 historical respondent。Theory-failure memory 保存 historical outcome、source HBM8 pattern 与 `P_pattern`、exception direction/severity、failure-profile tag、HBM score 和 error-analysis feature summary。

当前版本不要求 LLM 在 memory-building 阶段先生成一段“为什么错”的自由文本。原因是我们希望先让 **data 定义 observable failure signature**，再让 prediction-time LLM 判断这些 historical configuration 是否真正适用于当前 respondent。

### 4.7 Reward / Q 在这一版只负责 memory value，不负责概率

每条 ordinary / theory-failure memory 都得到一个 empirical Q-value。系统在其他 memory respondents 中寻找相似 neighbor，并进行一个小的 counterfactual test：如果按照该 historical case 的 outcome direction，对 neighbor 的 `P_pattern` 做一个很小的 increase / decrease，平均 log loss 是否下降？

简化表示：

```text
Q(memory)
≈ historical local loss improvement
when the memory's direction is applied to similar memory respondents
```

Q 高代表这条 historical evidence 在相似人群中更有方向性价值；Q 低或负则说明它可能更 idiosyncratic。

**Q 不修改 `P_pattern`，不输出 vaccination probability，也不训练 outcome classifier。**

### 4.8 Reward-guided contrastive retrieval

对新的 respondent，retrieval 默认同时考虑：

```text
0.45 × original V4/V5 similarity
+ 0.20 × error-analysis-feature similarity
+ 0.25 × empirical Q score
+ 0.07 × same-pattern bonus
+ 0.03 × failure-profile tag overlap
```

然后分别返回 top ordinary behavioral memories 与 top theory-failure memories。这两组 memory 构成 **contrastive evidence**：LLM 同时看到“通常遵循 theory 的历史人”和“在相似 observed configuration 下曾经偏离 theory 的历史人”。

### 4.9 LLM 负责最终 individualized integration

最终 LLM 输入：

```text
P_pattern
+
完整当前 respondent observed profile
+
当前 respondent 的 error-analysis descriptors / tags
+
reward-ranked ordinary memories
+
reward-ranked theory-failure memories
```

LLM 必须显式完成 theory-consistent evidence、theory-conflicting evidence、historical exception check、failure-memory apply/reject、individualized integration，并在 `P_pattern` 周围给出 bounded residual adjustment 和 `P_final`。

当前 division of labor 是：

```text
HBM
→ 定义 theory state 与 P_pattern anchor

Offline error analysis
→ 发现 theory compression / mismatch 的 observed signature

Reward/Q
→ 判断哪些历史 case 值得被检索

LLM
→ 综合 ordinary 与 theory-failure evidence，完成个体化判断
```

这里不再存在 learned `P_reward`、ML Failure Router 或 XGBoost residual expert。

### 4.10 主要输出

每个 V4/V5 run 会产生：

```text
theory_pattern_anchor_all_selected.csv
derived_failure_analysis_features.csv
error_analysis_pattern_summary.csv
error_analysis_same_pattern_signatures_detail.csv
error_analysis_stable_signatures.csv
error_analysis_tag_enrichment.csv
error_analysis_retrieval_feature_weights.csv
reward_valued_behavioral_memory.csv
error_analysis_report.md
```

每个模型目录另外保存 calibration/test prediction、retrieved memory ID、LLM exception check、threshold search 和 `summary.json`。

> **Evaluation caution.** 当前 feature family 的设计受到此前 post-hoc error analysis 启发。新脚本虽然把 signature discovery / validation 限制在 memory split 内，但最终 paper-level claim 仍应在冻结代码后使用新的 untouched holdout、另一个 NHIS 年份或独立样本确认。


## 5. Zero-shot / Few-shot baseline

代码：`scripts/60_llm_icl_benchmark_asu.py`

- Zero-shot：不提供示例。
- Random balanced 8-shot：固定 4 个 YES + 4 个 NO。
- Similarity-selected 8-shot：每个目标样本分别检索最相似的 4 个 YES + 4 个 NO。
- Representative 8-shot：每类用 KMeans 找 4 个代表中心，再选择最近的真实样本。
- Random 8-shot + generic CoT：与随机 8-shot 使用相同示例，但要求一般性的正反证据推理，不使用 HBM 理论。

这些 baseline 都不接收 HBM score、pattern、pattern prior、reflective memory 或 FLARE correction rule。

## 6. 其他论文方法的迁移

### 不需要微调大模型

**HBM-CoPB** — 原文：[Chain-of-Planned-Behaviour Workflow Elicits Few-Shot Mobility Generation in LLMs](https://arxiv.org/abs/2402.09836)。原文用 TPB 的 attitude、subjective norms、perceived behavioral control 组织移动意图推理。本项目将其替换成五阶段 HBM 疫苗推理，使用 8 个固定平衡示例，但不提供 FLARE 的确定性分数、pattern prior、retrieval 或 reflection。

**HBM-PB&J** — 原文：[Improving Language Model Personas via Rationalization with Psychological Scaffolds](https://aclanthology.org/2025.findings-emnlp.1187/)。原文先用心理学 scaffold 为既有判断生成 rationale，再用强化 persona 预测新偏好。本项目 V4 使用两次调用：第一次在不知道流感疫苗 label 的情况下构造 HBM health persona；第二次结合 persona 与 8 个 memory 示例进行预测。目前只实现 V4。

**SILIC-inspired** — 原文：[Where You Go is Who You Are](https://arxiv.org/abs/2505.17249)。原文结合 behavioral theory、LLM 引导和 IRL，从移动轨迹反推潜在 reward。本项目的初步 V4 版本把四类非目标疫苗决策视作 contextual binary choices，拟合五维 preventive reward vector，再由 LLM 完成最终预测。由于 NHIS 没有时间顺序状态转移，因此明确称为 inverse contextual choice，而不是 sequential IRL。LLM 不微调，只优化外部 latent parameters。**状态：代码已加入，完整结果待跑。**

### 需要监督微调

**Persona-aware and Explainable Bikeability Assessment** — 原文：[arXiv:2601.03534](https://arxiv.org/abs/2601.03534)。原文使用 cyclist persona conditioning、多粒度 supervised fine-tuning 和 AI data augmentation 完成可解释的 bikeability 评分。**状态：仅完成方法调研，尚未迁移到 FLARE-VAX。**

## 7. 结果

### 7.1 ML baseline

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

### 7.2 Zero-shot / Few-shot

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

上周尚未完成的 Llama 3 70B benchmark 现在已经补齐。可以看到，多组 direct 8-shot 的输出仍然接近“几乎全部预测为接种”，因此 balanced accuracy 约为 0.50；这里把它作为真实 benchmark 结果保留，而不是继续标记为未完成。`*` V5 70B generic-CoT 最终返回 12,851 个 test prediction（test success rate = 0.999922），比配置的 12,852 少 1 个。

### 7.3 CoPB / PB&J 迁移结果

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| V4 | Llama 3 70B | HBM-CoPB | 12853 | 51 | 0.6419 | 0.6464 | 0.6698 | 0.6593 | complete |
| V4 | Llama 4 Scout 17B | HBM-CoPB | 12853 | 51 | 0.6726 | 0.6753 | 0.7131 | 0.6776 | complete |
| V5 | Llama 3 70B | HBM-CoPB | 12852 | 51 | 0.5664 | 0.5634 | 0.5719 | 0.5258 | complete |
| V5 | Llama 4 Scout 17B | HBM-CoPB | 12852 | 61 | 0.6376 | 0.6378 | 0.6652 | 0.6268 | complete |
| V4 | Llama 3 70B | HBM-PB&J | 12853 | 76 | 0.5831 | 0.5935 | 0.6243 | 0.6425 | complete |
| V4 | Llama 4 Scout 17B | HBM-PB&J | 12853 | 61 | 0.7007 | 0.6970 | 0.7610 | 0.6651 | complete |

上周尚未完成的 V4 Llama 4 Scout 17B HBM-PB&J 已补跑完成，ROC-AUC 为 0.7610，balanced accuracy 为 0.6970。V5 HBM-PB&J、SILIC 和需要微调的 persona-aware 方法目前仍没有可汇报的 FLARE-VAX 结果。

### 7.4 FLARE-VAX 主方法与 ablation

| Version | Model | Method | Test N | Threshold | Accuracy | Balanced Acc. | ROC-AUC | F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| V4 | Llama 4 Scout 17B | FLARE-VAX full run | 12853 | 51 | 0.7298 | 0.7312 | 0.7789 | 0.7174 | complete |
| V4 | Llama 3 70B | FLARE-VAX full run | 12853 | 46 | 0.7278 | 0.7285 | 0.7546 | 0.7207 | complete |
| V4 | No LLM | HBM8 pattern-only ablation | 12853 | 50 | 0.7277 | 0.7284 | 0.7693 | 0.7206 | complete |
| V5 | Llama 4 Scout 17B | FLARE-VAX full run | 11057 | 47 | 0.6101 | 0.6179 | 0.6334 | 0.6265 | complete_final_summary_11057_evaluated |
| V5 | Llama 3 70B | FLARE-VAX full run | 12852 | 47 | 0.6255 | 0.6325 | 0.6763 | 0.6597 | complete |
| V5 | No LLM | HBM8 pattern-only ablation | 12852 | 37 | 0.6255 | 0.6325 | 0.6797 | 0.6597 | complete |

### 7.5 TFM-FLARE-VAX 新版结果

当前 repository 已包含完整代码与 notebook，但 **Llama 4 Scout 17B / Llama 4 Maverick 17B 的 full V4/V5 结果尚未写入 README**。运行完成后，`benchmark_results.csv` 会至少包含：

| Version | Model | Method | Accuracy | Balanced Acc. | ROC-AUC | F1 |
|---|---|---|---:|---:|---:|---:|
| V4 | Llama 4 Scout 17B | Pattern anchor | pending | pending | pending | pending |
| V4 | Llama 4 Scout 17B | TFM-FLARE-VAX | pending | pending | pending | pending |
| V5 | Llama 4 Scout 17B | Pattern anchor | pending | pending | pending | pending |
| V5 | Llama 4 Scout 17B | TFM-FLARE-VAX | pending | pending | pending | pending |
| V4 | Llama 4 Maverick 17B | Pattern anchor | pending | pending | pending | pending |
| V4 | Llama 4 Maverick 17B | TFM-FLARE-VAX | pending | pending | pending | pending |
| V5 | Llama 4 Maverick 17B | Pattern anchor | pending | pending | pending | pending |
| V5 | Llama 4 Maverick 17B | TFM-FLARE-VAX | pending | pending | pending | pending |

可选 `--run-ordinary-only-ablation` 会额外得到 `ordinary_memory_only`，用于判断 theory-failure memory 是否比只检索 ordinary historical cases 带来增量价值。

## 8. Curated repository structure / GitHub 中保留的内容

```text
scripts/
  01_ml_baselines.py
  40_flare_vax_v4_with_vaccine_history.py
  50_flare_vax_v5_without_vaccine_history.py
  60_llm_icl_benchmark_asu.py
  70_hbm_copb_pbj_baselines_asu.py
  80_silic_v4_inverse_contextual_reward_asu.py
  81_theory_failure_memory_flare_vax_asu.py
  90_collect_results.py
notebooks/
  81_theory_failure_memory_flare_vax_asu.ipynb
configs/
docs/
data/README.md
results/
  ml/
  icl/
  transfer_baselines/
  flare_vax/
  theory_failure_memory/
```

为了避免把 exploratory model lineage 误当成当前最终方法，公开 package 已移除旧 numerical-reward / theory-residual / ML-router / residual-expert script、notebook 与对应 public result folder。早期探索只在上面的 development-history 小节中保留简短方法学总结。

GitHub 中继续排除原始 NHIS 数据、逐 respondent API JSONL、local absolute path、credentials、checkpoint 和执行状态。

## 9. Reproduction / 复现

```bash
pip install -r requirements.txt
```

不调用 API 的完整 offline dry run：

```bash
python scripts/81_theory_failure_memory_flare_vax_asu.py \
  --input-csv /path/to/adult24.csv \
  --output-dir /path/to/tfm_results \
  --variants v4,v5 \
  --models llama4-scout-17b \
  --dry-run
```

正式运行两个主要模型：

```bash
python scripts/81_theory_failure_memory_flare_vax_asu.py \
  --input-csv /path/to/adult24.csv \
  --output-dir /path/to/tfm_results \
  --variants v4,v5 \
  --models llama4-scout-17b,llama4-maverick-17b \
  --base-url https://openai.rc.asu.edu/v1 \
  --threshold-metric accuracy
```

如需完全复用旧 V4/V5 source-row split：

```bash
--v4-reference-split /path/to/v4_split_assignments.csv \
--v5-reference-split /path/to/v5_split_assignments.csv
```

最重要的额外 ablation：`--run-ordinary-only-ablation`，比较 `Pattern anchor`、`Ordinary memory only` 与 `TFM-FLARE-VAX full contrastive memory`。

LLM runner 默认使用 ASU OpenAI-compatible endpoint，并从 `ASU_OPENAI_API_KEY`、`OPENAI_API_KEY` 或 `--api-key` 读取 credential。
