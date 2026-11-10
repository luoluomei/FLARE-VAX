# NHIS 2024 疫苗接种预测：三个数据约束实验的阶段性研究报告

**更新日期：2026-11-10**  
**研究任务：** 预测过去 12 个月是否接种流感疫苗（`SHTFLU12M_A`）  
**报告性质：** 组会汇报 / 内部研究总结（descriptive results；尚未进行正式显著性检验）

## 0. Executive summary

本项目围绕一个共同问题：**在输入信息或监督标签有限时，LLM 相比传统结构化机器学习模型是否具有优势？** 我们通过三个互补实验逐步检验不同的数据约束。

| 实验 | 控制或改变的维度 | 核心发现 |
|---|---|---|
| **I. Feature Sparsity** | 统一减少所有人可用的特征列（K=62→4）；Top-/Random-/Bottom-K | **特征质量比特征数量更重要。** Top-K 仅保留 8 个特征时，LR 为 76.24%、RF 为 75.95%、SFT 为 75.53%；没有发现持续的 LLM 优势。 |
| **II. Low-label / Few-label SFT** | 完整 62 特征不变；Train20 标签从 6,414 降到 50 | **低标签并没有自然带来 SFT 优势。** 50 labels 时 RF 为 73.29%，Qwen9B 为 67.75%；部分 SFT 预算出现明显非单调波动。 |
| **III. Instance-wise Missingness** | 每个人随机缺少不同特征；比较 Full-trained 与 Mixed-mask-trained | **缺失感知训练对所报告的模型均有效**，但经过同样训练的最强 ML 仍优于 SFT。K=16 时 XGB 从 57.31% 提高到 67.52%，SFT 从 59.79% 提高到 65.07%。 |

**总体结论：** 对现有纯结构化 NHIS 变量而言，预训练 LLM / Label-only SFT 还没有显示出系统性超过强表格 ML 的优势。最稳健的实验证据是：**(1) informative features 的保留非常重要；(2) low-label 的模型训练可能存在稳定性差异；(3) 训练时显式模拟部分可观察输入，能够显著缓解 test-time missingness 导致的性能退化。**

---

## 1. 统一数据、预测任务和评价口径

| 项目 | 定义 |
|---|---|
| 数据 | **NHIS 2024 adult survey**，总样本 **32,070** |
| 预测目标 | `SHTFLU12M_A`：过去 12 个月是否接种流感疫苗，二分类 |
| 原始训练/测试划分 | 固定 **Train20 = 6,414（20%）**；**Test80 = 25,656（80%）** |
| 完整输入 | 从 feature manifest 定义的 **62 个特征**，不包含结果标签或 `_legacy_origin` 元数据 |
| Test80 标签比例 | 接种约 **47.42%**；未接种约 **52.58%** |
| 主要 ML baseline | **XGBoost (XGB)、Random Forest (RF)、Logistic Regression (LR)** |
| 主要 SFT | **Qwen3.5-9B + Label-only LoRA**，使用受访者 profile 输出二分类预测；候选答案 likelihood 评分 |
| 冻结模型（仅实验 I） | **Qwen3.5-27B Direct** 和 **Qwen3.5-27B HBM-guided**（API） |
| 主指标 | **Accuracy (%)** 与 **Balanced Accuracy (%)**；有适当预测分数时补充 AUC、F1、正类预测比例 |

**Accuracy** 是所有受访者预测正确的比例；**Balanced Accuracy** 是正、负类召回率的平均值。后者对于判别模型是否退化为几乎全部预测“未接种”尤其重要：全预测负类在此 Test80 上仍能有约 **52.58% Accuracy**，但 **Balanced Accuracy = 50%**。

三组实验统一沿用既有 Train20/Test80、原始 NHIS 数值变量编码以及未加 survey weights 的测试指标。**指标相同并不意味着三组 SFT 是同一个训练模型**：在实验 I、II、III 的独立模型和训练种子之间，不能把分数差异全部解释为实验变量本身。

---

## 2. Experiment I — Feature Sparsity / Feature Budget Curve

### 2.1 研究问题与实验设置

**Research question:** 当所有受访者的可见特征列同时减少时，ML 和 LLM 谁更能保持预测能力？差异主要取决于可用特征数量，还是这些特征包含的信息？

- **六档预算：** K = **62、48、32、16、8、4**。
- **三种选择方法：**
  - **Top-K：** 基于 Train20 内的 XGBoost permutation importance 排名，保留最重要的 K 个变量。
  - **Random-K：** 随机选择 K 个变量；当前版本每档 K 仅一组随机特征集合，**不是多 seed 随机实验**。
  - **Bottom-K：** 保留排名最低的 K 个变量。
- 共 **3 × 6 = 18 个 feature configurations**。每种设置下所有受访者看到**同样的 K 个特征**；这属于 *dataset-level column selection*，而非个体级随机缺失。
- 对每个 configuration，重新拟合 **3 种 ML**、从原生 9B base 独立训练 **1 个 Label-only LoRA**；另运行 **Frozen Direct 27B、Frozen HBM-guided 27B**。
- 评估：同一固定 Test80，记录 Accuracy、Balanced Accuracy 等指标。筛选后报告 **108 条方法 × 配置结果**，两组 API 各有 1 项只完成 **25,654/25,656**，其余使用完整 Test80。

### 2.2 Top-K：仅保留高预测价值特征

每格为 **Accuracy / Balanced Accuracy（%，保留两位）**。

| K | XGB | RF | LR | Qwen9B SFT | Frozen Direct27B |
| --- | --- | --- | --- | --- | --- |
| 62 | 76.08 / 76.02 | 76.78 / 76.71 | 76.55 / 76.54 | 76.10 / 76.43 | 67.84 / 66.80 |
| 48 | 75.93 / 75.86 | 76.71 / 76.64 | 76.58 / 76.57 | 75.84 / 76.18 | 66.50 / 65.41 | 
| 32 | 75.80 / 75.72 | 76.59 / 76.51 | 76.59 / 76.58 | 75.95 / 76.33 | 66.62 / 65.31 |
| 16 | 75.89 / 75.82 | 76.38 / 76.28 | 76.57 / 76.57 | 75.93 / 76.30 | 65.36 / 63.94 | 
| 8 | 75.45 / 75.40 | 75.95 / 75.86 | 76.24 / 76.35 | 75.53 / 75.95 | 66.22 / 64.75 | 
| 4 | 74.86 / 75.14 | 75.02 / 75.27 | 75.00 / 75.30 | 74.78 / 75.22 | 52.72 / 50.16 |

**主要观察：**

1. **从 62 个特征减至 8 个高重要性特征，性能变化很小。** XGB 由 **76.08%** 变为 **75.45%**（-0.63 pp）；Qwen9B SFT 由 **76.10%** 变为 **75.53%**（-0.57 pp）。
2. **K=4 仍可维持约 75% 的监督预测 Accuracy。** RF 为 **75.02%**，SFT 为 **74.78%**；差距为 **0.24 pp**，但不能仅据此宣称两种方法统计等效或 LLM 有优势。
3. **冻结模型明显落后。** Top-K 62 下 Direct 为 **67.84%**、HBM 为 **63.61%**；Top-K 8 下 HBM 降至 **52.58% Accuracy / 50.00% Balanced Accuracy**，属于接近多数类预测的行为。

### 2.3 Random-K：随机保留特征

每格为 **Accuracy / Balanced Accuracy（%）**。

| K | XGB | RF | LR | Qwen9B SFT | Frozen Direct27B | Frozen HBM27B |
| --- | --- | --- | --- | --- | --- | --- |
| 62 | 76.08 / 76.02 | 76.78 / 76.71 | 76.55 / 76.54 | 75.18 / 75.66 | 67.88 / 66.83 | 63.86 / 62.80 |
| 48 | 75.23 / 75.21 | 75.89 / 75.87 | 75.89 / 75.93 | 75.41 / 75.73 | 61.13 / 59.30 | 55.42 / 53.22 |
| 32 | 72.71 / 72.62 | 73.32 / 73.23 | 73.36 / 73.25 | 72.10 / 72.69 | 58.45 / 57.05 | 62.08 / 60.71 |
| 16 | 73.28 / 73.31 | 74.15 / 74.17 | 74.27 / 74.32 | 72.66 / 73.25 | 51.77 / 49.37 | 52.32 / 49.77 |
| 8 | 71.10 / 70.59 | 71.46 / 70.96 | 71.68 / 71.14 | 71.45 / 70.98 | 55.68 / 53.58 | 52.59 / 50.00 |
| 4 | 55.75 / 54.36 | 55.76 / 54.37 | 55.75 / 54.36 | 53.02 / 55.06 | 52.58 / 50.00 | 52.58 / 50.00 |

**主要观察：**

- XGB 从 K=62 的 **76.08%** 降至 K=8 的 **71.10%**，再到 K=4 的 **55.75%**；SFT 相应为 **75.18%、71.45%、53.02%**。
- **K=16 的 Random-K 结果好于 K=32**（XGB：73.28% vs 72.71%），这是特征集合不同造成的可能现象；由于当前每档 K 只采样一次，不应把它解释为“16 特征比 32 特征天然更优”。
- Random-K 4 下，各 ML 的 Balanced Accuracy 约 **54.36%**，SFT 为 **55.06%**。即使 Accuracy 比多数类基线略高，实际区分能力已较弱。

### 2.4 Bottom-K：仅保留低重要性特征

每格为 **Accuracy / Balanced Accuracy（%）**。

| K | XGB | RF | LR | Qwen9B SFT | Frozen Direct27B | Frozen HBM27B |
| --- | --- | --- | --- | --- | --- | --- |
| 62 | 76.08 / 76.02 | 76.78 / 76.71 | 76.55 / 76.54 | 75.63 / 76.02 | 67.92 / 66.87 | 63.74 / 62.67 |
| 48 | 64.83 / 64.74 | 65.20 / 65.10 | 65.47 / 65.34 | 63.99 / 64.75 | 53.32 / 50.91 | 52.62 / 50.04 |
| 32 | 63.69 / 63.57 | 63.82 / 63.75 | 64.34 / 64.20 | 62.97 / 63.76 | 52.81 / 50.29 | 52.58 / 50.00 |
| 16 | 62.08 / 61.96 | 62.77 / 62.70 | 63.74 / 63.44 | 62.26 / 62.99 | 52.58 / 50.00 | 52.58 / 50.00 |
| 8 | 57.20 / 56.72 | 57.61 / 57.17 | 58.98 / 58.29 | 53.27 / 54.68 | 52.58 / 50.00 | 52.58 / 50.00 |
| 4 | 56.01 / 56.00 | 56.36 / 56.24 | 57.43 / 56.84 | 50.44 / 52.47 | 52.58 / 50.00 | 52.58 / 50.00 |

**主要观察：**

- 从 K=62 减至 K=48，XGB 由 **76.08%** 降至 **64.83%**（-11.25 pp）；相比之下 Top-K 48 的 XGB 仍为 **75.93%**。
- **同样 K=8，Top-K vs Bottom-K 的 XGB 是 75.45% vs 57.20%（差 18.25 pp），SFT 是 75.53% vs 53.27%（差 22.26 pp）**。
- Frozen Direct / HBM 在多个 Bottom-K 设置下接近 **52.58% Accuracy / 50.00% Balanced Accuracy**，并伴随几乎不预测正类的问题。

### 2.5 实验 I 的结论与限制

> **Preserving predictive information matters substantially more than merely preserving the number of features.** Both tabular ML and fine-tuned LLMs can retain much of their accuracy with a small set of highly informative features, but the LLM does not consistently outperform strong ML baselines.

**限制：** Top-/Bottom-K 特征重要性来自 XGB，具有一定 ML 方法依赖性；Random-K 缺少多次随机重复；不同配置分别训练 SFT，因此结果同时包含特征变化与训练随机性；K=62 在三类选择中是相同特征集合，但三个 SFT adapter 不是同一个随机运行。Frozen API 使用原始数值编码和生成式预测，HBM 的部分退化不等于理论引导本身被普遍证伪。

![Study I: Feature budget curves](figures/01_feature_budget.png)

---

## 3. Experiment II — Low-label Supervised Learning / Few-label SFT

### 3.1 研究问题与实验设置

**Research question:** 在样本特征完整、但可获得真实标签极少时，预训练 LLM 是否比传统 ML 更具 label efficiency？

本实验保持全部 **62 features** 与完整 **Test80=25,656** 不变，只改变用于训练的 Train20 有标签样本数。它是 **few-label supervised fine-tuning**，**不是**在冻结模型 prompt 中提供 2/4/8 个 examples 的 in-context few-shot。

- **10 档训练预算：** 50、64（1%）、100、128（2%）、192（3%）、257（4%）、321（5%）、641（10%）、1,283（20%）、6,414（100%）。百分比均相对于原始 Train20。
- **3 个训练/抽样 seeds：** 42、43、44；每个 seed 在 Train20 内分层、嵌套抽样，保证同一 seed 的较小子集包含于较大子集。四种保留方法使用同一批被选中的标签。
- **4 个对照模型：** XGB、RF、LR、Qwen3.5-9B Label-only LoRA；每次 SFT 从 base 初始化新 adapter。
- **共 10 × 3 × 4 = 120 组保留方法的模型评估；全部完成。** 每组使用相同完整 Test80。
- LoRA 采用小标签预算下的**自适应训练 epoch 计划**：50/64 用 15 epochs，100/128 用 8，192 用 5，257 用 4，321 用 3，641/1,283/6,414 用 2；这一策略缓解优化步数过少，但也意味着不同预算的训练日程并不完全一致。

### 3.2 Accuracy 与 Balanced Accuracy 结果

每格为 **Accuracy / Balanced Accuracy（%，保留两位）**，均由上传的 `all_metrics.csv` 对三个训练/抽样 seeds（42、43、44）取算术平均得到；所有模型使用相同的完整 Test80（25,656 人）。

| 预算 | Train20 标签数 | XGB | RF | LR | Qwen9B SFT |
| --- | ---: | --- | --- | --- | --- |
| 50 labels | 50 | 71.28 / 71.16 | 73.29 / 73.10 | 71.51 / 71.37 | 67.75 / 68.12 |
| 1% | 64 | 71.70 / 71.50 | 73.88 / 73.66 | 71.37 / 71.17 | 62.04 / 63.03 |
| 100 labels | 100 | 72.42 / 72.23 | 74.40 / 74.20 | 71.51 / 71.36 | 70.55 / 70.50 |
| 2% | 128 | 72.43 / 72.32 | 74.80 / 74.76 | 71.38 / 71.34 | 71.79 / 71.87 |
| 3% | 192 | 72.16 / 72.18 | 74.69 / 74.73 | 71.59 / 71.55 | 70.66 / 70.97 |
| 4% | 257 | 72.71 / 72.72 | 74.72 / 74.79 | 71.89 / 71.93 | 62.56 / 63.64 |
| 5% | 321 | 73.03 / 73.01 | 75.25 / 75.27 | 72.04 / 72.01 | 61.32 / 62.59 |
| 10% | 641 | 73.22 / 73.19 | 75.77 / 75.72 | 73.30 / 73.30 | 71.17 / 71.33 |
| 20% | 1,283 | 74.05 / 74.03 | 76.17 / 76.14 | 74.88 / 74.88 | 74.32 / 74.71 |
| 100% | 6,414 | 76.08 / 76.02 | 76.78 / 76.71 | 76.55 / 76.54 | 75.59 / 76.03 |

**Balanced Accuracy 的补充观察：** 50 labels 时，RF 为 **73.10%**，Qwen9B SFT 为 **68.12%**（差 **4.98 pp**）；100 labels 时分别为 **74.20%** 与 **70.50%**。在 SFT 波动较明显的 4%（257 labels）和 5%（321 labels）预算，Balanced Accuracy 仍分别只有 **63.64%** 和 **62.59%**，说明异常低的 Accuracy 并非仅由测试集类别比例导致。后续仍需查看各 seed 的预测正类比例及分数分布，以解释具体退化机制。

### 3.3 主要结果

1. **极少标签下 Random Forest 最强。** 50 个标签时，RF **73.29%**、XGB **71.28%**、LR **71.51%**、SFT **67.75%**。RF 领先 SFT **5.54 pp**。100 labels 时 RF **74.40%**，SFT **70.55%**。
2. **SFT 没有形成单调增长的学习曲线。** 从 128 labels 的 **71.79%** 降到 257 的 **62.56%** 和 321 的 **61.32%**，随后在 1,283 labels 回升至 **74.32%**。这提示优化稳定性、类别偏置或采样差异值得排查，但仅有三 seed 均值，不能确定是哪一个原因。
3. **更多标签下传统 ML 仍领先。** 1,283 labels 时 RF **76.17%**、SFT **74.32%**；6,414 labels 时 RF **76.78%**、SFT **75.59%**。
4. **ML 的标签效率本身很强。** 在这套 NHIS 结构化变量上，RF 仅用 321 labels 已达到 **75.25% Accuracy**，但这种结果不能外推到全部其他医疗/行为数据集。

### 3.4 实验 II 的结论与限制

> **Pretrained LLM fine-tuning does not automatically yield superior label efficiency on the current structured NHIS prediction task.** Random Forest is remarkably competitive under very small labeled training sets, while the SFT learning curve shows sensitivity to label budget and training conditions.

**限制：** 当前只有 3 个抽样/训练 seeds；不同预算采用不同 epochs，而非严格相同训练计算预算；虽然已补齐 Balanced Accuracy 均值，仍需检查各 seed 的 SFT 正类预测率、Balanced Accuracy 波动、AUC 和训练 loss，尤其是 64、257、321 labels；表中没有误差线或显著性结论。数据始终有完整 62 个特征，不能把本实验与 feature missingness 的因果效应混为一谈。

![Study II: Low-label learning curves](figures/02_low_label.png)

---

## 4. Experiment III — Instance-wise Missingness：A/B 随机个体缺失与缺失感知训练

### 4.1 研究问题

与实验 I 不同，这一次**每个 respondent 有不同的观察特征集合**；即使两个人都只保留 16 个特征，他们实际可见的变量也不一定相同。

研究同时检验两件事：

- **Experiment A / Test-time robustness：** 用完整 62 特征训练的模型，遇到个体级随机缺失输入时如何退化？
- **Experiment B / Missingness-aware adaptation：** 如果训练时就用随机遮蔽后的 profile 学习，ML 与 SFT 各能恢复多少预测能力？

### 4.2 实际运行的 Compact 测试设置

| 因素 | 设置 |
|---|---|
| 数据划分 | Train20=6,414；Test80=25,656，保持不变 |
| 完整特征 | 62 个 |
| 实际测试预算 | **K=62、32、16、8**（相应人工遮蔽比例约 0%、48.4%、74.2%、87.1%） |
| Test mask seeds | **K=62：1 个 seed；K=32/16/8：各 3 个 seeds（42/43/44）**，合计 **10 个 mask 条件** |
| Mask 机制 | 对每位受访者分别随机选择可见特征；同一个 seed 中 K 小的可见集合嵌套在 K 大的集合内 |
| 公平对照 | 同一受访者、同一 Test80 mask 同时用于所有方法；人工缺失与 NHIS 原始编码区分 |
| Full-trained ML | 3 个在完整 Train20 上训练的模型 |
| Full-trained SFT | 复用 Low-label 的 3 个 100%-label Qwen9B adapters |
| Mixed-mask-trained ML | 三种 ML × 3 个训练 seeds，共 9 个新模型；每个 respondent 使用两个 masked training views |
| Mixed-mask-trained SFT | 3 个从 base 新训练的 LoRA；与相应 ML 使用相同训练 masks，6,414 个真实标签不变 |
| ML Duplicate Control | 3 个模型只重复完整样本、不删除特征，用来检验重复数据的影响 |
| 规模与完成情况 | **210 条保留方法的评估记录，均为 COMPLETE**；相应 SFT 实际为 6 个 adapters × 10 条件 = 60 次完整 Test80 evaluation |

Mixed-mask training 随机采用 K∈{62,48,32,16,8,4}，即使第一轮评估只选 K=62/32/16/8，也没有改变训练中的覆盖预算。重复 masked views **不新增独立受访者或新的真实标签**。

### 4.3 主要结果：Full-trained vs Mixed-mask-trained

每格为 **Accuracy / Balanced Accuracy（%）**。对于 K=32/16/8，数值为已有 training seeds 与 test mask seeds 的均值；K=62 只有一个 test mask。不同模型的训练 seed 结构并不完全相同，不能把每条记录都视为独立训练重复。

| K | 训练条件 | XGB | RF | LR | Qwen9B SFT |
| --- | --- | --- | --- | --- | --- |
| 62 | Full | 76.08 / 76.02 | 76.78 / 76.71 | 76.55 / 76.54 | 75.59 / 76.03 |
| 62 | Mixed-mask | 75.88 / 75.86 | 76.62 / 76.62 | 76.37 / 76.37 | 75.35 / 75.72 |
| 32 | Full | 65.33 / 64.06 | 62.37 / 60.62 | 67.46 / 67.14 | 66.16 / 65.50 |
| 32 | Mixed-mask | 72.10 / 72.00 | 72.38 / 72.11 | 71.94 / 71.81 | 70.40 / 70.64 |
| 16 | Full | 57.31 / 55.19 | 54.04 / 51.57 | 61.81 / 61.25 | 59.79 / 58.65 |
| 16 | Mixed-mask | 67.52 / 67.29 | 67.37 / 66.93 | 66.72 / 66.40 | 65.07 / 65.40 |
| 8 | Full | 54.01 / 51.56 | 52.72 / 50.14 | 58.30 / 57.47 | 56.56 / 55.24 |
| 8 | Mixed-mask | 63.08 / 62.57 | 62.12 / 61.55 | 62.18 / 61.57 | 60.07 / 60.58 |

### 4.4 缺失感知训练带来的提升

每格为 **ΔAccuracy / ΔBalanced Accuracy（百分点，Mixed − Full）**。

| 模型 | K=32 | K=16 | K=8 |
| --- | --- | --- | --- |
| XGB | +6.77 / +7.93 | +10.21 / +12.10 | +9.07 / +11.01 |
| RF | +10.01 / +11.49 | +13.33 / +15.36 | +9.41 / +11.41 |
| LR | +4.48 / +4.68 | +4.91 / +5.14 | +3.88 / +4.10 |
| Qwen9B SFT | +4.24 / +5.14 | +5.28 / +6.75 | +3.52 / +5.35 |

**关键结果：**

1. **所有四类保留模型在 K=32、16、8 时都因 Mixed-mask Training 获益。** 这同时体现在 Accuracy 和 Balanced Accuracy 上。
2. **ML 经过适应训练后仍保持领先。** K=32，保留模型中最佳 RF Mixed 为 **72.38%**，Qwen Mixed 为 **70.40%**（差 1.98 pp）；K=16，最佳 XGB Mixed 为 **67.52%**，Qwen Mixed 为 **65.07%**（差 2.45 pp）；K=8，最佳 XGB Mixed 为 **63.08%**，Qwen Mixed 为 **60.07%**（差 3.01 pp）。
3. **相对改进并非 LLM 独有。** K=16 的 Accuracy 增益：RF **+13.33 pp**、XGB **+10.21 pp**、Qwen9B **+5.28 pp**、LR **+4.91 pp**。因此应表述为“多种 ML 的改进更大”，而非所有 ML 在每个预算下都超过 SFT 的增益。
4. **完整特征下的代价较小。** K=62 时 XGB 从 **76.08%** 略降至 **75.88%**（-0.20 pp），Qwen9B 从 **75.59%** 略降至 **75.35%**（-0.24 pp）。这是面向完整与部分可观察数据的一种性能权衡。

### 4.5 为什么 Balanced Accuracy 很重要：K=8 的类别偏置

| 模型 | Full-trained K=8 正类预测率 | Mixed-trained K=8 正类预测率 |
| --- | --- | --- |
| XGB | 2.59% | 39.60% |
| RF | 0.18% | 38.38% |
| LR | 33.42% | 37.63% |
| Qwen9B SFT | 24.23% | 59.32% |

- Full-trained RF 在 K=8 时仅预测 **0.18%** 为接种，其 **Accuracy = 52.72%**、**Balanced Accuracy = 50.14%**，接近完全预测多数类。
- Full-trained XGB 的正类预测率仅 **2.59%**，**Accuracy = 54.01%**、**Balanced Accuracy = 51.56%**。
- Mixed-mask Training 后，RF 和 XGB 的正类预测率分别上升至 **38.38%**、**39.60%**，Balanced Accuracy 也分别提高至 **61.55%**、**62.57%**。
- Qwen9B Mixed 在 K=8 时预测接种比例达到 **59.32%**（高于真实约 47.42%），其 **Accuracy = 60.07%**、**Balanced Accuracy = 60.58%**；改善明显，但预测偏置方向与部分树模型不同。

这提示：随机遮蔽造成的性能下降，一部分可能是**未见过的 missing-value 输入表示使分类阈值和预测分布发生变化**，不应全部归因于信息不可恢复。未来可额外报告 AUC / score calibration，并在独立 validation data 上调整阈值（不在 Test80 上调）。


---
