# NHIS Influenza Vaccination Prediction：Full SFT Baseline 与 Empirical Ceiling 分析

> 当前版本用于阶段性向导师汇报。  
> 本 README 主要总结最初的 **60% train / 40% test full-run**：数据划分、Qwen3.5-9B LoRA SFT 设置、XGBoost 对照、经验性 ceiling 分析，以及当前实验中遇到的工程问题。  
> 后续 1%–5% low-data / compute-matched 实验暂不在本版本展开。

---

## 1. 研究任务

当前任务是使用 **NHIS 2024 Sample Adult** 中的 respondent-level 信息，预测：

> **该 respondent 在过去 12 个月内是否接种过 influenza vaccine。**

二分类目标：

- `vaccinated = 1`：vaccinated
- `vaccinated = 0`：not vaccinated

清洗后的完整数据集：

- **N = 32,070**
- **69 columns**
- `vaccinated = 0`：16,864
- `vaccinated = 1`：15,206
- overall vaccination rate ≈ **47.42%**

因此该数据集不属于高度类别不平衡问题。

---

# 2. 固定的 Train/Test Split

为了让 SFT 与传统 ML 模型能够在完全相同的数据上公平比较，我们首先固定一次 train/test split，之后所有 full-run 模型都使用该划分。

## 2.1 Split 设置

采用：

- **60% train**
- **40% test**
- stratified split by `vaccinated`
- shuffle = True
- random seed = **42**

对应样本量：

```text
Total N = 32,070
Train N = 19,242
Test N  = 12,828
```

具体标签分布：

| Split | N | Not vaccinated | Vaccinated | Vaccination rate |
|---|---:|---:|---:|---:|
| Train | 19,242 | 10,118 | 9,124 | 47.42% |
| Test | 12,828 | 6,746 | 6,082 | 47.41% |
| Total | 32,070 | 16,864 | 15,206 | 47.42% |

可以看到，stratified split 后 train/test 的 outcome distribution 基本完全一致。

---

## 2.2 模型输入变量

清洗数据一共包含 69 个 columns。

其中排除以下 7 个字段：

| Variable | Exclusion reason |
|---|---|
| `vaccinated` | prediction target |
| `SHTFLU12M_A` | 原始 influenza vaccination target，存在直接 label leakage |
| `HHX` | respondent identifier |
| `PPSU` | survey design identifier |
| `PSTRAT` | survey design identifier |
| `WTFA_A` | survey analysis weight |
| `SRVY_YR` | constant survey year |

因此最终使用：

```text
62 model features
```

Qwen SFT 与 XGBoost **严格使用相同的 62 个 feature**。

---

# 3. Qwen3.5-9B SFT Baseline

## 3.1 Base model

使用：

```text
Qwen/Qwen3.5-9B
```

训练采用 parameter-efficient fine-tuning：

```text
LoRA
```

没有进行 full-parameter fine-tuning，也没有 quantization。

---

## 3.2 输入格式

每一个 respondent 被转换成一段结构化文本，例如：

```text
Respondent profile (NHIS variable=value):

RATCAT_A=...
HISPALLP_A=...
AGEP_A=...
...
PHSTAT_A=...

Did this respondent receive an influenza vaccine during the past 12 months?

Return only: vaccinated or not vaccinated.
```

当前版本直接使用：

```text
NHIS variable name = raw survey code/value
```

而不是把每一个 survey code 进一步翻译成自然语言含义。

System instruction 要求模型：

- 只使用给定 respondent information；
- 不补充未提供的信息；
- 最终只返回：
  - `vaccinated`
  - `not vaccinated`

---

# 4. SFT Training Objective

训练并不是让模型重新生成整个 prompt。

我们采用 **response-only SFT**：

- prompt / respondent profile token 不计算 loss；
- 只有 assistant 最后的正确 label token 计算 loss。

例如真实 label 为 vaccinated：

```text
Prompt:
Respondent profile ...
...

Assistant:
vaccinated
```

训练目标只作用于：

```text
vaccinated
```

对应的 assistant response tokens。

这样训练目标更加直接对应当前 binary classification task。

---

# 5. LoRA 设置

LoRA 参数：

| Parameter | Value |
|---|---:|
| LoRA rank `r` | **16** |
| LoRA alpha | **32** |
| LoRA dropout | **0.05** |
| Bias | none |
| Task type | causal LM |

LoRA target modules：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
in_proj_qkv
in_proj_z
in_proj_b
in_proj_a
out_proj
```

训练结束后保存：

```text
final_adapter/
```

因此实际保存的是：

> Qwen3.5-9B base model + trained LoRA adapter

而不是重新保存一整套 9B base weights。

---

# 6. Full SFT Training Hyperparameters

正式 full run 使用全部：

```text
19,242 training respondents
```

主要训练设置：

| Parameter | Value |
|---|---:|
| Base model | Qwen3.5-9B |
| Training rows | **19,242** |
| Test rows | **12,828** |
| Features | **62** |
| Epochs | **2** |
| Learning rate | **1e-4** |
| Micro batch size | **4** |
| Gradient accumulation | **8** |
| Effective batch size | **32** |
| Max sequence length | **1536** |
| Precision | **BF16** |
| Quantization | none |
| Optimizer | fused AdamW |
| LR scheduler | cosine |
| Warmup | **3%** |
| Weight decay | **0.01** |
| Max grad norm | **1.0** |
| Gradient checkpointing | enabled |
| Random seed | **42** |

训练过程中使用 dynamic padding。

本轮 full SFT 没有额外拆 validation set；固定的 60% train 被全部用于最终 full-run SFT benchmark。

正式 full run 总训练时间约：

```text
9,637 seconds ≈ 2.68 hours
```

使用一张 A100 80GB GPU。

---

# 7. SFT Inference / Binary Decision

测试时不让模型自由生成长文本，而是固定比较两个候选答案：

```text
not vaccinated
vaccinated
```

对于每一个候选答案，计算该 answer token 序列的平均 log-likelihood。

可以简单写成：

```text
candidate_score
= candidate answer 中所有 token log-probability 的平均值
```

即对于某个 candidate：

```text
score(candidate)
= mean log P(candidate token | respondent profile + previous candidate tokens)
```

最终 prediction 取两个 candidate 中 score 更高的一个：

```text
prediction
= argmax {
    score("vaccinated"),
    score("not vaccinated")
  }
```

同时，将两个 candidate score 做二项 softmax，构造一个相对的 vaccinated score：

```text
P_pair(vaccinated)
=
exp(score_vaccinated)
/
[
  exp(score_vaccinated)
  +
  exp(score_not_vaccinated)
]
```

需要强调：

> 这里的 `P_pair(vaccinated)` 是两个 candidate likelihood 的相对归一化值，并没有经过专门的 probability calibration。

因此不能直接把它解释成严格校准后的：

```text
真实个体 influenza vaccination probability
```

---

# 8. XGBoost Baseline

为了判断 LLM fine-tuning 是否真正具有 predictive advantage，我们使用完全相同的：

- 19,242 train respondents
- 12,828 test respondents
- 62 features

训练 XGBoost。

XGBoost 设置：

| Parameter | Value |
|---|---:|
| `n_estimators` | 600 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `min_child_weight` | 2 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `reg_alpha` | 0 |
| `reg_lambda` | 1 |
| objective | binary logistic |
| eval metric | log loss |
| tree method | hist |
| seed | 42 |

因此 SFT 与 XGBoost 的 comparison 是严格的：

```text
same train set
same test set
same 62 features
```

---

# 9. Full Test Results

测试集：

```text
N_test = 12,828
```

最终结果：

| Metric | Qwen3.5-9B SFT | XGBoost |
|---|---:|---:|
| Accuracy | **0.76715** | 0.76575 |
| Balanced Accuracy | **0.76849** | 0.76541 |
| Precision | 0.73558 | **0.74996** |
| Recall | **0.79448** | 0.75896 |
| F1 | **0.76389** | 0.75443 |
| ROC-AUC | **0.85002** | 0.84729 |
| Log Loss | 0.59516 | **0.48636** |

换成百分比：

```text
Qwen3.5-9B SFT Accuracy = 76.715%
XGBoost Accuracy         = 76.575%
```

Accuracy difference：

```text
76.715% - 76.575% = +0.140 percentage point
```

因此 full-data SFT 与 XGBoost 的 hard-label predictive performance 基本处于同一水平。

---

# 10. Paired Case-Level Comparison

因为两个模型测试的是完全相同的 12,828 respondents，可以做 paired comparison。

结果：

```text
SFT correct / XGBoost wrong = 482
XGBoost correct / SFT wrong = 464
```

因此：

```text
482 - 464 = 18
```

即 SFT 总共只比 XGBoost 多预测正确 18 个 respondent。

McNemar exact test：

```text
p ≈ 0.580
```

因此当前 full-run 下，没有证据表明 Qwen3.5-9B SFT 在 accuracy 上显著优于 XGBoost。

更合适的结论是：

> **SFT and XGBoost are approximately tied in final classification performance.**

同时：

```text
SFT ROC-AUC = 0.8500
XGB ROC-AUC = 0.8473
```

SFT 略高；但：

```text
SFT Log Loss = 0.5952
XGB Log Loss = 0.4864
```

XGBoost 明显更好。

因此当前结果说明：

> SFT 的 ranking / discrimination ability 已经达到强 tabular ML baseline 的水平，但当前 candidate-likelihood probability 的 calibration quality 仍明显弱于 XGBoost。

---

# 11. 为什么进一步分析 Dataset Ceiling

当 SFT 和 XGBoost 都稳定在约：

```text
76%–77%
```

时，一个关键问题是：

> 模型性能没有继续提高，到底是模型仍然没有学好，还是当前 62 个 observable features 本身只能支持有限程度的个体预测？

因此我们进一步估计：

> **在当前 feature representation 下，这个数据集可以支持的大致 empirical information ceiling。**

这里的 ceiling 不是严格数学意义上的绝对上限，而是一个：

```text
empirical / diagnostic upper-performance reference
```

主要用于判断：

> 当前模型距离数据本身可支持的 predictive information 还有多远。

---

# 12. Empirical Ceiling：1-Nearest-Neighbor / Cover–Hart Diagnostic

## 12.1 基本思想

如果两个 respondent 在当前 62 个 observable features 上非常相似，但 vaccination outcome 却经常不同，那么说明：

> 即使模型继续变复杂，仅根据当前 observable profile，也可能仍然无法完全区分这些个体。

因此我们采用 nearest-neighbor diagnostic。

对于每一个 test respondent：

1. 在 training set 中寻找 feature profile 最相似的 respondent；
2. 使用该 nearest training respondent 的 vaccination label 预测 test respondent；
3. 计算 1-NN error。

距离采用 62-feature profile 的 Hamming-style mismatch：

```text
distance(i, j)
=
两个 respondent 在 62 个 feature 中不相同的 feature 数
/
62
```

例如：

```text
如果有 10 / 62 个 feature 不一致：

distance = 10 / 62 ≈ 0.161
```

missing value 作为一个独立 category 处理。

---

## 12.2 Cover–Hart relation

经典的二分类 1-NN 结果可以写成：

```text
Bayes error
≤
asymptotic 1-NN error
≤
2 × Bayes error × (1 - Bayes error)
```

记：

```text
R*     = Bayes error
R_1NN  = asymptotic 1-NN error
```

则关系为：

```text
R* ≤ R_1NN ≤ 2 × R* × (1 - R*)
```

为了得到一个直观的 diagnostic scale，我们把 observed 1-NN error 近似代入右侧关系：

```text
R_1NN ≈ 2 × R* × (1 - R*)
```

解这个二次方程后：

```text
R*
≈
[1 - sqrt(1 - 2 × R_1NN)] / 2
```

然后将 error 转换为 accuracy：

```text
Diagnostic Bayes accuracy
≈
1 - R*
```

需要强调：

> 这里是为了获得一个 empirical diagnostic scale，并不是声称 observed finite-sample 1-NN error 就严格等于 asymptotic 1-NN error。

---

# 13. Ceiling Diagnostic Result

我们之前的 1-NN / sample-size-growth 分析给出的 practical empirical ceiling 大致位于：

```text
78.3%–79.2%
```

代表性结果中：

```text
1-NN accuracy ≈ 67%
1-NN error    ≈ 33%
```

将这一误差代入上面的 Cover–Hart diagnostic inversion 后，对应的 Bayes-accuracy scale 大约为：

```text
≈ 79%
```

同时，我们还观察了随着 reference-set size 增加，nearest-neighbor diagnostic 的变化。

增长外推后的结果大致稳定在：

```text
≈ 78.3%–78.7%
```

因此当前阶段采用：

```text
Empirical information-ceiling diagnostic range
≈ 78.3%–79.2%
```

作为当前 62-feature representation 的一个 practical reference。

---

# 14. Current Models vs Empirical Ceiling

当前主要结果：

```text
Majority-class baseline ≈ 52.59%

XGBoost Accuracy        = 76.575%
Qwen3.5-9B SFT Accuracy = 76.715%

Empirical ceiling
diagnostic              ≈ 78.3%–79.2%
```

因此 Qwen SFT 距离该 empirical ceiling diagnostic 大约还有：

```text
Lower end:
78.3% - 76.715% = 1.585 pp

Upper end:
79.2% - 76.715% = 2.485 pp
```

即：

```text
Qwen SFT is approximately
1.6–2.5 percentage points
below the empirical ceiling diagnostic.
```

XGBoost 则大约低：

```text
1.7–2.6 percentage points
```

---

## 14.1 相对于 majority baseline 的“已恢复提升”

如果以 majority baseline：

```text
≈ 52.59%
```

作为最简单参考，那么 SFT 已经获得的 improvement 为：

```text
76.715% - 52.588%
= 24.127 pp
```

如果采用 ceiling range 下端 78.3%：

```text
Maximum diagnostic improvement
= 78.3% - 52.588%
= 25.712 pp

Recovered fraction
= 24.127 / 25.712
≈ 93.8%
```

如果采用 ceiling range 上端 79.2%：

```text
Maximum diagnostic improvement
= 79.2% - 52.588%
= 26.612 pp

Recovered fraction
= 24.127 / 26.612
≈ 90.7%
```

因此可以粗略理解为：

> 当前 full-data SFT 已经恢复了相对于 majority baseline、由该 empirical ceiling diagnostic 所暗示的潜在提升中的约 **91%–94%**。

这里同样只是一个 diagnostic interpretation，而不是严格的理论比例。

---

# 15. 一个非常重要的 Ceiling Caveat

不能把：

```text
78.3%–79.2%
```

解释成严格、不可突破的“理论最大 accuracy”。

原因主要有两个。

### 第一，当前 ceiling 是 empirical diagnostic

它依赖于：

- finite-sample nearest neighbor；
- 当前 62-feature representation；
- 当前 Hamming-style distance；
- sample size；
- nearest-neighbor tie；
- representation quality。

因此它更适合被理解为：

```text
information-saturation reference
```

而不是：

```text
absolute mathematical upper bound
```

### 第二，SFT 与 XGBoost 的 error 并不完全重合

当前：

```text
SFT correct / XGB wrong = 482
XGB correct / SFT wrong = 464
```

如果构造一个实际无法直接获得的 oracle：

> 对每一个 test respondent，只要 SFT 或 XGBoost 任意一个预测正确，就选择那个正确预测。

那么该 oracle-union accuracy 大约可以达到：

```text
80.33%
```

已经超过约 79% 的 nearest-neighbor diagnostic。

因此：

> **78.3%–79.2% 应被理解为 empirical ceiling diagnostic，而不是 hard upper bound。**

它的主要作用是说明：

> 当前 76.7% 的模型已经进入接近当前数据可预测信息饱和的区域。

---

# 16. 当前 Full-Run 阶段的核心结论

## Finding 1 — SFT 已达到强 tabular ML baseline 水平

Qwen3.5-9B 经过 LoRA SFT 后：

```text
Accuracy = 76.715%
```

XGBoost：

```text
Accuracy = 76.575%
```

两者基本持平。

因此当前结果不是：

> “LLM 显著优于 XGBoost。”

而是：

> **经过 supervised adaptation 后，Qwen3.5-9B 可以在这一 structured health-behavior prediction task 上达到强 tabular ML baseline 的水平。**

---

## Finding 2 — 两个模型已经接近当前数据的信息饱和区

当前参考：

```text
SFT                  = 76.715%
XGBoost              = 76.575%
Empirical diagnostic = 78.3%–79.2%
```

这说明继续仅仅：

- 增加 LoRA rank；
- 多训练几个 epoch；
- 换更大的 LLM；

可能只能带来有限的 hard-label accuracy improvement。

---

## Finding 3 — Remaining errors 不完全是“模型不够强”

如果当前 62 个 observable features 本身不能充分区分两类 respondent，那么：

> 模型复杂度增加无法自动解决缺失信息问题。

因此后续研究需要进一步区分：

```text
model-limited errors
information-limited errors
highly ambiguous cases
```

不能把所有 prediction error 都视为同一种 failure。

---

## Finding 4 — SFT discrimination 较强，但 probability calibration 仍有明显空间

当前：

```text
SFT AUC = 0.8500
XGB AUC = 0.8473
```

SFT 略高。

但：

```text
SFT Log Loss = 0.5952
XGB Log Loss = 0.4864
```

XGBoost 明显更好。

因此当前结果暗示：

> Qwen SFT 已经学习到较强的 respondent ranking / discrimination signal，但当前 pairwise candidate likelihood 仍不能视作充分校准的 behavioral probability。

---

# 17. 当前阶段的整体 Interpretation

目前 full-data experiment 可以理解为建立了一个：

```text
FULL-INFORMATION / FULL-SUPERVISION REFERENCE
```

即：

```text
62 features
19,242 training respondents
Qwen3.5-9B LoRA SFT

Test Accuracy = 76.715%
```

同时：

```text
XGBoost Test Accuracy = 76.575%
```

而当前 feature representation 的 empirical information-ceiling diagnostic 大约是：

```text
78%–79%
```

因此后续研究的重点不应该只剩下：

> “怎样把 76.7% 再提高 0.3%？”

更重要的问题是：

1. **LLM 需要多少 supervision 才能接近 full-data performance？**
2. **哪些 training samples 对 SFT 真正具有高 marginal value？**
3. **哪些错误属于模型没有利用好已有信息？**
4. **哪些错误来自 respondent information 本身不足？**
5. **增加更 proximal / behavioral information 是否比增加更多 training respondents 更有效？**
6. **在接近 accuracy saturation 后，calibration、uncertainty 与 selective prediction 是否成为更合理的优化目标？**

这些问题构成了后续 low-data scaling、sample-value analysis 和 behavior-aware fine-tuning 的基础。

---

# 18. 当前工程困难：FLA 加速模块未成功启用

本轮实验中一个比较实际的工程困难是 **FLA（Flash Linear Attention）加速路径没有成功、稳定地启用**。

原本希望使用 Qwen3.5 对应的 FLA / fast-kernel implementation，以降低训练与推理时间。

但在 ASU GPU 环境中，FLA 相关依赖与 kernel 安装没有稳定完成，因此最终用于上述正式结果的运行方式是：

```text
Qwen3.5 standard/default Transformers implementation
```

而没有使用 FLA fast-kernel acceleration。

普通 attention backend 使用的是：

```text
SDPA
(Scaled Dot-Product Attention)
```

因此可以把当前运行方式理解为：

```text
Qwen3.5 standard Transformers path
+ no FLA acceleration
```

这不会改变：

- train/test split；
- training labels；
- LoRA 设置；
- SFT objective；
- inference candidate scoring；
- 最终 evaluation protocol。

主要影响的是：

```text
training / inference speed
```

也就是说：

> **当前主要限制是运行速度较慢，而不是因为 FLA 没有启用而改变了实验任务定义或 evaluation setup。**

因此 full SFT 在一张 A100 80GB 上仍需要约：

```text
2.68 hours
```

后续如果能够稳定安装和启用 FLA / optimized kernel path，预计主要收益会体现在工程效率和大规模 repeated experiments 的运行成本上，而不是把它作为当前方法本身的研究贡献。

---

# 19. 一句话阶段总结

> **Using a fixed 60/40 stratified split of 32,070 NHIS respondents, LoRA fine-tuning Qwen3.5-9B on 19,242 training cases achieved 76.72% test accuracy and 0.850 ROC-AUC, essentially matching XGBoost (76.57%, AUC 0.847). A nearest-neighbor/Cover–Hart-based empirical diagnostic places the usable information ceiling of the current 62-feature representation at roughly 78%–79%, suggesting that the full-data SFT model has already entered a performance-saturation regime. The main engineering limitation in the current implementation is that the FLA fast-kernel path was not successfully enabled, so the reported runs used the standard Qwen3.5/Transformers implementation and were correspondingly slower.**

