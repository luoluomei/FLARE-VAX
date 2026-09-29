# 本周进展 — Label Scaling、Tabular Foundation Models 与 Reasoning SFT

## 1. Label-only SFT 与 XGBoost：不同标注比例下的 Scaling 实验

在进入 teacher-reasoning SFT 之前，我们已经完成过一组 **label-only supervised fine-tuning scaling experiments**。

这一组实验不使用 teacher，也不生成 reasoning。Qwen3.5-9B 直接根据 respondent profile 学习最终 vaccination label，并与使用相同训练子集的 XGBoost 进行比较。

整体流程为：

```text
Labeled Training Subset
        ↓
 ┌───────────────┐
 │               │
Qwen3.5-9B      XGBoost
Label-only SFT   ML baseline
 │               │
 └───────┬───────┘
         ↓
   Fixed Test Set
```

实验逐步增加 labeled data 的比例，用于观察 Qwen3.5-9B 和传统 ML 随监督数据量增加时的变化趋势。

### 1.1 不同 Labeled Fraction 下的结果

| Labeled fraction | Qwen3.5-9B SFT | XGBoost |
|---|---:|---:|
| 1% | 0.643436 | **0.722482** |
| 5% | **0.758185** | 0.743842 |
| 10% | **0.758809** | 0.751871 |
| 20% | 0.756704 | **0.758731** |
| 30% | **0.762473** | 0.761927 |
| 40% | **0.766292** | 0.762551 |
| 50% | **0.766215** | 0.765279 |
| 60% | **0.767150** | 0.765747 |

### 1.2 初步观察

这组结果呈现出比较明显的 learning-curve pattern。

- 在 **1% labeled data** 下，Qwen3.5-9B 的表现明显不足，Accuracy 仅为 **0.6434**，低于 XGBoost 的 **0.7225**。
- 当 labeled data 增加到 **5%** 时，Qwen3.5-9B 的 Accuracy 快速提升到 **0.7582**，已经超过同样数据量下的 XGBoost（0.7438）。
- 从 **5% 到 20%**，Qwen3.5-9B 已经基本进入 0.756–0.759 的区间，继续增加训练数据带来的提升比较有限。
- 从 **30% 到 60%**，两类方法进一步趋于稳定，最终都接近 **76%–77% Accuracy**。

Qwen3.5-9B 的主要提升发生在：

```text
1% → 5%
0.6434 → 0.7582
```

而在更大的 labeled fraction 下，性能逐渐接近 saturation。

这一结果也为当前 reasoning SFT 提供了一个重要 baseline：后续需要判断 teacher-generated reasoning 是否能够在 **不增加更多 ground-truth labels** 的情况下，进一步提升 student model。

---

# 2. Tabular Foundation Model 实验

本周进一步测试了两个近期的 **Tabular Foundation Model**，用于 NHIS 2024 流感疫苗接种预测。

与 LLM 方法不同，这两个模型直接使用结构化 tabular features，不需要把 respondent profile 转换成自然语言。

当前实验使用：

- **5% training set**
- **15% validation set**
- **80% test set**
- Test N = **25,656**
- Input = raw numeric tabular features
- Outcome = 过去 12 个月是否接种 influenza vaccine

需要注意：这一组使用的是 **5/15/80 split**，而上面的 label-scaling experiment 使用的是另一套固定测试划分。因此两组结果可以用于观察整体性能水平和趋势，但不能视为完全相同 test respondents 上的严格 paired comparison。

## 2.1 TabPFN-3.5

**模型：** TabPFN-3.5  
**参考文献：** Jäger et al. (2026), *TabPFN-3.5: Technical Report*, arXiv:2609.17895.

TabPFN 是一种针对表格数据预训练的 foundation model。其主要思路是通过大量 synthetic tabular tasks 进行预训练，使模型能够在新的 tabular prediction task 上利用 labeled examples 进行 in-context prediction。

本实验测试：

- **Pretrained ICL**
- **Fine-tuned**

两种设置均使用相同的 raw numeric input features。

## 2.2 TabICLv2

**模型：** TabICLv2  
**参考文献：** Qu et al. (2026), *TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model*, arXiv:2602.11139.

TabICLv2 是 *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*（ICML 2025）的进一步扩展。

其核心同样是通过 synthetic tabular tasks 进行预训练，再在新的 tabular dataset 上通过 in-context learning 进行预测。

## 2.3 Test80 结果

### Fixed Threshold = 0.50

| Model | Accuracy | Balanced Accuracy | F1 | ROC-AUC | Log Loss |
|---|---:|---:|---:|---:|---:|
| **TabPFN-3.5 — Pretrained ICL** | **0.7671** | **0.7659** | **0.7513** | **0.8464** | 0.4853 |
| TabPFN-3.5 — Fine-tuned | 0.7665 | 0.7654 | 0.7512 | 0.8464 | **0.4852** |
| TabICLv2 — Pretrained ICL | 0.7647 | 0.7630 | 0.7464 | 0.8450 | 0.4884 |

在固定 threshold = 0.50 时，三个模型表现非常接近。

当前表现最好的是：

```text
TabPFN-3.5 — Pretrained ICL
Accuracy = 0.7671
Balanced Accuracy = 0.7659
ROC-AUC = 0.8464
```

TabPFN-3.5 在当前数据集上进一步 fine-tuning 后并没有带来明显提升。

### Validation-selected Threshold

| Model | Selected Threshold | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **TabPFN-3.5 — Pretrained ICL** | 0.465 | **0.7682** | **0.7685** | 0.7603 | **0.8464** |
| TabPFN-3.5 — Fine-tuned | 0.485 | 0.7676 | 0.7672 | 0.7561 | 0.8464 |
| TabICLv2 — Pretrained ICL | 0.445 | 0.7665 | 0.7672 | **0.7603** | 0.8450 |

使用 Validation15 选择 threshold 后，结果有小幅提升。

当前最高 Accuracy 为：

```text
TabPFN-3.5 — Pretrained ICL
Threshold = 0.465
Accuracy = 0.7682
Balanced Accuracy = 0.7685
ROC-AUC = 0.8464
```

## 2.4 与此前 Label-only SFT / XGBoost 的整体对照

为了更直观地观察不同方法目前达到的性能范围，可以把已有结果放在一起看：

| Method / Setting | Labeled Data | Accuracy |
|---|---:|---:|
| Qwen3.5-9B Label-only SFT | 1% | 0.6434 |
| XGBoost | 1% | 0.7225 |
| **Qwen3.5-9B Label-only SFT** | **5%** | **0.7582** |
| XGBoost | 5% | 0.7438 |
| Qwen3.5-9B Label-only SFT | 10% | 0.7588 |
| XGBoost | 10% | 0.7519 |
| Qwen3.5-9B Label-only SFT | 20% | 0.7567 |
| XGBoost | 20% | 0.7587 |
| Qwen3.5-9B Label-only SFT | 30% | 0.7625 |
| XGBoost | 30% | 0.7619 |
| Qwen3.5-9B Label-only SFT | 40% | 0.7663 |
| XGBoost | 40% | 0.7626 |
| Qwen3.5-9B Label-only SFT | 50% | 0.7662 |
| XGBoost | 50% | 0.7653 |
| Qwen3.5-9B Label-only SFT | 60% | 0.7672 |
| XGBoost | 60% | 0.7657 |
| TabICLv2 — Pretrained ICL | 5% train + 15% validation | 0.7665 |
| TabPFN-3.5 — Fine-tuned | 5% train + 15% validation | 0.7676 |
| **TabPFN-3.5 — Pretrained ICL** | **5% train + 15% validation** | **0.7682** |

需要强调的是：

> 上表中的 label-scaling experiment 与 Tabular Foundation Model experiment 使用的 test split 不完全相同，因此这里主要用于展示不同方法达到的整体性能水平，而不是严格的同样本 paired comparison。

不过，从整体趋势可以看到，目前几种较强的方法最终都逐渐集中在：

```text
Accuracy ≈ 0.76–0.77
ROC-AUC ≈ 0.845–0.85
```

这一范围。

这进一步提示当前数据集可能已经存在比较明显的 empirical performance ceiling。

---

# 3. Teacher-Reasoning SFT

本周另一个主要方向是测试：

> 相比只使用最终 vaccination label 进行 SFT，是否可以通过 teacher-generated reasoning 为较小的 student model 提供额外 supervision。

## 3.1 整体流程

当前 pipeline 为：

```text
NHIS Respondent Profile
        ↓
Qwen3.8-27B Teacher
        ↓
Teacher-generated Reasoning
+ Ground-truth Label
        ↓
Qwen3.5-9B Student
        ↓
LoRA Supervised Fine-Tuning
        ↓
Vaccinated / Not Vaccinated
```

当前 reasoning SFT 使用：

- **Train20：6,414 respondents**
- **Test80：25,656 respondents**
- **62 input features**

Teacher 只对 Train20 生成 reasoning annotation。

Student 随后学习：

```text
Profile
+ Teacher Reasoning
+ Ground-truth Label
```

最终仍然在独立 Test80 上预测 vaccination outcome。

## 3.2 Teacher Model

**Qwen3.8-27B**

Teacher 仅用于产生 reasoning annotation。

目前测试两种 reasoning generation mode。

### No Thinking

```text
enable_thinking = False
max_new_tokens = 384
```

### Thinking-Low

```text
enable_thinking = True
reasoning_effort = low
max_new_tokens = 640
```

因此，每种 reasoning strategy 都会分别得到：

```text
No Thinking
vs.
Thinking-Low
```

两个版本。

## 3.3 Student Model

**Qwen3.5-9B**

Student 使用简单 LoRA SFT。

主要训练参数：

| Parameter | Value |
|---|---:|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 1e-4 |
| Epochs | 2 |
| Micro batch size | 2 |
| Gradient accumulation | 16 |
| Maximum sequence length | 2048 |
| Seed | 42 |

不同 reasoning strategy 之间保持相同 student model 和 SFT settings。

## 3.4 Reasoning Strategy 1 — General Reasoning

第一种方法为相对自由的 **General Reasoning**。

Teacher 根据 respondent profile 分析：

- Supporting evidence
- Opposing evidence
- Conflicting signals
- Overall integration

不额外加入特定 behavioral theory。

整体形式：

```text
Respondent Profile
        ↓
General Reasoning
        ↓
Ground-truth Label
        ↓
Qwen3.5-9B SFT
```

分别生成：

```text
General Reasoning
├── No Thinking
└── Thinking-Low
```

这一组主要用于判断：

> 一般性的 teacher reasoning 是否比单纯 label supervision 提供更多有效训练信号。

## 3.5 Reasoning Strategy 2 — HBM-Light

第二种方法加入轻量级 **Health Belief Model (HBM)** 指导。

Teacher 在 reasoning 中可以考虑：

- perceived threat
- perceived benefits
- perceived barriers
- self-efficacy
- cues to action

但只有在 respondent profile 中存在相应 observable evidence 时才使用对应 HBM mechanism。

特别限制：

> 不允许仅根据 demographic variables 强行推断 respondent 未被直接观测的心理状态。

因此 HBM 在这里主要作为：

```text
reasoning scaffold
```

而不是：

```text
deterministic prediction rule
```

整体形式：

```text
Respondent Profile
        ↓
HBM-aware Reasoning
        ↓
Ground-truth Label
        ↓
Qwen3.5-9B SFT
```

同样包含：

```text
HBM-Light
├── No Thinking
└── Thinking-Low
```

## 3.6 当前实验矩阵

| Reasoning Strategy | Teacher Mode | Student |
|---|---|---|
| General Reasoning | No Thinking | Qwen3.5-9B |
| General Reasoning | Thinking-Low | Qwen3.5-9B |
| HBM-Light | No Thinking | Qwen3.5-9B |
| HBM-Light | Thinking-Low | Qwen3.5-9B |

后续主要比较：

1. **Label-only SFT vs. Reasoning SFT**
2. **General Reasoning vs. HBM-Light**
3. **No Thinking vs. Thinking-Low**
4. Reasoning supervision 是否能够进一步提升 Train20 条件下的 student performance
5. 是否能够接近或超过目前已经达到约 0.76–0.77 Accuracy 的 strong tabular baselines

## 3.7 当前状态

Teacher reasoning generation 与后续 student SFT pipeline 目前仍在运行。

因此，本周暂时不报告 reasoning SFT 的最终 performance。

后续将统一比较：

```text
Label-only SFT
        ↓
General Reasoning SFT
        ↓
HBM-Light Reasoning SFT
```

重点关注：

> 在不增加 ground-truth labeled data 的情况下，teacher-generated reasoning 是否能够为 Qwen3.5-9B 提供额外有效监督，并突破当前 label-only SFT 与 strong tabular models 已经接近的 performance range。
