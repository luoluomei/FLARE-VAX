# 本周进展 — Tabular Foundation Models 与 Reasoning SFT

## 1. Tabular Foundation Model 实验

本周首先测试了两个较新的 **Tabular Foundation Model**，用于 NHIS 2024 流感疫苗接种预测任务。与前面的 LLM 方法不同，这两个模型直接使用结构化 tabular features，不需要先把样本转换成自然语言描述。

### 1.1 TabPFN-3.5

**模型：** TabPFN-3.5  
**参考文献：** Jäger et al. (2026), *TabPFN-3.5: Technical Report*, arXiv:2609.17895.

TabPFN 是一种面向表格数据的 pretrained foundation model。其核心思路是通过大量 synthetic tabular tasks 进行预训练，使模型在面对新的 tabular prediction task 时，可以直接利用给定的 labeled examples 进行 in-context prediction。

本实验中测试了两种设置：

- **Pretrained ICL**：直接使用预训练模型进行 in-context learning；
- **Fine-tuned**：在当前 vaccination prediction task 上进一步 fine-tune。

两种设置都使用相同的 raw numeric features。

---

### 1.2 TabICLv2

**模型：** TabICLv2  
**参考文献：** Qu et al. (2026), *TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model*, arXiv:2602.11139.  
该模型是在 *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*（ICML 2025）基础上的进一步扩展。

TabICLv2 同样属于 tabular in-context learning 方法。它通过 synthetic-data pretraining 学习不同 tabular tasks 的结构，并通过更高效的模型设计提高在较大 tabular dataset 上的可扩展性。

在当前实验中，TabICLv2 直接读取 tabular features，并使用 training examples 作为 context 进行预测，而不需要传统的 task-specific classifier training。

---

### 1.3 实验设置

当前比较使用：

- **Test set：25,656 respondents**
- **Input：raw numeric tabular features**
- **Outcome：过去 12 个月是否接种 influenza vaccine**
- 使用两种 decision rule：
  - 固定 probability threshold = **0.50**
  - 使用 **15% validation set** 选择 threshold

---

### 1.4 主要结果

#### 固定 Threshold = 0.50

| Model | Accuracy | Balanced Accuracy | F1 | ROC-AUC | Log Loss |
|---|---:|---:|---:|---:|---:|
| **TabPFN-3.5 — Pretrained ICL** | **0.7671** | **0.7659** | **0.7513** | **0.8464** | 0.4853 |
| TabPFN-3.5 — Fine-tuned | 0.7665 | 0.7654 | 0.7512 | 0.8464 | **0.4852** |
| TabICLv2 — Pretrained ICL | 0.7647 | 0.7630 | 0.7464 | 0.8450 | 0.4884 |

在固定 0.50 threshold 下，三个模型的表现非常接近。

其中表现最好的为 **TabPFN-3.5 Pretrained ICL**：

- Accuracy = **0.7671**
- Balanced Accuracy = **0.7659**
- ROC-AUC = **0.8464**

一个比较明显的现象是，TabPFN-3.5 在当前任务上进行进一步 fine-tuning 后，并没有获得额外提升，整体结果与 pretrained ICL 基本一致。

---

#### 使用 Validation Set 选择 Threshold

| Model | Selected Threshold | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **TabPFN-3.5 — Pretrained ICL** | 0.465 | **0.7682** | **0.7685** | 0.7603 | **0.8464** |
| TabPFN-3.5 — Fine-tuned | 0.485 | 0.7676 | 0.7672 | 0.7561 | 0.8464 |
| TabICLv2 — Pretrained ICL | 0.445 | 0.7665 | 0.7672 | **0.7603** | 0.8450 |

使用 validation set 选择 threshold 后，模型结果有小幅提升。

当前最好的结果仍然来自 **TabPFN-3.5 Pretrained ICL**：

- **Accuracy = 0.7682**
- **Balanced Accuracy = 0.7685**
- **ROC-AUC = 0.8464**

因此，目前这两个 Tabular Foundation Model 可以作为额外的 strong tabular baseline。

从结果来看，TabPFN-3.5 在当前数据集上不需要额外 fine-tuning 就已经可以达到非常接近最优的表现。

---

# 2. Teacher-Reasoning SFT

本周另一个主要方向是测试：

> 相比只使用最终 vaccination label 进行 SFT，是否可以通过 teacher-generated reasoning 为较小的 student model 提供额外 supervision。

---

## 2.1 整体流程

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

当前数据划分为：

- **Train20：6,414 respondents**
- **Test80：25,656 respondents**
- **Input：62 features**

Teacher 只对 Train20 中的样本生成 reasoning annotation。

之后，Student 使用这些 teacher-generated rationales 和 ground-truth labels 进行 SFT，最后在完全独立的 Test80 上进行评估。

---

## 2.2 Teacher 与 Student

### Teacher Model

**Qwen3.8-27B**

Teacher 只负责生成 reasoning annotation，不直接参与最终 test prediction。

为了比较 teacher 自身 thinking mode 是否会影响 reasoning quality，目前分别生成两种版本。

#### No Thinking

```text
enable_thinking = False
max_new_tokens = 384
```

即不使用模型内部显式 thinking mode，直接生成 reasoning。

#### Thinking-Low

```text
enable_thinking = True
reasoning_effort = low
max_new_tokens = 640
```

允许 teacher 使用较低强度的 thinking，再生成最终 reasoning。

因此，每一种 reasoning strategy 都会分别产生：

```text
No Thinking
vs.
Thinking-Low
```

两套 teacher annotation。

---

### Student Model

**Qwen3.5-9B**

Student 使用与之前实验一致的简单 LoRA fine-tuning。

主要参数如下：

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

除了 teacher reasoning 内容不同之外，其余 student training settings 保持一致，从而尽可能保证不同 reasoning strategy 之间的公平比较。

---

## 2.3 Reasoning Strategy 1 — General Reasoning

第一种方法使用相对自由的 **General Reasoning**。

Teacher 不使用特定行为理论，而是根据 respondent profile 自由分析与最终 vaccination decision 有关的信息。

主要要求 teacher 考虑：

- Supporting evidence；
- Opposing evidence；
- Conflicting signals；
- 最后综合不同 evidence 得出整体判断。

整体形式为：

```text
Respondent Profile
        ↓
General Reasoning
        ↓
Ground-truth Label
        ↓
Student SFT
```

这一方法主要用于回答：

> 相比单纯使用 label supervision，加入一般性的 teacher reasoning 是否能够帮助 student 学习更有效的 decision pattern？

General Reasoning 同时生成两个版本：

```text
General Reasoning
├── No Thinking
└── Thinking-Low
```

---

## 2.4 Reasoning Strategy 2 — HBM-Light

第二种方法进一步引入 **Health Belief Model (HBM)**，使用一个相对轻量的 theory-guided reasoning prompt。

这里并不是要求 teacher 对每一个 respondent 都机械地分析所有 HBM dimensions。

相反，prompt 要求 teacher：

> 当 respondent profile 中存在对应 evidence 时，可以使用 HBM 作为 reasoning lens；如果没有 observable evidence，则不要强行推断对应的 psychological construct。

可能涉及的 HBM mechanisms 包括：

- perceived threat；
- perceived benefits；
- perceived barriers；
- self-efficacy；
- cues to action。

同时特别限制：

> 不允许仅根据 demographic variables 直接推断 respondent 的心理状态。

因此，HBM 在这里主要用于 **organize reasoning**，而不是作为 deterministic prediction rule。

整体形式为：

```text
Respondent Profile
        ↓
HBM-Aware Reasoning
        ↓
Ground-truth Label
        ↓
Student SFT
```

同样生成两种 teacher version：

```text
HBM-Light
├── No Thinking
└── Thinking-Low
```

---

## 2.5 当前实验设计

因此，目前 Teacher-Reasoning SFT 的主要实验矩阵为：

| Reasoning Strategy | Teacher Mode | Student |
|---|---|---|
| General Reasoning | No Thinking | Qwen3.5-9B |
| General Reasoning | Thinking-Low | Qwen3.5-9B |
| HBM-Light | No Thinking | Qwen3.5-9B |
| HBM-Light | Thinking-Low | Qwen3.5-9B |

后续主要希望比较：

1. **Reasoning supervision vs. Label-only SFT**
2. **General Reasoning vs. HBM-guided Reasoning**
3. **No Thinking vs. Thinking-Low**
4. Reasoning SFT 是否能够进一步接近或超过：
   - XGBoost；
   - TabPFN-3.5；
   - TabICLv2。

---

## 2.6 当前状态

目前 teacher reasoning generation 以及后续 student SFT pipeline 仍在运行中。

因此，本周暂时不汇报 reasoning SFT 的最终 performance。

后续将在所有实验完成后，统一比较：

```text
Label-only SFT
        ↓
General Reasoning SFT
        ↓
HBM-Light Reasoning SFT
```

并进一步分析不同 teacher reasoning strategy 是否真正能够改善 Qwen3.5-9B 在 vaccination behavior prediction 上的表现。
