# Fine-Tuning Strategies for Individual Vaccination Behavior Prediction

## Literature Review and Methodological Implications for the NHIS 2024 Project

> **Purpose of this README**\
> 这份文献综述不再按照"每篇论文用了哪些技术模块"机械罗列，而是统一回答五个问题：\
> **(1) 原论文研究的具体任务是什么？ (2) 作者发现现有方法哪里不够？ (3)
> 方法为什么这样设计？ (4) fine-tuning 过程中到底改变了什么？ (5)
> 对我们的 vaccination prediction 有什么可迁移的启示？**
>
> 因此，下面会尽量保留完整的研究逻辑，而只保留真正影响方法理解的技术细节。

------------------------------------------------------------------------

# 1. Our Research Context

我们的项目关注 **individual vaccination behavior prediction**：给定 NHIS
2024 respondent profile，预测该 respondent 在过去 12 个月内是否接种
influenza vaccine。

当前 supervised setting 可以概括为：

``` text
Respondent Profile
(62 observed survey features)
        ↓
Qwen3.5-9B + LoRA
        ↓
Vaccinated / Not Vaccinated
```

当前方法本质上属于 **label-only supervised fine-tuning
(SFT)**：模型在训练中看到 respondent profile，并学习直接输出最终
vaccination label。

因此，这次 literature review 真正想回答的不是：

> "还有哪些 fine-tuning algorithm 可以换？"

而是：

> **除了最终 label 之外，我们是否可以重新设计 fine-tuning
> 中模型看到的信息、学习的监督内容、训练样本的组织方式，以及 reasoning
> 在训练中的作用，从而让模型更好地学习 vaccination behavior？**

这也是为什么下面的六篇论文虽然来自不同任务，但可以放在同一条方法链中讨论：它们分别改变了
**input representation、training target、reasoning
supervision、training-data selection、error feedback、preference signal
或 context distillation**。

------------------------------------------------------------------------

# 2. Why These Papers Are Reviewed Together

The purpose of this review is not to identify one paper whose method can
be copied directly into the vaccination task. Instead, the six papers
are useful because they intervene at different points of the fine-tuning
pipeline.

The review follows the same progression as the presentation:

1.  **Wang & Ge** --- start from a task-specific failure and redesign
    fine-tuning around that failure.
2.  **From Numbers to Narratives** --- ask whether the model is seeing
    tabular information in an appropriate form.
3.  **TLRD** --- ask whether a final label alone is too weak as
    supervision and whether rationale supervision can provide richer
    learning signals.
4.  **MoRSD** --- once rationales are used, ask which rationales are
    actually reliable and useful for the student.
5.  **FAIR** --- move from generic supervision toward supervision
    targeted at the student's own failure modes.
6.  **OPCD** --- if external context is shown to be useful, ask whether
    that context can eventually be internalized into the student model.

This gives a single methodological progression:

``` text
Task-Specific Failure
        ↓
Input Representation
        ↓
Richer Supervision
        ↓
Rationale Quality
        ↓
Error-Aware Supervision
        ↓
Context Internalization
```

The important point is that these are not six competing methods. They
answer different questions about **what the model sees, what the model
is taught, which supervision is worth learning, and how useful external
knowledge might eventually be internalized**.

# 3. Wang & Ge

## *Align Generative Artificial Intelligence with Human Preferences: A Novel Large Language Model Fine-Tuning Method for Online Review Management*

**Authors:** Yanan Wang, Yong Ge\
**Source:** arXiv:2604.21209, 2026\
**Task:** Online review response generation / online review management

## 3.1 What problem does the paper study?

这篇论文研究的不是 classification，而是一个非常具体的 management task：

> **给定顾客发布的 online hotel review，让 LLM 自动生成合适的 managerial
> response。**

例如，一个顾客可能写：

``` text
The room was very noisy and I could not sleep.
```

酒店经理真实回复可能不仅会道歉，还会说：

``` text
We moved the guest to another room and contacted the night manager.
```

表面上看，这很适合直接做 supervised fine-tuning：

``` text
Customer Review
      ↓
Human Manager Response
```

但作者发现，真实 domain data 并不天然适合这样直接 fine-tune。

## 3.2 What is wrong with ordinary fine-tuning?

作者首先不是从"我要发明一个新的 DPO"开始，而是从 online review
management 的真实数据中识别 fine-tuning failure。

### Problem 1 --- The target may contain information absent from the input

Human manager response 经常包含：

-   酒店内部政策；
-   事件后续处理；
-   已采取的补救措施；
-   review 中没有写出的具体背景。

因此，如果直接训练：

``` text
Review → Human Response
```

模型实际上被要求学习生成一些 **无法从 review 本身推出的事实**。

这会制造一个结构性的 hallucination source：问题并不只是模型"喜欢
hallucinate"，而是 training target 本身就包含 input 无法支持的信息。

### Problem 2 --- Historical responses are not the same as explicit human preferences

即使解决 context 缺失，普通 SFT 学到的仍然主要是：

> "历史上经理是怎么回复的。"

但作者真正关心的是：

> "什么样的回复更符合 domain-specific human preference？"

原始数据只有 review--response pair，并没有天然的：

``` text
Preferred Response
vs.
Rejected Response
```

因此无法直接进行 preference learning。

### Problem 3 --- Offline preference optimization may be too conservative

即使构造了 preference pairs，offline preference fine-tuning
还面临另一个问题：为了不偏离 reference policy
太远，模型可能过度保守，从而限制它学习更好的 domain-specific response
behavior。

所以这篇论文真正的研究逻辑是：

``` text
先找出 domain FT 为什么失败
        ↓
再为每一种 failure 设计对应的训练模块
```

而不是先选择一个 generic FT algorithm 再硬套任务。

------------------------------------------------------------------------

## 3.3 How does the proposed method solve these problems?

### Stage 1 --- Context Augmentation

针对"human response 包含 review 中不存在的信息"，作者先从 training pair
中恢复 missing context。

训练数据原本是：

``` text
Review x + Human Response y
```

利用 GPT-4 分析二者，提取 **response 中需要、但 review
中没有的事实**，形成 context `c`。

于是训练材料从：

``` text
Review x → Response y
```

变成：

``` text
Review x + Context c → Response y
```

这里最重要的不是 GPT-4 本身，而是方法思想：

> **如果 target 依赖 input 中不存在的信息，就先修复 input--target
> information mismatch，再进行 fine-tuning。**

### Stage 2 --- Context-Aware SFT

在补全 context 后，模型进行标准 SFT：

``` text
Input:
Instruction + Review + Context

Target:
Human Manager Response
```

因此第一阶段的 fine-tuning 并没有发明新的 loss；创新首先来自 **training
data construction**。

### Stage 3 --- Theory-Driven Preference Pair Construction

接下来作者处理"历史 response ≠ 明确 preference"的问题。

他们保留经过筛选的 human response 作为 preferred response，并根据 online
review management 中的 domain theory 构造 less-preferred alternatives。

对于 negative reviews，作者使用与 complaint handling
相关的理论线索，例如 justice-related considerations 以及 rational /
emotional response cues；对于 positive reviews，则考虑 template-like 与
tailored response 的区别。

最终得到：

``` text
Review + Context
        ↓
Preferred Response
vs.
Rejected Response
```

因此，**domain theory 不只是写进 prompt
里告诉模型"请考虑理论"**，而是被用来定义 training signal 本身：什么
response 应该被偏好，什么 response 应该被拒绝。

### Stage 4 --- Preference Fine-Tuning

有了 preference pairs 后，作者再进行 preference learning。

他们还考虑不同 preference pair 的学习难度，通过 curriculum
的方式从更容易区分的 pair 逐步进入更困难的 pair。

最后，作者进一步修改 offline preference optimization 的约束，使 policy
可以在仍然受到历史数据支持的区域内更充分地偏离 reference
model，缓解过度保守的问题。

------------------------------------------------------------------------

## 3.4 What is the full training logic?

可以把整篇论文压缩成：

``` text
Historical Review–Response Data
        ↓
Identify missing context
        ↓
Review + Context → Response
        ↓
Context-Aware SFT
        ↓
Use domain theory to construct preference pairs
        ↓
Preferred vs. Rejected Responses
        ↓
Curriculum Preference Fine-Tuning
        ↓
Domain-aligned response generator
```

## 3.5 Why is this paper important for our project?

这篇论文对我们最重要的启示不是"我们也应该做 DPO"。

真正值得借鉴的是它的 **method development logic**：

> **先明确 label-only SFT 在 vaccination prediction
> 中到底缺了什么，再为这个具体缺陷设计 training supervision。**

对于我们的 NHIS task，可以问：

1.  **Information problem:** respondent profile 是否缺少真正决定
    vaccination 的 psychological constructs？
2.  **Representation problem:** 62 个 survey fields 是否只是被机械
    serialization，而没有形成清晰的 behavioral structure？
3.  **Supervision problem:** `Profile → Label` 是否提供的信息过于稀疏？
4.  **Reasoning problem:** 即使增加 rationale，什么样的 rationale
    才是我们希望 student 学习的？
5.  **Theory problem:** HBM 应该只是 prompt guidance，还是可以进一步参与
    supervision construction / rationale preference？

其中第 5 点是一个后续研究方向，而不是 Wang & Ge 已经证明的 vaccination
结论。

------------------------------------------------------------------------

# 4. Hayat, Tridle & Hasan

## *From Numbers to Narratives: Efficient Language Model-Based Detection for Safety-Critical Minority Classes*

**Authors:** Ahatsham Hayat, Hunter Tridle, Mohammad Rashedul Hasan\
**Venue:** Findings of EACL 2026\
**Task:** Safety-critical minority-class prediction from tabular data

## 4.1 What problem does the paper study?

这篇论文研究的是 **tabular classification**，尤其关注 safety-critical
datasets 中的 minority class。

典型场景包括：

-   machine failure detection；
-   semiconductor failure detection；
-   identifying at-risk students；
-   medical / safety-related classification。

作者指出，在 class imbalance 较强的任务中，一个模型可能有很高的 overall
accuracy，却仍然无法很好识别真正重要的 minority cases。

例如，如果绝大多数机器都不会故障，那么模型一直预测 "no failure"
也可以得到很高 accuracy，但这种模型在实际 safety-critical application
中没有足够价值。

## 4.2 What limitation do the authors identify?

论文的核心观察不仅是 class imbalance，还包括 **tabular representation 与
language model 的 mismatch**。

传统 tabular row 往往是：

``` text
Age = 42
Hypertension = 1
Glucose = 182
...
```

这些值对于 tree model 很自然，但对预训练 language model 来说：

-   feature-value pair 的语义并不充分显式；
-   数值之间的 domain relationship 没有自然语言上下文；
-   pretrained linguistic knowledge 很难直接与孤立数值连接。

所以作者没有首先修改 loss，而是先问：

> **如果我们把同一行数据改写成 LLM 更熟悉的语义形式，会不会让 downstream
> fine-tuning 更有效？**

------------------------------------------------------------------------

## 4.3 What does the method do?

### Stage 1 --- Structured Verbalization

作者使用 domain description 和 feature definitions，把 tabular row 转成
context-rich natural-language narrative。

例如：

``` text
Age = 42
Hypertension = 1
Glucose = 182
```

可以被表达为类似：

``` text
The individual is 42 years old, has hypertension,
and has an elevated glucose measurement.
```

关键点是：

> **改变 representation，但不应该改变 observed information。**

也就是说，verbalization 的目标不是让 LLM
自由"解释"数据，而是让原本抽象的 structured values 获得更明确的
linguistic semantics。

### Stage 2 --- Minority-Class Augmentation

由于论文特别关注 minority class，作者进一步只对 minority-class
narratives 进行语言层面的 augmentation，例如：

-   backtranslation；
-   contextual synonym replacement。

这样可以增加少数类 training examples 的语言变化，同时尽量保持
class-defining information 不变。

### Stage 3 --- Fine-Tune a Text Classifier

最后，fine-tuning task 仍然非常直接：

``` text
Narrative Representation
        ↓
Fine-Tuned Text Classifier
        ↓
Class Label
```

也就是说，这篇论文 **没有把 rationale 作为 target**。

它改变的是：

``` text
Raw Table → Label
```

中的左边：

``` text
Semantic Narrative → Label
```

而不是把右边变成 reasoning + label。

------------------------------------------------------------------------

## 4.4 What happens at inference?

新 tabular sample 在 inference 时也需要经过相同的 verbalization
pipeline：

``` text
New Tabular Row
        ↓
Structured Verbalization
        ↓
Fine-Tuned Classifier
        ↓
Prediction
```

因此，这里的 semantic representation 并不是只有 training 时存在的
auxiliary information，而是模型实际输入格式的一部分。

## 4.5 What does this suggest for NHIS?

我们的 respondent profile 同样是 structured survey
data，因此这篇论文给出了一个非常干净的 side experiment：

### Current representation

``` text
AGE = 68
INSURANCE = YES
DOCTOR_VISIT = FREQUENT
...
```

### Semantic representation

可以按我们已有的 feature categories 组织为：

``` text
Demographics:
...

Health status:
...

Healthcare access:
...

Healthcare utilization:
...

Previous vaccination history:
...
```

这里最重要的控制是：

> **只重新组织已经观察到的信息，不把没有测量的 HBM psychological
> constructs 写成 respondent 的真实心理状态。**

因此可以独立比较：

``` text
Raw Serialization → Label
vs.
Semantic / Grouped Serialization → Label
```

这样可以回答：

> **当前 SFT 的限制是否部分来自 input representation，而不只是
> supervision target？**

------------------------------------------------------------------------

# 5. Liang et al. 

## *TLRD: Teaching LLMs to Reason over Tabular Data with Tri-Level Rationale Distillation*

**Authors:** Tianyuan Liang, Xuwei Tan, Lei Shi, Junsheng Zhong, Ziyu
Hu, Tian Xie, Zhiqun Zuo, Xiaodong Yu, Xueru Zhang\
**Source:** arXiv:2606.08295, 2026\
**Task:** Tabular classification and regression with grounded
explanations

## 5.1 What problem does the paper study?

TLRD 与我们的 setting 最接近，因为它同样研究：

> **如何让 LLM 对 structured tabular features 做 prediction。**

作者希望模型不仅输出 prediction，还能够生成 readable、case-specific
explanation。

问题在于，LLM 虽然知道很多 feature 的一般语义，却不天然知道某一个
**specific dataset** 的统计规律。

例如，模型可能知道：

> "higher income" 是什么意思。

但它不一定知道：

> 这个 income 在当前 dataset 中到底算高还是低？\
> 哪些 feature combinations 在这个 dataset 中经常对应某个 label？\
> 与当前 sample 相似的人通常是什么 outcome？

这就是作者强调的 **dataset-specific patterns**。

------------------------------------------------------------------------

## 5.2 Why is label-only fine-tuning insufficient?

最简单的 tabular SFT 是：

``` text
Serialized Row → Label
```

这种训练可以让模型学习 prediction boundary，但 label
本身只提供非常稀疏的监督。

它没有显式告诉模型：

-   哪些 features 是支持证据；
-   哪些 features 与 outcome 冲突；
-   当前值在 population 中处于什么位置；
-   相似 historical cases 与当前 sample 有什么相同和不同。

因此 TLRD 的核心问题可以理解为：

> **能不能把训练数据中原本隐含的 dataset knowledge 先变成 explicit
> reasoning supervision，再让 student 学？**

------------------------------------------------------------------------

## 5.3 How does TLRD construct better supervision?

TLRD 不是直接让 teacher 看一行数据随便写 CoT，而是先给 teacher
三个层级的 evidence。

### Level 1 --- Instance-Level Evidence

Teacher 得到：

``` text
Current Row + Ground-Truth Label
```

因为这是 training-data construction 阶段，teacher 可以知道真实
outcome，再分析：

> 当前 sample 中哪些 observed features 支持这个 outcome？\
> 哪些 features 可能形成相反信号？

这里的 gold label 是为了生成 supervision，而不是 student inference
时的输入。

### Level 2 --- Dataset-Level Evidence

作者从 training data 中计算 population / class-conditional statistics。

这样 teacher 不再只凭一般常识说：

``` text
Feature X is high.
```

而可以把当前 sample 放回这个 dataset 的 distribution 中理解。

这一步的意义是：

> **让 explanation grounded in the actual dataset，而不是只依赖
> pretrained world knowledge。**

### Level 3 --- Comparison-Level Evidence

作者进一步检索与当前 sample 相似的 historical training cases，并让
teacher 比较：

-   与同类 outcome cases 有哪些相似点；
-   与不同 outcome cases 有哪些差异。

于是 teacher reasoning 不再只是单个 feature 的解释，而包含 **case-based
comparison**。

------------------------------------------------------------------------

## 5.4 How is the student actually fine-tuned?

Teacher 利用三层 evidence 生成 structured rationale，大体包含：

``` text
Instance reasoning
        ↓
Dataset-level reasoning
        ↓
Comparison with similar cases
        ↓
Final resolution / prediction
```

然后 student 的训练任务是：

``` text
Input:
Serialized Raw Row

Target:
Teacher Rationale + Label
```

所以 TLRD 最关键的设计不是"teacher 更大"，而是：

> **把原来的 label-only dataset 转换成 rationale-supervised dataset。**

值得特别注意的是，student 并不需要在 inference 时重新访问这些额外证据。

训练阶段 teacher 可以看到：

``` text
Row
+ Gold Label
+ Population Statistics
+ Retrieved Historical Cases
```

但 student 最终学习：

``` text
Raw Row
        ↓
Rationale + Prediction
```

也就是说，statistics 和 neighbors 是 **supervision construction
resources**，不是 deployment requirement。

------------------------------------------------------------------------

## 5.5 Why is TLRD especially relevant to our project?

我们当前的训练：

``` text
Profile → Vaccination Label
```

可以最直接地扩展成：

``` text
Profile + Gold Vaccination Label
        ↓
Teacher constructs rationale
        ↓
Profile → Rationale + Vaccination Label
```

但这里可以分阶段，而不需要一次把 TLRD 全部复制过来。

### First transfer: rationale supervision only

先固定 respondent profile，不加 population statistics /
neighbors，只比较：

``` text
Label-only SFT
vs.
Generic Rationale SFT
vs.
HBM-Guided Rationale SFT
```

这回答最基础的问题：

> **reasoning supervision 本身是否比 final-label supervision
> 提供更多价值？**

### Later transfer: stronger evidence

如果 rationale supervision 确实有效，再逐步加入：

``` text
Training-population statistics
```

以及：

``` text
Similar historical respondents
```

这样我们才能区分 improvement 到底来自：

-   多了 reasoning text；
-   多了 behavioral theory；
-   还是多了 dataset-specific evidence。

------------------------------------------------------------------------

# 6. Yan et al. 

## *Towards Efficient CoT Distillation: Self-Guided Rationale Selector for Better Performance with Fewer Rationales*

**Authors:** Jianzhi Yan, Le Liu, Youcheng Pan, Shiwei Chen, Yang Xiang,
Buzhou Tang\
**Venue:** Findings of EMNLP 2025\
**Method name:** Model-Oriented Rationale Selection Distillation
(MoRSD)\
**Task:** Mathematical, commonsense, and temporal/spatial reasoning

## 6.1 What problem does the paper study?

CoT distillation 的常见做法是：

``` text
Teacher generates rationales
        ↓
Student fine-tunes on rationales
```

一个自然想法是：

> teacher rationale 越多越好。

但 MoRSD 指出，这个假设并不成立。

同一个 question 可以生成很多 rationales，但它们可能：

-   reasoning 不正确；
-   最终答案碰巧正确但中间逻辑有问题；
-   内容高度重复；
-   对当前 student 来说过于困难；
-   对另一个 student 有帮助，但对当前 student 没帮助。

所以论文的核心问题不是：

> "怎么生成更多 rationale？"

而是：

> **哪些 rationale 值得真正进入 student 的 fine-tuning data？**

------------------------------------------------------------------------

## 6.2 How does MoRSD solve this?

### Stage 1 --- Generate multiple candidate rationales

对于同一个 question 和 gold answer，teacher 先生成多个 candidate
rationales。

### Stage 2 --- Remove unreliable rationales

首先检查 rationale 是否能够支持正确 answer。

这一步避免把明显错误的 reasoning 当成 supervision。

### Stage 3 --- Remove redundant rationales

如果多个 rationales 本质上表达相同 reasoning path，就没有必要全部保留。

因此方法进一步控制 diversity，减少重复 supervision。

### Stage 4 --- Evaluate usefulness from the student's perspective

这是 MoRSD 最重要的部分。

作者提出 **Rationale Difficulty (RD)**，核心思想是比较：

> student 在没有 rationale 时预测正确 answer 有多困难？

和：

> 给 student 这条 rationale 后，正确 answer 是否变得更容易？

因此 rationale quality 不再只由 teacher 或 external evaluator 决定，而与
**specific student model** 联系起来。

最后只用筛选后的 rationale 做标准 reasoning distillation。

------------------------------------------------------------------------

## 6.3 What is the main conceptual contribution?

MoRSD 最值得我们记住的一句话是：

> **Teacher-generated rationale is not automatically good supervision.
> Its value depends on whether it is correct, non-redundant, and useful
> for the student being trained.**

这对我们尤其重要，因为我们准备让较大的 teacher 根据：

``` text
Respondent Profile + Gold Vaccination Label
```

生成 reasoning。

Teacher 已知答案以后，很容易写出一个"听起来合理"的 explanation。

但：

> **label-consistent explanation ≠ evidence-grounded explanation ≠
> student-useful explanation**

这三者需要区分。

------------------------------------------------------------------------

## 6.4 What can we borrow?

如果 Generic / HBM-guided rationale SFT 有效，我们后续不应该简单扩大
teacher-generated rationale 数量，而可以建立 rationale quality
control，例如检查：

1.  是否与最终 label 一致；
2.  是否只引用 profile 中真实存在的信息；
3.  是否编造未测量的 psychological state；
4.  是否高度重复；
5.  是否真的帮助当前 Qwen3.5-9B 做 prediction。

因此 MoRSD 对我们的作用主要在：

> **rationale selection / quality control，而不是第一轮 rationale
> generation。**

------------------------------------------------------------------------

# 7. Li et al. 

## *Learning from Committee: Reasoning Distillation from a Mixture of Teachers with Peer-Review*

**Authors:** Zhuochun Li, Yuelyu Ji, Rui Meng, Daqing He\
**Venue:** Findings of ACL 2025\
**Method name:** Fault-Aware Distillation via Peer-Review (FAIR)\
**Task:** Mathematical, commonsense, and logical reasoning

## 7.1 What problem does the paper study?

普通 reasoning distillation 是 teacher-centered：

``` text
Teacher produces a correct rationale
        ↓
Student imitates it
```

但作者认为这种方式忽略了一个很重要的问题：

> **student 自己到底错在哪里？**

两个 student 即使在同一个 dataset 上训练，也可能有不同 weakness。

如果我们只是不断给 student
看"正确答案是怎么推出来的"，却不分析它自己的错误 reasoning，那么
supervision 并没有真正针对 student 的 failure mode。

------------------------------------------------------------------------

## 7.2 How does FAIR construct training data?

### Stage 1 --- Let the student produce its own reasoning

Student 先对 training examples 生成：

``` text
Student Rationale + Student Answer
```

然后与 gold answer 比较。

如果 student 做错，这个 case 就不只是一个 "wrong prediction"，而成为一个
**diagnostic example**。

### Stage 2 --- Ask teachers to explain the mistake

Teacher 不再只生成标准 gold rationale，而是看到：

``` text
Original Question
+ Student's Wrong Rationale
+ Gold Answer
```

然后生成两类信息：

1.  **Correct reasoning**
2.  **Mistake-specific feedback**

也就是说，teacher 不只是说：

> "正确应该怎么做。"

还要说：

> "你刚才具体错在哪里。"

### Stage 3 --- Peer review the teacher supervision

FAIR 还考虑 teacher 自己也可能生成 flawed rationale。

因此多个 teacher 之间进行 simulated peer review，对 candidate rationales
评分，只保留达到质量标准的内容。

### Stage 4 --- Joint fine-tuning

最终 student 同时学习：

``` text
Input → Correct Rationale
```

以及：

``` text
Input + Student Wrong Rationale
        → Mistake Feedback
```

因此 fine-tuning data 不再对所有 examples 一视同仁，而是显式包含
**student-specific corrective supervision**。

------------------------------------------------------------------------

## 7.3 Why is FAIR relevant to our existing error analysis?

这篇论文和我们已经做的 SFT vs. XGBoost error analysis 很容易连接起来。

我们已经可以把 test / analysis cases 分成：

``` text
Both correct
Both wrong
SFT-only correct
XGBoost-only correct
```

尤其值得分析：

``` text
SFT wrong / XGBoost correct
```

以及反方向：

``` text
XGBoost wrong / SFT correct
```

但如果要真正迁移 FAIR，不能直接拿 held-out test errors
重新训练，否则会破坏 evaluation independence。

更合理的方式是在 training / cross-fitted data 中：

1.  让 current SFT student 对 profile 做 prediction / reasoning；
2.  找出稳定出现的 student errors；
3.  让 teacher 针对这些 errors 生成 corrective reasoning；
4.  再构造 error-aware fine-tuning data。

因此 FAIR 给我们的不是一个简单的"再加一种
rationale"，而是一个更深的方向：

> **从 average supervision 转向 failure-targeted supervision。**

------------------------------------------------------------------------

# 8. Ye et al. 

## *On-Policy Context Distillation for Language Models*

**Authors:** Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, Furu Wei\
**Source:** arXiv:2602.12275, 2026\
**Method name:** On-Policy Context Distillation (OPCD)\
**Task:** Experiential knowledge distillation and system-prompt
distillation across reasoning, text-based games, and domain-specific
tasks

## 8.1 What problem does the paper study?

很多时候，我们已经知道额外 context 可以让模型表现更好。

例如，一个模型可能在得到：

-   optimized system prompt；
-   historical solution experience；
-   external guidance；

以后明显变强。

但如果每次 inference 都必须重新提供 context，那么 knowledge 始终存在于
prompt，而没有真正进入 model parameters。

因此 OPCD 研究：

> **能不能把"有 context 的 teacher 行为"蒸馏进一个 inference 时没有
> context 的 student？**

------------------------------------------------------------------------

## 8.2 Why is ordinary offline distillation not enough?

普通 distillation 常常是：

``` text
Teacher generates a fixed answer
        ↓
Student imitates teacher answer
```

但 student 真正在 inference 时会产生自己的 token trajectory。

如果 training 只在 teacher 的 trajectory 上学习，student 可能没有学会：

> 当它自己走到某个状态时，拥有额外 context 的 teacher 会怎么继续？

所以 OPCD 把 **on-policy learning** 与 **context distillation**
结合起来。

------------------------------------------------------------------------

## 8.3 How does OPCD work?

### Stage 1 --- Student generates its own trajectory

Student 只看到原始 input，不看到 extra context。

它自己生成 response trajectory。

### Stage 2 --- Context-conditioned teacher evaluates the same trajectory

Teacher 则看到：

``` text
Extra Context
+ Original Input
+ Student-generated Prefix
```

并在 student 实际走过的 trajectory 上提供 token-level distribution。

### Stage 3 --- Distill teacher behavior into student

训练目标让 student 的 token distribution 接近这个 **拥有 context 的
teacher**。

因此 student 学的不是一段固定 teacher answer，而是：

> **在我自己实际会到达的生成状态上，如果我拥有这些额外知识，我应该怎样行动？**

最终 inference 时：

``` text
Original Input
        ↓
Student
        ↓
Output
```

不再需要额外 context。

------------------------------------------------------------------------

## 8.4 What does this mean for our project?

OPCD 对我们不是第一阶段方案，因为它有一个前提：

> **我们必须先证明某一种 extra context 确实有价值。**

例如，我们未来可能发现：

``` text
HBM guidance
```

或：

``` text
Training-population statistics
```

或：

``` text
Similar historical respondents
```

在 teacher / prompted LLM 中能够稳定改善 vaccination prediction。

只有在这个前提成立以后，才值得进一步问：

> 能不能把这种 context 的 benefit internalize 到 Qwen3.5-9B 中，使
> inference 仍然只需要 respondent profile？

因此 OPCD 更像是我们方法路线后期的 **context → parameter
internalization** 方案。

------------------------------------------------------------------------

# 9. Putting the Six Papers Together

The six papers can be read as a sequence of increasingly specific
questions about fine-tuning design.

  -----------------------------------------------------------------------
  Paper                   Main Question           Relevance to Our
                                                  Project
  ----------------------- ----------------------- -----------------------
  **Wang & Ge**           What task-specific      Do not begin from a
                          failure are we actually favorite fine-tuning
                          trying to fix?          technique; begin from
                                                  the limitation of the
                                                  current vaccination
                                                  SFT.

  **From Numbers to       Is the model seeing the Test whether semantic
  Narratives**            data in the right form? organization of the
                                                  same 62 observed
                                                  features improves
                                                  learning.

  **TLRD**                Is the final label too  Test whether
                          weak as supervision?    rationale + label
                                                  supervision provides
                                                  more useful learning
                                                  signals than label-only
                                                  SFT.

  **MoRSD**               Which rationales are    If rationale SFT helps,
                          actually worth          introduce quality
                          learning?               control rather than
                                                  simply generating more
                                                  rationales.

  **FAIR**                Can supervision target  Use training or
                          the student's own       cross-fitted student
                          failure modes?          errors to construct
                                                  corrective supervision
                                                  for systematic failure
                                                  patterns.

  **OPCD**                Can useful external     Only after HBM
                          context become internal guidance, population
                          knowledge?              statistics, or
                                                  historical cases are
                                                  shown to help, consider
                                                  distilling that context
                                                  into the student.
  -----------------------------------------------------------------------

The resulting logic is:

``` text
Input Representation
        ↓
Supervision Enrichment
        ↓
Rationale Selection
        ↓
Failure-Targeted Supervision
        ↓
Context Internalization
```

Wang & Ge provides the broader methodological principle surrounding this
sequence: **identify the actual task failure first, and only then decide
which part of fine-tuning should change.**

------------------------------------------------------------------------

# 10. The Most Direct Experimental Design for Our Project

Based on the literature, the next step should not be a single complex
framework combining every idea at once. The cleaner strategy is to
separate the questions.

## 10.1 Input Representation

The first question is whether the current representation of the 62
survey features is limiting the language model.

### A0 --- Raw Profile

``` text
Raw / mechanically serialized profile
        ↓
Student
        ↓
Vaccination Label
```

### A1 --- Semantically Grouped Profile

``` text
Same observed features
organized by meaningful feature groups
        ↓
Student
        ↓
Vaccination Label
```

Possible groups include:

-   demographics;
-   socioeconomic characteristics;
-   health status and chronic conditions;
-   insurance and affordability;
-   healthcare access;
-   healthcare utilization;
-   previous preventive and vaccination behavior.

The critical constraint is that **A1 reorganizes existing information
but does not invent new information**.

Therefore:

> **A1 vs. A0 asks whether input representation itself matters.**

------------------------------------------------------------------------

## 10.2 Supervision Design

The more important immediate comparison is to keep the respondent
information fixed while changing what the student is asked to learn.

### B0 --- Label-Only SFT

``` text
Profile
   ↓
Student
   ↓
Label
```

This is the current baseline.

### B1 --- Generic Rationale SFT

Teacher:

``` text
Profile + Gold Label
        ↓
Evidence-Based Generic Rationale
```

Student:

``` text
Profile
   ↓
Generic Rationale + Label
```

This condition asks:

> **Does reasoning supervision itself provide value beyond the final
> label?**

### B2 --- HBM-Guided Rationale SFT

Teacher:

``` text
Profile + Gold Label + HBM Guidance
        ↓
HBM-Guided Evidence-Based Rationale
```

Student:

``` text
Profile
   ↓
HBM-Guided Rationale + Label
```

This condition asks:

> **Does behavioral theory add value beyond generic reasoning
> supervision?**

The cleanest immediate comparison is therefore:

``` text
B0: Label-Only SFT
        ↓
B1: Generic Rationale SFT
        ↓
B2: HBM-Guided Rationale SFT
```

Two research questions follow directly:

1.  **B1 vs. B0:** Does rationale supervision itself help?
2.  **B2 vs. B1:** Does HBM-guided reasoning add value beyond generic
    reasoning?

This staged comparison is easier to interpret than immediately combining
HBM, historical memory, population statistics, error correction, and
other mechanisms in one model.

------------------------------------------------------------------------

# 11. What Role Should HBM Play?

A critical issue is that NHIS does **not** directly measure the full set
of latent Health Belief Model constructs, such as complete perceived
susceptibility, perceived severity, perceived benefits, perceived
barriers, self-efficacy, or vaccine trust.

Therefore, HBM should initially be used as a **reasoning lens**, not as
permission to infer unobserved psychological states.

The teacher can use HBM to organize observed evidence, for example:

-   observed health and chronic-condition information as possible
    threat-related evidence;
-   insurance, affordability, and access variables as barrier-related
    evidence;
-   healthcare contact and utilization as possible cues or
    healthcare-support signals;
-   previous preventive or vaccination behavior as supporting or
    conflicting behavioral evidence.

However, if vaccine trust is not measured, the rationale should not
claim:

> "This respondent distrusts vaccines."

Likewise, the model should not convert a proxy into a certain latent
state merely because that interpretation fits the final label.

A suitable teacher instruction is:

> When reasoning about influenza vaccination, consider behavioral
> mechanisms from the Health Belief Model, including perceived threat,
> perceived benefits, barriers, self-efficacy, and cues to action when
> supported by the observed profile. Do not infer psychological states
> that are not supported by the data. Use HBM as a reasoning lens rather
> than a deterministic rule.

Thus, the role of theory is:

> **organize observed evidence, not invent latent psychological
> states.**

------------------------------------------------------------------------

# 12. Overall Methodological Roadmap

The literature suggests a staged progression rather than a single large
method.

``` text
Current:
Profile → Label

        ↓

Step 1:
Improve what the model sees
Input Representation

        ↓

Step 2:
Improve what the model is taught
Label → Rationale + Label

        ↓

Step 3:
Structure the reasoning
Generic Rationale → HBM-Guided Rationale

        ↓

Step 4:
Select which rationales are worth learning
MoRSD-style Rationale Quality Control

        ↓

Step 5:
Target systematic student failures
FAIR-style Error-Aware Supervision

        ↓

Step 6:
Add stronger dataset-specific evidence when justified
Population Statistics / Historical Cases

        ↓

Step 7:
If external context is demonstrably useful,
consider Context Internalization
OPCD
```

This roadmap deliberately keeps the early experiments simple.

The immediate goal is **not** to reproduce every mechanism from the
literature. It is to establish which source of improvement is actually
present in our task before adding additional complexity.

------------------------------------------------------------------------

# 13. Main Takeaway

The central conclusion from this literature review is not:

> "We found one paper whose method should be copied."

Instead, the papers suggest a way to progressively redesign the current
vaccination SFT.

The current baseline is:

``` text
Profile → Label
```

The next methodological question is whether we can construct supervision
that is:

-   more informative than the final label alone;
-   more structured in how evidence is organized;
-   constrained by observed survey information;
-   increasingly sensitive to rationale quality and student-specific
    failure modes.

The most natural immediate experiment is therefore:

``` text
Label-Only SFT
      vs.
Generic Rationale SFT
      vs.
HBM-Guided Rationale SFT
```

If rationale supervision does not improve over label-only SFT, there is
little reason to immediately add more complicated evidence or
distillation mechanisms.

If generic rationale supervision helps, then the next question is
whether HBM provides additional value.

If HBM-guided rationale supervision provides additional value, later
stages can examine rationale selection, error-aware supervision,
stronger dataset-specific evidence, and eventually context
internalization.

The methodological progression is therefore:

``` text
Profile → Label
      ↓
Semantic Input Representation
      ↓
Rationale + Label
      ↓
Generic → HBM-Guided Reasoning
      ↓
Rationale Quality Control
      ↓
Error-Aware Supervision
      ↓
Dataset-Specific Evidence
      ↓
Context Internalization
```

The core positioning is:

> **The goal is not to make the LLM generate longer reasoning. The goal
> is to construct more informative, theory-structured, and
> evidence-constrained fine-tuning supervision for individual
> vaccination behavior prediction.**

# 14. References

1.  **Wang, Y., & Ge, Y. (2026).** *Align Generative Artificial
    Intelligence with Human Preferences: A Novel Large Language Model
    Fine-Tuning Method for Online Review Management.* arXiv:2604.21209.

2.  **Hayat, A., Tridle, H., & Hasan, M. R. (2026).** *From Numbers to
    Narratives: Efficient Language Model-Based Detection for
    Safety-Critical Minority Classes.* Findings of the Association for
    Computational Linguistics: EACL 2026, 4920--4937.

3.  **Liang, T., Tan, X., Shi, L., Zhong, J., Hu, Z., Xie, T., Zuo, Z.,
    Yu, X., & Zhang, X. (2026).** *TLRD: Teaching LLMs to Reason over
    Tabular Data with Tri-Level Rationale Distillation.*
    arXiv:2606.08295.

4.  **Yan, J., Liu, L., Pan, Y., Chen, S., Xiang, Y., & Tang, B.
    (2025).** *Towards Efficient CoT Distillation: Self-Guided Rationale
    Selector for Better Performance with Fewer Rationales.* Findings of
    the Association for Computational Linguistics: EMNLP 2025,
    7818--7835.

5.  **Li, Z., Ji, Y., Meng, R., & He, D. (2025).** *Learning from
    Committee: Reasoning Distillation from a Mixture of Teachers with
    Peer-Review.* Findings of the Association for Computational
    Linguistics: ACL 2025, 4190--4205.\
    Method: **Fault-Aware Distillation via Peer-Review (FAIR).**

6.  **Ye, T., Dong, L., Wu, X., Huang, S., & Wei, F. (2026).**
    *On-Policy Context Distillation for Language Models.*
    arXiv:2602.12275.

------------------------------------------------------------------------
