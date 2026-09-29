# 个体疫苗接种行为预测的 Fine-Tuning 策略

## 文献综述及其对 NHIS 2024 项目的启示

> **这份 README 的目的**
>
> 这份文献综述按照实际汇报稿的叙述顺序组织。重点不是机械罗列每篇论文用了哪些技术，而是依次说明：
>
> **原论文在做什么任务 → 作者遇到了什么具体问题 → 针对这个问题怎么解决 →
> 这个设计为什么重要 → 对我们的 vaccination prediction 有什么启示。**
>
> 因此，每篇论文内部也尽量保持"提出一个问题，再讲对应解决方法"的顺序，而不是先把所有问题列完，再统一介绍所有方法。

------------------------------------------------------------------------

# 1. 为什么做这次 Literature Review

我们现在的 vaccination prediction 任务比较直接：给模型一个 respondent 的
profile，包括目前使用的 **62 个 survey features**，然后让 **Qwen3.5-9B**
通过 LoRA fine-tuning，最后预测这个人是 **vaccinated** 还是 **not
vaccinated**。

当前方法可以概括为：

``` text
Respondent Profile
(62 observed survey features)
        ↓
Qwen3.5-9B + LoRA
        ↓
Vaccinated / Not Vaccinated
```

本质上，我们现在做的还是比较标准的 **label-only SFT**：

``` text
Profile → Label
```

但目前单纯做 SFT 以后，performance
已经比较接近传统的机器学习模型。因此，这次 literature review
想看的并不是简单更换 backbone，或者继续调整 fine-tuning 参数，而是：

> **有没有可能改变 fine-tuning 过程中模型真正学习的内容？**

例如：

-   现在只学习最终 label，能不能提供更丰富的监督信息？
-   能不能改变 input 的组织方式，让 language model 更容易理解这些 survey
    features？
-   能不能让 teacher model 提供 reasoning，使 student 不只是学习
    prediction，也学习和预测相关的推理模式？
-   如果 reasoning 有用，什么样的 reasoning 才真正值得 student 学？
-   能不能针对 student 自己容易犯的错误进一步构造 supervision？

下面六篇文章虽然具体任务不同，但基本可以放在同一条方法线上理解。

有的改变 **input representation，也就是输入怎么表达**；有的改变
**training
target，也就是模型到底学什么**；有的研究如何筛选真正值得学习的
reasoning；还有一些直接针对 student 自己的错误构造新的监督信息。

------------------------------------------------------------------------

# 2. Wang & Ge

## *Align Generative Artificial Intelligence with Human Preferences: A Novel Large Language Model Fine-Tuning Method for Online Review Management*

**Authors:** Yanan Wang, Yong Ge\
**Source:** arXiv:2604.21209, 2026\
**Task:** 根据 online hotel review 自动生成 managerial response

我觉得这篇文章比较值得看的地方，不是它最后具体用了什么 preference
optimization，而是它整个方法是怎么一步一步设计出来的。

它研究的是一个非常具体的 management task：

> **给定一条酒店评论，让 LLM 自动生成酒店管理者的回复。**

最直接的方法当然是利用历史数据：

``` text
Customer Review → Human Manager Response
```

但是作者首先发现了一个很实际的问题。

## 2.1 问题一：Target 中可能包含 Input 没有的信息

例如 customer 只写：

> "房间很吵，我晚上睡不好。"

但是酒店经理真正的回复可能会说：

> "我们已经给客人换了房间，并且联系了 night manager。"

这里的"换房间"和"联系 manager"并没有出现在原始 review 中。

所以如果直接使用：

``` text
Review → Human Response
```

做 SFT，相当于要求模型根据一个信息不完整的 input，复现一个包含额外事实的
target。

作者认为，这本身就会形成一个结构性的 hallucination
source：模型需要生成一些 **无法从 review 本身推出的事实**。

### 对应解决方法：Context Augmentation

作者没有马上修改 loss，而是先补充缺失的 context。

他们利用：

``` text
Review x + Human Response y
```

分析 response 中有哪些信息是生成目标所需要、但原始 review
没有提供的，并将这些内容提取为额外 context `c`。

于是训练从：

``` text
Review → Response
```

变成：

``` text
Review + Context → Response
```

然后再进行正常的 SFT。

这里最重要的不是 GPT-4 或某个具体 extraction
procedure，而是这个方法逻辑：

> **在设计 fine-tuning 之前，先检查 input 和 target
> 之间的信息是不是匹配的。**

如果 target 依赖 input 中根本不存在的信息，那么首先应该修复 training
data，而不是直接要求模型学会复现 target。

------------------------------------------------------------------------

## 2.2 问题二：Historical Response 不等于 Explicit Preference

在补全 context 以后，作者进一步发现第二个问题。

即使训练数据已经变成：

``` text
Review + Context → Human Response
```

普通 SFT 学到的仍然主要是：

> **历史上的 manager 是怎么回复的。**

但这并不一定等于：

> **什么样的回复才是更好的回复。**

原始历史数据只有 review-response pair，并没有天然提供：

``` text
Preferred Response
vs.
Less-Preferred Response
```

### 对应解决方法：利用领域理论构造 Preference Pair

作者进一步利用领域理论去构造 preference pair，也就是偏好对。

对于同一个 review，他们保留更好的 preferred response，同时构造相对较差的
less-preferred response。

而且这种 preference 不是随机定义的。

例如对于 negative review，他们会参考 complaint handling、justice
theory，以及理性回应和情感回应等领域知识；对于 positive review，也会考虑
template-like response 和 tailored response 的区别。

因此，theory 的作用并不是简单地：

> "在 prompt 里面告诉模型，请考虑这个理论。"

而是：

> **理论直接参与定义什么样的 training signal
> 是好的，什么样的是不好的。**

在构造 preference pairs 以后，作者才进一步进行 preference
fine-tuning，并考虑不同 preference pair 的学习难度。

------------------------------------------------------------------------

## 2.3 对我们项目最重要的启示

所以这篇文章对我们最大的启示，并不是说我们也应该马上做 DPO。

它更重要的是提供了一个 **方法设计逻辑**：

``` text
先找清楚当前任务真正存在的问题
        ↓
再针对这个具体问题设计 fine-tuning 方法
```

对应到 vaccination prediction，我们首先应该问：

> **我们现在的 vaccination SFT 到底哪里不够？**

例如：

-   input 的表达方式是不是不够适合 language model？
-   `Profile → Label` 的监督是不是太弱？
-   reasoning 有没有足够结构？
-   teacher 生成的 rationale 是否真的 grounded in observed survey
    evidence？
-   HBM 是否能够进一步参与定义什么样的 reasoning 更值得 student 学？

这里尤其值得注意的是 HBM。

HBM 不一定只能作为一句 prompt。如果后面证明 **HBM-guided reasoning**
确实有价值，那么 theory 也可能进一步参与定义：

> **什么样的 reasoning 更值得 student 学。**

但这是我们后续可以研究的方向，并不是 Wang & Ge 已经在 vaccination task
上证明的结论。

------------------------------------------------------------------------

# 3. From Numbers to Narratives

## *From Numbers to Narratives: Efficient Language Model-Based Detection for Safety-Critical Minority Classes*

**Authors:** Ahatsham Hayat, Hunter Tridle, Mohammad Rashedul Hasan\
**Venue:** Findings of EACL 2026\
**Task:** Safety-critical minority-class prediction from tabular data

这篇和我们比较接近的一点是，它同样做 **tabular
classification，也就是表格数据分类**。

它关注一些比较重要的少数类预测问题，例如 machine failure、medical
classification，或者识别有风险的学生。

作者首先指出一个比较常见的问题：

> 如果数据类别不平衡，一个模型整体 accuracy
> 很高，并不代表它真的能够识别那些数量较少、但非常重要的 minority
> cases。

例如绝大多数机器都不会故障，那么模型一直预测 "no failure" 也可能得到很高
accuracy，但这种模型在实际 safety-critical application 中价值有限。

不过，对我们更有意思的是作者进一步提出的另一个问题。

## 3.1 问题：原始 Tabular Data 不是 Language Model 最熟悉的输入形式

原始输入可能是：

``` text
Age = 42
Hypertension = 1
Glucose = 182
```

对于 tree model 来说，这种 feature-value representation 没有什么问题。

但是对于 pretrained language model
来说，这些数字和变量之间的语义关系并不是特别明确。

因此作者没有先改变 training objective，而是先改变：

> **input representation，也就是数据怎么呈现给模型。**

### 对应解决方法：Structured Verbalization

他们利用 domain description 和 feature definitions，把每一行 tabular
data 转换成更自然的语言描述。

例如：

``` text
Age = 42
Hypertension = 1
Glucose = 182
```

可以表达成类似：

> "这个人 42 岁，有 hypertension，同时 glucose level 比较高。"

这里的重点不是让 LLM 自由发挥，而是：

> **把原来的 feature information
> 用更有语义的形式重新表达，同时不增加原始数据中不存在的信息。**

由于论文特别关注 minority class，作者之后还对 minority-class narratives
做语言层面的 augmentation，例如 backtranslation 和 contextual synonym
replacement。

最后训练仍然是：

``` text
Narrative → Label
```

所以这篇文章没有改变 reasoning，也没有改变最终监督目标。

它真正改变的是：

``` text
Raw Table → Label
```

中的左边，把它变成：

``` text
Semantic Narrative → Label
```

------------------------------------------------------------------------

## 3.2 对我们项目的启示

这个设计可以直接形成一个很干净的实验。

我们现在可能是把 62 个 feature 比较机械地 serialize 出来。

那么可以比较：

``` text
Raw Serialization → Label
```

和：

``` text
Semantically Grouped Profile → Label
```

例如按照我们已有的 feature categories 组织：

``` text
Demographics
Health status
Healthcare access
Healthcare utilization
Previous vaccination history
...
```

但这里不增加任何新的信息，也不把没有测量的 HBM psychological constructs
写成 respondent 的真实状态。

这样可以单独回答：

> **我们现在的问题，有没有一部分其实来自 input representation？**

------------------------------------------------------------------------

# 4. TLRD

## *TLRD: Teaching LLMs to Reason over Tabular Data with Tri-Level Rationale Distillation*

**Authors:** Tianyuan Liang, Xuwei Tan, Lei Shi, Junsheng Zhong, Ziyu
Hu, Tian Xie, Zhiqun Zuo, Xiaodong Yu, Xueru Zhang\
**Source:** arXiv:2606.08295, 2026\
**Task:** Tabular classification and regression with grounded
explanations

第三篇是我觉得和我们现在最直接相关的一篇。

它同样是在做 tabular classification 和 regression。

## 4.1 问题：LLM 不知道当前 Dataset 中 Feature 的具体含义

LLM 可能知道一个 feature 一般是什么意思，但它不一定知道这个 feature 在
**当前 dataset** 里面具体意味着什么。

例如它知道 income 高低的一般含义，但它不一定知道：

-   这个 income 在当前 dataset 中到底算高还是低；
-   和这个 sample 类似的人通常是什么 outcome；
-   哪些 feature combinations 在当前 dataset 中比较重要。

所以作者认为，只做：

``` text
Row → Label
```

supervision 太弱。

因为 label 只告诉模型最后答案，却没有告诉它：

> **为什么这个 sample 应该属于这个 label。**

### 对应解决方法：给 Teacher 三层 Evidence

TLRD 不是直接让 teacher 随便生成一段 CoT，而是先给 teacher 三层不同的
evidence。

### 第一层：Instance-Level Evidence

Teacher 会看到：

``` text
Current Row + Ground-Truth Label
```

因为这是 training-data construction 阶段，所以 teacher
可以知道正确答案，然后分析当前 features 中：

-   哪些支持这个 outcome；
-   哪些可能形成相反或冲突的 signal。

这里 gold label 的作用是帮助构造 supervision，而不是 student inference
时的输入。

### 第二层：Dataset-Level Evidence

作者进一步给 teacher 当前 training population 的统计信息。

因此 teacher 不只是说：

> "这个 feature 很高。"

而可以进一步根据当前 dataset 的 distribution 判断这个 value
到底处于什么位置。

这样 explanation 会更多 grounded in the actual dataset，而不是只依赖
pretrained world knowledge。

### 第三层：Comparison-Level Evidence

作者还会找和当前 sample 相似的 historical training cases。

Teacher 可以比较：

-   当前 sample 和同类 outcome cases 有哪些相似点；
-   和不同 outcome cases 有哪些差异。

这样 reasoning 就不再只是孤立解释单个 feature，而可以包含 case-based
comparison。

------------------------------------------------------------------------

## 4.2 Student 最终学习什么？

Teacher 把三层 evidence 整合成一个结构化 rationale：

``` text
Instance reasoning
        ↓
Dataset-level reasoning
        ↓
Comparison with similar cases
        ↓
Final resolution / prediction
```

但 student 真正训练时，并不需要看到这些 statistics 或 historical
neighbors。

Student 学的是：

``` text
Raw Row → Rationale + Label
```

所以 TLRD 真正改变的是：

``` text
Label-only supervision
```

变成：

``` text
Rationale + Label supervision
```

population statistics 和 historical cases 的主要作用，是帮助 teacher
构造一个更有依据、更 grounded 的 training target。

也就是说，它们首先是 **supervision construction resources**，而不是
student deployment 时必须重新访问的信息。

------------------------------------------------------------------------

## 4.3 对我们项目的启示

这个和我们现在的关系非常直接。

我们当前是：

``` text
Profile → Vaccination Label
```

最简单的扩展可以先让 teacher 看：

``` text
Profile + Gold Label
```

生成一段 rationale，再让 student 学：

``` text
Profile → Rationale + Label
```

而且第一步不需要马上把 TLRD 的 population statistics 和 historical
neighbors 全部加入。

我们可以先回答一个最基础的问题：

> **Reasoning supervision 本身到底有没有用？**

因此最值得先比较的是：

``` text
Label-only SFT
vs.
Generic Rationale SFT
vs.
HBM-Guided Rationale SFT
```

如果这一阶段已经没有 improvement，那么后面再加入大量复杂 evidence
的意义可能比较有限。

如果 rationale supervision 确实有效，再逐步加入：

``` text
Training-population statistics
```

以及：

``` text
Similar historical respondents
```

这样才能区分 improvement 到底来自 reasoning text、behavioral
theory，还是 dataset-specific evidence。

------------------------------------------------------------------------

# 5. MoRSD

## *Towards Efficient CoT Distillation: Self-Guided Rationale Selector for Better Performance with Fewer Rationales*

**Authors:** Jianzhi Yan, Le Liu, Youcheng Pan, Shiwei Chen, Yang Xiang,
Buzhou Tang\
**Venue:** Findings of EMNLP 2025\
**Method:** Model-Oriented Rationale Selection Distillation (MoRSD)

假设我们已经决定用 teacher-generated rationale 做
SFT，接下来就会自然出现另一个问题：

> **是不是 teacher 生成越多 rationale 就越好？**

作者的答案是否定的。

同一个问题可以生成很多 reasoning，但其中有些可能本身就是错的；有些 final
answer 虽然正确，中间 reasoning 却有问题；还有一些只是高度重复。

更重要的是：

> **一段 reasoning 对一个 student 有帮助，不代表对另一个 student
> 也同样有帮助。**

## 5.1 对应解决方法：筛选真正值得学习的 Rationale

所以 MoRSD 不是研究怎么生成更多 rationale，而是在研究：

> **哪些 rationale 真正值得放进 training data？**

他们先过滤掉错误 reasoning，再去掉高度重复的内容。

然后进一步从 student 的角度判断：

> 给 student 这条 rationale 以后，它预测正确答案是不是变得更容易？

因此 rationale quality 不再只由 teacher
决定，而是加入了一个很重要的标准：

> **它对当前 student 到底有没有实际帮助。**

------------------------------------------------------------------------

## 5.2 对我们项目的启示

我们现在准备让一个更大的 teacher，例如 Qwen3.8-32B，看：

``` text
Profile + Gold Label
```

然后生成 reasoning。

但 teacher 已经知道答案以后，其实很容易写出一段"听起来非常合理"的解释。

所以这里需要区分：

``` text
和 Label 一致
        ≠
有真实 Evidence 支持
        ≠
对 Student 学习有帮助
```

如果后面发现 rationale SFT 确实有效，下一步不应该只是简单生成更多
rationale，而应该开始做 **rationale quality
control，也就是推理质量筛选**。

尤其需要检查：

-   rationale 是否与 label 一致；
-   是否只使用 profile 中真实存在的信息；
-   是否编造未测量的 psychological state；
-   是否高度重复；
-   是否真的帮助当前 Qwen3.5-9B 做 prediction。

因此 MoRSD 更适合作为 rationale SFT
有效以后的下一步，而不是第一轮就加入。

------------------------------------------------------------------------

# 6. FAIR

## *Learning from Committee: Reasoning Distillation from a Mixture of Teachers with Peer-Review*

**Authors:** Zhuochun Li, Yuelyu Ji, Rui Meng, Daqing He\
**Venue:** Findings of ACL 2025\
**Method:** Fault-Aware Distillation via Peer-Review (FAIR)

前面的 reasoning distillation 基本都是以 teacher 为中心：

``` text
Teacher 给一个正确 reasoning
        ↓
Student 去模仿
```

但是 FAIR 进一步问：

> **这个 student 自己到底容易错在哪里？**

如果只是不断给 student 看正确 reasoning，却不分析 student
自己的错误，那么 supervision 并没有真正针对它的 failure mode。

## 6.1 对应解决方法：把 Student Error 变成新的 Supervision

FAIR 会先让 student 自己做题、自己生成 reasoning。

如果 student 做错，这个 case 就不只是一个 wrong label，而会成为一个
**diagnostic case，也就是诊断样本**。

Teacher 会看到：

``` text
Original Question
+ Student's Wrong Rationale
+ Gold Answer
```

然后提供两类信息：

1.  正确 reasoning 应该是什么；
2.  student 刚才具体错在哪里。

所以它构造的是一种 **针对具体错误的 feedback**。

论文还会进一步通过多个 teacher 的 peer review 筛选 supervision，降低
teacher 自己生成 flawed rationale 的风险。

------------------------------------------------------------------------

## 6.2 对我们项目的启示

这个和我们现在已经在做的 error analysis 可以很好地接起来。

我们正在比较：

``` text
SFT wrong / XGBoost correct
```

以及：

``` text
XGBoost wrong / SFT correct
```

目前这些 case 主要还是用于 error analysis。

但 FAIR 提供了进一步的思路：

> **这些 error cases 未来也可以成为新的 supervision source。**

例如，如果 SFT 经常在：

> 医疗接触很多，但是 previous vaccination 很少

这种存在 feature conflict 的 profile 上判断错误，那么可以让 teacher
专门解释：

-   student 为什么在这种冲突信号上容易判断错；
-   不同 evidence 应该怎样整合。

当然，正式训练时不能把 held-out test error 再拿回去训练。

因此真正迁移 FAIR 时，应该在 **training data 或 cross-fitted data**
上识别 student errors，再构造 corrective supervision。

整体思路就是从：

``` text
对所有样本提供比较平均的 supervision
```

进一步变成：

``` text
针对模型薄弱点提供 supervision
```

------------------------------------------------------------------------

# 7. OPCD

## *On-Policy Context Distillation for Language Models*

**Authors:** Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, Furu Wei\
**Source:** arXiv:2602.12275, 2026\
**Method:** On-Policy Context Distillation (OPCD)

这篇更适合作为比较后期的方法。

它研究的问题是：

> **假设我们已经知道某种额外 context
> 对模型确实有帮助，能不能把这种帮助真正学进 student model 的
> parameters？**

例如对我们来说，未来可能有：

``` text
HBM guidance
Population statistics
Similar historical cases
```

当然可以在每次 inference 时都把这些内容放进 prompt。

但这样 knowledge 一直存在于外部 context 里面，并没有真正进入 student
model。

## 7.1 对应解决方法：On-Policy Context Distillation

OPCD 比较特别的地方是 **on-policy**。

不是 teacher 先生成一批固定答案，然后 student 再模仿。

而是 student 先按照自己的模型生成自己的 trajectory。

随后，一个拥有额外 context 的 teacher 会在 student
实际生成到这些位置时告诉它：

> 如果我拥有这些额外信息，这里的行为应该是什么样的？

然后再把这种 context-conditioned behavior distill 给 student。

因此最终 inference 时：

``` text
Original Input
        ↓
Student
        ↓
Output
```

student 不再需要额外 context。

------------------------------------------------------------------------

## 7.2 对我们项目的启示

OPCD 对我们不是第一阶段方案，因为它有一个非常重要的前提：

> **我们必须先证明某一种 context 本身真的有用。**

例如，如果 HBM guidance 本身没有
improvement，那么就没有必要进一步研究怎么把 HBM context internalize 到
student model。

所以 OPCD 更适合放在整个 roadmap 的最后。

------------------------------------------------------------------------

# 8. 把这些文献放在一起

把这六篇文章放在一起以后，最重要的其实不是记住六个方法名字，而是它们刚好对应
fine-tuning pipeline 中不同的问题。

**From Numbers to Narratives** 问的是：

> 模型现在看到的数据表达方式是不是合适？

**TLRD** 问的是：

> 只有 final label 作为 supervision，是不是提供的信息太少？

**MoRSD** 接着问：

> teacher 生成的这些 reasoning，到底哪些真正值得 student 学？

**FAIR** 再进一步问：

> 我们能不能针对 student 自己真正容易犯的错误去设计新的 supervision？

**OPCD** 最后问的是：

> 如果某种额外 context 已经证明有效，能不能进一步把它真正学进 model
> parameters？

而 **Wang & Ge** 那篇更像是提供了一个总体的方法设计逻辑：

> **先找清楚当前任务真正存在的问题，再决定应该改变 fine-tuning
> 的哪一个部分。**

因此整体可以理解为：

``` text
当前任务到底哪里有问题？
        ↓
输入怎么表达？
        ↓
Supervision 怎么丰富？
        ↓
Rationale 怎么筛选？
        ↓
怎么针对 Student Error 构造 Supervision？
        ↓
怎么把已经证明有效的 Context 内化？
```

------------------------------------------------------------------------

# 9. 对我们项目最直接的实验设计

基于这些文献，我觉得现在不应该一下做一个特别复杂的 framework。

第一步最好还是把几个问题拆开。

## 9.1 Input Representation：输入怎么组织？

可以比较：

``` text
Raw Profile → Label
```

和：

``` text
Semantically Grouped Profile → Label
```

这里不改变 target，只看 input representation 有没有影响。

这样可以单独回答：

> **同样的 respondent information，用更适合 language model
> 的方式组织以后，prediction 会不会改善？**

------------------------------------------------------------------------

## 9.2 Supervision Design：到底让 Student 学什么？

这一条是目前更重要的实验。

这里保持 student 看到的 respondent information 完全一样。

### B0 --- Label-only

当前 baseline：

``` text
Profile → Label
```

### B1 --- Generic Rationale

Teacher：

``` text
Profile + Gold Label
        ↓
Generic Evidence-Based Rationale
```

Student：

``` text
Profile
        ↓
Generic Rationale + Label
```

### B2 --- HBM-Guided Rationale

Teacher：

``` text
Profile + Gold Label + HBM Guidance
        ↓
HBM-Guided Rationale
```

Student：

``` text
Profile
        ↓
HBM-Guided Rationale + Label
```

这样可以回答两个比较干净的问题。

第一个：

``` text
Generic Rationale vs. Label-only
```

也就是：

> **加入 reasoning supervision 本身到底有没有价值？**

第二个：

``` text
HBM-Guided vs. Generic Rationale
```

也就是：

> **Behavioral theory 能不能在普通 reasoning 之外进一步提供额外价值？**

这比一开始就把 HBM、historical memory、population statistics、error
correction 全部放进一个 model，更容易解释最后的结果。

------------------------------------------------------------------------

# 10. HBM 在这里应该是什么角色

这里需要特别强调：

我们现在不是想让 teacher 根据 survey features 去"猜"一个人的真实 HBM
psychological state。

因为 NHIS 并没有直接测量完整的：

-   perceived susceptibility；
-   perceived severity；
-   perceived benefits；
-   perceived barriers；
-   self-efficacy；
-   cues to action。

所以 HBM 在第一阶段更适合作为一个：

> **reasoning lens，也就是组织推理的框架。**

当 teacher 看 observed profile 时，HBM 帮助它更有结构地组织已有
evidence。

例如：

-   哪些 observed information 可能和 threat-related signals 有关；
-   哪些和 barriers 或 healthcare access 有关；
-   哪些可以看作 cues to action；
-   哪些 previous preventive behaviors 可以作为支持或冲突的 evidence。

但是，如果 profile 没有测量一个人的 vaccine trust，就不能让 teacher
直接写：

> "这个人不信任疫苗。"

因此 theory 的作用更多是：

> **组织我们真正观察到的
> evidence，而不是创造没有被测量的潜在心理状态。**

------------------------------------------------------------------------

# 11. 最后的整体思路

如果把这次 literature review 压缩成一条路线，大概是：

``` text
Profile → Label
        ↓
改善模型看到什么
Input Representation
        ↓
改善模型学习什么
Rationale + Label
        ↓
Reasoning 应该怎么组织
Generic → HBM-Guided
        ↓
哪些 Rationale 值得学习
MoRSD / Rationale Selection
        ↓
哪些模型错误值得专门监督
FAIR / Error-Aware Training
        ↓
加入更强的 Dataset-Specific Evidence
Population Statistics + Historical Cases
        ↓
如果 Context 已证明有效，再考虑内化
OPCD / Context Distillation
```

所以这次 literature review 最后给我们的核心结论不是：

> "我们找到了一篇 paper，可以直接把它的方法复制过来。"

而是：

> **我们可以把当前比较简单的 label-only vaccination
> SFT，逐步变成一个监督信息更加丰富、推理更加有结构，而且受到真实 survey
> evidence 约束的 fine-tuning 方法。**

而现在最自然的第一步，就是先把：

``` text
Label-only SFT
Generic Rationale SFT
HBM-Guided Rationale SFT
```

这三组真正跑出来。

然后根据这个结果，再决定后面的：

-   rationale selection；
-   针对错误的 supervision；
-   dataset-specific evidence；
-   context distillation；

有没有必要继续做。

------------------------------------------------------------------------

# 12. References

1.  **Wang, Y., & Ge, Y. (2026).** *Align Generative Artificial
    Intelligence with Human Preferences: A Novel Large Language Model
    Fine-Tuning Method for Online Review Management.* arXiv:2604.21209.

2.  **Hayat, A., Tridle, H., & Hasan, M. R. (2026).** *From Numbers to
    Narratives: Efficient Language Model-Based Detection for
    Safety-Critical Minority Classes.* Findings of the Association for
    Computational Linguistics: EACL 2026.

3.  **Liang, T., Tan, X., Shi, L., Zhong, J., Hu, Z., Xie, T., Zuo, Z.,
    Yu, X., & Zhang, X. (2026).** *TLRD: Teaching LLMs to Reason over
    Tabular Data with Tri-Level Rationale Distillation.*
    arXiv:2606.08295.

4.  **Yan, J., Liu, L., Pan, Y., Chen, S., Xiang, Y., & Tang, B.
    (2025).** *Towards Efficient CoT Distillation: Self-Guided Rationale
    Selector for Better Performance with Fewer Rationales.* Findings of
    the Association for Computational Linguistics: EMNLP 2025.

5.  **Li, Z., Ji, Y., Meng, R., & He, D. (2025).** *Learning from
    Committee: Reasoning Distillation from a Mixture of Teachers with
    Peer-Review.* Findings of the Association for Computational
    Linguistics: ACL 2025. Method: **Fault-Aware Distillation via
    Peer-Review (FAIR).**

6.  **Ye, T., Dong, L., Wu, X., Huang, S., & Wei, F. (2026).**
    *On-Policy Context Distillation for Language Models.*
    arXiv:2602.12275.

------------------------------------------------------------------------

## 一句话总结

> **这组文献共同给出了一条从 input representation、到 rationale
> supervision、再到 rationale selection、student-specific error
> correction，最后到 context internalization
> 的渐进路线，使我们可以在不一次堆叠复杂方法的情况下，系统地扩展当前
> label-only vaccination SFT。**
