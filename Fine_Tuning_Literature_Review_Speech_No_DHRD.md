# Fine-Tuning Literature Review 演讲稿（删除 DHRD 版本）

## 1. 开场：为什么做这次 Literature Review

这周我主要重新看了一下和 fine-tuning 相关的一些文献。

我们现在这个 vaccination prediction 的任务其实比较直接，就是给模型一个
respondent 的 profile，包括我们现在使用的 62 个 survey features，然后让
Qwen3.5-9B 通过 LoRA fine-tuning，最后预测这个人是 vaccinated 还是 not
vaccinated。

所以我们现在做的本质上还是一个比较标准的 **label-only SFT**，也就是：

**Profile → Label。**

但是现在的问题是，我们其实已经发现单纯做 SFT 之后，performance
已经比较接近传统的机器学习模型了。

所以我这次想看的不是简单换一个 backbone，或者继续调整一些 fine-tuning
参数，而是想看：

**有没有可能改变 fine-tuning 过程中模型真正学习的内容？**

比如现在它只学习一个最终
label，那我们能不能给它更丰富的监督信息？或者改变 input
的组织方式？再或者让 teacher model 提供 reasoning，让 student
不只是学习最后的 prediction，而是同时学习一些和预测相关的推理模式。

所以我最后找的这些文章虽然具体任务不完全一样，但是基本可以放到同一条线上理解。

有的文章是在改变 **input
representation，也就是输入怎么表达**；有的是改变 **training
target，也就是模型到底学什么**；有的是在筛选什么样的 reasoning
真正值得学习；还有一些是直接针对 student
自己容易犯的错误去构造新的监督信息。

------------------------------------------------------------------------

## 2. Wang & Ge

第一篇是 Wang 和 Ge 2026 年的：

**"Align Generative Artificial Intelligence with Human Preferences: A
Novel Large Language Model Fine-Tuning Method for Online Review
Management."**

我觉得这篇文章比较值得看的地方，其实不是它最后具体用了什么 preference
optimization，而是它整个方法是怎么一步一步设计出来的。

它研究的是一个很具体的管理任务：

**给定一条酒店评论，让 LLM 自动生成酒店管理者的回复。**

最直接的方法当然就是拿历史数据做：

**Customer Review → Human Manager Response。**

但是作者发现这里有一个很实际的问题。

比如 customer 只说：

"房间很吵，我晚上睡不好。"

但是酒店经理真正的回复可能会说：

"我们已经给客人换了房间，并且联系了 night manager。"

问题就在于，换房间和联系 manager 这些信息根本没有出现在原始 review
里面。

所以如果直接用 **Review → Response** 去做
SFT，相当于我们要求模型根据一个信息不完整的
input，去复现一个包含额外事实的 target。

作者认为这本身就容易造成 hallucination，也就是生成输入中没有依据的信息。

所以他们第一步没有马上去改 loss，而是先做 **Context
Augmentation，也就是补充上下文信息。**

他们利用 review 和真实的 human response，把 response 中需要、但是原始
review 里面没有的信息提取出来，作为额外的 context。

这样训练就从：

**Review → Response**

变成：

**Review + Context → Response。**

然后在这个基础上再做正常的 SFT。

所以我觉得这里第一个比较重要的启示就是：

**在设计 fine-tuning 之前，先检查 input 和 target
之间的信息是不是匹配的。**

但是这篇文章后面还有第二个问题。

即使我们已经把 context 补全了，SFT 学到的其实还是：

**历史上的 manager 是怎么回复的。**

但这并不一定等于：

**什么样的回复才是更好的回复。**

所以作者进一步利用领域理论去构造 **preference pair，也就是偏好对。**

对于同一个 review，他们会有一个更好的 preferred
response，再构造一个相对较差的 less-preferred response。

而且这个 preference 不是随机定义的。

比如对于 negative review，他们会参考 complaint handling、justice
theory，还有理性回应和情感回应这些领域知识。

所以这里 theory 的作用不是简单地：

"我在 prompt 里面告诉模型，请考虑这个理论。"

而是：

**理论直接参与定义什么样的 training signal 是好的，什么样的是不好的。**

然后他们才进一步做 preference fine-tuning。

所以我觉得这篇对我们最大的启示，其实不是说我们马上也去做 DPO。

而是它提供了一个比较好的 **方法设计逻辑**：

先问：

**我们现在的 vaccination SFT 到底哪里不够？**

比如可能是 input 的表达方式不够好，也可能是 label supervision
太弱，也可能是 reasoning 没有足够的结构。

然后再针对这个具体的问题去设计方法。

这个逻辑我觉得对我们后面的 HBM 也比较重要。

因为 HBM 不一定只能作为一句 prompt。

如果后面证明 HBM-guided reasoning
是有价值的，那么理论也可能进一步参与定义：

**什么样的 reasoning 更值得 student 学。**

------------------------------------------------------------------------

## 3. From Numbers to Narratives

第二篇是：

**"From Numbers to Narratives: Efficient Language Model-Based Detection
for Safety-Critical Minority Classes."**

这篇和我们比较接近的一点是，它也是做 **tabular
classification，也就是表格数据分类。**

它关注的是一些比较重要的少数类预测问题，比如 machine failure、medical
classification，或者识别有风险的学生。

作者首先指出一个比较常见的问题：

如果数据类别不平衡，一个模型整体 accuracy
很高，并不代表它真的能够识别那些数量比较少、但是又非常重要的 minority
cases。

但是对我们更有意思的是它提出的另外一个问题：

**原始的 tabular data，其实不是 language model 最熟悉的输入形式。**

比如原始输入可能是：

`Age = 42, Hypertension = 1, Glucose = 182`

对于 tree model 来说，这种输入没有什么问题。

但是对于 pretrained language model
来说，这些数字和变量之间的语义关系并不是特别明确。

所以这篇文章没有先改变 training objective，而是先改变 **input
representation，也就是数据怎么呈现给模型。**

他们把每一行 tabular data 转成一段自然语言描述。

比如刚才这些 feature 可以变成：

"这个人 42 岁，有 hypertension，同时 glucose level 比较高。"

当然它的重点不是让 LLM 自由发挥，而是把原来的 feature information
用更有语义的形式重新表达出来。

之后他们还针对 minority class 做了一些数据增强，最后还是正常做：

**Narrative → Label。**

所以这篇文章其实非常清楚：

它没有改变 reasoning，也没有改变最终的监督目标。

它只是在问：

**同样的信息，用一种更适合 language model
理解的方式表达，会不会让预测更好？**

这个对我们可以直接形成一个比较简单的实验。

我们现在可能是把 62 个 feature 比较机械地 serialize 出来。

那我们可以比较：

**Raw Serialization → Label**

和：

**Semantically Grouped Profile → Label。**

比如把 feature 按照 demographics、health
status、医疗可及性、医疗使用情况、previous vaccination history
这些类别组织起来。

但是不增加任何新的信息。

这样我们就可以单独回答：

**我们现在的问题，有没有一部分其实来自 input representation？**

------------------------------------------------------------------------

## 4. TLRD

第三篇是我觉得和我们现在最直接相关的一篇：

**"TLRD: Teaching LLMs to Reason over Tabular Data with Tri-Level
Rationale Distillation."**

它同样是在做 tabular classification 和 regression。

这篇文章提出的问题是：

LLM 可能知道一个 feature 一般是什么意思，但是它不一定知道这个 feature
在**当前 dataset 里面具体意味着什么。**

比如它知道 income 高低的一般含义。

但是它不知道：

这个 income 在当前 dataset 里面到底算高还是低；

和这个 sample 类似的人通常是什么 outcome；

或者哪些 feature combination 在当前 dataset 中比较重要。

所以作者认为，只做：

**Row → Label**

其实 supervision 太弱了。

因为 label 只告诉模型最后的答案，但是没有告诉它：

**为什么这个 sample 应该属于这个 label。**

所以 TLRD 的做法不是直接让 teacher 随便生成一段 CoT。

它先给 teacher 三层不同的 evidence。

第一层是 **Instance-Level Evidence，也就是当前样本本身的信息。**

这里 teacher 会看到当前 row 加上 ground-truth label。

因为这是在构造 training data 的阶段，所以 teacher
可以知道正确答案，然后分析当前 features 中哪些支持这个
outcome，哪些可能是相反或者冲突的 signal。

第二层是 **Dataset-Level Evidence，也就是整个数据集层面的信息。**

它会给 teacher 当前 training population 的一些统计信息。

所以 teacher 不只是简单说：

"这个 feature 很高。"

而是可以进一步说：

"相对于当前 dataset 中这一类人的分布，这个 value 是比较高的。"

第三层是 **Comparison-Level Evidence，也就是和相似历史样本做比较。**

这里会找和当前 sample 相似的 historical cases，让 teacher 比较当前
sample 和不同 outcome 的 cases 有什么相同或者不同。

最后 teacher 会把这三部分 information 整合成一个比较结构化的 rationale。

但是 student 真正训练的时候，并不需要看到这些 statistics 或者 historical
neighbors。

Student 做的是：

**Raw Row → Rationale + Label。**

所以它真正改变的是：

从原来的 **label-only supervision**

变成：

**rationale + label supervision。**

而 population statistics 和 historical cases，主要是帮助 teacher
构造一个更有依据、更 grounded 的 training target。

这个和我们现在的关系就非常直接。

因为我们完全可以先做一个最简单的版本。

我们现在是：

**Profile → Vaccination Label。**

我们可以先让 teacher 看：

**Profile + Gold Label**

然后生成一段 rationale。

再让 student 学：

**Profile → Rationale + Label。**

而且我觉得第一步不用马上把 TLRD 的 population statistics 和 historical
neighbors 全部加进来。

我们可以先回答一个最基础的问题：

**Reasoning supervision 本身到底有没有用？**

所以这里我觉得最值得先比较的是：

**Label-only SFT**

对比

**Generic Rationale SFT**

再对比

**HBM-Guided Rationale SFT。**

如果这个阶段已经没有 improvement，那后面再加入很多复杂 evidence
的意义可能也比较有限。

------------------------------------------------------------------------

## 5. MoRSD

下一篇是：

**"Towards Efficient CoT Distillation: Self-Guided Rationale Selector
for Better Performance with Fewer Rationales."**

它的方法叫 **MoRSD**。

这篇关注的问题是：

假设我们已经决定用 teacher-generated rationale 做 SFT，那么是不是
teacher 生成越多 rationale 就越好？

作者的答案其实是否定的。

因为同一个问题，teacher 可以生成很多 reasoning。

有些可能本身就是错的；

有些虽然 final answer 对，但是中间 reasoning 有问题；

还有一些其实只是高度重复。

更重要的是：

**一段 reasoning 对一个 student 有帮助，不代表对另外一个 student
也同样有帮助。**

所以这篇文章不是研究怎么生成更多 rationale，而是在研究：

**哪些 rationale 真正值得放进 training data。**

他们首先会过滤掉错误的 reasoning，再去掉高度重复的内容。

然后进一步比较：

**给 student 这条 rationale 之后，它预测正确答案是不是变得更容易。**

所以这里判断 rationale quality 的时候加入了一个很重要的因素：

**它对当前 student 到底有没有实际帮助。**

这个对我们比较重要。

因为我们现在准备让一个更大的 teacher，比如 Qwen3.8-32B，看 **Profile +
Gold Label**，然后生成 reasoning。

但是 teacher
已经知道答案以后，其实很容易写出一段"听起来非常合理"的解释。

所以这里其实要区分三个概念：

**和 Label 一致**

不一定等于

**有真实证据支持**

也不一定等于

**对 Student 学习有帮助。**

所以如果我们后面发现 rationale SFT
确实有效，那么下一步不应该只是简单生成更多 rationale，而是应该开始做
**rationale quality control，也就是推理质量筛选。**

特别是检查它有没有使用 profile
里面不存在的信息，或者有没有把我们没有真正测量的 psychological state
自己推断出来。

------------------------------------------------------------------------

## 6. FAIR

下一篇是：

**"Learning from Committee: Reasoning Distillation from a Mixture of
Teachers with Peer-Review."**

它里面的方法叫 **FAIR**。

这篇文章又进一步问了一个问题。

前面的 reasoning distillation 基本都是以 teacher 为中心：

**Teacher 给一个正确 reasoning，然后 Student 去模仿。**

但是作者认为，这样其实没有真正考虑：

**这个 student 自己到底容易错在哪里？**

所以 FAIR 会先让 student 自己做题、自己生成 reasoning。

如果 student 做错了，这个错误就不只是一个 wrong label，而是可以变成一个
**diagnostic case，也就是诊断样本。**

然后 teacher 会看到：

**原始问题 + Student 的错误 reasoning + Gold Answer**

再告诉 student 两件事情：

第一，正确 reasoning 应该是什么；

第二，**你刚才具体错在哪里。**

所以它实际上构造的是一种 **针对具体错误的 feedback。**

这个和我们现在做的 error analysis 其实可以很好地接起来。

因为我们已经在看：

**SFT wrong / XGBoost correct**

以及反方向：

**XGBoost wrong / SFT correct。**

现在我们主要还是把这些 case 当成 error analysis。

但是 FAIR 给了一个进一步的思路：

**这些 error cases 未来也可以成为新的 supervision source。**

比如如果 SFT 经常在：

**医疗接触很多，但是 previous vaccination 很少**

这种存在 feature conflict 的 profile 上判断错误，

那我们可以让 teacher 专门解释：

为什么 student 在这种冲突信号上容易判断错，以及这些 evidence
应该怎么整合。

当然正式训练的时候不能拿 test error 回去训练，所以应该在 training data
或者 cross-fitted data 上完成。

但是整体思路就是从：

**对所有样本提供比较平均的 supervision**

进一步变成：

**针对模型薄弱点提供 supervision。**

------------------------------------------------------------------------

## 7. OPCD

最后一篇是：

**"On-Policy Context Distillation for Language Models."**

也就是 **OPCD**。

这篇其实更适合作为比较后期的方法。

它研究的问题是：

假设我们已经知道某种额外 context 对模型确实有帮助。

比如对我们来说，未来可能是：

**HBM guidance、population statistics，或者 similar historical cases。**

那我们当然可以每次 inference 都把这些东西放进 prompt。

但是这样这些 knowledge 一直存在于外部 context 里面，并没有真正进入
student model 的 parameters。

所以 OPCD 问的是：

**能不能把一个拥有额外 context 的 teacher 的行为，蒸馏给一个没有这些
context 的 student？**

它比较特别的地方是 **on-policy，也就是训练会跟着 student
自己实际生成的轨迹走。**

不是 teacher 先生成一批固定答案，然后 student 再去模仿。

而是 student 先按照自己的模型生成。

然后拥有额外 context 的 teacher，在 student
实际生成到这些位置的时候告诉它：

如果我拥有这些额外信息，这里的预测应该是什么样的。

最后再把这种行为 distill 给 student。

所以 inference 的时候，student 就不需要再看到这些额外 context。

这个对我们来说，我觉得是一个比较后期的方向。

因为它有一个很重要的前提：

**我们必须先证明某种 context 本身真的有用。**

比如如果 HBM guidance 本身没有
improvement，那就没有必要再进一步研究怎么把 HBM context internalize 到
student model 里面。

所以 OPCD 我会放在整个 roadmap 的最后，而不是现在马上去做。

------------------------------------------------------------------------

## 8. 把这些文献放在一起

所以把这**六篇文章**放在一起以后，我觉得最重要的其实不是记住六个方法名字。

而是它们刚好对应了 fine-tuning pipeline 中不同的问题。

**Numbers to Narratives** 问的是：

模型现在看到的数据表达方式是不是合适？

**TLRD** 问的是：

只有 final label 作为 supervision，是不是提供的信息太少？

**MoRSD** 接着问：

teacher 生成的这些 reasoning，到底哪些真正值得 student 学？

**FAIR** 再进一步问：

我们能不能针对 student 自己真正容易犯的错误去设计新的 supervision？

**OPCD** 最后问的是：

如果某种额外 context 已经证明有效，能不能进一步把它真正学进 model
parameters？

而 Wang 和 Ge 那篇更像是提供了一个总体的方法设计逻辑：

**先找清楚当前任务真正存在的问题，再决定应该改变 fine-tuning
的哪一个部分。**

这样删掉 DHRD 以后，这条逻辑其实会更集中：

**输入怎么表达 → supervision 怎么丰富 → rationale 怎么筛选 →
怎么针对错误构造 supervision → 怎么把有效 context 内化。**

------------------------------------------------------------------------

## 9. 对我们项目最直接的实验设计

所以基于这些文献，我觉得现在其实不应该一下做一个特别复杂的 framework。

第一步最好还是把几个问题拆开。

首先是 **Input Representation，也就是输入怎么组织。**

我们可以比较：

**Raw Profile → Label**

和：

**Semantically Grouped Profile → Label。**

这里不改变 target，只看 input representation 有没有影响。

第二条，也是我觉得现在更重要的一条，是 **Supervision
Design，也就是到底让 student 学什么。**

这里保持 student 看到的 respondent information 完全一样。

第一组是：

**Label-only**

也就是我们现在的 baseline：

**Profile → Label。**

第二组是：

**Generic Rationale。**

Teacher 看：

**Profile + Gold Label**

然后生成一段基于现有 evidence 的 reasoning。

Student 再学习：

**Profile → Generic Rationale + Label。**

第三组是：

**HBM-Guided Rationale。**

Teacher 还是看到 **Profile + Gold Label**，但是额外使用 HBM 作为一个
reasoning lens。

然后 student 学：

**Profile → HBM-Guided Rationale + Label。**

这样我们实际上可以回答两个比较干净的问题。

第一个是：

**Generic Rationale vs. Label-only**

也就是：

**加入 reasoning supervision 本身到底有没有价值？**

第二个是：

**HBM-Guided vs. Generic Rationale**

也就是：

**Behavioral theory 能不能在普通 reasoning 之外进一步提供额外价值？**

我觉得这个比一开始就把 HBM、historical memory、population
statistics、error correction 全部放进一个 model
里面，更容易解释最后的结果。

------------------------------------------------------------------------

## 10. HBM 在这里应该是什么角色

这里还有一点我觉得需要特别强调。

我们现在不是想让 teacher 根据 survey features 去"猜"一个人的真实 HBM
psychological state。

因为 NHIS 并没有直接测量完整的 perceived
susceptibility、severity、benefits、barriers 这些心理变量。

所以 HBM 在第一阶段更适合作为一个 **reasoning
lens，也就是组织推理的框架。**

也就是说：

当 teacher 看 observed profile 的时候，HBM 帮助它更有结构地组织已有
evidence。

比如：

哪些信息可能和 threat-related signals 有关；

哪些和 barriers 或 healthcare access 有关；

哪些可以看作 cues to action；

哪些 previous preventive behaviors 可以作为支持或者冲突的 evidence。

但是如果 profile 没有测量一个人的 vaccine trust，我们就不能让 teacher
直接写：

"这个人不信任疫苗。"

所以这里 theory 的作用更多是：

**组织我们真正观察到的 evidence，而不是创造没有被测量的潜在心理状态。**

------------------------------------------------------------------------

## 11. 最后的整体思路

所以最后如果把这次 literature review 压缩成一条路线，我觉得大概是：

我们现在从：

**Profile → Label**

开始。

第一层可以改善：

**模型看到什么，也就是 Input Representation。**

第二层改善：

**模型学习什么，也就是从 Label 变成 Rationale + Label。**

第三层再问：

**Reasoning 应该怎么组织，也就是 Generic Reasoning 和 HBM-Guided
Reasoning。**

如果这些有效，再进入：

**哪些 Rationale 值得学习**

也就是 MoRSD 这种 rationale selection。

再进一步可以做：

**哪些模型错误值得专门提供额外监督**

也就是 FAIR 这种 error-aware training。

然后才是加入更强的 population statistics、historical cases，以及 OPCD
这种 context distillation 方法。

所以我觉得这次 literature review 最后给我们的核心结论不是：

"我们找到了一篇 paper，可以直接把它的方法复制过来。"

而是：

**我们可以把当前比较简单的 label-only vaccination
SFT，逐步变成一个监督信息更加丰富、推理更加有结构，而且受到真实 survey
evidence 约束的 fine-tuning 方法。**

而现在最自然的第一步，就是先把：

**Label-only SFT、Generic Rationale SFT、HBM-Guided Rationale SFT**

这三组真正跑出来。

然后根据这个结果，再决定后面的 rationale selection、针对错误的
supervision，以及 dataset-specific evidence 有没有必要继续做。
