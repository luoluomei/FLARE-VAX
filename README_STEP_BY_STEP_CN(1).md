# NHIS Vaccination — TabICLv2 + TabPFN-3.5 Benchmark（固定 5% Train / 15% Validation / 80% Test）

> 版本：v1，2026-09-23  
> 目标：在现有 NHIS 2024 influenza vaccination 项目上，使用专门面向表格数据的 foundation models 做强 baseline，并与此前 Qwen3.5-9B SFT / XGBoost 结果保持尽可能严格的 respondent-level 可比性。

---

# 0. 这套包到底做什么？

本包一次性完成以下工作：

1. **固定新的 5% / 15% / 80% 划分**，以后不再重复随机切分；
2. **5% train 完全复用之前 low-data p05 的 1,604 个 respondents**；
3. **旧的固定 test_40.csv 完整保留在新的 test_80 中**，因此新的模型仍可以和之前 Qwen / XGBoost 做 exact respondent-level comparison；
4. 跑 **TabICLv2**：
   - pretrained / in-context baseline；
   - 5% train + 15% validation 的 downstream fine-tuning；
5. 跑 **TabPFN-3.5**：
   - pretrained / in-context baseline；
   - 5% train + 15% validation 的 downstream fine-tuning；
6. 两个模型都提供：
   - `raw_numeric` 主实验；
   - `categorical_auto` sensitivity experiment；
7. 输出完整 prediction probabilities、classification metrics、calibration、FP/FN、error patterns、validation permutation sensitivity、high-confidence errors、boundary cases；
8. 自动和已有 5% Qwen / XGBoost、full-60% Qwen / XGBoost 在 **原来的 fixed test40** 上做 case-level paired comparison；
9. 最后生成 CSV + professor-facing Excel comparison workbook。

---

# 1. 为什么这样固定 5% / 15% / 80%？

完整 clean supervised dataset：

```text
Total N = 32,070
Target  = vaccinated
Features = 62
```

此前 full SFT benchmark 已经有：

```text
legacy train_60.csv = 19,242
legacy test_40.csv  = 12,828
```

此前 low-data experiment 又已经有 exact p05 subset：

```text
train_overall_p05.csv = 1,604 respondents
```

因此这里没有重新从零随机抽一个新的 5%。新的 split 被设计成：

```text
Train 5%:
N = 1,604
= EXACT previous lowdata p05 IDs

Validation 15%:
N = 4,810
= fixed stratified sample from the remaining old train60

Test 80%:
N = 25,656
= old fixed test40 (12,828)
+ all untouched remainder of old train60 (12,828)
```

标签分布：

| Split | N | Y=0 | Y=1 | Positive rate |
|---|---:|---:|---:|---:|
| Train 05 | 1,604 | 843 | 761 | 47.4439% |
| Validation 15 | 4,810 | 2,529 | 2,281 | 47.4220% |
| Test 80 | 25,656 | 13,492 | 12,164 | 47.4119% |
| └ Legacy Test40 | 12,828 | 6,746 | 6,082 | 47.4119% |
| └ Extra Holdout40 | 12,828 | 6,746 | 6,082 | 47.4119% |

因此：

```text
Train05 ∩ Validation15 = empty
Train05 ∩ Test80       = empty
Validation15 ∩ Test80  = empty

Train05 ∪ Validation15 ∪ Test80 = all 32,070 respondents
```

### 最重要的可比性设计

旧的：

```text
~/vax_sft/full/data/test_40.csv
```

一个 respondent 都没有被移进 train 或 validation。

它作为：

```text
origin = legacy_test40
```

完整保留在新的 `test_80.csv` 中。

因此我们既可以报告：

```text
新 benchmark:
25,656-row Test80
```

又可以单独切回：

```text
legacy Test40:
12,828 respondents
```

与以前的 Qwen / XGBoost 做完全相同 test respondents 的 paired comparison。

---

# 2. 一个必须提前讲清楚的公平性问题

本包会同时运行 **pretrained ICL** 和 **fine-tuned** 两类模型。

## 2.1 Pretrained TabICLv2 / TabPFN-3.5

默认模型只把：

```text
Train05 X + Y
```

作为 in-context learning context。

它不会使用 Validation15 来更新 foundation-model weights。

因此如果使用固定 `0.5` decision rule，可以将它理解为：

```text
5%-label-only pretrained tabular foundation model baseline
```

这是与以前 5% Qwen / 5% XGBoost 最接近的 supervised-data-budget comparison。

## 2.2 Fine-tuned TabICLv2 / TabPFN-3.5

Fine-tuning 时：

```text
Train05
→ actual parameter updates

Validation15
→ early stopping / best checkpoint selection
```

因此它必须明确标记为：

```text
5% train + 15% validation
```

而不能说它是“纯 5% supervision baseline”。

Validation 中包含 labels，因此即使它不直接做 training loss，也提供了 model-selection information。

## 2.3 Test80

Test80 在模型训练、early stopping 和 threshold selection 中都不会使用。

最终 test 只执行一次 prediction/evaluation。

---

# 3. 输入 feature 和之前保持一致

本包继续使用之前 clean SFT benchmark 的 **62 个 model features**。

排除：

```text
vaccinated      target
SHTFLU12M_A     raw target / direct leakage
HHX             respondent ID
PPSU            survey design ID
PSTRAT          survey design ID
WTFA_A          survey analysis weight
SRVY_YR         constant survey year
```

因此不会为了 TabICL / TabPFN 再偷偷加入额外变量。

完整 feature list 在：

```text
fixed_split/feature_manifest.json
```

---

# 4. 表格数据输入：为什么提供两个 representation？

NHIS 大量变量在 CSV 中保存成 integer / float code：

```text
1
2
3
7
8
9
```

这些数字很多实际上是 survey category code，并不一定是真正连续变量。

为了避免把 representation choice 和模型 performance 混为一谈，本包保留两种输入。

## 4.1 `raw_numeric` — 主实验

所有 62 个 feature 按当前 CSV numeric values 输入。

优点：

```text
与原 XGBoost 使用的 raw tabular representation 最接近
人工处理最少
最适合作为第一张 benchmark table
```

因此建议**先只跑 raw_numeric**。

## 4.2 `categorical_auto` — sensitivity analysis

一个启发式 representation：

```text
train + validation 中
integer-like 且 unique <= 20
→ categorical

AGEP_A
→ 强制 numeric
```

当前结果是：

```text
61 categorical-like features
1 numeric feature: AGEP_A
```

注意：

> 这只是 sensitivity representation，不是声称所有 61 个变量在 NHIS measurement semantics 上都应该被严格当作 nominal categories。

例如某些 healthcare-use count 可能具有 ordinal / count 含义。因此 categorical-auto 不能替代未来基于 NHIS codebook 的 manual feature typing。

具体分类见：

```text
fixed_split/feature_type_manifest.csv
```

---

# 5. TabICLv2 在这里怎么用？

本包使用两种模式。

### A. Pretrained / ICL

```text
Train05 X,Y
↓
TabICLv2 pretrained checkpoint
↓
Test predictions
```

默认不更新 foundation-model parameters。

### B. Fine-tuned TabICLv2

```text
Pretrained TabICLv2
+
Train05
↓
parameter fine-tuning

Validation15
↓
early stopping / best checkpoint

Test80
↓
final evaluation
```

当前 package 固定：

```text
epochs max = 50
learning rate = 1e-5
n_estimators_finetune = 2
n_estimators_validation = 2
n_estimators_inference = 8
early stopping = True
patience = 10
eval metric = ROC-AUC
seed = 42
```

这些设置与 TabICLv2 当前官方 fine-tuning example 保持接近。

---

# 6. TabPFN-3.5 在这里怎么用？

同样跑两种模式。

### A. Pretrained / ICL

```text
Train05 X,Y
↓
TabPFN-3.5 pretrained model
↓
Test predictions
```

### B. Fine-tuned TabPFN-3.5

```text
Train05
→ fine-tune model parameters

Validation15
→ early stopping / checkpoint selection

Test80
→ final held-out evaluation
```

当前 package：

```text
epochs max = 30
learning rate = 1e-5
weight decay = 0.01
n_estimators_finetune = 2
n_estimators_validation = 2
n_estimators_final_inference = 4
activation checkpointing = True
early stopping patience = 8
eval metric = ROC-AUC
seed = 42
```

TabPFN 官方 fine-tuning example 推荐 CUDA GPU，并特别建议 80GB VRAM；ASU Sol 的 A100 80GB 与此匹配。

---

# 7. 非常重要：TabPFN-3.5 权重许可

`tabpfn` Python source package 是开源代码，但 TabPFN-3.5 **model weights 有单独的 gated non-commercial license**。

官方模型页：

```text
https://huggingface.co/Prior-Labs/tabpfn_3_5
```

你需要：

1. 登录 Hugging Face；
2. 打开上面的模型页；
3. 阅读并接受 / request access；
4. 等账号获得 gated repo 权限；
5. 在 Sol 使用自己的 Hugging Face read token 登录。

当前权重许可允许 research / testing / evaluation / internal benchmarking，但禁止 commercial / production use。

**特别需要注意**：官方许可还对使用 TabPFN-3.5 outputs 去训练、微调或蒸馏与其竞争的模型有限制。

因此：

> 如果后续想让 Qwen/LLM “学习 TabPFN 输出的 patterns”，不要默认这一定被许可允许。

最稳妥的实践是：

```text
TabPFN patterns
→ 用于 benchmark / scientific analysis / error comparison

TabICLv2-derived patterns
→ 如果需要进一步作为 LLM method-development signal，会更容易处理许可问题
```

正式把 TabPFN outputs 用作另一个模型的 training/distillation target 前，应重新核对当时最新 license，并根据研究使用方式确认合规性。

---

# 8. 上传到 Sol

先在本地下载 zip：

```text
vax_tabular_fm_05_15_80_v1.zip
```

通过 Sol / Open OnDemand 的 File Browser 上传到：

```text
/home/yhong110/vax_sft/
```

然后 SSH 登录 Sol：

```bash
cd ~/vax_sft
unzip -q vax_tabular_fm_05_15_80_v1.zip
```

建议统一改成固定目录名：

```bash
rm -rf ~/vax_sft/tabular_fm_05_15_80
mv ~/vax_sft/vax_tabular_fm_05_15_80_v1 ~/vax_sft/tabular_fm_05_15_80
cd ~/vax_sft/tabular_fm_05_15_80
```

检查：

```bash
find . -maxdepth 2 -type f | sort
```

应该看到：

```text
README_STEP_BY_STEP_CN.md
requirements-tabicl.txt
requirements-tabpfn.txt

fixed_split/
code/
setup/
slurm/
```

---

# 9. 第一步：先验证你 Sol 上原数据还在

```bash
ls -lh ~/vax_sft/full/data/train_60.csv
ls -lh ~/vax_sft/full/data/test_40.csv
ls -lh ~/vax_sft/lowdata_steps/data/train_overall_p05.csv
```

必须三个都存在。

再看旧结果是否存在：

```bash
ls -lh ~/vax_sft/lowdata_steps/results/natural/p05/sft_test_predictions.csv
ls -lh ~/vax_sft/lowdata_steps/results/compute100/p05/sft_test_predictions.csv
ls -lh ~/vax_sft/lowdata_steps/results/natural/p05/xgb_test_predictions.csv
ls -lh ~/vax_sft/full/results/sft_test_predictions.csv
ls -lh ~/vax_sft/full/results/xgb_test_predictions.csv
```

旧 prediction 文件即使有部分不存在，也不会阻止新模型运行；只会在最后 comparison 阶段跳过对应旧模型。

---

# 10. 第二步：创建 TabICLv2 独立环境

不要修改现在已经工作的：

```text
~/.conda/envs/vax_sft_fla
```

运行：

```bash
cd ~/vax_sft/tabular_fm_05_15_80
bash setup/00_create_tabicl_env.sh
```

它会创建：

```text
vax_tabicl_v2
```

检查：

```bash
$HOME/.conda/envs/vax_tabicl_v2/bin/python - <<'PY'
import torch, tabicl
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("TabICL:", getattr(tabicl,"__version__","unknown"))
PY
```

登录节点上：

```text
cuda_available=False
```

通常没问题，因为登录节点没有分配 GPU。

---

# 11. 第三步：创建 TabPFN-3.5 独立环境

```bash
cd ~/vax_sft/tabular_fm_05_15_80
bash setup/01_create_tabpfn_env.sh
```

它会创建：

```text
vax_tabpfn35
```

检查：

```bash
$HOME/.conda/envs/vax_tabpfn35/bin/python - <<'PY'
import torch, tabpfn
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("TabPFN:", getattr(tabpfn,"__version__","unknown"))
PY
```

---

# 12. 第四步：固定并生成 5/15/80 CSV

这是 CPU 工作，不需要 A100。

推荐直接提交已有 CPU job：

```bash
cd ~/vax_sft/tabular_fm_05_15_80/slurm
J=$(sbatch --parsable 00_prepare_split_cpu.sh)
echo $J
```

查看：

```bash
squeue -u $USER
```

完成后：

```bash
ls -lh ~/vax_sft/tabular_fm_05_15_80/data
```

应该生成：

```text
train_05.csv
validation_15.csv
test_80.csv
split_assignments_05_15_80.csv
feature_manifest.json
feature_type_manifest.csv
split_manifest_05_15_80.json
```

检查行数：

```bash
$HOME/.conda/envs/vax_sft_fla/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
p=Path.home()/"vax_sft/tabular_fm_05_15_80/data"
for f in ["train_05.csv","validation_15.csv","test_80.csv"]:
    d=pd.read_csv(p/f,low_memory=False)
    print(f, len(d), d.vaccinated.value_counts().sort_index().to_dict(), d.vaccinated.mean())
PY
```

预期：

```text
train_05.csv       1604   {0:843, 1:761}
validation_15.csv  4810   {0:2529,1:2281}
test_80.csv       25656   {0:13492,1:12164}
```

如果任何数字不一致，**不要继续跑 GPU**。

---

# 13. 第五步：TabICLv2 checkpoint

通常第一次实例化会自动下载。

可以先在 CPU / login 上触发下载：

```bash
cd ~/vax_sft/tabular_fm_05_15_80
bash setup/02_download_tabicl.sh
```

cache 位置被设为：

```text
/scratch/$USER/vax_tabular_fm_cache/
```

如果 `/scratch` 暂时不可写，可以把脚本中的：

```text
/scratch/$USER/vax_tabular_fm_cache
```

换成：

```text
$HOME/vax_tabular_fm_cache
```

---

# 14. 第六步：获取 TabPFN-3.5 权重

先在浏览器打开：

```text
https://huggingface.co/Prior-Labs/tabpfn_3_5
```

接受 gated license / request access。

然后在 Sol：

```bash
export HF_HOME=/scratch/$USER/vax_tabular_fm_cache/huggingface
mkdir -p "$HF_HOME"
$HOME/.conda/envs/vax_tabpfn35/bin/hf auth login
```

粘贴你自己的 Hugging Face **read token**。

不要把 token 写进任何 SH 文件，也不要上传给我。

然后：

```bash
cd ~/vax_sft/tabular_fm_05_15_80
bash setup/03_download_tabpfn35.sh
```

目标文件：

```text
/scratch/$USER/vax_tabular_fm_cache/tabpfn/tabpfn-v3.5-20260909.safetensors
```

检查：

```bash
ls -lh /scratch/$USER/vax_tabular_fm_cache/tabpfn/
```

---

# 15. 可选：申请 interactive CPU 节点做检查

如果不想在 login node 上做 pandas 检查：

```bash
interactive -p lightwork -q public -t 120 -c 4 --mem=32G
```

进入 compute node 后：

```bash
cd ~/vax_sft/tabular_fm_05_15_80
ls -lh data
```

完成后：

```bash
exit
```

---

# 16. 必做：TabICLv2 A100 smoke test

不要一上来就 full run。

```bash
cd ~/vax_sft/tabular_fm_05_15_80/slurm
J=$(sbatch --parsable 01_tabicl_smoke_a100.sh)
echo $J
```

监控：

```bash
squeue -u $USER
```

任务结束后：

```bash
sacct -j $J --format=JobID,JobName,State,Elapsed,ExitCode
```

日志一般在当前 slurm 目录：

```bash
ls -lt | head
```

smoke test 只会使用：

```text
256 train
256 validation
512 test
max 2 epochs
```

所以目的只是验证：

```text
环境 OK
checkpoint OK
GPU OK
fit OK
fine-tuning OK
predict_proba OK
output pipeline OK
```

不是正式结果。

---

# 17. 必做：TabPFN-3.5 A100 smoke test

```bash
cd ~/vax_sft/tabular_fm_05_15_80/slurm
J=$(sbatch --parsable 04_tabpfn_smoke_a100.sh)
echo $J
```

检查：

```bash
sacct -j $J --format=JobID,JobName,State,Elapsed,ExitCode
```

TabPFN fine-tuning 官方推荐 80GB GPU；这里使用：

```text
A100 80GB
```

如果 smoke 出现 OOM，先不要随意改 train/val/test。优先减少：

```text
n_estimators_finetune
n_estimators_validation
n_estimators_inference
```

而不是改变数据 split。

---

# 18. 正式主实验：raw_numeric

这两个 job 都会分别跑：

```text
pretrained ICL
+
fine-tuned model
```

## TabICLv2

```bash
cd ~/vax_sft/tabular_fm_05_15_80/slurm
J_TABICL=$(sbatch --parsable 02_tabicl_full_raw_a100.sh)
echo "TabICL job = $J_TABICL"
```

## TabPFN-3.5

```bash
J_TABPFN=$(sbatch --parsable 05_tabpfn_full_raw_a100.sh)
echo "TabPFN job = $J_TABPFN"
```

两个可以并行排队。

监控：

```bash
squeue -u $USER
```

状态：

```bash
sacct -j $J_TABICL,$J_TABPFN --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

---

# 19. 或者：一键提交 core pipeline

在确认：

```text
环境已经安装
TabPFN license/access 已处理
模型下载成功
smoke test 成功
```

之后，可以：

```bash
cd ~/vax_sft/tabular_fm_05_15_80/slurm
bash submit_core.sh
```

它会建立 dependency：

```text
prepare split
   ↓
TabICL raw     TabPFN raw
      \         /
       \       /
        compare
```

注意：

> `submit_core.sh` 不会替你接受 TabPFN 的 Hugging Face license，也不会登录 HF。

这些必须提前做完。

---

# 20. Optional：categorical-aware sensitivity

主实验完成后再跑：

```bash
cd ~/vax_sft/tabular_fm_05_15_80/slurm
sbatch 03_tabicl_full_cat_a100.sh
sbatch 06_tabpfn_full_cat_a100.sh
```

不要让这个版本替代 raw numeric 主结果。

推荐报告形式：

```text
Primary:
raw_numeric

Sensitivity:
categorical_auto
```

---

# 21. 每个模型会输出什么？

例如 TabICL：

```text
results/tabicl/raw_numeric/pretrained_icl/
results/tabicl/raw_numeric/finetuned/
```

TabPFN：

```text
results/tabpfn35/raw_numeric/pretrained_icl/
results/tabpfn35/raw_numeric/finetuned/
```

每一个 model/mode 都会输出：

```text
predictions_validation15.csv
predictions_test80.csv

cases_with_features_validation15.csv
cases_with_features_test80.csv

metrics_validation15_threshold_0p5.json
metrics_validation15_threshold_from_validation.json
metrics_test80_threshold_0p5.json
metrics_test80_threshold_from_validation.json

validation_threshold_curve.csv
calibration_bins_validation15.csv
calibration_bins_test80.csv

feature_error_association_validation15.csv
feature_error_association_test80.csv
feature_value_patterns_validation15.csv
feature_value_patterns_test80.csv

permutation_importance_validation.csv

high_confidence_errors_validation15.csv
high_confidence_errors_test80.csv
boundary_cases_validation15.csv
boundary_cases_test80.csv

metrics_test80_by_origin.csv
run_config.json
DECISION_PATTERN_SUMMARY.md
```

fine-tuned model 还会有：

```text
checkpoints/
train_history.csv
```

---

# 22. 指标有哪些？

每个模型统一报告：

```text
Accuracy
Balanced Accuracy
Precision
Recall / Sensitivity
Specificity
F1
ROC-AUC
PR-AUC / Average Precision
Log Loss
Brier Score
MCC
ECE (10 bins)
Predicted Positive Rate
Observed Positive Rate
TN / FP / FN / TP
FPR / FNR
```

不会只看 Accuracy。

---

# 23. Threshold 怎么处理？

每个模型保存两套 classification results。

### A. Fixed 0.5

```text
prob >= 0.5 → vaccinated
```

这个最适合和旧 p05 Qwen / XGBoost 做最直接 comparison。

### B. Validation-selected threshold

只在：

```text
Validation15
```

搜索 threshold，并最大化 balanced accuracy。

然后 threshold 冻结，再应用到 Test80。

绝对不会：

```text
在 Test80 上寻找最好 threshold
```

因此避免 test leakage。

ROC-AUC、PR-AUC、log loss、Brier 等 probability metrics 不依赖这个 class threshold。

---

# 24. “模型为什么做这个决定”会输出到什么程度？

TabICLv2 / TabPFN 不是会自然输出 CoT 的语言模型。

因此本包**不会伪造或声称获取了模型 hidden reasoning**。

我们输出的是可观察、可验证的 decision diagnostics：

## 24.1 Validation permutation sensitivity

```text
permutation_importance_validation.csv
```

对 validation 中某个 feature 打乱，观察：

```text
ROC-AUC drop
Log-loss increase
```

用于回答：

> 这个模型对哪些变量的 predictive structure 最敏感？

这是 model-specific predictive sensitivity，不等于 causal importance。

## 24.2 Error association

```text
feature_error_association_validation15.csv
```

计算各 feature 与模型 error indicator 的 Cramér's V。

用于回答：

> 模型在哪些 feature regions 更容易犯错？

## 24.3 Feature-value patterns

```text
feature_value_patterns_validation15.csv
```

每个 sufficiently large subgroup 会保存：

```text
N
observed vaccination rate
mean predicted probability
predicted positive rate
error rate
FP
FN
calibration gap
```

用于发现例如：

```text
某些 vaccine-history level
某些 healthcare-use pattern
某些 insurance/access category
```

是不是反复造成 overprediction / underprediction。

## 24.4 High-confidence errors

```text
high_confidence_errors_validation15.csv
```

找：

```text
预测错
但 confidence 很高
```

这类 case 很适合后续人工/LLM qualitative review。

## 24.5 Boundary cases

```text
boundary_cases_validation15.csv
```

找接近 validation-selected threshold 的 respondent。

它们更可能体现：

```text
conflicting evidence
uncertain decision boundary
```

---

# 25. 后续让 LLM “参考 pattern”时怎样避免 test leakage？

这是非常重要的一条。

本包同时生成 validation 和 test diagnostics，但：

> **未来设计新 LLM prompt、reasoning trace、error correction 或 SFT rule 时，只能使用 Train/Validation-derived patterns。**

例如允许：

```text
permutation_importance_validation.csv
feature_error_association_validation15.csv
feature_value_patterns_validation15.csv
high_confidence_errors_validation15.csv
```

然后：

```text
冻结 pattern/rule
↓
最终只在 Test80 上 evaluate
```

不能：

```text
看 test error pattern
→ 设计 rule
→ 再回到同一个 test80 报 improvement
```

`*_test80.csv` 只用于**最终报告和事后解释**。

另外，TabPFN-3.5 的模型许可对 outputs 用于训练/蒸馏其他竞争模型有限制；因此如果 pattern 最后要真正进入 LLM training，优先考虑 TabICL / 自己的数据统计产生的 pattern，并对 TabPFN output 使用重新核对 license。

---

# 26. 正式比较结果

主 jobs 完成后：

```bash
cd ~/vax_sft/tabular_fm_05_15_80/slurm
sbatch 07_compare_cpu.sh
```

或者如果 `submit_core.sh` 已经跑了，comparison 会自动等待两个 raw jobs 成功后执行。

输出目录：

```text
~/vax_sft/tabular_fm_05_15_80/results/comparison/
```

核心文件：

```text
metrics_new_models_test80_fixed_0p5.csv
metrics_new_models_test80_validation_threshold.csv

case_predictions_new_models_test80.csv
pairwise_mcnemar_new_models_test80_fixed_0p5.csv
pairwise_mcnemar_new_models_test80_validation_threshold.csv

metrics_legacy_test40_strict_default_decision.csv
metrics_legacy_test40_with_validation_threshold.csv
pairwise_mcnemar_legacy_test40_strict_default_decision.csv
pairwise_mcnemar_legacy_test40_with_validation_threshold.csv
case_predictions_legacy_test40_all_available_models.csv

tabular_foundation_model_comparison.xlsx
```

---

# 27. 最值得先看的两张表

## A. 新 Test80

```text
metrics_new_models_test80_fixed_0p5.csv
```

回答：

> 在完全新固定 80% held-out test 上，TabICLv2 / TabPFN-3.5 的性能是多少？

## B. Legacy Test40 strict comparison

```text
metrics_legacy_test40_strict_default_decision.csv
```

回答：

> 在以前 Qwen3.5/XGBoost 已经预测过的完全相同 12,828 respondents 上，新 tabular foundation models 表现怎么样？

这里对新的 pretrained ICL 模型采用 fixed 0.5 decision，避免把 Validation15 threshold information 混入“纯 5% context”比较。

Fine-tuned models 本身已经用 Validation15 early stopping，因此它们无论如何必须单独注明：

```text
5% train + 15% validation
```

---

# 28. 推荐给老师的最终表格结构

建议不要把所有 setting 混成一个 leaderboard。

### Table A — 5% training-context / low-data benchmark on legacy Test40

```text
XGBoost p05
Qwen3.5-9B SFT p05 Natural
Qwen3.5-9B SFT p05 Compute100
TabICLv2 pretrained ICL (Train05 context)
TabPFN-3.5 pretrained ICL (Train05 context)
```

### Table B — 5% train + 15% validation adapted tabular foundation models on Test80

```text
TabICLv2 fine-tuned
TabPFN-3.5 fine-tuned
```

### Table C — Full-supervision references

```text
Qwen3.5-9B SFT 60%
XGBoost 60%
empirical ceiling diagnostic ~78–79%
```

这样不会把不同 supervision budget 偷偷当作完全公平比较。

---

# 29. 如果 job 失败，先看哪里？

## CUDA / GPU

```bash
cat slurm-<JOBID>.out
cat slurm-<JOBID>.err
```

如果提示 GPU 不存在：

```bash
sacct -j JOBID --format=JobID,Partition,State,Elapsed,ExitCode
```

## TabPFN unauthorized / gated repo

如果看到：

```text
401
403
GatedRepoError
```

重新：

```bash
export HF_HOME=/scratch/$USER/vax_tabular_fm_cache/huggingface
$HOME/.conda/envs/vax_tabpfn35/bin/hf auth login
```

并确认浏览器端已经被批准访问：

```text
Prior-Labs/tabpfn_3_5
```

## OOM

先降低 estimator counts，而不是改 split。

TabPFN 优先：

```text
n_estimators_finetune: 2 → 1
n_estimators_validation: 2 → 1
n_estimators_inference: 4 → 2
```

TabICL 优先：

```text
n_estimators_inference: 8 → 4
predict batch: 2048 → 1024 / 512
```

## Scratch 不可用

将：

```text
/scratch/$USER/vax_tabular_fm_cache
```

替换为：

```text
$HOME/vax_tabular_fm_cache
```

---

# 30. 当前 package 中已经固定的复现信息

```text
seed = 42
train05 = exact previous p05 IDs
validation15 = fixed IDs bundled in split_assignments_05_15_80.csv
test80 = fixed IDs
feature_count = 62
legacy test40 preserved exactly
```

Split assignment 文件已经直接打包，因此不同机器 / 不同时间运行：

```text
不会重新抽样 validation15
```

数据准备脚本只负责把当前 Sol CSV 按固定 HHX assignment 重建出来，并检查：

```text
row count
unique HHX
target consistency
overlap
full coverage
legacy test preservation
```

---

# 31. 推荐实际执行顺序

最稳的顺序是：

```text
1. 上传 zip
2. 解压 / rename
3. 检查旧 CSV
4. create TabICL env
5. create TabPFN env
6. prepare fixed split
7. download TabICL
8. accept + download TabPFN-3.5
9. TabICL smoke
10. TabPFN smoke
11. TabICL raw full
12. TabPFN raw full
13. compare
14. 下载 results/comparison + DECISION_PATTERN_SUMMARY.md
15. 我们一起分析
16. 最后再决定是否跑 categorical-auto
```

不要一开始同时跑所有 sensitivity experiments。

---

# 32. 最后希望得到的研究问题答案

这套 experiment 不只是问：

```text
谁的 Accuracy 最高？
```

而是同时问：

1. 只有 1,604 个 labeled respondents 时，专门的 tabular foundation model 能达到什么 performance？
2. 对同一个 5% context，TabICLv2 / TabPFN-3.5 与 Qwen SFT / XGBoost 谁更 data-efficient？
3. fine-tuning 后能否进一步接近此前约 78–79% 的 empirical information-saturation diagnostic？
4. 不同 model families 的 errors 是否高度重合？
5. 哪些 variables / subgroups 反复成为共同错误来源？
6. Tabular FM 能正确而 Qwen 错误的 cases 有什么 profile pattern？
7. Qwen 能正确而 Tabular FM 错误的 cases 又有什么 pattern？
8. 这些 patterns 是否提示：LLM 后续应该学到的是 structural tabular interaction，而不是增加更多 generic reasoning text？

这才是这个 benchmark 对后续方法设计最大的价值。

