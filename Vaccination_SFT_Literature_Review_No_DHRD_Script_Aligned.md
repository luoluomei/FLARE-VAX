# Fine-Tuning Strategies for Individual Vaccination Behavior Prediction

## Literature Review and Methodological Implications for the NHIS 2024 Project

This README follows the **same narrative order as the presentation
script**.\
The central question is not simply which fine-tuning algorithm to use,
but:

> **Can we improve individual vaccination behavior prediction by
> changing what the model sees and what it learns during fine-tuning?**

------------------------------------------------------------------------

# 1. Why This Literature Review?

Our current vaccination prediction task is straightforward:

-   **Input:** a respondent profile constructed from 62 observed NHIS
    survey features.
-   **Student model:** Qwen3.5-9B with LoRA fine-tuning.
-   **Output:** `Vaccinated` or `Not Vaccinated`.

The current setup is essentially:

``` text
Profile
   ↓
Qwen3.5-9B + LoRA
   ↓
Vaccination Label
```

or, more simply:

``` text
Profile → Label
```

This is a standard **label-only SFT** setting.

The motivation for this review is that our SFT performance is already
relatively close to conventional machine-learning models. Therefore, the
next question is not simply whether to change the backbone or continue
tuning fine-tuning hyperparameters.

Instead, we ask:

> **Can we change what the model actually learns during fine-tuning?**

Several possibilities follow from this question:

-   change how the input is represented;
-   provide richer supervision than a final label;
-   let a teacher model construct reasoning for the student;
-   structure that reasoning using behavioral theory;
-   select only useful rationales;
-   or construct additional supervision around the student's recurring
    errors.

The six papers below can therefore be understood as modifying different
parts of the same fine-tuning pipeline.

  -----------------------------------------------------------------------
  Paper                               Main question for this review
  ----------------------------------- -----------------------------------
  Wang & Ge                           What specific problem in the
                                      current task should fine-tuning
                                      actually solve?

  From Numbers to Narratives          Is the model seeing the same
                                      information in an appropriate
                                      representation?

  TLRD                                Is the final label alone too weak
                                      as supervision?

  MoRSD                               Which teacher-generated rationales
                                      are actually worth learning?

  FAIR                                Can supervision be targeted to the
                                      student's own errors?

  OPCD                                If external context is useful, can
                                      it eventually be internalized into
                                      model parameters?
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 2. Wang & Ge --- From a Task-Specific Problem to a Task-Specific Fine-Tuning Design

**Paper:** *Align Generative Artificial Intelligence with Human
Preferences: A Novel Large Language Model Fine-Tuning Method for Online
Review Management*\
**Authors:** Yanan Wang and Yong Ge\
**Year:** 2026

## 2.1 Concrete task

The paper studies a very specific management task:

> Given a customer hotel review, generate an appropriate managerial
> response.

The most direct training setup would be:

``` text
Customer Review → Human Manager Response
```

However, the authors first identify a practical problem with this
apparently natural training pair.

## 2.2 Problem 1 --- The input may not contain enough information to support the target

A customer review may say:

> "The room was very noisy and I could not sleep well."

But the actual manager response may additionally say that:

-   the guest was moved to another room;
-   the night manager was contacted;
-   or some other remedial action was taken.

Those facts are present in the **target response**, but not in the
original **customer review**.

Therefore, directly training:

``` text
Review → Response
```

can require the model to reproduce facts that cannot be supported by its
input.

This creates an **input--target information mismatch** and can encourage
unsupported generation.

## 2.3 Solution to Problem 1 --- Context Augmentation

The authors do not begin by changing the loss function.

Instead, they first repair the information mismatch.

They use the original review together with the human response to extract
information that is required by the response but missing from the
review. This becomes additional context.

The training pair changes from:

``` text
Review → Response
```

to:

``` text
Review + Context → Response
```

The model is then fine-tuned using this context-aware input.

### Immediate lesson for our project

The first lesson is therefore not about preference optimization.

It is:

> **Before designing a more complicated fine-tuning method, check
> whether the input contains the information needed to support the
> target.**

For NHIS, this means we should first ask whether our respondent profile
and our proposed training targets are informationally compatible.

------------------------------------------------------------------------

## 2.4 Problem 2 --- Historical human responses are not necessarily explicit preferences

After repairing the information mismatch, the paper introduces a second
problem.

Standard SFT still learns:

> **How managers historically responded.**

But this is not necessarily the same as:

> **What constitutes a better managerial response.**

In other words, historical demonstrations provide examples of behavior,
but they do not directly provide **preferred vs. less-preferred response
pairs**.

## 2.5 Solution to Problem 2 --- Theory-Driven Preference Construction

The authors therefore use domain theory to help construct preference
pairs.

For the same review, the training data contain:

``` text
Preferred Response
vs.
Less-Preferred Response
```

The preference relation is not defined arbitrarily.

For negative reviews, the construction draws on ideas such as:

-   complaint handling;
-   justice theory;
-   rational response cues;
-   emotional response cues.

For positive reviews, the comparison can distinguish more
generic/template-like responses from more tailored responses.

The important point is that **theory participates in defining the
training signal**.

Theory is therefore doing more than appearing as an instruction in a
prompt.

It helps determine:

> which response should be preferred during training.

The model can then undergo preference fine-tuning using these
constructed pairs.

## 2.6 Overall implication for NHIS

The main lesson from Wang & Ge is **not** that we should immediately
apply DPO.

The more important contribution for our project is its method-design
logic:

``` text
Identify a concrete task-specific failure
        ↓
Ask which part of the fine-tuning pipeline causes it
        ↓
Construct supervision that directly addresses that failure
        ↓
Only then choose the fine-tuning mechanism
```

For vaccination prediction, the relevant questions may therefore be:

-   Is the current input representation poorly suited to an LLM?
-   Is label-only supervision too weak?
-   Is teacher-generated reasoning useful?
-   Does reasoning need additional structure?
-   Can HBM help define what constitutes a better rationale?

This last point is particularly relevant to our project.

If **HBM-guided reasoning** proves useful, HBM does not necessarily need
to remain only a sentence in the teacher prompt. It could later help
define:

> **what kind of reasoning is more useful for the student to learn.**

That is a later hypothesis, not something assumed in advance.

------------------------------------------------------------------------

# 3. From Numbers to Narratives --- Change the Input Representation

**Paper:** *From Numbers to Narratives: Efficient Language Model-Based
Detection for Safety-Critical Minority Classes*

## 3.1 Concrete task

This paper studies **tabular classification**, especially
safety-critical settings in which minority cases can be important,
including examples such as machine failure, medical classification, and
at-risk student detection.

One concern is that high overall accuracy can hide poor performance on a
small but important minority class.

For our project, however, the more directly relevant question is about
**input representation**.

## 3.2 Problem --- Raw tabular data are not naturally expressed as language

A tabular row may look like:

``` text
Age = 42
Hypertension = 1
Glucose = 182
```

This format is natural for a tree-based model.

For a pretrained language model, however, the semantic relationship
among the variable name, value, and domain meaning may be less explicit.

## 3.3 Solution --- Convert the same information into a semantic narrative

The paper converts structured feature values into natural-language
descriptions.

Conceptually:

``` text
Raw Tabular Row
        ↓
Semantic Narrative
        ↓
Label
```

For example, the information above could be expressed as a description
such as:

> The individual is 42 years old, has hypertension, and has an elevated
> glucose level.

The purpose is not to allow the LLM to invent additional facts.

The same observed information is simply expressed in a form that may be
more compatible with the pretrained language model.

The paper also performs augmentation for minority-class examples, but
the final learning target remains the class label.

Therefore, this paper changes **how the input is represented**, not
whether the model generates a rationale.

## 3.4 Direct experiment for NHIS

This suggests a clean comparison:

``` text
A0. Raw Feature Serialization → Label
```

versus

``` text
A1. Semantically Grouped Profile → Label
```

For example, our 62 features could be organized into meaningful groups
such as:

-   demographics;
-   health status;
-   healthcare access;
-   healthcare utilization;
-   previous vaccination or preventive behavior.

The critical constraint is:

> **Do not add information that is not present in the original survey
> features.**

This experiment can isolate whether part of the current limitation comes
from **input representation** itself.

------------------------------------------------------------------------

# 4. TLRD --- From Label-Only Supervision to Evidence-Grounded Rationale Supervision

**Paper:** *TLRD: Teaching LLMs to Reason over Tabular Data with
Tri-Level Rationale Distillation*

## 4.1 Concrete task

TLRD studies tabular classification and regression.

It starts from a problem that is especially relevant to our current
project.

## 4.2 Problem --- Generic feature knowledge is not the same as dataset-specific knowledge

An LLM may understand the general meaning of a feature.

For example, it may know what income represents.

But it does not automatically know:

-   whether a particular income value is high or low in the current
    dataset;
-   what outcomes are common among similar observations;
-   or which combinations of features are especially informative in the
    current population.

Therefore:

``` text
Row → Label
```

may provide too little supervision.

The final label tells the model **what the answer is**, but not **why
this sample belongs to that label**.

## 4.3 Solution --- Give the teacher three levels of evidence

TLRD does not simply ask the teacher to generate an unconstrained CoT.

Instead, the teacher receives three types of evidence.

### Level 1 --- Instance-Level Evidence

The teacher receives the current row and the gold target.

Because this happens during training-data construction, the teacher can
use the correct label to identify:

-   features supporting the outcome;
-   features opposing it;
-   and potentially conflicting signals.

### Level 2 --- Dataset-Level Evidence

The teacher also receives statistics from the training population.

This allows the rationale to move beyond statements such as:

> "This value is high."

and instead make dataset-grounded comparisons such as:

> "This value is relatively high compared with the relevant distribution
> in this dataset."

### Level 3 --- Comparison-Level Evidence

The teacher additionally receives similar historical cases.

This allows comparison between the current sample and examples with
different outcomes.

The teacher can therefore reason about:

-   what the current case shares with similar cases;
-   where it differs;
-   and how those differences relate to the target.

## 4.4 Teacher constructs the rationale; student learns from the rationale

The teacher integrates these three evidence sources into a structured
rationale.

Importantly, the student does **not** need to receive the population
statistics or historical neighbors as input.

The student training objective is conceptually:

``` text
Raw Row → Rationale + Label
```

Therefore, TLRD changes the supervision from:

``` text
Label only
```

to:

``` text
Evidence-grounded Rationale + Label
```

The additional dataset statistics and historical cases are mainly used
to help the **teacher construct a better training target**.

## 4.5 Direct implication for NHIS

This is the paper most directly connected to our immediate experiment.

Our current setup is:

``` text
Profile → Vaccination Label
```

A simple first extension is:

``` text
Teacher:
Profile + Gold Label
        ↓
Rationale
```

followed by:

``` text
Student:
Profile → Rationale + Label
```

However, the first experiment does not need to reproduce the full TLRD
pipeline.

Before adding population statistics and historical neighbors, we can
first answer the simpler question:

> **Does rationale supervision itself help?**

This leads to the central three-way comparison:

``` text
B0. Label-only SFT
Profile → Label
```

``` text
B1. Generic Rationale SFT
Profile → Generic Rationale + Label
```

``` text
B2. HBM-Guided Rationale SFT
Profile → HBM-Guided Rationale + Label
```

If rationale supervision itself provides no benefit, immediately adding
more complicated evidence may have limited value.

------------------------------------------------------------------------

# 5. MoRSD --- Which Rationales Are Actually Worth Learning?

**Paper:** *Towards Efficient CoT Distillation: Self-Guided Rationale
Selector for Better Performance with Fewer Rationales*

**Method:** MoRSD

## 5.1 Problem --- More teacher rationales do not necessarily mean better supervision

Once we decide to use teacher-generated rationales, another question
appears:

> **Should every teacher-generated rationale be used for SFT?**

The answer in MoRSD is no.

For the same problem, teacher rationales may be:

-   incorrect;
-   correct in the final answer but flawed in intermediate reasoning;
-   highly redundant;
-   or simply not useful for the particular student model.

A rationale that helps one student does not necessarily help another.

## 5.2 Solution --- Select rationales before distillation

MoRSD therefore focuses not on generating more rationales, but on
selecting the rationales that are actually worth learning.

The process first removes incorrect reasoning and highly redundant
candidates.

It then considers whether giving a rationale to the current student
makes the correct answer easier for that student.

The important shift is:

``` text
Teacher says it
```

does not automatically imply:

``` text
Student should learn it
```

## 5.3 Implication for NHIS

This distinction is important for our planned teacher-generated
vaccination rationales.

A larger teacher may see:

``` text
Profile + Gold Label
```

and produce a very plausible explanation.

But because the teacher already knows the answer, a plausible
explanation can be constructed even when parts of the reasoning are
weakly grounded.

Therefore, we should distinguish:

``` text
Label-consistent explanation
        ≠
Evidence-grounded explanation
        ≠
Student-useful explanation
```

If rationale SFT proves useful, a later stage should therefore introduce
**rationale quality control**, including checks such as:

-   Does the rationale use only observed profile information?
-   Does it invent psychological states that were never measured?
-   Is it redundant with other rationales?
-   Does it actually help the student learn the prediction task?

------------------------------------------------------------------------

# 6. FAIR --- Turn Student Errors into Corrective Supervision

**Paper:** *Learning from Committee: Reasoning Distillation from a
Mixture of Teachers with Peer-Review*

**Method:** FAIR

## 6.1 Problem --- Teacher-centered rationales do not necessarily target student-specific failures

Standard reasoning distillation is often teacher-centered:

``` text
Teacher produces correct reasoning
        ↓
Student imitates it
```

But this does not directly ask:

> **Where does this particular student tend to fail?**

## 6.2 Solution --- Let the student's own mistake become a diagnostic case

FAIR first allows the student to solve the problem and generate its own
reasoning.

When the student is wrong, that failure becomes a diagnostic example.

The teacher can then receive:

``` text
Original Problem
+ Student's Wrong Reasoning
+ Gold Answer
```

and provide two kinds of information:

1.  the correct reasoning;
2.  feedback explaining what was wrong in the student's reasoning.

This changes supervision from generic correct demonstrations toward
**mistake-specific corrective feedback**.

## 6.3 Connection to our current error analysis

This connects naturally to our existing comparison between SFT and
XGBoost.

We are already examining cases such as:

``` text
SFT wrong / XGBoost correct
```

and

``` text
XGBoost wrong / SFT correct
```

At present, these cases are mainly used for error analysis.

FAIR suggests a later possibility:

> **Student errors themselves can become a source for constructing new
> supervision.**

For example, suppose the SFT model repeatedly fails on profiles with
conflicting signals such as:

``` text
High healthcare utilization
+
Low previous vaccination uptake
```

A teacher could be asked to explain:

-   why the student's original interpretation failed;
-   which signals conflict;
-   and how the evidence should be integrated more carefully.

For formal training, this should be constructed from **training data or
cross-fitted predictions**, not from the held-out test set.

The methodological progression is therefore:

``` text
Uniform supervision for all samples
        ↓
Identify recurring student failures
        ↓
Construct targeted corrective supervision
```

------------------------------------------------------------------------

# 7. OPCD --- Internalize Useful External Context

**Paper:** *On-Policy Context Distillation for Language Models*

**Method:** OPCD

## 7.1 Problem --- Useful context can remain permanently external to the student

Suppose later experiments show that additional context improves
vaccination prediction.

Potential examples include:

-   HBM guidance;
-   population statistics;
-   similar historical respondents.

One solution is to provide this context in the prompt every time.

However, in that case the knowledge remains external to the model.

The student still depends on repeated prompting with that context.

## 7.2 Solution --- Distill a context-informed teacher into a context-free student

OPCD asks whether a teacher that has access to additional context can
transfer its behavior to a student that does not receive that context.

A key characteristic is that the process is **on-policy**.

Rather than generating a fixed offline teacher trajectory and asking the
student to imitate it, the student first follows its own trajectory.

The context-informed teacher then provides guidance along the states
actually visited by the student.

Conceptually:

``` text
Student:
Input → Student trajectory
```

while the teacher has:

``` text
Context + Input + Student trajectory
```

and provides context-informed guidance that is distilled into the
student.

The eventual goal is:

``` text
Inference:
Input → Student
```

without repeatedly supplying the external context.

## 7.3 Implication for NHIS

This is a later-stage direction.

It only becomes meaningful after we establish that some external context
is genuinely useful.

For example:

``` text
If HBM guidance does not improve the task
        ↓
There is little reason to study how to internalize HBM context
```

Therefore, OPCD belongs near the end of the roadmap rather than in the
immediate experiment.

------------------------------------------------------------------------

# 8. Putting the Six Papers Together

The important point is not to memorize six method names.

Together, the papers form a sequence of increasingly specific questions
about the fine-tuning pipeline.

### Wang & Ge

> **What is the actual task-specific failure we are trying to fix?**

### From Numbers to Narratives

> **Is the model seeing the data in an appropriate form?**

### TLRD

> **Is the final label too weak as supervision?**

### MoRSD

> **Which teacher-generated rationales are actually worth learning?**

### FAIR

> **Can supervision directly target the student's recurring mistakes?**

### OPCD

> **If external context is useful, can it eventually be internalized
> into the student?**

This produces the following conceptual progression:

``` text
Task-specific failure
        ↓
Input representation
        ↓
Richer rationale supervision
        ↓
Rationale quality / selection
        ↓
Student-error-aware supervision
        ↓
Context internalization
```

Wang & Ge provides the broader design principle underlying the sequence:

> **First identify the real limitation of the current task; then decide
> which part of fine-tuning should change.**

------------------------------------------------------------------------

# 9. The Most Direct Experimental Design for Our Project

Based on these papers, the immediate goal should not be to combine every
idea into one large framework.

The first experiments should isolate different questions.

## 9.1 Experiment Line A --- Input Representation

Keep the prediction target unchanged.

Compare:

``` text
A0. Raw Profile → Label
```

with:

``` text
A1. Semantically Grouped Profile → Label
```

This isolates:

> **Does reorganizing the same observed information into a more semantic
> representation improve LLM prediction?**

No additional information should be introduced.

------------------------------------------------------------------------

## 9.2 Experiment Line B --- Supervision Design

Keep the respondent information available to the student unchanged.

### B0 --- Label-only SFT

``` text
Profile → Label
```

This is the current baseline.

### B1 --- Generic Rationale SFT

Teacher:

``` text
Profile + Gold Label
        ↓
Evidence-based Generic Rationale
```

Student:

``` text
Profile → Generic Rationale + Label
```

### B2 --- HBM-Guided Rationale SFT

Teacher:

``` text
Profile + Gold Label + HBM Guidance
        ↓
HBM-Guided Rationale
```

Student:

``` text
Profile → HBM-Guided Rationale + Label
```

This design answers two relatively clean questions.

### Question 1

``` text
Generic Rationale
vs.
Label-only
```

> **Does reasoning supervision itself add value?**

### Question 2

``` text
HBM-Guided Rationale
vs.
Generic Rationale
```

> **Does behavioral theory add value beyond generic reasoning
> supervision?**

This is easier to interpret than immediately combining HBM, historical
memory, population statistics, rationale filtering, and error correction
in a single model.

------------------------------------------------------------------------

# 10. What Role Should HBM Play?

HBM should not be used to ask the teacher to infer unobserved
psychological states from survey features.

NHIS does not directly measure the complete set of constructs such as:

-   perceived susceptibility;
-   perceived severity;
-   perceived benefits;
-   perceived barriers;
-   self-efficacy;
-   cues to action.

Therefore, in the first stage, HBM is better treated as a **reasoning
lens**.

Its role is to help the teacher organize **observed evidence**.

For example, the teacher may consider:

-   observed signals related to health threat;
-   observed barriers or healthcare access;
-   healthcare contact as possible cues to action;
-   previous preventive or vaccination behavior as supporting or
    conflicting evidence.

However, if vaccine trust is not measured, the rationale should not
claim:

> "This respondent distrusts vaccines."

A suitable principle is:

> **Use HBM to organize observed evidence, not to invent latent
> psychological states that were not measured.**

Conceptually:

``` text
Observed Survey Evidence
        ↓
HBM as a Reasoning Lens
        ↓
Structured, Evidence-Constrained Rationale
```

------------------------------------------------------------------------

# 11. Overall Research Progression

The literature review suggests the following progression.

We begin with:

``` text
Profile → Label
```

### Step 1 --- Improve what the model sees

``` text
Raw Profile
vs.
Semantically Grouped Profile
```

### Step 2 --- Improve what the model learns

``` text
Label
        ↓
Rationale + Label
```

### Step 3 --- Test how reasoning should be organized

``` text
Generic Rationale
vs.
HBM-Guided Rationale
```

### Step 4 --- If rationale supervision works, improve rationale quality

Following the logic of **MoRSD**:

``` text
Generate rationales
        ↓
Filter unsupported / redundant / unhelpful rationales
        ↓
Train student
```

### Step 5 --- Target recurring student errors

Following the logic of **FAIR**:

``` text
Identify recurring student failures
        ↓
Construct corrective teacher supervision
        ↓
Retrain / refine student
```

### Step 6 --- Add stronger dataset-specific evidence if needed

Following the logic of **TLRD**, later extensions may give the teacher:

``` text
Profile
+ Population Statistics
+ Similar Historical Respondents
+ Gold Label
        ↓
Evidence-Grounded Rationale
```

The student can still learn from:

``` text
Profile → Rationale + Label
```

without requiring all teacher-side evidence at inference.

### Step 7 --- Internalize useful external context

If HBM guidance, population statistics, historical cases, or another
context source proves useful, **OPCD** provides a later direction for
studying whether that context can be distilled into the student model
itself.

------------------------------------------------------------------------

# 12. Immediate Next Step

The most natural first experiment remains:

``` text
Label-only SFT
vs.
Generic Rationale SFT
vs.
HBM-Guided Rationale SFT
```

This comparison should come before building a large combined framework.

It directly tests:

1.  whether reasoning supervision helps beyond label-only SFT;
2.  whether HBM-guided reasoning adds value beyond generic reasoning.

Only after these results are available should we decide whether to
proceed toward:

-   rationale selection and quality control;
-   student-error-aware corrective supervision;
-   population statistics and historical cases;
-   or context distillation.

The central methodological idea is therefore not:

> **Make the LLM generate longer reasoning.**

It is:

> **Transform a simple label-only vaccination SFT setup into a
> fine-tuning process with richer, more structured, and
> evidence-constrained supervision.**
