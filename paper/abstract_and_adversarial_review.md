# Abstract and Adversarial Review

## Working Title

**Grounded Probabilistic Templates for Evidence-Conditioned Financial Decision Models**

## Draft Abstract

Small language models are attractive for financial decision support because they are cheaper, faster, and easier to deploy than frontier-scale models, but they often fail when answers depend on noisy external evidence such as SEC filings, financial tables, and time-sensitive disclosures. Existing retrieval-augmented and distillation methods typically provide raw context or transfer teacher answers, leaving unclear whether the student has learned to decide from evidence or merely imitate surface patterns. We introduce **GROUNDFIN**, a framework for converting raw financial evidence into reusable **grounded probabilistic templates** that decompose a decision into task variables, evidence units, support probabilities, computation traces, answer distributions, calibrated confidence, and abstention probability. Formally, GROUNDFIN factors evidence-conditioned decision-making as `P(y | x, E) = sum_z P(y | x, z) P(z | x, E)`, where `z` is a compact probabilistic evidence template derived from raw evidence `E`. This enables repeatable supervision and distillation from larger teacher models into smaller students without relying on ad hoc prompt engineering. On FinanceBench-style financial QA, our initial probes show that a 0.5B instruction model fails on raw gold evidence but succeeds on concise grounded and counterfactual evidence, suggesting that the bottleneck is evidence usability rather than evidence access alone. We propose benchmark splits for raw evidence, compressed evidence, missing evidence, and counterfactual evidence, and evaluate whether template-conditioned distillation improves accuracy, calibration, abstention, and robustness over raw retrieval, answer-only distillation, and rationale distillation. GROUNDFIN aims to show that structured grounding can close a substantial portion of the teacher-student gap in high-stakes evidence-dependent reasoning.

## One-Sentence Novelty Claim

GROUNDFIN is not another finance QA benchmark or RAG pipeline; it is a repeatable probabilistic template-and-distillation framework for teaching small models to transform noisy financial evidence into calibrated, abstention-aware decisions.

## Closest Prior-Art Threats

### RAG

Retrieval-augmented generation provides external context to a model, but usually does not define a repeatable probabilistic evidence object with support probabilities, answer distributions, computation traces, and abstention targets.

Threat:

> Reviewers may say this is just RAG with better prompting.

Response:

> We must show raw evidence/RAG fails while template-conditioned evidence succeeds, using identical models and examples.

### Knowledge-Augmented Reasoning Distillation

KARD-like work distills rationales from large models using retrieved knowledge for knowledge-intensive tasks.

Threat:

> Reviewers may say this is rationale distillation with financial data.

Response:

> We need ablations showing rationale-only distillation is weaker than probabilistic templates with evidence support, answer distribution, calibration, and abstention.

### Faithful / Context-Faithful RAG

Faithful RAG methods focus on resolving conflicts between parametric knowledge and retrieved context.

Threat:

> Reviewers may say groundedness and context-faithfulness are already studied.

Response:

> Our contribution is focused on small-model distillation through a reusable probabilistic template, not only inference-time conflict handling in large RAG systems.

### FinanceBench / Financial QA Benchmarks

FinanceBench already tests financial QA over evidence.

Threat:

> Reviewers may say the benchmark is not new.

Response:

> We are using FinanceBench as a base testbed, but introduce missing-evidence, compressed-evidence, and counterfactual-evidence splits to test whether models use evidence rather than memorized patterns.

## Adversarial Reviewer Critique

### Concern 1: The method is just prompt engineering.

Likely reviewer comment:

> The proposed templates appear to be hand-designed prompts or compressed rationales. The paper does not establish a principled method beyond prompt formatting.

Strengthening requirement:

- Define a formal schema used across all examples.
- Generate templates through fixed procedures.
- Report ablations: raw evidence, compact evidence, rationale, template without probabilities, full probabilistic template.
- Show the same template structure works across multiple task types.

### Concern 2: The probabilistic part is superficial.

Likely reviewer comment:

> The probabilities in the template seem heuristic and are not validated.

Strengthening requirement:

- Define how each probability is estimated.
- Evaluate calibration with ECE/Brier score.
- Show abstain probability correlates with insufficient-evidence cases.
- Include reliability diagrams or selective-risk curves.

### Concern 3: The benchmark leaks answers.

Likely reviewer comment:

> Using FinanceBench justifications or gold answers to build templates may leak the answer.

Strengthening requirement:

- Separate oracle-template experiments from deployment-like experiments.
- Clearly label oracle templates as upper bound.
- Build teacher/rule templates from raw evidence only for main claims.
- Use held-out examples and manual validation.

### Concern 4: Results may not generalize beyond finance.

Likely reviewer comment:

> The approach may be domain-specific to FinanceBench and financial filings.

Strengthening requirement:

- Frame finance as a high-stakes testbed.
- Add one small non-finance evidence-heavy dataset if feasible.
- Or show multiple financial task families: tables, ratios, textual comparisons, missing evidence, counterfactual numeric evidence.

### Concern 5: Automatic scoring is unreliable.

Likely reviewer comment:

> Weak string/numeric matching is not sufficient for financial QA evaluation.

Strengthening requirement:

- Add human evaluation on a stratified subset.
- Add semantic judge scoring with rubric, but keep human audit.
- Report numeric accuracy separately from text accuracy.
- Include error categories.

### Concern 6: Small model gains may come from shorter context, not templates.

Likely reviewer comment:

> The improvement may simply be caused by shortening the prompt.

Strengthening requirement:

- Compare against length-matched summaries.
- Compare rationale-only and extractive-summary baselines.
- Show probabilistic fields and abstention labels add value beyond compression.

### Concern 7: Distillation may only teach the benchmark.

Likely reviewer comment:

> The student may overfit to FinanceBench formats.

Strengthening requirement:

- Use counterfactual variants.
- Use held-out companies and years.
- Test missing/conflicting/stale evidence.
- Evaluate transfer to unseen metric types.

## Must-Have Experiments For Novelty

1. Raw evidence vs compact evidence vs probabilistic template.
2. Rationale distillation vs probabilistic-template distillation.
3. Original evidence vs counterfactual evidence.
4. Missing evidence and abstention.
5. Student size sweep.
6. Oracle-template upper bound vs teacher-generated template.
7. Manual validation subset.

## Protocol Updates From This Review

The experiment protocol has been strengthened with the following required controls:

1. **No-leakage labels**
   All template results must be labeled as `oracle_template`, `teacher_template`, `rule_template`, or `student_template`. Main claims cannot rely on oracle templates.

2. **Length-matched baselines**
   Template results must beat compact summaries and truncated evidence with similar token budgets.

3. **Rationale baselines**
   Template distillation must be compared against answer-only, rationale-only, and evidence-span distillation.

4. **Probability validation**
   Support probabilities, confidence, and abstention probabilities must be evaluated with calibration and selective-risk metrics.

5. **Human audit**
   A stratified subset must be manually labeled for correctness, unsupported answers, wrong evidence, and wrong computation.

6. **Separate upper-bound and deployment-like claims**
   Oracle-template results can show what is possible, but paper claims about the method must use templates generated from raw evidence without gold answers.

## Strongest Paper Framing

The strongest version of the paper is:

> Evidence access is not enough for small models. We show that financial decision accuracy depends heavily on evidence representation quality, and introduce a reusable probabilistic template for converting noisy evidence into calibrated, distillable decision objects.

## Weakest Paper Framing To Avoid

Avoid:

> We compress FinanceBench evidence and get better answers.

That sounds like prompt engineering.

Use:

> We introduce a repeatable probabilistic evidence representation and show it enables grounded distillation, counterfactual robustness, and abstention in small financial decision models.
