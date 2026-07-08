# Abstract and Adversarial Review

## Working Title

**Risk-Calibrated Evidence Templates for Distilling Small Financial Decision Models**

## Draft Abstract

Small language models are attractive for financial decision support because they are cheaper and easier to deploy than frontier-scale models, but they often fail when decisions depend on long, noisy evidence such as SEC filings, financial tables, and time-sensitive disclosures. Recent structured-trace and RAG-distillation methods improve small-model accuracy by transferring teacher-generated rationales, structured records, or retrieved evidence, but they do not directly optimize the reliability properties required in high-stakes decision settings: evidence support, calibrated confidence, abstention under insufficient evidence, and robustness to counterfactual evidence. We introduce **GROUNDFIN**, a framework for distilling small financial decision models with **risk-calibrated evidence templates**. Each template represents a decision through task variables, evidence units, support probabilities, computation traces, answer distributions, calibrated confidence, and abstention probability. Unlike deterministic structured traces, these templates make uncertainty and evidence sufficiency first-class supervision targets. GROUNDFIN evaluates models under a controlled evidence-condition benchmark with raw evidence, length-matched summaries, deterministic structured traces, probabilistic templates, missing evidence, and counterfactual evidence. Our central hypothesis is that probabilistic-template distillation can match or improve the accuracy of structured-trace distillation while substantially improving abstention, calibration, and counterfactual evidence-following. Initial FinanceBench probes show that a 0.5B instruction model fails on raw gold evidence but succeeds on concise grounded and counterfactual evidence, motivating the claim that small-model failures are partly evidence-representation failures rather than pure capacity failures. The paper’s main contribution is not another financial QA benchmark or compressed prompt, but a reliability-oriented distillation target for teaching small models when to answer, when to abstain, and how strongly evidence supports a financial decision.

## One-Sentence Novelty Claim

GROUNDFIN extends structured-trace and RAG distillation by making evidence support, calibrated confidence, and abstention explicit supervised fields, then testing whether these risk-calibrated templates improve small-model reliability under missing and counterfactual evidence.

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

### LiteCoST / Chain-of-Structured-Thought

LiteCoST is the strongest structured-trace prior art. It uses Chain-of-Structured-Thought templates to produce auditable supervision for SLMs on long-document QA, including financial settings.

Threat:

> Reviewers may say our templates are simply CoST with finance-specific packaging.

Response:

> We must compare against deterministic structured traces and show that first-class probabilistic fields improve calibration, abstention, and counterfactual robustness, not merely answer accuracy.

### DRAG / Evidence and Graph Distillation

DRAG distills RAG behavior into SLMs using ranked evidence and knowledge-graph distillation to reduce hallucination.

Threat:

> Reviewers may say our work is DRAG applied to FinanceBench.

Response:

> We need to show that support probabilities, answer distributions, and abstention supervision add reliability improvements beyond evidence alignment or graph/ranked-evidence distillation.

### Counterfactual Risk Control for RAG

RC-RAG and related counterfactual prompting work use counterfactual retrieval/use perturbations to estimate confidence and improve abstention.

Threat:

> Reviewers may say the risk-control and counterfactual pieces are already known.

Response:

> Our distinct claim must be distillation into small models through risk-calibrated templates, not inference-time counterfactual prompting alone.

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
