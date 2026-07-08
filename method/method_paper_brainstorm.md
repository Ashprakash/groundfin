# Method Paper Brainstorm

Working title:

**Grounded Probabilistic Templates for Evidence-Conditioned Decision Models**

Target venue:

**ACM KDD 2027** as the ambitious target. Backup targets: ACM Web Conference 2027, IEEE ICDM 2027.

## Core Problem

Small language models are attractive for cost, latency, privacy, and deployment, but they often lack the parametric knowledge and reasoning reliability of larger teacher models. Standard distillation transfers teacher behavior, but it can teach the student to imitate answers rather than learn when and how to use external grounded evidence.

The research problem:

> Can a smaller student model recover teacher-level decision quality on evidence-dependent tasks by learning a probabilistic mapping from question and grounded evidence to answer, confidence, and abstention behavior?

## Central Hypothesis

Grounded probabilistic templates can recover more of the teacher-student performance gap than answer-only distillation or naive RAG, especially on counterfactual and unseen evidence-conditioned tasks where memorized knowledge is unreliable.

Expected result pattern:

- Student without evidence performs poorly.
- Student with retrieval/evidence improves, but remains poorly calibrated.
- Answer-only distillation improves in-distribution accuracy but fails on counterfactual evidence.
- Grounded probabilistic distillation improves grounded accuracy, calibration, abstention, and robustness.

## Key Scientific Claim

The method separates raw evidence interpretation from answer prediction:

```text
P(y | x, E) = sum_z P(y | x, z) P(z | x, E)
```

where:

- `x` is the task/question,
- `E` is raw evidence,
- `z` is the grounded probabilistic template,
- `y` is the answer or decision.

The goal is to teach the student to rely on a repeatable template `z` when the answer depends on grounded external knowledge, and to abstain when `z` indicates missing, stale, or contradictory evidence.

## Proposed Method

Name candidates:

- Grounded Probabilistic Distillation, GPD
- Evidence-Conditioned Distillation, ECD
- Grounded Knowledge Distillation, GKD
- Probabilistic Evidence Distillation, PED

Current favorite:

**Grounded Probabilistic Templates (GP-Template) + Grounded Probabilistic Distillation (GPD)**

Teacher produces a structured supervision bundle:

```json
{
  "answer": "...",
  "answer_distribution": {"A": 0.72, "B": 0.18, "C": 0.07, "abstain": 0.03},
  "confidence": 0.72,
  "evidence_spans": ["..."],
  "evidence_support": "full | partial | none | conflicting",
  "abstain_label": false,
  "rationale_or_computation": "...",
  "temporal_validity": "valid | stale | unknown"
}
```

Student is trained with a multi-objective loss:

```text
L = L_answer + lambda_1 L_distribution + lambda_2 L_evidence
    + lambda_3 L_calibration + lambda_4 L_abstention
```

Possible loss components:

- `L_answer`: supervised answer loss against gold labels.
- `L_distribution`: KL divergence between teacher and student answer distributions.
- `L_evidence`: evidence-selection or citation-support loss.
- `L_calibration`: confidence calibration loss, such as Brier score or ECE-oriented objective.
- `L_abstention`: binary/multiclass loss for answer versus abstain.

## Benchmark Needed For The Method Paper

The method paper needs a benchmark that makes ordinary distillation look insufficient.

Use or adapt:

- FinanceBench as a known existing benchmark.
- FinGKD-Bench as our counterfactual extension.

Required benchmark splits:

1. **Original**
   Normal financial QA with evidence.

2. **Counterfactual numeric**
   Same question template, changed table values.

3. **Counterfactual entity/time**
   Same reasoning pattern, different company or year.

4. **Missing evidence**
   Correct behavior is abstention or uncertainty.

5. **Conflicting evidence**
   Model should detect contradiction or lower confidence.

6. **Stale evidence**
   Model must flag temporal invalidity.

7. **Distractor evidence**
   Similar-looking evidence appears but does not support the answer.

## Baselines

Minimum baselines:

- Student zero-shot / direct answer.
- Student with evidence in context.
- Student with RAG-style evidence.
- Student answer-only distilled from teacher.
- Student chain-of-thought or rationale-distilled from teacher.
- Student calibrated with temperature scaling.
- Teacher model.
- Our GPD method.

Optional stronger baselines:

- Self-consistency.
- Retrieval-augmented fine-tuning.
- DPO-style preference tuning using grounded versus ungrounded answers.
- Verifier-reranker pipeline.

## Models

Teacher candidates:

- GPT-4.1 / GPT-5-class API model if available for data generation.
- Claude/Gemini-class model if allowed as comparison.
- Large open model for reproducibility, such as Llama 3.1/3.3 70B or Qwen 2.5/3 72B.

Student candidates:

- Llama 3.2 1B/3B.
- Qwen 2.5/3 1.5B/3B/7B.
- Phi-class small model.
- Mistral 7B as a stronger student.

For a publishable paper, include at least one fully reproducible open teacher/student setting.

## Evaluation Metrics

Primary:

- Accuracy.
- Grounded accuracy: correct answer with valid supporting evidence.
- Counterfactual robustness: performance on perturbed evidence.
- Calibration: ECE, Brier score, reliability diagrams.
- Abstention F1 / selective risk.
- Hallucination rate: unsupported claims or fabricated numbers.

Secondary:

- Evidence precision/recall.
- Cost and latency.
- Teacher gap recovery:

```text
Gap recovery = (student_method - student_baseline) / (teacher - student_baseline)
```

- Robustness under missing/stale/conflicting evidence.

## Experiments Needed

1. **Known benchmark win**
   Show GPD improves over baselines on FinanceBench-style tasks.

2. **New benchmark diagnosis**
   Show existing methods that score well on original tasks fail on FinGKD counterfactual splits.

3. **Teacher-student gap recovery**
   Quantify how much performance gap GPD recovers.

4. **Calibration study**
   Compare confidence quality across methods.

5. **Evidence sensitivity**
   If evidence changes, does the answer change appropriately?

6. **Ablations**
   Remove distribution supervision, evidence loss, calibration loss, abstention loss.

7. **Scale study**
   Test at least two student sizes if compute allows.

8. **Error analysis**
   Categorize failures: retrieval failure, arithmetic failure, unsupported inference, stale evidence, overconfident wrong answer.

## Paper Contributions

Likely contribution list:

1. A probabilistic framework for distinguishing parametric decisions from grounded evidence-conditioned decisions.
2. Grounded Probabilistic Distillation, a multi-signal teacher-student method for evidence-conditioned decision models.
3. A counterfactual financial decision benchmark that tests whether models use evidence rather than memorized patterns.
4. Empirical evidence that GPD improves accuracy, calibration, abstention, and robustness over answer-only distillation and naive RAG.

## Risks

- If gains are only on finance, KDD may see it as too domain-specific.
- If benchmark construction is weak, reviewers may question whether counterfactuals are realistic.
- If method is just multi-task distillation without a crisp theory, novelty may feel incremental.
- If teacher labels are noisy, student may inherit teacher mistakes.
- If no open model reproduction exists, reviewers may worry about reproducibility.

## Mitigation

- Frame the method generally and use finance as a rigorous high-stakes testbed.
- Include at least one non-finance mini-domain if time allows, such as legal/regulatory QA or scientific tables.
- Open-source benchmark generation scripts and evaluation harness.
- Use human or rule-based verification for a carefully curated test set.
- Include strong ablations to prove which supervision signals matter.

## Immediate Work Needed

1. Finalize method name and mathematical framing.
2. Choose existing benchmark to beat, likely FinanceBench.
3. Define FinGKD-Bench schema and data generation procedure.
4. Build a small pilot dataset, around 100-300 examples.
5. Run baseline evaluations with one teacher and one student.
6. Validate whether counterfactual splits expose failures.
7. Decide whether we need fine-tuning, prompt-only distillation, or both.
8. Draft paper outline and related work map.

## Open Questions

- Should GPD be a fine-tuning method, an inference-time method, or both?
- Can we make the probabilistic theory precise enough to feel like more than engineering?
- What is the cheapest reproducible student/teacher pair that still shows a meaningful gap?
- How much human validation do we need for benchmark credibility?
- Can we create counterfactual financial evidence without introducing unrealistic artifacts?
- Should abstention be a separate class or derived from confidence and evidence support?
