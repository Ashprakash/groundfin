# GROUNDFIN Experiment Protocol

This document defines the staged experiment plan for GROUNDFIN.

The core research question is:

> Can a reusable probabilistic evidence template make smaller financial language models reliably decide from noisy external evidence, instead of relying on parametric memory, raw-context stuffing, or one-off prompt engineering?

## Method Thesis

Raw financial evidence is often long, noisy, table-heavy, time-sensitive, and numerically fragile. Large models can sometimes brute-force this context. Small models usually cannot.

GROUNDFIN tests whether the gap between small and large models is partly an **evidence representation gap**:

```text
small model + raw evidence                         -> weak
small model + compact grounded evidence            -> stronger
small model + probabilistic grounded template      -> repeatable and calibratable
small model + grounded distillation                -> target paper result
```

The method should not look like magic prompting. Every example should be transformed into the same repeatable probabilistic object.

## Probabilistic Template

For each task, raw evidence `E` is mapped into a grounded template `z`:

```text
P(y | x, E) = sum_z P(y | x, z) P(z | x, E)
```

where:

- `x` is the question or decision task,
- `E` is raw evidence,
- `z` is the grounded probabilistic template,
- `y` is the answer or decision.

Template schema:

```json
{
  "task": {
    "question": "...",
    "entity": "...",
    "period": "...",
    "metric": "...",
    "unit": "..."
  },
  "evidence_units": [
    {
      "claim": "...",
      "source": "...",
      "relevance_probability": 0.0,
      "support_probability": 0.0,
      "temporal_validity_probability": 0.0
    }
  ],
  "decision_variables": {
    "variable_name": "value"
  },
  "computation": {
    "formula": "...",
    "inputs": {},
    "result": "..."
  },
  "answer_distribution": {
    "answer": 0.0,
    "INSUFFICIENT_EVIDENCE": 0.0,
    "other": 0.0
  },
  "calibrated_confidence": 0.0,
  "abstain_probability": 0.0
}
```

The repeatability claim is that every condition and benchmark split can be evaluated through this same template family.

## Stage 0: Pipeline Sanity Check

Question:

> Can we load FinanceBench, construct prompts, run open models in Colab, score outputs, and export results?

Current status:

- FinanceBench open subset loads.
- Evidence prompt construction works.
- Open-model baseline runs with `Qwen/Qwen2.5-0.5B-Instruct`.
- Parsed-answer scoring works.
- Grounding probe writes `probe_summary.csv` and `probe_results.csv`.

Success criteria:

- Outputs include `parsed_answer`, `weak_accuracy_answer`, `numeric_accuracy_answer`, `refusal_rate`, and `n`.
- Manual preview agrees enough with automatic scores to use the metrics directionally.

Decision:

- If scoring is noisy, improve parsing or add manual labels.
- If pipeline is stable, move to Stage 1 and Stage 2.

## Stage 1: Raw Evidence Baseline

Question:

> Does raw gold evidence improve a small model compared with question-only prompting?

Conditions:

1. `question_only`
2. `with_gold_evidence`

Models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

Sample sizes:

- Pilot: `n=5`
- Short run: `n=20`
- Stable run: `n=50`
- Full open subset: `n=150`

Metrics:

- `weak_accuracy_answer`
- `numeric_accuracy_answer`
- `refusal_rate`
- manual error notes

Current signal:

```text
Qwen 0.5B, n=5:
question_only        weak_accuracy_answer = 0.20
with_gold_evidence   weak_accuracy_answer = 0.20
```

Interpretation:

- Raw evidence does not currently help the 0.5B model.
- This supports the hypothesis that raw context is not sufficient for small models.

Decision:

- If larger models improve with raw evidence, use them as teacher/reference.
- If small models remain weak, proceed to template/compression stages.

## Stage 2: Compact Evidence Probe

Question:

> Does compact evidence help more than raw evidence?

Conditions:

1. `gold_evidence`: original FinanceBench evidence.
2. `evidence_compressed`: compact FinanceBench justification as a proxy for teacher-compressed evidence.
3. `direct_grounded_evidence`: controlled evidence directly states the gold answer.

Expected pattern:

```text
gold_evidence < evidence_compressed < direct_grounded_evidence
```

Why this matters:

- If compact evidence beats raw evidence, the problem is evidence usability, not only model size.
- If direct evidence is easy but compact evidence is only partially successful, template quality becomes the method bottleneck.

Metrics:

- answer success rate
- numeric accuracy
- refusal rate
- delta from raw evidence

Decision:

- If compressed evidence helps, build the method around template generation.
- If compressed evidence does not help, inspect whether the justification field is too answer-like, too vague, or poorly aligned.

## Stage 3: Missing-Evidence / Abstention Test

Question:

> Does the model abstain when evidence is absent or insufficient?

Conditions:

1. `missing_evidence`
2. `question_only`
3. template with `abstain_probability > threshold`

Expected behavior:

- The model should return `INSUFFICIENT_EVIDENCE` when no valid evidence is present.

Metrics:

- `refusal_rate`
- abstention accuracy
- overconfident wrong rate

Current signal:

```text
Qwen 0.5B, n=3 missing_evidence:
refusal_rate = 0.667
```

Decision:

- If missing-evidence hallucination persists, include abstention probability as a template field and distillation target.

## Stage 4: Counterfactual Evidence Test

Question:

> Does the model follow changed evidence, or does it cling to the original answer pattern?

Conditions:

1. `gold_evidence`
2. `direct_grounded_evidence`
3. `counterfactual_direct_evidence`
4. later: `template_counterfactual`

Current signal:

```text
Qwen 0.5B, n=3:
direct_grounded_evidence           success_rate = 1.000
counterfactual_direct_evidence     success_rate = 1.000
gold_evidence                      success_rate = 0.000
```

Interpretation:

> The model can obey concise grounded evidence, including counterfactual evidence, but fails to extract the correct answer from raw FinanceBench evidence.

This is the first real support for the GROUNDFIN hypothesis.

Decision:

- If counterfactual direct evidence remains easy, focus on template extraction and distillation.
- If counterfactual template evidence fails, improve the probabilistic template schema.

## Stage 5: Probabilistic Template Generation

Question:

> Can we generate the same structured probabilistic template for every example in a repeatable way?

Inputs:

- question
- raw evidence
- gold answer during training/evaluation construction
- justification if available
- deterministic numeric extraction where possible

Template fields to generate:

- task fields: entity, period, metric, unit
- evidence units
- decision variables
- formula/computation if applicable
- answer distribution
- evidence support probability
- abstain probability
- calibrated confidence

Generation levels:

1. **Oracle template**
   Uses gold answer and/or FinanceBench justification. This tests the upper bound.

2. **Rule-assisted template**
   Uses regex/table/numeric extraction plus question parsing.

3. **Teacher template**
   Uses a larger open model or hosted model to produce the template.

4. **Student-predicted template**
   Student learns to produce/use templates.

Decision:

- If oracle templates reach high accuracy, the template form is useful.
- If rule/teacher templates approach oracle templates, the method is repeatable.
- If student-predicted templates work after distillation, we have the method-paper contribution.

## Stage 6: Grounded Probabilistic Distillation

Question:

> Does template-conditioned supervision improve a student model beyond answer-only distillation and raw evidence prompting?

Training variants:

1. `answer_only`
2. `answer_plus_rationale`
3. `answer_plus_compact_evidence`
4. `answer_plus_probabilistic_template`
5. `answer_plus_template_plus_abstention`

Loss components:

```text
L = L_answer
  + lambda_1 L_answer_distribution
  + lambda_2 L_evidence_support
  + lambda_3 L_abstention
  + lambda_4 L_calibration
```

Baselines:

1. Student question-only
2. Student raw evidence
3. Student compact evidence, prompt-only
4. Answer-only distillation
5. Rationale-only distillation
6. Template-conditioned distillation
7. Teacher model

Metrics:

- original accuracy
- counterfactual accuracy
- numeric accuracy
- abstention accuracy
- calibration quality
- hallucination rate
- teacher gap recovery

Teacher gap recovery:

```text
gap_recovery = (student_method - student_baseline) / (teacher - student_baseline)
```

Decision:

- If template-conditioned distillation improves accuracy, counterfactual robustness, and abstention, we have a method-paper result.
- If it only improves accuracy on original examples, reposition as benchmark/application paper.

## Stage 7: Paper-Grade Evaluation

Question:

> Are results stable enough for a recognizable international AI conference?

Target result shape:

| setting | target |
|---|---:|
| small model + raw evidence | low baseline |
| small model + compact evidence | clear improvement |
| small model + probabilistic template | strongest prompt-only result |
| small model + template distillation | target method result |
| large model + raw evidence | reference |
| oracle evidence / template | upper-bound reference |

Ambitious targets:

- 70%+ from raw-to-template pipeline.
- 90%+ in oracle/template-conditioned setting.

Requirements:

- At least two student model sizes.
- At least one reproducible open model setup.
- Clear ablations.
- Manual validation on a subset.
- Released template schema and evaluation harness.

Venue mapping:

- Method-first result: KDD / WebConf / ICDM.
- Benchmark-first result: ICAIF / IEEE BigData / ICDM.

## Immediate Next Step

Run the compressed-evidence grounding probe:

```python
%cd /content
!git -C groundfin pull
%cd /content/groundfin
```

Then rerun **6. Grounding Probe** and paste the printed `=== PROBE SUMMARY ===` block.

The important comparison is:

```text
gold_evidence vs evidence_compressed vs direct_grounded_evidence
```
