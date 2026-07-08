# GROUNDFIN Experiment Stages

This document defines the staged experiment plan for GROUNDFIN. The goal is to make each stage produce a clear decision about whether to continue, change model size, improve scoring, or revise the benchmark design.

## Stage 0: Pipeline Sanity Check

Question:

> Can we load FinanceBench, construct evidence prompts, run an open model in Colab, score outputs, and export results?

Current status:

- FinanceBench open subset loads.
- Evidence prompt construction works.
- Open-model baseline runs with `Qwen/Qwen2.5-0.5B-Instruct`.
- Initial weak scoring ran on 5 examples.
- Parsed-answer scoring has been added and needs rerun.

Success criteria:

- `hf_summary.csv` and `hf_results.csv` are produced.
- Results include `weak_accuracy_answer`, `numeric_accuracy_answer`, `refusal_rate`, and `n`.
- The preview includes `parsed_answer` for manual inspection.

Decision:

- If the pipeline is stable, move to Stage 1.
- If parsing/scoring is noisy, improve scoring before increasing sample size.

## Stage 1: Grounding Diagnostic

Question:

> Does providing gold evidence improve a small model's financial QA performance compared with question-only prompting?

Conditions:

1. `question_only`
2. `with_gold_evidence`

Models:

- Start: `Qwen/Qwen2.5-0.5B-Instruct`
- Next: `Qwen/Qwen2.5-1.5B-Instruct`
- Stretch: `Qwen/Qwen2.5-3B-Instruct`

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

Expected patterns:

- Very small model may not benefit from long evidence.
- Larger small models should improve with gold evidence.
- If evidence does not help any model, prompt formatting or context length may be the issue.

Decision:

- If `with_gold_evidence > question_only`, proceed to counterfactual tests.
- If equal or worse, test a larger model and shorter evidence windows.
- If scoring disagrees with manual inspection, improve scoring.

## Stage 2: Missing-Evidence / Abstention Test

Question:

> Does the model abstain when evidence is absent or insufficient?

Conditions:

1. `with_gold_evidence`
2. `missing_evidence`
3. `question_only`

Expected behavior:

- `with_gold_evidence`: answer if supported.
- `missing_evidence`: say `INSUFFICIENT_EVIDENCE`.
- `question_only`: ideally abstain for evidence-dependent questions.

Metrics:

- `refusal_rate`
- `overconfident_wrong_rate`
- answer accuracy where answer is allowed
- abstention accuracy where abstention is required

Decision:

- If models hallucinate in missing-evidence settings, this motivates the abstention component of grounded probabilistic distillation.

## Stage 3: Counterfactual Evidence Test

Question:

> Does the model follow changed evidence, or does it cling to the original answer pattern?

Conditions:

1. `original_evidence`
2. `numeric_counterfactual_evidence`
3. `distractor_evidence`

Example:

```text
Original evidence:
Purchases of property, plant and equipment (PP&E): (1,577)
Gold answer: $1,577 million

Counterfactual evidence:
Purchases of property, plant and equipment (PP&E): (2,143)
Gold answer: $2,143 million
```

Metrics:

- counterfactual accuracy
- original-answer cling rate
- evidence-following rate
- numeric accuracy

Decision:

- If models fail counterfactuals after passing originals, this becomes the benchmark paper's core empirical claim.

## Stage 4: Teacher Supervision Bundle Generation

Question:

> Can a teacher produce useful structured supervision beyond answer labels?

Teacher outputs:

```json
{
  "answer": "...",
  "confidence": 0.0,
  "evidence_support": "full | partial | none | conflicting",
  "evidence_spans": ["..."],
  "abstain_label": false,
  "temporal_validity": "valid | stale | unknown",
  "short_rationale": "..."
}
```

Possible teachers:

- Larger open model in Colab if feasible.
- Hosted open model endpoint if available.
- Manual/rule-based supervision for numeric counterfactuals.

Decision:

- If teacher supervision is reliable, proceed to distillation.
- If teacher labels are noisy, use deterministic/rule-based labels for numeric examples first.

## Stage 5: Grounded Probabilistic Distillation

Question:

> Does grounded probabilistic supervision improve a student model beyond answer-only distillation and evidence-only prompting?

Baselines:

1. Student question-only
2. Student with gold evidence
3. Answer-only distillation
4. Rationale-only distillation
5. Grounded probabilistic distillation
6. Teacher

Metrics:

- original accuracy
- counterfactual accuracy
- calibration / confidence quality
- abstention quality
- hallucination rate
- teacher gap recovery

Decision:

- If GPD improves counterfactual accuracy and abstention, we have a method-paper result.
- If it only improves original accuracy, reposition as benchmark/application paper.

## Stage 6: Paper-Grade Evaluation

Question:

> Are results stable enough for a recognizable international AI conference?

Requirements:

- At least two student sizes or model families.
- At least one reproducible open model setup.
- Strong ablations.
- Manual validation on a subset.
- Clear benchmark schema and released artifacts.

Decision:

- Method-first result: target KDD / WebConf / ICDM.
- Benchmark-first result: target ICAIF / IEEE BigData / ICDM.

## Current Next Step

Rerun Stage 0 with parsed-answer scoring:

```python
%cd /content
!git -C groundfin pull
%cd /content/groundfin
```

Then rerun **5. Open-Model Baseline** and paste the printed `=== HF SUMMARY ===` block.
