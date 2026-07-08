# Probabilistic Evidence Template

This document defines the repeatable template at the center of GROUNDFIN.

## Motivation

The method should not depend on ad hoc prompting. Every example should be mapped into a common probabilistic object that separates:

1. raw evidence interpretation,
2. decision-variable extraction,
3. answer generation,
4. confidence calibration,
5. abstention.

The core factorization is:

```text
P(y | x, E) = sum_z P(y | x, z) P(z | x, E)
```

where:

- `x` is the question or task,
- `E` is raw evidence,
- `z` is the probabilistic evidence template,
- `y` is the answer or decision.

The teacher estimates or supervises `P(z | x, E)`. The student learns `P(y | x, z)`.

## Template Schema

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

## Template Levels

### Oracle Template

Uses gold answer and/or FinanceBench justification. This tests the upper bound for a compact grounded representation.

Oracle templates are not allowed for the main method claim. They are used only for upper-bound analysis, debugging, and estimating how much accuracy is available when evidence representation is near-perfect.

### Rule-Assisted Template

Uses deterministic parsing, numeric extraction, entity/period matching, and simple formula rules.

### Teacher Template

Uses a larger model to generate the template from raw evidence. This is the primary distillation source.

Teacher templates must be generated from the question and raw evidence only. If the teacher sees the gold answer, the template must be labeled as oracle.

### Student Template

The student learns to consume or produce the template. This is the end method target.

## Evaluation Questions

1. Does the template improve small-model accuracy over raw evidence?
2. Does the template improve counterfactual evidence following?
3. Does the template improve abstention when evidence is missing?
4. Does the template improve calibration?
5. Can the template be generated repeatably without gold leakage?

## Required Controls

To show that the template is not just prompt engineering or summarization, evaluate against:

1. raw evidence,
2. truncated raw evidence with matched token budget,
3. extractive summary with matched token budget,
4. rationale-only compact evidence,
5. template without probabilities,
6. full probabilistic template.

The full template should improve not only answer accuracy, but also abstention and calibration.

## Paper Claim

The intended claim is:

> A reusable probabilistic evidence template can close much of the teacher-student gap in evidence-dependent financial reasoning by converting noisy external documents into compact, calibrated, decision-relevant representations.
