# Benchmark Workspace

This folder contains the FinanceBench pilot, Colab runner, task-typed evidence-bundle evaluation, and secondary SFT/GRPO experiments.

The current recommended experiment is **Problem-Typed Evidence Bundle Eval** in `groundfin_colab_runner.ipynb` section **7d**. It tests the current paper hypothesis before any training:

> FinanceBench reliability improves when raw evidence is converted into task-typed financial decision variables, not merely shortened into generic summaries.

The staged experiment plan is here:

- `experiment_stages.md`

Pilot result notes are here:

- `results_log.md`

## Current Notebooks And Code

- `groundfin_colab_runner.ipynb`: recommended stable Colab entry point.
- `financebench_colab_pilot.ipynb`: fuller exploratory notebook.
- `financebench_pilot.py`: shared pilot logic imported by the runner.

Purpose:

1. Load `PatronusAI/financebench` from Hugging Face.
2. Inspect the open FinanceBench subset.
3. Compare raw evidence, generic summaries, task-typed bundles, and oracle typed bundles.
4. Identify examples suitable for counterfactual, missing-evidence, stale-evidence, and numeric perturbation splits.
5. Export flattened pilot files for method experiments.

## Colab Workflow

Use `groundfin_colab_runner.ipynb` in Colab. The first cell clones or pulls `Ashprakash/groundfin`, then imports the latest code from `benchmark/financebench_pilot.py`.

When the code changes:

1. Push the updated `.py` file to GitHub.
2. Rerun the notebook's "Pull Latest Project Code" cell.
3. Rerun the experiment cells.

This avoids repeatedly replacing uploaded notebooks.

## Open-Model Baseline

If you do not have an API key, use the notebook section called **Open-Model Baseline**. It loads a small Hugging Face instruct model and runs the same question-only versus gold-evidence comparison.

Recommended runtime:

```text
Runtime -> Change runtime type -> T4 GPU
```

Default model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

## Grounding Probe

The runner notebook includes a **Grounding Probe** section that directly tests the core GROUNDFIN idea:

- Can the model use gold evidence?
- Does compact evidence help more than raw evidence?
- Does it abstain when evidence is missing?
- Can it copy a directly grounded answer?
- Can it follow a directly grounded counterfactual answer?

The probe writes:

```text
probe_summary.csv
probe_results.csv
```

## Template Reliability Comparison

The runner notebook also includes **Template Reliability Comparison**, which tests the method-facing conditions:

- `raw_gold_evidence`
- `length_matched_summary`
- `deterministic_trace`
- `template_no_probabilities`
- `risk_calibrated_template`
- `missing_risk_template`

It writes:

```text
template_summary.csv
template_results.csv
```

## Problem-Typed Evidence Bundle Eval

This is the current decision experiment.

Section **7d** compares:

- `raw_gold_evidence`,
- `generic_summary`,
- `task_typed_bundle`,
- `oracle_typed_bundle`.

Default smoke settings:

```python
PROBLEM_BUNDLE_READERS = ['Qwen/Qwen2.5-0.5B-Instruct']
PROBLEM_BUNDLE_N = 5
```

It writes:

```text
problem_bundle_summary.csv
problem_bundle_results.csv
```

The first pass/fail test is:

```text
task_typed_bundle > generic_summary > raw_gold_evidence
```

Do not spend more time on SFT/GRPO until this condition shows a useful signal.

## Immediate Pilot Goal

Run enough examples to answer:

> Does a finance-specific task-typed evidence interface outperform generic compression and raw evidence for small readers?

## Success Signal

The benchmark direction is promising if:

- `task_typed_bundle` beats `generic_summary`,
- `oracle_typed_bundle` is high enough to prove the reader can use explicit variables,
- numeric/table examples can be perturbed cleanly,
- missing-evidence variants trigger hallucination or inappropriate guessing,
- support/confidence metrics enable selective accuracy beyond full-coverage accuracy.
