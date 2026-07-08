# Benchmark Workspace

This folder contains artifacts for the FinanceBench pilot and the proposed FinGKD-Bench benchmark.

## Current Notebooks

- `groundfin_colab_runner.ipynb`: recommended stable Colab entry point.
- `financebench_colab_pilot.ipynb`: fuller exploratory notebook.
- `financebench_pilot.py`: shared pilot logic imported by the runner.

Purpose:

1. Load `PatronusAI/financebench` from Hugging Face.
2. Inspect the open FinanceBench subset.
3. Compare question-only vs gold-evidence prompting.
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

## Immediate Pilot Goal

Run enough examples to answer:

> Does evidence-conditioned prompting substantially outperform question-only prompting, and do FinanceBench examples contain enough numeric/table structure to build counterfactual grounded-evidence tests?

## Success Signal

The benchmark direction is promising if:

- question-only answers are overconfident or unsupported,
- gold-evidence answers improve but still fail on arithmetic or grounding,
- numeric/table examples can be perturbed cleanly,
- missing-evidence variants trigger hallucination or inappropriate guessing,
- the open 150 examples are enough for a pilot, even if the final paper needs a larger generated/curated benchmark.
