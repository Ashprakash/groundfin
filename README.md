# GROUNDFIN

**GROUNDFIN: Grounded Probabilistic Distillation for Evidence-Conditioned Financial Reasoning**

GROUNDFIN is a research project on teaching smaller language models to make financial decisions from grounded external evidence rather than memorized parametric knowledge or answer-only teacher imitation.

## Research Thesis

Existing financial QA and agent benchmarks often measure whether a model gives the right answer. They do not fully test whether the answer is grounded in the evidence available to the model, calibrated to uncertainty, robust to counterfactual evidence, and able to abstain when evidence is missing, stale, or contradictory.

GROUNDFIN studies:

```text
Parametric decision:          P(y | x)
Grounded decision:            P(y | x, e)
Grounded distilled decision:  P(y | x, e, t)
```

where:

- `x` is the question or decision task,
- `e` is the grounded evidence bundle,
- `t` is teacher supervision,
- `y` is the answer or decision.

## Planned Papers

### Method Paper

Working title:

**Grounded Probabilistic Distillation for Evidence-Conditioned Decision Models**

Target direction: ACM KDD / ACM Web Conference / IEEE ICDM.

Core contribution: a method for distilling grounded, calibrated, evidence-sensitive decision behavior from a larger teacher model into a smaller student model.

### Benchmark Paper

Working title:

**FinGKD-Bench: A Counterfactual Benchmark for Grounded Financial Decision Reasoning**

Target direction: ACM ICAIF / IEEE BigData / IEEE ICDM.

Core contribution: a benchmark that tests whether financial models answer from evidence rather than memorized patterns by using counterfactual, missing-evidence, stale-evidence, and distractor-evidence splits.

## Repository Structure

```text
benchmark/   FinanceBench pilot, FinGKD-Bench design, evaluation artifacts
method/      Grounded probabilistic distillation theory and method notes
paper/       Paper outlines, related work, figures, and drafts
```

Experiment roadmap:

[benchmark/experiment_stages.md](benchmark/experiment_stages.md)

Core method schema:

[method/probabilistic_template.md](method/probabilistic_template.md)

## Current Pilot

The recommended Colab entry point is:

```text
benchmark/groundfin_colab_runner.ipynb
```

It is a stable runner notebook. It clones or pulls the latest GitHub repo code, installs dependencies, and imports the Python pilot module. Most future fixes should happen in `.py` files, so Colab users can rerun the pull cell instead of replacing the notebook.

Open in Colab:

[groundfin_colab_runner.ipynb](https://colab.research.google.com/github/Ashprakash/groundfin/blob/main/benchmark/groundfin_colab_runner.ipynb)

The fuller exploratory notebook is:

```text
benchmark/financebench_colab_pilot.ipynb
```

Both notebooks use the same underlying pilot logic in:

```text
benchmark/financebench_pilot.py
```

The pilot loads `PatronusAI/financebench`, inspects the open FinanceBench subset, creates evidence-conditioned prompts, runs optional API baselines, and identifies candidate examples for counterfactual benchmark construction.

## No API Key Baseline

The runner notebook includes an open-model baseline that runs directly in Colab with Hugging Face models. No OpenAI API key is required.

Recommended Colab runtime:

```text
Runtime -> Change runtime type -> T4 GPU
```

Default pilot model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

This is intentionally small so the first baseline runs quickly. After the pipeline works, compare stronger students such as `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-3B-Instruct`.

## Immediate Next Steps

1. Rerun Stage 0 with parsed-answer scoring.
2. Compare question-only versus gold-evidence prompting.
3. Increase to 20 examples if the scoring is stable.
4. Add missing-evidence and counterfactual-evidence variants.
5. Generate teacher supervision bundles.
6. Run first grounded-distillation baseline.
