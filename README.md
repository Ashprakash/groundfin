# GEP

**Grounded Execution Precision (GEP)** is a research project on reliable financial question answering from grounded evidence.

The repository is still named `groundfin` for continuity with the existing GitHub/Colab workflow, but the research framing has moved to **GEP**.

The project started as grounded probabilistic distillation, but early experiments changed the thesis. The current, measurement-first direction is:

> Small financial models fail mainly because they cannot extract the decision variables from long, noisy filings. They can often answer when the relevant financial variables are exposed explicitly. The central question is whether task-typed evidence bundles beat generic summaries and raw evidence before any training.

## Current Hypothesis

**Grounded Execution Precision for Financial Decision Models**

GEP treats financial QA as an auditable execution problem rather than an end-to-end RAG generation problem. Correctness often depends on:

- exact line item,
- year or quarter,
- unit,
- sign convention,
- numerator and denominator,
- formula,
- cash-flow category,
- whether evidence is insufficient.

The current pre-training hypothesis is:

```text
task_typed_bundle > generic_summary > raw_gold_evidence
```

where `task_typed_bundle` is a no-leakage structured evidence object built from raw evidence only. It exposes the financial task type, required variables, candidate values, units, periods, answer rule, and support probability.

The hypothesis is intentionally falsifiable. If `task_typed_bundle` does not beat `generic_summary`, this direction is not strong enough and we should not spend more time on SFT/GRPO.

The stronger GEP method adds a deterministic execution boundary:

```text
raw evidence + question
  -> typed variable/formula extraction
  -> deterministic execution
  -> verification/confidence signal
  -> answer or abstain
```

## What We Already Learned

Early Colab experiments on `PatronusAI/financebench` with `Qwen/Qwen2.5-0.5B-Instruct` showed:

- Raw FinanceBench gold evidence did not improve parsed answer accuracy on the tiny baseline sample.
- The same model performed much better on compact/direct/counterfactual evidence.
- Generic probabilistic templates did not reliably improve deployment accuracy.
- Naive SFT with abstention overfit to refusal and hurt supported-question accuracy.

This killed the weak claim:

```text
generic probabilistic templates improve FinanceBench
```

The surviving signal is:

```text
small models may be usable readers, but poor extractors
```

The next experiment tests whether a finance-specific evidence interface and execution boundary are the missing pieces.

## Main Experiment: Section 7d

The recommended next run is notebook section:

```text
7d. Problem-Typed Evidence Bundle Eval
```

It compares four conditions:

| Condition | Meaning |
|---|---|
| `raw_gold_evidence` | The original FinanceBench evidence text. |
| `generic_summary` | Top evidence sentences by question overlap. |
| `task_typed_bundle` | No-leakage finance-specific bundle built from raw evidence only. |
| `oracle_typed_bundle` | Upper bound that intentionally includes the gold answer. Not a method claim. |

The key pass/fail pattern:

```text
task_typed_bundle accuracy > generic_summary accuracy > raw_gold_evidence accuracy
```

Ideal stronger signal:

```text
task_typed_bundle closes 40-60% of the oracle_typed_bundle gap
```

## Task Types

The current task taxonomy lives in `benchmark/financebench_pilot.py` as `TASK_SCHEMAS`.

Initial task types:

- `cash_flow_line_item`
- `ratio_calculation`
- `period_comparison`
- `cash_flow_category_selection`
- `guidance_delta`
- `line_item_lookup`
- `generic_financial_qa`

Each schema defines required variables and an answer rule. Example:

```json
{
  "task_type": "cash_flow_line_item",
  "required_variables": ["line_item_label", "period", "value", "unit"],
  "answer_rule": "For capital expenditure / capex, use purchases of property, plant and equipment (PP&E); report magnitude unless the question asks for sign."
}
```

## Repository Structure

```text
benchmark/   FinanceBench pilot, Colab runner, evaluation code, result logs
method/      Method notes and theory sketches
paper/       Abstracts, related work, adversarial review notes, paper drafts
```

Important files:

```text
benchmark/groundfin_colab_runner.ipynb   Recommended Colab entry point
benchmark/financebench_pilot.py          Shared evaluation logic
benchmark/train_groundfin.py             SFT/GRPO experiments, currently secondary
benchmark/results_log.md                 Running notes from pilot experiments
benchmark/experiment_stages.md           Older staged plan and controls
```

## Reproducing On Colab

Open:

[groundfin_colab_runner.ipynb](https://colab.research.google.com/github/Ashprakash/groundfin/blob/main/benchmark/groundfin_colab_runner.ipynb)

Use:

```text
Runtime -> Change runtime type -> T4 GPU
```

First cell:

```python
%cd /content
!test -d groundfin/.git && git -C groundfin fetch --all && git -C groundfin reset --hard origin/main || git clone https://github.com/Ashprakash/groundfin.git
%cd /content/groundfin
!pip -q install -r requirements-colab.txt
!git log --oneline -1
```

Then run:

1. **Load FinanceBench**
2. **7d. Problem-Typed Evidence Bundle Eval**

Default smoke settings:

```python
PROBLEM_BUNDLE_READERS = ['Qwen/Qwen2.5-0.5B-Instruct']
PROBLEM_BUNDLE_N = 5
```

The section writes:

```text
problem_bundle_summary.csv
problem_bundle_results.csv
```

Paste or save the printed table:

```text
=== PROBLEM-TYPED BUNDLE SUMMARY ===
```

## Scaling Plan

Do not run SFT/GRPO until Section 7d gives a useful signal.

Run in this order:

| Stage | Reader models | n | Purpose |
|---|---|---:|---|
| Smoke | Qwen 0.5B | 5 | Check code and table shape. |
| Pilot | Qwen 0.5B | 30 | Check whether task bundles beat summaries. |
| Capacity | Qwen 0.5B, Qwen 1.5B, Gemma small | 30-50 | Find reader capacity threshold. |
| Stability | 2-3 readers, 2-3 seeds | 100+ | Paper-grade trend check. |

Candidate readers:

```text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
google/gemma-2-2b-it or a comparable small Gemma instruction model
```

Use Qwen first for continuity, then add Gemma to show the effect is not model-family-specific.

## Interpreting Results

Promising:

```text
task_typed_bundle > generic_summary > raw_gold_evidence
oracle_typed_bundle is much higher than task_typed_bundle
```

This means:

- the reader can use finance-specific variables,
- generic compression is insufficient,
- extraction quality is the next bottleneck.

Not promising:

```text
task_typed_bundle <= generic_summary
```

This means the current bundle schema is not doing useful work yet. Improve task typing or stop this direction.

Bad but useful:

```text
oracle_typed_bundle is low
```

This means the reader itself is too weak. Try 1.5B or 3B before changing the method.

## Prior Work To Position Against

The novelty is not "small models can solve FinanceBench." Relevant prior work includes:

- **FinanceBench**: establishes the benchmark and shows that financial QA stresses retrieval, tabular reasoning, numeric reasoning, and hallucination.
- **KodeX**: finance-specialized 8B/70B models using RAG-aware LoRA and synthetic financial QA; strong FinanceBench-relevant baseline.
- **LiteCoST / structured-trace distillation**: close to the earlier template/distillation idea; motivates why generic structured traces are not enough as a novelty claim.

Our possible contribution is narrower:

> FinanceBench reliability improves when evidence is transformed into task-typed financial decision variables, not merely shortened or wrapped in generic traces.

## Local Machine Notes

An Apple Silicon machine such as an M4 Max / M5 Max with 64GB unified memory is useful for inference iteration:

- prompt tests,
- quantized model evaluation,
- Qwen/Gemma reader sweeps,
- cached evidence-bundle experiments.

It is not a drop-in replacement for CUDA workflows:

- `bitsandbytes`,
- QLoRA via CUDA,
- TRL/GRPO,
- heavy PEFT training.

For local Apple Silicon work, use MLX or llama.cpp/GGUF runners. The current repo's training code is CUDA/Colab-oriented, so local Apple Silicon support should be added as a separate runner rather than mixed into the Colab path.

## Secondary Ideas

### Delegated Reliability

If task-typed bundles help, the next GEP method is:

```text
strong extractor -> task-typed bundle + support probability -> small reader
```

Measure:

- bundle accuracy vs raw evidence,
- inherited support ECE vs reader verbal confidence ECE,
- selective accuracy by support probability.

### Decision-Preserving Quantization

A second possible paper:

> Domain-specific quantization calibration for financial decision models.

This only becomes strong if experiments show that finance-calibrated quantization preserves numeric/sign/unit/year correctness better than generic quantization calibration.

## Current Recommendation

Run Section 7d first. Training and quantization should wait until the evidence-interface result is clear.
