# GEP Abstract And Adversarial Review

## Current Abstract

Financial question answering over regulatory filings is often treated as a retrieval-augmented generation problem: retrieve the right context, then ask a language model to synthesize the answer. Our experiments suggest a different bottleneck for small and mid-sized open models. Even with gold evidence, models frequently fail when the task requires extracting typed financial variables, preserving units and periods, and performing deterministic computation. We introduce **Grounded Execution Precision (GEP)**, a zero-trust architecture that constrains the language model to typed extraction and delegates arithmetic to a deterministic executor. GEP reports answers only when the extracted variables and formula form an auditable execution trace; otherwise it abstains. Across FinanceBench-style financial QA runs and related numeric reasoning benchmarks, GEP improves committed-answer precision over raw evidence prompting while explicitly reporting the associated coverage tradeoff. The key contribution is not another RAG prompt, but an execution-centered reliability protocol for financial decision models: extract, execute, verify, and selectively commit.

## Reviewer-Safe Claim

GEP is strongest when framed as:

> a selective, auditable precision architecture for computational financial QA.

The safest claims are:

- GEP improves `acc@commit` over raw evidence prompting.
- GEP trades coverage for precision, so yield must be reported alongside committed accuracy.
- GEP applies most naturally to computational and lookup-style financial questions.
- Calibration and abstention remain open problems; current evidence suggests ensemble signals can be better calibrated than single signals, but no single selector universally transfers.

## Claims To Avoid Until More Evidence Lands

- Do not claim universal full-coverage accuracy improvement.
- Do not claim adversarial robustness without a dedicated adversarial table.
- Do not claim GEP "solves" FinanceBench.
- Do not claim the confidence ensemble is always best for ranking selective accuracy.
- Do not hide yield losses where low coverage makes committed precision misleading.

## Adversarial Reviewer Notes

Likely reviewer objections:

- "This is just program-of-thought / tool use."
- "The method wins by abstaining."
- "FinanceBench has heterogeneous question types; scalar execution is only a subset."
- "Committed accuracy is not enough; compare at matched coverage."
- "Where are seeds, confidence intervals, and human validation of the judge?"

Required defenses:

- Report both `acc@commit` and `yield = acc@commit * coverage`.
- Include matched-coverage baselines.
- Split computational/lookup questions from open-ended/list/selection questions.
- Present failure taxonomy as a contribution, not an embarrassment.
- Treat distillation and adversarial context injection as follow-up unless measured.
