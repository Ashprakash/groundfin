"""GEP training: SFT + GRPO for risk-calibrated evidence and execution traces.

This module turns the prompt-time template probe into the actual method paper
step. It builds supervised training examples from FinanceBench, LoRA-fine-tunes
a small student under several supervision variants, evaluates reliability on a
held-out split (supported / missing / counterfactual sub-tasks), and then runs
GRPO with a verifiable multi-part reward.

No-leakage policy (inherited from financebench_pilot):
- The model INPUT (prompt) is built from raw evidence only, never from the gold
  answer, using the deployment-mode rule templates.
- The training TARGET (completion) is the gold answer plus calibrated
  confidence / support / abstention labels. Using gold labels as the SFT target
  is standard supervision; it is not input leakage.

Heavy deps (torch, transformers, trl, peft, datasets) are imported lazily inside
the trainer functions so the example builder and reward can be unit-tested with
pandas alone. Tested against trl>=0.12, peft>=0.11, transformers>=4.44.
"""

import json
import re

import pandas as pd

from benchmark import financebench_pilot as pilot


# ---------------------------------------------------------------------------
# Supervision variants (the ablation ladder from the experiment protocol).
# ---------------------------------------------------------------------------

VARIANTS = [
    "answer_only",              # baseline: raw evidence -> answer
    "answer_rationale",         # rationale-distillation baseline
    "answer_compact",           # extractive compact evidence
    "answer_template",          # rule template, no probability supervision
    "answer_template_abstain",  # METHOD: template + support + abstention + counterfactual
]

SUPPORTED_CONFIDENCE = 0.85
ABSTAIN_CONFIDENCE = 0.15
COUNTERFACTUAL_CONFIDENCE = 0.85


def train_test_split(df, test_frac=0.25, seed=7):
    """Company-disjoint split so the student cannot memorize a filing.

    Falls back to a plain row split if company metadata is unavailable.
    """
    numeric = df[df["has_numeric_answer"]].copy()
    if "company" in numeric.columns and numeric["company"].notna().any():
        companies = (
            numeric["company"].dropna().drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
        )
        n_test = max(1, int(len(companies) * test_frac))
        test_companies = set(companies[:n_test])
        test = numeric[numeric["company"].isin(test_companies)]
        train = numeric[~numeric["company"].isin(test_companies)]
    else:
        shuffled = numeric.sample(frac=1.0, random_state=seed)
        n_test = max(1, int(len(shuffled) * test_frac))
        test = shuffled.iloc[:n_test]
        train = shuffled.iloc[n_test:]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_target_completion(answer, confidence, evidence_support, abstain, rationale=None):
    """Canonical JSON completion the student is trained to emit."""
    obj = {
        "answer": str(answer),
        "confidence": round(float(confidence), 2),
        "evidence_support": bool(evidence_support),
        "abstain": bool(abstain),
    }
    if rationale:
        obj["short_rationale"] = str(rationale)[:240]
    return json.dumps(obj, ensure_ascii=False)


def _supported_prompt(row, variant, max_evidence_chars):
    """Build the no-leakage input prompt for a supported example."""
    if variant in ("answer_only", "answer_rationale"):
        return pilot.make_prompt(row, with_evidence=True, max_evidence_chars=max_evidence_chars)
    if variant == "answer_compact":
        return pilot.make_template_comparison_prompt(
            row, "length_matched_summary", leakage_mode="deployment",
            max_evidence_chars=max_evidence_chars,
        )
    if variant == "answer_template":
        return pilot.make_template_comparison_prompt(
            row, "template_no_probabilities", leakage_mode="deployment",
            max_evidence_chars=max_evidence_chars,
        )
    # answer_template_abstain -> full risk-calibrated template
    return pilot.make_template_comparison_prompt(
        row, "risk_calibrated_template", leakage_mode="deployment",
        max_evidence_chars=max_evidence_chars,
    )


def _missing_prompt(row, variant, max_evidence_chars):
    if variant in ("answer_only", "answer_rationale", "answer_compact"):
        return pilot.make_prompt(row, with_evidence=False)
    return pilot.make_template_comparison_prompt(
        row, "missing_risk_template", leakage_mode="deployment",
        max_evidence_chars=max_evidence_chars,
    )


def _counterfactual_prompt(row, cf_answer, max_evidence_chars):
    """Controlled evidence stating a counterfactual value (obedience test)."""
    return f"""{pilot.SYSTEM_INSTRUCTION}

Question:
{row["question"]}

Controlled counterfactual evidence bundle:
The answer to the question is {cf_answer}.
This value intentionally differs from any prior or memorized value. Answer only from this evidence.

Return JSON with keys: answer, confidence, evidence_support, short_rationale."""


def build_sft_examples(
    df,
    variant="answer_template_abstain",
    max_evidence_chars=6000,
    add_missing=True,
    add_counterfactual=True,
    missing_frac=0.2,
    counterfactual_frac=0.35,
    seed=7,
):
    """Construct (prompt, completion) SFT pairs for one supervision variant.

    Supported rows always included. For the method variant (and any variant when
    the flags are on) a fraction of rows are duplicated as missing-evidence
    (abstain target) and counterfactual (obey-evidence target) examples so that
    abstention and counterfactual obedience are supervised, not just accuracy.
    """
    is_method = variant == "answer_template_abstain"
    include_rationale = variant == "answer_rationale"

    examples = []
    for _, row in df.iterrows():
        gold = str(row["answer"])
        rationale = None
        if include_rationale:
            just = row.get("justification", "")
            rationale = just.strip() if isinstance(just, str) and just.strip() else f"Answer grounded in evidence: {gold}."
        examples.append(
            {
                "variant": variant,
                "subtask": "supported",
                "prompt": _supported_prompt(row, variant, max_evidence_chars),
                "completion": make_target_completion(
                    gold, SUPPORTED_CONFIDENCE, evidence_support=True, abstain=False,
                    rationale=rationale,
                ),
                "gold_answer": gold,
                "target_answer": gold,
                "expected_behavior": "answer",
                "evidence_text": row.get("evidence_text", ""),
            }
        )

    # Abstention and counterfactual augmentation (method variant, or when asked).
    augment = df.sample(frac=1.0, random_state=seed)
    if add_missing and is_method:
        n_missing = int(len(augment) * missing_frac)
        for _, row in augment.head(n_missing).iterrows():
            examples.append(
                {
                    "variant": variant,
                    "subtask": "missing",
                    "prompt": _missing_prompt(row, variant, max_evidence_chars),
                    "completion": make_target_completion(
                        "INSUFFICIENT_EVIDENCE", ABSTAIN_CONFIDENCE,
                        evidence_support=False, abstain=True,
                    ),
                    "gold_answer": str(row["answer"]),
                    "target_answer": "INSUFFICIENT_EVIDENCE",
                    "expected_behavior": "abstain",
                    "evidence_text": "",
                }
            )

    if add_counterfactual and is_method:
        n_cf = int(len(augment) * counterfactual_frac)
        for _, row in augment.head(n_cf).iterrows():
            cf = pilot.format_counterfactual_answer(row["answer"])
            if not cf:
                continue
            examples.append(
                {
                    "variant": variant,
                    "subtask": "counterfactual",
                    "prompt": _counterfactual_prompt(row, cf, max_evidence_chars),
                    "completion": make_target_completion(
                        cf, COUNTERFACTUAL_CONFIDENCE, evidence_support=True, abstain=False,
                    ),
                    "gold_answer": str(row["answer"]),
                    "target_answer": cf,
                    "expected_behavior": "answer",
                    "evidence_text": "",
                }
            )

    return pd.DataFrame(examples)


def generate_teacher_targets(
    train_df,
    teacher_model_id="Qwen/Qwen2.5-7B-Instruct",
    max_evidence_chars=4000,
    keep_only_correct=True,
    add_missing=True,
    add_counterfactual=True,
    seed=7,
):
    """Filtered teacher distillation.

    Runs a strong teacher over the method (risk-calibrated template) prompts,
    parses its answer/confidence/rationale, and keeps rows where the teacher is
    verifiably correct (weak-matches gold). The student is then SFT-trained on
    the teacher's own correct, calibrated outputs rather than raw gold labels,
    giving a real teacher-gap-recovery story. Missing/counterfactual rows are
    added from the rule-defined augmentation (no teacher needed there).
    """
    generator = pilot.load_hf_generator(
        model_id=teacher_model_id, max_new_tokens=192,
        load_in_4bit=pilot.auto_4bit(teacher_model_id),
    )

    kept, dropped = [], 0
    for _, row in pilot.tqdm(train_df.iterrows(), total=len(train_df)):
        prompt = _supported_prompt(row, "answer_template_abstain", max_evidence_chars)
        pred = pilot.call_hf_generator(generator, prompt)
        answer = pilot.parse_model_answer(pred)
        confidence = pilot.parse_model_confidence(pred)
        gold = str(row["answer"])
        if keep_only_correct and not pilot.weak_answer_match(answer, gold):
            dropped += 1
            continue
        kept.append(
            {
                "variant": "teacher_distill",
                "subtask": "supported",
                "prompt": prompt,
                "completion": make_target_completion(
                    answer,
                    confidence if confidence is not None else SUPPORTED_CONFIDENCE,
                    evidence_support=True, abstain=False,
                ),
                "gold_answer": gold,
                "target_answer": gold,
                "expected_behavior": "answer",
                "evidence_text": row.get("evidence_text", ""),
            }
        )

    print(f"Teacher kept {len(kept)} correct / dropped {dropped} "
          f"(teacher supported accuracy ~= {len(kept) / max(1, len(train_df)):.2f}).")

    examples = pd.DataFrame(kept)
    if add_missing or add_counterfactual:
        aug = build_sft_examples(
            train_df, variant="answer_template_abstain",
            max_evidence_chars=max_evidence_chars,
            add_missing=add_missing, add_counterfactual=add_counterfactual, seed=seed,
        )
        aug = aug[aug["subtask"] != "supported"]
        examples = pd.concat([examples, aug], ignore_index=True)
    return examples


def audit_sft_leakage(examples):
    """Return supported rows whose PROMPT contains the target answer even though
    it is NOT present in the source evidence, i.e. the answer was injected rather
    than grounded. Should be empty: deployment templates never read the answer.

    A target value that appears in the prompt because it appears in the evidence
    is grounded extraction, not leakage, so it is not flagged. Missing rows
    (abstain) and counterfactual rows (controlled evidence intentionally states
    the value) are exempt.
    """
    bad = []
    for _, ex in examples.iterrows():
        if ex["subtask"] in ("missing", "counterfactual"):
            continue
        tgt = pilot.normalize_text(ex["target_answer"])
        if len(tgt) < 4:
            continue
        in_prompt = tgt in pilot.normalize_text(ex["prompt"])
        in_evidence = tgt in pilot.normalize_text(ex.get("evidence_text", ""))
        if in_prompt and not in_evidence:
            bad.append({"subtask": ex["subtask"], "target": ex["target_answer"]})
    return pd.DataFrame(bad)


# ---------------------------------------------------------------------------
# Verifiable GRPO reward.
# ---------------------------------------------------------------------------

def _score_completion(text, gold_answer, target_answer, expected_behavior):
    """Per-example verifiable reward in [0, 1].

    Correctness sets the band (correct >= 0.7, wrong <= 0.3, non-overlapping);
    confidence shapes within the band so that: confident-right > hesitant-right,
    hesitant-wrong > confident-wrong (calibration), correct abstention is high,
    and confident hallucination when abstention was required is near zero.
    """
    answer = pilot.parse_model_answer(text)
    confidence = pilot.parse_model_confidence(text)
    refusal = pilot.is_refusal(answer)

    if expected_behavior == "abstain":
        if refusal:
            c = 0.0 if confidence is None else confidence
            return round(0.8 + 0.2 * (1 - c), 3)  # honest (low-confidence) abstention best
        c = 0.5 if confidence is None else confidence
        return round(0.1 * (1 - c), 3)  # hallucinated instead of abstaining

    correct = pilot.weak_answer_match(answer, target_answer)
    if correct:
        c = 0.7 if confidence is None else confidence
        reward = 0.7 + 0.3 * c
    else:
        c = 0.5 if confidence is None else confidence
        reward = 0.3 * (1 - c)
    return round(max(0.0, min(1.0, reward)), 3)


def groundfin_reward(completions, gold_answer=None, target_answer=None,
                     expected_behavior=None, **kwargs):
    """TRL GRPO reward function.

    `completions` may be plain strings or chat-format [{"role","content"}].
    The dataset columns gold_answer/target_answer/expected_behavior arrive as
    parallel lists.
    """
    def as_text(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list) and c and isinstance(c[-1], dict):
            return c[-1].get("content", "")
        return str(c)

    n = len(completions)
    gold_answer = gold_answer or [""] * n
    target_answer = target_answer or gold_answer
    expected_behavior = expected_behavior or ["answer"] * n

    return [
        _score_completion(as_text(c), g, t, e)
        for c, g, t, e in zip(completions, gold_answer, target_answer, expected_behavior)
    ]


# ---------------------------------------------------------------------------
# LoRA SFT.
# ---------------------------------------------------------------------------

def _to_chat_text(tokenizer, prompt, completion):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"{prompt}\n{completion}"


def free_gpu():
    """Release cached VRAM held by earlier generators/models in the session.

    The eval/suite cells load pipelines that keep GPU memory cached; call this
    before training so SFT/GRPO do not OOM on a T4.
    """
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_memory():
    """Return (free_gb, total_gb, device_name) or (None, None, None) on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None, None
        free_b, total_b = torch.cuda.mem_get_info()
        return free_b / 1e9, total_b / 1e9, torch.cuda.get_device_name(0)
    except Exception:
        return None, None, None


def print_env():
    """Report the runtime so config decisions are visible and debuggable."""
    import platform

    info = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"
    free_gb, total_gb, name = gpu_memory()
    if name:
        info["gpu"] = f"{name} ({free_gb:.1f} GB free / {total_gb:.1f} GB)"
    else:
        info["gpu"] = "none (CPU)"
    print("=== ENVIRONMENT ===")
    for k, v in info.items():
        print(f"  {k:14s} {v}")
    return info


def _model_billions(model_id):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    return float(m.group(1)) if m else 0.5


def recommended_sft_config(model_id, free_gb=None):
    """Pick batch/seq/quantization from the model size and free VRAM.

    Conservative on purpose: the OOM-retry loop in run_sft will shrink further
    if these still do not fit. Returns a dict of training knobs.
    """
    if free_gb is None:
        free_gb, _, _ = gpu_memory()
    size = _model_billions(model_id)
    load_in_4bit = pilot.auto_4bit(model_id) or size >= 3

    if free_gb is None:  # CPU fallback (tiny, just so it runs)
        batch, seq, ev = 1, 512, 1200
    elif free_gb >= 30 and size < 3:
        batch, seq, ev = 4, 1536, 3000
    elif free_gb >= 12:
        batch, seq, ev = (2, 1024, 2500) if size < 3 else (1, 1024, 2500)
    elif free_gb >= 7:
        batch, seq, ev = 1, 1024, 2000
    elif free_gb >= 4:
        batch, seq, ev = 1, 768, 1500
    else:
        batch, seq, ev = 1, 512, 1200

    grad_accum = max(1, 16 // batch)
    return {
        "batch_size": batch,
        "grad_accum": grad_accum,
        "max_seq_len": seq,
        "max_evidence_chars": ev,
        "load_in_4bit": load_in_4bit,
    }


def _is_oom(err):
    try:
        import torch

        if isinstance(err, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(err).lower()


def _sft_once(dataset, model_id, output_dir, tokenizer, *, epochs, lr, batch_size,
              grad_accum, max_seq_len, load_in_4bit, seed):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("torch_dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.config.use_cache = False  # required with gradient checkpointing

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    try:
        import bitsandbytes  # noqa: F401

        optim = "paged_adamw_8bit"
    except Exception:
        optim = "adamw_torch"
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="epoch",
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        bf16=torch.cuda.is_available(),
        optim=optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    del trainer, model
    free_gpu()


def run_sft(
    train_df,
    variant="answer_template_abstain",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    output_dir=None,
    epochs=1,
    lr=1e-4,
    batch_size=None,
    grad_accum=None,
    max_seq_len=None,
    max_evidence_chars=None,
    seed=7,
    examples=None,
    max_oom_retries=3,
):
    """LoRA SFT with hardware-aware config and OOM-retry.

    Batch/seq/quantization default from recommended_sft_config (free VRAM +
    model size). If training still OOMs, VRAM is freed and batch/seq are
    reduced automatically until it fits or retries are exhausted.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    free_gpu()
    rec = recommended_sft_config(model_id)
    batch_size = rec["batch_size"] if batch_size is None else batch_size
    grad_accum = (max(1, 16 // batch_size)) if grad_accum is None else grad_accum
    max_seq_len = rec["max_seq_len"] if max_seq_len is None else max_seq_len
    max_evidence_chars = rec["max_evidence_chars"] if max_evidence_chars is None else max_evidence_chars
    load_in_4bit = rec["load_in_4bit"]
    free_gb, _, name = gpu_memory()
    print(f"[run_sft] gpu={name} free={None if free_gb is None else round(free_gb,1)}GB "
          f"-> batch={batch_size} grad_accum={grad_accum} seq={max_seq_len} "
          f"evidence={max_evidence_chars} 4bit={load_in_4bit}")

    output_dir = output_dir or f"groundfin_sft_{variant}_{model_id.split('/')[-1]}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if examples is None:
        examples = build_sft_examples(
            train_df, variant=variant, max_evidence_chars=max_evidence_chars, seed=seed
        )
    leak = audit_sft_leakage(examples)
    if len(leak):
        print(f"[warn] {len(leak)} supported prompts contain the target answer; inspect audit_sft_leakage output.")
    texts = [
        _to_chat_text(tokenizer, p, c)
        for p, c in zip(examples["prompt"], examples["completion"])
    ]
    dataset = Dataset.from_dict({"text": texts})

    attempt = 0
    while True:
        try:
            _sft_once(
                dataset, model_id, output_dir, tokenizer,
                epochs=epochs, lr=lr, batch_size=batch_size, grad_accum=grad_accum,
                max_seq_len=max_seq_len, load_in_4bit=load_in_4bit, seed=seed,
            )
            print(f"Saved SFT adapter to {output_dir} (n_examples={len(examples)}).")
            return output_dir
        except Exception as err:
            if not _is_oom(err):
                raise
            free_gpu()
            attempt += 1
            if attempt > max_oom_retries or (batch_size == 1 and max_seq_len <= 512):
                print(f"[run_sft] OOM after {attempt} retries at batch={batch_size} seq={max_seq_len}; giving up.")
                raise
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                grad_accum = max(1, 16 // batch_size)
            else:
                max_seq_len = max(512, int(max_seq_len * 0.75))
            print(f"[run_sft] OOM caught; retrying with batch={batch_size} "
                  f"grad_accum={grad_accum} seq={max_seq_len} (attempt {attempt}/{max_oom_retries}).")


# ---------------------------------------------------------------------------
# GRPO with the verifiable reward.
# ---------------------------------------------------------------------------

def build_grpo_dataset(train_df, max_evidence_chars=4000, seed=7):
    """Method-variant prompts as a conversational GRPO dataset."""
    examples = build_sft_examples(
        train_df, variant="answer_template_abstain",
        max_evidence_chars=max_evidence_chars, seed=seed,
    )
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": p}] for p in examples["prompt"]],
            "gold_answer": list(examples["gold_answer"]),
            "target_answer": list(examples["target_answer"]),
            "expected_behavior": list(examples["expected_behavior"]),
        }
    )


def _grpo_once(dataset, model_id, sft_adapter, output_dir, tokenizer, *, epochs, lr,
               batch_size, grad_accum, num_generations, max_prompt_len,
               max_completion_len, load_in_4bit, seed):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    load_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("torch_dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.config.use_cache = False

    peft_config = None
    if sft_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    try:
        import bitsandbytes  # noqa: F401

        optim = "paged_adamw_8bit"
    except Exception:
        optim = "adamw_torch"
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        logging_steps=5,
        bf16=torch.cuda.is_available(),
        optim=optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=seed,
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=groundfin_reward, args=grpo_config,
        train_dataset=dataset, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    del trainer, model
    free_gpu()


def run_grpo(
    train_df,
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    sft_adapter=None,
    output_dir=None,
    epochs=1,
    lr=1e-5,
    num_generations=None,
    batch_size=None,
    grad_accum=None,
    max_prompt_len=1024,
    max_completion_len=160,
    max_evidence_chars=None,
    seed=7,
    max_oom_retries=3,
):
    """GRPO with hardware-aware config and OOM-retry (heavier than SFT: it also
    samples num_generations completions per prompt, so it degrades batch and
    then num_generations before giving up)."""
    from transformers import AutoTokenizer

    free_gpu()
    rec = recommended_sft_config(model_id)
    load_in_4bit = rec["load_in_4bit"]
    max_evidence_chars = rec["max_evidence_chars"] if max_evidence_chars is None else max_evidence_chars
    # GRPO is heavier than SFT; start one notch below the SFT batch.
    if batch_size is None:
        batch_size = max(1, rec["batch_size"] // 2) if rec["batch_size"] > 1 else 1
    if num_generations is None:
        num_generations = 4
    if grad_accum is None:
        grad_accum = max(1, num_generations // batch_size)

    free_gb, _, name = gpu_memory()
    print(f"[run_grpo] gpu={name} free={None if free_gb is None else round(free_gb,1)}GB "
          f"-> batch={batch_size} grad_accum={grad_accum} num_gen={num_generations} "
          f"4bit={load_in_4bit}")

    output_dir = output_dir or f"groundfin_grpo_{model_id.split('/')[-1]}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_grpo_dataset(train_df, max_evidence_chars=max_evidence_chars, seed=seed)

    attempt = 0
    while True:
        # keep effective batch divisible by num_generations (TRL requirement)
        grad_accum = max(1, grad_accum)
        try:
            _grpo_once(
                dataset, model_id, sft_adapter, output_dir, tokenizer,
                epochs=epochs, lr=lr, batch_size=batch_size, grad_accum=grad_accum,
                num_generations=num_generations, max_prompt_len=max_prompt_len,
                max_completion_len=max_completion_len, load_in_4bit=load_in_4bit, seed=seed,
            )
            print(f"Saved GRPO model to {output_dir}.")
            return output_dir
        except Exception as err:
            if not _is_oom(err):
                raise
            free_gpu()
            attempt += 1
            if attempt > max_oom_retries or (batch_size == 1 and num_generations <= 2):
                print(f"[run_grpo] OOM after {attempt} retries; giving up.")
                raise
            if batch_size > 1:
                batch_size = 1
            else:
                num_generations = max(2, num_generations - 2)
            grad_accum = max(1, num_generations // batch_size)
            print(f"[run_grpo] OOM caught; retrying with batch={batch_size} "
                  f"num_gen={num_generations} (attempt {attempt}/{max_oom_retries}).")


# ---------------------------------------------------------------------------
# Reliability evaluation on a held-out split.
# ---------------------------------------------------------------------------

def _load_eval_generator(model_id, adapter=None, max_new_tokens=192):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(adapter or model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if pilot.auto_4bit(model_id) and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False,
    )


def evaluate_model(
    test_df,
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    adapter=None,
    eval_variant="answer_template_abstain",
    n_examples=None,
    max_evidence_chars=4000,
    max_new_tokens=192,
    label="model",
    seed=7,
):
    """Evaluate one model/adapter across supported, missing, and counterfactual
    sub-tasks, returning a one-row-per-subtask reliability summary."""
    generator = _load_eval_generator(model_id, adapter=adapter, max_new_tokens=max_new_tokens)
    sample = test_df if n_examples is None else test_df.head(n_examples)
    examples = build_sft_examples(
        sample, variant=eval_variant, max_evidence_chars=max_evidence_chars, seed=seed
    )

    rows = []
    for _, ex in pilot.tqdm(examples.iterrows(), total=len(examples)):
        pred = pilot.call_hf_generator(generator, ex["prompt"])
        scores = pilot.score_prediction(pred, ex["target_answer"])
        success = scores["refusal"] if ex["expected_behavior"] == "abstain" else scores["weak_match_answer"]
        rows.append(
            {
                "label": label,
                "variant": eval_variant,
                "subtask": ex["subtask"],
                "gold_answer": ex["gold_answer"],
                "target_answer": ex["target_answer"],
                "prediction": pred,
                "success": bool(success),
                **scores,
            }
        )
    results = pd.DataFrame(rows)
    summary = _summarize_eval(results)
    return results, summary


def _summarize_eval(results):
    metric_cols = ["success", "weak_match_answer", "numeric_match_answer", "refusal",
                   "confidence", "brier", "overconfident_wrong"]
    metric_cols = [c for c in metric_cols if c in results.columns]
    summary = results.groupby(["label", "subtask"])[metric_cols].mean()
    summary["n"] = results.groupby(["label", "subtask"]).size()
    summary["ece"] = results.groupby(["label", "subtask"]).apply(
        lambda g: pilot.expected_calibration_error(g)
    )
    return summary


def compare_models(test_df, specs, n_examples=None, max_evidence_chars=4000,
                   max_new_tokens=192, seed=7):
    """Evaluate several (label, model_id, adapter, eval_variant) specs and stack
    their reliability summaries for the paper table.

    specs: list of dicts with keys label, model_id, adapter (optional),
    eval_variant (optional, default method).
    """
    all_summaries = []
    all_results = []
    for spec in specs:
        results, summary = evaluate_model(
            test_df,
            model_id=spec["model_id"],
            adapter=spec.get("adapter"),
            eval_variant=spec.get("eval_variant", "answer_template_abstain"),
            n_examples=n_examples,
            max_evidence_chars=max_evidence_chars,
            max_new_tokens=max_new_tokens,
            label=spec["label"],
            seed=seed,
        )
        all_results.append(results)
        all_summaries.append(summary.reset_index())
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return pd.concat(all_results, ignore_index=True), pd.concat(all_summaries, ignore_index=True)
