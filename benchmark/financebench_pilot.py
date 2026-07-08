import json
import math
import os
import re

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm


DATASET_ID = "PatronusAI/financebench"

SYSTEM_INSTRUCTION = (
    "You are a careful financial QA assistant. Answer only from the provided "
    "evidence when evidence is provided. If the answer is not supported, say "
    "INSUFFICIENT_EVIDENCE. Return a concise answer and a confidence from 0 to 1."
)


def load_financebench():
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()
    df["evidence_text"] = df["evidence"].apply(flatten_evidence)
    df["evidence_chars"] = df["evidence_text"].str.len()
    add_candidate_flags(df)
    return df


def flatten_evidence(evidence):
    if evidence is None or (isinstance(evidence, float) and math.isnan(evidence)):
        return ""
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except Exception:
            return evidence

    parts = []
    for ev in evidence:
        if not isinstance(ev, dict):
            parts.append(str(ev))
            continue
        doc = ev.get("doc_name", "")
        page = ev.get("evidence_page_num", "")
        txt = ev.get("evidence_text", "") or ev.get("evidence_text_full_page", "")
        prefix = f"[doc={doc} page={page}]".strip()
        parts.append(f"{prefix} {txt}".strip())
    return "\n\n".join(parts)


def make_prompt(row, with_evidence=True, max_evidence_chars=None):
    if with_evidence:
        evidence_text = row["evidence_text"]
        if max_evidence_chars is not None and len(evidence_text) > max_evidence_chars:
            evidence_text = evidence_text[:max_evidence_chars] + "\n\n[TRUNCATED]"
        return f"""{SYSTEM_INSTRUCTION}

Question:
{row["question"]}

Evidence:
{evidence_text}

Return JSON with keys: answer, confidence, evidence_support, short_rationale."""

    return f"""{SYSTEM_INSTRUCTION}

Question:
{row["question"]}

No evidence is provided.

Return JSON with keys: answer, confidence, evidence_support, short_rationale."""


def normalize_text(s):
    s = str(s).lower().strip()
    s = re.sub(r"[$,%]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_numbers(s):
    return [
        float(x.replace(",", ""))
        for x in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(s))
    ]


def numeric_close(pred, gold, rel_tol=0.02, abs_tol=0.05):
    pnums = extract_numbers(pred)
    gnums = extract_numbers(gold)
    if not pnums or not gnums:
        return False

    for p in pnums:
        for g in gnums:
            if abs(p - g) <= max(abs_tol, rel_tol * max(1.0, abs(g))):
                return True
    return False


def weak_answer_match(pred, gold):
    pn = normalize_text(pred)
    gn = normalize_text(gold)
    if numeric_close(pred, gold):
        return True
    if len(pn) >= 12 and len(gn) >= 12 and (gn in pn or pn in gn):
        return True
    return False


def parse_model_answer(prediction):
    text = str(prediction).strip()
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"]).strip()
    except Exception:
        pass

    match = re.search(r'"answer"\s*:\s*"([^"]+)"', cleaned)
    if match:
        return match.group(1).strip()

    match = re.search(r"answer\s*:\s*(.+)", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip()

    return text


def is_refusal(answer_text):
    normalized = normalize_text(answer_text)
    refusal_markers = [
        "insufficient_evidence",
        "insufficient evidence",
        "not enough evidence",
        "cannot determine",
        "not provided",
        "no evidence",
    ]
    return any(marker in normalized for marker in refusal_markers)


def score_prediction(prediction, gold):
    answer_text = parse_model_answer(prediction)
    return {
        "parsed_answer": answer_text,
        "weak_match_raw": weak_answer_match(prediction, gold),
        "weak_match_answer": weak_answer_match(answer_text, gold),
        "numeric_match_answer": numeric_close(answer_text, gold),
        "refusal": is_refusal(answer_text),
    }


def format_counterfactual_answer(answer, factor=1.37):
    nums = extract_numbers(answer)
    if not nums:
        return None

    original = nums[0]
    changed = original * factor
    if abs(changed - original) < 1e-9:
        changed = original + 1

    answer_text = str(answer)
    if "%" in answer_text:
        rendered = f"{changed:.1f}%"
    elif "." in answer_text:
        rendered = f"{changed:.2f}"
    else:
        rendered = f"{changed:,.0f}"

    if "$" in answer_text:
        rendered = f"${rendered}"
    return rendered


def make_probe_prompt(row, max_evidence_chars=None):
    evidence_text = row.get("evidence_text", "")
    if max_evidence_chars is not None and len(evidence_text) > max_evidence_chars:
        evidence_text = evidence_text[:max_evidence_chars] + "\n\n[TRUNCATED]"

    if evidence_text:
        evidence_block = f"Evidence:\n{evidence_text}"
    else:
        evidence_block = "Evidence:\nNo evidence is provided."

    return f"""{SYSTEM_INSTRUCTION}

Question:
{row["question"]}

{evidence_block}

Return JSON with keys: answer, confidence, evidence_support, short_rationale."""


def build_grounding_probe(df, n_examples=5, random_state=7):
    base = df[df["has_numeric_answer"]].sample(
        min(n_examples, int(df["has_numeric_answer"].sum())),
        random_state=random_state,
    )

    rows = []
    for _, row in base.iterrows():
        common = {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "gold_answer": row["answer"],
        }

        rows.append(
            {
                **common,
                "condition": "gold_evidence",
                "target_answer": row["answer"],
                "expected_behavior": "answer",
                "evidence_text": row["evidence_text"],
            }
        )

        justification = row.get("justification", "")
        if isinstance(justification, str) and justification.strip():
            compressed_evidence = justification.strip()
            compression_source = "financebench_justification"
        else:
            compressed_evidence = (
                f"The answer to the question is {row['answer']}. "
                "This is an oracle compact evidence proxy for debugging."
            )
            compression_source = "oracle_answer_fallback"

        rows.append(
            {
                **common,
                "condition": "evidence_compressed",
                "target_answer": row["answer"],
                "expected_behavior": "answer",
                "compression_source": compression_source,
                "evidence_text": (
                    "Compact evidence bundle:\n"
                    f"{compressed_evidence}\n"
                    "Use this compact evidence to answer the question."
                ),
            }
        )

        rows.append(
            {
                **common,
                "condition": "missing_evidence",
                "target_answer": "INSUFFICIENT_EVIDENCE",
                "expected_behavior": "abstain",
                "evidence_text": "",
            }
        )

        rows.append(
            {
                **common,
                "condition": "direct_grounded_evidence",
                "target_answer": row["answer"],
                "expected_behavior": "answer",
                "evidence_text": (
                    "Controlled evidence bundle:\n"
                    f"The answer to the question is {row['answer']}.\n"
                    "Use this grounded evidence rather than prior knowledge."
                ),
            }
        )

        counterfactual = format_counterfactual_answer(row["answer"])
        if counterfactual:
            rows.append(
                {
                    **common,
                    "condition": "counterfactual_direct_evidence",
                    "target_answer": counterfactual,
                    "expected_behavior": "answer",
                    "evidence_text": (
                        "Controlled counterfactual evidence bundle:\n"
                        f"The answer to the question is {counterfactual}.\n"
                        "This value intentionally differs from any prior or memorized value. "
                        "Use this grounded evidence."
                    ),
                }
            )

    return pd.DataFrame(rows)


def summarize_probe_results(results, group_cols):
    metric_cols = [
        "probe_success",
        "weak_match_answer",
        "numeric_match_answer",
        "refusal",
    ]
    summary = results.groupby(group_cols)[metric_cols].mean()
    summary = summary.rename(
        columns={
            "probe_success": "success_rate",
            "weak_match_answer": "answer_weak_accuracy",
            "numeric_match_answer": "answer_numeric_accuracy",
            "refusal": "refusal_rate",
        }
    )
    summary["n"] = results.groupby(group_cols).size()
    return summary


def has_numeric_answer(answer):
    return len(extract_numbers(answer)) > 0


def add_candidate_flags(df):
    df["has_numeric_answer"] = df["answer"].apply(has_numeric_answer)
    df["candidate_numeric_counterfactual"] = (
        df["has_numeric_answer"] & (df["evidence_chars"] > 200)
    )
    df["candidate_missing_evidence"] = df["evidence_chars"] > 200
    df["candidate_temporal"] = df["question"].str.contains(
        "FY|Q[1-4]|year|2022|2023|2019|2018",
        case=False,
        regex=True,
        na=False,
    )


def summarize_dataset(df):
    print("Shape:", df.shape)
    print("Unique companies:", df["company"].nunique())

    for col in [
        "question_type",
        "question_reasoning",
        "dataset_subset_label",
        "gics_sector",
        "doc_type",
        "doc_period",
    ]:
        if col in df.columns:
            print(f"\n{col}")
            display(df[col].value_counts(dropna=False).head(20).to_frame("count"))

    flags = [
        "candidate_numeric_counterfactual",
        "candidate_missing_evidence",
        "candidate_temporal",
    ]
    print("\nCandidate flags")
    display(df[flags].sum().to_frame("count"))


def call_openai(prompt, model="gpt-4.1-mini"):
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def run_openai_baseline(df, n_examples=20, model="gpt-4.1-mini", random_state=7):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    rows = []
    pilot = df.sample(min(n_examples, len(df)), random_state=random_state).reset_index(
        drop=True
    )

    for _, row in tqdm(pilot.iterrows(), total=len(pilot)):
        for condition, with_evidence in [
            ("question_only", False),
            ("with_gold_evidence", True),
        ]:
            pred = call_openai(make_prompt(row, with_evidence=with_evidence), model)
            scores = score_prediction(pred, row["answer"])
            rows.append(
                {
                    "financebench_id": row["financebench_id"],
                    "condition": condition,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "prediction": pred,
                    "weak_match": scores["weak_match_answer"],
                    **scores,
                }
            )

    results = pd.DataFrame(rows)
    summary = summarize_results(results, ["condition"])
    return results, summary


def load_hf_generator(model_id="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=192):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )


def call_hf_generator(generator, prompt):
    messages = [{"role": "user", "content": prompt}]
    tokenizer = generator.tokenizer
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = prompt
    out = generator(rendered)
    return out[0]["generated_text"].strip()


def run_hf_baseline(
    df,
    n_examples=10,
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    random_state=7,
    max_new_tokens=192,
    max_evidence_chars=6000,
):
    generator = load_hf_generator(model_id=model_id, max_new_tokens=max_new_tokens)
    rows = []
    pilot = df.sample(min(n_examples, len(df)), random_state=random_state).reset_index(
        drop=True
    )

    for _, row in tqdm(pilot.iterrows(), total=len(pilot)):
        for condition, with_evidence in [
            ("question_only", False),
            ("with_gold_evidence", True),
        ]:
            pred = call_hf_generator(
                generator,
                make_prompt(
                    row,
                    with_evidence=with_evidence,
                    max_evidence_chars=max_evidence_chars,
                ),
            )
            scores = score_prediction(pred, row["answer"])
            rows.append(
                {
                    "financebench_id": row["financebench_id"],
                    "condition": condition,
                    "model_id": model_id,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "prediction": pred,
                    "weak_match": scores["weak_match_answer"],
                    **scores,
                }
            )

    results = pd.DataFrame(rows)
    summary = summarize_results(results, ["model_id", "condition"])
    return results, summary


def run_hf_grounding_probe(
    df,
    n_examples=5,
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    random_state=7,
    max_new_tokens=160,
    max_evidence_chars=6000,
):
    generator = load_hf_generator(model_id=model_id, max_new_tokens=max_new_tokens)
    probe_df = build_grounding_probe(
        df, n_examples=n_examples, random_state=random_state
    )

    rows = []
    for _, row in tqdm(probe_df.iterrows(), total=len(probe_df)):
        pred = call_hf_generator(
            generator, make_probe_prompt(row, max_evidence_chars=max_evidence_chars)
        )
        scores = score_prediction(pred, row["target_answer"])
        probe_success = (
            scores["refusal"]
            if row["expected_behavior"] == "abstain"
            else scores["weak_match_answer"]
        )
        rows.append(
            {
                "financebench_id": row["financebench_id"],
                "condition": row["condition"],
                "expected_behavior": row["expected_behavior"],
                "model_id": model_id,
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "target_answer": row["target_answer"],
                "prediction": pred,
                "weak_match": scores["weak_match_answer"],
                "probe_success": bool(probe_success),
                **scores,
            }
        )

    results = pd.DataFrame(rows)
    summary = summarize_probe_results(results, ["model_id", "condition"])
    return results, summary


def summarize_results(results, group_cols):
    metric_cols = [
        "weak_match_raw",
        "weak_match_answer",
        "numeric_match_answer",
        "refusal",
    ]
    summary = results.groupby(group_cols)[metric_cols].mean()
    summary = summary.rename(
        columns={
            "weak_match_raw": "weak_accuracy_raw",
            "weak_match_answer": "weak_accuracy_answer",
            "numeric_match_answer": "numeric_accuracy_answer",
            "refusal": "refusal_rate",
        }
    )
    summary["n"] = results.groupby(group_cols).size()
    return summary


def export_pilot_files(df, prefix="financebench_pilot_flat"):
    cols = [
        "financebench_id",
        "company",
        "doc_name",
        "question_type",
        "question_reasoning",
        "question",
        "answer",
        "justification",
        "evidence_text",
        "gics_sector",
        "doc_type",
        "doc_period",
        "candidate_numeric_counterfactual",
        "candidate_missing_evidence",
        "candidate_temporal",
    ]
    pilot_export = df[cols].copy()
    pilot_export.to_csv(f"{prefix}.csv", index=False)
    pilot_export.to_json(f"{prefix}.jsonl", orient="records", lines=True)
    return f"{prefix}.csv", f"{prefix}.jsonl"
