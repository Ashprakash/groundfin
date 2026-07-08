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


def make_prompt(row, with_evidence=True):
    if with_evidence:
        return f"""{SYSTEM_INSTRUCTION}

Question:
{row["question"]}

Evidence:
{row["evidence_text"]}

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
    if gn and (gn in pn or pn in gn):
        return True
    return numeric_close(pred, gold)


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
            rows.append(
                {
                    "financebench_id": row["financebench_id"],
                    "condition": condition,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "prediction": pred,
                    "weak_match": weak_answer_match(pred, row["answer"]),
                }
            )

    results = pd.DataFrame(rows)
    summary = results.groupby("condition")["weak_match"].mean().to_frame(
        "weak_accuracy"
    )
    return results, summary


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
