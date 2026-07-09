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


def parse_model_confidence(prediction):
    text = str(prediction).strip()
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            for key in ["confidence", "calibrated_confidence", "answer_confidence"]:
                if key in obj:
                    value = float(obj[key])
                    return max(0.0, min(1.0, value))
    except Exception:
        pass

    match = re.search(
        r"(?:confidence|calibrated_confidence|answer_confidence)\s*[:=]\s*([01](?:\.\d+)?)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        value = float(match.group(1))
        return max(0.0, min(1.0, value))
    return None


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
    confidence = parse_model_confidence(prediction)
    weak_match = weak_answer_match(answer_text, gold)
    numeric_match = numeric_close(answer_text, gold)
    refusal = is_refusal(answer_text)
    correctness = 1.0 if weak_match else 0.0
    brier = None if confidence is None else (confidence - correctness) ** 2
    return {
        "parsed_answer": answer_text,
        "weak_match_raw": weak_answer_match(prediction, gold),
        "weak_match_answer": weak_match,
        "numeric_match_answer": numeric_match,
        "refusal": refusal,
        "confidence": confidence,
        "brier": brier,
        "overconfident_wrong": bool(confidence is not None and confidence >= 0.8 and not weak_match),
    }


def expected_calibration_error(results, confidence_col="confidence", correctness_col="weak_match_answer", n_bins=5):
    scored = results.dropna(subset=[confidence_col]).copy()
    if scored.empty:
        return None

    ece = 0.0
    total = len(scored)
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (scored[confidence_col] >= lower) & (scored[confidence_col] <= upper)
        else:
            mask = (scored[confidence_col] >= lower) & (scored[confidence_col] < upper)
        bucket = scored[mask]
        if bucket.empty:
            continue
        accuracy = bucket[correctness_col].astype(float).mean()
        confidence = bucket[confidence_col].astype(float).mean()
        ece += (len(bucket) / total) * abs(accuracy - confidence)
    return ece


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
        "confidence",
        "brier",
        "overconfident_wrong",
    ]
    metric_cols = [col for col in metric_cols if col in results.columns]
    summary = results.groupby(group_cols)[metric_cols].mean()
    summary = summary.rename(
        columns={
            "probe_success": "success_rate",
            "weak_match_answer": "answer_weak_accuracy",
            "numeric_match_answer": "answer_numeric_accuracy",
            "refusal": "refusal_rate",
            "confidence": "avg_confidence",
            "brier": "brier_score",
            "overconfident_wrong": "overconfident_wrong_rate",
        }
    )
    summary["n"] = results.groupby(group_cols).size()
    ece = results.groupby(group_cols).apply(
        lambda group: expected_calibration_error(group)
    )
    summary["ece"] = ece
    return summary


def compact_evidence_for_row(row):
    justification = row.get("justification", "")
    if isinstance(justification, str) and justification.strip():
        return justification.strip(), "financebench_justification"
    return (
        f"The answer to the question is {row['answer']}. "
        "This is an oracle compact evidence proxy for debugging.",
        "oracle_answer_fallback",
    )


def build_risk_calibrated_template(row, include_probabilities=True):
    compact_evidence, source = compact_evidence_for_row(row)
    answer = str(row["answer"])
    nums = extract_numbers(answer)
    variables = {}
    if nums:
        variables["primary_numeric_value"] = nums[0]

    template = {
        "template_label": "oracle_template",
        "template_source": source,
        "task": {
            "question": row["question"],
            "entity": row.get("company", ""),
            "period": row.get("doc_period", ""),
            "metric": row.get("question_type", ""),
            "unit": "unknown",
        },
        "evidence_units": [
            {
                "claim": compact_evidence,
                "source": row.get("doc_name", ""),
            }
        ],
        "decision_variables": variables,
        "computation": {
            "formula": "not_provided",
            "inputs": variables,
            "result": answer,
        },
        "answer": answer,
    }

    if include_probabilities:
        template["evidence_units"][0].update(
            {
                "relevance_probability": 0.98,
                "support_probability": 0.98,
                "temporal_validity_probability": 0.95,
            }
        )
        template["answer_distribution"] = {
            answer: 0.94,
            "INSUFFICIENT_EVIDENCE": 0.03,
            "other": 0.03,
        }
        template["calibrated_confidence"] = 0.94
        template["abstain_probability"] = 0.03

    return template


# ---------------------------------------------------------------------------
# No-leakage (deployment) template construction.
#
# The oracle helpers above intentionally embed the gold answer to measure an
# upper bound. The functions below build templates from the raw evidence and
# question metadata ONLY. They must never read row["answer"], so that the
# main method claim is not answer-copying. This invariant is enforced by
# test_no_leakage() and by build_template_comparison's audit column.
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "what", "which", "how", "much", "many", "did", "does", "do",
    "that", "this", "with", "as", "at", "by", "from", "its", "it", "be", "been",
    "has", "have", "had", "s", "company", "year", "during", "period", "value",
    "amount", "total", "please", "based",
}


def _tokens(text):
    return [
        t
        for t in re.findall(r"[a-z0-9%.]+", str(text).lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def split_evidence_units(evidence_text, max_units=None):
    """Split raw evidence into sentence-level units, keeping the doc source."""
    if not evidence_text:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", str(evidence_text)) if b.strip()]
    units = []
    for block in blocks:
        source = ""
        body = block
        m = re.match(r"\[doc=([^\]]*?)(?:\s+page=[^\]]*)?\]\s*", block)
        if m:
            source = m.group(1).strip()
            body = block[m.end():]
        for sent in re.split(r"(?<=[.;])\s+(?=[A-Z0-9$(])", body):
            sent = sent.strip()
            if len(sent) >= 15:
                units.append({"text": sent, "source": source})
    if max_units:
        return units[:max_units]
    return units


def rank_evidence_units(question, units):
    """Rank evidence units by token overlap with the question (no answer used)."""
    q = set(_tokens(question))
    scored = []
    for u in units:
        toks = set(_tokens(u["text"]))
        overlap = len(q & toks)
        norm = overlap / max(1, len(q))
        scored.append((norm, overlap, u))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored


def extractive_summary(question, evidence_text, max_sentences=4, max_chars=None):
    """Length-matched compact summary drawn from raw evidence, no gold answer."""
    units = split_evidence_units(evidence_text)
    if not units:
        return ""
    scored = rank_evidence_units(question, units)
    top = [u["text"] for _, _, u in scored[:max_sentences]]
    summary = " ".join(top)
    if max_chars and len(summary) > max_chars:
        summary = summary[:max_chars] + " [TRUNCATED]"
    return summary


def build_rule_template(row, include_probabilities=True, max_units=6):
    """No-leakage template built only from raw evidence and question metadata.

    Never reads row["answer"]. The `answer` and `computation.result` fields are
    left as TO_BE_DETERMINED for the model to fill from the evidence units.
    Probabilities, when included, are heuristic priors derived from evidence
    coverage and unit overlap, not from knowledge of the gold answer.
    """
    question = row["question"]
    units = split_evidence_units(row.get("evidence_text", ""))
    scored = rank_evidence_units(question, units)
    kept = scored[:max_units]

    q_tokens = set(_tokens(question))
    covered = set()
    evidence_units = []
    candidate_values = []
    for norm, _overlap, u in kept:
        covered |= (set(_tokens(u["text"])) & q_tokens)
        nums = extract_numbers(u["text"])
        candidate_values.extend(nums[:4])
        unit = {"claim": u["text"], "source": u["source"]}
        if include_probabilities:
            unit["relevance_probability"] = round(min(0.95, 0.35 + 0.6 * norm), 3)
            unit["support_probability"] = round(
                min(0.9, 0.3 + 0.5 * norm + 0.1 * (len(nums) > 0)), 3
            )
            unit["temporal_validity_probability"] = 0.8
        evidence_units.append(unit)

    coverage = len(covered) / max(1, len(q_tokens))
    # De-duplicate candidate values while preserving order.
    seen = set()
    candidates = []
    for v in candidate_values:
        if v not in seen:
            seen.add(v)
            candidates.append(v)
    candidates = candidates[:8]

    template = {
        "template_label": "rule_template",
        "template_source": "rule_extraction_from_raw_evidence",
        "task": {
            "question": question,
            "entity": row.get("company", ""),
            "period": row.get("doc_period", ""),
            "metric": row.get("question_type", ""),
            "unit": "unknown",
        },
        "evidence_units": evidence_units,
        "decision_variables": {"candidate_values_from_evidence": candidates},
        "computation": {
            "formula": "not_provided",
            "inputs": {"candidate_values_from_evidence": candidates},
        },
    }

    if include_probabilities:
        abstain = round(max(0.02, 0.6 - 0.5 * coverage), 3)
        template["answer_distribution"] = {
            "ANSWER_FROM_EVIDENCE": round(1 - abstain - 0.03, 3),
            "INSUFFICIENT_EVIDENCE": abstain,
            "other": 0.03,
        }
        template["calibrated_confidence"] = round(min(0.9, 0.4 + 0.5 * coverage), 3)
        template["abstain_probability"] = abstain

    return template


def _raw_evidence_payload(row, max_evidence_chars):
    evidence_text = row["evidence_text"]
    if max_evidence_chars is not None and len(evidence_text) > max_evidence_chars:
        evidence_text = evidence_text[:max_evidence_chars] + "\n\n[TRUNCATED]"
    return f"Raw evidence:\n{evidence_text}"


def _missing_template_payload(question):
    template = {
        "template_label": "rule_template",
        "task": {"question": question},
        "evidence_units": [],
        "decision_variables": {},
        "answer_distribution": {"INSUFFICIENT_EVIDENCE": 0.94, "other": 0.06},
        "calibrated_confidence": 0.94,
        "abstain_probability": 0.94,
    }
    return (
        "Risk-calibrated evidence template:\n"
        f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
        "If evidence is insufficient, answer INSUFFICIENT_EVIDENCE."
    )


def _oracle_payload(row, condition):
    """Upper-bound conditions that intentionally embed the gold answer."""
    question = row["question"]
    if condition == "length_matched_summary":
        compact, source = compact_evidence_for_row(row)
        return (
            f"Length-matched compact summary ({source}):\n"
            f"{compact}\n"
            "This summary is not probabilistic. Use only the supplied summary."
        )
    if condition == "deterministic_trace":
        compact, source = compact_evidence_for_row(row)
        return (
            "Deterministic structured trace:\n"
            f"source: {source}\n"
            f"question: {question}\n"
            f"evidence_claim: {compact}\n"
            f"answer: {row['answer']}\n"
            "Use the trace to answer. The trace has no uncertainty fields."
        )
    if condition == "template_no_probabilities":
        template = build_risk_calibrated_template(row, include_probabilities=False)
        return (
            "Grounded evidence template without probabilities:\n"
            f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
            "Use the template to answer."
        )
    if condition == "risk_calibrated_template":
        template = build_risk_calibrated_template(row, include_probabilities=True)
        return (
            "Risk-calibrated evidence template:\n"
            f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
            "Use the template to answer. Your confidence should reflect the "
            "answer_distribution and evidence support."
        )
    raise ValueError(f"Unknown oracle condition: {condition}")


def _deployment_payload(row, condition, max_evidence_chars):
    """No-leakage conditions built only from raw evidence and metadata."""
    question = row["question"]
    if condition == "length_matched_summary":
        summary = extractive_summary(
            question, row["evidence_text"], max_sentences=4, max_chars=max_evidence_chars
        )
        return (
            "Length-matched extractive summary (top evidence sentences, no answer key):\n"
            f"{summary}\n"
            "This summary is not probabilistic. Use only the supplied summary."
        )
    if condition == "deterministic_trace":
        template = build_rule_template(row, include_probabilities=False)
        units_txt = "\n".join(f"- {u['claim']}" for u in template["evidence_units"])
        return (
            "Deterministic structured trace (extracted from raw evidence, "
            "no uncertainty fields, no answer key):\n"
            f"question: {question}\n"
            f"entity: {template['task']['entity']}\n"
            f"candidate_values: {template['decision_variables']['candidate_values_from_evidence']}\n"
            f"evidence:\n{units_txt}\n"
            "Compute the answer from the trace."
        )
    if condition == "template_no_probabilities":
        template = build_rule_template(row, include_probabilities=False)
        return (
            "Grounded evidence template without probabilities:\n"
            f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
            "The template does NOT contain the answer. Read the evidence_units and "
            "candidate_values_from_evidence, compute the answer yourself, and put the "
            "computed value in the answer field."
        )
    if condition == "risk_calibrated_template":
        template = build_rule_template(row, include_probabilities=True)
        return (
            "Risk-calibrated evidence template:\n"
            f"{json.dumps(template, ensure_ascii=False, indent=2)}\n"
            "The template does NOT contain the answer. Read the evidence_units and "
            "candidate_values_from_evidence, compute the answer yourself, and put the "
            "computed value in the answer field. Your confidence should reflect the "
            "support_probability of the evidence and the abstain_probability."
        )
    raise ValueError(f"Unknown deployment condition: {condition}")


def make_template_comparison_prompt(
    row, condition, leakage_mode="deployment", max_evidence_chars=None
):
    question = row["question"]
    target_instruction = "Return JSON with keys: answer, confidence, evidence_support, short_rationale."

    if condition == "raw_gold_evidence":
        payload = _raw_evidence_payload(row, max_evidence_chars)
    elif condition == "missing_risk_template":
        payload = _missing_template_payload(question)
    elif leakage_mode == "oracle":
        payload = _oracle_payload(row, condition)
    elif leakage_mode == "deployment":
        payload = _deployment_payload(row, condition, max_evidence_chars)
    else:
        raise ValueError(f"Unknown leakage_mode: {leakage_mode}")

    return f"""{SYSTEM_INSTRUCTION}

Question:
{question}

{payload}

{target_instruction}"""


def _answer_injected(prompt, condition, gold_answer, evidence_text, leakage_mode):
    """Audit flag: was the gold answer handed to the model beyond the evidence?

    A deployment template is built from raw evidence, so the gold value can
    legitimately appear in it because it appears in the evidence. That is
    grounded extraction, not leakage. So in deployment mode we only flag a row
    when the gold string is in the prompt but NOT in the raw evidence, i.e. it
    was injected from somewhere other than the evidence (a real bug/leak).

    In oracle mode we flag whenever the gold string is present, since the oracle
    conditions intentionally embed it as an upper-bound signal. raw_gold_evidence
    is always exempt.
    """
    if condition == "raw_gold_evidence":
        return False
    gold = normalize_text(gold_answer)
    if len(gold) < 4:
        return False
    in_prompt = gold in normalize_text(prompt)
    if leakage_mode == "oracle":
        return in_prompt
    return in_prompt and gold not in normalize_text(evidence_text)


def build_template_comparison(
    df, n_examples=5, random_state=7, leakage_mode="deployment", max_evidence_chars=6000
):
    base = df[df["has_numeric_answer"]].sample(
        min(n_examples, int(df["has_numeric_answer"].sum())),
        random_state=random_state,
    )
    conditions = [
        "raw_gold_evidence",
        "length_matched_summary",
        "deterministic_trace",
        "template_no_probabilities",
        "risk_calibrated_template",
        "missing_risk_template",
    ]

    rows = []
    for _, row in base.iterrows():
        for condition in conditions:
            expected_behavior = "abstain" if condition == "missing_risk_template" else "answer"
            target_answer = (
                "INSUFFICIENT_EVIDENCE"
                if expected_behavior == "abstain"
                else row["answer"]
            )
            if condition == "raw_gold_evidence":
                source = "raw_evidence"
            elif condition == "missing_risk_template":
                source = "missing"
            elif leakage_mode == "oracle":
                source = compact_evidence_for_row(row)[1]
            else:
                source = "rule_extraction_from_raw_evidence"
            prompt = make_template_comparison_prompt(
                row, condition, leakage_mode=leakage_mode, max_evidence_chars=max_evidence_chars
            )
            rows.append(
                {
                    "financebench_id": row["financebench_id"],
                    "condition": condition,
                    "leakage_mode": leakage_mode,
                    "template_source": source,
                    "expected_behavior": expected_behavior,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "target_answer": target_answer,
                    "answer_injected": _answer_injected(
                        prompt, condition, row["answer"], row["evidence_text"], leakage_mode
                    ),
                    "prompt": prompt,
                }
            )
    return pd.DataFrame(rows)


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
                "compression_source": row.get("compression_source", ""),
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


def run_hf_template_comparison(
    df,
    n_examples=3,
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    random_state=7,
    max_new_tokens=160,
    leakage_mode="deployment",
    max_evidence_chars=6000,
):
    generator = load_hf_generator(model_id=model_id, max_new_tokens=max_new_tokens)
    comparison_df = build_template_comparison(
        df,
        n_examples=n_examples,
        random_state=random_state,
        leakage_mode=leakage_mode,
        max_evidence_chars=max_evidence_chars,
    )

    rows = []
    for _, row in tqdm(comparison_df.iterrows(), total=len(comparison_df)):
        pred = call_hf_generator(generator, row["prompt"])
        scores = score_prediction(pred, row["target_answer"])
        template_success = (
            scores["refusal"]
            if row["expected_behavior"] == "abstain"
            else scores["weak_match_answer"]
        )
        rows.append(
            {
                "financebench_id": row["financebench_id"],
                "condition": row["condition"],
                "leakage_mode": row["leakage_mode"],
                "template_source": row["template_source"],
                "expected_behavior": row["expected_behavior"],
                "answer_injected": row["answer_injected"],
                "model_id": model_id,
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "target_answer": row["target_answer"],
                "prediction": pred,
                "template_success": bool(template_success),
                "weak_match": scores["weak_match_answer"],
                **scores,
            }
        )

    results = pd.DataFrame(rows)
    summary = summarize_template_results(results, ["model_id", "condition"])
    return results, summary


def summarize_template_results(results, group_cols):
    metric_cols = [
        "template_success",
        "weak_match_answer",
        "numeric_match_answer",
        "refusal",
        "confidence",
        "brier",
        "overconfident_wrong",
    ]
    metric_cols = [col for col in metric_cols if col in results.columns]
    summary = results.groupby(group_cols)[metric_cols].mean()
    summary = summary.rename(
        columns={
            "template_success": "success_rate",
            "weak_match_answer": "answer_weak_accuracy",
            "numeric_match_answer": "answer_numeric_accuracy",
            "refusal": "refusal_rate",
            "confidence": "avg_confidence",
            "brier": "brier_score",
            "overconfident_wrong": "overconfident_wrong_rate",
        }
    )
    summary["n"] = results.groupby(group_cols).size()
    summary["ece"] = results.groupby(group_cols).apply(
        lambda group: expected_calibration_error(group)
    )
    return summary


def summarize_results(results, group_cols):
    metric_cols = [
        "weak_match_raw",
        "weak_match_answer",
        "numeric_match_answer",
        "refusal",
        "confidence",
        "brier",
        "overconfident_wrong",
    ]
    metric_cols = [col for col in metric_cols if col in results.columns]
    summary = results.groupby(group_cols)[metric_cols].mean()
    summary = summary.rename(
        columns={
            "weak_match_raw": "weak_accuracy_raw",
            "weak_match_answer": "weak_accuracy_answer",
            "numeric_match_answer": "numeric_accuracy_answer",
            "refusal": "refusal_rate",
            "confidence": "avg_confidence",
            "brier": "brier_score",
            "overconfident_wrong": "overconfident_wrong_rate",
        }
    )
    summary["n"] = results.groupby(group_cols).size()
    summary["ece"] = results.groupby(group_cols).apply(
        lambda group: expected_calibration_error(group)
    )
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


# ---------------------------------------------------------------------------
# Multi-seed / multi-model template comparison suite.
#
# Loads each model once, then sweeps leakage modes and seeds, reusing the
# generator. Returns a flat per-example results frame plus aggregation helpers
# that report mean +/- std across seeds for each (model, leakage_mode,
# condition). This produces the paper-grade template reliability table.
# ---------------------------------------------------------------------------

TEMPLATE_METRIC_COLS = [
    "template_success",
    "weak_match_answer",
    "numeric_match_answer",
    "refusal",
    "confidence",
    "brier",
    "overconfident_wrong",
]


def _free_accelerator():
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_template_comparison_suite(
    df,
    model_ids=("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    seeds=(7, 13, 29),
    n_examples=50,
    leakage_modes=("deployment", "oracle"),
    max_new_tokens=160,
    max_evidence_chars=6000,
):
    all_rows = []
    for model_id in model_ids:
        generator = load_hf_generator(model_id=model_id, max_new_tokens=max_new_tokens)
        try:
            for leakage_mode in leakage_modes:
                for seed in seeds:
                    comparison_df = build_template_comparison(
                        df,
                        n_examples=n_examples,
                        random_state=seed,
                        leakage_mode=leakage_mode,
                        max_evidence_chars=max_evidence_chars,
                    )
                    short = model_id.split("/")[-1]
                    for _, row in tqdm(
                        comparison_df.iterrows(),
                        total=len(comparison_df),
                        desc=f"{short} {leakage_mode} seed{seed}",
                    ):
                        pred = call_hf_generator(generator, row["prompt"])
                        scores = score_prediction(pred, row["target_answer"])
                        success = (
                            scores["refusal"]
                            if row["expected_behavior"] == "abstain"
                            else scores["weak_match_answer"]
                        )
                        all_rows.append(
                            {
                                "model_id": model_id,
                                "leakage_mode": leakage_mode,
                                "seed": seed,
                                "financebench_id": row["financebench_id"],
                                "condition": row["condition"],
                                "template_source": row["template_source"],
                                "expected_behavior": row["expected_behavior"],
                                "answer_injected": row["answer_injected"],
                                "question": row["question"],
                                "gold_answer": row["gold_answer"],
                                "target_answer": row["target_answer"],
                                "prediction": pred,
                                "template_success": bool(success),
                                "weak_match": scores["weak_match_answer"],
                                **scores,
                            }
                        )
        finally:
            del generator
            _free_accelerator()

    return pd.DataFrame(all_rows)


def aggregate_template_suite(results, metric_cols=None):
    """Aggregate suite results to mean +/- std across seeds.

    Returns (agg, per_seed) where agg is indexed by
    (model_id, leakage_mode, condition) with (metric, {mean,std}) columns.
    """
    if metric_cols is None:
        metric_cols = TEMPLATE_METRIC_COLS
    metric_cols = [c for c in metric_cols if c in results.columns]
    group = ["model_id", "leakage_mode", "condition"]

    per_seed = results.groupby(group + ["seed"])[metric_cols].mean()
    ece = results.groupby(group + ["seed"]).apply(
        lambda g: expected_calibration_error(g)
    )
    per_seed["ece"] = ece
    per_seed["answer_injected_rate"] = results.groupby(group + ["seed"])[
        "answer_injected"
    ].mean()

    agg = per_seed.groupby(group).agg(["mean", "std"])
    agg["n_seeds"] = results.groupby(group)["seed"].nunique()
    agg["n_rows"] = results.groupby(group).size()
    return agg, per_seed


def format_template_suite_table(agg, keys=None):
    """Readable mean+/-std table for the headline reliability metrics."""
    if keys is None:
        keys = [
            "template_success",
            "confidence",
            "brier",
            "ece",
            "overconfident_wrong",
            "refusal",
            "answer_injected_rate",
        ]
    rows = []
    for idx, r in agg.iterrows():
        model_id, leakage_mode, condition = idx
        entry = {
            "model": model_id.split("/")[-1],
            "mode": leakage_mode,
            "condition": condition,
            "n_seeds": int(r[("n_seeds", "")]) if ("n_seeds", "") in agg.columns else "",
        }
        for k in keys:
            if (k, "mean") in agg.columns:
                m = r[(k, "mean")]
                s = r[(k, "std")]
                s = 0.0 if pd.isna(s) else s
                entry[k] = f"{m:.3f}+/-{s:.3f}"
        rows.append(entry)
    order = {
        "raw_gold_evidence": 0,
        "length_matched_summary": 1,
        "deterministic_trace": 2,
        "template_no_probabilities": 3,
        "risk_calibrated_template": 4,
        "missing_risk_template": 5,
    }
    table = pd.DataFrame(rows)
    table["_o"] = table["condition"].map(order).fillna(9)
    table = table.sort_values(["model", "mode", "_o"]).drop(columns="_o")
    return table.reset_index(drop=True)


def check_deployment_leakage(df, n_examples=25, seed=7):
    """Sanity check: no deployment condition injects the full gold answer.

    Returns a small frame of any violations (should be empty).
    """
    comparison_df = build_template_comparison(
        df, n_examples=n_examples, random_state=seed, leakage_mode="deployment"
    )
    violations = comparison_df[
        (comparison_df["condition"] != "raw_gold_evidence")
        & (comparison_df["answer_injected"])
    ]
    return violations[["financebench_id", "condition", "gold_answer"]]
