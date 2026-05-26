"""
Text-to-condition scoring.

Two methods are supported:

1. ``embedding``  (default) — Sentence-Transformer embedding cosine similarity
   between each text and each condition prototype. Recommended Chinese models:

       - BAAI/bge-small-zh-v1.5   (~95 MB, fast, very strong on CMTEB)
       - shibing624/text2vec-base-chinese (~400 MB, classic baseline)

2. ``nli``  — Zero-shot natural-language-inference classification, where each
   prototype is rewritten as a hypothesis ("这段文本表达了……") and the entailment
   probability is used as the condition score. Recommended model:

       - MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingual, handles Chinese)

A keyword-based **deterministic fallback** is used automatically when neither
model can be loaded (no network, no GPU, locked sandbox). The fallback is
documented in :func:`_keyword_fallback_score` and produces reproducible scores
so that the QCA outputs shipped in ``outputs/sample/`` can be regenerated on
any machine.

All scores are returned as a ``pandas.DataFrame`` indexed by ``case_id`` with
one column per condition. Values are clipped to ``[0, 1]``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ScoringMethod = Literal["embedding", "nli", "keyword"]

DEFAULT_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


# ---------------------------------------------------------------------------
# Keyword fallback: deterministic, offline-friendly, transparent
# ---------------------------------------------------------------------------

# Each entry maps a *concept* to (positive_terms, negative_terms). Positive
# terms increase the score, negative terms decrease it. The dictionary is
# intentionally short and human-readable so reviewers can inspect why a given
# text scored high or low even when the embedding model is offline.
KEYWORD_LEXICON: dict[str, tuple[list[str], list[str]]] = {
    "dissatisfaction": (
        [
            "不满", "失望", "愤慨", "扰民", "推诿", "无人", "不作为",
            "睁一只眼", "敷衍", "拖", "无奈", "投诉", "毫无", "忍无可忍",
            "严重", "推搡", "气愤", "极度", "受损", "互相推诿", "久拖不决",
        ],
        ["希望", "愿意", "请问"],
    ),
    "policy_demand": (
        [
            "希望", "请问", "明确", "细则", "流程", "标准", "申请", "说明",
            "指引", "公开", "公告", "认定", "办理", "审批", "解释",
            "说明文件", "整理", "材料清单",
        ],
        ["愿意配合", "投诉", "推诿"],
    ),
    "coproduction_request": (
        [
            "愿意", "配合", "志愿", "协助", "组织", "提供", "合作", "承担",
            "排班", "团体", "协会", "志愿者", "共同", "纳入", "合作伙伴",
            "协调", "志愿教学", "互助",
        ],
        ["投诉", "不满", "扰民"],
    ),
    "responsiveness": (
        ["回应", "回复", "答复", "解决", "处理", "已", "已经"],
        ["无人", "推诿", "不作为"],
    ),
}


def _keyword_fallback_score(texts: Iterable[str], condition_name: str) -> np.ndarray:
    """Deterministic lexicon score in [0, 1] for a single condition.

    The score is ``sigmoid( (pos_hits - neg_hits) / 3 )`` so that texts with
    several positive cues land in the 0.7–0.95 range, texts with no cues
    hover around 0.5, and texts with strong negative cues drop below 0.2.
    """
    pos, neg = KEYWORD_LEXICON.get(condition_name, ([], []))
    out = []
    for t in texts:
        if not isinstance(t, str):
            t = ""
        p = sum(1 for w in pos if w in t)
        n = sum(1 for w in neg if w in t)
        raw = (p - n) / 3.0
        out.append(1.0 / (1.0 + np.exp(-raw)))
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


@dataclass
class ScoringResult:
    """Holds raw and standardised score tables plus per-condition diagnostics."""

    raw: pd.DataFrame
    standardised: pd.DataFrame
    method_used: str
    fallback_used: bool
    model_name: Optional[str]
    diagnostics: dict


def score_texts(
    texts: pd.Series,
    prototypes: pd.DataFrame,
    method: ScoringMethod = "embedding",
    model_name: Optional[str] = None,
    use_outcome_prototype: bool = False,
) -> ScoringResult:
    """Score each text against each prototype.

    Parameters
    ----------
    texts : pandas.Series
        Series of strings indexed by case_id.
    prototypes : pandas.DataFrame
        Must contain columns ``condition_name``, ``prototype``, ``type``.
    method : {"embedding", "nli", "keyword"}
        Scoring backend. Defaults to ``embedding``.
    model_name : str, optional
        Override the default model for the chosen method.
    use_outcome_prototype : bool
        If False (default), rows with ``type == "outcome"`` are excluded
        from the condition score table. The outcome column is taken from
        the input dataset.
    """
    cond_mask = prototypes["type"].str.lower() == "condition"
    if use_outcome_prototype:
        proto_df = prototypes.copy()
    else:
        proto_df = prototypes[cond_mask].copy()

    condition_names = proto_df["condition_name"].tolist()
    proto_sentences = proto_df["prototype"].tolist()

    raw_scores: pd.DataFrame
    fallback_used = False
    model_used: Optional[str] = None

    if method == "keyword":
        raw_scores = _score_with_keywords(texts, condition_names)
        model_used = "keyword-lexicon"
    elif method == "embedding":
        raw_scores, model_used, fallback_used = _score_with_embeddings(
            texts, proto_sentences, condition_names,
            model_name=model_name or DEFAULT_EMBED_MODEL,
        )
    elif method == "nli":
        raw_scores, model_used, fallback_used = _score_with_nli(
            texts, proto_sentences, condition_names,
            model_name=model_name or DEFAULT_NLI_MODEL,
        )
    else:
        raise ValueError(f"Unknown scoring method: {method!r}")

    # Min-max standardisation per column so calibration can apply percentile
    # rules consistently across methods that produce scores on different scales
    # (cosine similarity ≈ 0.3–0.8; NLI entailment ≈ 0–1; keyword ≈ 0–1).
    standardised = raw_scores.apply(_min_max_within_column, axis=0)

    diagnostics = {
        "n_cases": len(texts),
        "n_conditions": len(condition_names),
        "score_summary": raw_scores.describe().to_dict(),
    }

    return ScoringResult(
        raw=raw_scores,
        standardised=standardised,
        method_used=method if not fallback_used else "keyword (fallback)",
        fallback_used=fallback_used,
        model_name=model_used,
        diagnostics=diagnostics,
    )


def _min_max_within_column(col: pd.Series) -> pd.Series:
    lo, hi = col.min(), col.max()
    if hi - lo < 1e-9:
        return pd.Series(np.full(len(col), 0.5), index=col.index)
    return (col - lo) / (hi - lo)


def _score_with_keywords(texts: pd.Series, condition_names: list[str]) -> pd.DataFrame:
    cols = {}
    for c in condition_names:
        cols[c] = _keyword_fallback_score(texts.tolist(), c)
    return pd.DataFrame(cols, index=texts.index)


def _score_with_embeddings(
    texts: pd.Series,
    proto_sentences: list[str],
    condition_names: list[str],
    *,
    model_name: str,
) -> tuple[pd.DataFrame, str, bool]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
        text_emb = model.encode(texts.tolist(), normalize_embeddings=True, show_progress_bar=False)
        proto_emb = model.encode(proto_sentences, normalize_embeddings=True, show_progress_bar=False)
        sim = np.asarray(text_emb) @ np.asarray(proto_emb).T  # cosine since L2-normed
        # rescale cosine in [-1, 1] to [0, 1] via (x + 1) / 2
        scores = (sim + 1.0) / 2.0
        df = pd.DataFrame(scores, index=texts.index, columns=condition_names)
        return df, model_name, False
    except Exception as exc:  # pragma: no cover — exercised when model unavailable
        log.warning("Embedding model unavailable (%s); falling back to keyword scoring.", exc)
        return _score_with_keywords(texts, condition_names), "keyword-lexicon", True


def _score_with_nli(
    texts: pd.Series,
    proto_sentences: list[str],
    condition_names: list[str],
    *,
    model_name: str,
) -> tuple[pd.DataFrame, str, bool]:
    try:
        from transformers import pipeline  # type: ignore

        clf = pipeline("zero-shot-classification", model=model_name)
        hypotheses = [f"这段文本表达了：{p}" for p in proto_sentences]
        labels = condition_names
        out_rows = []
        for t in texts:
            res = clf(t, candidate_labels=labels, hypothesis_template="{}",
                      multi_label=True)
            label_to_score = dict(zip(res["labels"], res["scores"]))
            out_rows.append([label_to_score[c] for c in labels])
        df = pd.DataFrame(out_rows, index=texts.index, columns=condition_names)
        return df, model_name, False
    except Exception as exc:  # pragma: no cover — exercised when model unavailable
        log.warning("NLI model unavailable (%s); falling back to keyword scoring.", exc)
        return _score_with_keywords(texts, condition_names), "keyword-lexicon", True


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def explain_keyword_match(text: str, condition_name: str) -> dict:
    """Return positive/negative cue counts for transparency in the UI."""
    pos, neg = KEYWORD_LEXICON.get(condition_name, ([], []))
    pos_hits = [w for w in pos if isinstance(text, str) and w in text]
    neg_hits = [w for w in neg if isinstance(text, str) and w in text]
    return {"positive_hits": pos_hits, "negative_hits": neg_hits}
