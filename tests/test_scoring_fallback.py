"""Smoke test: keyword fallback produces deterministic scores in [0, 1]."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import scoring


def test_keyword_fallback_is_deterministic():
    s = pd.Series([
        "我对垃圾分类设施严重不满，希望政府处理。",
        "我们愿意配合街道办志愿协助宣传活动。",
    ], index=[1, 2])
    protos = pd.DataFrame({
        "condition_name": ["dissatisfaction", "coproduction_request"],
        "prototype": ["不满或愤怒", "愿意合作"],
        "type": ["condition", "condition"],
    })
    r1 = scoring.score_texts(s, protos, method="keyword")
    r2 = scoring.score_texts(s, protos, method="keyword")
    pd.testing.assert_frame_equal(r1.raw, r2.raw)
    assert (r1.raw.values >= 0).all() and (r1.raw.values <= 1).all()


def test_keyword_signal_separates_concepts():
    s = pd.Series([
        "对此非常不满，部门长期不作为令人失望。",
        "我们愿意配合志愿者活动，提供协助。",
    ], index=[1, 2])
    protos = pd.DataFrame({
        "condition_name": ["dissatisfaction", "coproduction_request"],
        "prototype": ["不满", "愿意合作"],
        "type": ["condition", "condition"],
    })
    r = scoring.score_texts(s, protos, method="keyword")
    # Case 1 should score higher on dissatisfaction than coproduction
    assert r.raw.loc[1, "dissatisfaction"] > r.raw.loc[1, "coproduction_request"]
    # Case 2 should score higher on coproduction than dissatisfaction
    assert r.raw.loc[2, "coproduction_request"] > r.raw.loc[2, "dissatisfaction"]
