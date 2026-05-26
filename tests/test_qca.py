"""Unit tests for src.qca — checks textbook examples."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import qca as q


def test_necessity_textbook_example():
    # Outcome 1 only in rows where X is fully in.
    x = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    out = q.necessity(x, y)
    assert pytest.approx(out["consistency"], abs=1e-6) == 1.0
    assert pytest.approx(out["coverage"], abs=1e-6) == 1.0


def test_sufficiency_consistency_partial():
    x = np.array([1.0, 1.0, 0.0])
    y = np.array([1.0, 0.0, 0.0])
    out = q.sufficiency(x, y)
    # consistency = min(1,1)+min(1,0)+min(0,0) / sum(x) = 1 / 2 = 0.5
    assert pytest.approx(out["consistency"], abs=1e-6) == 0.5


def test_quine_mccluskey_two_var():
    # Three-of-four corners of (A, B): 01, 10, 11 → outcome = A + B
    primes = q.quine_mccluskey(["01", "10", "11"])
    assert sorted(primes) == ["-1", "1-"]


def test_quine_mccluskey_three_var():
    # 010, 011, 110, 111 collapse to -1- (i.e., middle bit set)
    primes = q.quine_mccluskey(["010", "011", "110", "111"])
    assert sorted(primes) == ["-1-"]


def test_essential_cover_picks_essential():
    primes = ["1-", "-1"]
    minterms = ["10", "11", "01"]
    # 10 only covered by 1- → essential; 01 only covered by -1 → essential.
    chosen = q.essential_cover(primes, minterms)
    assert sorted(chosen) == ["-1", "1-"]


def test_prime_to_expression():
    out = q.prime_to_expression("1-0", ["A", "B", "C"])
    assert out == "A * ~C"


def test_truth_table_structure():
    df = pd.DataFrame({
        "A": [0.9, 0.8, 0.1, 0.2],
        "B": [0.9, 0.1, 0.9, 0.1],
    })
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    tt = q.truth_table(df, y, consistency_cutoff=0.8, frequency_cutoff=1)
    # 2 conditions → 4 corners
    assert len(tt) == 4
    assert set(tt.columns) >= {"A", "B", "n", "consistency", "raw_coverage", "outcome", "flag"}


def test_run_qca_end_to_end_demo():
    # Plant a clear signal: outcome=1 iff A=1 or B=1 (and ~D otherwise)
    rng = np.random.default_rng(0)
    n = 20
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, n)
    c = rng.uniform(0, 1, n)
    y = np.clip(np.maximum(a, b) + rng.normal(0, 0.05, n), 0, 1)
    memb = pd.DataFrame({"A": a, "B": b, "C": c})
    outc = pd.Series(y, name="outcome")
    res = q.run_qca(memb, outc, consistency_cutoff=0.75, frequency_cutoff=1)
    assert res.necessity is not None
    assert res.solution_metrics["overall_consistency"] >= 0.7
