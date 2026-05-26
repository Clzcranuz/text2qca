"""Unit tests for src.calibration — verifies direct method against anchors."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import calibration as cal


def test_direct_method_anchors():
    x = np.array([0.2, 0.5, 0.8])
    m = cal._direct_calibration(x, 0.2, 0.5, 0.8)
    # full-out → 0.05, cross → 0.50, full-in → 0.95
    assert pytest.approx(m[0], abs=0.02) == 0.05
    assert pytest.approx(m[1], abs=0.02) == 0.50
    assert pytest.approx(m[2], abs=0.02) == 0.95


def test_direct_method_monotonic():
    x = np.linspace(0, 1, 50)
    m = cal._direct_calibration(x, 0.2, 0.5, 0.8)
    assert np.all(np.diff(m) >= -1e-9)


def test_direct_anchors_must_be_ordered():
    with pytest.raises(ValueError):
        cal._direct_calibration(np.array([0.5]), 0.6, 0.5, 0.4)


def test_calibrate_scores_returns_same_index():
    df = pd.DataFrame({"A": [0.1, 0.5, 0.9], "B": [0.2, 0.4, 0.8]},
                      index=["c1", "c2", "c3"])
    specs = [cal.CalibrationSpec(condition=c) for c in df.columns]
    m = cal.calibrate_scores(df, specs)
    assert list(m.index) == ["c1", "c2", "c3"]
    assert list(m.columns) == ["A", "B"]


def test_threshold_method_yields_binary():
    df = pd.DataFrame({"A": [0.1, 0.5, 0.9]})
    spec = cal.CalibrationSpec(condition="A", method="threshold",
                               crisp_cutoff=0.5)
    m = cal.calibrate_scores(df, [spec])
    assert sorted(m["A"].unique().tolist()) == [0.0, 1.0]


def test_percentile_method_runs():
    df = pd.DataFrame({"A": np.linspace(0, 1, 50)})
    spec = cal.CalibrationSpec(condition="A", method="percentile",
                               anchors=(0.10, 0.50, 0.90))
    m = cal.calibrate_scores(df, [spec])
    assert m["A"].min() >= 0 and m["A"].max() <= 1


def test_inverted_calibration():
    df = pd.DataFrame({"A": [0.1, 0.9]})
    spec = cal.CalibrationSpec(condition="A", method="threshold",
                               crisp_cutoff=0.5, invert=True)
    m = cal.calibrate_scores(df, [spec])
    assert m["A"].tolist() == [1.0, 0.0]
