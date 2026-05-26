"""
Calibration from raw / standardised condition scores to set-membership values.

This module implements three calibration strategies that researchers can pick
from in the UI:

* ``direct``   — Ragin's (2008) **direct** method based on three anchors
                 (full non-membership, crossover, full membership). The
                 mapping uses a log-odds transformation pinned to the
                 thresholds, producing genuinely fuzzy membership scores.

* ``percentile`` — Anchors are drawn from data percentiles (default 5/50/95).
                   This is useful when the researcher does not yet have
                   substantive thresholds.

* ``threshold`` — A simple crisp-set rule (``score >= cutoff → 1`` else 0).

All methods can be applied independently per condition with their own anchor
set. The module's public entry point is :func:`calibrate_scores` which takes a
score table plus a list of :class:`CalibrationSpec` objects and returns a
DataFrame of membership scores in ``[0, 1]``.

References
----------
Ragin, C. C. (2008). *Redesigning Social Inquiry: Fuzzy Sets and Beyond.*
Schneider, C. Q., & Wagemann, C. (2012). *Set-Theoretic Methods for the Social
Sciences.* Cambridge University Press, ch. 4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

CalibrationMethod = Literal["direct", "percentile", "threshold"]


@dataclass
class CalibrationSpec:
    """Per-condition calibration configuration."""

    condition: str
    method: CalibrationMethod = "direct"
    # For ``direct`` / ``percentile``: (full_out, crossover, full_in)
    anchors: tuple[float, float, float] = (0.05, 0.50, 0.95)
    # For ``threshold``: crisp cutoff
    crisp_cutoff: float = 0.5
    # ``percentile`` interprets ``anchors`` as percentile positions, then
    # converts them to value-anchors at calibration time.
    invert: bool = False  # set True for "absence" conditions

    def describe(self) -> str:
        if self.method == "threshold":
            return f"crisp ≥ {self.crisp_cutoff:.2f}"
        a = self.anchors
        if self.method == "percentile":
            return f"percentile anchors p{int(a[0]*100)}/p{int(a[1]*100)}/p{int(a[2]*100)}"
        return f"direct anchors out={a[0]:.2f} / cross={a[1]:.2f} / in={a[2]:.2f}"


def calibrate_scores(
    scores: pd.DataFrame,
    specs: Iterable[CalibrationSpec],
) -> pd.DataFrame:
    """Apply calibration specs to a score table.

    The returned DataFrame has the same index as ``scores`` and one column per
    spec, with values in [0, 1].
    """
    out = {}
    for spec in specs:
        col = scores[spec.condition]
        if spec.method == "direct":
            m = _direct_calibration(col.values, *spec.anchors)
        elif spec.method == "percentile":
            anchors = tuple(np.quantile(col.values, spec.anchors))  # type: ignore[arg-type]
            m = _direct_calibration(col.values, *anchors)
        elif spec.method == "threshold":
            m = (col.values >= spec.crisp_cutoff).astype(float)
        else:
            raise ValueError(f"Unknown calibration method: {spec.method}")
        if spec.invert:
            m = 1.0 - m
        out[spec.condition] = pd.Series(m, index=col.index)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Direct method (Ragin 2008)
# ---------------------------------------------------------------------------

def _direct_calibration(
    x: np.ndarray,
    full_out: float,
    crossover: float,
    full_in: float,
) -> np.ndarray:
    """Fuzzy direct calibration.

    Pieces a smooth log-odds curve onto three theoretical anchors:

    * ``full_out``: membership = 0.05
    * ``crossover``: membership = 0.50
    * ``full_in``: membership = 0.95

    Implementation follows Ragin (2008, ch. 5) with the log-odds metric.
    Values are clipped to [0, 1] at the end.
    """
    if not (full_out < crossover < full_in):
        raise ValueError(
            f"Anchors must satisfy full_out < crossover < full_in; got "
            f"({full_out}, {crossover}, {full_in})"
        )

    # log-odds scale: ln(0.95/0.05) = ln(19); ln(0.05/0.95) = -ln(19)
    LO_FULL = np.log(0.95 / 0.05)  # ≈ 2.944

    x = np.asarray(x, dtype=float)
    log_odds = np.where(
        x >= crossover,
        # upper branch — scale so x = full_in maps to +LO_FULL
        LO_FULL * (x - crossover) / max(full_in - crossover, 1e-12),
        # lower branch — scale so x = full_out maps to -LO_FULL
        -LO_FULL * (crossover - x) / max(crossover - full_out, 1e-12),
    )
    membership = 1.0 / (1.0 + np.exp(-log_odds))
    return np.clip(membership, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Auto-suggest anchors
# ---------------------------------------------------------------------------

def suggest_anchors(scores: pd.Series, method: CalibrationMethod = "percentile") -> tuple[float, float, float]:
    """Return a sensible default anchor triple for a given score column."""
    if method == "percentile":
        return tuple(np.quantile(scores.values, (0.10, 0.50, 0.90)).tolist())  # type: ignore[return-value]
    lo, hi = float(scores.min()), float(scores.max())
    span = max(hi - lo, 1e-6)
    return (lo + 0.15 * span, lo + 0.50 * span, lo + 0.85 * span)


def crispify(membership: pd.DataFrame, cutoff: float = 0.5) -> pd.DataFrame:
    """Convert fuzzy membership to crisp 0/1 at a given cutoff."""
    return (membership >= cutoff).astype(int)
