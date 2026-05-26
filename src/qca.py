"""
Fuzzy-set / crisp-set QCA core.

Implements the standard set-theoretic procedure described in Ragin (2008) and
Schneider & Wagemann (2012):

1. **Necessity** for each condition X (and its negation ~X) w.r.t. outcome Y::

       consistency(X ⇐ Y) = Σ min(Xᵢ, Yᵢ) / Σ Yᵢ
       coverage(X ⇐ Y)    = Σ min(Xᵢ, Yᵢ) / Σ Xᵢ

2. **Sufficiency** for each condition X w.r.t. Y::

       consistency(X ⇒ Y) = Σ min(Xᵢ, Yᵢ) / Σ Xᵢ
       coverage(X ⇒ Y)    = Σ min(Xᵢ, Yᵢ) / Σ Yᵢ

3. **Truth table** — fuzzy memberships are crispified at 0.5, every row is
   assigned to its closest configuration corner, and per-row consistency is
   computed as ``Σ min(corner_mᵢ, Yᵢ) / Σ corner_mᵢ`` where the corner
   membership is the minimum of (Xⱼ or 1-Xⱼ) across all conditions.

4. **Minimisation** — minterms (configurations whose outcome is coded 1) are
   minimised with the **Quine–McCluskey** algorithm to produce a parsimonious
   solution. The implementation is purely Python so the tool ships with no
   dependency on R's ``QCA`` package.

5. **Solution metrics** — each prime implicant in the solution receives a
   per-term consistency, raw coverage and unique coverage; the solution as a
   whole receives an overall consistency and coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Necessity / sufficiency
# ---------------------------------------------------------------------------

def _fuzzy_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)


def necessity(condition: np.ndarray, outcome: np.ndarray) -> dict:
    """Return necessity consistency & coverage for a single condition."""
    num = float(_fuzzy_and(condition, outcome).sum())
    cons = num / float(outcome.sum()) if outcome.sum() > 0 else 0.0
    cov = num / float(condition.sum()) if condition.sum() > 0 else 0.0
    return {"consistency": cons, "coverage": cov}


def sufficiency(condition: np.ndarray, outcome: np.ndarray) -> dict:
    """Return sufficiency consistency & coverage for a single condition."""
    num = float(_fuzzy_and(condition, outcome).sum())
    cons = num / float(condition.sum()) if condition.sum() > 0 else 0.0
    cov = num / float(outcome.sum()) if outcome.sum() > 0 else 0.0
    return {"consistency": cons, "coverage": cov}


def necessity_table(
    membership: pd.DataFrame,
    outcome: pd.Series,
    include_negation: bool = True,
) -> pd.DataFrame:
    """Necessity test for every condition (and optionally its negation)."""
    rows = []
    y = outcome.values.astype(float)
    for c in membership.columns:
        x = membership[c].values.astype(float)
        rows.append({"term": c, **necessity(x, y)})
        if include_negation:
            rows.append({"term": f"~{c}", **necessity(1.0 - x, y)})
    df = pd.DataFrame(rows).sort_values("consistency", ascending=False).reset_index(drop=True)
    df["necessary?"] = df["consistency"] >= 0.9
    return df


def sufficiency_table(
    membership: pd.DataFrame,
    outcome: pd.Series,
    include_negation: bool = True,
) -> pd.DataFrame:
    rows = []
    y = outcome.values.astype(float)
    for c in membership.columns:
        x = membership[c].values.astype(float)
        rows.append({"term": c, **sufficiency(x, y)})
        if include_negation:
            rows.append({"term": f"~{c}", **sufficiency(1.0 - x, y)})
    df = pd.DataFrame(rows).sort_values("consistency", ascending=False).reset_index(drop=True)
    df["sufficient?"] = df["consistency"] >= 0.8
    return df


# ---------------------------------------------------------------------------
# Truth table
# ---------------------------------------------------------------------------

def _corner_membership(memb_row: np.ndarray, corner: tuple[int, ...]) -> float:
    """Fuzzy membership of a case in a truth-table corner.

    ``corner`` is a tuple of 0/1 with length = number of conditions. The
    corner membership is the minimum of (Xⱼ if corner[j]==1 else 1-Xⱼ).
    """
    vals = np.where(np.asarray(corner) == 1, memb_row, 1.0 - memb_row)
    return float(vals.min())


def truth_table(
    membership: pd.DataFrame,
    outcome: pd.Series,
    *,
    consistency_cutoff: float = 0.80,
    frequency_cutoff: int = 1,
) -> pd.DataFrame:
    """Build a fuzzy-set truth table.

    Each row of the returned DataFrame corresponds to a configuration corner
    of the property space. The table reports:

    * Each condition (0/1)
    * ``n`` — number of cases assigned to that corner (membership > 0.5 in
              every condition cell)
    * ``consistency`` — sufficiency consistency of the corner
    * ``raw_coverage`` — sufficiency coverage of the corner
    * ``outcome`` — 1 if ``consistency >= consistency_cutoff`` and
                    ``n >= frequency_cutoff`` else 0
    * ``flag`` — ``"contradictory"`` if both outcome=1 and outcome=0 cases
                 share the same corner, ``"low_n"`` if n<frequency_cutoff,
                 else empty.
    """
    cond_names = list(membership.columns)
    k = len(cond_names)
    rows = []
    crisp_assignment = (membership >= 0.5).astype(int)

    for corner in _enumerate_corners(k):
        # cases assigned to this corner (crisp)
        mask = (crisp_assignment.values == np.array(corner)).all(axis=1)
        n = int(mask.sum())

        # fuzzy corner membership of every case
        corner_mem = np.array([
            _corner_membership(membership.values[i], corner)
            for i in range(len(membership))
        ])
        # cases with corner_mem > 0.5 are best fit; use for n_fuzzy
        n_fuzzy = int((corner_mem > 0.5).sum())

        denom = corner_mem.sum()
        y = outcome.values.astype(float)
        if denom > 0:
            num = _fuzzy_and(corner_mem, y).sum()
            cons = float(num / denom)
            raw_cov = float(num / y.sum()) if y.sum() > 0 else 0.0
        else:
            cons = np.nan
            raw_cov = 0.0

        if n_fuzzy < frequency_cutoff or np.isnan(cons):
            outcome_val: object = "?"
        else:
            outcome_val = int(cons >= consistency_cutoff)

        flag = ""
        if n_fuzzy < frequency_cutoff:
            flag = "low_n"
        elif not np.isnan(cons) and 0.6 <= cons < consistency_cutoff:
            flag = "contradictory"

        row = dict(zip(cond_names, corner))
        row.update({
            "n": n_fuzzy,
            "n_crisp": n,
            "consistency": round(cons, 3) if not np.isnan(cons) else np.nan,
            "raw_coverage": round(raw_cov, 3),
            "outcome": outcome_val,
            "flag": flag,
        })
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "consistency", ascending=False, na_position="last"
    ).reset_index(drop=True)


def _enumerate_corners(k: int) -> Iterable[tuple[int, ...]]:
    for i in range(2 ** k):
        yield tuple((i >> (k - 1 - j)) & 1 for j in range(k))


# ---------------------------------------------------------------------------
# Quine–McCluskey minimisation
# ---------------------------------------------------------------------------

def _to_bitstr(corner: tuple[int, ...]) -> str:
    return "".join(str(b) for b in corner)


def quine_mccluskey(minterms: list[str], dont_cares: Optional[list[str]] = None) -> list[str]:
    """Return prime implicants for ``minterms`` (plus optional don't-cares).

    Both inputs are lists of equal-length bit strings using '0' / '1'. Returns
    a list of prime implicants where each is a string using '0', '1', '-'
    (dash = either value).
    """
    if not minterms:
        return []
    n = len(minterms[0])
    all_terms = set(minterms) | set(dont_cares or [])
    current = {(t, frozenset([t])) for t in all_terms}
    primes: set[tuple[str, frozenset]] = set()

    while True:
        next_round: set[tuple[str, frozenset]] = set()
        combined_flags: set[tuple[str, frozenset]] = set()
        cur_list = list(current)
        for i in range(len(cur_list)):
            for j in range(i + 1, len(cur_list)):
                a, src_a = cur_list[i]
                b, src_b = cur_list[j]
                merged = _try_combine(a, b)
                if merged is not None:
                    next_round.add((merged, src_a | src_b))
                    combined_flags.add(cur_list[i])
                    combined_flags.add(cur_list[j])
        for term in cur_list:
            if term not in combined_flags:
                primes.add(term)
        if not next_round:
            break
        current = next_round

    # Drop primes that don't cover any actual minterm (only don't-cares)
    minterm_set = set(minterms)
    primes_filtered = [p for p, src in primes if src & minterm_set]
    # Deduplicate (different src may yield same pattern)
    primes_dedup = sorted(set(primes_filtered))
    return primes_dedup


def _try_combine(a: str, b: str) -> Optional[str]:
    if len(a) != len(b):
        return None
    diffs = 0
    out = []
    for ca, cb in zip(a, b):
        if ca == cb:
            out.append(ca)
        else:
            if ca == "-" or cb == "-":
                return None
            diffs += 1
            if diffs > 1:
                return None
            out.append("-")
    return "".join(out) if diffs == 1 else None


def _prime_covers(prime: str, minterm: str) -> bool:
    return all(p == "-" or p == m for p, m in zip(prime, minterm))


def essential_cover(primes: list[str], minterms: list[str]) -> list[str]:
    """Petrick-style selection: pick essentials, then greedy cover the rest."""
    if not primes:
        return []
    # Coverage map: which primes cover each minterm
    coverage = {m: [p for p in primes if _prime_covers(p, m)] for m in minterms}
    essentials: set[str] = set()
    covered: set[str] = set()
    for m, covering in coverage.items():
        if len(covering) == 1:
            essentials.add(covering[0])
    for p in essentials:
        for m in minterms:
            if _prime_covers(p, m):
                covered.add(m)

    remaining = [m for m in minterms if m not in covered]
    selected = set(essentials)
    while remaining:
        # Greedy: pick the prime that covers the most remaining minterms
        best, best_score = None, -1
        for p in primes:
            if p in selected:
                continue
            score = sum(1 for m in remaining if _prime_covers(p, m))
            if score > best_score:
                best, best_score = p, score
        if best is None or best_score <= 0:
            break
        selected.add(best)
        remaining = [m for m in remaining if not _prime_covers(best, m)]
    return sorted(selected)


def prime_to_expression(prime: str, condition_names: list[str]) -> str:
    """Convert a bitstring with ``-`` to a Boolean expression."""
    parts = []
    for c, name in zip(prime, condition_names):
        if c == "1":
            parts.append(name)
        elif c == "0":
            parts.append(f"~{name}")
    return " * ".join(parts) if parts else "<universal>"


# ---------------------------------------------------------------------------
# End-to-end QCA driver
# ---------------------------------------------------------------------------

@dataclass
class QCAResult:
    truth_table: pd.DataFrame
    necessity: pd.DataFrame
    sufficiency: pd.DataFrame
    prime_implicants: list[str]
    solution_terms: list[str]
    solution_table: pd.DataFrame
    solution_metrics: dict
    contradictions: pd.DataFrame
    config: dict = field(default_factory=dict)


def run_qca(
    membership: pd.DataFrame,
    outcome: pd.Series,
    *,
    consistency_cutoff: float = 0.80,
    frequency_cutoff: int = 1,
    solve_for_negation: bool = False,
) -> QCAResult:
    """End-to-end QCA pipeline.

    Parameters
    ----------
    membership
        Fuzzy membership table with one column per condition. Index = case_id.
    outcome
        Fuzzy or crisp outcome series, indexed by case_id.
    consistency_cutoff
        Truth-table rows with consistency >= cutoff are coded outcome = 1.
    frequency_cutoff
        Truth-table rows with fewer than ``frequency_cutoff`` cases are dropped
        before minimisation.
    solve_for_negation
        If True, the procedure is rerun with the outcome negated and the union
        of the two solution sets is reported (not commonly required but useful
        for asymmetric analyses).
    """
    cond_names = list(membership.columns)
    nec = necessity_table(membership, outcome)
    suf = sufficiency_table(membership, outcome)
    tt = truth_table(
        membership, outcome,
        consistency_cutoff=consistency_cutoff,
        frequency_cutoff=frequency_cutoff,
    )

    minterms = []
    contradictions = tt[tt["flag"] == "contradictory"].copy()
    for _, row in tt.iterrows():
        if row["outcome"] == 1 and row["n"] >= frequency_cutoff:
            bits = "".join(str(int(row[c])) for c in cond_names)
            minterms.append(bits)

    primes = quine_mccluskey(minterms) if minterms else []
    chosen = essential_cover(primes, minterms) if primes else []
    expressions = [prime_to_expression(p, cond_names) for p in chosen]

    # Per-term metrics in fuzzy data
    metrics_rows = []
    y = outcome.values.astype(float)
    overall_lhs = np.zeros_like(y)
    for p, expr in zip(chosen, expressions):
        memb_term = _prime_membership(p, membership.values)
        cons = float(_fuzzy_and(memb_term, y).sum() / max(memb_term.sum(), 1e-12))
        cov = float(_fuzzy_and(memb_term, y).sum() / max(y.sum(), 1e-12))
        # Unique coverage = covered by this term that no other term covers
        other_lhs = np.zeros_like(y)
        for q in chosen:
            if q != p:
                other_lhs = np.maximum(other_lhs, _prime_membership(q, membership.values))
        unique = float(_fuzzy_and(np.clip(memb_term - other_lhs, 0, 1), y).sum() / max(y.sum(), 1e-12))
        metrics_rows.append({
            "term": expr,
            "consistency": round(cons, 3),
            "raw_coverage": round(cov, 3),
            "unique_coverage": round(unique, 3),
        })
        overall_lhs = np.maximum(overall_lhs, memb_term)

    solution_table = pd.DataFrame(metrics_rows)

    if chosen:
        overall_cons = float(_fuzzy_and(overall_lhs, y).sum() / max(overall_lhs.sum(), 1e-12))
        overall_cov = float(_fuzzy_and(overall_lhs, y).sum() / max(y.sum(), 1e-12))
    else:
        overall_cons = float("nan")
        overall_cov = float("nan")

    return QCAResult(
        truth_table=tt,
        necessity=nec,
        sufficiency=suf,
        prime_implicants=primes,
        solution_terms=expressions,
        solution_table=solution_table,
        solution_metrics={
            "overall_consistency": round(overall_cons, 3),
            "overall_coverage": round(overall_cov, 3),
            "n_prime_implicants": len(primes),
            "n_solution_terms": len(chosen),
        },
        contradictions=contradictions,
        config={
            "consistency_cutoff": consistency_cutoff,
            "frequency_cutoff": frequency_cutoff,
            "solve_for_negation": solve_for_negation,
            "condition_order": cond_names,
        },
    )


def _prime_membership(prime: str, memb_matrix: np.ndarray) -> np.ndarray:
    """Fuzzy membership of every case in the conjunction represented by ``prime``."""
    out = np.ones(memb_matrix.shape[0])
    for j, c in enumerate(prime):
        if c == "1":
            out = np.minimum(out, memb_matrix[:, j])
        elif c == "0":
            out = np.minimum(out, 1.0 - memb_matrix[:, j])
        # '-' means free, do nothing
    return out
