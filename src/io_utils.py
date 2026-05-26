"""IO + lightweight validation helpers."""
from __future__ import annotations

import io
from pathlib import Path
from typing import IO

import pandas as pd


REQUIRED_TEXT_COLS = {"case_id", "text"}
REQUIRED_PROTO_COLS = {"condition_name", "prototype", "type"}


def load_texts(file_or_path) -> pd.DataFrame:
    df = _read_csv(file_or_path)
    missing = REQUIRED_TEXT_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Text file is missing required columns: {sorted(missing)}. "
            f"Expected at least: {sorted(REQUIRED_TEXT_COLS)}."
        )
    if "outcome" not in df.columns:
        raise ValueError(
            "Text file is missing an 'outcome' column. Add a numeric column "
            "named 'outcome' (binary or fuzzy in [0, 1])."
        )
    df = df.dropna(subset=["text"]).copy()
    df["case_id"] = df["case_id"].astype(int)
    df = df.set_index("case_id", drop=True).sort_index()
    return df


def load_prototypes(file_or_path) -> pd.DataFrame:
    df = _read_csv(file_or_path)
    missing = REQUIRED_PROTO_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Prototype file is missing required columns: {sorted(missing)}. "
            f"Expected at least: {sorted(REQUIRED_PROTO_COLS)}."
        )
    df["type"] = df["type"].str.lower().str.strip()
    invalid = ~df["type"].isin({"condition", "outcome"})
    if invalid.any():
        bad = df.loc[invalid, "type"].tolist()
        raise ValueError(
            f"Prototype 'type' must be 'condition' or 'outcome' (got {bad}). "
        )
    if not (df["type"] == "condition").any():
        raise ValueError("Prototype file must contain at least one 'condition' row.")
    return df


def _read_csv(file_or_path) -> pd.DataFrame:
    if hasattr(file_or_path, "read"):
        return pd.read_csv(file_or_path)
    return pd.read_csv(file_or_path)


def df_to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    return buf.getvalue().encode("utf-8-sig")


def df_to_excel_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            safe_sheet = sheet[:31]
            df.to_excel(writer, sheet_name=safe_sheet)
    return buf.getvalue()


def basic_health_checks(texts_df: pd.DataFrame) -> list[str]:
    """Return a list of warning messages about the uploaded dataset."""
    warnings: list[str] = []
    n = len(texts_df)
    if n < 10:
        warnings.append(
            f"Only {n} cases were uploaded. QCA results on fewer than ~10 "
            "cases are unstable; treat outputs as illustrative only."
        )
    if "outcome" in texts_df.columns:
        n_pos = int((texts_df["outcome"] > 0.5).sum())
        n_neg = n - n_pos
        if min(n_pos, n_neg) < 3:
            warnings.append(
                f"Outcome is heavily imbalanced (positives: {n_pos}, "
                f"non-positives: {n_neg}). Sufficiency and necessity tests "
                "may be unreliable."
            )
    avg_len = texts_df["text"].astype(str).str.len().mean()
    if avg_len < 8:
        warnings.append(
            f"Average text length is only {avg_len:.0f} characters. Very "
            "short texts may not contain enough information for the "
            "prototype-based scoring step to be reliable."
        )
    dup = texts_df["text"].duplicated().sum()
    if dup:
        warnings.append(f"{dup} duplicate text rows detected. Consider deduplication.")
    return warnings
