"""
Plotly + matplotlib figure helpers.

The Streamlit app reaches for ``plotly`` figures for interactive display in the
browser, and ``matplotlib`` figures for fixed PNG export to ``outputs/sample/``.
Every function below accepts a DataFrame and returns the appropriate figure
object — no global state, no side-effects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plotly is imported lazily inside each ``*_plotly`` function so the
# matplotlib-only path (used for sample-output generation and tests) works
# even in environments where Plotly is not installed.


def _safe_font():
    """Pick a CJK-capable font if available; otherwise fall back."""
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK", "PingFang SC", "Hiragino Sans GB",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return "DejaVu Sans"


def membership_heatmap_plotly(membership: pd.DataFrame):
    import plotly.express as px
    fig = px.imshow(
        membership.values,
        x=list(membership.columns),
        y=[f"case {i}" for i in membership.index],
        color_continuous_scale="Blues",
        zmin=0, zmax=1,
        aspect="auto",
        labels=dict(color="membership"),
    )
    fig.update_layout(
        title="Calibrated set membership",
        margin=dict(l=40, r=20, t=50, b=40),
        height=max(400, 18 * len(membership)),
    )
    return fig


def membership_heatmap_mpl(membership: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.family"] = _safe_font()
    fig, ax = plt.subplots(figsize=(6, 0.22 * len(membership) + 1.6))
    im = ax.imshow(membership.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(membership.columns)))
    ax.set_xticklabels(membership.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(membership)))
    ax.set_yticklabels([f"case {i}" for i in membership.index], fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("set membership")
    ax.set_title("Calibrated set membership")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def score_distribution_plotly(scores: pd.DataFrame):
    import plotly.express as px
    long = scores.reset_index().melt(id_vars=scores.index.name or "index",
                                     var_name="condition", value_name="score")
    fig = px.violin(
        long, x="condition", y="score",
        box=True, points="all",
        color="condition",
        title="Score distribution by condition (raw)",
    )
    fig.update_layout(showlegend=False, height=420)
    return fig


def score_distribution_mpl(scores: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.family"] = _safe_font()
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [scores[c].values for c in scores.columns]
    parts = ax.violinplot(data, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.55)
    ax.set_xticks(range(1, len(scores.columns) + 1))
    ax.set_xticklabels(scores.columns, rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Score distribution by condition")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def xy_plot_plotly(
    membership: pd.DataFrame,
    outcome: pd.Series,
    term_name: str,
    term_values: np.ndarray,
):
    """Sufficiency XY plot for one term."""
    import plotly.express as px
    df = pd.DataFrame({
        "x": term_values,
        "y": outcome.values,
        "case": membership.index,
    })
    fig = px.scatter(df, x="x", y="y", hover_data=["case"],
                     title=f"Sufficiency XY plot — {term_name}")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="grey"))
    fig.update_layout(xaxis_title=f"{term_name} (membership)",
                      yaxis_title="outcome (membership)",
                      xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.02]),
                      height=420)
    return fig


def xy_plot_mpl(values: np.ndarray, outcome: np.ndarray, term_name: str, path: Path) -> None:
    plt.rcParams["font.family"] = _safe_font()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(values, outcome, alpha=0.7)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"{term_name} (membership)")
    ax.set_ylabel("outcome (membership)")
    ax.set_title(f"Sufficiency XY plot — {term_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def consistency_coverage_bubble_plotly(suf: pd.DataFrame):
    import plotly.express as px
    fig = px.scatter(
        suf, x="coverage", y="consistency", text="term",
        size=np.clip(suf.get("coverage", 0).fillna(0) * 30 + 5, 5, 40),
        color="consistency",
        color_continuous_scale="Viridis",
        title="Sufficiency: consistency vs coverage",
    )
    fig.update_traces(textposition="top center")
    fig.add_shape(type="line", x0=0, y0=0.8, x1=1, y1=0.8,
                  line=dict(dash="dot", color="firebrick"))
    fig.update_layout(xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.02]),
                      height=460)
    return fig


def consistency_coverage_bubble_mpl(suf: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.family"] = _safe_font()
    fig, ax = plt.subplots(figsize=(6, 5))
    sizes = np.clip(suf["coverage"].fillna(0) * 800 + 60, 60, 900)
    sc = ax.scatter(suf["coverage"], suf["consistency"],
                    s=sizes, c=suf["consistency"], cmap="viridis", alpha=0.75)
    for _, row in suf.iterrows():
        ax.annotate(row["term"], (row["coverage"], row["consistency"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(0.8, color="firebrick", linestyle=":", linewidth=1,
               label="cons. cutoff = 0.80")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("coverage")
    ax.set_ylabel("consistency")
    ax.set_title("Sufficiency: consistency vs coverage")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("consistency")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
