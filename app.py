"""
text2qca — Streamlit Web UI.

Run locally with:

    streamlit run app.py

Deployed builds on HuggingFace Spaces and Streamlit Community Cloud use this
same file as the entry point.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import calibration as cal
from src import io_utils
from src import qca as qca_mod
from src import scoring as score_mod
from src import visualization as viz

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="text2qca — Citizen text to QCA conditions",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar — global controls and progress indicator
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("text2qca")
    st.caption(
        "Prototype-driven text-to-QCA pipeline for citizen–government "
        "communication research."
    )
    st.divider()
    st.subheader("1 · Data")
    use_demo = st.toggle("Use bundled demo data", value=True,
                         help="Toggle off to upload your own text + prototype files.")
    uploaded_texts = uploaded_prototypes = None
    if not use_demo:
        uploaded_texts = st.file_uploader(
            "Citizen-text CSV (`case_id`, `text`, `outcome`)",
            type=["csv"], key="texts_csv")
        uploaded_prototypes = st.file_uploader(
            "Prototypes CSV (`condition_name`, `prototype`, `type`)",
            type=["csv"], key="proto_csv")
    st.divider()
    st.subheader("2 · Scoring")
    scoring_method = st.radio(
        "Method", ["embedding", "nli", "keyword"], index=0,
        help=("`embedding` — cosine similarity over a Chinese sentence "
              "transformer (default). `nli` — zero-shot entailment. "
              "`keyword` — interpretable lexicon fallback that works offline.")
    )
    st.divider()
    st.subheader("3 · Calibration")
    calib_method = st.selectbox(
        "Default calibration method",
        ["direct (Ragin)", "percentile", "threshold (crisp)"],
        help=("All conditions start from this default. You can override each "
              "condition individually below.")
    )
    st.caption("Calibration anchors for direct fuzzy-set calibration.")

    full_out = st.slider("Full-out anchor", 0.0, 1.0, 0.30, step=0.01)
    cross = st.slider("Crossover anchor", 0.0, 1.0, 0.50, step=0.01)
    full_in = st.slider("Full-in anchor", 0.0, 1.0, 0.70, step=0.01)

    if not (full_out < cross < full_in):
        st.error("Please set anchors in ascending order: full-out < crossover < full-in.")
        st.stop()

    cutoff_direct = (full_out, cross, full_in)
    cutoff_crisp = st.slider("Crisp cutoff (if `threshold`)", 0.0, 1.0, 0.50, step=0.01)
    st.divider()
    st.subheader("4 · QCA")
    cons_cutoff = st.slider("Consistency cutoff", 0.5, 1.0, 0.80, step=0.01)
    freq_cutoff = st.number_input("Frequency cutoff (min cases per row)",
                                  min_value=1, max_value=10, value=1, step=1)


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_demo() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = io_utils.load_texts(DATA_DIR / "demo_citizen_texts.csv")
    p = io_utils.load_prototypes(DATA_DIR / "demo_prototypes.csv")
    return t, p


try:
    if use_demo:
        texts_df, proto_df = _load_demo()
    else:
        if not uploaded_texts or not uploaded_prototypes:
            st.info(
                "Upload both files in the sidebar to begin, or toggle the "
                "**Use bundled demo data** switch back on to explore the tool."
            )
            st.stop()
        texts_df = io_utils.load_texts(uploaded_texts)
        proto_df = io_utils.load_prototypes(uploaded_prototypes)
except ValueError as exc:
    st.error(f"Could not read uploaded files: {exc}")
    st.stop()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧭 text2qca")
st.markdown(
    "**From raw citizen-government text → set membership → QCA configurations.** "
    "Each tab corresponds to one transparent step in the research pipeline."
)

warnings = io_utils.basic_health_checks(texts_df)
if warnings:
    with st.expander(f"⚠ Data health warnings ({len(warnings)})", expanded=True):
        for w in warnings:
            st.warning(w)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "1 · Data preview",
    "2 · Text scoring",
    "3 · Calibration",
    "4 · QCA-ready table",
    "5 · Truth table & necessity",
    "6 · Solution",
    "7 · Export",
])


# ---------- Tab 1: Data preview --------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Citizen messages")
        st.dataframe(texts_df, use_container_width=True, height=420)
    with c2:
        st.subheader("Prototypes")
        st.dataframe(proto_df, use_container_width=True, height=420)
        st.markdown(
            f"**N cases:** {len(texts_df)}  ·  "
            f"**Conditions:** "
            f"{(proto_df['type']=='condition').sum()}  ·  "
            f"**Outcome col:** `outcome` in text file."
        )


# ---------- Tab 2: Scoring -------------------------------------------------
with tabs[1]:
    st.subheader("Prototype-based scoring")
    st.markdown(
        "Each text is scored against every condition prototype. Raw scores "
        "are shown on the left; the standardised (min-max) version on the "
        "right feeds the calibration step."
    )

    @st.cache_data(show_spinner="Scoring texts…")
    def _do_score(texts_index, texts_list, proto_records, method):
        s = pd.Series(texts_list, index=texts_index, name="text")
        p_df = pd.DataFrame(proto_records)
        result = score_mod.score_texts(s, p_df, method=method)
        return result.raw, result.standardised, result.method_used, result.model_name

    raw_scores, std_scores, method_used, model_name = _do_score(
        list(texts_df.index),
        texts_df["text"].tolist(),
        proto_df.to_dict("records"),
        scoring_method,
    )

    st.info(f"Method used: **{method_used}**  ·  Model: `{model_name}`")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw scores**")
        st.dataframe(raw_scores.round(3), use_container_width=True, height=380)
    with c2:
        st.markdown("**Standardised (min-max)**")
        st.dataframe(std_scores.round(3), use_container_width=True, height=380)

    st.plotly_chart(viz.score_distribution_plotly(raw_scores), use_container_width=True)

    with st.expander("Why did a single case score the way it did?"):
        case_pick = st.selectbox("Pick a case to inspect",
                                 list(texts_df.index), key="case_pick")
        st.markdown(f"**Text:**  {texts_df.loc[case_pick, 'text']}")
        st.markdown("**Per-condition scores:**")
        st.write(raw_scores.loc[case_pick].round(3).to_dict())
        st.markdown(
            "**Keyword cue audit** (works for any method — useful even when "
            "the embedding model is the active backend):"
        )
        for cond in raw_scores.columns:
            cues = score_mod.explain_keyword_match(
                texts_df.loc[case_pick, "text"], cond
            )
            st.write({cond: cues})

    st.session_state["raw_scores"] = raw_scores
    st.session_state["std_scores"] = std_scores


# ---------- Tab 3: Calibration ---------------------------------------------
with tabs[2]:
    st.subheader("Calibrate scores → set membership")
    st.markdown(
        "The default calibration method on the left of the sidebar applies to "
        "all conditions. To override a single condition, expand it below and "
        "adjust its anchors."
    )

    cond_cols = list(std_scores.columns)
    method_map = {
        "direct (Ragin)": "direct",
        "percentile": "percentile",
        "threshold (crisp)": "threshold",
    }
    base_method = method_map[calib_method]

    specs = []
    for cond in cond_cols:
        with st.expander(f"⚙ {cond}", expanded=False):
            override = st.checkbox(f"Override defaults for {cond}",
                                   key=f"ov_{cond}")
            if override:
                m = st.selectbox(
                    "Method", ["direct", "percentile", "threshold"],
                    key=f"m_{cond}",
                    index=["direct", "percentile", "threshold"].index(base_method),
                )
                if m == "threshold":
                    cut = st.slider("Crisp cutoff", 0.0, 1.0, 0.5, 0.01, key=f"cu_{cond}")
                    specs.append(cal.CalibrationSpec(
                        condition=cond, method="threshold", crisp_cutoff=cut))
                else:
                    a = st.slider("Anchors", 0.0, 1.0, cutoff_direct, 0.01, key=f"a_{cond}")
                    specs.append(cal.CalibrationSpec(
                        condition=cond, method=m, anchors=tuple(a)))
            else:
                if base_method == "threshold":
                    specs.append(cal.CalibrationSpec(
                        condition=cond, method="threshold",
                        crisp_cutoff=cutoff_crisp))
                else:
                    specs.append(cal.CalibrationSpec(
                        condition=cond, method=base_method,
                        anchors=tuple(cutoff_direct)))

    membership = cal.calibrate_scores(std_scores, specs)

    st.markdown("**Calibration choices**")
    st.table(pd.DataFrame([
        {"condition": s.condition,
         "method": s.method,
         "params": s.describe(),
         "invert": s.invert}
        for s in specs
    ]))

    st.markdown("**Calibrated membership table**")
    st.dataframe(membership.round(3), use_container_width=True, height=380)

    st.plotly_chart(viz.membership_heatmap_plotly(membership),
                    use_container_width=True)

    st.session_state["membership"] = membership
    st.session_state["specs"] = specs


# ---------- Tab 4: QCA-ready -----------------------------------------------
with tabs[3]:
    st.subheader("QCA-ready dataset")
    st.markdown(
        "One row per case, one column per condition (fuzzy), plus the outcome "
        "from the original upload. Download below."
    )

    qca_ready = membership.copy()
    qca_ready["outcome"] = texts_df["outcome"].astype(float).values
    st.dataframe(qca_ready.round(3), use_container_width=True, height=380)

    st.download_button(
        "⬇ Download QCA-ready CSV",
        data=io_utils.df_to_csv_bytes(qca_ready),
        file_name="qca_ready.csv",
        mime="text/csv",
    )
    st.session_state["qca_ready"] = qca_ready


# ---------- Tab 5: Truth table & necessity ---------------------------------
with tabs[4]:
    st.subheader("Truth table, necessity & sufficiency")
    outcome = pd.Series(texts_df["outcome"].astype(float).values,
                        index=texts_df.index, name="outcome")
    result = qca_mod.run_qca(
        membership, outcome,
        consistency_cutoff=cons_cutoff,
        frequency_cutoff=freq_cutoff,
    )
    st.session_state["qca_result"] = result

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Necessity test** (a condition is *necessary* when most "
                    "outcome-positive cases share it)")
        st.dataframe(result.necessity.round(3), use_container_width=True)
    with c2:
        st.markdown("**Sufficiency test** (each single condition alone)")
        st.dataframe(result.sufficiency.round(3), use_container_width=True)

    st.plotly_chart(viz.consistency_coverage_bubble_plotly(result.sufficiency),
                    use_container_width=True)

    st.markdown("**Truth table**")
    st.dataframe(result.truth_table, use_container_width=True, height=380)

    if not result.contradictions.empty:
        st.warning(
            f"{len(result.contradictions)} truth-table rows are flagged as "
            "potentially contradictory (consistency between 0.6 and the "
            "consistency cutoff). Review these rows before reporting."
        )
        st.dataframe(result.contradictions, use_container_width=True)


# ---------- Tab 6: Solution ------------------------------------------------
with tabs[5]:
    st.subheader("Parsimonious solution")
    result: qca_mod.QCAResult = st.session_state.get("qca_result")
    if not result or not result.solution_terms:
        st.info("No configuration passes the current consistency / frequency "
                "cutoffs. Lower the cutoffs in the sidebar to explore weaker "
                "configurations.")
    else:
        st.markdown("**Solution terms**")
        for t in result.solution_terms:
            st.markdown(f"- `{t}` → outcome")
        st.markdown("**Solution metrics**")
        st.json(result.solution_metrics, expanded=True)
        st.markdown("**Per-term metrics**")
        st.dataframe(result.solution_table, use_container_width=True)

        st.markdown("---")
        st.markdown("**Sufficiency XY plot** for each solution term")
        for term, prime in zip(result.solution_terms, result.prime_implicants[:len(result.solution_terms)]):
            term_values = qca_mod._prime_membership(prime, membership.values)
            fig = viz.xy_plot_plotly(membership, outcome, term, term_values)
            st.plotly_chart(fig, use_container_width=True)


# ---------- Tab 7: Export --------------------------------------------------
with tabs[6]:
    st.subheader("Export everything")
    result: qca_mod.QCAResult = st.session_state.get("qca_result")
    membership: pd.DataFrame = st.session_state.get("membership")
    raw_scores: pd.DataFrame = st.session_state.get("raw_scores")
    std_scores: pd.DataFrame = st.session_state.get("std_scores")
    qca_ready: pd.DataFrame = st.session_state.get("qca_ready")

    if not result:
        st.info("Run the pipeline through tab 5 first.")
    else:
        tables = {
            "raw_scores": raw_scores,
            "standardised_scores": std_scores,
            "membership": membership,
            "qca_ready": qca_ready,
            "truth_table": result.truth_table,
            "necessity": result.necessity,
            "sufficiency": result.sufficiency,
            "solution": result.solution_table,
        }
        st.download_button(
            "⬇ Download Excel workbook (all tables)",
            data=io_utils.df_to_excel_bytes(tables),
            file_name="text2qca_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        for name, df in tables.items():
            st.download_button(
                f"⬇ {name}.csv",
                data=io_utils.df_to_csv_bytes(df),
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"dl_{name}",
            )

        st.markdown(
            "**Reproducibility snapshot** — copy this JSON into a report or "
            "supplementary appendix so reviewers can reconstruct your run:"
        )
        snapshot = {
            "n_cases": len(texts_df),
            "scoring_method": scoring_method,
            "calibration_method": calib_method,
            "anchors_direct": list(cutoff_direct),
            "consistency_cutoff": cons_cutoff,
            "frequency_cutoff": int(freq_cutoff),
            "solution_terms": result.solution_terms,
            "solution_metrics": result.solution_metrics,
        }
        st.code(json.dumps(snapshot, ensure_ascii=False, indent=2), language="json")


st.divider()
st.caption(
    "Built for the *Digital Governance in an Age of AI and Big Data* "
    "research project. Code: github.com/<your-handle>/text2qca · "
    "MIT licensed."
)
