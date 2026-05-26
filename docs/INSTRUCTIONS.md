# How to run text2qca in 5 minutes

This is the short, reviewer-facing instructions file. The full README has more
detail.

## Option A — try the hosted demo (recommended for reviewers)

Open either of the public demo links:

- 🤗 HuggingFace Spaces — <https://huggingface.co/spaces/your-handle/text2qca>
- 🎈 Streamlit Community Cloud — <https://text2qca.streamlit.app>

In the sidebar, keep **Use bundled demo data = on** and click through tabs 1
through 7 in order. Each tab is one step of the research workflow.

## Option B — run locally

```bash
git clone https://github.com/your-handle/text2qca
cd text2qca
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>. Everything else is identical to Option A.

## Option C — regenerate the sample outputs offline

```bash
python scripts/generate_samples.py
```

This rebuilds every file in `outputs/sample/` using the deterministic
keyword-lexicon fallback, so the test runs even without internet access.

## What to look for in the demo

- **Tab 2** — note how each text is scored against every prototype. Open the
  *Why did this case score the way it did?* expander to inspect the
  per-keyword audit.
- **Tab 3** — drag the anchor sliders in the sidebar. The membership heatmap
  in this tab reacts immediately.
- **Tab 5** — necessity vs sufficiency results plus a fuzzy truth table.
  `~dissatisfaction` should appear as nearly necessary for the responsive
  outcome (consistency ≈ 0.99 in the demo).
- **Tab 6** — the parsimonious solution should land at roughly
  `~dissatisfaction * policy_demand * ~coproduction_request +
  ~dissatisfaction * ~policy_demand * coproduction_request`, with overall
  consistency ≈ 0.99 and coverage ≈ 0.84.
- **Tab 7** — download the Excel workbook to verify every intermediate
  artifact persists outside the app.
