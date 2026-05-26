---
title: text2qca — Citizen Text to QCA Conditions
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
license: mit
---

# text2qca

> Prototype-driven text-to-QCA toolchain for citizen–government communication
> research in Chinese and English. Submitted as the take-home assessment for
> *Digital Governance in an Age of AI and Big Data* (Ref. 260430011).

`text2qca` turns raw citizen-government text into a set-theoretic QCA dataset
in seven transparent steps — preview, prototype scoring, calibration,
QCA-ready table, truth table, parsimonious solution, export. Every
intermediate artifact is visible in the UI, can be downloaded as CSV, and is
shipped under `outputs/sample/` so reviewers can verify reproducibility before
running the tool themselves.

For methodological details, assumptions, limitations, and interpretation guidance, see [`docs/technical_note.md`](docs/technical_note.md).

| Step | What happens | Output |
|------|--------------|--------|
| 1 | Upload citizen-text + prototype CSVs | preview tables |
| 2 | Sentence-embedding (or zero-shot NLI) prototype scoring | raw + standardised score tables |
| 3 | Ragin direct / percentile / threshold calibration with adjustable anchors | fuzzy or crisp membership table |
| 4 | Stitch outcome back in | QCA-ready dataset |
| 5 | Necessity + sufficiency tests, truth table | three diagnostic tables, contradictions flag |
| 6 | Quine–McCluskey minimisation | parsimonious solution with per-term consistency, raw coverage, unique coverage |
| 7 | One-click Excel / CSV / JSON export | full reproducibility snapshot |

## Live demo

The same `app.py` is deployed to both platforms — pick whichever loads faster
for you:

- 🤗 **HuggingFace Spaces** — <https://huggingface.co/spaces/your-handle/text2qca>
- 🎈 **Streamlit Community Cloud** — <https://text2qca.streamlit.app>

*(Both links are placeholders. After cloning, follow the **Deployment** section
below to create your own.)*

## Quick start — local

```bash
git clone https://github.com/your-handle/text2qca
cd text2qca
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>. The sidebar's **Use bundled demo data**
toggle is on by default; flip it off to upload your own files.

### Minimum requirements

- Python ≥ 3.9
- ~600 MB free disk (sentence-transformer model is ~95 MB; `torch` is the
  largest dependency at ~250 MB CPU build)
- No GPU required

### Offline / restricted-network reviewers

If the reviewer cannot download the Chinese sentence-transformer model on
first run, the tool automatically falls back to a **deterministic
keyword-lexicon scorer** (`src/scoring.py::KEYWORD_LEXICON`). The fallback is
audited in the UI's *Why did this case score the way it did?* expander, and
the sample outputs in `outputs/sample/` are generated using exactly this
fallback so they reproduce on any machine.

## Input formats

### `texts.csv`

| column | type | description |
|--------|------|-------------|
| `case_id` | int | unique id |
| `text` | str | the citizen / government message |
| `outcome` | float / int | binary (0/1) or fuzzy ([0, 1]) outcome |

### `prototypes.csv`

| column | type | description |
|--------|------|-------------|
| `condition_name` | str | column name in the QCA dataset |
| `prototype` | str | natural-language description of the concept |
| `type` | str | `condition` or `outcome` |

Demo files are bundled under `data/`.

## What it produces

After a run, the following artifacts are downloadable from the **Export** tab
or already shipped under `outputs/sample/`:

- `raw_scores.csv` / `standardised_scores.csv`
- `membership.csv` — calibrated set-membership table
- `qca_ready.csv` — final QCA dataset
- `necessity.csv` / `sufficiency.csv`
- `truth_table.csv`
- `solution_table.csv` + `solution.txt`
- `snapshot.json` — full reproducibility record
- `figures/` — heatmap, score distribution, consistency-coverage bubble, XY plots

## Deployment

### Streamlit Community Cloud (≈ 3 minutes)

1. Push this repository to GitHub.
2. Sign in at <https://streamlit.io/cloud>.
3. **New app** → pick the repo, branch `main`, **main file path** `app.py`.
4. Done. The app builds itself from `requirements.txt`.

### HuggingFace Spaces (≈ 5 minutes)

1. Create a new Space (<https://huggingface.co/new-space>), SDK = **Streamlit**.
2. Clone the Space, copy this repository's files in (the YAML frontmatter at
   the top of this README is already in HF Spaces format), `git push`.
3. The Space builds from `requirements.txt`. The first build downloads the
   sentence-transformer model (~95 MB) into HF's persistent cache.

## Algorithmic notes

| Step | Reference |
|------|-----------|
| Fuzzy set membership via direct calibration | Ragin, *Redesigning Social Inquiry* (2008), ch. 5 |
| Necessity / sufficiency = Σmin(X,Y) / Σ· | Schneider & Wagemann (2012), ch. 5 |
| Truth-table corner membership = min over conditions | Ragin (2008), ch. 6 |
| Minimisation | Quine–McCluskey + greedy Petrick essential cover |
| Embedding scoring | Cosine similarity over L2-normed sentence embeddings |

The implementation is unit-tested against textbook examples; see
`tests/test_qca.py` and `tests/test_calibration.py`.

## Limitations (read this before reporting results)

- **Prototype quality is the bottleneck.** If the prototype sentence is
  vague, the embedding score is vague. Iterate on the prototype before
  re-running the pipeline.
- **The keyword fallback is interpretable but coarse.** It is shipped to make
  the demo reproducible offline; for substantive research, ensure the
  sentence-transformer model loads.
- **Small-N QCA is sensitive to calibration anchors.** Use the Calibration
  tab's per-condition override to stress-test alternative thresholds before
  reporting solutions.
- **The parsimonious solution is one of three QCA solution types.** A
  *complex* and *intermediate* solution would also require theoretical
  expectations about directional remainders, which are out of scope here.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you find the tool useful in your research please cite it as:

```bibtex
@software{liu2026text2qca,
  author = {Liu, Xuanchen},
  title  = {text2qca: Prototype-driven text-to-QCA toolchain for citizen-government communication research},
  year   = {2026},
  url    = {https://github.com/your-handle/text2qca}
}
```
