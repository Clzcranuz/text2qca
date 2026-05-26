# text2qca — Technical Note

*Submitted by Xuanchen Liu for the take-home assessment of the project
**"Digital Governance in an Age of AI and Big Data"** (Ref. 260430011),
Department of Applied Social Sciences, The Hong Kong Polytechnic University.*

## What does the tool do?

`text2qca` turns raw citizen-government text — citizen complaints, policy
clarification requests, consultation comments, government replies — into a
QCA-ready dataset, runs the standard set-theoretic procedure
(necessity, sufficiency, truth table, Quine–McCluskey minimisation), and
exports every intermediate artifact for transparent reporting. The whole
pipeline is exposed through a single Streamlit web app organised as seven
tabs that mirror the seven steps of the research workflow.

## What data does it require?

Two CSV files. The first contains one row per case with at minimum the
columns `case_id`, `text` (the citizen message), and `outcome` (a binary or
fuzzy value in `[0, 1]` recording, for example, whether the local government
provided a substantive, respectful reply). The second contains conceptual
prototypes — one row per condition plus optionally one outcome reference —
with a `condition_name`, a natural-language `prototype`, and a `type`
discriminator (`condition` or `outcome`). The bundled demo data (40 simulated
Chinese citizen messages) demonstrates the expected schema.

## What outputs does it produce?

Within the app: a raw and standardised similarity score table, a calibrated
fuzzy or crisp membership table, the QCA-ready dataset, separate necessity
and sufficiency tables, a fuzzy truth table with consistency, raw coverage,
and contradiction flags, a parsimonious solution with per-term consistency,
raw coverage, and unique coverage, and four diagnostic figures
(membership heatmap, score distribution violin, consistency-coverage bubble,
sufficiency XY plot per solution term). Outside the app: a single Excel
workbook with every table on its own sheet, individual CSVs, and a
`snapshot.json` capturing every parameter for reproducibility.

## How should the results be interpreted?

The truth table identifies *configurations* of conditions whose joint
presence is consistently associated with the outcome. The parsimonious
solution names the minimal Boolean expression of such configurations — for
example, the bundled demo yields
`~dissatisfaction * policy_demand * ~coproduction_request +
~dissatisfaction * ~policy_demand * coproduction_request → responsive`,
with overall consistency 0.99 and coverage 0.84. This reads as: across the
40 simulated cases, the local government tends to respond when the citizen
either (a) makes a clear policy demand without expressing dissatisfaction,
or (b) offers cooperation without expressing dissatisfaction. Pure
dissatisfaction without an actionable ask is associated with non-response.
Necessity results complement this: `~dissatisfaction` is *necessary*
(consistency 0.99) for a responsive reply — i.e., no responsive reply was
observed in the presence of strong complaint framing.

## What assumptions does the tool make?

Four. First, that the sentence-embedding cosine similarity between a text
and a prototype is a meaningful proxy for conceptual presence — this holds
well for the demo, but is a substantive assumption for new domains and
should be cross-checked with the optional zero-shot NLI backend. Second,
that calibration anchors are an honest theoretical choice rather than a
purely data-driven one; the UI exposes the anchors as sliders so the
researcher cannot hide them. Third, that the fuzzy QCA truth-table
procedure (corner membership = min over conditions) is the appropriate
operationalisation — fine for three to five conditions but combinatorially
expensive beyond that. Fourth, that the outcome variable is available
separately from the texts; the tool does not jointly model condition and
outcome from text.

## What are the main limitations?

The prototype is the bottleneck — a sloppy prototype sentence will yield
noisy scores regardless of the backbone model. Small-N QCA is intrinsically
sensitive to anchor placement and consistency cutoffs; the tool reports
warnings when sample size, outcome balance, or text length fall below
defensible thresholds, but it cannot recover statistical power that the
data do not have. The current build supports only the parsimonious
solution; complex and intermediate solutions, which require directional
expectations about logical remainders, are deferred to a future iteration.
Finally, the deterministic keyword-lexicon fallback is shipped to make the
demo reproducible offline, but it is intentionally simpler than the
embedding backend and should not be used for inferential work.

## What would I improve with more time?

Three additions. First, a Chinese-specific BERT-based classifier head
(BERT-wwm or ERNIE) fine-tuned on a small labelled sample, alongside the
zero-shot baseline, to demonstrate the empirical gain from supervised
calibration. Second, complex- and intermediate-solution support via
directional-expectations input, plus a sensitivity-analysis panel that
re-runs the pipeline across a grid of consistency cutoffs and anchor
choices to surface how robust the solution is. Third, a connector for live
Hong Kong / Mainland public-consultation portals so the tool ingests
documents directly rather than requiring a pre-cleaned CSV. Each of these
three additions corresponds to a "preferred qualification" listed in the
project's job description (Chinese BERT, configurational methods, and
large complex government-text corpora), and each could plausibly be
prototyped within a fortnight given access to a labelled pilot dataset.
