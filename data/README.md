# Demo Data

This folder contains the demo input files used for the Text-to-QCA tool.

## `demo_citizen_texts.csv`

40 simulated citizen-to-government messages in Chinese, designed to cover a
realistic mix of three latent concepts ("dissatisfaction", "policy demand",
"co-production request") and a binary outcome encoding whether the local
government provided a substantive, respectful reply (`outcome = 1`).

Columns:

| column   | type   | description |
|----------|--------|-------------|
| case_id  | int    | Unique case identifier |
| text     | string | Citizen message (Chinese) |
| outcome  | int    | 1 = responsive reply, 0 = non-responsive / no reply |

The cases are **simulated** rather than scraped, so the dataset is shareable
under the project's MIT license and contains no personally identifiable
information. The textual style is calibrated against the conceptual prototypes
in `demo_prototypes.csv`.

## `demo_prototypes.csv`

Four conceptual prototypes used to drive the prototype-based text scoring step.
Three rows are conditions and one row is the outcome reference. The Chinese
prototype sentences are intentionally written in neutral expository tone so
that sentence-embedding similarity reflects conceptual content rather than
shared stylistic features.

Columns:

| column         | type   | description |
|----------------|--------|-------------|
| condition_name | string | Internal name used as the column header in the QCA dataset |
| prototype      | string | Natural-language description of the concept |
| type           | string | `condition` or `outcome` |

## Reproducing

The exact files used to produce the sample outputs in `outputs/sample/` are the
ones shipped here. The tool ships a deterministic random-seed fallback for
score generation when the sentence-transformer model is unavailable, so the
truth-table results in `outputs/sample/` should reproduce locally even on a
machine without GPU or internet access.
