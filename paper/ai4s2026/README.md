# AI4S 2026 workshop-paper draft

Title: **Learning when black-hole hair is observable from synthetic ringdown
waves**

Target venue: **AI4S 2026 — 8th Workshop on Artificial Intelligence and Machine
Learning for Scientific Applications**

The source uses the IEEE conference two-column class. The main text is written
as a five-page workshop draft, with the bibliography forced to begin after the
body. Final pagination must be checked with the exact template distributed by
the workshop.

## Files

- `main.tex` — manuscript source
- `references.bib` — BibTeX database
- `figures/` — copied vector PDF figures used by the manuscript

## Compile

Install a TeX distribution containing `IEEEtran`, `graphicx`, `amsmath`,
`booktabs`, and `microtype`. Then run:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or, with `latexmk`:

```bash
latexmk -pdf main.tex
```

Or with the portable Tectonic compiler:

```bash
tectonic main.tex
```

Clean auxiliary files with:

```bash
latexmk -c
```

## Submission checks

1. Replace the email placeholder; the affiliation is Nagoya University,
   Nagoya, Japan.
2. Confirm the workshop's official IEEE template and 5–8 page rule.
3. Confirm that the body occupies at least five pages and that references begin
   afterward. Float placement can vary between TeX installations.
4. Inspect every figure at 100% zoom and check font readability.
5. Run a spelling and bibliography audit.
6. Retain all scope statements: synthetic leading-eikonal waves, synthetic
   geodesic proxies, no strain inference, no hair detection, no modified-gravity
   constraint, no full QNM solution, and no physical ray tracing.

## Local build status

The source and copied figure PDFs were compiled with Tectonic 0.16.9 on
2026-07-31. The resulting draft is five pages. The conclusion and references
flow together on page 5 without a forced bibliography page break.
