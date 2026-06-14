# Financial Analysis Skills

Stock-analysis skills vendored from Anthropic's
[Claude for Financial Services](https://github.com/anthropics/financial-services)
repo (Apache-2.0). Two verticals were installed:

- **equity-research**: `earnings-analysis`, `earnings-preview`, `initiating-coverage`,
  `model-update`, `sector-overview`, `thesis-tracker`, `catalyst-calendar`,
  `morning-note`, `idea-generation`
- **financial-analysis** (core): `dcf-model`, `comps-analysis`, `lbo-model`,
  `3-statement-model`, `competitive-analysis`, plus authoring/QC helpers
  (`xlsx-author`, `pptx-author`, `audit-xls`, `clean-data-xls`, `deck-refresh`,
  `ib-check-deck`, `ppt-template-creator`, `skill-creator`)

## Usage

After reopening the session, these appear as `/<skill-name>` commands (e.g.
`/dcf-model`, `/earnings-analysis`). Each skill's `SKILL.md` contains the full
workflow.

## Data sources (optional connectors)

The original plugin wires up authenticated MCP connectors (Daloopa, FactSet,
Morningstar, S&P Capital IQ, Moody's, PitchBook, LSEG, Aiera, MT Newswires,
Egnyte, Box). Those require paid credentials and are **not** configured here.
Without them, the skills run on data you provide or public web sources. To add a
connector, register the server in `.mcp.json` per its provider docs.
