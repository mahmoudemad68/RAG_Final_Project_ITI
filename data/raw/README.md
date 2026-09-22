# Source Documents

The PDFs in this directory are local educational inputs and are intentionally
ignored by Git until redistribution rights are confirmed.

Download these exact documents from their official publishers and save them
with the listed filenames:

| Filename | Official source |
|---|---|
| GINA_2026_Strategy_Report.pdf | https://ginasthma.org/2026-gina-strategy-report/ |
| NICE_NG245_Asthma_Guideline.pdf | https://www.nice.org.uk/guidance/ng245/resources/asthma-diagnosis-monitoring-and-chronic-asthma-management-bts-nice-sign-pdf-66143958279109 |
| NHLBI_Asthma_Care_Quick_Reference.pdf | https://www.nhlbi.nih.gov/sites/default/files/publications/12-5075.pdf |

After downloading, compare each SHA-256 and page count with data/manifest.json.
The notebook refuses to build an index when a required document is absent or
has the wrong hash.

The project cites one-based PDF page numbers, not printed footer numbers.
