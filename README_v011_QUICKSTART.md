# ERP v0.11 — Practical quickstart

**Recommended Colab notebook:** `notebooks/Equity_Factor_Discovery_and_Macro_v0_11_PORTABLE.ipynb`.

The notebook fully embeds the v0.11 source. Open it in Google Colab, run the
cells top to bottom. It downloads original Kenneth French + AQR series, performs
v0.10 unchanged Equity baseline, tests residual commonality and RP-PCA-style
factors, runs three-pass + SDF point-estimate diagnostics, and attempts to fetch
FRED macro/cross-asset inputs automatically.

If FRED is blocked: download the original six-series CSV/ZIP from
https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL%2CINDPRO%2CDGS10%2CDGS2%2CBAA10Y%2CVIXCLS
and set `FRED_LOCAL_PATH` in cell 1. The loader accepts the untouched wide file
(no manual merging). Existing FRED cached files also work offline. If AQR
workbooks are blocked, supply official originals via `AQR_PATHS`.

**Output folders:** `erp_equity_v011_results/equity_baseline_v010`,
`equity_extensions_v011`, `public_FRED_inputs`, `equity_economic_mapping`.

**Scientific cautions:** do not interpret large eigenvalues as priced factors,
FF/AQR pricing residuals as necessarily new systematic premia, macro changes
as identified surprises, or contemporary factor attribution as forecasting.

CLI: `python -m pip install -e .` then run the notebook or the separately
included v0.10 `scripts/run_equity_residual.py` and Python API.

Full methods and sources: `EQUITY_V011_RESEARCH.md`.
