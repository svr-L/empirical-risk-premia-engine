# ERP v0.10.0 validation report (22 Nov 2026)

- Based on the actual ERP v0.9 zip, not rewritten from memory.
- New scientific module: `src/erp/equity_residual.py`.
- Two standalone Equity notebooks compile their code cells and execute embedded ERP 0.10 bootstrap. Standalone research notebook was actually executed on **synthetic test data** through factor discovery, neutral FMP weights, OOS premium/pricing/SDF reports, optional economic mapping/state predictions and 17 exported files. These are functional tests, **not real-data economic results**.
- `PYTHONPATH=src pytest -q tests/test_equity_residual.py`: **11 passed**.
- `PYTHONPATH=src pytest -q -k 'not test_all_15_model_horizon_pairs and not test_full_five_family_grid_and_exports'`: **51 passed, 2 deselected**. The entire suite originally exceeded environment timeout late in legacy Rates computational tests, without recorded failures. The excluded two full-grid historical synthetic tests are not claimed passed in this execution.
- The existing original equity factor method and Rates source files are unchanged from ERP v0.9. Current historical real-data Equity v0.10 experiment **has not been executed**: Colab/public French/AQR data required.
- No validation of newly priced equity risks, timing or TAA alpha is claimed from synthetic tests.
