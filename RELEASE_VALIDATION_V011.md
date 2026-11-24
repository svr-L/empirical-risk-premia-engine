# ERP 0.11 — Release validation (22 November 2026)

- Repo based on the user-worked v0.10 repository ZIP, not a recreated codebase.
- Added `src/erp/equity_extensions.py` and `src/erp/equity_macro.py`; preserved Rates and original `equity.py` and `equity_residual.py` unchanged.
- The original equity residual v0.10 notebook remains in the repo and new v0.11 notebook embeds **all** ERP 0.11 sources. Bootstrap was executed and imported the declared v0.11 implementation (not the host repo).
- `PYTHONPATH=src pytest -q tests/test_equity_extensions.py tests/test_equity_residual.py`: **17 passed**.
- Other fast regression tests: **57 passed, 2 deselected**.
- Both full historical synthetic Rates tests rerun separately: **2 passed**; total **59 distinct tests passed**.
- `gamma=0` reproduces baseline residual PCA neutral FMP weights to 1.7e-14 on deterministic test data.
- Tests validate the residual eigenvalue empirical null, the global Holm correction, frozen weights and no OOS lookahead, three-pass output shapes, raw FRED parsing, local caching, calendar-month transformations, and no missing CPI interpolation.
- **No live historic dataset was fetched in this environment.** The notebook's FRED fetch with public provider is not validated from inside Colab; the original FRED CSV/ZIP fallback is provided. No real-data v0.11 alpha, priced-factor or TAA claims.
- Lettau–Pelger method is RP-PCA-inspired and GX method is point-estimate-style only; full published estimation/inference, identified structural macro shocks and stock-level testing remain work to do.
