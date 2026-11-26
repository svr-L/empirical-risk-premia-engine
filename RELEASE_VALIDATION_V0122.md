# Validation v0.12.2
- Final targeted suite: 32 passed (test_equity_shock_followup, test_equity_v012, test_equity_extensions, test_equity_residual).
- New tests cover common dates and missing current/lagged shocks, all rolling factors, same-sample pricing, shock contribution identity, no change in formation estimates from OOS perturbations, no RP backfill, and ignoring unrelated FRED downloads. Existing JK parser test updated to match the v0.12.1 missing-month policy and still rejects widespread corruption.
- Full collection: 74 tests. Regression excluding two long Rates grids emitted 71 passing dots but timed out before completion; no full-suite pass claimed. Rates source files unchanged.
- All new notebook code cells parsed successfully; embedded bootstrap actually imported ERP 0.12.2 in an empty temporary directory.
- No real market data run or live provider download performed in this validation.
