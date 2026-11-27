# ERP 0.13 validation
- 74 tests passed in the regression command excluding `test_rates_full_grid_short.py`, `test_rates_v08_port.py`, `test_rates_v09.py` (18.52 s). This is not the full suite.
- After adding hedge audit, all 10 dedicated v0.13 tests passed (one new test). Do not sum these overlapping counts.
- Tested portable notebook embedded runtime import and synthetic execution of sections 13, 14, 16 with exports on pre-final hedge-audit revision; code compiles on final revision, and added hedge audit passed focused tests.
- Rates source files verified byte-for-byte against ERP 0.12.2 ZIP. Heavy Rates tests not rerun.
- No live Yahoo Finance / FRED / JK or new real-market runs in this environment. Synthetic tests are not empirical results. CPI consensus remains optional and unavailable. No claims of weak-identification robust inference or investment performance.
