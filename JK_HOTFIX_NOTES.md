# ERP v0.12.1 — JK source-data import safety fix

The author-maintained JK monthly CSV has some months (e.g. 2008-01, 2008-10, 2020-03) where `MP_median + CBI_median != pc1_hf`. These months cannot supply identified median shocks and are **set missing**, not zero. Valid zero-event months stay zero. The full monthly calendar is retained and invalid dates are logged in warnings and `shock_provenance.json`. No equity RP/PCA/pricing or Rates estimation code changed.

For an already running Colab v0.12 kernel, upload `JK_shocks_safe_parser_existing_Colab.py` and `%run` it, then rerun the JK shock cell (section 10). For a fresh session, use `notebooks/Equity_Factor_Discovery_and_Pricing_v0_12_1_JK_FIX.ipynb`. This changes loader behaviour; it does not imply any new empirical JK estimates.

Verification: compile of embedded package, correct exclusion of three documented inconsistent source example rows with valid zero-event rows preserved, full synthetic mapping and FF/Jk cross-sectional diagnostics; `ERP 0.12.1` bootstrap passed. Original data full live run not verified in this environment.
