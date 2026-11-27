# Public release validation — ERP v0.13.0

This is a repository organization/documentation release, not a new estimator or a new market-data experiment. The Python source files are unchanged from the supplied ERP v0.13 archive. The three runnable notebooks use the checked-out package instead of separate embedded versions; their analytical cells are retained. Historical executed notebooks are separate evidence, with transient Colab UI metadata removed. No new license has been assigned.

## Checks performed for the public package

- **75 passed** in the fast offline regression suite (12.62 seconds in the packaging environment).
- All three current notebook bootstraps imported ERP 0.13.0 successfully; all code cells compiled.
- All current and three executed notebooks validated against the nbformat schema.
- `scripts/build_results.py` successfully regenerated summary CSVs and four figures from the actual displayed output tables, checking the headline numbers against the extracted values.
- Original historical evidence provenance is recorded in `results/evidence/manifest.json`.

Exact fast regression command:

```bash
python -m pytest -q --ignore=tests/test_rates_full_grid_short.py --ignore=tests/test_rates_v08_port.py --ignore=tests/test_rates_v09.py
```

The three excluded Rates test files remain included and runnable; their tests were not repeated in this packaging pass. See [original v0.13 validation record](validation-original.md). The GitHub Actions workflow runs the fast suite and report generation on push/PR; it has not yet run remotely.

## Empirical limits

No new historical or live market-data estimation was run in the packaging environment. The headline Equity figures are from the user's executed v0.13 notebook, the headline Rates forecast and allocation results from the executed v0.2.5 notebook. The v0.9 Rates common-sample experiment is implemented but has no supplied completed real-data run. The full Equity dynamic bootstrap and factor forecasting output files were not supplied. No statistical significance or investable alpha is inferred from the displayed mean pricing metrics.
