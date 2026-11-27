"""Empirical risk premia research toolkit — version 0.12.1."""
from .core import PremiaSnapshot, FactorDefinition, export_snapshot, load_snapshot
from .equity import EquityPremiaEngine, EquityData, EquityModel, run_equity_research
from .equity_residual import EquityResidualResult, analyze_equity_residuals
from .rates import RatesPremiaEngine, RatesData, run_rates_research
from .rates_research import RatesEmpiricalBondPremiaEngine, BondPremiaResearchResult, load_v08_macro_csv
from .macro import load_macro_panel, load_macro_series_files
__version__ = "0.11.0"
__all__ = ["PremiaSnapshot", "FactorDefinition", "export_snapshot", "load_snapshot", "EquityPremiaEngine", "EquityData", "EquityModel", "run_equity_research", "EquityResidualResult", "analyze_equity_residuals", "RatesPremiaEngine", "RatesData", "run_rates_research", "RatesEmpiricalBondPremiaEngine", "BondPremiaResearchResult", "load_v08_macro_csv", "load_macro_panel", "load_macro_series_files", "RatesV09Result", "analyze_rates_v09", "aligned_forecast_evaluation", "aligned_rotation_evaluation"]

from .rates_v09 import RatesV09Result, analyze_rates_v09, aligned_forecast_evaluation, aligned_rotation_evaluation
from .equity_extensions import EquityFactorExtensionsResult,analyze_equity_factor_extensions,residual_eigenvalue_null
from .equity_macro import EquityMacroPanel,load_equity_fred_macro,make_equity_macro_panel
__all__ += ['EquityFactorExtensionsResult','analyze_equity_factor_extensions','residual_eigenvalue_null',
           'EquityMacroPanel','load_equity_fred_macro','make_equity_macro_panel']
from .equity_v012 import EquityV012Result, analyze_equity_v012
from .equity_shocks import (EconomicShockPanel, load_jk_monetary_shocks,
                           load_inflation_release_surprises, analyze_equity_shock_mapping, estimate_jk_shock_prices)
__version__ = '0.12.1'
__all__ += ['EquityV012Result','analyze_equity_v012','EconomicShockPanel',
            'load_jk_monetary_shocks','load_inflation_release_surprises','analyze_equity_shock_mapping','estimate_jk_shock_prices']
from .equity_shock_followup import ShockFollowup, analyze_rolling_shocks
__version__ = '0.12.2'
__all__ += ['ShockFollowup','analyze_rolling_shocks']
from .equity_v013 import (analyze_dynamic_premia, constrained_rp_rolling, rp_basis,
    gx_three_pass, gx_bootstrap, economic_return_decomposition)
from .cross_asset import load_etf_adjusted_prices, etf_factor_returns, load_rates_factor_returns
__version__='0.13.0'
__all__ += ['analyze_dynamic_premia','constrained_rp_rolling','rp_basis','gx_three_pass','gx_bootstrap',
            'economic_return_decomposition','load_etf_adjusted_prices','etf_factor_returns','load_rates_factor_returns']
from .cross_asset import managed_factor_returns
from .equity_v013 import gx_initial_inference
__all__ += ['managed_factor_returns','gx_initial_inference']
