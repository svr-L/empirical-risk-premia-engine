"""Empirical risk premia research toolkit — version 0.2.4."""
from .core import PremiaSnapshot, FactorDefinition, export_snapshot, load_snapshot
from .equity import EquityPremiaEngine, EquityData, EquityModel, run_equity_research
from .rates import RatesPremiaEngine, RatesData, run_rates_research
from .rates_research import RatesEmpiricalBondPremiaEngine, BondPremiaResearchResult, load_v08_macro_csv
__version__ = "0.2.4"
__all__ = ["PremiaSnapshot", "FactorDefinition", "export_snapshot", "load_snapshot", "EquityPremiaEngine", "EquityData", "EquityModel", "run_equity_research", "RatesPremiaEngine", "RatesData", "run_rates_research", "RatesEmpiricalBondPremiaEngine", "BondPremiaResearchResult", "load_v08_macro_csv"]
