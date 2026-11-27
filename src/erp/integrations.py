"""Explicit opt-in hand-offs. Never change a consuming research engine's own rules."""
from __future__ import annotations
import pandas as pd
from .core import PremiaSnapshot


def to_alpha_slicer_inputs(snapshot:PremiaSnapshot,*,strategy_net:pd.Series,
                           benchmark:pd.Series,strategy_gross:pd.Series|None=None,
                           tail_factors:pd.DataFrame|None=None)->dict:
    """Build kwargs for AlphaSlicer.fit(...).

    Inputs MUST all represent the same calendar-month frequency/return convention.
    Returns can be strategy simple or excess as dictated by Alpha Slicer's own API;
    caller supplies its strategy and benchmark (we never synthesize a benchmark).
    """
    snapshot.validate()
    if not isinstance(strategy_net,pd.Series) or not isinstance(benchmark,pd.Series):
        raise TypeError('Strategy and benchmark must be explicit time series')
    idx=snapshot.factors.index.intersection(strategy_net.index).intersection(benchmark.index)
    if strategy_gross is not None:idx=idx.intersection(strategy_gross.index)
    if tail_factors is not None:idx=idx.intersection(tail_factors.index)
    if len(idx)<24:raise ValueError('Too few common observations for Slicer hand-off')
    result={"strategy_net":strategy_net.loc[idx].astype(float),
            "benchmark":benchmark.loc[idx].astype(float),
            "factors":snapshot.factors.loc[idx].astype(float)}
    if strategy_gross is not None:result['strategy_gross']=strategy_gross.loc[idx].astype(float)
    if tail_factors is not None:result['tail_factors']=tail_factors.loc[idx].astype(float)
    if any(v.isna().any().any() if isinstance(v,pd.DataFrame) else v.isna().any() for v in result.values()):
        raise ValueError('Missing observations in aligned Slicer payload')
    return result


def to_alpha_referee_factor_library(snapshot:PremiaSnapshot)->dict:
    """Offer a published-factor library without modifying Referee's certified pipeline.
    This is NOT a drop-in replacement for Referee's full data/signal setup.
    """
    return {"factors":snapshot.factors.copy(),"model":snapshot.model,
            "factor_definitions":[vars(f) for f in snapshot.factor_definitions]}
