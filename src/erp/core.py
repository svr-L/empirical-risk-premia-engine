"""Uniform, auditable outputs. Dates are monthly observation dates, NOT publication vintages.

A historical `as_of` removes observations after the cutoff, but downloaded French/AQR
files are currently revised vendor datasets, not archived point-in-time vintages.
Never describe these files as historical information sets without a separate vintage feed.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class FactorDefinition:
    name: str
    family: str
    source: str
    construction: str
    units: str = "simple_return_decimal_monthly"
    traded: bool = True
    convention: str = ""

@dataclass
class PremiaSnapshot:
    asset_class: str
    model: str
    universe: str
    as_of: pd.Timestamp
    factors: pd.DataFrame                  # realized factor return history, monthly decimals
    asset_excess_returns: pd.DataFrame     # realized asset excess returns
    betas: pd.DataFrame                    # current end-of-window exposures: asset x factor
    alpha: pd.Series                       # fitted monthly TS intercept (NOT an expected alpha forecast)
    residuals: pd.DataFrame                # TS residual = r_excess - alpha - beta*factor
    factor_premia: pd.Series               # trailing unconditional factor means (descriptive)
    fit_diagnostics: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    fmb_lambdas: pd.DataFrame = field(default_factory=pd.DataFrame)
    fmb_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_attribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_definitions: list[FactorDefinition] = field(default_factory=list)
    static_cs_premia: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    static_pricing_errors: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def validate(self):
        if self.factors.empty or self.betas.empty:
            raise ValueError("Missing factors or betas")
        if (self.factors.index.max() > self.as_of or
            self.asset_excess_returns.index.max() > self.as_of):
            raise ValueError("Look-ahead: an observation is after as_of")
        if not self.betas.columns.equals(self.factors.columns):
            raise ValueError("Beta/factor columns differ in name or order")
        if self.residuals.columns.difference(self.asset_excess_returns.columns).size:
            raise ValueError("Residual asset names do not match returns")
        return self

    def as_alpha_referee_payload(self) -> dict:
        """Neutral export; does not modify Alpha Referee's own statistical logic."""
        return {"benchmark_factor_returns": self.factors.copy(),
                "benchmark_asset_excess_returns": self.asset_excess_returns.copy(),
                "factor_exposures": self.betas.copy(),
                "factor_model_intercept": self.alpha.copy(),
                "benchmark_residual_returns": self.residuals.copy(),
                "model_metadata": dict(self.metadata)}

    def as_alpha_slicer_payload(self) -> dict:
        """Same objects, separate consumer convention; avoid conflating intercept with predictive alpha."""
        return self.as_alpha_referee_payload()

    def factor_contribution(self) -> pd.DataFrame:
        common = self.factors.index.intersection(self.asset_excess_returns.index)
        pred = self.factors.loc[common].to_numpy() @ self.betas.T.to_numpy()
        return pd.DataFrame(pred, index=common, columns=self.betas.index)


def export_snapshot(snapshot: PremiaSnapshot, destination: str | Path) -> Path:
    """A self-contained, neutral file contract consumable without importing erp."""
    snapshot.validate()
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    frames = {
        "factor_returns": snapshot.factors,
        "asset_excess_returns": snapshot.asset_excess_returns,
        "betas": snapshot.betas,
        "alpha": snapshot.alpha.to_frame("alpha_monthly"),
        "residual_returns": snapshot.residuals,
        "factor_premia": snapshot.factor_premia.to_frame("mean_monthly"),
        "fit_diagnostics": snapshot.fit_diagnostics,
        "fmb_lambdas": snapshot.fmb_lambdas,
        "fmb_diagnostics": snapshot.fmb_diagnostics,
        "oos_attribution": snapshot.oos_attribution,
        "static_cs_premia": snapshot.static_cs_premia.to_frame("lambda_monthly"),
        "static_pricing_errors": snapshot.static_pricing_errors.to_frame("pricing_error_monthly"),
    }
    for key, frame in frames.items():
        frame.to_csv(dest / f"{key}.csv", index_label="Date" if isinstance(frame.index, pd.DatetimeIndex) else "asset")
    info = {
        "schema_version": "0.2.0", "asset_class": snapshot.asset_class,
        "model": snapshot.model, "universe": snapshot.universe,
        "as_of": snapshot.as_of.strftime("%Y-%m-%d"),
        "units": "simple_monthly_returns_decimal",
        "factor_premia_meaning": "historical sample mean; not conditional forecast",
        "alpha_meaning": "fitted in-sample intercept; not investible future alpha",
        "point_in_time_note": "Observations are cut off at as_of, but French/AQR revisions are NOT historical vintages.",
        "factors": [vars(x) for x in snapshot.factor_definitions],
        "metadata": snapshot.metadata,
        "files": sorted(f"{key}.csv" for key in frames),
    }
    (dest / "manifest.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return dest


def load_snapshot(destination: str | Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """External consumers need only JSON and CSV; no project-level imports."""
    folder = Path(destination)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    tables = {}
    for name in manifest["files"]:
        tables[name[:-4]] = pd.read_csv(folder / name, index_col=0)
    return manifest, tables
