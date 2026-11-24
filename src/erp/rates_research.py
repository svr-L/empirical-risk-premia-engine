"""Rates v0.8 empirical bond-premia layer, ported to ERP v0.2.4.

The functions under 'Original v0.8 functional definitions' are exact text
from the executed notebook Rates_Factor_Timing_v0_8_EMPIRICAL_BOND_PREMIA_STANDALONE.
The wrapper below bridges the v0.2.3 funded Rates adapter to the v0.8
forward-return, prediction and rotation stack. Historical outputs have not
been recalibrated or re-optimized to target strategy performance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from .rates import RatesPremiaEngine, RatesData, FACTOR_NAMES as FACTORS

# Frozen original v0.8 research configuration; not tuned to portfolio returns.
FACTOR_NAMES = list(FACTORS)
PREDICTION_HORIZONS = (3, 6, 12)
MODEL_FAMILIES = ('BASE_STATE','FB','FB_STATE','CP','CP_STATE')
FB_FORWARD_MATURITIES = (2,3,4,5)
CP_FORWARD_MATURITIES = (1,2,3,4,5)
HORIZON_MIN_TRAIN = 60
HORIZON_RIDGE = 1.0
CP_FIRST_STAGE_RIDGE = 1e-10
MEAN_CONFIDENCE_THRESHOLD = 0.60
EXPECTED_RETURN_HURDLE = 0.0
FACTOR_VOL_LOOKBACK = 36
FACTOR_VOL_MIN_OBS = 24
MAX_FACTOR_ALLOCATION = 0.75
TOTAL_FACTOR_GROSS = 1.0
MAX_UNDERLYING_GROSS = 1.25
COST_SENSITIVITY_BPS = (1.,5.,10.)
V08_COST_BPS = 5.

# Original v0.8 functional definitions (extracted from notebook, unchanged):
def interp_curve_yield(
    curve_row: pd.Series,
    maturity: float,
) -> float:
    tenors = np.array(
        [
            float(c.replace("Y", ""))
            for c in curve_row.index
        ]
    )

    yields = curve_row.to_numpy(
        dtype=float
    )

    if maturity < tenors.min() or maturity > tenors.max():
        raise ValueError(
            f"Maturity {maturity:.4f} outside curve grid."
        )

    return float(
        np.interp(
            maturity,
            tenors,
            yields,
        )
    )

def zero_coupon_price_cc(
    y: float,
    maturity: float,
) -> float:
    return float(
        np.exp(-y * maturity)
    )
def signed_factor_signal(
    forecast: pd.DataFrame,
    confidence_threshold: float,
    expected_return_hurdle: float = 0.0,
):
    mu = (
        forecast[
            "total_mean"
        ]
        .astype(float)
    )

    mean_se = (
        forecast[
            "curve_mean_se"
        ]
        .astype(float)
        .clip(
            lower=1e-12
        )
    )

    realized_sigma = (
        forecast[
            "curve_sigma"
        ]
        .astype(float)
        .clip(
            lower=1e-12
        )
    )

    hurdle = float(
        expected_return_hurdle
    )

    p_positive_mean = (
        1.0
        - norm.cdf(
            (
                hurdle
                - mu
            )
            / mean_se
        )
    )

    p_negative_mean = (
        norm.cdf(
            (
                -hurdle
                - mu
            )
            / mean_se
        )
    )

    p_positive_mean = pd.Series(
        p_positive_mean,
        index=mu.index,
    )

    p_negative_mean = pd.Series(
        p_negative_mean,
        index=mu.index,
    )

    side = pd.Series(
        0.0,
        index=mu.index,
    )

    side[
        (
            p_positive_mean
            >= confidence_threshold
        )
        & (
            p_positive_mean
            >= p_negative_mean
        )
    ] = 1.0

    side[
        (
            p_negative_mean
            >= confidence_threshold
        )
        & (
            p_negative_mean
            > p_positive_mean
        )
    ] = -1.0

    # Diagnostic only: probability of the next realized curve return sign.
    predictive_sigma = np.sqrt(
        realized_sigma**2
        + mean_se**2
    )

    p_next_positive = (
        1.0
        - norm.cdf(
            (
                0.0
                - mu
            )
            / predictive_sigma
        )
    )

    return pd.DataFrame(
        {
            "total_mean": mu,
            "mean_se": mean_se,
            "curve_sigma": realized_sigma,
            "p_positive_mean":
                p_positive_mean,
            "p_negative_mean":
                p_negative_mean,
            "p_next_positive_return":
                p_next_positive,
            "side": side,
        }
    )

def trailing_factor_vol(
    factor_returns,
    dt,
    lookback,
    min_obs,
):
    hist = (
        factor_returns.loc[
            :dt,
            FACTOR_NAMES,
        ]
        .tail(
            lookback
        )
    )

    vol = hist.std(
        ddof=1
    )

    count = hist.count()

    vol[
        count
        < min_obs
    ] = np.nan

    return vol

def capped_signed_normalize(
    raw_score: pd.Series,
    max_abs_weight: float,
    gross_target: float,
):
    score = raw_score.copy()

    if (
        score.abs().sum()
        <= 0
    ):
        return pd.Series(
            0.0,
            index=score.index,
        )

    w = (
        gross_target
        * score
        / score.abs().sum()
    )

    # Iteratively cap while preserving sign.
    for _ in range(10):
        over = (
            w.abs()
            > max_abs_weight
        )

        if not over.any():
            break

        capped = (
            w.where(
                ~over,
                np.sign(w)
                * max_abs_weight,
            )
        )

        used = (
            capped[
                over
            ].abs().sum()
        )

        free = (
            ~over
        )

        residual = max(
            gross_target
            - used,
            0.0,
        )

        if (
            free.sum() == 0
            or residual <= 0
        ):
            w = capped
            break

        free_scores = (
            score[
                free
            ]
        )

        if (
            free_scores.abs().sum()
            <= 0
        ):
            w = capped
            break

        capped.loc[
            free
        ] = (
            residual
            * free_scores
            / free_scores.abs().sum()
        )

        w = capped

    # Final safety rescale.
    if (
        w.abs().sum()
        > gross_target
        + 1e-12
    ):
        w = (
            gross_target
            * w
            / w.abs().sum()
        )

    return w

def build_factor_allocations(
    total_forecasts,
    factor_returns,
):
    allocation_rows = []
    signal_rows = []

    dates = (
        total_forecasts
        .index
        .get_level_values(
            "forecast_date"
        )
        .unique()
        .sort_values()
    )

    for dt in dates:
        f = (
            total_forecasts
            .xs(
                dt,
                level="forecast_date",
            )
            .reindex(
                FACTOR_NAMES
            )
        )

        signal = (
            signed_factor_signal(
                f,
                confidence_threshold=
                    MEAN_CONFIDENCE_THRESHOLD,
                expected_return_hurdle=
                    EXPECTED_RETURN_HURDLE,
            )
        )

        vol = (
            trailing_factor_vol(
                factor_returns,
                dt,
                lookback=
                    FACTOR_VOL_LOOKBACK,
                min_obs=
                    FACTOR_VOL_MIN_OBS,
            )
        )

        raw_score = pd.Series(
            0.0,
            index=FACTOR_NAMES,
        )

        valid = (
            vol.notna()
            & (vol > 0)
        )

        raw_score.loc[
            valid
        ] = (
            signal.loc[
                valid,
                "total_mean",
            ]
            / vol.loc[
                valid
            ]
        )

        raw_score = (
            raw_score
            * signal[
                "side"
            ].abs()
        )

        allocation = (
            capped_signed_normalize(
                raw_score,
                max_abs_weight=
                    MAX_FACTOR_ALLOCATION,
                gross_target=
                    TOTAL_FACTOR_GROSS,
            )
        )

        allocation_rows.append(
            pd.Series(
                allocation,
                name=dt,
            )
        )

        tmp_signal = signal.copy()

        tmp_signal[
            "factor_vol"
        ] = vol

        tmp_signal[
            "raw_score"
        ] = raw_score

        tmp_signal[
            "allocation"
        ] = allocation

        tmp_signal[
            "forecast_date"
        ] = dt

        tmp_signal[
            "factor"
        ] = tmp_signal.index

        signal_rows.append(
            tmp_signal.reset_index(
                drop=True
            )
        )

    allocations = (
        pd.DataFrame(
            allocation_rows
        )
        .sort_index()
    )

    signal_detail = (
        pd.concat(
            signal_rows,
            ignore_index=True,
        )
        .set_index(
            [
                "forecast_date",
                "factor",
            ]
        )
        .sort_index()
    )

    return allocations, signal_detail

def combine_factor_and_maturity_weights(
    factor_allocations,
    mimic_weights,
    max_underlying_gross,
):
    rows = []

    for dt in factor_allocations.index:
        assets = next(
            iter(
                mimic_weights.values()
            )
        ).columns

        combined = pd.Series(
            0.0,
            index=assets,
        )

        for factor in FACTOR_NAMES:
            if (
                dt
                not in mimic_weights[
                    factor
                ].index
            ):
                continue

            w_factor = (
                mimic_weights[
                    factor
                ].loc[
                    dt,
                    assets,
                ]
                .fillna(
                    0.0
                )
            )

            combined = (
                combined
                + factor_allocations.loc[
                    dt,
                    factor,
                ]
                * w_factor
            )

        gross = float(
            combined.abs().sum()
        )

        if (
            gross
            > max_underlying_gross
            and gross > 0
        ):
            combined = (
                combined
                * max_underlying_gross
                / gross
            )

        rows.append(
            pd.Series(
                combined,
                name=dt,
            )
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_index()
    )

def apply_overlay_next_month(
    overlay_weights,
    asset_total_returns,
    asset_excess_returns,
    cash_returns,
    cost_bps,
    core_returns=None,
    overlay_scale=1.0,
):
    strategy = {}
    overlay_pnl = {}
    turnover = {}

    previous_w = None

    for dt in overlay_weights.index:
        future_dates = (
            asset_total_returns.index[
                asset_total_returns.index > dt
            ]
        )

        if len(future_dates) == 0:
            continue

        next_dt = future_dates[0]

        w = (
            overlay_weights.loc[dt]
            * overlay_scale
        )

        if previous_w is None:
            one_way_turnover = (
                0.5 * w.abs().sum()
            )
        else:
            aligned = pd.concat(
                [
                    previous_w.rename("prev"),
                    w.rename("curr"),
                ],
                axis=1,
            ).fillna(0.0)

            one_way_turnover = (
                0.5
                * (
                    aligned["curr"]
                    - aligned["prev"]
                )
                .abs()
                .sum()
            )

        # Self-financing overlay P&L uses excess returns.
        pnl = float(
            w
            @ asset_excess_returns.loc[
                next_dt,
                w.index,
            ]
        )

        cost = (
            one_way_turnover
            * cost_bps
            / 10000.0
        )

        if core_returns is None:
            base = float(
                cash_returns.loc[next_dt]
            )
        else:
            base = float(
                core_returns.loc[next_dt]
            )

        strategy[next_dt] = (
            base + pnl - cost
        )

        overlay_pnl[next_dt] = pnl
        turnover[next_dt] = one_way_turnover

        previous_w = w

    return (
        pd.Series(
            strategy,
            name="strategy_return",
        ).sort_index(),
        pd.Series(
            overlay_pnl,
            name="overlay_pnl",
        ).sort_index(),
        pd.Series(
            turnover,
            name="turnover",
        ).sort_index(),
    )

def max_drawdown(
    returns,
):
    wealth = (
        1.0
        + returns
    ).cumprod()

    peak = wealth.cummax()

    dd = (
        wealth
        / peak
        - 1.0
    )

    return float(
        dd.min()
    )

def performance_metrics(
    returns,
    cash=None,
    turnover=None,
):
    r = (
        returns
        .dropna()
    )

    if len(r) == 0:
        return pd.Series(
            dtype=float
        )

    years = (
        len(r)
        / 12.0
    )

    cagr = (
        (
            1.0
            + r
        ).prod()
        ** (
            1.0
            / years
        )
        - 1.0
    )

    ann_vol = (
        r.std(
            ddof=1
        )
        * np.sqrt(
            12.0
        )
    )

    if cash is not None:
        ex = (
            r
            - cash.reindex(
                r.index
            ).fillna(
                0.0
            )
        )
    else:
        ex = r

    ex_vol = ex.std(
        ddof=1
    )

    sharpe = (
        np.sqrt(
            12.0
        )
        * ex.mean()
        / ex_vol
        if ex_vol > 0
        else np.nan
    )

    downside = ex[
        ex < 0
    ]

    sortino = (
        np.sqrt(
            12.0
        )
        * ex.mean()
        / downside.std(
            ddof=1
        )
        if (
            len(
                downside
            ) > 1
            and downside.std(
                ddof=1
            ) > 0
        )
        else np.nan
    )

    avg_turnover = (
        float(
            turnover.reindex(
                r.index
            ).mean()
        )
        if turnover is not None
        else np.nan
    )

    return pd.Series(
        {
            "CAGR":
                cagr,
            "AnnVol":
                ann_vol,
            "Sharpe":
                sharpe,
            "Sortino":
                sortino,
            "MaxDD":
                max_drawdown(
                    r
                ),
            "AvgMonthlyTurnover":
                avg_turnover,
        }
    )

def build_forward_rate_panel(
    zero_curve: pd.DataFrame,
    max_maturity: int = 10,
):
    out = pd.DataFrame(index=zero_curve.index)

    out["f1"] = zero_curve["1Y"]

    for n in range(2, max_maturity + 1):
        out[f"f{n}"] = (
            n * zero_curve[f"{n}Y"]
            - (n - 1) * zero_curve[f"{n-1}Y"]
        )

    return out

def build_horizon_zero_coupon_data(
    zero_curve: pd.DataFrame,
    maturities: list[int],
    cash_return_monthly: pd.Series,
    current_funding_monthly: pd.Series,
    horizon_months: int,
):
    h_years = horizon_months / 12.0

    realized_rows = []
    excess_rows = []
    log_excess_rows = []
    carry_rows = []
    realized_cash_rows = []
    known_cash_rows = []

    idx = zero_curve.index

    for i in range(len(idx) - horizon_months):
        dt0 = idx[i]
        dt1 = idx[i + horizon_months]

        curve0 = zero_curve.loc[dt0]
        curve1 = zero_curve.loc[dt1]

        cash_dates = idx[i + 1 : i + horizon_months + 1]
        cash_path = cash_return_monthly.reindex(cash_dates)

        if cash_path.isna().any():
            continue

        realized_cash = float(
            (1.0 + cash_path).prod() - 1.0
        )

        funding_m = current_funding_monthly.loc[dt0]

        if pd.isna(funding_m):
            continue

        known_cash = float(
            (1.0 + funding_m) ** horizon_months - 1.0
        )

        realized = {}
        excess = {}
        log_excess = {}
        funded_carry = {}

        for tau in maturities:
            remaining = tau - h_years

            if remaining < 1.0:
                continue

            asset = f"ZC{tau}Y"

            y0_tau = float(
                curve0[f"{tau}Y"]
            )

            y0_remaining = interp_curve_yield(
                curve0,
                remaining,
            )

            y1_remaining = interp_curve_yield(
                curve1,
                remaining,
            )

            p0 = zero_coupon_price_cc(
                y0_tau,
                tau,
            )

            p1 = zero_coupon_price_cc(
                y1_remaining,
                remaining,
            )

            p_carry = zero_coupon_price_cc(
                y0_remaining,
                remaining,
            )

            r_total = p1 / p0 - 1.0
            r_bond_carry = p_carry / p0 - 1.0

            realized[asset] = r_total
            excess[asset] = r_total - realized_cash

            # Academic bond-return regressions are conventionally written
            # in log excess returns. Strategy P&L remains in simple returns.
            log_excess[asset] = (
                np.log1p(r_total)
                - np.log1p(realized_cash)
            )

            funded_carry[asset] = r_bond_carry - known_cash

        realized_rows.append(
            pd.Series(realized, name=dt0)
        )

        excess_rows.append(
            pd.Series(excess, name=dt0)
        )

        log_excess_rows.append(
            pd.Series(log_excess, name=dt0)
        )

        carry_rows.append(
            pd.Series(funded_carry, name=dt0)
        )

        realized_cash_rows.append(
            pd.Series({"cash_realized": realized_cash}, name=dt0)
        )

        known_cash_rows.append(
            pd.Series({"cash_known": known_cash}, name=dt0)
        )

    return {
        "realized": pd.DataFrame(realized_rows).sort_index(),
        "excess": pd.DataFrame(excess_rows).sort_index(),
        "log_excess": pd.DataFrame(log_excess_rows).sort_index(),
        "funded_carry": pd.DataFrame(carry_rows).sort_index(),
        "realized_cash": pd.DataFrame(realized_cash_rows)["cash_realized"].sort_index(),
        "known_cash": pd.DataFrame(known_cash_rows)["cash_known"].sort_index(),
    }

def build_horizon_factor_targets(
    mimic_weights,
    asset_data,
):
    excess_assets = asset_data["excess"]
    funded_carry_assets = asset_data["funded_carry"]

    total = pd.DataFrame(
        index=excess_assets.index,
        columns=FACTOR_NAMES,
        dtype=float,
    )

    carry = pd.DataFrame(
        index=excess_assets.index,
        columns=FACTOR_NAMES,
        dtype=float,
    )

    for dt in excess_assets.index:
        for factor in FACTOR_NAMES:
            if dt not in mimic_weights[factor].index:
                continue

            w = mimic_weights[factor].loc[dt].dropna()

            if len(w) == 0:
                continue

            assets = w.index.intersection(
                excess_assets.columns
            )

            if len(assets) == 0:
                continue

            total.loc[dt, factor] = float(
                w.loc[assets]
                @ excess_assets.loc[dt, assets]
            )

            carry.loc[dt, factor] = float(
                w.loc[assets]
                @ funded_carry_assets.loc[dt, assets]
            )

    curve = total - carry

    return {
        "total": total,
        "carry": carry,
        "curve": curve,
    }

def fama_bliss_univariate_diagnostics(
    asset_excess_h: pd.DataFrame,
    fb_panel: pd.DataFrame,
    maturities=(2, 3, 4, 5),
    hac_lags=11,
):
    rows = []

    for n in maturities:
        asset = f"ZC{n}Y"
        pred = f"FB{n}"

        frame = pd.concat(
            [
                asset_excess_h[asset].rename("rx"),
                fb_panel[pred].rename("fb"),
            ],
            axis=1,
        ).dropna()

        X = sm.add_constant(frame["fb"])

        fit = sm.OLS(
            frame["rx"],
            X,
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lags},
        )

        rows.append(
            pd.Series(
                {
                    "beta": fit.params["fb"],
                    "t_HAC": fit.tvalues["fb"],
                    "R2": fit.rsquared,
                    "n": len(frame),
                },
                name=asset,
            )
        )

    return pd.DataFrame(rows)

def fit_standardized_ridge_predictor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    x_current: pd.Series,
    ridge: float,
    min_obs: int = 60,
):
    frame = pd.concat(
        [y_train.rename("y"), X_train],
        axis=1,
    ).dropna()

    if len(frame) < max(min_obs, X_train.shape[1] + 5):
        return None

    features = list(X_train.columns)

    y = frame["y"].to_numpy(dtype=float)
    X_raw = frame[features].astype(float)

    mu_x = X_raw.mean(axis=0)
    sd_x = X_raw.std(axis=0, ddof=1).replace(0.0, 1.0)

    Xz = X_raw.subtract(mu_x, axis=1).divide(sd_x, axis=1)

    x0 = (
        x_current.reindex(features)
        .astype(float)
        .subtract(mu_x)
        .divide(sd_x)
    )

    if x0.isna().any():
        return None

    X = np.column_stack(
        [np.ones(len(Xz)), Xz.to_numpy(dtype=float)]
    )

    x = np.r_[1.0, x0.to_numpy(dtype=float)]

    XtX = X.T @ X

    penalty = np.eye(XtX.shape[0]) * ridge
    penalty[0, 0] = 0.0

    A = XtX + penalty
    A_inv = np.linalg.pinv(A)

    beta = A_inv @ X.T @ y
    resid = y - X @ beta

    dof = max(len(y) - len(beta), 1)
    s2 = float(resid @ resid) / dof

    Vb = s2 * A_inv @ XtX @ A_inv

    pred_mean = float(x @ beta)
    pred_mean_se = sqrt(max(float(x @ Vb @ x), 1e-18))
    resid_sigma = sqrt(max(s2, 1e-18))

    return {
        "mean": pred_mean,
        "mean_se": pred_mean_se,
        "sigma": resid_sigma,
        "n": len(y),
    }

def fit_cp_first_stage(
    train_dates,
    avg_excess_target: pd.Series,
    forward_panel: pd.DataFrame,
    current_date,
    ridge: float = 1e-10,
):
    features = list(forward_panel.columns)

    frame = pd.concat(
        [
            avg_excess_target.reindex(train_dates).rename("y"),
            forward_panel.reindex(train_dates),
        ],
        axis=1,
    ).dropna()

    if len(frame) < max(24, len(features) + 5):
        return None

    y = frame["y"].to_numpy(dtype=float)
    Xf = frame[features].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(Xf)), Xf])

    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0.0

    beta = (
        np.linalg.pinv(X.T @ X + penalty)
        @ X.T
        @ y
    )

    cp_train = pd.Series(
        X @ beta,
        index=frame.index,
        name="CP",
    )

    x_current_raw = forward_panel.loc[
        current_date,
        features,
    ].to_numpy(dtype=float)

    x_current = np.r_[1.0, x_current_raw]

    cp_current = float(x_current @ beta)

    coefficients = pd.Series(
        beta,
        index=["intercept", *features],
    )

    return {
        "cp_train": cp_train,
        "cp_current": cp_current,
        "coefficients": coefficients,
    }

def realization_cutoff_date(
    forecast_date,
    horizon_months,
    master_index,
):
    if forecast_date not in master_index:
        return None

    pos = master_index.get_loc(forecast_date)

    if isinstance(pos, slice) or pos < horizon_months:
        return None

    return master_index[pos - horizon_months]

def model_predictor_training_block(
    model_family,
    train_dates,
    current_date,
    state_panel,
    fb_panel,
    cp_payload=None,
):
    if model_family == "BASE_STATE":
        return (
            state_panel.reindex(train_dates),
            state_panel.loc[current_date],
        )

    if model_family == "FB":
        return (
            fb_panel.reindex(train_dates),
            fb_panel.loc[current_date],
        )

    if model_family == "FB_STATE":
        train = pd.concat(
            [
                fb_panel.reindex(train_dates),
                state_panel.reindex(train_dates),
            ],
            axis=1,
        )

        current = pd.concat(
            [
                fb_panel.loc[current_date],
                state_panel.loc[current_date],
            ]
        )

        return train, current

    if model_family in {"CP", "CP_STATE"}:
        if cp_payload is None:
            return None, None

        cp_train = cp_payload["cp_train"].to_frame("CP")
        cp_current = pd.Series(
            {"CP": cp_payload["cp_current"]}
        )

        if model_family == "CP":
            return cp_train, cp_current

        train = pd.concat(
            [
                cp_train,
                state_panel.reindex(cp_train.index),
            ],
            axis=1,
        )

        current = pd.concat(
            [
                cp_current,
                state_panel.loc[current_date],
            ]
        )

        return train, current

    raise ValueError(
        f"Unknown model family: {model_family}"
    )

def walk_forward_empirical_premia_model(
    horizon_months,
    model_family,
    curve_target_h,
    carry_h,
    avg_asset_excess_h,
    state_panel,
    fb_panel,
    cp_forward_panel,
    master_index,
    min_train=60,
    ridge=1.0,
):
    required_current = (
        state_panel.index
        if "STATE" in model_family
        else fb_panel.index
        if model_family in {"FB"}
        else cp_forward_panel.index
    )

    candidate_dates = (
        curve_target_h.index
        .intersection(carry_h.index)
        .intersection(required_current)
        .sort_values()
    )

    rows = []
    latest_cp_coefficients = None

    for current_date in candidate_dates:
        cutoff = realization_cutoff_date(
            current_date,
            horizon_months,
            master_index,
        )

        if cutoff is None:
            continue

        train_dates = curve_target_h.index[
            curve_target_h.index <= cutoff
        ]

        # All models need fully observed factor targets.
        train_dates = train_dates[
            curve_target_h.reindex(train_dates)
            .notna()
            .all(axis=1)
        ]

        if len(train_dates) < min_train:
            continue

        cp_payload = None

        if model_family in {"CP", "CP_STATE"}:
            cp_payload = fit_cp_first_stage(
                train_dates=train_dates,
                avg_excess_target=avg_asset_excess_h,
                forward_panel=cp_forward_panel,
                current_date=current_date,
                ridge=CP_FIRST_STAGE_RIDGE,
            )

            if cp_payload is None:
                continue

            latest_cp_coefficients = cp_payload["coefficients"]

        X_train, x_current = model_predictor_training_block(
            model_family=model_family,
            train_dates=train_dates,
            current_date=current_date,
            state_panel=state_panel,
            fb_panel=fb_panel,
            cp_payload=cp_payload,
        )

        if X_train is None or x_current is None:
            continue

        factor_rows = []

        for factor in FACTOR_NAMES:
            pred = fit_standardized_ridge_predictor(
                X_train=X_train,
                y_train=curve_target_h[factor].reindex(train_dates),
                x_current=x_current,
                ridge=ridge,
                min_obs=min_train,
            )

            if pred is None:
                continue

            factor_rows.append(
                pd.Series(
                    {
                        "curve_mean_h": pred["mean"],
                        "curve_mean_se_h": pred["mean_se"],
                        "curve_sigma_h": pred["sigma"],
                        "carry_h": carry_h.loc[current_date, factor],
                        "n_train": pred["n"],
                    },
                    name=factor,
                )
            )

        if len(factor_rows) != len(FACTOR_NAMES):
            continue

        block = pd.DataFrame(factor_rows)
        block["total_mean_h"] = (
            block["carry_h"]
            + block["curve_mean_h"]
        )

        # Convert the H-month expected-return object to a monthly-equivalent
        # signal for portfolio sizing. Sign confidence is invariant to this
        # positive scaling.
        block["total_mean"] = (
            block["total_mean_h"]
            / horizon_months
        )

        block["curve_mean"] = (
            block["curve_mean_h"]
            / horizon_months
        )

        block["curve_mean_se"] = (
            block["curve_mean_se_h"]
            / horizon_months
        )

        block["curve_sigma"] = (
            block["curve_sigma_h"]
            / np.sqrt(horizon_months)
        )

        block["forecast_date"] = current_date
        block["factor"] = block.index
        block["horizon_months"] = horizon_months
        block["model_family"] = model_family

        rows.append(
            block.reset_index(drop=True)
        )

    if not rows:
        return None, latest_cp_coefficients

    out = (
        pd.concat(rows, ignore_index=True)
        .set_index(["forecast_date", "factor"])
        .sort_index()
    )

    return out, latest_cp_coefficients

def build_horizon_unconditional_forecast(
    horizon_months,
    curve_target_h,
    carry_h,
    master_index,
    candidate_dates,
    min_train,
):
    rows = []

    for current_date in candidate_dates:
        cutoff = realization_cutoff_date(
            current_date,
            horizon_months,
            master_index,
        )

        if cutoff is None:
            continue

        hist = curve_target_h.loc[
            curve_target_h.index <= cutoff
        ].dropna()

        if len(hist) < min_train:
            continue

        mu = hist.mean()
        sig = hist.std(ddof=1)
        se = sig / np.sqrt(hist.count())

        block = pd.DataFrame(
            {
                "curve_mean_h": mu,
                "curve_mean_se_h": se,
                "curve_sigma_h": sig,
                "carry_h": carry_h.loc[current_date, FACTOR_NAMES],
            }
        )

        block["total_mean_h"] = (
            block["carry_h"]
            + block["curve_mean_h"]
        )

        block["total_mean"] = block["total_mean_h"] / horizon_months
        block["curve_mean"] = block["curve_mean_h"] / horizon_months
        block["curve_mean_se"] = block["curve_mean_se_h"] / horizon_months
        block["curve_sigma"] = block["curve_sigma_h"] / np.sqrt(horizon_months)
        block["forecast_date"] = current_date
        block["factor"] = block.index

        rows.append(block.reset_index(drop=True))

    return (
        pd.concat(rows, ignore_index=True)
        .set_index(["forecast_date", "factor"])
        .sort_index()
    )

def build_horizon_carry_only_forecast(
    horizon_months,
    carry_h,
    reference_dates,
):
    rows = []

    for current_date in reference_dates:
        if current_date not in carry_h.index:
            continue

        block = pd.DataFrame(
            index=FACTOR_NAMES
        )

        block["carry_h"] = carry_h.loc[
            current_date,
            FACTOR_NAMES,
        ]

        block["curve_mean_h"] = 0.0
        block["curve_mean_se_h"] = 1e-12
        block["curve_sigma_h"] = 1e-12
        block["total_mean_h"] = block["carry_h"]
        block["total_mean"] = block["carry_h"] / horizon_months
        block["curve_mean"] = 0.0
        block["curve_mean_se"] = 1e-12
        block["curve_sigma"] = 1e-12
        block["forecast_date"] = current_date
        block["factor"] = block.index

        rows.append(block.reset_index(drop=True))

    return (
        pd.concat(rows, ignore_index=True)
        .set_index(["forecast_date", "factor"])
        .sort_index()
    )

def predictive_diagnostics(
    model_forecast,
    uncond_forecast,
    realized_curve_h,
):
    rows = []

    for factor in FACTOR_NAMES:
        model = (
            model_forecast
            .xs(factor, level="factor")
            ["curve_mean_h"]
        )

        uncond = (
            uncond_forecast
            .xs(factor, level="factor")
            ["curve_mean_h"]
        )

        realized = realized_curve_h[factor]

        frame = pd.concat(
            [
                model.rename("model"),
                uncond.rename("uncond"),
                realized.rename("realized"),
            ],
            axis=1,
        ).dropna()

        if len(frame) == 0:
            continue

        sse_model = float(
            ((frame["realized"] - frame["model"]) ** 2).sum()
        )

        sse_uncond = float(
            ((frame["realized"] - frame["uncond"]) ** 2).sum()
        )

        oos_r2 = (
            1.0 - sse_model / sse_uncond
            if sse_uncond > 0
            else np.nan
        )

        rows.append(
            pd.Series(
                {
                    "corr": frame["model"].corr(frame["realized"]),
                    "OOS_R2_vs_uncond": oos_r2,
                    "MAE": (frame["model"] - frame["realized"]).abs().mean(),
                    "n": len(frame),
                },
                name=factor,
            )
        )

    return pd.DataFrame(rows)


# ===== ERP v0.2.4 bridge and shared research-result export =====

@dataclass
class BondPremiaResearchResult:
    """All v0.8 research tables, kept separate from historical factor snapshot."""
    snapshot: object
    factor_targets: dict
    forecasts: dict
    unconditional_forecasts: dict
    carry_only_forecasts: dict
    predictive_diagnostics: pd.DataFrame
    strategy_results: pd.DataFrame
    conditional_value_add: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    fb_12m_diagnostics: pd.DataFrame
    cp_coefficients: pd.DataFrame
    latest_allocations: pd.DataFrame
    backtests: dict
    metadata: dict

    def export(self, dest):
        """Stable CSV schema for Referee/Slicer and Rates TAA; no look-ahead predictions represented as P&L."""
        out=Path(dest);out.mkdir(parents=True,exist_ok=True)
        self.predictive_diagnostics.to_csv(out/'rates_v08_predictive_diagnostics.csv',index=False)
        self.strategy_results.to_csv(out/'rates_v08_strategy_results.csv')
        self.conditional_value_add.to_csv(out/'rates_v08_conditional_value_add.csv')
        self.cost_sensitivity.to_csv(out/'rates_v08_cost_sensitivity.csv',index=False)
        self.fb_12m_diagnostics.to_csv(out/'rates_v08_fb_12m_diagnostics.csv')
        self.cp_coefficients.to_csv(out/'rates_v08_cp_coefficients.csv')
        self.latest_allocations.to_csv(out/'rates_v08_latest_allocations.csv')
        for (h,model),panel in self.forecasts.items():
            panel.to_csv(out/f'rates_v08_forecast_{h:02d}m_{model}.csv')
        for h,panel in self.unconditional_forecasts.items():
            panel.to_csv(out/f'rates_v08_forecast_{h:02d}m_UNCONDITIONAL.csv')
        for h,panel in self.carry_only_forecasts.items():
            panel.to_csv(out/f'rates_v08_forecast_{h:02d}m_CARRY_ONLY.csv')
        for h,targets in self.factor_targets.items():
            for name,df in targets.items():
                df.to_csv(out/f'rates_v08_target_{h:02d}m_{name}.csv')
        # Return series and exact cash/funded-position records are exported separately.
        for (h,model),bt in self.backtests.items():
            bt['returns'].to_csv(out/f'rates_v08_strategy_returns_{h:02d}m_{model}.csv')
            bt['turnover'].to_csv(out/f'rates_v08_turnover_{h:02d}m_{model}.csv')
            bt['overlay_weights'].to_csv(out/f'rates_v08_underlying_weights_{h:02d}m_{model}.csv')
        with open(out/'rates_v08_manifest.json','w',encoding='utf8') as f:
            json.dump(self.metadata,f,indent=2,default=str,allow_nan=False)
        return out


def build_v08_state_panel(rates:RatesPremiaEngine, macro_raw:pd.DataFrame|None=None):
    """Exact v0.8 curve states and macro transformations; macro cannot be fabricated.

    macro_raw expects monthly OBSERVED CPIAUCSL, INDPRO (index levels) and
    T10YIE (percentage points), with columns CPI, INDPRO, BREAKEVEN10Y.
    FEDFUNDS comes from RatesData.fedfunds (annual decimal, observed monthly).
    Follows v0.8's one-period CPI/INDPRO lag and its current T10YIE choice.
    These are NOT true point-in-time macro release vintages.
    """
    if macro_raw is None:
        return None
    require={'CPI','INDPRO','BREAKEVEN10Y'}
    if not require.issubset(macro_raw.columns):
        raise ValueError(f'Macro input must contain original v0.8 series {sorted(require)}, missing {sorted(require-set(macro_raw.columns))}')
    z=rates.zero_curve_sample
    state=pd.DataFrame(index=z.index)
    state['curve_level']=z.mean(axis=1)
    state['curve_slope_10_2']=z['10Y']-z['2Y']
    state['curve_curvature']=2*z['5Y']-z['2Y']-z['10Y']
    vols=pd.DataFrame({dt:np.sqrt(np.diag(x)) for dt,x in rates.ewma_covariance_path.items()},index=z.columns).T
    state['rate_vol']=vols.mean(axis=1)
    for fac in FACTOR_NAMES:
        state['pc_'+fac.lower()]=rates.factor_scores_1sigma[fac]
    m=macro_raw.copy();m.index=pd.DatetimeIndex(m.index)+pd.offsets.MonthEnd(0)
    m=m.sort_index().loc[:z.index.max()].resample('ME').last()
    state['inflation_yoy']=m['CPI'].pct_change(12,fill_method=None).shift(1)
    state['growth_yoy']=m['INDPRO'].pct_change(12,fill_method=None).shift(1)
    cash=rates.data.fedfunds.copy();cash.index=pd.DatetimeIndex(cash.index)+pd.offsets.MonthEnd(0)
    state['fedfunds']=cash.resample('ME').last().shift(1)
    state['breakeven10y']=m['BREAKEVEN10Y']/100.
    return state.apply(pd.to_numeric,errors='coerce').pipe(_v08_rolling_zscore).dropna(how='any')


def _v08_rolling_zscore(x,window=60,min_periods=36):
    return (x-x.rolling(window,min_periods=min_periods).mean())/x.rolling(window,min_periods=min_periods).std(ddof=1)


def _stable_v08_ridge_state(rates,raw_macro):
    # Use precisely original v0.8 curve-state setup. The state block is only
    # available if the original macro fields were provided. No proxy substitution.
    return build_v08_state_panel(rates,raw_macro)


class RatesEmpiricalBondPremiaEngine:
    """Full v0.8 research grid port; consumes a fitted ERP funded Rates engine.

    In accordance with v0.8, forecasts and backtests are generated only for
    rows with observable realized H-month targets, and model training ends at
    t-H. A latest-date, not-yet-realized deployable forecast is a future module.
    """
    def __init__(self, rates:RatesPremiaEngine):
        self.rates=rates

    def fit(self, snapshot, macro_raw=None, horizons=(3,6,12),
            models=None, min_train=HORIZON_MIN_TRAIN, ridge=HORIZON_RIDGE,
            cost_bps=V08_COST_BPS, cost_sensitivity=(1.,5.,10.),
            require_full_grid=False):
        rates=self.rates
        if rates.zero_curve_sample is None or rates.mimic_weights_panel is None:
            raise ValueError('Call RatesPremiaEngine.fit() first on the same zero-coupon/funding data.')
        horizons=tuple(horizons)
        if any(h not in (3,6,12) for h in horizons):
            raise ValueError('The historical v0.8 grid supports horizons 3,6,12 months only.')
        z=rates.zero_curve_sample
        if z.index.max()>pd.Timestamp(snapshot.as_of):
            raise ValueError('Rates source exceeds as_of snapshot: look-ahead')
        state=_stable_v08_ridge_state(rates,macro_raw)
        valid_models=tuple(MODEL_FAMILIES if models is None else models)
        if set(valid_models)-set(MODEL_FAMILIES):
            raise ValueError(f'Unknown v0.8 model families {set(valid_models)-set(MODEL_FAMILIES)}')
        skipped=[]
        if state is None:
            absent=[m for m in valid_models if 'STATE' in m]
            if absent and require_full_grid:
                raise ValueError('BASE_STATE, FB_STATE and CP_STATE require original CPI, INDPRO and T10YIE series; supply macro_raw.')
            skipped=absent
            valid_models=tuple(m for m in valid_models if 'STATE' not in m)
            if skipped:
                warnings.warn('Missing original v0.8 macro series: skipped '+', '.join(skipped)+'. FB and CP are still run. Supply macro_raw for all five families.',RuntimeWarning)
        if not valid_models: raise ValueError('No available models to run')
        fwd=build_forward_rate_panel(z)
        fb=pd.DataFrame({f'FB{n}':fwd[f'f{n}']-fwd['f1'] for n in FB_FORWARD_MATURITIES},index=z.index)
        cp=fwd[[f'f{n}' for n in CP_FORWARD_MATURITIES]].copy()
        # Original v0.8 cash convention: next-month cash return uses the lagged
        # observed FEDFUNDS rate; the H-month carry uses today's observed rate.
        cash=rates.cash_returns.reindex(z.index)
        funding=rates.data.fedfunds.copy();funding.index=pd.DatetimeIndex(funding.index)+pd.offsets.MonthEnd(0)
        monthly_funding=funding.reindex(z.index)/12.
        for attr in ['asset_total_returns','asset_excess_returns','factor_returns']:
            if getattr(rates,attr) is None:raise ValueError(f'Rates engine missing {attr}')
        targets={}; forecasts={}; baseline={}; carry_only={}; cp_coefficients={}; backtests={};
        predictive_rows=[]; value_rows=[]; cost_rows=[]; strategy_rows=[];alloc_rows=[]
        all_horizon_asset_data={}
        for h in horizons:
            assets=build_horizon_zero_coupon_data(z,list(range(2,11)),cash,monthly_funding,h)
            all_horizon_asset_data[h]=assets
            targ=build_horizon_factor_targets(rates.mimic_weights_panel,assets)
            targets[h]=targ
            avg=assets['log_excess'][['ZC2Y','ZC3Y','ZC4Y','ZC5Y']].mean(axis=1)
            for family in valid_models:
                forecast,coef=walk_forward_empirical_premia_model(
                    horizon_months=h,model_family=family,
                    curve_target_h=targ['curve'],carry_h=targ['carry'],
                    avg_asset_excess_h=avg,state_panel=state,
                    fb_panel=fb,cp_forward_panel=cp,master_index=z.index,
                    min_train=min_train,ridge=ridge)
                if forecast is not None and len(forecast):forecasts[(h,family)]=forecast
                if coef is not None:cp_coefficients[(h,family)]=coef
            family_keys=[(h,f) for f in valid_models if (h,f) in forecasts]
            if not family_keys:
                warnings.warn(f'No walk-forward models produced forecasts for H={h}',RuntimeWarning)
                continue
            dates=pd.DatetimeIndex(sorted(set().union(*(set(forecasts[k].index.get_level_values('forecast_date')) for k in family_keys))))
            baseline[h]=build_horizon_unconditional_forecast(h,targ['curve'],targ['carry'],z.index,dates,min_train)
            carry_only[h]=build_horizon_carry_only_forecast(h,targ['carry'],dates)
            for family in valid_models:
                key=(h,family)
                if key not in forecasts: continue
                pdia=predictive_diagnostics(forecasts[key],baseline[h],targ['curve'])
                if not pdia.empty:
                    pdia['horizon']=h;pdia['model']=family;pdia['factor']=pdia.index
                    predictive_rows.append(pdia.reset_index(drop=True))
            # v0.8 portfolio construction exactly: signed mean-confidence gate,
            # rolling realized factor vol, capped factor weights, 2Y-10Y overlay,
            # next-month P&L and one-way turnover costs.
            for family,panel in [('UNCONDITIONAL',baseline[h]),('CARRY_ONLY',carry_only[h]),
                                 *[(m,forecasts[(h,m)]) for m in valid_models if (h,m) in forecasts]]:
                bt=self._backtest(panel,cost_bps)
                backtests[(h,family)]=bt
                metrics=performance_metrics(bt['returns'],cash=cash.reindex(bt['returns'].index),turnover=bt['turnover'])
                metrics['horizon']=h;metrics['model']=family
                strategy_rows.append(metrics)
                if len(bt['factor_alloc']):
                    last_dt=bt['factor_alloc'].index.max()
                    alloc_rows.append(pd.Series({'horizon':h,'model':family,'formation_date':last_dt,
                                                  **bt['factor_alloc'].loc[last_dt].to_dict()}))
            uncond=backtests[(h,'UNCONDITIONAL')]['returns']
            for family in valid_models:
                if (h,family) not in backtests: continue
                r=backtests[(h,family)]['returns']; common=r.index.intersection(uncond.index)
                diff=(r.loc[common]-uncond.loc[common]).dropna()
                if len(diff)<12:continue
                vol=diff.std(ddof=1)
                classic=float(diff.mean()/(vol/np.sqrt(len(diff)))) if vol>0 else np.nan
                hac_fit=sm.OLS(diff.to_numpy(),np.ones((len(diff),1))).fit(cov_type='HAC',cov_kwds={'maxlags':min(6,len(diff)-1)})
                value_rows.append(pd.Series({'annualized_mean_spread':diff.mean()*12,
                    'spread_t_stat':classic,'spread_t_hac':float(hac_fit.tvalues[0]),'n_months':len(diff)},name=(h,family)))
            for cbps in cost_sensitivity:
                for family in valid_models:
                    key=(h,family)
                    if key not in forecasts: continue
                    bt=self._backtest(forecasts[key],float(cbps))
                    m=performance_metrics(bt['returns'],cash=cash.reindex(bt['returns'].index),turnover=bt['turnover'])
                    cost_rows.append(pd.Series({'cost_bps':cbps,'horizon':h,'model':family,
                        'CAGR':m.get('CAGR',np.nan),'Sharpe':m.get('Sharpe',np.nan),
                        'MaxDD':m.get('MaxDD',np.nan),'Turnover':m.get('AvgMonthlyTurnover',np.nan)}))
        if 12 in all_horizon_asset_data:
            fb_diag=fama_bliss_univariate_diagnostics(all_horizon_asset_data[12]['log_excess'],fb)
        else:fb_diag=pd.DataFrame()
        strat=pd.DataFrame(strategy_rows)
        if len(strat):strat=strat.set_index(['horizon','model']).sort_index()
        vals=pd.DataFrame(value_rows)
        if len(vals):vals.index=pd.MultiIndex.from_tuples(vals.index,names=['horizon','model'])
        coef_rows=[]
        for (h,mod),coefs in cp_coefficients.items():
            coef_rows.append(pd.Series({'horizon':h,'model':mod,**coefs.to_dict()}))
        meta={'engine_version':'0.2.4','legacy_reference':'Rates Factor Timing v0.8',
              'factor_representation':'funded L/S/C PCA3 portfolios; not a guarantee of priced independent shocks',
              'models_requested':list(MODEL_FAMILIES if models is None else models),
              'models_run':list(valid_models),'models_skipped_for_missing_macro':skipped,
              'horizons_months':list(horizons),'as_of':str(snapshot.as_of),
              'historical_protocol':'v0.8 overlapping H-month forward targets; train date s <= forecast date t-H',
              'forecast_scope':'retrospective OOS grid only; last H months not forecast due to historical v0.8 candidate-date convention',
              'state_macro_source':'provided original CPI INDPRO BREAKEVEN10Y levels' if state is not None else None,
              'macro_vintage_warning':'monthly observations are revised, not publication-vintage PIT data',
              'forecast_model_ridge':ridge,'min_train':min_train,'cost_bps':cost_bps,
              'timing_rule':'v0.8 mean-confidence (not next-realization sign) and next-month executions',
              'forecast_return_units':'decimal holding-period returns; annualized in diagnostics only as specified',
              'no_data_imputation':'missing FEDFUNDS observations not forward-filled beyond latest known observation',
              'source_curve_dates':[str(z.index.min().date()),str(z.index.max().date())]}
        return BondPremiaResearchResult(snapshot=snapshot,factor_targets=targets,
            forecasts=forecasts,unconditional_forecasts=baseline,carry_only_forecasts=carry_only,
            predictive_diagnostics=pd.concat(predictive_rows,ignore_index=True) if predictive_rows else pd.DataFrame(),
            strategy_results=strat,conditional_value_add=vals,
            cost_sensitivity=pd.DataFrame(cost_rows),fb_12m_diagnostics=fb_diag,
            cp_coefficients=pd.DataFrame(coef_rows),latest_allocations=pd.DataFrame(alloc_rows),
            backtests=backtests,metadata=meta)

    def _backtest(self, forecasts, cost_bps):
        factor_alloc,signal_detail=build_factor_allocations(forecasts,self.rates.factor_returns)
        overlay=combine_factor_and_maturity_weights(factor_alloc,self.rates.mimic_weights_panel,MAX_UNDERLYING_GROSS)
        ret,pnl,turnover=apply_overlay_next_month(overlay,self.rates.asset_total_returns,
            self.rates.asset_excess_returns,self.rates.cash_returns,cost_bps,core_returns=None,overlay_scale=1.)
        return {'returns':ret,'turnover':turnover,'factor_alloc':factor_alloc,
                'overlay_weights':overlay,'signal_detail':signal_detail,'overlay_pnl':pnl}


def load_v08_macro_csv(path):
    """Load original v0.8 macro levels from CSV with Date,CPI,INDPRO,BREAKEVEN10Y.
    CPI/INDPRO are levels, BREAKEVEN10Y is PERCENT per original FRED T10YIE.
    Not silently overwritten with fallback variables or par yield data.
    """
    x=pd.read_csv(path)
    date_col=next((c for c in x.columns if c.lower().strip() in {'date','observation_date'}),None)
    if date_col is None:raise ValueError('CSV needs Date column')
    dt=pd.to_datetime(x.pop(date_col),errors='raise')+pd.offsets.MonthEnd(0)
    x.index=dt;x=x[['CPI','INDPRO','BREAKEVEN10Y']].apply(pd.to_numeric,errors='coerce')
    return x.sort_index()
