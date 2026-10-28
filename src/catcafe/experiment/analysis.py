"""Fixed-horizon readout at day 60: SRM check, primary metric, guardrail, CUPED."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def srm_check(df: pd.DataFrame, expected=0.5) -> dict:
    n1 = int(df["treatment"].sum())
    n0 = int(len(df) - n1)
    chi2, p = stats.chisquare([n0, n1], [len(df) * (1 - expected), len(df) * expected])
    return {"control": n0, "treatment": n1, "chi2": float(chi2), "p_value": float(p),
            "srm_detected": bool(p < 0.001)}


def two_proportion(y: np.ndarray, t: np.ndarray) -> dict:
    y, t = np.asarray(y), np.asarray(t).astype(bool)
    p1, p0 = y[t].mean(), y[~t].mean()
    n1, n0 = t.sum(), (~t).sum()
    diff = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    pp = y.mean()
    se_pooled = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    z = diff / se_pooled
    # relative change with delta-method CI on the log ratio
    log_rr = np.log(p1 / p0)
    se_log = np.sqrt((1 - p1) / (p1 * n1) + (1 - p0) / (p0 * n0))
    return {"control_rate": float(p0), "treatment_rate": float(p1), "diff": float(diff),
            "ci_low": float(diff - 1.96 * se), "ci_high": float(diff + 1.96 * se),
            "relative": float(p1 / p0 - 1),
            "relative_ci_low": float(np.exp(log_rr - 1.96 * se_log) - 1),
            "relative_ci_high": float(np.exp(log_rr + 1.96 * se_log) - 1),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z))),
            "n_control": int(n0), "n_treatment": int(n1)}


def regression_adjusted(df: pd.DataFrame, outcome: str) -> dict:
    """Linear probability model with pre-treatment covariates and HC2 errors.
    Randomisation makes the covariates orthogonal to treatment, so this keeps the
    estimate unbiased while removing variance explained by baseline risk."""
    m = smf.ols(f"{outcome} ~ treatment + p_cancel_30d + np.log1p(tenure_days) + np.log1p(visits_30d)"
                " + C(billing_cycle) + C(plan)", data=df).fit(cov_type="HC2")
    b, se = m.params["treatment"], m.bse["treatment"]
    return {"diff": float(b), "se": float(se), "ci_low": float(b - 1.96 * se),
            "ci_high": float(b + 1.96 * se), "p_value": float(m.pvalues["treatment"])}


def mean_diff(y: np.ndarray, t: np.ndarray) -> dict:
    y, t = np.asarray(y, float), np.asarray(t).astype(bool)
    d = y[t].mean() - y[~t].mean()
    se = np.sqrt(y[t].var(ddof=1) / t.sum() + y[~t].var(ddof=1) / (~t).sum())
    return {"control_mean": float(y[~t].mean()), "treatment_mean": float(y[t].mean()),
            "diff": float(d), "se": float(se), "ci_low": float(d - 1.96 * se),
            "ci_high": float(d + 1.96 * se), "p_value": float(2 * stats.norm.sf(abs(d / se)))}


def cuped(y: np.ndarray, x: np.ndarray, t: np.ndarray) -> dict:
    """CUPED with a single pre-period covariate (theta estimated on pooled data)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    y_adj = y - theta * (x - x.mean())
    raw, adj = mean_diff(y, t), mean_diff(y_adj, t)
    adj["theta"] = float(theta)
    adj["variance_reduction"] = float(1 - (adj["se"] / raw["se"]) ** 2)
    return {"raw": raw, "cuped": adj}


def readout(df: pd.DataFrame) -> dict:
    t = df["treatment"].to_numpy()
    return {
        "srm": srm_check(df),
        "primary_cancel_60d": two_proportion(df["cancelled_60d"], t),
        "primary_regression_adjusted": regression_adjusted(df, "cancelled_60d"),
        "guardrail_net_revenue_90d": mean_diff(df["net_revenue_90d"], t),
        "secondary_visits_60d": cuped(df["visits_60d"], df["pre_visits_30d"], t),
        "offer_cost_per_treated": float(df.loc[df["treatment"] == 1, "offer_cost"].mean()),
    }
