"""Translate calibrated churn probabilities into revenue at risk.

Definitions (stated, because the headline number depends on them):

* MRR at risk (next 30 days) = sum_i p_i * monthly_price_i
  Recurring revenue expected to walk out from requests submitted in January.
* 12-month revenue at risk     = sum_i monthly_price_i * sum_{j=1..11} (1 - (1 - p_i)^j)
  Expected revenue lost over the next 12 months versus "nobody leaves", assuming
  each member keeps their current monthly hazard and a request in month k
  removes revenue from month k+1 onward (30-day notice). Constant hazard
  overstates risk for new members (their hazard falls with tenure) - treat it
  as an upper-end planning figure, not a forecast.

Revenue at risk is not revenue you can save. What you can save is
risk x (treatment effect) - cost, and the treatment effect comes from the experiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import plotting as P
from .config import REVENUE_HORIZON_MONTHS


def tenure_band(days: pd.Series) -> pd.Series:
    return pd.cut(days, [-1, 90, 180, 365, 10_000], labels=["0-3 mo", "3-6 mo", "6-12 mo", "12+ mo"])


def revenue_at_risk(scored: pd.DataFrame, driver_share: pd.DataFrame | None = None) -> dict:
    df = scored.copy()
    p, mrr = df["p_cancel_30d"].to_numpy(), df["monthly_price"].to_numpy()
    j = np.arange(1, REVENUE_HORIZON_MONTHS)
    lost_months = (1 - (1 - p[:, None]) ** j[None, :]).sum(1)
    df["mrr_at_risk_30d"] = p * mrr
    df["revenue_at_risk_12m"] = mrr * lost_months
    df["tenure_band"] = tenure_band(df["tenure_days"])
    df["risk_decile"] = pd.qcut(df["p_cancel_30d"].rank(method="first"), 10,
                                labels=list(range(10, 0, -1))).astype(int)  # 1 = riskiest

    total_mrr = float(mrr.sum())
    tot30, tot12 = float(df["mrr_at_risk_30d"].sum()), float(df["revenue_at_risk_12m"].sum())
    by_decile = df.groupby("risk_decile").agg(
        members=("member_id", "size"), mean_p=("p_cancel_30d", "mean"),
        mrr=("monthly_price", "sum"), revenue_at_risk_12m=("revenue_at_risk_12m", "sum")).reset_index()
    by_decile["share_of_risk"] = by_decile["revenue_at_risk_12m"] / tot12

    def by(col):
        g = df.groupby(col, observed=True).agg(
            members=("member_id", "size"), mean_p=("p_cancel_30d", "mean"),
            mrr=("monthly_price", "sum"), mrr_at_risk_30d=("mrr_at_risk_30d", "sum"),
            revenue_at_risk_12m=("revenue_at_risk_12m", "sum")).reset_index()
        g["share_of_risk"] = g["revenue_at_risk_12m"] / tot12
        return g

    out = {
        "members": int(len(df)), "total_mrr": total_mrr,
        "mrr_at_risk_30d": tot30, "revenue_at_risk_12m": tot12,
        "annual_run_rate": total_mrr * 12,
        "share_of_annual_run_rate": tot12 / (total_mrr * 12),
        "top_decile_share": float(by_decile.loc[by_decile["risk_decile"] == 1, "share_of_risk"].iloc[0]),
        "top_3_deciles_share": float(by_decile.loc[by_decile["risk_decile"] <= 3, "share_of_risk"].sum()),
        "by_decile": by_decile, "by_plan": by("plan"), "by_tenure": by("tenure_band"),
        "by_billing": by("billing_cycle"), "frame": df,
    }
    if driver_share is not None:
        alloc = driver_share.mul(df["revenue_at_risk_12m"].to_numpy(), axis=0).sum()
        # Only the *excess* over a baseline member is attributable; we report the split of the total.
        out["by_driver"] = (alloc / alloc.sum() * tot12).sort_values(ascending=False)
    out["figures"] = {"concentration": _plot_concentration(df, tot12)} if tot12 > 0 else {}
    if driver_share is not None and tot12 > 0:
        out["figures"]["drivers"] = _plot_drivers(out["by_driver"])
    return out


def _plot_concentration(df: pd.DataFrame, tot: float) -> str:
    d = df.sort_values("p_cancel_30d", ascending=False)
    x = np.arange(1, len(d) + 1) / len(d) * 100
    y = d["revenue_at_risk_12m"].cumsum().to_numpy() / tot * 100
    fig, ax = P.plt.subplots(figsize=(6.6, 4.8))
    ax.plot(x, y, color=P.BLUE)
    ax.plot([0, 100], [0, 100], ls="--", lw=1, color=P.MUTED)
    for q in (10, 30):
        yi = float(np.interp(q, x, y))
        ax.scatter([q], [yi], color=P.BLUE, s=36, zorder=3, edgecolor=P.SURFACE, linewidth=2)
        ax.annotate(f"Top {q}% of members\n= {yi:.0f}% of revenue at risk", (q, yi),
                    xytext=(q + 8, yi - 14), color=P.INK_2, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=P.AXIS))
    ax.set_xlabel("% of active members, riskiest first")
    ax.set_ylabel("% of 12-month revenue at risk")
    ax.set_title("Revenue at risk is concentrated, but not extremely", pad=22)
    P.subtitle(ax, "Live base on 1-Jan-2026")
    return P.save(fig, "revenue_concentration")


def _plot_drivers(s: pd.Series) -> str:
    s = s[s > 0][::-1]
    fig, ax = P.plt.subplots(figsize=(7.4, 3.8))
    ax.barh(s.index, s.values / 1000, color=P.BLUE, height=0.6)
    for yi, v in enumerate(s.values):
        ax.text(v / 1000 + 1, yi, f"${v / 1000:,.0f}k", va="center", color=P.INK_2, fontsize=9)
    ax.set_xlabel("12-month revenue at risk attributed (thousand $)")
    ax.set_title("Revenue at risk by model driver", pad=22)
    P.subtitle(ax, "Each member's risk split by their positive SHAP contributions. Attribution, not causation.")
    ax.grid(axis="y", visible=False)
    return P.save(fig, "revenue_by_driver")
