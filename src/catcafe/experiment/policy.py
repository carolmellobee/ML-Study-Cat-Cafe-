"""Who should get the offer? Evaluate targeting policies on the randomised data.

Because treatment was assigned at random with p = 0.5, the value of *any*
targeting rule can be estimated without bias from the experiment itself:

    V(rule) = mean_i [ 1{T_i = rule(i)} * Y_i / 0.5 ]

Y_i is the member's 12-month value to the business: -(cancelled within 60d) *
12-month expected revenue, minus the offer cost if treated. Rules learned from
the data are cross-fitted (learn on one half, evaluate on the other) so they
do not grade their own homework.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import plotting as P
from ..config import REVENUE_HORIZON_MONTHS, SEED


def member_value(exp: pd.DataFrame, include_cost: bool = True) -> np.ndarray:
    p = exp["p_cancel_30d"].to_numpy()
    k = np.arange(REVENUE_HORIZON_MONTHS)
    clv12 = exp["monthly_price"].to_numpy() * ((1 - p[:, None]) ** k[None, :]).sum(1)
    v = -exp["cancelled_60d"].to_numpy() * clv12
    return v - exp["offer_cost"].to_numpy() if include_cost else v


def ipw_value(y: np.ndarray, t: np.ndarray, rule: np.ndarray) -> float:
    return float(np.mean((t == rule) * y / 0.5))


def _learn_cells(train: pd.DataFrame, y: np.ndarray, cell: str) -> set:
    keep = set()
    for c, g in train.assign(_y=y).groupby(cell):
        lift = g.loc[g["treatment"] == 1, "_y"].mean() - g.loc[g["treatment"] == 0, "_y"].mean()
        if lift > 0:
            keep.add(c)
    return keep


def evaluate_policies(exp: pd.DataFrame, boot: int = 1000, plot: bool = True) -> dict:
    exp = exp.copy()
    y = member_value(exp)
    t = exp["treatment"].to_numpy()
    p = exp["p_cancel_30d"].to_numpy()
    rules = {
        "Treat nobody": np.zeros(len(exp), int),
        "Treat everyone": np.ones(len(exp), int),
        "Top 30% by churn risk": (p >= np.quantile(p, 0.70)).astype(int),
        "Top 50% by churn risk": (p >= np.quantile(p, 0.50)).astype(int),
        "Tenure under 12 months": (exp["tenure_band"] != "12+ mo").astype(int).to_numpy(),
    }
    # cross-fitted rule: which tenure bands are worth treating?
    exp["cell"] = exp["tenure_band"]
    rng = np.random.default_rng(SEED + 40)
    fold = rng.random(len(exp)) < 0.5
    learned = np.zeros(len(exp), int)
    for a, b in ((fold, ~fold), (~fold, fold)):
        cells = _learn_cells(exp[a], y[a], "cell")
        learned[b] = exp.loc[b, "cell"].isin(cells).astype(int).to_numpy()
    rules["Learned tenure rule (cross-fitted)"] = learned
    full_cells = sorted(_learn_cells(exp, y, "cell"))

    cash = exp["net_revenue_90d"].to_numpy()
    y_ret = member_value(exp, include_cost=False)
    base = ipw_value(y, t, rules["Treat nobody"])
    base_cash = ipw_value(cash, t, rules["Treat nobody"])
    rows = []
    brng = np.random.default_rng(SEED + 41)
    bs = brng.integers(0, len(exp), (boot, len(exp)))
    for name, r in rules.items():
        v = ipw_value(y, t, r)
        diffs = np.array([ipw_value(y[b], t[b], r[b]) - ipw_value(y[b], t[b], rules["Treat nobody"][b])
                          for b in bs])
        rows.append({"policy": name, "share_treated": float(r.mean()),
                     "value_per_member_vs_nobody": v - base,
                     "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
                     "total_vs_nobody": (v - base) * len(exp),
                     "cash_90d_per_member_vs_nobody": ipw_value(cash, t, r) - base_cash,
                     # retention value created per treated member = the most an offer may cost
                     "break_even_cost_per_treated": (ipw_value(y_ret, t, r) - ipw_value(
                         y_ret, t, rules["Treat nobody"])) / r.mean() if r.mean() > 0 else np.nan})
    tab = pd.DataFrame(rows)
    return {"table": tab, "cells_worth_treating_full_sample": full_cells,
            "figure": _plot_policies(tab) if plot else None, "members": int(len(exp))}


def _plot_policies(tab: pd.DataFrame) -> str:
    t = tab[tab["policy"] != "Treat nobody"].iloc[::-1].reset_index(drop=True)
    fig, ax = P.plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(t))
    ax.hlines(y, t["ci_low"], t["ci_high"], color=P.AXIS, lw=2)
    colors = [P.RED if v < 0 else P.BLUE for v in t["value_per_member_vs_nobody"]]
    ax.scatter(t["value_per_member_vs_nobody"], y, color=colors, s=50, zorder=3,
               edgecolor=P.SURFACE, linewidth=2)
    for yi, (v, s) in enumerate(zip(t["value_per_member_vs_nobody"], t["share_treated"])):
        sign = "-" if v < 0 else "+"
        ax.text(v, yi + 0.24, f"{sign}${abs(v):.2f}  ({s:.0%} treated)", ha="center", color=P.INK_2,
                fontsize=8.5)
    ax.set_ylim(-0.5, len(t) - 0.2)
    ax.axvline(0, color=P.INK_2, lw=1)
    ax.set_yticks(y, t["policy"])
    ax.set_xlabel("12-month value per eligible member vs. no campaign ($, net of offer cost)")
    ax.set_title("Which members should get the offer?", pad=22)
    P.subtitle(ax, "Off-policy value estimated from the randomised test, 95% bootstrap CI")
    ax.grid(axis="y", visible=False)
    return P.save(fig, "exp_policy_value")
