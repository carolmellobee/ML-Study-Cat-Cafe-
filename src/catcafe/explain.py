"""SHAP explanations: global drivers, shape of each effect, single-member reason codes.

SHAP tells you how the *model* uses a feature, in log-odds. It is not a causal
effect: "booking failures raise predicted risk" does not prove that fixing
bookings would retain those members. That needs an experiment.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap

from . import plotting as P
from .config import SEED
from .features import CATEGORICAL, FEATURES, as_model_frame

LABELS = {
    "plan": "Plan", "billing_cycle": "Billing cycle", "location": "Location",
    "acquisition_channel": "Acquisition channel", "tenure_days": "Tenure (days)",
    "distance_km": "Distance to cafe (km)", "monthly_price": "Monthly price",
    "price_increase_exposed": "Hit by 2025 price increase",
    "price_increase_recent": "Price increase in last 60 days",
    "booking_failure_rate_90d": "Share of bookings that failed, 90d", "visits_30d": "Visits, last 30 days",
    "visits_31_90d": "Visits, days 31-90", "visit_trend": "Visit trend (30d vs prior)",
    "days_since_last_visit": "Days since last visit", "lifetime_visits": "Lifetime visits",
    "visits_per_month": "Visits per month of tenure", "booking_failures_90d": "Failed bookings, 90d",
    "support_tickets_90d": "Support tickets, 90d", "events_90d": "Events attended, 90d",
    "adopted_cat_90d": "Adopted a cat, last 90d", "adopted_cat_ever": "Has adopted a cat",
}

DRIVER_GROUPS = {
    "Low or falling engagement": ["visits_30d", "visits_31_90d", "visit_trend",
                                       "days_since_last_visit", "lifetime_visits",
                                       "visits_per_month", "events_90d"],
    "Capacity & service (failed bookings, tickets)": ["booking_failures_90d", "booking_failure_rate_90d",
                                                      "support_tickets_90d", "location"],
    "Monthly billing & plan": ["plan", "billing_cycle"],
    "Recent price increase": ["price_increase_recent"],
    "Adopted a cat": ["adopted_cat_90d", "adopted_cat_ever"],
    "Early tenure": ["tenure_days"],
    "Channel & distance": ["acquisition_channel", "distance_km"],
}


def _numeric_view(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    for c in CATEGORICAL:
        Xn[c] = Xn[c].cat.codes
    return Xn


def shap_values(model, frame: pd.DataFrame, features: list[str] | None = None) -> tuple[np.ndarray, float]:
    X = as_model_frame(frame, features)
    exp = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # shap notes its LightGBM output format
        sv = exp.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    base = exp.expected_value
    base = float(base[1] if np.ndim(base) else base)
    return np.asarray(sv), base


def explain_test(model, test_frame: pd.DataFrame, n: int = 5000) -> dict:
    sample = test_frame.sample(min(n, len(test_frame)), random_state=SEED)
    sv, base = shap_values(model, sample)
    X = as_model_frame(sample)
    imp = (pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(0)})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp["label"] = imp["feature"].map(LABELS)

    figs = {"importance": _plot_importance(imp), "beeswarm": _plot_beeswarm(sv, X, base),
            "dependence": _plot_dependence(sv, X)}
    adopt = FEATURES.index("adopted_cat_90d")
    a = X["adopted_cat_90d"].to_numpy() == 1
    odds_mult = float(np.exp(sv[a, adopt].mean() - sv[~a, adopt].mean())) if a.any() else float("nan")
    return {"importance": imp, "figures": figs, "base_log_odds": base,
            "adopted_cat_odds_multiplier": odds_mult}


def reason_codes(model, scored: pd.DataFrame, top_k: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    sv, base = shap_values(model, scored)
    names = np.array([LABELS[f] for f in FEATURES])
    order = np.argsort(-sv, axis=1)[:, :top_k]
    out = scored[["member_id", "plan", "billing_cycle", "tenure_days", "monthly_price",
                  "p_cancel_30d"]].copy()
    for k in range(top_k):
        out[f"reason_{k + 1}"] = names[order[:, k]]
    return out, sv


def driver_share(sv: np.ndarray) -> pd.DataFrame:
    """Per member, split the positive (risk-raising) SHAP mass across driver groups."""
    pos = np.clip(sv, 0, None)
    tot = pos.sum(1, keepdims=True)
    tot[tot == 0] = 1
    share = pos / tot
    cols = {}
    for g, feats in DRIVER_GROUPS.items():
        idx = [FEATURES.index(f) for f in feats]
        cols[g] = share[:, idx].sum(1)
    return pd.DataFrame(cols)


def waterfall_member(model, scored: pd.DataFrame, member_idx: int) -> str:
    row = scored.iloc[[member_idx]]
    sv, base = shap_values(model, row)
    sv = sv[0]
    X = as_model_frame(row).iloc[0]
    order = np.argsort(-np.abs(sv))[:8]
    rest = sv.sum() - sv[order].sum()
    labels = [f"{LABELS[FEATURES[j]]} = {_fmt(X[FEATURES[j]])}" for j in order] + ["All other features"]
    vals = list(sv[order]) + [rest]
    fig, ax = P.plt.subplots(figsize=(7.6, 4.8))
    y = np.arange(len(vals))[::-1]
    cum = base
    for yi, v, lab in zip(y, vals, labels):
        ax.barh(yi, v, left=cum, color=P.RED if v > 0 else P.BLUE, height=0.62)
        ax.text(cum + v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", color=P.INK_2, fontsize=8.5)
        cum += v
    ax.set_yticks(y, labels)
    ax.axvline(base, color=P.MUTED, lw=1, ls="--")
    p_base = 1 / (1 + np.exp(-base))
    p_final = 1 / (1 + np.exp(-cum))
    ax.set_xlabel("Contribution to log-odds of cancelling (model scale, before recalibration)")
    ax.set_title(f"Why member {row['member_id'].iloc[0]} is flagged", pad=22)
    P.subtitle(ax, f"Average member {p_base:.1%} -> this member {p_final:.1%} (raw model score)."
                   "  Red raises risk, blue lowers it.")
    ax.grid(axis="y", visible=False)
    return P.save(fig, "shap_waterfall_member")


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.1f}" if v != int(v) else f"{int(v)}"
    return str(v)


def _plot_importance(imp: pd.DataFrame) -> str:
    top = imp.head(12)[::-1]
    fig, ax = P.plt.subplots(figsize=(7, 4.8))
    ax.barh(top["label"], top["mean_abs_shap"], color=P.BLUE, height=0.62)
    for yi, v in enumerate(top["mean_abs_shap"]):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", color=P.INK_2, fontsize=8.5)
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("What the churn model relies on", pad=22)
    P.subtitle(ax, "Global importance on 5,000 test-month member snapshots")
    ax.grid(axis="y", visible=False)
    return P.save(fig, "shap_importance")


def _plot_beeswarm(sv, X, base) -> str:
    Xn = _numeric_view(X)
    exp = shap.Explanation(values=sv, base_values=np.full(len(sv), base), data=Xn.to_numpy(),
                           feature_names=[LABELS[f] for f in FEATURES])
    P.plt.figure(figsize=(8, 5.6))
    np.random.seed(SEED)  # shap's beeswarm jitters points with the global RNG
    shap.plots.beeswarm(exp, max_display=12, color=P.DIVERGING, show=False, plot_size=None)
    fig = P.plt.gcf()
    ax = P.plt.gca()
    ax.set_title("SHAP summary: direction and spread of each driver", pad=22, loc="left")
    P.subtitle(ax, "Each dot is a member-month. Right = higher predicted cancellation risk.")
    return P.save(fig, "shap_beeswarm")


def _plot_dependence(sv, X) -> str:
    feats = ["visits_per_month", "days_since_last_visit", "tenure_days", "distance_km"]
    fig, axes = P.plt.subplots(2, 2, figsize=(9.2, 6.8))
    rng = np.random.default_rng(SEED)
    for ax, f in zip(axes.ravel(), feats):
        j = FEATURES.index(f)
        x = X[f].astype(float).to_numpy()
        discrete = np.allclose(x, np.round(x)) and x.max() < 60
        jitter = rng.uniform(-0.25, 0.25, len(x)) if discrete else 0
        ax.scatter(x + jitter, sv[:, j], s=5, alpha=0.25, color=P.BLUE, linewidths=0)
        ax.axhline(0, color=P.AXIS, lw=1)
        q = np.quantile(x, 0.99)
        pad = 0.5 if discrete else q * 0.02
        ax.set_xlim(left=min(x.min(), 0) - pad, right=q * 1.05 + pad)
        ax.set_title(LABELS[f], fontsize=10.5, pad=6)
        ax.set_ylabel("SHAP (log-odds)")
    fig.suptitle("Shape of the main effects", x=0.06, ha="left", fontweight="bold", color=P.INK)
    fig.tight_layout()
    return P.save(fig, "shap_dependence")
