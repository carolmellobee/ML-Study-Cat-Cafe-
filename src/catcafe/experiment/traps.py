"""The three ways this experiment can fool you.

1. Peeking   - checking the p-value every day and stopping at the first p < 0.05.
2. Novelty   - a big early effect that fades; a week-1 readout overstates it.
3. Reversal  - a positive average hiding a segment where the campaign backfires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

from .. import plotting as P
from ..config import SEED
from .analysis import two_proportion
from .simulate import DAYS, daily_hazard

LOOK_DAYS = np.array([10, 20, 30, 40, 50, 60])  # planned interim looks (end of day)


# ------------------------------------------------------------------ peeking --
def _cum_z(day_t, day_c, n_t, n_c):
    """z statistic of cumulative cancellation rates after each day, fixed cohorts."""
    ct = np.cumsum(np.bincount(day_t, minlength=DAYS)[:DAYS])
    cc = np.cumsum(np.bincount(day_c, minlength=DAYS)[:DAYS])
    pt, pc = ct / n_t, cc / n_c
    pp = (ct + cc) / (n_t + n_c)
    se = np.sqrt(pp * (1 - pp) * (1 / n_t + 1 / n_c))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (pt - pc) / se, 0.0)
    return z, pt, pc


def aa_simulation(pop: pd.DataFrame, sims: int = 4000, seed: int = SEED) -> np.ndarray:
    """A/A tests on the real risk mix: returns z trajectories (sims x DAYS)."""
    rng = np.random.default_rng(seed + 20)
    h = daily_hazard(pop)
    n = len(pop)
    Z = np.empty((sims, DAYS))
    for s in range(sims):
        t = rng.random(n) < 0.5
        day = np.floor(rng.exponential(1 / h)).astype(int)
        ev = day < DAYS
        Z[s], _, _ = _cum_z(day[t & ev], day[~t & ev], t.sum(), (~t).sum())
    return Z


def obf_constant(Z_looks: np.ndarray, alpha: float = 0.05) -> float:
    """Calibrate c so that P(any |Z_k| > c * sqrt(K/k)) = alpha under the null."""
    K = Z_looks.shape[1]
    w = np.sqrt(K / np.arange(1, K + 1))
    stat = (np.abs(Z_looks) / w).max(1)
    return float(np.quantile(stat, 1 - alpha))


def peeking(pop: pd.DataFrame, exp: pd.DataFrame, sims: int = 4000) -> dict:
    Z = aa_simulation(pop, sims)
    crit = stats.norm.ppf(0.975)
    half = sims // 2
    looks_idx = LOOK_DAYS - 1
    c = obf_constant(Z[:half, looks_idx])
    K = len(LOOK_DAYS)
    bounds = c * np.sqrt(K / np.arange(1, K + 1))
    held = Z[half:]
    fpr = {
        "final_look_only": float((np.abs(held[:, -1]) > crit).mean()),
        "weekly_peeks": float((np.abs(held[:, 6::7]) > crit).any(1).mean()),
        "daily_peeks": float((np.abs(held[:, 3:]) > crit).any(1).mean()),
        "obf_6_looks": float((np.abs(held[:, looks_idx]) > bounds).any(1).mean()),
    }
    # the real experiment, watched every day
    t = exp["treatment"].to_numpy() == 1
    d = exp["cancel_day"].to_numpy()
    ev = d >= 0
    z, pt, pc = _cum_z(d[t & ev], d[~t & ev], t.sum(), (~t).sum())
    rel = np.where(pc > 0, pt / pc - 1, np.nan)
    # day numbers are 1-based: z[i] is the cumulative test after day i + 1
    first_naive = int(np.argmax(np.abs(z[3:]) > crit) + 3 + 1) if (np.abs(z[3:]) > crit).any() else None
    obf_hits = np.abs(z[looks_idx]) > bounds
    first_obf = int(LOOK_DAYS[np.argmax(obf_hits)]) if obf_hits.any() else None
    traj = pd.DataFrame({"day": np.arange(1, DAYS + 1), "z": z, "relative_effect": rel,
                         "treat_rate": pt, "control_rate": pc})
    out = {
        "false_positive_rate": fpr, "obf_constant": c,
        "obf_boundaries": dict(zip(LOOK_DAYS.tolist(), bounds.round(3).tolist())),
        "first_day_naive_p_below_05": first_naive,
        "effect_at_naive_stop": float(rel[first_naive - 1]) if first_naive else None,
        "first_obf_crossing_day": first_obf,
        "effect_at_obf_stop": float(rel[first_obf - 1]) if first_obf else None,
        "final_effect": float(rel[-1]), "final_z": float(z[-1]),
        "trajectory": traj,
    }
    out["figures"] = {"aa": _plot_aa(fpr), "trajectory": _plot_trajectory(traj, bounds, first_naive)}
    return out


def _plot_aa(fpr: dict) -> str:
    labels = ["Look once at day 60", "Peek weekly", "Peek daily", "6 planned looks\n(O'Brien-Fleming)"]
    vals = [fpr["final_look_only"], fpr["weekly_peeks"], fpr["daily_peeks"], fpr["obf_6_looks"]]
    colors = [P.BLUE, P.ORANGE, P.ORANGE, P.BLUE]
    fig, ax = P.plt.subplots(figsize=(6.8, 4.2))
    ax.bar(labels, np.array(vals) * 100, color=colors, width=0.58)
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 0.6, f"{v:.1%}", ha="center", color=P.INK_2, fontsize=9.5)
    ax.axhline(5, color=P.INK_2, lw=1, ls="--")
    ax.text(-0.42, 5.5, "promised 5%", color=P.INK_2, fontsize=8.5, ha="left")
    ax.set_ylabel("False positive rate in A/A tests (%)")
    ax.set_title(f"Daily peeking turns a 5% error rate into {fpr['daily_peeks']:.0%}", pad=22)
    P.subtitle(ax, "2,000 simulated A/A tests on the real member risk mix; nothing is different between arms")
    ax.grid(axis="x", visible=False)
    return P.save(fig, "exp_peeking_aa")


def _plot_trajectory(traj: pd.DataFrame, bounds: np.ndarray, first_naive) -> str:
    fig, ax = P.plt.subplots(figsize=(7.4, 4.6))
    ax.plot(traj["day"], -traj["z"], color=P.BLUE, label="Observed z (benefit = up)")
    ax.axhline(1.96, color=P.ORANGE, lw=1.2, ls="--", label="Naive threshold 1.96 every day")
    ax.step(np.r_[0, LOOK_DAYS], np.r_[bounds[0], bounds], where="pre", color=P.INK_2, lw=1.2,
            label="O'Brien-Fleming boundary (6 looks)")
    ax.scatter(LOOK_DAYS, bounds, color=P.INK_2, s=14, zorder=3)
    if first_naive:
        zi = -traj["z"].iloc[first_naive - 1]
        ax.scatter([first_naive], [zi], color=P.ORANGE, s=40, zorder=4, edgecolor=P.SURFACE, linewidth=2)
        ax.annotate(f"Day {first_naive}: first naive 'p < 0.05'.\nEstimate then: "
                    f"{traj['relative_effect'].iloc[first_naive - 1]:+.0%} cancellations",
                    (first_naive, zi), xytext=(first_naive + 14, max(zi - 1.6, 0.35)),
                    color=P.INK_2, fontsize=9, arrowprops=dict(arrowstyle="-", color=P.AXIS))
    ax.set_xlim(0, 61)
    ax.set_ylim(0, max(6.5, float(bounds[0]) + 0.5, float((-traj["z"]).max()) + 0.8))
    ax.set_xlabel("Day of experiment")
    ax.set_ylabel("z statistic, cumulative cancellations")
    ax.set_title("The real test, watched every day", pad=22)
    P.subtitle(ax, f"Final estimate at day 60: {traj['relative_effect'].iloc[-1]:+.0%} cancellations "
                   f"(z = {-traj['z'].iloc[-1]:.2f})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8.5)
    return P.save(fig, "exp_peeking_trajectory")


# ------------------------------------------------------------------ novelty --
def novelty(exp: pd.DataFrame, daily: pd.DataFrame) -> dict:
    t_map = exp.set_index("member_id")["treatment"]
    d = daily.assign(week=daily["day"] // 7)
    d = d[d["week"] < 8]
    wk = d.groupby(["member_id", "week"]).agg(visits=("visits", "sum"), days=("day", "size")).reset_index()
    wk = wk[wk["days"] == 7]  # members active the whole week
    wk["treatment"] = wk["member_id"].map(t_map)
    rows = []
    for w, g in wk.groupby("week"):
        a, b = g.loc[g["treatment"] == 1, "visits"], g.loc[g["treatment"] == 0, "visits"]
        r = a.mean() / b.mean()
        se_log = np.sqrt(a.var() / (len(a) * a.mean() ** 2) + b.var() / (len(b) * b.mean() ** 2))
        rows.append({"week": int(w) + 1, "lift": r - 1, "ci_low": r * np.exp(-1.96 * se_log) - 1,
                     "ci_high": r * np.exp(1.96 * se_log) - 1})
    visit = pd.DataFrame(rows)
    f = lambda w, a, b, tau: a + b * np.exp(-(w - 1) / tau)  # noqa: E731
    (a, b, tau), _ = optimize.curve_fit(f, visit["week"], visit["lift"], p0=[0.03, 0.4, 1.5],
                                        bounds=([-1, 0, 0.1], [1, 3, 20]))

    # cancellation hazard ratio by 10-day period, among members still at risk
    per = []
    day = exp["cancel_day"].to_numpy()
    tr = exp["treatment"].to_numpy() == 1
    for lo in range(0, DAYS, 10):
        at_risk = (day < 0) | (day >= lo)
        evt = (day >= lo) & (day < lo + 10)
        e1, e0 = evt[tr & at_risk].sum(), evt[~tr & at_risk].sum()
        n1, n0 = (tr & at_risk).sum(), (~tr & at_risk).sum()
        hr = (e1 / n1) / (e0 / n0)
        se = np.sqrt(1 / e1 + 1 / e0)
        per.append({"days": f"{lo + 1}-{lo + 10}", "hazard_ratio": hr,
                    "ci_low": hr * np.exp(-1.96 * se), "ci_high": hr * np.exp(1.96 * se),
                    "events_t": int(e1), "events_c": int(e0)})
    hz = pd.DataFrame(per)
    early = two_proportion((day >= 0) & (day < 14), tr)
    return {"visit_lift_by_week": visit, "hazard_ratio_by_period": hz,
            "visit_lift_fit": {"long_run": float(a), "initial_extra": float(b), "tau_weeks": float(tau),
                               "half_life_weeks": float(tau * np.log(2))},
            "week2_readout_cancel": early,
            "figures": {"novelty": _plot_novelty(visit, hz, (a, b, tau))}}


def _plot_novelty(visit, hz, fit) -> str:
    fig, (a1, a2) = P.plt.subplots(1, 2, figsize=(11, 4.4))
    a1.errorbar(visit["week"], visit["lift"] * 100,
                yerr=[(visit["lift"] - visit["ci_low"]) * 100, (visit["ci_high"] - visit["lift"]) * 100],
                fmt="o", color=P.BLUE, ecolor=P.AXIS, capsize=3, markersize=7)
    w = np.linspace(1, 8, 100)
    a, b, tau = fit
    a1.plot(w, (a + b * np.exp(-(w - 1) / tau)) * 100, color=P.BLUE, lw=1.5, alpha=0.6)
    a1.axhline(0, color=P.AXIS, lw=1)
    a1.set_xlabel("Week since campaign")
    a1.set_ylabel("Visits per member, treatment vs control (%)")
    a1.set_title("Engagement bump fades within weeks", pad=8)
    x = np.arange(len(hz))
    a2.errorbar(x, hz["hazard_ratio"], yerr=[hz["hazard_ratio"] - hz["ci_low"],
                                             hz["ci_high"] - hz["hazard_ratio"]],
                fmt="o", color=P.ORANGE, ecolor=P.AXIS, capsize=3, markersize=7)
    a2.axhline(1, color=P.INK_2, lw=1, ls="--")
    a2.set_xticks(x, hz["days"])
    a2.set_xlabel("Days since campaign")
    a2.set_ylabel("Cancellation hazard ratio (T / C)")
    a2.set_title("Cancellation hazard ratio by 10-day period", pad=8)
    fig.suptitle("Novelty effect", x=0.06, ha="left", fontweight="bold", color=P.INK)
    fig.tight_layout()
    return P.save(fig, "exp_novelty")


# ----------------------------------------------------------------- segments --
SEGMENT_CUTS = {"tenure_band": ["0-3 mo", "3-6 mo", "6-12 mo", "12+ mo"],
                "billing_cycle": ["monthly", "annual"], "plan": ["basic", "plus", "family"]}


def segments(exp: pd.DataFrame, plot: bool = True) -> dict:
    rows = []
    for cut, levels in SEGMENT_CUTS.items():
        for lv in levels:
            g = exp[exp[cut] == lv]
            r = two_proportion(g["cancelled_60d"], g["treatment"])
            rows.append({"cut": cut, "segment": lv, "members": len(g), **r})
    seg = pd.DataFrame(rows)
    seg["p_holm"] = multipletests(seg["p_value"], method="holm")[1]
    seg["reverses"] = seg["diff"] > 0
    # interaction test for the tenure cut (pre-registered as the primary heterogeneity hypothesis)
    full = smf.logit("cancelled_60d ~ treatment * C(tenure_band)", data=exp).fit(disp=0)
    red = smf.logit("cancelled_60d ~ treatment + C(tenure_band)", data=exp).fit(disp=0)
    lr = 2 * (full.llf - red.llf)
    p_int = stats.chi2.sf(lr, df=full.df_model - red.df_model)
    # out-of-sample replication: discover on one random half, confirm on the other
    rng = np.random.default_rng(SEED + 30)
    half = rng.random(len(exp)) < 0.5
    rep = []
    for name, part in (("discovery_half", exp[half]), ("confirmation_half", exp[~half])):
        g = part[part["tenure_band"] == "12+ mo"]
        r = two_proportion(g["cancelled_60d"], g["treatment"])
        rep.append({"sample": name, **{k: r[k] for k in ("diff", "ci_low", "ci_high", "p_value")}})
    overall = two_proportion(exp["cancelled_60d"], exp["treatment"])
    return {"table": seg, "interaction_lr": float(lr), "interaction_p": float(p_int),
            "replication_12plus": pd.DataFrame(rep),
            "figure": _plot_forest(seg, overall) if plot else None}


def _plot_forest(seg: pd.DataFrame, overall: dict) -> str:
    rows = [{"segment": "All members", "members": overall["n_control"] + overall["n_treatment"],
             "diff": overall["diff"], "ci_low": overall["ci_low"], "ci_high": overall["ci_high"],
             "p_holm": np.nan, "reverses": overall["diff"] > 0, "cut": "all"}]
    seg = pd.concat([pd.DataFrame(rows), seg], ignore_index=True).iloc[::-1].reset_index(drop=True)
    fig, ax = P.plt.subplots(figsize=(8.4, 5.6))
    y = np.arange(len(seg))
    colors = [P.RED if r else P.BLUE for r in seg["reverses"]]
    ax.hlines(y, seg["ci_low"] * 100, seg["ci_high"] * 100, color=P.AXIS, lw=2)
    ax.scatter(seg["diff"] * 100, y, color=colors, s=46, zorder=3, edgecolor=P.SURFACE, linewidth=2)
    ax.axvline(0, color=P.INK_2, lw=1)
    lbl = [f"{s}  (n={n:,})" for s, n in zip(seg["segment"], seg["members"])]
    ax.set_yticks(y, lbl)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("All members"):
            t.set_fontweight("bold")
    tr = ax.get_yaxis_transform()
    ax.text(1.02, len(seg) - 0.4, "diff (Holm p)", transform=tr, color=P.MUTED, fontsize=8.5)
    for yi, (d, p) in enumerate(zip(seg["diff"], seg["p_holm"])):
        ptxt = "" if np.isnan(p) else ("  (<0.001)" if p < 0.001 else f"  ({p:.3f})")
        ax.text(1.02, yi, f"{d * 100:+.1f} pp{ptxt}", transform=tr, va="center", ha="left",
                color=P.INK_2, fontsize=8.5)
    cuts = seg["cut"].to_numpy()
    for yi in range(1, len(seg)):
        if cuts[yi] != cuts[yi - 1]:
            ax.axhline(yi - 0.5, color=P.GRID, lw=1)
    ax.set_xlabel("Change in 60-day cancellation rate, treatment minus control (pp)")
    ax.set_title("The average hides a reversal", pad=22)
    P.subtitle(ax, "Pre-registered segments, 95% CI. Left of zero = campaign retains members.")
    ax.grid(axis="y", visible=False)
    return P.save(fig, "exp_segments_forest")
