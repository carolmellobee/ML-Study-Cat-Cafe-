"""Pre-test design: power analysis, done *before* any outcome data exists.

A cat cafe cannot "run the test longer to get more users" - the member base is
fixed (~7k). So the useful question is the reverse of the textbook one:
given the members we have, what is the smallest effect we can reliably detect,
and is that effect worth detecting?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from .. import plotting as P
from ..config import SEED
from ..config import REVENUE_HORIZON_MONTHS
from .simulate import DAYS, DISCOUNT, GUEST_PASS_COST, GUEST_PASS_REDEEM, daily_hazard, eligible_population

ALPHA, POWER = 0.05, 0.80


def n_per_arm(p0: float, rel_mde: float, alpha=ALPHA, power=POWER) -> int:
    p1 = p0 * (1 - rel_mde)
    h = abs(proportion_effectsize(p0, p1))
    return int(np.ceil(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power,
                                                    ratio=1.0, alternative="two-sided")))


def mde_given_n(p0: float, n: int, alpha=ALPHA, power=POWER) -> float:
    """Smallest relative reduction detectable with n per arm (normal approximation)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    lo, hi = 1e-4, 0.99
    for _ in range(60):
        mid = (lo + hi) / 2
        p1 = p0 * (1 - mid)
        se = np.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
        if (p0 - p1) / se >= za + zb:
            hi = mid
        else:
            lo = mid
    return hi


def simulated_power(p_member: np.ndarray, rel_effect: float, sims: int = 2000, seed: int = SEED) -> float:
    """Monte Carlo check of the analytic answer using the real heterogeneous risk mix."""
    rng = np.random.default_rng(seed + 10)
    n = len(p_member)
    hits = 0
    for _ in range(sims):
        t = rng.random(n) < 0.5
        y = rng.random(n) < np.where(t, p_member * (1 - rel_effect), p_member)
        a, b = y[t].mean(), y[~t].mean()
        pp = y.mean()
        se = np.sqrt(pp * (1 - pp) * (1 / t.sum() + 1 / (~t).sum()))
        hits += abs(a - b) / se > stats.norm.ppf(1 - ALPHA / 2)
    return hits / sims


def power_analysis(scored: pd.DataFrame) -> dict:
    pop = eligible_population(scored)
    h = daily_hazard(pop)
    p60 = 1 - np.exp(-h * DAYS)  # expected 60-day cancellation probability per member
    p0 = float(p60.mean())
    n_avail = len(pop) // 2

    grid = [0.10, 0.15, 0.20, 0.25, 0.30]
    table = pd.DataFrame({"relative_reduction": grid,
                          "absolute_pp": [p0 * g * 100 for g in grid],
                          "n_per_arm": [n_per_arm(p0, g) for g in grid]})
    table["feasible"] = table["n_per_arm"] <= n_avail
    mde = mde_given_n(p0, n_avail)

    seg = pop.groupby("tenure_band").agg(members=("member_id", "size"))
    seg["baseline_60d"] = pd.Series(p60, index=pop.index).groupby(pop["tenure_band"]).mean()
    seg["mde_relative"] = [mde_given_n(b, m // 2) for b, m in zip(seg["baseline_60d"], seg["members"])]
    seg = seg.reset_index()

    sim_pow = simulated_power(p60, mde)

    # Economics before the test: what effect would make "treat everyone" pay for itself?
    price = pop["monthly_price"].to_numpy()
    p30 = pop["p_cancel_30d"].to_numpy()
    clv12 = price * ((1 - p30[:, None]) ** np.arange(REVENUE_HORIZON_MONTHS)[None, :]).sum(1)
    cost = float(np.mean(DISCOUNT * price + GUEST_PASS_REDEEM * GUEST_PASS_COST))
    break_even_rel = cost / float(np.mean(p60 * clv12))
    fig = _plot_power_curve(p0, n_avail, mde)

    return {
        "eligible_members": int(len(pop)), "n_per_arm_available": int(n_avail),
        "baseline_60d_cancel_rate": p0, "alpha": ALPHA, "power": POWER,
        "requirements": table, "mde_relative": float(mde), "mde_absolute_pp": float(mde * p0 * 100),
        "simulated_power_at_mde": float(sim_pow), "segment_mde": seg, "figure": fig,
        "expected_cost_per_treated": cost, "mean_clv12": float(clv12.mean()),
        "break_even_relative_reduction": float(break_even_rel),
        # The interaction between treatment and a segment has ~2x the SE of a segment effect,
        # so detecting an interaction of size d needs ~4x the n of detecting d overall.
        "interaction_sample_multiplier": 4.0,
    }


def _plot_power_curve(p0: float, n_avail: int, mde: float) -> str:
    rel = np.linspace(0.02, 0.40, 200)
    za = stats.norm.ppf(1 - ALPHA / 2)
    fig, ax = P.plt.subplots(figsize=(6.8, 4.6))
    for n, color, label in ((n_avail, P.BLUE, f"Full base: {n_avail:,} per arm"),
                            (n_avail // 2, P.ORANGE, f"Half the base: {n_avail // 2:,} per arm"),
                            (n_avail // 4, P.AQUA, f"A 25% segment: {n_avail // 4:,} per arm")):
        p1 = p0 * (1 - rel)
        se = np.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
        pw = stats.norm.cdf((p0 - p1) / se - za)
        ax.plot(rel * 100, pw * 100, color=color, label=label)
    ax.axhline(80, color=P.AXIS, lw=1, ls="--")
    ax.scatter([mde * 100], [80], color=P.BLUE, s=40, zorder=3, edgecolor=P.SURFACE, linewidth=2)
    ax.annotate(f"MDE = {mde:.0%} relative\n({mde * p0 * 100:.1f} pp)", (mde * 100, 80),
                xytext=(mde * 100 + 4, 55), color=P.INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=P.AXIS))
    ax.set_xlabel("True relative reduction in 60-day cancellations (%)")
    ax.set_ylabel("Power (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Power analysis: what can a fixed member base detect?", pad=22)
    P.subtitle(ax, f"Baseline 60-day cancellation {p0:.1%}, alpha 0.05 two-sided")
    ax.legend(loc="lower right")
    return P.save(fig, "exp_power_curve")


def next_test_design(scored: pd.DataFrame, arms: int = 3) -> dict:
    """Follow-up test: under-12-month members only, control vs guest-pass-only vs current offer."""
    pop = eligible_population(scored)
    sub = pop[pop["tenure_band"] != "12+ mo"]
    p60 = 1 - np.exp(-daily_hazard(sub) * DAYS)
    p0 = float(p60.mean())
    n_arm = len(sub) // arms
    # Bonferroni across the 2 treatment-vs-control comparisons
    mde = mde_given_n(p0, n_arm, alpha=ALPHA / (arms - 1))
    return {"eligible_members": int(len(sub)), "arms": arms, "n_per_arm": int(n_arm),
            "baseline_60d": p0, "alpha_per_comparison": ALPHA / (arms - 1),
            "mde_relative": float(mde)}
