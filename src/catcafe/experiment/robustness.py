"""Is the story an artefact of one lucky random draw?

In real life you get one experiment. Here the data is simulated, so we can
re-draw the experiment 20 times with the same true effects and check how often
each headline conclusion would have come out the same way. If a conclusion
only holds for one seed, it should not be in the recommendation.
"""
from __future__ import annotations

import warnings

import pandas as pd

from . import analysis, policy, traps
from .simulate import simulate_experiment


def seed_sweep(scored: pd.DataFrame, seeds: int = 20) -> dict:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(seeds):
            exp, _ = simulate_experiment(scored, seed=1000 + s)
            r = analysis.two_proportion(exp["cancelled_60d"], exp["treatment"])
            seg = traps.segments(exp, plot=False)
            t = seg["table"].set_index("segment")
            pol = policy.evaluate_policies(exp, boot=1, plot=False)["table"].set_index("policy")
            v = pol["value_per_member_vs_nobody"]
            rows.append({
                "seed": 1000 + s, "relative_effect": r["relative"], "p_value": r["p_value"],
                "diff_12plus_pp": t.loc["12+ mo", "diff"] * 100, "holm_p_12plus": t.loc["12+ mo", "p_holm"],
                "interaction_p": seg["interaction_p"],
                "value_treat_everyone": v["Treat everyone"], "value_tenure_under_12m": v["Tenure under 12 months"],
                "value_top30_risk": v["Top 30% by churn risk"],
            })
    d = pd.DataFrame(rows)
    summary = {
        "seeds": seeds,
        "share_overall_significant": float((d["p_value"] < 0.05).mean()),
        "share_12plus_reversal_significant_holm": float(((d["holm_p_12plus"] < 0.05)
                                                         & (d["diff_12plus_pp"] > 0)).mean()),
        "share_interaction_significant": float((d["interaction_p"] < 0.05).mean()),
        "share_treat_everyone_loses_money": float((d["value_treat_everyone"] < 0).mean()),
        "share_under12_beats_everyone": float((d["value_tenure_under_12m"] > d["value_treat_everyone"]).mean()),
        "share_under12_beats_top30_risk": float((d["value_tenure_under_12m"] > d["value_top30_risk"]).mean()),
        "mean_relative_effect": float(d["relative_effect"].mean()),
        "sd_relative_effect": float(d["relative_effect"].std()),
        "mean_value_under12": float(d["value_tenure_under_12m"].mean()),
        "mean_value_everyone": float(d["value_treat_everyone"].mean()),
        "mean_value_top30": float(d["value_top30_risk"].mean()),
    }
    return {"table": d, "summary": summary}
