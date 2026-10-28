"""Run the whole project end to end:  python -m catcafe.pipeline

Order matters and mirrors how the work should be done in real life:
data -> leakage audit -> clean model -> SHAP -> revenue at risk ->
power analysis (before the test) -> experiment readout -> traps -> policy.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import churn, explain, features, leakage, revenue, simulate, warehouse
from .config import REPORTS_DIR, ensure_dirs
from .experiment import analysis, design, policy, robustness, traps
from .experiment import simulate as exp_sim
from .features import FEATURES_V1
from .warehouse import LEAKAGE_TRAPS

TABLES = REPORTS_DIR / "tables"


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return o.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(o, pd.Series):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def _log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def run() -> dict:
    t0 = time.time()
    ensure_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)

    simulate.save(simulate.simulate())
    raw = simulate.load_raw()
    _log("raw tables simulated", t0)

    panel = features.build_panel(raw)
    scoring = features.build_scoring_base(raw)
    features.save(panel, scoring)
    scoring = features.load_scoring_base()  # same CSV round trip as anyone re-running from disk
    panel = features.load_panel()
    export = warehouse.build_export(raw)
    warehouse.save(export)
    export = warehouse.load()
    _log("point-in-time features and warehouse export built", t0)

    # ---------------------------------------------------------------- leakage
    audit = leakage.run_audit(export, panel, raw)
    screen = audit["single_feature_screen"]
    flagged = int(screen["flag"].sum())
    audit["figures"] = {
        "waterfall": leakage.plot_waterfall(audit["waterfall"]),
        "screen": leakage.plot_screen(
            screen, f"Only {flagged} columns cross the line. The other leaks need timing and lineage checks."),
    }
    audit["traps_ground_truth"] = LEAKAGE_TRAPS
    _log("leakage audit done", t0)

    # ------------------------------------------------------------------ model
    out = churn.train_and_evaluate(panel, scoring)
    res_churn = out["results"]
    shap_res = explain.explain_test(out["eval_model"], out["test_frame"])
    sample = out["test_frame"].sample(4000, random_state=7)
    sv1, _ = explain.shap_values(out["v1_model"], sample, FEATURES_V1)
    v1_imp = (pd.DataFrame({"feature": FEATURES_V1, "mean_abs_shap": np.abs(sv1).mean(0)})
              .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    scored = out["scored_base"]
    rc, sv_live = explain.reason_codes(out["prod_model"], scored)
    top_pos = int(np.argmax(scored["p_cancel_30d"].to_numpy()))
    shap_res["figures"]["waterfall"] = explain.waterfall_member(out["prod_model"], scored, top_pos)
    _log("model trained, SHAP computed", t0)

    # ---------------------------------------------------------------- revenue
    rev = revenue.revenue_at_risk(scored, explain.driver_share(sv_live))
    canc = raw["cancellations"]
    recent = canc[canc["request_ts"] >= pd.Timestamp("2025-09-01")]
    rev["exit_reasons_sep_dec_2025"] = recent["reason"].value_counts(normalize=True)
    rev["exit_reasons_n"] = int(len(recent))
    frame = rev.pop("frame")
    rc = rc.merge(frame[["member_id", "revenue_at_risk_12m", "risk_decile", "tenure_band"]], on="member_id")
    rc.sort_values("revenue_at_risk_12m", ascending=False).head(250).round(4).to_csv(
        TABLES / "at_risk_members_top250.csv", index=False)
    _log("revenue at risk computed", t0)

    # ------------------------------------------------------------- experiment
    power = design.power_analysis(scored)          # before any outcome exists
    nxt = design.next_test_design(scored)
    exp, daily = exp_sim.simulate_experiment(scored)
    exp.to_csv(TABLES / "experiment_members.csv", index=False)
    read = analysis.readout(exp)
    pk = traps.peeking(exp_sim.eligible_population(scored), exp)
    nov = traps.novelty(exp, daily)
    seg = traps.segments(exp)
    pol = policy.evaluate_policies(exp)
    _log("experiment analysed", t0)
    rob = robustness.seed_sweep(scored)
    rob["table"].round(5).to_csv(TABLES / "experiment_seed_robustness.csv", index=False)
    _log("seed robustness sweep done", t0)

    results = {
        "leakage": audit,
        "churn": {**res_churn, "shap_importance": shap_res["importance"], "shap_figures": shap_res["figures"],
                  "adopted_cat_odds_multiplier": shap_res["adopted_cat_odds_multiplier"],
                  "v1_shap_importance": v1_imp.head(8)},
        "revenue": rev,
        "experiment": {"power": power, "next_test": nxt, "readout": read,
                       "peeking": {k: v for k, v in pk.items() if k != "trajectory"},
                       "novelty": nov, "segments": seg, "policy": pol,
                       "robustness": rob["summary"]},
    }
    pk["trajectory"].round(5).to_csv(TABLES / "peeking_trajectory.csv", index=False)
    for name, df in (("leakage_single_feature_screen", screen), ("leakage_waterfall", audit["waterfall"]),
                     ("shap_importance", shap_res["importance"]), ("revenue_by_decile", rev["by_decile"]),
                     ("experiment_segments", seg["table"]), ("experiment_policies", pol["table"])):
        df.round(5).to_csv(TABLES / f"{name}.csv", index=False)

    js = _jsonable(results)
    (REPORTS_DIR / "results.json").write_text(json.dumps(js, indent=2))
    _log("results.json written", t0)

    from . import report
    report.build(js)
    _log("reports written", t0)
    return js


if __name__ == "__main__":
    run()
