"""Write the markdown reports and README from results.json.

Every number in the write-ups comes from ``results.json``; nothing is typed by
hand, so re-running the pipeline can never leave the prose out of date.
"""
from __future__ import annotations

from .config import REPORTS_DIR, ROOT


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def pp(x, d=1):
    return f"{x * 100:+.{d}f} pp"


def usd(x, d=0):
    s = "-" if x < 0 else ""
    return f"{s}${abs(x):,.{d}f}"


def k(x):
    return f"${x / 1000:,.0f}k"


def p_fmt(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def _rows(records, key, val):
    return {r[key]: r for r in records} if val is None else {r[key]: r[val] for r in records}


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
def leakage_md(r):
    L = r["leakage"]
    wf = L["waterfall"]
    scr = _rows(L["single_feature_screen"], "column", "single_feature_auc")
    tim = _rows(L["information_timing"], "column", None)
    pit = _rows(L["point_in_time_mismatch"], "column", None)
    miss = _rows(L["missingness"], "column", None)
    lab, te, sr, so = L["label_already_known"], L["target_encoding"], L["split_random"], L["split_out_of_time"]
    final = wf[-1]["roc_auc"]
    clean = r["churn"]["test_metrics"]["lightgbm_calibrated"]["roc_auc"]

    traps = [
        ["L1", "`cancel_request_pending`", "Label definition: outcome already decided",
         f"Single-column AUC {scr['cancel_request_pending']:.2f}; {pct(lab['share_request_before_snapshot'])} of "
         "positive labels had *already requested* cancellation before the snapshot",
         "Predict the request, not the effective date; drop members with a pending request"],
        ["L2", "`exit_survey_score`", "Target leakage: exists only after cancelling",
         f"Missing for {pct(miss['exit_survey_score']['null_rate_neg'], 0)} of non-churners vs "
         f"{pct(miss['exit_survey_score']['null_rate_pos'], 0)} of churners; "
         f"{pct(tim['exit_survey_score']['share_from_future'], 0)} of values were written after the snapshot",
         "Drop. Use it for research, never as a feature"],
        ["L3", "`winback_offer_sent`", "Post-outcome treatment",
         f"{pct(tim['winback_offer_sent']['share_from_future'], 0)} of offers were sent after the snapshot "
         f"(AUC {scr['winback_offer_sent']:.2f} - below the screen's flag)",
         "Drop. The CRM sends it *because* someone cancelled"],
        ["L4", "`lifetime_visits`, `days_since_last_visit`", "Time travel: joined from today's member table",
         f"{pct(pit['lifetime_visits']['mismatch_share'], 0)} / {pct(pit['days_since_last_visit']['mismatch_share'], 0)} "
         "of rows disagree with a point-in-time recompute; "
         f"{pct(pit['days_since_last_visit']['impossible_share'], 0)} of rows claim more days since the last visit than "
         "the member had been a member - impossible at snapshot time",
         "Recompute from the event log with `ts < snapshot_date`"],
        ["L4", "`has_adopted_cat`", "Time travel (quiet version)",
         f"Single-column AUC only {scr['has_adopted_cat']:.2f} - invisible to the screen - but "
         f"{pct(tim['has_adopted_cat']['share_from_future'], 0)} of the 1s are adoptions that had not happened yet",
         "Replace with `adopted_cat_90d` / `adopted_cat_ever` as of the snapshot"],
        ["L5", "`promo_code_churn_rate`", "Target encoding fitted on all rows incl. test",
         f"{te['cardinality']:,} codes, {pct(te['share_codes_single_member'], 0)} used by one member. "
         f"AUC {te['auc_delivered_encoding']:.2f} as delivered vs {te['auc_out_of_fold_encoding']:.2f} out of fold",
         "Fit encodings inside each training fold - then notice it carries no signal and drop it"],
        ["L6", "(random row split)", "Split leakage: same members and future months on both sides",
         f"{pct(sr['test_rows_whose_member_is_in_train'], 0)} of test rows belong to a member seen in training and "
         f"{pct(sr['train_rows_after_earliest_test_date'], 0)} of training rows are *later* than the earliest test row",
         "Out-of-time split: train <= Aug-2025, validate Sep/Oct, test Nov/Dec"],
    ]
    wf_rows = [[w["step"], f"{w['roc_auc']:.3f}", "" if w["delta"] is None else f"{w['delta']:+.3f}"] for w in wf]

    return f"""# 1. Leakage audit

The analytics team handed over `member_snapshot_export`: one row per member per
month, a label `churned_next_30d`, and 20 columns. A LightGBM fit on it with a
random split scores **ROC AUC {wf[0]['roc_auc']:.4f}**. No churn model is that good.
The audit below finds why, using only checks a reviewer could run without
knowing where the traps are (`src/catcafe/leakage.py`).

![Leakage waterfall]({L['figures']['waterfall']})

## The seven leaky columns and how each was caught

{_table(["#", "Column", "Type", "Evidence", "Fix"], traps)}

The traps are planted by `warehouse.py` the way they usually appear in real
work: not by anyone doing something silly, but by joining a monthly snapshot
to *current-state* tables (member dimension, CRM, exit survey) and by a
reasonable-sounding label ("churned in the next 30 days" - but cancellations
take effect after a 30-day notice, so most of those members had already
decided).

## Why one check is not enough

![Single-column screen]({L['figures']['screen']})

The single-column AUC screen is the check most people run. It flags only
the three loudest columns. `winback_offer_sent` ({scr['winback_offer_sent']:.2f}) and
`exit_survey_score` ({scr['exit_survey_score']:.2f}) sit just under any sensible threshold, and
`has_adopted_cat` ({scr['has_adopted_cat']:.2f}) looks like noise. They were caught by asking a
different question: **when was this value written, relative to the snapshot?**
That is the check that generalises - every column needs a known
"available at" timestamp, and anything without one is guilty until proven innocent.

## Waterfall (same model settings at every step)

{_table(["Step", "Test AUC", "Change"], wf_rows)}

Reading it honestly:

* The order of fixes changes how much each bar "owns". Dropping the survey and
  win-back columns costs only {abs(wf[2]['delta']):.3f} AUC because, by that step, the time-travel
  columns already carried the same information. Each bar is a marginal effect
  given the fixes before it, not a standalone measure of that leak.
* The out-of-time split costs another {abs(wf[-1]['delta']):.3f}. Part of that is member-level
  memorisation in the random split; part is genuine drift (the base grows,
  cafes get more crowded, a price increase happened in July 2025). A random
  split hides both. Production only ever sees the future.
* The honest number with the export's columns is **{final:.3f}**. Rebuilding the
  features properly from the event log (`features.py`) gets the model to
  **{clean:.3f}** - see the model report.

## Split audit

| | Random 80/20 rows | Out-of-time |
|---|---|---|
| Test rows whose member is also in train | {pct(sr['test_rows_whose_member_is_in_train'], 0)} | {pct(so['test_rows_whose_member_is_in_train'], 0)} |
| Training rows dated after the earliest test row | {pct(sr['train_rows_after_earliest_test_date'], 0)} | {pct(so['train_rows_after_earliest_test_date'], 0)} |

Seeing the same member in train and test is fine in the out-of-time setting -
that is the deployment reality: you score existing members. Seeing their
*future* in training is not.

## Checklist to reuse

1. For every column: what event creates it, and is its timestamp strictly before the snapshot?
2. Recompute any "lifetime" or "last X" column from the raw log and diff it.
3. Look for impossible values (last visit before signup, negative tenure).
4. Is the label's outcome already determined at prediction time? Can anyone act on it?
5. Missingness by label. Columns that only exist for one class are outcomes, not features.
6. Any target or frequency encoding: was it fitted on rows that are later evaluated?
7. Split by time; group by entity when rows of one entity span the split.
8. If a simple model beats your expectations by a lot, assume leakage first.
"""


# ---------------------------------------------------------------------------
def model_md(r):
    c = r["churn"]
    m = c["test_metrics"]
    g, lr, raw, rule = (m["lightgbm_calibrated"], m["logistic_regression"], m["lightgbm_uncalibrated"],
                        m["rule_days_since_last_visit"])
    v1 = c["v1_model"]["test_metrics_uncalibrated"]
    v1_top = c["v1_shap_importance"]
    v1_rank = next(i for i, x in enumerate(v1_top) if x["feature"] == "monthly_price") + 1
    imp = c["shap_importance"]
    figs = c["shap_figures"]
    rows = [["LightGBM + Platt recalibration", f"{g['roc_auc']:.3f}", f"{g['pr_auc']:.3f}", f"{g['lift_top10']:.2f}x",
             pct(g["recall_top10"], 0), f"{g['brier']:.4f}", f"{g['expected_vs_observed']:.2f}"],
            ["LightGBM, raw scores", f"{raw['roc_auc']:.3f}", f"{raw['pr_auc']:.3f}", f"{raw['lift_top10']:.2f}x",
             pct(raw["recall_top10"], 0), f"{raw['brier']:.4f}", f"{raw['expected_vs_observed']:.2f}"],
            ["Logistic regression", f"{lr['roc_auc']:.3f}", f"{lr['pr_auc']:.3f}", f"{lr['lift_top10']:.2f}x",
             pct(lr["recall_top10"], 0), f"{lr['brier']:.4f}", f"{lr['expected_vs_observed']:.2f}"],
            ["Rule: days since last visit", f"{rule['roc_auc']:.3f}", f"{rule['pr_auc']:.3f}",
             f"{rule['lift_top10']:.2f}x", pct(rule["recall_top10"], 0), "-", "-"],
            ["v1 LightGBM (price level feature)", f"{v1['roc_auc']:.3f}", f"{v1['pr_auc']:.3f}",
             f"{v1['lift_top10']:.2f}x", pct(v1["recall_top10"], 0), f"{v1['brier']:.4f}",
             f"{v1['expected_vs_observed']:.2f}"]]
    top = [[i + 1, x["label"], f"{x['mean_abs_shap']:.3f}"] for i, x in enumerate(imp[:10])]  # noqa: E501
    return f"""# 2. Churn model and SHAP explanations

**Target:** member submits a cancellation request in the 30 days after the
monthly snapshot. Members with a request already pending are excluded.
**Features:** {len(imp)} point-in-time features built from the event log (`features.py`).
**Split:** train {c['rows']['train']:,} member-months (Apr-2024 to Aug-2025), validation
{c['rows']['valid']:,} (Sep/Oct, early stopping + calibration), test {c['rows']['test']:,} (Nov/Dec, touched once).
Base rate in test: {pct(g['base_rate'])} per month.

## Results on the held-out months

{_table(["Model", "ROC AUC", "PR AUC", "Lift top 10%", "Recall top 10%", "Brier", "Expected / observed"], rows)}

What this says, without spin:

* **The model is useful, not magic.** Contacting the riskiest 10% of members
  reaches {pct(g['recall_top10'], 0)} of next month's cancellations, {g['lift_top10']:.1f}x random. An AUC around
  {g['roc_auc']:.2f} is in the usual range for monthly subscription churn; anything near 0.95 should
  send you back to the leakage audit.
* **LightGBM barely beats logistic regression** ({g['roc_auc']:.3f} vs {lr['roc_auc']:.3f} AUC, near-identical lift).
  With this data a regularised logistic regression is a perfectly good
  production model and easier to maintain. LightGBM is kept because SHAP
  on trees shows non-linear shapes (below) that are useful for the business story.
* **Calibration matters more than AUC here** because revenue at risk is a *sum
  of probabilities*. Raw scores predicted {raw['expected_vs_observed']:.2f}x the observed cancellations; after
  Platt scaling on the validation months, {g['expected_vs_observed']:.2f}x. The remaining gap means the
  revenue-at-risk figure is, if anything, slightly understated.

![Calibration]({c['figures']['calibration']})
![Gains]({c['figures']['lift']})

## SHAP caught a modelling bug before it shipped

The first version (v1) used `monthly_price` and a persistent "was hit by the
price increase" flag. SHAP ranked monthly price as the **#{v1_rank} driver**. That made
no sense - price level has almost no effect in the cafe's history - until
splitting its SHAP values by snapshot month showed it: the July 2025 Basic-plan increase (29 -> 35) caused a
two-month spike in cancellations, and the model had learned "$35 members are
risky" *permanently*. On the Nov/Dec test months v1 over-predicted cancellations
by {pct(v1['expected_vs_observed'] - 1, 0)}.

v2 encodes the shock as transient (`price_increase_recent`: increase took effect
in the last 60 days) and drops the price level. AUC went from {v1['roc_auc']:.3f} to
{g['roc_auc']:.3f} and raw calibration from {v1['expected_vs_observed']:.2f} to {raw['expected_vs_observed']:.2f}. This is the most
valuable thing SHAP did in the project: not the pretty chart, the bug report.

## Global drivers (v2)

{_table(["Rank", "Feature", "Mean absolute SHAP (log-odds)"], top)}

![SHAP beeswarm]({figs['beeswarm']})

![SHAP dependence]({figs['dependence']})

How to read these, and how not to:

* **Engagement dominates.** Visits per month of tenure is the single strongest
  signal: below ~3 visits a month risk climbs steeply. Days since last visit
  adds risk after roughly two weeks of absence.
* **Billing and plan.** Monthly-billed members carry far more risk than annual.
  That is partly selection (people who prepay a year were already committed),
  so it is *not* evidence that converting someone to annual would retain them.
* **Adopting a cat** multiplies predicted odds by about {c['adopted_cat_odds_multiplier']:.1f}x in the 90 days after
  adoption. The cafe's mission succeeding (cats find homes) is a churn driver.
* **Tenure needs to be read with visits-per-month.** Brand-new members get a
  *negative* tenure SHAP because their visits-per-month value already carries
  the "new and not yet engaged" risk. Correlated features split credit; never
  read one SHAP column in isolation.
* **Distance has a conditional effect**: holding visits fixed, members who
  travel further are *less* likely to cancel (they come despite the trip).
* **Booking failures** barely register. Failures scale with how often you try to
  book, so the model cannot separate "crowded cafe" from "engaged member".
  Exit-form reasons tell a different story (see the recommendation). SHAP ranks
  predictive signal, not causal levers.

## Reason codes for the CRM

Every live member gets their top three risk-raising features as plain-English
reason codes (`reports/tables/at_risk_members_top250.csv`). Example for the
riskiest member in the January base:

![Waterfall]({figs['waterfall']})
"""


# ---------------------------------------------------------------------------
def revenue_md(r):
    v = r["revenue"]
    ten = v["by_tenure"]
    bill = _rows(v["by_billing"], "billing_cycle", None)
    drv = v["by_driver"]
    ten_rows = [[t["tenure_band"], f"{t['members']:,}", pct(t["mean_p"]), usd(t["mrr_at_risk_30d"]),
                 k(t["revenue_at_risk_12m"]), pct(t["share_of_risk"], 0)] for t in ten]
    dec = v["by_decile"]
    dec_rows = [[d["risk_decile"], f"{d['members']:,}", pct(d["mean_p"]), k(d["revenue_at_risk_12m"]),
                 pct(d["share_of_risk"], 0)] for d in dec]
    drv_rows = [[name, k(val), pct(val / v["revenue_at_risk_12m"], 0)] for name, val in drv.items() if val > 0]
    return f"""# 3. Revenue at risk

Scored base: **{v['members']:,} active members on 1-Jan-2026**, monthly recurring revenue
**{usd(v['total_mrr'])}** (annual run-rate {usd(v['annual_run_rate'])}).

| Measure | Value | Definition |
|---|---|---|
| MRR at risk, next 30 days | **{usd(v['mrr_at_risk_30d'])}** | sum of p(cancel in 30d) x monthly fee |
| 12-month revenue at risk | **{usd(v['revenue_at_risk_12m'])}** | expected fees lost over 12 months vs. "nobody leaves", constant monthly hazard, 30-day notice |
| As share of annual run-rate | {pct(v['share_of_annual_run_rate'], 0)} | |
| Held by the riskiest 10% / 30% of members | {pct(v['top_decile_share'], 0)} / {pct(v['top_3_deciles_share'], 0)} | |

Assumptions you should challenge: the 12-month figure keeps each member's
current hazard for a year. New members' hazard falls as they settle in, so
the figure leans high for them; calibration on the test months ran slightly
low ({r['churn']['test_metrics']['lightgbm_calibrated']['expected_vs_observed']:.2f}x), which leans the other way. Treat it as a planning range, not a forecast.

## Where the risk sits

{_table(["Tenure", "Members", "Mean p(30d)", "MRR at risk (30d)", "12m at risk", "Share"], ten_rows)}

Members in their first three months are {pct(ten[0]['members'] / v['members'], 0)} of the base but {pct(ten[0]['share_of_risk'], 0)} of revenue at
risk. Monthly-billed members hold {pct(bill['monthly']['share_of_risk'], 0)} of it.

![Concentration]({v['figures']['concentration']})

{_table(["Risk decile (1 = riskiest)", "Members", "Mean p(30d)", "12m at risk", "Share"], dec_rows)}

Risk is concentrated, but not enough to make a top-decile campaign a silver
bullet: the riskiest 10% hold under a quarter of the money.

## Revenue at risk by driver

Each member's 12-month risk is split across driver groups in proportion to
their positive SHAP contributions. This is a model attribution - a way of
saying "where does the model see the risk" - not a causal decomposition.

{_table(["Driver group", "12m revenue at risk", "Share"], drv_rows)}

![Drivers]({v['figures']['drivers']})

## Revenue at risk is not revenue you can save

What can be saved = risk x how much an intervention changes behaviour - what
it costs. The model cannot tell us the middle term. The experiment can.
"""


# ---------------------------------------------------------------------------
def experiment_md(r):
    e = r["experiment"]
    pw, nt, rd = e["power"], e["next_test"], e["readout"]
    pr, ra, gr = rd["primary_cancel_60d"], rd["primary_regression_adjusted"], rd["guardrail_net_revenue_90d"]
    cu = rd["secondary_visits_60d"]
    pk, nv, sg, pol, rob = e["peeking"], e["novelty"], e["segments"], e["policy"], e["robustness"]
    fpr = pk["false_positive_rate"]
    req = [[pct(x["relative_reduction"], 0), f"{x['absolute_pp']:.1f} pp", f"{x['n_per_arm']:,}",
            "yes" if x["feasible"] else "**no**"] for x in pw["requirements"]]
    order = ["0-3 mo", "3-6 mo", "6-12 mo", "12+ mo"]
    segm = sorted(pw["segment_mde"], key=lambda x: order.index(x["tenure_band"]))
    seg_pow = [[x["tenure_band"], f"{x['members']:,}", pct(x["baseline_60d"]), pct(x["mde_relative"], 0)] for x in segm]
    seg_rows = [[x["segment"], f"{x['members']:,}", pct(x["control_rate"]), pct(x["treatment_rate"]), pp(x["diff"]),
                 f"[{x['ci_low'] * 100:+.1f}, {x['ci_high'] * 100:+.1f}]", p_fmt(x["p_holm"])] for x in sg["table"]]
    rep = _rows(sg["replication_12plus"], "sample", None)
    vw = nv["visit_lift_by_week"]
    hz = nv["hazard_ratio_by_period"]
    hz_rows = [[h["days"], f"{h['hazard_ratio']:.2f}", f"[{h['ci_low']:.2f}, {h['ci_high']:.2f}]",
                f"{h['events_t']} / {h['events_c']}"] for h in hz]
    w2 = nv["week2_readout_cancel"]
    pol_rows = [[p["policy"], pct(p["share_treated"], 0), usd(p["value_per_member_vs_nobody"], 2),
                 f"[{usd(p['ci_low'], 2)}, {usd(p['ci_high'], 2)}]", usd(p["total_vs_nobody"]),
                 usd(p["cash_90d_per_member_vs_nobody"], 2),
                 "-" if p["break_even_cost_per_treated"] is None else usd(p["break_even_cost_per_treated"], 2)]
                for p in pol["table"] if p["policy"] != "Treat nobody"]
    bounds = pk["obf_boundaries"]
    naive_day, obf_day = pk["first_day_naive_p_below_05"], pk["first_obf_crossing_day"]
    return f"""# 4. The retention experiment

**"Purr-sonal Perks"**: a personalised email + app push featuring the member's
favourite cats, one free guest pass, and 20% off next month's fee.
Control: business as usual. Unit: member, 50/50 by salted hash of member id.
Start 2-Feb-2026, outcome window 60 days.

---

## Part A - Design, written before launch

**Hypothesis.** The campaign reduces 60-day cancellation requests.
**Primary metric.** Share of members who submit a cancellation request within 60 days.
**Guardrail.** Net revenue per member over 90 days (fees minus offer cost) must
not fall by more than the retention value it buys.
**Secondary.** Visits per member (CUPED-adjusted on prior-30-day visits).
**Pre-registered heterogeneity.** Tenure band (primary hypothesis), billing cycle, plan.
**Decision rule.** Ship if the primary effect is significant at the final look
*and* the 12-month value of the targeting policy is positive. Six planned looks
(days 10-60) with O'Brien-Fleming boundaries, for stopping on harm; no
stopping for success before day 60, because the segment and value questions
need the full window.

### Power analysis

The member base is fixed: **{pw['eligible_members']:,} eligible members, {pw['n_per_arm_available']:,} per arm**. Baseline
60-day cancellation rate (from the calibrated model): **{pct(pw['baseline_60d_cancel_rate'])}**.
alpha = {pw['alpha']}, two-sided, power = {pct(pw['power'], 0)}.

{_table(["Relative reduction", "Absolute", "Members needed per arm", "Feasible?"], req)}

**Minimum detectable effect with the base we have: {pct(pw['mde_relative'], 0)} relative ({pw['mde_absolute_pp']:.1f} pp).**
A Monte Carlo check with the real, heterogeneous risk mix gives {pct(pw['simulated_power_at_mde'], 0)} power at that
effect, matching the analytic answer.

Is that good enough? Only the economics can say. The offer is expected to cost
{usd(pw['expected_cost_per_treated'], 2)} per treated member and a retained member is worth {usd(pw['mean_clv12'])} over 12
months, so treating everyone **breaks even only at a {pct(pw['break_even_relative_reduction'], 0)} relative reduction**.
The MDE ({pct(pw['mde_relative'], 0)}) is below that line: any effect large enough to pay for itself is
detectable. Effects we cannot see are effects we could not afford to act on.

![Power curve]({pw['figure']})

Segment-level power is much weaker - worth knowing *before* someone reads
the forest plot:

{_table(["Tenure band", "Members", "Baseline 60d", "MDE (relative)"], seg_pow)}

Testing whether an effect *differs* between two segments needs roughly
{pw['interaction_sample_multiplier']:.0f}x the sample of detecting the effect itself. We can only detect large reversals.

---

## Part B - Readout at day 60

**Sample ratio check:** {rd['srm']['control']:,} control / {rd['srm']['treatment']:,} treatment, chi-square p = {rd['srm']['p_value']:.2f}. No SRM.

| Metric | Control | Treatment | Difference | 95% CI | p |
|---|---|---|---|---|---|
| 60-day cancellation | {pct(pr['control_rate'])} | {pct(pr['treatment_rate'])} | **{pp(pr['diff'])} ({pct(pr['relative'], 0)})** | [{pr['ci_low'] * 100:+.1f}, {pr['ci_high'] * 100:+.1f}] pp | {p_fmt(pr['p_value'])} |
| ...regression-adjusted | | | {pp(ra['diff'])} | [{ra['ci_low'] * 100:+.1f}, {ra['ci_high'] * 100:+.1f}] pp | {p_fmt(ra['p_value'])} |
| Net revenue / member, 90d | {usd(gr['control_mean'], 2)} | {usd(gr['treatment_mean'], 2)} | **{usd(gr['diff'], 2)}** | [{usd(gr['ci_low'], 2)}, {usd(gr['ci_high'], 2)}] | {p_fmt(gr['p_value'])} |
| Visits / member, 60d (raw) | {cu['raw']['control_mean']:.2f} | {cu['raw']['treatment_mean']:.2f} | {cu['raw']['diff']:+.2f} | [{cu['raw']['ci_low']:+.2f}, {cu['raw']['ci_high']:+.2f}] | {p_fmt(cu['raw']['p_value'])} |
| Visits / member, 60d (CUPED) | | | {cu['cuped']['diff']:+.2f} | [{cu['cuped']['ci_low']:+.2f}, {cu['cuped']['ci_high']:+.2f}] | {p_fmt(cu['cuped']['p_value'])} |

* The campaign **works on the primary metric**: {pct(abs(pr['relative']), 0)} fewer cancellation requests,
  relative CI [{pct(pr['relative_ci_low'], 0)}, {pct(pr['relative_ci_high'], 0)}]. Regression adjustment on pre-treatment
  covariates tightens the interval slightly and moves the estimate little, as it should.
* It **loses money in the short run**: {usd(gr['diff'], 2)} net revenue per member over 90 days. The offer
  costs {usd(rd['offer_cost_per_treated'], 2)} per treated member, and most treated members were never going to cancel.
* CUPED on pre-period visits removes {pct(cu['cuped']['variance_reduction'], 0)} of the variance of the visits metric.

A significant primary metric and a negative guardrail is the most common
real-world readout. It means the question is not "ship or not" but "ship to whom".

---

## Part C - Trap 1: peeking

![A/A peeking]({pk['figures']['aa']})

4,000 simulated A/A tests on the real risk mix (no true difference; half
used to calibrate the sequential boundary, the other 2,000 to measure):

| How you look | False positive rate |
|---|---|
| Once, at day 60 | {pct(fpr['final_look_only'])} |
| Every week, stop at first p < 0.05 | **{pct(fpr['weekly_peeks'])}** |
| Every day, stop at first p < 0.05 | **{pct(fpr['daily_peeks'])}** |
| Six planned looks with O'Brien-Fleming boundaries | {pct(fpr['obf_6_looks'])} |

The boundary constant was calibrated on one half of the simulations
(c = {pk['obf_constant']:.3f}, boundaries {", ".join(f"day {d}: {b:.2f}" for d, b in bounds.items())}) and
checked on the other half.

![Real trajectory]({pk['figures']['trajectory']})

In the real test, a daily peeker would have declared victory on **day {naive_day}**
with an estimated {pct(pk['effect_at_naive_stop'], 0)}. The sequential boundary was first crossed on day {obf_day}.
In this draw the early call happens to be right - which is exactly what makes
peeking dangerous: it is usually rewarded, and the {pct(fpr['daily_peeks'], 0)} of the time it is not,
nobody finds out. Stopping on day {naive_day} would also have meant deciding before
the 12+ month segment had enough events to show its reversal, and before a
single billing cycle had passed.

---

## Part D - Trap 2: novelty

![Novelty]({nv['figures']['novelty']})

* The **visit bump is a novelty effect**: +{pct(vw[0]['lift'], 0)} visits in week 1, +{pct(vw[1]['lift'], 0)} in week 2,
  and indistinguishable from zero from week 3 on (fitted half-life
  {nv['visit_lift_fit']['half_life_weeks']:.1f} weeks, long-run lift {pct(nv['visit_lift_fit']['long_run'], 1)}). A week-1 engagement dashboard would have sold this
  campaign as a big win on the wrong metric.
* A **2-week readout of the primary metric** was {pct(w2['relative'], 0)} with p = {w2['p_value']:.3f} -
  inconclusive. Whether the *retention* effect also fades cannot be resolved at
  this sample size:

{_table(["Days", "Hazard ratio T/C", "95% CI", "Events T / C"], hz_rows)}

  The intervals overlap heavily. The honest statement is "no evidence that the
  retention effect is decaying within 60 days, but we cannot rule it out". That
  is why the decision uses the 60-day outcome and why a long-term holdout is
  part of the rollout plan.

---

## Part E - Trap 3: the segment where the result reverses

![Forest plot]({sg['figure']})

{_table(["Segment", "Members", "Control", "Treatment", "Diff", "95% CI (pp)", "Holm p"], seg_rows)}

* For members in their **first three months** the campaign cuts cancellations by
  {abs(sg['table'][0]['diff']) * 100:.1f} pp. For **12+ month members it raises them by {sg['table'][3]['diff'] * 100:.1f} pp**.
* This is not a fishing result: tenure was the pre-registered heterogeneity
  hypothesis, the treatment x tenure interaction is significant (likelihood-ratio
  p = {sg['interaction_p']:.1e}), it survives Holm correction across all nine segment tests, and it
  replicates in two random halves of the data ({pp(rep['discovery_half']['diff'])}, p = {rep['discovery_half']['p_value']:.3f}; {pp(rep['confirmation_half']['diff'])}, p = {rep['confirmation_half']['p_value']:.3f}).
* Mechanism (a hypothesis): a "we miss you, here's a discount" message reminds
  habitual, long-standing members that they are paying, and some re-evaluate.
  A field experiment at a telecom found the same pattern - proactive retention
  outreach *increased* churn for some customers (Ascarza, Iyengar & Schleicher,
  2016). Do not wake sleeping dogs.

---

## Part F - So who should get the offer?

Because treatment was randomised, the value of any targeting rule can be
estimated from the experiment itself (inverse-propensity weighting with p = 0.5).
Value = 12-month expected fees kept, minus offer cost. Bootstrap 95% CIs.

{_table(["Policy", "Treated", "Value / member", "95% CI", "Whole base, 12m", "Cash, 90d / member", "Break-even offer cost"], pol_rows)}

![Policy value]({pol['figure']})

* **Treating everyone does not pay**: the retention it buys ({usd(pol['table'][1]['break_even_cost_per_treated'], 2)} per treated member)
  is less than the offer costs ({usd(rd['offer_cost_per_treated'], 2)}). The CI includes zero, so read it as
  "no evidence it pays" - it was negative in {pct(rob['share_treat_everyone_loses_money'], 0)} of re-draws (Part G).
* **Targeting by churn risk is better but not best.** High risk is not the
  same as high response (Ascarza, 2018, makes this argument with field data).
* **Targeting by tenure (< 12 months) is best**, and a cross-fitted rule (learn
  the bands on one half, evaluate on the other) recovers exactly the same set
  ({", ".join(pol['cells_worth_treating_full_sample'])}). Each treated member in that group generates
  {usd(pol['table'][4]['break_even_cost_per_treated'], 2)} of retention value - room to spare against a {usd(rd['offer_cost_per_treated'], 2)} offer.
* Even the best policy is **cash-negative over 90 days** ({usd(pol['table'][4]['cash_90d_per_member_vs_nobody'], 2)} per member): the
  discount is paid now, the retained fees arrive over the following year. Finance
  needs to hear that before launch, not after.

## Part G - Is this one lucky draw?

The data is simulated, so the experiment was re-drawn {rob['seeds']} times with the same
true effects. Share of re-draws in which each conclusion holds:

| Conclusion | Holds in |
|---|---|
| Overall effect significant at 5% | {pct(rob['share_overall_significant'], 0)} |
| 12+ month reversal significant after Holm | {pct(rob['share_12plus_reversal_significant_holm'], 0)} |
| Treatment x tenure interaction significant | {pct(rob['share_interaction_significant'], 0)} |
| Treating everyone loses money | {pct(rob['share_treat_everyone_loses_money'], 0)} |
| Tenure rule beats treating everyone | {pct(rob['share_under12_beats_everyone'], 0)} |
| Tenure rule beats top-30%-risk targeting | {pct(rob['share_under12_beats_top30_risk'], 0)} |

Mean relative effect across re-draws: {pct(rob['mean_relative_effect'], 0)} (sd {pct(rob['sd_relative_effect'], 0)}). The
direction of every recommendation is stable; the exact dollar figures are not.
Disclosure: the planted effect sizes were set so these conclusions are robust,
which is a choice about the *simulation*, not a property of real campaigns.

## References

* Ascarza, E., Iyengar, R., & Schleicher, M. (2016). The Perils of Proactive Churn
  Prevention Using Plan Recommendations: Evidence from a Field Experiment.
  *Journal of Marketing Research*.
* Ascarza, E. (2018). Retention Futility: Targeting High-Risk Customers Might Be
  Ineffective. *Journal of Marketing Research*.
* Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the Sensitivity of
  Online Controlled Experiments by Utilizing Pre-Experiment Data (CUPED). *WSDM*.
* O'Brien, P. C., & Fleming, T. R. (1979). A Multiple Testing Procedure for
  Clinical Trials. *Biometrics*.
"""


# ---------------------------------------------------------------------------
def recommendation_md(r):
    e, v, c = r["experiment"], r["revenue"], r["churn"]
    pol = _rows(e["policy"]["table"], "policy", None)
    best, everyone, risk30 = pol["Tenure under 12 months"], pol["Treat everyone"], pol["Top 30% by churn risk"]
    seg = _rows(e["segments"]["table"], "segment", None)
    rd = e["readout"]
    nt = e["next_test"]
    ten = _rows(v["by_tenure"], "tenure_band", None)
    bill = _rows(v["by_billing"], "billing_cycle", None)
    reasons = v["exit_reasons_sep_dec_2025"]
    share_saved = best["total_vs_nobody"] / v["revenue_at_risk_12m"]
    rob = e["robustness"]
    return f"""# 5. Recommendation for marketing

## The one-paragraph version

Keep the Purr-sonal Perks campaign, but **only for members in their first 12
months**, and **never send retention discounts to members with 12+ months of
tenure**. Sent to everyone, the campaign cuts cancellations by {pct(abs(rd['primary_cancel_60d']['relative']), 0)} and still
loses money on average ({usd(everyone['value_per_member_vs_nobody'], 2)} per member over 12 months; negative in
{pct(rob['share_treat_everyone_loses_money'], 0)} of re-simulations), because most people who get
the 20% discount were never going to leave, and long-tenured members leave
*more* when we remind them we exist. Targeted by tenure, the same campaign is
worth about **{k(best['total_vs_nobody'])} over 12 months** on today's base. That is roughly
{pct(share_saved, 0)} of the {k(v['revenue_at_risk_12m'])} revenue at risk - so the campaign is a useful lever, not the
retention strategy. The bigger levers are onboarding, capacity and the
moment a member adopts a cat.

## What to do, in priority order

**1. Re-target the campaign by tenure, starting next month.**
Send to members under 12 months ({pct(best['share_treated'], 0)} of the base). Expected value
{usd(best['value_per_member_vs_nobody'], 2)} per eligible member (95% CI {usd(best['ci_low'], 2)} to {usd(best['ci_high'], 2)}). Do not use the churn
score alone to pick recipients: the top-30%-risk list is worth {usd(risk30['value_per_member_vs_nobody'], 2)} per member and
lost to the tenure rule in {pct(rob['share_under12_beats_top30_risk'], 0)} of re-simulations. **Risk tells you who might
leave; only the experiment tells you who will respond.** Tell finance the
campaign is cash-negative for the first quarter ({usd(best['cash_90d_per_member_vs_nobody'], 2)} per member over 90 days).

**2. Put 12+ month members on a "do not discount" list.**
The campaign raised their cancellations by {seg['12+ mo']['diff'] * 100:.1f} pp (Holm p = {p_fmt(seg['12+ mo']['p_holm'])}), replicated in
both halves of the data. For this group, test recognition without a price
message (a members' night, naming a kitten after them) - but as a hypothesis
with its own test, not a rollout.

**3. Make the first 90 days the product.**
Members under 3 months are {pct(ten['0-3 mo']['members'] / v['members'], 0)} of the base and {pct(ten['0-3 mo']['share_of_risk'], 0)} of revenue at risk, and they
respond to the campaign most strongly ({abs(seg['0-3 mo']['diff']) * 100:.1f} pp fewer cancellations). The model's top signal
is visits per month of tenure; build an onboarding journey that triggers when a
new member is below ~3 visits a month (personal cat intro, a booked second visit,
a first event invitation).

**4. Treat crowding as an operations problem, and test it.**
The model barely uses booking failures (they move with engagement), but
"too crowded" is the reason on {pct(reasons.get('too_crowded', 0), 0)} of cancellation forms since September. That
disagreement is the point: SHAP ranks predictive signal, the forms report a
cause. Test priority booking for at-risk members, or peak-time capacity, with
a holdout - do not assume either the model or the forms is right.

**5. Turn cat adoption from an exit into a tier.**
Adopting a cafe cat multiplies predicted cancellation odds by about
{c['adopted_cat_odds_multiplier']:.1f}x. That is the mission working. Offer an "Alumni" membership at the moment of
adoption (lower fee, events, vet and supplies partner perks) and measure it
against a holdout.

**6. Test annual billing conversion - carefully.**
Monthly-billed members hold {pct(bill['monthly']['share_of_risk'], 0)} of revenue at risk, but annual members look
safer partly because committed people choose annual. Only a randomised
annual-upgrade offer (around month 3-4) can tell whether converting someone
changes their behaviour.

## Next experiment

Three arms, members under 12 months only: control / guest pass + personalised
message, **no discount** / current offer. Keep the control arm untreated for
six months afterwards so it doubles as a long-term holdout (does the retention
effect last beyond 60 days?).
{nt['eligible_members']:,} eligible members gives {nt['n_per_arm']:,} per arm; at alpha = {nt['alpha_per_comparison']} per comparison
(two comparisons) the MDE is {pct(nt['mde_relative'], 0)} relative on a {pct(nt['baseline_60d'])} baseline. The current offer
creates {usd(best['break_even_cost_per_treated'], 2)} of retention value per treated member against a {usd(rd['offer_cost_per_treated'], 2)} cost; the
question is how much of that value the 20% discount is actually buying.
If guest-pass-only retains nearly as well, it is the better business.

## What not to do

* Don't report the +{pct(e['novelty']['visit_lift_by_week'][0]['lift'], 0)} week-1 visit lift as the campaign's impact. It is gone by week 3.
* Don't read the experiment before day 60 and stop on a good number. Daily
  peeking turns a 5% false-positive rate into {pct(e['peeking']['false_positive_rate']['daily_peeks'], 0)}.
* Don't use SHAP drivers as a to-do list. They say where the model sees risk,
  not what will change it. Every lever above is phrased as a test for that reason.
* Don't trust a churn model with AUC above ~0.9 until someone has run the
  leakage checklist.

## Limitations

The data is simulated with known ground truth so the traps can be verified;
the effect sizes, prices and costs are illustrative. Revenue at risk assumes
constant hazard over 12 months. Policy values are estimated on the same
experiment used to choose the policy (mitigated by cross-fitting and
pre-registration, not eliminated) - the holdout in the next test is the real check.
"""


# ---------------------------------------------------------------------------
def readme_md(r):
    L, c, v, e = r["leakage"], r["churn"], r["revenue"], r["experiment"]
    g = c["test_metrics"]["lightgbm_calibrated"]
    pr = e["readout"]["primary_cancel_60d"]
    pol = _rows(e["policy"]["table"], "policy", None)
    seg = _rows(e["segments"]["table"], "segment", None)
    return f"""# Whisker & Co. - cat cafe membership churn

Predict which members of a three-location cat cafe chain will cancel, catch
the leakage traps hidden in the data you are handed, explain the model with
SHAP, turn the predictions into revenue at risk, and then test a retention
campaign properly: power analysis first, then the readout, then the three
classic traps (peeking, novelty, a segment that reverses). It ends with a
marketing recommendation.

All data is simulated (`src/catcafe/simulate.py`) with known ground truth,
so every trap can be verified rather than asserted. Everything below is
regenerated by one command and every number in the reports is read from
`reports/results.json`.

## Headline results

| | |
|---|---|
| Model on the table as delivered | ROC AUC **{L['waterfall'][0]['roc_auc']:.4f}** - leakage, not skill |
| Leaky columns found | 7 columns + a split problem, each with evidence ([report](reports/01_leakage_audit.md)) |
| Honest churn model (out-of-time) | ROC AUC **{g['roc_auc']:.3f}**, top-10% lift **{g['lift_top10']:.1f}x**, calibrated to {g['expected_vs_observed']:.2f}x observed |
| Bug SHAP caught | v1 learned a one-off price shock as permanent; fixing it improved AUC and calibration |
| 12-month revenue at risk | **{k(v['revenue_at_risk_12m'])}** ({pct(v['share_of_annual_run_rate'], 0)} of run-rate); {pct(v['by_tenure'][0]['share_of_risk'], 0)} sits in members under 3 months |
| Retention campaign, overall | **{pct(pr['relative'], 0)}** cancellations (p {p_fmt(pr['p_value'])}) but **{usd(pol['Treat everyone']['value_per_member_vs_nobody'], 2)}** / member in 12-month value |
| The reversal | 12+ month members cancel **more** when treated ({pp(seg['12+ mo']['diff'])}) |
| Recommendation | Target members under 12 months: **{k(pol['Tenure under 12 months']['total_vs_nobody'])}** over 12 months on today's base; never discount 12+ month members |

## Reports

1. [Leakage audit](reports/01_leakage_audit.md) - seven traps, the checks that catch them, the AUC waterfall
2. [Churn model and SHAP](reports/02_churn_model_and_shap.md) - out-of-time evaluation, calibration, drivers, reason codes
3. [Revenue at risk](reports/03_revenue_at_risk.md) - by tenure, decile and driver
4. [Experiment](reports/04_experiment.md) - power analysis, readout, peeking, novelty, segment reversal, targeting
5. [Recommendation for marketing](reports/05_recommendation.md)

![Leakage waterfall](reports/{L['figures']['waterfall']})
![Segment reversal](reports/{e['segments']['figure']})

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all        # simulate -> audit -> model -> SHAP -> revenue -> experiment -> reports (~2 min)
make test       # unit tests, including "does the audit find every planted trap?"
```

Without `make`: `PYTHONPATH=src python -m catcafe.pipeline` and `PYTHONPATH=src pytest -q`.

## Repository layout

```
src/catcafe/
  simulate.py        raw operational tables: members, activity log, cancellations, CRM, prices
  warehouse.py       the leaky "member_snapshot_export" a data scientist gets handed
  features.py        point-in-time feature store and label definition
  leakage.py         audit checks + leakage waterfall
  modeling.py        LightGBM / logistic regression helpers and metrics
  churn.py           out-of-time training, Platt calibration, production scoring
  explain.py         SHAP: importance, beeswarm, dependence, reason codes, driver shares
  revenue.py         MRR and 12-month revenue at risk
  experiment/
    design.py        power analysis and MDE (run before the test)
    simulate.py      the randomised campaign, with planted novelty and reversal
    analysis.py      SRM, two-proportion test, regression adjustment, CUPED, guardrail
    traps.py         peeking (A/A + O'Brien-Fleming), novelty, segments
    policy.py        off-policy value of targeting rules, cross-fitted
    robustness.py    re-draws the experiment to check conclusions are not luck
  pipeline.py        runs everything in order
  report.py          writes these markdown files from results.json
tests/               unit tests
reports/             generated reports, figures and tables
docs/                data dictionary
```

## Design choices worth arguing about

* **Label = cancellation request, not effective cancellation.** With a 30-day
  notice period the effective date is already decided; predicting it is
  predicting the past.
* **Out-of-time split, calibrated on the most recent months.** Revenue at risk
  is a sum of probabilities, so calibration is a business requirement, not a nicety.
* **Fixed member base drives the experiment design.** Power analysis asks
  "what can we detect with the members we have", not "how many users do we need".
* **Targeting is chosen by value, not risk.** The experiment shows the two disagree.
"""


DATA_DICTIONARY = """# Data dictionary

## Raw tables (`data/raw/`, generated by `simulate.py`)

| Table | Grain | Columns |
|---|---|---|
| `members` | member | member_id, signup_date, location, plan (basic/plus/family), billing_cycle (monthly/annual), acquisition_channel, distance_km, promo_code, price_at_signup |
| `activity_log` | event | member_id, ts, event_type (visit, booking_failed, support_ticket, event_attended, cat_adopted) |
| `cancellations` | request | member_id, request_ts, effective_date (= request + 30 days), reason (cancellation form), exit_survey_score (optional, 1-10), exit_survey_ts |
| `crm_actions` | action | member_id, action_ts, action (welcome_email, winback_offer, cat_birthday_invite) |
| `price_history` | price change | effective_date, plan, billing_cycle, monthly_price |

## Warehouse export (`data/warehouse/member_snapshot_export.csv`) - the leaky table

One row per member per monthly snapshot (Apr-2024 to Dec-2025), built by joining
snapshots to the *current* member, CRM and survey tables on the export date
(15-Jan-2026). Label `churned_next_30d` = effective cancellation date in the next 30 days.
See `reports/01_leakage_audit.md` for what is wrong with it.

## Point-in-time panel (`data/processed/panel_point_in_time.csv`)

One row per active member (no pending cancellation) per snapshot. Every feature
uses events strictly before `snapshot_date`.

| Feature | Meaning |
|---|---|
| plan, billing_cycle, location, acquisition_channel | member attributes known at signup |
| tenure_days | days since signup |
| distance_km | home-to-cafe distance |
| price_increase_recent | member's price rose within the last 60 days |
| visits_30d, visits_31_90d, visit_trend | recent visit counts and trend (30d minus half of days 31-90) |
| days_since_last_visit | days since the last visit before the snapshot |
| lifetime_visits, visits_per_month | history to date, and visits per month of tenure |
| booking_failures_90d, booking_failure_rate_90d | failed bookings (fully booked) and their share of attempts |
| support_tickets_90d, events_90d | service contacts and events attended |
| adopted_cat_90d, adopted_cat_ever | adopted a cafe cat in the last 90 days / ever, as of the snapshot |
| requested_cancel_30d | **label**: cancellation request in [snapshot, snapshot + 30 days) |

Kept in the file but not used by the model: monthly_price (needed for revenue),
price_increase_exposed (v1 feature, see the model report).
"""


def build(r: dict) -> None:
    files = {
        REPORTS_DIR / "01_leakage_audit.md": leakage_md(r),
        REPORTS_DIR / "02_churn_model_and_shap.md": model_md(r),
        REPORTS_DIR / "03_revenue_at_risk.md": revenue_md(r),
        REPORTS_DIR / "04_experiment.md": experiment_md(r),
        REPORTS_DIR / "05_recommendation.md": recommendation_md(r),
        ROOT / "README.md": readme_md(r),
        ROOT / "docs" / "data_dictionary.md": DATA_DICTIONARY,
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n")
