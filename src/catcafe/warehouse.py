"""The table a data scientist is usually handed: ``member_snapshot_export``.

It was built by joining monthly snapshots to the *current* member dimension
and the CRM / exit-survey tables, then refreshed on the export date. It looks
reasonable. It contains six leakage traps - see ``LEAKAGE_TRAPS`` - which the
audit in ``leakage.py`` has to find without being told where they are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABEL_WINDOW_DAYS, SIM, WAREHOUSE_DIR, snapshot_dates
from .features import CATEGORICAL, last_event, monthly_price_at, window_counts

NAIVE_LABEL = "churned_next_30d"

# Ground truth, used only by tests and by the report - never by the audit itself.
LEAKAGE_TRAPS = {
    "cancel_request_pending": "L1 label definition: outcome already decided at snapshot",
    "exit_survey_score": "L2 target leakage: only exists after a member cancels",
    "winback_offer_sent": "L3 post-outcome treatment: CRM sends it after the cancel request",
    "lifetime_visits": "L4 time travel: member dimension refreshed on export date",
    "days_since_last_visit": "L4 time travel: measured to the export date, not the snapshot",
    "has_adopted_cat": "L4 time travel: includes adoptions after the snapshot",
    "promo_code_churn_rate": "L5 target encoding fitted on all rows incl. the test set",
    "(random row split)": "L6 split leakage: same member and future months in train and test",
}


def build_export(raw: dict) -> pd.DataFrame:
    members, log, canc, crm = (raw["members"], raw["activity_log"],
                               raw["cancellations"], raw["crm_actions"])
    export_date = pd.Timestamp(SIM.export_date)

    # "Member dimension" - current state as of the export date
    dim = members.set_index("member_id")
    dim_lifetime = window_counts(log, "visit", None, export_date)
    dim_last_visit = last_event(log, "visit", export_date)
    dim_adopted = set(log.loc[log["event_type"] == "cat_adopted", "member_id"])
    survey = canc.set_index("member_id")["exit_survey_score"]
    winback = set(crm.loc[crm["action"] == "winback_offer", "member_id"])
    req = canc.set_index("member_id")["request_ts"]
    eff = canc.set_index("member_id")["effective_date"]

    rows = []
    for s in snapshot_dates():
        d30, d90 = s - pd.Timedelta(days=30), s - pd.Timedelta(days=90)
        m = members[members["signup_date"] < s]
        e = m["member_id"].map(eff)
        m = m[e.isna() | (e > s)]  # "still a paying member on the snapshot date"
        mid = m["member_id"]
        f = pd.DataFrame({"member_id": mid.to_numpy(), "snapshot_date": s})
        for c in CATEGORICAL + ["promo_code", "distance_km"]:
            f[c] = m[c].to_numpy()
        f["tenure_days"] = (s - m["signup_date"]).dt.days.to_numpy()
        f["monthly_price"] = monthly_price_at(m, s).to_numpy()

        def cnt(kind, lo):
            return mid.map(window_counts(log, kind, lo, s)).fillna(0).astype(int).to_numpy()

        # These windowed aggregates were built correctly (point in time).
        f["visits_30d"] = cnt("visit", d30)
        f["visits_90d"] = cnt("visit", d90)
        f["booking_failures_90d"] = cnt("booking_failed", d90)
        f["support_tickets_90d"] = cnt("support_ticket", d90)
        f["events_90d"] = cnt("event_attended", d90)
        # These came from the member dimension, joined as of the export date.
        f["lifetime_visits"] = mid.map(dim_lifetime).fillna(0).astype(int).to_numpy()
        f["days_since_last_visit"] = (export_date - mid.map(dim_last_visit)).dt.days.to_numpy()
        f["has_adopted_cat"] = mid.isin(dim_adopted).astype(int).to_numpy()
        # Account / CRM / survey columns
        r = mid.map(req)
        f["cancel_request_pending"] = (r < s).astype(int).to_numpy()
        f["exit_survey_score"] = mid.map(survey).to_numpy()
        f["winback_offer_sent"] = mid.isin(winback).astype(int).to_numpy()
        ee = mid.map(eff)
        f[NAIVE_LABEL] = ((ee > s) & (ee <= s + pd.Timedelta(days=LABEL_WINDOW_DAYS))).astype(int).to_numpy()
        rows.append(f)
    df = pd.concat(rows, ignore_index=True)

    # Analyst-added feature: churn rate per promo code, computed over the whole table.
    df["promo_code_churn_rate"] = df.groupby("promo_code")[NAIVE_LABEL].transform("mean")
    return df


def save(df: pd.DataFrame) -> None:
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WAREHOUSE_DIR / "member_snapshot_export.csv", index=False)


def load() -> pd.DataFrame:
    return pd.read_csv(WAREHOUSE_DIR / "member_snapshot_export.csv",
                       parse_dates=["snapshot_date"], dtype={"promo_code": str})
