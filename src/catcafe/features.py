"""Point-in-time feature store.

Rule: a feature for snapshot ``s`` may only use events with timestamp ``< s``.
The label is "member submits a cancellation request in [s, s + 30 days)".
Members who already have a pending request at ``s`` are excluded - their
outcome is already decided, so there is nothing to predict or prevent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABEL_WINDOW_DAYS, PROCESSED_DIR, SIM, SCORING_SNAPSHOT, snapshot_dates

CATEGORICAL = ["plan", "billing_cycle", "location", "acquisition_channel"]
NUMERIC = [
    "tenure_days", "distance_km", "price_increase_recent",
    "visits_30d", "visits_31_90d", "visit_trend", "days_since_last_visit",
    "lifetime_visits", "visits_per_month", "booking_failures_90d", "booking_failure_rate_90d",
    "support_tickets_90d", "events_90d", "adopted_cat_90d", "adopted_cat_ever",
]
FEATURES = CATEGORICAL + NUMERIC
# First model version, kept for the write-up: SHAP showed it had learned the
# one-off July 2025 price shock as a permanent property of "$35 members".
FEATURES_V1 = CATEGORICAL + [
    "tenure_days", "distance_km", "monthly_price", "price_increase_exposed",
    "visits_30d", "visits_31_90d", "visit_trend", "days_since_last_visit",
    "lifetime_visits", "visits_per_month", "booking_failures_90d",
    "support_tickets_90d", "events_90d", "adopted_cat_90d", "adopted_cat_ever",
]
PRICE_SHOCK_DAYS = 60
LABEL = "requested_cancel_30d"
CATEGORY_LEVELS = {
    "plan": ["basic", "plus", "family"], "billing_cycle": ["monthly", "annual"],
    "location": list(SIM.locations),
    "acquisition_channel": ["instagram", "walk_in", "google", "referral", "event"],
}


def monthly_price_at(members: pd.DataFrame, s: pd.Timestamp) -> pd.Series:
    price = members["price_at_signup"].astype(float).copy()
    hit = ((members["plan"] == SIM.price_increase_plan)
           & (members["billing_cycle"] == "monthly")
           & (s >= pd.Timestamp(SIM.price_increase_date)))
    price[hit] = SIM.price_increase_new_price
    return price


def window_counts(log: pd.DataFrame, kind: str, lo: pd.Timestamp | None,
                  hi: pd.Timestamp) -> pd.Series:
    """Count events of ``kind`` per member with lo <= ts < hi."""
    sel = log["event_type"].to_numpy() == kind
    ts = log["ts"].to_numpy()
    sel &= ts < hi.to_datetime64()
    if lo is not None:
        sel &= ts >= lo.to_datetime64()
    return log.loc[sel, "member_id"].value_counts()


def last_event(log: pd.DataFrame, kind: str, hi: pd.Timestamp) -> pd.Series:
    sel = (log["event_type"].to_numpy() == kind) & (log["ts"].to_numpy() < hi.to_datetime64())
    return log.loc[sel].groupby("member_id")["ts"].max()


def build_snapshot(raw: dict, s: pd.Timestamp, with_label: bool = True) -> pd.DataFrame:
    members, log, canc = raw["members"], raw["activity_log"], raw["cancellations"]
    req = canc.set_index("member_id")["request_ts"]
    m = members[members["signup_date"] < s].copy()
    r = m["member_id"].map(req)
    m = m[r.isna() | (r >= s)].copy()  # drop members whose cancellation is already pending

    d30, d90 = s - pd.Timedelta(days=30), s - pd.Timedelta(days=90)
    mid = m["member_id"]
    f = pd.DataFrame({"member_id": mid.to_numpy(), "snapshot_date": s})
    for c in CATEGORICAL:
        f[c] = m[c].to_numpy()
    f["tenure_days"] = (s - m["signup_date"]).dt.days.to_numpy()
    f["distance_km"] = m["distance_km"].to_numpy()
    price = monthly_price_at(m, s)
    f["monthly_price"] = price.to_numpy()
    exposed = (price > m["price_at_signup"]).to_numpy()
    f["price_increase_exposed"] = exposed.astype(int)
    since = (s - pd.Timestamp(SIM.price_increase_date)).days
    f["price_increase_recent"] = (exposed & (0 <= since < PRICE_SHOCK_DAYS)).astype(int)

    def cnt(kind, lo):
        return mid.map(window_counts(log, kind, lo, s)).fillna(0).astype(int).to_numpy()

    f["visits_30d"] = cnt("visit", d30)
    v90 = cnt("visit", d90)
    f["visits_31_90d"] = v90 - f["visits_30d"]
    f["visit_trend"] = f["visits_30d"] - f["visits_31_90d"] / 2.0
    last = mid.map(last_event(log, "visit", s))
    dslv = (s - last).dt.days
    f["days_since_last_visit"] = dslv.fillna(f["tenure_days"].set_axis(mid.index)).to_numpy()
    f["lifetime_visits"] = cnt("visit", None)
    f["visits_per_month"] = f["lifetime_visits"] / np.maximum(f["tenure_days"] / 30.4, 1.0)
    f["booking_failures_90d"] = cnt("booking_failed", d90)
    f["booking_failure_rate_90d"] = f["booking_failures_90d"] / np.maximum(
        f["booking_failures_90d"] + v90, 1)
    f["support_tickets_90d"] = cnt("support_ticket", d90)
    f["events_90d"] = cnt("event_attended", d90)
    f["adopted_cat_90d"] = (cnt("cat_adopted", d90) > 0).astype(int)
    f["adopted_cat_ever"] = (cnt("cat_adopted", None) > 0).astype(int)

    if with_label:
        rr = r.loc[m.index]
        f[LABEL] = ((rr >= s) & (rr < s + pd.Timedelta(days=LABEL_WINDOW_DAYS))).astype(int).to_numpy()
    return f


def build_panel(raw: dict) -> pd.DataFrame:
    return pd.concat([build_snapshot(raw, s) for s in snapshot_dates()], ignore_index=True)


def build_scoring_base(raw: dict) -> pd.DataFrame:
    return build_snapshot(raw, pd.Timestamp(SCORING_SNAPSHOT), with_label=False)


def as_model_frame(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    X = df[features or FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=CATEGORY_LEVELS[c])
    return X


def save(panel: pd.DataFrame, scoring: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED_DIR / "panel_point_in_time.csv", index=False)
    scoring.to_csv(PROCESSED_DIR / "scoring_base_2026_01.csv", index=False)


def load_panel() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "panel_point_in_time.csv", parse_dates=["snapshot_date"])


def load_scoring_base() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "scoring_base_2026_01.csv", parse_dates=["snapshot_date"])
