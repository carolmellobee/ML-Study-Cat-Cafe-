"""The clean feature store must never look past the snapshot."""
import numpy as np
import pandas as pd

from catcafe.features import FEATURES, LABEL
from catcafe.warehouse import LEAKAGE_TRAPS


def test_no_leaky_columns_in_model_features():
    # lifetime_visits / days_since_last_visit exist in both tables under the same name;
    # in the feature store they are recomputed point in time (tested below).
    recomputed = {"lifetime_visits", "days_since_last_visit"}
    leaky = {c for c in LEAKAGE_TRAPS if not c.startswith("(")} - recomputed
    assert not (set(FEATURES) & leaky)


def test_history_features_only_use_the_past(small):
    panel, log = small["panel"], small["raw"]["activity_log"]
    rows = panel.sample(300, random_state=0)
    visits = log[log["event_type"] == "visit"]
    for _, r in rows.iterrows():
        v = visits[(visits["member_id"] == r["member_id"]) & (visits["ts"] < r["snapshot_date"])]
        assert len(v) == r["lifetime_visits"]
        if len(v):
            assert r["days_since_last_visit"] == (r["snapshot_date"] - v["ts"].max()).days
    assert (panel["days_since_last_visit"] <= panel["tenure_days"]).all()


def test_pending_cancellations_are_excluded(small):
    panel, canc = small["panel"], small["raw"]["cancellations"]
    req = panel["member_id"].map(canc.set_index("member_id")["request_ts"])
    assert not (req < panel["snapshot_date"]).any()


def test_label_is_request_in_next_30_days(small):
    panel, canc = small["panel"], small["raw"]["cancellations"]
    req = panel["member_id"].map(canc.set_index("member_id")["request_ts"])
    expected = ((req >= panel["snapshot_date"]) &
                (req < panel["snapshot_date"] + pd.Timedelta(days=30))).astype(int)
    assert np.array_equal(expected.to_numpy(), panel[LABEL].to_numpy())
