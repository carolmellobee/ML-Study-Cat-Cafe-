"""The audit has to find every planted trap without being told where they are."""
from catcafe import leakage
from catcafe.warehouse import LEAKAGE_TRAPS, NAIVE_LABEL


def test_audit_catches_every_planted_trap(small):
    audit = leakage.run_audit(small["export"], small["panel"], small["raw"])
    flagged = set().union(*leakage.flagged_columns(audit).values())
    assert set(LEAKAGE_TRAPS) <= flagged, set(LEAKAGE_TRAPS) - flagged


def test_audit_does_not_condemn_clean_columns(small):
    audit = leakage.run_audit(small["export"], small["panel"], small["raw"])
    flagged = set().union(*leakage.flagged_columns(audit).values())
    clean = {"visits_30d", "visits_90d", "tenure_days", "plan", "billing_cycle", "distance_km",
             "booking_failures_90d", "support_tickets_90d", "events_90d", "monthly_price"}
    assert not (flagged & clean)


def test_waterfall_ends_far_below_where_it_starts(small):
    wf = leakage.leakage_waterfall(small["export"], small["panel"])
    assert wf["roc_auc"].iloc[0] > 0.95
    assert wf["roc_auc"].iloc[-1] < 0.85


def test_naive_label_is_mostly_already_decided(small):
    lab = leakage.label_already_known(small["export"], small["raw"])
    assert lab["share_request_before_snapshot"] > 0.9
    assert small["export"][NAIVE_LABEL].mean() > 0
