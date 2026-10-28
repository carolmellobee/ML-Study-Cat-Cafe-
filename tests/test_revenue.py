import pandas as pd
import pytest

from catcafe.revenue import revenue_at_risk


def _base(p):
    return pd.DataFrame({"member_id": [f"M{i}" for i in range(10)], "p_cancel_30d": [p] * 10,
                         "monthly_price": [30.0] * 10, "tenure_days": [100] * 10,
                         "plan": ["basic"] * 10, "billing_cycle": ["monthly"] * 10})


def test_zero_risk_means_zero_revenue_at_risk():
    r = revenue_at_risk(_base(0.0))
    assert r["revenue_at_risk_12m"] == 0 and r["mrr_at_risk_30d"] == 0


def test_certain_churn_loses_eleven_months():
    # a request in month 1 removes revenue from month 2 onward (30-day notice)
    r = revenue_at_risk(_base(1.0))
    assert r["revenue_at_risk_12m"] == pytest.approx(10 * 30.0 * 11)
