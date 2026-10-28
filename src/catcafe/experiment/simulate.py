"""Simulate the "Purr-sonal Perks" retention experiment (Feb-Mar 2026).

Treatment: a personalised email + app push featuring the member's favourite
cats, one free guest pass, and 20% off the next monthly fee.
Control: business as usual (no campaign).

Unit: member, randomised 50/50 by hashing the member id.
Population: every active member on 2-Feb-2026 who joined before 1-Jan-2026.
Primary metric: cancellation request within 60 days.

The *true* effects planted below are what the analysis has to recover - and
three of them are traps:

* Novelty: the visit bump and the retention effect are strongest in the first
  ~2 weeks and decay (time constant ``TAU_DAYS``).
* Reversal: for members with 12+ months of tenure the campaign *raises*
  cancellations (a "we miss you" discount reminds loyal, habitual members
  that they are paying - the sleeping-dog effect).
* Peeking: nothing is planted; peeking is a property of the analyst.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..config import SEED
from ..revenue import tenure_band

DAYS = 60
TAU_DAYS = 9.0
START = pd.Timestamp("2026-02-02")
# (long-run hazard multiplier, day-0 hazard multiplier) by tenure band
TRUE_HAZARD_MULT = {"0-3 mo": (0.45, 0.30), "3-6 mo": (0.55, 0.38),
                    "6-12 mo": (0.70, 0.48), "12+ mo": (1.55, 2.10)}
VISIT_LIFT_DAY0, VISIT_LIFT_LONG = 0.45, 0.03
DISCOUNT, GUEST_PASS_COST, GUEST_PASS_REDEEM = 0.20, 4.0, 0.40


def assign(member_ids: pd.Series, salt: str = "purrsonal-perks-2026") -> np.ndarray:
    h = member_ids.map(lambda m: int(hashlib.sha256(f"{salt}:{m}".encode()).hexdigest()[:8], 16))
    return (h % 2).to_numpy().astype(int)


def treatment_multiplier(band: np.ndarray, t: np.ndarray) -> np.ndarray:
    long = np.array([TRUE_HAZARD_MULT[b][0] for b in band])
    early = np.array([TRUE_HAZARD_MULT[b][1] for b in band])
    return long[:, None] + (early - long)[:, None] * np.exp(-t[None, :] / TAU_DAYS)


def eligible_population(scored: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Members scored on 1-Jan who did not cancel during January."""
    rng = np.random.default_rng(seed + 1)
    jan_cancel = rng.random(len(scored)) < scored["p_cancel_30d"].to_numpy()
    pop = scored.loc[~jan_cancel].copy().reset_index(drop=True)
    pop["tenure_days"] = pop["tenure_days"] + 32
    pop["tenure_band"] = tenure_band(pop["tenure_days"]).astype(str)
    pop["engagement"] = np.where(pop["visits_30d"] >= 2, "active", "dormant")
    return pop


def daily_hazard(pop: pd.DataFrame) -> np.ndarray:
    p30 = pop["p_cancel_30d"].to_numpy()
    return -np.log(1 - p30) / 30.0


def simulate_experiment(scored: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 2)
    pop = eligible_population(scored, seed)
    n = len(pop)
    pop["treatment"] = assign(pop["member_id"])
    t = np.arange(DAYS)
    h0 = daily_hazard(pop)
    mult = np.where(pop["treatment"].to_numpy()[:, None] == 1,
                    treatment_multiplier(pop["tenure_band"].to_numpy(), t), 1.0)
    h = h0[:, None] * mult
    # cancellation day: first day with an event (discrete-time hazard)
    ev = rng.random((n, DAYS)) < (1 - np.exp(-h))
    has = ev.any(1)
    day = np.where(has, ev.argmax(1), -1)
    pop["cancelled_60d"] = has.astype(int)
    pop["cancel_day"] = day

    # visits: Poisson per day until cancellation, with a decaying novelty bump for treated
    base_rate = (0.6 * pop["visits_30d"] + 0.4 * pop["visits_31_90d"] / 2).to_numpy() / 30 + 0.01
    lift = 1 + np.where(pop["treatment"].to_numpy()[:, None] == 1,
                        VISIT_LIFT_LONG + (VISIT_LIFT_DAY0 - VISIT_LIFT_LONG) * np.exp(-t / TAU_DAYS), 0)
    member_noise = rng.gamma(4, 1 / 4, n)[:, None]
    alive = (day[:, None] < 0) | (t[None, :] < day[:, None])
    visits = rng.poisson(base_rate[:, None] * member_noise * lift) * alive
    daily = pd.DataFrame({
        "member_id": np.repeat(pop["member_id"].to_numpy(), DAYS),
        "day": np.tile(t, n),
        "visits": visits.ravel(),
        "active": alive.ravel().astype(int),
    })
    daily = daily[daily["active"] == 1]
    pop["visits_60d"] = visits.sum(1)
    pop["pre_visits_30d"] = pop["visits_30d"]

    # money: charges on day 30 and day 60 happen unless the 30-day notice has run out
    price = pop["monthly_price"].to_numpy()
    charge_30 = np.ones(n)  # any request in the window has not taken effect yet by day 30
    charge_60 = np.where(has & (day < 30), 0, 1)
    charge_90 = np.where(has, 0, 1)
    treated = pop["treatment"].to_numpy() == 1
    discount = np.where(treated, DISCOUNT * price * charge_30, 0)
    guest = np.where(treated & (rng.random(n) < GUEST_PASS_REDEEM), GUEST_PASS_COST, 0)
    pop["offer_cost"] = discount + guest
    pop["net_revenue_90d"] = price * (charge_30 + charge_60 + charge_90) - pop["offer_cost"]
    return pop, daily
