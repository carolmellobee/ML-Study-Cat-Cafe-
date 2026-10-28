"""Simulate the raw operational data of "Whisker & Co.", a three-location cat cafe chain.

The simulator produces the tables a real cafe would have in its systems:

* ``members``        - one row per member, attributes known at signup
* ``activity_log``   - visits, failed bookings, support tickets, events, cat adoptions
* ``cancellations``  - cancellation *requests* (30-day notice) and exit surveys
* ``crm_actions``    - marketing touches, including win-back offers
* ``price_history``  - list prices over time (one mid-2025 increase)

Churn is generated month by month from a latent engagement process, so the
signals a model can legitimately use (falling visits, failed bookings, recent
cat adoption, early tenure, price increase) *precede* the cancellation request.
Nothing in here is leaky on purpose - the leakage is introduced later, in
``warehouse.py``, the way it usually is in real life: by how a table is built.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_DIR, SIM, SimConfig, ensure_dirs

PLAN_NAMES = ("basic", "plus", "family")
PLAN_VISIT_BASE = np.array([3.2, 6.0, 5.0])  # expected visits / month at average engagement
CHANNELS = ("instagram", "walk_in", "google", "referral", "event")
CHANNEL_PROBS = (0.35, 0.25, 0.15, 0.15, 0.10)
SEASON = {1: 1.10, 2: 0.95, 3: 0.95, 4: 1.0, 5: 1.0, 6: 1.05, 7: 1.12, 8: 1.0,
          9: 0.95, 10: 1.0, 11: 1.0, 12: 1.15}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _random_times(rng, starts, lengths_days, counts):
    """Expand per-row counts into uniformly placed timestamps within [start, start+len)."""
    idx = np.repeat(np.arange(len(counts)), counts)
    offs = rng.random(idx.size) * np.repeat(lengths_days, counts)
    ts = np.repeat(starts, counts) + (offs * 86400).astype("timedelta64[s]")
    return idx, ts


def simulate(cfg: SimConfig = SIM) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    months = pd.date_range(cfg.start, cfg.end, freq="MS")
    n_months = len(months)

    # ---- members ---------------------------------------------------------
    lam = cfg.base_joins_per_month * (1 + cfg.join_growth_per_month * np.arange(n_months))
    lam *= np.array([SEASON[m.month] for m in months])
    joins = rng.poisson(lam)
    n = int(joins.sum())
    join_month = np.repeat(np.arange(n_months), joins)
    days_in = np.array([m.days_in_month for m in months])
    signup = (months.values[join_month]
              + (rng.random(n) * days_in[join_month] * 86400).astype("timedelta64[s]"))
    signup = signup.astype("datetime64[s]")

    plan = rng.choice(3, size=n, p=cfg.plan_probs)
    annual = rng.random(n) < cfg.annual_prob
    loc = rng.integers(0, len(cfg.locations), size=n)
    channel = rng.choice(len(CHANNELS), size=n, p=CHANNEL_PROBS)
    distance = np.round(rng.lognormal(np.log(3.5), 0.7, size=n), 1)
    # Promo codes: heavy-tailed, most codes are used by one or two people (influencer links).
    # A few big campaign codes plus a long tail of one-off influencer / staff codes.
    has_code = rng.random(n) < 0.55
    big = rng.random(n) < 0.30
    code_num = np.where(big, rng.integers(1, 21, size=n), rng.integers(21, cfg.n_promo_codes + 1, size=n))
    promo = np.where(has_code, np.char.add("P", code_num.astype(str)), "NONE")
    # Code numbers are shuffled so the code id itself carries no information.
    perm = {f"P{i}": f"P{j:04d}" for i, j in zip(range(1, cfg.n_promo_codes + 1),
                                                  rng.permutation(cfg.n_promo_codes) + 1)}
    promo = np.array([perm.get(p, "NONE") for p in promo])

    # Share of visits a member tries to book at peak times (weekends, evenings).
    # Peak-time bookers hit "fully booked" far more often - independent of engagement.
    peak_pref = rng.beta(2, 3, size=n)
    e0 = rng.normal(0, 1, size=n) + 0.3 * (channel == CHANNELS.index("referral"))
    u = np.zeros(n)

    plan_price = np.array([cfg.plans[p] for p in PLAN_NAMES])
    price_at_signup = plan_price[plan].copy()
    inc_date = np.datetime64(cfg.price_increase_date)
    basic_idx = PLAN_NAMES.index(cfg.price_increase_plan)
    late_basic = (plan == basic_idx) & ~annual & (signup >= inc_date)
    price_at_signup[late_basic] = cfg.price_increase_new_price
    inc_month = int(np.where(months == pd.Timestamp(cfg.price_increase_date))[0][0])

    requested = np.zeros(n, dtype=bool)
    request_ts = np.full(n, np.datetime64("NaT", "s"), dtype="datetime64[s]")
    adopted_month = np.full(n, -99)
    reason = np.full(n, "", dtype=object)

    logs = []  # (member_idx, ts, event_type)

    for m, ms in enumerate(months):
        ms64 = np.datetime64(ms, "s")
        D = ms.days_in_month
        me64 = ms64 + np.timedelta64(D, "D")
        active = (signup < me64) & ~requested
        a = np.where(active)[0]
        if a.size == 0:
            continue
        start = np.maximum(signup[a], ms64)
        frac = (me64 - start).astype("timedelta64[s]").astype(float) / (D * 86400)

        u[a] = 0.85 * u[a] + rng.normal(0, 0.35, size=a.size)
        e = e0[a] + u[a]

        load = np.bincount(loc[a], minlength=len(cfg.locations)) / cfg.location_capacity
        fail_rate = (0.05 + 0.12 * np.maximum(load[loc[a]] - 0.6, 0)) * (0.3 + 1.75 * peak_pref[a])

        lam_v = (PLAN_VISIT_BASE[plan[a]] * np.exp(0.45 * e)
                 * np.exp(-0.07 * (distance[a] - 3.5)) * SEASON[ms.month])
        lam_v = np.minimum(lam_v, 22.0)  # nobody visits a cat cafe more than ~5x a week
        visits = rng.poisson(lam_v * frac)
        fails = rng.poisson(lam_v * frac * fail_rate)
        tickets = rng.poisson((0.03 + 0.12 * fails) * frac)
        events = rng.poisson(0.2 * np.exp(0.4 * e) * frac)

        can_adopt = adopted_month[a] < 0
        adopt = can_adopt & (rng.random(a.size) < 0.010 * np.exp(0.4 * e) * frac)
        adopted_month[a[adopt]] = m
        recent_adopt = adopted_month[a] >= m - 2

        tenure_m = m - join_month[a]
        price_shock = ((m == inc_month) | (m == inc_month + 1)) & (plan[a] == basic_idx) \
            & ~annual[a] & (signup[a] < inc_date)

        logit = (-3.2
                 + 0.9 * (tenure_m < 2) + 0.45 * ((tenure_m >= 2) & (tenure_m < 5))
                 - 1.1 * e
                 + 0.40 * np.minimum(fails, 4)
                 + 0.45 * np.minimum(tickets, 3)
                 + 1.4 * recent_adopt
                 + 0.8 * price_shock
                 - 1.0 * annual[a]
                 - 0.35 * (channel[a] == CHANNELS.index("referral"))
                 + 0.25 * (plan[a] == PLAN_NAMES.index("family"))
                 + 0.03 * (distance[a] - 3.5)
                 - 0.15 * (events > 0))
        p = _sigmoid(logit) * frac
        req = rng.random(a.size) < p
        window = (me64 - start).astype("timedelta64[s]").astype(float) / 86400
        r_ts = start + (rng.random(a.size) * window * 86400).astype("timedelta64[s]")
        ri = a[req]
        requested[ri] = True
        request_ts[ri] = r_ts[req]
        # Reason recorded on the exit form (dominant driver, with noise)
        rr = np.full(req.sum(), "other", dtype=object)
        rr[(e[req] < -0.8)] = "not_using_it"
        rr[(fails[req] >= 2)] = "too_crowded"
        rr[price_shock[req]] = "price"
        rr[recent_adopt[req]] = "adopted_a_cat"
        rr[rng.random(req.sum()) < 0.12] = "moved_away"
        reason[ri] = rr

        # Activity stops at the request timestamp for members who requested this month.
        end = np.where(req, r_ts, me64)
        length = (end - start).astype("timedelta64[s]").astype(float) / 86400
        for counts, kind in ((visits, "visit"), (fails, "booking_failed"),
                             (tickets, "support_ticket"), (events, "event_attended")):
            idx, ts = _random_times(rng, start, length, counts)
            logs.append(pd.DataFrame({"i": a[idx], "ts": ts, "event_type": kind}))
        idx, ts = _random_times(rng, start, length, adopt.astype(int))
        logs.append(pd.DataFrame({"i": a[idx], "ts": ts, "event_type": "cat_adopted"}))

    member_ids = np.array([f"M{i:05d}" for i in range(1, n + 1)])
    members = pd.DataFrame({
        "member_id": member_ids,
        "signup_date": pd.to_datetime(signup).floor("D"),
        "location": np.array(cfg.locations)[loc],
        "plan": np.array(PLAN_NAMES)[plan],
        "billing_cycle": np.where(annual, "annual", "monthly"),
        "acquisition_channel": np.array(CHANNELS)[channel],
        "distance_km": distance,
        "promo_code": promo,
        "price_at_signup": price_at_signup,
    })

    log = pd.concat(logs, ignore_index=True)
    log["member_id"] = member_ids[log["i"].to_numpy()]
    log = (log.drop(columns="i").assign(ts=lambda d: pd.to_datetime(d["ts"]))
           [["member_id", "ts", "event_type"]].sort_values(["ts", "member_id"])
           .reset_index(drop=True))

    # ---- cancellations ---------------------------------------------------
    ci = np.where(requested)[0]
    req_dt = pd.to_datetime(request_ts[ci])
    survey_filled = rng.random(ci.size) < 0.65
    base_score = {"too_crowded": 3.5, "price": 4.5, "not_using_it": 6.5,
                  "adopted_a_cat": 8.5, "moved_away": 7.5, "other": 6.0}
    score = np.clip(np.round(np.array([base_score[r] for r in reason[ci]])
                             + rng.normal(0, 1.5, ci.size)), 1, 10)
    cancellations = pd.DataFrame({
        "member_id": member_ids[ci],
        "request_ts": req_dt,
        "effective_date": (req_dt + pd.Timedelta(days=cfg.cancellation_notice_days)).floor("D"),
        "reason": reason[ci],
        "exit_survey_score": np.where(survey_filled, score, np.nan),
        "exit_survey_ts": np.where(survey_filled, req_dt + pd.Timedelta(hours=1), pd.NaT),
    }).sort_values("request_ts").reset_index(drop=True)
    cancellations["exit_survey_ts"] = pd.to_datetime(cancellations["exit_survey_ts"])

    # ---- CRM -------------------------------------------------------------
    welcome = pd.DataFrame({"member_id": member_ids, "action_ts": pd.to_datetime(signup)
                            + pd.Timedelta(hours=2), "action": "welcome_email"})
    gets_wb = rng.random(ci.size) < 0.85
    winback = pd.DataFrame({
        "member_id": member_ids[ci][gets_wb],
        "action_ts": req_dt[gets_wb] + pd.to_timedelta(rng.uniform(1, 3, gets_wb.sum()), unit="D"),
        "action": "winback_offer",
    })
    # Party invites go to random active members - pure noise, but realistic CRM volume.
    k = n * 2
    inv_i = rng.integers(0, n, k)
    inv_ts = pd.to_datetime(signup[inv_i]) + pd.to_timedelta(rng.uniform(5, 500, k), unit="D")
    alive = np.where(requested[inv_i], inv_ts < pd.to_datetime(request_ts[inv_i]), True)
    alive &= inv_ts < pd.Timestamp(cfg.end)
    invites = pd.DataFrame({"member_id": member_ids[inv_i][alive], "action_ts": inv_ts[alive],
                            "action": "cat_birthday_invite"})
    crm = (pd.concat([welcome, winback, invites], ignore_index=True)
           .sort_values("action_ts").reset_index(drop=True))

    price_history = pd.DataFrame({
        "effective_date": pd.to_datetime([cfg.start] * 3 + [cfg.price_increase_date]),
        "plan": list(PLAN_NAMES) + [cfg.price_increase_plan],
        "billing_cycle": ["monthly"] * 4,
        "monthly_price": list(plan_price) + [cfg.price_increase_new_price],
    })

    return {"members": members, "activity_log": log, "cancellations": cancellations,
            "crm_actions": crm, "price_history": price_history}


def save(tables: dict[str, pd.DataFrame]) -> None:
    ensure_dirs()
    for name, df in tables.items():
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)


def load_raw() -> dict[str, pd.DataFrame]:
    parse = {"members": ["signup_date"], "activity_log": ["ts"],
             "cancellations": ["request_ts", "effective_date", "exit_survey_ts"],
             "crm_actions": ["action_ts"], "price_history": ["effective_date"]}
    return {k: pd.read_csv(RAW_DIR / f"{k}.csv", parse_dates=v,
                           dtype={"promo_code": str} if k == "members" else None)
            for k, v in parse.items()}


if __name__ == "__main__":
    t = simulate()
    save(t)
    for k, v in t.items():
        print(f"{k:15s} {len(v):>9,d} rows")
