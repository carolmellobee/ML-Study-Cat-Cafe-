"""Central configuration: dates, prices, paths and split boundaries.

Everything that a reviewer might want to change lives here, so the rest of
the code never hard-codes a date or a price.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

SEED = 20260128


@dataclass(frozen=True)
class SimConfig:
    seed: int = SEED
    # The cafe chain opens three locations on day one.
    start: str = "2024-01-01"
    end: str = "2025-12-31"  # last day of observed history
    export_date: str = "2026-01-15"  # day the analytics team refreshed the warehouse table
    base_joins_per_month: float = 450.0
    join_growth_per_month: float = 0.04
    locations: tuple[str, ...] = ("downtown", "riverside", "uptown")
    location_capacity: float = 1300.0  # active members a location can serve before booking pressure bites
    plans: dict = field(
        default_factory=lambda: {"basic": 29.0, "plus": 49.0, "family": 79.0}
    )
    plan_probs: tuple[float, float, float] = (0.68, 0.24, 0.08)
    annual_prob: float = 0.18
    # Mid-2025 price increase on the monthly Basic plan
    price_increase_date: str = "2025-07-01"
    price_increase_plan: str = "basic"
    price_increase_new_price: float = 35.0
    cancellation_notice_days: int = 30
    n_promo_codes: int = 6000


SIM = SimConfig()

# --- modelling windows -------------------------------------------------------
# Snapshots are taken on the first day of each month. Features use only data
# strictly before the snapshot; the label looks at the following 30 days.
FIRST_SNAPSHOT = "2024-04-01"  # need 90 days of history before the first snapshot
LAST_LABELLED_SNAPSHOT = "2025-12-01"  # label window ends 2025-12-31 = end of history
SCORING_SNAPSHOT = "2026-01-01"  # the live base we score for revenue at risk

TRAIN_END = "2025-08-01"  # last training snapshot (inclusive)
VALID_SNAPSHOTS = ("2025-09-01", "2025-10-01")
TEST_SNAPSHOTS = ("2025-11-01", "2025-12-01")

LABEL_WINDOW_DAYS = 30
REVENUE_HORIZON_MONTHS = 12


def snapshot_dates() -> list[pd.Timestamp]:
    return list(pd.date_range(FIRST_SNAPSHOT, LAST_LABELLED_SNAPSHOT, freq="MS"))


def ensure_dirs() -> None:
    for d in (RAW_DIR, WAREHOUSE_DIR, PROCESSED_DIR, REPORTS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
