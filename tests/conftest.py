import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CATCAFE_FIG_DIR"] = tempfile.mkdtemp(prefix="catcafe-figs-")

from catcafe import features, simulate, warehouse  # noqa: E402
from catcafe.config import SIM  # noqa: E402


@pytest.fixture(scope="session")
def small():
    """A smaller cafe (about a quarter of the members) so the tests run in seconds."""
    cfg = replace(SIM, base_joins_per_month=110.0, location_capacity=320.0)
    raw = simulate.simulate(cfg)
    panel = features.build_panel(raw)
    export = warehouse.build_export(raw)
    return {"raw": raw, "panel": panel, "export": export}


@pytest.fixture(scope="session")
def small_scored(small):
    s = features.build_scoring_base(small["raw"])
    s["p_cancel_30d"] = 0.06
    return s.reset_index(drop=True)


__all__ = ["pd"]
