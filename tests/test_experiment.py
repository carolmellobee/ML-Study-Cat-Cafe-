import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from catcafe.experiment import analysis, design, policy, traps
from catcafe.experiment.simulate import assign, simulate_experiment


def test_n_per_arm_matches_statsmodels():
    h = proportion_effectsize(0.12, 0.12 * 0.8)
    ref = NormalIndPower().solve_power(abs(h), alpha=0.05, power=0.8)
    assert design.n_per_arm(0.12, 0.20) == int(np.ceil(ref))


def test_mde_inverts_sample_size():
    n = design.n_per_arm(0.12, 0.20)
    assert abs(design.mde_given_n(0.12, n) - 0.20) < 0.01


def test_assignment_is_deterministic_and_balanced():
    ids = pd.Series([f"M{i:05d}" for i in range(20000)])
    a, b = assign(ids), assign(ids)
    assert np.array_equal(a, b)
    assert abs(a.mean() - 0.5) < 0.015


def test_peeking_inflates_and_obf_controls_error(small_scored):
    pop = small_scored.copy()
    Z = traps.aa_simulation(pop, sims=1200)
    crit = 1.959964
    daily = (np.abs(Z[:, 3:]) > crit).any(1).mean()
    looks = Z[:, traps.LOOK_DAYS - 1]
    c = traps.obf_constant(looks[:600])
    K = len(traps.LOOK_DAYS)
    held = (np.abs(looks[600:]) > c * np.sqrt(K / np.arange(1, K + 1))).any(1).mean()
    assert daily > 0.12
    assert 0.02 < held < 0.09


def test_cuped_reduces_variance():
    rng = np.random.default_rng(0)
    x = rng.poisson(5, 4000).astype(float)
    t = rng.random(4000) < 0.5
    y = x * 1.5 + rng.normal(0, 2, 4000) + 0.3 * t
    res = analysis.cuped(y, x, t)
    assert res["cuped"]["se"] < res["raw"]["se"] * 0.6
    assert abs(res["cuped"]["diff"] - 0.3) < 0.2


def test_ipw_value_of_treat_all_equals_mean_difference():
    rng = np.random.default_rng(1)
    t = (rng.random(50000) < 0.5).astype(int)
    y = rng.normal(0, 1, 50000) + 2.0 * t
    v_all = policy.ipw_value(y, t, np.ones_like(t))
    v_none = policy.ipw_value(y, t, np.zeros_like(t))
    assert abs((v_all - v_none) - 2.0) < 0.05


def test_experiment_recovers_planted_reversal(small_scored):
    exp, _ = simulate_experiment(small_scored)
    seg = traps.segments(exp)["table"].set_index("segment")
    assert seg.loc["0-3 mo", "diff"] < 0
    assert seg.loc["12+ mo", "diff"] > 0
