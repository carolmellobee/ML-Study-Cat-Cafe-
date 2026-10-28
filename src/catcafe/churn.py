"""Train, calibrate and evaluate the clean churn model; then score the live base.

* Evaluation model  - train on snapshots <= Aug-2025, early-stop + calibrate on
  Sep/Oct, report on Nov/Dec (never touched before the final number).
* Production model  - same recipe moved forward one window: train <= Oct,
  calibrate on Nov/Dec, score the 1-Jan-2026 base.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import plotting as P
from .config import TEST_SNAPSHOTS, TRAIN_END, VALID_SNAPSHOTS
from .features import CATEGORICAL, FEATURES_V1, LABEL, as_model_frame
from .modeling import calibration_table, fit_lgbm, fit_logistic, metrics, predict_logistic


class PlattCalibrator:
    """Logistic recalibration on the logit of the raw score (2 parameters)."""

    def fit(self, p, y):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        self.lr = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y)
        return self

    def transform(self, p):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return self.lr.predict_proba(z.reshape(-1, 1))[:, 1]


def split(panel: pd.DataFrame):
    tr = panel[panel["snapshot_date"] <= pd.Timestamp(TRAIN_END)]
    va = panel[panel["snapshot_date"].isin(pd.to_datetime(list(VALID_SNAPSHOTS)))]
    te = panel[panel["snapshot_date"].isin(pd.to_datetime(list(TEST_SNAPSHOTS)))]
    return tr, va, te


def train_and_evaluate(panel: pd.DataFrame, scoring: pd.DataFrame) -> dict:
    tr, va, te = split(panel)
    Xtr, Xva, Xte = as_model_frame(tr), as_model_frame(va), as_model_frame(te)
    ytr, yva, yte = tr[LABEL].to_numpy(), va[LABEL].to_numpy(), te[LABEL].to_numpy()

    gbm = fit_lgbm(Xtr, ytr, Xva, yva)
    rounds = gbm.best_iteration
    raw_va, raw_te = gbm.predict(Xva, num_iteration=rounds), gbm.predict(Xte, num_iteration=rounds)
    cal = PlattCalibrator().fit(raw_va, yva)
    p_te = cal.transform(raw_te)

    logit = fit_logistic(Xtr, ytr, CATEGORICAL)
    p_lr = predict_logistic(logit, Xte, CATEGORICAL)

    # Naive baseline a cafe manager might use: "hasn't visited in a while"
    p_rule = te["days_since_last_visit"].to_numpy(dtype=float)

    results = {
        "rows": {"train": len(tr), "valid": len(va), "test": len(te)},
        "best_iteration": int(rounds),
        "test_metrics": {
            "lightgbm_calibrated": metrics(yte, p_te),
            "lightgbm_uncalibrated": metrics(yte, raw_te),
            "logistic_regression": metrics(yte, p_lr),
            "rule_days_since_last_visit": {k: v for k, v in metrics(yte, p_rule / p_rule.max()).items()
                                           if k in ("roc_auc", "pr_auc", "lift_top10", "recall_top10")},
        },
    }

    # --- v1 (kept for the write-up): price level + persistent exposure flag ----
    v1 = fit_lgbm(as_model_frame(tr, FEATURES_V1), ytr, as_model_frame(va, FEATURES_V1), yva)
    p_v1 = v1.predict(as_model_frame(te, FEATURES_V1), num_iteration=v1.best_iteration)
    results["v1_model"] = {"features": FEATURES_V1,
                           "test_metrics_uncalibrated": metrics(yte, p_v1)}

    # --- production refit, scored on the live base --------------------------
    trp = pd.concat([tr, va])
    gbm_prod = fit_lgbm(as_model_frame(trp), trp[LABEL].to_numpy(), rounds=rounds)
    cal_prod = PlattCalibrator().fit(gbm_prod.predict(Xte), yte)
    scored = scoring.copy()
    scored["p_cancel_30d"] = cal_prod.transform(gbm_prod.predict(as_model_frame(scoring)))
    results["scoring"] = {"members": int(len(scored)),
                          "mean_p": float(scored["p_cancel_30d"].mean()),
                          "expected_requests_jan26": float(scored["p_cancel_30d"].sum())}

    figs = {
        "calibration": _plot_calibration(yte, raw_te, p_te),
        "lift": _plot_gains(yte, {"LightGBM": p_te, "Logistic regression": p_lr,
                                  "Rule: days since last visit": p_rule}),
    }
    results["figures"] = figs
    test_scored = te.assign(p=p_te)
    return {"results": results, "eval_model": gbm, "prod_model": gbm_prod, "v1_model": v1,
            "test_frame": test_scored, "scored_base": scored}


def _plot_calibration(y, p_raw, p_cal) -> str:
    fig, ax = P.plt.subplots(figsize=(6.2, 5))
    hi = 0.0
    for p, color, name in ((p_raw, P.GRAY_BAR, "Before recalibration"),
                           (p_cal, P.BLUE, "After recalibration (Platt on Sep/Oct)")):
        t = calibration_table(y, p)
        ax.plot(t["mean_pred"], t["observed"], marker="o", color=color, label=name)
        hi = max(hi, t["mean_pred"].max(), t["observed"].max())
    lim = round(hi * 1.1 + 0.01, 2)
    ax.plot([0, lim], [0, lim], ls="--", lw=1, color=P.MUTED)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted 30-day cancellation probability (decile mean)")
    ax.set_ylabel("Observed cancellation rate")
    ax.set_title("Calibration on the held-out test months", pad=22)
    P.subtitle(ax, "Nov-Dec 2025. Dashed line = perfect calibration.")
    ax.legend(loc="upper left")
    return P.save(fig, "churn_calibration")


def _plot_gains(y, scores: dict) -> str:
    fig, ax = P.plt.subplots(figsize=(6.6, 5))
    y = np.asarray(y)
    x = np.linspace(0, 1, 101)
    for (name, p), color in zip(scores.items(), [P.BLUE, P.ORANGE, P.AQUA]):
        order = np.argsort(-np.asarray(p))
        cum = np.cumsum(y[order]) / y.sum()
        idx = np.clip((x * len(y)).astype(int) - 1, 0, len(y) - 1)
        g = np.where(x == 0, 0, cum[idx])
        ax.plot(x * 100, g * 100, color=color, label=name)
    ax.plot([0, 100], [0, 100], ls="--", lw=1, color=P.MUTED, label="Random")
    ax.axvline(10, color=P.AXIS, lw=1)
    ax.set_xlabel("% of members contacted (highest score first)")
    ax.set_ylabel("% of next-month cancellations captured")
    ax.set_title("Who should marketing call first?", pad=22)
    P.subtitle(ax, "Cumulative gains on the Nov-Dec 2025 test months")
    ax.legend(loc="lower right")
    return P.save(fig, "churn_gains")
