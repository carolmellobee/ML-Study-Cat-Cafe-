"""Model training and evaluation helpers shared by the audit and the final model."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEED

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=SEED, deterministic=True, force_row_wise=True, num_threads=4,
)
FIXED_ROUNDS = 400  # used where there is no validation set (leakage waterfall)


def fit_lgbm(X, y, X_valid=None, y_valid=None, rounds: int = FIXED_ROUNDS):
    dtrain = lgb.Dataset(X, y, free_raw_data=False)
    if X_valid is None:
        return lgb.train(LGB_PARAMS, dtrain, num_boost_round=rounds)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)
    return lgb.train(LGB_PARAMS, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(150, verbose=False)])


def fit_logistic(X: pd.DataFrame, y, categorical: list[str]):
    numeric = [c for c in X.columns if c not in categorical]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
    ])
    model = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=3000))
    Xc = X.copy()
    for c in categorical:
        Xc[c] = Xc[c].astype(str)
    model.fit(Xc, y)
    return model


def predict_logistic(model, X: pd.DataFrame, categorical: list[str]) -> np.ndarray:
    Xc = X.copy()
    for c in categorical:
        Xc[c] = Xc[c].astype(str)
    return model.predict_proba(Xc)[:, 1]


def metrics(y, p, top_frac: float = 0.10) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    k = max(int(round(len(p) * top_frac)), 1)
    top = np.argsort(-p)[:k]
    base = y.mean()
    return {
        "n": int(len(y)), "positives": int(y.sum()), "base_rate": float(base),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "precision_top10": float(y[top].mean()),
        "lift_top10": float(y[top].mean() / base),
        "recall_top10": float(y[top].sum() / y.sum()),
        "expected_vs_observed": float(p.sum() / y.sum()),
    }


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    return df.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"),
                                 n=("y", "size")).reset_index()
