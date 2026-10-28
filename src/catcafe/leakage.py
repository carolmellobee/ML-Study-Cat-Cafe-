"""Leakage audit for ``member_snapshot_export``.

The audit does *not* read ``warehouse.LEAKAGE_TRAPS``. It finds problems the way
a reviewer would: suspiciously strong single features, missingness that tracks
the label, values that need information from after the snapshot, labels whose
outcome was already known, target encodings fitted on the evaluation rows, and
splits that put the same member / future months on both sides.

Then it quantifies each trap with a "leakage waterfall": the same model refit
after each fix, so the inflated score is visibly deflated one step at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from . import plotting as P
from .config import SEED, TEST_SNAPSHOTS, TRAIN_END
from .features import CATEGORICAL, LABEL
from .modeling import fit_lgbm
from .warehouse import NAIVE_LABEL

AUC_FLAG = 0.80          # a single raw column this predictive deserves an explanation
MISSING_GAP_FLAG = 0.20  # |P(null | y=1) - P(null | y=0)|


# ----------------------------------------------------------------- checks ---
def _auc(y, x) -> float:
    a = roc_auc_score(y, x)
    return max(a, 1 - a)


def single_feature_screen(df: pd.DataFrame, label: str, exclude=()) -> pd.DataFrame:
    """AUC of each column on its own. Missing values get a sentinel so that
    'is missing' can show up as signal. Categoricals use out-of-fold label means."""
    y = df[label].to_numpy()
    out = []
    for c in df.columns:
        if c in (label, *exclude):
            continue
        s = df[c]
        if s.dtype == object or str(s.dtype) in ("category", "string", "str"):
            enc = np.zeros(len(df))
            for tr, te in GroupKFold(5).split(df, groups=df["member_id"]):
                means = df.iloc[tr].groupby(c)[label].mean()
                enc[te] = s.iloc[te].map(means).fillna(y[tr].mean()).to_numpy()
            x = enc
        else:
            x = s.astype(float).fillna(-999).to_numpy()
        if np.nanstd(x) == 0:
            continue
        out.append({"column": c, "single_feature_auc": _auc(y, x)})
    res = pd.DataFrame(out).sort_values("single_feature_auc", ascending=False)
    res["flag"] = res["single_feature_auc"] >= AUC_FLAG
    return res.reset_index(drop=True)


def missingness_vs_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    y = df[label] == 1
    rows = []
    for c in df.columns:
        if c == label or df[c].isna().sum() == 0:
            continue
        na = df[c].isna()
        rows.append({"column": c, "null_rate_pos": na[y].mean(), "null_rate_neg": na[~y].mean()})
    res = pd.DataFrame(rows, columns=["column", "null_rate_pos", "null_rate_neg"])
    res["gap"] = (res["null_rate_pos"] - res["null_rate_neg"]).abs()
    res["flag"] = res["gap"] >= MISSING_GAP_FLAG
    return res


def information_timing(export: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """For every column that is backed by a timestamped source event, measure the
    share of non-empty values whose source event happened on/after the snapshot.
    Any share above zero means the column knows the future."""
    canc, crm, log = raw["cancellations"], raw["crm_actions"], raw["activity_log"]
    first_adopt = log[log["event_type"] == "cat_adopted"].groupby("member_id")["ts"].min()
    first_winback = crm[crm["action"] == "winback_offer"].groupby("member_id")["action_ts"].min()
    survey_ts = canc.set_index("member_id")["exit_survey_ts"]
    s = export["snapshot_date"]
    checks = {
        "exit_survey_score": (export["exit_survey_score"].notna(), export["member_id"].map(survey_ts)),
        "winback_offer_sent": (export["winback_offer_sent"] == 1, export["member_id"].map(first_winback)),
        "has_adopted_cat": (export["has_adopted_cat"] == 1, export["member_id"].map(first_adopt)),
    }
    rows = []
    for col, (present, src_ts) in checks.items():
        n = int(present.sum())
        fut = int((present & (src_ts >= s)).sum())
        rows.append({"column": col, "non_empty_rows": n, "source_event_after_snapshot": fut,
                     "share_from_future": fut / n if n else 0.0})
    return pd.DataFrame(rows)


def point_in_time_mismatch(export: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Recompute history-based columns strictly before the snapshot and compare.
    Also count physically impossible values (e.g. a last visit before signup)."""
    keys = ["member_id", "snapshot_date"]
    j = export.merge(panel[keys + ["lifetime_visits", "days_since_last_visit", "adopted_cat_ever"]],
                     on=keys, suffixes=("", "_pit"))
    out = []
    for col, pit in (("lifetime_visits", "lifetime_visits_pit"),
                     ("days_since_last_visit", "days_since_last_visit_pit"),
                     ("has_adopted_cat", "adopted_cat_ever")):
        a, b = j[col], j[pit]
        out.append({"column": col, "rows_compared": len(j),
                    "mismatch_share": float((a.fillna(-1) != b.fillna(-1)).mean())})
    impossible = (export["days_since_last_visit"] > export["tenure_days"] + 0).mean()
    res = pd.DataFrame(out)
    res["impossible_share"] = np.where(res["column"] == "days_since_last_visit", impossible, np.nan)
    return res


def label_already_known(export: pd.DataFrame, raw: dict) -> dict:
    req = raw["cancellations"].set_index("member_id")["request_ts"]
    pos = export[export[NAIVE_LABEL] == 1]
    r = pos["member_id"].map(req)
    return {"positives": int(len(pos)),
            "share_request_before_snapshot": float((r < pos["snapshot_date"]).mean()),
            "pending_rows_in_population": int(export["cancel_request_pending"].sum())}


def target_encoding_check(df: pd.DataFrame, cat: str, te_col: str, label: str) -> dict:
    """Compare the delivered encoding with an honest out-of-fold one (grouped by member)."""
    y = df[label].to_numpy()
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(df, groups=df["member_id"]):
        means = df.iloc[tr].groupby(cat)[label].mean()
        oof[te] = df[cat].iloc[te].map(means).fillna(y[tr].mean()).to_numpy()
    counts = df.groupby(cat)["member_id"].nunique()
    return {"cardinality": int(df[cat].nunique()),
            "share_codes_single_member": float((counts == 1).mean()),
            "auc_delivered_encoding": _auc(y, df[te_col]),
            "auc_out_of_fold_encoding": _auc(y, oof)}


def split_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    tm, sm = set(train["member_id"]), set(test["member_id"])
    return {"test_rows_whose_member_is_in_train": float(test["member_id"].isin(tm).mean()),
            "test_members_seen_in_train": float(len(sm & tm) / len(sm)),
            "train_rows_after_earliest_test_date": float(
                (train["snapshot_date"] > test["snapshot_date"].min()).mean())}


# ------------------------------------------------------------- waterfall ---
BASE_COLS = CATEGORICAL + ["distance_km", "tenure_days", "monthly_price", "visits_30d",
                           "visits_90d", "booking_failures_90d", "support_tickets_90d", "events_90d"]
TIME_TRAVEL = ["lifetime_visits", "days_since_last_visit", "has_adopted_cat"]
POST_OUTCOME = ["exit_survey_score", "winback_offer_sent"]


def _xy(df, cols, label):
    X = df[cols].copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = X[c].astype("category")
    return X, df[label].to_numpy()


def _random_split(df, frac=0.2):
    rng = np.random.default_rng(SEED)
    m = rng.random(len(df)) < frac
    return df[~m], df[m]


def _time_split(df):
    tr = df[df["snapshot_date"] <= pd.Timestamp(TRAIN_END)]
    te = df[df["snapshot_date"].isin(pd.to_datetime(list(TEST_SNAPSHOTS)))]
    return tr, te


def _score(train, test, cols, label):
    Xtr, ytr = _xy(train, cols, label)
    Xte, yte = _xy(test, cols, label)
    model = fit_lgbm(Xtr, ytr)
    p = model.predict(Xte)
    return float(roc_auc_score(yte, p))


def leakage_waterfall(export: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    keys = ["member_id", "snapshot_date"]
    pit = panel[keys + [LABEL, "lifetime_visits", "days_since_last_visit", "adopted_cat_ever"]].rename(
        columns={"lifetime_visits": "lifetime_visits_pit",
                 "days_since_last_visit": "days_since_last_visit_pit"})
    steps = []

    # S0 - as delivered
    cols0 = BASE_COLS + TIME_TRAVEL + POST_OUTCOME + ["cancel_request_pending", "promo_code_churn_rate"]
    tr, te = _random_split(export)
    steps.append(("As delivered (all columns, random split)", _score(tr, te, cols0, NAIVE_LABEL)))

    # S1 - fix the label: predict the cancellation *request*, drop already-pending members
    df = export[export["cancel_request_pending"] == 0].merge(pit, on=keys, how="inner")
    df["promo_code_churn_rate"] = df.groupby("promo_code")[LABEL].transform("mean")  # analyst re-runs it
    cols1 = [c for c in cols0 if c != "cancel_request_pending"]
    tr, te = _random_split(df)
    steps.append(("Fix label: predict the request, drop pending", _score(tr, te, cols1, LABEL)))

    # S2 - drop columns that only exist after the outcome
    cols2 = [c for c in cols1 if c not in POST_OUTCOME]
    steps.append(("Drop post-outcome columns (survey, win-back)", _score(tr, te, cols2, LABEL)))

    # S3 - rebuild history columns point in time
    df = df.drop(columns=TIME_TRAVEL).rename(columns={
        "lifetime_visits_pit": "lifetime_visits", "days_since_last_visit_pit": "days_since_last_visit",
        "adopted_cat_ever": "has_adopted_cat"})
    tr, te = _random_split(df)
    steps.append(("Recompute history columns point in time", _score(tr, te, cols2, LABEL)))

    # S4 - target encoding must be fitted on training rows only (out of fold, by member)
    tr = tr.copy()
    te = te.copy()
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(tr, groups=tr["member_id"]):
        means = tr.iloc[a].groupby("promo_code")[LABEL].mean()
        oof[b] = tr["promo_code"].iloc[b].map(means).fillna(tr[LABEL].iloc[a].mean()).to_numpy()
    tr["promo_code_churn_rate"] = oof
    te["promo_code_churn_rate"] = te["promo_code"].map(tr.groupby("promo_code")[LABEL].mean()).fillna(
        tr[LABEL].mean())
    steps.append(("Fit target encoding on train folds only", _score(tr, te, cols2, LABEL)))
    cols4 = [c for c in cols2 if c != "promo_code_churn_rate"]
    steps.append(("...and since it adds nothing, drop it", _score(tr, te, cols4, LABEL)))

    # S5 - honest evaluation: train on the past, test on later months
    tr, te = _time_split(df)
    steps.append(("Out-of-time split (train <= Aug-25, test Nov/Dec-25)", _score(tr, te, cols4, LABEL)))

    res = pd.DataFrame(steps, columns=["step", "roc_auc"])
    res["delta"] = res["roc_auc"].diff()
    return res


def run_audit(export: pd.DataFrame, panel: pd.DataFrame, raw: dict) -> dict:
    exclude = ("member_id", "snapshot_date")
    screen = single_feature_screen(export, NAIVE_LABEL, exclude=exclude)
    tr_r, te_r = _random_split(export)
    tr_t, te_t = _time_split(panel)
    return {
        "single_feature_screen": screen,
        "missingness": missingness_vs_label(export, NAIVE_LABEL),
        "information_timing": information_timing(export, raw),
        "point_in_time_mismatch": point_in_time_mismatch(export, panel),
        "label_already_known": label_already_known(export, raw),
        "target_encoding": target_encoding_check(export, "promo_code", "promo_code_churn_rate",
                                                 NAIVE_LABEL),
        "split_random": split_audit(tr_r, te_r),
        "split_out_of_time": split_audit(tr_t, te_t),
        "waterfall": leakage_waterfall(export, panel),
    }


def plot_waterfall(wf: pd.DataFrame) -> str:
    fig, ax = P.plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(wf))
    prev = None
    last = len(wf) - 1
    for i, auc in enumerate(wf["roc_auc"]):
        if prev is None:
            ax.bar(i, auc - 0.5, bottom=0.5, color=P.RED, width=0.6)
            ax.text(i, auc + 0.008, f"{auc:.3f}", ha="center", color=P.INK, fontsize=9)
        else:
            lo, hi = sorted((prev, auc))
            ax.bar(i, max(hi - lo, 0.002), bottom=lo, color=P.GRAY_BAR, width=0.6)
            if i < last:
                ax.text(i, lo - 0.022, f"{auc:.3f}", ha="center", color=P.INK, fontsize=9)
        prev = auc
    final = wf["roc_auc"].iloc[-1]
    ax.bar(last, final - 0.5, bottom=0.5, color=P.BLUE, width=0.6)
    ax.text(last, final - 0.025, f"{final:.3f}", ha="center", color="#ffffff", fontsize=9,
            fontweight="bold")
    short = ["As delivered", "Fix label\ndefinition", "Drop post-\noutcome cols",
             "Point-in-time\nhistory", "Out-of-fold\ntarget enc.", "Drop target\nencoding",
             "Out-of-time\nsplit"]
    ax.set_xticks(x, short[:len(wf)], fontsize=8.5)
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("Test ROC AUC")
    ax.set_title("Leakage waterfall: from a 'perfect' model to the honest one", pad=22)
    P.subtitle(ax, "Same LightGBM settings at every step. Red = as delivered, blue = honest number.")
    ax.grid(axis="x", visible=False)
    return P.save(fig, "leakage_waterfall")


def plot_screen(screen: pd.DataFrame, note: str, top: int = 12) -> str:
    s = screen.head(top).iloc[::-1]
    fig, ax = P.plt.subplots(figsize=(7.2, 4.6))
    colors = [P.RED if f else P.GRAY_BAR for f in s["flag"]]
    ax.barh(s["column"], s["single_feature_auc"] - 0.5, left=0.5, color=colors, height=0.62)
    ax.axvline(AUC_FLAG, color=P.INK_2, lw=1, ls="--")
    ax.text(AUC_FLAG + 0.005, len(s) - 0.4, f"flag at {AUC_FLAG}", color=P.INK_2, fontsize=8.5)
    for yi, v in enumerate(s["single_feature_auc"]):
        ax.text(v + 0.005, yi, f"{v:.2f}", va="center", color=P.INK_2, fontsize=8.5)
    ax.set_xlim(0.5, 1.05)
    ax.set_xlabel("ROC AUC of the column on its own")
    ax.set_title("Check 1: single-column screen", pad=22)
    P.subtitle(ax, note)
    ax.grid(axis="y", visible=False)
    return P.save(fig, "leakage_single_feature_screen")


def flagged_columns(audit: dict, timing_flag: float = 0.05, mismatch_flag: float = 0.05,
                    te_gap_flag: float = 0.05) -> dict[str, list[str]]:
    """Which columns each check condemns. A column is leaky if any check flags it."""
    out = {
        "single_feature_screen": audit["single_feature_screen"].query("flag")["column"].tolist(),
        "missingness": audit["missingness"].query("flag")["column"].tolist(),
        "information_timing": audit["information_timing"].query(
            "share_from_future > @timing_flag")["column"].tolist(),
        "point_in_time_mismatch": audit["point_in_time_mismatch"].query(
            "mismatch_share > @mismatch_flag")["column"].tolist(),
        "label_definition": (["cancel_request_pending"]
                             if audit["label_already_known"]["share_request_before_snapshot"] > 0.5 else []),
        "target_encoding": (["promo_code_churn_rate"]
                            if audit["target_encoding"]["auc_delivered_encoding"]
                            - audit["target_encoding"]["auc_out_of_fold_encoding"] > te_gap_flag else []),
        "split": (["(random row split)"]
                  if audit["split_random"]["train_rows_after_earliest_test_date"] > 0 else []),
    }
    return out
