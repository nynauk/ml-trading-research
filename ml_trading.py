"""Can supervised models predict the next hour of EURUSD well enough to trade after costs?

Data: the HistData EURUSD one-minute bid bars from the SwiftPip research, 2004-2024, as hourly bars.
Features are built only from bars that have closed; the label is the return a position would actually earn
in the C++ engine (filled at the next open, held for one bar); models and trading thresholds are chosen on
2015-2019 by rules written to results/selection.json before 2020-2024 is evaluated; every policy is
backtested by the cpp-backtesting-engine with spread and commission.

  python ml_trading.py              # all experiments -> results/experiments.json (a few minutes)
  python make_figures.py            # charts -> results/figures/
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent
ENGINE_REPO = Path(os.environ.get("BT_ENGINE_REPO", ROOT.parent / "cpp-backtesting-engine"))
DATA = Path(os.environ.get("ML_DATA", ENGINE_REPO / "data" / "eurusd_1h.csv"))
RESULTS = ROOT / "results"
SEED = 42

PERIODS = {"train": ("2004-01-01", "2015-01-01"),
           "validation": ("2015-01-01", "2020-01-01"),
           "test": ("2020-01-01", "2025-01-01")}
WALK_FORWARD_YEARS = list(range(2008, 2025))  # retrain on every earlier bar, then predict the year

UNITS, CASH = 100_000.0, 100_000.0  # one standard lot on $100,000, as in the engine experiments
PIP = 0.0001
COMMISSION = 3.5 / 100_000  # $3.50 per side per 100,000 units
COST_LEVELS = {"no costs": (0.0, 0.0),
               "0.5 pip + commission": (0.5 * PIP, COMMISSION),
               "1 pip + commission": (1.0 * PIP, COMMISSION),
               "2 pips + commission": (2.0 * PIP, COMMISSION)}
BASELINE_COST = "0.5 pip + commission"
THRESHOLDS = [round(float(x), 3) for x in np.arange(0.0, 0.0801, 0.005)]  # trade when |p - 0.5| > threshold
MIN_VALIDATION_FILLS = 100  # a threshold must trade at least this often on validation to be eligible
BOOTSTRAP_REPS = 200  # resamples of whole days for confidence intervals

# SwiftPip's session clock, as in its research code (ftmotest/asia_cisd.py): HistData is EST without daylight
# saving, session hour = EST + 7 so hour 0 is the 17:00 rollover, Asia range = hours 2-9 (19:00-02:59 EST),
# London = hours 10-16.
SESSION_OFFSET_HOURS = 7
ASIA_FIRST_HOUR, ASIA_LAST_HOUR = 2, 9
MIN_ASIA_BARS = 6  # an Asia range needs at least 6 of its 8 hourly bars

PRICE = ["ret_1", "ret_3", "ret_12", "ret_24", "ma_dist_30", "ma_cross_10_30", "vol_20", "vol_ratio_20_100",
         "range_pos_24"]
CALENDAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
ASIA = ["asia_pos", "asia_high_dist", "asia_low_dist", "asia_width", "swept_high", "swept_low",
        "rejected_high", "rejected_low"]
FEATURE_SETS = {"price": PRICE, "price + calendar": PRICE + CALENDAR,
                "price + calendar + Asia range": PRICE + CALENDAR + ASIA}


def make_model(name: str):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if name == "gradient boosting":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              early_stopping=False, random_state=SEED)
    raise ValueError(name)


MODELS = ["logistic", "gradient boosting"]


# ---------------------------------------------------------------- data, features, labels
def load_bars(path: Path = DATA) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"missing {path}: run python/prepare_data.py in the cpp-backtesting-engine repository first")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not df["timestamp"].is_monotonic_increasing or df["timestamp"].duplicated().any():
        raise ValueError("bars must be sorted with unique timestamps")
    return df


def session_clock(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """SwiftPip session hour (0 = 17:00 EST rollover) and session day of each bar."""
    session = bars["timestamp"] + pd.Timedelta(hours=SESSION_OFFSET_HOURS)
    return session.dt.hour, session.dt.floor("D")


def asia_levels(bars: pd.DataFrame) -> pd.DataFrame:
    """For each bar, the SwiftPip Asia range (high/low of session hours 2-9) most recently completed by the
    bar's own close, and the session day it belongs to. A range only counts once its last bar has closed."""
    hour, day = session_clock(bars)
    in_asia = hour.between(ASIA_FIRST_HOUR, ASIA_LAST_HOUR)
    asia = (bars.loc[in_asia, ["high", "low"]].groupby(day[in_asia])
            .agg(asia_high=("high", "max"), asia_low=("low", "min"), n=("high", "size")))
    asia = asia[asia["n"] >= MIN_ASIA_BARS].drop(columns="n")
    asia["complete_at"] = asia.index + pd.Timedelta(hours=ASIA_LAST_HOUR + 1)  # session time the last bar closes
    asia["asia_day"] = asia.index
    closes_at = bars["timestamp"] + pd.Timedelta(hours=SESSION_OFFSET_HOURS + 1)
    left = pd.DataFrame({"closes_at": closes_at.to_numpy(), "row": np.arange(len(bars))})
    matched = pd.merge_asof(left, asia.reset_index(drop=True).sort_values("complete_at"),
                            left_on="closes_at", right_on="complete_at", direction="backward")
    matched.index = bars.index
    return matched[["asia_high", "asia_low", "asia_day"]]


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Every value in row t uses bars 0..t only (tests/test_ml_trading.py checks this by truncation)."""
    f = pd.DataFrame(index=bars.index)
    close, high, low = bars["close"], bars["high"], bars["low"]
    log_close = np.log(close)
    for k in (1, 3, 12, 24):
        f[f"ret_{k}"] = log_close.diff(k)
    ma10, ma30 = close.rolling(10).mean(), close.rolling(30).mean()
    f["ma_dist_30"] = close / ma30 - 1.0
    f["ma_cross_10_30"] = ma10 / ma30 - 1.0
    r1 = log_close.diff()
    f["vol_20"] = r1.rolling(20).std()
    f["vol_ratio_20_100"] = f["vol_20"] / r1.rolling(100).std()
    hi24, lo24 = high.rolling(24).max(), low.rolling(24).min()
    f["range_pos_24"] = (close - lo24) / (hi24 - lo24).where(hi24 > lo24)

    hour, day = session_clock(bars)
    dow = day.dt.dayofweek
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    matched = asia_levels(bars)
    a_hi, a_lo = matched["asia_high"], matched["asia_low"]
    atr24 = (high - low).rolling(24).mean()
    width = (a_hi - a_lo).where(a_hi > a_lo)
    f["asia_pos"] = (close - (a_hi + a_lo) / 2) / width
    f["asia_high_dist"] = (close - a_hi) / atr24
    f["asia_low_dist"] = (close - a_lo) / atr24
    f["asia_width"] = width / atr24
    # Sweeps: after today's range completed (session hours 10+), has price traded beyond it, and closed back?
    after_asia = (hour > ASIA_LAST_HOUR) & (matched["asia_day"] == day)
    post_high = high.where(after_asia).groupby(day).cummax()
    post_low = low.where(after_asia).groupby(day).cummin()
    f["swept_high"] = (after_asia & (post_high > a_hi)).astype(float)
    f["swept_low"] = (after_asia & (post_low < a_lo)).astype(float)
    f["rejected_high"] = ((f["swept_high"] > 0) & (close < a_hi)).astype(float)
    f["rejected_low"] = ((f["swept_low"] > 0) & (close > a_lo)).astype(float)
    f.loc[a_hi.isna(), ASIA] = np.nan
    return f


def make_labels(bars: pd.DataFrame) -> pd.DataFrame:
    """For a decision at bar t's close the engine buys or sells at bar t+1's open and the next decision can
    change the position at bar t+2's open, so the return earned is open[t+2] - open[t+1]. Ties are excluded
    from fitting and scoring but still traded through in backtests."""
    o = bars["open"].to_numpy()
    ret = np.full(len(o), np.nan)
    ret[:-2] = o[2:] - o[1:-1]
    y = np.where(ret > 0, 1.0, np.where(ret < 0, 0.0, np.nan))
    ends = bars["timestamp"].shift(-2)  # the label is not known until this bar opens
    return pd.DataFrame({"trade_return": ret, "y": y, "label_ends": ends}, index=bars.index)


def period_mask(bars: pd.DataFrame, labels: pd.DataFrame, start: str, end: str, purge: bool) -> pd.Series:
    """Rows whose decision falls in [start, end). With purge, also drop rows whose label reaches past `end`,
    so no training label uses prices from the next period."""
    ts = bars["timestamp"]
    m = (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))
    if purge:
        m &= labels["label_ends"] < pd.Timestamp(end)
    return m


# ---------------------------------------------------------------- evaluation helpers
def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"rows": int(len(y)), "up_share": float(y.mean()), "accuracy": float(accuracy_score(y, p > 0.5)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p > 0.5)), "auc": float(roc_auc_score(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "brier": float(brier_score_loss(y, p))}


def bootstrap_by_day(y: np.ndarray, p: np.ndarray, days: np.ndarray, reps: int = BOOTSTRAP_REPS) -> dict:
    """95% intervals for AUC and accuracy, resampling whole days (hours within a day are not independent)."""
    rng = np.random.default_rng(SEED)
    _, inverse = np.unique(days, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    groups = np.split(order, np.cumsum(np.bincount(inverse))[:-1])  # row indices of each day
    aucs, accs = [], []
    for _ in range(reps):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if 0 < y[idx].sum() < len(idx):
            aucs.append(roc_auc_score(y[idx], p[idx]))
            accs.append(accuracy_score(y[idx], p[idx] > 0.5))
    lo, hi = np.percentile(aucs, [2.5, 97.5]), np.percentile(accs, [2.5, 97.5])
    return {"auc_95": [float(lo[0]), float(lo[1])], "accuracy_95": [float(hi[0]), float(hi[1])]}


def policy_targets(p: np.ndarray, threshold: float) -> np.ndarray:
    """Long one lot when P(up) > 0.5 + threshold, short when < 0.5 - threshold, otherwise flat."""
    return np.where(p > 0.5 + threshold, UNITS, np.where(p < 0.5 - threshold, -UNITS, 0.0))


# ---------------------------------------------------------------- the C++ engine
def engine_binary() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    dirs = [Path(os.environ["BT_BUILD_DIR"])] if "BT_BUILD_DIR" in os.environ else []
    for d in dirs + [ENGINE_REPO / "build", ENGINE_REPO / "build" / "Release", ENGINE_REPO / "cmake-build-release"]:
        if (d / f"bt_run{suffix}").exists():
            return d / f"bt_run{suffix}"
    sys.exit(f"bt_run not found under {ENGINE_REPO}: build the engine's Release configuration first")


def backtest(bars: pd.DataFrame, rows: np.ndarray | None, targets: np.ndarray | None, cost: str, start: str,
             end: str, curve_to: Path | None = None, strategy: str = "file") -> dict:
    """Runs the engine on [start, end). With strategy "file", `targets` are positions decided at the closes
    of bars `rows`; otherwise one of the engine's own strategies runs and rows/targets are unused."""
    spread, commission = COST_LEVELS[cost]
    args = [str(engine_binary()), "--data", str(DATA), "--cash", repr(CASH), "--units", repr(UNITS),
            "--basis", "bid", "--spread", repr(spread), "--commission-per-unit", repr(commission),
            "--start", start, "--end", end]
    with tempfile.TemporaryDirectory() as tmp:
        if strategy == "file":
            path = Path(tmp) / "targets.csv"
            pd.DataFrame({"timestamp": bars["timestamp"].iloc[rows].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                          "target": targets}).to_csv(path, index=False)
            args += ["--strategy", "file", "--targets", str(path)]
        else:
            args += ["--strategy", strategy]  # "buyhold", "none" or "ma" (MA(20,50) by default)
        if curve_to is not None:
            args += ["--out", tmp]
        run = subprocess.run(args, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        m = json.loads(run.stdout)
        if curve_to is not None:
            pd.read_csv(Path(tmp) / "equity.csv", usecols=["timestamp", "equity"]).to_csv(curve_to, index=False)
    keep = ["net_pnl", "annual_return_on_capital", "sharpe_pnl_zero_rf", "max_drawdown_on_capital", "fills",
            "exposure", "fees_paid", "spread_slippage_cost", "stopped_out", "years"]
    out = {k: m.get(k) for k in keep}
    out["round_trips_per_year"] = m["fills"] / 2 / m["years"] if m.get("years") else None
    return out


# ---------------------------------------------------------------- experiments
def fit(name: str, cols: list[str], X: pd.DataFrame, y: pd.Series, rows: pd.Series):
    model = make_model(name)
    model.fit(X.loc[rows, cols].to_numpy(), y[rows].to_numpy().astype(int))
    return model


def p_up(model, cols: list[str], X: pd.DataFrame, rows) -> np.ndarray:
    """P(next tradable return > 0) for boolean-mask or positional rows."""
    data = X.loc[rows, cols] if isinstance(rows, pd.Series) else X.iloc[rows][cols]
    return model.predict_proba(data.to_numpy())[:, 1]


def main() -> None:
    t_start = time.time()
    (RESULTS / "curves").mkdir(parents=True, exist_ok=True)
    bars = load_bars()
    X, lab = make_features(bars), make_labels(bars)
    usable = X.notna().all(axis=1)  # warm-up rows (first ~100 bars, no Asia range yet) drop out here
    labelled = usable & lab["y"].notna()
    day = (bars["timestamp"] + pd.Timedelta(hours=SESSION_OFFSET_HOURS)).dt.floor("D").to_numpy()
    rows = {name: period_mask(bars, lab, *PERIODS[name], purge=(name != "test")) for name in PERIODS}
    report: dict = {"setup": {
        "data": DATA.name, "bars": int(len(bars)), "first_bar": str(bars["timestamp"].iloc[0]),
        "last_bar": str(bars["timestamp"].iloc[-1]), "periods": PERIODS, "feature_sets": FEATURE_SETS,
        "label": "sign of open[t+2] - open[t+1]: the one-bar return of a position filled at the next open",
        "rows": {k: int((v & labelled).sum()) for k, v in rows.items()},
        "up_share_train": float(lab.loc[rows["train"] & labelled, "y"].mean()),
        "costs": {k: {"spread": v[0], "commission_per_unit": v[1]} for k, v in COST_LEVELS.items()},
        "baseline_cost": BASELINE_COST, "units": UNITS, "capital": CASH, "seed": SEED,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__}}}

    # A. Models and feature sets: fit on train, score on train and validation.
    train_fit = rows["train"] & labelled
    a_rows, models = [], {}
    for fs, cols in FEATURE_SETS.items():
        for name in MODELS:
            model = fit(name, cols, X, lab["y"], train_fit)
            entry = {"feature_set": fs, "model": name}
            for period in ("train", "validation"):
                m = rows[period] & labelled
                entry[period] = classification_metrics(lab.loc[m, "y"].to_numpy(), p_up(model, cols, X, m))
            vm = rows["validation"] & labelled
            entry["validation"] |= bootstrap_by_day(lab.loc[vm, "y"].to_numpy(), p_up(model, cols, X, vm), day[vm.to_numpy()])
            a_rows.append(entry)
            models[(fs, name)] = model
            print(f"A  {fs:32s} {name:18s} val AUC {entry['validation']['auc']:.4f}  log loss {entry['validation']['log_loss']:.5f}")
    # Simple baselines on validation: a constant probability, and single features used as scores.
    vm = rows["validation"] & labelled
    yv = lab.loc[vm, "y"].to_numpy()
    base_rate = report["setup"]["up_share_train"]
    report["A_baselines_validation"] = {
        "constant train up-share": {"log_loss": float(log_loss(yv, np.full(len(yv), base_rate))),
                                    "accuracy": float(max(yv.mean(), 1 - yv.mean())), "auc": 0.5},
        "momentum score (ret_12)": {"auc": float(roc_auc_score(yv, X.loc[vm, "ret_12"]))},
        "reversal score (-ret_1)": {"auc": float(roc_auc_score(yv, -X.loc[vm, "ret_1"]))},
    }
    report["A_models"] = a_rows

    # Selection rule 1: lowest validation log loss.
    best = min(a_rows, key=lambda r: r["validation"]["log_loss"])
    fs, name = best["feature_set"], best["model"]
    cols, model = FEATURE_SETS[fs], models[(fs, name)]
    vrows = np.flatnonzero((rows["validation"] & usable).to_numpy())  # trade every usable bar, ties included
    p_val = p_up(model, cols, X, vrows)

    # D (validation). Thresholds x costs for the selected model.
    grid = []
    for thr in THRESHOLDS:
        for cost in COST_LEVELS:
            r = backtest(bars, vrows, policy_targets(p_val, thr), cost, *PERIODS["validation"])
            grid.append({"threshold": thr, "cost": cost, **r})
    report["D_validation_grid"] = grid
    eligible = [g for g in grid if g["cost"] == BASELINE_COST and g["fills"] >= MIN_VALIDATION_FILLS]
    pick = max(eligible, key=lambda g: g["sharpe_pnl_zero_rf"])
    selection = {
        "rules": ["model and feature set: lowest validation log loss",
                  f"threshold: highest validation Sharpe at {BASELINE_COST}, among thresholds with at least "
                  f"{MIN_VALIDATION_FILLS} validation fills"],
        "feature_set": fs, "model": name, "threshold": pick["threshold"],
        "validation_log_loss": best["validation"]["log_loss"], "validation_auc": best["validation"]["auc"],
        "validation_sharpe": pick["sharpe_pnl_zero_rf"], "validation_net_pnl": pick["net_pnl"],
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (RESULTS / "selection.json").write_text(json.dumps(selection, indent=2))  # fixed before any test-period run
    report["selection"] = selection
    print("selected:", selection)

    # C. The test period, once, for the selected policy and the baselines.
    trows = np.flatnonzero((rows["test"] & usable).to_numpy())
    p_test = p_up(model, cols, X, trows)
    test_period = PERIODS["test"]
    momentum = np.where(X["ret_12"].iloc[trows].to_numpy() > 0, UNITS, -UNITS)  # long after a rising 12 hours
    curves = RESULTS / "curves"
    report["C_test"] = [
        {"strategy": f"{name}, {fs}, threshold {pick['threshold']}", "role": "selected on validation",
         **backtest(bars, trows, policy_targets(p_test, pick["threshold"]), BASELINE_COST, *test_period,
                    curve_to=curves / "test_selected.csv")},
        {"strategy": "12-hour momentum rule", "role": "simple rule",
         **backtest(bars, trows, momentum, BASELINE_COST, *test_period, curve_to=curves / "test_momentum.csv")},
        {"strategy": "MA(20,50) crossover", "role": "engine default",
         **backtest(bars, None, None, BASELINE_COST, *test_period, curve_to=curves / "test_ma.csv", strategy="ma")},
        {"strategy": "Buy and hold", "role": "benchmark",
         **backtest(bars, None, None, BASELINE_COST, *test_period, curve_to=curves / "test_buyhold.csv",
                    strategy="buyhold")},
    ]
    tm = rows["test"] & labelled
    yt = lab.loc[tm, "y"].to_numpy()
    pt = p_up(model, cols, X, tm)
    report["C_test_classification"] = classification_metrics(yt, pt) | bootstrap_by_day(yt, pt, day[tm.to_numpy()])
    print("C  test:", report["C_test"][0])

    # After the selection, for diagnosis only: every model on the test period, and thresholds x costs there.
    report["C_test_all_models_after_selection"] = [
        {"feature_set": f, "model": n, **classification_metrics(yt, p_up(models[(f, n)], FEATURE_SETS[f], X, tm))}
        for f in FEATURE_SETS for n in MODELS]
    report["D_test_grid_after_selection"] = [
        {"threshold": thr, "cost": cost, **backtest(bars, trows, policy_targets(p_test, thr), cost, *test_period)}
        for thr in THRESHOLDS for cost in (BASELINE_COST, "no costs")]

    # Walk-forward: retrain both models on all earlier data each year, predict the year.
    wf = []
    for year in WALK_FORWARD_YEARS:
        start, end = f"{year}-01-01", f"{year + 1}-01-01"
        train_rows = period_mask(bars, lab, PERIODS["train"][0], start, purge=True) & labelled
        year_rows = period_mask(bars, lab, start, end, purge=False)
        for n in MODELS:
            m_y = fit(n, cols, X, lab["y"], train_rows)
            ym = year_rows & labelled
            py = p_up(m_y, cols, X, ym)
            yrows = np.flatnonzero((year_rows & usable).to_numpy())
            p_all = p_up(m_y, cols, X, yrows)
            wf.append({"year": year, "model": n, **classification_metrics(lab.loc[ym, "y"].to_numpy(), py),
                       "average_hourly_move_pips": float(lab.loc[year_rows, "trade_return"].abs().mean() / PIP),
                       "policy": backtest(bars, yrows, policy_targets(p_all, pick["threshold"]), BASELINE_COST, start, end),
                       "policy_no_costs": backtest(bars, yrows, policy_targets(p_all, pick["threshold"]), "no costs", start, end)})
        print(f"WF {year}: " + "  ".join(f"{w['model']} AUC {w['auc']:.3f}" for w in wf[-2:]))
    report["walk_forward"] = wf
    report["runtime_seconds"] = round(time.time() - t_start, 1)
    (RESULTS / "experiments.json").write_text(json.dumps(report, indent=2, default=float))
    print("wrote", RESULTS / "experiments.json", f"in {report['runtime_seconds']} s")


if __name__ == "__main__":
    main()
