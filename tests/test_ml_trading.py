"""Leakage, labelling and policy checks. These run on synthetic hourly bars shaped like HistData (EST, no
weekend hours), so they need no downloaded data; the engine test is skipped if bt_run has not been built."""
import numpy as np
import pandas as pd
import pytest

import ml_trading as ml


def synthetic_bars(n_hours: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2021-01-03 17:00", periods=n_hours, freq="1h")  # starts at the Sunday 17:00 open
    weekend = ((ts.dayofweek == 4) & (ts.hour >= 17)) | (ts.dayofweek == 5) | ((ts.dayofweek == 6) & (ts.hour < 17))
    ts = ts[~weekend]
    m = len(ts)
    close = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.001, m)))
    open_ = np.concatenate([[1.1], close[:-1]]) * (1 + rng.normal(0, 0.0001, m))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0005, m)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0005, m)))
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close})


def test_features_never_use_later_bars():
    bars = synthetic_bars()
    full = ml.make_features(bars)
    assert full.notna().all(axis=1).sum() > 2000  # the check below must cover real values, not just warm-up
    for cut in [130, 777, 1500, 2222, len(bars) - 2]:
        part = ml.make_features(bars.iloc[: cut + 1].copy())
        pd.testing.assert_frame_equal(part, full.iloc[: cut + 1], check_exact=False, rtol=0, atol=1e-12)


def test_features_ignore_changes_to_later_bars():
    bars = synthetic_bars()
    shocked = bars.copy()
    shocked.loc[1800:, ["open", "high", "low", "close"]] *= 1.5  # a huge move after row 1799
    pd.testing.assert_frame_equal(ml.make_features(shocked).iloc[:1800], ml.make_features(bars).iloc[:1800])


def test_asia_range_is_the_last_session_completed_by_each_close():
    bars = synthetic_bars()
    hour, day = ml.session_clock(bars)
    levels = ml.asia_levels(bars)
    asia = bars[hour.between(2, 9)].groupby(day[hour.between(2, 9)])
    highs, lows, counts = asia["high"].max(), asia["low"].min(), asia["high"].size()
    rng = np.random.default_rng(1)
    for i in rng.integers(200, len(bars), 300):
        # Brute force: the latest session day whose hours 2-9 had all closed by this bar's close.
        bar_close = bars["timestamp"].iloc[i] + pd.Timedelta(hours=8)  # session time at the bar's close
        done = [d for d in highs.index if counts[d] >= 6 and d + pd.Timedelta(hours=10) <= bar_close]
        if not done:
            assert np.isnan(levels["asia_high"].iloc[i])
            continue
        d = max(done)
        assert levels["asia_high"].iloc[i] == highs[d] and levels["asia_low"].iloc[i] == lows[d]
        if 2 <= hour.iloc[i] <= 8:
            assert d < day.iloc[i]  # mid-Asia bars must not see the range that is still forming
        if hour.iloc[i] == 9:
            assert d == day.iloc[i]  # at the close of hour 9 today's range has just completed


def test_sweep_flags_match_a_brute_force_scan():
    bars = synthetic_bars()
    f = ml.make_features(bars)
    hour, day = ml.session_clock(bars)
    levels = ml.asia_levels(bars)
    checked = 0
    for i in range(300, len(bars)):
        if hour.iloc[i] <= 9 or levels["asia_day"].iloc[i] != day.iloc[i] or np.isnan(f["asia_pos"].iloc[i]):
            continue
        since = bars.iloc[: i + 1][(day.iloc[: i + 1] == day.iloc[i]) & (hour.iloc[: i + 1] >= 10)]
        swept_high = since["high"].max() > levels["asia_high"].iloc[i]
        assert f["swept_high"].iloc[i] == float(swept_high)
        assert f["rejected_high"].iloc[i] == float(swept_high and bars["close"].iloc[i] < levels["asia_high"].iloc[i])
        swept_low = since["low"].min() < levels["asia_low"].iloc[i]
        assert f["swept_low"].iloc[i] == float(swept_low)
        checked += 1
    assert checked > 500


def test_label_is_the_return_of_a_position_filled_at_the_next_open():
    bars = synthetic_bars(200)
    lab = ml.make_labels(bars)
    o = bars["open"].to_numpy()
    for t in range(len(bars) - 2):
        assert lab["trade_return"].iloc[t] == pytest.approx(o[t + 2] - o[t + 1], abs=0)
        assert lab["y"].iloc[t] == (1.0 if o[t + 2] > o[t + 1] else 0.0)
    assert lab["y"].iloc[-2:].isna().all()  # the last two decisions have no outcome yet


def test_purge_drops_rows_whose_labels_reach_the_next_period():
    bars = synthetic_bars(1000)
    lab = ml.make_labels(bars)
    end = str(bars["timestamp"].iloc[600])
    kept = ml.period_mask(bars, lab, str(bars["timestamp"].iloc[0]), end, purge=True)
    unpurged = ml.period_mask(bars, lab, str(bars["timestamp"].iloc[0]), end, purge=False)
    assert unpurged.sum() == 600 and kept.sum() == 598  # rows 598 and 599 need opens 600 and 601
    assert (lab.loc[kept, "label_ends"] < pd.Timestamp(end)).all()


def test_policy_goes_flat_inside_the_threshold():
    p = np.array([0.40, 0.48, 0.495, 0.5, 0.505, 0.52, 0.60])
    assert list(ml.policy_targets(p, 0.0)) == [-ml.UNITS] * 3 + [0.0] + [ml.UNITS] * 3
    assert list(ml.policy_targets(p, 0.01)) == [-ml.UNITS, -ml.UNITS, 0, 0, 0, ml.UNITS, ml.UNITS]


def test_models_are_reproducible():
    bars = synthetic_bars()
    X, lab = ml.make_features(bars), ml.make_labels(bars)
    rows = X.notna().all(axis=1) & lab["y"].notna()
    cols = ml.FEATURE_SETS["price + calendar + Asia range"]
    a = ml.p_up(ml.fit("gradient boosting", cols, X, lab["y"], rows), cols, X, rows)
    b = ml.p_up(ml.fit("gradient boosting", cols, X, lab["y"], rows), cols, X, rows)
    np.testing.assert_array_equal(a, b)


def engine_available() -> bool:
    try:
        ml.engine_binary()
        return True
    except SystemExit:
        return False


@pytest.mark.skipif(not engine_available(), reason="cpp-backtesting-engine not built")
def test_engine_replays_targets_like_its_own_strategies(tmp_path, monkeypatch):
    bars = synthetic_bars(800)  # weekend hours are dropped, so fewer than 800 bars remain
    data = tmp_path / "bars.csv"
    out = bars.assign(timestamp=bars["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    out.to_csv(data, index=False, float_format="%.6f")
    monkeypatch.setattr(ml, "DATA", data)
    start, end = str(bars["timestamp"].iloc[100]), str(bars["timestamp"].iloc[len(bars) - 50])
    rows = np.arange(len(bars))
    always_long = ml.backtest(bars, rows, np.full(len(bars), ml.UNITS), ml.BASELINE_COST, start, end)
    hold = ml.backtest(bars, None, None, ml.BASELINE_COST, start, end, strategy="buyhold")
    assert always_long["net_pnl"] == pytest.approx(hold["net_pnl"], abs=1e-6)
    assert always_long["fills"] == hold["fills"] == 1
