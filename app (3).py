"""
================================================================================
 BINARY OPTIONS AI AGENT v4 - INSTITUTIONAL-GRADE PRECISION ENGINE
================================================================================
Run with: streamlit run app.py

Everything from v3 is preserved:
  - Dow Theory market structure (Uptrend / Downtrend / Sideways)
  - Dynamic trendline engine (least-squares fit through swing points)
  - Multi-timeframe Horizontal S/R (Daily, 4H, 30T, 10T, 5T)
  - Adaptive institutional Round Numbers
  - 7 candlestick reversal patterns (incl. Inverted Hammer)
  - RSI(14) momentum filter (oversold/overbought gating)
  - Volume confirmation filter (with auto-disable if feed volume is unusable)
  - 5m + 15m MTF trend alignment (structure OR 20-EMA)
  - Strict wick:body ratio (Hammer / Shooting Star / Inverted Hammer)
  - AI Market Grading + auto-lock (rolling win rate of last 20 signals)
  - ATR volatility regime filter (too-quiet / news-spike rejection)
  - Dual 1-min / 3-min expiry backtest, side-by-side

NEW in v4 (macro + chop-squelch layer):
  1. 200 EMA Macro Trend Filter (Multi-Timeframe) - fetches the SAME 5-min and
     15-min data already used for MTF alignment, and additionally computes a
     200-period EMA on each. A CALL is only valid if price is STRICTLY ABOVE
     the 200 EMA on BOTH the 5m and 15m charts. A PUT is only valid if price
     is STRICTLY BELOW the 200 EMA on BOTH. This is a harder, independent
     check than the existing structure/20-EMA MTF filter - both must agree.
  2. Bollinger Band (20, 2) Width Squeeze Filter - standard Bollinger Bands on
     the 1-minute chart. Bandwidth = (Upper - Lower) / Middle. When bandwidth
     falls below a configurable threshold, the market is read as a tight,
     choppy squeeze and the engine pauses new signals, displaying:
     "Market in tight squeeze (Chop/Sideways) - Signals Paused."

HONESTY NOTE (read before trusting any number this app shows you):
A rule-based 1-minute retail strategy sustaining a *consistent* 85%+ real
win rate across *all* currency pairs is an extraordinary claim that no
publicly verifiable strategy reliably clears, once broker payout structure,
spread, and the sheer noise of 1-minute price action are accounted for.
Adding more filters (as this version does) trades signal frequency for
signal quality on the signals that DO fire - it cannot guarantee any
specific number, and may produce very few or zero signals on quiet
pairs/sessions, which this dashboard will report honestly rather than
fabricate. Treat every number here as a hypothesis to validate further
(ideally on a demo account, across many pairs and weeks), not a guarantee.
================================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


# ==============================================================================
# 1. CONFIG
# ==============================================================================
@dataclass
class Config:
    # structure / trendline
    pivot_left: int = 3
    pivot_right: int = 3
    trend_lookback: int = 150
    trendline_points: int = 3
    # volatility unit
    atr_period: int = 14
    touch_tolerance_atr_mult: float = 0.25
    round_tolerance_atr_mult: float = 0.20
    marubozu_body_ratio: float = 0.90
    breakout_atr_mult: float = 0.15
    breakout_suppress_bars: int = 5
    # RSI filter
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    # volume filter
    volume_lookback: int = 2
    require_volume_confirmation: bool = True
    # NEW: strict wick:body ratio
    wick_body_ratio: float = 2.5
    # NEW: MTF alignment
    mtf_ema_period: int = 20
    mtf_trend_lookback: int = 60
    # NEW: ATR volatility regime
    atr_regime_window: int = 100
    atr_low_pct: float = 0.10
    atr_high_pct: float = 0.90
    # NEW: AI grading
    ai_grade_window: int = 20
    ai_grade_excellent: float = 75.0
    ai_grade_moderate: float = 60.0
    ai_grade_min_signals: int = 5
    # NEW: 200 EMA macro trend filter (5m & 15m)
    macro_ema_period: int = 200
    # NEW: Bollinger Band (20, 2) squeeze filter (1-minute)
    bb_period: int = 20
    bb_std_mult: float = 2.0
    bb_squeeze_threshold: float = 0.0008
    # execution
    expiries: Tuple[int, int] = (1, 3) # minutes


# ==============================================================================
# 2. INDICATORS
# ==============================================================================
def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean().bfill()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi.bfill().clip(0, 100)


def compute_atr_percentile(atr: pd.Series, window: int) -> pd.Series:
    """Where does TODAY's ATR rank versus the trailing `window` bars?
    0.0 = quietest bar in the window, 1.0 = most volatile. Used to detect
    both dead/choppy conditions (very low) and news-spike conditions
    (very high)."""
    return atr.rolling(window, min_periods=20).apply(lambda s: s.rank(pct=True).iloc[-1], raw=False)


def compute_bollinger_bands(close: pd.Series, period: int, std_mult: float) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Standard Bollinger Bands: middle = SMA(period), bands = middle +/- std_mult * stdev(period).
    Bandwidth = (upper - lower) / middle -- a scale-free measure of how
    'squeezed'price action currently is. A very low bandwidth means price is
    coiling in a tight range (chop), which is exactly the regime where 1-minute
    reversal signals are least reliable."""
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid
    return mid, upper, lower, width


def detect_pivots(df: pd.DataFrame, left: int, right: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    piv_high = np.zeros(n, dtype=bool)
    piv_low = np.zeros(n, dtype=bool)
    highs = df["high"].values
    lows = df["low"].values
    for i in range(left, n - right):
        wh = highs[i - left:i + right + 1]
        wl = lows[i - left:i + right + 1]
        if highs[i] == wh.max() and np.argmax(wh) == left:
            piv_high[i] = True
        if lows[i] == wl.min() and np.argmin(wl) == left:
            piv_low[i] = True
    return piv_high, piv_low


def classify_trend(hi_vals: List[float], lo_vals: List[float]) -> str:
    if len(hi_vals) >= 2 and len(lo_vals) >= 2:
        hh = hi_vals[-1] > hi_vals[-2]
        hl = lo_vals[-1] > lo_vals[-2]
        ll = lo_vals[-1] < lo_vals[-2]
        lh = hi_vals[-1] < hi_vals[-2]
        if hh and hl:
            return "UPTREND"
        if ll and lh:
            return "DOWNTREND"
    return "SIDEWAYS"


# ==============================================================================
# 3. DYNAMIC TRENDLINE
# ==============================================================================
def fit_trendline(x_points: List[int], y_points: List[float]) -> Tuple[float, float]:
    x = np.asarray(x_points, dtype=float)
    y = np.asarray(y_points, dtype=float)
    if len(x) == 1:
        return 0.0, y[0]
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def trendline_value_at(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept


# ==============================================================================
# 4. CANDLESTICK PATTERN LIBRARY (context-aware: SUPPORT vs RESISTANCE)
# ==============================================================================
def _ohlc(df, i):
    return df["open"].iat[i], df["high"].iat[i], df["low"].iat[i], df["close"].iat[i]


def _body(o, c):
    return abs(c - o)


def _range(h, l):
    return max(h - l, 1e-12)


def is_bull(o, c):
    return c > o


def is_bear(o, c):
    return c < o


def is_marubozu(df, i, bullish, ratio=0.90):
    o, h, l, c = _ohlc(df, i)
    if _body(o, c) / _range(h, l) < ratio:
        return False
    return is_bull(o, c) if bullish else is_bear(o, c)


def is_hammer_shape(df, i, ratio: float) -> bool:
    """Small body near the TOP of the range, lower wick >= ratio*body, tiny upper wick."""
    o, h, l, c = _ohlc(df, i)
    rng, body = _range(h, l), _body(o, c)
    upper, lower = h - max(o, c), min(o, c) - l
    return body / rng <= 0.35 and lower >= ratio * body and upper <= 0.15 * rng


def is_long_upper_wick_shape(df, i, ratio: float) -> bool:
    """Small body near the BOTTOM of the range, upper wick >= ratio*body, tiny lower wick.
    Interpreted as Inverted Hammer (bullish) at support, or Shooting Star (bearish) at resistance."""
    o, h, l, c = _ohlc(df, i)
    rng, body = _range(h, l), _body(o, c)
    upper, lower = h - max(o, c), min(o, c) - l
    return body / rng <= 0.35 and upper >= ratio * body and lower <= 0.15 * rng


def is_bullish_engulfing(df, i):
    if i < 1:
        return False
    o1, h1, l1, c1 = _ohlc(df, i - 1)
    o2, h2, l2, c2 = _ohlc(df, i)
    return is_bear(o1, c1) and is_bull(o2, c2) and c2 >= o1 and o2 <= c1


def is_bearish_engulfing(df, i):
    if i < 1:
        return False
    o1, h1, l1, c1 = _ohlc(df, i - 1)
    o2, h2, l2, c2 = _ohlc(df, i)
    return is_bull(o1, c1) and is_bear(o2, c2) and c2 <= o1 and o2 >= c1


def is_bullish_harami(df, i):
    if i < 1:
        return False
    o1, h1, l1, c1 = _ohlc(df, i - 1)
    o2, h2, l2, c2 = _ohlc(df, i)
    return is_bear(o1, c1) and is_bull(o2, c2) and o2 > c1 and c2 < o1 and _body(o2, c2) < _body(o1, c1)


def is_bearish_harami(df, i):
    if i < 1:
        return False
    o1, h1, l1, c1 = _ohlc(df, i - 1)
    o2, h2, l2, c2 = _ohlc(df, i)
    return is_bull(o1, c1) and is_bear(o2, c2) and o2 < c1 and c2 > o1 and _body(o2, c2) < _body(o1, c1)


def is_morning_star(df, i):
    if i < 2:
        return False
    o1, h1, l1, c1 = _ohlc(df, i - 2)
    o2, h2, l2, c2 = _ohlc(df, i - 1)
    o3, h3, l3, c3 = _ohlc(df, i)
    rng1, rng2 = _range(h1, l1), _range(h2, l2)
    mid1 = (o1 + c1) / 2
    return (is_bear(o1, c1) and _body(o1, c1) / rng1 > 0.5
            and _body(o2, c2) / rng2 <= 0.25
            and is_bull(o3, c3) and c3 > mid1)


def is_evening_star(df, i):
    if i < 2:
        return False
    o1, h1, l1, c1 = _ohlc(df, i - 2)
    o2, h2, l2, c2 = _ohlc(df, i - 1)
    o3, h3, l3, c3 = _ohlc(df, i)
    rng1, rng2 = _range(h1, l1), _range(h2, l2)
    mid1 = (o1 + c1) / 2
    return (is_bull(o1, c1) and _body(o1, c1) / rng1 > 0.5
            and _body(o2, c2) / rng2 <= 0.25
            and is_bear(o3, c3) and c3 < mid1)


def detect_pattern(df: pd.DataFrame, i: int, context: str, wick_ratio: float) -> Optional[str]:
    """context='SUPPORT' -> only bullish reversal candidates are checked.
    context='RESISTANCE' -> only bearish reversal candidates are checked.
    This is what makes the wick-ratio rule unambiguous: the same shape is
    never silently treated as both a bullish AND bearish signal."""
    if context == "SUPPORT":
        if is_morning_star(df, i):
            return "MORNING_STAR"
        if is_bullish_engulfing(df, i):
            return "BULLISH_ENGULFING"
        if is_bullish_harami(df, i):
            return "BULLISH_HARAMI"
        if is_hammer_shape(df, i, wick_ratio):
            return "HAMMER"
        if is_long_upper_wick_shape(df, i, wick_ratio):
            return "INVERTED_HAMMER"
        if is_marubozu(df, i, bullish=True):
            return "BULLISH_MARUBOZU"
        return None
    elif context == "RESISTANCE":
        if is_evening_star(df, i):
            return "EVENING_STAR"
        if is_bearish_engulfing(df, i):
            return "BEARISH_ENGULFING"
        if is_bearish_harami(df, i):
            return "BEARISH_HARAMI"
        if is_long_upper_wick_shape(df, i, wick_ratio):
            return "SHOOTING_STAR"
        if is_marubozu(df, i, bullish=False):
            return "BEARISH_MARUBOZU"
        return None
    return None


# ==============================================================================
# 5. MULTI-TIMEFRAME HORIZONTAL S/R + ADAPTIVE ROUND NUMBERS (unchanged)
# ==============================================================================
def build_mtf_levels(df: pd.DataFrame) -> pd.DataFrame:
    timeframe_rules = {"D": "1D", "4H": "4h", "30T": "30min", "10T": "10min", "5T": "5min"}
    out = {}
    for name, rule in timeframe_rules.items():
        agg = df.resample(rule, label="right", closed="left").agg({"high": "max", "low": "min"})
        agg = agg.shift(1)
        out[f"{name}_high"] = agg["high"].reindex(df.index, method="ffill")
        out[f"{name}_low"] = agg["low"].reindex(df.index, method="ffill")
    return pd.DataFrame(out, index=df.index)


def get_round_steps(price: float) -> Tuple[float, float, float]:
    if price >= 20:
        return (1.00, 0.50, 0.10)
    elif price >= 2:
        return (0.10, 0.05, 0.01)
    else:
        return (0.0100, 0.0050, 0.0010)


def get_round_number_zones(price: float, steps: Tuple[float, float, float]) -> List[float]:
    zones = []
    for step in steps:
        lower = np.floor(price / step) * step
        zones.extend([lower, lower + step])
    return sorted(set(round(z, 6) for z in zones))


# ==============================================================================
# 6. VOLUME CONFIRMATION FILTER (unchanged)
# ==============================================================================
def volume_confirmed(df: pd.DataFrame, i: int, lookback: int, volume_reliable: bool) -> bool:
    if not volume_reliable:
        return True
    if i - lookback < 0:
        return False
    avg_prev = df["volume"].iloc[i - lookback:i].mean()
    return df["volume"].iat[i] > avg_prev


# ==============================================================================
# 7. NEW: MULTI-TIMEFRAME TREND BIAS (5m & 15m, structure OR EMA20)
# ==============================================================================
def compute_tf_bias(df_tf: pd.DataFrame, cfg: Config) -> pd.Series:
    """For a higher-timeframe dataframe: bias = BULLISH if its own swing
    structure is an Uptrend OR price is above its 20-EMA; BEARISH mirrors
    this; otherwise NEUTRAL. Pivot confirmation is lag-delayed exactly like
    the 1-min engine, so this stays causal."""
    ema = df_tf["close"].ewm(span=cfg.mtf_ema_period, adjust=False).mean()
    piv_h, piv_l = detect_pivots(df_tf, cfg.pivot_left, cfg.pivot_right)
    n = len(df_tf)
    ch_idx, ch_val, cl_idx, cl_val = [], [], [], []
    bias = []
    for i in range(n):
        confirm_idx = i - cfg.pivot_right
        if confirm_idx >= 0:
            if piv_h[confirm_idx]:
                ch_idx.append(confirm_idx); ch_val.append(df_tf["high"].iat[confirm_idx])
            if piv_l[confirm_idx]:
                cl_idx.append(confirm_idx); cl_val.append(df_tf["low"].iat[confirm_idx])
        lo_cut = i - cfg.mtf_trend_lookback
        rh_val = [v for x, v in zip(ch_idx, ch_val) if x > lo_cut]
        rl_val = [v for x, v in zip(cl_idx, cl_val) if x > lo_cut]
        structure = classify_trend(rh_val, rl_val)
        price, ema_i = df_tf["close"].iat[i], ema.iat[i]
        if structure == "UPTREND"or price > ema_i:
            bias.append("BULLISH")
        elif structure == "DOWNTREND"or price < ema_i:
            bias.append("BEARISH")
        else:
            bias.append("NEUTRAL")
    return pd.Series(bias, index=df_tf.index, name="bias")


def align_bias_to_1m(bias_tf: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Shift by 1 higher-TF bar (use only the PREVIOUS closed bar's bias --
    no look-ahead) then forward-fill onto the 1-minute timeline."""
    shifted = bias_tf.shift(1)
    aligned = shifted.reindex(target_index, method="ffill")
    return aligned.fillna("NEUTRAL")


# ==============================================================================
# 7b. NEW: 200 EMA MACRO TREND FILTER (5m & 15m)
# ==============================================================================
def compute_macro_bias(df_tf: pd.DataFrame, period: int) -> pd.Series:
    """The 'big picture'institutional trend filter: price strictly ABOVE its
    own 200-EMA = macro BULLISH, strictly BELOW = macro BEARISH, equal/NaN
    (not enough history yet) = NEUTRAL. min_periods=period means this stays
    NEUTRAL (i.e. blocks trades) until a genuine 200-bar EMA has formed --
    no using an unstable, still-warming-up average to gate real trades."""
    ema200 = df_tf["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    price = df_tf["close"]
    bias = np.where(price > ema200, "BULLISH", np.where(price < ema200, "BEARISH", "NEUTRAL"))
    bias = pd.Series(bias, index=df_tf.index, name="macro_bias")
    bias[ema200.isna()] = "NEUTRAL"
    return bias


# ==============================================================================
# 8. NEW: AI MARKET GRADING (rolling win rate of last N graded 1-min signals)
# ==============================================================================
def compute_ai_grade(trades_1min: list, cfg: Config) -> dict:
    n_avail = len(trades_1min)
    if n_avail < cfg.ai_grade_min_signals:
        return {"status": "INSUFFICIENT_DATA", "win_rate": None, "n": n_avail, "locked": False}

    recent = trades_1min[-cfg.ai_grade_window:]
    wins = sum(1 for t in recent if t["result"] == "WIN")
    win_rate = round(100 * wins / len(recent), 2)

    if win_rate >= cfg.ai_grade_excellent:
        return {"status": "EXCELLENT", "win_rate": win_rate, "n": len(recent), "locked": False,
                "label": "EXCELLENT MARKET - SAFE TO TRADING"}
    elif win_rate >= cfg.ai_grade_moderate:
        return {"status": "MODERATE", "win_rate": win_rate, "n": len(recent), "locked": False,
                "label": "MODERATE MARKET - PROCEED WITH CAUTION"}
    else:
        return {"status": "BAD", "win_rate": win_rate, "n": len(recent), "locked": True,
                "label": "BAD MARKET CONDITIONS - SIGNAL ENGINE LOCKED"}


# ==============================================================================
# 9. CORE ENGINE
# ==============================================================================
def run_engine(df1m: pd.DataFrame, df5m: pd.DataFrame, df15m: pd.DataFrame, cfg: Config):
    df = df1m.copy()
    df["atr"] = compute_atr(df, cfg.atr_period)
    df["rsi"] = compute_rsi(df["close"], cfg.rsi_period)
    df["atr_pct"] = compute_atr_percentile(df["atr"], cfg.atr_regime_window)

    piv_high, piv_low = detect_pivots(df, cfg.pivot_left, cfg.pivot_right)
    mtf_levels = build_mtf_levels(df)
    volume_reliable = cfg.require_volume_confirmation and (df["volume"].fillna(0).sum() > 0)

    bias_5m_raw = compute_tf_bias(df5m, cfg)
    bias_15m_raw = compute_tf_bias(df15m, cfg)
    bias_5m = align_bias_to_1m(bias_5m_raw, df.index)
    bias_15m = align_bias_to_1m(bias_15m_raw, df.index)

    macro_5m_raw = compute_macro_bias(df5m, cfg.macro_ema_period)
    macro_15m_raw = compute_macro_bias(df15m, cfg.macro_ema_period)
    macro_5m = align_bias_to_1m(macro_5m_raw, df.index)
    macro_15m = align_bias_to_1m(macro_15m_raw, df.index)

    bb_mid, bb_upper, bb_lower, bb_width = compute_bollinger_bands(df["close"], cfg.bb_period, cfg.bb_std_mult)
    df["bb_width"] = bb_width

    confirmed_high_idx, confirmed_high_val = [], []
    confirmed_low_idx, confirmed_low_val = [], []
    breakout_block_until = -1
    trades = {exp: [] for exp in cfg.expiries}
    live_status = None

    n = len(df)
    warmup = max(cfg.pivot_left, cfg.atr_period, cfg.rsi_period, cfg.atr_regime_window // 2)
    if n <= warmup + 1:
        return trades, None, volume_reliable

    for i in range(warmup, n):
        confirm_idx = i - cfg.pivot_right
        if confirm_idx >= 0:
            if piv_high[confirm_idx]:
                confirmed_high_idx.append(confirm_idx)
                confirmed_high_val.append(df["high"].iat[confirm_idx])
            if piv_low[confirm_idx]:
                confirmed_low_idx.append(confirm_idx)
                confirmed_low_val.append(df["low"].iat[confirm_idx])

        lo_cut = i - cfg.trend_lookback
        rh_idx = [x for x in confirmed_high_idx if x > lo_cut]
        rh_val = [v for x, v in zip(confirmed_high_idx, confirmed_high_val) if x > lo_cut]
        rl_idx = [x for x in confirmed_low_idx if x > lo_cut]
        rl_val = [v for x, v in zip(confirmed_low_idx, confirmed_low_val) if x > lo_cut]

        trend_state = classify_trend(rh_val, rl_val)
        rsi_i = df["rsi"].iat[i]
        atr_pct_i = df["atr_pct"].iat[i]
        volatility_ok = bool(np.isfinite(atr_pct_i) and (cfg.atr_low_pct <= atr_pct_i <= cfg.atr_high_pct))

        bull_aligned = (bias_5m.iat[i] == "BULLISH") and (bias_15m.iat[i] == "BULLISH")
        bear_aligned = (bias_5m.iat[i] == "BEARISH") and (bias_15m.iat[i] == "BEARISH")

        macro_bull_aligned = (macro_5m.iat[i] == "BULLISH") and (macro_15m.iat[i] == "BULLISH")
        macro_bear_aligned = (macro_5m.iat[i] == "BEARISH") and (macro_15m.iat[i] == "BEARISH")

        bb_width_i = df["bb_width"].iat[i]
        squeeze_ok = bool(np.isfinite(bb_width_i) and bb_width_i >= cfg.bb_squeeze_threshold)

        final_bull_ok = bull_aligned and macro_bull_aligned
        final_bear_ok = bear_aligned and macro_bear_aligned

        atr_i = df["atr"].iat[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            atr_i = df["close"].iat[i] * 0.0005
        touch_tol = cfg.touch_tolerance_atr_mult * atr_i
        round_tol = cfg.round_tolerance_atr_mult * atr_i

        signal = category = pattern_name = None
        diag = {"trend_state": trend_state, "price": df["close"].iat[i], "rsi": rsi_i,
                "nearest_level": None, "level_type": None, "distance": None,
                "pattern_forming": None, "volume_ok": None,
                "mtf_5m": bias_5m.iat[i], "mtf_15m": bias_15m.iat[i],
                "mtf_bull_aligned": bull_aligned, "mtf_bear_aligned": bear_aligned,
                "macro_5m": macro_5m.iat[i], "macro_15m": macro_15m.iat[i],
                "macro_bull_aligned": macro_bull_aligned, "macro_bear_aligned": macro_bear_aligned,
                "atr_percentile": atr_pct_i, "volatility_ok": volatility_ok,
                "bb_width": bb_width_i, "squeeze_ok": squeeze_ok}

        vol_ok = volume_confirmed(df, i, cfg.volume_lookback, volume_reliable)

        # ---------------- UPTREND: ascending trendline through Higher Lows ----
        if trend_state == "UPTREND"and len(rl_idx) >= 2 and i > breakout_block_until:
            pts_idx, pts_val = rl_idx[-cfg.trendline_points:], rl_val[-cfg.trendline_points:]
            slope, intercept = fit_trendline(pts_idx, pts_val)
            line_val = trendline_value_at(slope, intercept, i)
            diag.update(nearest_level=line_val, level_type="Ascending Trendline (support)",
                        distance=df["close"].iat[i] - line_val)
            touched = (df["low"].iat[i] <= line_val + touch_tol) and (df["close"].iat[i] >= line_val - touch_tol)
            pattern = detect_pattern(df, i, "SUPPORT", cfg.wick_body_ratio)
            diag["pattern_forming"], diag["volume_ok"] = pattern, vol_ok
            if (touched and pattern is not None and rsi_i <= cfg.rsi_oversold and vol_ok
                    and final_bull_ok and volatility_ok and squeeze_ok):
                signal, category, pattern_name = "BUY", "TRENDLINE", pattern
            if is_marubozu(df, i, bullish=False) and df["close"].iat[i] < line_val - cfg.breakout_atr_mult * atr_i:
                breakout_block_until = i + cfg.breakout_suppress_bars

        # --------------- DOWNTREND: descending trendline through Lower Highs --
        elif trend_state == "DOWNTREND"and len(rh_idx) >= 2 and i > breakout_block_until:
            pts_idx, pts_val = rh_idx[-cfg.trendline_points:], rh_val[-cfg.trendline_points:]
            slope, intercept = fit_trendline(pts_idx, pts_val)
            line_val = trendline_value_at(slope, intercept, i)
            diag.update(nearest_level=line_val, level_type="Descending Trendline (resistance)",
                        distance=df["close"].iat[i] - line_val)
            touched = (df["high"].iat[i] >= line_val - touch_tol) and (df["close"].iat[i] <= line_val + touch_tol)
            pattern = detect_pattern(df, i, "RESISTANCE", cfg.wick_body_ratio)
            diag["pattern_forming"], diag["volume_ok"] = pattern, vol_ok
            if (touched and pattern is not None and rsi_i >= cfg.rsi_overbought and vol_ok
                    and final_bear_ok and volatility_ok and squeeze_ok):
                signal, category, pattern_name = "SELL", "TRENDLINE", pattern
            if is_marubozu(df, i, bullish=True) and df["close"].iat[i] > line_val + cfg.breakout_atr_mult * atr_i:
                breakout_block_until = i + cfg.breakout_suppress_bars

        # ----------------------- SIDEWAYS: Horizontal S/R + Round Numbers -----
        elif trend_state == "SIDEWAYS":
            price = df["close"].iat[i]
            row = mtf_levels.iloc[i]
            sr_levels = [row.get(c) for c in
                         ["D_high", "D_low", "4H_high", "4H_low", "30T_high", "30T_low",
                          "10T_high", "10T_low", "5T_high", "5T_low"]]
            sr_levels = [lvl for lvl in sr_levels if pd.notna(lvl)]
            round_levels = get_round_number_zones(price, get_round_steps(price))

            high_i, low_i = df["high"].iat[i], df["low"].iat[i]
            nearest_sr = min(sr_levels, key=lambda l: abs(price - l)) if sr_levels else None
            nearest_round = min(round_levels, key=lambda l: abs(price - l)) if round_levels else None

            if nearest_sr is not None and (nearest_round is None or abs(price - nearest_sr) <= abs(price - nearest_round)):
                diag.update(nearest_level=nearest_sr, level_type="Horizontal S/R", distance=price - nearest_sr)
            elif nearest_round is not None:
                diag.update(nearest_level=nearest_round, level_type="Round Number", distance=price - nearest_round)

            # check a SUPPORT-context bullish pattern (for BUY) ...
            support_pattern = detect_pattern(df, i, "SUPPORT", cfg.wick_body_ratio)
            # ... and a RESISTANCE-context bearish pattern (for SELL), independently
            resistance_pattern = detect_pattern(df, i, "RESISTANCE", cfg.wick_body_ratio)
            diag["pattern_forming"] = support_pattern or resistance_pattern
            diag["volume_ok"] = vol_ok

            if (support_pattern is not None and rsi_i <= cfg.rsi_oversold and vol_ok
                    and final_bull_ok and volatility_ok and squeeze_ok):
                if nearest_sr is not None and abs(low_i - nearest_sr) <= round_tol:
                    signal, category, pattern_name = "BUY", "HORIZONTAL_SR", support_pattern
                elif nearest_round is not None and abs(low_i - nearest_round) <= round_tol:
                    signal, category, pattern_name = "BUY", "ROUND_NUMBER", support_pattern
            elif (resistance_pattern is not None and rsi_i >= cfg.rsi_overbought and vol_ok
                    and final_bear_ok and volatility_ok and squeeze_ok):
                if nearest_sr is not None and abs(high_i - nearest_sr) <= round_tol:
                    signal, category, pattern_name = "SELL", "HORIZONTAL_SR", resistance_pattern
                elif nearest_round is not None and abs(high_i - nearest_round) <= round_tol:
                    signal, category, pattern_name = "SELL", "ROUND_NUMBER", resistance_pattern

        # ----- most recent bar: no future close yet -> "live", not graded ------
        if i == n - 1:
            diag["pending_signal"], diag["pending_category"], diag["pending_pattern"] = signal, category, pattern_name
            diag["timestamp"] = df.index[i]
            live_status = diag
            continue

        if signal is not None:
            entry_price = df["close"].iat[i]
            for exp in cfg.expiries:
                if i + exp >= n:
                    continue
                exit_price = df["close"].iat[i + exp]
                if signal == "BUY":
                    result = "WIN" if exit_price > entry_price else "LOSS"
                else:
                    result = "WIN" if exit_price < entry_price else "LOSS"
                trades[exp].append(dict(
                    index=i, timestamp=df.index[i], category=category, direction=signal,
                    pattern=pattern_name, trend_state=trend_state, rsi=round(float(rsi_i), 1),
                    entry_price=entry_price, exit_price=exit_price, result=result,
                ))

    return trades, live_status, volume_reliable


def summarize(trade_list: list) -> dict:
    if not trade_list:
        return {"total": 0}
    tdf = pd.DataFrame(trade_list)
    total = len(tdf)
    wins = int((tdf["result"] == "WIN").sum())

    cat_rows = []
    for cat in ["TRENDLINE", "HORIZONTAL_SR", "ROUND_NUMBER"]:
        sub = tdf[tdf["category"] == cat]
        c_total = len(sub)
        c_wins = int((sub["result"] == "WIN").sum())
        cat_rows.append({
            "Category": {"TRENDLINE": "Trendline (Trend Pullback)",
                         "HORIZONTAL_SR": "Horizontal Support/Resistance",
                         "ROUND_NUMBER": "Round Numbers"}[cat],
            "Signals": c_total, "Wins": c_wins, "Losses": c_total - c_wins,
            "Win Rate %": round(100 * c_wins / c_total, 2) if c_total else 0.0,
        })

    pat_rows = []
    for pat in sorted(tdf["pattern"].dropna().unique()):
        sub = tdf[tdf["pattern"] == pat]
        p_total = len(sub)
        p_wins = int((sub["result"] == "WIN").sum())
        pat_rows.append({
            "Pattern": pat.replace("_", " ").title(), "Signals": p_total,
            "Win Rate %": round(100 * p_wins / p_total, 2) if p_total else 0.0,
        })

    return {
        "total": total, "wins": wins, "losses": total - wins,
        "win_rate": round(100 * wins / total, 2) if total else 0.0,
        "by_category": pd.DataFrame(cat_rows),
        "by_pattern": pd.DataFrame(pat_rows).sort_values("Signals", ascending=False) if pat_rows else pd.DataFrame(),
        "trades_df": tdf,
    }


# ==============================================================================
# 10. DATA FETCH - yfinance, fully automatic, no file upload, 3 timeframes
# ==============================================================================
def normalize_forex_symbol(user_input: str) -> str:
    s = user_input.strip().upper().replace("/", "").replace(" ", "").replace("\\", "")
    if not s.endswith("=X"):
        s = s + "=X"
    return s


@st.cache_data(ttl=180, show_spinner=False)
def fetch_ohlcv(yf_symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Free OHLCV from Yahoo Finance for any interval (1m/5m/15m). Normalizes
    timezone to naive so the three timeframes can be reliably merged."""
    raw = yf.download(tickers=yf_symbol, period=period, interval=interval,
                       progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return None
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    df = raw[keep].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df if len(df) > 0 else None


# ==============================================================================
# 11. STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Binary Options AI Agent v4", layout="wide")

COMMON_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP", "EURAUD", "EURCAD", "AUDJPY", "CHFJPY",
    "NZDJPY", "GBPCAD", "GBPAUD", "GBPNZD", "AUDCAD", "AUDNZD", "Custom...",
]

st.sidebar.title("Settings")
st.sidebar.subheader("1. Choose a Forex Pair")
choice = st.sidebar.selectbox("Pick from common pairs, or choose Custom", COMMON_PAIRS, index=0)
if choice == "Custom...":
    raw_symbol = st.sidebar.text_input("Type ANY pair (e.g. EURNZD, USDMXN, USDTRY)", value="EURNZD")
else:
    raw_symbol = choice

period = st.sidebar.selectbox("2. Historical window (1-min data)", ["5d", "7d"], index=1,
                               help="Yahoo Finance only keeps ~7-8 days of 1-minute history - a free-data-source limit.")

with st.sidebar.expander("Advanced strategy parameters"):
    cfg = Config()
    cfg.pivot_left = st.slider("Swing pivot strength (bars left/right)", 2, 6, 3)
    cfg.pivot_right = cfg.pivot_left
    cfg.trend_lookback = st.slider("Trend lookback (candles)", 50, 300, 150, step=10)
    cfg.touch_tolerance_atr_mult = st.slider("Trendline touch tolerance (x ATR)", 0.10, 0.60, 0.25, step=0.05)
    cfg.round_tolerance_atr_mult = st.slider("S/R & round-number tolerance (x ATR)", 0.10, 0.60, 0.20, step=0.05)
    cfg.breakout_atr_mult = st.slider("Breakout confirmation (x ATR)", 0.05, 0.40, 0.15, step=0.05)
    st.markdown("**Momentum & volume filters**")
    cfg.rsi_oversold = st.slider("RSI oversold threshold (BUY needs RSI <= this)", 15, 40, 35)
    cfg.rsi_overbought = st.slider("RSI overbought threshold (SELL needs RSI >= this)", 60, 85, 65)
    cfg.require_volume_confirmation = st.checkbox("Require volume confirmation", value=True)
    st.markdown("**Institutional-grade filters**")
    cfg.wick_body_ratio = st.slider("Min wick:body ratio (Hammer/Shooting Star/Inverted Hammer)", 1.5, 4.0, 2.5, step=0.1)
    cfg.atr_low_pct = st.slider("Min ATR percentile (below = too quiet/choppy)", 0.0, 0.30, 0.10, step=0.01)
    cfg.atr_high_pct = st.slider("Max ATR percentile (above = news-spike volatility)", 0.70, 1.00, 0.90, step=0.01)
    cfg.atr_regime_window = st.slider("ATR regime lookback window (bars)", 50, 200, 100, step=10)
    st.markdown("**Macro Trend Layer (NEW)**")
    st.caption("200 EMA on 5m and 15m - price must be strictly above (CALL) or below (PUT) BOTH for a signal to be valid. Not a slider: fixed at a true 200-period EMA per the spec.")
    st.markdown("**Bollinger Band Squeeze Filter (NEW)**")
    st.caption("Standard Bollinger Bands (20, 2) on the 1-minute chart. Bandwidth = (Upper - Lower) / Middle.")
    cfg.bb_squeeze_threshold = st.slider(
        "Squeeze threshold (bandwidth below this = chop, signals paused)",
        0.0002, 0.0050, 0.0008, step=0.0001, format="%.4f",
        help="Lower = stricter (locks out more often). This value is a fraction of price (e.g. 0.0008 = 0.08%% bandwidth). "
             "Tune per pair: JPY pairs and exotics can have structurally different typical bandwidths than EURUSD/GBPUSD.",
    )

run_clicked = st.sidebar.button("Fetch Live Data & Run Analysis", type="primary", use_container_width=True)

st.title("Binary Options AI Agent v4 - Institutional-Grade Precision Engine")
st.caption("Dow Theory structure - Dynamic trendlines - Multi-timeframe Horizontal S/R - Adaptive Round Numbers - "
           "7 candlestick patterns (incl. Inverted Hammer) - RSI(14) - Volume confirmation - "
           "5m+15m MTF trend alignment - strict wick:body ratio - AI market grading + auto-lock - "
           "ATR volatility regime filter - **200 EMA Macro Trend Layer (5m+15m)** - "
           "**Bollinger Band (20,2) Squeeze filter** - 1-min vs 3-min expiry comparison. "
           "Data: Yahoo Finance (`yfinance`) - free, automatic, no upload.")

st.warning(
    "Important: **Read before trusting any win rate below:** a consistent 85%+ real-world win rate on 1-minute retail "
    "price action, across *all* currency pairs, is an extraordinary claim that no publicly verifiable rule-based "
    "strategy reliably sustains - especially once broker payout structure (you risk 100% to win ~70-90%) and "
    "bid/ask spread are factored in. Stacking more filters (as this version does) trades signal *frequency* for "
    "signal *quality* - it can genuinely raise the win rate on the signals that still fire, but it cannot "
    "guarantee a fixed number, and on quiet pairs/sessions it may produce zero signals at all, which this "
    "dashboard will show you honestly rather than fabricate a result. Use the AI Market Grade below as your "
    "ongoing, honest readout, and validate on a demo account across many pairs/weeks before risking real money.",
)

if run_clicked:
    yf_symbol = normalize_forex_symbol(raw_symbol)
    with st.spinner(f"Fetching 1-min, 5-min, and 15-min data for {yf_symbol} from Yahoo Finance..."):
        data_1m = fetch_ohlcv(yf_symbol, period, "1m")
        data_5m = fetch_ohlcv(yf_symbol, period, "5m")
        data_15m = fetch_ohlcv(yf_symbol, period, "15m")

    if data_1m is None or len(data_1m) < 250:
        st.error(
            f"Could not retrieve usable 1-minute data for **{yf_symbol}**.\n\n"
            "Likely causes: invalid/unsupported symbol, FX market currently closed (weekends), "
            "or Yahoo's free 1-min history is temporarily thin for this pair. "
            "Try EURUSD or GBPUSD first to confirm the app itself works."
        )
    elif data_5m is None or len(data_5m) < 220 or data_15m is None or len(data_15m) < 220:
        st.error(
            "1-minute data loaded fine, but the 5-min and/or 15-min history is too short for a genuine 200-period "
            "EMA (the Macro Trend Layer needs at least 200 closed bars on EACH of those timeframes, plus a margin). "
            "This is common with the 5-day window on less-liquid pairs. Try the 7-day window, or a major pair "
            "(EURUSD, GBPUSD, USDJPY) which reliably has enough 5m/15m history."
        )
    else:
        with st.spinner("Running structure, trendline, S/R, round-number, RSI, volume, MTF-alignment, "
                         "wick-ratio, ATR-regime, 200-EMA macro-trend, and Bollinger-squeeze engine..."):
            trades, live_status, volume_reliable = run_engine(data_1m, data_5m, data_15m, cfg)
            stats_by_exp = {exp: summarize(trade_list) for exp, trade_list in trades.items()}
            ai_grade = compute_ai_grade(trades.get(1, []), cfg)
        st.session_state["stats_by_exp"] = stats_by_exp
        st.session_state["live_status"] = live_status
        st.session_state["volume_reliable"] = volume_reliable
        st.session_state["ai_grade"] = ai_grade
        st.session_state["symbol"] = yf_symbol
        st.session_state["n_candles"] = len(data_1m)
        st.session_state["recent"] = data_1m.tail(30)
        st.session_state["cfg_snapshot"] = cfg

# ------------------------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------------------------
if "stats_by_exp"in st.session_state:
    stats_by_exp = st.session_state["stats_by_exp"]
    live_status = st.session_state["live_status"]
    symbol = st.session_state["symbol"]
    volume_reliable = st.session_state["volume_reliable"]
    ai_grade = st.session_state["ai_grade"]
    cfg = st.session_state["cfg_snapshot"]
    pip_size = 0.01 if "JPY"in symbol else 0.0001

    st.subheader(f"Results for {symbol.replace('=X', '')} - {st.session_state['n_candles']} 1-min candles analyzed")

    if not volume_reliable:
        st.warning("Note: Volume data from Yahoo Finance for this pair is zero/unreliable (typical for free spot-FX "
                   "feeds with no centralized exchange tape). The **volume confirmation filter has been "
                   "automatically disabled** for this run - every other filter (RSI, MTF, wick ratio, ATR regime) "
                   "is still fully active.")

    # ----------------------------- AI MARKET GRADE ----------------------------
    st.markdown("### AI Market Grade")
    if ai_grade["status"] == "INSUFFICIENT_DATA":
        st.info(f"Not enough graded signals yet ({ai_grade['n']}/{cfg.ai_grade_min_signals} minimum) to compute "
                f"a reliable AI grade for this pair/window. The filters are strict by design - try a more "
                f"active pair/session, or loosen tolerances in the sidebar.")
    else:
        wr = ai_grade["win_rate"]
        n = ai_grade["n"]
        if ai_grade["status"] == "EXCELLENT":
            st.success(f"**{ai_grade['label']}** - rolling win rate over last {n} signals: **{wr}%**")
        elif ai_grade["status"] == "MODERATE":
            st.warning(f"**{ai_grade['label']}** - rolling win rate over last {n} signals: **{wr}%**")
        else:
            st.error(f"**{ai_grade['label']}** - rolling win rate over last {n} signals: **{wr}%**")

    # ----------------------- SIDE-BY-SIDE EXPIRY COMPARISON ------------------
    st.markdown("### 1-Minute vs 3-Minute Expiry - Side-by-Side")
    exp_cols = st.columns(len(stats_by_exp))
    for col, (exp, stats) in zip(exp_cols, stats_by_exp.items()):
        with col:
            st.markdown(f"#### {exp}-Minute Expiry")
            if stats.get("total", 0) == 0:
                st.info("No signals generated at this expiry.")
            else:
                st.metric("Win Rate", f"{stats['win_rate']}%")
                sub1, sub2 = st.columns(2)
                sub1.metric("Signals", stats["total"])
                sub2.metric("Wins / Losses", f"{stats['wins']} / {stats['losses']}")

    st.markdown("---")

    # ------------------------- DETAILED BREAKDOWN PER EXPIRY ------------------
    tabs = st.tabs([f"{exp}-Min Detail"for exp in stats_by_exp.keys()])
    for tab, (exp, stats) in zip(tabs, stats_by_exp.items()):
        with tab:
            if stats.get("total", 0) == 0:
                st.warning("No signals at this expiry. With all 6 institutional filters active (MTF alignment, "
                           "200-EMA macro trend, 2.5x wick ratio, ATR regime, Bollinger squeeze, plus RSI/volume), "
                           "signal frequency drops sharply by design. Try loosening sidebar settings, or a more "
                           "active pair/session.")
            else:
                st.markdown("**Breakdown by Strategy Category**")
                st.dataframe(stats["by_category"], use_container_width=True, hide_index=True)
                st.markdown("**Breakdown by Candlestick Pattern**")
                if len(stats["by_pattern"]):
                    st.dataframe(stats["by_pattern"], use_container_width=True, hide_index=True)
                else:
                    st.info("No pattern data available.")
                with st.expander("View raw trade log"):
                    st.dataframe(stats["trades_df"], use_container_width=True)

    # --------------------------- LIVE SIGNAL ROOM ------------------------------
    st.markdown("---")
    st.markdown("### Live Status / Signal Room")

    if ai_grade.get("locked"):
        st.error("**SIGNAL ENGINE LOCKED** - the AI Market Grade for this pair is currently BAD "
                 "(rolling win rate below 60% over the last signals). Live signal alerts are suppressed until "
                 "conditions improve. You can still review the historical backtest tables above.")

    if live_status is not None and not live_status.get("squeeze_ok", True):
        st.warning("**Market in tight squeeze (Chop/Sideways) - Signals Paused.** "
                   f"Bollinger Bandwidth is {live_status.get('bb_width', float('nan')):.5f}, below the "
                   f"{cfg.bb_squeeze_threshold:.4f} squeeze threshold. The engine will not fire new signals "
                   "on the current candle until volatility expands.")

    if live_status is None:
        st.info("Not enough data yet to evaluate the current candle.")
    else:
        ts = live_status.get("timestamp")
        trend = live_status["trend_state"]
        price = live_status["price"]
        rsi_val = live_status["rsi"]
        level = live_status["nearest_level"]
        level_type = live_status["level_type"]
        distance = live_status["distance"]
        pattern = live_status["pattern_forming"]
        vol_ok = live_status["volume_ok"]
        mtf_5 = live_status["mtf_5m"]
        mtf_15 = live_status["mtf_15m"]
        macro_5 = live_status["macro_5m"]
        macro_15 = live_status["macro_15m"]
        vol_regime_ok = live_status["volatility_ok"]
        atr_pctile = live_status["atr_percentile"]
        squeeze_ok = live_status["squeeze_ok"]
        bb_width_val = live_status["bb_width"]
        pending = live_status.get("pending_signal")

        trend_emoji = {"UPTREND": "UP", "DOWNTREND": "DOWN", "SIDEWAYS": "FLAT"}[trend]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market State (1m)", f"{trend} ({trend_emoji})")
        c2.metric("Last Close", f"{price:.5f}")
        rsi_zone = "Oversold" if rsi_val <= cfg.rsi_oversold else ("Overbought" if rsi_val >= cfg.rsi_overbought else "Neutral")
        c3.metric("RSI(14)", f"{rsi_val:.1f}", rsi_zone)
        if level is not None:
            dist_pips = abs(distance) / pip_size
            c4.metric(f"Dist. to {level_type.split('(')[0].strip()}", f"{dist_pips:.1f} pips")
        else:
            c4.metric("Nearest level", "N/A")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("5m MTF Bias", mtf_5)
        c6.metric("15m MTF Bias", mtf_15)
        if np.isfinite(atr_pctile):
            vol_label = "Normal" if vol_regime_ok else ("Too Quiet" if atr_pctile < cfg.atr_low_pct else "News Spike")
            c7.metric("Volatility Regime", vol_label, f"{atr_pctile*100:.0f}th pct ATR")
        else:
            c7.metric("Volatility Regime", "Warming up...")
        if np.isfinite(bb_width_val):
            squeeze_label = "Open" if squeeze_ok else "Squeeze"
            c8.metric("BB(20,2) Width", squeeze_label, f"{bb_width_val:.4f}")
        else:
            c8.metric("BB(20,2) Width", "Warming up...")

        d1, d2 = st.columns(2)
        d1.metric("Macro Trend 5m (vs 200 EMA)", macro_5)
        d2.metric("Macro Trend 15m (vs 200 EMA)", macro_15)

        if level is not None:
            st.write(f"**Nearest key level:** `{level_type}` @ **{level:.5f}** "
                     f"(price is {'above' if distance >= 0 else 'below'} it by {abs(distance)/pip_size:.1f} pips)")

        if pattern:
            vol_text = "confirmed" if vol_ok else ("not confirmed" if vol_ok is not None else "n/a")
            macro_ok_text = "Yes" if (live_status["macro_bull_aligned"] or live_status["macro_bear_aligned"]) else "No"
            st.write(f"**Pattern forming:** {pattern.replace('_',' ').title()} - **Volume:** {vol_text} - "
                     f"**RSI condition:** {'met' if rsi_zone != 'Neutral' else 'not met'} - "
                     f"**MTF aligned:** {'Yes' if (live_status['mtf_bull_aligned'] or live_status['mtf_bear_aligned']) else 'No'} - "
                     f"**Macro (200 EMA) aligned:** {macro_ok_text} - "
                     f"**Volatility OK:** {'Yes' if vol_regime_ok else 'No'} - "
                     f"**No Squeeze:** {'Yes' if squeeze_ok else 'No'}")

        if pending and not ai_grade.get("locked") and squeeze_ok:
            direction_word = "BUY (CALL)" if pending == "BUY" else "SELL (PUT)"
            st.success(f"**LIVE SIGNAL - ALL FILTERS PASSED:** {direction_word} - "
                       f"pattern: **{(live_status.get('pending_pattern') or '').replace('_',' ').title()}** "
                       f"at the {level_type}, RSI={rsi_val:.1f}, MTF aligned, Macro (200 EMA) aligned, "
                       f"volatility normal, no squeeze. This is the most recent candle and is not yet graded "
                       f"(no future close available).")
        elif pending and ai_grade.get("locked"):
            st.info("A signal technically qualified on the current candle, but it is **withheld** because the "
                    "Signal Engine is locked (AI Market Grade = BAD). See the lock notice above.")
        elif pending and not squeeze_ok:
            st.info("A signal technically qualified on the current candle, but it is **withheld** because the "
                    "market is in a Bollinger Band squeeze. See the squeeze notice above.")
        elif pattern:
            st.info("A reversal pattern is forming, but one or more filters (RSI / volume / MTF alignment / "
                    "Macro 200-EMA alignment / volatility regime / squeeze / level-touch) isn't satisfied - "
                    "no signal triggered on the current candle.")
        else:
            st.write("No reversal pattern forming on the current candle. Engine is watching for the next setup.")

        st.caption(f"Snapshot as of candle timestamp: {ts}")

    with st.expander("Last 30 raw 1-min candles"):
        st.dataframe(st.session_state["recent"], use_container_width=True)

else:
    st.info("Pick a pair in the sidebar and click **'Fetch Live Data & Run Analysis'** to begin. "
             "1-min, 5-min, and 15-min data are pulled automatically from Yahoo Finance - no file upload needed.")
