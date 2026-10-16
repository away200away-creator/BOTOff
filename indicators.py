import numpy as np
import pandas as pd
from typing import Tuple


def ema(series: pd.Series, period: int) -> pd.Series:
    """Экспоненциальная скользящая средняя (EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Индекс относительной силы (RSI) со сглаживанием Уайлдера."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    res = 100 - (100 / (1 + rs))
    return res.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Канонический Average True Range (ATR)."""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def is_bullish_pin_bar(row: pd.Series, max_body_ratio: float = 0.33) -> bool:
    """Бычий пин-бар: маленькое тело + длинная нижняя тень."""
    rng = float(row['high'] - row['low'])
    if rng <= 0:
        return False
    body = abs(float(row['close'] - row['open']))
    lower_shadow = min(float(row['open']), float(row['close'])) - float(row['low'])
    upper_shadow = float(row['high']) - max(float(row['open']), float(row['close']))
    return (
        body / rng <= max_body_ratio and
        lower_shadow >= rng * 0.60 and
        upper_shadow <= rng * 0.25
    )


def is_bearish_pin_bar(row: pd.Series, max_body_ratio: float = 0.33) -> bool:
    """Медвежий пин-бар: маленькое тело + длинная верхняя тень."""
    rng = float(row['high'] - row['low'])
    if rng <= 0:
        return False
    body = abs(float(row['close'] - row['open']))
    upper_shadow = float(row['high']) - max(float(row['open']), float(row['close']))
    lower_shadow = min(float(row['open']), float(row['close'])) - float(row['low'])
    return (
        body / rng <= max_body_ratio and
        upper_shadow >= rng * 0.60 and
        lower_shadow <= rng * 0.25
    )


def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """Бычье свечное поглощение."""
    return (
        prev['open'] > prev['close'] and
        curr['open'] < curr['close'] and
        curr['open'] <= prev['close'] and
        curr['close'] >= prev['open']
    )


def is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """Медвежье свечное поглощение."""
    return (
        prev['open'] < prev['close'] and
        curr['open'] > curr['close'] and
        curr['open'] >= prev['close'] and
        curr['close'] <= prev['open']
    )


def volume_spike_ratio(volumes: pd.Series, lookback: int = 20) -> float:
    """Отношение объёма последнего бара к среднему за lookback предыдущих."""
    if len(volumes) < lookback + 1:
        return 1.0
    current = float(volumes.iloc[-1])
    avg = float(volumes.iloc[-(lookback + 1):-1].mean())
    return current / avg if avg > 0 else 1.0


def highest_high(series: pd.Series, period: int = 20) -> pd.Series:
    """Максимум за скользящий период period."""
    return series.rolling(window=period, min_periods=max(1, period // 2)).max()


def lowest_low(series: pd.Series, period: int = 20) -> pd.Series:
    """Минимум за скользящий период period."""
    return series.rolling(window=period, min_periods=max(1, period // 2)).min()


def donchian_channel(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Канал Дончяна (Donchian Channel):
    Возвращает кортеж (Upper Band, Lower Band, Middle Band).
    """
    upper = highest_high(df['high'], period)
    lower = lowest_low(df['low'], period)
    middle = (upper + lower) / 2.0
    return upper, lower, middle