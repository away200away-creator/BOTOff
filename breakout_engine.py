import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from indicators import ema, rsi, atr, volume_spike_ratio
from anti_crowd import AntiCrowdMetrics

logger = logging.getLogger("BreakoutEngine")

@dataclass
class FailedBreakoutEvent:
    symbol: str
    direction: str          # 'FAILED_LONG' или 'FAILED_SHORT'
    level: float            # Граница канала
    high_reached: float     # Экстремум закола
    close_price: float      # Закрытие бара внутри диапазона
    delta_oi_pct: float     # Изменение открытого интереса
    net_delta: float        # Тиковая дельта CVD
    volume_spike: float
    timestamp: float

@dataclass
class BreakoutSignal:
    symbol: str
    strategy_id: str        # 'BREAKOUT'
    direction: str          # 'LONG' или 'SHORT'
    price: float
    maker_price: float
    score: int
    win_prob: Optional[float]
    atr_pct: float
    channel_high: float
    channel_low: float
    volume_spike: float
    delta_oi_pct: float
    net_delta: float
    orderbook_imbalance: float
    tp1_roe: float
    stop_roe: float

class BreakoutEngine:
    """
    Движок импульсного пробоя диапазона (Donchian Expansion).
    Границы канала рассчитываются строго по предыдущим N барам (Look-ahead Free).
    """

    def __init__(
        self,
        lookback_bars: int = 20,
        min_vol_spike: float = 2.0,
        min_oi_growth: float = 1.0,
        min_score: int = 50,
        tp1_roe: float = 4.0,
        stop_roe: float = -10.0,
        max_24h_pump_for_short: float = 25.0
    ):
        self.lookback = lookback_bars
        self.min_vol_spike = min_vol_spike
        self.min_oi_growth = min_oi_growth
        self.min_score = min_score
        self.tp1_roe = tp1_roe
        self.stop_roe = stop_roe
        self.max_24h_pump_for_short = max_24h_pump_for_short

    @staticmethod
    def _klines_to_df(klines: list) -> pd.DataFrame:
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0)
        df.sort_values('timestamp', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def analyze(
        self,
        ticker: Dict[str, Any],
        klines: list,
        anti_crowd: AntiCrowdMetrics,
        oi_change: Optional[float],
        orderbook_imbalance: float = 0.0,
        spread_data: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[BreakoutSignal], Optional[FailedBreakoutEvent]]:
        if not ticker or not klines or len(klines) < (self.lookback + 25):
            return None, None

        symbol = ticker.get('symbol', 'UNKNOWN')
        df = self._klines_to_df(klines)

        df['ema20'] = ema(df['close'], 20)
        df['ema50'] = ema(df['close'], 50)
        df['rsi'] = rsi(df['close'], 14)
        df['atr'] = atr(df, 14)

        interval_ms = 5 * 60 * 1000
        if len(df) >= 2:
            diff_ts = int(df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2])
            if diff_ts > 0:
                interval_ms = diff_ts

        now_ms = float(ticker.get('timestamp', time.time() * 1000.0))
        last_bar_start = float(df['timestamp'].iloc[-1])
        is_forming_bar = (0 <= (now_ms - last_bar_start) < interval_ms)

        eval_idx = -2 if (is_forming_bar and len(df) >= (self.lookback + 26)) else -1
        eval_bar = df.iloc[eval_idx]
        eval_close = float(eval_bar['close'])
        eval_high = float(eval_bar['high'])
        eval_low = float(eval_bar['low'])

        eval_pos = len(df) + eval_idx
        window_start = max(0, eval_pos - self.lookback)
        prior_window = df.iloc[window_start:eval_pos]

        if len(prior_window) < self.lookback * 0.75:
            return None, None

        channel_high = float(prior_window['high'].max())
        channel_low = float(prior_window['low'].min())

        vol_series_to_eval = df['volume'].iloc[:eval_pos + 1]
        vol_spike = volume_spike_ratio(vol_series_to_eval, lookback=self.lookback)

        live_price = float(ticker.get('lastPrice', eval_close))
        if live_price <= 0:
            live_price = eval_close

        delta_oi = float(oi_change) if oi_change is not None else 0.0
        net_delta = anti_crowd.net_delta
        price_24h_pct = float(ticker.get('price24hPcnt', 0.0)) * 100.0
        atr_val = float(df['atr'].iloc[eval_idx])
        atr_pct = (atr_val / live_price * 100.0) if live_price > 0 else 2.0

        if eval_high > channel_high and eval_close <= channel_high:
            if delta_oi <= -0.2 and net_delta < 0:
                fakeout_event = FailedBreakoutEvent(
                    symbol=symbol,
                    direction='FAILED_LONG',
                    level=channel_high,
                    high_reached=eval_high,
                    close_price=eval_close,
                    delta_oi_pct=delta_oi,
                    net_delta=net_delta,
                    volume_spike=vol_spike,
                    timestamp=time.time()
                )
                return None, fakeout_event

        if eval_low < channel_low and eval_close >= channel_low:
            if delta_oi <= -0.2 and net_delta > 0:
                fakeout_event = FailedBreakoutEvent(
                    symbol=symbol,
                    direction='FAILED_SHORT',
                    level=channel_low,
                    high_reached=eval_low,
                    close_price=eval_close,
                    delta_oi_pct=delta_oi,
                    net_delta=net_delta,
                    volume_spike=vol_spike,
                    timestamp=time.time()
                )
                return None, fakeout_event

        is_breakout_long = (eval_close > channel_high)
        is_breakdown_short = (eval_close < channel_low)

        if not is_breakout_long and not is_breakdown_short:
            return None, None

        if delta_oi < self.min_oi_growth or vol_spike < self.min_vol_spike:
            return None, None

        long_score = 0.0
        short_score = 0.0

        if is_breakout_long and net_delta > 0:
            long_score += 40.0
            if delta_oi >= (self.min_oi_growth * 2.0):
                long_score += 20.0
            if orderbook_imbalance >= 0.10:
                long_score += 15.0
            if eval_close > float(df['ema20'].iloc[eval_idx]) > float(df['ema50'].iloc[eval_idx]):
                long_score += 15.0
            if anti_crowd.ls_ratio <= 1.0:
                long_score += 10.0

        if is_breakdown_short and net_delta < 0:
            if price_24h_pct < self.max_24h_pump_for_short:
                short_score += 40.0
                if delta_oi >= (self.min_oi_growth * 2.0):
                    short_score += 20.0
                if orderbook_imbalance <= -0.10:
                    short_score += 15.0
                if eval_close < float(df['ema20'].iloc[eval_idx]) < float(df['ema50'].iloc[eval_idx]):
                    short_score += 15.0
                if anti_crowd.ls_ratio >= 1.3:
                    short_score += 10.0

        direction = 'LONG' if long_score >= short_score else 'SHORT'
        score = max(0, min(100, round(max(long_score, short_score))))

        if score < self.min_score:
            return None, None

        if direction == 'SHORT' and price_24h_pct >= self.max_24h_pump_for_short:
            return None, None

        maker_price = live_price
        if spread_data and isinstance(spread_data, dict):
            maker_price = spread_data.get('bid' if direction == 'LONG' else 'ask', live_price)

        signal = BreakoutSignal(
            symbol=symbol,
            strategy_id='BREAKOUT',
            direction=direction,
            price=live_price,
            maker_price=maker_price,
            score=score,
            win_prob=None,
            atr_pct=round(atr_pct, 2),
            channel_high=round(channel_high, 4),
            channel_low=round(channel_low, 4),
            volume_spike=round(vol_spike, 2),
            delta_oi_pct=round(delta_oi, 2),
            net_delta=round(net_delta, 2),
            orderbook_imbalance=round(orderbook_imbalance, 3),
            tp1_roe=self.tp1_roe,
            stop_roe=self.stop_roe
        )
        return signal, None