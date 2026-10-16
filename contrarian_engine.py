import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

from indicators import (
    ema, rsi, atr,
    is_bullish_pin_bar, is_bearish_pin_bar,
    is_bullish_engulfing, is_bearish_engulfing
)
from anti_crowd import AntiCrowdMetrics
from breakout_engine import FailedBreakoutEvent

logger = logging.getLogger("ContrarianEngine")

@dataclass
class ContrarianSignal:
    symbol: str
    strategy_id: str        # 'CONTRARIAN'
    direction: str          # 'LONG' или 'SHORT'
    price: float
    maker_price: float
    score: int
    win_prob: Optional[float]
    atr_pct: float
    rsi: float
    m5_change: float
    atr: float
    funding_pct: float
    delta_oi_pct: float
    ls_ratio: float
    liq_z_score: float
    net_delta: float
    orderbook_imbalance: float
    is_fakeout_play: bool
    tp1_roe: float
    stop_roe: float

class ContrarianEngine:
    def __init__(
        self,
        liq_z_thresh: float = 1.5,
        oi_drop_thresh: float = -0.2,
        min_score: int = 50,
        max_24h_pump_for_short: float = 25.0,
        tp1_roe: float = 3.0,
        stop_roe: float = -12.0
    ):
        self.liq_z_thresh = liq_z_thresh
        self.oi_drop_thresh = oi_drop_thresh
        self.min_score = min_score
        self.max_24h_pump_for_short = max_24h_pump_for_short
        self.tp1_roe = tp1_roe
        self.stop_roe = stop_roe

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

    def evaluate_failed_breakout(
        self,
        fakeout: FailedBreakoutEvent,
        anti_crowd: AntiCrowdMetrics,
        ticker: Dict[str, Any],
        spread_data: Optional[Dict[str, float]] = None
    ) -> Optional[ContrarianSignal]:
        price_24h_pct = float(ticker.get('price24hPcnt', 0.0)) * 100.0
        live_price = float(ticker.get('lastPrice', fakeout.close_price))
        if live_price <= 0:
            live_price = fakeout.close_price

        # FAILED_LONG -> CONTRARIAN SHORT
        if fakeout.direction == 'FAILED_LONG':
            if price_24h_pct >= self.max_24h_pump_for_short:
                return None
            if anti_crowd.net_delta > 0 or fakeout.delta_oi_pct > 0.1:
                return None

            score = 55.0
            if anti_crowd.crowd_side == 'LONG_HEAVY' or anti_crowd.ls_ratio >= 1.3:
                score += 15.0
            if anti_crowd.liquidation_spike or anti_crowd.long_liq_vol_usd > 10000:
                score += 15.0
            if anti_crowd.orderbook_imbalance <= -0.10:
                score += 15.0

            final_score = max(0, min(100, round(score)))
            if final_score >= self.min_score:
                maker_price = spread_data.get('ask', live_price) if spread_data else live_price
                return ContrarianSignal(
                    symbol=fakeout.symbol,
                    strategy_id='CONTRARIAN',
                    direction='SHORT',
                    price=live_price,
                    maker_price=maker_price,
                    score=final_score,
                    win_prob=None,
                    atr_pct=2.0,
                    rsi=65.0,
                    m5_change=0.5,
                    atr=0.0,
                    funding_pct=anti_crowd.funding_rate_pct,
                    delta_oi_pct=fakeout.delta_oi_pct,
                    ls_ratio=anti_crowd.ls_ratio,
                    liq_z_score=anti_crowd.liq_z_score,
                    net_delta=anti_crowd.net_delta,
                    orderbook_imbalance=anti_crowd.orderbook_imbalance,
                    is_fakeout_play=True,
                    tp1_roe=self.tp1_roe,
                    stop_roe=self.stop_roe
                )

        # FAILED_SHORT -> CONTRARIAN LONG
        elif fakeout.direction == 'FAILED_SHORT':
            if anti_crowd.net_delta < 0 or fakeout.delta_oi_pct > 0.1:
                return None

            score = 55.0
            if anti_crowd.crowd_side == 'SHORT_HEAVY' or anti_crowd.ls_ratio <= 0.8:
                score += 15.0
            if anti_crowd.liquidation_spike or anti_crowd.short_liq_vol_usd > 10000:
                score += 15.0
            if anti_crowd.orderbook_imbalance >= 0.10:
                score += 15.0

            final_score = max(0, min(100, round(score)))
            if final_score >= self.min_score:
                maker_price = spread_data.get('bid', live_price) if spread_data else live_price
                return ContrarianSignal(
                    symbol=fakeout.symbol,
                    strategy_id='CONTRARIAN',
                    direction='LONG',
                    price=live_price,
                    maker_price=maker_price,
                    score=final_score,
                    win_prob=None,
                    atr_pct=2.0,
                    rsi=35.0,
                    m5_change=-0.5,
                    atr=0.0,
                    funding_pct=anti_crowd.funding_rate_pct,
                    delta_oi_pct=fakeout.delta_oi_pct,
                    ls_ratio=anti_crowd.ls_ratio,
                    liq_z_score=anti_crowd.liq_z_score,
                    net_delta=anti_crowd.net_delta,
                    orderbook_imbalance=anti_crowd.orderbook_imbalance,
                    is_fakeout_play=True,
                    tp1_roe=self.tp1_roe,
                    stop_roe=self.stop_roe
                )

        return None

    def analyze(
        self,
        ticker: Dict[str, Any],
        klines: list,
        anti_crowd: AntiCrowdMetrics,
        oi_change: Optional[float],
        orderbook_imbalance: float = 0.0,
        spread_data: Optional[Dict[str, float]] = None
    ) -> Optional[ContrarianSignal]:
        if not ticker or not klines or len(klines) < 60:
            return None

        symbol = ticker.get('symbol', 'UNKNOWN')
        df = self._klines_to_df(klines)

        df['ema20'] = ema(df['close'], 20)
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

        eval_idx = -2 if (is_forming_bar and len(df) >= 61) else -1
        prev_eval_idx = -3 if (is_forming_bar and len(df) >= 61) else -2

        eval_bar = df.iloc[eval_idx]
        prev_bar = df.iloc[prev_eval_idx]

        eval_close = float(eval_bar['close'])
        prev_close = float(prev_bar['close'])
        m5_change = ((eval_close - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0

        current_rsi = float(df['rsi'].iloc[eval_idx])
        atr_val = float(df['atr'].iloc[eval_idx])
        live_price = float(ticker.get('lastPrice', eval_close))
        if live_price <= 0:
            live_price = eval_close

        delta_oi = float(oi_change) if oi_change is not None else 0.0
        price_24h_pct = float(ticker.get('price24hPcnt', 0.0)) * 100.0
        atr_pct = (atr_val / live_price * 100.0) if live_price > 0 else 2.0

        if anti_crowd.liq_z_score < self.liq_z_thresh:
            return None
        if delta_oi > self.oi_drop_thresh and not (anti_crowd.liq_z_score >= (self.liq_z_thresh * 1.6) and delta_oi <= 0.05):
            return None

        has_bullish_pa = is_bullish_pin_bar(eval_bar, 0.33) or is_bullish_engulfing(prev_bar, eval_bar)
        has_bearish_pa = is_bearish_pin_bar(eval_bar, 0.33) or is_bearish_engulfing(prev_bar, eval_bar)

        long_score = 0.0
        short_score = 0.0

        if m5_change < -0.3 and current_rsi <= 40:
            if orderbook_imbalance >= -0.05:
                long_score += 20.0
                if anti_crowd.liq_z_score >= self.liq_z_thresh:
                    long_score += 20.0
                if anti_crowd.net_delta > 0:
                    long_score += 15.0
                if current_rsi <= 30:
                    long_score += 15.0
                if orderbook_imbalance >= 0.15:
                    long_score += 15.0

        if m5_change > 0.3 and current_rsi >= 60:
            if price_24h_pct < self.max_24h_pump_for_short and orderbook_imbalance <= 0.05:
                short_score += 20.0
                if anti_crowd.liq_z_score >= self.liq_z_thresh:
                    short_score += 20.0
                if anti_crowd.net_delta < 0:
                    short_score += 15.0
                if current_rsi >= 70:
                    short_score += 15.0
                if orderbook_imbalance <= -0.15:
                    short_score += 15.0

        if anti_crowd.funding_rate_pct > 0.01:
            short_score += 10.0
        elif anti_crowd.funding_rate_pct < -0.01:
            long_score += 10.0

        if anti_crowd.ls_ratio > 1.4:
            short_score += 10.0
        elif anti_crowd.ls_ratio < 0.7:
            long_score += 10.0

        if has_bullish_pa:
            long_score += 10.0
        if has_bearish_pa:
            short_score += 10.0

        direction = 'LONG' if long_score >= short_score else 'SHORT'
        score = max(0, min(100, round(max(long_score, short_score))))

        if score < self.min_score:
            return None

        if direction == 'SHORT' and price_24h_pct >= self.max_24h_pump_for_short:
            return None

        if direction == 'LONG' and orderbook_imbalance < -0.25:
            return None
        if direction == 'SHORT' and orderbook_imbalance > 0.25:
            return None

        maker_price = spread_data.get('bid' if direction == 'LONG' else 'ask', live_price) if spread_data else live_price

        return ContrarianSignal(
            symbol=symbol,
            strategy_id='CONTRARIAN',
            direction=direction,
            price=live_price,
            maker_price=maker_price,
            score=score,
            win_prob=None,
            atr_pct=round(atr_pct, 2),
            rsi=round(current_rsi, 1),
            m5_change=round(m5_change, 2),
            atr=round(atr_val, 4),
            funding_pct=anti_crowd.funding_rate_pct,
            delta_oi_pct=round(delta_oi, 2),
            ls_ratio=anti_crowd.ls_ratio,
            liq_z_score=anti_crowd.liq_z_score,
            net_delta=anti_crowd.net_delta,
            orderbook_imbalance=round(orderbook_imbalance, 3),
            is_fakeout_play=False,
            tp1_roe=self.tp1_roe,
            stop_roe=self.stop_roe
        )