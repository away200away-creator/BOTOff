import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from config import Config
from ml_model import TradeFilterModel
from indicators import (
    ema, rsi, atr,
    is_bullish_pin_bar, is_bearish_pin_bar,
    is_bullish_engulfing, is_bearish_engulfing,
    volume_spike_ratio
)


class ContrarianStrategy:
    def __init__(self, config: Config):
        self.cfg = config
        self.btc_regime = "NEUTRAL"
        self.btc_data = {}
        self.ml_filter = TradeFilterModel(model_path="trade_filter.cbm")
        self.ml_min_confidence = getattr(config, 'ML_MIN_CONFIDENCE', 0.60)

        self.liq_z_thresh = getattr(config, 'LIQ_Z_THRESHOLD', 1.5)
        self.oi_drop_thresh = getattr(config, 'OI_DROP_THRESHOLD', -0.2)
        self.require_liq_and_oi = getattr(config, 'REQUIRE_LIQ_AND_OI', True)
        self.max_24h_pump_for_short = getattr(config, 'MAX_24H_CHANGE_FOR_SHORT', 25.0)

    def _klines_to_df(self, klines: list) -> pd.DataFrame:
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0)
        df.sort_values('timestamp', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def analyze_btc(self, klines_5m: list, klines_15m: list) -> str:
        if not klines_5m or not klines_15m:
            return self.btc_regime

        df5 = self._klines_to_df(klines_5m)
        df15 = self._klines_to_df(klines_15m)

        if len(df5) < 30 or len(df15) < 30:
            return self.btc_regime

        df5['ema20'] = ema(df5['close'], 20)
        df5['ema50'] = ema(df5['close'], 50)
        df5['ema200'] = ema(df5['close'], min(200, len(df5)))
        df5['rsi'] = rsi(df5['close'], 14)

        price = float(df5['close'].iloc[-1])
        prev_price_5m = float(df5['close'].iloc[-2])
        prev_price_15m = float(df15['close'].iloc[-2])

        m5_change = ((price - prev_price_5m) / prev_price_5m * 100.0) if prev_price_5m > 0 else 0.0
        m15_change = ((float(df15['close'].iloc[-1]) - prev_price_15m) / prev_price_15m * 100.0) if prev_price_15m > 0 else 0.0

        ema200_val = float(df5['ema200'].iloc[-1])
        ema20_val = float(df5['ema20'].iloc[-1])
        ema50_val = float(df5['ema50'].iloc[-1])

        is_trend_up = (price > ema200_val) and (ema20_val > ema50_val)
        is_trend_down = (price < ema200_val) and (ema20_val < ema50_val)

        atr_series = atr(df5, 14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        atr_percent = (atr_val / price * 100.0) if price > 0 else 0.0
        is_ranging = (atr_percent < 2.5) and not is_trend_up and not is_trend_down

        if is_trend_up:
            regime = "BULL_STRONG"
        elif is_trend_down:
            regime = "BEAR_STRONG"
        elif is_ranging:
            regime = "RANGING"
        else:
            regime = "NEUTRAL"

        self.btc_regime = regime
        self.btc_data = {
            'regime': regime,
            'is_trend_up': is_trend_up,
            'is_trend_down': is_trend_down,
            'is_ranging': is_ranging,
            'm5': m5_change,
            'm15': m15_change,
            'rsi': float(df5['rsi'].iloc[-1]),
            'price': price
        }
        return self.btc_regime

    def _is_blocked(self, direction: str) -> bool:
        btc_m5 = abs(self.btc_data.get('m5', 0.0))
        if self.btc_regime == 'BULL_STRONG' and direction == 'SHORT' and btc_m5 > 1.2:
            return True
        if self.btc_regime == 'BEAR_STRONG' and direction == 'LONG' and btc_m5 > 1.2:
            return True

        against = ((direction == 'LONG' and self.btc_regime == 'BEAR_STRONG') or
                   (direction == 'SHORT' and self.btc_regime == 'BULL_STRONG'))
        if against and getattr(self.cfg, 'BTC_FILTER', 'hard') == 'hard':
            return True
        return False

    def analyze_symbol(
        self,
        ticker: dict,
        klines: list,
        oi_change: Optional[float],
        ls_ratio: Optional[float],
        order_flow_data: Optional[dict] = None,
        orderbook_imbalance: float = 0.0,
        spread_data: Optional[dict] = None,
        debug: bool = False
    ) -> Optional[dict]:
        if not ticker or not isinstance(ticker, dict) or not klines:
            return None

        symbol = ticker.get('symbol', 'UNKNOWN')

        try:
            volume_mln = float(ticker.get('turnover24h', 0.0)) / 1_000_000.0
        except (ValueError, TypeError):
            volume_mln = 0.0

        if volume_mln < self.cfg.MIN_VOLUME_MLN:
            return None

        df = self._klines_to_df(klines)
        if len(df) < 60:
            return None

        # Расчёт индикаторов
        df['ema20'] = ema(df['close'], self.cfg.EMA_FAST)
        df['rsi'] = rsi(df['close'], self.cfg.RSI_PERIOD)
        df['atr'] = atr(df, self.cfg.ATR_PERIOD)

        # ----------------------------------------------------------------------------------
        # ДЕТЕКЦИЯ СТАТУСА БАРА (Устранение Intrabar Repainting)
        # ----------------------------------------------------------------------------------
        interval_ms = int(getattr(self.cfg, 'KLINE_INTERVAL', 5)) * 60 * 1000
        if len(df) >= 2:
            diff_ts = int(df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2])
            if diff_ts > 0:
                interval_ms = diff_ts

        now_ms = ticker.get('timestamp')
        if now_ms is None:
            now_ms = time.time() * 1000.0
        else:
            try:
                now_ms = float(now_ms)
            except (ValueError, TypeError):
                now_ms = time.time() * 1000.0

        last_bar_start = float(df['timestamp'].iloc[-1])
        is_forming_bar = (0 <= (now_ms - last_bar_start) < interval_ms)

        if is_forming_bar and len(df) >= 61:
            eval_idx = -2
            prev_eval_idx = -3
        else:
            eval_idx = -1
            prev_eval_idx = -2

        eval_bar = df.iloc[eval_idx]
        prev_bar = df.iloc[prev_eval_idx]

        prev_close = float(prev_bar['close'])
        eval_close = float(eval_bar['close'])
        m5_change = ((eval_close - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
        current_rsi = float(df['rsi'].iloc[eval_idx])
        atr_val = float(df['atr'].iloc[eval_idx])
        ema20_val = float(df['ema20'].iloc[eval_idx])

        vol_series_to_eval = df['volume'].iloc[:(len(df) + eval_idx + 1)]
        vol_spike = volume_spike_ratio(vol_series_to_eval, lookback=20)

        # Цена исполнения
        try:
            live_price = float(ticker.get('lastPrice', df['close'].iloc[-1]))
        except (ValueError, TypeError):
            live_price = eval_close

        if live_price <= 0:
            live_price = eval_close

        # Метрики Order Flow
        liq_z = 0.0
        net_delta = 0.0
        if order_flow_data:
            try:
                liq_z = float(order_flow_data.get('liq_z_score', 0.0))
                net_delta = float(order_flow_data.get('net_delta', 0.0))
            except (ValueError, TypeError):
                pass

        try:
            oi_change_val = float(oi_change) if oi_change is not None else None
        except (ValueError, TypeError):
            oi_change_val = None

        try:
            ls_ratio_val = float(ls_ratio) if ls_ratio is not None else None
        except (ValueError, TypeError):
            ls_ratio_val = None

        try:
            ob_imbalance = float(orderbook_imbalance) if orderbook_imbalance is not None else 0.0
        except (ValueError, TypeError):
            ob_imbalance = 0.0

        require_gate = getattr(self.cfg, 'REQUIRE_LIQ_AND_OI', self.require_liq_and_oi)
        z_thresh = getattr(self.cfg, 'LIQ_Z_THRESHOLD', self.liq_z_thresh)
        oi_thresh = getattr(self.cfg, 'OI_DROP_THRESHOLD', self.oi_drop_thresh)

        # ----------------------------------------------------------------------------------
        # ВХОДНОЙ ШЛЮЗ ORDER FLOW
        # ----------------------------------------------------------------------------------
        if order_flow_data is not None:
            if require_gate:
                if liq_z < z_thresh:
                    return None
                if oi_change_val is not None and oi_change_val > oi_thresh:
                    if not (liq_z >= (z_thresh * 1.6) and oi_change_val <= 0.05):
                        return None
            else:
                is_liq_surge = (liq_z >= z_thresh) or (vol_spike >= 2.2 and abs(m5_change) >= 0.5)
                if not is_liq_surge:
                    return None
        else:
            if vol_spike < 1.3 and abs(m5_change) < 0.35:
                return None

        # Ставка финансирования и суточный памп
        try:
            fund_val = ticker.get('fundingRate', 0.0)
            funding_rate = float(fund_val) * 100.0 if fund_val is not None else 0.0
        except (ValueError, TypeError):
            funding_rate = 0.0

        try:
            price_24h_pct = float(ticker.get('price24hPcnt', 0.0)) * 100.0
        except (ValueError, TypeError):
            price_24h_pct = 0.0

        # Паттерны Price Action на закрытых барах
        pin_body_ratio = getattr(self.cfg, 'PIN_BAR_BODY_RATIO', 0.33)
        has_bullish_pa = is_bullish_pin_bar(eval_bar, pin_body_ratio) or is_bullish_engulfing(prev_bar, eval_bar)
        has_bearish_pa = is_bearish_pin_bar(eval_bar, pin_body_ratio) or is_bearish_engulfing(prev_bar, eval_bar)

        long_score = 0.0
        short_score = 0.0

        ob_enabled = getattr(self.cfg, 'ENABLE_ORDERBOOK_IMBALANCE', True)
        ob_thresh = getattr(self.cfg, 'OB_IMBALANCE_THRESHOLD', 0.15)
        rsi_oversold = getattr(self.cfg, 'RSI_OVERSOLD', 30)
        rsi_overbought = getattr(self.cfg, 'RSI_OVERBOUGHT', 70)

        # ----------------------------------------------------------------------------------
        # КАЛИБРОВАННЫЙ СКОРИНГ (Исключение Score 50 Trap)
        # ----------------------------------------------------------------------------------
        # LONG: Базовый (20) + Ликвидации (20) = 40. Требуется подтверждение для Score >= 50.
        if m5_change < -0.3 and current_rsi <= 40:
            if oi_change_val is None or oi_change_val <= oi_thresh or (liq_z >= z_thresh * 1.6 and oi_change_val <= 0.05):
                if not ob_enabled or ob_imbalance >= -0.05:
                    long_score += 20.0
                    if liq_z >= z_thresh:
                        long_score += 20.0
                    if net_delta > 0:
                        long_score += 15.0
                    if current_rsi <= rsi_oversold:
                        long_score += 15.0
                    if ob_enabled and ob_imbalance >= ob_thresh:
                        long_score += 15.0

        # SHORT: Базовый (20) + Ликвидации (20) = 40. Требуется подтверждение для Score >= 50.
        if m5_change > 0.3 and current_rsi >= 60:
            if price_24h_pct < self.max_24h_pump_for_short:
                if oi_change_val is None or oi_change_val <= oi_thresh or (liq_z >= z_thresh * 1.6 and oi_change_val <= 0.05):
                    if not ob_enabled or ob_imbalance <= 0.05:
                        short_score += 20.0
                        if liq_z >= z_thresh:
                            short_score += 20.0
                        if net_delta < 0:
                            short_score += 15.0
                        if current_rsi >= rsi_overbought:
                            short_score += 15.0
                        if ob_enabled and ob_imbalance <= -ob_thresh:
                            short_score += 15.0

        if funding_rate > self.cfg.FUNDING_LONG_THRESH:
            short_score += 10.0
        elif funding_rate < self.cfg.FUNDING_SHORT_THRESH:
            long_score += 10.0

        if ls_ratio_val is not None:
            if ls_ratio_val > self.cfg.LS_LONG_THRESH:
                short_score += 10.0
            elif ls_ratio_val < self.cfg.LS_SHORT_THRESH:
                long_score += 10.0

        if has_bullish_pa:
            long_score += 10.0
        if has_bearish_pa:
            short_score += 10.0

        direction = 'LONG' if long_score >= short_score else 'SHORT'
        raw_score = max(long_score, short_score)
        score = max(0, min(100, round(raw_score)))

        min_score_threshold = getattr(self.cfg, 'MIN_SCORE', 50)
        if score < min_score_threshold:
            return None

        if self._is_blocked(direction):
            return None

        # Предохранитель от шорта параболических ракет (Pump Shield)
        if direction == 'SHORT' and price_24h_pct >= self.max_24h_pump_for_short:
            return None

        # Вето стакана L2
        if ob_enabled:
            if direction == 'LONG' and ob_imbalance < -0.25:
                return None
            if direction == 'SHORT' and ob_imbalance > 0.25:
                return None

        ema_dist = ((live_price - ema20_val) / atr_val) if atr_val > 0 else 0.0
        atr_pct = ((atr_val / live_price) * 100.0) if live_price > 0 else 0.0

        # ----------------------------------------------------------------------------------
        # МАТРИЦА ML ИНФЕРЕНСА (Полный 13-факторный вектор)
        # ----------------------------------------------------------------------------------
        features_for_ml = {
            'direction': 1 if direction == 'LONG' else 0,
            'score': score,
            'rsi': current_rsi,
            'm5': m5_change,
            'atr_pct': atr_pct,
            'funding': funding_rate,
            'oi_change': oi_change_val if oi_change_val is not None else 0.0,
            'ls_ratio': ls_ratio_val if ls_ratio_val is not None else 1.0,
            'vol_spike': vol_spike,
            'ema_dist': ema_dist,
            'btc_regime': self.btc_regime,
            'liq_z': liq_z,
            'net_delta': net_delta
        }

        win_prob = self.ml_filter.predict_win_probability(features_for_ml)

        if self.ml_filter.is_loaded and win_prob < self.ml_min_confidence:
            return None

        maker_entry_price = live_price
        if spread_data and isinstance(spread_data, dict):
            if direction == 'LONG':
                maker_entry_price = float(spread_data.get('bid', live_price))
            else:
                maker_entry_price = float(spread_data.get('ask', live_price))

        if maker_entry_price <= 0:
            maker_entry_price = live_price

        return {
            'symbol': symbol,
            'price': live_price,
            'maker_price': maker_entry_price,
            'direction': direction,
            'score': score,
            'win_prob': win_prob,
            'volatility': atr_pct,
            'rsi': current_rsi,
            'm5': m5_change,
            'atr': atr_val,
            'funding': funding_rate,
            'oi_change': oi_change_val,
            'ls_ratio': ls_ratio_val,
            'volume_spike': vol_spike,
            'liq_z_score': liq_z,
            'net_delta': net_delta,
            'orderbook_imbalance': ob_imbalance,
            'ema20': ema20_val,
        }