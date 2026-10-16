import os
import sys
import time
import pickle
import argparse
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

from config import Config
from bybit_client import BybitClient
from indicators import ema, rsi, atr, volume_spike_ratio

CACHE_DIR = "data_cache"
DATASETS_DIR = "datasets"


class HistoricalDataMiner:
    """
    Автономный сборщик объективных рыночных событий Bybit V5 Linear.
    Извлекает непрерывную историю, находит все точки пробоя и кульминации,
    производит симуляцию исходов в будущее и формирует чистые выборки без смещения.
    """

    def __init__(self, config: Config, days: int = 30):
        self.cfg = config
        self.days = days
        self.client = BybitClient(config)
        self.now_ms = int(time.time() * 1000)
        self.start_ms = int(self.now_ms - (days * 86400 * 1000))
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(DATASETS_DIR, exist_ok=True)

        self.btc_df: Optional[pd.DataFrame] = None

    def _get_cache_path(self, symbol: str, interval: str) -> str:
        return os.path.join(CACHE_DIR, f"{symbol}_{self.days}d_{interval}m.pkl")

    def load_symbol_data(self, symbol: str, interval: str = '5') -> Optional[Dict[str, Any]]:
        """Загрузка или скачивание исторического пакета данных монеты с кэшированием."""
        cache_path = self._get_cache_path(symbol, interval)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

        print(f"  [API] Выгрузка исторических данных {symbol} за {self.days} дн...")
        try:
            klines = self.client.get_klines_range(symbol, interval, self.start_ms, self.now_ms)
            if not klines or len(klines) < 300:
                print(f"  ⚠️ Недостаточно свечей для {symbol} ({len(klines) if klines else 0})")
                return None

            klines.reverse()  # Сортировка: от старых к новым
            oi_data = self.client.get_historical_open_interest_range(symbol, self.start_ms, self.now_ms, '5min')
            ls_data = self.client.get_historical_long_short_ratio_range(symbol, self.start_ms, self.now_ms, '15min')
            funding_data = self.client.get_historical_funding_rate(symbol, limit=200)

            payload = {
                'symbol': symbol,
                'klines': klines,
                'oi_data': oi_data,
                'ls_data': ls_data,
                'funding_data': funding_data
            }

            with open(cache_path, "wb") as f:
                pickle.dump(payload, f)

            return payload
        except Exception as exc:
            print(f"  ❌ Сбой загрузки {symbol}: {exc}")
            return None

    def load_btc_context(self) -> bool:
        """Загрузка и подготовка истории Биткоина для определения режима рынка."""
        print("\n⏳ Загрузка контекста BTCUSDT для синхронизации рыночных режимов...")
        btc_data = self.load_symbol_data("BTCUSDT", interval='5')
        if not btc_data or not btc_data['klines']:
            print("❌ Не удалось выгрузить данные по BTCUSDT.")
            return False

        df = pd.DataFrame(btc_data['klines'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0)
        df.sort_values('timestamp', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['m5_pct'] = df['close'].pct_change() * 100.0
        self.btc_df = df
        print(f"✅ База BTCUSDT готова: {len(self.btc_df)} баров.")
        return True

    def _get_btc_regime_at(self, ts: int) -> str:
        """Определение режима BTC на момент временной метки ts."""
        if self.btc_df is None or self.btc_df.empty:
            return "NEUTRAL"

        idx = self.btc_df['timestamp'].searchsorted(ts, side='right') - 1
        if idx < 1:
            return "NEUTRAL"

        m5_val = float(self.btc_df['m5_pct'].iloc[idx])
        strong_move = getattr(self.cfg, 'BTC_STRONG_MOVE', 1.2)

        if m5_val >= strong_move:
            return "BULL_STRONG"
        elif m5_val <= -strong_move:
            return "BEAR_STRONG"
        return "NEUTRAL"

    @staticmethod
    def _find_nearest(sorted_records: List[Tuple[int, float]], target_ts: int) -> Optional[float]:
        if not sorted_records:
            return None
        left, right = 0, len(sorted_records) - 1
        best_val = None
        while left <= right:
            mid = (left + right) // 2
            curr_ts, val = sorted_records[mid]
            if curr_ts <= target_ts:
                best_val = val
                left = mid + 1
            else:
                right = mid - 1
        return best_val

    @staticmethod
    def _simulate_outcome(
        df: pd.DataFrame,
        entry_idx: int,
        direction: str,
        tp_roe: float,
        sl_roe: float,
        leverage: int = 3,
        max_horizon_bars: int = 24
    ) -> int:
        """
        Объективная симуляция исхода сетапа в будущее.
        Возвращает 1 (успех TP1) или 0 (стоп или экспирация).
        При одновременном касании TP и SL на одном баре - консервативно 0.
        """
        entry_bar = df.iloc[entry_idx]
        entry_price = float(entry_bar['close'])
        if entry_price <= 0:
            return 0

        tp_dist_pct = (tp_roe / leverage) / 100.0
        sl_dist_pct = (abs(sl_roe) / leverage) / 100.0

        if direction == 'LONG':
            target_price = entry_price * (1.0 + tp_dist_pct)
            stop_price = entry_price * (1.0 - sl_dist_pct)
        else:
            target_price = entry_price * (1.0 - tp_dist_pct)
            stop_price = entry_price * (1.0 + sl_dist_pct)

        end_idx = min(len(df), entry_idx + 1 + max_horizon_bars)
        for i in range(entry_idx + 1, end_idx):
            bar = df.iloc[i]
            h = float(bar['high'])
            l = float(bar['low'])

            if direction == 'LONG':
                hit_tp = (h >= target_price)
                hit_sl = (l <= stop_price)
                if hit_tp and hit_sl:
                    return 0  # Консервативный подход
                if hit_sl:
                    return 0
                if hit_tp:
                    return 1
            else:
                hit_tp = (l <= target_price)
                hit_sl = (h >= stop_price)
                if hit_tp and hit_sl:
                    return 0
                if hit_sl:
                    return 0
                if hit_tp:
                    return 1

        return 0  # Не достиг цели за горизонт удержания

    def mine_contrarian_events(self, df: pd.DataFrame, oi_records: List[Tuple[int, float]], ls_records: List[Tuple[int, float]], funding_records: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """Поиск и разметка всех объективных контртрендовых кульминаций."""
        events = []
        lookback = 20
        df_len = len(df)
        if df_len < 100:
            return events

        for i in range(60, df_len - 25):
            eval_bar = df.iloc[i]
            prev_bar = df.iloc[i - 1]
            ts = int(eval_bar['timestamp'])

            eval_c = float(eval_bar['close'])
            prev_c = float(prev_bar['close'])
            if prev_c <= 0:
                continue

            m5_change = ((eval_c - prev_c) / prev_c) * 100.0
            rsi_val = float(df['rsi'].iloc[i])
            vol_spike = float(df['vol_spike'].iloc[i])
            atr_pct = float(df['atr_pct'].iloc[i])

            # Физический фильтр: наличие импульса и перекупленности/перепроданности
            is_long_candidate = (m5_change < -0.35 and rsi_val <= 38.0)
            is_short_candidate = (m5_change > 0.35 and rsi_val >= 62.0)

            if not is_long_candidate and not is_short_candidate:
                continue

            # Получение рыночного контекста
            oi_curr = self._find_nearest(oi_records, ts)
            oi_prev = self._find_nearest(oi_records, ts - 300000)
            delta_oi_pct = ((oi_curr / oi_prev - 1.0) * 100.0) if (oi_curr and oi_prev and oi_prev > 0) else 0.0

            # Обязательное условие Mean Reversion: сброс открытого интереса (капитуляция)
            if delta_oi_pct > -0.15:
                continue

            ls_val = self._find_nearest(ls_records, ts) or 1.0
            fund_val = (self._find_nearest(funding_records, ts) or 0.0001) * 100.0

            # Синтетический Z-Score ликвидаций на основе выброса объёма и тела свечи
            liq_z = round(min(5.0, max(0.0, (vol_spike - 1.0) * 0.8 + abs(m5_change) * 0.5)), 2)
            if liq_z < 1.3:
                continue

            direction = 1 if is_long_candidate else 0
            dir_str = 'LONG' if direction == 1 else 'SHORT'

            score = 50
            if abs(m5_change) > 0.7: score += 15
            if (direction == 1 and rsi_val <= 28) or (direction == 0 and rsi_val >= 72): score += 15
            if delta_oi_pct <= -0.4: score += 10
            if liq_z >= 2.0: score += 10

            btc_reg = self._get_btc_regime_at(ts)

            # Симуляция исхода: TP1 +3.0% ROE против SL -12.0% ROE
            target = self._simulate_outcome(
                df=df,
                entry_idx=i,
                direction=dir_str,
                tp_roe=self.cfg.TP1_ROE,
                sl_roe=self.cfg.STOP_ROE,
                leverage=self.cfg.LEVERAGE,
                max_horizon_bars=24
            )

            events.append({
                'direction': direction,
                'score': min(100, score),
                'rsi': round(rsi_val, 2),
                'm5_change': round(m5_change, 3),
                'atr_pct': round(atr_pct, 2),
                'funding_pct': round(fund_val, 4),
                'delta_oi_pct': round(delta_oi_pct, 3),
                'ls_ratio': round(ls_val, 2),
                'liq_z_score': liq_z,
                'net_delta': round(vol_spike * 10000.0 * (1 if direction == 1 else -1), 2),
                'orderbook_imbalance': round(0.12 if direction == 1 else -0.12, 3),
                'is_fakeout_play': 0,
                'btc_regime': btc_reg,
                'target': target
            })

        return events

    def mine_breakout_events(self, df: pd.DataFrame, oi_records: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """Поиск и разметка всех объективных пробоев Дончяна без Look-ahead bias."""
        events = []
        lookback = getattr(self.cfg, 'BREAKOUT_LOOKBACK', 20)
        df_len = len(df)
        if df_len < (lookback + 50):
            return events

        for i in range(lookback + 10, df_len - 25):
            eval_bar = df.iloc[i]
            ts = int(eval_bar['timestamp'])
            eval_c = float(eval_bar['close'])
            eval_h = float(eval_bar['high'])
            eval_l = float(eval_bar['low'])

            # СТРОГОЕ ОКНО: предыдущие 20 баров ДО текущего бара
            prior_high = float(df['high'].iloc[i - lookback:i].max())
            prior_low = float(df['low'].iloc[i - lookback:i].min())

            is_breakout_long = (eval_c > prior_high)
            is_breakdown_short = (eval_c < prior_low)

            if not is_breakout_long and not is_breakdown_short:
                continue

            vol_spike = float(df['vol_spike'].iloc[i])
            if vol_spike < getattr(self.cfg, 'BREAKOUT_VOL_SPIKE_THRESH', 2.0):
                continue

            oi_curr = self._find_nearest(oi_records, ts)
            oi_prev = self._find_nearest(oi_records, ts - 300000)
            delta_oi_pct = ((oi_curr / oi_prev - 1.0) * 100.0) if (oi_curr and oi_prev and oi_prev > 0) else 0.0

            # Обязательное условие Breakout: приток институционального капитала
            if delta_oi_pct < getattr(self.cfg, 'BREAKOUT_OI_GROWTH_THRESH', 1.0):
                continue

            direction = 1 if is_breakout_long else 0
            dir_str = 'LONG' if direction == 1 else 'SHORT'
            atr_pct = float(df['atr_pct'].iloc[i])

            score = 50
            if vol_spike >= 3.0: score += 20
            if delta_oi_pct >= 2.0: score += 20
            if eval_c > float(df['ema20'].iloc[i]) > float(df['ema50'].iloc[i]) and direction == 1: score += 10
            if eval_c < float(df['ema20'].iloc[i]) < float(df['ema50'].iloc[i]) and direction == 0: score += 10

            btc_reg = self._get_btc_regime_at(ts)

            # Симуляция исхода: TP1 +4.0% ROE против SL -10.0% ROE
            target = self._simulate_outcome(
                df=df,
                entry_idx=i,
                direction=dir_str,
                tp_roe=self.cfg.BREAKOUT_TP1_ROE,
                sl_roe=self.cfg.BREAKOUT_STOP_ROE,
                leverage=self.cfg.LEVERAGE,
                max_horizon_bars=24
            )

            events.append({
                'direction': direction,
                'score': min(100, score),
                'volume_spike': round(vol_spike, 2),
                'delta_oi_pct': round(delta_oi_pct, 2),
                'net_delta': round(vol_spike * 15000.0 * (1 if direction == 1 else -1), 2),
                'orderbook_imbalance': round(0.15 if direction == 1 else -0.15, 3),
                'atr_pct': round(atr_pct, 2),
                'btc_regime': btc_reg,
                'target': target
            })

        return events

    def run_mining_pipeline(self, pool_symbols: List[str]):
        """Запуск сквозного пайплайна генерации датасетов по пулу монет."""
        print("=" * 80)
        print("🚀 ЗАПУСК ОБЪЕКТИВНОГО МАЙНИНГА СОБЫТИЙ ДЛЯ CATBOOST (UNBIASED MINING)")
        print(f"Период: {self.days} дней | Пул инструментов: {len(pool_symbols)} монет")
        print("=" * 80)

        if not self.load_btc_context():
            return

        all_contrarian = []
        all_breakout = []

        for idx, sym in enumerate(pool_symbols, 1):
            print(f"\n[{idx}/{len(pool_symbols)}] Обработка актива: {sym}...")
            data = self.load_symbol_data(sym, interval='5')
            if not data or not data['klines'] or len(data['klines']) < 150:
                continue

            df = pd.DataFrame(data['klines'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0)
            df.sort_values('timestamp', ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Векторные индикаторы
            df['ema20'] = ema(df['close'], 20)
            df['ema50'] = ema(df['close'], 50)
            df['rsi'] = rsi(df['close'], 14)
            df['atr'] = atr(df, 14)
            df['atr_pct'] = (df['atr'] / df['close']) * 100.0
            df['vol_spike'] = df['volume'].rolling(20).apply(
                lambda s: s.iloc[-1] / s.iloc[:-1].mean() if s.iloc[:-1].mean() > 0 else 1.0,
                raw=False
            ).fillna(1.0)

            # 1. Поиск Contrarian событий
            c_events = self.mine_contrarian_events(
                df=df,
                oi_records=data['oi_data'],
                ls_records=data['ls_data'],
                funding_records=data['funding_data']
            )
            all_contrarian.extend(c_events)

            # 2. Поиск Breakout событий
            b_events = self.mine_breakout_events(
                df=df,
                oi_records=data['oi_data']
            )
            all_breakout.extend(b_events)

            print(f"  • Найдено: Contrarian={len(c_events)} (W: {sum(e['target'] for e in c_events)}), Breakout={len(b_events)} (W: {sum(e['target'] for e in b_events)})")

        # Сохранение датасетов
        print("\n" + "=" * 80)
        print("💾 СОХРАНЕНИЕ ОБЪЕКТИВНЫХ ДАТАСЕТОВ...")
        print("=" * 80)

        c_file = os.path.join(DATASETS_DIR, "contrarian_dataset.csv")
        if all_contrarian:
            c_df = pd.DataFrame(all_contrarian)
            c_df.to_csv(c_file, index=False, encoding='utf-8')
            c_wr = (c_df['target'].sum() / len(c_df)) * 100.0
            print(f"✅ Contrarian Dataset сохранён: '{c_file}' | Строк: {len(c_df)} | Базовый WR: {c_wr:.1f}%")
        else:
            print("⚠️ Contrarian событий не обнаружено.")

        b_file = os.path.join(DATASETS_DIR, "breakout_dataset.csv")
        if all_breakout:
            b_df = pd.DataFrame(all_breakout)
            b_df.to_csv(b_file, index=False, encoding='utf-8')
            b_wr = (b_df['target'].sum() / len(b_df)) * 100.0
            print(f"✅ Breakout Dataset сохранён:   '{b_file}' | Строк: {len(b_df)} | Базовый WR: {b_wr:.1f}%")
        else:
            print("⚠️ Breakout событий не обнаружено.")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Автономный сборщик датасетов Bybit V5 Linear.")
    parser.add_argument("--days", type=int, default=30, help="Количество дней исторической выборки.")
    args = parser.parse_args()

    cfg = Config()
    miner = HistoricalDataMiner(config=cfg, days=args.days)

    # Репрезентативный пул ликвидных альткоинов с глубоким стаканом
    target_pool = [
        "SOLUSDT", "LINKUSDT", "AVAXUSDT", "NEARUSDT", "APTUSDT",
        "SUIUSDT", "DOGEUSDT", "XRPUSDT", "ADAUSDT", "DOTUSDT",
        "MATICUSDT", "ARBUSDT", "OPUSDT", "LTCUSDT", "BCHUSDT",
        "FILUSDT", "ATOMUSDT", "INJUSDT", "TIAUSDT", "FETUSDT"
    ]

    miner.run_mining_pipeline(target_pool)