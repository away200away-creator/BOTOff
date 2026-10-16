import time
import os
import sys
import pickle
import traceback
from typing import Dict, List, Optional, Any
import pandas as pd

from config import Config
from bybit_client import BybitClient
from strategy import ContrarianStrategy
from risk_manager import RiskManager, Position
import telegram_notifier

# Глушим все Telegram уведомления
telegram_notifier.send_telegram_message = lambda *args, **kwargs: None
if hasattr(telegram_notifier, 'notify_entry'):
    telegram_notifier.notify_entry = lambda *args, **kwargs: None
if hasattr(telegram_notifier, 'notify_tp1'):
    telegram_notifier.notify_tp1 = lambda *args, **kwargs: None
if hasattr(telegram_notifier, 'notify_close'):
    telegram_notifier.notify_close = lambda *args, **kwargs: None


class Backtester:
    def __init__(self, config: Config):
        self.cfg = config
        self.client = BybitClient(config)

        if hasattr(self.client, 'order_flow') and self.client.order_flow:
            try:
                self.client.order_flow.is_connected = False
            except Exception:
                pass

        self.strategy = ContrarianStrategy(config)
        self.cache_dir = "data_cache"

        # Изоляция журнала бэктеста
        try:
            self.risk = RiskManager(config, history_file=None)
        except TypeError:
            self.risk = RiskManager(config)
            self.risk.history_file = None

    def _load_market_data(self, symbol: str, days: int, timeframe: str) -> Optional[Dict[str, Any]]:
        cache_path = os.path.join(self.cache_dir, f"{symbol}_{days}d_{timeframe}m.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Ошибка чтения кэша {cache_path}: {e}")

        end_time = int(time.time() * 1000)
        start_time = int((time.time() - days * 86400) * 1000)
        try:
            klines = self.client.get_klines_range(symbol, timeframe, start_time, end_time)
            if klines:
                klines.reverse()
            oi_data = self.client.get_historical_open_interest_range(symbol, start_time, end_time, '5min')
            ls_data = self.client.get_historical_long_short_ratio_range(symbol, start_time, end_time, '15min')
            funding_data = self.client.get_historical_funding_rate(symbol, 200)

            payload = {
                'klines': klines,
                'oi_data': oi_data,
                'ls_data': ls_data,
                'funding_data': funding_data
            }
            if klines and len(klines) > 100:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(payload, f)
            return payload
        except Exception as e:
            print(f"❌ Ошибка выгрузки данных для {symbol}: {e}")
            return None

    def prepare_ticker(self, symbol: str, klines_rev: list, funding_rate: float = 0.0001) -> dict:
        candles_24h = 288
        if len(klines_rev) >= candles_24h:
            price_24h_ago = float(klines_rev[candles_24h - 1][4])
        else:
            price_24h_ago = float(klines_rev[-1][4])

        last_price = float(klines_rev[0][4])
        price_24h_pct = ((last_price - price_24h_ago) / price_24h_ago) if price_24h_ago > 0 else 0.0

        return {
            'symbol': symbol,
            'turnover24h': str(sum(float(c[5]) for c in klines_rev[:candles_24h]) * 1e6),
            'fundingRate': str(funding_rate),
            'price24hPcnt': str(price_24h_pct),
            'lastPrice': klines_rev[0][4],
        }

    def _find_nearest(self, data_map: dict, target_ts: int):
        if not data_map:
            return None
        best_ts = None
        best_val = None
        for ts, val in data_map.items():
            if ts <= target_ts and (best_ts is None or ts > best_ts):
                best_ts = ts
                best_val = val
        return best_val

    def run_single(self, symbol: str, days: int = 45, timeframe: str = '5', use_ml: bool = True) -> Optional[dict]:
        data = self._load_market_data(symbol, days, timeframe)
        if not data or not data['klines'] or len(data['klines']) < 120:
            return None

        all_klines = data['klines']
        oi_map = {ts: oi for ts, oi in data['oi_data']}
        ls_map = {ts: ratio for ts, ratio in data['ls_data']}
        funding_map = {ts: rate for ts, rate in data['funding_data']}

        if not hasattr(self, '_ml_was_loaded'):
            self._ml_was_loaded = getattr(self.strategy.ml_filter, 'is_loaded', False)

        self.strategy.ml_filter.is_loaded = bool(use_ml and self._ml_was_loaded)
        self.strategy.ml_min_confidence = getattr(self.cfg, 'ML_MIN_CONFIDENCE', 0.60)

        self.risk.balance = self.cfg.INITIAL_BALANCE
        self.risk.positions = []
        self.risk.history = []
        self.risk.wins = 0
        self.risk.cooldowns = {}
        self.risk.history_file = None

        last_signal_bar = -999

        for i in range(100, len(all_klines)):
            candle = all_klines[i]
            current_ts = int(candle[0])
            current_ts_sec = current_ts / 1000.0
            open_p, high_p, low_p, close_p = float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4])

            # 1. Симуляция с виртуальным временем
            for pos in self.risk.positions[:]:
                worst_price = low_p if pos.direction == 'LONG' else high_p
                pos.update_price(worst_price)
                self.risk.manage_position(pos, current_time=current_ts_sec)

                if pos in self.risk.positions:
                    best_price = high_p if pos.direction == 'LONG' else low_p
                    pos.update_price(best_price)
                    self.risk.manage_position(pos, current_time=current_ts_sec)

                if pos in self.risk.positions:
                    pos.update_price(close_p)
                    self.risk.manage_position(pos, current_time=current_ts_sec)

            # 2. Поиск новых сигналов
            if i - last_signal_bar >= 6 and self.risk.can_open():
                current_window = all_klines[i-100:i]
                current_window_rev = current_window[::-1]

                oi_val = self._find_nearest(oi_map, current_ts)
                ls_val = self._find_nearest(ls_map, current_ts)
                funding_val = self._find_nearest(funding_map, current_ts)

                oi_change = 0.0
                if oi_val is not None:
                    prev_ts = current_ts - 5 * 60 * 1000
                    prev_oi = self._find_nearest(oi_map, prev_ts)
                    if prev_oi and prev_oi != 0:
                        oi_change = (oi_val / prev_oi - 1) * 100

                ticker = self.prepare_ticker(
                    symbol,
                    current_window_rev,
                    funding_rate=funding_val if funding_val is not None else 0.0001
                )

                signal = self.strategy.analyze_symbol(
                    ticker,
                    current_window_rev,
                    oi_change,
                    ls_val
                )

                if signal and signal['score'] >= self.cfg.MIN_SCORE:
                    new_pos = self.risk.open_position(signal, current_time=current_ts_sec)
                    if new_pos:
                        last_signal_bar = i

        if self.risk.positions:
            last_close = float(all_klines[-1][4])
            last_ts_sec = int(all_klines[-1][0]) / 1000.0
            for pos in self.risk.positions[:]:
                pos.update_price(last_close)
                self.risk._close(pos, "Конец бэктеста", current_time=last_ts_sec)

        trades_count = len(self.risk.history)
        wins = sum(1 for t in self.risk.history if t['pnl'] > 0)
        losses = sum(1 for t in self.risk.history if t['pnl'] <= 0)
        winrate = (wins / trades_count * 100) if trades_count > 0 else 0.0
        total_pnl = sum(t['pnl'] for t in self.risk.history)
        gross_profit = sum(t['pnl'] for t in self.risk.history if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.risk.history if t['pnl'] < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        return {
            'symbol': symbol,
            'trades': trades_count,
            'wins': wins,
            'losses': losses,
            'winrate': winrate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_dd': self.calc_max_drawdown()
        }

    def calc_max_drawdown(self) -> float:
        if not self.risk.history:
            return 0.0
        balance = self.cfg.INITIAL_BALANCE
        peak = balance
        max_dd = 0.0
        for t in self.risk.history:
            balance += t['pnl']
            if balance > peak:
                peak = balance
            dd = peak - balance
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def run_ab_comparison(self, symbols: List[str], days: int = 45, timeframe: str = '5'):
        print("\n" + "=" * 95)
        print(f"🧪 ЗАПУСК СРАВНИТЕЛЬНОГО БЭКТЕСТА (A/B ТЕСТ) ЗА {days} ДНЕЙ")
        print(f"Параметры: TP1={self.cfg.TP1_ROE}% (закрытие {self.cfg.TP1_CLOSE_PCT}%), SL={self.cfg.STOP_ROE}%, Трейлинг={self.cfg.TRAILING_DISTANCE}%")
        print(f"Фильтр: MIN_SCORE={self.cfg.MIN_SCORE} | ML={self.cfg.ML_MIN_CONFIDENCE*100:.0f}% | Slip={getattr(self.cfg,'SLIPPAGE_PCT',0)}% | Z≥{self.cfg.LIQ_Z_THRESHOLD} & OI≤{self.cfg.OI_DROP_THRESHOLD}")
        print("=" * 95)

        results_no_ml = []
        results_with_ml = []

        print("\n⏳ [1/2] Прогон стратегии БЕЗ ИИ (чистая базовая логика)...")
        for sym in symbols:
            res = self.run_single(sym, days, timeframe, use_ml=False)
            if res:
                results_no_ml.append(res)
                print(f"  • {sym:10s} | Сделок: {res['trades']:3d} | WR: {res['winrate']:5.1f}% | PnL: {res['total_pnl']:+6.2f}$ | PF: {res['profit_factor']:.2f}")

        print(f"\n⏳ [2/2] Прогон стратегии С ML-ФИЛЬТРОМ (CatBoost > {self.cfg.ML_MIN_CONFIDENCE*100:.0f}%)...")
        for sym in symbols:
            res = self.run_single(sym, days, timeframe, use_ml=True)
            if res:
                results_with_ml.append(res)
                print(f"  • {sym:10s} | Сделок: {res['trades']:3d} | WR: {res['winrate']:5.1f}% | PnL: {res['total_pnl']:+6.2f}$ | PF: {res['profit_factor']:.2f}")

        self._print_comparison_table(results_no_ml, results_with_ml)

    def _print_comparison_table(self, no_ml: List[dict], with_ml: List[dict]):
        print("\n" + "=" * 95)
        print("📊 ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ ЭФФЕКТИВНОСТИ ИИ-ФИЛЬТРА")
        print("=" * 95)
        print(f"{'Символ':10} | {'Сделок (Без/С ИИ)':^18} | {'Winrate (Без/С ИИ)':^20} | {'PnL (Без/С ИИ)':^22} | {'PF (Без/С ИИ)':^15}")
        print("-" * 95)

        total_trades_no = sum(r['trades'] for r in no_ml)
        total_trades_ml = sum(r['trades'] for r in with_ml)
        total_pnl_no = sum(r['total_pnl'] for r in no_ml)
        total_pnl_ml = sum(r['total_pnl'] for r in with_ml)

        total_wins_no = sum(r['wins'] for r in no_ml)
        total_wins_ml = sum(r['wins'] for r in with_ml)
        avg_wr_no = (total_wins_no / total_trades_no * 100) if total_trades_no else 0
        avg_wr_ml = (total_wins_ml / total_trades_ml * 100) if total_trades_ml else 0

        for r_no, r_ml in zip(no_ml, with_ml):
            trades_str = f"{r_no['trades']} ➔ {r_ml['trades']}"
            wr_str = f"{r_no['winrate']:.1f}% ➔ {r_ml['winrate']:.1f}%"
            pnl_str = f"{r_no['total_pnl']:+.2f}$ ➔ {r_ml['total_pnl']:+.2f}$"
            pf_str = f"{r_no['profit_factor']:.2f} ➔ {r_ml['profit_factor']:.2f}"
            print(f"{r_no['symbol']:10} | {trades_str:^18} | {wr_str:^20} | {pnl_str:^22} | {pf_str:^15}")

        print("-" * 95)
        summary_trades = f"{total_trades_no} ➔ {total_trades_ml}"
        summary_wr = f"{avg_wr_no:.1f}% ➔ {avg_wr_ml:.1f}%"
        summary_pnl = f"{total_pnl_no:+.2f}$ ➔ {total_pnl_ml:+.2f}$"
        print(f"{'ИТОГО':10} | {summary_trades:^18} | {summary_wr:^20} | {summary_pnl:^22} |")
        print("=" * 95 + "\n")


if __name__ == "__main__":
    print("\n🚀 [Backtester] Инициализация процесса симуляции...")
    try:
        cfg = Config()
        bt = Backtester(cfg)
        test_symbols = ['SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT', 'APTUSDT', 'SUIUSDT']
        bt.run_ab_comparison(test_symbols, days=45, timeframe='5')
    except KeyboardInterrupt:
        print("\n🛑 Бэктест прерван пользователем.")
    except Exception as exc:
        print(f"\n❌ Критическая ошибка при выполнении бэктеста: {exc}")
        traceback.print_exc()