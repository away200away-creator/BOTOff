import time
import os
import sys
import json
import threading
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import pandas as pd

from config import Config
from bybit_client import BybitClient
from market_scanner import MarketScanner
from anti_crowd import AntiCrowdAnalyzer
from breakout_engine import BreakoutEngine, FailedBreakoutEvent
from contrarian_engine import ContrarianEngine
from decision_engine import DecisionEngine
from execution_engine import ExecutionEngine
from risk_manager import RiskManager
from ml import ContrarianModelFilter, BreakoutModelFilter

# --- ОДНОСТРОЧНЫЕ ЗАГЛУШКИ ДЛЯ ТЕЛЕГРАМА (ЗАЩИТА ОТ INDENTATION ERROR) ---
def send_telegram_message(msg, reply_markup=None): return None
def start_telegram_menu_thread(): return None
def set_bot_instance(instance): return None

try:
    from telegram_notifier import send_telegram_message, start_telegram_menu_thread, set_bot_instance
except ImportError:
    pass

GLOBAL_BOT_INSTANCE = None

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class BotHttpHandler(BaseHTTPRequestHandler):
    """Локальный веб-сервер для отображения дашборда и динамического управления."""
    
    def log_message(self, format, *args): return None
    
    def do_OPTIONS(self):
        """ОБЯЗАТЕЛЬНЫЙ обработчик для CORS (разблокирует POST-запросы из браузера)."""
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Cache-Control')
        self.end_headers()
        
    def _sanitize_payload(self, data):
        """Глубокая рекурсивная очистка словарей и списков от NaN/Infinity перед JSON сериализацией."""
        if isinstance(data, dict):
            return {k: self._sanitize_payload(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_payload(v) for v in data]
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                return 0.0
            return round(data, 6)
        return data

    def _send_json(self, payload: dict):
        """Безопасная отправка JSON с правильными CORS-заголовками и антикэшем."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.end_headers()
        clean_payload = self._sanitize_payload(payload)
        self.wfile.write(json.dumps(clean_payload, ensure_ascii=False).encode('utf-8'))

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store, no-cache')
            self.end_headers()
            if os.path.exists('dashboard.html'):
                with open('dashboard.html', 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<html><body><h3>dashboard.html not found</h3></body></html>")

        elif path == '/api/status':
            bot = GLOBAL_BOT_INSTANCE
            if bot:
                try:
                    status_data = bot.get_status_payload()
                    self._send_json(status_data)
                except Exception as e:
                    print(f"⚠️ Ошибка API статуса (Дашборд): {e}")
                    self._send_json({'is_running': bot.is_running, 'error': str(e)})
            else:
                self._send_json({'status': 'STOPPED'})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'

        try:
            data = json.loads(body)
        except Exception:
            data = {}

        bot = GLOBAL_BOT_INSTANCE
        if path == '/api/toggle':
            if bot:
                bot.is_running = not bot.is_running
            self._send_json({'success': True, 'is_running': bot.is_running if bot else False})
        elif path == '/api/settings':
            if bot:
                new_score = data.get('minScore') if data.get('minScore') is not None else data.get('min_score')
                if new_score is not None:
                    bot.cfg.MIN_SCORE = int(new_score)
                    bot.cfg.BREAKOUT_MIN_SCORE = int(new_score)
                    bot.contrarian_engine.min_score = int(new_score)
                    bot.breakout_engine.min_score = int(new_score)
                
                new_ml_conf = data.get('mlConfidence') if data.get('mlConfidence') is not None else data.get('ml_min_confidence')
                if new_ml_conf is not None:
                    bot.cfg.ML_MIN_CONFIDENCE = float(new_ml_conf)
                    bot.contrarian_ml.min_confidence = float(new_ml_conf)
                    bot.breakout_ml.min_confidence = float(new_ml_conf)
                    bot.decision_engine.min_ml_confidence = float(new_ml_conf)
            self._send_json({'success': True})
        else:
            self.send_response(404)
            self.end_headers()

def start_background_web_server(port: int = 8000):
    try:
        server = HTTPServer(('0.0.0.0', port), BotHttpHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"🌐 [Web Server] Локальный дашборд доступен: http://localhost:{port}")
    except Exception as e:
        print(f"⚠️ Не удалось запустить веб-сервер на порту {port}: {e}")

class LiveTradingBot:
    """Главный оркестратор институционального комплекса BOtoFF."""

    def __init__(self, config: Config):
        self.cfg = config
        self.client = BybitClient(config)
        self.stream = self.client.order_flow
        self.scanner = MarketScanner(self.client, self.stream, min_turnover_usd=getattr(config, 'MIN_VOLUME_MLN', 15.0) * 1_000_000.0)
        
        self.anti_crowd = AntiCrowdAnalyzer(
            liq_z_threshold=getattr(config, 'LIQ_Z_THRESHOLD', 1.5),
            funding_extreme_pct=getattr(config, 'FUNDING_LONG_THRESH', 0.01),
            ls_heavy_threshold=getattr(config, 'LS_LONG_THRESH', 1.4),
            ls_light_threshold=getattr(config, 'LS_SHORT_THRESH', 0.7),
            ob_imbalance_threshold=getattr(config, 'OB_IMBALANCE_THRESHOLD', 0.15)
        )
        
        self.breakout_engine = BreakoutEngine(
            lookback_bars=getattr(config, 'BREAKOUT_LOOKBACK', 20),
            min_vol_spike=getattr(config, 'BREAKOUT_VOL_SPIKE_THRESH', 2.0),
            min_oi_growth=getattr(config, 'BREAKOUT_OI_GROWTH_THRESH', 1.0),
            min_score=getattr(config, 'BREAKOUT_MIN_SCORE', 50),
            tp1_roe=getattr(config, 'BREAKOUT_TP1_ROE', 4.0),
            stop_roe=getattr(config, 'BREAKOUT_STOP_ROE', -10.0),
            max_24h_pump_for_short=getattr(config, 'MAX_24H_CHANGE_FOR_SHORT', 25.0)
        )
        
        self.contrarian_engine = ContrarianEngine(
            liq_z_thresh=getattr(config, 'LIQ_Z_THRESHOLD', 1.5),
            oi_drop_thresh=getattr(config, 'OI_DROP_THRESHOLD', -0.2),
            min_score=getattr(config, 'MIN_SCORE', 50),
            max_24h_pump_for_short=getattr(config, 'MAX_24H_CHANGE_FOR_SHORT', 25.0),
            tp1_roe=getattr(config, 'TP1_ROE', 3.0),
            stop_roe=getattr(config, 'STOP_ROE', -12.0)
        )

        self.decision_engine = DecisionEngine(config)
        self.execution_engine = ExecutionEngine(config, self.client)
        self.risk = RiskManager(config)

        self.contrarian_ml = ContrarianModelFilter("models/contrarian_filter.cbm", getattr(config, 'ML_MIN_CONFIDENCE', 0.60))
        self.breakout_ml = BreakoutModelFilter("models/breakout_filter.cbm", getattr(config, 'ML_MIN_CONFIDENCE', 0.60))

        # Флаги управления состоянием (Разделение паузы и полного отключения)
        self.is_running = True   # Пауза/Продолжить (работает для новых сделок)
        self._shutdown = False   # Убивает потоки и завершает программу
        
        self.btc_context = {'regime': 'NEUTRAL', 'm5': 0.0, 'price': 0.0}
        self.shortlist: list = []
        self.last_scan_time = 0.0
        self.last_btc_update = 0.0

    def update_btc_context(self):
        now = time.time()
        if now - self.last_btc_update < 15:
            return
        self.last_btc_update = now

        klines = self.client.get_klines("BTCUSDT", interval='5', limit=60)
        if not klines or len(klines) < 30:
            return

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df.sort_values('timestamp', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        price = float(df['close'].iloc[-1])
        prev_p = float(df['close'].iloc[-2])
        m5_pct = ((price - prev_p) / prev_p) * 100.0 if prev_p > 0 else 0.0
        strong_move = getattr(self.cfg, 'BTC_STRONG_MOVE', 1.2)

        regime = "NEUTRAL"
        if m5_pct >= strong_move:
            regime = "BULL_STRONG"
        elif m5_pct <= -strong_move:
            regime = "BEAR_STRONG"

        self.btc_context = {
            'regime': regime,
            'm5': m5_pct,
            'price': price
        }

    def _risk_sentinel_loop(self):
        """Высокочастотный независимый поток управления открытыми позициями (интервал 250 мс)."""
        print("⚡ [Risk Sentinel] Поток контроля рисков запущен (интервал 250 мс).")
        # Поток живет пока не будет включен _shutdown, игнорируя is_running
        while not self._shutdown:
            start_t = time.time()
            
            # 1. Получаем символы текущих позиций без долгого удержания блокировки
            active_symbols = []
            with self.risk.lock:
                active_symbols = [p.symbol for p in self.risk.positions]
            
            # 2. Выполняем медленные сетевые запросы REST ВНЕ блокировки
            live_prices = {}
            for sym in active_symbols:
                ticker = self.stream.get_cached_ticker(sym) if self.stream else {}
                price = float(ticker.get('lastPrice', 0.0))
                if price <= 0:
                    price = self.client.get_last_price(sym)
                live_prices[sym] = price

            # 3. Быстро применяем цены под мьютексом
            with self.risk.lock:
                if self.risk.positions:
                    for pos in self.risk.positions[:]:
                        price = live_prices.get(pos.symbol, 0.0)
                        if price > 0:
                            pos.update_price(price)
                            self.risk.manage_position(pos)

            elapsed = time.time() - start_t
            sleep_time = max(0.05, 0.25 - elapsed)
            time.sleep(sleep_time)

    def scan_market(self):
        print("🔍 [Market Scanner] Запуск воронки отбора монет...")
        self.shortlist = self.scanner.run_funnel()
        print(f"✅ [Market Scanner] Шортлист сформирован: {len(self.shortlist)} активов.")

        for sym in self.shortlist:
            # Защита: Если бот на паузе (is_running=False), не открываем новые сделки
            if not self.is_running or self._shutdown:
                break
                
            if not self.risk.can_open() or self.risk.is_cooldown(sym):
                continue
            if any(p.symbol == sym for p in self.risk.positions):
                continue

            ticker = self.stream.get_cached_ticker(sym) if self.stream else {}
            if not ticker or 'lastPrice' not in ticker:
                last_p = self.client.get_last_price(sym)
                if last_p > 0:
                    ticker = {'symbol': sym, 'lastPrice': str(last_p)}
                else:
                    continue

            klines = self.client.get_klines(sym, interval='5', limit=60)
            if not klines or len(klines) < 35:
                continue

            of_metrics = self.stream.get_order_flow_snapshot(sym) if self.stream else {'buckets': [], 'current_vol': 0.0, 'long_liq_5m': 0.0, 'short_liq_5m': 0.0, 'net_delta': 0.0}
            spread_data = self.client.get_spread_and_depth(sym)
            ob_imb = self.client.get_orderbook_imbalance(sym)

            try:
                oi_change = self.client.get_open_interest(sym)
            except Exception:
                oi_change = 0.0

            df_temp = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            for col in ['open', 'high', 'low', 'close']:
                df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0.0)
            df_temp.sort_values('timestamp', ascending=True, inplace=True)
            df_temp.reset_index(drop=True, inplace=True)

            candle_dir = 1 if float(df_temp['close'].iloc[-1]) >= float(df_temp['open'].iloc[-1]) else -1

            anti_crowd_metrics = self.anti_crowd.analyze(
                symbol=sym,
                liq_history_buckets=of_metrics['buckets'],
                current_liq_vol=of_metrics['current_vol'],
                long_liq_vol=of_metrics['long_liq_5m'],
                short_liq_vol=of_metrics['short_liq_5m'],
                net_delta=of_metrics['net_delta'],
                funding_rate_decimal=float(ticker.get('fundingRate', 0.0)),
                ls_ratio=float(self.client.get_long_short_ratio(sym) or 1.0),
                orderbook_imbalance=ob_imb,
                candle_direction=candle_dir
            )

            breakout_sig, fakeout_event = self.breakout_engine.analyze(
                ticker=ticker,
                klines=klines,
                anti_crowd=anti_crowd_metrics,
                oi_change=oi_change,
                orderbook_imbalance=ob_imb,
                spread_data=spread_data
            )

            contrarian_sig = None

            if fakeout_event:
                contrarian_sig = self.contrarian_engine.evaluate_failed_breakout(
                    fakeout=fakeout_event,
                    anti_crowd=anti_crowd_metrics,
                    ticker=ticker,
                    spread_data=spread_data
                )
            else:
                contrarian_sig = self.contrarian_engine.analyze(
                    ticker=ticker,
                    klines=klines,
                    anti_crowd=anti_crowd_metrics,
                    oi_change=oi_change,
                    orderbook_imbalance=ob_imb,
                    spread_data=spread_data
                )

            if contrarian_sig:
                feat_c = {
                    'direction': 1 if contrarian_sig.direction == 'LONG' else 0,
                    'score': contrarian_sig.score,
                    'rsi': contrarian_sig.rsi,
                    'm5_change': contrarian_sig.m5_change,
                    'atr_pct': contrarian_sig.atr_pct,
                    'funding_pct': contrarian_sig.funding_pct,
                    'delta_oi_pct': contrarian_sig.delta_oi_pct,
                    'ls_ratio': contrarian_sig.ls_ratio,
                    'liq_z_score': contrarian_sig.liq_z_score,
                    'net_delta': contrarian_sig.net_delta,
                    'orderbook_imbalance': contrarian_sig.orderbook_imbalance,
                    'is_fakeout_play': 1 if contrarian_sig.is_fakeout_play else 0,
                    'btc_regime': self.btc_context.get('regime', 'NEUTRAL')
                }
                contrarian_sig.win_prob = self.contrarian_ml.predict_win_probability(feat_c)

            if breakout_sig:
                feat_b = {
                    'direction': 1 if breakout_sig.direction == 'LONG' else 0,
                    'score': breakout_sig.score,
                    'volume_spike': breakout_sig.volume_spike,
                    'delta_oi_pct': breakout_sig.delta_oi_pct,
                    'net_delta': breakout_sig.net_delta,
                    'orderbook_imbalance': breakout_sig.orderbook_imbalance,
                    'atr_pct': breakout_sig.atr_pct,
                    'btc_regime': self.btc_context.get('regime', 'NEUTRAL')
                }
                breakout_sig.win_prob = self.breakout_ml.predict_win_probability(feat_b)

            decision = self.decision_engine.evaluate(
                symbol=sym,
                contrarian_sig=contrarian_sig,
                breakout_sig=breakout_sig,
                btc_context=self.btc_context
            )

            if decision.action == 'TRADE':
                exec_result = self.execution_engine.execute_decision(
                    decision=decision,
                    margin=self.risk.calculate_margin(),
                    leverage=self.cfg.LEVERAGE,
                    ticker=ticker,
                    spread_data=spread_data
                )
                if exec_result and exec_result.success:
                    signal_dict = self.execution_engine.to_signal_dict(exec_result)
                    self.risk.open_position(signal_dict)

            time.sleep(self.cfg.RATE_LIMIT_PAUSE)

    def get_status_payload(self) -> dict:
        with self.risk.lock:
            positions_data = []
            for p in self.risk.positions:
                positions_data.append({
                    'id': getattr(p, 'id', 'N/A'),
                    'symbol': getattr(p, 'symbol', 'UNKNOWN'),
                    'strategy_id': getattr(p, 'strategy_id', 'CONTRARIAN'),
                    'direction': getattr(p, 'direction', 'UNKNOWN'),
                    'entry_price': getattr(p, 'entry_price', 0.0),
                    'current_price': getattr(p, 'current_price', 0.0),
                    'margin': getattr(p, 'margin', 0.0),
                    'notional': getattr(p, 'notional', 0.0),
                    'leverage': int(getattr(p, 'leverage', 1)),
                    'roe': p.get_roe() if hasattr(p, 'get_roe') else 0.0,
                    'pnl': p.get_unrealized_pnl() if hasattr(p, 'get_unrealized_pnl') else 0.0,
                    'score': int(getattr(p, 'score', 0))
                })

            history_data = self.risk.history[-25:][::-1]
            total_pnl = sum(t.get('pnl', 0.0) for t in self.risk.history)
            wins = sum(1 for t in self.risk.history if t.get('pnl', 0.0) > 0)
            total_trades = len(self.risk.history)
            winrate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

            settings_data = {
                'minScore': getattr(self.cfg, 'MIN_SCORE', 38),
                'breakoutMinScore': getattr(self.cfg, 'BREAKOUT_MIN_SCORE', 38),
                'tp1Roe': getattr(self.cfg, 'TP1_ROE', 3.0),
                'stopRoe': getattr(self.cfg, 'STOP_ROE', -12.0),
                'leverage': getattr(self.cfg, 'LEVERAGE', 3),
                'riskPerTrade': getattr(self.cfg, 'RISK_PER_TRADE', 2.0),
                'mlConfidence': getattr(self.cfg, 'ML_MIN_CONFIDENCE', 0.54),
                'beTriggerRoe': getattr(self.cfg, 'BE_TRIGGER_ROE', 0.5),
                'atrMult': getattr(self.cfg, 'TRAILING_ATR_MULT', 1.0),
                'slippage': getattr(self.cfg, 'SLIPPAGE_PCT', 0.08)
            }

            return {
                'is_running': self.is_running,
                'balance': self.risk.balance,
                'total_pnl': total_pnl,
                'winrate': winrate,
                'total_trades': total_trades,
                'positions': positions_data,
                'history': history_data,
                'shortlist': self.shortlist,
                'btc_context': self.btc_context,
                'settings': settings_data
            }

    def start(self):
        clear_screen()
        print("=" * 85)
        c_status = f"✅ Загружена ({len(self.contrarian_ml.feature_names)} призн.)" if self.contrarian_ml.is_loaded else "⚠️ РЕЖИМ БЕЗ ВЕСОВ"
        b_status = f"✅ Загружена ({len(self.breakout_ml.feature_names)} призн.)" if self.breakout_ml.is_loaded else "⚠️ РЕЖИМ БЕЗ ВЕСОВ"
        print(f"🧠 [ML Contrarian] Модель: {self.contrarian_ml.model_path} | Загружена: {c_status}")
        print(f"🧠 [ML Breakout]   Модель: {self.breakout_ml.model_path}   | Загружена: {b_status}")
        ws_url = self.stream.WS_URL if self.stream else "Отключен"
        print(f"🌊 [WebSocket]     Шлюз Bybit V5 Linear: {ws_url}")
        print(f"🎯 [Воронка]       Universe USDT Linear ➔ Шортлист (20–40 монет)")
        print(f"⚖️ [Decision]      Mutual Exclusion Guard + max(P_win) Confluence + BTC Regime Filter")
        print(f"⚡ [Risk Sentinel] Изолированный поток: опрос позиций каждые 250 мс")
        print(f"💼 [Депозит]       Баланс: {self.cfg.INITIAL_BALANCE:.2f}$ | Плечо: {self.cfg.LEVERAGE}x | Слотов: {self.cfg.MAX_POSITIONS}")
        print("=" * 85 + "\n")

        start_background_web_server(8000)
        start_telegram_menu_thread()

        sentinel_thread = threading.Thread(target=self._risk_sentinel_loop, daemon=True)
        sentinel_thread.start()

        # Цикл не умирает при is_running=False, он просто ждет
        while not self._shutdown:
            try:
                # Если поставлен на паузу, просто спим и ждем команды "Продолжить"
                if not self.is_running:
                    time.sleep(1.0)
                    continue

                self.update_btc_context()
                self.scan_market()
                time.sleep(self.cfg.SCAN_INTERVAL_SEC)
            except KeyboardInterrupt:
                print("\n🛑 Остановка комплекса пользователем.")
                self._shutdown = True
                self.is_running = False
                break
            except Exception as exc:
                print(f"⚠️ Ошибка в главном цикле оркестратора: {exc}")
                time.sleep(5.0)

def main_menu():
    global GLOBAL_BOT_INSTANCE
    cfg = Config()

    while True:
        clear_screen()
        print("=" * 65)
        print("          ІНСТИТУЦІОНАЛЬНИЙ КОМПЛЕКС BOtoFF (QUANT V5)          ")
        print("=" * 65)
        print("  [1] Запуск Multi-Strategy Live Paper-Trading (Contrarian + Breakout)")
        print("  [2] Запуск бэктестера стратегий (backtester.py)")
        print("  [3] Запуск объективного майнинга датасетов (data_collector.py)")
        print("  [4] Обучение моделей CatBoost (train_contrarian & train_breakout)")
        print("  [5] Выход")
        print("=" * 65)
        choice = input("Выберите пункт меню [1-5]: ").strip()

        if choice == '1':
            bot = LiveTradingBot(cfg)
            GLOBAL_BOT_INSTANCE = bot
            set_bot_instance(bot)
            bot.start()
        elif choice == '2':
            os.system('python backtester.py')
            input("\nНажмите Enter для возврата в меню...")
        elif choice == '3':
            os.system('python data_collector.py --days 30')
            input("\nНажмите Enter для возврата в меню...")
        elif choice == '4':
            os.system('python ml/train_contrarian.py')
            os.system('python ml/train_breakout.py')
            input("\nНажмите Enter для возврата в меню...")
        elif choice == '5':
            print("До свидания!")
            sys.exit(0)

if __name__ == "__main__":
    main_menu()