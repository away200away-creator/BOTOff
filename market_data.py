import time
import json
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set, Any
import websocket

class MarketDataStream:
    """
    Высокочастотный потоковый шлюз Bybit V5 Linear WebSocket.
    Поддерживает динамическую подписку чанками по 15 тикеров для предотвращения
    ошибки 'Too many visits' и переполнения входного сокета.
    """
    WS_URL = "wss://stream.bybit.com/v5/public/linear"
    ANCHOR_SYMBOLS = ("BTCUSDT", "ETHUSDT")

    def __init__(self, liq_window_minutes: int = 15):
        self.liq_window_minutes = liq_window_minutes
        self.lock = threading.RLock()
        
        # In-Memory кэши биржевого клиринга
        self.tickers_cache: Dict[str, Dict[str, Any]] = {}
        self.liq_minute_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.liq_window_minutes))
        self.current_bucket_min: Dict[str, int] = defaultdict(int)
        self.current_bucket_vol: Dict[str, float] = defaultdict(float)
        
        self.long_liq_vol_5m: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        self.short_liq_vol_5m: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))
        
        self.delta_history: Dict[str, float] = defaultdict(float)
        self.delta_last_reset: Dict[str, float] = defaultdict(float)

        self.subscribed_symbols: Set[str] = set(self.ANCHOR_SYMBOLS)
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.is_connected = False
        self._last_ping_time = time.time()

    def start(self):
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={'ping_interval': 20, 'ping_timeout': 10, 'reconnect': 5},
            daemon=True
        )
        self.ws_thread.start()

    def _on_open(self, ws):
        self.is_connected = True
        with self.lock:
            syms = list(self.subscribed_symbols)
            self._batch_subscribe(syms)

    def _on_message(self, ws, message: str):
        try:
            msg = json.loads(message)
            topic = msg.get('topic', '')
            data = msg.get('data', None)
            if not data:
                return

            now = time.time()
            current_m = int(now // 60)

            # 1. Поток тикеров (lastPrice, turnover24h, fundingRate, bid1/ask1)
            if topic.startswith('tickers.'):
                sym = topic.split('.')[1]
                t_item = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})
                with self.lock:
                    if sym not in self.tickers_cache:
                        self.tickers_cache[sym] = {}
                    self.tickers_cache[sym].update(t_item)
                    self.tickers_cache[sym]['local_timestamp'] = now

            # 2. Поток ликвидаций розницы (allLiquidation)
            elif topic.startswith('allLiquidation.'):
                sym = topic.split('.')[1]
                items = data if isinstance(data, list) else [data]
                for item in items:
                    size = float(item.get('v', item.get('size', 0.0)))
                    price = float(item.get('p', item.get('price', 0.0)))
                    side = item.get('S', item.get('side', ''))
                    vol_usd = size * price if price > 0 else size

                    with self.lock:
                        # Bybit V5: S == 'Buy' -> ликвидирован LONG; S == 'Sell' -> ликвидирован SHORT
                        if side == 'Buy':
                            self.long_liq_vol_5m[sym].append((now, vol_usd))
                        elif side == 'Sell':
                            self.short_liq_vol_5m[sym].append((now, vol_usd))

                        if self.current_bucket_min[sym] != current_m:
                            if self.current_bucket_min[sym] > 0:
                                elapsed = current_m - self.current_bucket_min[sym]
                                self.liq_minute_buckets[sym].append(self.current_bucket_vol[sym])
                                for _ in range(min(elapsed - 1, self.liq_window_minutes)):
                                    self.liq_minute_buckets[sym].append(0.0)
                            self.current_bucket_min[sym] = current_m
                            self.current_bucket_vol[sym] = 0.0

                        self.current_bucket_vol[sym] += vol_usd

            # 3. Поток публичных сделок (тиковая дельта CVD)
            elif topic.startswith('publicTrade.'):
                sym = topic.split('.')[1]
                items = data if isinstance(data, list) else [data]
                with self.lock:
                    if now - self.delta_last_reset[sym] > 300:
                        self.delta_history[sym] = 0.0
                        self.delta_last_reset[sym] = now

                    for trade in items:
                        v = float(trade.get('v', 0.0))
                        p = float(trade.get('p', 0.0))
                        vol_usd = v * p if p > 0 else v
                        side = trade.get('S', '')
                        if side == 'Buy':
                            self.delta_history[sym] += vol_usd
                        elif side == 'Sell':
                            self.delta_history[sym] -= vol_usd

        except Exception:
            pass

    def _on_error(self, ws, error):
        pass

    def _on_close(self, ws, close_status_code, close_msg):
        self.is_connected = False

    def _batch_subscribe(self, symbols: List[str]):
        """Пакетная отправка подписок чанками по 15 штук с паузой 50 мс."""
        if not self.ws or not self.is_connected or not symbols:
            return
        
        topics = []
        for s in symbols:
            topics.append(f"tickers.{s}")
            topics.append(f"allLiquidation.{s}")
            topics.append(f"publicTrade.{s}")

        chunk_size = 15
        for i in range(0, len(topics), chunk_size):
            chunk = topics[i:i + chunk_size]
            payload = {"op": "subscribe", "args": chunk}
            try:
                self.ws.send(json.dumps(payload))
                time.sleep(0.05)
            except Exception:
                pass

    def _batch_unsubscribe(self, symbols: List[str]):
        """Пакетная отписка от тикеров, покинувших Shortlist."""
        if not self.ws or not self.is_connected or not symbols:
            return
        
        topics = []
        for s in symbols:
            if s not in self.ANCHOR_SYMBOLS:
                topics.append(f"tickers.{s}")
                topics.append(f"allLiquidation.{s}")
                topics.append(f"publicTrade.{s}")

        chunk_size = 15
        for i in range(0, len(topics), chunk_size):
            chunk = topics[i:i + chunk_size]
            payload = {"op": "unsubscribe", "args": chunk}
            try:
                self.ws.send(json.dumps(payload))
                time.sleep(0.05)
            except Exception:
                pass

    def sync_tracked_universe(self, shortlisted_symbols: List[str]):
        """Динамическая синхронизация подписок под свежий Shortlist воронки."""
        target_set = set(shortlisted_symbols) | set(self.ANCHOR_SYMBOLS)
        with self.lock:
            to_remove = list((self.subscribed_symbols - target_set) - set(self.ANCHOR_SYMBOLS))
            to_add = list(target_set - self.subscribed_symbols)

            if to_remove:
                self._batch_unsubscribe(to_remove)
                for s in to_remove:
                    self.subscribed_symbols.discard(s)

            if to_add:
                self._batch_subscribe(to_add)
                for s in to_add:
                    self.subscribed_symbols.add(s)

    def get_order_flow_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Мгновенный срез Order Flow из оперативной памяти без сетевых задержек."""
        now = time.time()
        current_m = int(now // 60)

        with self.lock:
            # Ролловер бакетов при отсутствии ликвидаций
            if self.current_bucket_min[symbol] > 0 and self.current_bucket_min[symbol] != current_m:
                elapsed = current_m - self.current_bucket_min[symbol]
                self.liq_minute_buckets[symbol].append(self.current_bucket_vol[symbol])
                for _ in range(min(elapsed - 1, self.liq_window_minutes)):
                    self.liq_minute_buckets[symbol].append(0.0)
                self.current_bucket_min[symbol] = current_m
                self.current_bucket_vol[symbol] = 0.0

            buckets = list(self.liq_minute_buckets[symbol])
            curr_vol = self.current_bucket_vol[symbol]
            delta = self.delta_history[symbol]

            # Сумма ликвидаций за последние 5 минут
            long_liqs = sum(v for t, v in self.long_liq_vol_5m[symbol] if now - t <= 300)
            short_liqs = sum(v for t, v in self.short_liq_vol_5m[symbol] if now - t <= 300)

        return {
            'buckets': buckets,
            'current_vol': curr_vol,
            'long_liq_5m': long_liqs,
            'short_liq_5m': short_liqs,
            'net_delta': delta
        }

    def get_cached_ticker(self, symbol: str) -> Dict[str, Any]:
        with self.lock:
            return dict(self.tickers_cache.get(symbol, {}))