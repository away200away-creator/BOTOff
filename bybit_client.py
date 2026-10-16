import requests
import time
import json
import threading
import math
import hmac
import hashlib
import os
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple, Any
import websocket
from config import Config
from dotenv import load_dotenv

load_dotenv()

class BybitOrderFlowTracker:
    WS_URL = "wss://stream.bybit.com/v5/public/linear"
    ANCHOR_SYMBOLS = ("BTCUSDT", "ETHUSDT")

    def __init__(self, window_minutes: int = 15, max_active_subs: int = 70):
        self.window_minutes = window_minutes
        self.max_active_subs = max_active_subs
        
        self.tickers_cache: Dict[str, Dict[str, Any]] = {}
        self.liq_history = defaultdict(lambda: deque(maxlen=400))
        self.liq_minute_buckets = defaultdict(lambda: deque(maxlen=self.window_minutes))
        self.current_bucket_min = defaultdict(int)
        self.current_bucket_vol = defaultdict(float)

        self.long_liq_vol_5m = defaultdict(lambda: deque(maxlen=300))
        self.short_liq_vol_5m = defaultdict(lambda: deque(maxlen=300))

        self.delta_history = defaultdict(float)
        self.delta_last_reset = defaultdict(float)

        self.subscribed_symbols = set(self.ANCHOR_SYMBOLS)
        self.ws = None
        self.ws_thread = None
        self.is_connected = False
        self.lock = threading.RLock()

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
            self._send_batch_subscribe(syms)

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
            topic = msg.get('topic', '')
            data = msg.get('data', None)
            if not data:
                return

            now = time.time()
            current_m = int(now // 60)

            if topic.startswith('tickers.'):
                sym = topic.split('.')[1]
                t_item = data if isinstance(data, dict) else (data[0] if isinstance(data, list) and data else {})
                with self.lock:
                    if sym not in self.tickers_cache:
                        self.tickers_cache[sym] = {}
                    self.tickers_cache[sym].update(t_item)
                    self.tickers_cache[sym]['local_timestamp'] = now

            elif topic.startswith('allLiquidation.'):
                sym = topic.split('.')[1]
                items = data if isinstance(data, list) else [data]
                for item in items:
                    size = float(item.get('v', item.get('size', 0)))
                    price = float(item.get('p', item.get('price', 0)))
                    side = item.get('S', item.get('side', ''))
                    vol_usd = size * price if price > 0 else size

                    with self.lock:
                        if side == 'Buy':
                            self.long_liq_vol_5m[sym].append((now, vol_usd))
                        elif side == 'Sell':
                            self.short_liq_vol_5m[sym].append((now, vol_usd))

                        self.liq_history[sym].append((now, vol_usd, side))
                        
                        if self.current_bucket_min[sym] != current_m:
                            if self.current_bucket_min[sym] > 0:
                                elapsed_mins = current_m - self.current_bucket_min[sym]
                                self.liq_minute_buckets[sym].append(self.current_bucket_vol[sym])
                                for _ in range(min(elapsed_mins - 1, self.window_minutes)):
                                    self.liq_minute_buckets[sym].append(0.0)
                            self.current_bucket_min[sym] = current_m
                            self.current_bucket_vol[sym] = 0.0

                        self.current_bucket_vol[sym] += vol_usd

            elif topic.startswith('publicTrade.'):
                sym = topic.split('.')[1]
                items = data if isinstance(data, list) else [data]
                with self.lock:
                    if now - self.delta_last_reset[sym] > 300:
                        self.delta_history[sym] = 0.0
                        self.delta_last_reset[sym] = now

                    for trade in items:
                        v = float(trade.get('v', 0))
                        p = float(trade.get('p', 0))
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

    def _send_batch_subscribe(self, symbols: list):
        if not self.ws or not self.is_connected or not symbols:
            return
        batch_size = 10
        topics = []
        for s in symbols:
            topics.append(f"tickers.{s}")
            topics.append(f"allLiquidation.{s}")
            topics.append(f"publicTrade.{s}")

        for i in range(0, len(topics), batch_size):
            chunk = topics[i:i + batch_size]
            payload = {"op": "subscribe", "args": chunk}
            try:
                self.ws.send(json.dumps(payload))
                time.sleep(0.05)
            except Exception:
                pass

    def _send_batch_unsubscribe(self, symbols: list):
        if not self.ws or not self.is_connected or not symbols:
            return
        batch_size = 10
        topics = []
        for s in symbols:
            if s not in self.ANCHOR_SYMBOLS:
                topics.append(f"tickers.{s}")
                topics.append(f"allLiquidation.{s}")
                topics.append(f"publicTrade.{s}")

        for i in range(0, len(topics), batch_size):
            chunk = topics[i:i + batch_size]
            payload = {"op": "unsubscribe", "args": chunk}
            try:
                self.ws.send(json.dumps(payload))
                time.sleep(0.05)
            except Exception:
                pass

    def sync_tracked_symbols(self, symbols: list):
        target_set = (set(symbols[:self.max_active_subs]) | set(self.ANCHOR_SYMBOLS))
        with self.lock:
            to_remove = list((self.subscribed_symbols - target_set) - set(self.ANCHOR_SYMBOLS))
            to_add = list(target_set - self.subscribed_symbols)

            if to_remove:
                self._send_batch_unsubscribe(to_remove)
                for s in to_remove:
                    self.subscribed_symbols.discard(s)

            if to_add:
                self._send_batch_subscribe(to_add)
                for s in to_add:
                    self.subscribed_symbols.add(s)

    def sync_tracked_universe(self, symbols: list):
        self.sync_tracked_symbols(symbols)

    def get_cached_ticker(self, symbol: str) -> dict:
        with self.lock:
            return dict(self.tickers_cache.get(symbol, {}))

    def get_order_flow_snapshot(self, symbol: str) -> dict:
        now = time.time()
        current_m = int(now // 60)

        with self.lock:
            if self.current_bucket_min[symbol] > 0 and self.current_bucket_min[symbol] != current_m:
                elapsed_mins = current_m - self.current_bucket_min[symbol]
                self.liq_minute_buckets[symbol].append(self.current_bucket_vol[symbol])
                for _ in range(min(elapsed_mins - 1, self.window_minutes)):
                    self.liq_minute_buckets[symbol].append(0.0)
                self.current_bucket_min[symbol] = current_m
                self.current_bucket_vol[symbol] = 0.0

            buckets = list(self.liq_minute_buckets[symbol])
            current_vol = self.current_bucket_vol[symbol]
            delta = self.delta_history[symbol]

            long_liqs = sum(v for t, v in self.long_liq_vol_5m[symbol] if now - t <= 300)
            short_liqs = sum(v for t, v in self.short_liq_vol_5m[symbol] if now - t <= 300)

        return {
            'buckets': buckets,
            'current_vol': current_vol,
            'long_liq_5m': long_liqs,
            'short_liq_5m': short_liqs,
            'net_delta': delta
        }

    def _calculate_series_z_score(self, buckets: list, current_val: float) -> float:
        if not buckets and current_val <= 0:
            return 0.0
        
        full_sample = list(buckets) if buckets else [0.0] * 5
        mean_val = sum(full_sample) / len(full_sample)
        variance = sum((x - mean_val) ** 2 for x in full_sample) / len(full_sample)
        std_val = math.sqrt(variance)

        if std_val == 0.0:
            if current_val > 1000.0:
                return 2.5
            return 0.0

        return round((current_val - mean_val) / std_val, 2)

    def get_order_flow_metrics(self, symbol: str) -> dict:
        now = time.time()
        current_m = int(now // 60)

        with self.lock:
            if self.current_bucket_min[symbol] > 0 and self.current_bucket_min[symbol] != current_m:
                elapsed_mins = current_m - self.current_bucket_min[symbol]
                self.liq_minute_buckets[symbol].append(self.current_bucket_vol[symbol])
                for _ in range(min(elapsed_mins - 1, self.window_minutes)):
                    self.liq_minute_buckets[symbol].append(0.0)
                self.current_bucket_min[symbol] = current_m
                self.current_bucket_vol[symbol] = 0.0

            buckets = list(self.liq_minute_buckets[symbol])
            current_vol = self.current_bucket_vol[symbol]
            delta = self.delta_history[symbol]

            recent_liqs = [item for item in self.liq_history[symbol] if now - item[0] <= 300]
            long_liq_vol = sum(item[1] for item in recent_liqs if item[2] == 'Buy')
            short_liq_vol = sum(item[1] for item in recent_liqs if item[2] == 'Sell')

            btc_buckets = list(self.liq_minute_buckets['BTCUSDT'])
            btc_vol = self.current_bucket_vol['BTCUSDT']

        z_score = self._calculate_series_z_score(buckets, current_vol)

        if z_score == 0.0 and len(btc_buckets) >= 2:
            btc_z = self._calculate_series_z_score(btc_buckets, btc_vol)
            if btc_z >= 1.5:
                z_score = round(btc_z * 0.85, 2)

        has_of = len(buckets) >= 2 or len(btc_buckets) >= 2 or current_vol > 0

        return {
            'liq_z_score': z_score,
            'long_liq_vol': long_liq_vol,
            'short_liq_vol': short_liq_vol,
            'net_delta': round(delta, 2),
            'has_order_flow': has_of
        }

class BybitClient:
    BASE_URL = "https://api.bybit.com"

    def __init__(self, config: Config):
        self.config = config
        self.timeout = (5, 10)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })

        self.api_key = os.getenv("BYBIT_API_KEY", "").strip(' "\'')
        self.api_secret = os.getenv("BYBIT_API_SECRET", "").strip(' "\'')
        self.is_authenticated = bool(self.api_key and self.api_secret and len(self.api_key) > 5)

        self._last_request_time = 0.0
        self._min_request_interval = 0.04 if self.is_authenticated else 0.10

        self.order_flow = None
        if getattr(config, 'ENABLE_ORDER_FLOW', True):
            self.order_flow = BybitOrderFlowTracker(
                window_minutes=getattr(config, 'LIQ_WINDOW', 15),
                max_active_subs=getattr(config, 'ORDER_FLOW_MAX_SUBS', 70)
            )
            self.order_flow.start()

    def _sign_headers(self, params_str: str) -> dict:
        if not self.is_authenticated:
            return {}

        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        param_to_sign = timestamp + self.api_key + recv_window + params_str

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_to_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': recv_window
        }

    def _get(self, path: str, params: dict, auth: bool = False):
        url = self.BASE_URL + path
        max_retries = getattr(self.config, 'MAX_RETRIES', 3)
        
        for attempt in range(max_retries):
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)

            try:
                self._last_request_time = time.time()
                
                headers = {}
                if auth and self.is_authenticated:
                    sorted_params = sorted(params.items()) if params else []
                    query_str = "&".join(f"{k}={v}" for k, v in sorted_params)
                    headers = self._sign_headers(query_str)

                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)

                if resp.status_code == 200:
                    data = resp.json()
                    ret_code = data.get('retCode')
                    ret_msg = data.get('retMsg', '')

                    if ret_code == 0:
                        return data.get('result', {})
                    elif ret_code in (10006, 10018) or 'Rate Limit' in ret_msg or 'Too many visits' in ret_msg:
                        backoff = 2.0 * (attempt + 1)
                        time.sleep(backoff)
                        continue
                    else:
                        return None
                elif resp.status_code == 429:
                    time.sleep(2.5 * (attempt + 1))
                    continue
                else:
                    return None
            except requests.exceptions.RequestException:
                if attempt == max_retries - 1:
                    return None
                time.sleep(1.0)
        return None

    def get_wallet_balance_usdt(self) -> float:
        if not self.is_authenticated:
            return 0.0

        res = self._get('/v5/account/wallet-balance', {'accountType': 'UNIFIED', 'coin': 'USDT'}, auth=True)
        if res and 'list' in res and len(res['list']) > 0:
            coins = res['list'][0].get('coin', [])
            for c in coins:
                if c.get('coin') == 'USDT':
                    try:
                        return float(c.get('walletBalance', 0.0))
                    except (ValueError, TypeError):
                        pass
        return 0.0

    def get_last_price(self, symbol: str) -> float:
        res = self._get('/v5/market/tickers', {'category': 'linear', 'symbol': symbol})
        if res and 'list' in res and len(res['list']) > 0:
            try:
                return float(res['list'][0].get('lastPrice', 0))
            except (ValueError, TypeError):
                pass
        return 0.0

    def get_spread_and_depth(self, symbol: str) -> dict:
        res = self._get('/v5/market/tickers', {'category': 'linear', 'symbol': symbol})
        if res and 'list' in res and len(res['list']) > 0:
            t = res['list'][0]
            try:
                bid1 = float(t.get('bid1Price', 0.0))
                ask1 = float(t.get('ask1Price', 0.0))
                last = float(t.get('lastPrice', 0.0))
                return {
                    'bid': bid1 if bid1 > 0 else last,
                    'ask': ask1 if ask1 > 0 else last,
                    'last': last
                }
            except Exception:
                pass
        return {'bid': 0.0, 'ask': 0.0, 'last': 0.0}

    def get_orderbook_imbalance(self, symbol: str, depth_pct: float = 0.8) -> float:
        res = self._get('/v5/market/orderbook', {
            'category': 'linear',
            'symbol': symbol,
            'limit': 50
        })
        if not res or 'b' not in res or 'a' not in res:
            return 0.0

        bids = res.get('b', [])
        asks = res.get('a', [])
        if not bids or not asks:
            return 0.0

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2.0
            if mid_price <= 0:
                return 0.0

            lower_bound = mid_price * (1.0 - depth_pct / 100.0)
            upper_bound = mid_price * (1.0 + depth_pct / 100.0)

            bid_volume = 0.0
            for price_str, size_str in bids:
                p = float(price_str)
                s = float(size_str)
                if p >= lower_bound:
                    bid_volume += p * s
                else:
                    break

            ask_volume = 0.0
            for price_str, size_str in asks:
                p = float(price_str)
                s = float(size_str)
                if p <= upper_bound:
                    ask_volume += p * s
                else:
                    break

            total_vol = bid_volume + ask_volume
            if total_vol <= 0:
                return 0.0

            imbalance = (bid_volume - ask_volume) / total_vol
            return round(imbalance, 3)
        except Exception:
            return 0.0

    def get_klines(self, symbol: str, interval: str = '5', limit: int = 200):
        res = self._get('/v5/market/kline', {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        })
        if res and 'list' in res:
            return res['list']
        return None

    def get_klines_range(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[list]:
        all_klines = []
        cur_end = end_time

        while cur_end > start_time:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': str(interval),
                'end': cur_end,
                'limit': 200
            }
            res = self._get('/v5/market/kline', params)
            if not res or 'list' not in res or not res['list']:
                break

            chunk = res['list']
            for candle in chunk:
                ts = int(candle[0])
                if start_time <= ts <= end_time:
                    all_klines.append(candle)

            oldest_ts = int(chunk[-1][0])
            if oldest_ts <= start_time or oldest_ts >= cur_end:
                break
            cur_end = oldest_ts - 1
            time.sleep(self._min_request_interval)

        seen_ts = set()
        unique_klines = []
        for c in all_klines:
            ts = int(c[0])
            if ts not in seen_ts:
                seen_ts.add(ts)
                unique_klines.append(c)

        unique_klines.sort(key=lambda x: int(x[0]), reverse=True)
        return unique_klines

    def get_historical_open_interest_range(self, symbol: str, start_time: int, end_time: int, period: str = '5min') -> List[Tuple[int, float]]:
        records = []
        cur_end = end_time

        while cur_end > start_time:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'intervalTime': period,
                'endTime': cur_end,
                'limit': 200
            }
            res = self._get('/v5/market/open-interest', params)
            if not res or 'list' not in res or not res['list']:
                break

            chunk = res['list']
            for item in chunk:
                ts = int(item.get('timestamp', 0))
                oi = float(item.get('openInterest', 0.0))
                if start_time <= ts <= end_time:
                    records.append((ts, oi))

            oldest_ts = int(chunk[-1].get('timestamp', 0))
            if oldest_ts <= start_time or oldest_ts >= cur_end:
                break
            cur_end = oldest_ts - 1
            time.sleep(self._min_request_interval)

        records_map = {ts: oi for ts, oi in records}
        sorted_records = sorted(records_map.items(), key=lambda x: x[0])
        return sorted_records

    def get_historical_long_short_ratio_range(self, symbol: str, start_time: int, end_time: int, period: str = '15min') -> List[Tuple[int, float]]:
        records = []
        cur_end = end_time

        while cur_end > start_time:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'period': period,
                'endTime': cur_end,
                'limit': 500
            }
            res = self._get('/v5/market/account-ratio', params)
            if not res or 'list' not in res or not res['list']:
                break

            chunk = res['list']
            for item in chunk:
                ts = int(item.get('timestamp', 0))
                ratio = float(item.get('buyRatio', 1.0))
                if start_time <= ts <= end_time:
                    records.append((ts, ratio))

            oldest_ts = int(chunk[-1].get('timestamp', 0))
            if oldest_ts <= start_time or oldest_ts >= cur_end:
                break
            cur_end = oldest_ts - 1
            time.sleep(self._min_request_interval)

        records_map = {ts: ratio for ts, ratio in records}
        sorted_records = sorted(records_map.items(), key=lambda x: x[0])
        return sorted_records

    def get_historical_funding_rate(self, symbol: str, limit: int = 200) -> List[Tuple[int, float]]:
        res = self._get('/v5/market/funding/history', {
            'category': 'linear',
            'symbol': symbol,
            'limit': min(200, limit)
        })
        if not res or 'list' not in res:
            return []

        result = []
        for item in res['list']:
            ts = int(item.get('fundingRateTimestamp', 0))
            rate = float(item.get('fundingRate', 0.0))
            result.append((ts, rate))

        result.sort(key=lambda x: x[0])
        return result

    def get_tickers(self, symbols=None):
        params = {'category': 'linear'}
        if symbols:
            if isinstance(symbols, str):
                params['symbol'] = symbols
            elif len(symbols) == 1:
                params['symbol'] = symbols[0]

        res = self._get('/v5/market/tickers', params)
        if res and 'list' in res:
            if symbols and isinstance(symbols, (list, tuple)) and len(symbols) > 1:
                sym_set = set(symbols)
                return [t for t in res['list'] if t.get('symbol') in sym_set]
            return res['list']
        return []

    def get_open_interest(self, symbol: str, period: str = '5min', limit: int = 4):
        res = self._get('/v5/market/open-interest', {
            'category': 'linear',
            'symbol': symbol,
            'intervalTime': period,
            'limit': limit
        })
        if res and 'list' in res and len(res['list']) >= 2:
            data = res['list'][::-1]
            old_oi = float(data[-2]['openInterest'])
            new_oi = float(data[-1]['openInterest'])
            if old_oi != 0:
                return (new_oi / old_oi - 1) * 100
        return None

    def get_long_short_ratio(self, symbol: str, period: str = '5min'):
        res = self._get('/v5/market/account-ratio', {
            'category': 'linear',
            'symbol': symbol,
            'period': period,
            'limit': 1
        })
        if res and 'list' in res and len(res['list']) > 0:
            return float(res['list'][0]['buyRatio'])
        return None

    def get_order_flow_data(self, symbol: str) -> dict:
        if self.order_flow:
            return self.order_flow.get_order_flow_metrics(symbol)
        return {'liq_z_score': 0.0, 'long_liq_vol': 0.0, 'short_liq_vol': 0.0, 'net_delta': 0.0, 'has_order_flow': False}

    def sync_order_flow_pool(self, symbols: list):
        if self.order_flow:
            self.order_flow.sync_tracked_symbols(symbols)