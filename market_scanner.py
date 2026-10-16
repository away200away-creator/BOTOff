import time
import logging
from typing import List, Dict, Any, Set, Optional
from bybit_client import BybitClient
from market_data import MarketDataStream

logger = logging.getLogger("MarketScanner")

class MarketScanner:
    """
    Институциональная динамическая воронка рынка Bybit V5 Linear.
    Universe (все USDT бессрочные) ──► Фильтр ликвидности ──► Фильтр спреда ──► Dynamic Shortlist (20-40)
    """

    def __init__(
        self,
        bybit_client: BybitClient,
        market_data_stream: MarketDataStream,
        min_turnover_usd: float = 15_000_000.0,
        max_spread_pct: float = 0.08,
        min_shortlist_size: int = 20,
        max_shortlist_size: int = 40,
        blacklist: Optional[List[str]] = None
    ):
        self.client = bybit_client
        self.stream = market_data_stream
        self.min_turnover_usd = min_turnover_usd
        self.max_spread_pct = max_spread_pct
        self.min_shortlist_size = min_shortlist_size
        self.max_shortlist_size = max_shortlist_size
        self.blacklist: Set[str] = set(blacklist or [
            'BTCUSDT', 'ETHUSDT',
            'VVVUSDT', 'SOPHUSDT', 'MARSCOINUSDT', 'USELESSUSDT', 'PONSUSDT',
            'SOXLUSDT', 'SKHYNIXUSDT', 'SKHYUSDT', 'SNDKUSDT', 'MUUSDT', 'KORUUSDT', 'XAUUSDT',
            'INTCUSDT'
        ])

        self.current_shortlist: List[str] = []
        self.last_scan_time: float = 0.0
        self.universe_cache: List[str] = []
        self.last_universe_refresh: float = 0.0

    def refresh_universe(self) -> List[str]:
        """Выгрузка полного списка активных Linear USDT деривативов Bybit."""
        try:
            tickers = self.client.get_tickers()
            if not tickers:
                return self.universe_cache

            all_linear = []
            for t in tickers:
                sym = t.get('symbol', '')
                if sym.endswith('USDT') and sym not in self.blacklist:
                    all_linear.append(sym)

            self.universe_cache = all_linear
            self.last_universe_refresh = time.time()
            return self.universe_cache
        except Exception as e:
            logger.error(f"Ошибка выгрузки Universe Bybit: {e}")
            return self.universe_cache

    def run_funnel(self) -> List[str]:
        """
        Прогон многоуровневой воронки фильтрации:
        300+ инструментов ──► 20-40 активных кандидатов шортлиста.
        """
        now = time.time()
        # Обновление полного списка деривативов раз в 60 минут
        if not self.universe_cache or (now - self.last_universe_refresh > 3600):
            self.refresh_universe()

        try:
            raw_tickers = self.client.get_tickers()
            if not raw_tickers:
                return self.current_shortlist

            candidates = []
            for t in raw_tickers:
                sym = t.get('symbol', '')
                if sym not in self.universe_cache:
                    continue

                try:
                    turnover = float(t.get('turnover24h', 0.0))
                except (ValueError, TypeError):
                    turnover = 0.0

                # 1. Жесткий фильтр ликвидности: суточный оборот >= $15M
                if turnover < self.min_turnover_usd:
                    continue

                # 2. Фильтр спреда: защита от неликвидных стаканов
                try:
                    bid = float(t.get('bid1Price', 0.0))
                    ask = float(t.get('ask1Price', 0.0))
                    if bid > 0 and ask > 0:
                        spread_pct = ((ask - bid) / bid) * 100.0
                        if spread_pct > self.max_spread_pct:
                            continue
                except (ValueError, TypeError):
                    pass

                candidates.append({
                    'symbol': sym,
                    'turnover': turnover,
                    'price24hPcnt': float(t.get('price24hPcnt', 0.0))
                })

            # Сортировка по обороту (наиболее ликвидные инструменты с глубокими стаканами)
            candidates.sort(key=lambda x: x['turnover'], reverse=True)

            # Формирование динамического шортлиста (от 20 до 40 монет)
            selected_count = max(self.min_shortlist_size, min(len(candidates), self.max_shortlist_size))
            shortlist = [c['symbol'] for c in candidates[:selected_count]]

            self.current_shortlist = shortlist
            self.last_scan_time = now

            # Синхронизация WebSocket: подписываемся только на отобранных кандидатов
            self.stream.sync_tracked_universe(self.current_shortlist)

            return self.current_shortlist

        except Exception as e:
            logger.error(f"Сбой выполнения воронки MarketScanner: {e}")
            return self.current_shortlist