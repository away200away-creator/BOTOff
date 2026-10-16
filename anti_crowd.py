import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class AntiCrowdMetrics:
    symbol: str
    liq_z_score: float
    net_delta: float
    funding_rate_pct: float
    ls_ratio: float
    orderbook_imbalance: float
    long_liq_vol_usd: float
    short_liq_vol_usd: float
    crowd_side: str          # 'LONG_HEAVY', 'SHORT_HEAVY', 'NEUTRAL'
    liquidation_spike: bool   # Z >= 1.5
    delta_divergence: bool    # Дельта противоположна движению свечи

class AntiCrowdAnalyzer:
    """
    Институциональный модуль анализа микроструктуры и сентимента толпы Bybit V5.
    Оценивает перегруз розничных трейдеров, кульминацию маржин-коллов и встречные айсберги.
    """

    def __init__(
        self,
        liq_z_threshold: float = 1.5,
        funding_extreme_pct: float = 0.01,
        ls_heavy_threshold: float = 1.4,
        ls_light_threshold: float = 0.7,
        ob_imbalance_threshold: float = 0.15
    ):
        self.liq_z_threshold = liq_z_threshold
        self.funding_extreme_pct = funding_extreme_pct
        self.ls_heavy_threshold = ls_heavy_threshold
        self.ls_light_threshold = ls_light_threshold
        self.ob_imbalance_threshold = ob_imbalance_threshold

    @staticmethod
    def calculate_series_z_score(buckets: List[float], current_val: float) -> float:
        """Расчет Z-Score выброса ликвидаций относительно скользящего окна."""
        if not buckets and current_val <= 0:
            return 0.0
        
        sample = list(buckets) if buckets else [0.0] * 5
        mean_val = sum(sample) / len(sample)
        variance = sum((x - mean_val) ** 2 for x in sample) / len(sample)
        std_val = math.sqrt(variance)

        if std_val == 0.0:
            # Если в стакане был мертвый штиль, а сейчас выбило свыше $1,000 маржи
            return 2.5 if current_val > 1000.0 else 0.0

        return round((current_val - mean_val) / std_val, 2)

    @staticmethod
    def calculate_orderbook_imbalance(bids: List[List[str]], asks: List[List[str]], depth_pct: float = 0.8) -> float:
        """
        Расчет взвешенного дисбаланса лимитной плотности стакана L2 в диапазоне depth_pct от спреда.
        Положительное значение: преобладание Bid (айсберг-поддержка покупателя).
        Отрицательное значение: преобладание Ask (лимитное давление продавца).
        """
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

            bid_vol = sum(float(p) * float(s) for p, s in bids if float(p) >= lower_bound)
            ask_vol = sum(float(p) * float(s) for p, s in asks if float(p) <= upper_bound)

            total_vol = bid_vol + ask_vol
            if total_vol <= 0:
                return 0.0

            return round((bid_vol - ask_vol) / total_vol, 3)
        except (ValueError, IndexError, TypeError):
            return 0.0

    def analyze(
        self,
        symbol: str,
        liq_history_buckets: List[float],
        current_liq_vol: float,
        long_liq_vol: float,
        short_liq_vol: float,
        net_delta: float,
        funding_rate_decimal: float,
        ls_ratio: float,
        orderbook_imbalance: float,
        candle_direction: int # +1 для зеленой свечи, -1 для красной свечи
    ) -> AntiCrowdMetrics:
        """Комплексная агрегация вектора сентимента толпы."""
        z_score = self.calculate_series_z_score(liq_history_buckets, current_liq_vol)
        funding_pct = funding_rate_decimal * 100.0

        # Определение позиционирования розницы
        if ls_ratio >= self.ls_heavy_threshold or funding_pct >= self.funding_extreme_pct:
            crowd_side = 'LONG_HEAVY'
        elif ls_ratio <= self.ls_light_threshold or funding_pct <= -self.funding_extreme_pct:
            crowd_side = 'SHORT_HEAVY'
        else:
            crowd_side = 'NEUTRAL'

        liq_spike = (z_score >= self.liq_z_threshold)
        
        # Дивергенция дельты: цена растет, а дельта отрицательна (лимитный айсберг продает)
        # либо цена падает, а дельта положительна (лимитный айсберг покупает)
        delta_div = False
        if candle_direction > 0 and net_delta < 0:
            delta_div = True
        elif candle_direction < 0 and net_delta > 0:
            delta_div = True

        return AntiCrowdMetrics(
            symbol=symbol,
            liq_z_score=z_score,
            net_delta=round(net_delta, 2),
            funding_rate_pct=round(funding_pct, 4),
            ls_ratio=round(ls_ratio, 2),
            orderbook_imbalance=round(orderbook_imbalance, 3),
            long_liq_vol_usd=round(long_liq_vol, 2),
            short_liq_vol_usd=round(short_liq_vol, 2),
            crowd_side=crowd_side,
            liquidation_spike=liq_spike,
            delta_divergence=delta_div
        )