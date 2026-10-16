import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from decision_engine import TradeDecision
from config import Config

logger = logging.getLogger("ExecutionEngine")


@dataclass
class ExecutionResult:
    """Результат выполнения ордера на биржевом или симуляционном контуре."""
    success: bool
    symbol: str
    strategy_id: str
    direction: str
    order_type: str         # 'MAKER_POST_ONLY' или 'AGGRESSIVE_LIMIT'
    requested_price: float
    executed_price: float
    slippage_pct: float
    fee_rate: float
    margin: float
    leverage: int
    notional: float
    score: int
    win_prob: float
    orderbook_imbalance: float
    volatility: float
    tp1_roe: float
    stop_roe: float
    reason: str
    timestamp: float


class ExecutionEngine:
    """
    Институциональный движок исполнения ордеров Bybit V5 Linear.
    Реализует специализированную асимметрию исполнения:
    - Contrarian: Maker Post-Only (Best Bid / Best Ask, нулевой slip, комиссия 0.020%)
    - Breakout: Aggressive Limit (Cross-the-Spread с жестким Max Slippage Guard <= 0.15%, комиссия 0.055%)
    """

    def __init__(self, config: Config, bybit_client=None):
        self.cfg = config
        self.client = bybit_client
        self.maker_fee_rate = getattr(config, 'MAKER_FEE_RATE', 0.00020)
        self.taker_fee_rate = getattr(config, 'TAKER_FEE_RATE', 0.00055)
        self.max_slippage_guard = 0.15  # Максимально допустимое проскальзывание входа для Breakout (%)
        self.max_spread_pct = getattr(config, 'MAX_SPREAD_PCT', 0.10)

    def _verify_spread_health(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float
    ) -> Tuple[bool, float, str]:
        """Проверка ликвидности и ширины спреда перед выставлением приказа."""
        if best_bid <= 0 or best_ask <= 0:
            return False, 0.0, "Некорректные цены спроса/предложения"

        spread_pct = ((best_ask - best_bid) / best_bid) * 100.0
        if spread_pct > self.max_spread_pct:
            return False, spread_pct, f"Спред {spread_pct:.3f}% превышает предел {self.max_spread_pct:.2f}%"

        return True, spread_pct, "Спред в пределах нормы"

    def execute_decision(
        self,
        decision: TradeDecision,
        margin: float,
        leverage: int,
        ticker: Dict[str, Any],
        spread_data: Optional[Dict[str, float]] = None
    ) -> Optional[ExecutionResult]:
        """
        Исполнение решения, сформированного DecisionEngine.
        Определяет оптимальный тип ордера и рассчитывает физику проскальзывания.
        """
        if not decision or decision.action != 'TRADE' or not decision.signal:
            return None

        sig = decision.signal
        symbol = sig.symbol
        direction = sig.direction
        strat_id = decision.selected_strategy
        now = time.time()

        # Получение лучших цен Bid/Ask
        live_price = float(ticker.get('lastPrice', sig.price))
        best_bid = float(spread_data.get('bid', live_price)) if spread_data else live_price
        best_ask = float(spread_data.get('ask', live_price)) if spread_data else live_price

        if best_bid <= 0:
            best_bid = live_price
        if best_ask <= 0:
            best_ask = live_price

        spread_ok, spread_pct, spread_msg = self._verify_spread_health(symbol, best_bid, best_ask)
        if not spread_ok:
            logger.warning("Отказ в исполнении %s [%s]: %s", symbol, strat_id, spread_msg)
            return None

        notional = margin * leverage

        # СЦЕНАРИЙ А: CONTRARIAN STRATEGY -> MAKER POST-ONLY
        if strat_id == 'CONTRARIAN':
            order_type = 'MAKER_POST_ONLY'
            # Вход пассивным лимитом: покупка по Best Bid, продажа по Best Ask
            requested_price = best_bid if direction == 'LONG' else best_ask
            executed_price = requested_price
            slippage_pct = 0.0
            fee_rate = self.maker_fee_rate

        # СЦЕНАРИЙ Б: BREAKOUT STRATEGY -> AGGRESSIVE LIMIT (CROSS-THE-SPREAD)
        elif strat_id == 'BREAKOUT':
            order_type = 'AGGRESSIVE_LIMIT'
            # Вход агрессивным лимитом: для лонга берем Best Ask, для шорта Best Bid
            requested_price = best_ask if direction == 'LONG' else best_bid
            
            # Моделирование микроструктурного проскальзывания в импульсе
            base_slip = min(self.max_slippage_guard, max(0.02, spread_pct / 2.0))
            if direction == 'LONG':
                executed_price = requested_price * (1.0 + base_slip / 100.0)
            else:
                executed_price = requested_price * (1.0 - base_slip / 100.0)

            slippage_pct = base_slip
            fee_rate = self.taker_fee_rate

        else:
            logger.error("Неизвестный идентификатор стратегии: %s", strat_id)
            return None

        result = ExecutionResult(
            success=True,
            symbol=symbol,
            strategy_id=strat_id,
            direction=direction,
            order_type=order_type,
            requested_price=requested_price,
            executed_price=executed_price,
            slippage_pct=round(slippage_pct, 4),
            fee_rate=fee_rate,
            margin=margin,
            leverage=leverage,
            notional=notional,
            score=sig.score,
            win_prob=getattr(sig, 'win_prob', 0.65),
            orderbook_imbalance=getattr(sig, 'orderbook_imbalance', 0.0),
            volatility=getattr(sig, 'atr_pct', 2.0),
            tp1_roe=sig.tp1_roe,
            stop_roe=sig.stop_roe,
            reason=decision.reason,
            timestamp=now
        )

        logger.info(
            "⚡ [ExecutionEngine] Исполнен %s %s [%s] по цене %.5f (%s, fee: %.3f%%, slip: %.3f%%)",
            symbol, direction, strat_id, executed_price, order_type, fee_rate * 100.0, slippage_pct
        )
        return result

    @staticmethod
    def to_signal_dict(exec_res: ExecutionResult) -> Dict[str, Any]:
        """Преобразование ExecutionResult в контракт для RiskManager."""
        return {
            'symbol': exec_res.symbol,
            'strategy_id': exec_res.strategy_id,
            'direction': exec_res.direction,
            'price': exec_res.executed_price,
            'maker_price': exec_res.executed_price,
            'order_type': exec_res.order_type,
            'fee_rate': exec_res.fee_rate,
            'slippage_pct': exec_res.slippage_pct,
            'score': exec_res.score,
            'win_prob': exec_res.win_prob,
            'volatility': exec_res.volatility,
            'orderbook_imbalance': exec_res.orderbook_imbalance,
            'tp1_roe': exec_res.tp1_roe,
            'stop_roe': exec_res.stop_roe,
            'reason': exec_res.reason
        }