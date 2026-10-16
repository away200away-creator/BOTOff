import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple

from breakout_engine import BreakoutSignal
from contrarian_engine import ContrarianSignal
from config import Config

logger = logging.getLogger("DecisionEngine")

@dataclass
class TradeDecision:
    action: str             # 'TRADE' или 'WAIT'
    signal: Optional[Union[BreakoutSignal, ContrarianSignal]]
    selected_strategy: str  # 'CONTRARIAN', 'BREAKOUT' или 'NONE'
    reason: str             # Подробное обоснование арбитра

class DecisionEngine:
    """
    Центральный арбитр принятия решений торгового комплекса BOtoFF.
    Обеспечивает:
    1. Защиту от фиктивных ML-сигналов: если весов нет и ALLOW_TRADING_WITHOUT_MODEL=False -> WAIT.
    2. Защиту от конфликтов гипотез (Mutual Exclusion): Long против Short -> WAIT.
    3. Сонаправленный арбитраж по калиброванной вероятности ML или скорингу.
    4. Фильтр генерального тренда BTC (BTC Regime Guard).
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.btc_filter_mode = getattr(config, 'BTC_FILTER', 'hard')
        self.btc_strong_thresh = getattr(config, 'BTC_STRONG_MOVE', 1.2)
        self.allow_without_ml = getattr(config, 'ALLOW_TRADING_WITHOUT_MODEL', False)
        self.min_ml_confidence = getattr(config, 'ML_MIN_CONFIDENCE', 0.60)

    def _is_btc_blocked(self, direction: str, btc_context: Dict[str, Any]) -> Tuple[bool, str]:
        if not btc_context:
            return False, "BTC контекст отсутствует"

        regime = btc_context.get('regime', 'NEUTRAL')
        btc_m5 = abs(btc_context.get('m5', 0.0))

        if regime == 'BULL_STRONG' and direction == 'SHORT' and btc_m5 > self.btc_strong_thresh:
            return True, f"BTC штормит вверх (M5={btc_m5:+.2f}%), SHORT заблокирован"
        if regime == 'BEAR_STRONG' and direction == 'LONG' and btc_m5 > self.btc_strong_thresh:
            return True, f"BTC штормит вниз (M5={btc_m5:+.2f}%), LONG заблокирован"

        if self.btc_filter_mode == 'hard':
            if direction == 'LONG' and regime == 'BEAR_STRONG':
                return True, "Генеральный тренд BTC: BEAR_STRONG, LONG запрещен"
            if direction == 'SHORT' and regime == 'BULL_STRONG':
                return True, "Генеральный тренд BTC: BULL_STRONG, SHORT запрещен"

        return False, "BTC фильтр пройден"

    def _verify_ml_gate(self, signal: Union[BreakoutSignal, ContrarianSignal], strat_name: str) -> Tuple[bool, str]:
        """Проверка допуска сетапа через ML-слой."""
        win_prob = getattr(signal, 'win_prob', None)

        # Модель отсутствует
        if win_prob is None:
            if not self.allow_without_ml:
                return False, f"Модель {strat_name} не обучена (ALLOW_TRADING_WITHOUT_MODEL=False) -> WAIT"
            return True, f"Режим отладки: модель {strat_name} не загружена, вход по правилам скоринга"

        # Модель обучена, проверяем уверенность
        if win_prob < self.min_ml_confidence:
            return False, f"Отказ ML {strat_name}: P(Win)={win_prob:.2f} ниже порога {self.min_ml_confidence:.2f}"

        return True, f"Одобрено ML {strat_name}: P(Win)={win_prob:.2f}"

    def evaluate(
        self,
        symbol: str,
        contrarian_sig: Optional[ContrarianSignal],
        breakout_sig: Optional[BreakoutSignal],
        btc_context: Dict[str, Any]
    ) -> TradeDecision:
        if not contrarian_sig and not breakout_sig:
            return TradeDecision(action='WAIT', signal=None, selected_strategy='NONE', reason="Нет сетапов")

        # 1. Конфликт гипотез (Mutual Exclusion)
        if contrarian_sig and breakout_sig:
            if contrarian_sig.direction != breakout_sig.direction:
                msg = f"🛑 Конфликт гипотез по {symbol}: Contrarian {contrarian_sig.direction} vs Breakout {breakout_sig.direction} -> WAIT"
                return TradeDecision(action='WAIT', signal=None, selected_strategy='NONE', reason=msg)

            # 2. Сонаправленный арбитраж
            p_c = getattr(contrarian_sig, 'win_prob', None) or 0.0
            p_b = getattr(breakout_sig, 'win_prob', None) or 0.0

            if p_c >= p_b:
                selected, strat_name = contrarian_sig, 'CONTRARIAN'
            else:
                selected, strat_name = breakout_sig, 'BREAKOUT'

            ml_ok, ml_reason = self._verify_ml_gate(selected, strat_name)
            if not ml_ok:
                return TradeDecision(action='WAIT', signal=None, selected_strategy=strat_name, reason=ml_reason)

            blocked, btc_reason = self._is_btc_blocked(selected.direction, btc_context)
            if blocked:
                return TradeDecision(action='WAIT', signal=None, selected_strategy=strat_name, reason=f"BTC: {btc_reason}")

            return TradeDecision(
                action='TRADE',
                signal=selected,
                selected_strategy=strat_name,
                reason=f"Сонаправленный конфлюэнс ({selected.direction}) [{strat_name}]: {ml_reason}"
            )

        # 3. Только Contrarian
        if contrarian_sig:
            ml_ok, ml_reason = self._verify_ml_gate(contrarian_sig, 'CONTRARIAN')
            if not ml_ok:
                return TradeDecision(action='WAIT', signal=None, selected_strategy='CONTRARIAN', reason=ml_reason)

            blocked, btc_reason = self._is_btc_blocked(contrarian_sig.direction, btc_context)
            if blocked:
                return TradeDecision(action='WAIT', signal=None, selected_strategy='CONTRARIAN', reason=f"BTC: {btc_reason}")

            tag = "Fakeout Reversal" if contrarian_sig.is_fakeout_play else "Mean Reversion"
            return TradeDecision(
                action='TRADE',
                signal=contrarian_sig,
                selected_strategy='CONTRARIAN',
                reason=f"Contrarian ({tag}) {contrarian_sig.direction}: {ml_reason}"
            )

        # 4. Только Breakout
        if breakout_sig:
            ml_ok, ml_reason = self._verify_ml_gate(breakout_sig, 'BREAKOUT')
            if not ml_ok:
                return TradeDecision(action='WAIT', signal=None, selected_strategy='BREAKOUT', reason=ml_reason)

            blocked, btc_reason = self._is_btc_blocked(breakout_sig.direction, btc_context)
            if blocked:
                return TradeDecision(action='WAIT', signal=None, selected_strategy='BREAKOUT', reason=f"BTC: {btc_reason}")

            return TradeDecision(
                action='TRADE',
                signal=breakout_sig,
                selected_strategy='BREAKOUT',
                reason=f"Breakout {breakout_sig.direction}: {ml_reason}"
            )

        return TradeDecision(action='WAIT', signal=None, selected_strategy='NONE', reason="Неопределённое состояние")