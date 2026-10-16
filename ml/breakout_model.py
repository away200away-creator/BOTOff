import os
import sys
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("BreakoutML")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [BreakoutML] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class BreakoutModelFilter:
    """
    Инференс модели CatBoost для фильтрации пробойных сетапов (Donchian Expansion).
    Строгое правило: при отсутствии файла весов P(Win) НЕ выдумывается.
    Возвращается None.
    """

    FEATURE_NAMES = [
        'direction',
        'score',
        'volume_spike',
        'delta_oi_pct',
        'net_delta',
        'orderbook_imbalance',
        'atr_pct',
        'btc_regime'
    ]

    def __init__(self, model_path: str = "models/breakout_filter.cbm", min_confidence: float = 0.60):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.model = None
        self.is_loaded = False
        self.feature_names: List[str] = list(self.FEATURE_NAMES)
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning("Файл весов '%s' отсутствует. Статус модели: NOT_READY.", self.model_path)
            self.is_loaded = False
            return

        try:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            self.is_loaded = True

            if hasattr(self.model, "feature_names_") and self.model.feature_names_:
                self.feature_names = list(self.model.feature_names_)
                logger.info("Модель '%s' успешно загружена. Признаков: %d", self.model_path, len(self.feature_names))
            else:
                self.feature_names = list(self.FEATURE_NAMES)

        except ImportError:
            logger.error("Библиотека 'catboost' не установлена.")
            self.is_loaded = False
        except Exception as exc:
            logger.exception("Ошибка при загрузке модели '%s': %s", self.model_path, exc)
            self.is_loaded = False

    def _prepare_feature_row(self, features: Dict[str, Any]) -> List[Any]:
        row: List[Any] = []
        for key in self.feature_names:
            val = features.get(key)
            if key == 'btc_regime':
                row.append(str(val) if val is not None else "NEUTRAL")
            elif key == 'direction':
                row.append(1 if val in (1, '1', 'LONG', True) else 0)
            else:
                try:
                    row.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    row.append(0.0)
        return row

    def predict_win_probability(self, features: Dict[str, Any]) -> Optional[float]:
        """
        Расчёт вероятности успеха пробоя P(Win) в диапазоне [0.0, 1.0].
        Если модель не загружена, возвращается строго None.
        """
        if not self.is_loaded or self.model is None:
            return None

        try:
            row_data = self._prepare_feature_row(features)
            probabilities = self.model.predict_proba([row_data])
            win_prob = float(probabilities[0][1])
            return max(0.0, min(1.0, win_prob))
        except Exception as exc:
            logger.debug("Сбой инференса BreakoutModel: %s", exc)
            return None