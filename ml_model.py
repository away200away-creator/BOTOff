import os
import sys
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("MLModel")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [MLModel] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class TradeFilterModel:
    """
    Класс для инференса обученной модели CatBoost (Meta-Labeling).
    Оценивает вероятность успешного завершения сделки P(Win) на основе рыночных факторов.
    
    Особенности реализации:
    1. Автоматическое определение сигнатуры признаков (model.feature_names_).
    2. Полная обратная совместимость со старыми моделями на 11 признаков
       и новыми моделями на 13 признаков (с liq_z и net_delta).
    3. Отказоустойчивость: при отсутствии файла или сбое инференса возвращает безопасный дефолт 0.50.
    4. Предотвращение падений при подаче непредвиденных типов данных или None.
    """

    DEFAULT_FEATURES_V1 = [
        'direction', 'score', 'rsi', 'm5', 'atr_pct',
        'funding', 'oi_change', 'ls_ratio', 'vol_spike',
        'ema_dist', 'btc_regime'
    ]

    DEFAULT_FEATURES_V2 = [
        'direction', 'score', 'rsi', 'm5', 'atr_pct',
        'funding', 'oi_change', 'ls_ratio', 'vol_spike',
        'ema_dist', 'btc_regime', 'liq_z', 'net_delta'
    ]

    def __init__(self, model_path: str = "trade_filter.cbm"):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.feature_names: List[str] = []
        self._load_model()

    def _load_model(self) -> None:
        """Загрузка бинарного файла весов CatBoost с извлечением метаданных признаков."""
        if not os.path.exists(self.model_path):
            logger.info("Модель '%s' не найдена. Работа в базовом режиме без ML-фильтрации.", self.model_path)
            self.is_loaded = False
            return

        try:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            self.is_loaded = True

            # Извлекаем точные имена и порядок признаков, запечённых внутри модели
            if hasattr(self.model, "feature_names_") and self.model.feature_names_:
                self.feature_names = list(self.model.feature_names_)
                logger.info(
                    "Модель '%s' успешно загружена. Признаков в модели: %d (%s)",
                    self.model_path, len(self.feature_names), ", ".join(self.feature_names)
                )
            else:
                self.feature_names = list(self.DEFAULT_FEATURES_V2)
                logger.warning(
                    "Имена признаков в модели '%s' не обнаружены. Применён резервный список V2 (%d признаков).",
                    self.model_path, len(self.feature_names)
                )

        except ImportError:
            logger.error("Библиотека 'catboost' не установлена. Запустите: pip install catboost")
            self.is_loaded = False
        except Exception as exc:
            logger.exception("Непредвиденная ошибка при загрузке модели '%s': %s", self.model_path, exc)
            self.is_loaded = False

    def _prepare_feature_row(self, features: Dict[str, Any]) -> List[Any]:
        """
        Преобразует входной словарь параметров в строго упорядоченный вектор признаков,
        соответствующий ожиданиям конкретной загруженной версии модели.
        """
        keys_to_use = self.feature_names if self.feature_names else self.DEFAULT_FEATURES_V2
        row: List[Any] = []

        for key in keys_to_use:
            val = features.get(key)

            if key == 'btc_regime':
                # Категориальный строковый признак
                if val is None or not isinstance(val, str):
                    row.append("NEUTRAL")
                else:
                    row.append(str(val))
            elif key == 'direction':
                # Бинарное направление (1 для LONG, 0 для SHORT)
                if val in (1, '1', 'LONG', True):
                    row.append(1)
                else:
                    row.append(0)
            else:
                # Все остальные факторы — числовые (float)
                if val is None:
                    row.append(0.0)
                else:
                    try:
                        row.append(float(val))
                    except (ValueError, TypeError):
                        row.append(0.0)

        return row

    def predict_win_probability(self, features: Dict[str, Any]) -> float:
        """
        Вычисляет прогнозируемую вероятность успешности сделки P(Win) в диапазоне [0.0, 1.0].
        При отсутствии модели или ошибке расчёта возвращает безопасную нейтральную оценку 0.50.
        """
        if not self.is_loaded or self.model is None:
            return 0.50

        try:
            row_data = self._prepare_feature_row(features)
            # CatBoostClassifier.predict_proba возвращает двумерный массив [[P(0), P(1)]]
            probabilities = self.model.predict_proba([row_data])
            win_prob = float(probabilities[0][1])

            # Защита от выхода за границы вероятностей
            return max(0.0, min(1.0, win_prob))

        except Exception as exc:
            logger.debug("Сбой инференса ML-модели для вектора %s: %s", features, exc)
            return 0.50

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 [Self-Test] Проверка модуля инференса ml_model.py...")
    print("=" * 70)

    model_engine = TradeFilterModel(model_path="trade_filter.cbm")
    print(f"Статус загрузки модели: {'✅ Загружена' if model_engine.is_loaded else '⚠️ Файл весов не найден (дефолтный режим)'}")
    print(f"Активный список признаков ({len(model_engine.feature_names)} шт.): {model_engine.feature_names}")

    # Тестовый входной словарь признаков
    sample_features = {
        'direction': 1,
        'score': 65,
        'rsi': 28.5,
        'm5': -0.85,
        'atr_pct': 2.1,
        'funding': -0.015,
        'oi_change': -0.45,
        'ls_ratio': 0.65,
        'vol_spike': 3.2,
        'ema_dist': -1.8,
        'btc_regime': 'BULL_STRONG',
        'liq_z': 2.4,
        'net_delta': 125000.0
    }

    predicted_prob = model_engine.predict_win_probability(sample_features)
    print("\n📊 Результат тестового инференса:")
    print(f"   Входные данные: direction=LONG, score=65, liq_z=+2.4σ, oi_change=-0.45%")
    print(f"   Расчётная вероятность P(Win): {predicted_prob:.4f} ({predicted_prob * 100:.1f}%)")
    print("=" * 70)