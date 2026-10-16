import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

def train_breakout():
    print("=" * 75)
    print("🚀 [1/4] Инициализация скрипта обучения Breakout Filter (Donchian Expansion)...")
    print("=" * 75)

    dataset_path = "datasets/breakout_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Файл '{dataset_path}' не найден!")
        print("Сначала сформируйте выборку через data_collector.py")
        return

    print(f"📂 [2/4] Загрузка датасета из '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    print(f"   Загружено строк: {len(df)}")

    if len(df) < 80:
        print("⚠️ Слишком мало данных для обучения. Требуется минимум 80 размеченных событий.")
        return

    feature_cols = [
        'direction',
        'score',
        'volume_spike',
        'delta_oi_pct',
        'net_delta',
        'orderbook_imbalance',
        'atr_pct',
        'btc_regime'
    ]
    target_col = 'target'

    for col in feature_cols:
        if col not in df.columns:
            df[col] = "NEUTRAL" if col == 'btc_regime' else 0.0

    df['btc_regime'] = df['btc_regime'].fillna('NEUTRAL').astype(str)
    for col in feature_cols:
        if col != 'btc_regime':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    X = df[feature_cols]
    y = df[target_col].astype(int)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"✂️  Time-Series Split: Обучение = {len(X_train)}, Валидация = {len(X_test)}")

    cat_features = ['btc_regime']
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    print("\n🧠 [3/4] Обучение модели CatBoost (Breakout Filter)...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        early_stopping_rounds=40,
        verbose=50
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    y_pred_proba = model.predict_proba(test_pool)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n📊 [4/4] Оценка качества классификации:")
    print(f"   Метрика разделения (AUC-ROC): {auc:.3f}")
    base_wr = (y_test.sum() / len(y_test)) * 100
    print(f"   Базовый винрейт без ML: {base_wr:.1f}%")

    os.makedirs("models", exist_ok=True)
    output_model = "models/breakout_filter.cbm"
    model.save_model(output_model)
    print(f"\n💾 Готово! Веса сохранены: '{output_model}'\n")

if __name__ == "__main__":
    train_breakout()