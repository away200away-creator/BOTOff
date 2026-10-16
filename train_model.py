import os
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from ml_model import DEFAULT_FEATURES


def _ensure_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Если в старом dataset.csv нет liq_z/net_delta — строим прокси из vol_spike/m5."""
    if 'liq_z' not in df.columns:
        if 'vol_spike' in df.columns:
            df['liq_z'] = ((df['vol_spike'] - 1.0).clip(lower=0) * 1.2).clip(upper=5.0)
        else:
            df['liq_z'] = 0.0
        print("ℹ️ Колонка liq_z отсутствовала — заполнена прокси из vol_spike.")
    if 'net_delta' not in df.columns:
        if 'm5' in df.columns and 'vol_spike' in df.columns:
            df['net_delta'] = df['m5'] * df['vol_spike']
        else:
            df['net_delta'] = 0.0
        print("ℹ️ Колонка net_delta отсутствовала — заполнена прокси из m5*vol_spike.")
    return df


def train():
    print("=" * 60)
    print("🚀 [1/4] Инициализация обучения ML-модели...")
    print("=" * 60)

    dataset_file = "dataset.csv"
    if not os.path.exists(dataset_file):
        print(f"❌ Файл '{dataset_file}' не найден. Сначала запустите сбор датасета [5].")
        return

    print(f"📂 [2/4] Загрузка '{dataset_file}'...")
    df = pd.read_csv(dataset_file)
    df = _ensure_orderflow_features(df)
    print(f"   Строк: {len(df)}")

    if len(df) < 100:
        print("⚠️ Слишком мало данных (нужно ≥ 100 строк).")
        return

    features = list(DEFAULT_FEATURES)
    cat_features = ['btc_regime']
    target = 'target'

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        print(f"❌ В датасете отсутствуют колонки: {missing}")
        return

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print(f"✂️  Train={len(train_df)}, Val={len(test_df)}")

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    print("\n🧠 [3/4] Обучение CatBoost...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        eval_metric='AUC',
        random_seed=42,
        early_stopping_rounds=40,
        verbose=100
    )
    model.fit(train_pool, eval_set=test_pool)

    print("\n📊 [4/4] Качество фильтра:")
    y_pred_proba = model.predict_proba(test_pool)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"   AUC-ROC: {auc:.3f}")

    base_wr = (y_test.sum() / len(y_test)) * 100
    print(f"   Базовый WR без ИИ: {base_wr:.1f}%")

    for thresh in [0.50, 0.55, 0.60]:
        mask = y_pred_proba >= thresh
        if mask.sum() > 0:
            filtered_wr = (y_test[mask].sum() / mask.sum()) * 100
            print(f"   🎯 WR @ >{int(thresh*100)}%: {filtered_wr:.1f}% ({mask.sum()}/{len(y_test)})")
        else:
            print(f"   ⚠️ Порог >{int(thresh*100)}% отсёк всё.")

    importance = model.get_feature_importance()
    print("\n🔍 Важность признаков:")
    for feat, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True)[:8]:
        print(f"   • {feat:12s}: {imp:.1f}")

    model_file = "trade_filter.cbm"
    model.save_model(model_file)
    print(f"\n💾 Модель сохранена: '{model_file}'\n")


if __name__ == "__main__":
    train()
