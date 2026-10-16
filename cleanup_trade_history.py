import csv
import os

HISTORY_FILE = "trade_history.csv"
CLEANED_FILE = "trade_history_cleaned.csv"

def clean_history():
    if not os.path.exists(HISTORY_FILE):
        print(f"❌ Файл '{HISTORY_FILE}' не найден.")
        return

    kept_rows = []
    removed_count = 0

    with open(HISTORY_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("Файл пуст.")
            return

        for row in reader:
            if not row or len(row) < 14:
                continue

            time_val = row[0].strip()
            duration_val = float(row[13].strip()) if row[13].strip() else 0.0

            # Признаки симуляционных данных бэктестера:
            # 1. Время свечи кратно ровным 5 минутам (например, '17:15:00', '01:30:00')
            # 2. Длительность кратна ровно 300, 1800, 2700 секундам без миллисекундных хвостов
            is_backtest_artifact = (
                time_val.endswith(":00") and 
                duration_val in (1800.0, 2100.0, 2400.0, 2700.0, 3000.0, 3300.0, 4200.0)
            )

            if is_backtest_artifact:
                removed_count += 1
            else:
                kept_rows.append(row)

    # Перезаписываем боевой файл чистыми данными
    with open(HISTORY_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept_rows)

    print("=" * 60)
    print(f"🧹 Очистка журнала завершена успешно!")
    print(f"   Удалено строк симуляции бэктестера: {removed_count}")
    print(f"   Сохранено реальных сделок в '{HISTORY_FILE}': {len(kept_rows)}")
    print("=" * 60)

if __name__ == "__main__":
    clean_history()