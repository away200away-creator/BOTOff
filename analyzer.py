import csv
import os
from collections import defaultdict
from datetime import datetime

try:
    from config import Config
    INITIAL_BALANCE = Config.INITIAL_BALANCE
except Exception:
    INITIAL_BALANCE = 100.0


def load_trades(filename='trade_history.csv'):
    if not os.path.exists(filename):
        print(f"❌ Файл {filename} не найден.")
        return []
    trades = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                trades.append({
                    'time': row['Time'],
                    'symbol': row['Symbol'],
                    'direction': row['Direction'],
                    'entry_price': float(row['Entry Price']),
                    'avg_price': float(row['Avg Price']),
                    'exit_price': float(row['Exit Price']),
                    'margin': float(row['Margin']),
                    'notional': float(row['Notional']),
                    'leverage': int(row['Leverage']),
                    'score': int(row['Score']),
                    'pnl': float(row['PnL ($)']),
                    'roe': float(row['ROE %']),
                    'reason': row['Reason'],
                    'duration': float(row['Duration (s)'])
                })
            except Exception as e:
                print(f"⚠️ Пропущена строка: {e}")
    return trades


def print_separator(char='=', length=80):
    print(char * length)


def calculate_max_drawdown(balance_curve):
    peak = balance_curve[0]
    max_dd = 0
    max_dd_pct = 0
    for bal in balance_curve:
        if bal > peak:
            peak = bal
        dd = peak - bal
        dd_pct = (dd / peak * 100) if peak != 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    return max_dd, max_dd_pct


def analyze(trades):
    if not trades:
        print("📭 Нет сделок для анализа.")
        return

    total = len(trades)
    wins = [t for t in trades if t['pnl'] >= 0]
    losses = [t for t in trades if t['pnl'] < 0]
    winrate = len(wins) / total * 100 if total else 0
    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl = total_pnl / total if total else 0
    max_win = max(t['pnl'] for t in trades)
    max_loss = min(t['pnl'] for t in trades)
    avg_roe = sum(t['roe'] for t in trades) / total if total else 0
    avg_duration = sum(t['duration'] for t in trades) / total if total else 0

    balance_curve = [INITIAL_BALANCE]
    for t in trades:
        balance_curve.append(balance_curve[-1] + t['pnl'])
    max_dd, max_dd_pct = calculate_max_drawdown(balance_curve)

    print_separator()
    print("📊 ОБЩАЯ СТАТИСТИКА")
    print_separator()
    print(f"Всего сделок:            {total}")
    print(f"Прибыльных:              {len(wins)} (Winrate: {winrate:.1f}%)")
    print(f"Убыточных:               {len(losses)}")
    print(f"Суммарный PnL:           {total_pnl:+.4f}$")
    print(f"Средний PnL на сделку:   {avg_pnl:+.4f}$")
    print(f"Максимальная прибыль:    {max_win:+.4f}$")
    print(f"Максимальный убыток:     {max_loss:+.4f}$")
    print(f"Средний ROE:             {avg_roe:+.2f}%")
    print(f"Средняя длительность:    {avg_duration:.1f} сек")
    print(f"Стартовый баланс:        {INITIAL_BALANCE:.2f}$")
    print(f"Максимальная просадка:   -{max_dd:.4f}$ ({max_dd_pct:.2f}%)")
    print(f"\nPnL за текущую сессию:   {total_pnl:+.4f}$ (все сделки в файле)")

    by_symbol = defaultdict(lambda: {'count': 0, 'pnl': 0.0, 'wins': 0})
    for t in trades:
        sym = t['symbol']
        by_symbol[sym]['count'] += 1
        by_symbol[sym]['pnl'] += t['pnl']
        if t['pnl'] >= 0:
            by_symbol[sym]['wins'] += 1

    print_separator()
    print("📈 ТОП-10 ПРИБЫЛЬНЫХ МОНЕТ")
    print_separator()
    sorted_profit = sorted(by_symbol.items(), key=lambda x: x[1]['pnl'], reverse=True)[:10]
    for sym, data in sorted_profit:
        wr = (data['wins'] / data['count'] * 100) if data['count'] else 0
        print(f"  {sym:15s} | Сделок: {data['count']:3d} | Winrate: {wr:.0f}% | PnL: {data['pnl']:+.4f}$")

    print_separator()
    print("📉 ТОП-10 УБЫТОЧНЫХ МОНЕТ")
    print_separator()
    sorted_loss = sorted(by_symbol.items(), key=lambda x: x[1]['pnl'])[:10]
    for sym, data in sorted_loss:
        wr = (data['wins'] / data['count'] * 100) if data['count'] else 0
        print(f"  {sym:15s} | Сделок: {data['count']:3d} | Winrate: {wr:.0f}% | PnL: {data['pnl']:+.4f}$")

    by_direction = defaultdict(lambda: {'count': 0, 'pnl': 0.0, 'wins': 0})
    for t in trades:
        d = t['direction']
        by_direction[d]['count'] += 1
        by_direction[d]['pnl'] += t['pnl']
        if t['pnl'] >= 0:
            by_direction[d]['wins'] += 1

    print_separator()
    print("🔀 НАПРАВЛЕНИЯ")
    print_separator()
    for d in sorted(by_direction.keys()):
        data = by_direction[d]
        wr = (data['wins'] / data['count'] * 100) if data['count'] else 0
        print(f"  {d:5s} | Сделок: {data['count']:3d} | Winrate: {wr:.0f}% | PnL: {data['pnl']:+.4f}$")

    by_reason = defaultdict(lambda: {'count': 0, 'pnl': 0.0})
    for t in trades:
        r = t['reason']
        by_reason[r]['count'] += 1
        by_reason[r]['pnl'] += t['pnl']

    print_separator()
    print("📋 ПРИЧИНЫ ЗАКРЫТИЯ")
    print_separator()
    for r in sorted(by_reason.keys()):
        data = by_reason[r]
        print(f"  {r:25s} | Сделок: {data['count']:3d} | PnL: {data['pnl']:+.4f}$")


def show_analytics():
    trades = load_trades()
    analyze(trades)


if __name__ == "__main__":
    show_analytics()
