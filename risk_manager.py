import time
import uuid
import csv
import os
import json
import threading
from typing import Dict, List, Optional
from config import Config

try:
    from telegram_notifier import send_telegram_message, notify_entry, notify_tp1, notify_close
except ImportError:
    def send_telegram_message(msg): pass
    def notify_entry(*args, **kwargs): pass
    def notify_tp1(*args, **kwargs): pass
    def notify_close(*args, **kwargs): pass

class Position:
    def __init__(self, signal: dict, margin: float, leverage: int, cfg: Optional[Config] = None, current_time: Optional[float] = None):
        self.id = str(uuid.uuid4())[:8]
        self.symbol = signal['symbol']
        self.direction = signal['direction']
        self.cfg = cfg
        self.is_maker = getattr(cfg, 'USE_MAKER_POST_ONLY', True) if cfg else False

        # ==================== ИСПОЛНЕНИЕ ВХОДА ====================
        raw_price = float(signal.get('maker_price', signal['price']))
        if self.is_maker:
            self.entry_price = raw_price
            self.entry_fee_rate = getattr(cfg, 'MAKER_FEE_RATE', 0.00020) if cfg else 0.00020
            self.entry_slippage_pct = 0.0
        else:
            slip = getattr(cfg, 'SLIPPAGE_PCT', 0.08) if cfg else 0.08
            self.entry_slippage_pct = slip
            if self.direction == 'LONG':
                self.entry_price = raw_price * (1.0 + self.entry_slippage_pct / 100.0)
            else:
                self.entry_price = raw_price * (1.0 - self.entry_slippage_pct / 100.0)
            self.entry_fee_rate = getattr(cfg, 'TAKER_FEE_RATE', 0.00055) if cfg else 0.00055

        self.taker_fee_rate = getattr(cfg, 'TAKER_FEE_RATE', 0.00055) if cfg else 0.00055
        self.exit_slippage_pct = getattr(cfg, 'SLIPPAGE_PCT', 0.08) if cfg else 0.08

        self.avg_price = self.entry_price
        self.margin = margin
        self.notional = margin * leverage
        self.leverage = leverage
        self.score = signal['score']
        self.win_prob = signal.get('win_prob', 0.5)
        self.sign = 1 if self.direction == 'LONG' else -1
        self.tp1_hit = False
        self.tp2_hit = False
        self.dca_done = False
        self.be_mode = False
        self.soft_be_mode = False
        self.orderbook_imbalance = signal.get('orderbook_imbalance', 0.0)
        
        atr_pct = float(signal.get('volatility', 1.0))
        atr_mult = getattr(cfg, 'TRAILING_ATR_MULT', 1.0) if cfg else 1.0
        raw_trail = atr_pct * atr_mult * self.leverage
        min_trail = getattr(cfg, 'TRAILING_DISTANCE_MIN', 1.5) if cfg else 1.5
        max_trail = getattr(cfg, 'TRAILING_DISTANCE_MAX', 4.0) if cfg else 4.0
        self.trailing_distance = max(min_trail, min(max_trail, raw_trail))

        self.entry_fee = self.notional * self.entry_fee_rate
        self.total_fees = self.entry_fee
        self.realized_pnl = -self.entry_fee

        self.current_price = self.entry_price
        self.created_at = current_time if current_time is not None else time.time()
        self.initial_margin = margin
        self.initial_notional = margin * leverage
        self.max_roe = 0.0
        self.trailing_active = False

    def update_price(self, price: float):
        self.current_price = price
        current_roe = self.get_roe()
        if current_roe > self.max_roe:
            self.max_roe = current_roe

    def get_roe(self) -> float:
        if self.avg_price == 0:
            return 0.0
        move = (self.current_price - self.avg_price) / self.avg_price * 100.0
        return self.sign * move * self.leverage

    def get_unrealized_pnl(self) -> float:
        roe = self.get_roe()
        gross_pnl = self.margin * roe / 100.0
        est_exit_fee = self.notional * self.taker_fee_rate
        return self.realized_pnl + gross_pnl - est_exit_fee

    def get_total_pnl(self) -> float:
        return self.realized_pnl + (self.margin * self.get_roe() / 100.0)


class RiskManager:
    def __init__(self, config: Config, history_file: Optional[str] = 'trade_history.csv'):
        self.cfg = config
        self.balance = config.INITIAL_BALANCE
        self.positions: List[Position] = []
        self.history: List[Dict] = []
        self.wins = 0
        self.lock = threading.RLock()
        self.cooldowns: Dict[str, float] = {}
        self.history_file = history_file
        self.cooldowns_file = getattr(config, 'COOLDOWNS_FILE', 'cooldowns.json')
        
        self.taker_fee_rate = getattr(config, 'TAKER_FEE_RATE', 0.00055)
        self.be_trigger_roe = getattr(config, 'BE_TRIGGER_ROE', 0.5)
        self.time_stop_minutes = getattr(config, 'TIME_STOP_MINUTES', 45)
        self.soft_be_minutes = getattr(config, 'SOFT_BE_MINUTES', 30)
        self.soft_be_trigger_roe = getattr(config, 'SOFT_BE_TRIGGER_ROE', 1.0)
        self.soft_be_level_roe = getattr(config, 'SOFT_BE_LEVEL_ROE', 0.5)
        self.max_directional = getattr(config, 'MAX_DIRECTIONAL_POSITIONS', 5)
        
        if self.history_file:
            self._init_history_file()
            self._load_cooldowns()

    def _init_history_file(self):
        if self.history_file and not os.path.exists(self.history_file):
            with open(self.history_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Time', 'Symbol', 'Direction', 'Entry Price', 'Avg Price',
                    'Exit Price', 'Margin', 'Notional', 'Leverage', 'Score',
                    'PnL ($)', 'ROE %', 'Reason', 'Duration (s)'
                ])

    def _load_cooldowns(self):
        with self.lock:
            if os.path.exists(self.cooldowns_file):
                try:
                    with open(self.cooldowns_file, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                        now = time.time()
                        self.cooldowns = {k: v for k, v in raw.items() if v > now}
                except Exception as e:
                    print(f"⚠️ Ошибка чтения {self.cooldowns_file}: {e}")
                    self.cooldowns = {}

    def _save_cooldowns(self):
        with self.lock:
            if not self.history_file:
                return
            try:
                now = time.time()
                active = {k: v for k, v in self.cooldowns.items() if v > now}
                with open(self.cooldowns_file, 'w', encoding='utf-8') as f:
                    json.dump(active, f, indent=2)
            except Exception as e:
                print(f"⚠️ Ошибка сохранения {self.cooldowns_file}: {e}")

    def can_open(self, direction: Optional[str] = None) -> bool:
        with self.lock:
            if len(self.positions) >= self.cfg.MAX_POSITIONS:
                return False
                
            if direction:
                same_dir_count = sum(1 for p in self.positions if p.direction == direction)
                if same_dir_count >= self.max_directional:
                    return False
                    
            return True

    def is_cooldown(self, symbol: str, current_time: Optional[float] = None) -> bool:
        now = current_time if current_time is not None else time.time()
        with self.lock:
            if symbol in self.cooldowns:
                if now < self.cooldowns[symbol]:
                    return True
                else:
                    del self.cooldowns[symbol]
                    self._save_cooldowns()
            return False

    def calculate_margin(self, signal: Optional[dict] = None) -> float:
        base_margin = self.balance * self.cfg.RISK_PER_TRADE / 100.0
        if not getattr(self.cfg, 'DYNAMIC_VAR_SIZING', True) or not signal:
            return base_margin

        atr_pct = float(signal.get('volatility', 2.0))
        base_atr = getattr(self.cfg, 'BASE_ATR_PCT', 2.0)
        
        vol_scalar = base_atr / max(0.5, atr_pct)
        adjusted_margin = base_margin * vol_scalar

        min_m = getattr(self.cfg, 'MIN_MARGIN', 1.0)
        max_m = getattr(self.cfg, 'MAX_MARGIN', 2.40)
        clamped_margin = max(min_m, min(max_m, adjusted_margin))
        return round(clamped_margin, 2)

    def open_position(self, signal: dict, current_time: Optional[float] = None) -> Optional[Position]:
        direction = signal['direction']
        with self.lock:
            if not self.can_open(direction):
                return None
            if self.is_cooldown(signal['symbol'], current_time=current_time):
                return None
            if any(p.symbol == signal['symbol'] for p in self.positions):
                return None

            margin = self.calculate_margin(signal)
            if margin <= 0 or margin > self.balance:
                return None

            self.balance -= margin
            position = Position(
                signal=signal,
                margin=margin,
                leverage=self.cfg.LEVERAGE,
                cfg=self.cfg,
                current_time=current_time
            )
            self.positions.append(position)

            long_count = sum(1 for p in self.positions if p.direction == 'LONG')
            short_count = sum(1 for p in self.positions if p.direction == 'SHORT')

        if self.history_file:
            notify_entry(
                symbol=position.symbol,
                direction=position.direction,
                price=position.entry_price,
                margin=position.margin,
                leverage=position.leverage,
                score=position.score,
                ml_prob=position.win_prob,
                liq_z=signal.get('liq_z_score', 0.0),
                oi_change=signal.get('oi_change', None),
                long_count=long_count,
                short_count=short_count,
                max_directional=self.max_directional
            )
        return position

    def manage_position(self, position: Position, current_time: Optional[float] = None):
        with self.lock:
            now = current_time if current_time is not None else time.time()
            roe = position.get_roe()
            holding_seconds = now - position.created_at
            time_stop_seconds = self.time_stop_minutes * 60

            # 1. ТАЙМ-МЕНЕДЖМЕНТ
            if not position.tp1_hit:
                if holding_seconds >= self.soft_be_minutes * 60:
                    if position.max_roe >= self.soft_be_trigger_roe or roe >= self.soft_be_trigger_roe:
                        position.soft_be_mode = True
                    
                    if position.soft_be_mode and roe <= self.soft_be_level_roe:
                        self._close(position, f"Soft BE (30+ мин, фиксация {roe:+.1f}% ROE)", current_time=current_time)
                        return

                time_stop_min_roe = getattr(self.cfg, 'TIME_STOP_MIN_ROE', 1.0)
                if holding_seconds >= time_stop_seconds:
                    if position.max_roe < time_stop_min_roe or roe <= 0.2 or holding_seconds >= time_stop_seconds * 1.5:
                        holding_mins = int(holding_seconds / 60)
                        self._close(position, f"Тайм-стоп ({holding_mins} мин без TP1, ROE {roe:+.1f}%)", current_time=current_time)
                        return

            # 2. TP1 (+3.0% ROE, ФИКСАЦИЯ 50%)
            if not position.tp1_hit and roe >= self.cfg.TP1_ROE:
                close_pct = self.cfg.TP1_CLOSE_PCT / 100.0
                close_notional = position.notional * close_pct
                close_margin = position.margin * close_pct
                gross_profit = close_margin * roe / 100.0
                
                tp1_fee = close_notional * self.taker_fee_rate
                position.total_fees += tp1_fee
                net_tp1_pnl = gross_profit - tp1_fee
                position.realized_pnl += net_tp1_pnl

                position.margin -= close_margin
                position.notional -= close_notional
                self.balance += close_margin
                position.tp1_hit = True
                position.be_mode = True
                
                position.trailing_active = True
                position.max_roe = roe
                
                if self.history_file:
                    notify_tp1(
                        symbol=position.symbol,
                        direction=position.direction,
                        roe=roe,
                        net_pnl=net_tp1_pnl,
                        be_level=self.be_trigger_roe
                    )

                if position.margin <= 0.01 or position.notional < 0.5:
                    self._close(position, "TP1 полный выход", current_time=current_time)
                    return

            # 3. БЕЗУБЫТОК ПОСЛЕ TP1
            if position.tp1_hit and not position.trailing_active:
                if roe <= self.be_trigger_roe:
                    self._close(position, f"Безубыток (+{roe:.1f}% ROE)", current_time=current_time)
                    return

            # 4. ДИНАМИЧЕСКИЙ АДАПТИВНЫЙ ТРЕЙЛИНГ-СТОП ВТОРОЙ ПОЛОВИНЫ
            if position.trailing_active:
                if roe > position.max_roe:
                    position.max_roe = roe
                
                trail_distance = getattr(position, 'trailing_distance', self.cfg.TRAILING_DISTANCE)
                trailing_stop_level = position.max_roe - trail_distance
                effective_stop = max(self.be_trigger_roe, trailing_stop_level)
                
                if roe <= effective_stop:
                    reason = f"Трейлинг-стоп (пик {position.max_roe:.1f}%, стоп {effective_stop:.1f}%, Δ{trail_distance:.1f})"
                    self._close(position, reason, current_time=current_time)
                    return

            # 5. ПЕРВИЧНЫЙ СТОП-ЛОСС (-12.0% ROE)
            if not position.tp1_hit and roe <= self.cfg.STOP_ROE:
                self._close(position, f"Стоп {self.cfg.STOP_ROE}% ROE", current_time=current_time)
                return

    def _close(self, position: Position, reason: str, current_time: Optional[float] = None):
        with self.lock:
            now = current_time if current_time is not None else time.time()
            exit_slip = getattr(position, 'exit_slippage_pct', 0.08)
            if position.direction == 'LONG':
                exec_exit_price = position.current_price * (1.0 - exit_slip / 100.0)
            else:
                exec_exit_price = position.current_price * (1.0 + exit_slip / 100.0)

            position.current_price = exec_exit_price
            exit_fee = position.notional * self.taker_fee_rate
            position.total_fees += exit_fee
            gross_remaining_pnl = position.margin * position.get_roe() / 100.0
            
            total_pnl = position.realized_pnl + gross_remaining_pnl - exit_fee
            total_roe = (total_pnl / position.initial_margin) * 100.0 if position.initial_margin > 0 else 0.0
            duration = now - position.created_at
            self.balance += position.margin + total_pnl

            if total_pnl >= 0:
                self.wins += 1

            trade = {
                'symbol': position.symbol,
                'direction': position.direction,
                'pnl': total_pnl,
                'roe': total_roe,
                'reason': reason,
                'score': position.score,
                'time': time.strftime('%H:%M:%S', time.localtime(now)),
                'entry_price': position.entry_price,
                'avg_price': position.avg_price,
                'exit_price': exec_exit_price,
                'margin': position.initial_margin,
                'notional': position.initial_notional,
                'leverage': position.leverage,
                'duration': duration
            }

            self.history.append(trade)
            if self.history_file:
                self._save_trade_to_csv(trade)

            if position in self.positions:
                self.positions.remove(position)

            is_hard_loss = ("Стоп" in reason and "Тайм" not in reason and "Трейлинг" not in reason) or total_pnl <= -0.10
            cooldown_mins = getattr(self.cfg, 'STOP_COOLDOWN_MINUTES', 360) if is_hard_loss else self.cfg.COOLDOWN_MINUTES
            self.cooldowns[position.symbol] = now + cooldown_mins * 60
            self._save_cooldowns()

            emoji = "💰" if total_pnl >= 0 else "🛑"
            maker_label = "[Maker Post-Only]" if getattr(position, 'is_maker', False) else "[Taker]"
            quarantine_note = f" (Карантин {cooldown_mins} мин / {cooldown_mins//60}ч)" if is_hard_loss else ""
            print(f"  {emoji} Закрыта {position.symbol} {maker_label} · {reason} · Чистый PnL {total_pnl:+.3f}$ (ROE {total_roe:+.1f}%){quarantine_note}")

        if self.history_file:
            notify_close(
                symbol=position.symbol,
                direction=position.direction,
                reason=reason + quarantine_note,
                pnl=total_pnl,
                roe=total_roe,
                fees=position.total_fees,
                balance=self.balance,
                duration_sec=duration
            )

    def _save_trade_to_csv(self, trade: Dict):
        if not self.history_file:
            return
        try:
            with open(self.history_file, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade['time'],
                    trade['symbol'],
                    trade['direction'],
                    f"{trade['entry_price']:.8f}",
                    f"{trade['avg_price']:.8f}",
                    f"{trade['exit_price']:.8f}",
                    f"{trade['margin']:.4f}",
                    f"{trade['notional']:.4f}",
                    trade['leverage'],
                    trade['score'],
                    f"{trade['pnl']:.4f}",
                    f"{trade['roe']:.2f}",
                    trade['reason'],
                    f"{trade['duration']:.1f}"
                ])
        except Exception as e:
            print(f"  ⚠️ Не удалось сохранить сделку в CSV: {e}")