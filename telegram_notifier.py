import os
import sys
import time
import logging
import threading
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Полное подавление внутреннего спама telebot в консоли
telebot_logger = logging.getLogger('TeleBot')
telebot_logger.setLevel(logging.CRITICAL)
telebot_logger.handlers = []
telebot_logger.propagate = False

TG_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip(' "\'')
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip(' "\'')
TG_PROXY = os.getenv("TG_PROXY", "").strip(' "\'')

tg_bot = None
bot_instance = None

try:
    import telebot
    from telebot import apihelper
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

    if TG_PROXY:
        apihelper.proxy = {'https': TG_PROXY, 'http': TG_PROXY}
        print(f"🔒 [Telegram] Активирован прокси: {TG_PROXY}")

    apihelper.CONNECT_TIMEOUT = 10
    apihelper.READ_TIMEOUT = 25

    if TG_TOKEN:
        tg_bot = telebot.TeleBot(TG_TOKEN)
        # Отключаем логирование на уровне экземпляра
        telebot.logger.setLevel(logging.CRITICAL)
        telebot.logger.handlers = []
        telebot.logger.propagate = False
except Exception:
    tg_bot = None

def set_bot_instance(instance):
    """Связывает Telegram-бота с главным торговым оркестратором."""
    global bot_instance
    bot_instance = instance

def get_main_keyboard():
    """Интерактивная клавиатура управления под сообщениями."""
    try:
        from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton("📊 Статус", callback_data="cmd_status"),
            InlineKeyboardButton("💰 Баланс", callback_data="cmd_balance")
        )
        markup.row(
            InlineKeyboardButton("⏸ Пауза", callback_data="cmd_pause"),
            InlineKeyboardButton("▶️ Старт", callback_data="cmd_resume")
        )
        return markup
    except Exception:
        return None

def send_telegram_message(msg: str, reply_markup=None):
    """Универсальная безопасная отправка сообщений администратору без падений."""
    if not tg_bot or not TG_CHAT_ID:
        return
    try:
        tg_bot.send_message(TG_CHAT_ID, msg, parse_mode='HTML', reply_markup=reply_markup, timeout=8)
    except Exception:
        # Полное игнорирование сетевых ошибок при отправке
        pass

def notify_entry(symbol, direction, price, margin, leverage, score, ml_prob, liq_z, oi_change, long_count, short_count, max_directional):
    """Оповещение об открытии новой сделки."""
    ml_display = f"{ml_prob:.2f}" if ml_prob is not None else "N/A"
    msg = (
        f"⚡️ <b>ОТКРЫТА ПОЗИЦИЯ</b>\n"
        f"Пара: <b>#{symbol}</b>\n"
        f"Тип: <b>{direction}</b> {leverage}x\n"
        f"Цена входа: <code>{price:.5f}</code>\n"
        f"Маржа: {margin:.2f}$\n\n"
        f"🎯 Score: {score} | P(win): {ml_display}\n"
    )
    send_telegram_message(msg, reply_markup=get_main_keyboard())

def notify_tp1(symbol, direction, roe, net_pnl, be_level):
    """Оповещение о фиксации 50% объема (TP1)."""
    msg = (
        f"✅ <b>TP1 ВЗЯТ (Сброс 50%)</b>\n"
        f"Пара: <b>#{symbol}</b> ({direction})\n"
        f"Прибыль: <b>+{net_pnl:.3f}$</b> (+{roe:.2f}% ROE)\n"
        f"🛡 <i>Остаток в безубытке: +{be_level}% ROE</i>"
    )
    send_telegram_message(msg)

def notify_close(symbol, direction, reason, pnl, roe, fees, balance, duration_sec):
    """Оповещение о полном закрытии сделки."""
    icon = "💰" if pnl >= 0 else "🛑"
    duration_min = int(duration_sec // 60)
    
    msg = (
        f"{icon} <b>ПОЗИЦИЯ ЗАКРЫТА</b>\n"
        f"Пара: <b>#{symbol}</b> ({direction})\n"
        f"Причина: <i>{reason}</i>\n\n"
        f"ROE: <b>{roe:+.2f}%</b>\n"
        f"Чистый PnL: <b>{pnl:+.3f}$</b>\n"
        f"Время в сделке: {duration_min} мин\n"
        f"💳 Баланс: {balance:.2f}$"
    )
    send_telegram_message(msg, reply_markup=get_main_keyboard())

def send_status(chat_id):
    """Сбор и отправка текущего статуса торгового оркестратора."""
    if not bot_instance:
        if tg_bot:
            try:
                tg_bot.send_message(chat_id, "⚠️ Торговый комплекс инициализируется...", timeout=8)
            except Exception:
                pass
        return

    try:
        payload = bot_instance.get_status_payload()
        bal = payload.get('balance', 0.0)
        is_run = payload.get('is_running', False)
        pos = payload.get('positions', [])
        winrate = payload.get('winrate', 0.0)
        pnl = payload.get('total_pnl', 0.0)
        
        status_text = "🟢 АКТИВЕН (Поиск сделок)" if is_run else "⏸ НА ПАУЗЕ"
        
        msg = f"📊 <b>Терминал BOtoFF Quant V5</b>\n\n"
        msg += f"Состояние: {status_text}\n"
        msg += f"Баланс: <b>{bal:.2f}$</b>\n"
        msg += f"Чистый PnL: <b>{pnl:+.2f}$</b>\n"
        msg += f"Винрейт: <b>{winrate:.1f}%</b>\n\n"
        msg += f"📦 Открытых позиций: {len(pos)}\n"
        
        for p in pos:
            sym = p.get('symbol', 'UNK')
            dir_ = p.get('direction', 'UNK')
            roe = p.get('roe', 0.0)
            pnl_pos = p.get('pnl', 0.0)
            icon = "🟢" if roe >= 0 else "🔴"
            msg += f"{icon} #{sym} {dir_}: <b>{roe:+.2f}%</b> ({pnl_pos:+.2f}$)\n"
            
        if not pos:
            msg += "<i>Нет активных позиций. Сканирование рынка...</i>\n"
            
        tg_bot.send_message(chat_id, msg, parse_mode='HTML', reply_markup=get_main_keyboard(), timeout=8)
    except Exception:
        pass

if tg_bot:
    @tg_bot.message_handler(commands=['start', 'help', 'status', 'balance'])
    def handle_commands(message):
        try:
            if TG_CHAT_ID and str(message.chat.id) != str(TG_CHAT_ID):
                return
            send_status(message.chat.id)
        except Exception:
            pass

    @tg_bot.callback_query_handler(func=lambda call: True)
    def handle_callback_query(call):
        try:
            if TG_CHAT_ID and str(call.message.chat.id) != str(TG_CHAT_ID):
                return
                
            if not bot_instance:
                try:
                    tg_bot.answer_callback_query(call.id, "Бот недоступен!", timeout=5)
                except Exception:
                    pass
                return

            if call.data in ["cmd_status", "cmd_balance"]:
                send_status(call.message.chat.id)
                try:
                    tg_bot.answer_callback_query(call.id, timeout=5)
                except Exception:
                    pass
                
            elif call.data == "cmd_pause":
                bot_instance.is_running = False
                try:
                    tg_bot.answer_callback_query(call.id, "⏸ Сканер на паузе", timeout=5)
                except Exception:
                    pass
                send_status(call.message.chat.id)
                
            elif call.data == "cmd_resume":
                bot_instance.is_running = True
                try:
                    tg_bot.answer_callback_query(call.id, "▶️ Сканер запущен", timeout=5)
                except Exception:
                    pass
                send_status(call.message.chat.id)
        except Exception:
            pass

def _poll_telegram():
    """
    Кастомный устойчивый цикл опроса.
    Полностью исключает падения с многостраничными трейсбеками telebot в терминал.
    """
    if not tg_bot:
        return

    # Первичная проверка доступности серверов Telegram
    is_online = False
    try:
        me = tg_bot.get_me()
        if me:
            print(f"📱 [Telegram] Бот @{me.username} успешно подключен к сети.")
            is_online = True
    except Exception:
        print("⚠️ [Telegram] Прямое соединение с api.telegram.org таймаутит (блокировка/проблемы сети).")
        print("ℹ️ [Telegram] Комплекс продолжает работу в штатном режиме, опрос Telegram переведен в бесшумный режим.")

    offset = 0

    while True:
        try:
            # Ручной запрос обновлений без infinity_polling, перехватывающий любые таймауты
            updates = tg_bot.get_updates(offset=offset, limit=20, timeout=15, long_polling_timeout=15)
            if updates:
                for update in updates:
                    offset = max(offset, update.update_id + 1)
                tg_bot.process_new_updates(updates)
            time.sleep(0.5)
        except Exception:
            # При таймауте спим 10 секунд без вывода трейсбеков
            time.sleep(10)

def start_telegram_menu_thread():
    """Запуск фонового демонического потока опроса Telegram."""
    if not tg_bot or not TG_CHAT_ID:
        print("⚠️ [Telegram] TG_BOT_TOKEN или TG_CHAT_ID не заданы в .env. Модуль Telegram выключен.")
        return

    t = threading.Thread(target=_poll_telegram, daemon=True)
    t.start()