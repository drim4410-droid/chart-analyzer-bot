import os
import requests
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not found")

bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]


def fetch_klines(symbol: str, interval: str, limit: int = 200):
    url_path = "/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    for base in BINANCE_BASES:
        try:
            r = requests.get(f"{base}{url_path}", params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                closes = np.array([float(x[4]) for x in data])
                highs = np.array([float(x[2]) for x in data])
                lows = np.array([float(x[3]) for x in data])
                return closes, highs, lows
        except:
            continue

    raise RuntimeError("Binance API недоступен")


def ema(data, period):
    return np.convolve(data, np.ones(period)/period, mode='valid')


def rsi(data, period=14):
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "Отправь скрин и подпись:\n\n"
        "<code>BTCUSDT 1H</code>"
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    caption = message.caption

    if not caption:
        await message.answer("Добавь подпись вида BTCUSDT 1H")
        return

    parts = caption.upper().split()

    if len(parts) != 2:
        await message.answer("Формат: BTCUSDT 1H")
        return

    symbol, interval = parts

    interval = interval.lower()

    await message.answer("⏳ Получаю данные...")

    try:
        closes, highs, lows = fetch_klines(symbol, interval)

        last_price = closes[-1]
        ema20 = ema(closes, 20)[-1]
        ema50 = ema(closes, 50)[-1]
        rsi_val = rsi(closes)

        trend = "Флет"
        if ema20 > ema50:
            trend = "Восходящий"
        elif ema20 < ema50:
            trend = "Нисходящий"

        response = (
            f"📊 {symbol} {interval.upper()}\n\n"
            f"Цена: {last_price:.2f}\n"
            f"Тренд: {trend}\n"
            f"EMA20: {ema20:.2f}\n"
            f"EMA50: {ema50:.2f}\n"
            f"RSI: {rsi_val:.1f}\n\n"
            f"⚠️ Не финсовет"
        )

        await message.answer(response)

    except Exception as e:
        await message.answer(f"Ошибка: {str(e)}")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
