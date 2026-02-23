import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import requests
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not found in environment variables")

bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]


# ---------- Parsing helpers ----------

_INTERVAL_ALIASES = {
    # minutes
    "1m": "1m", "1min": "1m",
    "3m": "3m", "3min": "3m",
    "5m": "5m", "5min": "5m",
    "15m": "15m", "15min": "15m",
    "30m": "30m", "30min": "30m",
    # hours
    "1h": "1h", "1hr": "1h", "60m": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    # days / weeks
    "1d": "1d", "1day": "1d", "d": "1d",
    "1w": "1w", "w": "1w",
}

_TF_PRETTY = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "8h": "8H", "12h": "12H",
    "1d": "1D", "1w": "1W",
}


def normalize_symbol(raw: str) -> str:
    s = raw.upper().strip()
    s = s.replace(" ", "")
    s = s.replace("-", "")
    s = s.replace("_", "")
    s = s.replace("/", "")
    # common: BTCUSDT, ETHUSDT
    return s


def parse_caption(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Expects like:
      BTCUSDT 1H
      BTC/USDT 4h
      ethusdt 15m
    """
    if not text:
        return None, None

    t = text.strip()
    t = re.sub(r"\s+", " ", t)

    # Find timeframe token
    tf = None
    for token in re.split(r"[ ,;|\n]+", t.lower()):
        token = token.strip()
        if not token:
            continue
        token = token.replace("hours", "h").replace("hour", "h")
        token = token.replace("mins", "m").replace("min", "m")
        token = token.replace("days", "d").replace("day", "d")
        token = token.replace("weeks", "w").replace("week", "w")
        token = token.replace("ч", "h").replace("м", "m").replace("д", "d")  # RU short
        token = token.replace("1ч", "1h").replace("4ч", "4h").replace("15м", "15m")
        if token in _INTERVAL_ALIASES:
            tf = _INTERVAL_ALIASES[token]
            break
        # allow "1H" "4H" etc
        if re.fullmatch(r"\d+[mhdw]", token):
            if token in _INTERVAL_ALIASES:
                tf = _INTERVAL_ALIASES[token]
                break
            # if not in aliases, still allow if binance supports
            if token in {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"}:
                tf = token
                break

    # Find symbol-like token
    sym = None
    for token in re.split(r"[ ,;|\n]+", t):
        token = token.strip()
        if not token:
            continue
        # skip timeframe tokens
        low = token.lower()
        low = low.replace("ч", "h").replace("м", "m").replace("д", "d")
        if low in _INTERVAL_ALIASES or re.fullmatch(r"\d+[mhdw]", low):
            continue

        candidate = normalize_symbol(token)
        # Basic heuristic: must end with USDT (crypto focus) and be reasonable length
        if candidate.endswith("USDT") and 6 <= len(candidate) <= 20:
            sym = candidate
            break

    return sym, tf


# ---------- Binance + indicators ----------

@dataclass
class MarketStats:
    symbol: str
    interval: str
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    last_price: float


def fetch_klines(symbol: str, interval: str, limit: int = 300) -> MarketStats:
    url_path = "/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    last_err = None

    for base in BINANCE_BASES:
        try:
            r = requests.get(f"{base}{url_path}", params=params, headers=headers, timeout=15)

            if r.status_code != 200:
                last_err = f"{base} -> HTTP {r.status_code}"
                continue

            data = r.json()

            closes = np.array([float(x[4]) for x in data], dtype=np.float64)
            highs = np.array([float(x[2]) for x in data], dtype=np.float64)
            lows = np.array([float(x[3]) for x in data], dtype=np.float64)
            last_price = float(closes[-1])

            return MarketStats(
                symbol=symbol,
                interval=interval,
                closes=closes,
                highs=highs,
                lows=lows,
                last_price=last_price,
            )

        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(f"Binance endpoints unavailable: {last_err}")
        symbol=symbol,
        interval=interval,
        closes=closes,
        highs=highs,
        lows=lows,
        last_price=last_price,
    )


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    alpha = 2 / (period + 1)
    out = np.empty_like(arr)
    out[:] = np.nan
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
    if len(arr) < period + 1:
        return np.full_like(arr, np.nan)
    diff = np.diff(arr)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)

    avg_gain = np.empty_like(arr)
    avg_loss = np.empty_like(arr)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan

    # seed
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    for i in range(period + 1, len(arr)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    rs = avg_gain / (avg_loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    out[:period] = np.nan
    return out


def find_swings(highs: np.ndarray, lows: np.ndarray, lookback: int = 3) -> Tuple[List[int], List[int]]:
    """
    Swing high: high[i] > highs[i-k..i-1] and high[i] > highs[i+1..i+k]
    Swing low: low[i] < lows[i-k..i-1] and low[i] < lows[i+1..i+k]
    """
    sh, sl = [], []
    n = len(highs)
    for i in range(lookback, n - lookback):
        h = highs[i]
        l = lows[i]
        if np.all(h > highs[i - lookback:i]) and np.all(h > highs[i + 1:i + 1 + lookback]):
            sh.append(i)
        if np.all(l < lows[i - lookback:i]) and np.all(l < lows[i + 1:i + 1 + lookback]):
            sl.append(i)
    return sh, sl


def nearest_levels(price: float, swing_highs: List[float], swing_lows: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (support, resistance) closest to current price.
    Support = nearest swing low below price
    Resistance = nearest swing high above price
    """
    support = None
    resistance = None

    below = [x for x in swing_lows if x < price]
    above = [x for x in swing_highs if x > price]

    if below:
        support = max(below)
    if above:
        resistance = min(above)

    return support, resistance


def trend_label(e20: float, e50: float, e200: float, rsi_last: float) -> str:
    if any(np.isnan(x) for x in [e20, e50, e200, rsi_last]):
        return "Недостаточно данных"

    if e20 > e50 > e200:
        if rsi_last >= 55:
            return "Восходящий (сильный)"
        return "Восходящий"
    if e20 < e50 < e200:
        if rsi_last <= 45:
            return "Нисходящий (сильный)"
        return "Нисходящий"
    return "Флет / переходная фаза"


def rr(entry: float, stop: float, target: float) -> Optional[float]:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    return reward / risk


def fmt_price(p: Optional[float]) -> str:
    if p is None:
        return "—"
    # Pretty formatting: fewer decimals for big prices
    if p >= 1000:
        return f"{p:,.0f}".replace(",", " ")
    if p >= 1:
        return f"{p:,.2f}".replace(",", " ")
    return f"{p:.6f}"


# ---------- Bot logic ----------

def build_report(stats: MarketStats) -> str:
    closes = stats.closes
    highs = stats.highs
    lows = stats.lows
    price = stats.last_price

    e20 = float(ema(closes, 20)[-1])
    e50 = float(ema(closes, 50)[-1])
    e200 = float(ema(closes, 200)[-1])
    rsi14 = float(rsi(closes, 14)[-1])

    sh_idx, sl_idx = find_swings(highs, lows, lookback=3)
    swing_high_vals = [float(highs[i]) for i in sh_idx[-20:]]  # last swings only
    swing_low_vals = [float(lows[i]) for i in sl_idx[-20:]]

    support, resistance = nearest_levels(price, swing_high_vals, swing_low_vals)

    trend = trend_label(e20, e50, e200, rsi14)

    # scenarios
    # Bullish: break & retest resistance or bounce from support
    bull_entry = resistance if resistance is not None else price
    bull_stop = support if support is not None else price * 0.98
    bull_target = (bull_entry * 1.02) if resistance is None else (resistance * 1.02)

    # Bearish: break support or reject from resistance
    bear_entry = support if support is not None else price
    bear_stop = resistance if resistance is not None else price * 1.02
    bear_target = (bear_entry * 0.98) if support is None else (support * 0.98)

    bull_rr = rr(bull_entry, bull_stop, bull_target)
    bear_rr = rr(bear_entry, bear_stop, bear_target)

    tf_pretty = _TF_PRETTY.get(stats.interval, stats.interval)

    levels_line = (
        f"🧱 <b>Поддержка:</b> {fmt_price(support)}\n"
        f"🧱 <b>Сопротивление:</b> {fmt_price(resistance)}"
    )

    indicators_line = (
        f"📌 <b>EMA20/50/200:</b> {fmt_price(e20)} / {fmt_price(e50)} / {fmt_price(e200)}\n"
        f"📌 <b>RSI(14):</b> {rsi14:.1f}"
    )

    def rr_text(x: Optional[float]) -> str:
        return "—" if x is None else f"~1:{x:.2f}"

    msg = (
        f"📊 <b>{stats.symbol}</b> — <b>{tf_pretty}</b>\n"
        f"💰 <b>Цена:</b> {fmt_price(price)}\n\n"
        f"🔎 <b>Тренд:</b> {trend}\n\n"
        f"{indicators_line}\n\n"
        f"{levels_line}\n\n"
        f"🧠 <b>Сценарии</b>\n"
        f"✅ <b>Bullish:</b>\n"
        f"• Вход: {fmt_price(bull_entry)} (закрепление/ретест)\n"
        f"• Invalidation (SL): {fmt_price(bull_stop)}\n"
        f"• Цель (TP): {fmt_price(bull_target)}\n"
        f"• RR: {rr_text(bull_rr)}\n\n"
        f"🟥 <b>Bearish:</b>\n"
        f"• Вход: {fmt_price(bear_entry)} (пробой/подтверждение)\n"
        f"• Invalidation (SL): {fmt_price(bear_stop)}\n"
        f"• Цель (TP): {fmt_price(bear_target)}\n"
        f"• RR: {rr_text(bear_rr)}\n\n"
        f"⚠️ <i>Не финсовет. Это алгоритмический теханализ по данным Binance.</i>"
    )
    return msg


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "Привет 👋\n\n"
        "Отправь <b>скриншот графика</b> и добавь подпись:\n"
        "<code>BTCUSDT 1H</code>\n\n"
        "Примеры:\n"
        "• <code>ETHUSDT 15m</code>\n"
        "• <code>SOLUSDT 4h</code>\n\n"
        "Я подтяну свечи с Binance и сделаю полный теханализ (EMA/RSI/уровни/сценарии)."
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    sym, tf = parse_caption(message.caption)

    if not sym or not tf:
        await message.answer(
            "Нужна подпись к скрину, чтобы я понял что анализировать.\n\n"
            "Напиши в подписи, например:\n"
            "<code>BTCUSDT 1H</code>\n"
            "или\n"
            "<code>ETHUSDT 15m</code>"
        )
        return

    # Скачаем фото (чтобы сохранить привычное поведение; в этой версии фото не анализируем)
    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        os.makedirs("tmp", exist_ok=True)
        path = f"tmp/{photo.file_id}.jpg"
        await bot.download_file(file.file_path, destination=path)
    except Exception:
        # Даже если скачивание не удалось — анализ по данным можно сделать
        pass

    await message.answer("⏳ Анализирую данные Binance...")

    try:
        stats = fetch_klines(sym, tf, limit=300)
        report = build_report(stats)
        await message.answer(report)
    except requests.HTTPError:
        await message.answer(
            "❌ Не смог получить данные с Binance.\n"
            "Проверь тикер и формат, например: <code>BTCUSDT 1H</code>\n"
            "Важно: сейчас поддерживаются пары, которые заканчиваются на <b>USDT</b>."
        )
    except Exception as e:
        await message.answer(f"❌ Ошибка анализа: <code>{type(e).__name__}</code>")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
