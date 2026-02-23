import os
import re
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from aiogram.enums import ParseMode


# =========================
# CONFIG
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not found")

BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]

SUPPORTED_INTERVALS = {
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","1w"
}
TF_PRETTY = {k: k.upper() for k in SUPPORTED_INTERVALS}

MIN_RR_OK = 1.2
MIN_RR_STRONG = 2.0

# =========================
# BOT
# =========================
bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()


# =========================
# HELPERS
# =========================
def fmt(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if abs(p) >= 1000:
        return f"{p:,.0f}".replace(",", " ")
    if abs(p) >= 1:
        return f"{p:,.2f}".replace(",", " ")
    return f"{p:.6f}"


def rr(entry: float, sl: float, tp: float) -> Optional[float]:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    return reward / risk


def parse_caption(caption: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Examples:
      BTCUSDT 1H
      GUAUSDT 1h
      ETHUSDT 15m
    """
    if not caption:
        return None, None
    t = re.sub(r"\s+", " ", caption.strip())
    parts = t.split(" ")
    if len(parts) < 2:
        return None, None

    symbol = parts[0].upper().replace("/", "").replace("-", "").replace("_", "")
    tf = parts[1].lower().replace("ч", "h").replace("м", "m").replace("д", "d")

    if tf not in SUPPORTED_INTERVALS:
        return symbol, None
    return symbol, tf


# =========================
# BINANCE (SPOT + FUTURES)
# =========================
def fetch_klines(symbol: str, interval: str, limit: int = 400):
    """
    1) пробуем SPOT: /api/v3/klines
    2) если 400/404 -> пробуем FUTURES: /fapi/v1/klines
    Возвращает: (open, high, low, close, market_name)
    """
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    def _get(base: str, path: str):
        return requests.get(f"{base}{path}", params=params, headers=headers, timeout=12)

    last_err = None

    # --- SPOT ---
    for base in BINANCE_BASES:
        try:
            r = _get(base, "/api/v3/klines")
            if r.status_code == 200:
                data = r.json()
                o = np.array([float(x[1]) for x in data], dtype=np.float64)
                h = np.array([float(x[2]) for x in data], dtype=np.float64)
                l = np.array([float(x[3]) for x in data], dtype=np.float64)
                c = np.array([float(x[4]) for x in data], dtype=np.float64)
                return o, h, l, c, "SPOT"

            if r.status_code in (400, 404):
                last_err = f"SPOT {base} -> HTTP {r.status_code}"
                break

            last_err = f"SPOT {base} -> HTTP {r.status_code}"
        except Exception as e:
            last_err = f"SPOT {base} -> {e}"
            continue

    # --- FUTURES ---
    for base in BINANCE_BASES:
        try:
            r = _get(base, "/fapi/v1/klines")
            if r.status_code == 200:
                data = r.json()
                o = np.array([float(x[1]) for x in data], dtype=np.float64)
                h = np.array([float(x[2]) for x in data], dtype=np.float64)
                l = np.array([float(x[3]) for x in data], dtype=np.float64)
                c = np.array([float(x[4]) for x in data], dtype=np.float64)
                return o, h, l, c, "FUTURES"

            last_err = f"FUTURES {base} -> HTTP {r.status_code}"
        except Exception as e:
            last_err = f"FUTURES {base} -> {e}"
            continue

    raise RuntimeError(f"Binance API недоступен / символ не найден. Last: {last_err}")


# =========================
# INDICATORS
# =========================
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

    avg_gain = np.empty(len(arr))
    avg_loss = np.empty(len(arr))
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan

    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    for i in range(period + 1, len(arr)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    rs = avg_gain / (avg_loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    out[:period] = np.nan
    return out


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < period + 1:
        return np.full_like(closes, np.nan)

    prev_close = closes[:-1]
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
    )

    out = np.empty_like(closes)
    out[:] = np.nan
    out[period] = np.mean(tr[:period])
    for i in range(period + 1, len(closes)):
        out[i] = (out[i - 1] * (period - 1) + tr[i - 1]) / period
    return out


# =========================
# MARKET STRUCTURE + LEVELS
# =========================
def swings(highs: np.ndarray, lows: np.ndarray, k: int = 3) -> Tuple[List[int], List[int]]:
    sh, sl = [], []
    n = len(highs)
    for i in range(k, n - k):
        if np.all(highs[i] > highs[i - k : i]) and np.all(highs[i] > highs[i + 1 : i + 1 + k]):
            sh.append(i)
        if np.all(lows[i] < lows[i - k : i]) and np.all(lows[i] < lows[i + 1 : i + 1 + k]):
            sl.append(i)
    return sh, sl


def structure_type(highs: np.ndarray, lows: np.ndarray) -> str:
    sh, sl = swings(highs, lows, k=3)
    if len(sh) < 2 or len(sl) < 2:
        return "Неопределённо"
    last_h, prev_h = highs[sh[-1]], highs[sh[-2]]
    last_l, prev_l = lows[sl[-1]], lows[sl[-2]]

    if last_h < prev_h and last_l < prev_l:
        return "LL/LH (нисходящая структура)"
    if last_h > prev_h and last_l > prev_l:
        return "HH/HL (восходящая структура)"
    return "Переходная"


def pick_levels(price: float, highs: np.ndarray, lows: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Return (S1, S2, R1, R2) based on swing lows/highs.
    S1 closest support below price, S2 next lower.
    R1 closest resistance above price, R2 next higher.
    """
    sh, sl = swings(highs, lows, k=3)
    sh_vals = sorted([float(highs[i]) for i in sh[-60:]])
    sl_vals = sorted([float(lows[i]) for i in sl[-60:]])

    supports = [x for x in sl_vals if x < price]
    resists = [x for x in sh_vals if x > price]

    s1 = supports[-1] if len(supports) >= 1 else None
    s2 = supports[-2] if len(supports) >= 2 else None

    r1 = resists[0] if len(resists) >= 1 else None
    r2 = resists[1] if len(resists) >= 2 else None

    return s1, s2, r1, r2


# =========================
# IMAGE DRAWING (PREMIUM)
# =========================
def crop_chart_area(img: np.ndarray) -> np.ndarray:
    """
    Light crop for Telegram/TradingView mobile screenshots.
    """
    h, w = img.shape[:2]
    x0 = int(w * 0.30) if w > 700 else 0
    y1 = int(h * 0.86) if h > 700 else h
    return img[:y1, x0:w].copy()


def price_to_y(price: float, p_min: float, p_max: float, h: int) -> int:
    if p_max <= p_min:
        return int(h * 0.5)
    t = (price - p_min) / (p_max - p_min)
    y = int((1 - t) * (h - 1))
    return max(0, min(h - 1, y))


def overlay_rect(img: np.ndarray, y0: int, y1: int, color_bgr: Tuple[int, int, int], alpha: float = 0.18):
    h, w = img.shape[:2]
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if y1 < y0:
        y0, y1 = y1, y0
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, y1), color_bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hline(img: np.ndarray, y: int, color_bgr: Tuple[int, int, int], thickness: int = 2):
    h, w = img.shape[:2]
    y = max(0, min(h - 1, y))
    cv2.line(img, (0, y), (w, y), color_bgr, thickness)


def draw_label_left(img: np.ndarray, y: int, text: str, color_bgr: Tuple[int, int, int]):
    """
    Label on LEFT side to avoid price scale on RIGHT.
    """
    h, w = img.shape[:2]
    y = max(18, min(h - 8, y))
    x0 = 10
    x1 = int(w * 0.45)

    cv2.rectangle(img, (x0, y - 18), (x1, y + 8), (255, 255, 255), -1)
    cv2.putText(img, text, (x0 + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)


def draw_direction_arrow(img: np.ndarray, side: str):
    h, w = img.shape[:2]
    x = int(w * 0.93)
    if side == "LONG":
        cv2.arrowedLine(img, (x, int(h * 0.78)), (x, int(h * 0.55)), (0, 160, 0), 4, tipLength=0.25)
    elif side == "SHORT":
        cv2.arrowedLine(img, (x, int(h * 0.55)), (x, int(h * 0.78)), (0, 0, 220), 4, tipLength=0.25)
    else:
        cv2.arrowedLine(img, (x, int(h * 0.66)), (x, int(h * 0.60)), (40, 40, 40), 4, tipLength=0.25)


def draw_plan_on_screenshot(in_path: str, plan: dict, highs: np.ndarray, lows: np.ndarray) -> str:
    img_full = cv2.imread(in_path)
    if img_full is None:
        raise RuntimeError("cv2.imread: не смог прочитать изображение")
    img = crop_chart_area(img_full)

    h, w = img.shape[:2]
    p_min = float(np.min(lows[-200:]))
    p_max = float(np.max(highs[-200:]))

    zone_pad = float(plan["zone_pad"])

    # zones: S1/R1
    s1 = plan["S1"]
    r1 = plan["R1"]

    if s1 is not None:
        y = price_to_y(s1, p_min, p_max, h)
        y0 = price_to_y(s1 - zone_pad, p_min, p_max, h)
        y1 = price_to_y(s1 + zone_pad, p_min, p_max, h)
        overlay_rect(img, y0, y1, (0, 200, 0), alpha=0.16)
        draw_hline(img, y, (0, 170, 0), thickness=2)
        draw_label_left(img, y, f"S1 {fmt(s1)}", (0, 140, 0))

    if r1 is not None:
        y = price_to_y(r1, p_min, p_max, h)
        y0 = price_to_y(r1 - zone_pad, p_min, p_max, h)
        y1 = price_to_y(r1 + zone_pad, p_min, p_max, h)
        overlay_rect(img, y0, y1, (0, 0, 220), alpha=0.14)
        draw_hline(img, y, (0, 0, 220), thickness=2)
        draw_label_left(img, y, f"R1 {fmt(r1)}", (0, 0, 220))

    # lines: entry/sl/tp1/tp2/tp3
    def draw_price_line(key: str, color: Tuple[int, int, int]):
        val = plan.get(key)
        if val is None:
            return
        y = price_to_y(float(val), p_min, p_max, h)
        draw_hline(img, y, color, thickness=1)
        draw_label_left(img, y, f"{key.upper()} {fmt(float(val))}", color)

    draw_price_line("entry", (40, 40, 40))
    draw_price_line("sl", (0, 0, 220))
    draw_price_line("tp1", (0, 140, 0))
    draw_price_line("tp2", (0, 140, 0))
    draw_price_line("tp3", (0, 140, 0))

    draw_direction_arrow(img, plan["side"])

    out_path = in_path.replace(".jpg", "_maxpro.jpg")
    cv2.imwrite(out_path, img)
    return out_path


# =========================
# PLAN LOGIC (ONE SCENARIO)
# =========================
def classify_trend(e20: float, e50: float, e200: float, r: float) -> Tuple[str, str]:
    """
    Returns (trend_text, side)
    """
    if any(math.isnan(x) for x in [e20, e50, e200, r]):
        return "Недостаточно данных", "NEUTRAL"

    if e20 > e50 > e200 and r >= 50:
        return "Восходящий", "LONG"
    if e20 < e50 < e200 and r <= 50:
        return "Нисходящий", "SHORT"
    return "Флет/переход", "NEUTRAL"


def build_plan(symbol: str, tf: str, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, market: str) -> dict:
    price = float(closes[-1])
    e20 = float(ema(closes, 20)[-1])
    e50 = float(ema(closes, 50)[-1])
    e200 = float(ema(closes, 200)[-1])
    r = float(rsi(closes, 14)[-1])
    a = float(atr(highs, lows, closes, 14)[-1])

    trend_text, side = classify_trend(e20, e50, e200, r)
    structure = structure_type(highs, lows)

    S1, S2, R1, R2 = pick_levels(price, highs, lows)

    # ATR-based pads
    zone_pad = (a * 0.35) if not math.isnan(a) else price * 0.003
    stop_pad = (a * 0.90) if not math.isnan(a) else price * 0.008

    # -------------------------
    # Entry logic (more pro):
    # If price is close to the "better" level -> use it.
    # LONG: prefer entry at R1 (breakout/retest) if close; else at S1 (pullback).
    # SHORT: prefer entry at R1 (sell rally) if close; else at S1 (breakdown).
    # -------------------------
    entry = sl = tp1 = tp2 = tp3 = None

    def _is_close(level: Optional[float], to_price: float, thresh: float) -> bool:
        return level is not None and abs(level - to_price) <= thresh

    close_thresh = (a * 0.55) if not math.isnan(a) else price * 0.0055

    if side == "LONG":
        if _is_close(S1, price, close_thresh) and S1 is not None:
            # pullback long
            entry = S1
            sl = S2 - stop_pad if S2 is not None else (S1 - stop_pad)
        elif R1 is not None:
            # breakout/retest long
            entry = R1
            sl = S1 - stop_pad if S1 is not None else (price - stop_pad)
        else:
            side = "NEUTRAL"

        if entry is not None and sl is not None:
            risk = max(1e-9, entry - sl)
            tp1 = entry + risk * 1.0
            tp2 = entry + risk * 2.0
            tp3 = entry + risk * 3.0

    elif side == "SHORT":
        if _is_close(R1, price, close_thresh) and R1 is not None:
            # sell rally
            entry = R1
            sl = R2 + stop_pad if R2 is not None else (R1 + stop_pad)
        elif S1 is not None:
            # breakdown
            entry = S1
            sl = R1 + stop_pad if R1 is not None else (price + stop_pad)
        else:
            side = "NEUTRAL"

        if entry is not None and sl is not None:
            risk = max(1e-9, sl - entry)
            tp1 = entry - risk * 1.0
            tp2 = entry - risk * 2.0
            tp3 = entry - risk * 3.0

    # quality
    strength = "—"
    rr2 = None
    if entry is not None and sl is not None and tp2 is not None:
        rr2 = rr(entry, sl, tp2)
        if rr2 is not None and rr2 >= MIN_RR_STRONG:
            strength = "🔥 High probability"
        elif rr2 is not None and rr2 >= MIN_RR_OK:
            strength = "⚖️ Рабочий"
        else:
            strength = "⚠️ Слабый"

    # If NEUTRAL -> explain
    if side == "NEUTRAL":
        strength = "⚠️ Сетап не сформирован (флет/нет уровней)"

    report = (
        f"📊 <b>{symbol}</b> — <b>{TF_PRETTY.get(tf, tf)}</b> <i>({market})</i>\n"
        f"💰 Цена: <b>{fmt(price)}</b>\n\n"
        f"🔻 Тренд: <b>{trend_text}</b>\n"
        f"📐 Структура: <b>{structure}</b>\n"
        f"📉 RSI: <b>{r:.1f}</b>\n\n"
        f"🧱 Уровни:\n"
        f"• S1: {fmt(S1)}   S2: {fmt(S2)}\n"
        f"• R1: {fmt(R1)}   R2: {fmt(R2)}\n\n"
        f"🎯 Основной сценарий: <b>{side}</b>\n"
        f"Entry: <b>{fmt(entry)}</b>\n"
        f"SL: <b>{fmt(sl)}</b>\n"
        f"TP1: <b>{fmt(tp1)}</b>\n"
        f"TP2: <b>{fmt(tp2)}</b>\n"
        f"TP3: <b>{fmt(tp3)}</b>\n"
        f"{strength}\n\n"
        f"⚠️ <i>Не финсовет.</i>"
    )

    return {
        "symbol": symbol,
        "tf": tf,
        "market": market,
        "price": price,
        "trend": trend_text,
        "side": side,
        "structure": structure,
        "rsi": r,
        "ema20": e20,
        "ema50": e50,
        "ema200": e200,
        "atr": a,
        "S1": S1,
        "S2": S2,
        "R1": R1,
        "R2": R2,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "zone_pad": zone_pad,
        "rr2": rr2,
        "report": report,
    }


# =========================
# HANDLERS
# =========================
@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "Отправь скрин графика + подпись, например:\n"
        "<code>BTCUSDT 1H</code>\n\n"
        "Я верну:\n"
        "• MAX PRO разметку (zones + entry/sl/tp1/tp2/tp3)\n"
        "• Один главный сценарий (A) по тренду\n"
        "• Авто SPOT/FUTURES"
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("⏳ Анализирую рынок...")

    symbol, tf = parse_caption(message.caption)
    if not symbol or not tf:
        await message.answer("Нужна подпись вида <code>BTCUSDT 1H</code> (или ETHUSDT 15m).")
        return
    if not symbol.endswith("USDT"):
        await message.answer("Сейчас поддерживаются пары, которые заканчиваются на <b>USDT</b>.")
        return

    try:
        # download image
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)

        os.makedirs("tmp", exist_ok=True)
        in_path = f"tmp/{photo.file_id}.jpg"
        await bot.download_file(file.file_path, destination=in_path)

        # market data (spot -> futures fallback)
        o, h, l, c, market = fetch_klines(symbol, tf, limit=400)

        plan = build_plan(symbol, tf, o, h, l, c, market)
        out_path = draw_plan_on_screenshot(in_path, plan, h, l)

        await message.answer_photo(photo=FSInputFile(out_path), caption="🧠 MAX PRO разметка готова")
        await message.answer(plan["report"])

    except Exception as e:
        await message.answer(f"❌ Ошибка: <code>{type(e).__name__}: {str(e)[:250]}</code>")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
