import os
import re
import math
import cv2
import numpy as np
import requests
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
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

SUPPORTED_INTERVALS = {
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","1w"
}
TF_PRETTY = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1H","2h":"2H","4h":"4H","6h":"6H","8h":"8H","12h":"12H",
    "1d":"1D","1w":"1W"
}


# ----------------- Parsing -----------------

def parse_caption(caption: str | None):
    """
    Expected: BTCUSDT 1H, ETHUSDT 15m, SOLUSDT 4h
    """
    if not caption:
        return None, None
    t = re.sub(r"\s+", " ", caption.strip())
    parts = t.split(" ")
    if len(parts) < 2:
        return None, None

    symbol = parts[0].upper().replace("/", "").replace("-", "").replace("_", "")
    tf = parts[1].lower().replace("ч", "h").replace("м", "m").replace("д", "d")

    # allow "1H" -> "1h"
    tf = tf.lower()
    if tf not in SUPPORTED_INTERVALS:
        return symbol, None

    return symbol, tf


# ----------------- Binance data -----------------

def fetch_klines(symbol: str, interval: str, limit: int = 350):
    path = "/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_err = None

    for base in BINANCE_BASES:
        try:
            r = requests.get(f"{base}{path}", params=params, timeout=12)
            if r.status_code != 200:
                last_err = f"{base} -> HTTP {r.status_code}"
                continue

            data = r.json()
            opens  = np.array([float(x[1]) for x in data], dtype=np.float64)
            highs  = np.array([float(x[2]) for x in data], dtype=np.float64)
            lows   = np.array([float(x[3]) for x in data], dtype=np.float64)
            closes = np.array([float(x[4]) for x in data], dtype=np.float64)
            return opens, highs, lows, closes

        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(f"Binance API недоступен: {last_err}")


# ----------------- Indicators -----------------

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
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)))

    out = np.empty_like(closes)
    out[:] = np.nan
    out[period] = np.mean(tr[:period])
    for i in range(period + 1, len(closes)):
        out[i] = (out[i - 1] * (period - 1) + tr[i - 1]) / period
    return out

def swings(highs: np.ndarray, lows: np.ndarray, k: int = 3):
    sh, sl = [], []
    n = len(highs)
    for i in range(k, n - k):
        if np.all(highs[i] > highs[i-k:i]) and np.all(highs[i] > highs[i+1:i+1+k]):
            sh.append(i)
        if np.all(lows[i] < lows[i-k:i]) and np.all(lows[i] < lows[i+1:i+1+k]):
            sl.append(i)
    return sh, sl

def nearest_levels(price: float, sh_vals: list[float], sl_vals: list[float]):
    support = None
    resistance = None
    below = [x for x in sl_vals if x < price]
    above = [x for x in sh_vals if x > price]
    if below:
        support = max(below)
    if above:
        resistance = min(above)
    return support, resistance


# ----------------- Formatting -----------------

def fmt(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p >= 1000:
        return f"{p:,.0f}".replace(",", " ")
    if p >= 1:
        return f"{p:,.2f}".replace(",", " ")
    return f"{p:.6f}"

def rr(entry: float, sl: float, tp: float) -> float | None:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    return reward / risk


# ----------------- Drawing helpers -----------------

def crop_chart_area(img: np.ndarray) -> np.ndarray:
    """
    Обрезаем фон/панели. Под моб. TradingView + Telegram обычно подходит.
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

def overlay_rect(img: np.ndarray, y0: int, y1: int, color_bgr: tuple[int,int,int], alpha: float = 0.25):
    h, w = img.shape[:2]
    y0 = max(0, min(h-1, y0))
    y1 = max(0, min(h-1, y1))
    if y1 < y0:
        y0, y1 = y1, y0
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, y1), color_bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_label(img: np.ndarray, y: int, text: str, color_bgr: tuple[int,int,int]):
    h, w = img.shape[:2]
    y = max(0, min(h-1, y))
    x = int(w * 0.60)
    cv2.rectangle(img, (x, y - 18), (w - 10, y + 8), (255, 255, 255), -1)
    cv2.putText(img, text, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)

def draw_line(img: np.ndarray, y: int, color_bgr: tuple[int,int,int], thickness: int = 3):
    h, w = img.shape[:2]
    y = max(0, min(h-1, y))
    cv2.line(img, (0, y), (w, y), color_bgr, thickness)

def draw_arrow(img: np.ndarray, direction: str):
    h, w = img.shape[:2]
    if direction == "LONG":
        cv2.arrowedLine(img, (int(w*0.94), int(h*0.75)), (int(w*0.94), int(h*0.55)), (0, 160, 0), 4, tipLength=0.25)
    elif direction == "SHORT":
        cv2.arrowedLine(img, (int(w*0.94), int(h*0.55)), (int(w*0.94), int(h*0.75)), (0, 0, 220), 4, tipLength=0.25)
    else:
        cv2.arrowedLine(img, (int(w*0.94), int(h*0.65)), (int(w*0.94), int(h*0.60)), (0, 0, 0), 4, tipLength=0.25)


# ----------------- Analysis -----------------

def classify_trend(e20: float, e50: float, e200: float, r: float) -> tuple[str, str]:
    """
    Returns: (trend_text, main_side)
    main_side: LONG/SHORT/NEUTRAL
    """
    if any(math.isnan(x) for x in [e20, e50, e200, r]):
        return "Недостаточно данных", "NEUTRAL"

    if e20 > e50 > e200 and r >= 50:
        return "Восходящий", "LONG"
    if e20 < e50 < e200 and r <= 50:
        return "Нисходящий", "SHORT"
    return "Флет / переходная фаза", "NEUTRAL"


def build_plan(symbol: str, tf: str, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
    price = float(closes[-1])

    e20 = float(ema(closes, 20)[-1])
    e50 = float(ema(closes, 50)[-1])
    e200 = float(ema(closes, 200)[-1])
    r = float(rsi(closes, 14)[-1])
    a = float(atr(highs, lows, closes, 14)[-1])

    sh_idx, sl_idx = swings(highs, lows, k=3)
    sh_vals = [float(highs[i]) for i in sh_idx[-30:]]
    sl_vals = [float(lows[i]) for i in sl_idx[-30:]]

    support, resistance = nearest_levels(price, sh_vals, sl_vals)
    trend_text, side = classify_trend(e20, e50, e200, r)

    # ширина зон и стопов через ATR (если ATR не посчитался — запас 1% цены)
    zone_pad = a * 0.35 if not math.isnan(a) else price * 0.0035
    stop_pad = a * 0.80 if not math.isnan(a) else price * 0.008

    # План: делаем основной сценарий по стороне тренда
    if side == "LONG":
        entry = resistance if resistance else price
        sl = (support - stop_pad) if support else (price - stop_pad)
        risk = max(1e-9, entry - sl)
        tp1 = entry + risk * 1.0
        tp2 = entry + risk * 2.0

        secondary = "SHORT"

    elif side == "SHORT":
        entry = support if support else price
        sl = (resistance + stop_pad) if resistance else (price + stop_pad)
        risk = max(1e-9, sl - entry)
        tp1 = entry - risk * 1.0
        tp2 = entry - risk * 2.0

        secondary = "LONG"

    else:
        # нейтрально: дадим аккуратно лонг от поддержки и шорт от сопротивления
        entry = price
        sl = price - stop_pad
        tp1 = price + stop_pad
        tp2 = price + stop_pad * 2
        secondary = "BOTH"

    report = (
        f"📊 <b>{symbol}</b> — <b>{TF_PRETTY.get(tf, tf)}</b>\n"
        f"💰 <b>Цена:</b> {fmt(price)}\n\n"
        f"🔎 <b>Тренд:</b> {trend_text}\n"
        f"📌 <b>EMA20/50/200:</b> {fmt(e20)} / {fmt(e50)} / {fmt(e200)}\n"
        f"📌 <b>RSI(14):</b> {r:.1f}\n"
        f"📌 <b>ATR(14):</b> {fmt(a) if not math.isnan(a) else '—'}\n\n"
        f"🧱 <b>Support:</b> {fmt(support)}\n"
        f"🧱 <b>Resistance:</b> {fmt(resistance)}\n\n"
        f"🧠 <b>Основной план ({'LONG' if side=='LONG' else 'SHORT' if side=='SHORT' else 'NEUTRAL'}):</b>\n"
        f"• Entry: {fmt(entry)}\n"
        f"• SL: {fmt(sl)}\n"
        f"• TP1: {fmt(tp1)} (RR {('—' if rr(entry, sl, tp1) is None else f'~1:{rr(entry, sl, tp1):.2f}')})\n"
        f"• TP2: {fmt(tp2)} (RR {('—' if rr(entry, sl, tp2) is None else f'~1:{rr(entry, sl, tp2):.2f}')})\n\n"
        f"⚠️ <i>Не финсовет. Алгоритмический анализ.</i>"
    )

    return {
        "price": price,
        "support": support,
        "resistance": resistance,
        "trend_text": trend_text,
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "zone_pad": zone_pad,
        "report": report
    }


def draw_premium(in_path: str, plan: dict, highs: np.ndarray, lows: np.ndarray) -> str:
    img_full = cv2.imread(in_path)
    if img_full is None:
        raise RuntimeError("cv2.imread: не смог прочитать изображение")
    img = crop_chart_area(img_full)

    h, w = img.shape[:2]

    # price range for mapping (берём последние 200 свечей)
    p_min = float(np.min(lows[-200:]))
    p_max = float(np.max(highs[-200:]))

    support = plan["support"]
    resistance = plan["resistance"]
    zone_pad = float(plan["zone_pad"])

    # --- зоны ---
    if support:
        y_sup = price_to_y(support, p_min, p_max, h)
        y0 = price_to_y(support - zone_pad, p_min, p_max, h)
        y1 = price_to_y(support + zone_pad, p_min, p_max, h)
        overlay_rect(img, y0, y1, (0, 200, 0), alpha=0.22)
        draw_line(img, y_sup, (0, 170, 0), thickness=3)
        draw_label(img, y_sup, f"SUP {fmt(support)}", (0, 130, 0))

    if resistance:
        y_res = price_to_y(resistance, p_min, p_max, h)
        y0 = price_to_y(resistance - zone_pad, p_min, p_max, h)
        y1 = price_to_y(resistance + zone_pad, p_min, p_max, h)
        overlay_rect(img, y0, y1, (0, 0, 220), alpha=0.18)
        draw_line(img, y_res, (0, 0, 220), thickness=3)
        draw_label(img, y_res, f"RES {fmt(resistance)}", (0, 0, 220))

    # --- метки Entry/SL/TP1/TP2 ---
    y_entry = price_to_y(plan["entry"], p_min, p_max, h)
    y_sl    = price_to_y(plan["sl"], p_min, p_max, h)
    y_tp1   = price_to_y(plan["tp1"], p_min, p_max, h)
    y_tp2   = price_to_y(plan["tp2"], p_min, p_max, h)

    draw_line(img, y_entry, (40, 40, 40), thickness=2)
    draw_label(img, y_entry, f"ENTRY {fmt(plan['entry'])}", (0, 0, 0))

    draw_line(img, y_sl, (0, 0, 220), thickness=2)
    draw_label(img, y_sl, f"SL {fmt(plan['sl'])}", (0, 0, 220))

    draw_line(img, y_tp1, (0, 140, 0), thickness=2)
    draw_label(img, y_tp1, f"TP1 {fmt(plan['tp1'])}", (0, 140, 0))

    draw_line(img, y_tp2, (0, 140, 0), thickness=2)
    draw_label(img, y_tp2, f"TP2 {fmt(plan['tp2'])}", (0, 140, 0))

    # --- стрелка направления ---
    draw_arrow(img, plan["side"])

    out_path = in_path.replace(".jpg", "_premium.jpg")
    cv2.imwrite(out_path, img)
    return out_path


# ----------------- Bot handlers -----------------

@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "Отправь скрин графика + подпись:\n"
        "<code>BTCUSDT 1H</code>\n\n"
        "Я верну:\n"
        "• премиум-разметку (зоны, entry/sl/tp1/tp2)\n"
        "• полный текстовый анализ"
    )

@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("⏳ Обрабатываю скрин + тяну данные Binance...")

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

        # fetch market data
        opens, highs, lows, closes = fetch_klines(symbol, tf, limit=350)

        plan = build_plan(symbol, tf, highs, lows, closes)
        out_path = draw_premium(in_path, plan, highs, lows)

        await message.answer_photo(photo=FSInputFile(out_path), caption="🧠 Премиум-разметка готова")
        await message.answer(plan["report"])

    except Exception as e:
        await message.answer(f"❌ Ошибка: <code>{type(e).__name__}: {str(e)[:250]}</code>")


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
