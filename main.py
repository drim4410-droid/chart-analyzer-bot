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


def parse_caption(caption: str | None):
    """
    BTCUSDT 1H, ETHUSDT 15m, SOLUSDT 4h
    """
    if not caption:
        return None, None
    t = re.sub(r"\s+", " ", caption.strip())
    parts = t.split(" ")
    if len(parts) < 2:
        return None, None
    symbol = parts[0].upper().replace("/", "").replace("-", "").replace("_", "")
    tf = parts[1].lower()
    tf = tf.replace("ч", "h").replace("м", "m").replace("д", "d")
    if tf not in SUPPORTED_INTERVALS:
        return symbol, None
    return symbol, tf


def fetch_klines(symbol: str, interval: str, limit: int = 300):
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
            opens = np.array([float(x[1]) for x in data], dtype=np.float64)
            highs = np.array([float(x[2]) for x in data], dtype=np.float64)
            lows  = np.array([float(x[3]) for x in data], dtype=np.float64)
            closes= np.array([float(x[4]) for x in data], dtype=np.float64)
            return opens, highs, lows, closes
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"Binance API недоступен: {last_err}")


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


def fmt(p: float | None):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p >= 1000:
        return f"{p:,.0f}".replace(",", " ")
    if p >= 1:
        return f"{p:,.2f}".replace(",", " ")
    return f"{p:.6f}"


def rr(entry: float, sl: float, tp: float):
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return None
    return reward / risk


# ---------- Image drawing helpers ----------

def crop_chart_area(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(w * 0.30) if w > 700 else 0
    y1 = int(h * 0.86) if h > 700 else h
    return img[:y1, x0:w].copy()

def price_to_y(price: float, p_min: float, p_max: float, h: int) -> int:
    # higher price -> smaller y
    if p_max <= p_min:
        return int(h * 0.5)
    t = (price - p_min) / (p_max - p_min)
    y = int((1 - t) * (h - 1))
    return max(0, min(h - 1, y))

def draw_level(img: np.ndarray, y: int, text: str, color: tuple[int,int,int], thickness: int = 3):
    h, w = img.shape[:2]
    cv2.line(img, (0, y), (w, y), color, thickness)
    cv2.putText(img, text, (10, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def draw_marker(img: np.ndarray, y: int, label: str, color: tuple[int,int,int]):
    h, w = img.shape[:2]
    x = int(w * 0.72)
    cv2.rectangle(img, (x, y - 18), (w - 10, y + 8), (255,255,255), -1)
    cv2.putText(img, label, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def build_analysis(symbol: str, tf: str, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
    price = float(closes[-1])
    e20 = float(ema(closes, 20)[-1])
    e50 = float(ema(closes, 50)[-1])
    e200= float(ema(closes, 200)[-1])
    r = float(rsi(closes, 14)[-1])

    sh_idx, sl_idx = swings(highs, lows, k=3)
    sh_vals = [float(highs[i]) for i in sh_idx[-25:]]
    sl_vals = [float(lows[i]) for i in sl_idx[-25:]]

    support, resistance = nearest_levels(price, sh_vals, sl_vals)

    trend = "Флет/переход"
    if e20 > e50 > e200:
        trend = "Восходящий"
    elif e20 < e50 < e200:
        trend = "Нисходящий"

    # сценарии (простые, но полезные)
    bull_entry = resistance if resistance else price
    bull_sl = support if support else price * 0.98
    bull_tp = (bull_entry * 1.02) if resistance else (price * 1.02)
    bull_rr = rr(bull_entry, bull_sl, bull_tp)

    bear_entry = support if support else price
    bear_sl = resistance if resistance else price * 1.02
    bear_tp = (bear_entry * 0.98) if support else (price * 0.98)
    bear_rr = rr(bear_entry, bear_sl, bear_tp)

    text = (
        f"📊 <b>{symbol}</b> — <b>{TF_PRETTY.get(tf, tf)}</b>\n"
        f"💰 <b>Цена:</b> {fmt(price)}\n\n"
        f"🔎 <b>Тренд:</b> {trend}\n"
        f"📌 <b>EMA20/50/200:</b> {fmt(e20)} / {fmt(e50)} / {fmt(e200)}\n"
        f"📌 <b>RSI(14):</b> {r:.1f}\n\n"
        f"🧱 <b>Поддержка:</b> {fmt(support)}\n"
        f"🧱 <b>Сопротивление:</b> {fmt(resistance)}\n\n"
        f"🧠 <b>Сценарии</b>\n"
        f"✅ <b>Bullish:</b> entry {fmt(bull_entry)}, SL {fmt(bull_sl)}, TP {fmt(bull_tp)}, RR {('—' if bull_rr is None else f'~1:{bull_rr:.2f}')}\n"
        f"🟥 <b>Bearish:</b> entry {fmt(bear_entry)}, SL {fmt(bear_sl)}, TP {fmt(bear_tp)}, RR {('—' if bear_rr is None else f'~1:{bear_rr:.2f}')}\n\n"
        f"⚠️ <i>Не финсовет. Алгоритмический анализ.</i>"
    )

    return {
        "price": price,
        "support": support,
        "resistance": resistance,
        "trend": trend,
        "bull": (bull_entry, bull_sl, bull_tp),
        "bear": (bear_entry, bear_sl, bear_tp),
        "text": text
    }


def draw_on_screenshot(in_path: str, analysis: dict, p_min: float, p_max: float) -> str:
    img_full = cv2.imread(in_path)
    if img_full is None:
        raise RuntimeError("cv2.imread: не смог прочитать изображение")

    img = crop_chart_area(img_full)

    h, w = img.shape[:2]
    price = analysis["price"]
    support = analysis["support"]
    resistance = analysis["resistance"]
    bull_entry, bull_sl, bull_tp = analysis["bull"]

    # уровни (как зоны/толстые линии)
    if support:
        y = price_to_y(support, p_min, p_max, h)
        draw_level(img, y, f"SUP {fmt(support)}", (0, 200, 0), thickness=4)

    if resistance:
        y = price_to_y(resistance, p_min, p_max, h)
        draw_level(img, y, f"RES {fmt(resistance)}", (0, 0, 220), thickness=4)

    # метки entry/sl/tp (лонг-сценарий)
    y_entry = price_to_y(bull_entry, p_min, p_max, h)
    y_sl    = price_to_y(bull_sl, p_min, p_max, h)
    y_tp    = price_to_y(bull_tp, p_min, p_max, h)

    draw_marker(img, y_entry, f"ENTRY {fmt(bull_entry)}", (0, 0, 0))
    draw_marker(img, y_sl,    f"SL {fmt(bull_sl)}", (0, 0, 220))
    draw_marker(img, y_tp,    f"TP {fmt(bull_tp)}", (0, 140, 0))

    # стрелка направления
    if analysis["trend"].startswith("Восход"):
        cv2.arrowedLine(img, (int(w*0.92), int(h*0.75)), (int(w*0.92), int(h*0.55)), (0,140,0), 4, tipLength=0.25)
    elif analysis["trend"].startswith("Нисход"):
        cv2.arrowedLine(img, (int(w*0.92), int(h*0.55)), (int(w*0.92), int(h*0.75)), (0,0,220), 4, tipLength=0.25)
    else:
        cv2.arrowedLine(img, (int(w*0.92), int(h*0.65)), (int(w*0.92), int(h*0.60)), (0,0,0), 4, tipLength=0.25)

    out_path = in_path.replace(".jpg", "_full.jpg")
    cv2.imwrite(out_path, img)
    return out_path


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "Отправь скрин графика и подпись, например:\n"
        "<code>BTCUSDT 1H</code>\n\n"
        "Я верну:\n"
        "• размеченный скрин (уровни/entry/sl/tp)\n"
        "• полный текстовый анализ"
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("⏳ Обрабатываю скрин и данные...")

    symbol, tf = parse_caption(message.caption)
    if not symbol or not tf:
        await message.answer("Нужна подпись вида <code>BTCUSDT 1H</code> (или ETHUSDT 15m).")
        return

    try:
        # download image
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        os.makedirs("tmp", exist_ok=True)
        in_path = f"tmp/{photo.file_id}.jpg"
        await bot.download_file(file.file_path, destination=in_path)

        # market data
        opens, highs, lows, closes = fetch_klines(symbol, tf, limit=300)

        analysis = build_analysis(symbol, tf, highs, lows, closes)

        # price range for mapping to Y
        p_min = float(np.min(lows[-200:]))
        p_max = float(np.max(highs[-200:]))

        out_path = draw_on_screenshot(in_path, analysis, p_min, p_max)

        await message.answer_photo(photo=FSInputFile(out_path), caption="🧠 Разметка готова")
        await message.answer(analysis["text"])

    except Exception as e:
        await message.answer(f"❌ Ошибка: <code>{type(e).__name__}: {str(e)[:250]}</code>")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
