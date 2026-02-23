# main.py — MAX PRO+++ (stable, no "hangs", Binance SPOT+Futures auto, ticker hints)
# Replace the whole file with this and commit.

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

SUPPORTED_INTERVALS = {
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","1w"
}
TF_PRETTY = {k: k.upper() for k in SUPPORTED_INTERVALS}

MIN_RR_OK = 1.2
MIN_RR_STRONG = 2.0

# Use stable primary domains only (mirrors sometimes return 202/edge issues)
SPOT_BASES = ["https://api.binance.com", "https://data-api.binance.vision"]
FUT_USDT_BASES = ["https://fapi.binance.com"]   # USDT-M futures
FUT_COIN_BASES = ["https://dapi.binance.com"]   # COIN-M futures

REQ_TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# Caches (in-memory)
_EXINFO_CACHE = {"spot": None, "usdtm": None, "coinm": None}
_EXINFO_TS = {"spot": 0.0, "usdtm": 0.0, "coinm": 0.0}
_EXINFO_TTL_SEC = 30 * 60  # 30 min


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
    ap = abs(float(p))
    if ap >= 1000:
        return f"{p:,.0f}".replace(",", " ")
    if ap >= 1:
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
    Expected:
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


def _parse_klines(data):
    o = np.array([float(x[1]) for x in data], dtype=np.float64)
    h = np.array([float(x[2]) for x in data], dtype=np.float64)
    l = np.array([float(x[3]) for x in data], dtype=np.float64)
    c = np.array([float(x[4]) for x in data], dtype=np.float64)
    return o, h, l, c


def _safe_get(url: str, params: dict):
    # 2 attempts total, treat 202 as retry
    last = None
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=REQ_TIMEOUT)
            if r.status_code == 202:
                last = r
                continue
            return r
        except Exception as e:
            last = e
            continue
    raise RuntimeError(f"Request failed: {url} ({last})")


# =========================
# EXCHANGEINFO (hints)
# =========================
def _now_ts() -> float:
    # avoid importing time in constrained envs
    import time
    return time.time()


def _get_exchangeinfo(kind: str):
    """
    kind: spot | usdtm | coinm
    caches for 30 minutes
    """
    ts = _now_ts()
    if _EXINFO_CACHE.get(kind) is not None and (ts - _EXINFO_TS.get(kind, 0.0)) < _EXINFO_TTL_SEC:
        return _EXINFO_CACHE[kind]

    if kind == "spot":
        url = f"{SPOT_BASES[0]}/api/v3/exchangeInfo"
    elif kind == "usdtm":
        url = f"{FUT_USDT_BASES[0]}/fapi/v1/exchangeInfo"
    elif kind == "coinm":
        url = f"{FUT_COIN_BASES[0]}/dapi/v1/exchangeInfo"
    else:
        return None

    try:
        r = _safe_get(url, params={})
        if r.status_code != 200:
            return None
        js = r.json()
        _EXINFO_CACHE[kind] = js
        _EXINFO_TS[kind] = ts
        return js
    except Exception:
        return None


def suggest_symbols(user_symbol: str, limit: int = 10) -> List[str]:
    """
    Suggest real Binance symbols close to base.
    """
    base = user_symbol
    if user_symbol.endswith("USDT"):
        base = user_symbol[:-4]
    elif user_symbol.endswith("USDC"):
        base = user_symbol[:-4]

    suggestions: List[str] = []

    for kind in ("spot", "usdtm", "coinm"):
        js = _get_exchangeinfo(kind)
        if not js:
            continue
        for s in js.get("symbols", []):
            sym = s.get("symbol", "")
            if not sym:
                continue
            if base and sym.startswith(base):
                # show mostly relevant quotes / perp
                if ("USDT" in sym) or ("USDC" in sym) or ("PERP" in sym) or (sym.endswith("USD")):
                    suggestions.append(sym)

    # unique
    uniq = []
    seen = set()
    for x in suggestions:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
        if len(uniq) >= limit:
            break
    return uniq


# =========================
# BINANCE AUTO FETCH (spot/usdt-m/coin-m)
# =========================
def fetch_klines_auto(symbol: str, interval: str, limit: int = 400):
    """
    Returns: (open, high, low, close, market_name, used_symbol)
    """
    # build candidate tickers
    base = symbol
    quote = ""
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        quote = "USDT"
    elif symbol.endswith("USDC"):
        base = symbol[:-4]
        quote = "USDC"

    candidates = [symbol]
    if quote == "USDT":
        candidates += [f"1000{base}USDT", f"{base}USDC", f"1000{base}USDC"]
    elif quote == "USDC":
        candidates += [f"1000{base}USDC", f"{base}USDT", f"1000{base}USDT"]

    # de-dup
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    def try_bases(bases: List[str], path: str, sym: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        for b in bases:
            url = f"{b}{path}"
            params = {"symbol": sym, "interval": interval, "limit": limit}
            try:
                r = _safe_get(url, params=params)
                if r.status_code == 200:
                    return _parse_klines(r.json())
                # 4xx: symbol/interval issues, try next symbol
                # 5xx: server issue, try next base
                continue
            except Exception:
                continue
        return None

    # 1) SPOT
    for sym in candidates:
        res = try_bases(SPOT_BASES, "/api/v3/klines", sym)
        if res is not None:
            o, h, l, c = res
            return o, h, l, c, "SPOT", sym

    # 2) USDT-M futures
    for sym in candidates:
        res = try_bases(FUT_USDT_BASES, "/fapi/v1/klines", sym)
        if res is not None:
            o, h, l, c = res
            return o, h, l, c, "FUTURES-USDTM", sym

    # 3) COIN-M futures
    coin_candidates = list(candidates)
    # common COIN-M perpetual format
    if quote in ("USDT", "USDC"):
        coin_candidates += [f"{base}USD_PERP", f"1000{base}USD_PERP"]

    seen2 = set()
    coin_candidates = [x for x in coin_candidates if not (x in seen2 or seen2.add(x))]

    for sym in coin_candidates:
        res = try_bases(FUT_COIN_BASES, "/dapi/v1/klines", sym)
        if res is not None:
            o, h, l, c = res
            return o, h, l, c, "FUTURES-COINM", sym

    # Not found -> suggestions
    sugg = suggest_symbols(symbol, limit=10)
    if sugg:
        raise RuntimeError(
            f"Символ <b>{symbol}</b> не найден.\n"
            f"Похожие тикеры на Binance: <code>{', '.join(sugg)}</code>\n"
            f"Попробуй, например: <code>{sugg[0]} {TF_PRETTY.get(interval, interval)}</code>"
        )
    raise RuntimeError(f"Символ <b>{symbol}</b> не найден на Binance (spot/usdt-m/coin-m).")


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
# STRUCTURE + LEVELS
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


def pick_levels(price: float, highs: np.ndarray, lows: np.ndarray):
    sh, sl = swings(highs, lows, k=3)
    sh_vals = sorted([float(highs[i]) for i in sh[-80:]])
    sl_vals = sorted([float(lows[i]) for i in sl[-80:]])

    supports = [x for x in sl_vals if x < price]
    resists = [x for x in sh_vals if x > price]

    s1 = supports[-1] if len(supports) >= 1 else None
    s2 = supports[-2] if len(supports) >= 2 else None
    r1 = resists[0] if len(resists) >= 1 else None
    r2 = resists[1] if len(resists) >= 2 else None
    return s1, s2, r1, r2


# =========================
# IMAGE DRAW
# =========================
def crop_chart_area(img: np.ndarray) -> np.ndarray:
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


def overlay_rect(img: np.ndarray, y0: int, y1: int, color_bgr, alpha: float = 0.18):
    h, w = img.shape[:2]
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if y1 < y0:
        y0, y1 = y1, y0
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, y1), color_bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hline(img: np.ndarray, y: int, color_bgr, thickness: int = 2):
    h, w = img.shape[:2]
    y = max(0, min(h - 1, y))
    cv2.line(img, (0, y), (w, y), color_bgr, thickness)


def draw_label_left(img: np.ndarray, y: int, text: str, color_bgr):
    h, w = img.shape[:2]
    y = max(18, min(h - 8, y))
    x0 = 10
    x1 = int(w * 0.46)
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

    s1, r1 = plan["S1"], plan["R1"]

    if s1 is not None:
        y = price_to_y(s1, p_min, p_max, h)
        overlay_rect(img,
                     price_to_y(s1 - zone_pad, p_min, p_max, h),
                     price_to_y(s1 + zone_pad, p_min, p_max, h),
                     (0, 200, 0),
                     alpha=0.16)
        draw_hline(img, y, (0, 170, 0), 2)
        draw_label_left(img, y, f"S1 {fmt(s1)}", (0, 140, 0))

    if r1 is not None:
        y = price_to_y(r1, p_min, p_max, h)
        overlay_rect(img,
                     price_to_y(r1 - zone_pad, p_min, p_max, h),
                     price_to_y(r1 + zone_pad, p_min, p_max, h),
                     (0, 0, 220),
                     alpha=0.14)
        draw_hline(img, y, (0, 0, 220), 2)
        draw_label_left(img, y, f"R1 {fmt(r1)}", (0, 0, 220))

    def draw_key(key: str, color):
        val = plan.get(key)
        if val is None:
            return
        y = price_to_y(float(val), p_min, p_max, h)
        draw_hline(img, y, color, 1)
        draw_label_left(img, y, f"{key.upper()} {fmt(float(val))}", color)

    draw_key("entry", (40, 40, 40))
    draw_key("sl", (0, 0, 220))
    draw_key("tp1", (0, 140, 0))
    draw_key("tp2", (0, 140, 0))
    draw_key("tp3", (0, 140, 0))

    draw_direction_arrow(img, plan["side"])

    out_path = in_path.replace(".jpg", "_maxpro_stable.jpg")
    cv2.imwrite(out_path, img)
    return out_path


# =========================
# PLAN LOGIC (ONE SCENARIO A)
# =========================
def classify_trend(e20: float, e50: float, e200: float, r: float) -> Tuple[str, str]:
    if any(math.isnan(x) for x in [e20, e50, e200, r]):
        return "Недостаточно данных", "NEUTRAL"
    if e20 > e50 > e200 and r >= 50:
        return "Восходящий", "LONG"
    if e20 < e50 < e200 and r <= 50:
        return "Нисходящий", "SHORT"
    return "Флет/переход", "NEUTRAL"


def build_plan(symbol_raw: str, used_symbol: str, tf: str, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, market: str) -> dict:
    price = float(closes[-1])

    e20 = float(ema(closes, 20)[-1])
    e50 = float(ema(closes, 50)[-1])
    e200 = float(ema(closes, 200)[-1])
    r = float(rsi(closes, 14)[-1])
    a = float(atr(highs, lows, closes, 14)[-1])

    trend_text, side = classify_trend(e20, e50, e200, r)
    structure = structure_type(highs, lows)

    S1, S2, R1, R2 = pick_levels(price, highs, lows)

    zone_pad = (a * 0.35) if not math.isnan(a) else price * 0.003
    stop_pad = (a * 0.90) if not math.isnan(a) else price * 0.008

    entry = sl = tp1 = tp2 = tp3 = None

    if side == "LONG":
        if S1 is not None:
            entry = S1
            sl = (S2 - stop_pad) if S2 is not None else (S1 - stop_pad)
        elif R1 is not None:
            entry = R1
            sl = (price - stop_pad)
        else:
            side = "NEUTRAL"

        if entry is not None and sl is not None:
            risk = max(1e-9, entry - sl)
            tp1 = entry + risk * 1
            tp2 = entry + risk * 2
            tp3 = entry + risk * 3

    elif side == "SHORT":
        if R1 is not None:
            entry = R1
            sl = (R2 + stop_pad) if R2 is not None else (R1 + stop_pad)
        elif S1 is not None:
            entry = S1
            sl = (price + stop_pad)
        else:
            side = "NEUTRAL"

        if entry is not None and sl is not None:
            risk = max(1e-9, sl - entry)
            tp1 = entry - risk * 1
            tp2 = entry - risk * 2
            tp3 = entry - risk * 3

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

    if side == "NEUTRAL":
        strength = "⚠️ Сетап не сформирован (флет/нет уровней)"

    report = (
        f"📊 <b>{symbol_raw}</b> — <b>{TF_PRETTY.get(tf, tf)}</b>\n"
        f"🔗 Использую: <code>{used_symbol}</code> <i>({market})</i>\n"
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
        "symbol_raw": symbol_raw,
        "used_symbol": used_symbol,
        "tf": tf,
        "market": market,
        "price": price,
        "trend": trend_text,
        "side": side,
        "structure": structure,
        "rsi": r,
        "atr": a,
        "S1": S1, "S2": S2, "R1": R1, "R2": R2,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
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
        "MAX PRO+++ (stable):\n"
        "• авто SPOT/USDT-M/COIN-M\n"
        "• подсказки тикеров если не найдено\n"
        "• один сценарий по тренду\n"
        "• зоны + TP1/TP2/TP3"
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("⏳ Анализирую (stable)...")

    symbol, tf = parse_caption(message.caption)
    if not symbol or not tf:
        await message.answer("Нужна подпись вида <code>BTCUSDT 1H</code>.")
        return

    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)

        os.makedirs("tmp", exist_ok=True)
        in_path = f"tmp/{photo.file_id}.jpg"
        await bot.download_file(file.file_path, destination=in_path)

        o, h, l, c, market, used_symbol = fetch_klines_auto(symbol, tf, limit=400)
        plan = build_plan(symbol, used_symbol, tf, h, l, c, market)

        out_path = draw_plan_on_screenshot(in_path, plan, h, l)

        await message.answer_photo(photo=FSInputFile(out_path), caption="🧠 MAX PRO+++ разметка готова")
        await message.answer(plan["report"])

    except Exception as e:
        await message.answer(f"❌ Ошибка: <code>{type(e).__name__}: {str(e)[:900]}</code>")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
