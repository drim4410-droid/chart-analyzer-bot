import os
import re
import math
import cv2
import numpy as np
import requests
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from aiogram.enums import ParseMode

# ================== CONFIG ==================
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

MIN_RR_STRONG = 2.0
MIN_RR_WEAK = 1.2

# ================== BOT ==================
bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

# ================== UTILS ==================

def parse_caption(caption: str | None):
    if not caption:
        return None, None
    t = re.sub(r"\s+", " ", caption.strip())
    parts = t.split(" ")
    if len(parts) < 2:
        return None, None
    symbol = parts[0].upper().replace("/", "").replace("-", "").replace("_", "")
    tf = parts[1].lower().replace("ч","h").replace("м","m").replace("д","d")
    if tf not in SUPPORTED_INTERVALS:
        return symbol, None
    return symbol, tf

def fetch_klines(symbol: str, interval: str, limit: int = 400):
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
            o = np.array([float(x[1]) for x in data], dtype=np.float64)
            h = np.array([float(x[2]) for x in data], dtype=np.float64)
            l = np.array([float(x[3]) for x in data], dtype=np.float64)
            c = np.array([float(x[4]) for x in data], dtype=np.float64)
            return o, h, l, c
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"Binance API недоступен: {last_err}")

def ema(arr: np.ndarray, period: int):
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    alpha = 2/(period+1)
    out = np.empty_like(arr)
    out[:] = np.nan
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha*arr[i] + (1-alpha)*out[i-1]
    return out

def rsi(arr: np.ndarray, period: int = 14):
    if len(arr) < period+1:
        return np.full_like(arr, np.nan)
    diff = np.diff(arr)
    gain = np.where(diff>0, diff, 0.0)
    loss = np.where(diff<0, -diff, 0.0)
    avg_gain = np.empty(len(arr)); avg_loss = np.empty(len(arr))
    avg_gain[:] = np.nan; avg_loss[:] = np.nan
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period+1, len(arr)):
        avg_gain[i] = (avg_gain[i-1]*(period-1)+gain[i-1])/period
        avg_loss[i] = (avg_loss[i-1]*(period-1)+loss[i-1])/period
    rs = avg_gain/(avg_loss+1e-12)
    out = 100 - (100/(1+rs))
    out[:period] = np.nan
    return out

def atr(highs, lows, closes, period=14):
    if len(closes) < period+1:
        return np.full_like(closes, np.nan)
    prev_close = closes[:-1]
    tr = np.maximum(highs[1:]-lows[1:], np.maximum(
        np.abs(highs[1:]-prev_close),
        np.abs(lows[1:]-prev_close)
    ))
    out = np.empty_like(closes); out[:] = np.nan
    out[period] = np.mean(tr[:period])
    for i in range(period+1, len(closes)):
        out[i] = (out[i-1]*(period-1)+tr[i-1])/period
    return out

def swings(highs, lows, k=3):
    sh, sl = [], []
    n = len(highs)
    for i in range(k, n-k):
        if np.all(highs[i]>highs[i-k:i]) and np.all(highs[i]>highs[i+1:i+1+k]):
            sh.append(i)
        if np.all(lows[i]<lows[i-k:i]) and np.all(lows[i]<lows[i+1:i+1+k]):
            sl.append(i)
    return sh, sl

def structure_type(highs, lows):
    sh, sl = swings(highs, lows, k=3)
    if len(sh)<2 or len(sl)<2:
        return "Неопределённо"
    if highs[sh[-1]] < highs[sh[-2]] and lows[sl[-1]] < lows[sl[-2]]:
        return "LL/LH (нисходящая структура)"
    if highs[sh[-1]] > highs[sh[-2]] and lows[sl[-1]] > lows[sl[-2]]:
        return "HH/HL (восходящая структура)"
    return "Переходная"

def nearest_levels(price, highs, lows):
    sh, sl = swings(highs, lows, k=3)
    sh_vals = [float(highs[i]) for i in sh[-30:]]
    sl_vals = [float(lows[i]) for i in sl[-30:]]
    support = max([x for x in sl_vals if x<price], default=None)
    resistance = min([x for x in sh_vals if x>price], default=None)
    return support, resistance

def fmt(p):
    if p is None or (isinstance(p,float) and math.isnan(p)):
        return "—"
    if p>=1000:
        return f"{p:,.0f}".replace(",", " ")
    if p>=1:
        return f"{p:,.2f}".replace(",", " ")
    return f"{p:.6f}"

def rr(entry, sl, tp):
    risk = abs(entry-sl)
    reward = abs(tp-entry)
    if risk<=0: return None
    return reward/risk

# ================== DRAW ==================

def crop_chart_area(img):
    h,w = img.shape[:2]
    x0 = int(w*0.30) if w>700 else 0
    y1 = int(h*0.86) if h>700 else h
    return img[:y1, x0:w].copy()

def price_to_y(price, p_min, p_max, h):
    if p_max<=p_min:
        return int(h*0.5)
    t = (price-p_min)/(p_max-p_min)
    y = int((1-t)*(h-1))
    return max(0,min(h-1,y))

def overlay_rect(img, y0, y1, color, alpha=0.20):
    h,w = img.shape[:2]
    y0=max(0,min(h-1,y0)); y1=max(0,min(h-1,y1))
    if y1<y0: y0,y1=y1,y0
    overlay = img.copy()
    cv2.rectangle(overlay,(0,y0),(w,y1),color,-1)
    cv2.addWeighted(overlay,alpha,img,1-alpha,0,img)

def draw_line(img,y,color,th=2):
    h,w=img.shape[:2]
    y=max(0,min(h-1,y))
    cv2.line(img,(0,y),(w,y),color,th)

def draw_label(img,y,text,color):
    h,w=img.shape[:2]
    y=max(0,min(h-1,y))
    x=int(w*0.60)
    cv2.rectangle(img,(x,y-18),(w-10,y+8),(255,255,255),-1)
    cv2.putText(img,text,(x+6,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)

# ================== CORE ==================

def build_plan(symbol, tf, highs, lows, closes):
    price = float(closes[-1])
    e20 = float(ema(closes,20)[-1])
    e50 = float(ema(closes,50)[-1])
    e200= float(ema(closes,200)[-1])
    r = float(rsi(closes,14)[-1])
    a = float(atr(highs,lows,closes,14)[-1])

    structure = structure_type(highs,lows)
    support, resistance = nearest_levels(price, highs, lows)

    trend = "Флет"
    side = "NEUTRAL"

    if e20>e50>e200 and r>=50:
        trend="Восходящий"; side="LONG"
    elif e20<e50<e200 and r<=50:
        trend="Нисходящий"; side="SHORT"

    zone_pad = a*0.35 if not math.isnan(a) else price*0.003
    stop_pad = a*0.9 if not math.isnan(a) else price*0.008

    if side=="LONG" and resistance:
        entry = resistance
        sl = (support-stop_pad) if support else price-stop_pad
        risk = max(1e-9, entry-sl)
        tp1 = entry+risk*1
        tp2 = entry+risk*2
        tp3 = entry+risk*3
    elif side=="SHORT" and support:
        entry = support
        sl = (resistance+stop_pad) if resistance else price+stop_pad
        risk = max(1e-9, sl-entry)
        tp1 = entry-risk*1
        tp2 = entry-risk*2
        tp3 = entry-risk*3
    else:
        entry=sl=tp1=tp2=tp3=None

    strength="—"
    if entry and sl and tp2:
        r2=rr(entry,sl,tp2)
        if r2 and r2>=MIN_RR_STRONG:
            strength="🔥 High probability"
        elif r2 and r2>=MIN_RR_WEAK:
            strength="⚖️ Рабочий"
        else:
            strength="⚠️ Слабый"

    report=(
        f"📊 <b>{symbol}</b> — <b>{TF_PRETTY.get(tf,tf)}</b>\n"
        f"💰 Цена: {fmt(price)}\n\n"
        f"🔻 Тренд: {trend}\n"
        f"📐 Структура: {structure}\n"
        f"📉 RSI: {r:.1f}\n\n"
        f"🧱 Support: {fmt(support)}\n"
        f"🧱 Resistance: {fmt(resistance)}\n\n"
        f"🎯 Основной сценарий: {side}\n"
        f"Entry: {fmt(entry)}\n"
        f"SL: {fmt(sl)}\n"
        f"TP1: {fmt(tp1)}\n"
        f"TP2: {fmt(tp2)}\n"
        f"TP3: {fmt(tp3)}\n"
        f"{strength}\n\n"
        f"⚠️ Не финсовет."
    )

    return {
        "price":price,
        "support":support,
        "resistance":resistance,
        "entry":entry,
        "sl":sl,
        "tp1":tp1,
        "tp2":tp2,
        "tp3":tp3,
        "zone_pad":zone_pad,
        "side":side,
        "report":report
    }

def draw_premium(in_path, plan, highs, lows):
    img_full=cv2.imread(in_path)
    img=crop_chart_area(img_full)
    h,w=img.shape[:2]

    p_min=float(np.min(lows[-200:]))
    p_max=float(np.max(highs[-200:]))

    for level,color in [(plan["support"],(0,170,0)),(plan["resistance"],(0,0,220))]:
        if level:
            y=price_to_y(level,p_min,p_max,h)
            overlay_rect(img,
                price_to_y(level-plan["zone_pad"],p_min,p_max,h),
                price_to_y(level+plan["zone_pad"],p_min,p_max,h),
                color,0.18)
            draw_line(img,y,color,2)
            draw_label(img,y,f"{fmt(level)}",color)

    for key,color in [("entry",(50,50,50)),("sl",(0,0,220)),("tp1",(0,140,0)),("tp2",(0,140,0)),("tp3",(0,140,0))]:
        val=plan.get(key)
        if val:
            y=price_to_y(val,p_min,p_max,h)
            draw_line(img,y,color,1)
            draw_label(img,y,f"{key.upper()} {fmt(val)}",color)

    out=in_path.replace(".jpg","_pro.jpg")
    cv2.imwrite(out,img)
    return out

# ================== HANDLERS ==================

@dp.message(F.text=="/start")
async def start(message:Message):
    await message.answer("Отправь скрин + подпись вида:\n<code>BTCUSDT 1H</code>")

@dp.message(F.photo)
async def handle_photo(message:Message):
    await message.answer("⏳ Анализирую рынок...")

    symbol,tf=parse_caption(message.caption)
    if not symbol or not tf:
        await message.answer("Нужна подпись <code>BTCUSDT 1H</code>")
        return
    if not symbol.endswith("USDT"):
        await message.answer("Поддерживаются только пары USDT.")
        return

    try:
        photo=message.photo[-1]
        file=await bot.get_file(photo.file_id)
        os.makedirs("tmp",exist_ok=True)
        in_path=f"tmp/{photo.file_id}.jpg"
        await bot.download_file(file.file_path,destination=in_path)

        o,h,l,c=fetch_klines(symbol,tf)
        plan=build_plan(symbol,tf,h,l,c)
        out_path=draw_premium(in_path,plan,h,l)

        await message.answer_photo(photo=FSInputFile(out_path),caption="🧠 PRO-разметка")
        await message.answer(plan["report"])

    except Exception as e:
        await message.answer(f"❌ Ошибка: <code>{type(e).__name__}: {str(e)[:250]}</code>")

async def main():
    await dp.start_polling(bot)

if __name__=="__main__":
    import asyncio
    asyncio.run(main())
