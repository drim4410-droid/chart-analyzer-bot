import os
import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from aiogram.enums import ParseMode

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not found")

bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

def crop_chart_area(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    # Если это узкий скрин — не режем агрессивно
    x0 = int(w * 0.30) if w > 600 else 0
    y1 = int(h * 0.86) if h > 600 else h

    cropped = img[:y1, x0:w].copy()
    return cropped

def detect_levels(edges: np.ndarray, top_k: int = 3) -> list[int]:
    projection = edges.sum(axis=1)
    if projection.size == 0:
        return []
    idx = np.argsort(projection)[-top_k:]
    return sorted([int(y) for y in idx])

def detect_trend_line(edges: np.ndarray):
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=120,
        minLineLength=120,
        maxLineGap=20
    )
    if lines is None:
        return None

    # берём самую длинную линию
    longest = max(
        lines,
        key=lambda l: float(np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1]))
    )
    return longest[0]

def analyze_and_draw(path: str) -> tuple[str, str]:
    img_full = cv2.imread(path)
    if img_full is None:
        raise RuntimeError("cv2.imread не смог прочитать изображение")

    img = crop_chart_area(img_full)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    # уровни
    levels = detect_levels(edges, top_k=3)
    for y in levels:
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 2)

    # тренд-линия
    trend = "Не определён"
    tl = detect_trend_line(edges)
    if tl is not None:
        x1, y1, x2, y2 = tl
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        trend = "Нисходящий" if slope > 0.10 else ("Восходящий" if slope < -0.10 else "Флет")

    # стрелка “внимание”
    cv2.arrowedLine(
        img,
        (int(w * 0.85), int(h * 0.65)),
        (int(w * 0.85), int(h * 0.55)),
        (0, 0, 255),
        3,
        tipLength=0.25
    )

    out_path = path.replace(".jpg", "_ai.jpg")
    cv2.imwrite(out_path, img)
    return out_path, trend

@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer("Отправь скрин графика. Я разметлю уровни и тренд 😈")

@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("⏳ Обрабатываю скрин...")

    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)

        os.makedirs("tmp", exist_ok=True)
        path = f"tmp/{photo.file_id}.jpg"

        await bot.download_file(file.file_path, destination=path)

        analyzed_path, trend = analyze_and_draw(path)

        await message.answer_photo(
            photo=FSInputFile(analyzed_path),
            caption=f"🧠 AI-разметка готова\nТренд: <b>{trend}</b>"
        )

    except Exception as e:
        await message.answer(f"❌ Ошибка обработки: <code>{type(e).__name__}: {str(e)[:200]}</code>")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
