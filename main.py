import os
import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not found")

bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()


def analyze_image(path: str):
    img = cv2.imread(path)
    if img is None:
        return {"error": "Не удалось обработать изображение"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=120,
                            minLineLength=80,
                            maxLineGap=10)

    slopes = []
    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0]:
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)

    trend = "Флет / неопределённо"
    if slopes:
        m = float(np.median(slopes))
        if m < -0.15:
            trend = "Восходящий тренд"
        elif m > 0.15:
            trend = "Нисходящий тренд"

    return {
        "trend": trend
    }


def format_answer(result: dict):
    if "error" in result:
        return f"❌ {result['error']}"

    return (
        f"📊 <b>Анализ графика</b>\n\n"
        f"🔎 Тренд: <b>{result['trend']}</b>\n\n"
        f"🧠 Сценарии:\n"
        f"• При подтверждении импульса — работа по тренду\n"
        f"• При сломе структуры — возможен разворот\n\n"
        f"⚠️ Это автоматический анализ по изображению."
    )


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer(
        "Привет 👋\n\n"
        "Отправь скриншот графика монеты.\n"
        "Я сделаю автоматический анализ."
    )


@dp.message(F.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    os.makedirs("tmp", exist_ok=True)
    path = f"tmp/{photo.file_id}.jpg"

    await bot.download_file(file.file_path, path)

    result = analyze_image(path)

    await message.answer(format_answer(result))


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
