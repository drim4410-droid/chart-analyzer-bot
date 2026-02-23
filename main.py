import os
import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode

BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()


def crop_chart_area(img):
    h, w = img.shape[:2]

    # обрезаем Telegram фон (левая часть)
    img = img[:, int(w*0.35):w]

    # обрезаем нижнюю панель
    img = img[0:int(h*0.85), :]

    return img


def detect_levels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    projection = edges.sum(axis=1)
    strongest = np.argsort(projection)[-4:]
    return sorted(strongest)


def detect_trend_line(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=120,
                            minLineLength=100,
                            maxLineGap=20)

    if lines is None:
        return None

    longest = max(lines, key=lambda l: np.linalg.norm(
        (l[0][2]-l[0][0], l[0][3]-l[0][1])
    ))

    return longest[0]


def analyze_and_draw(path):
    img_full = cv2.imread(path)
    img = crop_chart_area(img_full.copy())

    h, w = img.shape[:2]

    levels = detect_levels(img)

    for y in levels:
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 2)

    trend_line = detect_trend_line(img)
    if trend_line is not None:
        x1, y1, x2, y2 = trend_line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        direction = "Нисходящий" if slope > 0 else "Восходящий"
    else:
        direction = "Не определён"

    # стрелка
    cv2.arrowedLine(img,
                    (int(w*0.8), int(h*0.6)),
                    (int(w*0.8), int(h*0.5)),
                    (0, 0, 255),
                    3)

    output_path = path.replace(".jpg", "_ai.jpg")
    cv2.imwrite(output_path, img)

    return output_path, direction


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer("Отправь скрин графика. AI разметит уровни 😈")


@dp.message(F.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    os.makedirs("tmp", exist_ok=True)
    path = f"tmp/{photo.file_id}.jpg"

    await bot.download_file(file.file_path, destination=path)

    analyzed_path, trend = analyze_and_draw(path)

    await message.answer_photo(
        photo=open(analyzed_path, "rb"),
        caption=f"🧠 AI-анализ\nТренд: {trend}"
    )


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
