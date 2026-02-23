import os
import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode

BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()


def analyze_and_draw(path: str) -> str:
    img = cv2.imread(path)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # -------- Горизонтальные уровни --------
    projection = edges.sum(axis=1)
    idx = np.argsort(projection)[-3:]
    levels = sorted(idx)

    for y in levels:
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 2)

    # -------- Поиск тренд линии --------
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150,
                            minLineLength=100, maxLineGap=20)

    if lines is not None:
        longest = max(lines, key=lambda l: np.linalg.norm(
            (l[0][2]-l[0][0], l[0][3]-l[0][1])
        ))
        x1, y1, x2, y2 = longest[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # -------- Стрелка входа --------
    cv2.arrowedLine(img,
                    (int(w*0.8), int(h*0.6)),
                    (int(w*0.8), int(h*0.5)),
                    (0, 0, 255),
                    3)

    output_path = path.replace(".jpg", "_analyzed.jpg")
    cv2.imwrite(output_path, img)

    return output_path


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer("Отправь скрин графика. Я разрисую его 😈")


@dp.message(F.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    os.makedirs("tmp", exist_ok=True)
    path = f"tmp/{photo.file_id}.jpg"

    await bot.download_file(file.file_path, destination=path)

    analyzed_path = analyze_and_draw(path)

    await message.answer_photo(
        photo=open(analyzed_path, "rb"),
        caption="🧠 AI разметил уровни и тренд"
    )


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
