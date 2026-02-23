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
    """
    Пытаемся вырезать область с графиком.
    Под твои скрины (TradingView/мобилка) подходит хорошо:
    - убираем левую часть (фон Telegram)
    - убираем нижнюю панель (кнопки)
    """
    h, w = img.shape[:2]

    # Если скрин широкий — обычно слева фон, справа сам график
    x0 = int(w * 0.30) if w > 700 else 0
    # Снизу часто панель, обрежем
    y1 = int(h * 0.86) if h > 700 else h

    cropped = img[:y1, x0:w].copy()
    return cropped


def detect_levels(edges: np.ndarray, top_k: int = 3) -> list[int]:
    """
    Ищем горизонтальные "скопления" по сумме edge-пикселей по строкам.
    Берём top_k самых сильных строк как грубые уровни.
    """
    proj = edges.sum(axis=1)
    if proj.size == 0:
        return []
    idx = np.argsort(proj)[-top_k:]
    return sorted([int(y) for y in idx])


def detect_trend_line(edges: np.ndarray):
    """
    Поиск заметной линии через HoughLinesP.
    Берём самую длинную.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=120,
        maxLineGap=20,
    )
    if lines is None:
        return None

    longest = max(
        lines,
        key=lambda l: float(np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1])),
    )
    return longest[0]


def analyze_and_draw(in_path: str) -> tuple[str, str]:
    img_full = cv2.imread(in_path)
    if img_full is None:
        raise RuntimeError("cv2.imread: не смог прочитать изображение")

    img = crop_chart_area(img_full)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    # --- уровни ---
    levels = detect_levels(edges, top_k=3)
    for y in levels:
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 2)  # зелёные уровни

    # --- тренд линия ---
    trend = "Не определён"
    tl = detect_trend_line(edges)
    if tl is not None:
        x1, y1, x2, y2 = tl
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # синяя линия

        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        # В координатах изображения y растёт вниз:
        # slope > 0 -> линия "вниз" вправо (нисходящий)
        if slope > 0.12:
            trend = "Нисходящий"
        elif slope < -0.12:
            trend = "Восходящий"
        else:
            trend = "Флет"

    # --- стрелка (просто визуальный маркер) ---
    cv2.arrowedLine(
        img,
        (int(w * 0.85), int(h * 0.65)),
        (int(w * 0.85), int(h * 0.55)),
        (0, 0, 255),
        3,
        tipLength=0.25,
    )

    out_path = in_path.replace(".jpg", "_ai.jpg")
    cv2.imwrite(out_path, img)
    return out_path, trend


@dp.message(F.text == "/start")
async def start(message: Message):
    await message.answer("Отправь скрин графика. Я разрисую уровни и тренд 😈")


@dp.message(F.photo)
async def handle_photo(message: Message):
    # Сразу отвечаем, чтобы ты видел что бот живой
    await message.answer("⏳ Обрабатываю скрин...")

    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)

        os.makedirs("tmp", exist_ok=True)
        in_path = f"tmp/{photo.file_id}.jpg"

        await bot.download_file(file.file_path, destination=in_path)

        out_path, trend = analyze_and_draw(in_path)

        await message.answer_photo(
            photo=FSInputFile(out_path),
            caption=f"🧠 <b>AI-разметка</b>\nТренд: <b>{trend}</b>",
        )

    except Exception as e:
        await message.answer(
            f"❌ Ошибка обработки: <code>{type(e).__name__}: {str(e)[:250]}</code>"
        )


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
