import asyncio
import io

from aiogram import F, Router, types

from ..config import MODE_SERIOUS, VISION_MODELS
from ..runtime import logger
from ..state import state
from ..services.formatter import format_for_telegram
from ..services.vision import analyze_image

router = Router()


@router.message(F.photo)
async def handle_photo(message: types.Message) -> None:
    chat_id = message.chat.id
    mode = state.user_modes.get(chat_id, "worker")

    if mode != MODE_SERIOUS:
        await message.answer("📷 Отправка картинок доступна только в серьезном режиме.")
        return

    selected_model = state.user_models.get(chat_id, "gemini-2.5-flash")
    if selected_model not in VISION_MODELS:
        await message.answer(
            "📷 Анализ изображения доступен для AUTO (он использует `gemini-2.5-flash`), "
            "`gemini-2.5-flash`, `gpt-4o` и `@cf/moonshotai/kimi-k2.6`.",
            parse_mode="Markdown",
        )
        return

    await message.bot.send_chat_action(chat_id, "typing")

    photo = message.photo[-1]
    file_info = await message.bot.get_file(photo.file_id)
    buffer = io.BytesIO()
    await message.bot.download_file(file_info.file_path, destination=buffer)
    image_bytes = buffer.getvalue()
    caption = message.caption or "Что на этом изображении? Опиши подробно."

    try:
        answer = await analyze_image(selected_model, caption, image_bytes)
        if not answer:
            await message.answer("🤔 Модель не смогла дать ответ на это изображение.")
            return

        raw_text = f"🤖 **[{selected_model}]** 📷 [Vision]\n\n{answer}"
        formatted_text = format_for_telegram(raw_text)
        try:
            await message.answer(formatted_text, parse_mode="MarkdownV2")
        except Exception:
            await message.answer(answer)
    except asyncio.TimeoutError:
        await message.answer("⏱ Обработка изображения заняла слишком много времени. Попробуй еще раз.")
    except Exception as error:
        logger.warning("Vision error on %s: %s", selected_model, str(error)[:150])
        await message.answer(f"❌ Ошибка при анализе изображения: {str(error)[:120]}")
