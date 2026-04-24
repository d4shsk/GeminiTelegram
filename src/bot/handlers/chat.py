import json

from aiogram import F, Router, types

from ..config import MENU_CHANGE_MODE, MENU_CHANGE_MODEL, MENU_CLEAR_HISTORY, MODE_SERIOUS
from ..keyboards import build_main_menu, build_mode_picker, build_model_picker, model_picker_text
from ..state import state
from ..services.formatter import format_for_telegram, split_message
from ..services.llm_text import generate_text_reply

router = Router()


@router.message(F.text)
async def handle_message(message: types.Message) -> None:
    chat_id = message.chat.id
    user_input = message.text
    mode = state.user_modes.get(chat_id, MODE_SERIOUS)

    if user_input == MENU_CHANGE_MODE:
        await message.answer(
            "Выберите режим работы бота:\n"
            "1. Шуточный режим (RolePlay)\n"
            "2. Серьезный режим",
            reply_markup=build_mode_picker(),
        )
        return

    if user_input == MENU_CHANGE_MODEL:
        if mode != MODE_SERIOUS:
            await message.answer("Выбор модели доступен только в серьезном режиме.")
        else:
            await message.answer(model_picker_text(), reply_markup=build_model_picker(mode), parse_mode="HTML")
        return

    if user_input == MENU_CLEAR_HISTORY:
        state.clear_history(chat_id)
        await message.answer("🧹 История очищена.", reply_markup=build_main_menu())
        return

    await message.bot.send_chat_action(chat_id, "typing")
    result = await generate_text_reply(chat_id, user_input, mode)
    if not result:
        await message.answer("🤯 Сейчас перегрузка по всем моделям. Попробуй позже.")
        return

    final_text = result["text"]
    used_model = result["used_model"]
    response_text_to_save = result["save_text"]
    requested_name = result["requested_name"]
    response_meta_raw = result.get("response_mode", "{}")
    try:
        response_meta = json.loads(response_meta_raw)
    except Exception:
        response_meta = {}

    answer_type = response_meta.get("answer_type", "быстрый ответ")
    if mode == MODE_SERIOUS:
        header = f"🤖 **[{used_model}]** 🧐 [{answer_type}]"
    else:
        header = f"🤖 **[{used_model}]** 🤪 [RolePlay]"

    raw_full_text = f"{header}\n\n{final_text}"
    if requested_name and requested_name.lower() not in used_model.lower():
        fallback_name = used_model.split("-")[0].capitalize()
        raw_full_text += (
            f"\n\n*(P.S. Вы звали {requested_name}, но она сейчас недоступна из-за API-ошибки, "
            f"поэтому ответила {fallback_name})*"
        )

    formatted_text = format_for_telegram(raw_full_text)
    state.update_history(chat_id, "user", user_input)
    state.update_history(chat_id, "model", response_text_to_save)

    try:
        for part in split_message(formatted_text):
            await message.answer(part, parse_mode="MarkdownV2")
    except Exception:
        for part in split_message(final_text):
            await message.answer(part)
