from aiogram import F, Router, types
from aiogram.filters import Command
from aiogram.types import CallbackQuery

from ..config import MODE_WORKER
from ..keyboards import build_main_menu, build_mode_picker, build_model_picker, model_picker_text
from ..state import state

router = Router()


@router.message(Command("start"))
async def cmd_start(message: types.Message) -> None:
    await message.answer(
        "Выберите режим работы бота:\n"
        "1. Шуточный режим (RolePlay)\n"
        "2. Серьезный режим",
        reply_markup=build_mode_picker(),
    )


@router.callback_query(F.data.startswith("mode_"))
async def handle_mode_selection(callback: CallbackQuery) -> None:
    if not callback.message:
        await callback.answer()
        return

    mode = callback.data.split("_")[1]
    chat_id = callback.message.chat.id
    state.user_modes[chat_id] = mode
    state.clear_history(chat_id)

    if mode == MODE_WORKER:
        await callback.message.edit_text(
            "✅ Выбран режим RolePlay.\n"
            "История очищена. Бот хранит недавний диалог и краткое резюме более раннего контекста."
        )
        await callback.message.answer("Меню всегда под рукой 👇", reply_markup=build_main_menu())
    else:
        state.user_models.pop(chat_id, None)
        await callback.message.edit_text(
            "✅ Выбран серьезный режим. История очищена.\n"
            "По умолчанию включен AUTO: бот сам выбирает между быстрым, сильным и fallback-маршрутом."
            + model_picker_text().replace("Выберите модель:", "\n\nВыберите модель или оставьте AUTO:"),
            reply_markup=build_model_picker(mode),
            parse_mode="HTML",
        )
        await callback.message.answer("Меню всегда под рукой 👇", reply_markup=build_main_menu())

    await callback.answer()
