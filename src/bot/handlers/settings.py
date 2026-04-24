from aiogram import F, Router, types
from aiogram.filters import Command
from aiogram.types import CallbackQuery

from ..config import MODE_SERIOUS
from ..keyboards import build_main_menu, build_model_picker, model_picker_text
from ..state import state

router = Router()


@router.message(Command("model"))
async def cmd_model(message: types.Message) -> None:
    chat_id = message.chat.id
    if state.user_modes.get(chat_id) != MODE_SERIOUS:
        await message.answer("Команда /model доступна только в серьезном режиме. Выберите режим через /start.")
        return
    await message.answer(model_picker_text(), reply_markup=build_model_picker(), parse_mode="HTML")


@router.callback_query(F.data.startswith("setmodel_"))
async def handle_model_selection(callback: CallbackQuery) -> None:
    if not callback.message:
        await callback.answer()
        return

    model_id = callback.data.split("_", 1)[1]
    chat_id = callback.message.chat.id
    if model_id == "auto":
        state.user_models.pop(chat_id, None)
        selected_label = "AUTO"
    else:
        state.user_models[chat_id] = model_id
        selected_label = model_id
    state.clear_history(chat_id)
    await callback.message.edit_text(f"✅ Модель изменена на: {selected_label}\nИстория очищена для чистого переключения.")
    await callback.answer()


@router.message(Command("clear"))
async def cmd_clear(message: types.Message) -> None:
    state.clear_history(message.chat.id)
    await message.answer("🧹 История очищена.", reply_markup=build_main_menu())
