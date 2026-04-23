from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from .config import (
    MENU_CHANGE_MODE,
    MENU_CHANGE_MODEL,
    MENU_CLEAR_HISTORY,
    MODEL_PRIORITY,
    MODEL_RATING_TEXT,
)


def build_main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=MENU_CHANGE_MODE), KeyboardButton(text=MENU_CHANGE_MODEL)],
            [KeyboardButton(text=MENU_CLEAR_HISTORY)],
        ],
        resize_keyboard=True,
    )


def build_mode_picker() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Шуточный режим (RolePlay)🤪", callback_data="mode_worker")],
            [InlineKeyboardButton(text="Серьезный режим 🧐", callback_data="mode_serious")],
        ]
    )


def build_model_buttons() -> list[list[InlineKeyboardButton]]:
    buttons: list[list[InlineKeyboardButton]] = []
    for model_info in MODEL_PRIORITY:
        label = model_info["model"].replace(":free", "")
        if model_info.get("serious_only"):
            label = "🔬 " + label
        buttons.append([InlineKeyboardButton(text=label, callback_data=f"setmodel_{model_info['model']}")])
    return buttons


def build_model_picker() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=build_model_buttons())


def model_picker_text() -> str:
    return "Выберите модель:" + MODEL_RATING_TEXT
