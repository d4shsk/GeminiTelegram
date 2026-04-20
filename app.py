import os
import asyncio
import random
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from google import genai
from aiogram.client.session.aiohttp import AiohttpSession

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- КОНФИГУРАЦИЯ ---
MODEL_PRIORITY = [
    "gemini-flash-latest", 
    "gemini-2.5-flash", 
    "gemini-2.0-flash", 
    "gemma-3-27b-it"
]
MAX_HISTORY = 30
GEMMA_PHRASES = [
    "Гемини ушёл распространять демократию, за пультом Гемма!",
    "Старший брат спит, отдуваюсь я. Погнали!",
    "Система перегружена, вызвали стажёра Гемму. Слушаю!"
]

# Инициализация
tg_token = os.environ.get("TELEGRAM_TOKEN")
gemini_key = os.environ.get("GEMINI_API_KEY")

bot = Bot(token=tg_token)
dp = Dispatcher()
client = genai.Client(api_key=gemini_key)

sessions = {}

def update_history(chat_id, role, text):
    if chat_id not in sessions:
        sessions[chat_id] = []
    sessions[chat_id].append({"role": role, "parts": [{"text": text}]})
    if len(sessions[chat_id]) > MAX_HISTORY:
        sessions[chat_id] = sessions[chat_id][-MAX_HISTORY:]

# --- ОБРАБОТЧИКИ ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("МегаМоZг на связи! Я могу упасть, так как дондолет захостил меня на железной дороге. Помню 30 сообщений. Сброс: /clear")

@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    sessions.pop(message.chat.id, None)
    await message.answer("🧹 Память очищена.")

@dp.message(F.text)
async def handle_message(message: types.Message):
    chat_id = message.chat.id
    user_input = message.text
    await bot.send_chat_action(chat_id, "typing")

    final_text = None
    used_model = ""

    for model_id in MODEL_PRIORITY:
        try:
            chat = client.chats.create(model=model_id, history=sessions.get(chat_id, []))
            response = chat.send_message(user_input)
            if response.text:
                final_text = response.text
                used_model = model_id
                break
        except Exception as e:
            logger.warning(f"⚠️ Ошибка на {model_id}: {str(e)[:50]}")
            continue

    if final_text:
        if "gemma" in used_model.lower():
            final_text = f"⚠️ **{random.choice(GEMMA_PHRASES)}**\n\n{final_text}"

        update_history(chat_id, "user", user_input)
        update_history(chat_id, "model", response.text)

        if len(final_text) > 4000:
            for i in range(0, len(final_text), 4000):
                await message.answer(final_text[i:i+4000])
        else:
            await message.answer(final_text, parse_mode="Markdown")
    else:
        await message.answer("🤯 Перегрузка всех систем. Попробуй позже.")

# --- ЗАПУСК ---
async def main():
    logger.info("📡 Запуск MegaMoZg в Telegram...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())