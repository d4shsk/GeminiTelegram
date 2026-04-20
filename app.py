import os
import asyncio
import random
import logging
import re
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from google import genai
from aiogram.client.session.aiohttp import AiohttpSession

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Отключаем спам от сторонних библиотек
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

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

def format_for_telegram(text: str) -> str:
    # Конвертируем Markdown от Gemini в поддерживаемый Telegram HTML
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'```(?:.*?)\n(.*?)\n?```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    text = re.sub(r'```(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
    text = re.sub(r'(?m)^\s*\*\s', '• ', text)
    text = re.sub(r'(?m)^#+\s+(.*)', r'<b>\1</b>', text)
    return text

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
    response_text_to_save = ""

    for model_id in MODEL_PRIORITY:
        try:
            # Асинхронный клиент + таймаут для быстрого переключения
            chat = client.aio.chats.create(model=model_id, history=sessions.get(chat_id, []))
            response = await asyncio.wait_for(
                chat.send_message(user_input),
                timeout=10.0 # Ждем 10 секунд, затем переходим к следующей модели
            )
            if response.text:
                final_text = response.text
                response_text_to_save = response.text
                used_model = model_id
                break
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Таймаут: модель {model_id} думала слишком долго.")
            continue
        except Exception as e:
            logger.warning(f"⚠️ Ошибка на {model_id}: {str(e)[:50]}")
            continue

    if final_text:
        formatted_text = format_for_telegram(final_text)

        if "gemma" in used_model.lower():
            formatted_text = f"⚠️ <b>{random.choice(GEMMA_PHRASES)}</b>\n\n{formatted_text}"

        update_history(chat_id, "user", user_input)
        update_history(chat_id, "model", response_text_to_save)

        if len(formatted_text) > 4000:
            for i in range(0, len(formatted_text), 4000):
                try:
                    await message.answer(formatted_text[i:i+4000], parse_mode="HTML")
                except Exception:
                    await message.answer(final_text[i:i+4000])
        else:
            try:
                await message.answer(formatted_text, parse_mode="HTML")
            except Exception:
                # Если HTML кривой, отправляем чистый текст
                await message.answer(final_text)
    else:
        await message.answer("🤯 Перегрузка всех систем. Попробуй позже.")

# --- ЗАПУСК ---
async def main():
    logger.info("📡 Запуск MegaMoZg в Telegram...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())