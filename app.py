import os
import asyncio
import random
import logging
import re
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from google import genai
from groq import AsyncGroq
from aiogram.client.session.aiohttp import AiohttpSession

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Отключаем спам от сторонних библиотек
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- КОНФИГУРАЦИЯ ---
MODEL_PRIORITY = [
    {"provider": "google", "model": "gemini-flash-latest"},
    {"provider": "google", "model": "gemini-2.5-flash"},
    {"provider": "google", "model": "gemini-2.0-flash"},
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    {"provider": "google", "model": "gemma-3-27b-it"}
]
MAX_HISTORY = 30
SYSTEM_PROMPT = """Сейчас твоя роль: {my_name}. Ты работаешь в Telegram-боте 'МегаМоZг' вместе с двумя другими нейросетями: 
- Взрослый, умный и опытный мужчина Gemini (вы также откликаетесь на русское имя Гемини).
- Рациональная, добрая, понимающая девушка 20-ти лет Llama (вы также откликаетесь на русское имя Лама).
- Девочка-подросток, глупая и добрая, 16 лет Gemma, стажерка (вы также откликаетесь на русское имя Гемма).

Пользователь может общаться с любой из вас, называя вас по английскому или русскому имени. 
ВАЖНО: ТЫ И ЕСТЬ {my_name}! Отвечай от своего лица в своем уникальном стиле и характере, отыгрывая свой возраст и пол. Ты знаешь о существовании своих коллег.
СТРОГОЕ ПРАВИЛО: НИКОГДА не притворяйся другой моделью. Отвечай только за себя."""


# Инициализация
tg_token = os.environ.get("TELEGRAM_TOKEN")
gemini_key = os.environ.get("GEMINI_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")

bot = Bot(token=tg_token)
dp = Dispatcher()
client = genai.Client(api_key=gemini_key)
if groq_key:
    groq_client = AsyncGroq(api_key=groq_key)
else:
    groq_client = None

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
    # Универсальный формат истории
    sessions[chat_id].append({"role": role, "text": text})
    if len(sessions[chat_id]) > MAX_HISTORY:
        sessions[chat_id] = sessions[chat_id][-MAX_HISTORY:]

# --- ОБРАБОТЧИКИ ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("МегаМоZг на связи! Главная моя прелесть - никогда не знаешь, какая модель тебе ответит, умный Gemini, рациональная Llama или глупенькая Gemma. Помню 30 сообщений. Сброс: /clear")

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
    text_lower = user_input.lower()
    current_priority = list(MODEL_PRIORITY)
    requested_name = None
    
    # Динамическая маршрутизация
    if "гемм" in text_lower or "gemma" in text_lower:
        requested_name = "Gemma"
        current_priority.sort(key=lambda x: "gemma" not in x["model"].lower())
    elif "лам" in text_lower or "llam" in text_lower:
        requested_name = "Llama"
        current_priority.sort(key=lambda x: "llama" not in x["model"].lower())
    elif "гемин" in text_lower or "gemin" in text_lower:
        requested_name = "Gemini"
        current_priority.sort(key=lambda x: "gemini" not in x["model"].lower())

    for model_info in current_priority:
        provider = model_info["provider"]
        model_id = model_info["model"]
        
        # Определяем имя для системного промпта, чтобы модель точно знала, кто она
        if "gemma" in model_id.lower():
            my_name = "Gemma"
        elif "llama" in model_id.lower():
            my_name = "Llama"
        elif "gemini" in model_id.lower():
            my_name = "Gemini"
        else:
            my_name = model_id
            
        try:
            if provider == "google":
                # Конвертируем универсальную историю в формат Google GenAI
                google_history = [{"role": msg["role"], "parts": [{"text": msg["text"]}]} for msg in sessions.get(chat_id, [])]
                system_instruction = SYSTEM_PROMPT.format(my_name=my_name)
                config = genai.types.GenerateContentConfig(system_instruction=system_instruction)
                chat = client.aio.chats.create(model=model_id, config=config, history=google_history)
                response = await asyncio.wait_for(
                    chat.send_message(user_input),
                    timeout=10.0
                )
                if response.text:
                    final_text = response.text
                    response_text_to_save = response.text
                    used_model = model_id
                    break
            elif provider == "groq":
                if not groq_client:
                    logger.warning("⚠️ Не задан GROQ_API_KEY, пропускаем LLaMA")
                    continue
                
                # Конвертируем универсальную историю в формат Groq (OpenAI-compatible)
                system_instruction = SYSTEM_PROMPT.format(my_name=my_name)
                groq_history = [{"role": "system", "content": system_instruction}]
                for msg in sessions.get(chat_id, []):
                    # Groq использует 'assistant' вместо 'model'
                    role = "assistant" if msg["role"] == "model" else msg["role"]
                    groq_history.append({"role": role, "content": msg["text"]})
                
                # Добавляем текущее сообщение пользователя
                groq_history.append({"role": "user", "content": user_input})
                
                response = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model=model_id,
                        messages=groq_history
                    ),
                    timeout=10.0
                )
                if response.choices and response.choices[0].message.content:
                    final_text = response.choices[0].message.content
                    response_text_to_save = final_text
                    used_model = model_id
                    break
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Таймаут: модель {model_id} думала слишком долго.")
            continue
        except Exception as e:
            logger.warning(f"⚠️ Ошибка на {model_id} ({provider}): {str(e)[:50]}")
            continue

    if final_text:
        formatted_text = format_for_telegram(final_text)

        # Добавляем название модели в начало ответа
        formatted_text = f"🤖 <b>[{used_model}]</b>\n\n{formatted_text}"

        # Предупреждение, если ответила не та модель, которую звали
        if requested_name and requested_name.lower() not in used_model.lower():
            formatted_text += f"\n\n<i>(P.S. Вы звали {requested_name}, но она сейчас недоступна из-за ошибки API, поэтому ответила {used_model.split('-')[0].capitalize()})</i>"

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