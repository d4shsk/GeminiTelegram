import os
import asyncio
import random
import logging
import re
import json
from datetime import datetime
import pytz
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from google import genai
from groq import AsyncGroq
from aiogram.client.session.aiohttp import AiohttpSession
from ddgs import DDGS

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
    {"provider": "groq", "model": "meta-llama/llama-4-scout-17b-16e-instruct"},
    {"provider": "google", "model": "gemma-3-27b-it"}
]
MAX_HISTORY = 30
SYSTEM_PROMPT = """Сейчас твоя роль: {my_name}. Ты работаешь в Telegram-боте 'МегаМоZг' вместе с двумя другими нейросетями: 
- Взрослый, умный и опытный мужчина Gemini (вы также откликаетесь на русское имя Гемини).
- Рациональная, добрая, понимающая девушка 20-ти лет Llama (вы также откликаетесь на русское имя Лама).
- Девочка-подросток, глупая и добрая, 16 лет Gemma, стажерка (вы также откликаетесь на русское имя Гемма).

Пользователь может общаться с любой из вас, называя вас по английскому или русскому имени. 
ВАЖНО: ТЫ И ЕСТЬ {my_name}! Отвечай от своего лица в своем уникальном стиле и характере, отыгрывая свой возраст и пол. Ты знаешь о существовании своих коллег.
СТРОГОЕ ПРАВИЛО: НИКОГДА не притворяйся другой моделью. Отвечай только за себя.

Текущие дата и время (МСК): {current_datetime}"""

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "Искать информацию в интернете (DuckDuckGo). Используй это для поиска новостей, актуальных событий или фактов, которых нет в твоей базе знаний. НЕ используй для вопросов о текущей дате/времени — эти данные уже есть в системном промпте.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def sync_web_search(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "Ничего не найдено."
        return "\n".join([f"Название: {r.get('title', '')}\nОписание: {r.get('body', '')}\nСсылка: {r.get('href', '')}" for r in results])
    except Exception as e:
        return f"Ошибка поиска: {str(e)}"

async def perform_web_search(query: str) -> str:
    return await asyncio.to_thread(sync_web_search, query)

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
active_models = {}
user_modes = {}
user_models = {}


def format_for_telegram(text: str) -> str:
    if not text:
        return ""
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
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Режим МегаМоZг 🤪", callback_data="mode_worker")],
        [InlineKeyboardButton(text="Серьезный режим 🧐", callback_data="mode_serious")]
    ])
    await message.answer("Выберите режим работы бота:", reply_markup=keyboard)

@dp.callback_query(F.data.startswith("mode_"))
async def handle_mode_selection(callback: CallbackQuery):
    mode = callback.data.split("_")[1]
    chat_id = callback.message.chat.id
    user_modes[chat_id] = mode
    
    if mode == "worker":
        await callback.message.edit_text("✅ Выбран режим МегаМоZг!\nГлавная моя прелесть - никогда не знаешь, какая модель тебе ответит, умный Gemini, рациональная Llama или глупенькая Gemma. Помню 30 сообщений. Сброс: /clear")
    else:
        user_models[chat_id] = "gemini-2.5-flash"
        
        buttons = []
        for model_info in MODEL_PRIORITY:
            buttons.append([InlineKeyboardButton(text=model_info["model"], callback_data=f"setmodel_{model_info['model']}")])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
        await callback.message.edit_text("✅ Выбран Серьезный режим!\nСистемный промпт отключен. Выберите предпочитаемую модель (при недоступности будет использована следующая по приоритету):", reply_markup=keyboard)
    await callback.answer()

@dp.message(Command("model"))
async def cmd_model(message: types.Message):
    chat_id = message.chat.id
    if user_modes.get(chat_id) != "serious":
        await message.answer("Команда /model доступна только в Серьезном режиме. Выберите режим через /start")
        return
        
    buttons = []
    for model_info in MODEL_PRIORITY:
        buttons.append([InlineKeyboardButton(text=model_info["model"], callback_data=f"setmodel_{model_info['model']}")])
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.answer("Выберите модель:", reply_markup=keyboard)

@dp.callback_query(F.data.startswith("setmodel_"))
async def handle_model_selection(callback: CallbackQuery):
    model_id = callback.data.split("_", 1)[1]
    user_models[callback.message.chat.id] = model_id
    await callback.message.edit_text(f"✅ Модель изменена на: {model_id}")
    await callback.answer()

@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    chat_id = message.chat.id
    sessions.pop(chat_id, None)
    active_models.pop(chat_id, None)
    await message.answer("🧹 Память и настройки переписки очищены.")

@dp.message(F.text)
async def handle_message(message: types.Message):
    chat_id = message.chat.id
    user_input = message.text
    mode = user_modes.get(chat_id, "worker")
    
    await bot.send_chat_action(chat_id, "typing")

    final_text = None
    used_model = ""
    response_text_to_save = ""
    text_lower = user_input.lower()
    
    requested_name = None
    
    if mode == "worker":
        current_priority = list(MODEL_PRIORITY)
        # Запоминаем выбор пользователя, если он явно позвал модель
        if "гемм" in text_lower or "gemma" in text_lower:
            active_models[chat_id] = "gemma"
        elif "лам" in text_lower or "llam" in text_lower:
            active_models[chat_id] = "llama"
        elif "гемин" in text_lower or "gemin" in text_lower:
            active_models[chat_id] = "gemini"
            
        saved_model = active_models.get(chat_id)
        
        # Динамическая маршрутизация на основе сохраненного или нового выбора
        if saved_model == "gemma":
            requested_name = "Gemma"
            current_priority.sort(key=lambda x: "gemma" not in x["model"].lower())
        elif saved_model == "llama":
            requested_name = "Llama"
            current_priority.sort(key=lambda x: "llama" not in x["model"].lower())
        elif saved_model == "gemini":
            requested_name = "Gemini"
            current_priority.sort(key=lambda x: "gemini" not in x["model"].lower())
    else:
        # Серьезный режим
        selected_model_id = user_models.get(chat_id, "gemini-2.5-flash")
        
        current_priority = []
        # Сначала добавляем выбранную модель
        for m in MODEL_PRIORITY:
            if m["model"] == selected_model_id:
                current_priority.append(m)
                break
        
        # Затем добавляем остальные модели как запасные
        for m in MODEL_PRIORITY:
            if m["model"] != selected_model_id:
                current_priority.append(m)

    for model_info in current_priority:
        provider = model_info["provider"]
        model_id = model_info["model"]
        
        # Определяем имя для системного промпта, чтобы модель точно знала, кто она (только для worker)
        if mode == "worker":
            if "gemma" in model_id.lower():
                my_name = "Gemma"
            elif "llama" in model_id.lower():
                my_name = "Llama"
            elif "gemini" in model_id.lower():
                my_name = "Gemini"
            else:
                my_name = model_id
            tz_moscow = pytz.timezone("Europe/Moscow")
            current_datetime = datetime.now(tz_moscow).strftime("%d %B %Y, %H:%M МСК")
            system_instruction = SYSTEM_PROMPT.format(my_name=my_name, current_datetime=current_datetime)
        else:
            system_instruction = None
            
        try:
            if provider == "google":
                # Конвертируем универсальную историю в формат Google GenAI
                google_history = [{"role": msg["role"], "parts": [{"text": msg["text"]}]} for msg in sessions.get(chat_id, [])]
                
                if "gemma" in model_id.lower():
                    # Gemma в Google API часто не поддерживает config/tools, передаем текстом
                    chat = client.aio.chats.create(model=model_id, history=google_history)
                    if system_instruction:
                        current_input = f"[{system_instruction}]\n\nСообщение пользователя: {user_input}"
                    else:
                        current_input = user_input
                    response = await asyncio.wait_for(
                        chat.send_message(current_input),
                        timeout=15.0
                    )
                else:
                    # Для Gemini используем штатный config и добавляем поиск в интернете
                    if system_instruction:
                        config = genai.types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            tools=[{"google_search": {}}]
                        )
                    else:
                        config = genai.types.GenerateContentConfig(
                            tools=[{"google_search": {}}]
                        )
                        
                    chat = client.aio.chats.create(model=model_id, config=config, history=google_history)
                    response = await asyncio.wait_for(
                        chat.send_message(user_input),
                        timeout=15.0
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
                
                # Конвертируем универсальную историю в формат Groq
                groq_history = []
                tz_moscow = pytz.timezone("Europe/Moscow")
                current_datetime = datetime.now(tz_moscow).strftime("%d %B %Y, %H:%M МСК")
                if system_instruction:
                    groq_history.append({"role": "system", "content": system_instruction})
                else:
                    # В серьёзном режиме системного промпта нет — добавляем только дату
                    groq_history.append({"role": "system", "content": f"Текущие дата и время (МСК): {current_datetime}"})
                    
                for msg in sessions.get(chat_id, []):
                    role = "assistant" if msg["role"] == "model" else msg["role"]
                    groq_history.append({"role": role, "content": msg["text"]})
                
                # Добавляем текущее сообщение пользователя
                groq_history.append({"role": "user", "content": user_input})
                
                response = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model=model_id,
                        messages=groq_history,
                        tools=GROQ_TOOLS,
                        tool_choice="auto"
                    ),
                    timeout=15.0
                )
                
                response_message = response.choices[0].message
                
                if response_message.tool_calls:
                    tool_calls_dicts = []
                    for tc in response_message.tool_calls:
                        tool_calls_dicts.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        })
                    
                    groq_history.append({
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": tool_calls_dicts
                    })
                    
                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "search_internet":
                            args = json.loads(tool_call.function.arguments)
                            search_query = args.get("query", "")
                            logger.info(f"Groq tool call: search_internet for '{search_query}'")
                            search_result = await perform_web_search(search_query)

                            print(f"DEBUG: Search results: {search_result}", flush=True)
                            
                            groq_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": search_result
                            })
                            
                    # Второй вызов после получения результатов поиска
                    response = await asyncio.wait_for(
                        groq_client.chat.completions.create(
                            model=model_id,
                            messages=groq_history,
                        ),
                        timeout=15.0
                    )
                    response_message = response.choices[0].message

                if response_message and response_message.content:
                    final_text = response_message.content
                    response_text_to_save = final_text
                    used_model = model_id
                    break
                    
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Таймаут: модель {model_id} думала слишком долго.")
            continue
        except Exception as e:
            if "tool_use_failed" in str(e):
                logger.warning("⚠️ Ошибка парсинга инструмента Groq, повторяем без интернета...")
                try:
                    response = await asyncio.wait_for(
                        groq_client.chat.completions.create(
                            model=model_id,
                            messages=groq_history
                        ),
                        timeout=15.0
                    )
                    response_message = response.choices[0].message
                    if response_message and response_message.content:
                        final_text = response_message.content
                        response_text_to_save = final_text
                        used_model = model_id
                        break
                except Exception as ex:
                    logger.warning(f"⚠️ Ошибка повторного вызова {model_id}: {str(ex)[:50]}")
                    continue
            else:
                logger.warning(f"⚠️ Ошибка на {model_id} ({provider}): {str(e)[:50]}")
                continue

    if final_text:
        formatted_text = format_for_telegram(final_text)

        # Добавляем название модели в начало ответа
        mode_indicator = "🧐 [Серьезный]" if mode == "serious" else "🤪 [МегаМоZг]"
        formatted_text = f"🤖 <b>[{used_model}]</b> {mode_indicator}\n\n{formatted_text}"

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