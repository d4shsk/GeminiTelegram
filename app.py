import os
import asyncio
import random
import logging
import re
import json
import io
import base64
import httpx
from datetime import datetime
import pytz
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from google import genai
from groq import AsyncGroq
import openai as openai_lib
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
    {"provider": "groq",        "model": "llama-3.3-70b-versatile"},
    {"provider": "google",      "model": "gemini-2.5-flash"},
    {"provider": "google",      "model": "gemma-3-27b-it"},
    {"provider": "cloudflare",  "model": "@cf/moonshotai/kimi-k2.6", "serious_only": True},
    {"provider": "github",      "model": "gpt-4o", "serious_only": True},
]
MAX_HISTORY = 30

MODEL_RATING_TEXT = (
    "\n\n🏆 <b>Рейтинг моделей:</b>\n"
    "1. <code>kimi-k2.6</code> — Видит картинки\n"
    "2. <code>gemini-2.5-flash</code>\n"
    "3. <code>gpt-4o</code> — Видит картинки\n"
    "4. <code>llama-3.3-70b-versatile</code>\n"
    "5. <code>gemma-3-27b-it</code>\n"
    "\n🎨 <b>FLUX</b> — генерация картинок (выберите в меню)"
)
SYSTEM_PROMPT = """Сейчас твоя роль: {my_name}. Ты работаешь в Telegram-боте 'DummyLLM' (Дамми ЛЛМ) вместе с двумя другими нейросетями: 
- Взрослый, умный и опытный мужчина Gemini (вы также откликаетесь на русское имя Гемини).
- Рациональная, добрая, понимающая девушка 20-ти лет Llama (вы также откликаетесь на русское имя Лама).
- Девочка-подросток, глупая и добрая, 16 лет Gemma, стажерка (вы также откликаетесь на русское имя Гемма).

Пользователь может общаться с любой из вас, называя вас по английскому или русскому имени. 
ВАЖНО: ТЫ И ЕСТЬ {my_name}! Отвечай от своего лица в своем уникальном стиле и характере, отыгрывая свой возраст и пол. Ты знаешь о существовании своих коллег.
СТРОГОЕ ПРАВИЛО: НИКОГДА не притворяйся другой моделью. Отвечай только за себя.

Текущие дата и время (МСК): {current_datetime}"""

SEARCH_TOOLS = [
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
github_token = os.environ.get("GITHUB_TOKEN")
cf_account_id = os.environ.get("CF_ACCOUNT_ID")
cf_api_token = os.environ.get("CF_API_TOKEN")

bot = Bot(token=tg_token)
dp = Dispatcher()
client = genai.Client(api_key=gemini_key)
if groq_key:
    groq_client = AsyncGroq(api_key=groq_key)
else:
    groq_client = None
if github_token:
    github_ai_client = openai_lib.AsyncOpenAI(
        api_key=github_token,
        base_url="https://models.inference.ai.azure.com",
    )
else:
    github_ai_client = None
if cf_account_id and cf_api_token:
    cf_client = openai_lib.AsyncOpenAI(
        api_key=cf_api_token,
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/v1",
    )
else:
    cf_client = None

sessions = {}
active_models = {}
user_modes = {}
user_models = {}
# Словарь для отслеживания пользователей, ожидающих промт для FLUX
awaiting_flux_prompt = {}


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

def get_main_menu() -> ReplyKeyboardMarkup:
    """Постоянное нижнее меню, доступное всегда."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Сменить режим"), KeyboardButton(text="🤖 Сменить модель")],
            [KeyboardButton(text="🧹 Очистить историю")]
        ],
        resize_keyboard=True
    )

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    inline_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Шуточный режим (RolePlay)🤪", callback_data="mode_worker")],
        [InlineKeyboardButton(text="Серьезный режим 🧐", callback_data="mode_serious")]
    ])
    await message.answer(
        "Выберите режим работы бота:\n"
        "1. Шуточный режим. Нейросети думают, что они реальные люди и работают в телеграмм боте.\n"
        "2. Серьезный режим. Нейросети отвечают стандартно.",
        reply_markup=inline_kb
    )

@dp.callback_query(F.data.startswith("mode_"))
async def handle_mode_selection(callback: CallbackQuery):
    mode = callback.data.split("_")[1]
    chat_id = callback.message.chat.id
    user_modes[chat_id] = mode

    # Очищаем историю при смене режима
    sessions.pop(chat_id, None)
    active_models.pop(chat_id, None)
    awaiting_flux_prompt.pop(chat_id, None)

    if mode == "worker":
        await callback.message.edit_text(
            "✅ Выбран режим RolePlay!\n"
            "Никогда не знаешь, какая модель ответит: умный Gemini, рациональная Llama или глупенькая Gemma.\n"
            "История очищена. Помню до 30 сообщений."
        )
        # Показываем постоянное меню отдельным сообщением
        await callback.message.answer("Меню всегда под рукой 👇", reply_markup=get_main_menu())
    else:
        user_models[chat_id] = "gemini-2.5-flash"

        model_buttons = _build_model_buttons()
        inline_kb = InlineKeyboardMarkup(inline_keyboard=model_buttons)
        await callback.message.edit_text(
            "✅ Выбран Серьезный режим! История очищена.\n"
            "Системный промпт отключен. Выберите предпочитаемую модель (при недоступности будет использована следующая по приоритету):"
            + MODEL_RATING_TEXT,
            reply_markup=inline_kb,
            parse_mode="HTML"
        )
        # Показываем постоянное меню отдельным сообщением
        await callback.message.answer("Меню всегда под рукой 👇", reply_markup=get_main_menu())
    await callback.answer()

def _build_model_buttons() -> list:
    """Строит список кнопок для выбора модели, включая FLUX."""
    model_buttons = []
    for model_info in MODEL_PRIORITY:
        label = model_info["model"].replace(":free", "")
        if model_info.get("serious_only"):
            label = "🔬 " + label
        model_buttons.append([InlineKeyboardButton(text=label, callback_data=f"setmodel_{model_info['model']}")])
    # Добавляем кнопку FLUX
    model_buttons.append([InlineKeyboardButton(text="🎨 FLUX (генерация картинок)", callback_data="setmodel_flux")])
    return model_buttons

def show_model_picker_text() -> str:
    return "Выберите модель:" + MODEL_RATING_TEXT

async def send_model_picker(target, chat_id: int):
    """Отправляет меню выбора модели. target — message или callback.message."""
    inline_kb = InlineKeyboardMarkup(inline_keyboard=_build_model_buttons())
    await target.answer(show_model_picker_text(), reply_markup=inline_kb, parse_mode="HTML")

@dp.message(Command("model"))
async def cmd_model(message: types.Message):
    chat_id = message.chat.id
    if user_modes.get(chat_id) != "serious":
        await message.answer("Команда /model доступна только в Серьезном режиме. Выберите режим через /start")
        return
    await send_model_picker(message, chat_id)

@dp.callback_query(F.data.startswith("setmodel_"))
async def handle_model_selection(callback: CallbackQuery):
    model_id = callback.data.split("_", 1)[1]
    chat_id = callback.message.chat.id
    user_models[chat_id] = model_id
    awaiting_flux_prompt.pop(chat_id, None)

    if model_id == "flux":
        awaiting_flux_prompt[chat_id] = True
        await callback.message.edit_text(
            "🎨 <b>FLUX — генерация картинок</b>\n\n"
            "Напишите промт для генерации изображения.\n"
            "<i>Например: закат над горами в стиле аниме</i>",
            parse_mode="HTML"
        )
    else:
        await callback.message.edit_text(f"✅ Модель изменена на: {model_id}")
    await callback.answer()

@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    chat_id = message.chat.id
    sessions.pop(chat_id, None)
    active_models.pop(chat_id, None)
    awaiting_flux_prompt.pop(chat_id, None)
    await message.answer("🧹 История очищена.", reply_markup=get_main_menu())

# --- ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ (FLUX) ---

async def generate_image_cf(prompt: str) -> bytes | None:
    """Генерирует изображение через Cloudflare flux-1-schnell. Возвращает PNG байты."""
    if not (cf_account_id and cf_api_token):
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    headers = {"Authorization": f"Bearer {cf_api_token}"}
    payload = {"prompt": prompt}
    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            ct = resp.headers.get("content-type", "")
            if "image" in ct:
                return resp.content
            # Cloudflare может вернуть JSON с base64
            try:
                data = resp.json()
                b64 = (data.get("result") or {}).get("image") or data.get("image")
                if b64:
                    return base64.b64decode(b64)
            except Exception:
                pass
        logger.warning(f"⚠️ Flux ошибка {resp.status_code}: {resp.text[:200]}")
        return None

@dp.message(Command("image"))
async def cmd_image(message: types.Message):
    """Генерирует изображение по тексту через FLUX-1-schnell (Cloudflare)."""
    prompt = message.text.partition(" ")[2].strip()
    if not prompt:
        await message.answer("✏️ Укажи запрос после команды, например:\n<code>/image закат над горами</code>", parse_mode="HTML")
        return
    if not (cf_account_id and cf_api_token):
        await message.answer("⚠️ Не заданы CF_ACCOUNT_ID / CF_API_TOKEN, генерация недоступна.")
        return
    await bot.send_chat_action(message.chat.id, "upload_photo")
    img_bytes = await generate_image_cf(prompt)
    if img_bytes:
        buf = io.BytesIO(img_bytes)
        buf.name = "image.png"
        await message.answer_photo(
            types.BufferedInputFile(buf.getvalue(), filename="image.png"),
            caption=f"🎨 <b>FLUX-1-schnell</b>\n<i>{prompt[:200]}</i>",
            parse_mode="HTML"
        )
    else:
        await message.answer("❌ Не удалось сгенерировать изображение. Попробуй позже.")

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    """Обработчик фото — видение через gpt-4o (GitHub Models) или kimi-k2.6 (Cloudflare)."""
    chat_id = message.chat.id
    mode = user_modes.get(chat_id, "worker")

    if mode != "serious":
        await message.answer("📷 Отправка картинок доступна только в Серьёзном режиме с моделью gpt-4o или kimi-k2.6.")
        return

    selected_model = user_models.get(chat_id, "")
    vision_models = {"gpt-4o", "@cf/moonshotai/kimi-k2.6"}

    if selected_model not in vision_models:
        await message.answer(
            "📷 Отправка картинок работает только с моделями <code>gpt-4o</code> или <code>🔬 @cf/moonshotai/kimi-k2.6</code>.\n"
            "Смените модель через 🤖 Сменить модель.",
            parse_mode="HTML"
        )
        return

    await bot.send_chat_action(chat_id, "typing")

    # Скачиваем крупнейшее фото (file_id последнее = максимальное разрешение)
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    caption = message.caption or "Что на этом изображении? Опиши подробно."
    tz_moscow = pytz.timezone("Europe/Moscow")
    current_datetime = datetime.now(tz_moscow).strftime("%d %B %Y, %H:%M МСК")

    if selected_model == "gpt-4o":
        if not github_ai_client:
            await message.answer("⚠️ Не задан GITHUB_TOKEN, зрение недоступно.")
            return
        try:
            gh_response = await asyncio.wait_for(
                github_ai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Отвечай на русском языке. Описывай изображение подробно и чётко. "
                                f"Текущие дата и время (МСК): {current_datetime}"
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": caption},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1024,
                    temperature=0.3,
                ),
                timeout=45.0
            )
            answer = (gh_response.choices[0].message.content or "").strip()
            if answer:
                formatted = format_for_telegram(answer)
                formatted = f"🤖 <b>[gpt-4o]</b> 📷 [Vision]\n\n{formatted}"
                try:
                    await message.answer(formatted, parse_mode="HTML")
                except Exception:
                    await message.answer(answer)
            else:
                await message.answer("🤔 Модель не смогла дать ответ на это изображение.")
        except asyncio.TimeoutError:
            await message.answer("⏱ Обработка изображения заняла слишком много времени. Попробуй ещё раз.")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка Vision gpt-4o: {str(e)[:100]}")
            await message.answer(f"❌ Ошибка при анализе изображения: {str(e)[:80]}")

    elif selected_model == "@cf/moonshotai/kimi-k2.6":
        if not cf_client:
            await message.answer("⚠️ Не заданы CF_ACCOUNT_ID / CF_API_TOKEN, зрение недоступно.")
            return
        try:
            kimi_response = await asyncio.wait_for(
                cf_client.chat.completions.create(
                    model="@cf/moonshotai/kimi-k2.6",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Отвечай на русском языке. Описывай изображение подробно и чётко. "
                                f"Текущие дата и время (МСК): {current_datetime}"
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": caption},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1024,
                ),
                timeout=45.0
            )
            answer = (kimi_response.choices[0].message.content or "").strip()
            if answer:
                formatted = format_for_telegram(answer)
                formatted = f"🤖 <b>[kimi-k2.6]</b> 📷 [Vision]\n\n{formatted}"
                try:
                    await message.answer(formatted, parse_mode="HTML")
                except Exception:
                    await message.answer(answer)
            else:
                await message.answer("🤔 Модель не смогла дать ответ на это изображение.")
        except asyncio.TimeoutError:
            await message.answer("⏱ Обработка изображения заняла слишком много времени. Попробуй ещё раз.")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка Vision kimi-k2.6: {str(e)[:100]}")
            await message.answer(f"❌ Ошибка при анализе изображения: {str(e)[:80]}")

@dp.message(F.text)
async def handle_message(message: types.Message):
    chat_id = message.chat.id
    user_input = message.text
    mode = user_modes.get(chat_id, "worker")

    # --- Обработка кнопок постоянного меню ---
    if user_input == "🔄 Сменить режим":
        inline_kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Шуточный режим (RolePlay)🤪", callback_data="mode_worker")],
            [InlineKeyboardButton(text="Серьезный режим 🧐", callback_data="mode_serious")]
        ])
        await message.answer(
            "Выберите режим работы бота:\n"
            "1. Шуточный режим. Нейросети думают, что они реальные люди и работают в телеграмм боте.\n"
            "2. Серьезный режим. Нейросети отвечают стандартно.",
            reply_markup=inline_kb
        )
        return

    if user_input == "🤖 Сменить модель":
        if mode != "serious":
            await message.answer("Выбор модели доступен только в Серьезном режиме. Сначала выберите режим через 🔄 Сменить режим.")
        else:
            await send_model_picker(message, chat_id)
        return

    if user_input == "🧹 Очистить историю":
        sessions.pop(chat_id, None)
        active_models.pop(chat_id, None)
        awaiting_flux_prompt.pop(chat_id, None)
        await message.answer("🧹 История очищена.", reply_markup=get_main_menu())
        return

    # --- Обработка режима ожидания промта для FLUX ---
    if awaiting_flux_prompt.get(chat_id):
        awaiting_flux_prompt.pop(chat_id, None)
        if not (cf_account_id and cf_api_token):
            await message.answer("⚠️ Не заданы CF_ACCOUNT_ID / CF_API_TOKEN, генерация недоступна.")
            return
        await bot.send_chat_action(chat_id, "upload_photo")
        img_bytes = await generate_image_cf(user_input)
        if img_bytes:
            buf = io.BytesIO(img_bytes)
            buf.name = "image.png"
            await message.answer_photo(
                types.BufferedInputFile(buf.getvalue(), filename="image.png"),
                caption=f"🎨 <b>FLUX-1-schnell</b>\n<i>{user_input[:200]}</i>",
                parse_mode="HTML"
            )
        else:
            await message.answer("❌ Не удалось сгенерировать изображение. Попробуй позже.")
        return

    await bot.send_chat_action(chat_id, "typing")

    final_text = None
    used_model = ""
    response_text_to_save = ""
    text_lower = user_input.lower()
    
    requested_name = None
    
    if mode == "worker":
        # В шуточном режиме не используем serious_only модели
        current_priority = [m for m in MODEL_PRIORITY if not m.get("serious_only")]
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

        # Если выбран FLUX — предложить написать промт
        if selected_model_id == "flux":
            awaiting_flux_prompt[chat_id] = True
            await message.answer(
                "🎨 <b>FLUX — генерация картинок</b>\n\nНапишите промт для генерации изображения.",
                parse_mode="HTML"
            )
            return

        current_priority = []
        # Сначала добавляем выбранную модель
        for m in MODEL_PRIORITY:
            if m["model"] == selected_model_id:
                current_priority.append(m)
                break

        # Затем добавляем запасные модели: сначала gpt-4o, потом остальные (llama последней)
        # Порядок запасных: gpt-4o → gemini-2.5-flash → gemma → kimi → llama
        fallback_order = [
            "gpt-4o",
            "gemini-2.5-flash",
            "@cf/moonshotai/kimi-k2.6",
            "gemma-3-27b-it",
            "llama-3.3-70b-versatile",
        ]
        added = {selected_model_id}
        for fallback_id in fallback_order:
            if fallback_id not in added:
                for m in MODEL_PRIORITY:
                    if m["model"] == fallback_id:
                        current_priority.append(m)
                        added.add(fallback_id)
                        break
        # Добавляем всё, что не попало (на случай новых моделей)
        for m in MODEL_PRIORITY:
            if m["model"] not in added:
                current_priority.append(m)
                added.add(m["model"])

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
                        tools=SEARCH_TOOLS,
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
                            logger.info(f"Search results: {search_result[:200]}")
                            
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

            elif provider == "cloudflare":
                if not cf_client:
                    logger.warning("⚠️ Не заданы CF_ACCOUNT_ID / CF_API_TOKEN, пропускаем Kimi")
                    continue

                tz_moscow = pytz.timezone("Europe/Moscow")
                current_datetime = datetime.now(tz_moscow).strftime("%d %B %Y, %H:%M МСК")
                cf_history = [{"role": "system", "content": f"Ты — умная языковая модель. Отвечай чётко и полезно. Текущие дата и время (МСК): {current_datetime}"}]
                for msg in sessions.get(chat_id, []):
                    role = "assistant" if msg["role"] == "model" else msg["role"]
                    cf_history.append({"role": role, "content": msg["text"]})
                cf_history.append({"role": "user", "content": user_input})

                cf_text = ""
                
                
                # Делаем максимум 7 обращений к API (6 попыток поиска + 1 принудительный ответ)
                for attempt in range(7):
                    # Отправляем статус "печатает", чтобы юзер понимал, что процесс идет
                    await bot.send_chat_action(chat_id, "typing")
                    
                    if attempt < 6:
                        # Первые 6 попыток: разрешаем модели использовать поиск
                        cf_response = await asyncio.wait_for(
                            cf_client.chat.completions.create(
                                model=model_id,
                                messages=cf_history,
                                tools=SEARCH_TOOLS, 
                                tool_choice="auto",
                                max_tokens=2048,
                            ),
                            timeout=60.0
                        )
                    else:
                        # 7-я попытка (финальная): отключаем tools! 
                        # Теперь модель ОБЯЗАНА выдать текстовый ответ на основе того, что нашла за 6 раз.
                        cf_response = await asyncio.wait_for(
                            cf_client.chat.completions.create(
                                model=model_id,
                                messages=cf_history,
                                max_tokens=2048,
                            ),
                            timeout=60.0
                        )

                    cf_message = cf_response.choices[0].message
                    cf_text = (cf_message.content or "").strip()

                    # 1. Проверяем стандартные tool_calls
                    if getattr(cf_message, "tool_calls", None):
                        tool_calls_dicts = []
                        for tc in cf_message.tool_calls:
                            tool_calls_dicts.append({
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                            })
                        cf_history.append({
                            "role": "assistant",
                            "content": cf_message.content,
                            "tool_calls": tool_calls_dicts
                        })
                        
                        for tool_call in cf_message.tool_calls:
                            if tool_call.function.name == "search_internet":
                                try:
                                    args = json.loads(tool_call.function.arguments)
                                    search_query = args.get("query", "")
                                    logger.info(f"Kimi tool call: search_internet for '{search_query}'")
                                    search_result = await perform_web_search(search_query)
                                    cf_history.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": tool_call.function.name,
                                        "content": search_result
                                    })
                                except Exception as e:
                                    logger.error(f"Ошибка парсинга аргументов Kimi: {e}")
                        continue 

                    # 2. Резервный парсинг сырых токенов
                    elif "<|toolcallbegin|>" in cf_text and "searchinternet" in cf_text.lower():
                        match = re.search(r'\{"query":\s*"(.*?)"\}', cf_text)
                        if match:
                            search_query = match.group(1)
                            logger.info(f"Kimi (raw token) tool call: search_internet for '{search_query}'")
                            search_result = await perform_web_search(search_query)
                            cf_history.append({"role": "assistant", "content": cf_text})
                            cf_history.append({"role": "user", "content": f"Результаты поиска:\n{search_result}"})
                            continue

                    # 3. Текстовый ответ без попыток поиска! Выходим из цикла.
                    else:
                        break
                    
                # Финальная проверка перед сохранением ответа
                if cf_text and "<|toolcallssectionbegin|>" not in cf_text:
                    final_text = cf_text
                    response_text_to_save = cf_text
                    used_model = model_id
                    break
            
            
            elif provider == "github":
                if not github_ai_client:
                    logger.warning("⚠️ Не задан GITHUB_TOKEN, пропускаем GitHub Models")
                    continue

                # Собираем историю в формате OpenAI
                gh_history = []
                tz_moscow = pytz.timezone("Europe/Moscow")
                current_datetime = datetime.now(tz_moscow).strftime("%d %B %Y, %H:%M МСК")
                gh_history.append({
                    "role": "system",
                    "content": (
                        "Ты — умная языковая модель. Отвечай чётко, думай аналитически, "
                        "объясняй своё рассуждение. "
                        f"Текущие дата и время (МСК): {current_datetime}"
                    )
                })
                for msg in sessions.get(chat_id, []):
                    role = "assistant" if msg["role"] == "model" else msg["role"]
                    gh_history.append({"role": role, "content": msg["text"]})
                gh_history.append({"role": "user", "content": user_input})

                gh_response = await asyncio.wait_for(
                    github_ai_client.chat.completions.create(
                        model=model_id,
                        messages=gh_history,
                        tools=SEARCH_TOOLS,
                        tool_choice="auto",
                        max_tokens=1024,
                        temperature=0.3,
                    ),
                    timeout=30.0
                )
                gh_message = gh_response.choices[0].message

                if gh_message.tool_calls:
                    tool_calls_dicts = []
                    for tc in gh_message.tool_calls:
                        tool_calls_dicts.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        })
                    gh_history.append({
                        "role": "assistant",
                        "content": gh_message.content,
                        "tool_calls": tool_calls_dicts
                    })
                    for tool_call in gh_message.tool_calls:
                        if tool_call.function.name == "search_internet":
                            args = json.loads(tool_call.function.arguments)
                            search_query = args.get("query", "")
                            logger.info(f"GPT-4o tool call: search_internet for '{search_query}'")
                            search_result = await perform_web_search(search_query)
                            gh_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": search_result
                            })
                    gh_response2 = await asyncio.wait_for(
                        github_ai_client.chat.completions.create(
                            model=model_id,
                            messages=gh_history,
                            max_tokens=1024,
                            temperature=0.3,
                        ),
                        timeout=30.0
                    )
                    gh_message = gh_response2.choices[0].message

                gh_text = (gh_message.content or "").strip()
                if gh_text:
                    final_text = gh_text
                    response_text_to_save = gh_text
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
        mode_indicator = "🧐 [Стандартный]" if mode == "serious" else "🤪 [RolePlay]"
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
    logger.info("📡 Запуск бота в Telegram...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
