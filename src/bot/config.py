MODEL_PRIORITY = [
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    {"provider": "google", "model": "gemini-2.5-flash"},
    {"provider": "google", "model": "gemma-3-27b-it"},
    {"provider": "cloudflare", "model": "@cf/moonshotai/kimi-k2.6", "serious_only": True},
    {"provider": "github", "model": "gpt-4o", "serious_only": True},
]

MAX_HISTORY = 30

MODE_WORKER = "worker"
MODE_SERIOUS = "serious"

MENU_CHANGE_MODE = "🔄 Сменить режим"
MENU_CHANGE_MODEL = "🤖 Сменить модель"
MENU_CLEAR_HISTORY = "🧹 Очистить историю"

VISION_MODELS = {"gemini-2.5-flash", "gpt-4o", "@cf/moonshotai/kimi-k2.6"}

SERIOUS_FALLBACK_ORDER = [
    "gpt-4o",
    "gemini-2.5-flash",
    "@cf/moonshotai/kimi-k2.6",
    "gemma-3-27b-it",
    "llama-3.3-70b-versatile",
]

MODEL_RATING_TEXT = (
    "\n\n🏆 <b>Рейтинг моделей:</b>\n"
    "1. <code>kimi-k2.6</code> — Видит картинки\n"
    "2. <code>gemini-2.5-flash</code> — Видит картинки\n"
    "3. <code>gpt-4o</code> — Видит картинки\n"
    "4. <code>llama-3.3-70b-versatile</code>\n"
    "5. <code>gemma-3-27b-it</code>\n"
)

SYSTEM_PROMPT = """Сейчас твоя роль: {my_name}. Ты работаешь в Telegram-боте DummyLLM вместе с другими моделями.
Пользователь может обращаться к тебе по имени на русском или английском.
Важно: ты и есть {my_name}. Никогда не притворяйся другой моделью.
Отвечай на русском языке, соблюдай структуру, избегай markdown-таблиц с символом |.

Текущие дата и время (МСК): {current_datetime}"""

SERIOUS_SYSTEM_PROMPT = """Ты — экспертный и объективный ИИ-ассистент.
Отвечай точно, структурированно и по делу. Избегай markdown-таблиц с символом |.

Текущие дата и время (МСК): {current_datetime}"""

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": (
                "Искать информацию в интернете (DuckDuckGo). "
                "Используй для новостей, актуальных событий и свежих фактов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос",
                    }
                },
                "required": ["query"],
            },
        },
    }
]
