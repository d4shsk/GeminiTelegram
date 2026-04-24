MODEL_PRIORITY = [
    {"provider": "github", "model": "gpt-4o", "serious_only": True},
    {"provider": "google", "model": "gemini-2.5-flash-lite"},
    {"provider": "google", "model": "gemini-2.5-flash"},
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    {"provider": "cloudflare", "model": "@cf/moonshotai/kimi-k2.6", "serious_only": True},
    {"provider": "google", "model": "gemma-3-27b-it"},
]

MAX_HISTORY = 30
HISTORY_KEEP_MESSAGES = 8
HISTORY_TOKEN_BUDGET = 3600
SUMMARY_CHAR_BUDGET = 1200

MODE_WORKER = "worker"
MODE_SERIOUS = "serious"

MENU_CHANGE_MODE = "🔄 Сменить режим"
MENU_CHANGE_MODEL = "🤖 Сменить модель"
MENU_CLEAR_HISTORY = "🧹 Очистить историю"

VISION_MODELS = {"gemini-2.5-flash", "gpt-4o", "@cf/moonshotai/kimi-k2.6"}

SERIOUS_FALLBACK_ORDER = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gpt-4o",
    "llama-3.3-70b-versatile",
    "@cf/moonshotai/kimi-k2.6",
    "gemma-3-27b-it",
]

MODEL_RATING_TEXT = (
    "\n\n🏆 <b>Рейтинг моделей:</b>\n"
    "1. <code>gemini-2.5-flash-lite</code> — ответ по умолчанию\n"
    "2. <code>gpt-4o</code> — авто-подъем для сложных запросов\n"
    "3. <code>gemini-2.5-flash</code> — запасной Google fallback\n"
    "4. <code>llama-3.3-70b-versatile</code>\n"
    "5. <code>kimi-k2.6</code> — резервный multimodal fallback\n"
    "6. <code>gemma-3-27b-it</code> — крайний резерв\n"
)

SYSTEM_PROMPT = """Сейчас твоя роль: {my_name}. Ты работаешь в Telegram-боте DummyLLM вместе с другими моделями.
Пользователь может обращаться к тебе по имени на русском или английском.
Важно: ты и есть {my_name}. Никогда не притворяйся другой моделью.
Отвечай на русском языке, соблюдай структуру. Markdown-таблицы допустимы, если они действительно улучшают ответ: бот адаптирует их для мобильного Telegram.

Текущие дата и время (МСК): {current_datetime}"""

SERIOUS_SYSTEM_PROMPT = """Ты — экспертный и объективный ИИ-ассистент.
Отвечай точно, структурированно и по делу.

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

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression deterministically. "
            "Use this for arithmetic and exact calculations instead of mental math. "
            "Supported notation: n! means factorial, !n means subfactorial."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate. Example: (2+3)^2 is written as (2+3)**2, 7! is factorial, !7 is subfactorial.",
                }
            },
            "required": ["expression"],
        },
    },
}
