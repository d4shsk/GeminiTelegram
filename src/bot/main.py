from aiogram import Dispatcher

from .handlers import all_routers
from .runtime import logger, runtime
from .state import state


async def run() -> None:
    if not runtime.bot:
        raise RuntimeError("Не найден TELEGRAM_TOKEN.")

    state.initialize()
    dispatcher = Dispatcher()
    for router in all_routers:
        dispatcher.include_router(router)

    logger.info("Запуск Telegram-бота...")
    await dispatcher.start_polling(runtime.bot)
