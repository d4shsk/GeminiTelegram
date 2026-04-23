import asyncio

from ddgs import DDGS


def sync_web_search(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "Ничего не найдено."
        formatted = []
        for item in results:
            formatted.append(
                f"Название: {item.get('title', '')}\n"
                f"Описание: {item.get('body', '')}\n"
                f"Ссылка: {item.get('href', '')}"
            )
        return "\n".join(formatted)
    except Exception as error:
        return f"Ошибка поиска: {error}"


async def perform_web_search(query: str) -> str:
    return await asyncio.to_thread(sync_web_search, query)
