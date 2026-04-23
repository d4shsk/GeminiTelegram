import telegramify_markdown


def format_for_telegram(text: str) -> str:
    if not text:
        return ""
    return telegramify_markdown.markdownify(text)


def split_message(text: str, chunk_size: int = 4000) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
