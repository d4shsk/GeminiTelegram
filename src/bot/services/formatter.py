import telegramify_markdown


TELEGRAM_SPACER_LINE = "\u2063"


def format_for_telegram(text: str) -> str:
    if not text:
        return ""
    return telegramify_markdown.markdownify(adapt_markdown_tables(text))


def adapt_markdown_tables(text: str) -> str:
    """Convert pipe tables to a mobile-friendly vertical list before Telegram escaping."""
    lines = text.splitlines()
    result: list[str] = []
    index = 0
    in_code_block = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith(("```", "~~~")):
            in_code_block = not in_code_block
            result.append(line)
            index += 1
            continue

        if (
            not in_code_block
            and index + 1 < len(lines)
            and _is_table_row(line)
            and _is_table_separator(lines[index + 1])
        ):
            table_lines = [line, lines[index + 1]]
            index += 2

            while index < len(lines) and _is_table_row(lines[index]):
                table_lines.append(lines[index])
                index += 1

            result.extend(_table_to_mobile_cards(table_lines))
            continue

        result.append(line)
        index += 1

    return "\n".join(result)


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and not stripped.startswith(">")


def _is_table_separator(line: str) -> bool:
    cells = _split_table_row(line)
    if len(cells) < 2:
        return False
    return all(_is_separator_cell(cell) for cell in cells)


def _is_separator_cell(cell: str) -> bool:
    value = cell.strip()
    if value.startswith(":"):
        value = value[1:]
    if value.endswith(":"):
        value = value[:-1]
    return len(value) >= 3 and set(value) == {"-"}


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False

    for char in stripped:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == "|":
            cells.append("".join(current).strip().replace("\\|", "|"))
            current = []
            continue
        current.append(char)

    cells.append("".join(current).strip().replace("\\|", "|"))
    return cells


def _table_to_mobile_cards(table_lines: list[str]) -> list[str]:
    headers = _split_table_row(table_lines[0])
    rows = [_split_table_row(line) for line in table_lines[2:]]
    cards: list[str] = []

    if len(headers) < 2 or not rows:
        return table_lines

    for row_number, row in enumerate(rows, start=1):
        normalized_row = (row + [""] * len(headers))[: len(headers)]
        title = normalized_row[0] or f"Строка {row_number}"
        title_header = headers[0] or "Пункт"

        if cards:
            cards.extend(["", TELEGRAM_SPACER_LINE, ""])

        cards.append(f"**{title_header}: {title}**")
        for header, value in zip(headers[1:], normalized_row[1:]):
            if not header and not value:
                continue
            label = header or "Значение"
            cards.append(f"- **{label}:** {value or '-'}")

    return cards


def split_message(text: str, chunk_size: int = 4000) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > chunk_size:
        split_at = remaining.rfind("\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = chunk_size

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n")

    if remaining:
        chunks.append(remaining)

    return chunks
