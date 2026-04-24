# Style and conventions

## General code style
- Python code uses type hints on functions and methods (`-> None`, tuple unions, etc.).
- Async-first style: handlers and service entrypoints are `async def` where they perform I/O.
- Imports are grouped with stdlib first, then third-party, then local relative imports.
- Module layout is simple and direct; there is little abstraction beyond handlers/services/runtime/state separation.
- Naming is mostly snake_case for functions/variables and PascalCase for classes.
- Constants are uppercase and concentrated in `src/bot/config.py`.

## Project-specific patterns
- Telegram routes live under `src/bot/handlers` and expose a module-level `router`.
- External provider clients/tokens are initialized centrally in `AppRuntime` (`src/bot/runtime.py`).
- User interaction state is stored in an in-memory shared state object.
- Provider fallback/model priority logic is centralized in `src/bot/services/llm_text.py`.
- Russian user-facing strings are common in bot messages and log/error text, so preserve existing language unless intentionally changing UX copy.

## Conventions to preserve
- Prefer small handlers delegating real work to services.
- Add new provider/runtime config through `runtime.py` and `config.py`, not ad hoc in handlers.
- Keep Telegram-specific formatting concerns out of core business logic when possible.
- Avoid introducing heavy framework/config churn unless the repo first adopts dedicated tooling.