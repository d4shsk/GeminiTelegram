# GeminiTelegram project overview

## Purpose
A Telegram bot built on aiogram that routes user text and photos through multiple LLM providers. The bot supports at least two user modes (roleplay/fun and serious), model selection, chat history, image analysis, and optional web search/tool use.

## Tech stack
- Python
- aiogram for Telegram bot routing and polling
- google-genai for Gemini
- groq async client
- openai Python client pointed at GitHub Models inference endpoint
- aiohttp/httpx for HTTP integrations
- ddgs for web search
- telegramify-markdown for Telegram-friendly formatting

## Runtime/env
Main environment variables discovered in code:
- TELEGRAM_TOKEN
- GEMINI_API_KEY
- GROQ_API_KEY
- GITHUB_TOKEN
- CF_ACCOUNT_IDS / CF_ACCOUNT_ID
- CF_API_TOKENS / CF_API_TOKEN

## Rough codebase structure
- app.py: process entrypoint, runs src.bot.main.run via asyncio
- src/bot/main.py: creates Dispatcher, registers routers, starts polling
- src/bot/runtime.py: central runtime object, clients/tokens/provider rotation
- src/bot/config.py: model constants, prompts, menu labels, search tool config
- src/bot/state.py: in-memory user state/history/mode/model tracking
- src/bot/keyboards.py: Telegram keyboards and picker text
- src/bot/handlers/: Telegram command and message handlers
- src/bot/services/llm_text.py: text generation/provider fallback/tool-call flow
- src/bot/services/vision.py: image analysis across providers
- src/bot/services/search.py: web search helpers
- src/bot/services/formatter.py: Telegram formatting helpers
- test_gpt.py, test_groq.py, test_github.py: standalone manual/integration test scripts

## Notes
- No README was found in the repo root.
- No pyproject.toml, ruff, black, mypy, or pytest configuration was found during onboarding.
- .gitignore explicitly excludes the test_*.py files because they may contain keys/secrets.
- Project appears to target Windows development currently.