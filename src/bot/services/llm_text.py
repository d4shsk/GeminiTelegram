import asyncio
import json
from typing import Any

from google import genai
import openai as openai_lib

from ..config import (
    MODEL_PRIORITY,
    MODE_SERIOUS,
    SEARCH_TOOLS,
    SERIOUS_FALLBACK_ORDER,
    SERIOUS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from ..runtime import logger, runtime
from ..state import state
from ..utils import moscow_datetime
from .search import perform_web_search


def _resolve_worker_priority(chat_id: int, user_input: str) -> tuple[list[dict[str, Any]], str | None]:
    priority = [model for model in MODEL_PRIORITY if not model.get("serious_only")]
    requested_name = None
    text_lower = user_input.lower()

    if "гемм" in text_lower or "gemma" in text_lower:
        state.active_models[chat_id] = "gemma"
    elif "лам" in text_lower or "llam" in text_lower:
        state.active_models[chat_id] = "llama"
    elif "гемин" in text_lower or "gemin" in text_lower:
        state.active_models[chat_id] = "gemini"

    selected = state.active_models.get(chat_id)
    if selected == "gemma":
        requested_name = "Gemma"
        priority.sort(key=lambda item: "gemma" not in item["model"].lower())
    elif selected == "llama":
        requested_name = "Llama"
        priority.sort(key=lambda item: "llama" not in item["model"].lower())
    elif selected == "gemini":
        requested_name = "Gemini"
        priority.sort(key=lambda item: "gemini" not in item["model"].lower())

    return priority, requested_name


def _resolve_serious_priority(chat_id: int) -> list[dict[str, Any]]:
    selected_model_id = state.user_models.get(chat_id, "gpt-4o")
    resolved: list[dict[str, Any]] = []
    added = set()

    for model in MODEL_PRIORITY:
        if model["model"] == selected_model_id:
            resolved.append(model)
            added.add(model["model"])
            break

    for fallback_model_id in SERIOUS_FALLBACK_ORDER:
        if fallback_model_id in added:
            continue
        for model in MODEL_PRIORITY:
            if model["model"] == fallback_model_id:
                resolved.append(model)
                added.add(model["model"])
                break

    for model in MODEL_PRIORITY:
        if model["model"] not in added:
            resolved.append(model)
            added.add(model["model"])

    return resolved


def _build_priority(chat_id: int, user_input: str, mode: str) -> tuple[list[dict[str, Any]], str | None]:
    if mode == MODE_SERIOUS:
        return _resolve_serious_priority(chat_id), None
    return _resolve_worker_priority(chat_id, user_input)


def _build_system_instruction(mode: str, model_id: str) -> str:
    now = moscow_datetime()
    if mode != MODE_SERIOUS:
        if "gemma" in model_id.lower():
            my_name = "Gemma"
        elif "llama" in model_id.lower():
            my_name = "Llama"
        elif "gemini" in model_id.lower():
            my_name = "Gemini"
        else:
            my_name = model_id
        return SYSTEM_PROMPT.format(my_name=my_name, current_datetime=now)
    return SERIOUS_SYSTEM_PROMPT.format(current_datetime=now)


def _universal_history(chat_id: int) -> list[dict[str, str]]:
    return state.sessions.get(chat_id, [])


def _to_assistant_role(role: str) -> str:
    return "assistant" if role == "model" else role


async def _ask_google(chat_id: int, model_id: str, user_input: str, system_instruction: str) -> str | None:
    if not runtime.gemini_client:
        return None

    history = [{"role": msg["role"], "parts": [{"text": msg["text"]}]} for msg in _universal_history(chat_id)]
    if "gemma" in model_id.lower():
        chat = runtime.gemini_client.aio.chats.create(model=model_id, history=history)
        current_input = user_input
        if system_instruction:
            current_input = f"[{system_instruction}]\n\nСообщение пользователя: {user_input}"
        response = await asyncio.wait_for(chat.send_message(current_input), timeout=15.0)
    else:
        config_data: dict[str, Any] = {"tools": [{"google_search": {}}]}
        if system_instruction:
            config_data["system_instruction"] = system_instruction
        config = genai.types.GenerateContentConfig(**config_data)
        chat = runtime.gemini_client.aio.chats.create(model=model_id, config=config, history=history)
        response = await asyncio.wait_for(chat.send_message(user_input), timeout=15.0)

    text = (response.text or "").strip()
    return text or None


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    normalized = []
    for tool_call in tool_calls:
        normalized.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    return normalized


async def _apply_search_tools(history: list[dict[str, Any]], tool_calls: Any) -> None:
    for tool_call in tool_calls:
        if tool_call.function.name != "search_internet":
            continue
        try:
            args = json.loads(tool_call.function.arguments)
            search_query = args.get("query", "")
        except Exception:
            search_query = ""
        logger.info("Tool call search_internet: %s", search_query)
        result = await perform_web_search(search_query)
        history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": result,
            }
        )


async def _ask_groq(chat_id: int, model_id: str, user_input: str, system_instruction: str) -> str | None:
    if not runtime.groq_client:
        return None

    history: list[dict[str, Any]] = []
    if system_instruction:
        history.append({"role": "system", "content": system_instruction})
    else:
        history.append({"role": "system", "content": f"Текущие дата и время (МСК): {moscow_datetime()}"})

    for message in _universal_history(chat_id):
        history.append({"role": _to_assistant_role(message["role"]), "content": message["text"]})
    history.append({"role": "user", "content": user_input})

    try:
        response = await asyncio.wait_for(
            runtime.groq_client.chat.completions.create(
                model=model_id,
                messages=history,
                tools=SEARCH_TOOLS,
                tool_choice="auto",
            ),
            timeout=15.0,
        )
        response_message = response.choices[0].message
        if response_message.tool_calls:
            history.append(
                {
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": _normalize_tool_calls(response_message.tool_calls),
                }
            )
            await _apply_search_tools(history, response_message.tool_calls)
            response = await asyncio.wait_for(
                runtime.groq_client.chat.completions.create(model=model_id, messages=history),
                timeout=15.0,
            )
            response_message = response.choices[0].message
        text = (response_message.content or "").strip()
        return text or None
    except Exception as error:
        if "tool_use_failed" not in str(error):
            raise
        fallback_response = await asyncio.wait_for(
            runtime.groq_client.chat.completions.create(model=model_id, messages=history),
            timeout=15.0,
        )
        fallback_message = fallback_response.choices[0].message
        text = (fallback_message.content or "").strip()
        return text or None


async def _ask_cloudflare(chat_id: int, model_id: str, user_input: str, system_instruction: str) -> str | None:
    if not runtime.cf_accounts or not runtime.cf_tokens:
        return None

    history: list[dict[str, str]] = []
    history.append(
        {"role": "system", "content": system_instruction or f"Текущие дата и время (МСК): {moscow_datetime()}"}
    )
    for message in _universal_history(chat_id):
        history.append({"role": _to_assistant_role(message["role"]), "content": message["text"]})
    history.append({"role": "user", "content": user_input})

    attempts = 0
    while attempts < len(runtime.cf_tokens):
        account_id, api_token = runtime.get_current_cf_credentials()
        if not account_id or not api_token:
            return None

        cf_client = openai_lib.AsyncOpenAI(
            api_key=api_token,
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        )
        try:
            response = await asyncio.wait_for(
                cf_client.chat.completions.create(
                    model=model_id,
                    messages=history,
                    max_tokens=2048,
                ),
                timeout=60.0,
            )
            text = (response.choices[0].message.content or "").strip()
            return text or None
        except Exception as error:
            error_text = str(error).lower()
            if "429" in error_text or "limit" in error_text or "quota" in error_text:
                runtime.rotate_cf_credentials()
                attempts += 1
                continue
            raise
    return None


async def _ask_github(chat_id: int, model_id: str, user_input: str, system_instruction: str) -> str | None:
    if not runtime.github_client:
        return None

    history: list[dict[str, Any]] = [
        {"role": "system", "content": system_instruction or f"Текущие дата и время (МСК): {moscow_datetime()}"}
    ]
    for message in _universal_history(chat_id):
        history.append({"role": _to_assistant_role(message["role"]), "content": message["text"]})
    history.append({"role": "user", "content": user_input})

    response = await asyncio.wait_for(
        runtime.github_client.chat.completions.create(
            model=model_id,
            messages=history,
            tools=SEARCH_TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.3,
        ),
        timeout=30.0,
    )
    message = response.choices[0].message
    if message.tool_calls:
        history.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": _normalize_tool_calls(message.tool_calls),
            }
        )
        await _apply_search_tools(history, message.tool_calls)
        response = await asyncio.wait_for(
            runtime.github_client.chat.completions.create(
                model=model_id,
                messages=history,
                max_tokens=1024,
                temperature=0.3,
            ),
            timeout=30.0,
        )
        message = response.choices[0].message
    text = (message.content or "").strip()
    return text or None


async def generate_text_reply(chat_id: int, user_input: str, mode: str) -> dict[str, str] | None:
    priority, requested_name = _build_priority(chat_id, user_input, mode)

    for model_info in priority:
        provider = model_info["provider"]
        model_id = model_info["model"]
        system_instruction = _build_system_instruction(mode, model_id)

        try:
            text = None
            if provider == "google":
                text = await _ask_google(chat_id, model_id, user_input, system_instruction)
            elif provider == "groq":
                text = await _ask_groq(chat_id, model_id, user_input, system_instruction)
            elif provider == "cloudflare":
                text = await _ask_cloudflare(chat_id, model_id, user_input, system_instruction)
            elif provider == "github":
                text = await _ask_github(chat_id, model_id, user_input, system_instruction)

            if text:
                return {
                    "text": text,
                    "used_model": model_id,
                    "save_text": text,
                    "requested_name": requested_name or "",
                }
        except asyncio.TimeoutError:
            logger.warning("Таймаут на модели %s", model_id)
            continue
        except Exception as error:
            logger.warning("Ошибка на %s (%s): %s", model_id, provider, str(error)[:120])
            continue

    return None
