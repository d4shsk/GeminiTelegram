import asyncio
import json
from typing import Any

from google import genai
import openai as openai_lib

from ..config import SEARCH_TOOLS
from ..runtime import logger, runtime
from ..state import state
from ..utils import moscow_datetime
from .search import perform_web_search


def _universal_history(chat_id: int) -> list[dict[str, str]]:
    return state.get_history(chat_id)


def _to_assistant_role(role: str) -> str:
    return "assistant" if role == "model" else role


def _openai_history(chat_id: int, system_instruction: str) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [
        {"role": "system", "content": system_instruction or f"Current Moscow datetime: {moscow_datetime()}"}
    ]
    for message in _universal_history(chat_id):
        history.append({"role": _to_assistant_role(message["role"]), "content": message["text"]})
    return history


def _gemini_history(chat_id: int) -> list[dict[str, Any]]:
    return [{"role": msg["role"], "parts": [{"text": msg["text"]}]} for msg in _universal_history(chat_id)]


def is_rate_limit_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(
        marker in error_text
        for marker in ("429", "rate limit", "too many requests", "quota", "resource exhausted", "limit exceeded")
    )


def log_provider_cooldown(provider: str, cooldown_seconds: float, reason: str) -> None:
    if cooldown_seconds > 0:
        logger.warning("Provider %s cooldown for %.0fs after %s", provider, cooldown_seconds, reason)


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


async def _apply_search_tools(history: list[dict[str, Any]], tool_calls: Any) -> bool:
    used_search = False
    for tool_call in tool_calls:
        if tool_call.function.name != "search_internet":
            continue
        used_search = True
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
    return used_search


async def _ask_google(
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
    *,
    use_tools: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any] | None:
    if not runtime.gemini_client:
        return None

    history = _gemini_history(chat_id)
    if "gemma" in model_id.lower():
        chat = runtime.gemini_client.aio.chats.create(model=model_id, history=history)
        current_input = user_input
        if system_instruction:
            current_input = f"[{system_instruction}]\n\nUser message: {user_input}"
        response = await asyncio.wait_for(chat.send_message(current_input), timeout=15.0)
        text = (response.text or "").strip()
        return {"text": text or None, "search_enabled": False}

    config_data: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if use_tools:
        config_data["tools"] = [{"google_search": {}}]
    if system_instruction:
        config_data["system_instruction"] = system_instruction
    config = genai.types.GenerateContentConfig(**config_data)
    chat = runtime.gemini_client.aio.chats.create(model=model_id, config=config, history=history)
    response = await asyncio.wait_for(chat.send_message(user_input), timeout=15.0)
    text = (response.text or "").strip()
    return {"text": text or None, "search_enabled": use_tools}


async def _ask_groq(
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
    *,
    use_tools: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any] | None:
    if not runtime.groq_client:
        return None

    history = _openai_history(chat_id, system_instruction)
    history.append({"role": "user", "content": user_input})

    try:
        request_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": history,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_tools:
            request_kwargs["tools"] = SEARCH_TOOLS
            request_kwargs["tool_choice"] = "auto"
        response = await asyncio.wait_for(
            runtime.groq_client.chat.completions.create(**request_kwargs),
            timeout=15.0,
        )
        response_message = response.choices[0].message
        used_search = False
        if use_tools and response_message.tool_calls:
            history.append(
                {
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": _normalize_tool_calls(response_message.tool_calls),
                }
            )
            used_search = await _apply_search_tools(history, response_message.tool_calls)
            response = await asyncio.wait_for(
                runtime.groq_client.chat.completions.create(
                    model=model_id,
                    messages=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=15.0,
            )
            response_message = response.choices[0].message
        text = (response_message.content or "").strip()
        return {"text": text or None, "search_enabled": used_search}
    except Exception as error:
        if "tool_use_failed" not in str(error):
            raise
        fallback_response = await asyncio.wait_for(
            runtime.groq_client.chat.completions.create(
                model=model_id,
                messages=history,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=15.0,
        )
        fallback_message = fallback_response.choices[0].message
        text = (fallback_message.content or "").strip()
        return {"text": text or None, "search_enabled": False}


async def _ask_cloudflare(
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
    *,
    use_tools: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any] | None:
    if not runtime.cf_accounts or not runtime.cf_tokens:
        return None

    history = _openai_history(chat_id, system_instruction)
    history.append({"role": "user", "content": user_input})

    attempts = 0
    rate_limited = False
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
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=60.0,
            )
            text = (response.choices[0].message.content or "").strip()
            return {"text": text or None, "search_enabled": False}
        except Exception as error:
            if is_rate_limit_error(error):
                rate_limited = True
                runtime.rotate_cf_credentials()
                attempts += 1
                continue
            raise

    if rate_limited:
        raise RuntimeError("Cloudflare rate limit exhausted")
    return None


async def ask_github(
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
    *,
    use_tools: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any] | None:
    if not runtime.github_client:
        return None

    history = _openai_history(chat_id, system_instruction)
    history.append({"role": "user", "content": user_input})

    request_kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": history,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if use_tools:
        request_kwargs["tools"] = SEARCH_TOOLS
        request_kwargs["tool_choice"] = "auto"

    response = await asyncio.wait_for(
        runtime.github_client.chat.completions.create(**request_kwargs),
        timeout=30.0,
    )
    message = response.choices[0].message
    used_search = False
    if use_tools and message.tool_calls:
        history.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": _normalize_tool_calls(message.tool_calls),
            }
        )
        used_search = await _apply_search_tools(history, message.tool_calls)
        response = await asyncio.wait_for(
            runtime.github_client.chat.completions.create(
                model=model_id,
                messages=history,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=30.0,
        )
        message = response.choices[0].message
    text = (message.content or "").strip()
    return {"text": text or None, "search_enabled": used_search}


async def ask_model(
    provider: str,
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
    *,
    use_tools: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any] | None:
    if provider == "google":
        return await _ask_google(
            chat_id,
            model_id,
            user_input,
            system_instruction,
            use_tools=use_tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "groq":
        return await _ask_groq(
            chat_id,
            model_id,
            user_input,
            system_instruction,
            use_tools=use_tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "cloudflare":
        return await _ask_cloudflare(
            chat_id,
            model_id,
            user_input,
            system_instruction,
            use_tools=use_tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "github":
        return await ask_github(
            chat_id,
            model_id,
            user_input,
            system_instruction,
            use_tools=use_tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return None
