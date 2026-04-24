import asyncio
import json
import re
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

MIXED_SCRIPT_RE = re.compile(r"\b(?=\w*[A-Za-zÀ-ÿ])(?=\w*[А-Яа-яЁё])[A-Za-zÀ-ÿА-Яа-яЁё]+\b")
SUSPICIOUS_GARBLED_CHARS = {"Ã", "Ð", "Ñ", "Õ", "õ", "ã", "č", "š", "ž", "�"}
COMPLEX_REQUEST_KEYWORDS = (
    "разбери",
    "архитектур",
    "пошагово",
    "подробно",
    "сравни",
    "сравнение",
    "обоснуй",
    "докажи",
    "проанализ",
    "analysis",
    "analyze",
    "compare",
    "tradeoff",
    "refactor",
    "architecture",
    "debug",
    "bug",
    "review",
    "code review",
    "оптимиз",
    "алгоритм",
    "алгоритма",
)
SEARCH_REQUEST_KEYWORDS = (
    "сегодня",
    "сейчас",
    "последн",
    "новост",
    "актуаль",
    "курс",
    "цена",
    "цены",
    "погода",
    "расписан",
    "score",
    "latest",
    "recent",
    "today",
    "current",
    "news",
    "price",
    "version",
    "release",
)
LOW_CONFIDENCE_MARKERS = (
    "возможно",
    "кажется",
    "не уверен",
    "могу ошибаться",
    "скорее всего",
    "не могу проверить",
    "maybe",
    "probably",
    "not sure",
    "i might be wrong",
)
TOPIC_KEYWORDS = {
    "coding": ("код", "python", "api", "debug", "bug", "refactor", "programming"),
    "telegram": ("telegram", "бот", "bot", "aiogram"),
    "llm": ("llm", "model", "модель", "prompt", "gemini", "gpt", "groq", "kimi"),
    "infra": ("railway", "docker", "deploy", "sqlite", "postgres", "redis", "server"),
}
STRONG_REASONING_MODELS = {"gemini-2.5-flash", "@cf/moonshotai/kimi-k2.6"}


def _resolve_worker_priority(chat_id: int, user_input: str) -> tuple[list[dict[str, Any]], str | None, bool]:
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

    return priority, requested_name, bool(selected)


def _is_complex_request(user_input: str) -> bool:
    text = user_input.strip()
    if not text:
        return False

    score = 0
    lower_text = text.lower()

    if len(text) >= 700:
        score += 3
    elif len(text) >= 350:
        score += 2
    elif len(text) >= 180:
        score += 1

    line_count = text.count("\n") + 1
    if line_count >= 10:
        score += 2
    elif line_count >= 5:
        score += 1

    if "```" in text:
        score += 3

    punctuation_count = sum(text.count(marker) for marker in ("?", ":", ";"))
    if punctuation_count >= 6:
        score += 1

    if re.search(r"\b\d+[.)]\s", text):
        score += 1

    if re.search(r"[{}[\]()<>_=/*\\]", text):
        score += 1

    keyword_hits = sum(1 for keyword in COMPLEX_REQUEST_KEYWORDS if keyword in lower_text)
    score += min(keyword_hits, 3)

    return score >= 3


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_profile_updates(chat_id: int, user_input: str) -> dict[str, Any]:
    profile = state.get_user_profile(chat_id)
    text = user_input.strip()
    lower_text = text.lower()
    updates: dict[str, Any] = {}

    if re.search(r"на русском|по-русски|in russian", lower_text):
        updates["language"] = "ru"
    elif re.search(r"на английском|in english", lower_text):
        updates["language"] = "en"
    else:
        cyr = len(re.findall(r"[А-Яа-яЁё]", text))
        lat = len(re.findall(r"[A-Za-z]", text))
        if cyr > lat * 2 and cyr >= 8:
            updates["language"] = "ru"
        elif lat > cyr * 2 and lat >= 8:
            updates["language"] = "en"

    if re.search(r"коротко|кратко|briefly|concise|без воды", lower_text):
        updates["verbosity"] = "short"
    elif re.search(r"подробно|детально|развернуто|in detail|thorough", lower_text):
        updates["verbosity"] = "detailed"

    if re.search(r"по пунктам|списком|bullet", lower_text):
        updates["format"] = "bullets"
    elif re.search(r"обычным текстом|сплошным текстом|plain prose", lower_text):
        updates["format"] = "prose"

    if re.search(r"сначала вывод|итог сначала|bottom line first|summary first", lower_text):
        updates["structure"] = "summary_first"

    if re.search(r"без таблиц|no tables", lower_text):
        updates["tables"] = "avoid"
    elif re.search(r"с таблиц|таблиц|table", lower_text):
        updates["tables"] = "allow"

    if re.search(r"с примерами|examples", lower_text):
        updates["examples"] = "prefer"
    elif re.search(r"без примеров|no examples", lower_text):
        updates["examples"] = "avoid"

    if re.search(r"без воды|по делу|direct", lower_text):
        updates["tone"] = "direct"

    topics = list(profile.get("topics", []))
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lower_text for keyword in keywords):
            topics.append(topic)
    if topics:
        updates["topics"] = _dedupe(topics)[:6]

    return updates


def _profile_to_prompt(profile: dict[str, Any]) -> str:
    if not profile:
        return ""

    lines: list[str] = ["Known user profile:"]
    if profile.get("language"):
        lines.append(f"- Preferred language: {profile['language']}")
    if profile.get("verbosity"):
        lines.append(f"- Preferred verbosity: {profile['verbosity']}")
    if profile.get("format"):
        lines.append(f"- Preferred format: {profile['format']}")
    if profile.get("structure"):
        lines.append(f"- Preferred structure: {profile['structure']}")
    if profile.get("tables"):
        lines.append(f"- Tables: {profile['tables']}")
    if profile.get("examples"):
        lines.append(f"- Examples: {profile['examples']}")
    if profile.get("tone"):
        lines.append(f"- Tone: {profile['tone']}")
    if profile.get("topics"):
        lines.append(f"- Frequent topics: {', '.join(profile['topics'])}")
    return "\n".join(lines)


def _analyze_request(user_input: str, mode: str, profile: dict[str, Any]) -> dict[str, Any]:
    lower_text = user_input.lower()
    is_complex = _is_complex_request(user_input)
    search_signal = any(keyword in lower_text for keyword in SEARCH_REQUEST_KEYWORDS)
    code_signal = "```" in user_input or bool(re.search(r"[{}[\]()<>_=/*\\]", user_input))

    needs_search = mode == MODE_SERIOUS and search_signal
    needs_plan = mode == MODE_SERIOUS and (is_complex or len(user_input) >= 220 or code_signal)
    needs_critic = mode == MODE_SERIOUS and (is_complex or needs_search or code_signal)
    confidence_risk = needs_search or is_complex or code_signal

    return {
        "is_complex": is_complex,
        "needs_search": needs_search,
        "needs_plan": needs_plan,
        "needs_critic": needs_critic,
        "confidence_risk": confidence_risk,
        "prefer_strong_model": mode == MODE_SERIOUS and (is_complex or needs_search),
        "profile_hint": profile.get("verbosity", ""),
    }


def _is_weak_reasoning_model(model_id: str) -> bool:
    return model_id not in STRONG_REASONING_MODELS


def _resolve_serious_priority(
    chat_id: int, user_input: str, analysis: dict[str, Any]
) -> tuple[list[dict[str, Any]], str | None, bool]:
    explicit_model_id = state.user_models.get(chat_id)
    manually_selected = explicit_model_id is not None

    if explicit_model_id:
        selected_model_id = explicit_model_id
    elif analysis["prefer_strong_model"]:
        selected_model_id = "gpt-4o"
        logger.info("Escalating request to gpt-4o due to complexity/search risk")
    else:
        selected_model_id = "gemini-2.5-flash-lite"

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

    return resolved, None, manually_selected


def _build_priority(
    chat_id: int, user_input: str, mode: str, analysis: dict[str, Any]
) -> tuple[list[dict[str, Any]], str | None, bool]:
    if mode == MODE_SERIOUS:
        return _resolve_serious_priority(chat_id, user_input, analysis)
    return _resolve_worker_priority(chat_id, user_input)


def _build_system_instruction(
    chat_id: int, mode: str, model_id: str, profile: dict[str, Any], analysis: dict[str, Any]
) -> str:
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
        base_instruction = SYSTEM_PROMPT.format(my_name=my_name, current_datetime=now)
    else:
        base_instruction = SERIOUS_SYSTEM_PROMPT.format(current_datetime=now)

    profile_prompt = _profile_to_prompt(profile)
    summary = state.get_summary(chat_id)
    extras: list[str] = []
    if profile_prompt:
        extras.append(profile_prompt)
    if summary:
        extras.append(
            "Earlier conversation summary:\n"
            f"{summary}"
        )
    if analysis["needs_search"]:
        extras.append(
            "If the answer may depend on recent or changing facts, verify before answering rather than guessing."
        )

    if not extras:
        return base_instruction
    return f"{base_instruction}\n\n" + "\n\n".join(extras)


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


def _is_rate_limit_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(
        marker in error_text
        for marker in ("429", "rate limit", "too many requests", "quota", "resource exhausted", "limit exceeded")
    )


def _looks_garbled_text(text: str) -> bool:
    if not text or not re.search(r"[А-Яа-яЁё]", text):
        return False
    if MIXED_SCRIPT_RE.search(text):
        return True
    suspicious_count = sum(text.count(char) for char in SUSPICIOUS_GARBLED_CHARS)
    return suspicious_count >= 2


def _estimate_confidence(user_input: str, text: str, analysis: dict[str, Any], search_enabled: bool) -> str:
    lower_text = text.lower()
    if any(marker in lower_text for marker in LOW_CONFIDENCE_MARKERS):
        return "low"
    if _looks_incomplete_answer(text):
        return "medium"
    if analysis["is_complex"] and len(text) < 260:
        return "low"
    if analysis["needs_search"] and not search_enabled:
        return "medium"
    if analysis["is_complex"] and len(text) < 450:
        return "medium"
    if len(text) < max(90, len(user_input) // 3):
        return "medium"
    return "high"


def _log_provider_cooldown(provider: str, cooldown_seconds: float, reason: str) -> None:
    if cooldown_seconds > 0:
        logger.warning("Provider %s cooldown for %.0fs after %s", provider, cooldown_seconds, reason)


def _looks_incomplete_answer(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False

    if stripped.endswith((":", "(", "[", "{", "-", "—")):
        return True

    last_line = stripped.splitlines()[-1].strip()
    if not last_line:
        return False

    if re.match(r"^[•⦁*-]\s+\S+", last_line) and not re.search(r"[.!?…»\])\"]$", last_line):
        return True

    if len(last_line) <= 18 and not re.search(r"[.!?…»\])\"]$", last_line):
        return True

    return False


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


def _build_plan_prompt(user_input: str) -> str:
    return (
        "Create a short internal answer plan with 3-6 concise bullets. "
        "Do not answer the user. Focus on checks, structure, and reasoning steps.\n\n"
        f"User request:\n{user_input}"
    )


def _merge_user_prompt(user_input: str, plan: str | None) -> str:
    if not plan:
        return user_input
    return (
        f"User request:\n{user_input}\n\n"
        f"Internal answer plan:\n{plan}\n\n"
        "Write only the final answer for the user. Do not expose the internal plan unless it is directly useful."
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _sanitize_profile_updates(raw_updates: Any) -> dict[str, Any]:
    if not isinstance(raw_updates, dict):
        return {}

    allowed_keys = {"language", "verbosity", "format", "structure", "tables", "examples", "tone", "topics"}
    sanitized: dict[str, Any] = {}
    for key, value in raw_updates.items():
        if key not in allowed_keys:
            continue
        if key == "topics" and isinstance(value, list):
            sanitized[key] = [str(item) for item in value[:6]]
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
    return sanitized


def _build_goal_prompt(user_input: str) -> str:
    return (
        "Extract the practical goal behind the user's request. Return strict JSON with keys:\n"
        "goal: string\n"
        "required_objects: array of strings\n"
        "must_remain_true: array of strings\n"
        "invalid_outcomes: array of strings\n\n"
        "Focus on what must still be possible after following the advice.\n\n"
        f"User request:\n{user_input}"
    )


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
            if _is_rate_limit_error(error):
                rate_limited = True
                runtime.rotate_cf_credentials()
                attempts += 1
                continue
            raise

    if rate_limited:
        raise RuntimeError("Cloudflare rate limit exhausted")
    return None


async def _ask_github(
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


async def _ask_model(
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
        return await _ask_github(
            chat_id,
            model_id,
            user_input,
            system_instruction,
            use_tools=use_tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return None


async def _build_answer_plan(
    provider: str,
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
) -> str | None:
    plan_response = await _ask_model(
        provider,
        chat_id,
        model_id,
        _build_plan_prompt(user_input),
        system_instruction,
        use_tools=False,
        temperature=0.1,
        max_tokens=220,
    )
    if not plan_response:
        return None
    plan_text = (plan_response["text"] or "").strip()
    return plan_text or None


async def _extract_goal_contract(
    provider: str,
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
) -> dict[str, Any] | None:
    response = await _ask_model(
        provider,
        chat_id,
        model_id,
        _build_goal_prompt(user_input),
        system_instruction,
        use_tools=False,
        temperature=0.1,
        max_tokens=260,
    )
    if not response or not response.get("text"):
        return None
    payload = _extract_json_object(str(response["text"]))
    if not isinstance(payload, dict):
        return None

    goal = str(payload.get("goal", "")).strip()
    required_objects = payload.get("required_objects", [])
    must_remain_true = payload.get("must_remain_true", [])
    invalid_outcomes = payload.get("invalid_outcomes", [])

    contract = {
        "goal": goal,
        "required_objects": [str(item).strip() for item in required_objects if str(item).strip()][:5],
        "must_remain_true": [str(item).strip() for item in must_remain_true if str(item).strip()][:5],
        "invalid_outcomes": [str(item).strip() for item in invalid_outcomes if str(item).strip()][:5],
    }
    if not contract["goal"]:
        return None
    return contract


async def _continue_incomplete_answer(
    provider: str,
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
    partial_answer: str,
) -> str | None:
    continuation_prompt = (
        "Continue the answer exactly from where it stopped. "
        "Do not restart, do not repeat earlier sentences, and finish the thought cleanly.\n\n"
        f"User request:\n{user_input}\n\n"
        f"Partial answer:\n{partial_answer}"
    )
    continuation = await _ask_model(
        provider,
        chat_id,
        model_id,
        continuation_prompt,
        system_instruction,
        use_tools=False,
        temperature=0.2,
        max_tokens=900,
    )
    if not continuation or not continuation.get("text"):
        return None
    extra = str(continuation["text"]).strip()
    if not extra:
        return None
    joiner = "" if partial_answer.endswith(("\n", " ")) else " "
    return partial_answer + joiner + extra


async def _apply_goal_guard(
    provider: str,
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
    draft_answer: str,
    goal_contract: dict[str, Any],
) -> dict[str, Any] | None:
    guard_prompt = (
        "Check whether the draft answer preserves the user's real-world goal. "
        "Return strict JSON with keys:\n"
        "is_consistent: boolean\n"
        "problem: string\n"
        "rewritten_answer: string\n\n"
        "If the draft is already goal-consistent, keep rewritten_answer empty.\n\n"
        f"Goal contract:\n{json.dumps(goal_contract, ensure_ascii=False)}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Draft answer:\n{draft_answer}"
    )
    response = await _ask_model(
        provider,
        chat_id,
        model_id,
        guard_prompt,
        system_instruction,
        use_tools=False,
        temperature=0.1,
        max_tokens=700,
    )
    if not response or not response.get("text"):
        return None

    payload = _extract_json_object(str(response["text"]))
    if not isinstance(payload, dict):
        return None

    rewritten_answer = str(payload.get("rewritten_answer", "")).strip()
    return {
        "is_consistent": bool(payload.get("is_consistent")),
        "problem": str(payload.get("problem", "")).strip(),
        "rewritten_answer": rewritten_answer,
    }


async def _critic_and_refine(
    chat_id: int,
    user_input: str,
    draft_answer: str,
    profile: dict[str, Any],
    analysis: dict[str, Any],
    route_mode: str,
    draft_model: str,
) -> dict[str, Any] | None:
    if not runtime.github_client or not runtime.is_provider_available("github"):
        return None

    critic_prompt = (
        "Review the draft answer. Return a strict JSON object with keys:\n"
        'confidence: "high" | "medium" | "low"\n'
        "needs_refine: boolean\n"
        "reason: string\n"
        "improved_answer: string\n"
        "profile_updates: object\n\n"
        "Only add profile_updates for stable user preferences explicitly implied by the user's wording.\n"
        "If the draft is already good, keep improved_answer empty.\n\n"
        f"Request mode: {route_mode}\n"
        f"Draft model: {draft_model}\n"
        f"Needs search: {analysis['needs_search']}\n"
        f"Known profile: {json.dumps(profile, ensure_ascii=False)}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Draft answer:\n{draft_answer}"
    )

    try:
        critic_response = await _ask_github(
            chat_id,
            "gpt-4o",
            critic_prompt,
            "You are a concise response critic. Output strict JSON only.",
            use_tools=False,
            temperature=0.1,
            max_tokens=900,
        )
    except Exception as error:
        rate_limited = _is_rate_limit_error(error)
        cooldown = runtime.mark_provider_failure("github", rate_limited=rate_limited)
        _log_provider_cooldown("github", cooldown, "critic failure")
        return None

    if not critic_response or not critic_response.get("text"):
        return None

    payload = _extract_json_object(critic_response["text"])
    if not payload:
        return None

    runtime.mark_provider_success("github")
    profile_updates = _sanitize_profile_updates(payload.get("profile_updates"))
    if profile_updates:
        state.update_user_profile(chat_id, profile_updates)

    confidence = str(payload.get("confidence", "medium")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    improved_answer = str(payload.get("improved_answer", "")).strip()
    return {
        "confidence": confidence,
        "needs_refine": bool(payload.get("needs_refine")),
        "reason": str(payload.get("reason", "")).strip(),
        "improved_answer": improved_answer,
        "profile_updates": profile_updates,
    }


def _build_response_mode_label(
    mode: str,
    manually_selected: bool,
    analysis: dict[str, Any],
    used_model: str,
    confidence: str,
    *,
    plan_used: bool,
    critic_used: bool,
    search_enabled: bool,
) -> str:
    route_mode = "вручную" if manually_selected else "авто"
    answer_type = "быстрый ответ"
    if search_enabled:
        answer_type = "с поиском"
    elif critic_used or confidence == "high":
        answer_type = "проверенный ответ"

    return json.dumps(
        {
            "route_mode": route_mode,
            "answer_type": answer_type,
            "model": used_model,
        },
        ensure_ascii=False,
    )


async def generate_text_reply(chat_id: int, user_input: str, mode: str) -> dict[str, str] | None:
    profile_updates = _extract_profile_updates(chat_id, user_input)
    profile = state.update_user_profile(chat_id, profile_updates) if profile_updates else state.get_user_profile(chat_id)
    analysis = _analyze_request(user_input, mode, profile)
    priority, requested_name, manually_selected = _build_priority(chat_id, user_input, mode, analysis)

    for model_info in priority:
        provider = model_info["provider"]
        model_id = model_info["model"]

        if not runtime.is_provider_available(provider):
            logger.info(
                "Skipping %s (%s) because provider cooldown still has %.0fs left",
                model_id,
                provider,
                runtime.provider_cooldown_remaining(provider),
            )
            continue

        system_instruction = _build_system_instruction(chat_id, mode, model_id, profile, analysis)
        max_tokens = 2600 if analysis["is_complex"] or manually_selected else 1600
        temperature = 0.2 if mode == MODE_SERIOUS else 0.5
        plan_text: str | None = None
        plan_used = False
        goal_contract: dict[str, Any] | None = None

        try:
            if analysis["needs_plan"]:
                try:
                    plan_text = await _build_answer_plan(provider, chat_id, model_id, user_input, system_instruction)
                except Exception as error:
                    logger.info("Planning step skipped for %s: %s", model_id, str(error)[:100])
                if plan_text:
                    plan_used = True

            if mode == MODE_SERIOUS and _is_weak_reasoning_model(model_id):
                try:
                    goal_contract = await _extract_goal_contract(
                        provider,
                        chat_id,
                        model_id,
                        user_input,
                        system_instruction,
                    )
                except Exception as error:
                    logger.info("Goal extraction skipped for %s: %s", model_id, str(error)[:100])

            response = await _ask_model(
                provider,
                chat_id,
                model_id,
                _merge_user_prompt(user_input, plan_text),
                system_instruction,
                use_tools=analysis["needs_search"],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not response or not response.get("text"):
                continue

            text = str(response["text"]).strip()
            search_enabled = bool(response.get("search_enabled"))
            if _looks_garbled_text(text):
                cooldown = runtime.mark_provider_failure(provider)
                logger.warning("Discarding garbled response from %s", model_id)
                _log_provider_cooldown(provider, cooldown, "garbled output")
                continue

            if _looks_incomplete_answer(text):
                continued_text = await _continue_incomplete_answer(
                    provider,
                    chat_id,
                    model_id,
                    user_input,
                    system_instruction,
                    text,
                )
                if continued_text and not _looks_garbled_text(continued_text):
                    text = continued_text

            if goal_contract:
                try:
                    goal_guard = await _apply_goal_guard(
                        provider,
                        chat_id,
                        model_id,
                        user_input,
                        system_instruction,
                        text,
                        goal_contract,
                    )
                except Exception as error:
                    logger.info("Goal guard skipped for %s: %s", model_id, str(error)[:100])
                    goal_guard = None
                if goal_guard and not goal_guard["is_consistent"]:
                    rewritten_answer = goal_guard["rewritten_answer"]
                    if rewritten_answer and not _looks_garbled_text(rewritten_answer):
                        logger.info("Rewriting goal-inconsistent answer from %s", model_id)
                        text = rewritten_answer
                    else:
                        logger.info("Escalating from %s due to goal inconsistency", model_id)
                        continue

            confidence = _estimate_confidence(user_input, text, analysis, search_enabled)
            if confidence == "low" and provider != "github" and not manually_selected:
                logger.info("Escalating from %s due to low draft confidence", model_id)
                continue

            final_text = text
            critic_used = False
            final_model = model_id
            critic_result: dict[str, Any] | None = None

            if analysis["needs_critic"]:
                critic_result = await _critic_and_refine(
                    chat_id,
                    user_input,
                    final_text,
                    profile,
                    analysis,
                    "manual" if manually_selected else "auto",
                    model_id,
                )
                if critic_result:
                    confidence = critic_result["confidence"]
                    improved_answer = critic_result["improved_answer"]
                    if improved_answer:
                        final_text = improved_answer
                        critic_used = True
                        final_model = f"{model_id} -> gpt-4o critic"

            runtime.mark_provider_success(provider)
            response_mode = _build_response_mode_label(
                mode,
                manually_selected,
                analysis,
                final_model,
                confidence,
                plan_used=plan_used,
                critic_used=critic_used,
                search_enabled=search_enabled,
            )

            return {
                "text": final_text,
                "used_model": final_model,
                "save_text": final_text,
                "requested_name": requested_name or "",
                "response_mode": response_mode,
                "confidence": confidence,
            }
        except asyncio.TimeoutError:
            cooldown = runtime.mark_provider_failure(provider)
            logger.warning("Timeout on model %s", model_id)
            _log_provider_cooldown(provider, cooldown, "timeout")
            continue
        except Exception as error:
            rate_limited = _is_rate_limit_error(error)
            cooldown = runtime.mark_provider_failure(provider, rate_limited=rate_limited)
            logger.warning("Error on %s (%s): %s", model_id, provider, str(error)[:120])
            _log_provider_cooldown(provider, cooldown, "rate limit" if rate_limited else "repeated failures")
            continue

    return None
