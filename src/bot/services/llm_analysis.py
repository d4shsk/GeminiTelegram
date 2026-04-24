import json
import re
from typing import Any

from ..config import MODEL_PRIORITY, MODE_SERIOUS, SERIOUS_FALLBACK_ORDER, SERIOUS_SYSTEM_PROMPT, SYSTEM_PROMPT
from ..runtime import logger
from ..state import state
from ..utils import moscow_datetime

MIXED_SCRIPT_RE = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[\u0400-\u04FF])[A-Za-z\u0400-\u04FF]+\b")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
SUSPICIOUS_GARBLED_CHARS = {"Гѓ", "Гђ", "Г‘", "Г•", "Гµ", "ГЈ", "ДЌ", "ЕЎ", "Еѕ", "пїЅ"}
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
PRACTICAL_REASONING_KEYWORDS = (
    "лучше",
    "стоит ли",
    "что выбрать",
    "как поступить",
    "что делать",
    "пойти или",
    "поехать или",
    "идти или",
    "брать или",
    "нужно ли",
    "do i",
    "should i",
    "which is better",
    "better to",
    "go or",
    "walk or",
    "drive or",
)


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


def extract_profile_updates(chat_id: int, user_input: str) -> dict[str, Any]:
    profile = state.get_user_profile(chat_id)
    text = user_input.strip()
    lower_text = text.lower()
    updates: dict[str, Any] = {}

    if re.search(r"на русском|по-русски|in russian", lower_text):
        updates["language"] = "ru"
    elif re.search(r"на английском|in english", lower_text):
        updates["language"] = "en"
    else:
        cyr = len(re.findall(r"[\u0400-\u04FF]", text))
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


def analyze_request(user_input: str, mode: str, profile: dict[str, Any]) -> dict[str, Any]:
    lower_text = user_input.lower()
    is_complex = _is_complex_request(user_input)
    search_signal = any(keyword in lower_text for keyword in SEARCH_REQUEST_KEYWORDS)
    code_signal = "```" in user_input or bool(re.search(r"[{}[\]()<>_=/*\\]", user_input))
    practical_reasoning = needs_goal_guard(user_input)

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
        "practical_reasoning": practical_reasoning,
    }


def is_weak_reasoning_model(model_id: str) -> bool:
    return model_id not in STRONG_REASONING_MODELS


def needs_goal_guard(user_input: str) -> bool:
    text = user_input.strip().lower()
    if not text:
        return False
    if re.search(r"\b\d+\s*[+\-*/]\s*\d+\b", text):
        return False
    if len(text) < 20:
        return False

    keyword_match = any(keyword in text for keyword in PRACTICAL_REASONING_KEYWORDS)
    choice_pattern = bool(re.search(r"\bили\b", text) or re.search(r"\bor\b", text))
    action_pattern = bool(
        re.search(
            r"пойти|поехать|идти|ехать|взять|оставить|купить|продать|нести|везти|дойти|walk|drive|go|take|leave",
            text,
        )
    )
    return keyword_match or (choice_pattern and action_pattern)


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


def build_priority(
    chat_id: int, user_input: str, mode: str, analysis: dict[str, Any]
) -> tuple[list[dict[str, Any]], str | None, bool]:
    if mode == MODE_SERIOUS:
        return _resolve_serious_priority(chat_id, user_input, analysis)
    return _resolve_worker_priority(chat_id, user_input)


def build_system_instruction(
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
        extras.append(f"Earlier conversation summary:\n{summary}")
    if analysis["needs_search"]:
        extras.append("If the answer may depend on recent or changing facts, verify before answering rather than guessing.")

    if not extras:
        return base_instruction
    return f"{base_instruction}\n\n" + "\n\n".join(extras)


def looks_garbled_text(text: str) -> bool:
    if not text or not CYRILLIC_RE.search(text):
        return False
    if MIXED_SCRIPT_RE.search(text):
        return True
    suspicious_count = sum(text.count(char) for char in SUSPICIOUS_GARBLED_CHARS)
    return suspicious_count >= 2


def looks_incomplete_answer(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False

    if stripped.endswith((":", "(", "[", "{", "-", "—")):
        return True

    last_line = stripped.splitlines()[-1].strip()
    if not last_line:
        return False

    if re.match(r"^[•▪*-]\s+\S+", last_line) and not re.search(r"[.!?…»\])\"]$", last_line):
        return True

    if len(last_line) <= 18 and not re.search(r"[.!?…»\])\"]$", last_line):
        return True

    return False


def estimate_confidence(user_input: str, text: str, analysis: dict[str, Any], search_enabled: bool) -> str:
    lower_text = text.lower()
    if any(marker in lower_text for marker in LOW_CONFIDENCE_MARKERS):
        return "low"
    if looks_incomplete_answer(text):
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


def _normalize_for_comparison(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_duplicate_or_echo(original: str, candidate: str) -> bool:
    normalized_original = _normalize_for_comparison(original)
    normalized_candidate = _normalize_for_comparison(candidate)
    if not normalized_original or not normalized_candidate:
        return False
    if normalized_original == normalized_candidate:
        return True
    if normalized_candidate == normalized_original * 2:
        return True
    return normalized_candidate.startswith(normalized_original) and normalized_candidate.endswith(normalized_original)


def build_response_mode_label(
    mode: str,
    manually_selected: bool,
    analysis: dict[str, Any],
    used_model: str,
    confidence: str,
    *,
    plan_used: bool,
    critic_used: bool,
    search_enabled: bool,
    checker_model: str = "",
    self_check: bool = False,
) -> str:
    route_mode = "вручную" if manually_selected else "авто"
    answer_type = "быстрый ответ"
    if search_enabled:
        answer_type = "с поиском"
    elif critic_used:
        answer_type = "самопроверка" if self_check else f"проверено: {checker_model}"

    return json.dumps(
        {
            "route_mode": route_mode,
            "answer_type": answer_type,
            "model": used_model,
            "checker_model": checker_model,
            "self_check": self_check,
            "confidence": confidence,
            "plan_used": plan_used,
            "needs_critic": analysis.get("needs_critic", False),
        },
        ensure_ascii=False,
    )
