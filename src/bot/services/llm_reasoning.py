import json
from typing import Any

from ..config import MODEL_PRIORITY
from ..runtime import runtime
from ..state import state
from .llm_clients import ask_model, is_rate_limit_error, log_provider_cooldown

CRITIC_MODELS = (
    ("google", "gemini-2.5-flash"),
    ("cloudflare", "@cf/moonshotai/kimi-k2.6"),
)


def _build_plan_prompt(user_input: str) -> str:
    return (
        "Create a short internal answer plan with 3-6 concise bullets. "
        "Do not answer the user. Focus on checks, structure, and reasoning steps.\n\n"
        f"User request:\n{user_input}"
    )


def merge_user_prompt(user_input: str, plan: str | None, goal_contract: dict[str, Any] | None = None) -> str:
    sections = [f"User request:\n{user_input}"]

    if goal_contract:
        sections.append(
            "Practical reasoning guardrails:\n"
            f"- Goal: {goal_contract.get('goal', '')}\n"
            f"- Required objects: {', '.join(goal_contract.get('required_objects', [])) or 'none'}\n"
            f"- Must remain true: {', '.join(goal_contract.get('must_remain_true', [])) or 'none'}\n"
            f"- Invalid outcomes: {', '.join(goal_contract.get('invalid_outcomes', [])) or 'none'}\n"
            "- Internally verify that your advice still lets the user achieve the goal.\n"
            "- If one option breaks the goal, reject it and explain briefly why."
        )

    if plan:
        sections.append(f"Internal answer plan:\n{plan}")

    sections.append("Write only the final answer for the user. Do not expose the internal plan.")
    return "\n\n".join(sections)


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


async def build_answer_plan(
    provider: str,
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
) -> str | None:
    plan_response = await ask_model(
        provider,
        chat_id,
        model_id,
        _build_plan_prompt(user_input),
        system_instruction,
        use_search_tool=False,
        use_calculator_tool=False,
        temperature=0.1,
        max_tokens=220,
    )
    if not plan_response:
        return None
    plan_text = (plan_response["text"] or "").strip()
    return plan_text or None


async def extract_goal_contract(
    provider: str,
    chat_id: int,
    model_id: str,
    user_input: str,
    system_instruction: str,
) -> dict[str, Any] | None:
    response = await ask_model(
        provider,
        chat_id,
        model_id,
        _build_goal_prompt(user_input),
        system_instruction,
        use_search_tool=False,
        use_calculator_tool=False,
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


async def continue_incomplete_answer(
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
    continuation = await ask_model(
        provider,
        chat_id,
        model_id,
        continuation_prompt,
        system_instruction,
        use_search_tool=False,
        use_calculator_tool=False,
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


async def apply_goal_guard(
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
        'verdict: "pass" | "rewrite" | "fail"\n'
        "problem: string\n"
        "rewritten_answer: string\n\n"
        "Use verdict=fail when the draft breaks the goal and you cannot repair it safely.\n"
        "Use verdict=rewrite only when you can fix the answer directly.\n"
        "If the draft is already goal-consistent, use verdict=pass and keep rewritten_answer empty.\n\n"
        f"Goal contract:\n{json.dumps(goal_contract, ensure_ascii=False)}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Draft answer:\n{draft_answer}"
    )
    response = await ask_model(
        provider,
        chat_id,
        model_id,
        guard_prompt,
        system_instruction,
        use_search_tool=False,
        use_calculator_tool=False,
        temperature=0.1,
        max_tokens=700,
    )
    if not response or not response.get("text"):
        return None

    payload = _extract_json_object(str(response["text"]))
    if not isinstance(payload, dict):
        return None

    verdict = str(payload.get("verdict", "fail")).strip().lower()
    if verdict not in {"pass", "rewrite", "fail"}:
        verdict = "fail"

    rewritten_answer = str(payload.get("rewritten_answer", "")).strip()
    return {
        "verdict": verdict,
        "problem": str(payload.get("problem", "")).strip(),
        "rewritten_answer": rewritten_answer,
    }


def _critic_candidates(draft_model: str) -> list[tuple[str, str]]:
    model_provider = {item["model"]: item["provider"] for item in MODEL_PRIORITY}
    candidates: list[tuple[str, str]] = []

    if draft_model in model_provider and draft_model in {model_id for _, model_id in CRITIC_MODELS}:
        candidates.append((model_provider[draft_model], draft_model))

    for provider, model_id in CRITIC_MODELS:
        if (provider, model_id) not in candidates:
            candidates.append((provider, model_id))
    return candidates


async def _run_critic(
    chat_id: int,
    prompt: str,
    draft_model: str,
) -> dict[str, Any] | None:
    for provider, critic_model in _critic_candidates(draft_model):
        if not runtime.is_provider_available(provider):
            continue

        try:
            critic_response = await ask_model(
                provider,
                chat_id,
                critic_model,
                prompt,
                "You are a concise response critic. Output strict JSON only.",
                use_search_tool=False,
                use_calculator_tool=False,
                temperature=0.1,
                max_tokens=900,
            )
        except Exception as error:
            rate_limited = is_rate_limit_error(error)
            cooldown = runtime.mark_provider_failure(provider, rate_limited=rate_limited)
            log_provider_cooldown(provider, cooldown, f"critic failure on {critic_model}")
            continue

        if not critic_response or not critic_response.get("text"):
            continue

        payload = _extract_json_object(str(critic_response["text"]))
        if not isinstance(payload, dict):
            continue

        runtime.mark_provider_success(provider)
        payload["_checker_model"] = critic_model
        payload["_self_check"] = critic_model == draft_model
        return payload

    return None


async def review_practical_answer(
    chat_id: int,
    user_input: str,
    draft_answer: str,
    goal_contract: dict[str, Any],
    draft_model: str,
) -> dict[str, Any] | None:
    reviewer_prompt = (
        "Review the draft answer for practical reasoning correctness. Return strict JSON with keys:\n"
        'verdict: "pass" | "rewrite" | "fail"\n'
        "reason: string\n"
        "improved_answer: string\n"
        'confidence: "high" | "medium" | "low"\n\n'
        "Use fail when the draft breaks the user's goal or drops a required object.\n"
        "Use rewrite when the answer can be fixed directly.\n"
        "Use pass only when the answer clearly preserves the goal.\n\n"
        f"Goal contract:\n{json.dumps(goal_contract, ensure_ascii=False)}\n\n"
        f"User request:\n{user_input}\n\n"
        f"Draft answer:\n{draft_answer}"
    )
    payload = await _run_critic(chat_id, reviewer_prompt, draft_model)
    if not payload:
        return None

    verdict = str(payload.get("verdict", "fail")).strip().lower()
    if verdict not in {"pass", "rewrite", "fail"}:
        verdict = "fail"

    confidence = str(payload.get("confidence", "medium")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    return {
        "verdict": verdict,
        "reason": str(payload.get("reason", "")).strip(),
        "improved_answer": str(payload.get("improved_answer", "")).strip(),
        "confidence": confidence,
        "checker_model": str(payload.get("_checker_model", "")).strip(),
        "self_check": bool(payload.get("_self_check")),
    }


async def critic_and_refine(
    chat_id: int,
    user_input: str,
    draft_answer: str,
    profile: dict[str, Any],
    analysis: dict[str, Any],
    route_mode: str,
    draft_model: str,
) -> dict[str, Any] | None:
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

    payload = await _run_critic(chat_id, critic_prompt, draft_model)
    if not payload:
        return None

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
        "checker_model": str(payload.get("_checker_model", "")).strip(),
        "self_check": bool(payload.get("_self_check")),
    }
