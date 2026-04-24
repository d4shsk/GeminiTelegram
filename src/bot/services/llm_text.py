import asyncio

from ..config import MODE_SERIOUS
from ..runtime import logger, runtime
from ..state import state
from .llm_analysis import (
    analyze_request,
    build_priority,
    build_response_mode_label,
    build_system_instruction,
    estimate_confidence,
    extract_profile_updates,
    is_duplicate_or_echo,
    is_weak_reasoning_model,
    looks_garbled_text,
    looks_incomplete_answer,
)
from .llm_clients import ask_model, is_rate_limit_error, log_provider_cooldown
from .llm_reasoning import (
    apply_goal_guard,
    build_answer_plan,
    continue_incomplete_answer,
    critic_and_refine,
    extract_goal_contract,
    merge_user_prompt,
)


async def generate_text_reply(chat_id: int, user_input: str, mode: str) -> dict[str, str] | None:
    profile_updates = extract_profile_updates(chat_id, user_input)
    profile = state.update_user_profile(chat_id, profile_updates) if profile_updates else state.get_user_profile(chat_id)
    analysis = analyze_request(user_input, mode, profile)
    priority, requested_name, manually_selected = build_priority(chat_id, user_input, mode, analysis)

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

        system_instruction = build_system_instruction(chat_id, mode, model_id, profile, analysis)
        max_tokens = 2600 if analysis["is_complex"] or manually_selected else 1600
        temperature = 0.2 if mode == MODE_SERIOUS else 0.5
        plan_text: str | None = None
        plan_used = False
        goal_contract: dict[str, object] | None = None

        try:
            if analysis["needs_plan"]:
                try:
                    plan_text = await build_answer_plan(provider, chat_id, model_id, user_input, system_instruction)
                except Exception as error:
                    logger.info("Planning step skipped for %s: %s", model_id, str(error)[:100])
                if plan_text:
                    plan_used = True

            if mode == MODE_SERIOUS and analysis["practical_reasoning"] and is_weak_reasoning_model(model_id):
                try:
                    goal_contract = await extract_goal_contract(
                        provider,
                        chat_id,
                        model_id,
                        user_input,
                        system_instruction,
                    )
                except Exception as error:
                    logger.info("Goal extraction skipped for %s: %s", model_id, str(error)[:100])

            response = await ask_model(
                provider,
                chat_id,
                model_id,
                merge_user_prompt(user_input, plan_text),
                system_instruction,
                use_tools=analysis["needs_search"],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not response or not response.get("text"):
                continue

            text = str(response["text"]).strip()
            search_enabled = bool(response.get("search_enabled"))
            if looks_garbled_text(text):
                cooldown = runtime.mark_provider_failure(provider)
                logger.warning("Discarding garbled response from %s", model_id)
                log_provider_cooldown(provider, cooldown, "garbled output")
                continue

            if looks_incomplete_answer(text):
                continued_text = await continue_incomplete_answer(
                    provider,
                    chat_id,
                    model_id,
                    user_input,
                    system_instruction,
                    text,
                )
                if continued_text and not looks_garbled_text(continued_text):
                    text = continued_text

            if goal_contract:
                try:
                    goal_guard = await apply_goal_guard(
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
                    rewritten_answer = str(goal_guard["rewritten_answer"]).strip()
                    if (
                        rewritten_answer
                        and not looks_garbled_text(rewritten_answer)
                        and not is_duplicate_or_echo(text, rewritten_answer)
                    ):
                        logger.info("Rewriting goal-inconsistent answer from %s", model_id)
                        text = rewritten_answer
                    else:
                        logger.info("Escalating from %s due to goal inconsistency", model_id)
                        continue

            confidence = estimate_confidence(user_input, text, analysis, search_enabled)
            if confidence == "low" and provider != "github" and not manually_selected:
                logger.info("Escalating from %s due to low draft confidence", model_id)
                continue

            final_text = text
            critic_used = False
            final_model = model_id

            if analysis["needs_critic"]:
                critic_result = await critic_and_refine(
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
            response_mode = build_response_mode_label(
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
            log_provider_cooldown(provider, cooldown, "timeout")
            continue
        except Exception as error:
            rate_limited = is_rate_limit_error(error)
            cooldown = runtime.mark_provider_failure(provider, rate_limited=rate_limited)
            logger.warning("Error on %s (%s): %s", model_id, provider, str(error)[:120])
            log_provider_cooldown(provider, cooldown, "rate limit" if rate_limited else "repeated failures")
            continue

    return None
