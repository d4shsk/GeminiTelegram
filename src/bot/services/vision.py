import asyncio
import base64

from google import genai
import openai as openai_lib

from ..runtime import runtime
from ..utils import moscow_datetime


def _vision_system_prompt() -> str:
    return (
        "Отвечай на русском языке. Описывай изображение подробно и четко. "
        f"Текущие дата и время (МСК): {moscow_datetime()}"
    )


async def _analyze_with_gemini(caption: str, image_bytes: bytes) -> str:
    if not runtime.gemini_client:
        raise RuntimeError("Не задан GEMINI_API_KEY.")

    config = genai.types.GenerateContentConfig(system_instruction=_vision_system_prompt(), temperature=0.3)
    try:
        image_part = genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        response = await asyncio.wait_for(
            runtime.gemini_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[caption, image_part],
                config=config,
            ),
            timeout=45.0,
        )
    except Exception:
        image_b64 = base64.b64encode(image_bytes).decode()
        response = await asyncio.wait_for(
            runtime.gemini_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": caption},
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                        ],
                    }
                ],
                config=config,
            ),
            timeout=45.0,
        )

    return (response.text or "").strip()


async def _analyze_with_gpt4o(caption: str, image_b64: str) -> str:
    if not runtime.github_client:
        raise RuntimeError("Не задан GITHUB_TOKEN.")

    response = await asyncio.wait_for(
        runtime.github_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _vision_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
                        },
                    ],
                },
            ],
            max_tokens=1024,
            temperature=0.3,
        ),
        timeout=45.0,
    )
    return (response.choices[0].message.content or "").strip()


async def _analyze_with_kimi(caption: str, image_b64: str) -> str:
    if not runtime.cf_accounts or not runtime.cf_tokens:
        raise RuntimeError("Не заданы CF_ACCOUNT_ID/CF_API_TOKEN.")

    attempts = 0
    while attempts < len(runtime.cf_tokens):
        account_id, token = runtime.get_current_cf_credentials()
        if not account_id or not token:
            raise RuntimeError("Нет валидных Cloudflare credentials.")

        cf_client = openai_lib.AsyncOpenAI(
            api_key=token,
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        )
        try:
            response = await asyncio.wait_for(
                cf_client.chat.completions.create(
                    model="@cf/moonshotai/kimi-k2.6",
                    messages=[
                        {"role": "system", "content": _vision_system_prompt()},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": caption},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=1024,
                ),
                timeout=45.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as error:
            text = str(error).lower()
            if "429" in text or "limit" in text or "quota" in text:
                runtime.rotate_cf_credentials()
                attempts += 1
                continue
            raise

    raise RuntimeError("Все Cloudflare-токены недоступны.")


async def analyze_image(model_id: str, caption: str, image_bytes: bytes) -> str:
    image_b64 = base64.b64encode(image_bytes).decode()
    if model_id == "gemini-2.5-flash":
        return await _analyze_with_gemini(caption, image_bytes)
    if model_id == "gpt-4o":
        return await _analyze_with_gpt4o(caption, image_b64)
    if model_id == "@cf/moonshotai/kimi-k2.6":
        return await _analyze_with_kimi(caption, image_b64)
    raise RuntimeError(f"Модель {model_id} не поддерживает vision.")
