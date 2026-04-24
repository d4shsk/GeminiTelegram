import logging
import os
import time

from aiogram import Bot
from google import genai
from groq import AsyncGroq
import openai as openai_lib


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class AppRuntime:
    def __init__(self) -> None:
        self.telegram_token = os.environ.get("TELEGRAM_TOKEN")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.github_token = os.environ.get("GITHUB_TOKEN")

        self.bot = Bot(token=self.telegram_token) if self.telegram_token else None
        self.gemini_client = genai.Client(api_key=self.gemini_key) if self.gemini_key else None
        self.groq_client = AsyncGroq(api_key=self.groq_key) if self.groq_key else None
        self.github_client = (
            openai_lib.AsyncOpenAI(
                api_key=self.github_token,
                base_url="https://models.inference.ai.azure.com",
            )
            if self.github_token
            else None
        )

        account_ids_raw = os.environ.get("CF_ACCOUNT_IDS", os.environ.get("CF_ACCOUNT_ID", ""))
        api_tokens_raw = os.environ.get("CF_API_TOKENS", os.environ.get("CF_API_TOKEN", ""))

        self.cf_accounts = [account.strip() for account in account_ids_raw.split(",") if account.strip()]
        self.cf_tokens = [token.strip() for token in api_tokens_raw.split(",") if token.strip()]

        if len(self.cf_accounts) == 1 and len(self.cf_tokens) > 1:
            self.cf_accounts = self.cf_accounts * len(self.cf_tokens)

        if self.cf_accounts and self.cf_tokens and len(self.cf_accounts) != len(self.cf_tokens):
            logger.warning("Количество CF_ACCOUNT_IDS не совпадает с CF_API_TOKENS.")

        self.current_cf_idx = 0
        self.provider_failures: dict[str, int] = {}
        self.provider_cooldowns: dict[str, float] = {}

    def get_current_cf_credentials(self) -> tuple[str | None, str | None]:
        if not self.cf_accounts or not self.cf_tokens:
            return None, None

        index = self.current_cf_idx % len(self.cf_tokens)
        token = self.cf_tokens[index]
        account = self.cf_accounts[index] if index < len(self.cf_accounts) else self.cf_accounts[-1]
        return account, token

    def rotate_cf_credentials(self) -> None:
        if self.cf_tokens:
            self.current_cf_idx = (self.current_cf_idx + 1) % len(self.cf_tokens)

    def is_provider_available(self, provider: str) -> bool:
        cooldown_until = self.provider_cooldowns.get(provider, 0.0)
        if cooldown_until <= time.monotonic():
            self.provider_cooldowns.pop(provider, None)
            return True
        return False

    def provider_cooldown_remaining(self, provider: str) -> float:
        cooldown_until = self.provider_cooldowns.get(provider, 0.0)
        return max(0.0, cooldown_until - time.monotonic())

    def mark_provider_success(self, provider: str) -> None:
        self.provider_failures.pop(provider, None)
        self.provider_cooldowns.pop(provider, None)

    def mark_provider_failure(self, provider: str, *, rate_limited: bool = False) -> float:
        failures = self.provider_failures.get(provider, 0) + 1
        self.provider_failures[provider] = failures

        cooldown_seconds = 0.0
        if rate_limited:
            cooldown_seconds = 600.0
        elif failures >= 3:
            cooldown_seconds = min(120.0 * (failures - 2), 900.0)

        if cooldown_seconds:
            self.provider_cooldowns[provider] = time.monotonic() + cooldown_seconds

        return cooldown_seconds


runtime = AppRuntime()
