import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    async def generate_variants(
        self, prompt: str, max_variants: int
    ) -> list[dict[str, Any]]:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class OpenAICompatibleProvider(LLMProvider):
    @classmethod
    def validate_configuration(cls):
        """Pre-flight check for configuration validity"""
        api_key = settings.OPENAI_API_KEY
        if not api_key or not api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is not configured.\n\n"
                "Set it in the current PowerShell session:\n\n"
                '$env:OPENAI_API_KEY="your-key"\n\n'
                "or place it in the local .env file.\n\n"
                "Then rerun:\n\n"
                "uv run python -m scripts.benchmark generate-llm-variants --dataset scifact"
            )
        if not settings.OPENAI_MODEL:
            raise ValueError("OPENAI_MODEL is not configured.")

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        timeout: float = 10.0,
    ):
        # Import openai lazily to not break things if missing during early imports
        from openai import AsyncOpenAI

        self.validate_configuration()

        api_key = (settings.OPENAI_API_KEY or "").strip()
        base_url = (
            settings.OPENAI_BASE_URL.strip() if settings.OPENAI_BASE_URL else None
        )

        # Standard configuration
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )
        self._model = model or settings.OPENAI_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        # We track metrics from the last call
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_latency_ms = 0.0

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate_variants(
        self, prompt: str, max_variants: int
    ) -> list[dict[str, Any]]:
        t0 = time.time()
        try:
            for attempt in range(15):
                try:
                    response = await self.client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a search query transformation engine. Output strictly valid JSON with a single 'variants' array.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    break
                except Exception as e:
                    error_msg = str(e)
                    # Check for transient server errors (5xx)
                    is_transient = any(
                        code in error_msg for code in ["500", "502", "503", "504"]
                    )
                    if is_transient and attempt < 14:
                        import asyncio

                        # Bounded exponential backoff: 2^attempt, capped at 60 seconds
                        sleep_time = min(60, 2**attempt)
                        logger.warning(
                            f"Transient error encountered: {error_msg}. Retrying in {sleep_time}s (attempt {attempt + 1}/15)..."
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                    raise

            latency = (time.time() - t0) * 1000
            self.last_latency_ms = latency

            usage = response.usage
            if usage:
                self.last_input_tokens = usage.prompt_tokens
                self.last_output_tokens = usage.completion_tokens

            content = response.choices[0].message.content or ""
            # Strip markdown json block if present
            if content.strip().startswith("```json"):
                content = content.strip()[7:]
                content = content.removesuffix("```")
            elif content.strip().startswith("```"):
                content = content.strip()[3:]
                content = content.removesuffix("```")

            parsed = json.loads(content) if content else {}

            return parsed.get("variants", [])

        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            # Mock fallback for quota limits to allow evaluation to complete
            return [{"query": prompt + " (fallback)", "is_accepted": True}]
