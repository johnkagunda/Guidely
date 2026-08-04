"""LLM generation service.

Supports two interchangeable providers, selected via the LLM_PROVIDER
environment variable:

    LLM_PROVIDER=ollama  (default) - calls a local Ollama server
    LLM_PROVIDER=openai            - calls an OpenAI-compatible /chat/completions API

Both providers implement the same generate(prompt) -> str interface so
the RAG pipeline never needs to know which one is active.
"""
from __future__ import annotations

import os
from typing import Optional

from utils.logger import get_logger, log_event

logger = get_logger("guidely.llm")

SYSTEM_PROMPT_TEMPLATE = """You are Guidely, an internal knowledge assistant.

Answer the user's question using only the supplied context.

Do not invent information.

If the context does not contain enough information, clearly say that the \
answer could not be found in the available documents.

Keep the answer concise and easy to understand.

Question:
{question}

Context:
{context}"""


class LLMError(Exception):
    """Raised for provider errors, missing config, or timeouts."""


class LLMTimeoutError(LLMError):
    pass


class LLMConfigError(LLMError):
    pass


def build_prompt(question: str, context: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(question=question, context=context)


class OllamaProvider:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    def generate(self, prompt: str) -> str:
        import httpx

        url = f"{self.base_url.rstrip('/')}/api/generate"
        try:
            resp = httpx.post(
                url,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self.timeout_s}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMConfigError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Is `ollama serve` running?"
            ) from exc

        if resp.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        answer = data.get("response", "").strip()
        if not answer:
            raise LLMError("Ollama returned an empty response")
        return answer


class OpenAICompatibleProvider:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise LLMConfigError(
                "OPENAI_API_KEY is not set. Configure it in your .env file, "
                "or switch LLM_PROVIDER=ollama."
            )

        import httpx

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=self.timeout_s)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM request timed out after {self.timeout_s}s"
            ) from exc

        if resp.status_code != 200:
            raise LLMError(f"LLM API returned HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        try:
            answer = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError("Unexpected LLM API response shape") from exc
        if not answer:
            raise LLMError("LLM returned an empty response")
        return answer


_provider_singleton: Optional[object] = None


def get_llm_provider():
    global _provider_singleton
    if _provider_singleton is None:
        provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()
        if provider_name == "openai":
            _provider_singleton = OpenAICompatibleProvider()
        else:
            _provider_singleton = OllamaProvider()
        log_event(logger, "llm_provider_selected", provider=provider_name)
    return _provider_singleton


def generate_answer(question: str, context: str) -> str:
    provider = get_llm_provider()
    prompt = build_prompt(question, context)
    return provider.generate(prompt)
