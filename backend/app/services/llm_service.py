"""
LLM provider abstraction.

This is the swap point the project's "Model Abstraction" principle
calls for: nothing above this module (rag_service, the API layer)
should know or care whether answers come from a local Ollama model, a
cloud API, or anything else. They call `get_llm_provider().generate(prompt)`
and get a string back.

Today only Ollama is implemented (matches this machine's hardware --
8GB RAM, no usable GPU -- and keeps everything local/free). Adding a
cloud provider (Anthropic, OpenAI, ...) later means writing one more
class here and adding one branch to get_llm_provider(); nothing that
calls generate() needs to change.
"""

from abc import ABC, abstractmethod

import requests

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Given a fully-constructed prompt, return the model's text
        response. Implementations own their own error handling and
        should raise a clear exception on failure -- callers don't
        retry or interpret provider-specific errors."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """
    Talks to a local Ollama server over its HTTP API
    (POST /api/generate), with streaming disabled -- we want one
    complete answer back, not a token stream, since Phase 4 doesn't
    have a streaming response layer yet.
    """

    def generate(self, prompt: str) -> str:
        url = f"{settings.ollama_base_url}/api/generate"
        try:
            response = requests.post(
                url,
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=settings.llm_request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_base_url}. "
                "Is the Ollama service running? (Run `ollama serve` or "
                "start the Ollama app.)"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama did not respond within {settings.llm_request_timeout_seconds}s. "
                "The model may still be loading, or the machine is overloaded."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Ollama returned an error: {exc}") from exc

        data = response.json()
        return data.get("response", "").strip()


def get_llm_provider() -> LLMProvider:
    """Factory: the one place that maps config -> concrete provider."""
    if settings.llm_provider == "ollama":
        return OllamaProvider()
    raise ValueError(
        f"Unknown llm_provider '{settings.llm_provider}'. "
        "Supported: 'ollama'."
    )
