"""
llm_tools.py
A lightweight, robust multi-provider LLM client with a classical chat interface.

Supported providers:
- OLLAMA (local)            -> http://localhost:11434/api/chat
- OPENAI (official API)     -> https://api.openai.com/v1/chat/completions
- HUGGINGFACE_INFERENCE_API -> https://api-inference.huggingface.co/models/{model}

Features:
- Classical chat structure: [system?, user, ...]
- Optional system prompt with sensible default
- Tidy configuration via LLMConfig
- Provider-agnostic interface: LLMClient.generate()
- Graceful error handling + simple retries
- Minimal dependencies (only 'requests')
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time
import json

try:
    import requests
except ImportError as e:
    raise ImportError("Please install 'requests' to use llm_tools: pip install requests") from e


# --------------------------
# Public Types
# --------------------------

class Provider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMConfig:
    provider: Provider
    model: str
    # Common gen params
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None  # some OSS models support it
    # Provider specifics
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # e.g., Ollama: "http://localhost:11434"
    # Networking
    timeout: int = 60  # seconds
    retries: int = 2
    retry_backoff_sec: float = 1.0
    # Extra headers (if needed)
    extra_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMResult:
    text: str
    model: str
    provider: Provider
    raw: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)


# --------------------------
# Exceptions
# --------------------------

class LLMRouterError(Exception):
    pass


class ProviderError(LLMRouterError):
    pass


class NetworkError(LLMRouterError):
    pass


# --------------------------
# Backend Interfaces
# --------------------------

class BaseBackend:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        raise NotImplementedError


# --------------------------
# Utilities
# --------------------------

DEFAULT_SYSTEM = (
    "You are a helpful, concise assistant. "
    "Follow the user's instructions carefully and answer step-by-step when helpful."
)

def _build_messages(
    user_prompt: str,
    system_prompt: Optional[str],
    extra_messages: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """
    Classical structure: system? -> user -> (optional extras like few-shot or assistant turns)
    """
    msgs: List[Dict[str, str]] = []
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM
    msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    if extra_messages:
        # ensure each has 'role' and 'content'
        for m in extra_messages:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise LLMRouterError("extra_messages entries must be dicts with 'role' and 'content'")
            msgs.append(m)
    return msgs


def _do_request_with_retries(method, url, *, headers=None, json_body=None, timeout=60, retries=2, backoff=1.0):
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method=method, url=url, headers=headers, json=json_body, timeout=timeout
            )
            return resp
        except requests.RequestException as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
            else:
                raise NetworkError(f"Network error after {retries+1} attempts: {e}") from e
    # unreachable, but keeps mypy happy
    raise NetworkError(str(last_err))


# --------------------------
# Ollama Backend
# --------------------------

class OllamaBackend(BaseBackend):
    """
    Uses Ollama's /api/chat endpoint.
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#chat
    """

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        base = self.cfg.base_url or "http://localhost:11434"
        url = f"{base}/api/chat"

        messages = _build_messages(user_prompt, system_prompt, extra_messages)

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
            },
        }
        if self.cfg.max_tokens is not None:
            payload["options"]["num_predict"] = self.cfg.max_tokens
        if self.cfg.top_k is not None:
            payload["options"]["top_k"] = self.cfg.top_k
        if stop:
            payload["options"]["stop"] = stop

        headers = {"Content-Type": "application/json", **self.cfg.extra_headers}

        resp = _do_request_with_retries(
            "POST",
            url,
            headers=headers,
            json_body=payload,
            timeout=self.cfg.timeout,
            retries=self.cfg.retries,
            backoff=self.cfg.retry_backoff_sec,
        )

        if resp.status_code != 200:
            raise ProviderError(f"Ollama error {resp.status_code}: {resp.text}")

        data = resp.json()
        text = data.get("message", {}).get("content", "") or ""
        usage = {}
        return LLMResult(text=text, model=self.cfg.model, provider=Provider.OLLAMA, raw=data, usage=usage)


# --------------------------
# OpenAI Backend
# --------------------------

class OpenAIBackend(BaseBackend):
    """
    Official OpenAI Chat Completions API.
    """

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        if not self.cfg.api_key:
            raise ProviderError("OpenAI requires api_key in LLMConfig")

        base = self.cfg.base_url or "https://api.openai.com/v1"
        url = f"{base}/chat/completions"

        messages = _build_messages(user_prompt, system_prompt, extra_messages)

        body: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        if self.cfg.max_tokens is not None:
            body["max_tokens"] = self.cfg.max_tokens
        if stop:
            body["stop"] = stop

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
            **self.cfg.extra_headers,
        }

        resp = _do_request_with_retries(
            "POST",
            url,
            headers=headers,
            json_body=body,
            timeout=self.cfg.timeout,
            retries=self.cfg.retries,
            backoff=self.cfg.retry_backoff_sec,
        )

        if resp.status_code != 200:
            raise ProviderError(f"OpenAI error {resp.status_code}: {resp.text}")

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = choice.get("message", {}).get("content", "") or ""
        usage = data.get("usage", {}) or {}
        return LLMResult(text=text, model=self.cfg.model, provider=Provider.OPENAI, raw=data, usage=usage)


# --------------------------
# Hugging Face Inference API Backend
# --------------------------

class HFBackend(BaseBackend):
    """
    Hugging Face Inference API for text generation / chat.
    Works best with chat-ready models that accept messages-like input,
    but can also do plain text prompting by concatenating prompts.

    Endpoint: https://api-inference.huggingface.co/models/{model}
    """

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        if not self.cfg.api_key:
            raise ProviderError("Hugging Face Inference API requires api_key in LLMConfig")

        base = self.cfg.base_url or "https://api-inference.huggingface.co/models"
        url = f"{base}/{self.cfg.model}"

        # For HF Inference API, many models are not chat-native.
        # We'll build a simple composite prompt:
        # [SYSTEM]\n{system}\n\n[USER]\n{user}\n\n plus any extras.
        system_text = system_prompt or DEFAULT_SYSTEM
        composite = f"[SYSTEM]\n{system_text}\n\n[USER]\n{user_prompt}\n"

        if extra_messages:
            for m in extra_messages:
                role = m.get("role", "user").upper()
                content = m.get("content", "")
                composite += f"\n[{role}]\n{content}\n"

        gen_params = {
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
        }
        if self.cfg.max_tokens is not None:
            gen_params["max_new_tokens"] = self.cfg.max_tokens
        if self.cfg.top_k is not None:
            gen_params["top_k"] = self.cfg.top_k
        if stop:
            gen_params["stop_sequences"] = stop  # not all models honor this

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            **self.cfg.extra_headers,
        }
        body = {"inputs": composite, "parameters": gen_params, "options": {"wait_for_model": True}}

        resp = _do_request_with_retries(
            "POST",
            url,
            headers=headers,
            json_body=body,
            timeout=self.cfg.timeout,
            retries=self.cfg.retries,
            backoff=self.cfg.retry_backoff_sec,
        )

        if resp.status_code != 200:
            raise ProviderError(f"HuggingFace error {resp.status_code}: {resp.text}")

        data = resp.json()
        # HF returns a list of dicts with 'generated_text' OR a dict with 'error'
        text = ""
        if isinstance(data, list) and data:
            # many models return [{"generated_text": "..."}]
            item = data[0]
            text = item.get("generated_text") or item.get("summary_text") or ""
            # If the model returns the full prompt + completion, strip the prompt:
            if text.startswith(composite):
                text = text[len(composite):].strip()
        elif isinstance(data, dict) and "generated_text" in data:
            text = data.get("generated_text", "")
        else:
            # Some models return token-stream-like outputs or other shapes
            text = json.dumps(data)  # last resort: return raw json as text

        usage = {}
        return LLMResult(text=text, model=self.cfg.model, provider=Provider.HUGGINGFACE, raw=data, usage=usage)


# --------------------------
# Public Client
# --------------------------

class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if cfg.provider == Provider.OLLAMA:
            self._backend = OllamaBackend(cfg)
        elif cfg.provider == Provider.OPENAI:
            self._backend = OpenAIBackend(cfg)
        elif cfg.provider == Provider.HUGGINGFACE:
            self._backend = HFBackend(cfg)
        else:
            raise LLMRouterError(f"Unsupported provider: {cfg.provider}")

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        *,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        if not user_prompt or not isinstance(user_prompt, str):
            raise LLMRouterError("user_prompt must be a non-empty string")

        return self._backend.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            extra_messages=extra_messages,
            stop=stop,
        )


# --------------------------
# Quick self-test (optional)
# --------------------------
if __name__ == "__main__":
    # Example: Ollama local (mistral:latest)
    try:
        cfg = LLMConfig(
            provider=Provider.OLLAMA,
            model="mistral:latest",
            temperature=0.2,
            top_p=0.95,
            max_tokens=256,
            base_url="http://localhost:11434",
        )
        client = LLMClient(cfg)
        res = client.generate(
            user_prompt="Explain diffusion models in one paragraph.",
            system_prompt=None,  # will use default
        )
        print(f"[{res.provider}] {res.model} ->\n{res.text}\n")
    except Exception as e:
        print("Self-test (Ollama) skipped or failed:", e)
