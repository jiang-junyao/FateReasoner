"""Provider adapters for FateReasoner's LLM scoring prompts.

No provider SDK is needed; requests and responses use the documented REST APIs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

PROVIDERS = ("deepseek", "gpt", "claude", "gemini", "glm", "qwen")
DEFAULTS = {
    "deepseek": ("deepseek-flash", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "gpt": ("gpt-4.1", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "claude": ("claude-sonnet-4-5", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "gemini": ("gemini-3.5-flash", "https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
    "glm": ("glm-4.5", "https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
    "qwen": ("qwen-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
}


def read_api_key(api_key: str | None, api_key_file: str | None, provider: str | None = None) -> str:
    if provider is None:
        raise ValueError(f"No LLM provider specified. Pass provider= (Python) or --provider (CLI); choose from {', '.join(PROVIDERS)}.")
    if provider not in DEFAULTS:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose from {', '.join(PROVIDERS)}")
    if api_key:
        return api_key.strip()
    if api_key_file:
        path = Path(api_key_file)
        if not path.is_file():
            raise FileNotFoundError(f"API key file not found: {path}")
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
        raise ValueError(f"API key file is empty: {path}")
    env_name = DEFAULTS[provider][2]
    key = os.environ.get(env_name, "").strip()
    if not key and provider == "glm":
        key = os.environ.get("GLM_API_KEY", "").strip()
    if not key and provider == "gemini":
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise ValueError(f"{provider} API key required: use --api-key-file or set {env_name}.")
    return key


def call_llm_chat(
    *, prompt: str, api_key: str, provider: str | None = None, model: str | None = None,
    base_url: str | None = None, thinking: str = "disabled", temperature: float = 0.0,
    max_tokens: int = 8000, timeout: float = 120.0, max_retries: int = 3,
) -> tuple[str, dict[str, Any]]:
    """Return the generated text and unmodified provider response JSON."""
    if provider is None:
        raise ValueError(f"No LLM provider specified. Pass provider= (Python) or --provider (CLI); choose from {', '.join(PROVIDERS)}.")
    if provider not in DEFAULTS:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose from {', '.join(PROVIDERS)}")
    if max_retries < 1 or max_tokens < 1:
        raise ValueError("max_retries and max_tokens must be positive")
    default_model, default_url, _ = DEFAULTS[provider]
    model = model or default_model
    base = (base_url or default_url).rstrip("/")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if provider == "claude":
        url = base + "/messages"
        headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        payload: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                                   "messages": [{"role": "user", "content": prompt}]}
        if temperature != 0.0:
            payload["temperature"] = temperature
    elif provider == "gemini":
        url = base + f"/models/{model}:generateContent"
        headers["x-goog-api-key"] = api_key
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"maxOutputTokens": max_tokens}}
        if temperature != 0.0:
            payload["generationConfig"]["temperature"] = temperature
    else:
        url = base + "/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if provider == "gpt":
            payload["max_completion_tokens"] = max_tokens
            # Reasoning models may reject temperature; omit the default value.
            if temperature != 0.0:
                payload["temperature"] = temperature
        else:
            payload.update({"temperature": temperature, "max_tokens": max_tokens})
            if provider == "deepseek" and thinking == "disabled":
                payload["thinking"] = {"type": "disabled"}
    session = requests.Session()
    session.trust_env = False
    try:
        for attempt in range(max_retries):
            try:
                response = session.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                if provider == "claude":
                    parts = [part.get("text", "") for part in data.get("content", []) if part.get("type") == "text"]
                elif provider == "gemini":
                    candidates = data.get("candidates", [])
                    parts = [part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])] if candidates else []
                else:
                    choices = data.get("choices", [])
                    parts = [choices[0].get("message", {}).get("content", "")] if choices else []
                result = "".join(str(part) for part in parts if part)
                if not result:
                    raise ValueError(f"{provider} returned no text; inspect the saved API response or token limit")
                return result, data
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status not in (408, 409, 429) and status < 500:
                    raise RuntimeError(f"{provider} API returned HTTP {status}") from exc
                error: Exception = RuntimeError(f"{provider} API returned HTTP {status}")
            except (requests.ConnectionError, requests.Timeout) as exc:
                error = exc
            if attempt + 1 < max_retries:
                time.sleep(min(10.0, 1.5 * (attempt + 1)))
        raise RuntimeError(f"{provider} request failed after {max_retries} attempts: {error}")
    finally:
        session.close()
