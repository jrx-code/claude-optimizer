"""Core optimization logic — Ollama client, OpenRouter fallback, and prompt processing."""

import json
import re
import urllib.request
import urllib.error
from optimizer.config import OptimizerConfig

# Patterns that should skip optimization
SKIP_PATTERNS = re.compile(
    r"^(/\w|yes$|no$|y$|n$|ok$|tak$|nie$)",
    re.IGNORECASE,
)


def should_skip(prompt: str, config: OptimizerConfig) -> bool:
    """Return True if prompt should NOT be optimized."""
    stripped = prompt.strip()
    if len(stripped) < config.min_prompt_length:
        return True
    if SKIP_PATTERNS.match(stripped):
        return True
    return False


def call_ollama(prompt: str, config: OptimizerConfig) -> str | None:
    """Send prompt to Ollama API. Returns optimized text or None on failure."""
    payload = json.dumps({
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }).encode("utf-8")

    url = f"{config.ollama_url.rstrip('/')}/api/chat"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "").strip()
            return content if content else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def call_openrouter(prompt: str, config: OptimizerConfig) -> str | None:
    """Send prompt to OpenRouter API. Returns optimized text or None on failure.

    Free tier models (Google AI Studio) don't support the system role,
    so we merge the system prompt into the user message.
    """
    if not config.openrouter_api_key:
        return None

    merged_content = (
        f"[Instructions]\n{config.system_prompt}\n\n"
        f"[Prompt to optimize]\n{prompt}"
    )

    payload = json.dumps({
        "model": config.openrouter_model,
        "messages": [
            {"role": "user", "content": merged_content},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        config.openrouter_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.openrouter_api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=config.openrouter_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                return content if content else None
            return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def optimize(prompt: str, config: OptimizerConfig | None = None) -> str | None:
    """Optimize a prompt. Tries Ollama first, falls back to OpenRouter."""
    if config is None:
        config = OptimizerConfig.load()
    if not config.enabled:
        return None
    if should_skip(prompt, config):
        return None

    result = call_ollama(prompt, config)
    if result:
        return result

    if config.openrouter_enabled:
        return call_openrouter(prompt, config)

    return None
