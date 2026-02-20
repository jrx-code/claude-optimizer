"""Configuration for the prompt optimizer."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "optimizer_config.json"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

SYSTEM_PROMPT = (
    "You are a prompt preprocessor for an AI coding assistant. "
    "Your task: take the user's prompt (possibly in Polish or another language) "
    "and return an optimized English version. Rules:\n"
    "1. Translate to English if not already English.\n"
    "2. Remove filler words and redundancy.\n"
    "3. Keep all technical terms, file paths, code snippets, and variable names EXACTLY as-is.\n"
    "4. Preserve the original intent completely.\n"
    "5. Output ONLY the optimized prompt, nothing else — no explanations, no markdown."
)


def _load_env_api_key() -> str:
    """Load OPENROUTER_API_KEY from .env file or environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


@dataclass
class OptimizerConfig:
    enabled: bool = True
    ollama_url: str = "http://ai.home.lan:11434"
    model: str = "gemma3:4b"
    timeout: int = 12
    min_prompt_length: int = 10
    system_prompt: str = field(default=SYSTEM_PROMPT)
    # OpenRouter fallback
    openrouter_enabled: bool = True
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_model: str = "google/gemma-3-4b-it:free"
    openrouter_api_key: str = field(default_factory=_load_env_api_key)
    openrouter_timeout: int = 15

    @classmethod
    def load(cls, path: Path | str | None = None) -> "OptimizerConfig":
        """Load config from JSON file. Falls back to defaults if file missing."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            return cls()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            known_fields = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in known_fields}
            return cls(**filtered)
        except (json.JSONDecodeError, TypeError):
            return cls()
