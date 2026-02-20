"""Tests for optimizer.config."""

import json
import tempfile
from pathlib import Path

from optimizer.config import OptimizerConfig


def test_defaults():
    cfg = OptimizerConfig()
    assert cfg.enabled is True
    assert cfg.model == "gemma3:4b"
    assert cfg.timeout == 12
    assert cfg.min_prompt_length == 10
    assert "ai.home.lan" in cfg.ollama_url


def test_openrouter_defaults():
    cfg = OptimizerConfig()
    assert cfg.openrouter_enabled is True
    assert "openrouter.ai" in cfg.openrouter_url
    assert cfg.openrouter_model == "google/gemma-3-4b-it:free"
    assert cfg.openrouter_timeout == 15


def test_load_from_file():
    data = {"enabled": False, "model": "llama3:8b", "timeout": 20}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        cfg = OptimizerConfig.load(f.name)
    assert cfg.enabled is False
    assert cfg.model == "llama3:8b"
    assert cfg.timeout == 20


def test_load_openrouter_from_file():
    data = {"openrouter_enabled": False, "openrouter_model": "meta-llama/llama-3.1-8b-instruct"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        cfg = OptimizerConfig.load(f.name)
    assert cfg.openrouter_enabled is False
    assert cfg.openrouter_model == "meta-llama/llama-3.1-8b-instruct"


def test_load_missing_file():
    cfg = OptimizerConfig.load("/tmp/nonexistent_optimizer_config.json")
    assert cfg.enabled is True
    assert cfg.model == "gemma3:4b"


def test_load_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not json")
        f.flush()
        cfg = OptimizerConfig.load(f.name)
    assert cfg.enabled is True


def test_unknown_fields_ignored():
    data = {"enabled": True, "unknown_field": "value"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        cfg = OptimizerConfig.load(f.name)
    assert cfg.enabled is True
    assert not hasattr(cfg, "unknown_field")


def test_load_default_path():
    cfg = OptimizerConfig.load()
    assert isinstance(cfg, OptimizerConfig)


def test_api_key_loaded_from_env():
    """API key should be loaded from .env file at project root."""
    cfg = OptimizerConfig.load()
    assert cfg.openrouter_api_key.startswith("sk-or-")
