"""Tests for optimizer.core (Ollama and OpenRouter mocked)."""

import json
from unittest.mock import patch, MagicMock

from optimizer.config import OptimizerConfig
from optimizer.core import should_skip, call_ollama, call_openrouter, optimize


# --- should_skip tests ---

def test_should_skip_short():
    cfg = OptimizerConfig()
    assert should_skip("hi", cfg) is True
    assert should_skip("", cfg) is True


def test_should_skip_slash_command():
    cfg = OptimizerConfig()
    assert should_skip("/compact", cfg) is True
    assert should_skip("/help", cfg) is True


def test_should_skip_yes_no():
    cfg = OptimizerConfig()
    assert should_skip("yes", cfg) is True
    assert should_skip("no", cfg) is True
    assert should_skip("y", cfg) is True
    assert should_skip("n", cfg) is True
    assert should_skip("tak", cfg) is True
    assert should_skip("nie", cfg) is True


def test_should_not_skip_normal_prompt():
    cfg = OptimizerConfig()
    assert should_skip("Dodaj nowy endpoint do API użytkowników", cfg) is False


# --- call_ollama tests ---

def test_call_ollama_success():
    cfg = OptimizerConfig()
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "message": {"content": "Add a new endpoint to the users API"}
    }).encode("utf-8")
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("optimizer.core.urllib.request.urlopen", return_value=mock_response):
        result = call_ollama("Dodaj nowy endpoint do API użytkowników", cfg)
    assert result == "Add a new endpoint to the users API"


def test_call_ollama_timeout():
    cfg = OptimizerConfig(timeout=1)
    with patch("optimizer.core.urllib.request.urlopen", side_effect=TimeoutError):
        result = call_ollama("test prompt", cfg)
    assert result is None


def test_call_ollama_connection_error():
    cfg = OptimizerConfig()
    with patch("optimizer.core.urllib.request.urlopen", side_effect=OSError("refused")):
        result = call_ollama("test prompt", cfg)
    assert result is None


# --- call_openrouter tests ---

def test_call_openrouter_success():
    cfg = OptimizerConfig(openrouter_api_key="test-key")
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [{"message": {"content": "Add a new endpoint to the users API"}}]
    }).encode("utf-8")
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("optimizer.core.urllib.request.urlopen", return_value=mock_response):
        result = call_openrouter("Dodaj nowy endpoint", cfg)
    assert result == "Add a new endpoint to the users API"


def test_call_openrouter_no_api_key():
    cfg = OptimizerConfig(openrouter_api_key="")
    result = call_openrouter("test prompt", cfg)
    assert result is None


def test_call_openrouter_timeout():
    cfg = OptimizerConfig(openrouter_api_key="test-key", openrouter_timeout=1)
    with patch("optimizer.core.urllib.request.urlopen", side_effect=TimeoutError):
        result = call_openrouter("test prompt", cfg)
    assert result is None


def test_call_openrouter_connection_error():
    cfg = OptimizerConfig(openrouter_api_key="test-key")
    with patch("optimizer.core.urllib.request.urlopen", side_effect=OSError("refused")):
        result = call_openrouter("test prompt", cfg)
    assert result is None


def test_call_openrouter_empty_choices():
    cfg = OptimizerConfig(openrouter_api_key="test-key")
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({"choices": []}).encode("utf-8")
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("optimizer.core.urllib.request.urlopen", return_value=mock_response):
        result = call_openrouter("test prompt", cfg)
    assert result is None


# --- optimize tests (fallback logic) ---

def test_optimize_disabled():
    cfg = OptimizerConfig(enabled=False)
    result = optimize("Dodaj endpoint", cfg)
    assert result is None


def test_optimize_skips_short():
    cfg = OptimizerConfig()
    result = optimize("ok", cfg)
    assert result is None


def test_optimize_ollama_success_no_fallback():
    """When Ollama succeeds, OpenRouter should NOT be called."""
    cfg = OptimizerConfig()
    with patch("optimizer.core.call_ollama", return_value="Add endpoint") as mock_ollama, \
         patch("optimizer.core.call_openrouter") as mock_openrouter:
        result = optimize("Dodaj nowy endpoint do API", cfg)
    assert result == "Add endpoint"
    mock_ollama.assert_called_once()
    mock_openrouter.assert_not_called()


def test_optimize_ollama_fails_openrouter_fallback():
    """When Ollama fails, OpenRouter should be called as fallback."""
    cfg = OptimizerConfig(openrouter_enabled=True)
    with patch("optimizer.core.call_ollama", return_value=None), \
         patch("optimizer.core.call_openrouter", return_value="Add endpoint") as mock_or:
        result = optimize("Dodaj nowy endpoint do API", cfg)
    assert result == "Add endpoint"
    mock_or.assert_called_once()


def test_optimize_ollama_fails_openrouter_disabled():
    """When Ollama fails and OpenRouter is disabled, returns None."""
    cfg = OptimizerConfig(openrouter_enabled=False)
    with patch("optimizer.core.call_ollama", return_value=None), \
         patch("optimizer.core.call_openrouter") as mock_or:
        result = optimize("Dodaj nowy endpoint do API", cfg)
    assert result is None
    mock_or.assert_not_called()


def test_optimize_both_fail():
    """When both Ollama and OpenRouter fail, returns None (graceful)."""
    cfg = OptimizerConfig(openrouter_enabled=True)
    with patch("optimizer.core.call_ollama", return_value=None), \
         patch("optimizer.core.call_openrouter", return_value=None):
        result = optimize("Dodaj nowy endpoint do API", cfg)
    assert result is None
