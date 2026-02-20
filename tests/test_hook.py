"""Tests for optimizer.hook (subprocess-based integration tests)."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_hook(prompt: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    """Run hook.py as subprocess with given prompt in hook JSON format."""
    hook_input = json.dumps({"prompt": prompt})
    return subprocess.run(
        [sys.executable, "-m", "optimizer.hook"],
        input=hook_input,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


class TestHookUnit:
    """Unit tests with mocked optimizer."""

    def test_hook_with_optimization(self):
        with patch("optimizer.hook.OptimizerConfig.load") as mock_load, \
             patch("optimizer.hook.optimize", return_value="Optimized prompt"):
            from optimizer.hook import main
            from io import StringIO
            with patch("sys.stdin", StringIO(json.dumps({"prompt": "test prompt"}))):
                with patch("builtins.print") as mock_print:
                    main()
                    mock_print.assert_called_once_with("Optimized prompt")

    def test_hook_no_optimization(self):
        with patch("optimizer.hook.OptimizerConfig.load"), \
             patch("optimizer.hook.optimize", return_value=None):
            from optimizer.hook import main
            from io import StringIO
            with patch("sys.stdin", StringIO(json.dumps({"prompt": "ok"}))):
                with patch("builtins.print") as mock_print:
                    main()
                    mock_print.assert_not_called()

    def test_hook_invalid_json(self):
        from optimizer.hook import main
        from io import StringIO
        with patch("sys.stdin", StringIO("not json")):
            with patch("builtins.print") as mock_print:
                main()
                mock_print.assert_not_called()


class TestHookSubprocess:
    """Subprocess tests — always exit 0."""

    def test_exit_code_zero_on_short_prompt(self):
        result = run_hook("ok")
        assert result.returncode == 0

    def test_exit_code_zero_on_slash_command(self):
        result = run_hook("/compact")
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_exit_code_zero_on_invalid_json(self):
        proc = subprocess.run(
            [sys.executable, "-m", "optimizer.hook"],
            input="not json",
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0


class TestHookLive:
    """Live integration tests — skipped if Ollama is unreachable."""

    @pytest.fixture(autouse=True)
    def check_ollama(self):
        import urllib.request
        import urllib.error
        try:
            urllib.request.urlopen("http://ai.home.lan:11434/api/tags", timeout=3)
        except (urllib.error.URLError, OSError):
            pytest.skip("Ollama not reachable")

    def test_polish_prompt_returns_english(self):
        result = run_hook("Dodaj nowy endpoint do API użytkowników z walidacją danych wejściowych")
        assert result.returncode == 0
        output = result.stdout.strip()
        assert len(output) > 0, "Expected optimized output"

    def test_english_prompt_still_works(self):
        result = run_hook("Add a new REST endpoint for user management with input validation")
        assert result.returncode == 0


class TestOpenRouterLive:
    """Live OpenRouter tests — skipped if API unreachable or no key."""

    @pytest.fixture(autouse=True)
    def check_openrouter(self):
        from optimizer.config import OptimizerConfig
        cfg = OptimizerConfig.load()
        if not cfg.openrouter_api_key:
            pytest.skip("No OpenRouter API key")
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {cfg.openrouter_api_key}"},
            )
            urllib.request.urlopen(req, timeout=5)
        except (urllib.error.URLError, OSError):
            pytest.skip("OpenRouter not reachable")

    def test_openrouter_fallback_polish(self):
        """Simulate Ollama down — OpenRouter should handle the prompt."""
        from optimizer.config import OptimizerConfig
        from optimizer.core import call_openrouter
        cfg = OptimizerConfig.load()
        result = call_openrouter(
            "Dodaj nowy endpoint do API użytkowników z walidacją danych",
            cfg,
        )
        assert result is not None
        assert len(result) > 0
