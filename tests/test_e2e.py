"""End-to-end tests for the full hook flow."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_hook(prompt: str, timeout: float = 20.0, env=None) -> subprocess.CompletedProcess:
    """Run hook.py as subprocess with given prompt."""
    import os
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "optimizer.hook"],
        input=json.dumps({"prompt": prompt}),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env=run_env,
    )


def ollama_available() -> bool:
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen("http://ai.home.lan:11434/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


def openrouter_available() -> bool:
    from optimizer.config import OptimizerConfig
    cfg = OptimizerConfig.load()
    if not cfg.openrouter_api_key:
        return False
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {cfg.openrouter_api_key}"},
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


class TestSkipCases:
    """Prompts that should pass through without optimization."""

    def test_short_prompt(self):
        result = run_hook("ok")
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_slash_command(self):
        result = run_hook("/compact")
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_yes_no(self):
        for word in ("yes", "no", "y", "n", "tak", "nie"):
            result = run_hook(word)
            assert result.returncode == 0
            assert result.stdout.strip() == ""

    def test_invalid_json(self):
        proc = subprocess.run(
            [sys.executable, "-m", "optimizer.hook"],
            input="not json at all",
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""


@pytest.mark.skipif(not ollama_available(), reason="Ollama not reachable")
class TestOllamaE2E:
    """Full flow with live Ollama."""

    def test_polish_prompt_translated(self):
        result = run_hook(
            "Zrefaktoryzuj klasę UserService w pliku src/services/user.py "
            "żeby używała dependency injection zamiast hardkodowanych zależności"
        )
        assert result.returncode == 0
        output = result.stdout.strip()
        assert len(output) > 0
        assert "UserService" in output
        assert "src/services/user.py" in output

    def test_english_prompt_passthrough(self):
        result = run_hook(
            "Add a new REST endpoint for user management with input validation"
        )
        assert result.returncode == 0
        output = result.stdout.strip()
        assert len(output) > 0

    def test_technical_terms_preserved(self):
        result = run_hook(
            "Napraw błąd w pliku db/connection.py gdzie DatabaseConnection "
            "nie zamyka połączenia po TimeoutError"
        )
        assert result.returncode == 0
        output = result.stdout.strip()
        assert "db/connection.py" in output
        assert "DatabaseConnection" in output
        assert "TimeoutError" in output


@pytest.mark.skipif(not openrouter_available(), reason="OpenRouter not reachable")
class TestFallbackE2E:
    """Ollama dead → OpenRouter fallback."""

    def test_fallback_translates_polish(self):
        from optimizer.config import OptimizerConfig
        from optimizer.core import optimize

        cfg = OptimizerConfig.load()
        cfg.ollama_url = "http://127.0.0.1:1"  # dead port
        cfg.timeout = 2

        result = optimize(
            "Napisz testy jednostkowe dla modułu płatności "
            "w tests/test_payments.py z mockowaniem zewnętrznego API",
            cfg,
        )
        assert result is not None
        assert len(result) > 0
        assert "test" in result.lower()

    def test_fallback_preserves_paths(self):
        from optimizer.config import OptimizerConfig
        from optimizer.core import optimize

        cfg = OptimizerConfig.load()
        cfg.ollama_url = "http://127.0.0.1:1"
        cfg.timeout = 2

        result = optimize(
            "Sprawdź dlaczego docker-compose up -d nie startuje "
            "serwis redis — logi pokazują 'Permission denied' na /data/redis.conf",
            cfg,
        )
        assert result is not None
        assert "/data/redis.conf" in result
