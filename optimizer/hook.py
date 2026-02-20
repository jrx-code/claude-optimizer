#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook entry point.

Reads hook JSON from stdin, optimizes the prompt via Ollama,
and prints the optimized version as plain text to stdout.

Exit code 0 always — never block the user's prompt.
"""

import json
import sys

from optimizer.config import OptimizerConfig
from optimizer.core import optimize


def main() -> None:
    try:
        raw = sys.stdin.read()
        hook_data = json.loads(raw)
        prompt = hook_data.get("prompt", "")
    except (json.JSONDecodeError, KeyError):
        return

    config = OptimizerConfig.load()
    result = optimize(prompt, config)

    if result:
        print(result)


if __name__ == "__main__":
    main()
