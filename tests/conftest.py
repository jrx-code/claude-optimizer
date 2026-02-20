"""Shared fixtures for tests."""

import pytest
from optimizer.config import OptimizerConfig


@pytest.fixture
def default_config():
    return OptimizerConfig()


@pytest.fixture
def disabled_config():
    return OptimizerConfig(enabled=False)
