import importlib
import sys

import pytest


def test_python_version() -> None:
    assert sys.version_info[:2] == (3, 12)


@pytest.mark.parametrize(
    "module",
    [
        "alpaca",
        "numpy",
        "pandas",
        "pydantic",
        "pydantic_settings",
        "quantstats",
        "structlog",
        "vectorbt",
    ],
)
def test_stack_importable(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module",
    [
        "bot.data",
        "bot.strategy",
        "bot.risk",
        "bot.execution",
        "bot.journal",
        "audit",
    ],
)
def test_project_packages_importable(module: str) -> None:
    importlib.import_module(module)
