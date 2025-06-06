"""Загрузка конфигурации из YAML."""

import yaml
from pathlib import Path


CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Возвращает словарь с настройками."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
