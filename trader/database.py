"""Модуль для сохранения истории сделок в SQLite."""

import sqlite3
from pathlib import Path

DB_PATH = Path("trades.db")
_conn = None


def get_conn(path: Path = DB_PATH) -> sqlite3.Connection:
    """Возвращает соединение с базой данных, создавая файл при необходимости."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(path)
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS trades (
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            price REAL,
            expected_price REAL
        )"""
        )
        _conn.commit()
    return _conn


def log_trade(symbol: str, side: str, price: float, expected: float) -> None:
    """Сохраняет информацию о сделке."""
    conn = get_conn()
    conn.execute(
        "INSERT INTO trades (timestamp, symbol, side, price, expected_price) "
        "VALUES (datetime('now'), ?, ?, ?, ?)",
        (symbol, side, price, expected),
    )
    conn.commit()
