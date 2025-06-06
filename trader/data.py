"""Модуль загрузки исторических данных с Binance."""

import pandas as pd  # библиотека для работы с таблицами
from binance.client import Client  # клиент Binance


def fetch_data(client: Client, symbol: str, interval: str, lookback: int = 100) -> pd.DataFrame:
    """Возвращает DataFrame с историческими ценами."""
    klines = client.get_historical_klines(symbol, interval, f"{lookback} hour ago UTC")  # запрос свечей
    df = pd.DataFrame(
        klines,  # список свечей
        columns=[  # названия столбцов
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ],
    )
    df['close'] = df['close'].astype(float)  # преобразуем цену в float
    return df  # возвращаем таблицу
