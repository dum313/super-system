"""Модуль принятия торговых решений."""

from binance.client import Client  # клиент Binance
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET  # типы ордеров
from sklearn.linear_model import LinearRegression  # модель для прогнозирования
import pandas as pd  # работа с данными
import logging  # вывод в лог


def trade(
    client: Client,
    model: LinearRegression,
    df: pd.DataFrame,
    symbol: str,
    buy_thr: float,
    sell_thr: float,
) -> str:
    """Совершает тестовую сделку и возвращает описание действия."""
    next_idx = len(df)  # индекс следующей свечи
    predicted_price = model.predict([[next_idx]])[0]  # предсказание цены
    last_price = df['close'].iloc[-1]  # последняя цена

    if predicted_price > last_price * buy_thr:  # ожидается рост цены
        client.create_test_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=0.001,
        )  # тестовый ордер на покупку
        action = f"Покупаем по {last_price:.2f}, ожидание {predicted_price:.2f}"  # сообщение о покупке
        logging.info(action)
    elif predicted_price < last_price * sell_thr:  # ожидается падение цены
        client.create_test_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=0.001,
        )  # тестовый ордер на продажу
        action = f"Продаем по {last_price:.2f}, ожидание {predicted_price:.2f}"  # сообщение о продаже
        logging.info(action)
    else:
        action = f"Нет действия. Цена {last_price:.2f}, ожидание {predicted_price:.2f}"  # остаёмся в позиции
        logging.info(action)
    return action  # возвращаем описание совершённого действия
