"""Модуль принятия торговых решений."""

from binance.client import Client  # клиент Binance
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET  # типы ордеров
from sklearn.linear_model import LinearRegression  # модель для прогнозирования
import pandas as pd  # работа с данными


def trade(client: Client, model: LinearRegression, df: pd.DataFrame, symbol: str) -> str:
    """Совершает тестовую сделку и возвращает описание действия."""
    next_idx = len(df)  # индекс следующей свечи
    predicted_price = model.predict([[next_idx]])[0]  # предсказание цены
    last_price = df['close'].iloc[-1]  # последняя цена

    if predicted_price > last_price * 1.002:  # ожидается рост цены
        client.create_test_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=0.001,
        )  # тестовый ордер на покупку
        action = f"Покупаем по {last_price:.2f}, ожидание {predicted_price:.2f}"  # сообщение о покупке
    elif predicted_price < last_price * 0.998:  # ожидается падение цены
        client.create_test_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=0.001,
        )  # тестовый ордер на продажу
        action = f"Продаем по {last_price:.2f}, ожидание {predicted_price:.2f}"  # сообщение о продаже
    else:
        action = f"Нет действия. Цена {last_price:.2f}, ожидание {predicted_price:.2f}"  # остаёмся в позиции
    return action  # возвращаем описание совершённого действия
