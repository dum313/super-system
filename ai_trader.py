import os  # для работы с переменными окружения
import time  # модуль для ожидания между итерациями
import numpy as np  # библиотека для работы с массивами
import pandas as pd  # библиотека для работы с таблицами
from binance.client import Client  # клиент Binance API
from binance.enums import *  # типы ордеров и прочие константы
from sklearn.linear_model import LinearRegression  # простая модель машинного обучения

# Загрузка ключей API из переменных окружения
api_key = os.getenv('BINANCE_API_KEY')  # API ключ
api_secret = os.getenv('BINANCE_API_SECRET')  # секретный ключ

# Создание клиента Binance
client = Client(api_key, api_secret)  # объект для работы с биржей

# Торгуемая пара и интервал
symbol = 'BTCUSDT'  # пара биткоин/доллар
interval = Client.KLINE_INTERVAL_1HOUR  # интервал данных 1 час
lookback_hours = 100  # глубина исторических данных

# Функция загрузки исторических данных
def fetch_data(symbol: str, interval: str, lookback: int = 100) -> pd.DataFrame:
    """Загружаем исторические данные по указанной паре и интервалу."""
    klines = client.get_historical_klines(symbol, interval, f"{lookback} hour ago UTC")
    df = pd.DataFrame(
        klines,
        columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ],
    )
    df['close'] = df['close'].astype(float)
    return df

# Функция обучения модели на исторических данных
def train_model(df: pd.DataFrame) -> LinearRegression:
    df['index'] = np.arange(len(df))
    X = df[['index']]
    y = df['close']
    model = LinearRegression().fit(X, y)
    return model

# Функция принятия решения о покупке или продаже
def trade(model: LinearRegression, df: pd.DataFrame, symbol: str):
    next_idx = len(df)
    predicted_price = model.predict([[next_idx]])[0]
    last_price = df['close'].iloc[-1]

    if predicted_price > last_price * 1.002:
        client.create_test_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=0.001,
        )
        print(f"Покупаем {symbol} по цене {last_price}, ожидание {predicted_price}")
    elif predicted_price < last_price * 0.998:
        client.create_test_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=0.001,
        )
        print(f"Продаем {symbol} по цене {last_price}, ожидание {predicted_price}")
    else:
        print(f"Нет действия. Текущая цена: {last_price}, ожидание: {predicted_price}")

# Основной цикл работы бота
while True:
    data = fetch_data(symbol, interval, lookback_hours)
    model = train_model(data)
    trade(model, data, symbol)
    time.sleep(60 * 60)
