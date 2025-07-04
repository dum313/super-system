"""Точка входа и простой графический интерфейс."""

import os  # модуль для работы с переменными окружения
import logging  # модуль логирования
import tkinter as tk  # стандартная библиотека для создания GUI
from binance.client import Client  # клиент для API Binance

from trader.data import fetch_data  # функция получения данных
from trader.model import train_model  # функция обучения модели
from trader.trading import trade  # функция для совершения сделок
from trader.config import load_config  # загрузка настроек


def run_cycle(
    client: Client,
    symbol: str,
    interval: str,
    lookback: int,
    buy_thr: float,
    sell_thr: float,
    result_var: tk.StringVar,
) -> None:
    """Выполняет один цикл обучения и торговли."""
    data = fetch_data(client, symbol, interval, lookback)  # загружаем данные
    model = train_model(data)  # обучаем модель
    action = trade(client, model, data, symbol, buy_thr, sell_thr)  # действие по модели
    result_var.set(action)  # выводим результат в интерфейс


def main() -> None:
    """Запуск приложения."""
    logging.basicConfig(
        filename="bot.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
    )  # настройка логов

    api_key = os.getenv('BINANCE_API_KEY')  # ключ API
    api_secret = os.getenv('BINANCE_API_SECRET')  # секретный ключ
    if not api_key or not api_secret:
        raise RuntimeError('Не заданы ключи API Binance')  # проверяем ключи

    client = Client(api_key, api_secret)  # создаём клиента Binance

    config = load_config()  # читаем настройки из YAML

    symbol = config.get('symbol', 'BTCUSDT')  # торгуемая пара
    interval = getattr(Client, config.get('interval', 'KLINE_INTERVAL_1HOUR'))  # таймфрейм
    lookback_hours = int(config.get('lookback_hours', 100))  # количество часов истории
    buy_thr = float(config.get('buy_threshold', 1.002))  # порог покупки
    sell_thr = float(config.get('sell_threshold', 0.998))  # порог продажи

    root = tk.Tk()  # создаём главное окно
    root.title('AI Trader')  # заголовок окна

    result_var = tk.StringVar(value='Нажмите кнопку для старта')  # переменная для вывода результата
    result_label = tk.Label(root, textvariable=result_var)  # метка с результатом
    result_label.pack(pady=10)  # размещаем метку

    def on_run() -> None:
        run_cycle(
            client,
            symbol,
            interval,
            lookback_hours,
            buy_thr,
            sell_thr,
            result_var,
        )  # запуск цикла по нажатию

    run_button = tk.Button(root, text='Запустить цикл', command=on_run)  # кнопка запуска
    run_button.pack(pady=10)  # размещаем кнопку

    root.mainloop()  # запускаем цикл обработки событий


if __name__ == '__main__':  # запускаем только при прямом вызове
    main()  # вход в программу