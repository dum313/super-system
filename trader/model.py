"""Модуль обучения модели."""

import numpy as np  # работа с массивами
import pandas as pd  # обработка данных
from sklearn.linear_model import LinearRegression  # модель линейной регрессии


def train_model(df: pd.DataFrame) -> LinearRegression:
    """Обучает модель линейной регрессии и возвращает её."""
    df['index'] = np.arange(len(df))  # создаём признак-порядковый номер
    X = df[['index']]  # признаки для обучения
    y = df['close']  # целевая переменная
    model = LinearRegression().fit(X, y)  # обучаем модель
    return model  # возвращаем обученную модель
